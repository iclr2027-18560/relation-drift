"""Elliptic Bitcoin benchmark for Temporal Sheaf V3.5 (+ fast ρ routing).

Protocol (matches Week-7 / V3.7 Elliptic setup):
    - Train timesteps 1–34, test timesteps 35–49
    - Edge labels derived from node classes (both endpoints known)
    - Report per-timestep AUC, mean test AUC, pre/post regime t=43

Usage:
    python src/run/run_elliptic_benchmark.py
    python src/run/run_elliptic_benchmark.py --data_dir data/elliptic --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor

_FIX35FINAL = Path(__file__).resolve().parent
_SRC = _FIX35FINAL.parent
_ROOT = _SRC.parent
for _p in (_SRC, _FIX35FINAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from temporal_sheaf_v35 import (  # noqa: E402
    SheafConfig,
    TrainConfig,
    TemporalSheafV35,
)
from train_v35 import train_temporal_v35  # noqa: E402


# ─── Elliptic constants ─────────────────────────────────────────────────
ELLIPTIC_NUM_FEATURES_DEFAULT = 166
ELLIPTIC_TRAIN_END = 34
ELLIPTIC_TEST_START = 35
ELLIPTIC_LABEL_LICIT = "1"
ELLIPTIC_LABEL_ILLICIT = "2"
REGIME_CHANGE_TIMESTEP = 43
NUM_EDGE_TYPES = 1
SEED = 42
POS_WEIGHT = 9.0
STATIC_EPOCHS = 200
ONLINE_LR = 1e-3
ONLINE_SUPERVISION_RATE = 0.10
DEFAULT_DATA_DIR = _ROOT / "data" / "elliptic"


class TemporalSnapshot:
    """One Elliptic timestep (node features + intra-timestep edges)."""

    def __init__(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        edge_labels: Tensor,
        timestep: int,
    ) -> None:
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.edge_labels = edge_labels
        self.timestep = timestep


class EllipticTemporalDataset:
    """Load Elliptic CSVs into per-timestep snapshots for V3.5."""

    def __init__(
        self,
        data_dir: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.device = device
        self.num_features = ELLIPTIC_NUM_FEATURES_DEFAULT
        self.snapshots: List[TemporalSnapshot] = []
        self.train_snapshots: List[TemporalSnapshot] = []
        self.test_snapshots: List[TemporalSnapshot] = []
        self._load_and_process()

    def _paths(self) -> Tuple[Path, Path, Path]:
        features = self.data_dir / "elliptic_txs_features.csv"
        edges = self.data_dir / "elliptic_txs_edgelist.csv"
        classes = self.data_dir / "elliptic_txs_classes.csv"
        for path in (features, edges, classes):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        return features, edges, classes

    @staticmethod
    def _edge_labels_both_known(
        node_labels: Tensor,
        mask: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Keep edges where both endpoints are labeled; illicit if either is."""
        if edge_index.numel() == 0:
            empty_e = torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)
            empty_y = torch.zeros(0, dtype=torch.float32, device=edge_index.device)
            return empty_e, empty_y

        src, dst = edge_index[0], edge_index[1]
        both = mask[src] & mask[dst]
        if both.sum() == 0:
            empty_e = torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)
            empty_y = torch.zeros(0, dtype=torch.float32, device=edge_index.device)
            return empty_e, empty_y

        kept = edge_index[:, both]
        y = torch.clamp(
            node_labels[kept[0]] + node_labels[kept[1]],
            max=1.0,
        )
        return kept, y

    def _load_and_process(self) -> None:
        features_path, edges_path, classes_path = self._paths()

        print("Loading features...")
        features_df = pd.read_csv(features_path, header=None)
        node_ids = features_df.iloc[:, 0].values
        timesteps = features_df.iloc[:, 1].values.astype(int)
        features = features_df.iloc[:, 2:].values.astype(np.float32)
        self.num_features = int(features.shape[1])
        print(
            f"  {features.shape[0]} txs × {self.num_features} feats "
            f"(CSV cols={features_df.shape[1]})"
        )

        print("Loading edges...")
        edges_df = pd.read_csv(edges_path)
        src_ids = edges_df.iloc[:, 0].values
        dst_ids = edges_df.iloc[:, 1].values

        print("Loading classes...")
        classes_df = pd.read_csv(classes_path)
        class_map: Dict[int, str] = {
            int(row.iloc[0]): str(row.iloc[1]) for _, row in classes_df.iterrows()
        }

        print("Building temporal snapshots...")
        for ts in sorted(set(timesteps)):
            ts_mask = timesteps == ts
            ts_node_ids = node_ids[ts_mask]
            ts_features = features[ts_mask]
            id_to_local = {
                int(nid): i for i, nid in enumerate(ts_node_ids)
            }
            ts_set = set(id_to_local.keys())

            local_src: List[int] = []
            local_dst: List[int] = []
            for s, d in zip(src_ids, dst_ids):
                s_i, d_i = int(s), int(d)
                if s_i in ts_set and d_i in ts_set:
                    local_src.append(id_to_local[s_i])
                    local_dst.append(id_to_local[d_i])

            n = len(ts_node_ids)
            labels = torch.zeros(n, device=self.device)
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            for idx, nid in enumerate(ts_node_ids):
                cls = class_map.get(int(nid))
                if cls == ELLIPTIC_LABEL_ILLICIT:
                    labels[idx] = 1.0
                    mask[idx] = True
                elif cls == ELLIPTIC_LABEL_LICIT:
                    labels[idx] = 0.0
                    mask[idx] = True

            x = torch.tensor(ts_features, dtype=torch.float32, device=self.device)
            if local_src:
                edge_index = torch.tensor(
                    [local_src, local_dst],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

            edge_index, edge_labels = self._edge_labels_both_known(
                labels, mask, edge_index
            )
            edge_types = torch.zeros(
                edge_index.shape[1], dtype=torch.long, device=self.device
            )

            snap = TemporalSnapshot(
                node_features=x,
                edge_index=edge_index,
                edge_types=edge_types,
                edge_labels=edge_labels,
                timestep=int(ts),
            )
            self.snapshots.append(snap)

        self.train_snapshots = [
            s for s in self.snapshots if s.timestep <= ELLIPTIC_TRAIN_END
        ]
        self.test_snapshots = [
            s for s in self.snapshots if s.timestep >= ELLIPTIC_TEST_START
        ]
        print(
            f"Loaded {len(self.snapshots)} timesteps: "
            f"{len(self.train_snapshots)} train, "
            f"{len(self.test_snapshots)} test"
        )

    def to_v35_sequence(
        self,
        snaps: List[TemporalSnapshot],
    ) -> List[Dict[str, Tensor]]:
        """Map Elliptic snapshots to TemporalSheafV35 schema."""
        out: List[Dict[str, Tensor]] = []
        for s in snaps:
            out.append(
                {
                    "x": s.node_features,
                    "edge_index": s.edge_index,
                    "edge_types": s.edge_types,
                    "labels": s.edge_labels,
                    "t": torch.tensor(s.timestep),
                }
            )
        return out


def compute_auc(scores: Tensor, labels: Tensor) -> float:
    """ROC-AUC; 0.5 if undefined."""
    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


def train_static_elliptic(
    train_seq: List[Dict[str, Tensor]],
    input_dim: int,
    device: torch.device,
    epochs: int = STATIC_EPOCHS,
) -> TemporalSheafV35:
    """Train static proxy (no fast encoder, gate forced open, reset each t)."""
    cfg = SheafConfig(
        node_feat_dim=input_dim,
        num_edge_types=NUM_EDGE_TYPES,
        use_fast_encoder=False,
        pos_weight=POS_WEIGHT,
    )
    model = TemporalSheafV35(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, device=device)
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batch = 0
        for snap in train_seq:
            if snap["edge_index"].shape[1] == 0:
                continue
            model.reset_state()
            scores, _ = model.forward_timestep(
                snap["x"],
                snap["edge_index"],
                snap["edge_types"],
                force_gate_open=True,
            )
            loss = loss_fn(scores, snap["labels"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batch += 1
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"  [Static] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {epoch_loss / max(n_batch, 1):.4f}"
            )
    return model


def evaluate_static(
    model: TemporalSheafV35,
    test_seq: List[Dict[str, Tensor]],
) -> List[float]:
    """Per-timestep AUC for static model."""
    model.eval()
    aucs: List[float] = []
    with torch.no_grad():
        for snap in test_seq:
            if snap["edge_index"].shape[1] == 0:
                aucs.append(0.5)
                continue
            model.reset_state()
            scores, _ = model.forward_timestep(
                snap["x"],
                snap["edge_index"],
                snap["edge_types"],
                force_gate_open=True,
            )
            aucs.append(compute_auc(torch.sigmoid(scores), snap["labels"]))
    return aucs


def train_temporal_elliptic(
    train_seq: List[Dict[str, Tensor]],
    static_model: TemporalSheafV35,
    input_dim: int,
    device: torch.device,
    train_cfg: TrainConfig,
) -> TemporalSheafV35:
    """Train V3.5 on Elliptic train timesteps only (no test leakage)."""
    sheaf_cfg = SheafConfig(
        node_feat_dim=input_dim,
        num_edge_types=NUM_EDGE_TYPES,
        use_fast_encoder=True,
        pos_weight=POS_WEIGHT,
    )
    model = TemporalSheafV35(sheaf_cfg).to(device)
    train_ts = list(range(len(train_seq)))
    # No GT jump times on Elliptic — rely on jerk/recovery + curriculum P1
    model = train_temporal_v35(
        model=model,
        static_model=static_model,
        dataset=[train_seq],
        config=train_cfg,
        sheaf_config=sheaf_cfg,
        train_timesteps=train_ts,
        jump_times=[],
    )
    return model


def evaluate_temporal(
    model: TemporalSheafV35,
    train_seq: List[Dict[str, Tensor]],
    test_seq: List[Dict[str, Tensor]],
    online_adapt: bool = True,
    online_lr: float = ONLINE_LR,
    supervision_rate: float = ONLINE_SUPERVISION_RATE,
) -> Tuple[List[float], List[float], List[float]]:
    """Warm on train, then score test with optional light online adaptation.

    Returns:
        (aucs, fusion_gates, adapt_losses)
    """
    model.eval()
    model.reset_state()

    # Carry temporal state through train window
    with torch.no_grad():
        for snap in train_seq:
            if snap["edge_index"].shape[1] == 0:
                continue
            model.forward_timestep(
                snap["x"],
                snap["edge_index"],
                snap["edge_types"],
            )

    aucs: List[float] = []
    gates: List[float] = []
    adapt_losses: List[float] = []
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            POS_WEIGHT, device=next(model.parameters()).device
        )
    )

    for snap in test_seq:
        if snap["edge_index"].shape[1] == 0:
            aucs.append(0.5)
            gates.append(0.0)
            adapt_losses.append(0.0)
            continue

        with torch.no_grad():
            scores, diag = model.forward_timestep(
                snap["x"],
                snap["edge_index"],
                snap["edge_types"],
            )
            aucs.append(compute_auc(torch.sigmoid(scores), snap["labels"]))
            gates.append(float(diag.get("fusion_gate", 0.0)))

        # Light online adaptation (10% edges) — same spirit as V3.7
        adapt_loss = 0.0
        if online_adapt and snap["edge_index"].shape[1] > 0:
            model.train()
            n_e = snap["edge_index"].shape[1]
            n_lab = max(1, int(n_e * supervision_rate))
            idx = torch.randperm(n_e, device=snap["labels"].device)[:n_lab]
            opt = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=online_lr,
            )
            scores_a, _ = model.forward_timestep(
                snap["x"],
                snap["edge_index"],
                snap["edge_types"],
            )
            loss = loss_fn(scores_a[idx], snap["labels"][idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            adapt_loss = float(loss.detach().item())
            model.eval()
            # Detach carried state after adapt step
            model.current_rho = model.current_rho.detach()
            if model.rho_evolver.h_state is not None:
                model.rho_evolver.h_state = model.rho_evolver.h_state.detach()
        adapt_losses.append(adapt_loss)

    return aucs, gates, adapt_losses


def run_elliptic_benchmark(
    data_dir: str | Path,
    quick: bool = False,
    online_adapt: bool = False,
) -> Dict[str, Any]:
    """Run Static vs V3.5 Elliptic benchmark.

    The fair default does not use labels from test snapshots. Online adaptation
    is an explicit diagnostic and must be enabled from the command line.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    static_epochs = 40 if quick else STATIC_EPOCHS
    if quick:
        train_cfg = TrainConfig(
            epochs_phase1=20,
            epochs_phase2=0,
            epochs_phase3=30,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=0.0,  # no GT jumps on Elliptic
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=False,
        )
    else:
        train_cfg = TrainConfig(
            epochs_phase1=60,
            epochs_phase2=0,
            epochs_phase3=120,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=0.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=False,
        )

    print("=" * 72)
    print("TEMPORAL SHEAF V3.5 — ELLIPTIC BITCOIN BENCHMARK")
    print("  Model: GRU ρ_slow + FastRotationEncoder + geodesic fuse")
    print(f"  Device: {device}")
    print(f"  Data:   {data_dir}")
    print(
        f"  Curriculum: P1={train_cfg.epochs_phase1} "
        f"P2={train_cfg.epochs_phase2} P3={train_cfg.epochs_phase3}"
    )
    print(f"  Online adapt: {online_adapt}")
    print("=" * 72)

    dataset = EllipticTemporalDataset(data_dir=data_dir, device=device)
    train_seq = dataset.to_v35_sequence(dataset.train_snapshots)
    test_seq = dataset.to_v35_sequence(dataset.test_snapshots)
    input_dim = dataset.num_features

    print(f"\n[1/4] Training static ({static_epochs} epochs)...")
    static_model = train_static_elliptic(
        train_seq, input_dim, device, epochs=static_epochs
    )

    print("\n[2/4] Evaluating static...")
    static_aucs = evaluate_static(static_model, test_seq)
    static_mean = float(np.mean(static_aucs))
    print(f"  Static mean AUC: {static_mean:.4f}")

    print("\n[3/4] Training V3.5 temporal...")
    temporal_model = train_temporal_elliptic(
        train_seq, static_model, input_dim, device, train_cfg
    )

    print("\n[4/4] Evaluating V3.5 on test (warm-start from train)...")
    temporal_aucs, fusion_gates, adapt_losses = evaluate_temporal(
        temporal_model,
        train_seq,
        test_seq,
        online_adapt=online_adapt,
    )
    temporal_mean = float(np.mean(temporal_aucs))
    print(f"  V3.5 mean AUC: {temporal_mean:.4f}")

    regime_idx = REGIME_CHANGE_TIMESTEP - ELLIPTIC_TEST_START
    pre_s = static_aucs[:regime_idx]
    post_s = static_aucs[regime_idx:]
    pre_t = temporal_aucs[:regime_idx]
    post_t = temporal_aucs[regime_idx:]

    results: Dict[str, Any] = {
        "model": "TemporalSheafV35+fast",
        "num_features": input_dim,
        "static_epochs": static_epochs,
        "curriculum": {
            "phase1": train_cfg.epochs_phase1,
            "phase2": train_cfg.epochs_phase2,
            "phase3": train_cfg.epochs_phase3,
        },
        "online_adapt": online_adapt,
        "static_aucs": static_aucs,
        "static_mean_auc": static_mean,
        "temporal_aucs": temporal_aucs,
        "temporal_mean_auc": temporal_mean,
        "fusion_gates": fusion_gates,
        "adaptation_losses": adapt_losses,
        "improvement": temporal_mean - static_mean,
        "pre_regime_static_mean": float(np.mean(pre_s)) if pre_s else 0.0,
        "post_regime_static_mean": float(np.mean(post_s)) if post_s else 0.0,
        "pre_regime_temporal_mean": float(np.mean(pre_t)) if pre_t else 0.0,
        "post_regime_temporal_mean": float(np.mean(post_t)) if post_t else 0.0,
        "test_timesteps": [
            ELLIPTIC_TEST_START + i for i in range(len(test_seq))
        ],
        "regime_change_timestep": REGIME_CHANGE_TIMESTEP,
    }
    results["static_degradation"] = (
        results["pre_regime_static_mean"] - results["post_regime_static_mean"]
    )
    results["temporal_degradation"] = (
        results["pre_regime_temporal_mean"]
        - results["post_regime_temporal_mean"]
    )

    print("\n" + "=" * 72)
    print("ELLIPTIC V3.5 RESULTS")
    print("=" * 72)
    print(f"Overall Static AUC:  {static_mean:.4f}")
    print(f"Overall V3.5 AUC:    {temporal_mean:.4f}")
    print(f"Improvement:         {results['improvement']:+.4f}")
    print(f"\nAround regime change (t={REGIME_CHANGE_TIMESTEP}):")
    print(
        f"  Static pre/post: "
        f"{results['pre_regime_static_mean']:.4f} → "
        f"{results['post_regime_static_mean']:.4f} "
        f"(Δ={results['static_degradation']:+.4f})"
    )
    print(
        f"  V3.5  pre/post: "
        f"{results['pre_regime_temporal_mean']:.4f} → "
        f"{results['post_regime_temporal_mean']:.4f} "
        f"(Δ={results['temporal_degradation']:+.4f})"
    )
    print("\nPer-timestep AUC:")
    print(f"{'Timestep':<10} | {'Static':>7} | {'V3.5':>7} | {'Gate':>6}")
    print("-" * 40)
    for i, (s_auc, t_auc) in enumerate(zip(static_aucs, temporal_aucs)):
        ts = ELLIPTIC_TEST_START + i
        g = fusion_gates[i] if i < len(fusion_gates) else 0.0
        marker = " <- regime" if ts == REGIME_CHANGE_TIMESTEP else ""
        print(
            f"t={ts:<7} | {s_auc:>7.4f} | {t_auc:>7.4f} | "
            f"{g:>6.3f}{marker}"
        )
    print("=" * 72)

    out_path = _ROOT / "results" / "elliptic_v35_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elliptic Bitcoin benchmark for Temporal Sheaf V3.5"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory with elliptic_txs_{features,edgelist,classes}.csv",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter epochs for a smoke run",
    )
    parser.add_argument(
        "--online-adapt",
        action="store_true",
        help="Enable diagnostic adaptation using 10%% of each test snapshot",
    )
    args = parser.parse_args()
    run_elliptic_benchmark(
        data_dir=args.data_dir,
        quick=args.quick,
        online_adapt=args.online_adapt,
    )


if __name__ == "__main__":
    main()

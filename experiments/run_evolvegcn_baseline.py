# EvolveGCN baseline for temporal edge anomaly detection.
# Pareja et al. (AAAI 2020) via PyG-style sparse GCN with RNN-evolved weights
# (same design as pytorch_geometric_temporal EvolveGCNH / EvolveGCNO).
# Same Week-7 data pipeline and temporal holdout as V3.5 (src/run).

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn import GRU, Parameter
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)

# ---------------------------------------------------------------------------
# Constants (aligned with V3.5 / TGN baseline)
# ---------------------------------------------------------------------------
TRAIN_TIMESTEP_RATIO: float = 0.5
POS_WEIGHT: float = 9.0
GRAD_CLIP: float = 1.0
NUM_EPOCHS: int = 200
HIDDEN_DIM: int = 64
NUM_LAYERS: int = 2
LR: float = 1e-3
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMPLEMENTATION_NAME = "EvolveGCN (Pareja et al.; PyG sparse GCN + RNN weights)"


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------
def _adapt_sheaf_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Map Week-7 snapshots for EvolveGCN edge anomaly scoring.

    Structure edges (union of typed edge_index) are used for GCN message
    passing; query_edges + labels are the supervised anomaly targets.
    """
    if not dataset:
        return []

    edge_type_names: List[str] = list(dataset[0]["edge_types"])
    snapshots: List[Dict[str, Tensor]] = []
    for t_idx, snap in enumerate(dataset):
        # Union of typed structural edges (undirected for GCN).
        typed = snap["edge_index"]
        parts = [typed[etype].long() for etype in edge_type_names]
        struct = torch.cat(parts, dim=1)
        # Add reverse edges for undirected message passing.
        struct = torch.cat([struct, struct.flip(0)], dim=1)

        snapshots.append(
            {
                "x": snap["node_features"].float(),
                "struct_edge_index": struct,
                "edge_index": snap["query_edges"].long(),
                "labels": snap["labels"].float(),
                "clean_labels": snap["clean_labels"].float(),
                "t": torch.tensor(t_idx),
                "t_normalized": torch.tensor(float(snap["t"])),
            }
        )
    return snapshots


def load_temporal_dataset(
    noise_name: str,
    drift_name: str,
    seed: int = 42,
) -> List[Dict[str, Tensor]]:
    raw = generate_temporal_dataset(
        DRIFT_SCENARIOS[drift_name],
        NOISE_CONFIGS[noise_name],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    return _adapt_sheaf_dataset(raw)


def compute_auc(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_scores_np = (
        y_scores.cpu().numpy() if torch.is_tensor(y_scores) else np.array(y_scores)
    )
    if len(np.unique(y_true_np)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_np, y_scores_np))


# ---------------------------------------------------------------------------
# EvolveGCN layers (pytorch_geometric_temporal-compatible)
# ---------------------------------------------------------------------------


class GCNConvFixedW(MessagePassing):
    """GCN convolution with externally provided (non-trainable) weight W."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def forward(
        self,
        W: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                self.improved,
                self.add_self_loops,
                dtype=x.dtype,
            )
        x = torch.matmul(x, W)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class EvolveGCNH(nn.Module):
    """EvolveGCN-H: TopK graph summary → GRU → evolved GCN weight (Pareja)."""

    def __init__(self, num_nodes: int, channels: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.initial_weight = Parameter(torch.empty(1, channels, channels))
        self.weight: Optional[Tensor] = None
        # Pool roughly ``channels`` nodes to form the GRU input summary.
        ratio = min(1.0, max(channels / max(num_nodes, 1), 1.0 / max(num_nodes, 1)))
        self.pooling_layer = TopKPooling(channels, ratio)
        self.recurrent_layer = GRU(
            input_size=channels, hidden_size=channels, num_layers=1
        )
        self.conv_layer = GCNConvFixedW(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.initial_weight)
        self.weight = None

    def reinitialize_weight(self) -> None:
        self.weight = None

    def detach_weight(self) -> None:
        if self.weight is not None:
            self.weight = self.weight.detach()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x_tilde, _, _, _, _, _ = self.pooling_layer(x, edge_index)
        # TopK may return fewer than ``channels`` rows; pad/truncate for GRU.
        if x_tilde.size(0) < self.channels:
            pad = x.new_zeros(self.channels - x_tilde.size(0), self.channels)
            x_tilde = torch.cat([x_tilde, pad], dim=0)
        elif x_tilde.size(0) > self.channels:
            x_tilde = x_tilde[: self.channels]
        x_tilde = x_tilde.unsqueeze(0)  # (1, channels, channels)

        if self.weight is None:
            _, self.weight = self.recurrent_layer(x_tilde, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(x_tilde, self.weight)

        return self.conv_layer(self.weight.squeeze(0), x, edge_index, edge_weight)


class EvolveGCNO(nn.Module):
    """EvolveGCN-O: LSTM/GRU evolves W without node-feature conditioning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.initial_weight = Parameter(torch.empty(1, channels, channels))
        self.weight: Optional[Tensor] = None
        self.recurrent_layer = GRU(
            input_size=channels, hidden_size=channels, num_layers=1
        )
        self.conv_layer = GCNConvFixedW(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.initial_weight)
        self.weight = None

    def reinitialize_weight(self) -> None:
        self.weight = None

    def detach_weight(self) -> None:
        if self.weight is not None:
            self.weight = self.weight.detach()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if self.weight is None:
            _, self.weight = self.recurrent_layer(
                self.initial_weight, self.initial_weight
            )
        else:
            _, self.weight = self.recurrent_layer(self.weight, self.weight)
        return self.conv_layer(self.weight.squeeze(0), x, edge_index, edge_weight)


class EdgeAnomalyPredictor(nn.Module):
    """MLP on concatenated endpoint embeddings (+ optional raw features)."""

    def __init__(self, emb_dim: int, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + feat_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, z_src: Tensor, z_dst: Tensor, x_src: Tensor, x_dst: Tensor
    ) -> Tensor:
        return self.mlp(torch.cat([z_src, z_dst, x_src, x_dst], dim=-1)).squeeze(-1)


class EvolveGCNModel(nn.Module):
    """Stacked EvolveGCN + edge anomaly head."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        variant: str = "H",
    ):
        super().__init__()
        if variant not in {"H", "O"}:
            raise ValueError(f"variant must be 'H' or 'O', got {variant}")
        self.variant = variant
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            if variant == "H":
                layers.append(EvolveGCNH(num_nodes, hidden_dim))
            else:
                layers.append(EvolveGCNO(hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.predictor = EdgeAnomalyPredictor(hidden_dim, in_channels)

    def reinitialize_weights(self) -> None:
        for layer in self.layers:
            layer.reinitialize_weight()

    def detach_weights(self) -> None:
        for layer in self.layers:
            layer.detach_weight()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x: Tensor, struct_edge_index: Tensor) -> Tensor:
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = F.relu(layer(h, struct_edge_index))
        return h

    def forward_timestep(
        self,
        x: Tensor,
        struct_edge_index: Tensor,
        query_edge_index: Tensor,
    ) -> Tensor:
        z = self.encode(x, struct_edge_index)
        src, dst = query_edge_index[0], query_edge_index[1]
        return self.predictor(z[src], z[dst], x[src], x[dst])


# ---------------------------------------------------------------------------
# Static baseline
# ---------------------------------------------------------------------------


class StaticEdgeClassifier(nn.Module):
    def __init__(self, node_feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, src_feats: Tensor, dst_feats: Tensor) -> Tensor:
        return self.mlp(torch.cat([src_feats, dst_feats], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train_and_evaluate_evolvegcn(
    timesteps: List[dict],
    num_nodes: int,
    node_feat_dim: int,
    num_epochs: int = NUM_EPOCHS,
    variant: str = "H",
    train_end: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    n_t = len(timesteps)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    model = EvolveGCNModel(
        num_nodes=num_nodes,
        in_channels=node_feat_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        variant=variant,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=DEVICE)

    print(
        f"    EvolveGCN-{variant} parameters: {model.count_parameters():,} "
        f"(~{model.count_parameters()/1000:.1f}K)"
    )

    model.train()
    for epoch in range(num_epochs):
        model.reinitialize_weights()
        total_loss = 0.0
        total_edges = 0

        for t in range(train_end):
            x = timesteps[t]["x"].to(DEVICE)
            struct = timesteps[t]["struct_edge_index"].to(DEVICE)
            query = timesteps[t]["edge_index"].to(DEVICE)
            y = timesteps[t]["labels"].to(DEVICE)
            if query.shape[1] == 0:
                continue

            optimizer.zero_grad()
            logits = model.forward_timestep(x, struct, query)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            model.detach_weights()

            total_loss += float(loss.item()) * query.shape[1]
            total_edges += int(query.shape[1])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = total_loss / max(total_edges, 1)
            print(f"      Epoch {epoch+1:3d}/{num_epochs}: loss={avg:.4f}")

    # Eval: warm weight evolution on train, score test.
    model.eval()
    model.reinitialize_weights()
    all_labels: List[Tensor] = []
    all_scores: List[Tensor] = []
    per_timestep_auc: Dict[int, float] = {}

    with torch.no_grad():
        for t in range(n_t):
            x = timesteps[t]["x"].to(DEVICE)
            struct = timesteps[t]["struct_edge_index"].to(DEVICE)
            query = timesteps[t]["edge_index"].to(DEVICE)
            y = timesteps[t]["labels"].to(DEVICE)
            if query.shape[1] == 0:
                if t >= train_end:
                    per_timestep_auc[t] = 0.5
                continue

            logits = model.forward_timestep(x, struct, query)
            if t < train_end:
                continue
            scores = torch.sigmoid(logits)
            per_timestep_auc[t] = compute_auc(y, scores)
            all_labels.append(y.cpu())
            all_scores.append(scores.cpu())

    if not all_labels:
        return 0.5, per_timestep_auc
    return compute_auc(torch.cat(all_labels), torch.cat(all_scores)), per_timestep_auc


def train_and_evaluate_static(
    timesteps: List[dict],
    node_feat_dim: int,
    num_epochs: int = NUM_EPOCHS,
    train_end: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    n_t = len(timesteps)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    model = StaticEdgeClassifier(node_feat_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=DEVICE)

    train_src: List[Tensor] = []
    train_dst: List[Tensor] = []
    train_y: List[Tensor] = []
    for t in range(train_end):
        x = timesteps[t]["x"]
        ei = timesteps[t]["edge_index"]
        y = timesteps[t]["labels"]
        if ei.shape[1] == 0:
            continue
        train_src.append(x[ei[0]])
        train_dst.append(x[ei[1]])
        train_y.append(y)

    if not train_y:
        return 0.5, {t: 0.5 for t in range(train_end, n_t)}

    src = torch.cat(train_src).to(DEVICE)
    dst = torch.cat(train_dst).to(DEVICE)
    y_all = torch.cat(train_y).to(DEVICE)

    model.train()
    for epoch in range(num_epochs):
        logits = model(src, dst)
        loss = F.binary_cross_entropy_with_logits(
            logits, y_all, pos_weight=pos_weight_tensor
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"      Epoch {epoch + 1:3d}/{num_epochs}: loss={loss.item():.4f}")

    model.eval()
    all_labels: List[Tensor] = []
    all_scores: List[Tensor] = []
    per_timestep_auc: Dict[int, float] = {}
    with torch.no_grad():
        for t in range(train_end, n_t):
            x = timesteps[t]["x"].to(DEVICE)
            ei = timesteps[t]["edge_index"].to(DEVICE)
            y = timesteps[t]["labels"].to(DEVICE)
            if ei.shape[1] == 0:
                per_timestep_auc[t] = 0.5
                continue
            scores = torch.sigmoid(model(x[ei[0]], x[ei[1]]))
            per_timestep_auc[t] = compute_auc(y, scores)
            all_labels.append(y.cpu())
            all_scores.append(scores.cpu())
    if not all_labels:
        return 0.5, per_timestep_auc
    return compute_auc(torch.cat(all_labels), torch.cat(all_scores)), per_timestep_auc


# ---------------------------------------------------------------------------
# V3.5 comparison + benchmark harness
# ---------------------------------------------------------------------------


def load_v35_results() -> Optional[dict]:
    candidates = [
        _ROOT / "results" / "week8_v35_routing.json",
        _ROOT / "results" / "week8_v35_final.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            print(f"  [INFO] Loaded V3.5 results from: {path}")
            return data
    print("  [INFO] V3.5 results not found; comparison column will be N/A.")
    return None


def get_v35_post_auc(
    v35_data: Optional[dict], noise_name: str, drift_name: str
) -> Optional[float]:
    if v35_data is None:
        return None
    entries = v35_data.get("results", v35_data)
    if isinstance(entries, list):
        for row in entries:
            if (
                isinstance(row, dict)
                and row.get("noise_config") == noise_name
                and row.get("drift_scenario") == drift_name
            ):
                for key in ("v35_post", "mean_post", "auc", "test_auc"):
                    if key in row:
                        return float(row[key])
    return None


def run_benchmark(
    noise_filter: Optional[List[str]] = None,
    drift_filter: Optional[List[str]] = None,
    quick: bool = False,
    variant: str = "H",
    seed: int = 42,
) -> Dict[str, Any]:
    epochs = 40 if quick else NUM_EPOCHS
    noise_items = list(NOISE_CONFIGS.keys())
    drift_items = list(DRIFT_SCENARIOS.keys())
    if noise_filter:
        noise_items = [n for n in noise_items if n in set(noise_filter)]
    if drift_filter:
        drift_items = [d for d in drift_items if d in set(drift_filter)]

    impl = f"{IMPLEMENTATION_NAME} [{variant}]"
    print("=" * 80)
    print("EVOLVEGCN BASELINE BENCHMARK (fair vs V3.5)")
    print(f"  Implementation: {impl}")
    print(f"  Device: {DEVICE}")
    print(
        f"  Timesteps: {NUM_TIMESTEPS} "
        f"(train 0-{int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO) - 1}, "
        f"test {int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO)}-{NUM_TIMESTEPS - 1})"
    )
    print(
        f"  Epochs: {epochs}, pos_weight: {POS_WEIGHT}, lr: {LR}, "
        f"hidden={HIDDEN_DIM}, layers={NUM_LAYERS}"
    )
    print("=" * 80)

    v35_data = load_v35_results()
    results: Dict[str, Dict[str, dict]] = {}
    all_per_timestep: Dict[str, Dict[str, dict]] = {}
    flat_rows: List[dict] = []

    for noise_name in noise_items:
        results[noise_name] = {}
        all_per_timestep[noise_name] = {}

        for drift_name in drift_items:
            print(f"\n{'─' * 60}")
            print(f"  Setting: {noise_name}/{drift_name}")
            print(f"{'─' * 60}")

            timesteps = load_temporal_dataset(noise_name, drift_name, seed=seed)
            num_nodes = int(max(ts["x"].shape[0] for ts in timesteps))
            node_feat_dim = int(timesteps[0]["x"].shape[1])

            print("  Training Static MLP baseline...")
            static_auc, static_per_t = train_and_evaluate_static(
                timesteps, node_feat_dim, num_epochs=epochs
            )
            print(f"    Static AUC: {static_auc:.4f}")

            print(f"  Training EvolveGCN-{variant}...")
            egcn_auc, egcn_per_t = train_and_evaluate_evolvegcn(
                timesteps,
                num_nodes,
                node_feat_dim,
                num_epochs=epochs,
                variant=variant,
            )
            print(f"    EvolveGCN-{variant} AUC: {egcn_auc:.4f}")

            v35_auc = get_v35_post_auc(v35_data, noise_name, drift_name)
            if v35_auc is not None:
                print(f"    V3.5 post AUC: {v35_auc:.4f}")

            results[noise_name][drift_name] = {
                "static_auc": static_auc,
                "evolvegcn_auc": egcn_auc,
                "v35_post": v35_auc,
            }
            all_per_timestep[noise_name][drift_name] = {
                "static": {str(k): v for k, v in static_per_t.items()},
                "evolvegcn": {str(k): v for k, v in egcn_per_t.items()},
            }
            flat_rows.append(
                {
                    "noise_config": noise_name,
                    "drift_scenario": drift_name,
                    "static_auc": static_auc,
                    "evolvegcn_auc": egcn_auc,
                    "v35_post": v35_auc,
                }
            )

            train_end = int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO)
            print(f"    Per-timestep AUC (test t={train_end}-{NUM_TIMESTEPS - 1}):")
            for t in range(train_end, NUM_TIMESTEPS):
                print(
                    f"      t={t}: Static={static_per_t.get(t, 0.5):.4f}  "
                    f"EvolveGCN={egcn_per_t.get(t, 0.5):.4f}"
                )

    print("\n" + "=" * 88)
    print(f"COMPARISON: Static MLP vs EvolveGCN-{variant} vs V3.5 post")
    print("=" * 88)
    header = (
        f"{'Noise':<12} {'Drift':<16} {'Static':>8} {'EGCN':>8} "
        f"{'V3.5':>8} {'EGCN-Static':>12} {'EGCN-V3.5':>12}"
    )
    print(header)
    print("-" * len(header))

    static_aucs: List[float] = []
    egcn_aucs: List[float] = []
    v35_aucs: List[float] = []
    for row in flat_rows:
        static = row["static_auc"]
        egcn = row["evolvegcn_auc"]
        v35 = row["v35_post"]
        static_aucs.append(static)
        egcn_aucs.append(egcn)
        v35_str = f"{v35:.4f}" if v35 is not None else "N/A"
        diff_v35 = f"{egcn - v35:+.4f}" if v35 is not None else "N/A"
        if v35 is not None:
            v35_aucs.append(v35)
        print(
            f"{row['noise_config']:<12} {row['drift_scenario']:<16} "
            f"{static:>8.4f} {egcn:>8.4f} {v35_str:>8} "
            f"{egcn - static:>+12.4f} {diff_v35:>12}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"  Static MLP mean: {np.mean(static_aucs):.4f} "
        f"(std {np.std(static_aucs):.4f})"
    )
    print(
        f"  EvolveGCN-{variant} mean: {np.mean(egcn_aucs):.4f} "
        f"(std {np.std(egcn_aucs):.4f})"
    )
    if v35_aucs:
        print(
            f"  V3.5 post mean:  {np.mean(v35_aucs):.4f} "
            f"(std {np.std(v35_aucs):.4f})"
        )

    output = {
        "config": {
            "model": impl,
            "variant": variant,
            "num_timesteps": NUM_TIMESTEPS,
            "train_timestep_ratio": TRAIN_TIMESTEP_RATIO,
            "pos_weight": POS_WEIGHT,
            "num_epochs": epochs,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "lr": LR,
            "device": str(DEVICE),
            "note": (
                "Pareja EvolveGCN with PyG sparse GCNConv_Fixed_W. "
                "H: TopK+GRU conditioned on pooled nodes; O: GRU on W only. "
                "Predict on query edges; structure = union of typed edges. "
                "Static is feature MLP (not sheaf static)."
            ),
        },
        "results": results,
        "results_flat": flat_rows,
        "per_timestep_auc": all_per_timestep,
        "summary": {
            "static_mean_auc": float(np.mean(static_aucs)),
            "evolvegcn_mean_auc": float(np.mean(egcn_aucs)),
            "v35_mean_post": float(np.mean(v35_aucs)) if v35_aucs else None,
        },
    }

    out_path = _ROOT / "results" / "evolvegcn_baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvolveGCN baseline on Week-7 / V3.5 fair protocol"
    )
    parser.add_argument("--quick", action="store_true", help="40 epochs")
    parser.add_argument(
        "--variant",
        choices=["H", "O"],
        default="H",
        help="EvolveGCN-H (default) or EvolveGCN-O",
    )
    parser.add_argument("--noise", nargs="*", default=None)
    parser.add_argument("--drift", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(
        noise_filter=args.noise,
        drift_filter=args.drift,
        quick=args.quick,
        variant=args.variant,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

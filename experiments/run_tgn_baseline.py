# Actual PyG TGN baseline for temporal edge anomaly detection.
# Uses torch_geometric TGNMemory + IdentityMessage + LastAggregator +
# LastNeighborLoader + TransformerConv embedding (Rossi et al. 2020 / PyG examples/tgn.py).
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
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

# ---------------------------------------------------------------------------
# Path setup: experiments/ → repo root + src/
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
# Constants (aligned with V3.5)
# ---------------------------------------------------------------------------
TRAIN_TIMESTEP_RATIO: float = 0.5
POS_WEIGHT: float = 9.0
GRAD_CLIP: float = 1.0
NUM_EPOCHS: int = 200
MEMORY_DIM: int = 64
EMBEDDING_DIM: int = 64
TIME_DIM: int = 64
NEIGHBOR_SIZE: int = 10
LR: float = 1e-4
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMPLEMENTATION_NAME = "TGN (PyG TGNMemory + TransformerConv)"


# ---------------------------------------------------------------------------
# Dataset adapter (identical to src/run/run_v35_benchmark.py)
# ---------------------------------------------------------------------------
def _adapt_sheaf_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Map Week-7 snapshots to the V3.5 train/eval schema."""
    if not dataset:
        return []

    edge_type_names: List[str] = list(dataset[0]["edge_types"])
    edge_index_by_type: Dict[str, Tensor] = dataset[0]["edge_index"]
    etype_to_id = {etype: i for i, etype in enumerate(edge_type_names)}
    edge_type_ids = torch.cat(
        [
            torch.full(
                (edge_index_by_type[etype].shape[1],),
                etype_to_id[etype],
                dtype=torch.long,
            )
            for etype in edge_type_names
        ],
        dim=0,
    )

    snapshots: List[Dict[str, Tensor]] = []
    for t_idx, snap in enumerate(dataset):
        snapshots.append(
            {
                "x": snap["node_features"].float(),
                "edge_index": snap["query_edges"].long(),
                "edge_types": edge_type_ids,
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
    """Same generator call as V3.5 fair benchmark."""
    raw = generate_temporal_dataset(
        DRIFT_SCENARIOS[drift_name],
        NOISE_CONFIGS[noise_name],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    return _adapt_sheaf_dataset(raw)


def snapshots_to_temporal_data(
    timesteps: List[Dict[str, Tensor]],
) -> Tuple[TemporalData, int, int]:
    """Flatten snapshot edges into a chronological TemporalData event stream.

    Raw message = concat(src_feat, dst_feat), matching common TGN feature usage.
    Timestamps are integer snapshot indices (required by TGNMemory).
    """
    srcs: List[Tensor] = []
    dsts: List[Tensor] = []
    ts: List[Tensor] = []
    msgs: List[Tensor] = []
    ys: List[Tensor] = []
    snap_ids: List[Tensor] = []

    num_nodes = 0
    msg_dim = 0
    for t_idx, snap in enumerate(timesteps):
        x = snap["x"]
        edge_index = snap["edge_index"]
        labels = snap["labels"]
        offset = int(snap.get("node_offset", 0))
        num_nodes = max(num_nodes, int(x.shape[0]) + offset)
        if edge_index.shape[1] == 0:
            continue
        local_src = edge_index[0].long()
        local_dst = edge_index[1].long()
        src = local_src + offset
        dst = local_dst + offset
        num_nodes = max(num_nodes, int(src.max().item()) + 1, int(dst.max().item()) + 1)
        msg = torch.cat([x[local_src], x[local_dst]], dim=-1)
        msg_dim = int(msg.shape[1])
        e = src.shape[0]
        srcs.append(src)
        dsts.append(dst)
        ts.append(torch.full((e,), t_idx, dtype=torch.long))
        msgs.append(msg)
        ys.append(labels.float())
        snap_ids.append(torch.full((e,), t_idx, dtype=torch.long))

    if not srcs:
        raise RuntimeError("No edges found in temporal dataset")

    data = TemporalData(
        src=torch.cat(srcs),
        dst=torch.cat(dsts),
        t=torch.cat(ts),
        msg=torch.cat(msgs),
        y=torch.cat(ys),
        snap_id=torch.cat(snap_ids),
    )
    # TemporalData has no num_nodes field by default; store explicitly.
    data.num_nodes = num_nodes
    return data, num_nodes, msg_dim


def compute_auc(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Compute ROC AUC; returns 0.5 if only one class present."""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_scores_np = (
        y_scores.cpu().numpy() if torch.is_tensor(y_scores) else np.array(y_scores)
    )
    if len(np.unique(y_true_np)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_np, y_scores_np))


# ---------------------------------------------------------------------------
# PyG TGN modules (from examples/tgn.py, adapted for edge anomaly)
# ---------------------------------------------------------------------------


class GraphAttentionEmbedding(nn.Module):
    """Temporal graph attention over recent neighbors (PyG TGN example)."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            # Cold start: no historical neighbors — TransformerConv root path only.
            edge_dim = msg.size(-1) + self.time_enc.out_channels
            empty_attr = x.new_zeros((0, edge_dim))
            return self.conv(x, edge_index, empty_attr)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class EdgeAnomalyPredictor(nn.Module):
    """Score anomalous edges from TGN embeddings (+ raw message features)."""

    def __init__(self, emb_dim: int, msg_dim: int):
        super().__init__()
        self.lin_src = Linear(emb_dim, emb_dim)
        self.lin_dst = Linear(emb_dim, emb_dim)
        self.lin_msg = Linear(msg_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            Linear(emb_dim, emb_dim),
            nn.ReLU(),
            Linear(emb_dim, 1),
        )

    def forward(self, z_src: Tensor, z_dst: Tensor, msg: Tensor) -> Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst) + self.lin_msg(msg)
        return self.mlp(h).squeeze(-1)


class ActualTGN(nn.Module):
    """Full PyG TGN stack for supervised edge anomaly detection."""

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        memory_dim: int = MEMORY_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        time_dim: int = TIME_DIM,
        neighbor_size: int = NEIGHBOR_SIZE,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        )
        self.predictor = EdgeAnomalyPredictor(embedding_dim, msg_dim)
        self.neighbor_loader = LastNeighborLoader(
            num_nodes, size=neighbor_size, device=DEVICE
        )
        self.register_buffer(
            "assoc", torch.empty(num_nodes, dtype=torch.long), persistent=False
        )

    def reset_state(self) -> None:
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed_batch(
        self,
        src: Tensor,
        dst: Tensor,
        data_t: Tensor,
        data_msg: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (z_src, z_dst, msg) using pre-update memory + neighbor GNN."""
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        z, last_update = self.memory(n_id)
        if e_id.numel() > 0:
            z = self.gnn(
                z,
                last_update,
                edge_index,
                data_t[e_id].to(z.device),
                data_msg[e_id].to(z.device),
            )
        else:
            z = self.gnn(
                z,
                last_update,
                edge_index,
                torch.empty(0, dtype=torch.long, device=z.device),
                torch.empty(0, self.msg_dim, device=z.device),
            )

        return z[self.assoc[src]], z[self.assoc[dst]], data_msg

    def predict_then_update(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
        data_t: Tensor,
        data_msg: Tensor,
        update: bool = True,
    ) -> Tensor:
        """Score edges with current memory, then ingest events (no same-edge leak)."""
        # For the current batch, predictor uses raw msg of *this* batch;
        # neighbor GNN uses historical data_t / data_msg via e_id.
        z_src, z_dst, _ = self.embed_batch(src, dst, data_t, data_msg)
        logits = self.predictor(z_src, z_dst, msg)
        if update:
            self.memory.update_state(src, dst, t, msg)
            self.neighbor_loader.insert(src, dst)
        return logits


# ---------------------------------------------------------------------------
# Static baseline (MLP on concatenated node features, no temporal info)
# ---------------------------------------------------------------------------


class StaticEdgeClassifier(nn.Module):
    """Simple MLP baseline that ignores temporal information."""

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
# Training and evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate_tgn(
    timesteps: List[dict],
    num_epochs: int = NUM_EPOCHS,
    train_end: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """Train actual PyG TGN on t<train_end, evaluate on later snapshots."""
    n_t = len(timesteps)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    data, num_nodes, msg_dim = snapshots_to_temporal_data(timesteps)
    data = data.to(DEVICE)

    model = ActualTGN(num_nodes=num_nodes, msg_dim=msg_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=DEVICE)

    print(f"    TGN parameters: {model.count_parameters():,} (~{model.count_parameters()/1000:.1f}K)")
    print(f"    Events: {int(data.num_events)}, nodes: {num_nodes}, msg_dim: {msg_dim}")

    model.train()
    for epoch in range(num_epochs):
        model.reset_state()
        total_loss = 0.0
        total_edges = 0

        # Chronological snapshot batches (matches V3.5 temporal protocol).
        for t_snap in range(train_end):
            idx = (data.snap_id == t_snap).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            src = data.src[idx]
            dst = data.dst[idx]
            t = data.t[idx]
            msg = data.msg[idx]
            y = data.y[idx]

            optimizer.zero_grad()
            logits = model.predict_then_update(
                src, dst, t, msg, data.t, data.msg, update=True
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            model.memory.detach()

            total_loss += float(loss.item()) * idx.numel()
            total_edges += int(idx.numel())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / max(total_edges, 1)
            print(f"      Epoch {epoch+1:3d}/{num_epochs}: loss={avg_loss:.4f}")

    # Eval: warm memory on train, then score test (predict-then-update).
    model.eval()
    model.reset_state()
    all_labels: List[Tensor] = []
    all_scores: List[Tensor] = []
    per_timestep_auc: Dict[int, float] = {}

    with torch.no_grad():
        for t_snap in range(n_t):
            idx = (data.snap_id == t_snap).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                if t_snap >= train_end:
                    per_timestep_auc[t_snap] = 0.5
                continue

            src = data.src[idx]
            dst = data.dst[idx]
            t = data.t[idx]
            msg = data.msg[idx]
            y = data.y[idx]

            logits = model.predict_then_update(
                src, dst, t, msg, data.t, data.msg, update=True
            )

            if t_snap < train_end:
                continue

            scores = torch.sigmoid(logits)
            per_timestep_auc[t_snap] = compute_auc(y, scores)
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
    """Train static MLP on t<train_end; evaluate on later snapshots."""
    n_t = len(timesteps)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    model = StaticEdgeClassifier(node_feat_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=DEVICE)

    train_src_feats: List[Tensor] = []
    train_dst_feats: List[Tensor] = []
    train_labels: List[Tensor] = []

    for t in range(train_end):
        node_feats = timesteps[t]["x"]
        edge_index = timesteps[t]["edge_index"]
        edge_labels = timesteps[t]["labels"]
        if edge_index.shape[1] == 0:
            continue
        train_src_feats.append(node_feats[edge_index[0]])
        train_dst_feats.append(node_feats[edge_index[1]])
        train_labels.append(edge_labels)

    if len(train_labels) == 0:
        return 0.5, {t: 0.5 for t in range(train_end, n_t)}

    train_src = torch.cat(train_src_feats).to(DEVICE)
    train_dst = torch.cat(train_dst_feats).to(DEVICE)
    train_y = torch.cat(train_labels).to(DEVICE)

    model.train()
    for epoch in range(num_epochs):
        logits = model(train_src, train_dst)
        loss = F.binary_cross_entropy_with_logits(
            logits, train_y, pos_weight=pos_weight_tensor
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
            node_feats = timesteps[t]["x"].to(DEVICE)
            edge_index = timesteps[t]["edge_index"].to(DEVICE)
            edge_labels = timesteps[t]["labels"].to(DEVICE)
            if edge_index.shape[1] == 0:
                per_timestep_auc[t] = 0.5
                continue
            logits = model(node_feats[edge_index[0]], node_feats[edge_index[1]])
            scores = torch.sigmoid(logits)
            per_timestep_auc[t] = compute_auc(edge_labels, scores)
            all_labels.append(edge_labels.cpu())
            all_scores.append(scores.cpu())

    if not all_labels:
        return 0.5, per_timestep_auc
    return compute_auc(torch.cat(all_labels), torch.cat(all_scores)), per_timestep_auc


# ---------------------------------------------------------------------------
# V3.5 comparison helpers + main benchmark
# ---------------------------------------------------------------------------


def load_v35_results() -> Optional[dict]:
    """Load V3.5 routing results for side-by-side comparison."""
    candidates = [
        _ROOT / "results" / "week8_v35_routing.json",
        _ROOT / "results" / "week8_v35_final.json",
        Path("results/week8_v35_routing.json"),
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
    """Extract v35_post from week8_v35_routing.json-style results."""
    if v35_data is None:
        return None

    entries = v35_data.get("results", v35_data)
    if isinstance(entries, list):
        for row in entries:
            if not isinstance(row, dict):
                continue
            if (
                row.get("noise_config") == noise_name
                and row.get("drift_scenario") == drift_name
            ):
                for key in ("v35_post", "mean_post", "auc", "test_auc"):
                    if key in row:
                        return float(row[key])
        return None

    if isinstance(entries, dict):
        if noise_name in entries and isinstance(entries[noise_name], dict):
            sub = entries[noise_name].get(drift_name)
            if isinstance(sub, dict):
                for key in ("v35_post", "mean_post", "auc", "test_auc"):
                    if key in sub:
                        return float(sub[key])
            elif isinstance(sub, (int, float)):
                return float(sub)
    return None


def run_benchmark(
    noise_filter: Optional[List[str]] = None,
    drift_filter: Optional[List[str]] = None,
    quick: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run actual PyG TGN baseline across noise × drift settings."""
    epochs = 40 if quick else NUM_EPOCHS
    noise_items = list(NOISE_CONFIGS.keys())
    drift_items = list(DRIFT_SCENARIOS.keys())
    if noise_filter:
        noise_items = [n for n in noise_items if n in set(noise_filter)]
    if drift_filter:
        drift_items = [d for d in drift_items if d in set(drift_filter)]

    print("=" * 80)
    print("TGN BASELINE BENCHMARK (PyG actual TGN, fair vs V3.5)")
    print(f"  Implementation: {IMPLEMENTATION_NAME}")
    print(f"  Device: {DEVICE}")
    print(
        f"  Timesteps: {NUM_TIMESTEPS} "
        f"(train 0-{int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO) - 1}, "
        f"test {int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO)}-{NUM_TIMESTEPS - 1})"
    )
    print(f"  Epochs: {epochs}, pos_weight: {POS_WEIGHT}, lr: {LR}")
    print(
        f"  Memory/Embed/Time dims: {MEMORY_DIM}/{EMBEDDING_DIM}/{TIME_DIM}, "
        f"neighbors={NEIGHBOR_SIZE}"
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
            node_feat_dim = int(timesteps[0]["x"].shape[1])

            print("  Training Static MLP baseline...")
            static_auc, static_per_t = train_and_evaluate_static(
                timesteps, node_feat_dim, num_epochs=epochs
            )
            print(f"    Static AUC: {static_auc:.4f}")

            print("  Training actual PyG TGN...")
            tgn_auc, tgn_per_t = train_and_evaluate_tgn(timesteps, num_epochs=epochs)
            print(f"    TGN AUC: {tgn_auc:.4f}")

            v35_auc = get_v35_post_auc(v35_data, noise_name, drift_name)
            if v35_auc is not None:
                print(f"    V3.5 post AUC: {v35_auc:.4f}")

            results[noise_name][drift_name] = {
                "static_auc": static_auc,
                "tgn_auc": tgn_auc,
                "v35_post": v35_auc,
            }
            all_per_timestep[noise_name][drift_name] = {
                "static": {str(k): v for k, v in static_per_t.items()},
                "tgn": {str(k): v for k, v in tgn_per_t.items()},
            }
            flat_rows.append(
                {
                    "noise_config": noise_name,
                    "drift_scenario": drift_name,
                    "static_auc": static_auc,
                    "tgn_auc": tgn_auc,
                    "v35_post": v35_auc,
                }
            )

            train_end = int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO)
            print(f"    Per-timestep AUC (test t={train_end}-{NUM_TIMESTEPS - 1}):")
            for t in range(train_end, NUM_TIMESTEPS):
                print(
                    f"      t={t}: Static={static_per_t.get(t, 0.5):.4f}  "
                    f"TGN={tgn_per_t.get(t, 0.5):.4f}"
                )

    print("\n" + "=" * 88)
    print("COMPARISON: Static MLP vs PyG TGN vs V3.5 post")
    print("=" * 88)
    header = (
        f"{'Noise':<12} {'Drift':<16} {'Static':>8} {'TGN':>8} "
        f"{'V3.5':>8} {'TGN-Static':>12} {'TGN-V3.5':>12}"
    )
    print(header)
    print("-" * len(header))

    static_aucs: List[float] = []
    tgn_aucs: List[float] = []
    v35_aucs: List[float] = []

    for row in flat_rows:
        static = row["static_auc"]
        tgn = row["tgn_auc"]
        v35 = row["v35_post"]
        static_aucs.append(static)
        tgn_aucs.append(tgn)
        v35_str = f"{v35:.4f}" if v35 is not None else "N/A"
        diff_v35 = f"{tgn - v35:+.4f}" if v35 is not None else "N/A"
        if v35 is not None:
            v35_aucs.append(v35)
        print(
            f"{row['noise_config']:<12} {row['drift_scenario']:<16} "
            f"{static:>8.4f} {tgn:>8.4f} {v35_str:>8} "
            f"{tgn - static:>+12.4f} {diff_v35:>12}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"  Static MLP mean: {np.mean(static_aucs):.4f} "
        f"(std {np.std(static_aucs):.4f})"
    )
    print(
        f"  PyG TGN mean:    {np.mean(tgn_aucs):.4f} "
        f"(std {np.std(tgn_aucs):.4f})"
    )
    if v35_aucs:
        print(
            f"  V3.5 post mean:  {np.mean(v35_aucs):.4f} "
            f"(std {np.std(v35_aucs):.4f})"
        )

    output = {
        "config": {
            "model": IMPLEMENTATION_NAME,
            "num_timesteps": NUM_TIMESTEPS,
            "train_timestep_ratio": TRAIN_TIMESTEP_RATIO,
            "pos_weight": POS_WEIGHT,
            "num_epochs": epochs,
            "memory_dim": MEMORY_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "time_dim": TIME_DIM,
            "neighbor_size": NEIGHBOR_SIZE,
            "lr": LR,
            "device": str(DEVICE),
            "note": (
                "Actual PyG TGN (TGNMemory + IdentityMessage + LastAggregator + "
                "LastNeighborLoader + TransformerConv). Predict-then-update. "
                "Static is feature MLP (not sheaf static)."
            ),
        },
        "results": results,
        "results_flat": flat_rows,
        "per_timestep_auc": all_per_timestep,
        "summary": {
            "static_mean_auc": float(np.mean(static_aucs)),
            "tgn_mean_auc": float(np.mean(tgn_aucs)),
            "v35_mean_post": float(np.mean(v35_aucs)) if v35_aucs else None,
        },
    }

    out_path = _ROOT / "results" / "tgn_baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 80)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Actual PyG TGN baseline on Week-7 / V3.5 fair protocol"
    )
    parser.add_argument("--quick", action="store_true", help="40 epochs")
    parser.add_argument("--noise", nargs="*", default=None)
    parser.add_argument("--drift", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(
        noise_filter=args.noise,
        drift_filter=args.drift,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

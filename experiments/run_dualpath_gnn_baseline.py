"""Euclidean dual-path GNN baseline (no sheaf, no SO(d)).

Answers the ICLR review question: do jump gains come from adaptive
routing between a recurrent path and a snapshot path, or from
sheaf / SO(d) geometry?

Architecture (mirrors V3.5 routing, Euclidean throughout):
  - Slow: GRU over graph-level embeddings (long memory)
  - Fast: history-free snapshot GNN
  - Gate: cosine-shift detector on an EMA of graph embeddings
  - Fuse: convex combination in R^h (not a geodesic on SO(d))
  - Head: MLP on concatenated endpoint embeddings + raw features

Modes isolate the routing hierarchy:
  gru   — slow path only (g=0)
  snap  — fast path only (g=1)
  dual  — gated fusion (the generic dual-path baseline)

Same Week-7 generator, temporal holdout, pos_weight, and split as
src/run, TGN, and EvolveGCN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    INPUT_DIM,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)

TRAIN_TIMESTEP_RATIO = 0.5
POS_WEIGHT = 9.0
GRAD_CLIP = 1.0
NUM_EPOCHS = 200
HIDDEN_DIM = 64
LR = 1e-3
GATE_EMA = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V35_POST = {
    ("clean", "slow_smooth"): 0.976138767361111,
    ("clean", "fast_smooth"): 0.9145309375,
    ("clean", "single_jump"): 0.8445506944444444,
    ("clean", "multi_jump"): 0.6718411805555555,
}


def _adapt_sheaf_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Map Week-7 snapshots to the shared train/eval schema."""
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
            }
        )
    return snapshots


def load_temporal_dataset(
    noise_name: str,
    drift_name: str,
    seed: int = 42,
    jump_mode: str = "additive",
) -> List[Dict[str, Tensor]]:
    """Same generator call as the V3.5 fair benchmark."""
    drift_params = dict(DRIFT_SCENARIOS[drift_name])
    drift_params["jump_mode"] = jump_mode
    raw = generate_temporal_dataset(
        drift_params,
        NOISE_CONFIGS[noise_name],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    return _adapt_sheaf_dataset(raw)


def relabel_affine_mismatch(
    snapshots: List[Dict[str, Tensor]],
    jump_t: int = 5,
    seed: int = 0,
) -> List[Dict[str, Tensor]]:
    """Replace sheaf labels with a GL(d) affine rule that jumps at ``jump_t``.

    Compliant edges satisfy A_t x_u ≈ x_v with A_t invertible but *not*
    constrained to SO(d). This is a mismatched data-generating process:
    the true operator is not a sheaf restriction map on SO(d).
    """
    gen = torch.Generator().manual_seed(seed)
    feat_dim = int(snapshots[0]["x"].shape[1])
    a_pre = torch.randn(feat_dim, feat_dim, generator=gen) + 2.0 * torch.eye(feat_dim)
    a_post = torch.randn(feat_dim, feat_dim, generator=gen) + 2.0 * torch.eye(feat_dim)
    out: List[Dict[str, Tensor]] = []
    for t, snap in enumerate(snapshots):
        a_t = a_pre if t < jump_t else a_post
        x = snap["x"]
        ei = snap["edge_index"]
        viol = ((x[ei[0]] @ a_t.T) - x[ei[1]]).norm(dim=1)
        thresh = torch.quantile(viol, 0.90)
        labels = (viol > thresh).float()
        new_snap = dict(snap)
        new_snap["labels"] = labels
        new_snap["clean_labels"] = labels
        out.append(new_snap)
    return out


def compute_auc(y: Tensor, scores: Tensor) -> float:
    """ROC AUC; 0.5 if a split has a single class."""
    y_np = y.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return 0.5
    return float(roc_auc_score(y_np, s_np))


class SnapshotEncoder(nn.Module):
    """Two-layer mean-aggregation GNN producing node and graph embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        h = F.relu(self.lin1(x))
        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        if src.numel() > 0:
            msg = h[src]
            agg = torch.zeros(num_nodes, h.size(1), device=h.device)
            counts = torch.zeros(num_nodes, 1, device=h.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            counts.scatter_add_(
                0, dst.unsqueeze(1), torch.ones(src.size(0), 1, device=h.device)
            )
            agg = agg / counts.clamp(min=1.0)
        else:
            agg = torch.zeros_like(h)
        h2 = F.relu(self.lin2(torch.cat([h, agg], dim=-1)))
        graph_emb = h2.mean(dim=0)
        return h2, graph_emb


class DualPathGNN(nn.Module):
    """GRU + snapshot GNN + Euclidean fusion. No sheaf, no Cayley."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        mode: str = "dual",
    ) -> None:
        super().__init__()
        if mode not in {"gru", "snap", "dual"}:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.encoder = SnapshotEncoder(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("ema_emb", torch.zeros(hidden_dim))
        self.register_buffer("ema_ready", torch.tensor(False))

    def reset_state(self) -> None:
        """Clear recurrent and EMA state between sequences."""
        self.ema_emb.zero_()
        self.ema_ready.fill_(False)
        self._hidden: Optional[Tensor] = None

    def _fusion_gate(self, graph_emb: Tensor) -> Tensor:
        if self.mode == "gru":
            return graph_emb.new_zeros(())
        if self.mode == "snap":
            return graph_emb.new_ones(())
        if not bool(self.ema_ready):
            self.ema_emb.copy_(graph_emb.detach())
            self.ema_ready.fill_(True)
            return graph_emb.new_zeros(())
        gate_in = torch.cat([self.ema_emb, graph_emb], dim=0)
        g = torch.sigmoid(self.gate(gate_in)).squeeze()
        self.ema_emb.mul_(GATE_EMA).add_((1.0 - GATE_EMA) * graph_emb.detach())
        return g

    def forward_timestep(
        self,
        x: Tensor,
        edge_index: Tensor,
        query: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Score query edges; update GRU/EMA state."""
        node_emb, graph_emb = self.encoder(x, edge_index)
        gru_in = graph_emb.view(1, 1, -1)
        if not hasattr(self, "_hidden") or self._hidden is None:
            gru_out, self._hidden = self.gru(gru_in)
        else:
            gru_out, self._hidden = self.gru(gru_in, self._hidden)
        slow = gru_out.view(-1)
        gate = self._fusion_gate(graph_emb)
        fused_graph = (1.0 - gate) * slow + gate * graph_emb
        # Broadcast a fused residual onto node embeddings.
        node_h = node_emb + fused_graph.unsqueeze(0)
        src, dst = query[0], query[1]
        logits = self.head(
            torch.cat([node_h[src], node_h[dst], x[src], x[dst]], dim=-1)
        ).squeeze(-1)
        return logits, gate


def train_and_evaluate(
    snapshots: List[Dict[str, Tensor]],
    mode: str,
    num_epochs: int,
    train_end: Optional[int] = None,
) -> Tuple[float, Dict[int, float], float]:
    """Train on t<train_end, evaluate post-split AUC on later snapshots."""
    in_dim = int(snapshots[0]["x"].shape[1])
    n_t = len(snapshots)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    model = DualPathGNN(in_dim, HIDDEN_DIM, mode=mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)

    model.train()
    for epoch in range(num_epochs):
        model.reset_state()
        optimizer.zero_grad()
        losses: List[Tensor] = []
        for t in range(train_end):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            if ei.shape[1] == 0:
                continue
            logits, _ = model.forward_timestep(x, ei, ei)
            losses.append(
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            )
        if not losses:
            return 0.5, {}, 0.0
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"      [{mode}] epoch {epoch+1:3d}/{num_epochs} loss={loss.item():.4f}")

    model.eval()
    model.reset_state()
    all_y: List[Tensor] = []
    all_s: List[Tensor] = []
    per_t: Dict[int, float] = {}
    gates: List[float] = []
    with torch.no_grad():
        for t in range(n_t):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            if ei.shape[1] == 0:
                if t >= train_end:
                    per_t[t] = 0.5
                continue
            logits, gate = model.forward_timestep(x, ei, ei)
            scores = torch.sigmoid(logits)
            gates.append(float(gate.detach().cpu()))
            per_t[t] = compute_auc(y, scores)
            if t < train_end:
                continue
            all_y.append(y.cpu())
            all_s.append(scores.cpu())
    post = compute_auc(torch.cat(all_y), torch.cat(all_s)) if all_y else 0.5
    mean_gate = float(sum(gates) / max(len(gates), 1))
    return post, per_t, mean_gate


def run_benchmark(
    noise_filter: Optional[List[str]],
    drift_filter: Optional[List[str]],
    modes: List[str],
    epochs: int,
    mismatch: bool,
    jump_mode: str = "additive",
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run requested noise/drift/mode grid."""
    noises = list(NOISE_CONFIGS.keys())
    drifts = list(DRIFT_SCENARIOS.keys())
    if noise_filter:
        noises = [n for n in noises if n in set(noise_filter)]
    if drift_filter:
        drifts = [d for d in drifts if d in set(drift_filter)]
    seed_list = seeds or [42]

    results: List[Dict[str, Any]] = []
    for noise in noises:
        for drift in drifts:
            for seed in seed_list:
                print(
                    f"\n=== {noise}/{drift} jump_mode={jump_mode} "
                    f"seed={seed} mismatch={mismatch} ==="
                )
                snaps = load_temporal_dataset(
                    noise, drift, seed=seed, jump_mode=jump_mode
                )
                if mismatch:
                    snaps = relabel_affine_mismatch(snaps, seed=seed)
                for mode in modes:
                    post, per_t, mean_gate = train_and_evaluate(
                        snaps, mode, epochs
                    )
                    v35 = V35_POST.get((noise, drift))
                    row = {
                        "noise": noise,
                        "drift": drift,
                        "jump_mode": jump_mode,
                        "seed": seed,
                        "mode": mode,
                        "mismatch": mismatch,
                        "post_auc": post,
                        "mean_gate": mean_gate,
                        "per_timestep_auc": {
                            str(k): v for k, v in per_t.items()
                        },
                        "v35_post": v35,
                    }
                    results.append(row)
                    print(
                        f"  {mode:4s} post-AUC={post:.4f}  "
                        f"gate={mean_gate:.3f}  v35={v35}"
                    )
    return {
        "config": {
            "epochs": epochs,
            "hidden_dim": HIDDEN_DIM,
            "modes": modes,
            "mismatch": mismatch,
            "jump_mode": jump_mode,
            "seeds": seed_list,
            "device": str(DEVICE),
            "note": (
                "Euclidean dual-path GNN: GRU + snapshot + cosine-shift "
                "gate; fusion in R^h; no sheaf / SO(d)."
            ),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", nargs="*", default=["clean"])
    parser.add_argument(
        "--drift",
        nargs="*",
        default=["slow_smooth", "fast_smooth", "single_jump", "multi_jump"],
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["gru", "snap", "dual"],
        help="gru | snap | dual",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument(
        "--jump-mode",
        type=str,
        default="additive",
        choices=["additive", "group"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument(
        "--out",
        type=str,
        default="results/dualpath_gnn_baseline_results.json",
    )
    args = parser.parse_args()
    payload = run_benchmark(
        args.noise,
        args.drift,
        args.modes,
        args.epochs,
        args.mismatch,
        jump_mode=args.jump_mode,
        seeds=args.seeds,
    )
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

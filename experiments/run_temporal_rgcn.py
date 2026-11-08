"""Relation-aware temporal control: per-relation snapshot aggregation + GRU.

This is not STHN (Li et al., CIKM 2023). STHN is a continuous-time
heterogeneous link predictor; adapting it to snapshot edge-anomaly scoring
would change the task. This script uses the DualPath routing skeleton with
typed message passing, i.e. the per-relation aggregation Theorem 2(i) does
not cover.

Protocol matches DualPath / Table 1: leak-free train t<0.5, post-drift AUROC,
clean noise, slow_smooth and single_jump.
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

from run_dualpath_gnn_baseline import (  # noqa: E402
    DEVICE,
    GATE_EMA,
    GRAD_CLIP,
    HIDDEN_DIM,
    LR,
    NUM_EPOCHS,
    POS_WEIGHT,
    TRAIN_TIMESTEP_RATIO,
    load_temporal_dataset,
)

IMPLEMENTATION_NAME = (
    "temporal R-GCN control: per-relation mean aggregation + GRU "
    "(not STHN; snapshot edge-anomaly protocol)"
)


def compute_auc(y: Tensor, scores: Tensor) -> float:
    y_np = y.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return 0.5
    return float(roc_auc_score(y_np, s_np))


class TypedSnapshotEncoder(nn.Module):
    """Two-layer GNN with a separate linear map per edge type."""

    def __init__(self, in_dim: int, hidden_dim: int, n_types: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.rel = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_types)]
        )
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_types: Tensor
    ) -> Tuple[Tensor, Tensor]:
        h = F.relu(self.lin1(x))
        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        counts = torch.zeros(num_nodes, 1, device=h.device)
        if src.numel() > 0:
            for r, layer in enumerate(self.rel):
                mask = edge_types == r
                if not bool(mask.any()):
                    continue
                msg = layer(h[src[mask]])
                dst_r = dst[mask].unsqueeze(1).expand_as(msg)
                agg.scatter_add_(0, dst_r, msg)
                counts.scatter_add_(
                    0,
                    dst[mask].unsqueeze(1),
                    torch.ones(int(mask.sum()), 1, device=h.device),
                )
            agg = agg / counts.clamp(min=1.0)
        h2 = F.relu(self.lin2(torch.cat([h, agg], dim=-1)))
        return h2, h2.mean(dim=0)


class TemporalRGCN(nn.Module):
    def __init__(
        self, in_dim: int, n_types: int, hidden_dim: int = HIDDEN_DIM
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TypedSnapshotEncoder(in_dim, hidden_dim, n_types)
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
        self.ema_emb.zero_()
        self.ema_ready.fill_(False)
        self._hidden: Optional[Tensor] = None

    def _fusion_gate(self, graph_emb: Tensor) -> Tensor:
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
        edge_types: Tensor,
        query: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        node_emb, graph_emb = self.encoder(x, edge_index, edge_types)
        gru_in = graph_emb.view(1, 1, -1)
        if not hasattr(self, "_hidden") or self._hidden is None:
            gru_out, self._hidden = self.gru(gru_in)
        else:
            gru_out, self._hidden = self.gru(gru_in, self._hidden)
        slow = gru_out.view(-1)
        gate = self._fusion_gate(graph_emb)
        fused_graph = (1.0 - gate) * slow + gate * graph_emb
        node_h = node_emb + fused_graph.unsqueeze(0)
        src, dst = query[0], query[1]
        logits = self.head(
            torch.cat([node_h[src], node_h[dst], x[src], x[dst]], dim=-1)
        ).squeeze(-1)
        return logits, gate


def train_and_evaluate(
    snapshots: List[Dict[str, Tensor]],
    num_epochs: int,
    train_end: Optional[int] = None,
) -> Tuple[float, Dict[int, float], float]:
    in_dim = int(snapshots[0]["x"].shape[1])
    n_types = int(snapshots[0]["edge_types"].max().item()) + 1
    n_t = len(snapshots)
    if train_end is None:
        train_end = int(n_t * TRAIN_TIMESTEP_RATIO)
    model = TemporalRGCN(in_dim, n_types).to(DEVICE)
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
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            if ei.shape[1] == 0:
                continue
            logits, _ = model.forward_timestep(x, ei, et, ei)
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
            print(f"      [rgcn] epoch {epoch+1:3d}/{num_epochs} loss={loss.item():.4f}")

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
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            if ei.shape[1] == 0:
                if t >= train_end:
                    per_t[t] = 0.5
                continue
            logits, gate = model.forward_timestep(x, ei, et, ei)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--drifts", nargs="*", default=["slow_smooth", "single_jump"]
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--out",
        type=str,
        default="results/temporal_rgcn.json",
    )
    args = parser.parse_args()

    cells: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for drift in args.drifts:
            print(f"=== seed={seed} drift={drift} ===")
            snaps = load_temporal_dataset("clean", drift, seed=seed)
            post, per_t, gate = train_and_evaluate(snaps, args.epochs)
            cell = {
                "seed": seed,
                "drift": drift,
                "model": "temporal_rgcn",
                "implementation": IMPLEMENTATION_NAME,
                "post_auc": post,
                "mean_gate": gate,
                "per_timestep_auc": {str(k): v for k, v in per_t.items()},
            }
            cells.append(cell)
            print(f"  post-AUC={post:.3f} gate={gate:.3f}")

    by_drift: Dict[str, List[float]] = {}
    for cell in cells:
        by_drift.setdefault(cell["drift"], []).append(cell["post_auc"])
    summary = {
        drift: {
            "mean": float(sum(v) / len(v)),
            "std": float(
                (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(len(v) - 1, 1))
                ** 0.5
            ),
            "n": len(v),
        }
        for drift, v in by_drift.items()
    }
    payload = {
        "implementation": IMPLEMENTATION_NAME,
        "not_sthn": True,
        "protocol": "leak-free t<0.5, clean noise, DualPath epochs/split",
        "cells": cells,
        "summary": summary,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

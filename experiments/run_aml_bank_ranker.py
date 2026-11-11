"""Bank-inspired synthetic AML ranker: DualPath vs residual-augmented DualPath.

Scenarios are abstract plants motivated by typed-ledger constraints
(FX conversion, settlement identity, competing rate). They are not
confidential bank records and are not independently validated FX or
settlement-exception tickets.

Leak-free: train t<0.5; unlabeled per-snapshot IT-GLLS residuals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from graph_generator import (  # noqa: E402
    EdgeTypeConfig,
    HeterogeneousGraphGenerator,
    NodeTypeConfig,
)
from run_dualpath_gnn_baseline import (  # noqa: E402
    DEVICE,
    DualPathGNN,
    GRAD_CLIP,
    HIDDEN_DIM,
    LR,
    NUM_EPOCHS,
    POS_WEIGHT,
    TRAIN_TIMESTEP_RATIO,
)
from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_itglls_contamination import givens, random_so  # noqa: E402
from run_jump_estimator_ablation import _score_maps  # noqa: E402
from run_sheaf_benchmark import (  # noqa: E402
    NUM_TIMESTEPS,
    _aggregate_node_features,
    _build_global_ids,
)
from run_sheaf_residual_ablation import attach_pair_residuals  # noqa: E402

DIM = 12
RAIL_FX = 0
RAIL_SETTLE = 2

SCENARIOS = [
    {
        "name": "fx_regime_jump",
        "purpose": "FX conversion scale jumps at the split; 10% independent anomalies",
        "kind": "fx_jump",
        "alpha": 0.10,
        "fx_scale_post": 1.4,
    },
    {
        "name": "settlement_exception",
        "purpose": "identity settlement rail; planted additive exceptions",
        "kind": "settlement",
        "alpha": 0.10,
        "break_norm": 2.5,
    },
    {
        "name": "competing_fx",
        "purpose": "wrong-rate contamination on the FX rail",
        "kind": "competing_fx",
        "alpha": 0.15,
        "theta_deg": 15.0,
    },
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics(y: Tensor, scores: Tensor) -> Dict[str, float]:
    y_np = y.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return {"auroc": 0.5, "auprc": float(y_np.mean())}
    return {
        "auroc": float(roc_auc_score(y_np, s_np)),
        "auprc": float(average_precision_score(y_np, s_np)),
    }


class DualPathResidual(DualPathGNN):
    """DualPath head with an optional concatenated pair residual."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        mode: str = "dual",
        residual_dim: int = 0,
    ) -> None:
        super().__init__(in_dim, hidden_dim, mode)
        self.residual_dim = int(residual_dim)
        extra = self.residual_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_dim * 2 + extra, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward_timestep(
        self,
        x: Tensor,
        edge_index: Tensor,
        query: Tensor,
        pair_residual: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        node_emb, graph_emb = self.encoder(x, edge_index)
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
        parts = [node_h[src], node_h[dst], x[src], x[dst]]
        if self.residual_dim > 0:
            n_e = int(src.shape[0])
            if pair_residual is None:
                residual = torch.zeros(
                    n_e, self.residual_dim, device=x.device, dtype=x.dtype
                )
            else:
                residual = pair_residual.to(device=x.device, dtype=x.dtype)
                if residual.dim() == 1:
                    residual = residual.unsqueeze(-1)
                residual = residual[:n_e, : self.residual_dim]
            parts.append(residual)
        logits = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return logits, gate


def _static_topology(
    seed: int,
    node_counts: Tuple[int, int, int],
    densities: Tuple[float, float, float],
    dim: int,
) -> Dict[str, Any]:
    n_account, n_entity, n_tx = node_counts
    node_configs = [
        NodeTypeConfig(name="Account", count=n_account, feature_dim=dim),
        NodeTypeConfig(name="Entity", count=n_entity, feature_dim=dim),
        NodeTypeConfig(name="Transaction", count=n_tx, feature_dim=dim),
    ]
    edge_configs = [
        EdgeTypeConfig(
            name="fx_wire",
            source_type="Account",
            target_type="Account",
            density=densities[0],
        ),
        EdgeTypeConfig(
            name="owns",
            source_type="Entity",
            target_type="Account",
            density=densities[1],
        ),
        EdgeTypeConfig(
            name="ach_settle",
            source_type="Account",
            target_type="Transaction",
            density=densities[2],
        ),
    ]
    graph_gen = HeterogeneousGraphGenerator(
        node_configs=node_configs, edge_configs=edge_configs, seed=seed
    )
    snapshot = graph_gen.generate()
    global_graph = _build_global_ids(snapshot, edge_configs)
    edge_types = list(global_graph["edge_types"])
    type_ids = torch.cat(
        [
            torch.full(
                (global_graph["edge_index"][etype].shape[1],),
                i,
                dtype=torch.long,
            )
            for i, etype in enumerate(edge_types)
        ],
        dim=0,
    )
    query = torch.stack([global_graph["src_ids"], global_graph["dst_ids"]])
    return {
        "num_nodes": global_graph["num_nodes"],
        "src_ids": global_graph["src_ids"],
        "dst_ids": global_graph["dst_ids"],
        "type_ids": type_ids,
        "query": query,
        "n_edges": int(query.shape[1]),
        "n_types": int(type_ids.max().item()) + 1,
    }


def _rail_maps(
    dim: int,
    t: float,
    scenario: Dict[str, Any],
    q_own: Tensor,
    q_fx: Tensor,
) -> List[Tensor]:
    """Typed maps: FX conversion, ownership rotation, settlement identity."""
    kind = scenario["kind"]
    maps = [torch.eye(dim) for _ in range(3)]
    maps[1] = q_own
    maps[0] = q_fx
    maps[2] = torch.eye(dim)
    if kind == "fx_jump":
        scale = 1.0 if t < 0.5 else float(scenario["fx_scale_post"])
        maps[0] = scale * q_fx
    return maps


def generate_bank_inspired_dataset(
    seed: int,
    scenario: Dict[str, Any],
    num_timesteps: int = NUM_TIMESTEPS,
    dim: int = DIM,
    noise: float = 0.1,
    node_counts: Tuple[int, int, int] = (500, 300, 500),
    densities: Tuple[float, float, float] = (0.01, 0.02, 0.01),
) -> List[Dict[str, Tensor]]:
    _set_seed(seed)
    topo = _static_topology(seed, node_counts, densities, dim)
    n_edges = topo["n_edges"]
    type_ids = topo["type_ids"]
    q_own = random_so(dim)
    q_fx = random_so(dim)
    g_alt = givens(dim, math.radians(float(scenario.get("theta_deg", 15.0))))
    times = np.linspace(0.0, 1.0, num_timesteps)
    alpha = float(scenario["alpha"])
    kind = scenario["kind"]
    dataset: List[Dict[str, Tensor]] = []
    for t_idx, t in enumerate(times):
        maps = _rail_maps(dim, float(t), scenario, q_own, q_fx)
        src = torch.randn(n_edges, dim)
        dst = torch.zeros_like(src)
        anom = torch.zeros(n_edges, dtype=torch.bool)
        for r in range(int(type_ids.max().item()) + 1):
            mask = type_ids == r
            n_r = int(mask.sum())
            if n_r == 0:
                continue
            dst[mask] = src[mask] @ maps[r].T + noise * torch.randn(n_r, dim)
            if kind == "competing_fx" and r == RAIL_FX:
                flag = torch.rand(n_r) < alpha
                n_a = int(flag.sum())
                if n_a:
                    alt = maps[r] @ g_alt
                    dst_r = dst[mask]
                    src_r = src[mask]
                    dst_r[flag] = src_r[flag] @ alt.T + noise * torch.randn(n_a, dim)
                    dst[mask] = dst_r
                    anom[mask] = flag
            elif kind == "settlement" and r == RAIL_SETTLE:
                flag = torch.rand(n_r) < alpha
                n_a = int(flag.sum())
                if n_a:
                    brk = torch.zeros(n_a, dim)
                    brk[:, 0] = float(scenario["break_norm"])
                    dst_r = dst[mask]
                    dst_r[flag] = dst_r[flag] + brk
                    dst[mask] = dst_r
                    anom[mask] = flag
            else:
                flag = torch.rand(n_r) < alpha
                n_a = int(flag.sum())
                if n_a:
                    dst_r = dst[mask]
                    dst_r[flag] = noise * 0 + 2.0 * torch.randn(n_a, dim)
                    dst[mask] = dst_r
                    anom[mask] = flag
        labels = anom.float()
        node_features = _aggregate_node_features(
            topo["num_nodes"],
            topo["src_ids"],
            topo["dst_ids"],
            src,
            dst,
        )
        dataset.append(
            {
                "x": node_features.float(),
                "edge_index": topo["query"].long(),
                "edge_types": type_ids.clone(),
                "labels": labels,
                "clean_labels": labels.clone(),
                "t": torch.tensor(t_idx),
                "t_normalized": torch.tensor(float(t)),
                "edge_src_feats": src.float(),
                "edge_tgt_feats": dst.float(),
            }
        )
    return dataset


def train_dualpath(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    num_epochs: int,
    residual_dim: int,
) -> Tuple[Dict[str, float], List[Tensor], List[Tensor]]:
    in_dim = int(snapshots[0]["x"].shape[1])
    model = DualPathResidual(
        in_dim, HIDDEN_DIM, mode="dual", residual_dim=residual_dim
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    model.train()
    for epoch in range(num_epochs):
        model.reset_state()
        opt.zero_grad()
        losses: List[Tensor] = []
        for t in range(train_end):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            residual = snapshots[t].get("pair_residual")
            if residual_dim > 0 and residual is not None:
                residual = residual.to(DEVICE)
            else:
                residual = None
            logits, _ = model.forward_timestep(x, ei, ei, pair_residual=residual)
            losses.append(
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            tag = "dual+res" if residual_dim else "dual"
            print(
                f"      [{tag}] epoch {epoch+1:3d}/{num_epochs} loss={loss.item():.4f}",
                flush=True,
            )
    model.eval()
    model.reset_state()
    logits_all: List[Tensor] = []
    ys: List[Tensor] = []
    ss: List[Tensor] = []
    with torch.no_grad():
        for t, snap in enumerate(snapshots):
            x = snap["x"].to(DEVICE)
            ei = snap["edge_index"].to(DEVICE)
            residual = snap.get("pair_residual")
            if residual_dim > 0 and residual is not None:
                residual = residual.to(DEVICE)
            else:
                residual = None
            logits, _ = model.forward_timestep(x, ei, ei, pair_residual=residual)
            logits_all.append(logits.cpu())
            if t < train_end:
                continue
            ys.append(snap["labels"].cpu())
            ss.append(torch.sigmoid(logits).cpu())
    return _metrics(torch.cat(ys), torch.cat(ss)), logits_all, ys


def late_fusion(
    snapshots: List[Dict[str, Tensor]],
    logits_all: List[Tensor],
    train_end: int,
) -> Dict[str, float]:
    x_tr = torch.cat(
        [
            torch.stack(
                [logits_all[t], snapshots[t]["pair_residual"]], dim=1
            )
            for t in range(train_end)
        ],
        dim=0,
    ).numpy()
    y_tr = torch.cat([snapshots[t]["labels"] for t in range(train_end)]).numpy()
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    x_te = torch.cat(
        [
            torch.stack(
                [logits_all[t], snapshots[t]["pair_residual"]], dim=1
            )
            for t in range(train_end, len(snapshots))
        ],
        dim=0,
    )
    y_te = torch.cat(
        [snapshots[t]["labels"] for t in range(train_end, len(snapshots))]
    )
    scores = torch.tensor(clf.predict_proba(x_te.numpy())[:, 1])
    return _metrics(y_te, scores)


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def summarize(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    names = sorted({c["scenario"] for c in cells})
    keys = [
        "residual_alone",
        "dualpath",
        "dualpath_residual",
        "late_fusion",
    ]
    for name in names:
        subset = [c for c in cells if c["scenario"] == name]
        block: Dict[str, Any] = {}
        for key in keys:
            auroc = [c[key]["auroc"] for c in subset if key in c]
            auprc = [c[key]["auprc"] for c in subset if key in c]
            block[key] = {"auroc": mean_std(auroc), "auprc": mean_std(auprc)}
        deltas = [
            c["dualpath_residual"]["auroc"] - c["dualpath"]["auroc"]
            for c in subset
            if "dualpath_residual" in c and "dualpath" in c
        ]
        block["dualpath_residual_minus_dualpath"] = mean_std(deltas)
        out[name] = block
    return out


def run_cell(
    seed: int,
    scenario: Dict[str, Any],
    epochs: int,
    node_counts: Tuple[int, int, int],
    densities: Tuple[float, float, float],
) -> Dict[str, Any]:
    snaps = generate_bank_inspired_dataset(
        seed=seed,
        scenario=scenario,
        node_counts=node_counts,
        densities=densities,
    )
    train_end = int(len(snaps) * TRAIN_TIMESTEP_RATIO)
    residual_diag = attach_pair_residuals(snaps, train_end)
    cell: Dict[str, Any] = {
        "seed": seed,
        "scenario": scenario["name"],
        "kind": scenario["kind"],
        "residual_alone": residual_diag["residual_only"],
    }
    print(
        f"  residual-alone AUROC={cell['residual_alone']['auroc']:.3f} "
        f"AUPRC={cell['residual_alone']['auprc']:.3f}",
        flush=True,
    )
    dual, logits_all, _ = train_dualpath(
        snaps, train_end, epochs, residual_dim=0
    )
    cell["dualpath"] = dual
    print(
        f"  dualpath AUROC={dual['auroc']:.3f} AUPRC={dual['auprc']:.3f}",
        flush=True,
    )
    dual_res, _, _ = train_dualpath(
        snaps, train_end, epochs, residual_dim=1
    )
    cell["dualpath_residual"] = dual_res
    print(
        f"  dual+res AUROC={dual_res['auroc']:.3f} AUPRC={dual_res['auprc']:.3f}",
        flush=True,
    )
    fusion = late_fusion(snaps, logits_all, train_end)
    cell["late_fusion"] = fusion
    print(
        f"  late-fusion AUROC={fusion['auroc']:.3f} AUPRC={fusion['auprc']:.3f}",
        flush=True,
    )
    return cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[s["name"] for s in SCENARIOS],
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default="results/aml_bank_ranker.json")
    args = parser.parse_args()
    selected = [s for s in SCENARIOS if s["name"] in args.scenarios]
    node_counts = (80, 40, 80) if args.quick else (500, 300, 500)
    densities = (0.04, 0.06, 0.04) if args.quick else (0.01, 0.02, 0.01)
    epochs = 4 if args.quick else args.epochs
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    payload: Dict[str, Any] = {
        "protocol": (
            "Bank-inspired synthetic plants (FX conversion jump, settlement "
            "exception, competing FX rate). Not confidential records. "
            "Unlabeled per-snapshot IT-GLLS; DualPath leak-free t<0.5. "
            "Late fusion is logistic regression on DualPath logits and the "
            "pair residual, fit on train snapshots only."
        ),
        "scenarios_precommitted": selected,
        "seeds": args.seeds,
        "epochs": epochs,
        "dim": DIM,
        "node_counts": list(node_counts),
        "densities": list(densities),
        "quick": args.quick,
        "device": str(DEVICE),
        "cells": [],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for seed in args.seeds:
        for scenario in selected:
            print(
                f"\n=== seed={seed} scenario={scenario['name']} ===",
                flush=True,
            )
            cell = run_cell(
                seed=seed,
                scenario=scenario,
                epochs=epochs,
                node_counts=node_counts,
                densities=densities,
            )
            payload["cells"].append(cell)
            payload["summary"] = summarize(payload["cells"])
            out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

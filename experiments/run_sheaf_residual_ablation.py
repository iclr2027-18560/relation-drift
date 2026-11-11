"""Controlled ablation: does the temporal sheaf add detection value beyond IT-GLLS?

Precommitted conditions (chosen before running; not selected post hoc):
  A  competing map, 15°, 15% contamination, stationary
  B  competing map, 15°, 25% contamination, stationary
  C  competing map, 15°, 15% contamination, 45° smooth drift of R★

Five models, identical snapshots, identical IT-GLLS residuals, five seeds.
Residuals are unlabeled per-snapshot fits (no labels, no future snapshots).
Neural weights are trained on t<0.5 only and frozen at test; recurrent state
may update from test features. Concatenating a residual does not refine the
IT-GLLS map; this experiment is detection only.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "fix-35-final"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from graph_generator import (  # noqa: E402
    EdgeTypeConfig,
    HeterogeneousGraphGenerator,
    NodeTypeConfig,
)
from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_jump_estimator_ablation import _score_maps  # noqa: E402
from run_itglls_contamination import givens, random_so  # noqa: E402
from run_jump_estimator_ablation import leak_free_train_config  # noqa: E402
from run_sheaf_benchmark import (  # noqa: E402
    INPUT_DIM,
    NUM_TIMESTEPS,
    _aggregate_node_features,
    _build_global_ids,
)
from run_temporal_rgcn import (  # noqa: E402
    DEVICE,
    GRAD_CLIP,
    HIDDEN_DIM,
    LR,
    NUM_EPOCHS,
    POS_WEIGHT,
    TRAIN_TIMESTEP_RATIO,
    TemporalRGCN,
)
from run_v35_benchmark import BenchmarkConfig, train_static_sheaf  # noqa: E402
from temporal_sheaf_v35 import SheafConfig, TemporalSheafV35  # noqa: E402
from train_v35 import train_temporal_v35  # noqa: E402

CONDITIONS = [
    {
        "name": "A",
        "theta_deg": 15.0,
        "alpha": 0.15,
        "drift_total_deg": 0.0,
        "purpose": "stationary; residual informative but not saturated",
    },
    {
        "name": "B",
        "theta_deg": 15.0,
        "alpha": 0.25,
        "drift_total_deg": 0.0,
        "purpose": "higher contamination",
    },
    {
        "name": "C",
        "theta_deg": 15.0,
        "alpha": 0.15,
        "drift_total_deg": 45.0,
        "purpose": "smooth drift; temporal memory has a chance to help",
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


def generate_competing_map_dataset(
    seed: int,
    alpha: float,
    theta_deg: float,
    drift_total_deg: float,
    num_timesteps: int = NUM_TIMESTEPS,
    dim: int = INPUT_DIM,
    noise: float = 0.1,
    node_counts: Tuple[int, int, int] = (500, 300, 500),
    densities: Tuple[float, float, float] = (0.01, 0.02, 0.01),
) -> List[Dict[str, Tensor]]:
    """Graph snapshots with the competing-map pair law on a static topology.

    Independent per-snapshot contamination (no fixed edge identity).
    Compliant: x_v = R★_t x_u + ξ. Contaminated: x_v = R_alt_t x_u + ξ
    with R_alt_t = R★_t Givens(θ). If drift_total_deg>0,
    R★_t = R★_0 Givens(drift_total_deg * t).
    """
    _set_seed(seed)
    n_account, n_entity, n_tx = node_counts
    node_configs = [
        NodeTypeConfig(name="Account", count=n_account, feature_dim=dim),
        NodeTypeConfig(name="Entity", count=n_entity, feature_dim=dim),
        NodeTypeConfig(name="Transaction", count=n_tx, feature_dim=dim),
    ]
    edge_configs = [
        EdgeTypeConfig(
            name="sends_to",
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
            name="transfers",
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
    n_edges = int(global_graph["src_ids"].shape[0])
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
    r0 = random_so(dim)
    g_alt = givens(dim, math.radians(theta_deg))
    times = np.linspace(0.0, 1.0, num_timesteps)
    dataset: List[Dict[str, Tensor]] = []
    for t_idx, t in enumerate(times):
        r_star = r0 @ givens(dim, math.radians(drift_total_deg * float(t)))
        r_alt = r_star @ g_alt
        src = torch.randn(n_edges, dim)
        anom = torch.rand(n_edges) < alpha
        dst = src @ r_star.T + noise * torch.randn(n_edges, dim)
        dst[anom] = src[anom] @ r_alt.T + noise * torch.randn(int(anom.sum()), dim)
        labels = anom.float()
        node_features = _aggregate_node_features(
            global_graph["num_nodes"],
            global_graph["src_ids"],
            global_graph["dst_ids"],
            src,
            dst,
        )
        dataset.append(
            {
                "x": node_features.float(),
                "edge_index": query.long(),
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


def attach_pair_residuals(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
) -> Dict[str, Any]:
    """Unlabeled IT-GLLS residual per snapshot; no labels and no future snaps."""
    only_y: List[Tensor] = []
    only_s: List[Tensor] = []
    per_t: Dict[str, Dict[str, float]] = {}
    for t, snap in enumerate(snapshots):
        src = snap["edge_src_feats"].float()
        dst = snap["edge_tgt_feats"].float()
        types = snap["edge_types"].long()
        maps = it_glls_maps(src, dst, types)
        residual = _score_maps(src, dst, maps, types)
        snap["pair_residual"] = torch.log1p(residual).detach()
        m = _metrics(snap["labels"].float(), residual)
        per_t[f"t_{t}"] = m
        if t >= train_end:
            only_y.append(snap["labels"].float())
            only_s.append(residual)
    post = _metrics(torch.cat(only_y), torch.cat(only_s))
    return {"residual_only": post, "residual_only_per_timestep": per_t}


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, residual: Tensor) -> Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(-1)
        return self.net(residual).squeeze(-1)


def train_residual_mlp(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    num_epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    model = ResidualMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    train_r = torch.cat(
        [snapshots[t]["pair_residual"] for t in range(train_end)]
    ).to(device)
    train_y = torch.cat([snapshots[t]["labels"] for t in range(train_end)]).to(
        device
    )
    model.train()
    for epoch in range(num_epochs):
        opt.zero_grad()
        logits = model(train_r)
        loss = F.binary_cross_entropy_with_logits(
            logits, train_y, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"      [mlp] epoch {epoch+1:3d}/{num_epochs} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y = torch.cat(
            [snapshots[t]["labels"] for t in range(train_end, len(snapshots))]
        )
        scores = torch.sigmoid(
            model(
                torch.cat(
                    [
                        snapshots[t]["pair_residual"]
                        for t in range(train_end, len(snapshots))
                    ]
                ).to(device)
            )
        ).cpu()
    return _metrics(y, scores)


def train_rgcn_residual(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    num_epochs: int,
    residual_dim: int,
) -> Dict[str, float]:
    in_dim = int(snapshots[0]["x"].shape[1])
    n_types = int(snapshots[0]["edge_types"].max().item()) + 1
    model = TemporalRGCN(in_dim, n_types, residual_dim=residual_dim).to(DEVICE)
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
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            residual = snapshots[t].get("pair_residual")
            if residual is not None:
                residual = residual.to(DEVICE)
            logits, _ = model.forward_timestep(
                x, ei, et, ei, pair_residual=residual
            )
            losses.append(
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"      [rgcn] epoch {epoch+1:3d}/{num_epochs} loss={loss.item():.4f}")
    model.eval()
    model.reset_state()
    ys: List[Tensor] = []
    ss: List[Tensor] = []
    with torch.no_grad():
        for t in range(len(snapshots)):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            residual = snapshots[t].get("pair_residual")
            if residual is not None:
                residual = residual.to(DEVICE)
            logits, _ = model.forward_timestep(
                x, ei, et, ei, pair_residual=residual
            )
            if t < train_end:
                continue
            ys.append(y.cpu())
            ss.append(torch.sigmoid(logits).cpu())
    return _metrics(torch.cat(ys), torch.cat(ss))


def eval_sheaf_post(
    model: TemporalSheafV35,
    snapshots: List[Dict[str, Tensor]],
    device: str,
    train_end: int,
    use_residual: bool,
) -> Dict[str, float]:
    model.eval()
    model.reset_state()
    ys: List[Tensor] = []
    ss: List[Tensor] = []
    with torch.no_grad():
        for t, snap in enumerate(snapshots):
            x = snap["x"].to(device)
            edge_index = snap["edge_index"].to(device)
            edge_types = snap["edge_types"].to(device)
            pair_residual = None
            if use_residual:
                pair_residual = snap["pair_residual"].to(device)
            logits, _ = model.forward_timestep(
                x,
                edge_index,
                edge_types,
                pair_residual=pair_residual,
            )
            if t < train_end:
                continue
            ys.append(snap["labels"].cpu())
            ss.append(torch.sigmoid(logits).cpu())
    return _metrics(torch.cat(ys), torch.cat(ss))


def train_sheaf(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    device: str,
    use_residual: bool,
    quick: bool,
    static_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    config = BenchmarkConfig(device=device)
    config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
    config.node_feat_dim = int(snapshots[0]["x"].shape[1])
    train_ts = list(range(train_end))
    if static_model is None:
        static_epochs = 8 if quick else 200
        static_model = train_static_sheaf(
            snapshots, config, train_ts, num_epochs=static_epochs
        )
    sheaf_config = SheafConfig(
        node_feat_dim=config.node_feat_dim,
        num_edge_types=config.num_edge_types,
        use_fast_encoder=True,
        pair_residual_dim=1 if use_residual else 0,
    )
    train_cfg = leak_free_train_config()
    if quick:
        train_cfg.epochs_phase1 = 2
        train_cfg.epochs_phase3 = 2
    model = TemporalSheafV35(sheaf_config).to(device)
    model = train_temporal_v35(
        model=model,
        static_model=static_model,
        dataset=[snapshots],
        config=train_cfg,
        sheaf_config=sheaf_config,
        train_timesteps=train_ts,
        jump_times=[],
    )
    return eval_sheaf_post(
        model, snapshots, device, train_end, use_residual=use_residual
    )


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def summarize(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = [
        "residual_alone",
        "residual_mlp",
        "rgcn_residual",
        "sheaf_residual",
        "sheaf_only",
    ]
    out: List[Dict[str, Any]] = []
    names = sorted({c["condition"] for c in cells})
    for name in names:
        rows = [c for c in cells if c["condition"] == name]
        entry: Dict[str, Any] = {
            "condition": name,
            "n_seeds": len(rows),
            "theta_deg": rows[0]["theta_deg"],
            "alpha": rows[0]["alpha"],
            "drift_total_deg": rows[0]["drift_total_deg"],
        }
        for model in models:
            for metric in ("auroc", "auprc"):
                vals = [r[model][metric] for r in rows if model in r]
                entry[f"{model}_{metric}"] = mean_std(vals)
        paired = [
            ("sheaf_residual_minus_alone", "sheaf_residual", "residual_alone"),
            ("sheaf_residual_minus_mlp", "sheaf_residual", "residual_mlp"),
            ("sheaf_residual_minus_rgcn", "sheaf_residual", "rgcn_residual"),
        ]
        for key, left, right in paired:
            for metric in ("auroc", "auprc"):
                diffs = [
                    r[left][metric] - r[right][metric]
                    for r in rows
                    if left in r and right in r
                ]
                entry[f"{key}_{metric}"] = mean_std(diffs)
        out.append(entry)
    return out


def run_cell(
    seed: int,
    condition: Dict[str, Any],
    device: str,
    epochs: int,
    quick: bool,
    models: List[str],
    node_counts: Tuple[int, int, int],
    densities: Tuple[float, float, float],
) -> Dict[str, Any]:
    _set_seed(seed)
    snaps = generate_competing_map_dataset(
        seed=seed,
        alpha=float(condition["alpha"]),
        theta_deg=float(condition["theta_deg"]),
        drift_total_deg=float(condition["drift_total_deg"]),
        node_counts=node_counts,
        densities=densities,
    )
    train_end = int(len(snaps) * TRAIN_TIMESTEP_RATIO)
    residual_diag = attach_pair_residuals(snaps, train_end)
    cell: Dict[str, Any] = {
        "seed": seed,
        "condition": condition["name"],
        "theta_deg": condition["theta_deg"],
        "alpha": condition["alpha"],
        "drift_total_deg": condition["drift_total_deg"],
        "residual_alone": residual_diag["residual_only"],
    }
    print(
        f"  residual-alone AUROC={cell['residual_alone']['auroc']:.3f} "
        f"AUPRC={cell['residual_alone']['auprc']:.3f}"
    )
    if "residual_mlp" in models:
        cell["residual_mlp"] = train_residual_mlp(
            snaps, train_end, epochs, torch.device(device)
        )
        print(
            f"  residual-mlp AUROC={cell['residual_mlp']['auroc']:.3f} "
            f"AUPRC={cell['residual_mlp']['auprc']:.3f}"
        )
    if "rgcn_residual" in models:
        cell["rgcn_residual"] = train_rgcn_residual(
            snaps, train_end, epochs, residual_dim=1
        )
        print(
            f"  rgcn+res AUROC={cell['rgcn_residual']['auroc']:.3f} "
            f"AUPRC={cell['rgcn_residual']['auprc']:.3f}"
        )
    static_model = None
    need_sheaf = "sheaf_residual" in models or "sheaf_only" in models
    if need_sheaf:
        config = BenchmarkConfig(device=device)
        config.num_edge_types = int(snaps[0]["edge_types"].max().item()) + 1
        config.node_feat_dim = int(snaps[0]["x"].shape[1])
        static_epochs = 8 if quick else 200
        static_model = train_static_sheaf(
            snaps, config, list(range(train_end)), num_epochs=static_epochs
        )
    if "sheaf_residual" in models:
        cell["sheaf_residual"] = train_sheaf(
            snaps,
            train_end,
            device,
            use_residual=True,
            quick=quick,
            static_model=static_model,
        )
        print(
            f"  sheaf+res AUROC={cell['sheaf_residual']['auroc']:.3f} "
            f"AUPRC={cell['sheaf_residual']['auprc']:.3f}"
        )
    if "sheaf_only" in models:
        cell["sheaf_only"] = train_sheaf(
            snaps,
            train_end,
            device,
            use_residual=False,
            quick=quick,
            static_model=static_model,
        )
        print(
            f"  sheaf-only AUROC={cell['sheaf_only']['auroc']:.3f} "
            f"AUPRC={cell['sheaf_only']['auprc']:.3f}"
        )
    return cell


def write_payload(
    path: Path,
    cells: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> None:
    payload = {
        **extra,
        "cells": cells,
        "summary": summarize(cells),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["A", "B", "C"],
        help="Precommitted condition names.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "residual_mlp",
            "rgcn_residual",
            "sheaf_residual",
            "sheaf_only",
        ],
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", type=str, default=str(DEVICE))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/sheaf_itglls_ablation.json",
    )
    args = parser.parse_args()
    cond_by_name = {c["name"]: c for c in CONDITIONS}
    selected = [cond_by_name[n] for n in args.conditions]
    node_counts = (80, 40, 80) if args.quick else (500, 300, 500)
    densities = (0.04, 0.06, 0.04) if args.quick else (0.01, 0.02, 0.01)
    epochs = 4 if args.quick else args.epochs
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    protocol = {
        "protocol": (
            "Competing-map graph DGP. Unlabeled per-snapshot IT-GLLS residuals "
            "(no labels, no future). Neural weights trained t<0.5 and frozen; "
            "recurrent state may update from test features. Same residual "
            "values for every residual-augmented model. Detection only: "
            "concatenating the residual does not refine the IT-GLLS map."
        ),
        "conditions_precommitted": selected,
        "seeds": args.seeds,
        "epochs_mlp_rgcn": epochs,
        "sheaf_train": "leak-free P1=80 P3=160, static=200 (2/2/8 if --quick)",
        "pos_weight": POS_WEIGHT,
        "hidden_dim_rgcn": HIDDEN_DIM,
        "node_counts": list(node_counts),
        "densities": list(densities),
        "quick": args.quick,
        "device": args.device,
        "label_trained_models": [
            "residual_mlp",
            "rgcn_residual",
            "sheaf_residual",
            "sheaf_only",
        ],
        "unsupervised_models": ["residual_alone"],
    }
    cells: List[Dict[str, Any]] = []
    write_payload(out, cells, protocol)
    for seed in args.seeds:
        for condition in selected:
            print(
                f"\n=== seed={seed} condition={condition['name']} "
                f"θ={condition['theta_deg']} α={condition['alpha']} "
                f"drift={condition['drift_total_deg']} ==="
            )
            cell = run_cell(
                seed=seed,
                condition=condition,
                device=args.device,
                epochs=epochs,
                quick=args.quick,
                models=list(args.models),
                node_counts=node_counts,
                densities=densities,
            )
            cells.append(cell)
            write_payload(out, cells, protocol)
            print(f"  wrote {out}")
    print(json.dumps(summarize(cells), indent=2))


if __name__ == "__main__":
    main()

"""2x2 connection ablation: static vs time-varying, orthogonal vs unconstrained.

Fusion off for all four variants. No pair residual. Leak-free t<0.5.
Static means one learned map, identical at every snapshot (not Table 1 Static).
Unconstrained uses GL energy ||s_u ρ - s_v||, not the SO(d) coboundary ρᵀ.
Time-varying unconstrained: ρ_t = ρ_{t-1}(I + g A_t), A_t from the GRU.

Does not report connection-to-R* error. Pre-registered cells: no_drift,
slow_smooth, fast_smooth. Seeds 42-46 share the same generated graphs
across variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX35 = _SRC / "fix-35-final"
for _p in (_SRC, _FIX35, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)
from run_v35_benchmark import (  # noqa: E402
    BenchmarkConfig,
    compute_auc,
    get_temporal_split,
    _adapt_sheaf_dataset,
)
from temporal_sheaf_v35 import SheafConfig, TemporalSheafV35  # noqa: E402
from train_v35 import compute_orthogonality_penalty, compute_smoothness_penalty  # noqa: E402

VARIANTS = {
    "A_static_orthogonal": {
        "static_connection": True,
        "connection_family": "orthogonal",
    },
    "B_temporal_orthogonal": {
        "static_connection": False,
        "connection_family": "orthogonal",
    },
    "C_static_unconstrained": {
        "static_connection": True,
        "connection_family": "unconstrained",
    },
    "D_temporal_unconstrained": {
        "static_connection": False,
        "connection_family": "unconstrained",
    },
}

DRIFT_PARAMS = {
    "no_drift": {"omega": 0.0, "jumps": [], "magnitudes": []},
    "slow_smooth": dict(DRIFT_SCENARIOS["slow_smooth"]),
    "fast_smooth": dict(DRIFT_SCENARIOS["fast_smooth"]),
}

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_DRIFTS = ["no_drift", "slow_smooth", "fast_smooth"]
POS_WEIGHT = 9.0
GRAD_CLIP = 1.0
LR = 1e-3
ORTHO_PENALTY = 0.01
SMOOTH_PENALTY = 0.005


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_snapshots(drift: str, seed: int) -> List[Dict[str, Tensor]]:
    params = dict(DRIFT_PARAMS[drift])
    params["jump_mode"] = "additive"
    raw = generate_temporal_dataset(
        params,
        NOISE_CONFIGS["clean"],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    return _adapt_sheaf_dataset(raw)


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _rho_diagnostics(rho: Tensor) -> Dict[str, float]:
    mat = rho[0].detach()
    eye = torch.eye(mat.size(0), device=mat.device, dtype=mat.dtype)
    ortho = torch.linalg.norm(mat.T @ mat - eye, ord="fro")
    sv = torch.linalg.svdvals(mat)
    return {
        "ortho_fro": float(ortho.item()),
        "fro_norm": float(torch.linalg.norm(mat, ord="fro").item()),
        "sv_min": float(sv.min().item()),
        "sv_max": float(sv.max().item()),
        "cond": float((sv.max() / sv.clamp_min(1e-8).min()).item()),
    }


def _auc_pr(labels: Tensor, scores: Tensor) -> Tuple[float, float]:
    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    if y.min() == y.max():
        return 0.5, float(y.mean())
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def build_model(variant: str, node_feat_dim: int, num_edge_types: int) -> TemporalSheafV35:
    spec = VARIANTS[variant]
    cfg = SheafConfig(
        node_feat_dim=node_feat_dim,
        num_edge_types=num_edge_types,
        use_fast_encoder=False,
        pair_residual_dim=0,
        connection_family=spec["connection_family"],
        static_connection=spec["static_connection"],
    )
    return TemporalSheafV35(cfg)


def train_variant(
    model: TemporalSheafV35,
    snapshots: List[Dict[str, Tensor]],
    train_ts: List[int],
    device: str,
    epochs: int,
    family: str,
    static: bool,
) -> Dict[str, float]:
    model.to(device)
    pos_weight = torch.tensor(POS_WEIGHT, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    last_loss = 0.0
    last_train_auc = 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        model.reset_state()
        epoch_loss = 0.0
        n_batch = 0
        rho_prev: Optional[Tensor] = None
        train_scores: List[Tensor] = []
        train_labels: List[Tensor] = []
        for t in train_ts:
            snap = snapshots[t]
            scores, _ = model.forward_timestep(
                snap["x"].to(device),
                snap["edge_index"].to(device),
                snap["edge_types"].to(device),
            )
            labels = snap["labels"].to(device)
            loss = loss_fn(scores, labels)
            rho_now = model.get_current_rho()
            if family == "orthogonal":
                loss = loss + ORTHO_PENALTY * compute_orthogonality_penalty(rho_now)
            if (not static) and SMOOTH_PENALTY > 0:
                loss = loss + SMOOTH_PENALTY * compute_smoothness_penalty(
                    rho_now, rho_prev
                )
            rho_prev = rho_now.detach().clone()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batch += 1
            train_scores.append(scores.detach())
            train_labels.append(labels.detach())
        last_loss = epoch_loss / max(n_batch, 1)
        last_train_auc = compute_auc(
            torch.cat(train_scores), torch.cat(train_labels)
        )
        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:3d}/{epochs} loss={last_loss:.4f} "
                f"train_auc={last_train_auc:.3f}"
            )
    return {"train_loss": last_loss, "train_auc": last_train_auc}


@torch.no_grad()
def evaluate_variant(
    model: TemporalSheafV35,
    snapshots: List[Dict[str, Tensor]],
    timesteps: List[int],
    device: str,
    static: bool,
) -> Dict[str, Any]:
    model.eval()
    model.reset_state()
    scores_all: List[Tensor] = []
    labels_all: List[Tensor] = []
    diags: List[Dict[str, float]] = []
    rhos: List[Tensor] = []
    prev_rho: Optional[Tensor] = None
    for t in range(len(snapshots)):
        snap = snapshots[t]
        scores, _ = model.forward_timestep(
            snap["x"].to(device),
            snap["edge_index"].to(device),
            snap["edge_types"].to(device),
        )
        rho = model.get_current_rho().detach().cpu()
        d = _rho_diagnostics(rho)
        if prev_rho is None:
            d["delta_fro"] = 0.0
        else:
            d["delta_fro"] = float(
                torch.linalg.norm(rho[0] - prev_rho[0], ord="fro").item()
            )
        prev_rho = rho.clone()
        rhos.append(rho[0].clone())
        if t in timesteps:
            scores_all.append(torch.sigmoid(scores).cpu())
            labels_all.append(snap["labels"].cpu())
            diags.append(d)
    y = torch.cat(labels_all)
    s = torch.cat(scores_all)
    auroc, auprc = _auc_pr(y, s)
    static_ok = True
    if static and rhos:
        stacked = torch.stack(rhos)
        static_ok = bool(
            (stacked - stacked[:1]).abs().max().item() < 1e-5
        )
    mean_diag = {
        k: float(np.mean([d[k] for d in diags])) for k in diags[0]
    }
    return {
        "auroc": auroc,
        "auprc": auprc,
        "diagnostics": mean_diag,
        "static_rho_constant": static_ok,
    }


def _cell_key(seed: int, drift: str, variant: str) -> str:
    return f"{seed}/{drift}/{variant}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--drifts", nargs="*", default=DEFAULT_DRIFTS)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "results" / "connection_ablation.json"),
    )
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 4)
        if args.seeds == DEFAULT_SEEDS:
            args.seeds = [42]
        if args.drifts == DEFAULT_DRIFTS:
            args.drifts = ["slow_smooth"]

    out_path = Path(args.out)
    payload: Dict[str, Any]
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    else:
        payload = {
            "protocol": {
                "fusion": "off",
                "residual": "off",
                "split": "t<0.5 train, t>=0.5 test",
                "static": "one learned map, identical across snapshots",
                "orthogonal_energy": "s_u rho - s_v rho^T",
                "unconstrained_energy": "s_u rho - s_v",
                "unconstrained_update": "rho <- rho (I + g A)",
                "ortho_penalty": "orthogonal variants only",
                "note": (
                    "B is slow-path Cayley, not Table 1 dual-path sheaf. "
                    "Table 1 Static resets rho to I each snapshot; A does not."
                ),
            },
            "cells": [],
        }
    done = {_cell_key(c["seed"], c["drift"], c["variant"]) for c in payload["cells"]}
    device = _device()
    print(f"device={device} epochs={args.epochs}")

    for seed in args.seeds:
        for drift in args.drifts:
            snaps = load_snapshots(drift, seed)
            train_ts, test_ts = get_temporal_split(len(snaps))
            bcfg = BenchmarkConfig(
                node_feat_dim=int(snaps[0]["x"].shape[1]),
                num_edge_types=int(snaps[0]["edge_types"].max().item()) + 1,
                seed=seed,
                device=device,
            )
            for variant in args.variants:
                key = _cell_key(seed, drift, variant)
                if key in done:
                    print(f"skip {key}")
                    continue
                spec = VARIANTS[variant]
                print(f"\n=== {variant} seed={seed} {drift} ===")
                torch.manual_seed(seed)
                model = build_model(
                    variant, bcfg.node_feat_dim, bcfg.num_edge_types
                )
                n_params = _count_params(model)
                train_stats = train_variant(
                    model,
                    snaps,
                    train_ts,
                    device,
                    args.epochs,
                    family=spec["connection_family"],
                    static=spec["static_connection"],
                )
                train_eval = evaluate_variant(
                    model, snaps, train_ts, device, spec["static_connection"]
                )
                test_eval = evaluate_variant(
                    model, snaps, test_ts, device, spec["static_connection"]
                )
                cell = {
                    "seed": seed,
                    "drift": drift,
                    "variant": variant,
                    "n_params": n_params,
                    "train_loss": train_stats["train_loss"],
                    "train_rollout_auc": train_stats["train_auc"],
                    "train_auroc": train_eval["auroc"],
                    "train_auprc": train_eval["auprc"],
                    "test_auroc": test_eval["auroc"],
                    "test_auprc": test_eval["auprc"],
                    "train_diag": train_eval["diagnostics"],
                    "test_diag": test_eval["diagnostics"],
                    "static_rho_constant": test_eval["static_rho_constant"],
                }
                payload["cells"].append(cell)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2, default=str))
                done.add(key)
                print(
                    f"  params={n_params} test_auroc={cell['test_auroc']:.3f} "
                    f"auprc={cell['test_auprc']:.3f} "
                    f"static_ok={cell['static_rho_constant']}"
                )

    _summarize(payload)


def _summarize(payload: Dict[str, Any]) -> None:
    cells = payload["cells"]
    if not cells:
        return
    print("\n===== summary (test AUROC mean±std, paired vs B) =====")
    drifts = sorted({c["drift"] for c in cells})
    variants = list(VARIANTS)
    for drift in drifts:
        print(f"\n{drift}")
        by_var: Dict[str, Dict[int, float]] = {v: {} for v in variants}
        for c in cells:
            if c["drift"] != drift:
                continue
            by_var[c["variant"]][c["seed"]] = float(c["test_auroc"])
        b = by_var.get("B_temporal_orthogonal", {})
        for v in variants:
            vals = list(by_var[v].values())
            if not vals:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            extra = ""
            if v != "B_temporal_orthogonal" and b:
                seeds = sorted(set(by_var[v]) & set(b))
                if seeds:
                    diffs = [by_var[v][s] - b[s] for s in seeds]
                    extra = (
                        f"  paired {v}-B {np.mean(diffs):+.3f}"
                        f"±{(np.std(diffs, ddof=1) if len(diffs)>1 else 0):.3f}"
                    )
            print(f"  {v:28s} {mu:.3f}±{sd:.3f} n={len(vals)}{extra}")


if __name__ == "__main__":
    main()

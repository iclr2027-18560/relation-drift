"""Classical operator-estimation baselines on the Week-7 synthetic graphs.

Answers the ICLR review question: if compliant pairs satisfy
x(v) ≈ pinv(ρ_tgt) ρ_src x(u) with ρ in a block-orthogonal class,
does a neural snapshot encoder beat orthogonal Procrustes on the
current snapshot?

Variants (all history-free except EMA and frozen):
  snapshot_proc  — independent SO(d) Procrustes per snapshot and type
  ema_proc       — polar factor of an EMA of the cross-covariance
  lstsq          — unconstrained least squares (GL(d))
  frozen_proc    — Procrustes fit on t<0.5, frozen on test
  oracle         — planted (ρ_src, ρ_tgt) coboundary (upper bound)

Scoring: coboundary energy ||ρ_src x_u − ρ_tgt x_v|| (oracle) or
||R x_u − x_v|| (single-map estimators). Higher = more anomalous.
Same temporal split as V3.5 / TGN / DualPath (train t<0.5, test t≥0.5).
No labels are used to fit snapshot/EMA/lstsq/oracle maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
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
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)

TRAIN_TIMESTEP_RATIO = 0.5
EMA_ALPHA = 0.5
V35_POST = {
    ("clean", "slow_smooth"): 0.976138767361111,
    ("clean", "fast_smooth"): 0.9145309375,
    ("clean", "single_jump"): 0.8445506944444444,
    ("clean", "multi_jump"): 0.6718411805555555,
}
DUAL_POST = {
    ("clean", "slow_smooth"): 0.8563,
    ("clean", "fast_smooth"): 0.8086,
    ("clean", "single_jump"): 0.5977,
    ("clean", "multi_jump"): 0.5246,
}


def compute_auc(y: Tensor, scores: Tensor) -> float:
    """ROC AUC; 0.5 if a split has a single class."""
    y_np = y.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return 0.5
    return float(roc_auc_score(y_np, s_np))


def orthogonal_procrustes(src: Tensor, dst: Tensor) -> Tensor:
    """Special-orthogonal Procrustes: min_{R in SO(d)} ||R src − dst||_F.

    Solves R = U diag(1,...,det(UV^T)) V^T from SVD of dst^T src.
    """
    dim = src.shape[1]
    if src.shape[0] < 2:
        return torch.eye(dim, dtype=src.dtype)
    cov = dst.T @ src
    u_mat, _, vh = torch.linalg.svd(cov, full_matrices=False)
    r_mat = u_mat @ vh
    if torch.det(r_mat) < 0:
        u_fix = u_mat.clone()
        u_fix[:, -1] *= -1
        r_mat = u_fix @ vh
    return r_mat


def unconstrained_lstsq(src: Tensor, dst: Tensor) -> Tensor:
    """Unconstrained R minimizing ||src R^T − dst|| (GL(d) analogue)."""
    dim = src.shape[1]
    if src.shape[0] < dim:
        return torch.eye(dim, dtype=src.dtype)
    solution = torch.linalg.lstsq(src, dst).solution
    return solution.T


def polar_so(matrix: Tensor) -> Tensor:
    """Nearest SO(d) factor of a square matrix via SVD."""
    u_mat, _, vh = torch.linalg.svd(matrix, full_matrices=True)
    r_mat = u_mat @ vh
    if torch.det(r_mat) < 0:
        u_fix = u_mat.clone()
        u_fix[:, -1] *= -1
        r_mat = u_fix @ vh
    return r_mat


def pair_features(
    snap: Dict[str, Any],
    use_edge_pairs: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Endpoint features, labels, and per-edge type ids.

    ``use_edge_pairs`` scores the generator's raw (src,tgt) features
    before node-level aggregation — the planted DGP pairs.
    """
    ei = snap["query_edges"].long()
    y = snap["labels"].float()
    edge_type_names: List[str] = list(snap["edge_types"])
    type_ids = []
    for i, etype in enumerate(edge_type_names):
        n_e = int(snap["edge_index"][etype].shape[1])
        type_ids.append(torch.full((n_e,), i, dtype=torch.long))
    types = torch.cat(type_ids, dim=0)
    if use_edge_pairs:
        return (
            snap["edge_src_feats"].float(),
            snap["edge_tgt_feats"].float(),
            y,
            types,
        )
    x = snap["node_features"].float()
    return x[ei[0]], x[ei[1]], y, types


def score_single_map(
    src: Tensor,
    dst: Tensor,
    rot: Tensor,
    types: Tensor,
) -> Tensor:
    """Per-edge ||R_r x_u − x_v|| with one R per type."""
    scores = torch.zeros(src.shape[0], dtype=src.dtype)
    for r in types.unique().tolist():
        mask = types == r
        delta = src[mask] @ rot[r].T - dst[mask]
        scores[mask] = delta.norm(dim=1)
    return scores


def score_oracle(
    src: Tensor,
    dst: Tensor,
    rho_src: Tensor,
    rho_tgt: Tensor,
    types: Tensor,
) -> Tensor:
    """Planted coboundary ||ρ_src x_u − ρ_tgt x_v||."""
    scores = torch.zeros(src.shape[0], dtype=src.dtype)
    for r in types.unique().tolist():
        mask = types == r
        delta = src[mask] @ rho_src[r].T - dst[mask] @ rho_tgt[r].T
        scores[mask] = delta.norm(dim=1)
    return scores


def fit_type_maps(
    src: Tensor,
    dst: Tensor,
    types: Tensor,
    kind: str,
) -> Tensor:
    """Fit one map per edge type."""
    n_types = int(types.max().item()) + 1
    dim = src.shape[1]
    maps = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    for r in range(n_types):
        mask = types == r
        if int(mask.sum()) < 2:
            continue
        if kind == "proc":
            maps[r] = orthogonal_procrustes(src[mask], dst[mask])
        else:
            maps[r] = unconstrained_lstsq(src[mask], dst[mask])
    return maps


def evaluate_variant(
    dataset: List[Dict[str, Any]],
    variant: str,
) -> Tuple[float, Dict[int, float]]:
    """Return post-drift AUC and per-timestep AUC."""
    train_end = int(NUM_TIMESTEPS * TRAIN_TIMESTEP_RATIO)
    n_types = len(dataset[0]["edge_types"])
    dim = int(dataset[0]["node_features"].shape[1])
    frozen: Optional[Tensor] = None
    ema_cov = torch.zeros(n_types, dim, dim)
    ema_ready = False

    per_t: Dict[int, float] = {}
    post_y: List[Tensor] = []
    post_s: List[Tensor] = []

    for t, snap in enumerate(dataset):
        use_pairs = variant.startswith("edge_")
        src, dst, y, types = pair_features(snap, use_edge_pairs=use_pairs)
        kind = variant.replace("edge_", "")
        if kind == "oracle":
            scores = score_oracle(
                src, dst, snap["rho_src"].float(), snap["rho_tgt"].float(), types
            )
        elif kind == "frozen_proc":
            if t < train_end:
                if frozen is None:
                    frozen = fit_type_maps(src, dst, types, "proc")
                else:
                    frozen = 0.5 * frozen + 0.5 * fit_type_maps(
                        src, dst, types, "proc"
                    )
                    frozen = torch.stack([polar_so(m) for m in frozen])
                continue
            assert frozen is not None
            scores = score_single_map(src, dst, frozen, types)
        elif kind == "ema_proc":
            cov = torch.zeros(n_types, dim, dim)
            for r in range(n_types):
                mask = types == r
                if int(mask.sum()) >= 2:
                    cov[r] = dst[mask].T @ src[mask]
            if not ema_ready:
                ema_cov = cov
                ema_ready = True
            else:
                ema_cov = EMA_ALPHA * cov + (1.0 - EMA_ALPHA) * ema_cov
            maps = torch.stack([polar_so(m) for m in ema_cov])
            scores = score_single_map(src, dst, maps, types)
        elif kind == "lstsq":
            maps = fit_type_maps(src, dst, types, "lstsq")
            scores = score_single_map(src, dst, maps, types)
        else:
            maps = fit_type_maps(src, dst, types, "proc")
            scores = score_single_map(src, dst, maps, types)

        per_t[t] = compute_auc(y, scores)
        if t >= train_end:
            post_y.append(y)
            post_s.append(scores)

    post = (
        compute_auc(torch.cat(post_y), torch.cat(post_s)) if post_y else 0.5
    )
    return post, per_t


def run_benchmark(
    noise_filter: Optional[List[str]],
    drift_filter: Optional[List[str]],
) -> Dict[str, Any]:
    """Run the classical estimator grid."""
    noises = list(NOISE_CONFIGS.keys())
    drifts = list(DRIFT_SCENARIOS.keys())
    if noise_filter:
        noises = [n for n in noises if n in set(noise_filter)]
    if drift_filter:
        drifts = [d for d in drifts if d in set(drift_filter)]
    variants = [
        "snapshot_proc",
        "ema_proc",
        "lstsq",
        "frozen_proc",
        "oracle",
        "edge_snapshot_proc",
        "edge_oracle",
    ]
    rows: List[Dict[str, Any]] = []
    for noise in noises:
        for drift in drifts:
            print(f"\n=== {noise}/{drift} ===")
            raw = generate_temporal_dataset(
                DRIFT_SCENARIOS[drift],
                NOISE_CONFIGS[noise],
                num_timesteps=NUM_TIMESTEPS,
            )
            for variant in variants:
                post, per_t = evaluate_variant(raw, variant)
                row = {
                    "noise": noise,
                    "drift": drift,
                    "variant": variant,
                    "post_auc": post,
                    "per_timestep_auc": {str(k): v for k, v in per_t.items()},
                    "v35_post": V35_POST.get((noise, drift)),
                    "dualpath_post": DUAL_POST.get((noise, drift)),
                }
                rows.append(row)
                print(f"  {variant:14s} post-AUC={post:.4f}")
    return {
        "config": {
            "note": (
                "Unsupervised operator estimators. snapshot_proc is the "
                "classical SO(d) Procrustes control for the neural fast path."
            ),
            "ema_alpha": EMA_ALPHA,
        },
        "results": rows,
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
        "--out",
        type=str,
        default="results/procrustes_baseline_results.json",
    )
    args = parser.parse_args()
    payload = run_benchmark(args.noise, args.drift)
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

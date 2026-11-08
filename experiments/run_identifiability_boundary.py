"""Problem B: identifiability of post-jump R from aggregated node features.

Separate from Problem A (smooth typed drift). Does not mix narratives.

Claim: mean-aggregated node features are insufficient to identify typed
R ∈ SO(d) after a group-valued jump. The minimal extra statistic is the
per-type cross-moment

    S_r(u) = (1/|N_r(u)|) Σ_{v ∈ N_r(u)} x_v x_u^T

which is a compressed pair statistic, not a TGN node table.

Cells (group jump, leak-free, unlabeled):
  1. Mean-aggregated endpoints → SO(d) Procrustes  (expected ceiling ~0.55)
  2. Pooled per-type S_r → polar / IT-GLLS         (expected ~1.0)
  3. Rank diagnostic: stacked {x̄_u} vs vec(S_r(u))
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "run"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_jump_estimator_ablation import (  # noqa: E402
    TRAIN_END,
    _fro_err,
    _raw_pairs,
    _score_maps,
    _true_relative,
    compute_auc,
    fit_type_maps,
)
from run_procrustes_baseline import pair_features, polar_so
from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _group_drift(drift: str) -> Dict[str, Any]:
    params = dict(DRIFT_SCENARIOS[drift])
    params["jump_mode"] = "group"
    return params


def _node_typed_moments(
    snap: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-node per-type S_r(u) from raw pairs, then degree-weighted pool.

    Returns (pooled_maps_proc, stacked_mean_features, stacked_vec_moments).
    """
    src, dst, _y, _clean, types = _raw_pairs(snap)
    ei = snap["query_edges"].long()
    n_nodes = int(snap["num_nodes"])
    dim = src.shape[1]
    n_types = int(types.max().item()) + 1

    # Mixed-type mean table (what TGN / EvolveGCN see).
    mean_x = snap["node_features"].float()

    pooled = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    vec_rows: List[Tensor] = []
    for r in range(n_types):
        mask = types == r
        if int(mask.sum()) < 2:
            vec_rows.append(torch.zeros(n_nodes, dim * dim))
            continue
        src_idx = ei[0][mask]
        src_r, dst_r = src[mask], dst[mask]
        # Degree-weighted pool of S_r(u) = mean_v x_v x_u^T equals pair S_r.
        cov = dst_r.T @ src_r
        pooled[r] = polar_so(cov)
        # Per-source-node moments for the rank diagnostic.
        acc = torch.zeros(n_nodes, dim, dim)
        cnt = torch.zeros(n_nodes, 1, 1)
        outer = dst_r.unsqueeze(2) * src_r.unsqueeze(1)  # [E, d, d]  x_v x_u^T
        acc.index_add_(0, src_idx, outer)
        cnt.index_add_(0, src_idx, torch.ones(src_idx.shape[0], 1, 1))
        s_u = acc / cnt.clamp(min=1.0)
        vec_rows.append(s_u.reshape(n_nodes, dim * dim))

    stacked_moments = torch.cat(vec_rows, dim=1)  # [N, n_types * d^2]
    return pooled, mean_x, stacked_moments


def _matrix_rank(mat: Tensor, tol: float = 1e-6) -> int:
    if mat.numel() == 0:
        return 0
    # Center columns; rank of the observation design.
    centered = mat - mat.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    peak = float(svals[0].item()) if svals.numel() else 0.0
    thresh = tol * max(peak, 1.0)
    return int((svals > thresh).sum().item())


def run_seed(seed: int, drift: str) -> Dict[str, Any]:
    _set_seed(seed)
    raw = generate_temporal_dataset(
        _group_drift(drift),
        NOISE_CONFIGS["clean"],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )

    post_mean_y: List[Tensor] = []
    post_mean_s: List[Tensor] = []
    post_moment_y: List[Tensor] = []
    post_moment_s: List[Tensor] = []
    post_glls_s: List[Tensor] = []
    post_mean_fro: List[float] = []
    post_moment_fro: List[float] = []
    ranks: List[Dict[str, float]] = []
    per_t: Dict[str, Any] = {}

    for t, snap in enumerate(raw):
        target = _true_relative(snap)
        n_types = int(target.shape[0])
        dim = int(target.shape[-1])
        so_dim = n_types * dim * (dim - 1) // 2

        # (1) mean-aggregated node table
        src_m, dst_m, y_m, types_m = pair_features(snap, use_edge_pairs=False)
        mean_maps = fit_type_maps(src_m, dst_m, types_m, "proc")
        mean_scores = _score_maps(src_m, dst_m, mean_maps, types_m)
        mean_fro = [_fro_err(mean_maps[r], target[r]) for r in range(n_types)]

        # (2) typed cross-moment from raw pairs (minimal sufficient statistic)
        src_p, dst_p, y_p, _clean, types_p = _raw_pairs(snap)
        moment_maps, mean_x, stacked_S = _node_typed_moments(snap)
        glls_maps = it_glls_maps(src_p, dst_p, types_p)
        moment_scores = _score_maps(src_p, dst_p, moment_maps, types_p)
        glls_scores = _score_maps(src_p, dst_p, glls_maps, types_p)
        moment_fro = [_fro_err(moment_maps[r], target[r]) for r in range(n_types)]

        rank_mean = _matrix_rank(mean_x)
        rank_S = _matrix_rank(stacked_S)
        row = {
            "mean_agg_auc": compute_auc(y_m, mean_scores),
            "moment_auc": compute_auc(y_p, moment_scores),
            "it_glls_auc": compute_auc(y_p, glls_scores),
            "mean_fro": float(sum(mean_fro) / n_types),
            "moment_fro": float(sum(moment_fro) / n_types),
            "rank_mean_features": rank_mean,
            "rank_typed_moments": rank_S,
            "so_param_dim": so_dim,
            "rank_mean_vs_params": rank_mean / max(so_dim, 1),
            "rank_S_vs_params": rank_S / max(so_dim, 1),
        }
        per_t[str(t)] = row
        ranks.append(row)
        if t >= TRAIN_END:
            post_mean_y.append(y_m)
            post_mean_s.append(mean_scores)
            post_moment_y.append(y_p)
            post_moment_s.append(moment_scores)
            post_glls_s.append(glls_scores)
            post_mean_fro.append(row["mean_fro"])
            post_moment_fro.append(row["moment_fro"])

    out = {
        "seed": seed,
        "drift": drift,
        "jump_mode": "group",
        "mean_agg_post_auc": compute_auc(torch.cat(post_mean_y), torch.cat(post_mean_s)),
        "moment_post_auc": compute_auc(
            torch.cat(post_moment_y), torch.cat(post_moment_s)
        ),
        "it_glls_post_auc": compute_auc(
            torch.cat(post_moment_y), torch.cat(post_glls_s)
        ),
        "mean_agg_post_fro": float(sum(post_mean_fro) / max(len(post_mean_fro), 1)),
        "moment_post_fro": float(sum(post_moment_fro) / max(len(post_moment_fro), 1)),
        "rank_mean_features": float(
            sum(r["rank_mean_features"] for r in ranks) / max(len(ranks), 1)
        ),
        "rank_typed_moments": float(
            sum(r["rank_typed_moments"] for r in ranks) / max(len(ranks), 1)
        ),
        "so_param_dim": ranks[-1]["so_param_dim"] if ranks else None,
        "per_timestep": per_t,
    }
    print(
        f"  [seed={seed}] mean-agg AUC={out['mean_agg_post_auc']:.3f} "
        f"F={out['mean_agg_post_fro']:.3f} | "
        f"S_r AUC={out['moment_post_auc']:.3f} F={out['moment_post_fro']:.3f} | "
        f"IT-GLLS={out['it_glls_post_auc']:.3f} | "
        f"rank x̄={out['rank_mean_features']:.1f} "
        f"rank S={out['rank_typed_moments']:.1f} "
        f"params={out['so_param_dim']}"
    )
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / max(n, 1)
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--drift", type=str, default="single_jump")
    parser.add_argument(
        "--out",
        type=str,
        default="results/identifiability_boundary.json",
    )
    args = parser.parse_args()

    rows = [run_seed(seed, args.drift) for seed in args.seeds]
    summary = {
        "mean_agg_auc": _mean_std([r["mean_agg_post_auc"] for r in rows]),
        "moment_auc": _mean_std([r["moment_post_auc"] for r in rows]),
        "it_glls_auc": _mean_std([r["it_glls_post_auc"] for r in rows]),
        "mean_agg_fro": _mean_std([r["mean_agg_post_fro"] for r in rows]),
        "moment_fro": _mean_std([r["moment_post_fro"] for r in rows]),
        "rank_mean_features": _mean_std([r["rank_mean_features"] for r in rows]),
        "rank_typed_moments": _mean_std([r["rank_typed_moments"] for r in rows]),
    }
    mean_ok = summary["mean_agg_auc"][0] < 0.80
    moment_ok = summary["moment_auc"][0] >= 0.95
    summary["success"] = {
        "mean_agg_is_ceiling": mean_ok,
        "cross_moment_recovers": moment_ok,
        "rank_S_exceeds_rank_mean": (
            summary["rank_typed_moments"][0] > summary["rank_mean_features"][0]
        ),
    }
    payload = {
        "protocol": (
            "Group-valued jump so R stays in SO(d). Mean-agg uses the TGN node "
            "table. S_r is the per-type cross-moment from pairs (compressed pair "
            "statistic). IT-GLLS is the GL(d) control on the same pairs. "
            "Not mixed with the adversarial-smooth typed-sheaf experiment."
        ),
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

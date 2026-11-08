"""Leak-free V3.5 with unlabeled IT-GLLS pair residual as an edge feature.

Per snapshot, fit C_t in GL(d) on raw (src,tgt) pairs with no labels,
score r_e = ||x_v - C_t x_u||, and concatenate r_e onto the sheaf edge head.
Cayley/GRU stay unchanged. This is the architectural wiring of the pair-level
estimator that already recovers post-jump constraints (AUC ~1 on raw pairs).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from run_jump_estimator_ablation import (  # noqa: E402
    TRAIN_END,
    _raw_pairs,
    _score_maps,
    compute_auc,
    leak_free_train_config,
)
from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset,
)
from run_v35_benchmark import (  # noqa: E402
    BenchmarkConfig,
    evaluate_model_detailed,
    get_temporal_split,
    load_temporal_dataset,
    train_static_sheaf,
)
from temporal_sheaf_v35 import SheafConfig, TemporalSheafV35  # noqa: E402
from train_v35 import train_temporal_v35  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def it_glls_maps(
    src: Tensor,
    dst: Tensor,
    types: Tensor,
    alpha: float = 0.15,
    iters: int = 3,
) -> Tensor:
    """Per-type trimmed GL(d) least squares, no labels."""
    n_types = int(types.max().item()) + 1
    dim = src.shape[1]
    maps = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    for r in range(n_types):
        mask = types == r
        if int(mask.sum()) < dim + 2:
            continue
        src_r, dst_r = src[mask], dst[mask]
        keep = torch.ones(src_r.shape[0], dtype=torch.bool)
        fitted = torch.eye(dim, dtype=src.dtype)
        for _ in range(iters):
            if int(keep.sum()) < dim + 2:
                break
            fitted = torch.linalg.lstsq(src_r[keep], dst_r[keep]).solution.T
            resid = (src_r @ fitted.T - dst_r).norm(dim=1)
            thresh = torch.quantile(resid, 1.0 - alpha)
            keep = resid <= thresh
        maps[r] = fitted
    return maps


def attach_pair_residuals(snapshots: List[Dict[str, Tensor]]) -> Dict[str, float]:
    """Write log-scaled unlabeled GL residuals onto each snapshot."""
    only_y: List[Tensor] = []
    only_s: List[Tensor] = []
    per_t: Dict[str, float] = {}
    for t, snap in enumerate(snapshots):
        src = snap["edge_src_feats"].float()
        dst = snap["edge_tgt_feats"].float()
        types = snap["edge_types"].long()
        y = snap["labels"].float()
        maps = it_glls_maps(src, dst, types)
        residual = _score_maps(src, dst, maps, types)
        # log1p keeps the head numerically stable; ranking is unchanged
        snap["pair_residual"] = torch.log1p(residual).detach()
        auc = compute_auc(y, residual)
        per_t[f"t_{t}"] = auc
        if t >= TRAIN_END:
            only_y.append(y)
            only_s.append(residual)
    return {
        "residual_only_post_auc": compute_auc(torch.cat(only_y), torch.cat(only_s)),
        "residual_only_per_timestep": per_t,
    }


def run_one(
    seed: int,
    drift: str,
    device: str,
    quick: bool,
) -> Dict[str, Any]:
    _set_seed(seed)
    config = BenchmarkConfig(seed=seed, device=device)
    train_ts, test_ts = get_temporal_split(config.num_timesteps)
    snapshots = load_temporal_dataset("clean", drift, seed=seed)
    config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
    config.node_feat_dim = int(snapshots[0]["x"].shape[1])

    residual_diag = attach_pair_residuals(snapshots)
    print(
        f"  residual-only post-AUC={residual_diag['residual_only_post_auc']:.3f}"
    )

    train_cfg = leak_free_train_config()
    static_epochs = 8 if quick else 200
    if quick:
        train_cfg.epochs_phase1 = 2
        train_cfg.epochs_phase3 = 2

    static_model = train_static_sheaf(
        snapshots, config, train_ts, num_epochs=static_epochs
    )
    sheaf_config = SheafConfig(
        node_feat_dim=config.node_feat_dim,
        num_edge_types=config.num_edge_types,
        use_fast_encoder=True,
        pair_residual_dim=1,
    )
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
    detail = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        jump_times=[],
        hard_handoff=False,
    )
    post = [detail["per_timestep_auc"][f"t_{t}"] for t in test_ts]
    pre = [detail["per_timestep_auc"][f"t_{t}"] for t in train_ts]
    out = {
        "seed": seed,
        "drift": drift,
        **residual_diag,
        "v35_residual_pre_auc": sum(pre) / max(len(pre), 1),
        "v35_residual_post_auc": sum(post) / max(len(post), 1),
        "per_timestep_auc": detail["per_timestep_auc"],
    }
    print(
        f"  V3.5+IT-GLLS pre={out['v35_residual_pre_auc']:.3f} "
        f"post={out['v35_residual_post_auc']:.3f}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--drift",
        nargs="*",
        default=["single_jump", "slow_smooth"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/gld_residual_v35.json",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for drift in args.drift:
            print(f"\n=== seed={seed} {drift} ===")
            rows.append(run_one(seed, drift, args.device, args.quick))

    payload = {
        "protocol": (
            "Leak-free V3.5. Unlabeled IT-GLLS on raw pairs; log1p residual "
            "concatenated to the edge head. No jump labels."
        ),
        "results": rows,
    }
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

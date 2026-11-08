"""Leak-free multi-seed jump protocol: V3.5 vs DualPath.

Temporal holdout only (t<0.5). No planted jump BCE, no GT hard-handoff,
no post-split task loss. Records per-timestep AUC for the recovery curve.
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

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "run"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_dualpath_gnn_baseline import (  # noqa: E402
    load_temporal_dataset as load_dualpath,
    train_and_evaluate as train_dualpath,
)
from run_v35_benchmark import (  # noqa: E402
    get_temporal_split,
    load_temporal_dataset as load_v35,
    train_and_eval_static,
    train_and_eval_v35,
    train_static_sheaf,
    BenchmarkConfig,
)
from temporal_sheaf_v35 import TrainConfig  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_v35_seed(seed: int) -> Dict[str, Any]:
    _set_seed(seed)
    config = BenchmarkConfig(seed=seed)
    train_ts, test_ts = get_temporal_split(config.num_timesteps)
    snapshots = load_v35("clean", "single_jump", seed=seed)
    config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
    config.node_feat_dim = int(snapshots[0]["x"].shape[1])
    train_cfg = TrainConfig(
        epochs_phase1=80,
        epochs_phase2=0,
        epochs_phase3=160,
        use_routing_curriculum=True,
        fusion_gate_penalty=0.0,
        jump_gate_bce_weight=0.0,
        jump_gate_window=0,
        fast_cb_penalty=0.1,
        hard_handoff_on_jump=False,
        train_window_only=True,
        use_gt_jump_at_eval=False,
    )
    print("  Training static sheaf...")
    static_results = train_and_eval_static(
        snapshots, config, train_ts, test_ts, static_epochs=200
    )
    static_model = train_static_sheaf(
        snapshots, config, train_ts, num_epochs=200
    )
    print("  Training V3.5 (leak-free)...")
    v35 = train_and_eval_v35(
        snapshots,
        config,
        static_model,
        train_ts,
        test_ts,
        jump_times=[],
        train_config=train_cfg,
    )
    return {
        "seed": seed,
        "model": "v35",
        "static_pre": static_results["mean_pre"],
        "static_post": static_results["mean_post"],
        "pre": v35["mean_pre"],
        "post": v35["mean_post"],
        "per_timestep_auc": v35["per_timestep_auc"],
        "fusion_gates": v35["fusion_gates"],
    }


def run_dualpath_seed(seed: int) -> Dict[str, Any]:
    _set_seed(seed)
    snaps = load_dualpath("clean", "single_jump", seed=seed)
    post, per_t, mean_gate = train_dualpath(snaps, "dual", 200)
    return {
        "seed": seed,
        "model": "dualpath",
        "post": post,
        "mean_gate": mean_gate,
        "per_timestep_auc": {f"t_{k}": v for k, v in per_t.items()},
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    v35 = [r for r in rows if r["model"] == "v35"]
    dp = [r for r in rows if r["model"] == "dualpath"]
    summary: Dict[str, Any] = {}
    if v35:
        summary["v35_post"] = _mean_std([r["post"] for r in v35])
        summary["static_post"] = _mean_std([r["static_post"] for r in v35])
        keys = sorted(v35[0]["per_timestep_auc"].keys())
        summary["v35_per_timestep"] = {
            k: _mean_std([r["per_timestep_auc"][k] for r in v35]) for k in keys
        }
    if dp:
        summary["dualpath_post"] = _mean_std([r["post"] for r in dp])
        keys = sorted(dp[0]["per_timestep_auc"].keys())
        summary["dualpath_per_timestep"] = {
            k: _mean_std([r["per_timestep_auc"][k] for r in dp]) for k in keys
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--out",
        type=str,
        default="results/jump_multiseed_unsupervised.json",
    )
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n======== seed {seed} V3.5 ========")
        rows.append(run_v35_seed(seed))
        print(f"  V3.5 post={rows[-1]['post']:.4f}")
        print(f"\n======== seed {seed} DualPath ========")
        rows.append(run_dualpath_seed(seed))
        print(f"  DualPath post={rows[-1]['post']:.4f}")
    payload = {
        "protocol": "leak-free: train t<0.5, no jump BCE, no GT handoff",
        "seeds": args.seeds,
        "summary": summarize(rows),
        "runs": rows,
    }
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

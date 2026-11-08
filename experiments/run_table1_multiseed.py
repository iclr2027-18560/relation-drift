"""Five-seed Table 1: MLP, TGN, EvolveGCN-H, static sheaf, DualPath, V3.5.

Clean noise, four drifts, seeds 42–46, leak-free V3.5 (--no-jump-labels).
Saves after every cell to results/table1_multiseed.json.

Usage:
  python experiments/run_table1_multiseed.py
  python experiments/run_table1_multiseed.py --quick --seeds 42 --drifts single_jump --models mlp dual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX35 = _SRC / "run"
for _p in (_SRC, _FIX35, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sheaf_benchmark import DRIFT_SCENARIOS  # noqa: E402
from run_tgn_baseline import (  # noqa: E402
    NUM_EPOCHS as TGN_EPOCHS,
    load_temporal_dataset as load_tgn,
    train_and_evaluate_static,
    train_and_evaluate_tgn,
)
from run_evolvegcn_baseline import (  # noqa: E402
    NUM_EPOCHS as EGCN_EPOCHS,
    load_temporal_dataset as load_egcn,
    train_and_evaluate_evolvegcn,
)
from run_dualpath_gnn_baseline import (  # noqa: E402
    NUM_EPOCHS as DP_EPOCHS,
    load_temporal_dataset as load_dp,
    train_and_evaluate as train_dp,
)
from run_v35_benchmark import (  # noqa: E402
    STATIC_EPOCHS,
    TEMPORAL_PHASE1_EPOCHS,
    TEMPORAL_PHASE2_EPOCHS,
    TEMPORAL_PHASE3_EPOCHS,
    BenchmarkConfig,
    get_temporal_split,
    load_temporal_dataset as load_v35,
    train_and_eval_v35,
    train_static_sheaf,
    train_and_eval_static,
)
from temporal_sheaf_v35 import TrainConfig  # noqa: E402

DEFAULT_DRIFTS = ["slow_smooth", "fast_smooth", "single_jump", "multi_jump"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _load_existing(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"cells": [], "config": {}}


def _cell_key(seed: int, drift: str, model: str) -> str:
    return f"{seed}/{drift}/{model}"


def _done(cells: List[dict]) -> set:
    return {_cell_key(c["seed"], c["drift"], c["model"]) for c in cells}


def _append(path: Path, payload: Dict[str, Any], cell: dict) -> None:
    payload["cells"].append(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(
        f"  saved {cell['model']} {cell['drift']} seed={cell['seed']} "
        f"post={cell['post_auc']:.4f} -> {path}"
    )


def _v35_train_cfg(quick: bool) -> TrainConfig:
    if quick:
        cfg = TrainConfig(
            epochs_phase1=40,
            epochs_phase2=0,
            epochs_phase3=40,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=0.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=False,
        )
    else:
        cfg = TrainConfig(
            epochs_phase1=TEMPORAL_PHASE1_EPOCHS,
            epochs_phase2=TEMPORAL_PHASE2_EPOCHS,
            epochs_phase3=TEMPORAL_PHASE3_EPOCHS,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=0.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=False,
        )
        if cfg.epochs_phase2 > 0:
            cfg.epochs_phase3 += cfg.epochs_phase2
            cfg.epochs_phase2 = 0
    cfg.use_gt_jump_at_eval = False
    cfg.train_window_only = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--drifts", nargs="*", default=DEFAULT_DRIFTS)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["mlp", "tgn", "evolvegcn", "static_sheaf", "dual", "v35"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default="results/table1_multiseed.json")
    args = parser.parse_args()

    out_path = _ROOT / args.out
    payload = _load_existing(out_path)
    payload["config"] = {
        "seeds": args.seeds,
        "drifts": args.drifts,
        "models": args.models,
        "quick": args.quick,
        "noise": "clean",
        "no_jump_labels": True,
    }
    done = _done(payload["cells"])

    epochs_tgn = 40 if args.quick else TGN_EPOCHS
    epochs_eg = 40 if args.quick else EGCN_EPOCHS
    epochs_dp = 40 if args.quick else DP_EPOCHS
    static_epochs = 40 if args.quick else STATIC_EPOCHS
    wanted = [m.lower() for m in args.models]
    train_cfg = _v35_train_cfg(args.quick)

    for seed in args.seeds:
        torch.manual_seed(seed)
        for drift in args.drifts:
            if drift not in DRIFT_SCENARIOS:
                raise ValueError(drift)

            need_tgn_data = any(m in wanted for m in ("mlp", "tgn"))
            tgn_snaps = (
                load_tgn("clean", drift, seed=seed) if need_tgn_data else None
            )

            if "mlp" in wanted and _cell_key(seed, drift, "mlp") not in done:
                print(f"\n=== MLP seed={seed} {drift} ===")
                feat = int(tgn_snaps[0]["x"].shape[1])
                auc, _ = train_and_evaluate_static(
                    tgn_snaps, feat, num_epochs=epochs_tgn
                )
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "mlp",
                        "post_auc": float(auc),
                    },
                )
                done = _done(payload["cells"])

            if "tgn" in wanted and _cell_key(seed, drift, "tgn") not in done:
                print(f"\n=== TGN seed={seed} {drift} ===")
                auc, _ = train_and_evaluate_tgn(
                    tgn_snaps, num_epochs=epochs_tgn
                )
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "tgn",
                        "post_auc": float(auc),
                    },
                )
                done = _done(payload["cells"])

            if (
                "evolvegcn" in wanted
                and _cell_key(seed, drift, "evolvegcn") not in done
            ):
                print(f"\n=== EvolveGCN-H seed={seed} {drift} ===")
                snaps = load_egcn("clean", drift, seed=seed)
                n = int(max(s["x"].shape[0] for s in snaps))
                feat = int(snaps[0]["x"].shape[1])
                auc, _ = train_and_evaluate_evolvegcn(
                    snaps, n, feat, num_epochs=epochs_eg, variant="H"
                )
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "evolvegcn",
                        "post_auc": float(auc),
                    },
                )
                done = _done(payload["cells"])

            if "dual" in wanted and _cell_key(seed, drift, "dual") not in done:
                print(f"\n=== DualPath seed={seed} {drift} ===")
                snaps = load_dp("clean", drift, seed=seed, jump_mode="additive")
                auc, _, _ = train_dp(snaps, "dual", epochs_dp)
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "dual",
                        "post_auc": float(auc),
                    },
                )
                done = _done(payload["cells"])

            need_v35 = any(m in wanted for m in ("static_sheaf", "v35"))
            if need_v35:
                bcfg = BenchmarkConfig(seed=seed)
                torch.manual_seed(seed)
                snaps = load_v35("clean", drift, seed=seed)
                bcfg.node_feat_dim = int(snaps[0]["x"].shape[1])
                bcfg.num_edge_types = int(snaps[0]["edge_types"].max().item()) + 1
                train_ts, test_ts = get_temporal_split(bcfg.num_timesteps)

            if (
                "static_sheaf" in wanted
                and _cell_key(seed, drift, "static_sheaf") not in done
            ):
                print(f"\n=== Static sheaf seed={seed} {drift} ===")
                static_res = train_and_eval_static(
                    snaps,
                    bcfg,
                    train_ts,
                    test_ts,
                    static_epochs=static_epochs,
                )
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "static_sheaf",
                        "post_auc": float(static_res["mean_post"]),
                    },
                )
                done = _done(payload["cells"])

            if "v35" in wanted and _cell_key(seed, drift, "v35") not in done:
                print(f"\n=== V3.5 leak-free seed={seed} {drift} ===")
                static_model = train_static_sheaf(
                    snaps, bcfg, train_ts, num_epochs=static_epochs
                )
                v35_res = train_and_eval_v35(
                    snaps,
                    bcfg,
                    static_model,
                    train_ts,
                    test_ts,
                    jump_times=[],
                    train_config=train_cfg,
                )
                _append(
                    out_path,
                    payload,
                    {
                        "seed": seed,
                        "drift": drift,
                        "model": "v35",
                        "post_auc": float(v35_res["mean_post"]),
                    },
                )
                done = _done(payload["cells"])

    print(f"\nComplete: {len(payload['cells'])} cells in {out_path}")


if __name__ == "__main__":
    main()

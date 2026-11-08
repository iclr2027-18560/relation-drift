"""Problem A: typed sheaf vs global W_t on smooth Lipschitz drift.

Default slow_smooth uses correlated Givens planes, so EvolveGCN's single
W_t can track all types at once. This script does NOT replace that DGP.

It adds a separate adversarial-type cell: per-type signed ω in opposing
planes so a global map must average conflicting rotations.

Compares, leak-free, unlabeled:
  - V3.5 Cayley sheaf (typed restriction maps)
  - EvolveGCN-H (single global W_t)
  - DualPath GNN
  - Classical typed vs mixed-type Procrustes operator error vs R*

Success: sheaf post-AUC >= EvolveGCN + 0.02, OR typed map error
< 0.5 * global-map error.
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

from run_dualpath_gnn_baseline import (  # noqa: E402
    _adapt_sheaf_dataset as adapt_dualpath,
)
from run_dualpath_gnn_baseline import train_and_evaluate as train_dualpath
from run_evolvegcn_baseline import (  # noqa: E402
    _adapt_sheaf_dataset as adapt_evolvegcn,
)
from run_evolvegcn_baseline import train_and_evaluate_evolvegcn
from run_jump_estimator_ablation import (  # noqa: E402
    TRAIN_END,
    _fro_err,
    _raw_pairs,
    _score_maps,
    _true_relative,
    compute_auc,
    leak_free_train_config,
)
from run_procrustes_baseline import orthogonal_procrustes  # noqa: E402
from run_sheaf_benchmark import NOISE_CONFIGS, NUM_TIMESTEPS, generate_temporal_dataset
from run_v35_benchmark import (  # noqa: E402
    BenchmarkConfig,
    _adapt_sheaf_dataset as adapt_v35,
    evaluate_model_detailed,
    get_temporal_split,
    train_static_sheaf,
)
from temporal_sheaf_v35 import SheafConfig, TemporalSheafV35  # noqa: E402
from train_v35 import train_temporal_v35  # noqa: E402

# Opposing planes: sends_to and transfers share (0,1) and counter-rotate;
# owns drifts in the orthogonal (2,3) plane at >=2x rate.
ADVERSARIAL_DRIFT: Dict[str, Any] = {
    "omega": 0.05,
    "jumps": [],
    "magnitudes": [],
    "adversarial_types": True,
    "angular_velocities": {
        "sends_to": 0.05,
        "owns": -0.12,
        "transfers": -0.08,
    },
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_adversarial(seed: int) -> List[Dict[str, Any]]:
    return generate_temporal_dataset(
        ADVERSARIAL_DRIFT,
        NOISE_CONFIGS["clean"],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )


def _type_disagreement(rel: Tensor) -> Dict[str, float]:
    """Pairwise ||R_i - R_j||_F on planted relative maps."""
    out: Dict[str, float] = {}
    n = int(rel.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            out[f"{i}_vs_{j}"] = float((rel[i] - rel[j]).square().sum().sqrt().item())
    return out


def run_classical(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Typed vs global Procrustes on raw pairs (no neural model)."""
    post_typed_err: List[float] = []
    post_global_err: List[float] = []
    post_y: List[Tensor] = []
    post_typed_s: List[Tensor] = []
    post_global_s: List[Tensor] = []
    per_t: Dict[str, Any] = {}
    last_disagree: Dict[str, float] = {}

    for t, snap in enumerate(raw):
        src, dst, y, _clean, types = _raw_pairs(snap)
        target = _true_relative(snap)
        n_types = int(target.shape[0])
        typed = torch.eye(src.shape[1]).unsqueeze(0).repeat(n_types, 1, 1)
        for r in range(n_types):
            mask = types == r
            if int(mask.sum()) >= 2:
                typed[r] = orthogonal_procrustes(src[mask], dst[mask])
        mixed = orthogonal_procrustes(src, dst)
        global_maps = mixed.unsqueeze(0).expand(n_types, -1, -1).contiguous()

        typed_err = [
            _fro_err(typed[r], target[r]) for r in range(n_types)
        ]
        global_err = [
            _fro_err(global_maps[r], target[r]) for r in range(n_types)
        ]
        typed_scores = _score_maps(src, dst, typed, types)
        global_scores = _score_maps(src, dst, global_maps, types)
        row = {
            "typed_auc": compute_auc(y, typed_scores),
            "global_auc": compute_auc(y, global_scores),
            "typed_fro_mean": float(sum(typed_err) / n_types),
            "global_fro_mean": float(sum(global_err) / n_types),
            "typed_fro_per_type": typed_err,
            "global_fro_per_type": global_err,
            "type_disagreement": _type_disagreement(target),
        }
        per_t[str(t)] = row
        last_disagree = row["type_disagreement"]
        if t >= TRAIN_END:
            post_typed_err.extend(typed_err)
            post_global_err.extend(global_err)
            post_y.append(y)
            post_typed_s.append(typed_scores)
            post_global_s.append(global_scores)

    typed_fro = float(sum(post_typed_err) / max(len(post_typed_err), 1))
    global_fro = float(sum(post_global_err) / max(len(post_global_err), 1))
    return {
        "post_typed_auc": compute_auc(torch.cat(post_y), torch.cat(post_typed_s)),
        "post_global_auc": compute_auc(torch.cat(post_y), torch.cat(post_global_s)),
        "post_typed_fro": typed_fro,
        "post_global_fro": global_fro,
        "fro_ratio_typed_over_global": typed_fro / max(global_fro, 1e-8),
        "type_disagreement_final": last_disagree,
        "per_timestep": per_t,
    }


def run_sheaf(
    raw: List[Dict[str, Any]],
    seed: int,
    device: str,
    quick: bool,
) -> Dict[str, Any]:
    _set_seed(seed)
    snapshots = adapt_v35(raw)
    config = BenchmarkConfig(seed=seed, device=device)
    train_ts, test_ts = get_temporal_split(config.num_timesteps)
    config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
    config.node_feat_dim = int(snapshots[0]["x"].shape[1])
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
    return {
        "post_auc": sum(post) / max(len(post), 1),
        "per_timestep_auc": detail["per_timestep_auc"],
    }


def run_evolvegcn(
    raw: List[Dict[str, Any]],
    quick: bool,
) -> Dict[str, Any]:
    snaps = adapt_evolvegcn(raw)
    epochs = 8 if quick else 200
    post, per_t = train_and_evaluate_evolvegcn(
        snaps,
        num_nodes=int(snaps[0]["x"].shape[0]),
        node_feat_dim=int(snaps[0]["x"].shape[1]),
        num_epochs=epochs,
        variant="H",
    )
    return {"post_auc": float(post), "per_timestep_auc": {str(k): v for k, v in per_t.items()}}


def run_dualpath(
    raw: List[Dict[str, Any]],
    quick: bool,
) -> Dict[str, Any]:
    snaps = adapt_dualpath(raw)
    epochs = 8 if quick else 200
    post, per_t, gate = train_dualpath(snaps, mode="dual", num_epochs=epochs)
    return {
        "post_auc": float(post),
        "mean_gate": float(gate),
        "per_timestep_auc": {str(k): v for k, v in per_t.items()},
    }


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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/typed_adversarial_smooth.json",
    )
    args = parser.parse_args()

    classical_rows: List[Dict[str, Any]] = []
    neural_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        _set_seed(seed)
        raw = generate_adversarial(seed)
        classical = run_classical(raw)
        classical_rows.append({"seed": seed, **classical})
        print(
            f"  [classical seed={seed}] typed-AUC={classical['post_typed_auc']:.3f} "
            f"global-AUC={classical['post_global_auc']:.3f} "
            f"typed-F={classical['post_typed_fro']:.3f} "
            f"global-F={classical['post_global_fro']:.3f} "
            f"disagree={classical['type_disagreement_final']}"
        )
        if args.skip_neural:
            continue
        sheaf = run_sheaf(raw, seed, args.device, args.quick)
        egcn = run_evolvegcn(raw, args.quick)
        dual = run_dualpath(raw, args.quick)
        delta = sheaf["post_auc"] - egcn["post_auc"]
        neural_rows.append(
            {
                "seed": seed,
                "sheaf_post_auc": sheaf["post_auc"],
                "evolvegcn_post_auc": egcn["post_auc"],
                "dualpath_post_auc": dual["post_auc"],
                "sheaf_minus_evolvegcn": delta,
                "sheaf": sheaf,
                "evolvegcn": egcn,
                "dualpath": dual,
            }
        )
        print(
            f"  [neural seed={seed}] sheaf={sheaf['post_auc']:.3f} "
            f"EvolveGCN={egcn['post_auc']:.3f} DualPath={dual['post_auc']:.3f} "
            f"Δ={delta:+.3f}"
        )

    summary: Dict[str, Any] = {
        "typed_proc_auc": _mean_std([r["post_typed_auc"] for r in classical_rows]),
        "global_proc_auc": _mean_std([r["post_global_auc"] for r in classical_rows]),
        "typed_fro": _mean_std([r["post_typed_fro"] for r in classical_rows]),
        "global_fro": _mean_std([r["post_global_fro"] for r in classical_rows]),
        "fro_ratio": _mean_std(
            [r["fro_ratio_typed_over_global"] for r in classical_rows]
        ),
    }
    if neural_rows:
        sheaf_m, sheaf_s = _mean_std([r["sheaf_post_auc"] for r in neural_rows])
        egcn_m, egcn_s = _mean_std([r["evolvegcn_post_auc"] for r in neural_rows])
        dual_m, dual_s = _mean_std([r["dualpath_post_auc"] for r in neural_rows])
        summary["sheaf_auc"] = (sheaf_m, sheaf_s)
        summary["evolvegcn_auc"] = (egcn_m, egcn_s)
        summary["dualpath_auc"] = (dual_m, dual_s)
        summary["sheaf_minus_evolvegcn"] = _mean_std(
            [r["sheaf_minus_evolvegcn"] for r in neural_rows]
        )
        typed_fro_m = summary["typed_fro"][0]
        global_fro_m = summary["global_fro"][0]
        summary["success"] = {
            "auc_gap_ge_0.02": (sheaf_m - egcn_m) >= 0.02,
            "typed_fro_lt_half_global": typed_fro_m < 0.5 * global_fro_m,
        }

    payload = {
        "protocol": (
            "Adversarial-type smooth DGP (separate from default slow_smooth). "
            "Types counter-rotate in opposing Givens planes. Leak-free train t<0.5. "
            "Classical operator error uses raw pairs; neural models see aggregated nodes."
        ),
        "drift": ADVERSARIAL_DRIFT,
        "summary": summary,
        "classical": classical_rows,
        "neural": neural_rows,
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

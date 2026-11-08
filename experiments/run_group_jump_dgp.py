"""Group-multiply vs additive jump DGP (same seeds).

Keeps the default additive jump. Adds a group-valued cell:
    ρ(τ+) = Cayley(A) ρ(τ-)
so the relative map stays in SO(d).

Compares, leak-free, no jump labels:
  - two-map oracle
  - pair SO(d) Procrustes (all pairs)
  - pair GL(d) IT-GLLS (all pairs)
  - node-aggregated SO(d) Procrustes
  - leak-free aggregated-node Cayley sheaf (V3.5)

Hypothesis: when R* ∈ SO(d), pair Procrustes recovers the jump;
aggregated Cayley still fails because aggregation, not the group, is
the blocker.
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
    _raw_pairs,
    _score_maps,
    _score_oracle,
    compute_auc,
    evaluate_classical,
    leak_free_train_config,
)
from run_procrustes_baseline import pair_features  # noqa: E402
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


def _drift_params(drift: str, jump_mode: str) -> Dict[str, Any]:
    params = dict(DRIFT_SCENARIOS[drift])
    params["jump_mode"] = jump_mode
    return params


def _so_deviation(snap: Dict[str, Any]) -> float:
    """||R R^T - I||_F for R = pinv(ρ_tgt) ρ_src on type 0."""
    rho_src = snap["rho_src"].float()[0]
    rho_tgt = snap["rho_tgt"].float()[0]
    rel = torch.linalg.pinv(rho_tgt) @ rho_src
    eye = torch.eye(rel.shape[0], dtype=rel.dtype)
    return float((rel @ rel.T - eye).square().sum().sqrt().item())


def _node_agg_proc(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    post_y: List[Tensor] = []
    post_s: List[Tensor] = []
    per_t: Dict[str, float] = {}
    for t, snap in enumerate(dataset):
        src, dst, y, types = pair_features(snap, use_edge_pairs=False)
        from run_jump_estimator_ablation import fit_type_maps

        fitted = fit_type_maps(src, dst, types, "proc")
        scores = _score_maps(src, dst, fitted, types)
        auc = compute_auc(y, scores)
        per_t[str(t)] = auc
        if t >= TRAIN_END:
            post_y.append(y)
            post_s.append(scores)
    return {
        "post_auc": compute_auc(torch.cat(post_y), torch.cat(post_s)),
        "per_timestep_auc": per_t,
    }


def _it_glls_classical(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    post_y: List[Tensor] = []
    post_s: List[Tensor] = []
    per_t: Dict[str, float] = {}
    for t, snap in enumerate(dataset):
        src, dst, y, _clean, types = _raw_pairs(snap)
        maps = it_glls_maps(src, dst, types)
        scores = _score_maps(src, dst, maps, types)
        auc = compute_auc(y, scores)
        per_t[str(t)] = auc
        if t >= TRAIN_END:
            post_y.append(y)
            post_s.append(scores)
    return {
        "post_auc": compute_auc(torch.cat(post_y), torch.cat(post_s)),
        "per_timestep_auc": per_t,
    }


def run_classical(seed: int, drift: str, jump_mode: str) -> Dict[str, Any]:
    _set_seed(seed)
    raw = generate_temporal_dataset(
        _drift_params(drift, jump_mode),
        NOISE_CONFIGS["clean"],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    post_dev = _so_deviation(raw[TRAIN_END]) if TRAIN_END < len(raw) else float("nan")
    pre_dev = _so_deviation(raw[max(TRAIN_END - 1, 0)])
    out = {
        "seed": seed,
        "drift": drift,
        "jump_mode": jump_mode,
        "so_deviation_pre": pre_dev,
        "so_deviation_post": post_dev,
        "edge_oracle": evaluate_classical(raw, "oracle", False),
        "edge_proc_all": evaluate_classical(raw, "proc", False),
        "edge_lstsq_all": evaluate_classical(raw, "lstsq", False),
        "edge_it_glls": _it_glls_classical(raw),
        "node_agg_proc": _node_agg_proc(raw),
    }
    print(
        f"  [{jump_mode} seed={seed}] "
        f"SO-dev pre={pre_dev:.3f} post={post_dev:.3f} | "
        f"oracle={out['edge_oracle']['post_auc']:.3f} "
        f"pair-SO={out['edge_proc_all']['post_auc']:.3f} "
        f"pair-GL={out['edge_lstsq_all']['post_auc']:.3f} "
        f"IT-GLLS={out['edge_it_glls']['post_auc']:.3f} "
        f"node-agg={out['node_agg_proc']['post_auc']:.3f}"
    )
    return out


def run_aggregated_cayley(
    seed: int,
    drift: str,
    jump_mode: str,
    device: str,
    quick: bool,
) -> Dict[str, Any]:
    _set_seed(seed)
    config = BenchmarkConfig(seed=seed, device=device)
    train_ts, test_ts = get_temporal_split(config.num_timesteps)
    snapshots = load_temporal_dataset(
        "clean", drift, seed=seed, jump_mode=jump_mode
    )
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
    out = {
        "seed": seed,
        "drift": drift,
        "jump_mode": jump_mode,
        "post_auc": sum(post) / max(len(post), 1),
        "per_timestep_auc": detail["per_timestep_auc"],
        "fusion_gates": detail["fusion_gates"],
    }
    print(
        f"  [Cayley {jump_mode} seed={seed}] post={out['post_auc']:.3f}"
    )
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / max(n, 1)
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [float(r[key]["post_auc"] if isinstance(r[key], dict) else r[key]) for r in rows]
    mean, std = _mean_std(vals)
    return {"mean": mean, "std": std, "n": len(vals), "values": vals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--drift", type=str, default="single_jump")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["additive", "group"],
        help="Jump modes to run (additive is the existing DGP).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/group_jump_dgp.json",
    )
    args = parser.parse_args()

    classical: List[Dict[str, Any]] = []
    neural: List[Dict[str, Any]] = []
    for mode in args.modes:
        print(f"\n==== jump_mode={mode} ====")
        for seed in args.seeds:
            classical.append(run_classical(seed, args.drift, mode))
        if not args.skip_neural:
            for seed in args.seeds:
                neural.append(
                    run_aggregated_cayley(
                        seed, args.drift, mode, args.device, args.quick
                    )
                )

    summary: Dict[str, Any] = {}
    for mode in args.modes:
        c_rows = [r for r in classical if r["jump_mode"] == mode]
        n_rows = [r for r in neural if r["jump_mode"] == mode]
        block = {
            "so_deviation_post": _mean_std([r["so_deviation_post"] for r in c_rows]),
            "edge_oracle": summarize(c_rows, "edge_oracle"),
            "edge_proc_all": summarize(c_rows, "edge_proc_all"),
            "edge_lstsq_all": summarize(c_rows, "edge_lstsq_all"),
            "edge_it_glls": summarize(c_rows, "edge_it_glls"),
            "node_agg_proc": summarize(c_rows, "node_agg_proc"),
        }
        if n_rows:
            block["aggregated_cayley"] = summarize(n_rows, "post_auc")
        summary[mode] = block

    payload = {
        "protocol": (
            "Leak-free. Additive jump is the original DGP (may leave SO(d)). "
            "Group jump is Cayley left-multiply (R stays near SO(d)). "
            "Pair estimators see raw (x_u,x_v); Cayley sheaf sees aggregated nodes."
        ),
        "summary": summary,
        "classical": classical,
        "neural": neural,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

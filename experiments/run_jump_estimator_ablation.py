"""Jump estimator ablation: Exp 1 / 1b / 2.

Separates fusion capacity from unsupervised operator estimation.

Exp 1   Leak-free V3.5 + force-handoff of SO(d) Procrustes fitted on
        oracle-compliant pairs (stalk space). Tests whether Cayley fusion
        and the scoring head can ingest a correct jump.

Exp 1b  Classical controls, no neural net. Fit on raw generator pairs:
        - SO(d) Procrustes on compliant pairs only
        - unconstrained GL(d) least squares on compliant pairs
        - two-map planted oracle
        If 1b Procrustes stays ~0.58 after the jump, SO(d) is the wrong
        model class and IRLS cannot save it.

Exp 2   Unlabeled robust Procrustes (IRLS, RANSAC) on all raw pairs,
        scored classically and force-handoff'd into leak-free V3.5.

Leak-free protocol: train t<0.5, no jump BCE, no GT handoff.
Default grid: clean / single_jump, plus slow_smooth as a sanity check
for 1b. Classical runs are cheap and use seeds 42-46; neural handoff
defaults to seed 42.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

from run_procrustes_baseline import (  # noqa: E402
    compute_auc,
    orthogonal_procrustes,
    polar_so,
    unconstrained_lstsq,
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
from temporal_sheaf_v35 import (  # noqa: E402
    SheafConfig,
    TemporalSheafV35,
    TrainConfig,
)
from train_v35 import train_temporal_v35  # noqa: E402

TRAIN_END = int(NUM_TIMESTEPS * 0.5)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _type_ids(snap: Dict[str, Any]) -> Tensor:
    names: List[str] = list(snap["edge_types"])
    parts = []
    for i, etype in enumerate(names):
        n_e = int(snap["edge_index"][etype].shape[1])
        parts.append(torch.full((n_e,), i, dtype=torch.long))
    return torch.cat(parts, dim=0)


def _raw_pairs(snap: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    src = snap["edge_src_feats"].float()
    dst = snap["edge_tgt_feats"].float()
    y = snap["labels"].float()
    clean = snap["clean_labels"].float()
    types = _type_ids(snap)
    return src, dst, y, clean, types


def _score_maps(src: Tensor, dst: Tensor, rot: Tensor, types: Tensor) -> Tensor:
    scores = torch.zeros(src.shape[0], dtype=src.dtype)
    for r in types.unique().tolist():
        mask = types == r
        delta = src[mask] @ rot[r].T - dst[mask]
        scores[mask] = delta.norm(dim=1)
    return scores


def _score_oracle(
    src: Tensor,
    dst: Tensor,
    rho_src: Tensor,
    rho_tgt: Tensor,
    types: Tensor,
) -> Tensor:
    scores = torch.zeros(src.shape[0], dtype=src.dtype)
    for r in types.unique().tolist():
        mask = types == r
        delta = src[mask] @ rho_src[r].T - dst[mask] @ rho_tgt[r].T
        scores[mask] = delta.norm(dim=1)
    return scores


def _true_relative(snap: Dict[str, Any]) -> Tensor:
    """Polar factor of pinv(ρ_tgt) ρ_src, one map per type."""
    rho_src = snap["rho_src"].float()
    rho_tgt = snap["rho_tgt"].float()
    maps = []
    for r in range(rho_src.shape[0]):
        rel = torch.linalg.pinv(rho_tgt[r]) @ rho_src[r]
        maps.append(polar_so(rel))
    return torch.stack(maps)


def _fro_err(fitted: Tensor, target: Tensor) -> float:
    return float((fitted - target).square().mean().sqrt().item())


def irls_procrustes(
    src: Tensor,
    dst: Tensor,
    n_iter: int = 20,
) -> Tensor:
    """Huber-IRLS special-orthogonal Procrustes."""
    if src.shape[0] < 2:
        return torch.eye(src.shape[1], dtype=src.dtype)
    rot = orthogonal_procrustes(src, dst)
    for _ in range(n_iter):
        resid = (src @ rot.T - dst).norm(dim=1)
        scale = resid.median().clamp(min=1e-6)
        cutoff = 1.345 * scale
        weights = torch.ones_like(resid)
        large = resid > cutoff
        weights[large] = cutoff / resid[large].clamp(min=1e-8)
        sw = weights.sqrt().unsqueeze(1)
        rot = orthogonal_procrustes(src * sw, dst * sw)
    return rot


def ransac_procrustes(
    src: Tensor,
    dst: Tensor,
    n_iter: int = 96,
    min_sample: Optional[int] = None,
) -> Tensor:
    """RANSAC SO(d) Procrustes; refit on inliers of the best model."""
    n_pts, dim = src.shape
    if n_pts < 2:
        return torch.eye(dim, dtype=src.dtype)
    sample_n = min_sample or max(dim, 8)
    sample_n = min(sample_n, n_pts)
    best_inliers = -1
    best_mask = torch.ones(n_pts, dtype=torch.bool)
    best_rot = orthogonal_procrustes(src, dst)
    for _ in range(n_iter):
        idx = torch.randint(0, n_pts, (sample_n,))
        rot = orthogonal_procrustes(src[idx], dst[idx])
        resid = (src @ rot.T - dst).norm(dim=1)
        thresh = (resid.median() * 1.5).clamp(min=1e-6)
        mask = resid < thresh
        n_in = int(mask.sum().item())
        if n_in > best_inliers:
            best_inliers = n_in
            best_mask = mask
            best_rot = rot
    if int(best_mask.sum().item()) >= 2:
        return orthogonal_procrustes(src[best_mask], dst[best_mask])
    return best_rot


def fit_type_maps(
    src: Tensor,
    dst: Tensor,
    types: Tensor,
    kind: str,
    compliant: Optional[Tensor] = None,
) -> Tensor:
    """Fit one map per edge type.

    kind: proc | lstsq | irls | ransac
    If ``compliant`` is set (1 = anomalous), keep rows with compliant < 0.5.
    """
    n_types = int(types.max().item()) + 1
    dim = src.shape[1]
    maps = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    for r in range(n_types):
        mask = types == r
        if compliant is not None:
            mask = mask & (compliant < 0.5)
        if int(mask.sum()) < 2:
            continue
        src_r, dst_r = src[mask], dst[mask]
        if kind == "proc":
            maps[r] = orthogonal_procrustes(src_r, dst_r)
        elif kind == "lstsq":
            maps[r] = unconstrained_lstsq(src_r, dst_r)
        elif kind == "irls":
            maps[r] = irls_procrustes(src_r, dst_r)
        elif kind == "ransac":
            maps[r] = ransac_procrustes(src_r, dst_r)
        else:
            raise ValueError(f"unknown kind {kind}")
    return maps


def evaluate_classical(
    dataset: List[Dict[str, Any]],
    kind: str,
    use_compliant_mask: bool,
) -> Dict[str, Any]:
    """Independent-per-snapshot estimator; report pre/post AUC."""
    per_t: Dict[str, float] = {}
    fro: Dict[str, float] = {}
    post_y: List[Tensor] = []
    post_s: List[Tensor] = []
    pre_y: List[Tensor] = []
    pre_s: List[Tensor] = []

    for t, snap in enumerate(dataset):
        src, dst, y, clean, types = _raw_pairs(snap)
        if kind == "oracle":
            scores = _score_oracle(
                src, dst, snap["rho_src"].float(), snap["rho_tgt"].float(), types
            )
            fitted = _true_relative(snap)
        else:
            mask = clean if use_compliant_mask else None
            fitted = fit_type_maps(src, dst, types, kind, compliant=mask)
            scores = _score_maps(src, dst, fitted, types)
        target = _true_relative(snap)
        fro[str(t)] = _fro_err(polar_so(fitted[0]), target[0]) if fitted.shape[-1] == target.shape[-1] else float("nan")
        auc = compute_auc(y, scores)
        per_t[str(t)] = auc
        if t >= TRAIN_END:
            post_y.append(y)
            post_s.append(scores)
        else:
            pre_y.append(y)
            pre_s.append(scores)

    return {
        "pre_auc": compute_auc(torch.cat(pre_y), torch.cat(pre_s)),
        "post_auc": compute_auc(torch.cat(post_y), torch.cat(post_s)),
        "per_timestep_auc": per_t,
        "frobenius_to_true_polar_type0": fro,
    }


CLASSICAL_VARIANTS = [
    ("edge_oracle", "oracle", False),
    ("edge_proc_all", "proc", False),
    ("edge_proc_compliant", "proc", True),
    ("edge_lstsq_compliant", "lstsq", True),
    ("edge_irls_all", "irls", False),
    ("edge_ransac_all", "ransac", False),
]


def run_classical(
    seeds: List[int],
    drifts: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for drift in drifts:
            print(f"\n=== classical seed={seed} {drift} ===")
            raw = generate_temporal_dataset(
                DRIFT_SCENARIOS[drift],
                NOISE_CONFIGS["clean"],
                num_timesteps=NUM_TIMESTEPS,
                seed=seed,
            )
            for name, kind, compliant in CLASSICAL_VARIANTS:
                result = evaluate_classical(raw, kind, compliant)
                row = {
                    "seed": seed,
                    "drift": drift,
                    "variant": name,
                    **result,
                }
                rows.append(row)
                print(
                    f"  {name:24s} pre={result['pre_auc']:.3f} "
                    f"post={result['post_auc']:.3f} "
                    f"t4={result['per_timestep_auc']['4']:.3f} "
                    f"t5={result['per_timestep_auc']['5']:.3f}"
                )
    return rows


def _stalk_pairs(
    snap: Dict[str, Tensor],
    model: Any,
    device: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = snap["x"].to(device)
    with torch.no_grad():
        stalk = model.input_proj(x)
    ei = snap["edge_index"].to(device)
    src = stalk[ei[0]]
    dst = stalk[ei[1]]
    types = snap["edge_types"].to(device)
    clean = snap["clean_labels"].to(device)
    return src, dst, types, clean


def make_stalk_provider(
    kind: str,
    use_compliant_mask: bool,
    split: int,
    device: str,
) -> Callable[[int, Dict[str, Tensor], Any], Optional[Tensor]]:
    """Fit algebraic maps in stalk space from the jump onward."""

    def provider(t: int, snap: Dict[str, Tensor], model: Any) -> Optional[Tensor]:
        if t < split:
            return None
        src, dst, types, clean = _stalk_pairs(snap, model, device)
        mask = clean if use_compliant_mask else None
        maps = fit_type_maps(src.cpu(), dst.cpu(), types.cpu(), kind, compliant=mask.cpu() if mask is not None else None)
        return maps.to(device)

    return provider


def leak_free_train_config() -> TrainConfig:
    return TrainConfig(
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


def _summarize_eval(detail: Dict[str, Any], test_ts: List[int]) -> Dict[str, Any]:
    post = [detail["per_timestep_auc"][f"t_{t}"] for t in test_ts]
    return {
        "post_auc": sum(post) / max(len(post), 1),
        "per_timestep_auc": detail["per_timestep_auc"],
        "fusion_gates": detail["fusion_gates"],
        "hard_handoffs": detail["hard_handoffs"],
    }


def run_neural(
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

    train_cfg = leak_free_train_config()
    if quick:
        train_cfg.epochs_phase1 = 2
        train_cfg.epochs_phase3 = 2

    print(f"\n=== neural leak-free seed={seed} {drift} ===")
    static_model = train_static_sheaf(
        snapshots, config, train_ts, num_epochs=8 if quick else 200
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

    baseline = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        jump_times=[],
        hard_handoff=False,
    )
    exp1 = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        jump_times=[],
        hard_handoff=False,
        external_rho_provider=make_stalk_provider(
            "proc", True, TRAIN_END, device
        ),
    )
    exp2_irls = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        jump_times=[],
        hard_handoff=False,
        external_rho_provider=make_stalk_provider(
            "irls", False, TRAIN_END, device
        ),
    )
    exp2_ransac = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        jump_times=[],
        hard_handoff=False,
        external_rho_provider=make_stalk_provider(
            "ransac", False, TRAIN_END, device
        ),
    )

    out = {
        "seed": seed,
        "drift": drift,
        "leak_free": _summarize_eval(baseline, test_ts),
        "exp1_compliant_proc_handoff": _summarize_eval(exp1, test_ts),
        "exp2_irls_handoff": _summarize_eval(exp2_irls, test_ts),
        "exp2_ransac_handoff": _summarize_eval(exp2_ransac, test_ts),
    }
    for name, block in (
        ("leak_free", out["leak_free"]),
        ("exp1_compliant_proc", out["exp1_compliant_proc_handoff"]),
        ("exp2_irls", out["exp2_irls_handoff"]),
        ("exp2_ransac", out["exp2_ransac_handoff"]),
    ):
        print(f"  {name:24s} post={block['post_auc']:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip leak-free V3.5 training",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Skip classical pair estimators",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classical-seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=str,
        default="results/jump_estimator_ablation.json",
    )
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "protocol": (
            "Exp1: leak-free V3.5 + compliant-pair Procrustes handoff in "
            "stalk space from t>=0.5. Exp1b: classical pair estimators "
            "(compliant vs all, SO(d) vs GL(d), oracle two-map). "
            "Exp2: unlabeled IRLS/RANSAC on all pairs, classical and neural."
        ),
        "classical": [],
        "neural": None,
    }
    out_path = _ROOT / args.out
    if out_path.exists() and (args.neural_only or args.classical_only):
        try:
            payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            pass
    if not args.neural_only:
        drifts = ["slow_smooth", "single_jump"]
        payload["classical"] = run_classical(args.classical_seeds, drifts)
    if not args.classical_only:
        payload["neural"] = run_neural(
            args.seed, "single_jump", args.device, args.quick
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

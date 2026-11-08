"""Table 1b: RMS operator error on the adversarial-type smooth cell.

Classical estimators only. Neural methods (Cayley sheaf, EvolveGCN, TGN,
DualPath) do not emit an evaluable per-type relative map and are not scored.

Operator error is the same RMS used in typed_adversarial_smooth.json:
    sqrt(mean((Rhat - R*)^2))
not the Frobenius sqrt(sum((Rhat - R*)^2)).

Target R* is the polar factor of pinv(ρ_tgt) ρ_src (same as
run_typed_adversarial_smooth). IT-GLLS is also scored against the
unpolarised relative map, which is close on this SO(d) cell.
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
    _true_relative,
)
from run_procrustes_baseline import (  # noqa: E402
    fit_type_maps,
    orthogonal_procrustes,
    pair_features,
)
from run_typed_adversarial_smooth import generate_adversarial  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _raw_relative(snap: Dict[str, Any]) -> Tensor:
    rho_src = snap["rho_src"].float()
    rho_tgt = snap["rho_tgt"].float()
    maps = []
    for r in range(rho_src.shape[0]):
        maps.append(torch.linalg.pinv(rho_tgt[r]) @ rho_src[r])
    return torch.stack(maps)


def _mean_types(fitted: Tensor, target: Tensor) -> float:
    n_types = int(target.shape[0])
    return float(sum(_fro_err(fitted[r], target[r]) for r in range(n_types)) / n_types)


def run_seed(seed: int) -> Dict[str, Any]:
    _set_seed(seed)
    raw = generate_adversarial(seed)
    post: Dict[str, List[float]] = {
        "typed": [],
        "mixed": [],
        "node_agg": [],
        "it_glls_polar": [],
        "it_glls_raw": [],
    }
    per_t: Dict[str, Any] = {}

    for t, snap in enumerate(raw):
        polar_tgt = _true_relative(snap)
        raw_tgt = _raw_relative(snap)
        n_types = int(polar_tgt.shape[0])
        src, dst, _y, _clean, types = _raw_pairs(snap)

        typed = torch.eye(src.shape[1]).unsqueeze(0).repeat(n_types, 1, 1)
        for r in range(n_types):
            mask = types == r
            if int(mask.sum()) >= 2:
                typed[r] = orthogonal_procrustes(src[mask], dst[mask])
        mixed = orthogonal_procrustes(src, dst)
        mixed_maps = mixed.unsqueeze(0).expand(n_types, -1, -1).contiguous()

        src_m, dst_m, _y_m, types_m = pair_features(snap, use_edge_pairs=False)
        node_maps = fit_type_maps(src_m, dst_m, types_m, "proc")
        glls = it_glls_maps(src, dst, types)

        row = {
            "typed": _mean_types(typed, polar_tgt),
            "mixed": _mean_types(mixed_maps, polar_tgt),
            "node_agg": _mean_types(node_maps, polar_tgt),
            "it_glls_polar": _mean_types(glls, polar_tgt),
            "it_glls_raw": _mean_types(glls, raw_tgt),
        }
        per_t[str(t)] = row
        if t >= TRAIN_END:
            for key, val in row.items():
                post[key].append(val)

    out = {
        "seed": seed,
        "post": {k: float(sum(v) / max(len(v), 1)) for k, v in post.items()},
        "per_timestep": per_t,
    }
    print(
        f"  [seed={seed}] typed={out['post']['typed']:.3f} "
        f"mixed={out['post']['mixed']:.3f} "
        f"node-agg={out['post']['node_agg']:.3f} "
        f"IT-GLLS(polar)={out['post']['it_glls_polar']:.3f} "
        f"IT-GLLS(raw)={out['post']['it_glls_raw']:.3f}"
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
    parser.add_argument(
        "--out",
        type=str,
        default="results/operator_recovery_adversarial.json",
    )
    args = parser.parse_args()

    rows = [run_seed(seed) for seed in args.seeds]
    summary = {
        key: _mean_std([r["post"][key] for r in rows])
        for key in rows[0]["post"]
    }
    payload = {
        "protocol": (
            "Adversarial-type smooth DGP (opposing Givens planes). "
            "Leak-free post-split t>=0.5. RMS operator error vs planted R* "
            "(polar of pinv(ρ_tgt) ρ_src), same helper as typed_adversarial_smooth. "
            "Neural models are not scored: they do not emit an evaluable typed map."
        ),
        "metric": "sqrt(mean((Rhat-R*)^2)); not Frobenius sqrt(sum(.))",
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

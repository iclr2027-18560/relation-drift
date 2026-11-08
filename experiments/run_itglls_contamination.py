"""Structured-contamination stress test for untrimmed OLS vs IT-GLLS vs Procrustes.

Paper DGP (independent anomalies): anomalous targets are independent of the
source, so they contribute zero cross-moment and the population OLS target is
(1-α)R, while polar is unchanged.

This DGP (competing linear map): anomalous pairs obey a different linear map
R_alt, with the same source second moment. Polar is then a mixture and is no
longer guaranteed to recover R_star.

Reports compliant-map RMS error and detection AUROC. Not a native-dataset run
and not a heterogeneous TGNN baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_procrustes_baseline import (  # noqa: E402
    compute_auc,
    orthogonal_procrustes,
    unconstrained_lstsq,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_so(dim: int) -> Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim))
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, -1] *= -1
    return q


def givens(dim: int, theta: float) -> Tensor:
    rot = torch.eye(dim)
    c, s = math.cos(theta), math.sin(theta)
    rot[0, 0], rot[1, 1] = c, c
    rot[0, 1], rot[1, 0] = -s, s
    return rot


def rms(fitted: Tensor, target: Tensor) -> float:
    return float((fitted - target).square().mean().sqrt().item())


def residual_scores(src: Tensor, dst: Tensor, rot: Tensor) -> Tensor:
    return (src @ rot.T - dst).norm(dim=1)


def fit_it_glls(src: Tensor, dst: Tensor) -> Tensor:
    types = torch.zeros(src.shape[0], dtype=torch.long)
    return it_glls_maps(src, dst, types)[0]


def make_pairs(
    n: int,
    dim: int,
    alpha: float,
    r_star: Tensor,
    r_alt: Tensor,
    dgp: str,
    noise: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    n_anom = int(round(alpha * n))
    n_ok = n - n_anom
    src = torch.randn(n, dim)
    dst = torch.empty(n, dim)
    y = torch.zeros(n)
    dst[:n_ok] = src[:n_ok] @ r_star.T + noise * torch.randn(n_ok, dim)
    if n_anom == 0:
        return src, dst, y
    y[n_ok:] = 1.0
    src_a = src[n_ok:]
    if dgp == "competing_map":
        dst[n_ok:] = src_a @ r_alt.T + noise * torch.randn(n_anom, dim)
    elif dgp == "independent":
        dst[n_ok:] = 2.0 * torch.randn(n_anom, dim)
    else:
        raise ValueError(dgp)
    perm = torch.randperm(n)
    return src[perm], dst[perm], y[perm]


def run_cell(
    seed: int,
    alpha: float,
    theta_deg: float,
    dgp: str,
    n: int,
    dim: int,
    noise: float,
) -> Dict[str, Any]:
    _set_seed(seed)
    r_star = random_so(dim)
    r_alt = r_star @ givens(dim, math.radians(theta_deg))
    src, dst, y = make_pairs(n, dim, alpha, r_star, r_alt, dgp, noise)
    fits = {
        "ols": unconstrained_lstsq(src, dst),
        "it_glls": fit_it_glls(src, dst),
        "procrustes": orthogonal_procrustes(src, dst),
    }
    row: Dict[str, Any] = {
        "seed": seed,
        "alpha": alpha,
        "theta_deg": theta_deg,
        "dgp": dgp,
    }
    for name, rot in fits.items():
        row[f"{name}_rms"] = rms(rot, r_star)
        row[f"{name}_auc"] = compute_auc(y, residual_scores(src, dst, rot))
    return row


def summarize(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in cells:
        key = (row["dgp"], row["alpha"], row["theta_deg"])
        keys.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    metrics = [
        "ols_rms",
        "it_glls_rms",
        "procrustes_rms",
        "ols_auc",
        "it_glls_auc",
        "procrustes_auc",
    ]
    for (dgp, alpha, theta), rows in sorted(keys.items()):
        entry: Dict[str, Any] = {
            "dgp": dgp,
            "alpha": alpha,
            "theta_deg": theta,
            "n_seeds": len(rows),
        }
        for m in metrics:
            vals = np.array([r[m] for r in rows], dtype=float)
            entry[m + "_mean"] = float(vals.mean())
            entry[m + "_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.25, 0.40]
    )
    parser.add_argument(
        "--thetas", type=float, nargs="*", default=[15.0, 45.0, 90.0]
    )
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument(
        "--out",
        type=str,
        default="results/itglls_contamination.json",
    )
    args = parser.parse_args()

    cells: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for alpha in args.alphas:
            for theta in args.thetas:
                for dgp in ("competing_map", "independent"):
                    if dgp == "independent" and theta != args.thetas[0]:
                        continue
                    row = run_cell(
                        seed, alpha, theta, dgp, args.n, args.dim, args.noise
                    )
                    cells.append(row)
                    print(
                        f"seed={seed} {dgp} α={alpha:.2f} θ={theta:.0f} "
                        f"ols_rms={row['ols_rms']:.3f} glls_rms={row['it_glls_rms']:.3f} "
                        f"proc_rms={row['procrustes_rms']:.3f} "
                        f"glls_auc={row['it_glls_auc']:.3f}"
                    )

    payload = {
        "protocol": {
            "n": args.n,
            "dim": args.dim,
            "noise": args.noise,
            "it_glls_trim": 0.15,
            "it_glls_iters": 3,
            "seeds": args.seeds,
            "note": (
                "competing_map: anomalous pairs use R_alt = R_star Givens(θ), "
                "same source Gram. independent: anomalous targets ~ N(0,4I), "
                "as in the paper generator."
            ),
        },
        "cells": cells,
        "summary": summarize(cells),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

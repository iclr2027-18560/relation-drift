"""MathOverflow SNAP: Gate 0 on lagged activity, then planted typed-map jump.

SNAP a2q/c2q/c2a topology and times. No native pair law is assumed.
Gate 0 uses lagged monthly activity vs a destination shuffle.
The inject plants x_v≈R_r(t) x_u in 12-d and jumps comment-to-answer only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathoverflow_inject import (  # noqa: E402
    AFFECTED_TYPE,
    N_TYPES,
    TYPE_NAMES,
    MonthSnap,
    inject_mathoverflow,
    lagged_activity_features,
    load_mathoverflow_months,
)
from run_elliptic_inject import eval_break, series_delta  # noqa: E402
from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_kaggle_inject import (  # noqa: E402
    account_total_series,
    amount_volume_series,
    new_pair_series,
)


def _source_cond(src: Tensor) -> float:
    if src.shape[0] < 2:
        return float("inf")
    evals = torch.linalg.eigvalsh(src.T @ src)
    return float((evals[-1] / (evals[0].abs() + 1e-12)).item())


def typed_itglls(
    src: Tensor,
    dst: Tensor,
    types: Tensor,
    prev: Tensor | None,
    min_pairs: int,
    max_cond: float,
) -> Tensor:
    dim = int(src.shape[1])
    maps = (
        prev.clone()
        if prev is not None
        else torch.eye(dim).unsqueeze(0).repeat(N_TYPES, 1, 1)
    )
    if src.shape[0] < 2:
        return maps
    fitted = it_glls_maps(src, dst, types)
    for r in range(N_TYPES):
        mask = types == r
        if int(mask.sum()) < min_pairs:
            continue
        if _source_cond(src[mask]) > max_cond:
            continue
        maps[r] = fitted[r]
    return maps


def residual_auc(snap: MonthSnap, maps: Tensor, type_id: int) -> float:
    mask = snap.edge_types == type_id
    if int(mask.sum()) < 4:
        return float("nan")
    y = snap.labels[mask].cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    src, dst = snap.pair_src[mask], snap.pair_dst[mask]
    resid = (dst - (maps[type_id] @ src.T).T).norm(dim=1).cpu().numpy()
    return float(roc_auc_score(y, resid))


def gate0_cell(src: Tensor, dst: Tensor, rng: np.random.Generator) -> Dict[str, float]:
    n, d = int(src.shape[0]), int(src.shape[1])
    if n < d + 2:
        return {"n": n, "too_few": True}
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    fitted, _, _, _ = np.linalg.lstsq(src_np, dst_np, rcond=None)
    pred = src_np @ fitted
    resid = np.sqrt(((pred - dst_np) ** 2).sum(axis=1)).mean()
    dest_rms = float(np.sqrt((dst_np**2).sum(axis=1)).mean() + 1e-8)
    shuf = dst_np[rng.permutation(n)]
    shuf_resid = np.sqrt(((pred - shuf) ** 2).sum(axis=1)).mean()
    ratio_dest = float(resid / dest_rms)
    ratio_shuf = float(resid / (shuf_resid + 1e-8))
    return {
        "n": n,
        "too_few": False,
        "rmse": float(resid),
        "rmse_over_dest_rms": ratio_dest,
        "rmse_over_shuffle": ratio_shuf,
        "pass_dest": bool(ratio_dest < 0.25),
        "pass_shuffle": bool(ratio_shuf < 0.50),
        "passed": bool(ratio_dest < 0.25 and ratio_shuf < 0.50),
    }


def run_gate0(snaps: Sequence[MonthSnap], train_end: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells: Dict[str, Any] = {}
    eval_snaps = [s for i, s in enumerate(snaps) if i > train_end]
    if not eval_snaps:
        eval_snaps = list(snaps[1:])
    for r, name in enumerate(TYPE_NAMES):
        src = torch.cat([s.native_src[s.edge_types == r] for s in eval_snaps], 0)
        dst = torch.cat([s.native_dst[s.edge_types == r] for s in eval_snaps], 0)
        cells[name] = gate0_cell(src, dst, rng)
    src = torch.cat([s.native_src for s in eval_snaps], 0)
    dst = torch.cat([s.native_dst for s in eval_snaps], 0)
    cells["pooled"] = gate0_cell(src, dst, rng)
    passed = bool(cells["pooled"].get("passed")) or (
        sum(1 for n in TYPE_NAMES if cells[n].get("passed")) >= 2
    )
    return {
        "stalk": "lagged monthly log1p in/out degree and per-type send/receive counts",
        "pass_rule": "held-out RMSE < 0.25 dest RMS and < 0.50 destination shuffle",
        "cells": cells,
        "passed": passed,
        "open_plant_native": False,
        "note": (
            "SNAP MathOverflow has no user features. This Gate 0 is lagged activity "
            "only. A fail means native activity is not a typed linear pair law. "
            "The inject below is a planted subspace on real topology, as in IBM HI-Small."
        ),
    }


def write_plot(
    path: Path,
    months: Sequence[str],
    delta_glls: Dict[str, List[float]],
    baselines: Dict[str, List[float]],
    t_star_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def zscore(vals: List[float]) -> List[float]:
        arr = np.array(vals, dtype=float)
        tail = arr[1:] if len(arr) > 2 else arr
        mu = float(np.mean(tail))
        sd = float(np.std(tail)) or 1.0
        return [float((v - mu) / sd) for v in vals]

    xs = list(range(len(months)))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    colors = {"a2q": "#6b7280", "c2q": "#9ca3af", "c2a": "#1d4ed8"}
    for name, series in delta_glls.items():
        lw = 2.4 if name == TYPE_NAMES[AFFECTED_TYPE] else 1.1
        ax.plot(xs, zscore(series), label=name, color=colors.get(name, "#111"), linewidth=lw)
    styles = {
        "interaction_volume": ("#a16207", ":"),
        "user_total": ("#b45309", "--"),
        "new_pair": ("#111827", "-."),
    }
    for name, series in baselines.items():
        c, ls = styles.get(name, ("#444", ":"))
        ax.plot(xs, zscore(series), label=name, color=c, linestyle=ls, linewidth=1.3)
    ax.axvline(t_star_index, color="#b91c1c", linestyle=":", linewidth=1.2, label=r"$t^*$")
    ax.set_xlabel("Month index (MathOverflow SNAP)")
    ax.set_ylabel("z-scored month-to-month change")
    ax.set_title("Planted c2a map jump; users and typed edges unchanged")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = Path(args.data)
    if not data.is_absolute():
        data = _ROOT / data
    months, by_month, user_to_i, _raw = load_mathoverflow_months(
        data,
        n_snaps=args.n_snaps,
        min_per_type=args.min_per_type,
        max_pairs_per_type=args.max_pairs,
        seed=args.seed,
    )
    n_users = len(user_to_i)
    native_feats = lagged_activity_features(months, by_month, n_users)
    t_star_index = int(args.t_star_index)
    train_end = int(args.train_end)
    snaps, _proj, true_R = inject_mathoverflow(
        months,
        by_month,
        n_users,
        native_feats,
        feat_dim=args.feat_dim,
        proj_dim=args.dim,
        t_star_index=t_star_index,
        anomaly_frac=args.anomaly_frac,
        noise=args.noise,
        omega=args.omega,
        jump_mag=args.jump_mag,
        seed=args.seed,
    )
    type_counts = {
        TYPE_NAMES[r]: int(sum(int((s.edge_types == r).sum()) for s in snaps))
        for r in range(N_TYPES)
    }
    print(f"  months={months} t*={months[t_star_index]} types={type_counts}")

    gate0 = run_gate0(snaps, train_end, args.seed)
    print("  Gate 0 passed=", gate0["passed"], gate0["cells"]["pooled"])

    min_pairs = max(4 * args.dim, args.min_pairs)
    glls_by_t: List[Tensor] = []
    prev = None
    for snap in snaps:
        prev = typed_itglls(
            snap.pair_src, snap.pair_dst, snap.edge_types, prev, min_pairs, args.max_cond
        )
        glls_by_t.append(prev)
    delta_glls = {
        TYPE_NAMES[r]: series_delta([m[r] for m in glls_by_t]) for r in range(N_TYPES)
    }
    baselines = {
        "interaction_volume": amount_volume_series(snaps),
        "user_total": account_total_series(snaps),
        "new_pair": new_pair_series(snaps),
    }
    times = list(range(len(months)))
    eval_start = train_end + 1
    tests_glls = {
        name: eval_break(series, times, t_star_index, eval_start)
        for name, series in delta_glls.items()
    }
    tests_base = {
        name: eval_break(series, times, t_star_index, eval_start)
        for name, series in baselines.items()
    }
    aff = TYPE_NAMES[AFFECTED_TYPE]
    una = [n for n in TYPE_NAMES if n != aff]
    ratio = tests_glls[aff]["delta_at_tstar"] / max(
        float(np.mean([tests_glls[n]["delta_at_tstar"] for n in una])), 1e-8
    )
    pre_idx = [i for i in times if i <= train_end]
    post_idx = [i for i in times if i >= t_star_index]
    pre_glls = torch.stack([glls_by_t[i] for i in pre_idx]).mean(0)
    aucs: Dict[str, Any] = {}
    for r, name in enumerate(TYPE_NAMES):
        scores = [residual_auc(snaps[i], pre_glls, r) for i in post_idx]
        aucs[name] = {"itglls_frozen_pre": float(np.nanmean(scores))}

    plot_path = _ROOT / "results" / "mathoverflow_inject_delta.png"
    write_plot(plot_path, months, delta_glls, baselines, t_star_index)

    success = {
        "itglls_unique_max_at_tstar": bool(tests_glls[aff]["unique_max"]),
        "itglls_ratio_ge_3": bool(ratio >= 3.0),
        "volume_not_unique_max": not bool(tests_base["interaction_volume"]["unique_max"]),
        "user_total_not_unique_max": not bool(tests_base["user_total"]["unique_max"]),
        "new_pair_not_unique_max": not bool(tests_base["new_pair"]["unique_max"]),
        "affected_auc_ge_0.80": bool(aucs[aff]["itglls_frozen_pre"] >= 0.80),
    }
    success["all"] = all(success.values())
    return {
        "protocol": (
            "MathOverflow SNAP (Paranjape et al.). Typed temporal edges a2q/c2q/c2a. "
            "Gate 0: lagged monthly activity, no plant. Inject: random user features, "
            "planted x_v≈R_r(t) x_u in 12-d, additive jump on c2a only. Counterparties "
            "unchanged. Not native friend-semantics drift."
        ),
        "months": months,
        "t_star_month": months[t_star_index],
        "t_star_index": t_star_index,
        "train_end": train_end,
        "n_users": n_users,
        "type_counts": type_counts,
        "gate0": gate0,
        "delta_itglls": delta_glls,
        "delta_baselines": baselines,
        "break_itglls": tests_glls,
        "break_baselines": tests_base,
        "type_selective_ratio_itglls": ratio,
        "post_inject_auc_frozen_pre": aucs,
        "success": success,
        "plot": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/mathoverflow")
    parser.add_argument("--n-snaps", type=int, default=12)
    parser.add_argument("--min-per-type", type=int, default=100)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--feat-dim", type=int, default=16)
    parser.add_argument("--t-star-index", type=int, default=7)
    parser.add_argument("--train-end", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--anomaly-frac", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.03)
    parser.add_argument("--jump-mag", type=float, default=0.8)
    parser.add_argument("--min-pairs", type=int, default=80)
    parser.add_argument("--max-cond", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/mathoverflow_inject.json")
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")
    print(
        json.dumps(
            {
                "t_star": payload["t_star_month"],
                "months": payload["months"],
                "type_counts": payload["type_counts"],
                "gate0_passed": payload["gate0"]["passed"],
                "gate0_pooled": payload["gate0"]["cells"]["pooled"],
                "break_itglls": payload["break_itglls"],
                "break_baselines": payload["break_baselines"],
                "ratio_itglls": payload["type_selective_ratio_itglls"],
                "auc": payload["post_inject_auc_frozen_pre"],
                "success": payload["success"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

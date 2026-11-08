"""Test A: planted typed-map jump on IBM HI-Small topology.

Does not change amounts or counterparties. Baselines that watch money or
new names should not uniquely peak at t*. Pair IT-GLLS should.
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

from kaggle_inject import (  # noqa: E402
    AFFECTED_TYPE,
    N_TYPES,
    TYPE_NAMES,
    DaySnap,
    inject_ibm,
    load_ibm_days,
)
from run_elliptic_inject import eval_break, fro_delta, series_delta  # noqa: E402
from run_gld_residual_v35 import it_glls_maps  # noqa: E402


def _source_cond(src: Tensor) -> float:
    if src.shape[0] < 2:
        return float("inf")
    evals = torch.linalg.eigvalsh(src.T @ src)
    return float((evals[-1] / (evals[0].abs() + 1e-12)).item())


def typed_itglls(snap: DaySnap, prev: Tensor | None, min_pairs: int, max_cond: float) -> Tensor:
    dim = int(snap.pair_src.shape[1])
    maps = (
        prev.clone()
        if prev is not None
        else torch.eye(dim).unsqueeze(0).repeat(N_TYPES, 1, 1)
    )
    if snap.pair_src.shape[0] < 2:
        return maps
    fitted = it_glls_maps(snap.pair_src, snap.pair_dst, snap.edge_types)
    for r in range(N_TYPES):
        mask = snap.edge_types == r
        if int(mask.sum()) < min_pairs:
            continue
        if _source_cond(snap.pair_src[mask]) > max_cond:
            continue
        maps[r] = fitted[r]
    return maps


def residual_auc(snap: DaySnap, maps: Tensor, type_id: int) -> float:
    mask = snap.edge_types == type_id
    if int(mask.sum()) < 4:
        return float("nan")
    y = snap.labels[mask].cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    src, dst = snap.pair_src[mask], snap.pair_dst[mask]
    resid = (dst - (maps[type_id] @ src.T).T).norm(dim=1).cpu().numpy()
    return float(roc_auc_score(y, resid))


def amount_volume_series(snaps: Sequence[DaySnap]) -> List[float]:
    vols = [float(s.amounts.sum().item()) for s in snaps]
    return [0.0] + [abs(vols[t] - vols[t - 1]) for t in range(1, len(vols))]


def account_total_series(snaps: Sequence[DaySnap]) -> List[float]:
    """Day-to-day change of per-sender amount totals (shared senders)."""
    out = [0.0]
    prev: Dict[int, float] = {}
    for t, snap in enumerate(snaps):
        cur: Dict[int, float] = {}
        src = snap.src_ids.cpu().numpy()
        amt = snap.amounts.cpu().numpy()
        for i, a in zip(src, amt):
            cur[int(i)] = cur.get(int(i), 0.0) + float(a)
        if t == 0:
            prev = cur
            continue
        keys = set(prev) | set(cur)
        delta = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        out.append(delta / max(len(keys), 1))
        prev = cur
    return out


def new_pair_series(snaps: Sequence[DaySnap]) -> List[float]:
    seen: set[tuple[int, int]] = set()
    out: List[float] = []
    for snap in snaps:
        src = snap.src_ids.cpu().numpy()
        dst = snap.dst_ids.cpu().numpy()
        pairs = list(zip(src.tolist(), dst.tolist()))
        if not pairs:
            out.append(0.0)
            continue
        new = sum(1 for p in pairs if p not in seen)
        out.append(new / len(pairs))
        seen.update(pairs)
    # first day is all-new; convert to day-to-day |diff| for eval_break
    return [0.0] + [abs(out[t] - out[t - 1]) for t in range(1, len(out))]


def write_plot(
    path: Path,
    dates: Sequence[str],
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

    xs = list(range(len(dates)))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    colors = {
        "ACH": "#1d4ed8",
        "Wire": "#6b7280",
        "Cheque": "#9ca3af",
        "Credit Card": "#d1d5db",
        "Other": "#e5e7eb",
    }
    for name, series in delta_glls.items():
        lw = 2.4 if name == TYPE_NAMES[AFFECTED_TYPE] else 1.1
        ax.plot(xs, zscore(series), label=name, color=colors.get(name, "#111"), linewidth=lw)
    styles = {
        "amount_volume": ("#a16207", ":"),
        "account_total": ("#b45309", "--"),
        "new_pair": ("#111827", "-."),
    }
    for name, series in baselines.items():
        c, ls = styles.get(name, ("#444", ":"))
        ax.plot(xs, zscore(series), label=name, color=c, linestyle=ls, linewidth=1.3)
    ax.axvline(t_star_index, color="#b91c1c", linestyle=":", linewidth=1.2, label=r"$t^*$")
    ax.set_xlabel("Day index (IBM HI-Small)")
    ax.set_ylabel("z-scored day-to-day change")
    ax.set_title("Test A: planted ACH map jump; amounts/names unchanged")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = Path(args.data)
    if not data.is_absolute():
        data = _ROOT / data
    dates, by_day, acct_to_i = load_ibm_days(
        data,
        max_pairs_per_type=args.max_pairs,
        min_day_txs=args.min_day_txs,
        seed=args.seed,
    )
    t_star_index = int(args.t_star_index)
    if t_star_index < 0 or t_star_index >= len(dates):
        raise ValueError(f"t_star_index {t_star_index} out of {len(dates)} days")
    train_end = int(args.train_end)
    snaps, _proj, true_R = inject_ibm(
        dates,
        by_day,
        n_accounts=len(acct_to_i),
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
    print(f"  days={len(dates)} t*={dates[t_star_index]} types={type_counts}")

    min_pairs = max(4 * args.dim, args.min_pairs)
    glls_by_t: List[Tensor] = []
    prev = None
    for snap in snaps:
        prev = typed_itglls(snap, prev, min_pairs, args.max_cond)
        glls_by_t.append(prev)
    delta_glls = {
        TYPE_NAMES[r]: series_delta([m[r] for m in glls_by_t]) for r in range(N_TYPES)
    }
    true_delta = {
        TYPE_NAMES[r]: series_delta([true_R[t, r] for t in range(len(dates))])
        for r in range(N_TYPES)
    }
    baselines = {
        "amount_volume": amount_volume_series(snaps),
        "account_total": account_total_series(snaps),
        "new_pair": new_pair_series(snaps),
    }

    # eval_break wants integer times; use day indices
    times = list(range(len(dates)))
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

    plot_path = _ROOT / "results" / "kaggle_inject_delta.png"
    write_plot(plot_path, dates, delta_glls, baselines, t_star_index)

    success = {
        "itglls_unique_max_at_tstar": bool(tests_glls[aff]["unique_max"]),
        "itglls_ratio_ge_3": bool(ratio >= 3.0),
        "amount_volume_not_unique_max": not bool(tests_base["amount_volume"]["unique_max"]),
        "account_total_not_unique_max": not bool(tests_base["account_total"]["unique_max"]),
        "new_pair_not_unique_max": not bool(tests_base["new_pair"]["unique_max"]),
        "affected_auc_ge_0.80": bool(aucs[aff]["itglls_frozen_pre"] >= 0.80),
    }
    success["all"] = all(success.values())

    return {
        "protocol": (
            "Test A. IBM HI-Small topology, calendar days, payment-kind types. "
            "Random account features; planted x_v≈R_r(t) x_u in 12-d. Additive "
            "jump on ACH only. Amounts and counterparties unchanged. Not a claim "
            "that IBM natively has relational-semantics drift or that labels are fraud."
        ),
        "dates": dates,
        "t_star_date": dates[t_star_index],
        "t_star_index": t_star_index,
        "train_end": train_end,
        "type_counts": type_counts,
        "delta_itglls": delta_glls,
        "delta_true_R": true_delta,
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
    parser.add_argument("--data", type=str, default="data/kaggle/HI-Small_Trans.csv")
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--feat-dim", type=int, default=16)
    parser.add_argument("--t-star-index", type=int, default=6)
    parser.add_argument("--train-end", type=int, default=4)
    parser.add_argument("--min-day-txs", type=int, default=10000)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--anomaly-frac", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.03)
    parser.add_argument("--jump-mag", type=float, default=0.8)
    parser.add_argument("--min-pairs", type=int, default=80)
    parser.add_argument("--max-cond", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/kaggle_inject.json")
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
                "t_star": payload["t_star_date"],
                "type_counts": payload["type_counts"],
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

"""Elliptic++ typed-constraint inject (real topology, planted ρ).

Pre-registered protocol:
  Train t=1..25, inject t*=30 on hub_to_leaf only, eval t=26..49.
  Degree-ratio typing from topology. Plant in a 12-dim projected subspace.

Typed Δ_r is polar(S_r) / IT-GLLS on raw projected pairs — the identifying
estimator in the paper. Implemented V3.5 coboundary uses a single ρ and is
not scored for type-selective Δ (same limitation as Table 1b).
EvolveGCN is the global W_t control on the same projected node features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "run"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from elliptic_inject import (  # noqa: E402
    AFFECTED_TYPE,
    N_TYPES,
    TYPE_NAMES,
    InjectedSnapshot,
    inject_sequence,
    load_elliptic_snapshots,
)
from run_evolvegcn_baseline import DEVICE, EvolveGCNModel  # noqa: E402
from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_procrustes_baseline import orthogonal_procrustes, polar_so  # noqa: E402


def fro_delta(a: Tensor, b: Tensor) -> float:
    return float((a - b).square().sum().sqrt().item())


def fit_polar(src: Tensor, dst: Tensor, dim: int) -> Tensor:
    if src.shape[0] < 2:
        return torch.eye(dim)
    cov = (dst.T @ src) / float(src.shape[0])
    return polar_so(cov + 1e-4 * torch.eye(dim))


def _source_cond(src: Tensor) -> float:
    if src.shape[0] < 2:
        return float("inf")
    cov = src.T @ src
    evals = torch.linalg.eigvalsh(cov)
    return float((evals[-1] / (evals[0].abs() + 1e-12)).item())


def typed_maps(
    snap: InjectedSnapshot,
    kind: str,
    prev: Tensor | None,
    min_pairs: int,
    max_cond: float,
) -> Tensor:
    dim = snap.pair_src.shape[1] if snap.pair_src.numel() else 12
    maps = prev.clone() if prev is not None else torch.eye(dim).unsqueeze(0).repeat(N_TYPES, 1, 1)
    if snap.pair_src.shape[0] < 2:
        return maps
    if kind == "itglls":
        fitted = it_glls_maps(snap.pair_src, snap.pair_dst, snap.edge_types)
        for r in range(N_TYPES):
            mask = snap.edge_types == r
            if int(mask.sum()) < min_pairs:
                continue
            if _source_cond(snap.pair_src[mask]) > max_cond:
                continue
            maps[r] = fitted[r]
        return maps
    for r in range(N_TYPES):
        mask = snap.edge_types == r
        if int(mask.sum()) < min_pairs:
            continue
        if _source_cond(snap.pair_src[mask]) > max_cond:
            continue
        if kind == "proc":
            maps[r] = orthogonal_procrustes(snap.pair_src[mask], snap.pair_dst[mask])
        else:
            maps[r] = fit_polar(snap.pair_src[mask], snap.pair_dst[mask], dim)
    return maps


def mixed_map(snap: InjectedSnapshot) -> Tensor:
    dim = snap.pair_src.shape[1] if snap.pair_src.numel() else 12
    if snap.pair_src.shape[0] < 2:
        return torch.eye(dim)
    return orthogonal_procrustes(snap.pair_src, snap.pair_dst)


def residual_auc(snap: InjectedSnapshot, maps: Tensor, type_id: int) -> float:
    mask = snap.edge_types == type_id
    if int(mask.sum()) < 4:
        return float("nan")
    y = snap.labels[mask].cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    src = snap.pair_src[mask]
    dst = snap.pair_dst[mask]
    rho = maps[type_id]
    resid = (dst - (rho @ src.T).T).norm(dim=1).cpu().numpy()
    return float(roc_auc_score(y, resid))


def series_delta(maps: Sequence[Tensor]) -> List[float]:
    out = [0.0]
    for t in range(1, len(maps)):
        out.append(fro_delta(maps[t], maps[t - 1]))
    return out


def eval_break(
    delta: List[float],
    times: Sequence[int],
    t_star: int,
    eval_start: int,
) -> Dict[str, Any]:
    idx = times.index(t_star)
    eval_idx = [i for i, t in enumerate(times) if t >= eval_start and i > 0]
    eval_vals = [delta[i] for i in eval_idx]
    at = delta[idx]
    rest = [delta[i] for i in eval_idx if i != idx]
    rank = 1 + sum(1 for v in eval_vals if v > at + 1e-12)
    unique_max = rank == 1 and sum(1 for v in eval_vals if abs(v - at) < 1e-8) == 1
    return {
        "delta_at_tstar": at,
        "mean_rest_eval": float(np.mean(rest)) if rest else float("nan"),
        "z_vs_eval": (
            (at - float(np.mean(rest))) / (float(np.std(rest)) + 1e-8) if rest else float("nan")
        ),
        "rank": rank,
        "n_eval": len(eval_vals),
        "rank_p": rank / max(len(eval_vals), 1),
        "unique_max": unique_max,
    }


def evolvegcn_delta(
    snaps: List[InjectedSnapshot],
    times: Sequence[int],
    train_end: int,
    epochs: int,
    seed: int,
) -> List[float]:
    torch.manual_seed(seed)
    dim = int(snaps[0].x_proj.shape[1])
    max_n = max(int(s.x_proj.shape[0]) for s in snaps)
    padded = []
    for snap in snaps:
        n = snap.x_proj.shape[0]
        x = torch.zeros(max_n, dim)
        x[:n] = snap.x_proj
        ei = snap.edge_index
        if ei.numel():
            ei = torch.cat([ei, ei.flip(0)], dim=1)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        padded.append({"x": x, "ei": ei, "n": n})

    train_end_i = max(1, times.index(train_end) + 1 if train_end in times else len(times) // 2)
    model = EvolveGCNModel(
        num_nodes=max_n,
        in_channels=dim,
        hidden_dim=min(32, dim),
        num_layers=1,
        variant="H",
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    weights: List[Tensor] = [torch.zeros(1) for _ in snaps]
    for epoch in range(epochs):
        model.reinitialize_weights()
        last = epoch == epochs - 1
        for t in range(train_end_i):
            x = padded[t]["x"].to(DEVICE)
            struct = padded[t]["ei"].to(DEVICE)
            if struct.shape[1] == 0:
                continue
            n_neg = min(256, max(struct.shape[1], 8))
            neg = torch.stack(
                [
                    torch.randint(0, padded[t]["n"], (n_neg,)),
                    torch.randint(0, padded[t]["n"], (n_neg,)),
                ],
                dim=0,
            ).to(DEVICE)
            query = torch.cat([struct, neg], dim=1)
            y = torch.cat(
                [
                    torch.ones(struct.shape[1], device=DEVICE),
                    torch.zeros(n_neg, device=DEVICE),
                ]
            )
            opt.zero_grad()
            logits = model.forward_timestep(x, struct, query)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            model.detach_weights()
            if last:
                w = model.layers[0].weight
                if w is not None:
                    weights[t] = w.detach().cpu().squeeze(0).clone()

    model.eval()
    with torch.no_grad():
        for t in range(train_end_i, len(snaps)):
            x = padded[t]["x"].to(DEVICE)
            struct = padded[t]["ei"].to(DEVICE)
            dummy = struct if struct.shape[1] else torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
            _ = model.forward_timestep(x, struct if struct.shape[1] else dummy, dummy)
            w = model.layers[0].weight
            if w is not None:
                weights[t] = w.detach().cpu().squeeze(0).clone()
    return series_delta(weights)


def write_plot(
    path: Path,
    times: Sequence[int],
    delta_typed: Dict[str, List[float]],
    delta_mixed: List[float],
    delta_egcn: List[float],
    t_star: int,
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

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = {"hub_to_leaf": "#1d4ed8", "peer": "#6b7280", "leaf_to_hub": "#9ca3af"}
    for name, series in delta_typed.items():
        lw = 2.3 if name == TYPE_NAMES[AFFECTED_TYPE] else 1.2
        ax.plot(times, zscore(series), label=name, color=colors.get(name, "#111"), linewidth=lw)
    ax.plot(times, zscore(delta_mixed), label="mixed Procrustes", color="#a16207", linestyle=":", linewidth=1.4)
    ax.plot(times, zscore(delta_egcn), label=r"EvolveGCN $\|\Delta W\|_F$", color="#111827", linestyle="--", linewidth=1.4)
    ax.axvline(t_star, color="#b91c1c", linestyle=":", linewidth=1.2, label=rf"$t^*={t_star}$")
    ax.axvline(43, color="#6d28d9", linestyle="-.", linewidth=1.0, label="market $t=43$")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"z-scored $\Delta_r(t)=\|\hat\rho_r(t)-\hat\rho_r(t-1)\|_F$")
    ax.set_title("Planted hub-to-leaf jump on Elliptic topology")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _ROOT / data_dir
    print(f"  loading Elliptic from {data_dir}")
    times, xs, eidx = load_elliptic_snapshots(data_dir)
    print(f"  {len(times)} timesteps, feat_dim={xs[0].shape[1]}")
    snaps, _proj, true_R = inject_sequence(
        times,
        xs,
        eidx,
        proj_dim=args.dim,
        t_star=args.t_star,
        train_end=args.train_end,
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
    print(f"  type counts {type_counts}")

    polar_by_t: List[Tensor] = []
    glls_by_t: List[Tensor] = []
    prev_p = prev_g = None
    min_pairs = max(4 * args.dim, args.min_pairs)
    max_cond = args.max_cond
    for snap in snaps:
        prev_p = typed_maps(snap, "polar", prev_p, min_pairs, max_cond)
        prev_g = typed_maps(snap, "itglls", prev_g, min_pairs, max_cond)
        polar_by_t.append(prev_p)
        glls_by_t.append(prev_g)
    mixed_by_t = [mixed_map(s) for s in snaps]

    delta_polar = {
        TYPE_NAMES[r]: series_delta([m[r] for m in polar_by_t]) for r in range(N_TYPES)
    }
    delta_glls = {
        TYPE_NAMES[r]: series_delta([m[r] for m in glls_by_t]) for r in range(N_TYPES)
    }
    delta_mixed = series_delta(mixed_by_t)
    true_delta = {
        TYPE_NAMES[r]: series_delta([true_R[t, r] for t in range(len(times))])
        for r in range(N_TYPES)
    }

    print("  fitting EvolveGCN global W_t")
    delta_egcn = evolvegcn_delta(
        snaps, times, args.train_end, epochs=args.egcn_epochs, seed=args.seed
    )

    t_star = args.t_star
    eval_start = args.train_end + 1
    tests_polar = {
        name: eval_break(series, times, t_star, eval_start)
        for name, series in delta_polar.items()
    }
    tests_glls = {
        name: eval_break(series, times, t_star, eval_start)
        for name, series in delta_glls.items()
    }
    tests_mixed = eval_break(delta_mixed, times, t_star, eval_start)
    tests_egcn = eval_break(delta_egcn, times, t_star, eval_start)

    aff = TYPE_NAMES[AFFECTED_TYPE]
    una = [n for n in TYPE_NAMES if n != aff]
    ratio_polar = tests_polar[aff]["delta_at_tstar"] / max(
        float(np.mean([tests_polar[n]["delta_at_tstar"] for n in una])), 1e-8
    )
    ratio_glls = tests_glls[aff]["delta_at_tstar"] / max(
        float(np.mean([tests_glls[n]["delta_at_tstar"] for n in una])), 1e-8
    )

    # Frozen pre-inject maps (train window) for residual AUC after t*.
    pre_idx = [i for i, t in enumerate(times) if t <= args.train_end]
    post_idx = [i for i, t in enumerate(times) if t >= t_star]
    pre_polar = torch.stack([polar_by_t[i] for i in pre_idx]).mean(0)
    pre_glls = torch.stack([glls_by_t[i] for i in pre_idx]).mean(0)
    aucs: Dict[str, Any] = {}
    for r, name in enumerate(TYPE_NAMES):
        polar_scores = [residual_auc(snaps[i], pre_polar, r) for i in post_idx]
        glls_scores = [residual_auc(snaps[i], pre_glls, r) for i in post_idx]
        aucs[name] = {
            "polar_frozen_pre": float(np.nanmean(polar_scores)),
            "itglls_frozen_pre": float(np.nanmean(glls_scores)),
        }

    plot_path = _ROOT / "results" / "elliptic_inject_delta.png"
    write_plot(plot_path, times, delta_glls, delta_mixed, delta_egcn, t_star)

    success = {
        "itglls_unique_max_at_tstar": bool(tests_glls[aff]["unique_max"]),
        "itglls_ratio_ge_3": bool(ratio_glls >= 3.0),
        "evolvegcn_not_unique_max": not bool(tests_egcn["unique_max"]),
        "affected_auc_ge_0.80": bool(aucs[aff]["itglls_frozen_pre"] >= 0.80),
        "polar_unique_max_at_tstar": bool(tests_polar[aff]["unique_max"]),
        "polar_ratio_ge_3": bool(ratio_polar >= 3.0),
    }
    success["all"] = all(
        [
            success["itglls_unique_max_at_tstar"],
            success["itglls_ratio_ge_3"],
            success["evolvegcn_not_unique_max"],
            success["affected_auc_ge_0.80"],
        ]
    )

    return {
        "protocol": (
            "Real Elliptic topology and 165-dim features. Degree-ratio typing "
            "(no labels). Planted typed constraint in a 12-dim projected "
            "subspace; additive jump on hub_to_leaf at t*=30 (before market "
            "shift t=43). Typed Δ_r from IT-GLLS on raw pairs with carry-forward "
            "when a type has fewer than min_pairs (additive jump leaves SO(d)). "
            "V3.5 coboundary is a single ρ and is not used for type-selective Δ. "
            "Planted constraint is synthetic; Elliptic is not claimed to natively "
            "exhibit typed relational-semantics drift."
        ),
        "times": list(times),
        "t_star": t_star,
        "train_end": args.train_end,
        "type_counts": type_counts,
        "delta_polar": delta_polar,
        "delta_itglls": delta_glls,
        "delta_mixed_procrustes": delta_mixed,
        "delta_evolvegcn": delta_egcn,
        "delta_true_R": true_delta,
        "break_polar": tests_polar,
        "break_itglls": tests_glls,
        "break_mixed": tests_mixed,
        "break_evolvegcn": tests_egcn,
        "type_selective_ratio_polar": ratio_polar,
        "type_selective_ratio_itglls": ratio_glls,
        "post_inject_auc_frozen_pre": aucs,
        "success": success,
        "plot": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/elliptic")
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--t-star", type=int, default=30)
    parser.add_argument("--train-end", type=int, default=25)
    parser.add_argument("--anomaly-frac", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.03)
    parser.add_argument("--jump-mag", type=float, default=0.8)
    parser.add_argument("--min-pairs", type=int, default=80)
    parser.add_argument("--max-cond", type=float, default=1e5)
    parser.add_argument("--egcn-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/elliptic_inject.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")
    print(json.dumps({
        "type_counts": payload["type_counts"],
        "break_polar": payload["break_polar"],
        "break_itglls": payload["break_itglls"],
        "break_mixed": payload["break_mixed"],
        "break_evolvegcn": payload["break_evolvegcn"],
        "ratio_polar": payload["type_selective_ratio_polar"],
        "ratio_itglls": payload["type_selective_ratio_itglls"],
        "auc": payload["post_inject_auc_frozen_pre"],
        "success": payload["success"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

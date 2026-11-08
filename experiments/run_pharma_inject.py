"""Test P-A: planted typed-map jump on a public protein contact graph.

Coordinates (hence RMSD / RMSF / contacts) are not changed by the plant
beyond tiny thermal noise. Leftover should uniquely peak on the jumped
contact type and name that type. This is a mechanism test, not biology.
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

from pharma_inject import (  # noqa: E402
    AFFECTED_TYPE,
    CONTACT_CUTOFF,
    N_TYPES,
    PDB_ID,
    TYPE_NAMES,
    FrameSnap,
    fetch_pdb,
    inject_protein,
    load_protein_graph,
)
from run_elliptic_inject import eval_break, fro_delta, series_delta  # noqa: E402
from run_gld_residual_v35 import it_glls_maps  # noqa: E402


def _source_cond(src: Tensor) -> float:
    if src.shape[0] < 2:
        return float("inf")
    evals = torch.linalg.eigvalsh(src.T @ src)
    return float((evals[-1] / (evals[0].abs() + 1e-12)).item())


def typed_itglls(snap: FrameSnap, prev: Tensor | None, min_pairs: int, max_cond: float) -> Tensor:
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


def residual_auc(snap: FrameSnap, maps: Tensor, type_id: int) -> float:
    mask = snap.edge_types == type_id
    if int(mask.sum()) < 4:
        return float("nan")
    y = snap.labels[mask].cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    src, dst = snap.pair_src[mask], snap.pair_dst[mask]
    resid = (dst - (maps[type_id] @ src.T).T).norm(dim=1).cpu().numpy()
    return float(roc_auc_score(y, resid))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    cov = ac.T @ bc
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    rot = ac @ r
    return float(np.sqrt(np.mean(np.sum((rot - bc) ** 2, axis=1))))


def rmsd_series(snaps: Sequence[FrameSnap]) -> List[float]:
    ref = snaps[0].ca.cpu().numpy()
    vals = [kabsch_rmsd(s.ca.cpu().numpy(), ref) for s in snaps]
    return [0.0] + [abs(vals[t] - vals[t - 1]) for t in range(1, len(vals))]


def rmsf_series(snaps: Sequence[FrameSnap], window: int = 3) -> List[float]:
    """Mean per-residue fluctuation in a short window; then |diff|."""
    coords = [s.ca.cpu().numpy() for s in snaps]
    out = [0.0]
    for t in range(1, len(coords)):
        lo = max(0, t - window + 1)
        block = np.stack(coords[lo : t + 1], axis=0)
        rmsf = np.sqrt(np.mean(np.var(block, axis=0), axis=1)).mean()
        prev_lo = max(0, t - window)
        prev = np.stack(coords[prev_lo:t], axis=0)
        prev_rmsf = np.sqrt(np.mean(np.var(prev, axis=0), axis=1)).mean()
        out.append(abs(float(rmsf) - float(prev_rmsf)))
    return out


def residue_motion_series(snaps: Sequence[FrameSnap]) -> List[float]:
    out = [0.0]
    prev = snaps[0].ca.cpu().numpy()
    for t in range(1, len(snaps)):
        cur = snaps[t].ca.cpu().numpy()
        out.append(float(np.linalg.norm(cur - prev, axis=1).mean()))
        prev = cur
    return out


def _contact_set(ca: np.ndarray, cutoff: float = CONTACT_CUTOFF) -> set[tuple[int, int]]:
    n = ca.shape[0]
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(ca[i] - ca[j])) < cutoff:
                pairs.add((i, j))
    return pairs


def contact_count_series(snaps: Sequence[FrameSnap]) -> List[float]:
    counts = [float(len(_contact_set(s.ca.cpu().numpy()))) for s in snaps]
    return [0.0] + [abs(counts[t] - counts[t - 1]) for t in range(1, len(counts))]


def new_contact_series(snaps: Sequence[FrameSnap]) -> List[float]:
    out = [0.0]
    prev = _contact_set(snaps[0].ca.cpu().numpy())
    for t in range(1, len(snaps)):
        cur = _contact_set(snaps[t].ca.cpu().numpy())
        new = len(cur - prev)
        out.append(new / max(len(cur), 1))
        prev = cur
    return out


def localization(delta_glls: Dict[str, List[float]], t_star: int) -> Dict[str, Any]:
    at = {name: series[t_star] for name, series in delta_glls.items()}
    ranked = sorted(at.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "argmax_type_at_tstar": ranked[0][0],
        "values_at_tstar": at,
        "identifies_affected_type": ranked[0][0] == TYPE_NAMES[AFFECTED_TYPE],
    }


def write_plot(
    path: Path,
    n_steps: int,
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

    xs = list(range(n_steps))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    colors = {
        "backbone": "#9ca3af",
        "short_range": "#6b7280",
        "long_range": "#1d4ed8",
        "distal": "#d1d5db",
    }
    for name, series in delta_glls.items():
        lw = 2.4 if name == TYPE_NAMES[AFFECTED_TYPE] else 1.1
        ax.plot(xs, zscore(series), label=name, color=colors.get(name, "#111"), linewidth=lw)
    styles = {
        "ca_rmsd": ("#a16207", ":"),
        "rmsf": ("#b45309", "--"),
        "contact_count": ("#111827", "-."),
        "new_contact": ("#7c2d12", ":"),
        "residue_motion": ("#92400e", "--"),
    }
    for name, series in baselines.items():
        c, ls = styles.get(name, ("#444", ":"))
        ax.plot(xs, zscore(series), label=name, color=c, linestyle=ls, linewidth=1.3)
    ax.axvline(t_star_index, color="#b91c1c", linestyle=":", linewidth=1.2, label=r"$t^*$")
    ax.set_xlabel("Frame index (crystal + thermal noise)")
    ax.set_ylabel("z-scored frame-to-frame change")
    ax.set_title("Test P-A: planted long-range map jump; coordinates unchanged")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache = Path(args.pdb_cache)
    if not cache.is_absolute():
        cache = _ROOT / cache
    pdb_path = fetch_pdb(args.pdb_id, cache)
    graph = load_protein_graph(pdb_path, chain=args.chain)
    type_counts = {
        TYPE_NAMES[r]: int((graph["types"] == r).sum()) for r in range(N_TYPES)
    }
    n_res = int(graph["ca"].shape[0])
    print(f"  pdb={args.pdb_id} chain={args.chain} residues={n_res} types={type_counts}")

    t_star_index = int(args.t_star_index)
    train_end = int(args.train_end)
    snaps, _proj, true_R, gate0 = inject_protein(
        graph,
        n_steps=args.n_steps,
        t_star_index=t_star_index,
        feat_dim=args.feat_dim,
        proj_dim=args.dim,
        anomaly_frac=args.anomaly_frac,
        noise=args.noise,
        omega=args.omega,
        jump_mag=args.jump_mag,
        coord_noise=args.coord_noise,
        seed=args.seed,
    )

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
        TYPE_NAMES[r]: series_delta([true_R[t, r] for t in range(args.n_steps)])
        for r in range(N_TYPES)
    }
    baselines = {
        "ca_rmsd": rmsd_series(snaps),
        "rmsf": rmsf_series(snaps),
        "contact_count": contact_count_series(snaps),
        "new_contact": new_contact_series(snaps),
        "residue_motion": residue_motion_series(snaps),
    }

    times = list(range(args.n_steps))
    eval_start = train_end + 1
    tests_glls = {
        name: eval_break(series, times, t_star_index, eval_start)
        for name, series in delta_glls.items()
    }
    tests_base = {
        name: eval_break(series, times, t_star_index, eval_start)
        for name, series in baselines.items()
    }
    loc = localization(delta_glls, t_star_index)

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

    plot_path = _ROOT / "results" / "pharma_inject_delta.png"
    write_plot(plot_path, args.n_steps, delta_glls, baselines, t_star_index)

    gate0_native_ok = any(
        bool(v.get("usable_linear_pair_law")) for v in gate0.values()
    )

    success = {
        "itglls_unique_max_at_tstar": bool(tests_glls[aff]["unique_max"]),
        "itglls_ratio_ge_3": bool(ratio >= 3.0),
        "identifies_affected_type": bool(loc["identifies_affected_type"]),
        "ca_rmsd_not_unique_max": not bool(tests_base["ca_rmsd"]["unique_max"]),
        "rmsf_not_unique_max": not bool(tests_base["rmsf"]["unique_max"]),
        "contact_count_not_unique_max": not bool(tests_base["contact_count"]["unique_max"]),
        "new_contact_not_unique_max": not bool(tests_base["new_contact"]["unique_max"]),
        "residue_motion_not_unique_max": not bool(
            tests_base["residue_motion"]["unique_max"]
        ),
        "affected_auc_ge_0.80": bool(aucs[aff]["itglls_frozen_pre"] >= 0.80),
    }
    success["all"] = all(success.values())

    return {
        "protocol": (
            "Test P-A (mechanism only). Public crystal 1UBQ chain A. Contact types "
            "are sequence-separation + Cα distance: backbone (|i-j|=1); short_range "
            "(2≤|i-j|≤4, d<8Å); long_range (|i-j|≥5, d<8Å, jumped); distal "
            "(|i-j|≥5, 8≤d<12Å). Random residue features; planted x_v≈R_r(t) x_u "
            "in 12-d. Additive jump on long_range only. Cα coordinates get thermal "
            "noise only; RMSD/RMSF/contacts are not rewritten by the plant. "
            "Success is (i) leftover unique max on the jumped type and (ii) leftover "
            "names that type at t* — relationship-specific information, not 'beat "
            "RMSD at RMSD's global task'. Not a native conformational transition, "
            "not allostery, not drug design. Gate 0 on native Cα endpoints is "
            "reported separately and is not a P-A pass condition."
        ),
        "pdb_id": args.pdb_id,
        "pdb_path": str(pdb_path),
        "n_residues": n_res,
        "n_steps": args.n_steps,
        "t_star_index": t_star_index,
        "train_end": train_end,
        "type_definition": {
            "backbone": "|i-j|=1",
            "short_range": f"2<=|i-j|<={4}, Cα<{CONTACT_CUTOFF}Å",
            "long_range": f"|i-j|>=5, Cα<{CONTACT_CUTOFF}Å (jumped)",
            "distal": f"|i-j|>=5, {CONTACT_CUTOFF}<=Cα<{12}Å",
        },
        "type_counts": type_counts,
        "gate0_native_ca_linear_map": gate0,
        "gate0_any_usable_linear_pair_law": gate0_native_ok,
        "delta_itglls": delta_glls,
        "delta_true_R": true_delta,
        "delta_baselines": baselines,
        "break_itglls": tests_glls,
        "break_baselines": tests_base,
        "localization": loc,
        "type_selective_ratio_itglls": ratio,
        "post_inject_auc_frozen_pre": aucs,
        "success": success,
        "plot": str(plot_path),
        "non_claim": (
            "Does not show a real ligand- or mutant-induced transition (P-B), "
            "does not show drug-discovery utility (P-C). Native Cα endpoints are "
            "not a usable shared linear pair law (Gate 0); P-A therefore plants a "
            "synthetic subspace, same as the IBM inject."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-id", type=str, default=PDB_ID)
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--pdb-cache", type=str, default="data/pdb")
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--feat-dim", type=int, default=16)
    parser.add_argument("--t-star-index", type=int, default=7)
    parser.add_argument("--train-end", type=int, default=4)
    parser.add_argument("--anomaly-frac", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.03)
    parser.add_argument("--jump-mag", type=float, default=0.8)
    parser.add_argument("--coord-noise", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--max-cond", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/pharma_inject.json")
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
                "pdb": payload["pdb_id"],
                "n_residues": payload["n_residues"],
                "type_counts": payload["type_counts"],
                "gate0_any_usable_linear_pair_law": payload["gate0_any_usable_linear_pair_law"],
                "break_itglls": payload["break_itglls"],
                "break_baselines": payload["break_baselines"],
                "localization": payload["localization"],
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

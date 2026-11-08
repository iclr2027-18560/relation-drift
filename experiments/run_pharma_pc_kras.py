"""Test P-C: KRAS G12C GDP vs AMG 510 — localize cryptic-pocket interface change.

Win is not AUC and not 'hinge highest'. Win: pair-local leftover marks
Switch I–II / SII-P contacts more than CORE contacts, and is not the same
ranking as global Cα displacement, |Δdistance|, or floppy termini.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adk_pb import (  # noqa: E402
    align_intersection,
    contact_pairs,
    high_b_set,
    kabsch_displace,
    pair_leftover,
    residue_scores,
    termini_set,
)
from kras_pc import (  # noqa: E402
    CRYPTIC_SUBPOCKET,
    POCKET,
    SWITCH_I,
    SWITCH_II,
    is_core_pair,
    is_pocket_pair,
    is_si_sii_pair,
    kras_region,
)
from pharma_inject import backbone_frames, fetch_pdb, parse_backbone  # noqa: E402


def _mean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def pair_distance_change(ca_a: np.ndarray, ca_h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    da = np.linalg.norm(ca_a[src] - ca_a[dst], axis=1)
    dh = np.linalg.norm(ca_h[src] - ca_h[dst], axis=1)
    return np.abs(dh - da)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache = Path(args.pdb_cache)
    if not cache.is_absolute():
        cache = _ROOT / cache
    apo = parse_backbone(fetch_pdb(args.apo, cache).read_text(errors="replace"), chain=args.chain)
    holo = parse_backbone(fetch_pdb(args.holo, cache).read_text(errors="replace"), chain=args.chain)
    aln = align_intersection(apo, holo)
    resseq = aln["resseq"]
    # Drop expression-tag residue numbers ≤ 0 (6OIM has Gly 0 / His-tag)
    keep_num = resseq >= 1
    if int(keep_num.sum()) < 50:
        raise ValueError("too few positive-numbered residues")
    idx = np.where(keep_num)[0]
    resseq = resseq[idx]
    for key in ("ca_apo", "ca_holo", "n_apo", "n_holo", "c_apo", "c_holo", "b_apo", "b_holo"):
        aln[key] = aln[key][idx]
    aln["resseq"] = resseq

    frames_a = backbone_frames(aln["n_apo"], aln["ca_apo"], aln["c_apo"])
    frames_h = backbone_frames(aln["n_holo"], aln["ca_holo"], aln["c_holo"])
    src, dst = contact_pairs(aln["ca_apo"], aln["ca_holo"])
    leftover = pair_leftover(frames_a, frames_h, aln["ca_apo"], aln["ca_holo"], src, dst)
    dchg = pair_distance_change(aln["ca_apo"], aln["ca_holo"], src, dst)

    term = termini_set(resseq)
    hib = high_b_set(resseq, np.fmax(aln["b_apo"], aln["b_holo"]), q=args.b_quantile)
    # Termini always dropped. High B is dropped only outside the pre-registered
    # functional sites: Switch II is high-B by nature; masking it would delete
    # the cryptic-pocket hypothesis (Reviewer-2), not "floppy undruggable termini".
    functional = {
        int(r)
        for r in resseq
        if kras_region(int(r)) in ("pocket", "switch_i", "switch_ii", "p_loop")
    }
    hib_drop = hib - functional
    drop = term | hib_drop
    keep_res = np.array([int(r) not in drop for r in resseq])
    keep_pair = keep_res[src] & keep_res[dst]

    combined = leftover["combined"]
    res_left = residue_scores(len(resseq), src, dst, combined, keep_pair)
    res_dist = residue_scores(len(resseq), src, dst, dchg, keep_pair)
    glob = kabsch_displace(aln["ca_apo"], aln["ca_holo"])
    core_mask = np.array([kras_region(int(r)) == "core" for r in resseq])
    core_disp = kabsch_displace(aln["ca_apo"], aln["ca_holo"], mask=core_mask)

    def pair_mean(pred) -> float:
        vals = [
            float(combined[k])
            for k, (s, d) in enumerate(zip(src, dst))
            if keep_pair[k] and pred(int(resseq[s]), int(resseq[d]))
        ]
        return _mean(vals)

    pair_groups = {
        "si_sii": pair_mean(is_si_sii_pair),
        "pocket": pair_mean(is_pocket_pair),
        "core": pair_mean(is_core_pair),
    }
    pair_n = {
        name: int(
            sum(
                bool(keep_pair[k] and pred(int(resseq[src[k]]), int(resseq[dst[k]])))
                for k in range(len(src))
            )
        )
        for name, pred in (
            ("si_sii", is_si_sii_pair),
            ("pocket", is_pocket_pair),
            ("core", is_core_pair),
        )
    }

    by_region: Dict[str, Dict[str, float]] = {}
    for name in ("pocket", "switch_i", "switch_ii", "p_loop", "core"):
        ix = [
            i
            for i, r in enumerate(resseq)
            if kras_region(int(r)) == name and keep_res[i] and np.isfinite(res_left[i])
        ]
        by_region[name] = {
            "n": float(len(ix)),
            "leftover_mean": _mean([float(res_left[i]) for i in ix]),
            "global_disp_mean": _mean([float(glob[i]) for i in ix]),
            "core_aligned_disp_mean": _mean([float(core_disp[i]) for i in ix]),
            "abs_distance_change_mean": _mean([float(res_dist[i]) for i in ix]),
        }

    finite = keep_res & np.isfinite(res_left) & np.isfinite(glob)
    spear_glob = (
        float(
            np.corrcoef(
                np.argsort(np.argsort(res_left[finite])),
                np.argsort(np.argsort(glob[finite])),
            )[0, 1]
        )
        if int(finite.sum()) >= 8
        else float("nan")
    )
    finite_d = keep_res & np.isfinite(res_left) & np.isfinite(res_dist)
    spear_dist = (
        float(
            np.corrcoef(
                np.argsort(np.argsort(res_left[finite_d])),
                np.argsort(np.argsort(res_dist[finite_d])),
            )[0, 1]
        )
        if int(finite_d.sum()) >= 8
        else float("nan")
    )

    pocket_l = by_region["pocket"]["leftover_mean"]
    core_l = by_region["core"]["leftover_mean"]
    pocket_g = by_region["pocket"]["global_disp_mean"]
    core_g = by_region["core"]["global_disp_mean"]
    si_sii_l = pair_groups["si_sii"]
    core_pair_l = pair_groups["core"]

    # Localization vs baselines: leftover enrichment at pocket vs core, while
    # global displacement enrichment at pocket vs core is smaller or leftover
    # is not a copy of displacement/distance ranks.
    leftover_enrich = pocket_l / max(core_l, 1e-8)
    disp_enrich = pocket_g / max(core_g, 1e-8)

    success = {
        "si_sii_leftover_gt_core_pairs": bool(si_sii_l > core_pair_l) if np.isfinite(si_sii_l) else False,
        "pocket_leftover_gt_core_residues": bool(pocket_l > core_l),
        "leftover_not_same_rank_as_global_disp": bool(abs(spear_glob) < 0.7),
        "leftover_not_same_rank_as_distance_change": bool(abs(spear_dist) < 0.7),
        "leftover_enriches_pocket_more_than_global_disp": bool(leftover_enrich > disp_enrich),
    }
    success["all"] = all(success.values())

    plot_path = _ROOT / "results" / "pharma_pc_kras.png"
    _write_plot(plot_path, resseq, res_left, glob, keep_res)

    return {
        "protocol": (
            "Test P-C (application branch, not the paper leftover). Human KRAS "
            "G12C: GDP-bound 4L8G vs AMG 510 complex 6OIM. Identifying object is "
            "per-contact relative pose in the sender local frame (apo vs holo). "
            "Pre-registered: Switch I 25–40, Switch II 58–72, cryptic subpocket "
            "H95/Y96/Q99, SII-P set {12,58,59,60,61,64,68,69,72,92,95,96,99}. "
            "Termini and high B-factor dropped. No whole-protein superposition in "
            "leftover. Win: leftover marks SI–SII / pocket interactions more than "
            "CORE, and is not the same ranking as global Cα displacement or "
            "|Δdistance|. Not hinge-highest. Not 'predicts which drugs work'. "
            "Not typed shared R_r (Gate 0 abandoned for this branch)."
        ),
        "apo": args.apo,
        "holo": args.holo,
        "n_shared_residues": int(len(resseq)),
        "n_contacts": int(src.shape[0]),
        "n_contacts_kept": int(keep_pair.sum()),
        "n_dropped_termini": len(term),
        "n_dropped_high_b_nonfunctional": len(hib_drop),
        "n_high_b_kept_functional": len(hib & functional),
        "pre_registered": {
            "switch_i": [25, 40],
            "switch_ii": [58, 72],
            "cryptic_subpocket": sorted(CRYPTIC_SUBPOCKET),
            "pocket": sorted(POCKET),
        },
        "pair_leftover_by_group": pair_groups,
        "pair_n_by_group": pair_n,
        "by_region": by_region,
        "leftover_pocket_over_core": leftover_enrich,
        "global_disp_pocket_over_core": disp_enrich,
        "spearman_leftover_vs_global_disp": spear_glob,
        "spearman_leftover_vs_distance_change": spear_dist,
        "success": success,
        "plot": str(plot_path),
        "non_claim": (
            "Does not claim allostery detection as the paper result, does not "
            "rank ligands (P-D), does not restore Gate 0 typed maps."
        ),
    }


def _write_plot(path: Path, resseq, leftover, glob, keep) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.asarray(resseq)
    left = np.asarray(leftover, dtype=float)
    g = np.asarray(glob, dtype=float)
    m = np.asarray(keep) & np.isfinite(left)
    ax.plot(
        x[m],
        left[m] / (np.nanmax(left[m]) + 1e-8),
        label="pair leftover (local frames)",
        color="#1d4ed8",
    )
    ax.plot(
        x[m],
        g[m] / (np.nanmax(g[m]) + 1e-8),
        label="Cα disp after global superposition",
        color="#a16207",
        linestyle=":",
    )
    ax.axvspan(25, 40, color="#fde68a", alpha=0.3, label="Switch I")
    ax.axvspan(58, 72, color="#bfdbfe", alpha=0.3, label="Switch II")
    for r in sorted(POCKET):
        ax.axvline(r, color="#9ca3af", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Residue (human KRAS)")
    ax.set_ylabel("normalized score")
    ax.set_title("P-C KRAS: local pair leftover vs global displacement")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apo", type=str, default="4L8G")
    parser.add_argument("--holo", type=str, default="6OIM")
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--pdb-cache", type=str, default="data/pdb")
    parser.add_argument("--b-quantile", type=float, default=0.90)
    parser.add_argument("--out", type=str, default="results/pharma_pc_kras.json")
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")
    print(
        json.dumps(
            {
                "pdbs": [payload["apo"], payload["holo"]],
                "n_shared": payload["n_shared_residues"],
                "pair_n": payload["pair_n_by_group"],
                "pair_leftover": payload["pair_leftover_by_group"],
                "by_region": payload["by_region"],
                "enrich": {
                    "leftover": payload["leftover_pocket_over_core"],
                    "global_disp": payload["global_disp_pocket_over_core"],
                },
                "spearman": {
                    "vs_disp": payload["spearman_leftover_vs_global_disp"],
                    "vs_dist": payload["spearman_leftover_vs_distance_change"],
                },
                "success": payload["success"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

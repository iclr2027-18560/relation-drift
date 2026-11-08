"""Test P-B: AdK apo 4AKE vs holo 1AKE, per-contact leftover vs own past.

Expert protocol: no typed shared R_r; local sender frames only; no whole-protein
superposition inside leftover. Hinges pre-registered. Filter termini and high B.
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
    CORE_RANGES,
    HINGE_RES,
    LID_RANGE,
    NMP_RANGE,
    align_intersection,
    contact_pairs,
    high_b_set,
    kabsch_displace,
    pair_leftover,
    region_of,
    residue_scores,
    termini_set,
)
from pharma_inject import backbone_frames, fetch_pdb, parse_backbone  # noqa: E402


def _mean(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache = Path(args.pdb_cache)
    if not cache.is_absolute():
        cache = _ROOT / cache
    apo_path = fetch_pdb(args.apo, cache)
    holo_path = fetch_pdb(args.holo, cache)
    apo = parse_backbone(apo_path.read_text(errors="replace"), chain=args.chain)
    holo = parse_backbone(holo_path.read_text(errors="replace"), chain=args.chain)
    aln = align_intersection(apo, holo)
    resseq = aln["resseq"]
    frames_a = backbone_frames(aln["n_apo"], aln["ca_apo"], aln["c_apo"])
    frames_h = backbone_frames(aln["n_holo"], aln["ca_holo"], aln["c_holo"])
    src, dst = contact_pairs(aln["ca_apo"], aln["ca_holo"])
    leftover = pair_leftover(
        frames_a, frames_h, aln["ca_apo"], aln["ca_holo"], src, dst
    )

    term = termini_set(resseq)
    hib = high_b_set(resseq, np.fmax(aln["b_apo"], aln["b_holo"]), q=args.b_quantile)
    drop = term | hib
    keep_res = np.array([int(r) not in drop for r in resseq])
    keep_pair = keep_res[src] & keep_res[dst]

    combined = leftover["combined"]
    res_left = residue_scores(len(resseq), src, dst, combined, keep_pair)
    glob = kabsch_displace(aln["ca_apo"], aln["ca_holo"])
    core_mask = np.array([region_of(int(r)) == "core_interior" for r in resseq])
    core_disp = kabsch_displace(aln["ca_apo"], aln["ca_holo"], mask=core_mask)

    by_region: Dict[str, Dict[str, float]] = {}
    for name in ("hinge", "lid_interior", "nmp_interior", "core_interior"):
        idx = [
            i
            for i, r in enumerate(resseq)
            if region_of(int(r)) == name and keep_res[i] and np.isfinite(res_left[i])
        ]
        by_region[name] = {
            "n": float(len(idx)),
            "leftover_mean": _mean([float(res_left[i]) for i in idx]),
            "global_disp_mean": _mean([float(glob[i]) for i in idx]),
            "core_aligned_disp_mean": _mean([float(core_disp[i]) for i in idx]),
        }

    hinge_l = by_region["hinge"]["leftover_mean"]
    lid_l = by_region["lid_interior"]["leftover_mean"]
    nmp_l = by_region["nmp_interior"]["leftover_mean"]
    lid_g = by_region["lid_interior"]["global_disp_mean"]
    hinge_g = by_region["hinge"]["global_disp_mean"]

    finite = keep_res & np.isfinite(res_left)
    if int(finite.sum()) >= 8:
        spearman = float(
            np.corrcoef(
                np.argsort(np.argsort(res_left[finite])),
                np.argsort(np.argsort(glob[finite])),
            )[0, 1]
        )
    else:
        spearman = float("nan")

    success = {
        "hinge_leftover_gt_lid_interior": bool(hinge_l > lid_l),
        "hinge_leftover_gt_nmp_interior": bool(hinge_l > nmp_l),
        "lid_global_disp_gt_hinge": bool(lid_g > hinge_g),
        "lid_interior_low_leftover_vs_its_global_disp": bool(
            lid_l < hinge_l and lid_g > hinge_g
        ),
    }
    success["all"] = all(success.values())

    plot_path = _ROOT / "results" / "pharma_pb_adk.png"
    _write_plot(plot_path, resseq, res_left, glob, keep_res)

    return {
        "protocol": (
            "Test P-B. E. coli adenylate kinase apo 4AKE vs holo 1AKE (Ap5A). "
            "Identifying object is per-contact relative pose of residue j in "
            "sender i's local backbone frame, apo vs holo — not a typed shared "
            "map R_r. No whole-protein superposition is used in leftover. "
            "Pre-registered regions: NMP 30–59, LID 122–159, CORE 1–29/60–121/"
            "160–214; hinge band junctions ±1. Termini (5 residues each end) and "
            "top B-factor quantile dropped. Success: leftover higher at hinges "
            "than at LID/NMP interiors, while global Cα displacement is larger "
            "in the swinging domains. Not drug design. Not a shared-type Gate 0 pass."
        ),
        "apo": args.apo,
        "holo": args.holo,
        "n_shared_residues": int(len(resseq)),
        "n_contacts": int(src.shape[0]),
        "n_contacts_kept": int(keep_pair.sum()),
        "n_dropped_termini": len(term),
        "n_dropped_high_b": len(hib),
        "pre_registered": {
            "core": CORE_RANGES,
            "nmp": NMP_RANGE,
            "lid": LID_RANGE,
            "hinge": HINGE_RES,
        },
        "by_region": by_region,
        "spearman_leftover_vs_global_disp": spearman,
        "success": success,
        "plot": str(plot_path),
        "non_claim": (
            "Does not restore typed shared R_r. Does not claim a medicine. "
            "Hinge labels are literature junctions, not leftover-selected."
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
    ax.plot(x[m], left[m] / (np.nanmax(left[m]) + 1e-8), label="pair leftover (local frames)", color="#1d4ed8")
    ax.plot(x[m], g[m] / (np.nanmax(g[m]) + 1e-8), label="Cα disp after global superposition", color="#a16207", linestyle=":")
    for h in HINGE_RES:
        ax.axvline(h, color="#9ca3af", linewidth=0.6, alpha=0.7)
    ax.axvspan(NMP_RANGE[0], NMP_RANGE[1], color="#fde68a", alpha=0.25, label="NMP")
    ax.axvspan(LID_RANGE[0], LID_RANGE[1], color="#bfdbfe", alpha=0.25, label="LID")
    ax.set_xlabel("Residue (E. coli AdK)")
    ax.set_ylabel("normalized score")
    ax.set_title("P-B AdK: local pair leftover vs global displacement")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apo", type=str, default="4AKE")
    parser.add_argument("--holo", type=str, default="1AKE")
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--pdb-cache", type=str, default="data/pdb")
    parser.add_argument("--b-quantile", type=float, default=0.90)
    parser.add_argument("--out", type=str, default="results/pharma_pb_adk.json")
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
                "n_shared": payload["n_shared_residues"],
                "n_contacts_kept": payload["n_contacts_kept"],
                "by_region": payload["by_region"],
                "spearman": payload["spearman_leftover_vs_global_disp"],
                "success": payload["success"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

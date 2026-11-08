"""Gate 0: does a native molecular pair law exist on 1UBQ?

Pass rules are fixed before looking at numbers.

Paper leftover (IT-GLLS on residue stalks) needs:
  held-out RMSE < 0.25 × feature RMS  and  < 0.50 × shuffled RMSE
  on a non-backbone contact type (short_range / long_range / distal).

Relative pose is a different object (how frame j sits in frame i). It does
not authorize the paper leftover even if concentrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pharma_inject import (  # noqa: E402
    PDB_ID,
    TYPE_NAMES,
    evaluate_gate0,
    fetch_pdb,
    gate0_verdict,
    load_protein_graph,
)


def write_plot(path: Path, report: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    feats = ["ca_world", "frame_6", "frame_x_axis"]
    x = np.arange(len(TYPE_NAMES))
    width = 0.12
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    colors = {"ca_world": "#a16207", "frame_6": "#1d4ed8", "frame_x_axis": "#0f766e"}
    for i, feat in enumerate(feats):
        native = [report[feat][n].get("rmse_over_scale") or float("nan") for n in TYPE_NAMES]
        ax.bar(x + (i - 1) * width, native, width, label=feat, color=colors[feat])
    ax.axhline(0.25, color="#b91c1c", linestyle=":", linewidth=1.2, label="pass line 0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_NAMES)
    ax.set_ylabel("held-out RMSE / scale")
    ax.set_title("Gate 0: native linear pair law on 1UBQ")
    ax.legend(fontsize=8)
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
    report = evaluate_gate0(graph, seed=args.seed)
    verdict = gate0_verdict(report)
    plot_path = _ROOT / "results" / "pharma_gate0.png"
    write_plot(plot_path, report)
    return {
        "protocol": (
            "Gate 0 on public crystal 1UBQ chain A. Contact types are sequence-"
            "separation + Cα distance. Features: (i) world-frame Cα, (ii) 6-dim "
            "backbone frame (first two columns), (iii) frame x-axis. Linear pair "
            "law x_v≈A x_u is fit on 70% of pairs and scored on 30% vs a "
            "destination shuffle (5 splits). Pass for the paper leftover if a "
            "non-backbone type on frame_6 or frame_x_axis has held-out RMSE < 0.25× "
            "feature RMS and < 0.50×shuffle. World-frame Cα is a control (scale = "
            "pair length). Relative pose is reported separately. Backbone is not "
            "enough to pass: peptide geometry is expected to be stereotyped."
        ),
        "pdb_id": args.pdb_id,
        "pdb_path": str(pdb_path),
        "n_residues": int(graph["ca"].shape[0]),
        "type_counts": {
            name: int((graph["types"] == i).sum()) for i, name in enumerate(TYPE_NAMES)
        },
        "pass_rule": {
            "linear": "holdout RMSE < 0.25 * scale AND < 0.50 * shuffle; scale is pair length for Cα and feature RMS for frames; pass requires a non-backbone *frame* type",
            "relative_pose": "geodesic < 0.50 * random pairs AND translation RMSE < 0.25 * |t|; non-backbone",
        },
        "report": report,
        "verdict": verdict,
        "plot": str(plot_path),
        "non_claim": (
            "A pass does not show allostery or drug utility. A fail blocks P-B "
            "for that feature family."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-id", type=str, default=PDB_ID)
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--pdb-cache", type=str, default="data/pdb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/pharma_gate0.json")
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")
    slim = {
        "verdict": payload["verdict"],
        "linear": {
            feat: {
                name: {
                    "usable": rec.get("usable_linear_pair_law"),
                    "rmse_over_rms": rec.get("rmse_over_feature_rms"),
                    "native_over_shuffle": rec.get("native_over_shuffle"),
                }
                for name, rec in payload["report"][feat].items()
            }
            for feat in ("ca_world", "frame_6", "frame_x_axis")
        },
        "relative_pose": {
            name: {
                "usable": rec.get("usable_shared_relative_pose"),
                "geo_over_random": rec.get("geodesic_over_random"),
                "t_over_scale": rec.get("translation_over_scale"),
            }
            for name, rec in payload["report"]["relative_pose"].items()
        },
    }
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

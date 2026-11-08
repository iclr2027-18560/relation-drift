"""Gate 0: does a native robot motion→force pair law exist?

Pass rules are fixed before looking at numbers.

Paper leftover (IT-GLLS on motion/wrench stalks) needs:
  held-out RMSE < 0.25 × wrench RMS  and  < 0.50 × shuffled RMSE
  on the contact type, using a motion stalk (delta pose, affine delta pose,
  or measured twist). Free-space cannot pass the claim. Absolute pose is a
  control and does not authorize leftover.
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

from robotics_inject import (  # noqa: E402
    CLAIM_STALKS,
    CONTACT_EXCESS_N,
    TASK,
    TYPE_NAMES,
    evaluate_gate0,
    fetch_forceflow_parquet,
    gate0_verdict,
    load_plug_pairs,
)


def write_plot(path: Path, report: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stalks = [
        "delta_pose",
        "delta_pose_affine",
        "measured_twist",
        "delta_pose_to_delta_wrench",
        "abs_pose",
    ]
    x = np.arange(len(TYPE_NAMES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    colors = {
        "delta_pose": "#1d4ed8",
        "delta_pose_affine": "#0f766e",
        "measured_twist": "#7c3aed",
        "delta_pose_to_delta_wrench": "#a16207",
        "abs_pose": "#9ca3af",
    }
    for i, stalk in enumerate(stalks):
        native = [
            report[stalk][n].get("rmse_over_scale") or float("nan") for n in TYPE_NAMES
        ]
        ax.bar(x + (i - 2) * width, native, width, label=stalk, color=colors[stalk])
    ax.axhline(0.25, color="#b91c1c", linestyle=":", linewidth=1.2, label="pass line 0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_NAMES)
    ax.set_ylabel("held-out RMSE / wrench RMS")
    ax.set_title("Gate 0: native linear pair law on ForceFlow plug")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _ROOT / cache
    parquet = fetch_forceflow_parquet(args.task, cache)
    pairs = load_plug_pairs(parquet)
    report = evaluate_gate0(pairs, seed=args.seed)
    verdict = gate0_verdict(report)
    plot_path = _ROOT / "results" / "robotics_gate0.png"
    write_plot(plot_path, report)
    n = int(pairs["types"].shape[0])
    n_contact = int((pairs["types"] == 1).sum())
    return {
        "protocol": (
            "Gate 0 on public ForceFlow plug-insertion (xArm6, 6-axis wrist F/T). "
            "Pair law is wrench ≈ A · motion. Types are episode-baseline excess "
            f"xyz force: free < {CONTACT_EXCESS_N} N, contact ≥ {CONTACT_EXCESS_N} N. "
            "Stalks: (i) commanded 6D delta pose, (ii) affine delta pose [Δx; 1], "
            "(iii) measured twist from consecutive poses, (iv) delta pose → delta "
            "wrench, (v) absolute pose (control). Linear map is fit on 70% of "
            "episodes and scored on 30% vs a destination shuffle (5 splits). "
            "Pass for the paper leftover if a contact cell on a motion stalk has "
            "held-out RMSE < 0.25× wrench RMS and < 0.50×shuffle. Free-space is "
            "not enough to pass: near-zero force is expected to be stereotyped."
        ),
        "task": args.task,
        "parquet_path": str(parquet),
        "n_frames": n,
        "n_episodes": int(np_unique_len(pairs["episodes"])),
        "type_counts": {
            name: int((pairs["types"] == i).sum()) for i, name in enumerate(TYPE_NAMES)
        },
        "contact_fraction": n_contact / max(n, 1),
        "pass_rule": {
            "linear": (
                "episode-holdout RMSE < 0.25 * dest RMS AND < 0.50 * shuffle; "
                f"pass requires contact on one of {list(CLAIM_STALKS)}"
            ),
            "free_cannot_pass": True,
            "abs_pose_is_control": True,
        },
        "report": report,
        "verdict": verdict,
        "plot": str(plot_path),
        "non_claim": (
            "A pass does not show a robot product, failure detection SOTA, or "
            "transfer of the banking typed-leftover theorem. A fail blocks P-A/P-B "
            "for this leftover on this stalk."
        ),
    }


def np_unique_len(arr) -> int:
    import numpy as np

    return int(np.unique(arr).size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--cache", type=str, default="data/forceflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/robotics_gate0.json")
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
        "type_counts": payload["type_counts"],
        "linear": {
            stalk: {
                name: {
                    "usable": rec.get("usable_linear_pair_law"),
                    "rmse_over_scale": rec.get("rmse_over_scale"),
                    "native_over_shuffle": rec.get("native_over_shuffle"),
                    "n": rec.get("n"),
                }
                for name, rec in payload["report"][stalk].items()
            }
            for stalk in payload["report"]
        },
    }
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

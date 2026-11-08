"""Gate 0: does a native A2A style pair law exist on CAMEL chats?

Pass rules are fixed before looking at numbers.

Paper leftover on this domain needs:
  held-out RMSE < 0.25 × dest RMS  and  < 0.50 × shuffled RMSE
  on USER→ASSISTANT turns, conversation holdout, after stripping
  Instruction:/Solution: templates.

If this fails, stop. Do not plant a mapping jump.
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

from a2a_inject import (  # noqa: E402
    CLAIM_CHANNEL,
    CLAIM_STALKS,
    CONTROL_CHANNEL,
    FEATURE_NAMES,
    N_ROLE_PAIRS,
    evaluate_gate0,
    fetch_camel_archive,
    gate0_verdict,
    load_turn_pairs,
)


def write_plot(path: Path, report: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["pooled"] + list(report[CLAIM_CHANNEL]["role_pair"].keys())[:8]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(labels))
    width = 0.35
    native = []
    shuffle = []
    for lab in labels:
        cell = (
            report[CLAIM_CHANNEL]["pooled"]
            if lab == "pooled"
            else report[CLAIM_CHANNEL]["role_pair"][lab]
        )
        rec = cell["style_affine"]
        native.append(rec.get("rmse_over_scale") or float("nan"))
        shuffle.append(rec.get("native_over_shuffle") or float("nan"))
    ax.bar(x - width / 2, native, width, label="RMSE / dest RMS", color="#1d4ed8")
    ax.bar(x + width / 2, shuffle, width, label="native / shuffle", color="#0f766e")
    ax.axhline(0.25, color="#b91c1c", linestyle=":", linewidth=1.2, label="scale pass 0.25")
    ax.axhline(0.50, color="#a16207", linestyle="--", linewidth=1.0, label="shuffle pass 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Gate 0 ratios (affine style)")
    ax.set_title("A2A Gate 0: user→assistant linear pair law")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _slim_cell(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "usable": rec.get("usable_linear_pair_law"),
        "too_few_pairs": rec.get("too_few_pairs"),
        "n": rec.get("n"),
        "n_episodes": rec.get("n_episodes"),
        "rmse_over_scale": rec.get("rmse_over_scale"),
        "native_over_shuffle": rec.get("native_over_shuffle"),
        "native_rmse": rec.get("native_rmse"),
        "shuffle_rmse": rec.get("shuffle_rmse"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _ROOT / cache
    archive = fetch_camel_archive(cache)
    pairs = load_turn_pairs(archive)
    report = evaluate_gate0(pairs, seed=args.seed)
    verdict = gate0_verdict(report)
    plot_path = _ROOT / "results" / "a2a_gate0.png"
    write_plot(plot_path, report)
    return {
        "protocol": (
            "Gate 0 on public CAMEL AI Society two-agent chats. Pair law is "
            "assistant style ≈ A · user style on consecutive turns. Features "
            f"(pre-registered): {list(FEATURE_NAMES)}. Instruction/Solution "
            "prefixes and <CAMEL_TASK_DONE> are stripped. Holdout is by "
            "conversation (70/30, 5 splits) vs destination shuffle. Claim "
            "channel is USER→ASSISTANT; ASSISTANT→USER is a control. Pooled "
            f"map plus the first {N_ROLE_PAIRS} role pairs with at least 8 "
            "conversations. Pass if pooled user→assistant meets "
            "RMSE < 0.25×dest RMS and < 0.50×shuffle, or if at least half of "
            "those role-pair cells do. A fail blocks any mapping-jump plant."
        ),
        "archive_path": str(archive),
        "n_conversations": pairs["n_conversations"],
        "n_skipped_short": pairs["n_skipped_short"],
        "n_user_to_assistant": int(pairs[CLAIM_CHANNEL]["src"].shape[0]),
        "n_assistant_to_user": int(pairs[CONTROL_CHANNEL]["src"].shape[0]),
        "feature_names": list(FEATURE_NAMES),
        "pass_rule": {
            "linear": (
                "conversation-holdout RMSE < 0.25 * dest RMS AND < 0.50 * shuffle"
            ),
            "claim_channel": CLAIM_CHANNEL,
            "control_cannot_authorize": True,
            "plant_only_if_pass": True,
        },
        "report": report,
        "verdict": verdict,
        "plot": str(plot_path),
        "non_claim": (
            "A pass does not show an A2A security product, jailbreak detector, "
            "or transfer of the banking typed leftover. A fail means there is "
            "no native linear pair law on this stalk; do not plant."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="data/camel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/a2a_gate0.json")
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
        "n_conversations": payload["n_conversations"],
        "n_user_to_assistant": payload["n_user_to_assistant"],
        "n_assistant_to_user": payload["n_assistant_to_user"],
        "linear": {
            channel: {
                "pooled": {
                    stalk: _slim_cell(payload["report"][channel]["pooled"][stalk])
                    for stalk in CLAIM_STALKS
                },
                "role_pair": {
                    rp: {
                        stalk: _slim_cell(cells[stalk])
                        for stalk in CLAIM_STALKS
                    }
                    for rp, cells in payload["report"][channel]["role_pair"].items()
                },
            }
            for channel in (CLAIM_CHANNEL, CONTROL_CHANNEL)
        },
    }
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

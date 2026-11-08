"""Equilibrium AdK: can CORE predict LID/NMP on ordinary fluctuations?"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adk_md_coupling import run_equilibrium  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dcd",
        type=str,
        default="data/adk_md/equilibrium/1ake_007-nowater-core-dt240ps.dcd",
    )
    parser.add_argument(
        "--psf",
        type=str,
        default="data/adk_md/equilibrium/adk4AKE.psf",
    )
    parser.add_argument("--out", type=str, default="results/adk_eq_coupling.json")
    args = parser.parse_args()
    dcd = Path(args.dcd)
    psf = Path(args.psf)
    if not dcd.is_absolute():
        dcd = _ROOT / dcd
    if not psf.is_absolute():
        psf = _ROOT / psf
    payload = run_equilibrium(dcd, psf)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")
    slim = {
        "source": payload["source"],
        "verdict": payload["verdict"],
        "tau_1": {
            block: {
                pair: {
                    "usable": rec["gate0"]["gate0_pass"],
                    "rmse_over_scale": rec["gate0"]["rmse_over_scale"],
                    "native_over_shuffle": rec["gate0"]["native_over_shuffle"],
                    "shuffle_distinguishes": rec["shuffle_distinguishes"],
                }
                for pair, rec in payload["blocks"][block]["tau_1"].items()
            }
            for block in payload["blocks"]
        },
    }
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

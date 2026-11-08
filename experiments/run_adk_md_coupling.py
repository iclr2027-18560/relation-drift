"""AdK LID–CORE / NMP–CORE dynamical coupling test (DIMS paths)."""

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

from adk_md_coupling import N_PATHS, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default="data/adk_md/DIMS/trajectories")
    parser.add_argument("--psf", type=str, default="data/adk_md/DIMS/topologies/adk4ake.psf")
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument("--out", type=str, default="results/adk_md_coupling.json")
    args = parser.parse_args()
    traj = Path(args.traj_dir)
    psf = Path(args.psf)
    if not traj.is_absolute():
        traj = _ROOT / traj
    if not psf.is_absolute():
        psf = _ROOT / psf
    payload = run(traj, psf, n_paths=args.n_paths)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")
    slim = {
        "verdict": payload["verdict"],
        "go_no_go": payload["go_no_go"],
        "gate0": {
            rep: {
                pair: rec["gate0"]
                for pair, rec in pairs.items()
            }
            for rep, pairs in payload["replicas"].items()
        },
        "D": {
            rep: {pair: rec["D_frobenius"] for pair, rec in pairs.items()}
            for rep, pairs in payload["replicas"].items()
        },
        "incremental": {
            rep: {pair: rec["incremental"] for pair, rec in pairs.items()}
            for rep, pairs in payload["replicas"].items()
        },
        "baselines_delta_d": {
            rep: {
                pair: rec["baselines"]["delta_distance"]
                for pair, rec in pairs.items()
            }
            for rep, pairs in payload["replicas"].items()
        },
    }
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Elliptic++ TGN / EvolveGCN-H / DualPath on the fair V3.5 split.

Protocol (matches src/run/run_elliptic_benchmark.py):
  - Train timesteps 1–34, test 35–49
  - No test-label online adaptation
  - Report overall / pre-43 / post-43 / degradation

Usage:
  python experiments/run_elliptic_dynamic_baselines.py --models dual
  python experiments/run_elliptic_dynamic_baselines.py --models dual evolvegcn tgn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX35 = _SRC / "run"
for _p in (_SRC, _FIX35, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_elliptic_benchmark import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ELLIPTIC_TEST_START,
    ELLIPTIC_TRAIN_END,
    REGIME_CHANGE_TIMESTEP,
    EllipticTemporalDataset,
)
from run_dualpath_gnn_baseline import (  # noqa: E402
    NUM_EPOCHS as DP_EPOCHS,
    train_and_evaluate as train_dualpath,
)
from run_evolvegcn_baseline import (  # noqa: E402
    NUM_EPOCHS as EGCN_EPOCHS,
    train_and_evaluate_evolvegcn,
)
from run_tgn_baseline import (  # noqa: E402
    NUM_EPOCHS as TGN_EPOCHS,
    train_and_evaluate_tgn,
)

TRAIN_END = ELLIPTIC_TRAIN_END  # 34 snapshots if timesteps are 1..49


def _as_cpu_snap(snap: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "x": snap["x"].detach().cpu().float(),
        "edge_index": snap["edge_index"].detach().cpu().long(),
        "labels": snap["labels"].detach().cpu().float(),
        "t": snap["t"].detach().cpu(),
    }


def elliptic_to_dualpath(seq: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
    return [_as_cpu_snap(s) for s in seq]


def elliptic_to_evolvegcn(seq: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
    out: List[Dict[str, Tensor]] = []
    for s in seq:
        item = _as_cpu_snap(s)
        ei = item["edge_index"]
        if ei.shape[1] == 0:
            item["struct_edge_index"] = ei
        else:
            item["struct_edge_index"] = torch.cat([ei, ei.flip(0)], dim=1)
        out.append(item)
    return out


def elliptic_to_tgn(seq: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
    """Unique node IDs across time (Elliptic txs do not persist)."""
    out: List[Dict[str, Tensor]] = []
    offset = 0
    for s in seq:
        item = _as_cpu_snap(s)
        item["node_offset"] = offset
        offset += int(item["x"].shape[0])
        out.append(item)
    return out


def _timestep_of(snapshots: List[Dict[str, Tensor]], idx: int) -> int:
    t = snapshots[idx]["t"]
    if torch.is_tensor(t):
        return int(t.item())
    return int(t)


def regime_summary(
    per_t: Dict[int, float],
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    overall: float,
    regime: int = REGIME_CHANGE_TIMESTEP,
) -> Dict[str, Any]:
    pre: List[float] = []
    post: List[float] = []
    per_test: Dict[str, float] = {}
    for idx, auc in per_t.items():
        i = int(idx)
        if i < train_end:
            continue
        ts = _timestep_of(snapshots, i)
        per_test[str(ts)] = float(auc)
        if ts < regime:
            pre.append(float(auc))
        else:
            post.append(float(auc))
    pre_m = float(sum(pre) / len(pre)) if pre else 0.5
    post_m = float(sum(post) / len(post)) if post else 0.5
    return {
        "overall_auc": float(overall),
        "pre_regime_mean": pre_m,
        "post_regime_mean": post_m,
        "degradation": pre_m - post_m,
        "per_test_timestep_auc": per_test,
    }


def _save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["dual", "evolvegcn", "tgn"],
        help="Subset: dual evolvegcn tgn",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="40 epochs")
    parser.add_argument(
        "--out",
        type=str,
        default="results/elliptic_dynamic_baselines.json",
    )
    args = parser.parse_args()

    epochs_dp = 40 if args.quick else (args.epochs or DP_EPOCHS)
    epochs_eg = 40 if args.quick else (args.epochs or EGCN_EPOCHS)
    epochs_tgn = 40 if args.quick else (args.epochs or TGN_EPOCHS)

    ds = EllipticTemporalDataset(args.data_dir, device=torch.device("cpu"))
    seq = ds.to_v35_sequence(ds.snapshots)
    n_t = len(seq)
    assert n_t >= TRAIN_END, f"expected >= {TRAIN_END} snapshots, got {n_t}"
    train_end = TRAIN_END
    first_t = _timestep_of(seq, 0)
    if first_t == 1:
        pass
    elif first_t == 0:
        train_end = ELLIPTIC_TRAIN_END + 1
    print(
        f"Elliptic snapshots={n_t} first_t={first_t} "
        f"train_end_idx={train_end} (exclusive) "
        f"train_last_t={_timestep_of(seq, train_end - 1)} "
        f"test_first_t={_timestep_of(seq, train_end)}"
    )

    payload: Dict[str, Any] = {
        "protocol": {
            "train_timesteps": f"1–{ELLIPTIC_TRAIN_END}",
            "test_timesteps": f"{ELLIPTIC_TEST_START}–49",
            "regime_change_timestep": REGIME_CHANGE_TIMESTEP,
            "online_adapt": False,
            "train_end_index": train_end,
        },
        "models": {},
    }
    out_path = _ROOT / args.out

    wanted = {m.lower() for m in args.models}

    if "dual" in wanted:
        print("\n=== DualPath (gated) ===")
        snaps = elliptic_to_dualpath(seq)
        overall, per_t, mean_gate = train_dualpath(
            snaps, "dual", epochs_dp, train_end=train_end
        )
        payload["models"]["dualpath"] = {
            **regime_summary(per_t, snaps, train_end, overall),
            "mean_gate": mean_gate,
            "epochs": epochs_dp,
        }
        _save(out_path, payload)

    if "evolvegcn" in wanted:
        print("\n=== EvolveGCN-H ===")
        snaps = elliptic_to_evolvegcn(seq)
        num_nodes = int(max(s["x"].shape[0] for s in snaps))
        feat = int(snaps[0]["x"].shape[1])
        overall, per_t = train_and_evaluate_evolvegcn(
            snaps,
            num_nodes,
            feat,
            num_epochs=epochs_eg,
            variant="H",
            train_end=train_end,
        )
        payload["models"]["evolvegcn_h"] = {
            **regime_summary(per_t, snaps, train_end, overall),
            "epochs": epochs_eg,
            "num_nodes": num_nodes,
        }
        _save(out_path, payload)

    if "tgn" in wanted:
        print("\n=== TGN (unique node IDs per tx) ===")
        snaps = elliptic_to_tgn(seq)
        overall, per_t = train_and_evaluate_tgn(
            snaps, num_epochs=epochs_tgn, train_end=train_end
        )
        payload["models"]["tgn"] = {
            **regime_summary(per_t, snaps, train_end, overall),
            "epochs": epochs_tgn,
        }
        _save(out_path, payload)

    print("\nDone.")
    print(json.dumps(payload["models"], indent=2))


if __name__ == "__main__":
    main()

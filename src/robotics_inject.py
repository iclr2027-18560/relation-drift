"""Gate 0: does a native motion→force pair law exist on a public robot?

The paper leftover is a typed shared linear map x_v ≈ A x_u. Robotics is a
candidate only if that object exists natively, not because a plant can be
detected (Test P-A) and not because force magnitude already flags contact.

Pre-registered native object: wrist wrench vs end-effector motion on the
public ForceFlow plug-insertion task (xArm6 + 6-axis F/T). Free-space cannot
pass the claim: near-zero force is stereotyped, like backbone geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

TASK = "plug"
REPO_ID = "JokerESC/ForceFlow"

TYPE_NAMES: Tuple[str, ...] = ("free", "contact")
CONTACT_EXCESS_N = 2.0
BASELINE_STEPS = 10
N_SPLITS = 5
HOLDOUT = 0.3

ACTION_COLS = [f"action_{i}" for i in range(6)]
POS_COLS = [f"pos_{i}" for i in range(6)]
FORCE_COLS = [f"force_{i}" for i in range(6)]
DELTA_FORCE_COLS = [f"delta_force_{i}" for i in range(6)]

# Paper leftover needs a motion stalk on contact, not absolute pose.
CLAIM_STALKS = ("delta_pose", "delta_pose_affine", "measured_twist")


def fetch_forceflow_parquet(
    task: str = TASK,
    cache_dir: Path | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir) if cache_dir is not None else Path("data/forceflow")
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"data/{task}/train-00000-of-00001.parquet"
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=name,
        local_dir=str(cache_dir),
    )
    return Path(path)


def _episode_baseline_xyz(force_xyz: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """Median xyz wrench of the first BASELINE_STEPS of each episode."""
    baseline = np.zeros_like(force_xyz)
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        k = min(BASELINE_STEPS, int(idx.size))
        baseline[idx] = np.median(force_xyz[idx[:k]], axis=0)
    return baseline


def load_plug_pairs(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    episodes = df["episode_index"].to_numpy(dtype=np.int64)
    action = df[ACTION_COLS].to_numpy(dtype=np.float64)
    pos = df[POS_COLS].to_numpy(dtype=np.float64)
    wrench = df[FORCE_COLS].to_numpy(dtype=np.float64)
    delta_wrench = df[DELTA_FORCE_COLS].to_numpy(dtype=np.float64)

    twist = np.zeros_like(pos)
    twist[1:] = pos[1:] - pos[:-1]
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        twist[idx[0]] = 0.0

    excess = np.linalg.norm(
        wrench[:, :3] - _episode_baseline_xyz(wrench[:, :3], episodes), axis=1
    )
    types = np.where(excess >= CONTACT_EXCESS_N, 1, 0).astype(np.int64)
    return {
        "episodes": episodes,
        "action": action,
        "pos": pos,
        "twist": twist,
        "wrench": wrench,
        "delta_wrench": delta_wrench,
        "excess_force": excess,
        "types": types,
    }


def _lstsq_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    a = np.linalg.lstsq(src, dst, rcond=None)[0].T
    return float(np.linalg.norm(dst - (a @ src.T).T, axis=1).mean())


def linear_pair_law_grouped(
    src: np.ndarray,
    dst: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    n_splits: int = N_SPLITS,
    holdout: float = HOLDOUT,
) -> Dict[str, float]:
    """Shared map x_v ≈ A x_u with episode holdout (not random timesteps).

    Pre-registered pass (same inequalities as the pharma Gate 0 leftover):
      holdout native RMSE < 0.25 × RMS(destination)  and
      holdout native RMSE < 0.50 × holdout shuffle RMSE.
    """
    n, dim_src = src.shape
    dim_dst = dst.shape[1]
    feat_rms = float(np.sqrt(np.mean(dst * dst)))
    uniq = np.unique(groups)
    min_train = max(8, 4 * dim_src)
    if n < min_train or uniq.size < 4:
        return {
            "n": float(n),
            "n_episodes": float(uniq.size),
            "dim_src": float(dim_src),
            "dim_dst": float(dim_dst),
            "native_rmse": float("nan"),
            "shuffle_rmse": float("nan"),
            "feature_rms": feat_rms,
            "usable_linear_pair_law": False,
            "too_few_pairs": True,
        }

    native_scores, shuffle_scores = [], []
    for _ in range(n_splits):
        perm_g = rng.permutation(uniq.size)
        n_hold_g = max(int(round(holdout * uniq.size)), 1)
        if n_hold_g >= uniq.size - 1:
            n_hold_g = max(uniq.size // 3, 1)
        hold_g = set(uniq[perm_g[:n_hold_g]].tolist())
        hold = np.array([g in hold_g for g in groups])
        train = ~hold
        if int(train.sum()) < min_train or int(hold.sum()) < dim_src + 2:
            native_scores.append(_lstsq_rmse(src, dst))
            perm = rng.permutation(n)
            shuffle_scores.append(_lstsq_rmse(src, dst[perm]))
            continue
        a = np.linalg.lstsq(src[train], dst[train], rcond=None)[0].T
        native_scores.append(
            float(np.linalg.norm(dst[hold] - (a @ src[hold].T).T, axis=1).mean())
        )
        perm = rng.permutation(n)
        dst_sh = dst[perm]
        a_null = np.linalg.lstsq(src[train], dst_sh[train], rcond=None)[0].T
        shuffle_scores.append(
            float(np.linalg.norm(dst_sh[hold] - (a_null @ src[hold].T).T, axis=1).mean())
        )

    native = float(np.mean(native_scores))
    shuffle = float(np.mean(shuffle_scores))
    usable = bool(native < 0.25 * feat_rms and native < 0.50 * shuffle)
    return {
        "n": float(n),
        "n_episodes": float(uniq.size),
        "n_splits": float(n_splits),
        "dim_src": float(dim_src),
        "dim_dst": float(dim_dst),
        "native_rmse": native,
        "shuffle_rmse": shuffle,
        "feature_rms": feat_rms,
        "rmse_over_scale": native / max(feat_rms, 1e-8),
        "native_over_shuffle": native / max(shuffle, 1e-8),
        "native_better_than_shuffle": bool(native < 0.8 * shuffle),
        "usable_linear_pair_law": usable,
        "too_few_pairs": False,
    }


def _affine(src: np.ndarray) -> np.ndarray:
    return np.concatenate([src, np.ones((src.shape[0], 1), dtype=src.dtype)], axis=1)


def evaluate_gate0(
    pairs: Dict[str, np.ndarray],
    *,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Native pair-law checks. Pass rules are fixed before looking at numbers."""
    rng = np.random.default_rng(seed)
    episodes = pairs["episodes"]
    types = pairs["types"]
    stalks = {
        "delta_pose": (pairs["action"], pairs["wrench"]),
        "delta_pose_affine": (_affine(pairs["action"]), pairs["wrench"]),
        "measured_twist": (pairs["twist"], pairs["wrench"]),
        "delta_pose_to_delta_wrench": (pairs["action"], pairs["delta_wrench"]),
        "abs_pose": (pairs["pos"], pairs["wrench"]),
    }
    out: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in stalks}
    for stalk, (src, dst) in stalks.items():
        for r, name in enumerate(TYPE_NAMES):
            m = types == r
            out[stalk][name] = linear_pair_law_grouped(
                src[m], dst[m], episodes[m], rng
            )
            out[stalk][name]["mean_excess_force"] = float(pairs["excess_force"][m].mean())
    return out


def gate0_verdict(report: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Paper leftover needs a usable motion→wrench map on contact.

    abs_pose is a control (world-frame analog). free cannot pass the claim.
    """
    contact_ok = False
    free_ok = False
    pose_control_ok = False
    passing: list[str] = []
    for stalk in CLAIM_STALKS:
        if stalk not in report:
            continue
        if bool(report[stalk]["contact"].get("usable_linear_pair_law")):
            contact_ok = True
            passing.append(f"{stalk}:contact")
        if bool(report[stalk]["free"].get("usable_linear_pair_law")):
            free_ok = True
    if bool(report.get("abs_pose", {}).get("contact", {}).get("usable_linear_pair_law")):
        pose_control_ok = True
    return {
        "linear_pair_law_contact": contact_ok,
        "linear_pair_law_free": free_ok,
        "abs_pose_contact_usable": pose_control_ok,
        "passing_claim_cells": passing,
        "gate0_pass_for_paper_leftover": contact_ok,
    }

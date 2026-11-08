"""MathOverflow SNAP topology: Gate 0 on lagged activity, then a planted typed map.

Public edges from SNAP (a2q / c2q / c2a). SNAP files have no node features.
Gate 0 uses lagged monthly activity. The inject overwrites a 12-d random
subspace so x_v ≈ R_r(t) x_u, jumping only comment-to-answer. Counterparties
and interaction counts are unchanged.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from elliptic_inject import cayley, random_ortho_rows, random_skew

TYPE_NAMES: Tuple[str, ...] = ("a2q", "c2q", "c2a")
AFFECTED_TYPE = 2  # comment-to-answer
N_TYPES = len(TYPE_NAMES)
_FILES = {
    0: "sx-mathoverflow-a2q.txt.gz",
    1: "sx-mathoverflow-c2q.txt.gz",
    2: "sx-mathoverflow-c2a.txt.gz",
}


@dataclass
class MonthSnap:
    month: str
    pair_src: Tensor
    pair_dst: Tensor
    edge_types: Tensor
    labels: Tensor
    src_ids: Tensor
    dst_ids: Tensor
    amounts: Tensor  # interaction counts (ones); volume baseline
    R: Tensor
    native_src: Tensor
    native_dst: Tensor


def _read_typed_edges(data_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for type_id, fname in _FILES.items():
        path = Path(data_dir) / fname
        src, dst, ts = [], [], []
        with gzip.open(path, "rt") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                src.append(int(parts[0]))
                dst.append(int(parts[1]))
                ts.append(int(parts[2]))
        months = [
            datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m") for t in ts
        ]
        rows.append(
            pd.DataFrame(
                {"src": src, "dst": dst, "month": months, "type": type_id}
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_mathoverflow_months(
    data_dir: Path,
    *,
    n_snaps: int = 12,
    min_per_type: int = 100,
    max_pairs_per_type: int = 8000,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[int, int], pd.DataFrame]:
    raw = _read_typed_edges(data_dir)
    months = sorted(raw["month"].unique())
    ok: List[str] = []
    for m in months:
        block = raw[raw["month"] == m]
        if all(int((block["type"] == r).sum()) >= min_per_type for r in range(N_TYPES)):
            ok.append(m)
    if len(ok) < n_snaps:
        raise RuntimeError(
            f"only {len(ok)} months have ≥{min_per_type} edges per type; need {n_snaps}"
        )

    def score(window: Sequence[str]) -> int:
        return min(
            int((raw[(raw["month"] == m) & (raw["type"] == r)].shape[0]))
            for m in window
            for r in range(N_TYPES)
        )

    best = max(
        (ok[i : i + n_snaps] for i in range(len(ok) - n_snaps + 1)),
        key=score,
    )
    raw = raw[raw["month"].isin(best)]
    users = pd.unique(pd.concat([raw["src"], raw["dst"]], ignore_index=True))
    user_to_i = {int(u): i for i, u in enumerate(users.tolist())}
    raw = raw.copy()
    raw["src_i"] = raw["src"].map(user_to_i).astype(np.int32)
    raw["dst_i"] = raw["dst"].map(user_to_i).astype(np.int32)

    rng = np.random.default_rng(seed)
    by_month: Dict[str, pd.DataFrame] = {}
    for m in best:
        day = raw.loc[raw["month"] == m, ["src_i", "dst_i", "type"]]
        parts = []
        for r in range(N_TYPES):
            block = day[day["type"] == r]
            if len(block) > max_pairs_per_type:
                take = rng.choice(len(block), size=max_pairs_per_type, replace=False)
                block = block.iloc[take]
            parts.append(block)
        by_month[m] = pd.concat(parts, ignore_index=True)
        print(f"    {m} edges={len(by_month[m])} (from {len(day)} interactions)")
    return best, by_month, user_to_i, raw


def lagged_activity_features(
    months: Sequence[str],
    by_month: Dict[str, pd.DataFrame],
    n_users: int,
) -> Dict[str, Tensor]:
    """8-d lagged activity: log1p in/out degree and per-type send/receive counts."""
    feats: Dict[str, Tensor] = {}
    prev = torch.zeros(n_users, 8)
    feats[months[0]] = prev.clone()
    for m in months:
        cur = torch.zeros(n_users, 8)
        day = by_month[m]
        src = torch.from_numpy(day["src_i"].to_numpy())
        dst = torch.from_numpy(day["dst_i"].to_numpy())
        types = torch.from_numpy(day["type"].to_numpy().astype(np.int64))
        ones = torch.ones(src.shape[0])
        cur[:, 0].scatter_add_(0, src, ones)
        cur[:, 1].scatter_add_(0, dst, ones)
        for r in range(N_TYPES):
            mask = types == r
            if int(mask.sum()) == 0:
                continue
            cur[:, 2 + r].scatter_add_(0, src[mask], ones[mask])
            cur[:, 5 + r].scatter_add_(0, dst[mask], ones[mask])
        feats[m] = prev.clone()
        prev = torch.log1p(cur)
    return feats


def build_R_trajectory(
    n_steps: int,
    dim: int,
    t_star_index: int,
    affected: int,
    seed: int,
    omega: float,
    jump_mag: float,
    n_types: int = N_TYPES,
) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    maps = torch.zeros(n_steps, n_types, dim, dim)
    skews = [random_skew(dim, gen, scale=1.0) for _ in range(n_types)]
    current = [cayley(0.25 * random_skew(dim, gen, scale=1.0)) for _ in range(n_types)]
    jump_delta = random_skew(dim, gen, scale=1.0)
    for t in range(n_steps):
        for r in range(n_types):
            current[r] = current[r] @ cayley(omega * skews[r])
        if t == t_star_index:
            current[affected] = current[affected] + jump_mag * jump_delta
        for r in range(n_types):
            maps[t, r] = current[r]
    return maps


def inject_mathoverflow(
    months: Sequence[str],
    by_month: Dict[str, pd.DataFrame],
    n_users: int,
    native_feats: Dict[str, Tensor],
    *,
    feat_dim: int = 16,
    proj_dim: int = 12,
    t_star_index: int = 7,
    anomaly_frac: float = 0.10,
    noise: float = 0.05,
    omega: float = 0.03,
    jump_mag: float = 0.8,
    seed: int = 42,
) -> Tuple[List[MonthSnap], Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_users, feat_dim, generator=gen)
    proj = random_ortho_rows(proj_dim, feat_dim, gen)
    x_proj = x @ proj.T
    true_R = build_R_trajectory(
        len(months), proj_dim, t_star_index, AFFECTED_TYPE, seed + 1, omega, jump_mag
    )
    snaps: List[MonthSnap] = []
    for t_i, m in enumerate(months):
        day = by_month[m]
        src = torch.from_numpy(day["src_i"].to_numpy())
        dst = torch.from_numpy(day["dst_i"].to_numpy())
        types = torch.from_numpy(day["type"].to_numpy().astype(np.int64))
        n_e = int(src.shape[0])
        pair_src = x_proj[src]
        native = native_feats[m]
        g = torch.Generator().manual_seed(seed + 17 * t_i)
        labels = torch.zeros(n_e)
        for r in range(N_TYPES):
            idx = (types == r).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            n_anom = max(1, int(round(anomaly_frac * idx.numel())))
            perm = torch.randperm(idx.numel(), generator=g)[:n_anom]
            labels[idx[perm]] = 1.0
        noise_t = noise * torch.randn(n_e, proj_dim, generator=g)
        planted = torch.zeros(n_e, proj_dim)
        for r in range(N_TYPES):
            mask = types == r
            if int(mask.sum()) == 0:
                continue
            planted[mask] = (true_R[t_i, r] @ pair_src[mask].T).T + noise_t[mask]
        pair_dst_nat = x_proj[dst]
        pair_dst = torch.where(labels.unsqueeze(1) < 0.5, planted, pair_dst_nat)
        snaps.append(
            MonthSnap(
                month=m,
                pair_src=pair_src,
                pair_dst=pair_dst,
                edge_types=types,
                labels=labels,
                src_ids=src,
                dst_ids=dst,
                amounts=torch.ones(n_e),
                R=true_R[t_i],
                native_src=native[src],
                native_dst=native[dst],
            )
        )
    return snaps, proj, true_R

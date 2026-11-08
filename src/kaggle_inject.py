"""Plant a typed linear map on IBM HI-Small topology (Test A).

Real payment graph, times, and payment kinds. Account features are random;
a 12-dim subspace is overwritten so x_v ≈ R_r(t) x_u. One kind (ACH) jumps
at t*. Amounts and counterparties are not changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from elliptic_inject import cayley, random_ortho_rows, random_skew

TYPE_NAMES: Tuple[str, ...] = ("ACH", "Wire", "Cheque", "Credit Card", "Other")
AFFECTED_TYPE = 0  # ACH
N_TYPES = len(TYPE_NAMES)
_TYPE_INDEX = {name: i for i, name in enumerate(TYPE_NAMES)}


def format_to_type(name: str) -> int:
    return _TYPE_INDEX.get(str(name), _TYPE_INDEX["Other"])


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


@dataclass
class DaySnap:
    date: str
    pair_src: Tensor
    pair_dst: Tensor
    edge_types: Tensor
    labels: Tensor
    src_ids: Tensor
    dst_ids: Tensor
    amounts: Tensor
    R: Tensor


def load_ibm_days(
    path: Path,
    *,
    max_pairs_per_type: int = 8000,
    min_day_txs: int = 10000,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, int]]:
    """Load HI-Small; return dates, per-day sampled edges, account id map."""
    path = Path(path)
    print(f"  reading {path}")
    raw = pd.read_csv(
        path,
        usecols=[0, 2, 4, 5, 9, 10],
        header=0,
        names=["ts", "src", "dst", "amount", "fmt", "laundering"],
        dtype={
            "src": str,
            "dst": str,
            "amount": np.float32,
            "fmt": str,
            "laundering": np.int8,
        },
    )
    raw["date"] = pd.to_datetime(raw["ts"], format="%Y/%m/%d %H:%M").dt.strftime(
        "%Y-%m-%d"
    )
    counts = raw.groupby("date").size()
    keep = set(counts[counts >= min_day_txs].index.tolist())
    raw = raw[raw["date"].isin(keep)]
    dates = sorted(raw["date"].unique())
    print(f"  kept {len(dates)} days with ≥{min_day_txs} txs (dropped {int((counts < min_day_txs).sum())})")
    raw["type"] = raw["fmt"].map(format_to_type).astype(np.int16)
    accounts = pd.unique(pd.concat([raw["src"], raw["dst"]], ignore_index=True))
    acct_to_i = {a: i for i, a in enumerate(accounts.tolist())}
    raw["src_i"] = raw["src"].map(acct_to_i).astype(np.int32)
    raw["dst_i"] = raw["dst"].map(acct_to_i).astype(np.int32)

    rng = np.random.default_rng(seed)
    by_day: Dict[str, pd.DataFrame] = {}
    for d in dates:
        day = raw.loc[
            raw["date"] == d, ["src_i", "dst_i", "amount", "type", "laundering"]
        ]
        parts = []
        for r in range(N_TYPES):
            block = day[day["type"] == r]
            if len(block) > max_pairs_per_type:
                take = rng.choice(len(block), size=max_pairs_per_type, replace=False)
                block = block.iloc[take]
            parts.append(block)
        by_day[d] = pd.concat(parts, ignore_index=True)
        print(f"    {d} edges={len(by_day[d])} (from {len(day)} txs)")
    return dates, by_day, acct_to_i


def inject_ibm(
    dates: Sequence[str],
    by_day: Dict[str, pd.DataFrame],
    n_accounts: int,
    *,
    feat_dim: int = 16,
    proj_dim: int = 12,
    t_star_index: int = 10,
    anomaly_frac: float = 0.10,
    noise: float = 0.05,
    omega: float = 0.03,
    jump_mag: float = 0.8,
    seed: int = 42,
) -> Tuple[List[DaySnap], Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_accounts, feat_dim, generator=gen)
    proj = random_ortho_rows(proj_dim, feat_dim, gen)
    x_proj = x @ proj.T
    true_R = build_R_trajectory(
        len(dates), proj_dim, t_star_index, AFFECTED_TYPE, seed + 1, omega, jump_mag
    )
    snaps: List[DaySnap] = []
    for t_i, d in enumerate(dates):
        day = by_day[d]
        src = torch.from_numpy(day["src_i"].to_numpy())
        dst = torch.from_numpy(day["dst_i"].to_numpy())
        types = torch.from_numpy(day["type"].to_numpy().astype(np.int64))
        amounts = torch.from_numpy(day["amount"].to_numpy())
        n_e = int(src.shape[0])
        pair_src = x_proj[src]
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
            DaySnap(
                date=d,
                pair_src=pair_src,
                pair_dst=pair_dst,
                edge_types=types,
                labels=labels,
                src_ids=src,
                dst_ids=dst,
                amounts=amounts,
                R=true_R[t_i],
            )
        )
    return snaps, proj, true_R

"""Plant a typed SO(d)/GL(d) constraint on real Elliptic topology.

Degree-ratio edge types from topology only (no label leakage). A fixed
orthogonal projection P carries the planted map; the orthogonal complement
of the 165-dim features is left unchanged.

Raw pair observations (x_u_proj, planted or frozen x_v_proj) are the
identifying object. Node features get a mixed-mean write-back of incoming
planted targets so a snapshot GNN sees the same subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

TYPE_NAMES: Tuple[str, ...] = ("hub_to_leaf", "peer", "leaf_to_hub")
HUB_TO_LEAF = 0
PEER = 1
LEAF_TO_HUB = 2
AFFECTED_TYPE = HUB_TO_LEAF
N_TYPES = 3


def cayley(skew: Tensor) -> Tensor:
    """Cayley map R = (I - A)(I + A)^{-1} for skew A."""
    dim = skew.shape[-1]
    eye = torch.eye(dim, dtype=skew.dtype, device=skew.device)
    return torch.linalg.solve(eye + skew, eye - skew)


def random_skew(dim: int, generator: torch.Generator, scale: float = 1.0) -> Tensor:
    params = torch.randn(dim * (dim - 1) // 2, generator=generator)
    skew = torch.zeros(dim, dim)
    idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            skew[i, j] = params[idx]
            skew[j, i] = -params[idx]
            idx += 1
    norm = float(skew.square().sum().sqrt().item())
    if norm > 1e-8:
        skew = skew * (scale / norm)
    return skew


def random_ortho_rows(out_dim: int, in_dim: int, generator: torch.Generator) -> Tensor:
    """P with shape [out_dim, in_dim] and orthonormal rows."""
    raw = torch.randn(in_dim, out_dim, generator=generator)
    q_mat, _ = torch.linalg.qr(raw, mode="reduced")
    return q_mat.T.contiguous()


def degree_ratio_types(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Type edges by deg(u)/(deg(v)+1) on undirected snapshot degree."""
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    ones = torch.ones(src.shape[0], dtype=torch.float32)
    deg.scatter_add_(0, src, ones)
    deg.scatter_add_(0, dst, ones)
    ratio = deg[src] / (deg[dst] + 1.0)
    types = torch.full((src.shape[0],), PEER, dtype=torch.long)
    types[ratio > 2.0] = HUB_TO_LEAF
    types[ratio < 0.5] = LEAF_TO_HUB
    return types


def build_R_trajectory(
    n_steps: int,
    dim: int,
    t_star_index: int,
    affected: int,
    seed: int,
    omega: float,
    jump_mag: float,
) -> Tensor:
    """R[t, r] in R^{d,d}. Smooth Cayley drift; additive jump on one type."""
    gen = torch.Generator().manual_seed(seed)
    maps = torch.zeros(n_steps, N_TYPES, dim, dim)
    skews = [random_skew(dim, gen, scale=1.0) for _ in range(N_TYPES)]
    current = [cayley(0.25 * random_skew(dim, gen, scale=1.0)) for _ in range(N_TYPES)]
    jump_delta = random_skew(dim, gen, scale=1.0)
    for t in range(n_steps):
        for r in range(N_TYPES):
            current[r] = current[r] @ cayley(omega * skews[r])
        if t == t_star_index:
            current[affected] = current[affected] + jump_mag * jump_delta
        for r in range(N_TYPES):
            maps[t, r] = current[r]
    return maps


@dataclass
class InjectedSnapshot:
    timestep: int
    x_original: Tensor
    x_injected: Tensor
    x_proj: Tensor
    edge_index: Tensor
    edge_types: Tensor
    pair_src: Tensor
    pair_dst: Tensor
    labels: Tensor  # 1 = planted anomaly (frozen real target)
    R: Tensor  # [n_types, d, d]


def load_elliptic_snapshots(
    data_dir: Path,
) -> Tuple[List[int], List[Tensor], List[Tensor]]:
    """Load full intra-timestep graphs (all edges, not label-filtered)."""
    data_dir = Path(data_dir)
    features_df = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    edges_df = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    node_ids = features_df.iloc[:, 0].to_numpy()
    timesteps = features_df.iloc[:, 1].to_numpy().astype(int)
    feats = features_df.iloc[:, 2:].to_numpy(dtype=np.float32)

    times = sorted(set(int(t) for t in timesteps))
    local: Dict[int, int] = {}
    counters: Dict[int, int] = {t: 0 for t in times}
    ts_of: Dict[int, int] = {}
    for nid, ts in zip(node_ids, timesteps):
        nid_i, ts_i = int(nid), int(ts)
        local[nid_i] = counters[ts_i]
        ts_of[nid_i] = ts_i
        counters[ts_i] += 1

    xs: List[Tensor] = []
    for ts in times:
        xs.append(torch.from_numpy(feats[timesteps == ts]))

    src_ids = edges_df.iloc[:, 0].to_numpy()
    dst_ids = edges_df.iloc[:, 1].to_numpy()
    buckets: Dict[int, List[Tuple[int, int]]] = {t: [] for t in times}
    for s_raw, d_raw in zip(src_ids, dst_ids):
        s_i, d_i = int(s_raw), int(d_raw)
        ts_s = ts_of.get(s_i)
        ts_d = ts_of.get(d_i)
        if ts_s is None or ts_s != ts_d:
            continue
        buckets[ts_s].append((local[s_i], local[d_i]))

    eidx: List[Tensor] = []
    for ts in times:
        pairs = buckets[ts]
        if pairs:
            eidx.append(torch.tensor(pairs, dtype=torch.long).t().contiguous())
        else:
            eidx.append(torch.zeros(2, 0, dtype=torch.long))
    return times, xs, eidx


def inject_sequence(
    times: Sequence[int],
    features: Sequence[Tensor],
    edges: Sequence[Tensor],
    *,
    proj_dim: int = 12,
    t_star: int = 30,
    train_end: int = 25,
    anomaly_frac: float = 0.10,
    noise: float = 0.05,
    omega: float = 0.03,
    jump_mag: float = 0.8,
    blend: float = 1.0,
    seed: int = 42,
) -> Tuple[List[InjectedSnapshot], Tensor, Tensor]:
    """Return injected snapshots, projection P, and true R[t,r]."""
    del train_end  # documented in the experiment protocol; unused in planting
    gen = torch.Generator().manual_seed(seed)
    in_dim = int(features[0].shape[1])
    proj = random_ortho_rows(proj_dim, in_dim, gen)
    n_steps = len(times)
    t_star_index = times.index(t_star) if t_star in times else n_steps // 2
    true_R = build_R_trajectory(
        n_steps, proj_dim, t_star_index, AFFECTED_TYPE, seed + 1, omega, jump_mag
    )
    out: List[InjectedSnapshot] = []
    for t_i, (ts, x, eidx) in enumerate(zip(times, features, edges)):
        n = x.shape[0]
        types = degree_ratio_types(eidx, n)
        src_idx = eidx[0] if eidx.numel() else torch.zeros(0, dtype=torch.long)
        dst_idx = eidx[1] if eidx.numel() else torch.zeros(0, dtype=torch.long)
        x_proj = x @ proj.T
        n_e = int(src_idx.shape[0])
        labels = torch.zeros(n_e, dtype=torch.float32)
        pair_src = x_proj[src_idx] if n_e else torch.zeros(0, proj_dim)
        pair_dst = x_proj[dst_idx] if n_e else torch.zeros(0, proj_dim)
        if n_e:
            g_edge = torch.Generator().manual_seed(seed + 17 * int(ts))
            for r in range(N_TYPES):
                mask = types == r
                idx = mask.nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                n_anom = max(1, int(round(anomaly_frac * idx.numel())))
                perm = torch.randperm(idx.numel(), generator=g_edge)[:n_anom]
                labels[idx[perm]] = 1.0
            compliant = labels < 0.5
            noise_t = noise * torch.randn(n_e, proj_dim, generator=g_edge)
            planted = torch.zeros(n_e, proj_dim)
            for r in range(N_TYPES):
                mask = types == r
                if int(mask.sum()) == 0:
                    continue
                planted[mask] = (true_R[t_i, r] @ pair_src[mask].T).T + noise_t[mask]
            pair_dst = torch.where(compliant.unsqueeze(1), planted, pair_dst)

        x_inj = x.clone()
        if n_e:
            acc = torch.zeros(n, proj_dim)
            cnt = torch.zeros(n, 1)
            acc.index_add_(0, dst_idx, pair_dst)
            cnt.index_add_(0, dst_idx, torch.ones(n_e, 1))
            has = (cnt.squeeze(1) > 0).unsqueeze(1)
            mean_proj = acc / cnt.clamp(min=1.0)
            new_proj = torch.where(has, mean_proj, x_proj)
            x_inj = x + blend * ((new_proj - x_proj) @ proj)

        out.append(
            InjectedSnapshot(
                timestep=int(ts),
                x_original=x,
                x_injected=x_inj,
                x_proj=x_inj @ proj.T,
                edge_index=eidx,
                edge_types=types,
                pair_src=pair_src,
                pair_dst=pair_dst,
                labels=labels,
                R=true_R[t_i],
            )
        )
    return out, proj, true_R

"""Test P-A: plant a typed linear map on a public protein contact graph.

Crystal Cα coordinates and contact types come from a PDB file. Residue
features are random; a 12-dim subspace is overwritten so x_v ≈ R_r(t) x_u.
One contact type (long-range Cα contacts) jumps at t*. Coordinates, and
therefore RMSD / RMSF / contact-count / new-contact, are not rewritten by
the plant.

This is a mechanism / sanity check, not a biological discovery and not a
drug-design result. Pair types are sequence-separation + Cα distance, not
an allostery label.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from elliptic_inject import cayley, random_ortho_rows, random_skew

TYPE_NAMES: Tuple[str, ...] = (
    "backbone",
    "short_range",
    "long_range",
    "distal",
)
AFFECTED_TYPE = 2  # long_range: |i-j| >= 5 and Cα–Cα < 8 Å
N_TYPES = len(TYPE_NAMES)

CONTACT_CUTOFF = 8.0
DISTAL_CUTOFF = 12.0
SHORT_SEQ_MAX = 4

PDB_ID = "1UBQ"
PDB_URL = "https://files.rcsb.org/download/1UBQ.pdb"


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


def fetch_pdb(pdb_id: str = PDB_ID, cache_dir: Path | None = None) -> Path:
    cache_dir = Path(cache_dir) if cache_dir is not None else Path("data/pdb")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdb_id.upper()}.pdb"
    if path.exists() and path.stat().st_size > 200:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, path)
    return path


def parse_backbone(pdb_text: str, chain: str = "A") -> Dict[str, np.ndarray]:
    """First MODEL, one atom per (residue, name). AltLoc blank or A."""
    atoms: Dict[Tuple[int, str], np.ndarray] = {}
    bfactors: Dict[int, float] = {}
    resseq_order: List[int] = []
    seen_res: set[int] = set()
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[21].strip() != chain:
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C"):
            continue
        resseq = int(line[22:26])
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        key = (resseq, name)
        if key in atoms:
            continue
        atoms[key] = xyz
        if name == "CA":
            try:
                bfactors[resseq] = float(line[60:66])
            except ValueError:
                bfactors[resseq] = float("nan")
        if resseq not in seen_res:
            seen_res.add(resseq)
            resseq_order.append(resseq)

    ca, n_xyz, c_xyz, resseqs, bfac = [], [], [], [], []
    for r in resseq_order:
        if (r, "CA") not in atoms or (r, "N") not in atoms or (r, "C") not in atoms:
            continue
        ca.append(atoms[(r, "CA")])
        n_xyz.append(atoms[(r, "N")])
        c_xyz.append(atoms[(r, "C")])
        resseqs.append(r)
        bfac.append(bfactors.get(r, float("nan")))
    if len(ca) < 20:
        raise ValueError(f"too few complete backbone residues: {len(ca)}")
    return {
        "ca": np.stack(ca, axis=0),
        "n": np.stack(n_xyz, axis=0),
        "c": np.stack(c_xyz, axis=0),
        "resseq": np.array(resseqs, dtype=np.int32),
        "ca_b": np.array(bfac, dtype=np.float64),
    }


def backbone_frames(n_xyz: np.ndarray, ca: np.ndarray, c_xyz: np.ndarray) -> np.ndarray:
    """Orthonormal frames at Cα: x along Cα→C, z ~ (Cα→C)×(Cα→N), y = z×x."""
    x = c_xyz - ca
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(xn, 1e-8, None)
    nvec = n_xyz - ca
    z = np.cross(x, nvec)
    zn = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(zn, 1e-8, None)
    y = np.cross(z, x)
    frames = np.stack([x, y, z], axis=2)  # [n, 3, 3] columns
    return frames


def type_contacts(
    ca: np.ndarray,
    *,
    contact: float = CONTACT_CUTOFF,
    distal: float = DISTAL_CUTOFF,
    short_seq_max: int = SHORT_SEQ_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected residue pairs, stored i < j. Types from sequence gap + Cα distance."""
    n = ca.shape[0]
    src, dst, types = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ca[i] - ca[j]))
            gap = j - i
            if gap == 1:
                t = 0
            elif gap <= short_seq_max and d < contact:
                t = 1
            elif gap >= 5 and d < contact:
                t = 2
            elif gap >= 5 and d < distal:
                t = 3
            else:
                continue
            src.append(i)
            dst.append(j)
            types.append(t)
    if not src:
        raise ValueError("no typed contacts")
    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(types, dtype=np.int64),
    )


def native_linear_fit_residual(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Legacy Cα Gate 0 used by Test P-A. Prefer evaluate_gate0()."""
    return linear_pair_law(
        src_xyz, dst_xyz, rng, n_splits=1, holdout=0.0, scale_mode="pair_length"
    )


def _lstsq_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    src_t = torch.from_numpy(src.astype(np.float32))
    dst_t = torch.from_numpy(dst.astype(np.float32))
    a = torch.linalg.lstsq(src_t, dst_t).solution.T
    return float((dst_t - (a @ src_t.T).T).norm(dim=1).mean().item())


def linear_pair_law(
    src: np.ndarray,
    dst: np.ndarray,
    rng: np.random.Generator,
    *,
    n_splits: int = 5,
    holdout: float = 0.3,
    scale_mode: str = "feature_rms",
) -> Dict[str, float]:
    """Shared map x_v ≈ A x_u. Fit on a train split; score held-out pairs.

    Pre-registered pass (paper model):
      holdout native RMSE < 0.25 × RMS(destination)  and
      holdout native RMSE < 0.50 × holdout shuffle RMSE.
    """
    n, dim = src.shape
    feat_rms = float(np.sqrt(np.mean(dst * dst)))
    pair_len = float(np.linalg.norm(dst - src, axis=1).mean()) if dim == 3 else feat_rms
    if n < max(8, 4 * dim):
        return {
            "n": float(n),
            "dim": float(dim),
            "native_rmse": float("nan"),
            "shuffle_rmse": float("nan"),
            "feature_rms": feat_rms,
            "mean_pair_distance": pair_len,
            "usable_linear_pair_law": False,
            "too_few_pairs": True,
        }

    native_scores, shuffle_scores = [], []
    for _ in range(n_splits):
        n_hold = max(int(round(holdout * n)), dim + 2) if holdout > 0 else 0
        if n_hold == 0 or n_hold >= n - (dim + 2):
            native_scores.append(_lstsq_rmse(src, dst))
            perm = rng.permutation(n)
            shuffle_scores.append(_lstsq_rmse(src, dst[perm]))
            continue
        perm = rng.permutation(n)
        hold, train = perm[:n_hold], perm[n_hold:]
        a = np.linalg.lstsq(src[train], dst[train], rcond=None)[0].T
        native_scores.append(
            float(np.linalg.norm(dst[hold] - (a @ src[hold].T).T, axis=1).mean())
        )
        perm_all = rng.permutation(n)
        dst_sh = dst[perm_all]
        a_null = np.linalg.lstsq(src[train], dst_sh[train], rcond=None)[0].T
        shuffle_scores.append(
            float(np.linalg.norm(dst_sh[hold] - (a_null @ src[hold].T).T, axis=1).mean())
        )

    native = float(np.mean(native_scores))
    shuffle = float(np.mean(shuffle_scores))
    # World-frame coordinates: scale is pair length, not ||x|| from the origin.
    # Unit frames: scale is feature RMS.
    scale = pair_len if scale_mode == "pair_length" else feat_rms
    usable = bool(native < 0.25 * scale and native < 0.50 * shuffle)
    return {
        "n": float(n),
        "dim": float(dim),
        "n_splits": float(n_splits),
        "native_rmse": native,
        "shuffle_rmse": shuffle,
        "feature_rms": feat_rms,
        "mean_pair_distance": pair_len,
        "rmse_over_feature_rms": native / max(feat_rms, 1e-8),
        "rmse_over_scale": native / max(scale, 1e-8),
        "native_over_shuffle": native / max(shuffle, 1e-8),
        "native_better_than_shuffle": bool(native < 0.8 * shuffle),
        "usable_linear_pair_law": usable,
        "too_few_pairs": False,
    }


def _so3_geodesic(r: np.ndarray) -> float:
    tr = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(tr))


def _chordal_mean_so3(rots: np.ndarray) -> np.ndarray:
    m = rots.sum(axis=0)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def relative_pose_concentration(
    frames: np.ndarray,
    ca: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Typed contacts share a typical relative pose (rotation + local translation).

    This is not the paper's x_v ≈ A x_u on node stalks. It is the natural
    molecular pair object: how frame j sits in frame i.

    Pre-registered pass:
      mean geodesic to type mean < 0.50 × random residue pairs, and
      translation RMSE to type mean < 0.25 × mean |t|.
    """
    n_pairs = int(src_idx.shape[0])
    n_res = int(ca.shape[0])
    if n_pairs < 8:
        return {"n": float(n_pairs), "usable_shared_relative_pose": False, "too_few_pairs": True}

    rel = np.einsum("nij,njk->nik", frames[src_idx].transpose(0, 2, 1), frames[dst_idx])
    t_loc = np.einsum("nji,nj->ni", frames[src_idx], ca[dst_idx] - ca[src_idx])
    r_mean = _chordal_mean_so3(rel)
    geo = np.array([_so3_geodesic(r_mean.T @ rel[k]) for k in range(n_pairs)])
    t_mean = t_loc.mean(axis=0)
    t_rmse = float(np.linalg.norm(t_loc - t_mean, axis=1).mean())
    t_scale = float(np.linalg.norm(t_loc, axis=1).mean())
    geo_mean = float(geo.mean())

    rand_geo, rand_t = [], []
    for _ in range(8):
        i = rng.integers(0, n_res, size=n_pairs)
        j = rng.integers(0, n_res, size=n_pairs)
        same = i == j
        j = np.where(same, (j + 1) % n_res, j)
        rel_r = np.einsum("nij,njk->nik", frames[i].transpose(0, 2, 1), frames[j])
        t_r = np.einsum("nji,nj->ni", frames[i], ca[j] - ca[i])
        r_m = _chordal_mean_so3(rel_r)
        rand_geo.append(np.mean([_so3_geodesic(r_m.T @ rel_r[k]) for k in range(n_pairs)]))
        rand_t.append(float(np.linalg.norm(t_r - t_r.mean(axis=0), axis=1).mean()))

    geo_null = float(np.mean(rand_geo))
    t_null = float(np.mean(rand_t))
    usable = bool(geo_mean < 0.50 * geo_null and t_rmse < 0.25 * t_scale)
    return {
        "n": float(n_pairs),
        "geodesic_mean_rad": geo_mean,
        "geodesic_random_rad": geo_null,
        "geodesic_over_random": geo_mean / max(geo_null, 1e-8),
        "translation_rmse": t_rmse,
        "translation_scale": t_scale,
        "translation_over_scale": t_rmse / max(t_scale, 1e-8),
        "translation_random_rmse": t_null,
        "usable_shared_relative_pose": usable,
        "too_few_pairs": False,
    }


def frame_features(frames: np.ndarray) -> np.ndarray:
    """6-dim stalk: first two columns of the Cα backbone frame."""
    return np.concatenate([frames[:, :, 0], frames[:, :, 1]], axis=1)


def evaluate_gate0(
    graph: Dict[str, np.ndarray],
    *,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Native pair-law checks. Pass rules are fixed before looking at numbers."""
    rng = np.random.default_rng(seed)
    ca = graph["ca"].astype(np.float64)
    frames = graph["frames"].astype(np.float64)
    src, dst, types = graph["src"], graph["dst"], graph["types"]
    feat6 = frame_features(frames)
    axis = frames[:, :, 0]

    out: Dict[str, Dict[str, Dict[str, float]]] = {
        "ca_world": {},
        "frame_6": {},
        "frame_x_axis": {},
        "relative_pose": {},
    }
    for r, name in enumerate(TYPE_NAMES):
        m = types == r
        out["ca_world"][name] = linear_pair_law(
            ca[src[m]], ca[dst[m]], rng, scale_mode="pair_length"
        )
        out["frame_6"][name] = linear_pair_law(feat6[src[m]], feat6[dst[m]], rng)
        out["frame_x_axis"][name] = linear_pair_law(axis[src[m]], axis[dst[m]], rng)
        out["relative_pose"][name] = relative_pose_concentration(
            frames, ca, src[m], dst[m], rng
        )
    return out


def gate0_verdict(report: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, bool]:
    """Paper leftover needs a usable linear pair law on node features.

    relative_pose is a different object (typical contact geometry). It does
    not by itself authorize P-B with IT-GLLS on residue stalks.
    """
    linear_ok = False
    linear_contact_ok = False
    frame_contact_ok = False
    pose_ok = False
    pose_contact_ok = False
    for name in TYPE_NAMES:
        for feat in ("ca_world", "frame_6", "frame_x_axis"):
            if bool(report[feat][name].get("usable_linear_pair_law")):
                linear_ok = True
                if name != "backbone":
                    linear_contact_ok = True
        for feat in ("frame_6", "frame_x_axis"):
            if name != "backbone" and bool(report[feat][name].get("usable_linear_pair_law")):
                frame_contact_ok = True
        if bool(report["relative_pose"][name].get("usable_shared_relative_pose")):
            pose_ok = True
            if name != "backbone":
                pose_contact_ok = True
    return {
        "linear_pair_law_any_type": linear_ok,
        "linear_pair_law_non_backbone": linear_contact_ok,
        "frame_linear_law_non_backbone": frame_contact_ok,
        "relative_pose_any_type": pose_ok,
        "relative_pose_non_backbone": pose_contact_ok,
        "gate0_pass_for_paper_leftover": frame_contact_ok,
        "gate0_pass_for_pose_leftover": pose_contact_ok,
    }


@dataclass
class FrameSnap:
    t: int
    pair_src: Tensor
    pair_dst: Tensor
    edge_types: Tensor
    labels: Tensor
    src_ids: Tensor
    dst_ids: Tensor
    ca: Tensor
    R: Tensor


def load_protein_graph(pdb_path: Path, chain: str = "A") -> Dict[str, np.ndarray]:
    text = Path(pdb_path).read_text(errors="replace")
    bb = parse_backbone(text, chain=chain)
    src, dst, types = type_contacts(bb["ca"])
    frames = backbone_frames(bb["n"], bb["ca"], bb["c"])
    return {**bb, "src": src, "dst": dst, "types": types, "frames": frames}


def inject_protein(
    graph: Dict[str, np.ndarray],
    *,
    n_steps: int = 12,
    t_star_index: int = 7,
    feat_dim: int = 16,
    proj_dim: int = 12,
    anomaly_frac: float = 0.10,
    noise: float = 0.05,
    omega: float = 0.03,
    jump_mag: float = 0.8,
    coord_noise: float = 0.05,
    seed: int = 42,
) -> Tuple[List[FrameSnap], Tensor, Tensor, Dict[str, Dict[str, float]]]:
    n_res = int(graph["ca"].shape[0])
    src_np = graph["src"]
    dst_np = graph["dst"]
    types_np = graph["types"]
    ca0 = graph["ca"].astype(np.float64)

    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.randn(n_res, feat_dim, generator=gen)
    proj = random_ortho_rows(proj_dim, feat_dim, gen)
    x_proj = x @ proj.T
    true_R = build_R_trajectory(
        n_steps, proj_dim, t_star_index, AFFECTED_TYPE, seed + 1, omega, jump_mag
    )

    gate0: Dict[str, Dict[str, float]] = {}
    for r, name in enumerate(TYPE_NAMES):
        m = types_np == r
        gate0[name] = native_linear_fit_residual(
            ca0[src_np[m]], ca0[dst_np[m]], rng
        )

    snaps: List[FrameSnap] = []
    src = torch.from_numpy(src_np)
    dst = torch.from_numpy(dst_np)
    types = torch.from_numpy(types_np)
    n_e = int(src.shape[0])
    pair_src = x_proj[src]

    for t_i in range(n_steps):
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
        ca_t = ca0 + coord_noise * rng.normal(size=ca0.shape)
        snaps.append(
            FrameSnap(
                t=t_i,
                pair_src=pair_src,
                pair_dst=pair_dst,
                edge_types=types,
                labels=labels,
                src_ids=src,
                dst_ids=dst,
                ca=torch.from_numpy(ca_t.astype(np.float32)),
                R=true_R[t_i],
            )
        )
    return snaps, proj, true_R, gate0

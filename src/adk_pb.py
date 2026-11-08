"""P-B object for AdK: per-contact leftover vs its own apo→holo past.

Not a typed shared R_r. Local sender frames only; no whole-protein superposition
in the leftover. Domain/hinge labels are pre-registered from the 4AKE/1AKE
literature, not from leftover.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from pharma_inject import _so3_geodesic, backbone_frames

# Pre-registered before leftover. E. coli AdK, Muller 1996 / standard 4AKE–1AKE split
# (LID 122–159, NMP 30–59, CORE the rest). Junctions ±1 are the hinge band.
CORE_RANGES: Tuple[Tuple[int, int], ...] = ((1, 29), (60, 121), (160, 214))
NMP_RANGE = (30, 59)
LID_RANGE = (122, 159)
HINGE_RES: Tuple[int, ...] = (
    28, 29, 30, 31,
    58, 59, 60, 61,
    120, 121, 122, 123,
    158, 159, 160, 161,
)
TERMINI_N = 5
TERMINI_C = 5
CONTACT_CUTOFF = 8.0
MIN_SEQ_GAP = 3


def in_ranges(resseq: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(a <= resseq <= b for a, b in ranges)


def region_of(resseq: int) -> str:
    if resseq in HINGE_RES:
        return "hinge"
    if NMP_RANGE[0] <= resseq <= NMP_RANGE[1]:
        return "nmp_interior"
    if LID_RANGE[0] <= resseq <= LID_RANGE[1]:
        return "lid_interior"
    if in_ranges(resseq, CORE_RANGES):
        return "core_interior"
    return "other"


def termini_set(resseqs: np.ndarray) -> Set[int]:
    lo = int(resseqs.min())
    hi = int(resseqs.max())
    return set(range(lo, lo + TERMINI_N)) | set(range(hi - TERMINI_C + 1, hi + 1))


def high_b_set(resseqs: np.ndarray, bfac: np.ndarray, q: float = 0.90) -> Set[int]:
    ok = np.isfinite(bfac)
    if int(ok.sum()) < 10:
        return set()
    thresh = float(np.quantile(bfac[ok], q))
    return {int(r) for r, b, k in zip(resseqs, bfac, ok) if k and b >= thresh}


def align_intersection(
    apo: Dict[str, np.ndarray],
    holo: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Match residues by resseq; keep only complete backbone in both states."""
    apo_map = {int(r): i for i, r in enumerate(apo["resseq"])}
    holo_map = {int(r): i for i, r in enumerate(holo["resseq"])}
    common = sorted(set(apo_map) & set(holo_map))
    if len(common) < 50:
        raise ValueError(f"too few shared residues: {len(common)}")
    ia = [apo_map[r] for r in common]
    ih = [holo_map[r] for r in common]
    return {
        "resseq": np.array(common, dtype=np.int32),
        "ca_apo": apo["ca"][ia],
        "ca_holo": holo["ca"][ih],
        "n_apo": apo["n"][ia],
        "n_holo": holo["n"][ih],
        "c_apo": apo["c"][ia],
        "c_holo": holo["c"][ih],
        "b_apo": apo["ca_b"][ia],
        "b_holo": holo["ca_b"][ih],
    }


def kabsch_displace(ref: np.ndarray, mov: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Cα displacement of mov onto ref. If mask is set, fit only on those atoms."""
    if mask is None:
        mask = np.ones(ref.shape[0], dtype=bool)
    a = ref[mask]
    b = mov[mask]
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    cov = ac.T @ bc
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = a.mean(axis=0) - (b.mean(axis=0) @ r)
    aligned = mov @ r + t
    return np.linalg.norm(aligned - ref, axis=1)


def contact_pairs(ca_a: np.ndarray, ca_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = ca_a.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(i + MIN_SEQ_GAP, n):
            da = float(np.linalg.norm(ca_a[i] - ca_a[j]))
            dh = float(np.linalg.norm(ca_b[i] - ca_b[j]))
            if da < CONTACT_CUTOFF or dh < CONTACT_CUTOFF:
                src.append(i)
                dst.append(j)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def pair_leftover(
    frames_a: np.ndarray,
    frames_h: np.ndarray,
    ca_a: np.ndarray,
    ca_h: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Relative pose of j in i's local frame, apo vs holo. No global superposition."""
    rel_a = np.einsum("nij,njk->nik", frames_a[src].transpose(0, 2, 1), frames_a[dst])
    rel_h = np.einsum("nij,njk->nik", frames_h[src].transpose(0, 2, 1), frames_h[dst])
    t_a = np.einsum("nji,nj->ni", frames_a[src], ca_a[dst] - ca_a[src])
    t_h = np.einsum("nji,nj->ni", frames_h[src], ca_h[dst] - ca_h[src])
    geo = np.array([_so3_geodesic(rel_a[k].T @ rel_h[k]) for k in range(src.shape[0])])
    dt = np.linalg.norm(t_h - t_a, axis=1)
    # Translation in Å; rotation in rad. Combined leftover uses 1 rad ~ 3.8 Å.
    combined = geo + dt / 3.8
    return {"geodesic": geo, "dtrans": dt, "combined": combined}


def residue_scores(
    n_res: int,
    src: np.ndarray,
    dst: np.ndarray,
    pair_score: np.ndarray,
    keep_pair: np.ndarray,
) -> np.ndarray:
    acc = np.zeros(n_res)
    cnt = np.zeros(n_res)
    for s, d, v, k in zip(src, dst, pair_score, keep_pair):
        if not k:
            continue
        acc[s] += v
        acc[d] += v
        cnt[s] += 1
        cnt[d] += 1
    out = np.full(n_res, np.nan)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out

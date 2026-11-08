"""AdK domain-coupling test on DIMS closed→open paths.

Pre-registered pairs: LID–CORE and NMP–CORE (Muller / 4AKE–1AKE cuts).
Not a typed shared map. Not two crystal snapshots.

Data: Seyler et al. DIMS MD, CORE-fitted DCDs (closed=1AKE-like start,
open=4AKE-like end). This is targeted transition sampling, not equilibrium
apo/holo MD; still a many-frame dynamical ensemble with replicas.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from adk_pb import CORE_RANGES, LID_RANGE, NMP_RANGE

TAU = 1
PCA_DIM = 8
BASIN_FRAC = 0.25
N_PATHS = 10
HOLDOUT = 0.30
N_SPLITS = 5
SEED = 42
N_SHUFFLE = 8

PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("lid_core", "lid", "core"),
    ("nmp_core", "nmp", "core"),
)


def parse_psf_ca(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return CA atom indices (0-based) and residue numbers."""
    text = path.read_text(errors="replace").splitlines()
    n_atom = None
    start = None
    for i, line in enumerate(text):
        if "!NATOM" in line:
            n_atom = int(line.split()[0])
            start = i + 1
            break
    if n_atom is None or start is None:
        raise ValueError("no NATOM in psf")
    idx, res = [], []
    for line in text[start : start + n_atom]:
        parts = line.split()
        if len(parts) < 5:
            continue
        serial = int(parts[0]) - 1
        resid = int(parts[2])
        name = parts[4]
        if name == "CA":
            idx.append(serial)
            res.append(resid)
    if len(idx) < 50:
        raise ValueError(f"too few CA: {len(idx)}")
    return np.asarray(idx, dtype=np.int64), np.asarray(res, dtype=np.int32)


def read_dcd_xyz(path: Path) -> np.ndarray:
    """CHARMM CORD DCD → [F, N, 3] float64 (all atoms)."""
    with path.open("rb") as f:

        def rec() -> bytes:
            raw = f.read(4)
            if len(raw) < 4:
                raise EOFError
            n = struct.unpack("<i", raw)[0]
            data = f.read(n)
            n2 = struct.unpack("<i", f.read(4))[0]
            if n != n2 or len(data) != n:
                raise ValueError("corrupt dcd record")
            return data

        hdr = rec()
        if hdr[:4] != b"CORD":
            raise ValueError("not a CORD dcd")
        nset = struct.unpack_from("<i", hdr, 4)[0]
        rec()  # title
        n_atoms = struct.unpack("<i", rec())[0]
        frames = []
        while True:
            pos = f.tell()
            peek = f.read(4)
            if len(peek) < 4:
                break
            n = struct.unpack("<i", peek)[0]
            f.seek(pos)
            if n == 48:
                rec()
            x = np.frombuffer(rec(), dtype="<f4")
            y = np.frombuffer(rec(), dtype="<f4")
            z = np.frombuffer(rec(), dtype="<f4")
            if x.size != n_atoms:
                raise ValueError("atom count mismatch in frame")
            frames.append(np.stack([x, y, z], axis=1))
            if nset > 0 and len(frames) >= nset:
                break
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames, axis=0).astype(np.float64)


def region_mask(resseq: np.ndarray, name: str) -> np.ndarray:
    r = resseq.astype(int)
    if name == "lid":
        return (r >= LID_RANGE[0]) & (r <= LID_RANGE[1])
    if name == "nmp":
        return (r >= NMP_RANGE[0]) & (r <= NMP_RANGE[1])
    if name == "core":
        m = np.zeros(r.shape, dtype=bool)
        for a, b in CORE_RANGES:
            m |= (r >= a) & (r <= b)
        return m
    raise ValueError(name)


def _com(xyz: np.ndarray) -> np.ndarray:
    return xyz.mean(axis=1)


def flatten_region(ca: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[F, 3*n_res] Cα of a region, minus that region's COM each frame."""
    sub = ca[:, mask]
    return (sub - sub.mean(axis=1, keepdims=True)).reshape(ca.shape[0], -1)


def fit_pca(x: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    xc = x - mu
    # SVD on centered features
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(dim, vt.shape[0])
    basis = vt[:k]
    return mu, basis, xc @ basis.T


def apply_pca(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mu) @ basis.T


def fit_A(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """dst ≈ A @ [src; 1]. Returns A with shape [d_dst, d_src+1]."""
    n = src.shape[0]
    aug = np.concatenate([src, np.ones((n, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(aug, dst, rcond=None)
    return coef.T


def predict_A(A: np.ndarray, src: np.ndarray) -> np.ndarray:
    aug = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    return (A @ aug.T).T


def rmse_map(A: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    err = dst - predict_A(A, src)
    return float(np.linalg.norm(err, axis=1).mean())


def frobenius_linear(A: np.ndarray, B: np.ndarray) -> float:
    """Distance on the linear block, not the intercept."""
    return float(np.linalg.norm(A[:, :-1] - B[:, :-1], ord="fro"))


def time_holdout_rmse(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    holdout: float = HOLDOUT,
) -> Dict[str, float]:
    n = src.shape[0]
    n_hold = max(int(round(holdout * n)), src.shape[1] + 3)
    if n_hold >= n - (src.shape[1] + 3):
        n_hold = max(n // 4, src.shape[1] + 2)
    train, hold = slice(0, n - n_hold), slice(n - n_hold, n)
    A = fit_A(src[train], dst[train])
    native = rmse_map(A, src[hold], dst[hold])
    rng = np.random.default_rng(SEED)
    shuf = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(n)
        dst_s = dst[perm]
        A_s = fit_A(src[train], dst_s[train])
        shuf.append(rmse_map(A_s, src[hold], dst_s[hold]))
    shuffle = float(np.mean(shuf))
    scale = float(np.sqrt(np.mean(dst[hold] * dst[hold])))
    return {
        "n": float(n),
        "n_hold": float(n_hold),
        "native_rmse": native,
        "shuffle_rmse": shuffle,
        "feature_rms": scale,
        "rmse_over_scale": native / max(scale, 1e-8),
        "native_over_shuffle": native / max(shuffle, 1e-8),
        "gate0_pass": bool(native < 0.25 * scale and native < 0.50 * shuffle),
    }


def lagged(x: np.ndarray, tau: int = TAU) -> Tuple[np.ndarray, np.ndarray]:
    return x[:-tau], x[tau:]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if d < 1e-12:
        return float("nan")
    return float((a * b).sum() / d)


def mi_gaussian(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian MI of two 1-d series (nats)."""
    r = pearson(a, b)
    if not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    return float(-0.5 * np.log(max(1.0 - r * r, 1e-12)))


def region_rmsd(ca: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    sub = ca[:, mask]
    r = ref[mask]
    return np.linalg.norm(sub - r, axis=2).mean(axis=1)


def logistic_auc(X: np.ndarray, y: np.ndarray) -> float:
    if len(set(y.tolist())) < 2 or X.shape[0] < 20:
        return float("nan")
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def build_pair_features(
    ca_closed: np.ndarray,
    ca_open: np.ndarray,
    resseq: np.ndarray,
    src_name: str,
    dst_name: str,
) -> Dict[str, np.ndarray]:
    ms = region_mask(resseq, src_name)
    md = region_mask(resseq, dst_name)
    src_c = flatten_region(ca_closed, ms)
    dst_c = flatten_region(ca_closed, md)
    src_o = flatten_region(ca_open, ms)
    dst_o = flatten_region(ca_open, md)
    mu, basis, src_c_p = fit_pca(src_c, PCA_DIM)
    _, _, dst_c_p = fit_pca(dst_c, PCA_DIM)
    # share source PCA; destination PCA from closed only
    mu_d = dst_c.mean(axis=0)
    xc = dst_c - mu_d
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(PCA_DIM, vt.shape[0])
    basis_d = vt[:k]
    dst_c_p = xc @ basis_d.T
    src_o_p = apply_pca(src_o, mu, basis)
    dst_o_p = apply_pca(dst_o, mu_d, basis_d)
    return {
        "src_closed": src_c_p,
        "dst_closed": dst_c_p,
        "src_open": src_o_p,
        "dst_open": dst_o_p,
        "com_src_c": _com(ca_closed[:, ms]),
        "com_dst_c": _com(ca_closed[:, md]),
        "com_src_o": _com(ca_open[:, ms]),
        "com_dst_o": _com(ca_open[:, md]),
        "ca_closed": ca_closed,
        "ca_open": ca_open,
        "mask_src": ms,
        "mask_dst": md,
    }


def evaluate_pair(feat: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, object]:
    src_c, dst_c = lagged(feat["src_closed"])
    src_o, dst_o = lagged(feat["src_open"])
    gate0 = time_holdout_rmse(src_c, dst_c)

    A_c = fit_A(src_c, dst_c)
    A_o = fit_A(src_o, dst_o)
    D = frobenius_linear(A_c, A_o)

    # shuffle: destroy temporal pairing in closed
    shuf_D = []
    shuf_rmse = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(src_c.shape[0])
        A_s = fit_A(src_c, dst_c[perm])
        shuf_rmse.append(rmse_map(A_s, src_c, dst_c[perm]))
        shuf_D.append(frobenius_linear(A_c, A_s))
    coupling_vs_shuffle = {
        "closed_rmse": rmse_map(A_c, src_c, dst_c),
        "shuffled_rmse": float(np.mean(shuf_rmse)),
        "D_to_shuffled_A": float(np.mean(shuf_D)),
        "D_apo_holo": D,
        "shuffle_distinguishes": bool(
            rmse_map(A_c, src_c, dst_c) < 0.8 * float(np.mean(shuf_rmse))
        ),
    }

    d_c = np.linalg.norm(feat["com_dst_c"] - feat["com_src_c"], axis=1)
    d_o = np.linalg.norm(feat["com_dst_o"] - feat["com_src_o"], axis=1)
    ref = feat["ca_closed"][0]
    rmsd_c = region_rmsd(feat["ca_closed"], feat["mask_dst"], ref)
    rmsd_o = region_rmsd(feat["ca_open"], feat["mask_dst"], ref)

    def rmsf(ca: np.ndarray, mask: np.ndarray) -> float:
        sub = ca[:, mask]
        mu = sub.mean(axis=0, keepdims=True)
        return float(np.sqrt(((sub - mu) ** 2).sum(axis=2).mean()))

    lid_c = flatten_region(feat["ca_closed"], feat["mask_dst"])
    mu = lid_c.mean(axis=0)
    _, _, vt = np.linalg.svd(lid_c - mu, full_matrices=False)
    pc_c = (lid_c - mu) @ vt[0]
    lid_o = flatten_region(feat["ca_open"], feat["mask_dst"])
    pc_o = (lid_o - mu) @ vt[0]

    r_c = np.linalg.norm(dst_c - predict_A(A_c, src_c), axis=1)
    r_o = np.linalg.norm(dst_o - predict_A(A_c, src_o), axis=1)

    # frame-level closed=0 open=1 (drop last tau frames already in r_*)
    y = np.concatenate(
        [np.zeros(r_c.shape[0], dtype=int), np.ones(r_o.shape[0], dtype=int)]
    )
    d_all = np.concatenate([d_c[TAU:], d_o[TAU:]])
    rmsd_all = np.concatenate([rmsd_c[TAU:], rmsd_o[TAU:]])
    pc_all = np.concatenate([pc_c[TAU:], pc_o[TAU:]])
    r_all = np.concatenate([r_c, r_o])
    model_a = np.stack([d_all, rmsd_all, pc_all], axis=1)
    model_b = np.concatenate([model_a, r_all[:, None]], axis=1)
    auc_a = logistic_auc(model_a, y)
    auc_b = logistic_auc(model_b, y)

    return {
        "gate0": gate0,
        "D_frobenius": D,
        "coupling_vs_shuffle": coupling_vs_shuffle,
        "baselines": {
            "mean_interregion_distance_closed": float(d_c.mean()),
            "mean_interregion_distance_open": float(d_o.mean()),
            "delta_distance": float(d_o.mean() - d_c.mean()),
            "mean_rmsd_dst_closed": float(rmsd_c.mean()),
            "mean_rmsd_dst_open": float(rmsd_o.mean()),
            "rmsf_dst_closed": rmsf(feat["ca_closed"], feat["mask_dst"]),
            "rmsf_dst_open": rmsf(feat["ca_open"], feat["mask_dst"]),
            "dcc_com": pearson(
                np.linalg.norm(np.diff(feat["com_src_c"], axis=0), axis=1),
                np.linalg.norm(np.diff(feat["com_dst_c"], axis=0), axis=1),
            ),
            "mi_com_distance_vs_src_speed_closed": mi_gaussian(
                d_c[1:], np.linalg.norm(np.diff(feat["com_src_c"], axis=0), axis=1)
            ),
        },
        "incremental": {
            "auc_distance_rmsd_pca": auc_a,
            "auc_plus_coupling_residual": auc_b,
            "delta_auc": (auc_b - auc_a) if np.isfinite(auc_a) and np.isfinite(auc_b) else float("nan"),
            "spearman_residual_vs_distance": float(
                _spearman(r_all, d_all)
            ),
        },
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(v)).astype(np.float64)

    return pearson(rank(a), rank(b))


def load_paths(
    traj_dir: Path,
    psf: Path,
    n_paths: int = N_PATHS,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    ca_idx, resseq = parse_psf_ca(psf)
    trajs = []
    for i in range(1, n_paths + 1):
        dcd = traj_dir / f"dims{i:04d}_fit-core.dcd"
        xyz = read_dcd_xyz(dcd)
        trajs.append(xyz[:, ca_idx])
    return resseq, trajs


def split_basin(ca: np.ndarray, frac: float = BASIN_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    n = ca.shape[0]
    k = max(int(round(frac * n)), 8)
    return ca[:k], ca[-k:]


def run(traj_dir: Path, psf: Path, n_paths: int = N_PATHS, seed: int = SEED) -> Dict:
    resseq, trajs = load_paths(traj_dir, psf, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    closed_rep1 = np.concatenate([split_basin(t)[0] for t in trajs[:5]], axis=0)
    open_rep1 = np.concatenate([split_basin(t)[1] for t in trajs[:5]], axis=0)
    closed_rep2 = np.concatenate([split_basin(t)[0] for t in trajs[5:]], axis=0)
    open_rep2 = np.concatenate([split_basin(t)[1] for t in trajs[5:]], axis=0)

    out: Dict = {
        "protocol": (
            "AdK LID–CORE and NMP–CORE coupling on DIMS closed→open paths "
            "(Seyler 2015; CORE-fitted). Closed = first 25% of each path, "
            "open = last 25%. X_j(t+1)≈A X_i(t) in 8-d PCA of COM-centered "
            "region Cα. Gate 0 is time holdout on closed vs destination shuffle. "
            "Replicas: paths 1–5 vs 6–10. Incremental: frame-level closed vs open "
            "logistic AUC of [distance, RMSD, PC1] vs same plus closed-map residual."
        ),
        "data_note": (
            "DIMS is importance-sampled transitions, not equilibrium apo vs holo MD."
        ),
        "n_paths": n_paths,
        "n_ca": int(resseq.size),
        "pre_registered_pairs": [p[0] for p in PAIRS],
        "tau": TAU,
        "pca_dim": PCA_DIM,
        "basin_frac": BASIN_FRAC,
        "replicas": {},
        "go_no_go": {},
    }

    for rep_name, closed, opened in (
        ("paths_1_5", closed_rep1, open_rep1),
        ("paths_6_10", closed_rep2, open_rep2),
    ):
        rec = {}
        for pair_id, src_n, dst_n in PAIRS:
            feat = build_pair_features(closed, opened, resseq, src_n, dst_n)
            rec[pair_id] = evaluate_pair(feat, rng)
        out["replicas"][rep_name] = rec

    # go/no-go from replica 1, reproducibility vs replica 2
    gng = {}
    for pair_id, _, _ in PAIRS:
        r1 = out["replicas"]["paths_1_5"][pair_id]
        r2 = out["replicas"]["paths_6_10"][pair_id]
        A = bool(r1["D_frobenius"] > 0.0)  # operator numbers differ; weak
        A = bool(abs(r1["D_frobenius"]) > 1e-6)
        B = bool(
            np.sign(r1["baselines"]["delta_distance"])
            == np.sign(r2["baselines"]["delta_distance"])
            and r1["D_frobenius"] > 0
            and r2["D_frobenius"] > 0
        )
        C = bool(
            np.isfinite(r1["incremental"]["delta_auc"])
            and r1["incremental"]["delta_auc"] > 0.02
            and abs(r1["incremental"]["spearman_residual_vs_distance"]) < 0.7
        )
        gng[pair_id] = {
            "A_operator_changes": A,
            "B_reproducible_across_path_groups": B
            and (
                abs(r1["D_frobenius"] - r2["D_frobenius"])
                / max(r1["D_frobenius"], 1e-8)
                < 1.0
            ),
            "C_incremental_beyond_motion": C,
            "gate0_closed": bool(r1["gate0"]["gate0_pass"]),
            "shuffle_control": bool(
                r1["coupling_vs_shuffle"]["shuffle_distinguishes"]
            ),
            "proceed": False,
        }
        gng[pair_id]["proceed"] = bool(
            gng[pair_id]["gate0_closed"]
            and gng[pair_id]["shuffle_control"]
            and gng[pair_id]["A_operator_changes"]
            and gng[pair_id]["B_reproducible_across_path_groups"]
            and gng[pair_id]["C_incremental_beyond_motion"]
        )
    out["go_no_go"] = gng
    out["verdict"] = {
        "any_pair_proceed": any(v["proceed"] for v in gng.values()),
        "md_claim_supported": any(v["proceed"] for v in gng.values()),
    }
    return out


EQ_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("core_to_lid", "core", "lid"),
    ("core_to_nmp", "core", "nmp"),
    ("lid_to_core", "lid", "core"),
    ("nmp_to_core", "nmp", "core"),
)
EQ_TAUS: Tuple[int, ...] = (1, 5)


def run_equilibrium(
    dcd: Path,
    psf: Path,
    *,
    seed: int = SEED,
) -> Dict:
    """Single-state equilibrium: can X_i(t) predict X_j(t+τ)?

    Beckstein 1 µs AdK (no water, CORE-fitted, 240 ps/frame). First vs second
    half are two blocks in the *same* basin, not closed vs open.
    """
    ca_idx, resseq = parse_psf_ca(psf)
    xyz = read_dcd_xyz(dcd)[:, ca_idx]
    n_f = xyz.shape[0]
    mid = n_f // 2
    blocks = {"first_half": xyz[:mid], "second_half": xyz[mid:]}
    rng = np.random.default_rng(seed)

    out: Dict = {
        "protocol": (
            "Equilibrium AdK (Beckstein 1 µs, CORE-fitted DCD, dt=240 ps). "
            "Same LID/NMP/CORE cuts as DIMS. Question only: does X_i(t) linearly "
            "predict X_j(t+τ) on ordinary fluctuations? Gate 0 = last-30% time "
            "holdout RMSE < 0.25× dest RMS and < 0.50× destination shuffle. "
            "Primary maps: CORE→LID and CORE→NMP. Reverse maps reported. "
            "τ=1 (240 ps) is the pass cell; τ=5 is diagnostic. Halves are "
            "same-state replicas, not a transition."
        ),
        "source": {
            "citation": "MDAnalysis AdK equilibrium (figshare 5108170); 1ake_007-nowater-core-dt240ps.dcd",
            "n_frames": int(n_f),
            "dt_ps": 240,
            "n_ca": int(resseq.size),
        },
        "blocks": {},
        "verdict": {},
    }

    for block_name, ca in blocks.items():
        rec: Dict = {}
        for tau in EQ_TAUS:
            rec[f"tau_{tau}"] = {}
            for pair_id, src_n, dst_n in EQ_PAIRS:
                ms = region_mask(resseq, src_n)
                md = region_mask(resseq, dst_n)
                src_f = flatten_region(ca, ms)
                dst_f = flatten_region(ca, md)
                mu_s, basis_s, src_p = fit_pca(src_f, PCA_DIM)
                mu_d = dst_f.mean(axis=0)
                xc = dst_f - mu_d
                _, _, vt = np.linalg.svd(xc, full_matrices=False)
                k = min(PCA_DIM, vt.shape[0])
                dst_p = xc @ vt[:k].T
                s_lag, d_lag = src_p[:-tau], dst_p[tau:]
                gate0 = time_holdout_rmse(s_lag, d_lag)
                A = fit_A(s_lag, d_lag)
                perm = rng.permutation(s_lag.shape[0])
                A_sh = fit_A(s_lag, d_lag[perm])
                rec[f"tau_{tau}"][pair_id] = {
                    "src": src_n,
                    "dst": dst_n,
                    "gate0": gate0,
                    "closed_rmse": rmse_map(A, s_lag, d_lag),
                    "shuffled_rmse": rmse_map(A_sh, s_lag, d_lag[perm]),
                    "shuffle_distinguishes": bool(
                        rmse_map(A, s_lag, d_lag)
                        < 0.8 * rmse_map(A_sh, s_lag, d_lag[perm])
                    ),
                }
        out["blocks"][block_name] = rec

    primary = ("core_to_lid", "core_to_nmp")
    pass_tau1 = {}
    for pair_id in primary:
        g1 = out["blocks"]["first_half"]["tau_1"][pair_id]["gate0"]["gate0_pass"]
        g2 = out["blocks"]["second_half"]["tau_1"][pair_id]["gate0"]["gate0_pass"]
        pass_tau1[pair_id] = bool(g1 and g2)
    out["verdict"] = {
        "question": "Can X_i(t) linearly predict X_j(t+τ) on equilibrium fluctuations?",
        "primary_tau_ps": 240,
        "gate0_pass_both_halves_tau1": pass_tau1,
        "Aij_works_on_equilibrium": any(pass_tau1.values()),
        "interpretation": (
            "If Aij_works_on_equilibrium is true, DIMS incremental-C failure is "
            "largely a transition-data issue. If false, linear A_ij is the wrong "
            "coupling object even in one basin."
        ),
    }
    return out

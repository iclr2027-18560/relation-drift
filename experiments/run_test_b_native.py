"""Test B: native labels, no plant.

IBM HI-Small: Is Laundering vs amount / sender-total / velocity / new-pair
and vs unlabeled typed pair leftover on leak-free account history features.

Elliptic: illicit nodes vs degree / feature-norm and vs pair leftover on
native (PCA-compressed) features. No planted map.

This is Claim 2. A lose is an acceptable outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from elliptic_inject import (  # noqa: E402
    load_elliptic_snapshots,
    degree_ratio_types,
)
from kaggle_inject import N_TYPES, format_to_type  # noqa: E402
from run_gld_residual_v35 import it_glls_maps  # noqa: E402

FEAT_DIM = 8


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max() or len(y) < 4:
        return float("nan")
    return float(roc_auc_score(y, s))


def _z(arr: np.ndarray) -> np.ndarray:
    mu = float(np.mean(arr))
    sd = float(np.std(arr)) or 1.0
    return (arr - mu) / sd


def incremental(
    y: np.ndarray,
    base: np.ndarray,
    extra: np.ndarray,
    frac: float,
) -> Dict[str, Any]:
    n = len(y)
    k = max(1, int(round(frac * n)))
    b_idx = np.argpartition(-base, k - 1)[:k]
    e_idx = np.argpartition(-extra, k - 1)[:k]
    b_set, e_set = set(b_idx.tolist()), set(e_idx.tolist())
    both = b_set & e_set
    only_e = e_set - b_set
    only_b = b_set - e_set
    y_at = y.astype(np.int8)

    def hits(ix: set[int]) -> int:
        return int(sum(y_at[i] for i in ix))

    return {
        "alert_frac": frac,
        "n_alerts": k,
        "n_pos": int(y.sum()),
        "baseline_hits": hits(b_set),
        "residual_hits": hits(e_set),
        "both_hits": hits(both),
        "residual_only_hits": hits(only_e),
        "baseline_only_hits": hits(only_b),
        "residual_only_alerts": len(only_e),
        "precision_baseline": hits(b_set) / max(k, 1),
        "precision_residual": hits(e_set) / max(k, 1),
        "precision_residual_only": hits(only_e) / max(len(only_e), 1),
    }


def _account_features(
    sent: np.ndarray,
    recv: np.ndarray,
    n_out: np.ndarray,
    n_in: np.ndarray,
    n_cout: np.ndarray,
    n_cin: np.ndarray,
    today_out: np.ndarray,
    today_in: np.ndarray,
    today_nout: np.ndarray,
    today_nin: np.ndarray,
) -> np.ndarray:
    """History plus same-day volume so new accounts are not the zero vector."""
    return np.stack(
        [
            np.log1p(sent),
            np.log1p(recv),
            np.log1p(n_out),
            np.log1p(n_in),
            np.log1p(n_cout),
            np.log1p(n_cin),
            np.log1p(today_out),
            np.log1p(today_in),
            np.log1p(today_nout),
            np.log1p(today_nin),
        ],
        axis=1,
    ).astype(np.float32)


def run_ibm(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.ibm_data)
    if not path.is_absolute():
        path = _ROOT / path
    print(f"  IBM reading {path}")
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
    keep = set(counts[counts >= args.min_day_txs].index.tolist())
    raw = raw[raw["date"].isin(keep)]
    dates = sorted(raw["date"].unique())
    raw["type"] = raw["fmt"].map(format_to_type).astype(np.int16)
    accounts = pd.unique(pd.concat([raw["src"], raw["dst"]], ignore_index=True))
    acct_to_i = {a: i for i, a in enumerate(accounts.tolist())}
    n_acct = len(acct_to_i)
    raw["src_i"] = raw["src"].map(acct_to_i).astype(np.int32)
    raw["dst_i"] = raw["dst"].map(acct_to_i).astype(np.int32)
    print(f"  {len(dates)} days, {len(raw)} txs, {n_acct} accounts")

    sent = np.zeros(n_acct, dtype=np.float64)
    recv = np.zeros(n_acct, dtype=np.float64)
    n_out = np.zeros(n_acct, dtype=np.float64)
    n_in = np.zeros(n_acct, dtype=np.float64)
    n_cout = np.zeros(n_acct, dtype=np.float64)
    n_cin = np.zeros(n_acct, dtype=np.float64)
    seen_out: List[set] = [set() for _ in range(n_acct)]
    seen_in: List[set] = [set() for _ in range(n_acct)]
    seen_pairs: set[Tuple[int, int]] = set()
    prev_src_daily: Dict[int, List[float]] = defaultdict(list)
    prev_src_cnt: Dict[int, List[float]] = defaultdict(list)
    type_logamt: Dict[int, List[float]] = defaultdict(list)

    ys: List[np.ndarray] = []
    scores: Dict[str, List[np.ndarray]] = {
        "amount_z": [],
        "src_total_z": [],
        "velocity_z": [],
        "new_pair": [],
        "baseline_max": [],
        "baseline_no_newpair": [],
        "pair_residual": [],
        "both_new": [],
    }

    eval_start = int(args.ibm_eval_start)
    frozen_maps: Tensor | None = None
    diag_pos_new = 0
    diag_neg_new = 0
    diag_n_pos = 0
    diag_n_neg = 0
    for d_i, d in enumerate(dates):
        day = raw.loc[raw["date"] == d]
        src = day["src_i"].to_numpy()
        dst = day["dst_i"].to_numpy()
        amt = day["amount"].to_numpy()
        types = day["type"].to_numpy()
        y = day["laundering"].to_numpy().astype(np.float64)
        n = len(day)
        print(f"    {d} n={n} pos={int(y.sum())}")

        today_out = np.zeros(n_acct, dtype=np.float64)
        today_in = np.zeros(n_acct, dtype=np.float64)
        today_nout = np.zeros(n_acct, dtype=np.float64)
        today_nin = np.zeros(n_acct, dtype=np.float64)
        np.add.at(today_out, src, amt.astype(np.float64))
        np.add.at(today_in, dst, amt.astype(np.float64))
        np.add.at(today_nout, src, 1.0)
        np.add.at(today_nin, dst, 1.0)
        feat = _account_features(
            sent, recv, n_out, n_in, n_cout, n_cin,
            today_out, today_in, today_nout, today_nin,
        )
        x = torch.from_numpy(feat)
        src_t = torch.from_numpy(src.astype(np.int64))
        dst_t = torch.from_numpy(dst.astype(np.int64))
        typ_t = torch.from_numpy(types.astype(np.int64))
        pair_src, pair_dst = x[src_t], x[dst_t]
        dim = int(pair_src.shape[1])
        if frozen_maps is None:
            frozen_maps = torch.eye(dim).unsqueeze(0).repeat(N_TYPES, 1, 1)
        resid = torch.zeros(n)
        for r in range(N_TYPES):
            mask = typ_t == r
            if int(mask.sum()) == 0:
                continue
            resid[mask] = (
                pair_dst[mask] - (frozen_maps[r] @ pair_src[mask].T).T
            ).norm(dim=1)
        hist_norm = np.linalg.norm(feat[src][:, :6], axis=1) + np.linalg.norm(
            feat[dst][:, :6], axis=1
        )
        both_new = hist_norm < 1e-8

        log_amt = np.log1p(np.clip(amt, 0, None))
        amount_z = np.zeros(n, dtype=np.float64)
        for r in range(N_TYPES):
            mask = types == r
            if not np.any(mask):
                continue
            hist = np.array(type_logamt[r], dtype=np.float64)
            if hist.size < 8:
                amount_z[mask] = 0.0
            else:
                amount_z[mask] = (log_amt[mask] - hist.mean()) / (hist.std() or 1.0)

        today_tot: Dict[int, float] = defaultdict(float)
        today_cnt: Dict[int, float] = defaultdict(float)
        for i, a in zip(src, amt):
            today_tot[int(i)] += float(a)
            today_cnt[int(i)] += 1.0
        src_total_z = np.zeros(n, dtype=np.float64)
        velocity_z = np.zeros(n, dtype=np.float64)
        for k, i in enumerate(src):
            ii = int(i)
            hist_t = prev_src_daily[ii]
            hist_c = prev_src_cnt[ii]
            if len(hist_t) >= 2:
                src_total_z[k] = (today_tot[ii] - float(np.mean(hist_t))) / (
                    float(np.std(hist_t)) or 1.0
                )
                velocity_z[k] = (today_cnt[ii] - float(np.mean(hist_c))) / (
                    float(np.std(hist_c)) or 1.0
                )
        new_pair = np.array(
            [(int(s), int(t)) not in seen_pairs for s, t in zip(src, dst)],
            dtype=np.float64,
        )
        base_vol = np.maximum.reduce(
            [_z(amount_z), _z(src_total_z), _z(velocity_z)]
        )
        base = np.maximum(base_vol, new_pair)

        if d_i >= eval_start:
            ys.append(y)
            scores["amount_z"].append(amount_z)
            scores["src_total_z"].append(src_total_z)
            scores["velocity_z"].append(velocity_z)
            scores["new_pair"].append(new_pair)
            scores["baseline_max"].append(base)
            scores["baseline_no_newpair"].append(base_vol)
            scores["pair_residual"].append(resid.detach().cpu().numpy())
            scores["both_new"].append(both_new.astype(np.float64))
            diag_n_pos += int(y.sum())
            diag_n_neg += int((1.0 - y).sum())
            diag_pos_new += int(((y > 0.5) & both_new).sum())
            diag_neg_new += int(((y < 0.5) & both_new).sum())

        frozen_maps = it_glls_maps(pair_src, pair_dst, typ_t)

        for i, a, t in zip(src, amt, types):
            sent[int(i)] += float(a)
            n_out[int(i)] += 1
            type_logamt[int(t)].append(float(np.log1p(max(a, 0.0))))
        for j, a in zip(dst, amt):
            recv[int(j)] += float(a)
            n_in[int(j)] += 1
        for s, t in zip(src, dst):
            si, ti = int(s), int(t)
            if ti not in seen_out[si]:
                seen_out[si].add(ti)
                n_cout[si] += 1
            if si not in seen_in[ti]:
                seen_in[ti].add(si)
                n_cin[ti] += 1
            seen_pairs.add((si, ti))
        for i, tot in today_tot.items():
            prev_src_daily[i].append(tot)
            prev_src_cnt[i].append(today_cnt[i])

    y = np.concatenate(ys)
    packed = {k: np.concatenate(v) for k, v in scores.items()}
    aucs = {
        k: _auc(y, packed[k])
        for k in packed
        if k != "both_new"
    }
    old = packed["both_new"] < 0.5
    aucs["pair_residual_old_accounts"] = (
        _auc(y[old], packed["pair_residual"][old]) if old.sum() > 8 else float("nan")
    )
    incs = {
        f"top_{int(100 * f)}pct_vs_full_baseline": incremental(
            y, packed["baseline_max"], packed["pair_residual"], f
        )
        for f in (0.01, 0.02, 0.05)
    }
    incs.update(
        {
            f"top_{int(100 * f)}pct_vs_amount_velocity": incremental(
                y, packed["baseline_no_newpair"], packed["pair_residual"], f
            )
            for f in (0.01, 0.02, 0.05)
        }
    )
    print(f"  IBM AUC {aucs}")
    return {
        "dataset": "IBM HI-Small",
        "dates": dates,
        "eval_start_day_index": eval_start,
        "n_eval": int(len(y)),
        "n_pos": int(y.sum()),
        "auc": aucs,
        "incremental": incs,
        "diagnostics": {
            "frac_pos_both_accounts_new": diag_pos_new / max(diag_n_pos, 1),
            "frac_neg_both_accounts_new": diag_neg_new / max(diag_n_neg, 1),
            "mean_residual_pos": float(packed["pair_residual"][y > 0.5].mean()),
            "mean_residual_neg": float(packed["pair_residual"][y < 0.5].mean()),
            "v1_bug": (
                "v1 used history-only features (new accounts were the zero "
                "vector, leftover≈0) and fitted the map on the same day it scored."
            ),
            "v2_fix": (
                "Same-day in/out volume in the feature; map frozen from the "
                "previous day; leftover is out-of-sample."
            ),
        },
        "note": (
            "No plant. v2: leak-free history plus same-day volume; frozen "
            "yesterday map. IBM labels are simulator typologies."
        ),
    }


def run_elliptic(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.elliptic_dir)
    if not data_dir.is_absolute():
        data_dir = _ROOT / data_dir
    print(f"  Elliptic loading {data_dir}")
    times, xs, eidx = load_elliptic_snapshots(data_dir)
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    feats = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    nid = feats.iloc[:, 0].to_numpy().astype(np.int64)
    ts = feats.iloc[:, 1].to_numpy().astype(int)
    tx_col, cls_col = classes.columns[0], classes.columns[1]
    cls_map: Dict[int, float] = {}
    for txid, lab in zip(classes[tx_col].to_numpy(), classes[cls_col].to_numpy()):
        s = str(lab)
        if s == "1":
            cls_map[int(txid)] = 1.0
        elif s == "2":
            cls_map[int(txid)] = 0.0
    local: Dict[int, Dict[int, int]] = {}
    counters: Dict[int, int] = {t: 0 for t in times}
    node_y: Dict[int, List[float]] = {t: [] for t in times}
    for n, t in zip(nid, ts):
        t = int(t)
        local.setdefault(t, {})[int(n)] = counters[t]
        counters[t] += 1
        node_y[t].append(cls_map.get(int(n), np.nan))

    # PCA on train-window features only (no illicit labels).
    train_end = int(args.elliptic_train_end)
    train_x = torch.cat([xs[i] for i, t in enumerate(times) if t <= train_end], dim=0)
    pca = PCA(n_components=args.elliptic_pca, random_state=0)
    pca.fit(train_x.numpy())

    ys: List[np.ndarray] = []
    deg_s: List[np.ndarray] = []
    nrm_s: List[np.ndarray] = []
    res_s: List[np.ndarray] = []
    for i, t in enumerate(times):
        if t <= train_end:
            continue
        x = xs[i]
        ei = eidx[i]
        y_node = np.array(node_y[t], dtype=np.float64)
        labeled = ~np.isnan(y_node)
        if int(labeled.sum()) < 8 or ei.numel() == 0:
            continue
        xp = torch.from_numpy(pca.transform(x.numpy()).astype(np.float32))
        types = degree_ratio_types(ei, x.shape[0])
        src, dst = ei[0], ei[1]
        maps = it_glls_maps(xp[src], xp[dst], types)
        edge_res = torch.zeros(src.shape[0])
        for r in range(int(types.max().item()) + 1 if types.numel() else 0):
            mask = types == r
            if int(mask.sum()) == 0:
                continue
            edge_res[mask] = (
                xp[dst[mask]] - (maps[r] @ xp[src[mask]].T).T
            ).norm(dim=1)
        node_res = torch.zeros(x.shape[0])
        cnt = torch.zeros(x.shape[0])
        node_res.index_add_(0, dst, edge_res)
        node_res.index_add_(0, src, edge_res)
        cnt.index_add_(0, dst, torch.ones_like(edge_res))
        cnt.index_add_(0, src, torch.ones_like(edge_res))
        node_res = node_res / cnt.clamp(min=1.0)
        deg = torch.zeros(x.shape[0])
        deg.index_add_(0, src, torch.ones(src.shape[0]))
        deg.index_add_(0, dst, torch.ones(dst.shape[0]))
        nrm = x.norm(dim=1)
        ys.append(y_node[labeled])
        deg_s.append(deg.numpy()[labeled])
        nrm_s.append(nrm.numpy()[labeled])
        res_s.append(node_res.detach().numpy()[labeled])
        print(f"    t={t} labeled={int(labeled.sum())} illicit={int(y_node[labeled].sum())}")

    y = np.concatenate(ys)
    deg = np.concatenate(deg_s)
    nrm = np.concatenate(nrm_s)
    res = np.concatenate(res_s)
    # feat_norm is anti-correlated with illicit (v1 AUC 0.15); flip for a fair max.
    base = np.maximum(_z(deg), _z(-nrm))
    aucs = {
        "degree": _auc(y, deg),
        "feat_norm": _auc(y, nrm),
        "feat_norm_flipped": _auc(y, -nrm),
        "baseline_max": _auc(y, base),
        "pair_residual": _auc(y, res),
    }
    incs = {
        f"top_{int(100 * f)}pct": incremental(y, base, res, f)
        for f in (0.01, 0.02, 0.05)
    }
    print(f"  Elliptic AUC {aucs}")
    return {
        "dataset": "Elliptic illicit (native features, PCA, no plant)",
        "times_eval": [int(t) for t in times if t > train_end],
        "train_end": train_end,
        "pca_dim": args.elliptic_pca,
        "n_eval": int(len(y)),
        "n_pos": int(y.sum()),
        "auc": aucs,
        "incremental": incs,
        "note": (
            "No plant. Pair leftover on PCA of native 165-d features. "
            "Paper already states the linear pair model does not hold here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm-data", type=str, default="data/kaggle/HI-Small_Trans.csv")
    parser.add_argument("--ibm-eval-start", type=int, default=4)
    parser.add_argument("--min-day-txs", type=int, default=10000)
    parser.add_argument("--elliptic-dir", type=str, default="data/elliptic")
    parser.add_argument("--elliptic-train-end", type=int, default=25)
    parser.add_argument("--elliptic-pca", type=int, default=12)
    parser.add_argument("--skip-elliptic", action="store_true")
    parser.add_argument("--skip-ibm", action="store_true")
    parser.add_argument("--out", type=str, default="results/test_b_native.json")
    args = parser.parse_args()
    payload: Dict[str, Any] = {
        "protocol": (
            "Test B v2. No planted map. v1 bugs: history-only features made "
            "new accounts the zero vector (leftover≈0 on the new-pair cases); "
            "map fitted on the scored day. v2: same-day volume in features; "
            "yesterday map frozen; Elliptic residual on both endpoints; "
            "feat-norm flipped in the Elliptic baseline."
        )
    }
    if not args.skip_ibm:
        payload["ibm"] = run_ibm(args)
    if not args.skip_elliptic:
        payload["elliptic"] = run_elliptic(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")
    brief = {
        k: payload[k]
        for k in payload
        if k in {"ibm", "elliptic"}
    }
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()

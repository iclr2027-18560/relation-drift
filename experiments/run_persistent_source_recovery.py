"""Persistent-source recovery sensitivity (Concern 2 / Remark 2).

Fixed mixed-degree bipartite topology: each target has several incoming
edges spanning multiple types. Only source persistence (and optional
anisotropy) changes. No anomalies.

Pipelines: raw typed pairs; mixed mean (paper _aggregate_node_features);
relation-aware mean (same aggregator, one type at a time).

Source-identity split before aggregation. Train and test aggregates are
built separately so shared targets cannot mix splits.

This is estimator sensitivity, not a proof extending Theorem 2(i), and
not a fully persistent-node graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_itglls_contamination import random_so  # noqa: E402
from run_procrustes_baseline import (  # noqa: E402
    orthogonal_procrustes,
    unconstrained_lstsq,
)
from run_sheaf_benchmark import _aggregate_node_features  # noqa: E402

CONDITIONS = [
    {"name": "reference", "persistent": False, "cond": 1.0},
    {"name": "persistence", "persistent": True, "cond": 1.0},
    {"name": "anisotropy", "persistent": False, "cond": 10.0},
    {"name": "combined", "persistent": True, "cond": 10.0},
]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_cov(dim: int, cond: float) -> Tensor:
    if cond <= 1.0 + 1e-8:
        return torch.eye(dim)
    eigs = torch.linspace(1.0, float(cond), dim)
    eigs = eigs * (dim / eigs.sum())
    q = random_so(dim)
    return q @ torch.diag(eigs) @ q.T


def sample_rows(n: int, cov: Tensor) -> Tensor:
    dim = cov.shape[0]
    chol = torch.linalg.cholesky(cov + 1e-6 * torch.eye(dim))
    return torch.randn(n, dim) @ chol.T


def make_mixed_topology(
    n_src: int,
    n_tgt: int,
    n_types: int,
    in_degree: int,
    seed: int,
) -> Dict[str, Tensor]:
    """Each target gets `in_degree` incoming edges and all relation types."""
    rng = np.random.default_rng(seed)
    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []
    for v in range(n_tgt):
        types = np.arange(n_types)
        extra = rng.integers(0, n_types, size=in_degree - n_types)
        edge_types = np.concatenate([types, extra])
        rng.shuffle(edge_types)
        sources = rng.integers(0, n_src, size=in_degree)
        for u, r in zip(sources, edge_types):
            src_list.append(int(u))
            dst_list.append(n_src + v)
            type_list.append(int(r))
    return {
        "src": torch.tensor(src_list, dtype=torch.long),
        "dst": torch.tensor(dst_list, dtype=torch.long),
        "types": torch.tensor(type_list, dtype=torch.long),
        "n_nodes": n_src + n_tgt,
        "n_src": n_src,
        "n_tgt": n_tgt,
        "n_types": n_types,
    }


def topology_stats(src: Tensor, dst: Tensor, types: Tensor, n_types: int) -> Dict[str, float]:
    tgt_ids = dst.unique()
    degrees: List[int] = []
    n_rel: List[int] = []
    for v in tgt_ids.tolist():
        mask = dst == v
        degrees.append(int(mask.sum()))
        n_rel.append(int(types[mask].unique().numel()))
    return {
        "n_edges": int(src.numel()),
        "n_targets_used": int(tgt_ids.numel()),
        "mean_in_degree": float(np.mean(degrees)) if degrees else 0.0,
        "frac_in_degree_gt1": float(np.mean([d > 1 for d in degrees])) if degrees else 0.0,
        "mean_types_per_target": float(np.mean(n_rel)) if n_rel else 0.0,
        "frac_multitype_targets": float(np.mean([k > 1 for k in n_rel])) if n_rel else 0.0,
        "n_types": n_types,
    }


def generate_pairs(
    topo: Dict[str, Tensor],
    maps: Tensor,
    persistent: bool,
    cov: Tensor,
    noise: float,
) -> Tuple[Tensor, Tensor]:
    src = topo["src"]
    types = topo["types"]
    dim = maps.shape[-1]
    n_src = int(topo["n_src"])
    if persistent:
        x_nodes = sample_rows(n_src, cov)
        x = x_nodes[src]
    else:
        x = sample_rows(int(src.numel()), cov)
    y = torch.empty_like(x)
    for r in range(int(topo["n_types"])):
        mask = types == r
        y[mask] = x[mask] @ maps[r].T
    y = y + noise * torch.randn_like(y)
    return x, y


def source_split(
    src: Tensor, n_src: int, train_frac: float, seed: int
) -> Tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(n_src)
    n_train = int(round(train_frac * n_src))
    train_src = torch.tensor(order[:n_train], dtype=torch.long)
    test_src = torch.tensor(order[n_train:], dtype=torch.long)
    train_mask = torch.isin(src, train_src)
    test_mask = torch.isin(src, test_src)
    return train_mask, test_mask


def fit_typed(src_f: Tensor, dst_f: Tensor, types: Tensor, n_types: int, dim: int) -> Dict[str, Tensor]:
    ols = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    proc = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    for r in range(n_types):
        mask = types == r
        if int(mask.sum()) < 2:
            continue
        xs, ys = src_f[mask], dst_f[mask]
        ols[r] = unconstrained_lstsq(xs, ys)
        proc[r] = orthogonal_procrustes(xs, ys)
    return {"ols": ols, "procrustes": proc}


def op_error(fitted: Tensor, truth: Tensor) -> float:
    nums = (fitted - truth).square().sum(dim=(-2, -1)).sqrt()
    dens = truth.square().sum(dim=(-2, -1)).sqrt().clamp(min=1e-8)
    return float((nums / dens).mean().item())


def pred_error(fitted: Tensor, x: Tensor, y: Tensor, types: Tensor) -> float:
    pred = torch.empty_like(y)
    for r in range(int(fitted.shape[0])):
        mask = types == r
        pred[mask] = x[mask] @ fitted[r].T
    return float((pred - y).square().mean().sqrt().item())


def gram_cond(x: Tensor) -> float:
    if x.shape[0] < 2:
        return float("nan")
    gram = (x.T @ x) / max(x.shape[0], 1)
    eigs = torch.linalg.eigvalsh(gram).clamp(min=1e-12)
    return float((eigs.max() / eigs.min()).item())


def mixed_vs_raw(
    bar: Tensor, src: Tensor, dst: Tensor, x: Tensor, y: Tensor
) -> Dict[str, float]:
    src_gap = float((bar[src] - x).norm(dim=1).mean().item())
    dst_gap = float((bar[dst] - y).norm(dim=1).mean().item())
    return {
        "mean_l2_bar_src_minus_x": src_gap,
        "mean_l2_bar_dst_minus_y": dst_gap,
        "mixed_differs_from_raw": bool(dst_gap > 1e-6 or src_gap > 1e-6),
    }


def run_pipelines(
    topo: Dict[str, Tensor],
    x: Tensor,
    y: Tensor,
    maps: Tensor,
    train_mask: Tensor,
    test_mask: Tensor,
) -> Dict[str, Any]:
    src, dst, types = topo["src"], topo["dst"], topo["types"]
    n_nodes = int(topo["n_nodes"])
    n_types = int(topo["n_types"])
    dim = int(x.shape[1])
    out: Dict[str, Any] = {
        "n_train_edges": int(train_mask.sum()),
        "n_test_edges": int(test_mask.sum()),
        "n_train_sources": int(src[train_mask].unique().numel()),
        "n_test_sources": int(src[test_mask].unique().numel()),
        "train_source_gram_cond": gram_cond(x[train_mask]),
        "train_topology": topology_stats(
            src[train_mask], dst[train_mask], types[train_mask], n_types
        ),
        "test_topology": topology_stats(
            src[test_mask], dst[test_mask], types[test_mask], n_types
        ),
    }
    raw = fit_typed(x[train_mask], y[train_mask], types[train_mask], n_types, dim)
    bar_train = _aggregate_node_features(
        n_nodes, src[train_mask], dst[train_mask], x[train_mask], y[train_mask]
    )
    mixed = fit_typed(
        bar_train[src[train_mask]],
        bar_train[dst[train_mask]],
        types[train_mask],
        n_types,
        dim,
    )
    typed_ols = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    typed_proc = torch.eye(dim).unsqueeze(0).repeat(n_types, 1, 1)
    for r in range(n_types):
        mask = train_mask & (types == r)
        bar_r = _aggregate_node_features(
            n_nodes, src[mask], dst[mask], x[mask], y[mask]
        )
        typed_fit = fit_typed(
            bar_r[src[mask]], bar_r[dst[mask]], types[mask], n_types, dim
        )
        typed_ols[r] = typed_fit["ols"][r]
        typed_proc[r] = typed_fit["procrustes"][r]
    rel = {"ols": typed_ols, "procrustes": typed_proc}

    out["mixed_vs_raw_train"] = mixed_vs_raw(
        bar_train, src[train_mask], dst[train_mask], x[train_mask], y[train_mask]
    )
    xt, yt, tt = x[test_mask], y[test_mask], types[test_mask]
    for name, fitted in (
        ("raw_ols", raw["ols"]),
        ("raw_procrustes", raw["procrustes"]),
        ("mixed_ols", mixed["ols"]),
        ("mixed_procrustes", mixed["procrustes"]),
        ("rel_ols", rel["ols"]),
        ("rel_procrustes", rel["procrustes"]),
    ):
        out[name] = {
            "op_error": op_error(fitted, maps),
            "pred_error": pred_error(fitted, xt, yt, tt),
        }
    return out


def summarize(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "raw_ols",
        "raw_procrustes",
        "mixed_ols",
        "mixed_procrustes",
        "rel_ols",
        "rel_procrustes",
    ]
    out: List[Dict[str, Any]] = []
    names = [c["name"] for c in CONDITIONS]
    for name in names:
        rows = [c for c in cells if c["condition"] == name]
        if not rows:
            continue
        entry: Dict[str, Any] = {
            "condition": name,
            "n_seeds": len(rows),
            "persistent": rows[0]["persistent"],
            "cond": rows[0]["cond"],
        }
        for k in keys:
            for m in ("op_error", "pred_error"):
                vals = np.array([r["pipelines"][k][m] for r in rows], dtype=float)
                entry[f"{k}_{m}"] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                }
        gaps = np.array(
            [
                r["pipelines"]["mixed_vs_raw_train"]["mean_l2_bar_dst_minus_y"]
                for r in rows
            ],
            dtype=float,
        )
        entry["mean_mixed_dst_gap"] = {
            "mean": float(gaps.mean()),
            "std": float(gaps.std(ddof=1) if len(gaps) > 1 else 0.0),
        }
        entry["all_mixed_differs_from_raw"] = all(
            r["pipelines"]["mixed_vs_raw_train"]["mixed_differs_from_raw"] for r in rows
        )
        deg = np.array(
            [r["full_topology"]["mean_in_degree"] for r in rows], dtype=float
        )
        multi = np.array(
            [r["full_topology"]["frac_multitype_targets"] for r in rows], dtype=float
        )
        entry["mean_in_degree"] = float(deg.mean())
        entry["frac_multitype_targets"] = float(multi.mean())
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-src", type=int, default=200)
    parser.add_argument("--n-tgt", type=int, default=80)
    parser.add_argument("--n-types", type=int, default=3)
    parser.add_argument("--in-degree", type=int, default=6)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=str,
        default="results/persistent_source_recovery.json",
    )
    args = parser.parse_args()
    cells: List[Dict[str, Any]] = []
    for seed in args.seeds:
        _set_seed(seed)
        topo = make_mixed_topology(
            args.n_src, args.n_tgt, args.n_types, args.in_degree, seed
        )
        full_stats = topology_stats(
            topo["src"], topo["dst"], topo["types"], args.n_types
        )
        maps = torch.stack([random_so(args.dim) for _ in range(args.n_types)])
        train_mask, test_mask = source_split(
            topo["src"], args.n_src, args.train_frac, seed
        )
        for cond in CONDITIONS:
            cov = random_cov(args.dim, float(cond["cond"]))
            x, y = generate_pairs(
                topo, maps, bool(cond["persistent"]), cov, args.noise
            )
            pipes = run_pipelines(topo, x, y, maps, train_mask, test_mask)
            cell = {
                "seed": seed,
                "condition": cond["name"],
                "persistent": cond["persistent"],
                "cond": cond["cond"],
                "full_topology": full_stats,
                "pipelines": pipes,
            }
            cells.append(cell)
            print(
                f"seed={seed} {cond['name']:12s} "
                f"rawP={pipes['raw_procrustes']['op_error']:.3f} "
                f"mixP={pipes['mixed_procrustes']['op_error']:.3f} "
                f"relP={pipes['rel_procrustes']['op_error']:.3f} "
                f"dst_gap={pipes['mixed_vs_raw_train']['mean_l2_bar_dst_minus_y']:.3f} "
                f"multi={full_stats['frac_multitype_targets']:.2f}"
            )
    summary = summarize(cells)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    payload = {
        "protocol": (
            "Mixed-degree bipartite targets; persistent-source vs fresh per-edge "
            "sources; optional anisotropy (randomly oriented covariance, trace=d). "
            "No anomalies. Source-identity split before aggregation. Mixed uses "
            "paper _aggregate_node_features on split edges only. Relation-aware "
            "uses the same aggregator per type. Prediction error is on raw pairs "
            "from held-out sources. Not Theorem 2(i) and not fully persistent nodes."
        ),
        "n_src": args.n_src,
        "n_tgt": args.n_tgt,
        "n_types": args.n_types,
        "in_degree": args.in_degree,
        "dim": args.dim,
        "noise": args.noise,
        "train_frac": args.train_frac,
        "seeds": args.seeds,
        "cells": cells,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

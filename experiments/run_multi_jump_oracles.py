"""Privileged vs unlabeled pair oracles on multi-jump.

For each snapshot and edge type, compare three maps on the *same* raw
pairs and residual ||R x_u - x_v||:

  true_R   unpolarised R* = pinv(ρ_tgt) ρ_src          (privileged GL)
  polar_Q  nearest SO(d) factor of R* (Kabsch / polar) (privileged SO)
  it_glls  unlabeled per-type trimmed LS               (unsupervised GL)

Also reports two-map coboundary ||ρ_src x_u - ρ_tgt x_v|| (label plant)
and ||(R*)^T R* - I||_F at every t.

Kill criterion: if true_R post-AUC is not ~1.0, the scorer is wrong.
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
_FIX = _SRC / "run"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_gld_residual_v35 import it_glls_maps  # noqa: E402
from run_jump_estimator_ablation import (  # noqa: E402
    TRAIN_END,
    _score_maps,
    _score_oracle,
    compute_auc,
)
from run_procrustes_baseline import polar_so  # noqa: E402
from run_v35_benchmark import load_temporal_dataset  # noqa: E402


def _true_relative_raw(snap: Dict[str, Any]) -> Tensor:
    rho_src = snap["rho_src"].float()
    rho_tgt = snap["rho_tgt"].float()
    maps = []
    for r in range(rho_src.shape[0]):
        maps.append(torch.linalg.pinv(rho_tgt[r]) @ rho_src[r])
    return torch.stack(maps)


def _non_orthogonality(rel: Tensor) -> List[float]:
    eye = torch.eye(rel.shape[-1], dtype=rel.dtype)
    out = []
    for r in range(rel.shape[0]):
        gram = rel[r].T @ rel[r]
        out.append(float((gram - eye).norm().item()))
    return out


def _singular_values(rel: Tensor) -> List[List[float]]:
    out = []
    for r in range(rel.shape[0]):
        svals = torch.linalg.svdvals(rel[r])
        out.append([float(v) for v in svals.tolist()])
    return out


def _post_auc(per_t: Dict[int, float], ys: List[Tensor], ss: List[Tensor]) -> float:
    post_y = [ys[t] for t in range(TRAIN_END, len(ys))]
    post_s = [ss[t] for t in range(TRAIN_END, len(ss))]
    return compute_auc(torch.cat(post_y), torch.cat(post_s))


def evaluate_cell(
    drift: str,
    jump_mode: str,
    seed: int,
) -> Dict[str, Any]:
    snapshots = load_temporal_dataset(
        "clean", drift, seed=seed, jump_mode=jump_mode
    )
    methods = ("two_map", "true_R", "polar_Q", "it_glls")
    scores: Dict[str, List[Tensor]] = {k: [] for k in methods}
    labels: List[Tensor] = []
    per_t: Dict[str, Dict[str, float]] = {k: {} for k in methods}
    departure: List[Dict[str, Any]] = []
    glls_innov: List[float] = []
    prev_glls: Tensor | None = None

    for t, snap in enumerate(snapshots):
        src = snap["edge_src_feats"].float()
        dst = snap["edge_tgt_feats"].float()
        y = snap["labels"].float()
        types = snap["edge_types"].long()
        rho_src = snap["rho_src"].float()
        rho_tgt = snap["rho_tgt"].float()
        rel = _true_relative_raw(snap)
        q_star = torch.stack([polar_so(rel[r]) for r in range(rel.shape[0])])
        glls = it_glls_maps(src, dst, types)

        pack = {
            "two_map": _score_oracle(src, dst, rho_src, rho_tgt, types),
            "true_R": _score_maps(src, dst, rel, types),
            "polar_Q": _score_maps(src, dst, q_star, types),
            "it_glls": _score_maps(src, dst, glls, types),
        }
        labels.append(y)
        for name, sc in pack.items():
            scores[name].append(sc)
            per_t[name][f"t_{t}"] = compute_auc(y, sc)

        non_orth = _non_orthogonality(rel)
        polar_gap = [
            float((q_star[r] - rel[r]).norm().item()) for r in range(rel.shape[0])
        ]
        departure.append(
            {
                "t": t,
                "t_normalized": float(snap["t_normalized"].item())
                if torch.is_tensor(snap["t_normalized"])
                else float(snap["t"]),
                "non_orth_F_mean": sum(non_orth) / len(non_orth),
                "non_orth_F_per_type": non_orth,
                "polar_gap_F_mean": sum(polar_gap) / len(polar_gap),
                "singular_values_per_type": _singular_values(rel),
            }
        )
        if prev_glls is not None:
            glls_innov.append(float((glls - prev_glls).norm().item()))
        else:
            glls_innov.append(0.0)
        prev_glls = glls.detach().clone()

    result: Dict[str, Any] = {
        "drift": drift,
        "jump_mode": jump_mode,
        "seed": seed,
        "train_end": TRAIN_END,
        "post_auc": {
            name: _post_auc(per_t[name], labels, scores[name]) for name in methods
        },
        "per_timestep_auc": per_t,
        "departure": departure,
        "it_glls_innovation_F": glls_innov,
    }
    return result


def _print_cell(out: Dict[str, Any]) -> None:
    print(
        f"\n=== {out['drift']} / {out['jump_mode']} / seed {out['seed']} ==="
    )
    print("post-AUC (t>=0.5):")
    for name, auc in out["post_auc"].items():
        print(f"  {name:10s} {auc:.4f}")
    print("per-t true_R / polar_Q / it_glls / two_map:")
    pt = out["per_timestep_auc"]
    for t in range(10):
        key = f"t_{t}"
        print(
            f"  t={t}  "
            f"R={pt['true_R'][key]:.3f}  "
            f"Q={pt['polar_Q'][key]:.3f}  "
            f"GL={pt['it_glls'][key]:.3f}  "
            f"2map={pt['two_map'][key]:.3f}  "
            f"||R'R-I||_F={out['departure'][t]['non_orth_F_mean']:.3f}"
        )
    print("non-orthogonality at jump-adjacent t=3,5,7:")
    for t in (3, 5, 7):
        d = out["departure"][t]
        print(
            f"  t={t} mean={d['non_orth_F_mean']:.4f} "
            f"per_type={['%.3f' % x for x in d['non_orth_F_per_type']]} "
            f"polar_gap={d['polar_gap_F_mean']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--out",
        type=str,
        default="results/multi_jump_oracles.json",
    )
    args = parser.parse_args()

    cells = [
        ("multi_jump", "additive"),
        ("multi_jump", "group"),
        ("single_jump", "additive"),
    ]
    all_out: List[Dict[str, Any]] = []
    for drift, mode in cells:
        for seed in args.seeds:
            print(f"running {drift} {mode} seed={seed}", flush=True)
            out = evaluate_cell(drift, mode, seed)
            _print_cell(out)
            all_out.append(out)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

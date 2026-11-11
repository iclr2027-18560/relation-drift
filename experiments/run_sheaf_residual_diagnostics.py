"""Fusion diagnostics for the competing-map sheaf vs IT-GLLS ablation.

Checks (condition A, five seeds), not used to pick the original table:
  1. Residual length, indexing, and log1p rank identity vs the raw residual.
  2. Residual-only copies of the sheaf and R-GCN fusion heads (graph channels
     zeroed) recover the residual baseline.
  3. Train vs test AUROC and train-loss start/end for the live fusion models.
  4. R-GCN checkpointed on a held-out train-window snapshot (t=4), not test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
_SRC = _ROOT / "src"
_FIX = _SRC / "fix-35-final"
for _p in (_SRC, _FIX, _EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_jump_estimator_ablation import leak_free_train_config  # noqa: E402
from run_sheaf_residual_ablation import (  # noqa: E402
    CONDITIONS,
    _metrics,
    _set_seed,
    attach_pair_residuals,
    generate_competing_map_dataset,
)
from run_temporal_rgcn import (  # noqa: E402
    DEVICE,
    GRAD_CLIP,
    HIDDEN_DIM,
    LR,
    NUM_EPOCHS,
    POS_WEIGHT,
    TRAIN_TIMESTEP_RATIO,
    TemporalRGCN,
)
from run_v35_benchmark import BenchmarkConfig, train_static_sheaf  # noqa: E402
from temporal_sheaf_v35 import SheafConfig, TemporalSheafV35  # noqa: E402
from train_v35 import train_temporal_v35  # noqa: E402


def _auroc(y: Tensor, s: Tensor) -> float:
    y_np = y.detach().cpu().numpy()
    s_np = s.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return 0.5
    return float(roc_auc_score(y_np, s_np))


def residual_identity(snapshots: List[Dict[str, Tensor]], train_end: int) -> Dict[str, Any]:
    lengths_ok = True
    spearmans: List[float] = []
    raw_post: List[Tensor] = []
    log_post: List[Tensor] = []
    y_post: List[Tensor] = []
    for snap in snapshots:
        n_e = int(snap["edge_index"].shape[1])
        raw = (torch.expm1(snap["pair_residual"])).clamp(min=0.0)
        log_r = snap["pair_residual"]
        if int(log_r.shape[0]) != n_e:
            lengths_ok = False
        if int(snap["edge_src_feats"].shape[0]) != n_e:
            lengths_ok = False
        r_np = raw.detach().cpu().numpy()
        l_np = log_r.detach().cpu().numpy()
        if r_np.std() > 0 and l_np.std() > 0:
            spearmans.append(float(np.corrcoef(r_np.argsort().argsort(), l_np.argsort().argsort())[0, 1]))
        else:
            spearmans.append(1.0)
    for t, snap in enumerate(snapshots):
        if t < train_end:
            continue
        raw_post.append(torch.expm1(snap["pair_residual"]).clamp(min=0.0))
        log_post.append(snap["pair_residual"])
        y_post.append(snap["labels"].float())
    y = torch.cat(y_post)
    raw_m = _metrics(y, torch.cat(raw_post))
    log_m = _metrics(y, torch.cat(log_post))
    return {
        "edge_length_matches_residual": lengths_ok,
        "mean_rank_corr_raw_vs_log1p": float(np.mean(spearmans)),
        "post_auroc_raw": raw_m["auroc"],
        "post_auroc_log1p": log_m["auroc"],
        "log1p_preserves_auroc": abs(raw_m["auroc"] - log_m["auroc"]) < 1e-12,
    }


class ResidualOnlyHead(nn.Module):
    """Same MLP as a fusion head; graph channels are zeros."""

    def __init__(self, graph_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        if hidden == 64:
            self.head = nn.Sequential(
                nn.Linear(graph_dim + 1, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(graph_dim + 1, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, residual: Tensor) -> Tensor:
        if residual.dim() == 1:
            residual = residual.unsqueeze(-1)
        zeros = residual.new_zeros(residual.shape[0], self.graph_dim)
        return self.head(torch.cat([zeros, residual], dim=-1)).squeeze(-1)


def train_residual_only_head(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    graph_dim: int,
    hidden: int,
    dropout: float,
    num_epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    model = ResidualOnlyHead(graph_dim, hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    train_r = torch.cat([snapshots[t]["pair_residual"] for t in range(train_end)]).to(device)
    train_y = torch.cat([snapshots[t]["labels"] for t in range(train_end)]).to(device)
    losses: List[float] = []
    model.train()
    for _ in range(num_epochs):
        opt.zero_grad()
        logits = model(train_r)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        train_s = torch.sigmoid(model(train_r)).cpu()
        test_r = torch.cat(
            [snapshots[t]["pair_residual"] for t in range(train_end, len(snapshots))]
        ).to(device)
        test_y = torch.cat(
            [snapshots[t]["labels"] for t in range(train_end, len(snapshots))]
        )
        test_s = torch.sigmoid(model(test_r)).cpu()
    return {
        "train_auroc": _auroc(train_y.cpu(), train_s),
        "test_auroc": _auroc(test_y, test_s),
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
    }


def _rgcn_split_metrics(
    model: TemporalRGCN,
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
) -> Tuple[float, float]:
    model.eval()
    model.reset_state()
    train_y: List[Tensor] = []
    train_s: List[Tensor] = []
    test_y: List[Tensor] = []
    test_s: List[Tensor] = []
    with torch.no_grad():
        for t, snap in enumerate(snapshots):
            x = snap["x"].to(DEVICE)
            ei = snap["edge_index"].to(DEVICE)
            et = snap["edge_types"].to(DEVICE)
            residual = snap["pair_residual"].to(DEVICE)
            logits, _ = model.forward_timestep(x, ei, et, ei, pair_residual=residual)
            scores = torch.sigmoid(logits).cpu()
            y = snap["labels"].cpu()
            if t < train_end:
                train_y.append(y)
                train_s.append(scores)
            else:
                test_y.append(y)
                test_s.append(scores)
    return _auroc(torch.cat(train_y), torch.cat(train_s)), _auroc(
        torch.cat(test_y), torch.cat(test_s)
    )


def train_rgcn_original(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    num_epochs: int,
) -> Dict[str, Any]:
    in_dim = int(snapshots[0]["x"].shape[1])
    n_types = int(snapshots[0]["edge_types"].max().item()) + 1
    model = TemporalRGCN(in_dim, n_types, residual_dim=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    first_loss = 0.0
    last_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        model.reset_state()
        opt.zero_grad()
        losses: List[Tensor] = []
        for t in range(train_end):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            residual = snapshots[t]["pair_residual"].to(DEVICE)
            logits, _ = model.forward_timestep(x, ei, et, ei, pair_residual=residual)
            losses.append(
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        last_loss = float(loss.item())
        if epoch == 0:
            first_loss = last_loss
    train_auc, test_auc = _rgcn_split_metrics(model, snapshots, train_end)
    return {
        "train_loss_start": first_loss,
        "train_loss_end": last_loss,
        "last_epoch_train_auroc": train_auc,
        "last_epoch_test_auroc": test_auc,
        "selected_on": "last epoch, original t<0.5 train window",
    }


def train_rgcn_live(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    num_epochs: int,
    val_t: int,
) -> Dict[str, Any]:
    in_dim = int(snapshots[0]["x"].shape[1])
    n_types = int(snapshots[0]["edge_types"].max().item()) + 1
    model = TemporalRGCN(in_dim, n_types, residual_dim=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    fit_end = val_t
    best_state = None
    best_val = -1.0
    last_train_loss = 0.0
    first_train_loss = 0.0
    val_losses: List[float] = []
    train_losses: List[float] = []
    for epoch in range(num_epochs):
        model.train()
        model.reset_state()
        opt.zero_grad()
        losses: List[Tensor] = []
        for t in range(fit_end):
            x = snapshots[t]["x"].to(DEVICE)
            ei = snapshots[t]["edge_index"].to(DEVICE)
            et = snapshots[t]["edge_types"].to(DEVICE)
            y = snapshots[t]["labels"].to(DEVICE)
            residual = snapshots[t]["pair_residual"].to(DEVICE)
            logits, _ = model.forward_timestep(x, ei, et, ei, pair_residual=residual)
            losses.append(
                F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        last_train_loss = float(loss.item())
        if epoch == 0:
            first_train_loss = last_train_loss
        train_losses.append(last_train_loss)
        model.eval()
        model.reset_state()
        with torch.no_grad():
            for t in range(val_t + 1):
                x = snapshots[t]["x"].to(DEVICE)
                ei = snapshots[t]["edge_index"].to(DEVICE)
                et = snapshots[t]["edge_types"].to(DEVICE)
                residual = snapshots[t]["pair_residual"].to(DEVICE)
                logits, _ = model.forward_timestep(
                    x, ei, et, ei, pair_residual=residual
                )
                if t != val_t:
                    continue
                y = snapshots[t]["labels"].to(DEVICE)
                val_loss = float(
                    F.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=pos_weight
                    ).item()
                )
                val_auc = _auroc(y.cpu(), torch.sigmoid(logits).cpu())
        val_losses.append(val_loss)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(
                f"      [rgcn-diag] epoch {epoch+1:3d} train_loss={last_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.3f}"
            )
    last_train, last_test = _rgcn_split_metrics(model, snapshots, train_end)
    if best_state is not None:
        model.load_state_dict(best_state)
    best_train, best_test = _rgcn_split_metrics(model, snapshots, train_end)
    return {
        "train_loss_start": first_train_loss,
        "train_loss_end": last_train_loss,
        "val_loss_end": val_losses[-1],
        "last_epoch_train_auroc": last_train,
        "last_epoch_test_auroc": last_test,
        "best_val_train_auroc": best_train,
        "best_val_test_auroc": best_test,
        "best_val_auroc": best_val,
        "selected_on": "snapshot t=4 inside the original train window",
    }


def _sheaf_split_metrics(
    model: TemporalSheafV35,
    snapshots: List[Dict[str, Tensor]],
    device: str,
    train_end: int,
) -> Tuple[float, float]:
    model.eval()
    model.reset_state()
    train_y: List[Tensor] = []
    train_s: List[Tensor] = []
    test_y: List[Tensor] = []
    test_s: List[Tensor] = []
    with torch.no_grad():
        for t, snap in enumerate(snapshots):
            logits, _ = model.forward_timestep(
                snap["x"].to(device),
                snap["edge_index"].to(device),
                snap["edge_types"].to(device),
                pair_residual=snap["pair_residual"].to(device),
            )
            scores = torch.sigmoid(logits).cpu()
            y = snap["labels"].cpu()
            if t < train_end:
                train_y.append(y)
                train_s.append(scores)
            else:
                test_y.append(y)
                test_s.append(scores)
    return _auroc(torch.cat(train_y), torch.cat(train_s)), _auroc(
        torch.cat(test_y), torch.cat(test_s)
    )


def train_sheaf_live(
    snapshots: List[Dict[str, Tensor]],
    train_end: int,
    device: str,
) -> Dict[str, Any]:
    config = BenchmarkConfig(device=device)
    config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
    config.node_feat_dim = int(snapshots[0]["x"].shape[1])
    train_ts = list(range(train_end))
    static_model = train_static_sheaf(
        snapshots, config, train_ts, num_epochs=200
    )
    sheaf_config = SheafConfig(
        node_feat_dim=config.node_feat_dim,
        num_edge_types=config.num_edge_types,
        use_fast_encoder=True,
        pair_residual_dim=1,
    )
    model = TemporalSheafV35(sheaf_config).to(device)
    model = train_temporal_v35(
        model=model,
        static_model=static_model,
        dataset=[snapshots],
        config=leak_free_train_config(),
        sheaf_config=sheaf_config,
        train_timesteps=train_ts,
        jump_times=[],
    )
    train_auc, test_auc = _sheaf_split_metrics(model, snapshots, device, train_end)
    return {
        "last_epoch_train_auroc": train_auc,
        "last_epoch_test_auroc": test_auc,
        "selected_on": "last epoch; sheaf trainer has no validation checkpoint",
    }


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", type=str, default=str(DEVICE))
    parser.add_argument(
        "--out",
        type=str,
        default="results/sheaf_itglls_ablation_diagnostics.json",
    )
    args = parser.parse_args()
    condition = CONDITIONS[0]
    cells: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n=== diagnostic seed={seed} condition=A ===")
        _set_seed(seed)
        snaps = generate_competing_map_dataset(
            seed=seed,
            alpha=float(condition["alpha"]),
            theta_deg=float(condition["theta_deg"]),
            drift_total_deg=float(condition["drift_total_deg"]),
        )
        train_end = int(len(snaps) * TRAIN_TIMESTEP_RATIO)
        residual_diag = attach_pair_residuals(snaps, train_end)
        identity = residual_identity(snaps, train_end)
        print(
            f"  identity length_ok={identity['edge_length_matches_residual']} "
            f"rank_corr={identity['mean_rank_corr_raw_vs_log1p']:.6f} "
            f"raw={identity['post_auroc_raw']:.6f} "
            f"log1p={identity['post_auroc_log1p']:.6f}"
        )
        sheaf_cfg = SheafConfig()
        sheaf_graph_dim = 2 * int(sheaf_cfg.stalk_dim)
        rgcn_graph_dim = HIDDEN_DIM * 2 + int(snaps[0]["x"].shape[1]) * 2
        sheaf_head = train_residual_only_head(
            snaps,
            train_end,
            sheaf_graph_dim,
            hidden=64,
            dropout=float(sheaf_cfg.dropout),
            num_epochs=args.epochs,
            device=torch.device(args.device),
        )
        rgcn_head = train_residual_only_head(
            snaps,
            train_end,
            rgcn_graph_dim,
            hidden=HIDDEN_DIM,
            dropout=0.1,
            num_epochs=args.epochs,
            device=torch.device(args.device),
        )
        print(
            f"  sheaf-head-only test={sheaf_head['test_auroc']:.3f} "
            f"(train={sheaf_head['train_auroc']:.3f}) "
            f"rgcn-head-only test={rgcn_head['test_auroc']:.3f} "
            f"(train={rgcn_head['train_auroc']:.3f})"
        )
        rgcn_orig = train_rgcn_original(snaps, train_end, args.epochs)
        print(
            f"  rgcn+res original last train/test="
            f"{rgcn_orig['last_epoch_train_auroc']:.3f}/"
            f"{rgcn_orig['last_epoch_test_auroc']:.3f} "
            f"loss {rgcn_orig['train_loss_start']:.3f}->{rgcn_orig['train_loss_end']:.3f}"
        )
        rgcn_live = train_rgcn_live(snaps, train_end, args.epochs, val_t=train_end - 1)
        print(
            f"  rgcn+res val-select last test={rgcn_live['last_epoch_test_auroc']:.3f} "
            f"best-val test={rgcn_live['best_val_test_auroc']:.3f}"
        )
        sheaf_live = train_sheaf_live(snaps, train_end, args.device)
        print(
            f"  sheaf+res last train/test={sheaf_live['last_epoch_train_auroc']:.3f}/"
            f"{sheaf_live['last_epoch_test_auroc']:.3f}"
        )
        cells.append(
            {
                "seed": seed,
                "residual_alone": residual_diag["residual_only"]["auroc"],
                "identity": identity,
                "sheaf_fusion_head_residual_only": sheaf_head,
                "rgcn_fusion_head_residual_only": rgcn_head,
                "rgcn_original": rgcn_orig,
                "rgcn_live": rgcn_live,
                "sheaf_live": sheaf_live,
            }
        )
    summary = {
        "residual_alone": mean_std([c["residual_alone"] for c in cells]),
        "identity_rank_corr": mean_std(
            [c["identity"]["mean_rank_corr_raw_vs_log1p"] for c in cells]
        ),
        "sheaf_head_only_test": mean_std(
            [c["sheaf_fusion_head_residual_only"]["test_auroc"] for c in cells]
        ),
        "rgcn_head_only_test": mean_std(
            [c["rgcn_fusion_head_residual_only"]["test_auroc"] for c in cells]
        ),
        "rgcn_original_train": mean_std(
            [c["rgcn_original"]["last_epoch_train_auroc"] for c in cells]
        ),
        "rgcn_original_test": mean_std(
            [c["rgcn_original"]["last_epoch_test_auroc"] for c in cells]
        ),
        "rgcn_best_val_test": mean_std(
            [c["rgcn_live"]["best_val_test_auroc"] for c in cells]
        ),
        "sheaf_last_train": mean_std(
            [c["sheaf_live"]["last_epoch_train_auroc"] for c in cells]
        ),
        "sheaf_last_test": mean_std(
            [c["sheaf_live"]["last_epoch_test_auroc"] for c in cells]
        ),
        "all_lengths_ok": all(
            c["identity"]["edge_length_matches_residual"] for c in cells
        ),
        "log1p_preserves_auroc": all(
            c["identity"]["log1p_preserves_auroc"] for c in cells
        ),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    payload = {
        "condition": "A",
        "protocol_note": (
            "Diagnostics only. Original ablation table used last epoch and "
            "t<0.5 train. R-GCN best-val uses t=4 inside that window; test is "
            "still t>=0.5. Sheaf has no validation checkpoint in this trainer."
        ),
        "cells": cells,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

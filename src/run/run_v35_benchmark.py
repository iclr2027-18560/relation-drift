"""Unified fair benchmark: Static Sheaf vs Temporal Sheaf V3.5 + fast ρ.

Data pipeline: reuse the validated Week-7 sheaf generator from
``run_sheaf_benchmark.generate_temporal_dataset`` (same stack as
fix-3-5). Local synthetic rotation/noise generation is intentionally
not used — that path made post-drift features unlearnable (~0.50 AUC).

V3.5 path adds FastRotationEncoder + ChangeDetector geodesic fusion
on top of the GRU ρ_slow evolver (snapshot rescue at jumps).

Grid: 3 noise levels x 4 drift scenarios = 12 settings.
Temporal holdout: train on t < 0.5, test on t >= 0.5.
NO edge masking — evaluation is purely temporal holdout.
"""

# ============ correct test so far ============

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Ensure `src/` and `src/run/` are importable when launched as
# `python src/run/run_v35_benchmark.py`
_FIX35FINAL = Path(__file__).resolve().parent
_SRC = _FIX35FINAL.parent
for _p in (_SRC, _FIX35FINAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sheaf_benchmark import (  # noqa: E402
    DRIFT_SCENARIOS,
    INPUT_DIM,
    NOISE_CONFIGS,
    NUM_TIMESTEPS,
    generate_temporal_dataset as generate_sheaf_temporal_dataset,
)
from temporal_sheaf_v35 import (  # noqa: E402
    SheafConfig,
    TrainConfig,
    TemporalSheafV35,
)
from train_v35 import build_jump_targets, train_temporal_v35  # noqa: E402


# ─── Constants ───────────────────────────────────────────────────────────
BENCHMARK_SEED = 42
STATIC_EPOCHS = 200
TEMPORAL_PHASE1_EPOCHS = 80
TEMPORAL_PHASE2_EPOCHS = 40
TEMPORAL_PHASE3_EPOCHS = 120
TRAIN_TIMESTEP_RATIO = 0.5


@dataclass
class BenchmarkConfig:
    """Configuration for the fair benchmark experiment.

    Graph structure and drift come from ``run_sheaf_benchmark``;
    these fields mainly configure the V3.5 model and device.
    """

    num_timesteps: int = NUM_TIMESTEPS
    node_feat_dim: int = INPUT_DIM
    num_edge_types: int = 3
    seed: int = BENCHMARK_SEED
    device: str = "cpu"


def _adapt_sheaf_dataset(
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Tensor]]:
    """Map Week-7 sheaf snapshots to the V3.5 train/eval schema.

    ``run_sheaf_benchmark.generate_temporal_dataset`` returns
    ``node_features``, ``query_edges``, and string edge-type lists.
    TemporalSheafV35 expects ``x``, a single ``edge_index`` tensor,
    and integer ``edge_types`` ids.
    """
    if not dataset:
        return []

    edge_type_names: List[str] = list(dataset[0]["edge_types"])
    edge_index_by_type: Dict[str, Tensor] = dataset[0]["edge_index"]
    etype_to_id = {etype: i for i, etype in enumerate(edge_type_names)}
    edge_type_ids = torch.cat(
        [
            torch.full(
                (edge_index_by_type[etype].shape[1],),
                etype_to_id[etype],
                dtype=torch.long,
            )
            for etype in edge_type_names
        ],
        dim=0,
    )

    snapshots: List[Dict[str, Tensor]] = []
    for t_idx, snap in enumerate(dataset):
        snapshots.append(
            {
                "x": snap["node_features"],
                "edge_index": snap["query_edges"],
                "edge_types": edge_type_ids,
                "labels": snap["labels"].float(),
                "clean_labels": snap["clean_labels"].float(),
                "t": torch.tensor(t_idx),
                "t_normalized": torch.tensor(float(snap["t"])),
                "rho_src": snap["rho_src"],
                "rho_tgt": snap["rho_tgt"],
                "edge_src_feats": snap["edge_src_feats"],
                "edge_tgt_feats": snap["edge_tgt_feats"],
            }
        )
    return snapshots


def load_temporal_dataset(
    noise_name: str,
    drift_name: str,
    seed: int = BENCHMARK_SEED,
    jump_mode: str = "additive",
) -> List[Dict[str, Tensor]]:
    """Load one setting via the proven Week-7 / fix-3-5 data pipeline.

    Does not reimplement generation — delegates to
    ``run_sheaf_benchmark.generate_temporal_dataset``.
    ``jump_mode`` is ``additive`` (default) or ``group`` (SO(d) left-multiply).
    """
    drift_params = dict(DRIFT_SCENARIOS[drift_name])
    drift_params["jump_mode"] = jump_mode
    raw = generate_sheaf_temporal_dataset(
        drift_params,
        NOISE_CONFIGS[noise_name],
        num_timesteps=NUM_TIMESTEPS,
        seed=seed,
    )
    return _adapt_sheaf_dataset(raw)


def get_temporal_split(
    num_timesteps: int,
) -> Tuple[List[int], List[int]]:
    """Split timesteps into train (t<0.5) and test (t>=0.5).

    Args:
        num_timesteps: Total number of timesteps.

    Returns:
        Tuple of (train_timesteps, test_timesteps).
    """
    split_t = int(num_timesteps * TRAIN_TIMESTEP_RATIO)
    train_ts = list(range(split_t))
    test_ts = list(range(split_t, num_timesteps))
    return train_ts, test_ts


def compute_auc(scores: Tensor, labels: Tensor) -> float:
    """Compute AUC-ROC from predicted scores and true labels.

    Args:
        scores: Predicted anomaly scores [num_edges].
        labels: Ground-truth binary labels [num_edges].

    Returns:
        AUC-ROC value (0.5 if only one class present).
    """
    from sklearn.metrics import roc_auc_score

    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    if len(set(labels_np)) < 2:
        return 0.5

    return float(roc_auc_score(labels_np, scores_np))


def train_static_sheaf(
    snapshots: List[Dict[str, Tensor]],
    config: BenchmarkConfig,
    train_timesteps: List[int],
    num_epochs: Optional[int] = None,
) -> nn.Module:
    """Train a static sheaf model on training timesteps.

    The static model uses fixed restriction maps (no temporal
    evolution). Trained only on t < 0.5 timesteps.

    Args:
        snapshots: Full temporal dataset.
        config: Benchmark configuration.
        train_timesteps: Timestep indices for training.
        num_epochs: Number of training epochs.

    Returns:
        Trained static sheaf model.
    """
    if num_epochs is None:
        num_epochs = STATIC_EPOCHS
    device = config.device
    sheaf_config = SheafConfig(
        node_feat_dim=config.node_feat_dim,
        num_edge_types=config.num_edge_types,
        use_fast_encoder=False,  # true static proxy (no ρ_fast confound)
    )

    # TemporalSheafV35 with gate open, but no fast encoder
    model = TemporalSheafV35(sheaf_config).to(device)

    pos_weight = torch.tensor(sheaf_config.pos_weight, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for t in train_timesteps:
            snap = snapshots[t]
            model.reset_state()

            scores, _ = model.forward_timestep(
                snap["x"].to(device),
                snap["edge_index"].to(device),
                snap["edge_types"].to(device),
                force_gate_open=True,
            )

            loss = loss_fn(scores, snap["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0:
            avg = epoch_loss / max(num_batches, 1)
            print(f"    [Static] Epoch {epoch:3d}/" f"{num_epochs}, Loss: {avg:.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    snapshots: List[Dict[str, Tensor]],
    timesteps: List[int],
    device: str = "cpu",
    force_fusion_open: bool = False,
) -> List[float]:
    """Evaluate model on specified timesteps (temporal holdout).

    NO edge masking — all edges at each timestep are scored.
    """
    detail = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        force_fusion_open=force_fusion_open,
        timesteps=timesteps,
    )
    return detail["aucs"]


def evaluate_model_detailed(
    model: nn.Module,
    snapshots: List[Dict[str, Tensor]],
    device: str = "cpu",
    force_fusion_open: bool = False,
    timesteps: Optional[List[int]] = None,
    jump_times: Optional[Sequence[float]] = None,
    hard_handoff: bool = True,
    jump_gate_window: int = 0,
    external_rho_provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Full-sequence eval with per-timestep AUC and fusion gates.

    Args:
        model: Trained sheaf model.
        snapshots: Full temporal dataset.
        device: Compute device.
        force_fusion_open: Force g=1 (ρ_fast ceiling diagnostic).
        timesteps: If set, ``aucs`` only includes these indices (in order).
        jump_times: Normalized jump times for GT pulses / hard handoff.
        hard_handoff: Reset GRU and switch to ρ_fast on jump pulses.
        jump_gate_window: Extra steps marked as jump (0 = pulse only).
        external_rho_provider: Optional ``(t, snap, model) -> Tensor|None``
            that supplies an algebraic ρ to hard-handoff from that step.

    Returns:
        Dict with per_timestep_auc, fusion_gates, and optionally filtered aucs.
    """
    model.eval()
    model.reset_state()
    per_t_auc: Dict[str, float] = {}
    fusion_gates: Dict[str, float] = {}
    handoffs: Dict[str, float] = {}
    filtered: List[float] = []
    want = set(timesteps) if timesteps is not None else None
    jump_targets = build_jump_targets(
        len(snapshots),
        jump_times or [],
        window_after=jump_gate_window,
    )

    with torch.no_grad():
        for t in range(len(snapshots)):
            snap = snapshots[t]
            x = snap["x"].to(device)
            edge_index = snap["edge_index"].to(device)
            edge_types = snap["edge_types"].to(device)
            labels = snap["labels"].to(device)
            jump_active = jump_targets[t] > 0.5

            external_rho = None
            if external_rho_provider is not None:
                external_rho = external_rho_provider(t, snap, model)

            pair_residual = snap.get("pair_residual")
            if pair_residual is not None:
                pair_residual = pair_residual.to(device)

            if hasattr(model, "forward_timestep"):
                logits, diag = model.forward_timestep(
                    x,
                    edge_index,
                    edge_types,
                    force_fusion_open=force_fusion_open,
                    jump_active=jump_active,
                    hard_handoff=hard_handoff and jump_active,
                    external_rho=external_rho,
                    pair_residual=pair_residual,
                )
                scores = torch.sigmoid(logits)
                gate = float(diag.get("fusion_gate", 0.0))
                handoffs[f"t_{t}"] = float(diag.get("hard_handoff", 0.0))
            else:
                scores = model.score_edges(x, edge_index, edge_types)
                gate = 0.0
                handoffs[f"t_{t}"] = 0.0

            auc = compute_auc(scores, labels)
            per_t_auc[f"t_{t}"] = auc
            fusion_gates[f"t_{t}"] = gate
            if want is None or t in want:
                filtered.append(auc)

    return {
        "aucs": filtered,
        "per_timestep_auc": per_t_auc,
        "fusion_gates": fusion_gates,
        "hard_handoffs": handoffs,
    }


def train_and_eval_static(
    snapshots: List[Dict[str, Tensor]],
    config: BenchmarkConfig,
    train_ts: List[int],
    test_ts: List[int],
    static_epochs: Optional[int] = None,
) -> Dict[str, float]:
    """Train static sheaf and evaluate on temporal holdout."""
    model = train_static_sheaf(
        snapshots, config, train_ts, num_epochs=static_epochs
    )

    pre_aucs = evaluate_model(model, snapshots, train_ts, config.device)
    post_aucs = evaluate_model(model, snapshots, test_ts, config.device)

    return {
        "mean_pre": (sum(pre_aucs) / max(len(pre_aucs), 1)),
        "mean_post": (sum(post_aucs) / max(len(post_aucs), 1)),
    }


def train_and_eval_v35(
    snapshots: List[Dict[str, Tensor]],
    config: BenchmarkConfig,
    static_model: nn.Module,
    train_ts: List[int],
    test_ts: List[int],
    jump_times: Optional[Sequence[float]] = None,
    train_config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Train V3.5 temporal sheaf and evaluate on temporal holdout.

    Also reports force-g=1 (ρ_fast ceiling) post-drift metrics.
    """
    device = config.device
    sheaf_config = SheafConfig(
        node_feat_dim=config.node_feat_dim,
        num_edge_types=config.num_edge_types,
        use_fast_encoder=True,
    )
    if train_config is None:
        train_config = TrainConfig(
            epochs_phase1=TEMPORAL_PHASE1_EPOCHS,
            epochs_phase2=TEMPORAL_PHASE2_EPOCHS,
            epochs_phase3=TEMPORAL_PHASE3_EPOCHS,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=1.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=True,
        )

    model = TemporalSheafV35(sheaf_config).to(device)
    dataset = [snapshots]

    model = train_temporal_v35(
        model=model,
        static_model=static_model,
        dataset=dataset,
        config=train_config,
        sheaf_config=sheaf_config,
        train_timesteps=train_ts,
        jump_times=jump_times or [],
    )

    jw = train_config.jump_gate_window
    eval_jumps = (
        jump_times
        if train_config.use_gt_jump_at_eval
        else []
    )
    learned = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        force_fusion_open=False,
        jump_times=eval_jumps,
        hard_handoff=(
            train_config.hard_handoff_on_jump
            and train_config.use_gt_jump_at_eval
        ),
        jump_gate_window=jw,
    )
    ceiling = evaluate_model_detailed(
        model,
        snapshots,
        device=device,
        force_fusion_open=True,
        jump_times=eval_jumps,
        hard_handoff=(
            train_config.hard_handoff_on_jump
            and train_config.use_gt_jump_at_eval
        ),
        jump_gate_window=jw,
    )

    def _mean_for(ts: List[int], per_t: Dict[str, float]) -> float:
        vals = [per_t[f"t_{t}"] for t in ts if f"t_{t}" in per_t]
        return sum(vals) / max(len(vals), 1)

    jump_targets = build_jump_targets(
        len(snapshots), jump_times or [], window_after=jw
    )

    return {
        "mean_pre": _mean_for(train_ts, learned["per_timestep_auc"]),
        "mean_post": _mean_for(test_ts, learned["per_timestep_auc"]),
        "ceiling_mean_post": _mean_for(test_ts, ceiling["per_timestep_auc"]),
        "per_timestep_auc": learned["per_timestep_auc"],
        "fusion_gates": learned["fusion_gates"],
        "hard_handoffs": learned.get("hard_handoffs", {}),
        "ceiling_per_timestep_auc": ceiling["per_timestep_auc"],
        "ceiling_fusion_gates": ceiling["fusion_gates"],
        "jump_targets": {
            f"t_{i}": jump_targets[i] for i in range(len(jump_targets))
        },
    }


def print_comparison_table(
    results: List[Dict],
) -> None:
    """Print formatted comparison table of all results.

    Args:
        results: List of result dictionaries from each setting.
    """
    print("\n" + "=" * 88)
    print(
        f"{'Setting':<25} {'Static Pre':>10} "
        f"{'Static Post':>11} {'V3.5 Pre':>9} "
        f"{'V3.5 Post':>10} {'Δ Post':>8}"
    )
    print("-" * 88)

    for r in results:
        setting = f"{r['noise_config']}/{r['drift_scenario']}"
        print(
            f"{setting:<25} "
            f"{r['static_pre']:>10.4f} "
            f"{r['static_post']:>11.4f} "
            f"{r['v35_pre']:>9.4f} "
            f"{r['v35_post']:>10.4f} "
            f"{r['delta_post']:>+8.4f}"
        )

    print("=" * 88)


def run_benchmark(
    noise_filter: Optional[Sequence[str]] = None,
    drift_filter: Optional[Sequence[str]] = None,
    ceiling_only: bool = False,
    quick_epochs: bool = False,
    no_jump_labels: bool = False,
    seed: int = BENCHMARK_SEED,
) -> Dict:
    """Run the fair benchmark across settings.

    Args:
        noise_filter: Subset of noise names (default: all).
        drift_filter: Subset of drift names (default: all).
        ceiling_only: Only run clean diagnostics for ρ_fast ceiling.
        quick_epochs: Shorter curriculum for smoke / ceiling checks.

    Returns:
        Dictionary with complete results and summary statistics.
    """
    config = BenchmarkConfig(seed=seed)
    torch.manual_seed(seed)
    train_ts, test_ts = get_temporal_split(config.num_timesteps)

    noise_items = list(NOISE_CONFIGS.keys())
    drift_items = list(DRIFT_SCENARIOS.keys())
    if noise_filter:
        noise_items = [n for n in noise_items if n in set(noise_filter)]
    if drift_filter:
        drift_items = [d for d in drift_items if d in set(drift_filter)]
    if ceiling_only:
        noise_items = ["clean"]
        drift_items = ["single_jump"]

    if quick_epochs or ceiling_only:
        train_cfg = TrainConfig(
            epochs_phase1=30 if ceiling_only else 40,
            epochs_phase2=15 if ceiling_only else 20,
            epochs_phase3=0 if ceiling_only else 40,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=1.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=True,
        )
    else:
        train_cfg = TrainConfig(
            epochs_phase1=TEMPORAL_PHASE1_EPOCHS,
            epochs_phase2=TEMPORAL_PHASE2_EPOCHS,
            epochs_phase3=TEMPORAL_PHASE3_EPOCHS,
            use_routing_curriculum=True,
            fusion_gate_penalty=0.0,
            jump_gate_bce_weight=1.0,
            jump_gate_window=0,
            fast_cb_penalty=0.1,
            hard_handoff_on_jump=True,
        )

    if no_jump_labels:
        train_cfg.jump_gate_bce_weight = 0.0
        train_cfg.hard_handoff_on_jump = False
        train_cfg.use_gt_jump_at_eval = False
        train_cfg.train_window_only = True
        if train_cfg.epochs_phase2 > 0:
            train_cfg.epochs_phase3 += train_cfg.epochs_phase2
            train_cfg.epochs_phase2 = 0

    results: Dict = {"summary": {}, "results": []}
    all_result_entries: List[Dict] = []
    all_deltas: List[float] = []

    print("=" * 88)
    print(
        "WEEK 8 BENCHMARK: V3.5 + Fast Encoder "
        "(Lie slow update + routing curriculum)"
    )
    print("  Data: run_sheaf_benchmark.generate_temporal_dataset")
    print("  Static: use_fast_encoder=False")
    print("  Eval: learned gate + force g=1 ceiling")
    print(f"  Train timesteps: {train_ts}")
    print(f"  Test timesteps:  {test_ts}")
    print(
        f"  Curriculum epochs: "
        f"P1={train_cfg.epochs_phase1} "
        f"P2={train_cfg.epochs_phase2} "
        f"P3={train_cfg.epochs_phase3}"
    )
    print("=" * 88)

    for noise_name in noise_items:
        for drift_name in drift_items:
            setting = f"{noise_name}/{drift_name}"
            print(f"\n{'─' * 70}")
            print(f"  Setting: {setting}")
            print(f"{'─' * 70}")

            snapshots = load_temporal_dataset(noise_name, drift_name, seed=seed)
            config.num_edge_types = int(snapshots[0]["edge_types"].max().item()) + 1
            config.node_feat_dim = int(snapshots[0]["x"].shape[1])
            jump_times = list(DRIFT_SCENARIOS[drift_name].get("jumps", []))
            if no_jump_labels:
                jump_times = []

            static_epochs = 40 if (quick_epochs or ceiling_only) else STATIC_EPOCHS
            print("  Training static sheaf...")
            static_results = train_and_eval_static(
                snapshots,
                config,
                train_ts,
                test_ts,
                static_epochs=static_epochs,
            )
            static_model = train_static_sheaf(
                snapshots, config, train_ts, num_epochs=static_epochs
            )

            print("  Training temporal V3.5 (routing curriculum)...")
            v35_results = train_and_eval_v35(
                snapshots,
                config,
                static_model,
                train_ts,
                test_ts,
                jump_times=jump_times,
                train_config=train_cfg,
            )

            delta = v35_results["mean_post"] - static_results["mean_post"]
            all_deltas.append(delta)

            result_entry = {
                "noise_config": noise_name,
                "drift_scenario": drift_name,
                "static_pre": static_results["mean_pre"],
                "static_post": static_results["mean_post"],
                "v35_pre": v35_results["mean_pre"],
                "v35_post": v35_results["mean_post"],
                "v35_ceiling_post": v35_results["ceiling_mean_post"],
                "delta_post": delta,
                "per_timestep_auc": v35_results["per_timestep_auc"],
                "fusion_gates": v35_results["fusion_gates"],
                "hard_handoffs": v35_results.get("hard_handoffs", {}),
                "ceiling_per_timestep_auc": v35_results[
                    "ceiling_per_timestep_auc"
                ],
                "jump_targets": v35_results["jump_targets"],
            }
            all_result_entries.append(result_entry)

            print(
                f"  Static: "
                f"{static_results['mean_pre']:.4f} → "
                f"{static_results['mean_post']:.4f}"
            )
            print(
                f"  V3.5:   "
                f"{v35_results['mean_pre']:.4f} → "
                f"{v35_results['mean_post']:.4f}"
            )
            print(
                f"  Ceiling (g=1) post: "
                f"{v35_results['ceiling_mean_post']:.4f}"
            )
            print(f"  Δ post-drift: {delta:+.4f}")
            print(f"  fusion_gates: {v35_results['fusion_gates']}")
            print(
                f"  per_t_auc:    {v35_results['per_timestep_auc']}"
            )
            print(
                f"  ceiling_auc:  {v35_results['ceiling_per_timestep_auc']}"
            )

    print_comparison_table(all_result_entries)

    wins = sum(1 for d in all_deltas if d > 0.0)
    mean_delta = sum(all_deltas) / max(len(all_deltas), 1)
    mean_ceiling = sum(
        r["v35_ceiling_post"] for r in all_result_entries
    ) / max(len(all_result_entries), 1)

    results["results"] = all_result_entries
    results["summary"] = {
        "mean_delta_post": mean_delta,
        "mean_ceiling_post": mean_ceiling,
        "win_count": wins,
        "total_settings": len(all_deltas),
        "train_timesteps": train_ts,
        "test_timesteps": test_ts,
        "curriculum": {
            "phase1": train_cfg.epochs_phase1,
            "phase2": train_cfg.epochs_phase2,
            "phase3": train_cfg.epochs_phase3,
        },
        "no_jump_labels": no_jump_labels,
    }

    print(f"\n  V3.5 wins: {wins}/{len(all_deltas)}")
    print(f"  Mean Δ post-drift: {mean_delta:+.4f}")
    print(f"  Mean ceiling post (g=1): {mean_ceiling:.4f}")
    print("=" * 88)

    if ceiling_only:
        out_name = "week8_v35_ceiling.json"
    elif no_jump_labels:
        out_name = "week8_v35_unsupervised_jump.json"
    elif drift_filter == ["multi_jump"] or (
        drift_filter is not None
        and list(drift_filter) == ["multi_jump"]
    ):
        out_name = "week8_v35_multijump.json"
    else:
        out_name = "week8_v35_routing.json"
    output_path = Path("results") / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="V3.5 + fast encoder benchmark / ceiling diagnostic"
    )
    parser.add_argument(
        "--ceiling-only",
        action="store_true",
        help="Phase-1 style diagnostic on clean/single_jump only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter curriculum epochs",
    )
    parser.add_argument(
        "--noise",
        nargs="*",
        default=None,
        help="Subset of noise configs",
    )
    parser.add_argument(
        "--drift",
        nargs="*",
        default=None,
        help="Subset of drift scenarios",
    )
    parser.add_argument(
        "--no-jump-labels",
        action="store_true",
        help="No GT jump BCE, no GT hard-handoff, train on t<0.5 only",
    )
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    args = parser.parse_args()
    run_benchmark(
        noise_filter=args.noise,
        drift_filter=args.drift,
        ceiling_only=args.ceiling_only,
        quick_epochs=args.quick,
        no_jump_labels=args.no_jump_labels,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
Week 7 Deliverable: Static Sheaf Network benchmark + motivation figure.

Run:
    python src/run_sheaf_benchmark.py
    python src/run_sheaf_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_generator import (
    EdgeTypeConfig,
    HeterogeneousGraphGenerator,
    NodeTypeConfig,
)
from restriction_map_generator import RestrictionMapGenerator, RestrictionMapConfig
from temporal_evolution import TemporalEvolution, TemporalEvolutionConfig
from jump_process import JumpProcess, JumpConfig
from feature_generator import FeatureGenerator, FeatureGeneratorConfig
from noise_injector import NoiseInjector, NoiseConfig

from sheaf_diffusion import StaticSheafNetwork, StaticSheafConfig


# ======================================================================
# CONFIGURATION
# ======================================================================

DRIFT_SCENARIOS = {
    "slow_smooth": {"omega": 0.05, "jumps": [], "magnitudes": []},
    "fast_smooth": {"omega": 1.5, "jumps": [], "magnitudes": []},
    "single_jump": {"omega": 0.1, "jumps": [0.5], "magnitudes": [1.0]},
    "multi_jump": {
        "omega": 0.2,
        "jumps": [0.25, 0.5, 0.75],
        "magnitudes": [0.8, 1.2, 0.8],
    },
}

NOISE_CONFIGS = {
    "clean": NoiseConfig(
        label_noise_rate=0.0,
        feature_noise_snr=float("inf"),
        supervision_fraction=1.0,
    ),
    "moderate": NoiseConfig(
        label_noise_rate=0.1,
        feature_noise_snr=10.0,
        supervision_fraction=0.1,
    ),
    "extreme": NoiseConfig(
        label_noise_rate=0.2,
        feature_noise_snr=3.0,
        supervision_fraction=0.01,
    ),
}

NUM_TIMESTEPS = 10
NUM_EPOCHS = 200
INPUT_DIM = 16
HIDDEN_DIM = 32
STALK_DIM = 8


# ======================================================================
# DATA GENERATION
# ======================================================================

def _build_global_ids(snapshot, edge_configs: List[EdgeTypeConfig]):
    """Map heterogeneous local indices to a global node space."""
    offsets: Dict[str, int] = {}
    cursor = 0
    for ntype in snapshot.node_types:
        offsets[ntype] = cursor
        cursor += snapshot.node_features[ntype].shape[0]

    cfg_by_name = {ec.name: ec for ec in edge_configs}
    edge_index: Dict[str, torch.Tensor] = {}
    src_all, dst_all = [], []
    for etype in snapshot.edge_types:
        local = snapshot.edge_indices[etype]
        cfg = cfg_by_name[etype]
        src = local[0] + offsets[cfg.source_type]
        dst = local[1] + offsets[cfg.target_type]
        edge_index[etype] = torch.stack([src, dst])
        src_all.append(src)
        dst_all.append(dst)

    return {
        "num_nodes": cursor,
        "offsets": offsets,
        "edge_index": edge_index,
        "src_ids": torch.cat(src_all),
        "dst_ids": torch.cat(dst_all),
        "edge_types": list(snapshot.edge_types),
    }


def _aggregate_node_features(
    num_nodes: int,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    src_feats: torch.Tensor,
    tgt_feats: torch.Tensor,
) -> torch.Tensor:
    feat_dim = src_feats.shape[1]
    acc = torch.zeros(num_nodes, feat_dim)
    cnt = torch.zeros(num_nodes, 1)
    acc.index_add_(0, src_ids, src_feats)
    cnt.index_add_(0, src_ids, torch.ones(src_ids.shape[0], 1))
    acc.index_add_(0, dst_ids, tgt_feats)
    cnt.index_add_(0, dst_ids, torch.ones(dst_ids.shape[0], 1))
    return acc / cnt.clamp(min=1.0)


def generate_temporal_dataset(
    drift_params: Dict,
    noise_config: NoiseConfig,
    *,
    num_timesteps: int = NUM_TIMESTEPS,
    node_counts: Tuple[int, int, int] = (500, 300, 500),
    densities: Tuple[float, float, float] = (0.01, 0.02, 0.01),
    seed: int = 42,
) -> List[Dict]:
    """Generate temporal snapshots compatible with StaticSheafNetwork."""
    n_account, n_entity, n_tx = node_counts
    d_sends, d_owns, d_transfers = densities

    node_configs = [
        NodeTypeConfig(name="Account", count=n_account, feature_dim=INPUT_DIM),
        NodeTypeConfig(name="Entity", count=n_entity, feature_dim=INPUT_DIM),
        NodeTypeConfig(name="Transaction", count=n_tx, feature_dim=INPUT_DIM),
    ]
    edge_configs = [
        EdgeTypeConfig(
            name="sends_to",
            source_type="Account",
            target_type="Account",
            density=d_sends,
        ),
        EdgeTypeConfig(
            name="owns",
            source_type="Entity",
            target_type="Account",
            density=d_owns,
        ),
        EdgeTypeConfig(
            name="transfers",
            source_type="Account",
            target_type="Transaction",
            density=d_transfers,
        ),
    ]

    graph_gen = HeterogeneousGraphGenerator(
        node_configs=node_configs, edge_configs=edge_configs, seed=seed
    )
    snapshot = graph_gen.generate()
    global_graph = _build_global_ids(snapshot, edge_configs)

    rmap_gen = RestrictionMapGenerator(
        config=RestrictionMapConfig(block_size=4, num_blocks=4, seed=seed),
        edge_types=list(snapshot.edge_indices.keys()),
    )
    temporal_kwargs = dict(default_omega=drift_params["omega"], seed=seed)
    if drift_params.get("angular_velocities"):
        temporal_kwargs["angular_velocities"] = dict(
            drift_params["angular_velocities"]
        )
    if drift_params.get("adversarial_types"):
        temporal_kwargs["adversarial_type_drift"] = True
    temporal = TemporalEvolution(
        rmap_gen,
        config=TemporalEvolutionConfig(**temporal_kwargs),
    )
    jump = JumpProcess(
        temporal_evolution=temporal,
        config=JumpConfig(
            jump_times=list(drift_params.get("jumps", [])),
            jump_magnitudes=list(drift_params.get("magnitudes", [])),
            seed=seed,
            jump_mode=str(drift_params.get("jump_mode", "additive")),
        ),
    )
    feat_gen = FeatureGenerator(
        jump_process=jump,
        config=FeatureGeneratorConfig(anomaly_ratio=0.1, seed=seed),
    )

    num_edges_per_type = {
        etype: snapshot.edge_indices[etype].shape[1]
        for etype in snapshot.edge_indices
    }
    edge_types = global_graph["edge_types"]

    # Plant fixed anomaly membership / sources at t=0
    src0, tgt0, masks0, _ = feat_gen.generate_all_features_at_time(
        0.0, num_edges_per_type
    )
    anomalous_targets = {
        etype: tgt0[etype][masks0[etype]] for etype in edge_types
    }

    noise_injector = NoiseInjector(noise_config)
    timesteps = np.linspace(0.0, 1.0, num_timesteps)
    dataset: List[Dict] = []

    for t in timesteps:
        src_feats, tgt_feats, masks, _violations = (
            feat_gen.generate_all_features_at_time(
                float(t),
                num_edges_per_type,
                source_features_per_type=src0,
                anomaly_masks=masks0,
                anomalous_target_features=anomalous_targets,
            )
        )
        src_cat = torch.cat([src_feats[etype] for etype in edge_types], dim=0)
        tgt_cat = torch.cat([tgt_feats[etype] for etype in edge_types], dim=0)
        clean_labels = torch.cat(
            [masks[etype].float() for etype in edge_types], dim=0
        )

        node_features = _aggregate_node_features(
            global_graph["num_nodes"],
            global_graph["src_ids"],
            global_graph["dst_ids"],
            src_cat,
            tgt_cat,
        )
        noisy = noise_injector.apply_all(clean_labels, node_features)

        query_edges = torch.stack(
            [global_graph["src_ids"], global_graph["dst_ids"]]
        )
        maps_at_t = jump.get_maps_at_time(float(t))
        rho_src = torch.stack([maps_at_t[et][0] for et in edge_types])
        rho_tgt = torch.stack([maps_at_t[et][1] for et in edge_types])
        dataset.append(
            {
                "t": float(t),
                "edge_index": global_graph["edge_index"],
                "edge_types": edge_types,
                "node_features": noisy["noisy_features"],
                "num_nodes": global_graph["num_nodes"],
                "query_edges": query_edges,
                "query_edge_type": edge_types[0],
                "labels": noisy["noisy_labels"].float(),
                "clean_labels": clean_labels,
                "supervision_mask": noisy["supervision_mask"],
                "rho_src": rho_src.detach().cpu(),
                "rho_tgt": rho_tgt.detach().cpu(),
                "edge_src_feats": src_cat.detach().cpu(),
                "edge_tgt_feats": tgt_cat.detach().cpu(),
            }
        )

    return dataset


# ======================================================================
# TRAINING / EVAL
# ======================================================================

def train_static_sheaf(
    dataset: List[Dict],
    train_on: str = "early",
    num_epochs: int = NUM_EPOCHS,
) -> StaticSheafNetwork:
    """Train StaticSheafNetwork on early / late / all timesteps."""
    edge_types = dataset[0]["edge_types"]
    config = StaticSheafConfig(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        stalk_dim=STALK_DIM,
        edge_types=edge_types,
        num_layers=3,
        output_dim=1,
        dropout=0.1,
    )
    model = StaticSheafNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 1)
    )
    # AnomalyHead emits logits (no sigmoid)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))

    if train_on == "early":
        train_data = [d for d in dataset if d["t"] < 0.5]
    elif train_on == "late":
        train_data = [d for d in dataset if d["t"] >= 0.5]
    else:
        train_data = dataset
    if not train_data:
        train_data = dataset[: max(1, len(dataset) // 2)]

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch in train_data:
            optimizer.zero_grad()
            scores = model(batch)
            labels = batch["labels"]
            mask = batch.get("supervision_mask")
            if mask is not None and mask.any():
                loss = criterion(scores[mask], labels[mask])
            else:
                loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch == 1 or epoch == num_epochs or epoch % max(1, num_epochs // 5) == 0:
            print(
                f"    epoch {epoch:03d}/{num_epochs} "
                f"loss={epoch_loss / max(len(train_data), 1):.4f}"
            )

    return model


def evaluate_at_timesteps(model: StaticSheafNetwork, dataset: List[Dict]) -> List[Dict]:
    """Evaluate model at each timestep (clean labels)."""
    model.eval()
    results_per_t = []

    with torch.no_grad():
        for batch in dataset:
            scores = model(batch)
            labels = batch["clean_labels"]
            probs = torch.sigmoid(scores)

            y_true = labels.cpu().numpy()
            y_scores = probs.cpu().numpy()
            if len(np.unique(y_true)) < 2:
                auc_roc, auc_pr, f1 = 0.5, float(y_true.mean()), 0.0
            else:
                y_pred = (y_scores >= 0.5).astype(int)
                auc_roc = float(roc_auc_score(y_true, y_scores))
                auc_pr = float(average_precision_score(y_true, y_scores))
                f1 = float(f1_score(y_true, y_pred, zero_division=0))

            results_per_t.append(
                {
                    "t": batch["t"],
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "f1": f1,
                }
            )

    return results_per_t


# ======================================================================
# MAIN
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 7 static sheaf benchmark")
    p.add_argument("--quick", action="store_true", help="Small smoke run")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        num_epochs = args.epochs if args.epochs is not None else 20
        num_timesteps = 6
        node_counts = (80, 40, 80)
        densities = (0.03, 0.05, 0.03)
        noise_items = [("clean", NOISE_CONFIGS["clean"])]
        drift_items = [
            ("slow_smooth", DRIFT_SCENARIOS["slow_smooth"]),
            ("single_jump", DRIFT_SCENARIOS["single_jump"]),
        ]
        mode = "QUICK"
    else:
        num_epochs = args.epochs if args.epochs is not None else NUM_EPOCHS
        num_timesteps = NUM_TIMESTEPS
        node_counts = (500, 300, 500)
        densities = (0.01, 0.02, 0.01)
        noise_items = list(NOISE_CONFIGS.items())
        drift_items = list(DRIFT_SCENARIOS.items())
        mode = "FULL"

    print("=" * 70)
    print(f"WEEK 7: STATIC SHEAF NETWORK BENCHMARK ({mode})")
    print(f"Epochs={num_epochs}, timesteps={num_timesteps}")
    print("=" * 70)

    all_results: Dict[str, Dict] = {}

    for drift_name, drift_params in drift_items:
        for noise_name, noise_config in noise_items:
            setting = f"{noise_name}__{drift_name}"
            print(f"\n{'─' * 70}")
            print(f"Setting: {setting}")
            print(f"{'─' * 70}")

            dataset = generate_temporal_dataset(
                drift_params,
                noise_config,
                num_timesteps=num_timesteps,
                node_counts=node_counts,
                densities=densities,
            )

            print("  Training static sheaf on t < 0.5...")
            start = time.time()
            model = train_static_sheaf(
                dataset, train_on="early", num_epochs=num_epochs
            )
            train_time = time.time() - start

            results_per_t = evaluate_at_timesteps(model, dataset)
            pre_drift = [r for r in results_per_t if r["t"] < 0.5]
            post_drift = [r for r in results_per_t if r["t"] >= 0.5]
            pre_auc = float(np.mean([r["auc_roc"] for r in pre_drift]))
            post_auc = float(np.mean([r["auc_roc"] for r in post_drift]))
            degradation = pre_auc - post_auc

            print(f"  Pre-drift AUC-ROC:  {pre_auc:.4f}")
            print(f"  Post-drift AUC-ROC: {post_auc:.4f}")
            print(f"  Degradation:        {degradation:.4f}")
            print(f"  Train time:         {train_time:.1f}s")
            print(f"  {'t':>6} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8}")
            for r in results_per_t:
                print(
                    f"  {r['t']:>6.2f} {r['auc_roc']:>8.4f} "
                    f"{r['auc_pr']:>8.4f} {r['f1']:>8.4f}"
                )

            all_results[setting] = {
                "per_timestep": results_per_t,
                "pre_drift_auc": pre_auc,
                "post_drift_auc": post_auc,
                "degradation": degradation,
                "train_time": train_time,
            }

    print("\n\n" + "=" * 70)
    print("MOTIVATION FIGURE DATA")
    print("Static Sheaf: trained on t<0.5, evaluated across full timeline")
    print("=" * 70)
    print(f"{'Setting':<30} {'Pre-drift':>10} {'Post-drift':>10} {'Degradation':>12}")
    print("─" * 70)
    for setting, res in all_results.items():
        print(
            f"{setting:<30} "
            f"{res['pre_drift_auc']:>10.4f} "
            f"{res['post_drift_auc']:>10.4f} "
            f"{res['degradation']:>12.4f}"
        )

    print("\n\n" + "=" * 70)
    print("RISK CHECK: Is drift-induced degradation meaningful?")
    print("=" * 70)
    jump_settings = [k for k in all_results if "jump" in k]
    smooth_settings = [k for k in all_results if "smooth" in k]
    avg_jump = (
        float(np.mean([all_results[k]["degradation"] for k in jump_settings]))
        if jump_settings
        else float("nan")
    )
    avg_smooth = (
        float(np.mean([all_results[k]["degradation"] for k in smooth_settings]))
        if smooth_settings
        else float("nan")
    )
    print(f"  Avg degradation on SMOOTH drift: {avg_smooth:.4f}")
    print(f"  Avg degradation on JUMP drift:   {avg_jump:.4f}")
    if avg_jump == avg_jump:  # not NaN
        if avg_jump < 0.05:
            print("\n  WARNING: Jump degradation < 5%")
        elif avg_jump < 0.15:
            print("\n  MODERATE: Jump degradation 5-15%")
        else:
            print("\n  GOOD: Jump degradation > 15%")


if __name__ == "__main__":
    main()

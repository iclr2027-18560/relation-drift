"""
===============================================================================
Feature Generator: Compliant & Anomalous Node Features
===============================================================================
Module: feature_generator.py
Project: Learning Time-Varying Relational Constraints on Dynamic Heterogeneous Graphs

This module implements Step 5 of the synthetic data generation pipeline:
  - Generate node features where compliant edges have low coboundary violation
  - Anomalous edges receive independent random features → high violation
  - Controllable anomaly ratio and noise levels

Connection to paper:
  The feature generation process creates the observation space. Compliant
  edges satisfy ρ_source @ x(u) ≈ ρ_target @ x(v) (low coboundary),
  while anomalous edges violate this constraint (high coboundary).
  The model's task is to recover these constraints from observations alone.

Dependencies:
  - jump_process.py (for JumpProcess)
  - restriction_map_generator.py (for RestrictionMapGenerator, RestrictionMapConfig)
  - temporal_evolution.py (for TemporalEvolution, TemporalEvolutionConfig)
===============================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from jump_process import JumpProcess, JumpConfig
    from temporal_evolution import TemporalEvolution, TemporalEvolutionConfig
    from restriction_map_generator import (
        RestrictionMapGenerator,
        RestrictionMapConfig,
    )
except ImportError:
    JumpProcess = None
    JumpConfig = None
    TemporalEvolution = None
    TemporalEvolutionConfig = None
    RestrictionMapGenerator = None
    RestrictionMapConfig = None


# ==============================================================================
# STEP 5: FEATURE GENERATION WITH PLANTED COMPLIANCE / ANOMALY
# ==============================================================================
#
# WHY THIS EXISTS:
#   We need node features such that the coboundary operator (violation score)
#   cleanly separates normal from anomalous edges. This module controls that
#   separation by:
#     - Normal edges: target features are constructed from source features
#       via the inverse restriction map, plus small Gaussian noise.
#     - Anomalous edges: target features are drawn independently, so the
#       restriction map constraint is violated.
#
# WHY PINV FOR COMPLIANT FEATURES:
#   For an edge (u, v) to be compliant:
#       ρ_src @ x(u) ≈ ρ_tgt @ x(v)
#   Solving for x(v):
#       x(v) ≈ pinv(ρ_tgt) @ ρ_src @ x(u)
#   Adding small noise simulates real-world imperfect compliance.
#
# WHY INDEPENDENT RANDOM FOR ANOMALOUS:
#   If x(v) is drawn independently of x(u), then ρ_src @ x(u) and
#   ρ_tgt @ x(v) are uncorrelated → large expected ||δ||².
#   This creates a clear signal for anomaly detection.
# ==============================================================================


@dataclass
class FeatureGeneratorConfig:
    """
    Configuration for feature generation.

    Attributes:
        anomaly_ratio: Fraction of edges that are anomalous (default 10%).
        compliance_noise_std: Std of Gaussian noise added to compliant target
            features. Controls how "perfectly" normal edges satisfy constraints.
        anomaly_noise_std: Std of random features for anomalous edges.
            Larger values → easier separation from normal.
        seed: Random seed for reproducibility.
    """
    anomaly_ratio: float = 0.1
    compliance_noise_std: float = 0.05
    anomaly_noise_std: float = 2.0
    seed: int = 42


class FeatureGenerator:
    """
    Generates node features with planted compliance and anomaly signals.

    For each edge at time t:
      - With probability (1 - anomaly_ratio): generate compliant target features
        such that ρ_src @ x(u) ≈ ρ_tgt @ x(v)
      - With probability anomaly_ratio: generate independent random target features
        so that the coboundary violation is large.

    Pipeline position:
        Step 1 (HeterogeneousGraphGenerator) → graph topology
        Step 2 (RestrictionMapGenerator) → initial restriction maps ρ₀
        Step 3 (TemporalEvolution) → smooth evolution ρ(t)
        Step 4 (JumpProcess) → smooth + discrete jumps
        Step 5 (this class) → node features with planted anomalies
        Step 6 (LabelGenerator) → binary anomaly labels
    """

    def __init__(
        self,
        jump_process: "JumpProcess",
        config: FeatureGeneratorConfig
    ):
        """
        Args:
            jump_process: Provides restriction maps at any time t (Steps 2-4).
            config: Feature generation parameters.
        """
        self.jump_process = jump_process
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        # Structural info
        self.edge_types = jump_process.edge_types
        self.map_dim = jump_process.map_dim

    def generate_compliant_features(
        self,
        source_features: torch.Tensor,
        rho_src: torch.Tensor,
        rho_tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate target features that satisfy the restriction map constraint.

        Computes:
            x(v) = pinv(ρ_tgt) @ ρ_src @ x(u) + ε
        where ε ~ N(0, compliance_noise_std²·I)

        Args:
            source_features: (num_edges, map_dim) — features of source nodes.
            rho_src: (map_dim, map_dim) — source restriction map.
            rho_tgt: (map_dim, map_dim) — target restriction map.

        Returns:
            (num_edges, map_dim) — compliant target features.
        """
        # Project source into edge space
        projected = source_features @ rho_src.T  # (num_edges, map_dim)

        # Invert target map to find compliant target features
        rho_tgt_inv = torch.linalg.pinv(rho_tgt)
        compliant_target = projected @ rho_tgt_inv.T  # (num_edges, map_dim)

        # Add small compliance noise
        noise = self.config.compliance_noise_std * torch.randn_like(compliant_target)
        compliant_target = compliant_target + noise

        return compliant_target

    def generate_anomalous_features(
        self,
        num_edges: int,
        dim: int
    ) -> torch.Tensor:
        """
        Generate independent random features for anomalous edges.

        These features have no relationship to the source features,
        causing high coboundary violation.

        Args:
            num_edges: Number of anomalous edges.
            dim: Feature dimension.

        Returns:
            (num_edges, dim) — random target features.
        """
        return self.config.anomaly_noise_std * torch.randn(num_edges, dim)

    def generate_all_features_at_time(
        self,
        t: float,
        num_edges_per_type: Dict[str, int],
        source_features_per_type: Optional[Dict[str, torch.Tensor]] = None,
        anomaly_masks: Optional[Dict[str, torch.Tensor]] = None,
        anomalous_target_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor]
    ]:
        """
        Generate source and target features for all edge types at time t.

        For each edge type:
          1. Generate (or accept) source features
          2. Decide which edges are normal vs. anomalous
          3. Generate compliant target features for normal edges under ρ_t
          4. Generate (or reuse) random target features for anomalous edges

        Pass fixed ``anomaly_masks`` / ``source_features_per_type`` across
        timesteps so that only ρ_t (and therefore compliant targets) changes —
        this is what makes drift scenarios produce different data.

        Args:
            t: Time at which to evaluate restriction maps.
            num_edges_per_type: Dict mapping edge_type -> number of edges.
            source_features_per_type: Optional pre-generated source features.
                If None, random source features are generated.
            anomaly_masks: Optional fixed planted anomaly masks per edge type.
                If None, a new random mask is sampled.
            anomalous_target_features: Optional fixed anomalous target rows
                (one row per anomalous edge, in mask order). If None, sampled.

        Returns:
            Tuple of:
                source_features: Dict[edge_type -> (num_edges, map_dim)]
                target_features: Dict[edge_type -> (num_edges, map_dim)]
                anomaly_masks: Dict[edge_type -> (num_edges,) bool tensor]
                    True = anomalous, False = normal
                violations: Dict[edge_type -> (num_edges,) float tensor]
        """
        # Get restriction maps at time t
        maps_at_t = self.jump_process.get_maps_at_time(t)

        source_features = {}
        target_features = {}
        out_masks: Dict[str, torch.Tensor] = {}

        for edge_type in self.edge_types:
            num_edges = num_edges_per_type.get(edge_type, 0)
            if num_edges == 0:
                continue

            rho_src, rho_tgt = maps_at_t[edge_type]

            # Source features
            if source_features_per_type is not None and edge_type in source_features_per_type:
                src_feats = source_features_per_type[edge_type]
            else:
                src_feats = torch.randn(num_edges, self.map_dim)
            source_features[edge_type] = src_feats

            # Anomaly membership (fixed across time when masks are provided)
            if anomaly_masks is not None and edge_type in anomaly_masks:
                mask = anomaly_masks[edge_type].bool()
            else:
                num_anomalous = int(num_edges * self.config.anomaly_ratio)
                perm = torch.randperm(num_edges)
                mask = torch.zeros(num_edges, dtype=torch.bool)
                mask[perm[:num_anomalous]] = True
            out_masks[edge_type] = mask

            anomaly_idx = torch.where(mask)[0]
            normal_idx = torch.where(~mask)[0]
            num_anomalous = int(anomaly_idx.numel())
            num_normal = int(normal_idx.numel())

            # Generate target features under current ρ_t
            tgt_feats = torch.zeros(num_edges, self.map_dim)

            if num_normal > 0:
                compliant_tgt = self.generate_compliant_features(
                    src_feats[normal_idx], rho_src, rho_tgt
                )
                tgt_feats[normal_idx] = compliant_tgt

            if num_anomalous > 0:
                if (
                    anomalous_target_features is not None
                    and edge_type in anomalous_target_features
                ):
                    anomalous_tgt = anomalous_target_features[edge_type]
                else:
                    anomalous_tgt = self.generate_anomalous_features(
                        num_anomalous, self.map_dim
                    )
                tgt_feats[anomaly_idx] = anomalous_tgt

            target_features[edge_type] = tgt_feats

        # Compute violations for all edges under current ρ_t
        violations = self.compute_all_violations(
            source_features, target_features, t
        )

        return source_features, target_features, out_masks, violations

    def compute_all_violations(
        self,
        source_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
        t: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute coboundary violation for all edges at time t.

        violation(e) = ||ρ_src @ x(u) - ρ_tgt @ x(v)||²

        Args:
            source_features: Dict[edge_type -> (num_edges, map_dim)]
            target_features: Dict[edge_type -> (num_edges, map_dim)]
            t: Time at which to evaluate restriction maps.

        Returns:
            Dict[edge_type -> (num_edges,)] violation scores.
        """
        maps_at_t = self.jump_process.get_maps_at_time(t)
        violations = {}

        for edge_type in self.edge_types:
            if edge_type not in source_features:
                continue

            rho_src, rho_tgt = maps_at_t[edge_type]
            src_feats = source_features[edge_type]
            tgt_feats = target_features[edge_type]

            # Project into edge space
            src_projected = src_feats @ rho_src.T  # (num_edges, map_dim)
            tgt_projected = tgt_feats @ rho_tgt.T  # (num_edges, map_dim)

            # Coboundary
            coboundary = src_projected - tgt_projected

            # Violation = squared L2 norm
            violation = (coboundary ** 2).sum(dim=-1)
            violations[edge_type] = violation

        return violations


# ==============================================================================
# USAGE EXAMPLE & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Setup: Full pipeline (Steps 2-4)
    # ------------------------------------------------------------------
    if RestrictionMapGenerator is None or TemporalEvolution is None or JumpProcess is None:
        print("[ERROR] Required modules not found.")
        print("        This module requires restriction_map_generator.py,")
        print("        temporal_evolution.py, and jump_process.py for execution.")
        exit(1)

    edge_types = ["sends_to", "owns", "transfers"]

    rmap_config = RestrictionMapConfig(
        block_size=4,
        num_blocks=4,
        spectral_range=(0.5, 2.0),
        condition_number=4.0,
        orthogonal_init=True,
        seed=42
    )
    rmap_gen = RestrictionMapGenerator(config=rmap_config, edge_types=edge_types)

    evo_config = TemporalEvolutionConfig(
        angular_velocities={
            "sends_to": 0.1,
            "owns": 0.05,
            "transfers": 0.15,
        },
        default_omega=0.1,
        seed=42
    )
    temporal_evo = TemporalEvolution(rmap_generator=rmap_gen, config=evo_config)

    jump_config = JumpConfig(
        jump_times=[3.0, 7.0],
        jump_magnitudes=[0.5, 0.8],
        seed=42
    )
    jump_proc = JumpProcess(temporal_evolution=temporal_evo, config=jump_config)

    # ------------------------------------------------------------------
    # Step 5: Feature generation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 5: FEATURE GENERATION WITH PLANTED ANOMALIES")
    print("=" * 60)

    feat_config = FeatureGeneratorConfig(
        anomaly_ratio=0.1,
        compliance_noise_std=0.05,
        anomaly_noise_std=2.0,
        seed=42
    )

    feat_gen = FeatureGenerator(jump_process=jump_proc, config=feat_config)

    # Generate features at several time points
    num_edges_per_type = {
        "sends_to": 5000,
        "owns": 4000,
        "transfers": 15000,
    }

    test_times = [1.0, 5.0, 9.0]

    for t in test_times:
        print(f"\n  Time t={t:.1f}:")
        print("  " + "-" * 56)

        src_feats, tgt_feats, masks, violations = feat_gen.generate_all_features_at_time(
            t=t,
            num_edges_per_type=num_edges_per_type
        )

        for edge_type in edge_types:
            v = violations[edge_type]
            m = masks[edge_type]

            v_normal = v[~m]
            v_anomalous = v[m]

            print(f"\n    [{edge_type}] ({num_edges_per_type[edge_type]} edges, "
                  f"{m.sum().item()} anomalous)")
            print(f"      Normal edges:    mean={v_normal.mean():.4f}, "
                  f"std={v_normal.std():.4f}")
            print(f"      Anomalous edges: mean={v_anomalous.mean():.4f}, "
                  f"std={v_anomalous.std():.4f}")
            separation = v_anomalous.mean() / max(v_normal.mean().item(), 1e-8)
            print(f"      Separation ratio: {separation:.1f}x")
            status = "✓ GOOD" if separation > 5 else "⚠ WEAK"
            print(f"      → {status} separation")

    print("\n" + "=" * 60)
    print("  ✓ Feature generation validation complete.")
    print("=" * 60)
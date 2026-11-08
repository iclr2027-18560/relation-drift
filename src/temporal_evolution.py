"""
===============================================================================
Temporal Evolution of Restriction Maps via Smooth Rotations
===============================================================================
Module: temporal_evolution.py
Project: Learning Time-Varying Relational Constraints on Dynamic Heterogeneous Graphs

This module implements Step 3 of the synthetic data generation pipeline:
  - Smooth temporal evolution of restriction maps via Givens rotations
  - Controllable angular velocity (Lipschitz constant) per edge type
  - Deterministic and exactly computable evolution at any continuous time t

Connection to paper:
  Between regime changes, restriction maps evolve smoothly. The Lipschitz
  constant L = ω * ||ρ_0|| bounds how fast constraints can change, which
  determines how quickly the model must adapt.

Dependencies:
  - restriction_map_generator.py (for RestrictionMapGenerator, RestrictionMapConfig)
===============================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    from restriction_map_generator import (
        RestrictionMapGenerator,
        RestrictionMapConfig,
    )
except ImportError:
    RestrictionMapGenerator = None
    RestrictionMapConfig = None


# ==============================================================================
# STEP 3: SMOOTH TEMPORAL EVOLUTION VIA GIVENS ROTATIONS
# ==============================================================================
#
# WHY THIS EXISTS:
#   Real-world compliance rules don't change instantaneously — they drift
#   gradually as regulations evolve, market conditions shift, or adversaries
#   adapt. This module models that smooth drift.
#
# WHY GIVENS ROTATIONS:
#   A Givens rotation in the (i, j) plane is an orthogonal transformation
#   that rotates only coordinates i and j by angle θ:
#       G(i, j, θ) is identity except:
#           G[i,i] = cos(θ), G[i,j] = -sin(θ)
#           G[j,i] = sin(θ), G[j,j] = cos(θ)
#
#   Composing multiple Givens rotations with angle ωt gives:
#       R(t) = G(0,1, ω₁t) @ G(2,3, ω₂t) @ ... @ G(b-2,b-1, ω_{b/2}t)
#
#   Properties:
#   - R(t) is orthogonal for all t → preserves spectral properties of ρ
#   - R(0) = I → ρ(0) = ρ₀ (starts at planted map)
#   - Smooth in t → small Δt gives small ||ρ(t+Δt) - ρ(t)||_F
#   - Deterministic → given ρ₀ and ω, ρ(t) is exactly computable
#
# LIPSCHITZ CONSTANT:
#   ||ρ(t₁) - ρ(t₂)||_F ≤ L * |t₁ - t₂|
#   where L = ω * ||ρ₀||_F (for small angular velocities)
#   This bounds the "speed of change" and determines detection difficulty.
# ==============================================================================


@dataclass
class TemporalEvolutionConfig:
    """
    Configuration for smooth temporal evolution of restriction maps.

    Attributes:
        angular_velocities: Dict mapping edge_type -> angular velocity ω (radians/time)
            Higher ω = faster drift = harder to track.
            If not specified for an edge type, uses default_omega.
        default_omega: Default angular velocity for unspecified edge types.
        num_rotation_planes: Number of Givens rotation planes per block.
            For block_size b, max is b//2.
        seed: Random seed for rotation plane selection.
    """
    angular_velocities: Dict[str, float] = field(default_factory=dict)
    default_omega: float = 0.1
    num_rotation_planes: Optional[int] = None  # None = use max (b//2)
    seed: int = 42
    # Source/target must evolve differently; identical rotations cancel when
    # regenerating compliant features (x_v = pinv(ρ_tgt) @ ρ_src @ x_u).
    target_omega_scale: float = 0.35
    # If True, each edge type uses its own Givens planes and signed ω so
    # types can counter-rotate. Default False keeps the original shared-plane DGP.
    adversarial_type_drift: bool = False


class TemporalEvolution:
    """
    Smooth temporal evolution of restriction maps via Givens rotations.

    Given initial restriction maps ρ₀ (from RestrictionMapGenerator),
    produces smoothly evolving maps ρ(t) = R(t) @ ρ₀ where R(t) is
    composed of Givens rotations with controllable angular velocity.

    Pipeline position:
        Step 1 (HeterogeneousGraphGenerator) → graph topology
        Step 2 (RestrictionMapGenerator) → initial restriction maps ρ₀
        Step 3 (this class) → smooth evolution ρ(t) = R(t) @ ρ₀
        Step 4 (JumpProcess) → adds discrete regime changes
    """

    def __init__(
        self,
        rmap_generator: "RestrictionMapGenerator",
        config: TemporalEvolutionConfig
    ):
        """
        Args:
            rmap_generator: Provides initial restriction maps (ρ₀) and structure info.
            config: Evolution parameters (angular velocities, etc.)
        """
        self.rmap_generator = rmap_generator
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Extract structural info from the restriction map generator
        self.edge_types = rmap_generator.edge_types
        self.block_size = rmap_generator.config.block_size
        self.num_blocks = rmap_generator.config.num_blocks
        self.map_dim = rmap_generator.map_dim

        # Number of Givens rotation planes per block
        max_planes = self.block_size // 2
        self.num_planes = config.num_rotation_planes or max_planes
        assert self.num_planes <= max_planes, \
            f"num_rotation_planes ({self.num_planes}) exceeds max ({max_planes}) for block_size={self.block_size}"

        # Store initial maps as reference
        self.initial_maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type in self.edge_types:
            rho_src, rho_tgt = rmap_generator.get_maps(edge_type)
            self.initial_maps[edge_type] = (rho_src.clone(), rho_tgt.clone())

        # Precompute rotation plane indices for each block
        # Each block gets num_planes pairs of (i, j) indices for Givens rotations
        self._rotation_planes = self._select_rotation_planes()

        # Per-plane angular velocity multipliers (slight variation per plane)
        self._plane_omega_multipliers = self._generate_omega_multipliers()

        # Optional per-type planes for adversarial / counter-rotating drift
        self._type_planes: Dict[str, List[List[Tuple[int, int]]]] = {}
        if config.adversarial_type_drift:
            self._type_planes = self._adversarial_type_planes()
            self._plane_omega_multipliers = torch.ones(
                self.num_blocks, self.num_planes, dtype=torch.float32
            )

    def _select_rotation_planes(self) -> List[List[Tuple[int, int]]]:
        """
        Select which pairs of dimensions to rotate within each block.

        Returns:
            List of length num_blocks, each containing a list of (i, j) pairs
            where i, j are local indices within that block (0-indexed).
        """
        planes_per_block = []
        for _ in range(self.num_blocks):
            # Generate all possible pairs and select num_planes of them
            all_pairs = [(i, j) for i in range(self.block_size)
                         for j in range(i + 1, self.block_size)]
            selected_indices = self.rng.choice(
                len(all_pairs), size=self.num_planes, replace=False
            )
            selected_pairs = [all_pairs[idx] for idx in selected_indices]
            planes_per_block.append(selected_pairs)
        return planes_per_block

    def _adversarial_type_planes(self) -> Dict[str, List[List[Tuple[int, int]]]]:
        """Disjoint / opposing Givens planes per edge type (first two coords vs last two)."""
        b = self.block_size
        plane_a = [(0, 1)] if b >= 2 else [(0, 0)]
        plane_b = [(2, 3)] if b >= 4 else plane_a
        templates = [plane_a, plane_b, plane_a]
        out: Dict[str, List[List[Tuple[int, int]]]] = {}
        for i, etype in enumerate(self.edge_types):
            planes = templates[i % len(templates)]
            out[etype] = [list(planes) for _ in range(self.num_blocks)]
        return out

    def _generate_omega_multipliers(self) -> torch.Tensor:
        """
        Generate per-plane angular velocity multipliers.

        Each rotation plane gets a slightly different speed to create
        complex, non-trivial evolution patterns.

        Returns:
            Tensor of shape (num_blocks, num_planes) with multipliers in [0.5, 1.5]
        """
        multipliers = self.rng.uniform(0.5, 1.5, size=(self.num_blocks, self.num_planes))
        return torch.tensor(multipliers, dtype=torch.float32)

    def _build_rotation_matrix(self, t: float, edge_type: str) -> torch.Tensor:
        """
        Build the full block-diagonal rotation matrix R(t) for a given edge type.

        R(t) = diag(R_1(t), R_2(t), ..., R_k(t))
        where each R_i(t) is a product of Givens rotations:
            R_i(t) = G(p₁, q₁, ω·m₁·t) @ G(p₂, q₂, ω·m₂·t) @ ...

        Args:
            t: Continuous time value
            edge_type: Which edge type (determines angular velocity)

        Returns:
            Rotation matrix of shape (map_dim, map_dim)
        """
        omega = self.config.angular_velocities.get(
            edge_type, self.config.default_omega
        )

        R_full = torch.eye(self.map_dim)
        planes_all = self._type_planes.get(edge_type, self._rotation_planes)

        for block_idx in range(self.num_blocks):
            block_start = block_idx * self.block_size

            # Build rotation for this block by composing Givens rotations
            R_block = torch.eye(self.block_size)
            planes = planes_all[block_idx]
            for plane_idx, (i, j) in enumerate(planes):
                if i == j:
                    continue
                multiplier = 1.0
                if plane_idx < self._plane_omega_multipliers.shape[1]:
                    multiplier = self._plane_omega_multipliers[block_idx, plane_idx].item()
                angle = omega * multiplier * t

                # Givens rotation in (i, j) plane
                G = torch.eye(self.block_size)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                G[i, i] = cos_a
                G[i, j] = -sin_a
                G[j, i] = sin_a
                G[j, j] = cos_a

                R_block = R_block @ G

            # Place block rotation into full matrix
            R_full[block_start:block_start + self.block_size,
                   block_start:block_start + self.block_size] = R_block

        return R_full

    def get_maps_at_time(
        self, t: float
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get restriction maps at continuous time t.

        Computes: ρ(t) = R(t) @ ρ₀
        where R(t) is the composed Givens rotation matrix.

        Args:
            t: Continuous time value (can be any float ≥ 0)

        Returns:
            Dict mapping edge_type -> (ρ_source(t), ρ_target(t))
        """
        evolved_maps = {}
        for edge_type in self.edge_types:
            R_src = self._build_rotation_matrix(t, edge_type)
            # Asymmetric target evolution so compliant features change with t
            scale = self.config.target_omega_scale
            R_tgt = self._build_rotation_matrix(t * scale, edge_type)
            rho_src_0, rho_tgt_0 = self.initial_maps[edge_type]

            rho_src_t = R_src @ rho_src_0
            rho_tgt_t = R_tgt @ rho_tgt_0

            evolved_maps[edge_type] = (rho_src_t, rho_tgt_t)

        return evolved_maps

    def get_trajectory(
        self,
        t_start: float,
        t_end: float,
        n_steps: int
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get a sequence of restriction maps at evenly spaced time points.

        Args:
            t_start: Start time
            t_end: End time
            n_steps: Number of time steps (inclusive of start and end)

        Returns:
            List of length n_steps, each element is a dict:
                edge_type -> (ρ_source(t), ρ_target(t))
        """
        times = np.linspace(t_start, t_end, n_steps)
        trajectory = []
        for t in times:
            trajectory.append(self.get_maps_at_time(float(t)))
        return trajectory

    def compute_lipschitz_constant(self, edge_type: str) -> float:
        """
        Compute the theoretical Lipschitz constant for a given edge type.

        L = ω * ||ρ₀||_F

        This bounds the rate of change:
            ||ρ(t₁) - ρ(t₂)||_F ≤ L * |t₁ - t₂|   (approximate, for small ω)

        Args:
            edge_type: Which relationship type

        Returns:
            Lipschitz constant L
        """
        omega = self.config.angular_velocities.get(
            edge_type, self.config.default_omega
        )
        rho_src_0, rho_tgt_0 = self.initial_maps[edge_type]

        # Use the larger norm between source and target
        norm_src = torch.linalg.norm(rho_src_0, ord='fro').item()
        norm_tgt = torch.linalg.norm(rho_tgt_0, ord='fro').item()
        max_norm = max(norm_src, norm_tgt)

        return omega * max_norm

    def compute_frobenius_difference(
        self, t1: float, t2: float, edge_type: str
    ) -> Tuple[float, float]:
        """
        Compute ||ρ(t₁) - ρ(t₂)||_F for both source and target maps.

        Args:
            t1, t2: Two time points
            edge_type: Which relationship type

        Returns:
            (source_diff, target_diff) — Frobenius norm of the difference
        """
        maps_t1 = self.get_maps_at_time(t1)
        maps_t2 = self.get_maps_at_time(t2)

        src_diff = torch.linalg.norm(
            maps_t1[edge_type][0] - maps_t2[edge_type][0], ord='fro'
        ).item()
        tgt_diff = torch.linalg.norm(
            maps_t1[edge_type][1] - maps_t2[edge_type][1], ord='fro'
        ).item()

        return src_diff, tgt_diff


# ==============================================================================
# USAGE EXAMPLE & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Setup: Create restriction maps (Step 2) to evolve
    # ------------------------------------------------------------------
    if RestrictionMapGenerator is None or RestrictionMapConfig is None:
        print("[ERROR] restriction_map_generator.py not found.")
        print("        This module requires it for execution.")
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

    # ------------------------------------------------------------------
    # Step 3: Temporal evolution
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 3: TEMPORAL EVOLUTION OF RESTRICTION MAPS")
    print("=" * 60)

    evo_config = TemporalEvolutionConfig(
        angular_velocities={
            "sends_to": 0.1,
            "owns": 0.05,
            "transfers": 0.15,
        },
        default_omega=0.1,
        num_rotation_planes=None,  # Use max (b//2 = 2)
        seed=42
    )

    temporal_evo = TemporalEvolution(rmap_generator=rmap_gen, config=evo_config)

    # ------------------------------------------------------------------
    # Validation 1: Maps at t=0 should equal initial maps
    # ------------------------------------------------------------------
    print("\n  Validation 1: ρ(0) == ρ₀")
    print("  " + "-" * 56)
    maps_t0 = temporal_evo.get_maps_at_time(0.0)
    for edge_type in edge_types:
        src_diff = torch.linalg.norm(
            maps_t0[edge_type][0] - temporal_evo.initial_maps[edge_type][0]
        ).item()
        tgt_diff = torch.linalg.norm(
            maps_t0[edge_type][1] - temporal_evo.initial_maps[edge_type][1]
        ).item()
        status = "✓" if (src_diff < 1e-6 and tgt_diff < 1e-6) else "✗"
        print(f"    {status} [{edge_type}] src_diff={src_diff:.2e}, tgt_diff={tgt_diff:.2e}")

    # ------------------------------------------------------------------
    # Validation 2: Smooth evolution — consecutive Frobenius differences bounded
    # ------------------------------------------------------------------
    print("\n  Validation 2: Smooth evolution (bounded Frobenius differences)")
    print("  " + "-" * 56)

    n_steps = 20
    trajectory = temporal_evo.get_trajectory(0.0, 10.0, n_steps)
    dt = 10.0 / (n_steps - 1)

    for edge_type in edge_types:
        diffs = []
        for i in range(1, n_steps):
            src_prev = trajectory[i - 1][edge_type][0]
            src_curr = trajectory[i][edge_type][0]
            diff = torch.linalg.norm(src_curr - src_prev, ord='fro').item()
            diffs.append(diff)

        L = temporal_evo.compute_lipschitz_constant(edge_type)
        max_diff = max(diffs)
        mean_diff = np.mean(diffs)
        bound = L * dt

        print(f"    [{edge_type}] ω={evo_config.angular_velocities[edge_type]:.3f}")
        print(f"      Lipschitz L = {L:.4f}")
        print(f"      Δt = {dt:.4f}")
        print(f"      Max ||Δρ||_F = {max_diff:.4f} (bound ≈ {bound:.4f})")
        print(f"      Mean ||Δρ||_F = {mean_diff:.4f}")
        print(f"      → {'✓ BOUNDED' if max_diff < 3 * bound else '⚠ EXCEEDS BOUND'}")

    # ------------------------------------------------------------------
    # Validation 3: Rotation preserves spectral properties
    # ------------------------------------------------------------------
    print("\n  Validation 3: Spectral preservation (orthogonal rotation)")
    print("  " + "-" * 56)

    for edge_type in edge_types:
        svs_t0 = torch.linalg.svdvals(temporal_evo.initial_maps[edge_type][0])
        maps_t5 = temporal_evo.get_maps_at_time(5.0)
        svs_t5 = torch.linalg.svdvals(maps_t5[edge_type][0])

        sv_diff = torch.linalg.norm(svs_t0 - svs_t5).item()
        status = "✓" if sv_diff < 1e-4 else "⚠"
        print(f"    {status} [{edge_type}] SV difference at t=5.0: {sv_diff:.6f}")

    print("\n" + "=" * 60)
    print("  ✓ Temporal evolution validation complete.")
    print("=" * 60)
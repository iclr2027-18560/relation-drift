"""
===============================================================================
Jump Process: Discrete Regime Changes in Restriction Maps
===============================================================================
Module: jump_process.py
Project: Learning Time-Varying Relational Constraints on Dynamic Heterogeneous Graphs

This module implements Step 4 of the synthetic data generation pipeline:
  - Discrete jumps (regime changes) at specified timepoints
  - Combines smooth evolution (Step 3) with sudden shifts
  - Configurable jump magnitudes for different regime change severities

Connection to paper:
  In practice, compliance rules sometimes change abruptly (new regulation,
  fraud ring activation, policy update). The model must detect these regime
  changes as spikes in coboundary violation. Between jumps, smooth evolution
  applies (Step 3). At jump times, ρ_{t+} ≠ ρ_{t-}.

Dependencies:
  - temporal_evolution.py (for TemporalEvolution, TemporalEvolutionConfig)
  - restriction_map_generator.py (for RestrictionMapGenerator, RestrictionMapConfig)
===============================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    from temporal_evolution import TemporalEvolution, TemporalEvolutionConfig
    from restriction_map_generator import (
        RestrictionMapGenerator,
        RestrictionMapConfig,
    )
except ImportError:
    TemporalEvolution = None
    TemporalEvolutionConfig = None
    RestrictionMapGenerator = None
    RestrictionMapConfig = None


# ==============================================================================
# STEP 4: JUMP PROCESS — DISCRETE REGIME CHANGES
# ==============================================================================
#
# WHY THIS EXISTS:
#   Real-world systems experience both gradual drift AND sudden shifts:
#   - Gradual: market conditions evolve, adversaries slowly adapt
#   - Sudden: new regulation enacted, fraud ring activated, policy update
#
#   The jump process models these sudden shifts as discrete perturbations
#   applied at specific timepoints, on top of the smooth evolution.
#
# HOW IT WORKS:
#   Between jumps: ρ(t) = R(t - t_last_jump) @ ρ_after_last_jump
#   At jump time τ: ρ(τ+) = ρ(τ-) + magnitude * ΔR
#   where ΔR is a random orthogonal perturbation (preserves structure).
#
# WHY ORTHOGONAL PERTURBATION:
#   An arbitrary perturbation could destroy the block-diagonal structure
#   or push singular values to extreme ranges. By using an orthogonal
#   perturbation ΔR (constructed via skew-symmetric exponentiation),
#   we ensure the perturbed map retains similar spectral properties
#   while being meaningfully different in direction.
# ==============================================================================


@dataclass
class JumpConfig:
    """
    Configuration for the jump process.

    Attributes:
        jump_times: Sorted list of times at which regime changes occur.
        jump_magnitudes: Magnitude of each jump. If a single float,
            applies to all jumps. If a list, must match length of jump_times.
        seed: Random seed for perturbation generation.
        jump_mode: ``additive`` (default) does ρ ← ρ + mag·Δ with Δ skew
            and may leave SO(d). ``group`` left-multiplies by Cayley(A)∈SO(d).
    """
    jump_times: List[float] = field(default_factory=lambda: [3.0, 7.0])
    jump_magnitudes: List[float] = field(default_factory=lambda: [0.5, 0.8])
    seed: int = 42
    jump_mode: str = "additive"

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.jump_times) == len(self.jump_magnitudes), \
            (f"jump_times (len={len(self.jump_times)}) and jump_magnitudes "
             f"(len={len(self.jump_magnitudes)}) must have same length")
        assert all(t > 0 for t in self.jump_times), \
            "All jump times must be positive"
        assert self.jump_times == sorted(self.jump_times), \
            "jump_times must be sorted in ascending order"
        mode = str(self.jump_mode).lower()
        assert mode in {"additive", "group"}, \
            f"jump_mode must be additive or group, got {self.jump_mode}"
        self.jump_mode = mode


class JumpProcess:
    """
    Wraps a TemporalEvolution and injects discrete regime changes.

    Between jumps, maps evolve smoothly via Givens rotations (Step 3).
    At jump times, either an additive skew increment (default) or a
    group-valued left multiply is applied:
        additive: ρ(τ+) = ρ(τ-) + magnitude * ΔR   (may leave SO(d))
        group:    ρ(τ+) = Q ρ(τ-) with Q = Cayley(A) ∈ SO(d)

    The combined trajectory is:
        - [0, τ₁): smooth evolution from ρ₀
        - τ₁: jump → ρ₁ = ρ(τ₁-) + mag₁ * ΔR₁
        - [τ₁, τ₂): smooth evolution from ρ₁
        - τ₂: jump → ρ₂ = ρ(τ₂-) + mag₂ * ΔR₂
        - ...

    Pipeline position:
        Step 1 (HeterogeneousGraphGenerator) → graph topology
        Step 2 (RestrictionMapGenerator) → initial restriction maps ρ₀
        Step 3 (TemporalEvolution) → smooth evolution
        Step 4 (this class) → smooth + discrete jumps
    """

    def __init__(
        self,
        temporal_evolution: "TemporalEvolution",
        config: JumpConfig
    ):
        """
        Args:
            temporal_evolution: Provides smooth evolution between jumps.
            config: Jump times and magnitudes.
        """
        self.temporal_evolution = temporal_evolution
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Structural info from the temporal evolution
        self.edge_types = temporal_evolution.edge_types
        self.block_size = temporal_evolution.block_size
        self.num_blocks = temporal_evolution.num_blocks
        self.map_dim = temporal_evolution.map_dim

        # Precompute jump perturbations for reproducibility
        self._jump_perturbations: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        self._precompute_perturbations()

        # Cache: post-jump base maps (computed lazily)
        self._post_jump_maps: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def _generate_orthogonal_perturbation(self, magnitude: float) -> torch.Tensor:
        """
        Generate a random orthogonal perturbation matrix scaled by magnitude.

        Algorithm:
            1. Generate random skew-symmetric matrix A (A = -A^T)
            2. Compute orthogonal perturbation: ΔR = magnitude * A
               (For small magnitude, exp(A) ≈ I + A, so ρ + mag*A ≈ exp(mag*A')@ρ)
            3. Maintain block-diagonal structure

        Args:
            magnitude: Scale of the perturbation (0 = no change, 1 = large change)

        Returns:
            Block-diagonal perturbation matrix of shape (map_dim, map_dim)
        """
        perturbation = torch.zeros(self.map_dim, self.map_dim)

        for block_idx in range(self.num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            b = self.block_size

            # Generate random skew-symmetric matrix
            random_mat = torch.tensor(
                self.rng.standard_normal((b, b)), dtype=torch.float32
            )
            skew = (random_mat - random_mat.T) / 2.0

            # Scale by magnitude and normalize
            # Normalize so that ||perturbation_block||_F ≈ magnitude
            skew_norm = torch.linalg.norm(skew, ord='fro')
            if skew_norm > 1e-10:
                skew = skew * (magnitude / skew_norm)

            perturbation[start:end, start:end] = skew

        return perturbation

    def _cayley_so(self, skew: torch.Tensor) -> torch.Tensor:
        """Cayley map (I - A)(I + A)^{-1} from skew A to SO(d)."""
        eye = torch.eye(skew.shape[0], dtype=skew.dtype)
        return torch.linalg.solve(eye + skew, eye - skew)

    def _generate_group_rotation(self, magnitude: float) -> torch.Tensor:
        """Cayley rotation with the same skew generator as the additive jump."""
        skew = self._generate_orthogonal_perturbation(magnitude)
        return self._cayley_so(skew)

    def _precompute_perturbations(self):
        """
        Precompute all jump perturbations at initialization for reproducibility.

        Each jump gets a perturbation for each edge type (source and target maps).
        """
        for jump_idx, magnitude in enumerate(self.config.jump_magnitudes):
            perturbation_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for edge_type in self.edge_types:
                if self.config.jump_mode == "group":
                    delta_src = self._generate_group_rotation(magnitude)
                    delta_tgt = self._generate_group_rotation(magnitude)
                else:
                    delta_src = self._generate_orthogonal_perturbation(magnitude)
                    delta_tgt = self._generate_orthogonal_perturbation(magnitude)
                perturbation_dict[edge_type] = (delta_src, delta_tgt)
            self._jump_perturbations.append(perturbation_dict)

    def _get_regime_info(self, t: float) -> Tuple[int, float]:
        """
        Determine which regime a time t falls in and the local time within that regime.

        Returns:
            (regime_index, t_local)
            regime_index: -1 for before first jump, 0 for after first jump, etc.
                Actually: number of jumps that have occurred before time t.
            t_local: Time elapsed since the start of the current regime.
        """
        num_jumps_before = 0
        for jump_time in self.config.jump_times:
            if t >= jump_time:
                num_jumps_before += 1
            else:
                break

        if num_jumps_before == 0:
            t_local = t
        else:
            t_local = t - self.config.jump_times[num_jumps_before - 1]

        return num_jumps_before, t_local

    def _get_post_jump_base_maps(
        self, regime_index: int
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the base maps immediately after the regime_index-th jump.

        This is computed recursively:
            base_0 = initial maps (no jumps yet)
            base_i = evolve(base_{i-1}, from t_{i-1} to t_i) + perturbation_i

        Uses caching to avoid recomputation.

        Args:
            regime_index: How many jumps have occurred (0 = no jumps yet)

        Returns:
            Dict mapping edge_type -> (ρ_source, ρ_target) at regime start
        """
        if regime_index == 0:
            return self.temporal_evolution.initial_maps

        if regime_index in self._post_jump_maps:
            return self._post_jump_maps[regime_index]

        # Get base maps from previous regime
        prev_base = self._get_post_jump_base_maps(regime_index - 1)

        # Compute time duration of previous regime
        if regime_index == 1:
            t_prev_start = 0.0
        else:
            t_prev_start = self.config.jump_times[regime_index - 2]
        t_jump = self.config.jump_times[regime_index - 1]
        duration = t_jump - t_prev_start

        # Evolve previous base maps for the duration of that regime
        # (asymmetric source/target rotations — see TemporalEvolutionConfig)
        evolved_maps = {}
        tgt_scale = self.temporal_evolution.config.target_omega_scale
        for edge_type in self.edge_types:
            R_src = self.temporal_evolution._build_rotation_matrix(duration, edge_type)
            R_tgt = self.temporal_evolution._build_rotation_matrix(
                duration * tgt_scale, edge_type
            )
            rho_src_prev, rho_tgt_prev = prev_base[edge_type]
            evolved_src = R_src @ rho_src_prev
            evolved_tgt = R_tgt @ rho_tgt_prev
            evolved_maps[edge_type] = (evolved_src, evolved_tgt)

        # Apply jump perturbation
        jump_idx = regime_index - 1
        perturbation = self._jump_perturbations[jump_idx]

        post_jump_maps = {}
        for edge_type in self.edge_types:
            evolved_src, evolved_tgt = evolved_maps[edge_type]
            delta_src, delta_tgt = perturbation[edge_type]
            if self.config.jump_mode == "group":
                post_jump_maps[edge_type] = (
                    delta_src @ evolved_src,
                    delta_tgt @ evolved_tgt,
                )
            else:
                post_jump_maps[edge_type] = (
                    evolved_src + delta_src,
                    evolved_tgt + delta_tgt,
                )

        # Cache result
        self._post_jump_maps[regime_index] = post_jump_maps
        return post_jump_maps

    def get_maps_at_time(
        self, t: float
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get restriction maps at time t, combining smooth evolution + jumps.

        Algorithm:
            1. Determine which regime t is in (how many jumps have occurred)
            2. Get the base maps for that regime (post-jump state)
            3. Apply smooth rotation for the local time within the regime

        Args:
            t: Continuous time value (≥ 0)

        Returns:
            Dict mapping edge_type -> (ρ_source(t), ρ_target(t))
        """
        assert t >= 0, f"Time must be non-negative, got {t}"

        regime_index, t_local = self._get_regime_info(t)
        base_maps = self._get_post_jump_base_maps(regime_index)

        # Apply asymmetric smooth evolution for t_local from regime base
        result = {}
        tgt_scale = self.temporal_evolution.config.target_omega_scale
        for edge_type in self.edge_types:
            R_src = self.temporal_evolution._build_rotation_matrix(t_local, edge_type)
            R_tgt = self.temporal_evolution._build_rotation_matrix(
                t_local * tgt_scale, edge_type
            )
            rho_src_base, rho_tgt_base = base_maps[edge_type]
            result[edge_type] = (R_src @ rho_src_base, R_tgt @ rho_tgt_base)

        return result

    def get_full_trajectory(
        self,
        t_start: float,
        t_end: float,
        n_steps: int
    ) -> Tuple[np.ndarray, List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Get the full trajectory of maps with jumps visible.

        Args:
            t_start: Start time
            t_end: End time
            n_steps: Number of time steps

        Returns:
            (times, trajectory) where:
                times: numpy array of time points
                trajectory: list of maps at each time point
        """
        times = np.linspace(t_start, t_end, n_steps)
        trajectory = []
        for t in times:
            trajectory.append(self.get_maps_at_time(float(t)))
        return times, trajectory

    def get_jump_times(self) -> List[float]:
        """Return the configured jump times."""
        return list(self.config.jump_times)

    def get_jump_magnitudes(self) -> List[float]:
        """Return the configured jump magnitudes."""
        return list(self.config.jump_magnitudes)

    def is_jump_time(self, t: float, tolerance: float = 1e-6) -> bool:
        """Check if a given time is (approximately) a jump time."""
        for jump_t in self.config.jump_times:
            if abs(t - jump_t) < tolerance:
                return True
        return False


# ==============================================================================
# USAGE EXAMPLE & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Setup: Create restriction maps (Step 2) and evolution (Step 3)
    # ------------------------------------------------------------------
    if RestrictionMapGenerator is None or TemporalEvolution is None:
        print("[ERROR] Required modules not found.")
        print("        This module requires restriction_map_generator.py")
        print("        and temporal_evolution.py for execution.")
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

    # ------------------------------------------------------------------
    # Step 4: Jump process
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 4: JUMP PROCESS — DISCRETE REGIME CHANGES")
    print("=" * 60)

    jump_config = JumpConfig(
        jump_times=[3.0, 7.0],
        jump_magnitudes=[0.5, 0.8],
        seed=42
    )

    jump_proc = JumpProcess(temporal_evolution=temporal_evo, config=jump_config)

    # ------------------------------------------------------------------
    # Validation 1: Smooth regions have small differences
    # ------------------------------------------------------------------
    print("\n  Validation 1: Smooth regions (small Frobenius differences)")
    print("  " + "-" * 56)

    # Sample within first regime (t=0 to t=3, no jumps)
    n_samples = 10
    times_smooth = np.linspace(0.5, 2.5, n_samples)

    for edge_type in edge_types:
        diffs = []
        for i in range(1, n_samples):
            maps_prev = jump_proc.get_maps_at_time(float(times_smooth[i - 1]))
            maps_curr = jump_proc.get_maps_at_time(float(times_smooth[i]))
            diff = torch.linalg.norm(
                maps_curr[edge_type][0] - maps_prev[edge_type][0], ord='fro'
            ).item()
            diffs.append(diff)

        dt = times_smooth[1] - times_smooth[0]
        print(f"    [{edge_type}] regime 1 (t∈[0.5, 2.5], Δt={dt:.3f})")
        print(f"      Max ||Δρ||_F = {max(diffs):.4f}")
        print(f"      Mean ||Δρ||_F = {np.mean(diffs):.4f}")

    # ------------------------------------------------------------------
    # Validation 2: Jump points have large differences
    # ------------------------------------------------------------------
    print("\n  Validation 2: Jump points (large Frobenius differences)")
    print("  " + "-" * 56)

    epsilon = 0.001  # Small offset to measure just before/after jump

    for jump_idx, jump_time in enumerate(jump_config.jump_times):
        print(f"\n    Jump at t={jump_time} (magnitude={jump_config.jump_magnitudes[jump_idx]})")
        maps_before = jump_proc.get_maps_at_time(jump_time - epsilon)
        maps_after = jump_proc.get_maps_at_time(jump_time + epsilon)

        for edge_type in edge_types:
            diff = torch.linalg.norm(
                maps_after[edge_type][0] - maps_before[edge_type][0], ord='fro'
            ).item()
            print(f"      [{edge_type}] ||ρ(τ+ε) - ρ(τ-ε)||_F = {diff:.4f}")

    # ------------------------------------------------------------------
    # Validation 3: Smooth-vs-jump comparison
    # ------------------------------------------------------------------
    print("\n  Validation 3: Smooth vs. Jump magnitude comparison")
    print("  " + "-" * 56)

    # Compare: smooth difference over same dt vs. jump difference
    dt_compare = 2 * epsilon  # Same time difference as jump measurement
    for edge_type in edge_types:
        # Smooth difference (within regime, same dt)
        smooth_diff = torch.linalg.norm(
            jump_proc.get_maps_at_time(1.0 + dt_compare)[edge_type][0]
            - jump_proc.get_maps_at_time(1.0)[edge_type][0],
            ord='fro'
        ).item()

        # Jump difference
        jump_diff = torch.linalg.norm(
            jump_proc.get_maps_at_time(3.0 + epsilon)[edge_type][0]
            - jump_proc.get_maps_at_time(3.0 - epsilon)[edge_type][0],
            ord='fro'
        ).item()

        ratio = jump_diff / max(smooth_diff, 1e-10)
        status = "✓" if ratio > 10 else "⚠"
        print(f"    {status} [{edge_type}] smooth={smooth_diff:.6f}, "
              f"jump={jump_diff:.4f}, ratio={ratio:.1f}x")

    # ------------------------------------------------------------------
    # Validation 4: Full trajectory visualization data
    # ------------------------------------------------------------------
    print("\n  Validation 4: Full trajectory overview")
    print("  " + "-" * 56)

    times, trajectory = jump_proc.get_full_trajectory(0.0, 10.0, 50)

    for edge_type in edge_types:
        norms = []
        for maps in trajectory:
            norm = torch.linalg.norm(maps[edge_type][0], ord='fro').item()
            norms.append(norm)

        diffs = []
        for i in range(1, len(trajectory)):
            diff = torch.linalg.norm(
                trajectory[i][edge_type][0] - trajectory[i - 1][edge_type][0],
                ord='fro'
            ).item()
            diffs.append(diff)

        # Find the largest differences (should be at jump points)
        top_diff_indices = np.argsort(diffs)[-3:][::-1]
        print(f"\n    [{edge_type}]")
        print(f"      Frobenius norm range: [{min(norms):.3f}, {max(norms):.3f}]")
        print(f"      Top 3 largest Δρ steps:")
        for idx in top_diff_indices:
            t_val = (times[idx] + times[idx + 1]) / 2
            is_jump = jump_proc.is_jump_time(t_val, tolerance=0.3)
            marker = " ← JUMP" if is_jump else ""
            print(f"        t≈{t_val:.2f}: ||Δρ||_F = {diffs[idx]:.4f}{marker}")

    print("\n" + "=" * 60)
    print("  ✓ Jump process validation complete.")
    print("=" * 60)


# ##############################################################################
# ##############################################################################
# ##                                                                          ##
# ##  FILE 5: feature_generator.py                                            ##
# ##                                                                          ##
# ##############################################################################
# ##############################################################################

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
        source_features_per_type: Optional[Dict[str, torch.Tensor]] = None
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
          3. Generate compliant target features for normal edges
          4. Generate random target features for anomalous edges

        Args:
            t: Time at which to evaluate restriction maps.
            num_edges_per_type: Dict mapping edge_type -> number of edges.
            source_features_per_type: Optional pre-generated source features.
                If None, random source features are generated.

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
        anomaly_masks = {}

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

            # Decide anomaly membership
            num_anomalous = int(num_edges * self.config.anomaly_ratio)
            num_normal = num_edges - num_anomalous

            # Shuffle indices to randomly assign anomaly status
            perm = torch.randperm(num_edges)
            anomaly_idx = perm[:num_anomalous]
            normal_idx = perm[num_anomalous:]

            # Create anomaly mask
            mask = torch.zeros(num_edges, dtype=torch.bool)
            mask[anomaly_idx] = True
            anomaly_masks[edge_type] = mask

            # Generate target features
            tgt_feats = torch.zeros(num_edges, self.map_dim)

            # Compliant targets for normal edges
            if num_normal > 0:
                compliant_tgt = self.generate_compliant_features(
                    src_feats[normal_idx], rho_src, rho_tgt
                )
                tgt_feats[normal_idx] = compliant_tgt

            # Anomalous targets for anomalous edges
            if num_anomalous > 0:
                anomalous_tgt = self.generate_anomalous_features(
                    num_anomalous, self.map_dim
                )
                tgt_feats[anomaly_idx] = anomalous_tgt

            target_features[edge_type] = tgt_feats

        # Compute violations for all edges
        violations = self.compute_all_violations(
            source_features, target_features, t
        )

        return source_features, target_features, anomaly_masks, violations

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
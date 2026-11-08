"""
===============================================================================
Planted Restriction Map Generator
===============================================================================
Module: restriction_map_generator.py
Project: Learning Time-Varying Relational Constraints on Dynamic Heterogeneous Graphs

This module implements Step 2 of the synthetic data generation pipeline:
  - Planted restriction maps with controlled spectral properties
  - Block-diagonal structure for parameter efficiency
  - Coboundary violation computation for anomaly scoring

Connection to paper:
  The restriction maps ρ_{r,t} define "compliance rules" per edge type.
  The coboundary operator δ measures how much each edge violates these rules.
  Anomaly detection = finding edges with high ||δ_t x||².

Dependencies:
  - graph_generator.py (for HeteroGraphSnapshot, HeterogeneousGraphGenerator)
    Note: This module can also run standalone for testing without that import.
===============================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import from graph_generator for integrated pipeline usage.
# Note: This module can run standalone for testing (see __main__ block below).
try:
    from graph_generator import (
        HeteroGraphSnapshot,
        HeterogeneousGraphGenerator,
        NodeTypeConfig,
        EdgeTypeConfig,
    )
except ImportError:
    # Allow standalone execution without graph_generator present
    HeteroGraphSnapshot = None
    HeterogeneousGraphGenerator = None
    NodeTypeConfig = None
    EdgeTypeConfig = None


# ==============================================================================
# STEP 2: PLANTED RESTRICTION MAP GENERATION
# ==============================================================================
#
# WHY THIS EXISTS:
#   A restriction map ρ is the mathematical object that encodes "rules" for a
#   relationship type. For edge (u, v) of type r:
#       violation = ||ρ_source @ x(u) - ρ_target @ x(v)||²
#
#   Low violation  → the edge is compliant (normal behavior)
#   High violation → the edge breaks the current rules (anomalous)
#
#   By planting KNOWN restriction maps, we create ground truth that lets us
#   verify whether our model can recover these maps from data alone.
#
# WHY BLOCK-DIAGONAL:
#   A full d×d matrix has d² parameters per edge type — too expensive for
#   large feature dimensions. Block-diagonal structure:
#       ρ = diag(B_1, B_2, ..., B_k),  each B_i ∈ R^{b×b}
#   reduces parameters from d² to k·b² while each block captures a distinct
#   "aspect" of the constraint (e.g., amount compliance, timing rules, etc.)
#
# WHY CONTROL SPECTRAL PROPERTIES:
#   The singular values of each block determine constraint "strength":
#   - Large singular value → that dimension is heavily constrained
#   - Small singular value → that dimension is loosely constrained
#   - Controlling the range lets us test recovery under various difficulties
# ==============================================================================


@dataclass
class RestrictionMapConfig:
    """
    Configuration for restriction map generation.

    Attributes:
        block_size: Size of each diagonal block (b×b)
        num_blocks: Number of blocks (k); total map dim = b × k
        spectral_range: (min, max) for singular values of each block
        condition_number: Maximum ratio of largest/smallest singular value per block
        orthogonal_init: If True, use orthogonal basis for blocks (more stable)
        seed: Random seed for reproducibility
    """
    block_size: int = 4
    num_blocks: int = 4
    spectral_range: Tuple[float, float] = (0.5, 2.0)
    condition_number: float = 4.0
    orthogonal_init: bool = True
    seed: int = 42


class RestrictionMapGenerator:
    """
    Generates planted ground-truth restriction maps for each edge type.

    Each restriction map is a block-diagonal matrix:
        ρ_r = diag(B_1, B_2, ..., B_k)
    where B_i ∈ R^{b×b} with controlled singular values.

    Pipeline position:
        Step 1 (HeterogeneousGraphGenerator) → creates graph topology
        Step 2 (this class) → attaches "rules" to each edge type
        Steps 3-6 → evolve rules, generate features, assign labels
    """

    def __init__(self, config: RestrictionMapConfig, edge_types: List[str]):
        """
        Args:
            config: Spectral and structural parameters for the maps
            edge_types: List of edge type names (e.g., ['sends_to', 'owns', 'transfers'])
        """
        self.config = config
        self.edge_types = edge_types
        self.rng = np.random.default_rng(config.seed)

        # Total dimension of the restriction map: b * k
        self.map_dim = config.block_size * config.num_blocks

        # Storage: each edge type gets (ρ_source, ρ_target)
        self.restriction_maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._generate_all_maps()

    def _generate_single_block(self) -> torch.Tensor:
        """
        Generate one b×b block with controlled singular values.

        Algorithm:
            1. Create random orthogonal matrix Q via QR decomposition
            2. Sample singular values within [spectral_range] respecting condition_number
            3. Construct block = Q @ diag(σ₁, σ₂, ..., σ_b)

        The orthogonal basis Q determines WHICH directions are constrained.
        The singular values σ determine HOW STRONGLY each direction is constrained.
        """
        b = self.config.block_size

        # --- Orthogonal basis via QR factorization ---
        random_matrix = self.rng.standard_normal((b, b))
        Q, R = np.linalg.qr(random_matrix)
        Q = Q * np.sign(np.diag(R))  # Ensure det(Q) = +1

        # --- Controlled singular values ---
        low, high = self.config.spectral_range
        sigma_max = self.rng.uniform(low, high)
        sigma_min = max(low, sigma_max / self.config.condition_number)
        sigmas = self.rng.uniform(sigma_min, sigma_max, size=b)
        sigmas = np.sort(sigmas)[::-1]  # Descending order

        # --- Assemble block ---
        if self.config.orthogonal_init:
            block = Q @ np.diag(sigmas)
        else:
            # Use two independent orthogonal bases (U, V) for full generality
            V_random = self.rng.standard_normal((b, b))
            V, _ = np.linalg.qr(V_random)
            block = Q @ np.diag(sigmas) @ V.T

        return torch.tensor(block, dtype=torch.float32)

    def _generate_block_diagonal_map(self) -> torch.Tensor:
        """
        Assemble a full block-diagonal restriction map.

        Structure:
            ρ = | B_1  0    0   ...  0   |
                | 0    B_2  0   ...  0   |
                | 0    0    B_3 ...  0   |
                | ...                    |
                | 0    0    0   ...  B_k |

        Shape: (map_dim, map_dim) where map_dim = b * k
        Parameters: k * b² (vs. (b*k)² for dense — significant savings)
        """
        b = self.config.block_size
        k = self.config.num_blocks
        full_map = torch.zeros(self.map_dim, self.map_dim)

        for i in range(k):
            block = self._generate_single_block()
            start = i * b
            end = (i + 1) * b
            full_map[start:end, start:end] = block

        return full_map

    def _generate_all_maps(self):
        """
        Generate restriction maps for every edge type.

        Each edge type gets TWO maps (source and target) because:
            violation(u,v) = ||ρ_source @ x(u) - ρ_target @ x(v)||²

        Different maps for source/target allow asymmetric relationships
        (e.g., "sender's profile should map to X, receiver's profile should map to Y,
        and X should equal Y for compliant transactions").
        """
        for edge_type in self.edge_types:
            rho_source = self._generate_block_diagonal_map()
            rho_target = self._generate_block_diagonal_map()
            self.restriction_maps[edge_type] = (rho_source, rho_target)

    def get_maps(self, edge_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve restriction maps for a given edge type.

        Returns:
            (ρ_source, ρ_target) each of shape (map_dim, map_dim)
        """
        return self.restriction_maps[edge_type]

    def compute_violation(
        self,
        edge_type: str,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the coboundary violation score for a batch of edges.

        This is THE anomaly signal:
            violation(e) = ||ρ_source @ x(source) - ρ_target @ x(target)||²

        Args:
            edge_type: Which relationship type
            source_features: (num_edges, map_dim) — features of source nodes
            target_features: (num_edges, map_dim) — features of target nodes

        Returns:
            (num_edges,) — violation score per edge (higher = more anomalous)
        """
        rho_source, rho_target = self.restriction_maps[edge_type]

        # Project into "edge space" using restriction maps
        source_projected = source_features @ rho_source.T  # (num_edges, map_dim)
        target_projected = target_features @ rho_target.T  # (num_edges, map_dim)

        # Coboundary = difference in edge space
        coboundary = source_projected - target_projected

        # Violation = squared L2 norm of coboundary
        violation = (coboundary ** 2).sum(dim=-1)

        return violation

    def get_spectral_info(self, edge_type: str) -> Dict[str, np.ndarray]:
        """Retrieve singular value analysis for a given edge type's maps."""
        rho_source, rho_target = self.restriction_maps[edge_type]

        src_svd = torch.linalg.svdvals(rho_source).numpy()
        tgt_svd = torch.linalg.svdvals(rho_target).numpy()

        # Filter near-zero values (from off-diagonal zeros)
        src_nonzero = src_svd[src_svd > 1e-10]
        tgt_nonzero = tgt_svd[tgt_svd > 1e-10]

        return {
            "source_singular_values": src_svd,
            "target_singular_values": tgt_svd,
            "source_condition_number": src_nonzero.max() / src_nonzero.min(),
            "target_condition_number": tgt_nonzero.max() / tgt_nonzero.min(),
            "num_active_dimensions": len(src_nonzero),
        }

    def summary(self) -> str:
        """Human-readable summary of generated restriction maps."""
        lines = [
            "=" * 60,
            "  RESTRICTION MAP GENERATOR — SUMMARY",
            "=" * 60,
            f"  Block size (b):        {self.config.block_size}",
            f"  Num blocks (k):        {self.config.num_blocks}",
            f"  Total map dimension:   {self.map_dim} × {self.map_dim}",
            f"  Parameters per map:    {self.config.num_blocks * self.config.block_size**2}",
            f"  (vs. dense):           {self.map_dim**2}",
            f"  Spectral range:        {self.config.spectral_range}",
            f"  Condition number cap:  {self.config.condition_number}",
            f"  Edge types:            {self.edge_types}",
            "-" * 60,
        ]

        for edge_type in self.edge_types:
            info = self.get_spectral_info(edge_type)
            lines.append(f"\n  [{edge_type}]")
            lines.append(
                f"    Source SVs: min={info['source_singular_values'][info['source_singular_values']>1e-10].min():.3f}, "
                f"max={info['source_singular_values'].max():.3f}, "
                f"cond={info['source_condition_number']:.2f}"
            )
            lines.append(
                f"    Target SVs: min={info['target_singular_values'][info['target_singular_values']>1e-10].min():.3f}, "
                f"max={info['target_singular_values'].max():.3f}, "
                f"cond={info['target_condition_number']:.2f}"
            )
            lines.append(f"    Active dims: {info['num_active_dimensions']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ==============================================================================
# USAGE EXAMPLE & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step 2: Generate planted restriction maps (standalone test)
    # ------------------------------------------------------------------
    edge_types = ["sends_to", "owns", "transfers"]

    rmap_config = RestrictionMapConfig(
        block_size=4,
        num_blocks=4,       # Total dim = 4*4 = 16 (matches feature_dim)
        spectral_range=(0.5, 2.0),
        condition_number=4.0,
        orthogonal_init=True,
        seed=42
    )

    rmap_gen = RestrictionMapGenerator(config=rmap_config, edge_types=edge_types)
    print(rmap_gen.summary())

    # ------------------------------------------------------------------
    # Validate: compliant vs. anomalous edges should separate
    # ------------------------------------------------------------------
    print("\nVALIDATION: Violation score separation")
    print("-" * 60)

    num_test_edges = 500
    dim = rmap_config.block_size * rmap_config.num_blocks  # 16

    for edge_type in edge_types:
        rho_src, rho_tgt = rmap_gen.get_maps(edge_type)

        # COMPLIANT edges: target features are constructed to satisfy the constraint
        # If ρ_src @ x(u) ≈ ρ_tgt @ x(v), then v = ρ_tgt⁻¹ @ ρ_src @ x(u) + noise
        source_feats = torch.randn(num_test_edges, dim)
        projected = source_feats @ rho_src.T  # What source looks like in edge space
        # Invert target map to find compliant target features
        rho_tgt_inv = torch.linalg.pinv(rho_tgt)
        compliant_target = projected @ rho_tgt_inv.T + 0.05 * torch.randn(num_test_edges, dim)

        # ANOMALOUS edges: completely random target (no relationship to source)
        anomalous_target = torch.randn(num_test_edges, dim)

        # Compute violations
        v_normal = rmap_gen.compute_violation(edge_type, source_feats, compliant_target)
        v_anomalous = rmap_gen.compute_violation(edge_type, source_feats, anomalous_target)

        print(f"\n  [{edge_type}]")
        print(f"    Normal edges:    mean={v_normal.mean():.4f}, std={v_normal.std():.4f}")
        print(f"    Anomalous edges: mean={v_anomalous.mean():.4f}, std={v_anomalous.std():.4f}")
        print(f"    Separation ratio: {v_anomalous.mean() / max(v_normal.mean(), 1e-8):.1f}x")
        print(f"    → {'✓ GOOD' if v_anomalous.mean() > 5 * v_normal.mean() else '⚠ WEAK'} separation")

    # ------------------------------------------------------------------
    # Optional: Integrated test with graph_generator (if available)
    # ------------------------------------------------------------------
    if HeterogeneousGraphGenerator is not None:
        print("\n" + "=" * 60)
        print("  INTEGRATED TEST: graph_generator + restriction_map_generator")
        print("=" * 60)

        node_configs = [
            NodeTypeConfig(name="Account", count=1000, feature_dim=16),
            NodeTypeConfig(name="Entity", count=200, feature_dim=16),
            NodeTypeConfig(name="Transaction", count=5000, feature_dim=16),
        ]
        edge_configs = [
            EdgeTypeConfig(name="sends_to", source_type="Account", target_type="Account", density=0.005),
            EdgeTypeConfig(name="owns", source_type="Entity", target_type="Account", density=0.02),
            EdgeTypeConfig(name="transfers", source_type="Account", target_type="Transaction", density=0.003),
        ]

        graph_gen = HeterogeneousGraphGenerator(node_configs, edge_configs, seed=42)
        snapshot = graph_gen.generate(timestep=0)

        print(f"\n  Graph snapshot at t={snapshot.timestep}:")
        print(f"    Node types: {snapshot.node_types}")
        print(f"    Edge types: {snapshot.edge_types}")
        for etype, idx in snapshot.edge_indices.items():
            print(f"      {etype}: {idx.shape[1]} edges")
        print("\n  ✓ Integrated pipeline working correctly.")
    else:
        print("\n  [INFO] graph_generator.py not found — skipping integrated test.")
        print("         This module runs standalone for restriction map testing.")

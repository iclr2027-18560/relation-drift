"""
===============================================================================
Synthetic Temporal Heterogeneous Graph Generator
===============================================================================
Module: graph_generator.py
Project: Learning Time-Varying Relational Constraints on Dynamic Heterogeneous Graphs

This module implements Step 1 of the synthetic data generation pipeline:
  - Base heterogeneous graph topology (nodes, edges, types)
  - Temporal sequence generation with gradual evolution

Connection to paper:
  The graph topology defines the structure over which restriction maps and
  coboundary operators will be defined in subsequent pipeline steps.
===============================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# ==============================================================================
# STEP 1: BASE HETEROGENEOUS GRAPH GENERATOR
# ==============================================================================


@dataclass
class NodeTypeConfig:
    """Configuration for a single node type."""
    name: str
    count: int
    feature_dim: int


@dataclass
class EdgeTypeConfig:
    """Configuration for a single edge type."""
    name: str
    source_type: str
    target_type: str
    density: float = 0.01  # Fraction of possible edges that exist


@dataclass
class HeteroGraphSnapshot:
    """
    A single snapshot of a heterogeneous graph at a given timestep.

    Attributes:
        node_features: Dict mapping node_type -> tensor of shape (num_nodes, feat_dim)
        edge_indices: Dict mapping edge_type -> tensor of shape (2, num_edges)
        node_types: List of node type names
        edge_types: List of edge type names
        timestep: Integer timestep identifier
        metadata: Optional dict for additional info
    """
    node_features: Dict[str, torch.Tensor]
    edge_indices: Dict[str, torch.Tensor]
    node_types: List[str]
    edge_types: List[str]
    timestep: int
    metadata: Optional[Dict] = None


class HeterogeneousGraphGenerator:
    """
    Generates synthetic heterogeneous graphs with multiple node/edge types.

    This is Step 1 of the pipeline: creating the graph topology that
    subsequent steps will attach restriction maps and features to.

    Typical financial crime graph:
        Node types: Account, Entity, Transaction
        Edge types: sends_to (Account->Account),
                    owns (Entity->Account),
                    transfers (Account->Transaction)
    """

    def __init__(
        self,
        node_configs: List[NodeTypeConfig],
        edge_configs: List[EdgeTypeConfig],
        seed: int = 42
    ):
        self.node_configs = {nc.name: nc for nc in node_configs}
        self.edge_configs = {ec.name: ec for ec in edge_configs}
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Validate edge types reference valid node types
        valid_node_types = set(self.node_configs.keys())
        for ec in edge_configs:
            assert ec.source_type in valid_node_types, \
                f"Edge type '{ec.name}' references unknown source type '{ec.source_type}'"
            assert ec.target_type in valid_node_types, \
                f"Edge type '{ec.name}' references unknown target type '{ec.target_type}'"

    def generate(self, timestep: int = 0) -> HeteroGraphSnapshot:
        """Generate a single graph snapshot at the given timestep."""
        node_features = {}
        for name, config in self.node_configs.items():
            node_features[name] = torch.randn(config.count, config.feature_dim)

        edge_indices = {}
        for name, config in self.edge_configs.items():
            src_count = self.node_configs[config.source_type].count
            tgt_count = self.node_configs[config.target_type].count
            num_possible = src_count * tgt_count
            num_edges = int(num_possible * config.density)
            num_edges = max(num_edges, 1)

            src_indices = self.rng.integers(0, src_count, size=num_edges)
            tgt_indices = self.rng.integers(0, tgt_count, size=num_edges)
            edge_indices[name] = torch.tensor(
                np.stack([src_indices, tgt_indices]), dtype=torch.long
            )

        return HeteroGraphSnapshot(
            node_features=node_features,
            edge_indices=edge_indices,
            node_types=list(self.node_configs.keys()),
            edge_types=list(self.edge_configs.keys()),
            timestep=timestep,
            metadata={"generator": "HeterogeneousGraphGenerator"}
        )

    def generate_temporal_sequence(
        self,
        num_timesteps: int,
        node_growth_rate: float = 0.02,
        edge_rewire_rate: float = 0.05
    ) -> List[HeteroGraphSnapshot]:
        """
        Generate a sequence of graph snapshots with gradual evolution.

        Args:
            num_timesteps: Number of snapshots to generate
            node_growth_rate: Fraction of new nodes added per step
            edge_rewire_rate: Fraction of edges rewired per step

        Returns:
            List of HeteroGraphSnapshot objects
        """
        snapshots = []
        current_counts = {name: cfg.count for name, cfg in self.node_configs.items()}

        for t in range(num_timesteps):
            # Grow node counts slightly
            if t > 0:
                for name in current_counts:
                    growth = int(current_counts[name] * node_growth_rate)
                    current_counts[name] += max(growth, 0)

            # Generate features for current counts
            node_features = {}
            for name, count in current_counts.items():
                feat_dim = self.node_configs[name].feature_dim
                node_features[name] = torch.randn(count, feat_dim)

            # Generate edges with current topology
            edge_indices = {}
            for name, config in self.edge_configs.items():
                src_count = current_counts[config.source_type]
                tgt_count = current_counts[config.target_type]
                num_possible = src_count * tgt_count
                num_edges = int(num_possible * config.density)
                num_edges = max(num_edges, 1)

                src_indices = self.rng.integers(0, src_count, size=num_edges)
                tgt_indices = self.rng.integers(0, tgt_count, size=num_edges)
                edge_indices[name] = torch.tensor(
                    np.stack([src_indices, tgt_indices]), dtype=torch.long
                )

            snapshots.append(HeteroGraphSnapshot(
                node_features=node_features,
                edge_indices=edge_indices,
                node_types=list(self.node_configs.keys()),
                edge_types=list(self.edge_configs.keys()),
                timestep=t,
                metadata={"node_counts": dict(current_counts)}
            ))

        return snapshots


# ==============================================================================
# USAGE EXAMPLE & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step 1: Generate graph topology
    # ------------------------------------------------------------------
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

    print("STEP 1: Graph Topology")
    print(f"  Node types: {snapshot.node_types}")
    print(f"  Edge types: {snapshot.edge_types}")
    for etype, idx in snapshot.edge_indices.items():
        print(f"    {etype}: {idx.shape[1]} edges")
    print()

    # ------------------------------------------------------------------
    # Temporal sequence validation
    # ------------------------------------------------------------------
    print("TEMPORAL SEQUENCE:")
    snapshots = graph_gen.generate_temporal_sequence(num_timesteps=5)
    for snap in snapshots:
        total_edges = sum(idx.shape[1] for idx in snap.edge_indices.values())
        print(f"  t={snap.timestep}: nodes={snap.metadata['node_counts']}, total_edges={total_edges}")

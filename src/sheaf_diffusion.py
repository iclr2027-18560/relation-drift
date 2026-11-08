# ============================================================================
# FILE 2: sheaf_diffusion.py
# ============================================================================
# Static Sheaf Neural Network - Diffusion Layers and Full Network
# Implements the sheaf diffusion process and the complete static sheaf
# network for anomaly detection on heterogeneous graphs.
#
# Architecture:
#   1. Input projection to stalk dimension
#   2. Stack of SheafDiffusionLayers: X^(l+1) = σ((I - L_F) X^(l) W^(l))
#   3. Edge representation via src/tgt concatenation
#   4. AnomalyHead MLP for binary classification
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Local imports (same package)
from sheaf_laplacian import RestrictionMapBank, SheafLaplacian, SheafLaplacianConfig


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StaticSheafConfig:
    """Configuration for the Static Sheaf Network.

    Attributes:
        input_dim: Dimension of input node features.
        hidden_dim: Dimension of hidden layers.
        output_dim: Dimension of output (1 for binary anomaly score).
        stalk_dim: Dimension of the sheaf stalks.
        num_layers: Number of sheaf diffusion layers.
        edge_types: List of edge type identifiers in the heterogeneous graph.
        dropout: Dropout probability.
        num_heads: Number of attention heads (for attention variant).
    """

    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 1
    stalk_dim: int = 4
    num_layers: int = 3
    edge_types: List[str] = field(default_factory=lambda: ["default"])
    dropout: float = 0.1
    num_heads: int = 4


# ============================================================================
# Anomaly Detection Head
# ============================================================================


class AnomalyHead(nn.Module):
    """MLP head for edge-level anomaly scoring.

    Takes concatenated source and target node embeddings and produces
    a scalar anomaly score per edge.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize anomaly detection head.

        Args:
            input_dim: Dimension of concatenated edge representation.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (1 for binary classification).
            dropout: Dropout probability.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, edge_representations: Tensor) -> Tensor:
        """Compute anomaly scores.

        Args:
            edge_representations: Concatenated src/tgt embeddings,
                shape (num_edges, input_dim).

        Returns:
            Anomaly scores, shape (num_edges, 1).
        """
        return self.mlp(edge_representations)


# ============================================================================
# Sheaf Diffusion Layer
# ============================================================================


class SheafDiffusionLayer(nn.Module):
    """Single sheaf diffusion layer.

    Implements: X^(l+1) = σ(LayerNorm((I - L_F) X^(l) W^(l)))

    This applies one step of sheaf Laplacian diffusion followed by
    a learned linear transformation and nonlinearity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stalk_dim: int,
        restriction_maps: RestrictionMapBank,
        dropout: float = 0.1,
    ) -> None:
        """Initialize sheaf diffusion layer.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Output feature dimension.
            stalk_dim: Stalk dimension for the sheaf.
            restriction_maps: The shared restriction map bank.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.dropout = dropout

        # Sheaf Laplacian for diffusion
        self.sheaf_laplacian = SheafLaplacian(restriction_maps)

        # Linear transformation W^(l)
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_types: List[str],
        edge_indices: Dict[str, Tensor],
    ) -> Tensor:
        """Apply one sheaf diffusion step.

        Computes: X^(l+1) = σ(LayerNorm((I - L_F) X^(l) W^(l)))

        Args:
            x: Node features, shape (num_nodes, input_dim).
            edge_types: List of edge type identifiers.
            edge_indices: Dict mapping edge type to edge_index tensors.

        Returns:
            Updated node features, shape (num_nodes, hidden_dim).
        """
        # Step 1: Sheaf diffusion — (I - L_F) @ X
        diffused = self.sheaf_laplacian.compute_diffusion(x, edge_types, edge_indices)

        # Step 2: Linear transformation
        out = self.linear(diffused)

        # Step 3: Normalization + activation + dropout
        out = self.layer_norm(out)
        out = F.elu(out)
        out = self.drop(out)

        return out


# ============================================================================
# Static Sheaf Network (Full Model)
# ============================================================================


class StaticSheafNetwork(nn.Module):
    """Complete Static Sheaf Neural Network for anomaly detection.

    Architecture:
        1. Input projection to stalk dimension
        2. Stack of SheafDiffusionLayers
        3. Edge representation via src/tgt concatenation
        4. AnomalyHead MLP for binary classification

    This implements Bodnar et al. 2022 adapted for heterogeneous graphs
    WITHOUT temporal evolution — restriction maps are static.
    """

    def __init__(self, config: StaticSheafConfig) -> None:
        """Initialize the static sheaf network.

        Args:
            config: Configuration dataclass with all hyperparameters.
        """
        super().__init__()
        self.config = config

        # Restriction map bank (shared across all layers)
        self.restriction_maps = RestrictionMapBank(
            edge_types=config.edge_types,
            input_dim=config.hidden_dim,
            stalk_dim=config.stalk_dim,
        )

        # Input projection: map raw features to hidden_dim (stalk working space)
        self.input_proj_src = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_proj_tgt = nn.Linear(config.input_dim, config.hidden_dim)

        # Stack of sheaf diffusion layers
        self.diffusion_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = SheafDiffusionLayer(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                stalk_dim=config.stalk_dim,
                restriction_maps=self.restriction_maps,
                dropout=config.dropout,
            )
            self.diffusion_layers.append(layer)

        # Anomaly detection head (input: concatenation of src and tgt embeddings)
        self.anomaly_head = AnomalyHead(
            input_dim=config.hidden_dim * 2,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        snapshot: Dict,
        src_features: Optional[Tensor] = None,
        tgt_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass: compute edge-level anomaly scores.

        Args:
            snapshot: Dictionary containing graph data with keys:
                - 'edge_index': Dict[str, Tensor] mapping edge type to indices
                - 'edge_types': List[str] of edge types present
                - 'node_features': Tensor of shape (num_nodes, input_dim)
                - 'num_nodes': int
                - 'query_edges': Tensor of shape (2, num_query) edges to score
                - 'query_edge_type': str type of query edges
            src_features: Optional source node features override.
            tgt_features: Optional target node features override.

        Returns:
            Anomaly scores, shape (num_query_edges,).
        """
        # Extract graph data from snapshot
        edge_indices = snapshot.get("edge_index", {})
        edge_types = snapshot.get("edge_types", self.config.edge_types)
        num_nodes = snapshot.get("num_nodes", 0)
        query_edges = snapshot.get("query_edges")

        # Get node features
        if src_features is not None and tgt_features is not None:
            node_features = (
                src_features
                if src_features.shape[0] >= num_nodes
                else torch.cat([src_features, tgt_features], dim=0)
            )
        elif "node_features" in snapshot:
            node_features = snapshot["node_features"]
        else:
            node_features = torch.zeros(
                num_nodes, self.config.input_dim, device=self._get_device()
            )

        # Step 1: Project node features to hidden dimension
        x = self.input_proj_src(node_features)  # (num_nodes, hidden_dim)

        # Step 2: Apply sheaf diffusion layers with residual connections
        for layer in self.diffusion_layers:
            x_new = layer(x, edge_types, edge_indices)
            if x_new.shape == x.shape:
                x = x + x_new  # Residual connection
            else:
                x = x_new

        # Step 3: Compute edge representations for query edges
        if query_edges is not None:
            src_idx = query_edges[0]  # (num_query,)
            tgt_idx = query_edges[1]  # (num_query,)
        else:
            # Score all edges of all types if no query edges specified
            all_src = []
            all_tgt = []
            for etype, eidx in edge_indices.items():
                all_src.append(eidx[0])
                all_tgt.append(eidx[1])
            if all_src:
                src_idx = torch.cat(all_src)
                tgt_idx = torch.cat(all_tgt)
            else:
                return torch.tensor([], device=self._get_device())

        src_emb = x[src_idx]  # (num_query, hidden_dim)
        tgt_emb = x[tgt_idx]  # (num_query, hidden_dim)

        # Concatenate source and target embeddings
        edge_repr = torch.cat([src_emb, tgt_emb], dim=-1)  # (num_query, 2*hidden_dim)

        # Step 4: Anomaly scoring
        scores = self.anomaly_head(edge_repr).squeeze(-1)  # (num_query,)

        return scores

    def get_learned_maps(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        """Extract learned restriction maps for analysis.

        Returns:
            Dictionary mapping edge type to (rho_src, rho_tgt) tensor pairs.
        """
        return self.restriction_maps.get_all_maps()

    def _get_device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device
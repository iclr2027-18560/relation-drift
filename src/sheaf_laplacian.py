# ============================================================================
# FILE 1: sheaf_laplacian.py
# ============================================================================
# Static Sheaf Neural Network - Sheaf Laplacian Construction
# Implements restriction maps, coboundary operators, and sheaf Laplacian
# for heterogeneous graphs following Bodnar et al. 2022.
#
# Math Overview:
#   - Restriction Maps: ρ_src, ρ_tgt learned per edge type
#   - Coboundary: (δx)_e = ρ_src @ x(u) - ρ_tgt @ x(v)
#   - Sheaf Laplacian: L_F = δ^T δ (computed via message passing)
#   - Diffusion: X^(l+1) = σ((I - L_F) X^(l) W^(l))
# ============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SheafLaplacianConfig:
    """Configuration for the Sheaf Laplacian construction.

    Attributes:
        stalk_dim: Dimension of the stalks in the cellular sheaf.
        num_layers: Number of diffusion layers to apply.
        learn_maps: Whether restriction maps are learnable parameters.
    """

    stalk_dim: int = 4
    num_layers: int = 3
    learn_maps: bool = True


# ============================================================================
# Restriction Map Bank
# ============================================================================


class RestrictionMapBank(nn.Module):
    """Stores learned restriction maps ρ_r per edge type.

    For each edge type r, we maintain two matrices:
        - ρ_src(r): maps source node features to the stalk space
        - ρ_tgt(r): maps target node features to the stalk space

    These restriction maps define how node stalks attach to edge stalks
    in the cellular sheaf structure.
    """

    def __init__(self, edge_types: List[str], input_dim: int, stalk_dim: int) -> None:
        """Initialize restriction map bank.

        Args:
            edge_types: List of edge type identifiers (e.g., ['user_item', 'item_user']).
            input_dim: Dimension of input node features.
            stalk_dim: Dimension of the stalk (fiber) space.
        """
        super().__init__()
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim

        # Store restriction maps as ParameterDict for named access.
        # Each edge type has a source and target restriction map.
        self.restriction_maps_src = nn.ParameterDict()
        self.restriction_maps_tgt = nn.ParameterDict()

        for etype in edge_types:
            key = self._sanitize_key(etype)
            # Initialize with orthogonal initialization for stability
            src_map = nn.Parameter(torch.empty(stalk_dim, input_dim))
            tgt_map = nn.Parameter(torch.empty(stalk_dim, input_dim))
            nn.init.orthogonal_(src_map)
            nn.init.orthogonal_(tgt_map)
            self.restriction_maps_src[key] = src_map
            self.restriction_maps_tgt[key] = tgt_map

    def _sanitize_key(self, edge_type: str) -> str:
        """Sanitize edge type string for use as ParameterDict key."""
        return edge_type.replace(" ", "_").replace("-", "_").replace(".", "_")

    def forward(self, edge_type: str) -> Tuple[Tensor, Tensor]:
        """Retrieve restriction maps for a given edge type.

        Args:
            edge_type: The edge type identifier.

        Returns:
            Tuple of (rho_src, rho_tgt), each of shape (stalk_dim, input_dim).
        """
        key = self._sanitize_key(edge_type)
        rho_src = self.restriction_maps_src[key]
        rho_tgt = self.restriction_maps_tgt[key]
        return rho_src, rho_tgt

    def get_all_maps(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        """Extract all learned restriction maps for analysis.

        Returns:
            Dictionary mapping edge type to (rho_src, rho_tgt) tensors.
        """
        result: Dict[str, Tuple[Tensor, Tensor]] = {}
        for etype in self.edge_types:
            key = self._sanitize_key(etype)
            result[etype] = (
                self.restriction_maps_src[key].detach().clone(),
                self.restriction_maps_tgt[key].detach().clone(),
            )
        return result


# ============================================================================
# Coboundary Operator
# ============================================================================


class CoboundaryOperator(nn.Module):
    """Computes the coboundary operator δ for a given graph.

    The coboundary maps node signals to edge signals:
        (δx)_e = ρ_src(e) @ x_u - ρ_tgt(e) @ x_v

    where e = (u, v) is an edge, and ρ_src, ρ_tgt are restriction maps.
    """

    def __init__(self, restriction_maps: RestrictionMapBank) -> None:
        """Initialize coboundary operator.

        Args:
            restriction_maps: The restriction map bank providing ρ per edge type.
        """
        super().__init__()
        self.restriction_maps = restriction_maps

    def compute_coboundary(
        self,
        edge_type: str,
        edge_index: Tensor,
        node_features_src: Tensor,
        node_features_tgt: Tensor,
    ) -> Tensor:
        """Compute coboundary δx for edges of a given type.

        Computes: (δx)_e = ρ_src @ x_u - ρ_tgt @ x_v for each edge (u, v).

        Args:
            edge_type: The edge type identifier.
            edge_index: Edge indices of shape (2, num_edges).
            node_features_src: Source node features, shape (num_src_nodes, input_dim).
            node_features_tgt: Target node features, shape (num_tgt_nodes, input_dim).

        Returns:
            Coboundary values of shape (num_edges, stalk_dim).
        """
        rho_src, rho_tgt = self.restriction_maps(edge_type)

        src_indices = edge_index[0]  # (num_edges,)
        tgt_indices = edge_index[1]  # (num_edges,)

        # Gather source and target node features for each edge
        x_src = node_features_src[src_indices]  # (num_edges, input_dim)
        x_tgt = node_features_tgt[tgt_indices]  # (num_edges, input_dim)

        # Apply restriction maps:
        #   x @ ρ^T : (num_edges, input_dim) @ (input_dim, stalk_dim) -> (num_edges, stalk_dim)
        mapped_src = x_src @ rho_src.t()  # (num_edges, stalk_dim)
        mapped_tgt = x_tgt @ rho_tgt.t()  # (num_edges, stalk_dim)

        # Coboundary: difference in stalk space
        coboundary = mapped_src - mapped_tgt  # (num_edges, stalk_dim)

        return coboundary


# ============================================================================
# Sheaf Laplacian
# ============================================================================


class SheafLaplacian(nn.Module):
    """Constructs and applies the Sheaf Laplacian L_F = δ^T δ.

    Rather than materializing the full Laplacian matrix (which can be very large),
    this class provides methods to compute L_F @ X directly via message passing,
    and a diffusion operation (I - L_F) @ X.
    """

    def __init__(self, restriction_maps: RestrictionMapBank) -> None:
        """Initialize sheaf Laplacian.

        Args:
            restriction_maps: The restriction map bank.
        """
        super().__init__()
        self.restriction_maps = restriction_maps
        self.coboundary = CoboundaryOperator(restriction_maps)

    def compute(
        self,
        edge_types: List[str],
        edge_indices: Dict[str, Tensor],
        num_nodes: int,
    ) -> Tensor:
        """Compute the full sheaf Laplacian matrix (dense).

        WARNING: This materializes the full (N*d) x (N*d) matrix.
        Use compute_diffusion for efficiency in practice.

        The block structure of L_F for edge (u, v):
            L_F[u*d:(u+1)*d, u*d:(u+1)*d] += ρ_src^T @ ρ_src
            L_F[v*d:(v+1)*d, v*d:(v+1)*d] += ρ_tgt^T @ ρ_tgt
            L_F[u*d:(u+1)*d, v*d:(v+1)*d] -= ρ_src^T @ ρ_tgt
            L_F[v*d:(v+1)*d, u*d:(u+1)*d] -= ρ_tgt^T @ ρ_src

        Args:
            edge_types: List of edge type identifiers.
            edge_indices: Dict mapping edge type to edge_index tensors.
            num_nodes: Total number of nodes in the graph.

        Returns:
            Dense sheaf Laplacian of shape (num_nodes * stalk_dim, num_nodes * stalk_dim).
        """
        d = self.restriction_maps.stalk_dim
        N = num_nodes
        L = torch.zeros(N * d, N * d, device=self._get_device())

        for etype in edge_types:
            if etype not in edge_indices:
                continue
            edge_index = edge_indices[etype]
            rho_src, rho_tgt = self.restriction_maps(etype)

            num_edges = edge_index.shape[1]
            for k in range(num_edges):
                u = edge_index[0, k].item()
                v = edge_index[1, k].item()

                rho_src_T_rho_src = rho_src.t() @ rho_src
                rho_tgt_T_rho_tgt = rho_tgt.t() @ rho_tgt
                rho_src_T_rho_tgt = rho_src.t() @ rho_tgt
                rho_tgt_T_rho_src = rho_tgt.t() @ rho_src

                L[u * d:(u + 1) * d, u * d:(u + 1) * d] += rho_src_T_rho_src
                L[v * d:(v + 1) * d, v * d:(v + 1) * d] += rho_tgt_T_rho_tgt
                L[u * d:(u + 1) * d, v * d:(v + 1) * d] -= rho_src_T_rho_tgt
                L[v * d:(v + 1) * d, u * d:(u + 1) * d] -= rho_tgt_T_rho_src

        return L

    def compute_laplacian_times_x(
        self,
        node_features: Tensor,
        edge_types: List[str],
        edge_indices: Dict[str, Tensor],
    ) -> Tensor:
        """Compute L_F @ X via message passing without materializing L_F.

        For each edge (u, v) of type r:
            Contribution to node u: ρ_src^T (ρ_src x_u - ρ_tgt x_v)
            Contribution to node v: ρ_tgt^T (ρ_tgt x_v - ρ_src x_u)

        Args:
            node_features: Node feature matrix, shape (num_nodes, stalk_dim).
            edge_types: List of edge type identifiers.
            edge_indices: Dict mapping edge type to edge_index tensors.

        Returns:
            Result of L_F @ X, shape (num_nodes, stalk_dim).
        """
        result = torch.zeros_like(node_features)

        for etype in edge_types:
            if etype not in edge_indices:
                continue
            edge_index = edge_indices[etype]
            rho_src, rho_tgt = self.restriction_maps(etype)

            src_indices = edge_index[0]  # (num_edges,)
            tgt_indices = edge_index[1]  # (num_edges,)

            # Gather features
            x_src = node_features[src_indices]  # (num_edges, stalk_dim)
            x_tgt = node_features[tgt_indices]  # (num_edges, stalk_dim)

            # Map to stalk space via restriction maps
            mapped_src = x_src @ rho_src.t()  # (num_edges, stalk_dim)
            mapped_tgt = x_tgt @ rho_tgt.t()  # (num_edges, stalk_dim)
            diff = mapped_src - mapped_tgt  # (num_edges, stalk_dim)

            # Backpropagate through restriction maps (adjoint / transpose)
            contrib_src = diff @ rho_src  # (num_edges, stalk_dim)
            contrib_tgt = -diff @ rho_tgt  # (num_edges, stalk_dim)

            # Scatter-add contributions back to nodes
            result.scatter_add_(
                0, src_indices.unsqueeze(1).expand_as(contrib_src), contrib_src
            )
            result.scatter_add_(
                0, tgt_indices.unsqueeze(1).expand_as(contrib_tgt), contrib_tgt
            )

        return result

    def compute_diffusion(
        self,
        node_features: Tensor,
        edge_types: List[str],
        edge_indices: Dict[str, Tensor],
        normalize: bool = True,
    ) -> Tensor:
        """Compute (I - L_F) @ X without materializing L_F.

        Implements one step of sheaf diffusion, smoothing node features
        according to the sheaf structure.

        Args:
            node_features: Node feature matrix, shape (num_nodes, stalk_dim).
            edge_types: List of edge type identifiers.
            edge_indices: Dict mapping edge type to edge_index tensors.
            normalize: Whether to normalize the Laplacian by node degree.

        Returns:
            Diffused features, shape (num_nodes, stalk_dim).
        """
        lf_x = self.compute_laplacian_times_x(node_features, edge_types, edge_indices)

        if normalize:
            # Compute degree-based normalization to prevent numerical issues
            num_nodes = node_features.shape[0]
            degree = torch.zeros(num_nodes, device=node_features.device)
            for etype in edge_types:
                if etype not in edge_indices:
                    continue
                edge_index = edge_indices[etype]
                src_indices = edge_index[0]
                tgt_indices = edge_index[1]
                degree.scatter_add_(
                    0, src_indices, torch.ones_like(src_indices, dtype=torch.float)
                )
                degree.scatter_add_(
                    0, tgt_indices, torch.ones_like(tgt_indices, dtype=torch.float)
                )

            # Avoid division by zero
            degree = degree.clamp(min=1.0)
            # Normalize: scale L_F @ X by 1/degree for stability
            norm_factor = (1.0 / degree).unsqueeze(1)  # (num_nodes, 1)
            lf_x = lf_x * norm_factor

        # Diffusion step: (I - L_F) @ X
        diffused = node_features - lf_x
        return diffused

    def _get_device(self) -> torch.device:
        """Get the device of the restriction map parameters."""
        for param in self.restriction_maps.parameters():
            return param.device
        return torch.device("cpu")

"""Temporal Sheaf Network V3.5 with jerk detection and fast rho encoder.

Key mathematical components:
    - Cayley map for orthogonal restriction maps
    - Jerk (2nd derivative) detection of coboundary norm changes
    - Adaptive gate with bias=+2.0 and floor=0.1
    - Linear decay recovery window (6 steps)
    - Residual rho update: rho_new = rho_old + gate * delta_rho
    - FastRotationEncoder: ρ_fast = F(G_t, X_t) (snapshot, no history)
    - ChangeDetector + geodesic fusion of ρ_slow and ρ_fast on SO(d)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── Constants ───────────────────────────────────────────────────────────
CAYLEY_EPS = 1e-8
COBOUNDARY_EPS = 1e-8
DEFAULT_STALK_DIM = 12
DEFAULT_HIDDEN_DIM = 128
DEFAULT_GRU_LAYERS = 2
DEFAULT_NUM_DIFFUSION_LAYERS = 3
DEFAULT_NUM_EDGE_TYPES = 3
DEFAULT_RECOVERY_STEPS = 6
DEFAULT_RECOVERY_DECAY_RATE = 1.0 / 6.0
DEFAULT_JERK_EMA_ALPHA = 0.3
DEFAULT_JERK_THRESHOLD = 2.0
DEFAULT_GATE_BIAS_INIT = 2.0
DEFAULT_GATE_FLOOR = 0.1
DEFAULT_DROPOUT = 0.1
DEFAULT_POS_WEIGHT = 9.0
DEFAULT_FAST_ENCODER_HIDDEN = 128
DEFAULT_GATE_EMA_ALPHA = 0.9


@dataclass
class SheafConfig:
    """Configuration for the temporal sheaf network architecture.

    Attributes:
        node_feat_dim:         Input feature dimension per node.
        stalk_dim:             Dimension of sheaf stalks.
        hidden_dim:            GRU hidden state dimension.
        gru_layers:            Number of GRU layers.
        num_diffusion_layers:  Number of sheaf diffusion layers.
        num_edge_types:        Number of heterogeneous edge types.
        recovery_steps:        Forced-open steps after jump detection.
        jerk_ema_alpha:        EMA smoothing for jerk detector.
        jerk_threshold:        Threshold for 2nd-derivative detection.
        gate_bias_init:        Initial bias for adaptive gate.
        gate_floor:            Minimum gate activation value.
        dropout:               Dropout rate.
        pos_weight:            Positive class weight for BCE loss.
        use_fast_encoder:      Enable snapshot ρ_fast + geodesic fuse.
        fast_encoder_hidden:   Hidden dim for FastRotationEncoder.
        gate_ema_alpha:        EMA alpha for ChangeDetector embedding.
        connection_family:     ``orthogonal`` (Cayley / SO(d)) or
                               ``unconstrained`` (GL analog).
        static_connection:     If True, one learned map used at every t.
    """

    node_feat_dim: int = 16
    stalk_dim: int = DEFAULT_STALK_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    gru_layers: int = DEFAULT_GRU_LAYERS
    num_diffusion_layers: int = DEFAULT_NUM_DIFFUSION_LAYERS
    num_edge_types: int = DEFAULT_NUM_EDGE_TYPES
    recovery_steps: int = DEFAULT_RECOVERY_STEPS
    jerk_ema_alpha: float = DEFAULT_JERK_EMA_ALPHA
    jerk_threshold: float = DEFAULT_JERK_THRESHOLD
    gate_bias_init: float = DEFAULT_GATE_BIAS_INIT
    gate_floor: float = DEFAULT_GATE_FLOOR
    dropout: float = DEFAULT_DROPOUT
    pos_weight: float = DEFAULT_POS_WEIGHT
    use_fast_encoder: bool = True
    fast_encoder_hidden: int = DEFAULT_FAST_ENCODER_HIDDEN
    gate_ema_alpha: float = DEFAULT_GATE_EMA_ALPHA
    pair_residual_dim: int = 0
    connection_family: str = "orthogonal"
    static_connection: bool = False


@dataclass
class TrainConfig:
    """Configuration for training the temporal sheaf model.

    Routing curriculum (breaks fast-encoder / gate collapse):
        Phase 1: Freeze GRU; force fusion g=1; train ρ_fast (+ head).
        Phase 2: Freeze trunks; train change detector via jump BCE.
        Phase 3: End-to-end fine-tune at low LR.

    Attributes:
        epochs_phase1:       Pre-train fast encoder (fusion forced open).
        epochs_phase2:       Train gate only (jump-indicator BCE).
        epochs_phase3:       Joint fine-tune.
        learning_rate:       Main parameter learning rate.
        gate_learning_rate:  Learning rate for gate/jump params.
        finetune_learning_rate: LR for phase-3 E2E.
        ortho_penalty:       Orthogonality regularization weight.
        smooth_penalty:      Temporal smoothness penalty weight.
        fusion_gate_penalty: L1 on fusion gate (0 disables; fights spikes).
        jump_gate_bce_weight: Weight for BCE(gate, jump_indicator).
        jump_gate_window:    Extra steps after jump to keep gate target=1
                             (0 = pulse only at crossing — better for multi-jump).
        fast_cb_penalty:     Unsupervised coboundary loss under ρ_fast.
        hard_handoff_on_jump: Reset GRU and ρ_slow←ρ_fast on jump pulses.
        train_window_only:   If True, never use post-split snapshots in training
                             (labels or unsupervised coboundary). Default True.
        use_gt_jump_at_eval: If False, evaluation does not receive jump times.
        label_smoothing:     Label smoothing factor.
        grad_clip_norm:      Maximum gradient norm for clipping.
        use_routing_curriculum: Enable 3-phase routing schedule.
    """

    epochs_phase1: int = 80
    epochs_phase2: int = 40
    epochs_phase3: int = 120
    learning_rate: float = 1e-3
    gate_learning_rate: float = 5e-3
    finetune_learning_rate: float = 3e-4
    ortho_penalty: float = 0.01
    smooth_penalty: float = 0.005
    fusion_gate_penalty: float = 0.0
    jump_gate_bce_weight: float = 1.0
    jump_gate_window: int = 0
    fast_cb_penalty: float = 0.1
    hard_handoff_on_jump: bool = True
    train_window_only: bool = True
    use_gt_jump_at_eval: bool = True
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    use_routing_curriculum: bool = True


class CoboundaryOperator(nn.Module):
    """Computes the coboundary operator norm for sheaf consistency.

    Asymmetric discrete connection, not a shared left action.
    For an edge (u → v) and ρ ∈ SO(d), in the row-vector layout of
    this module,
        (δ s)(u→v) = s(u) ρ − s(v) ρᵀ.
    If both maps were the same ρ, orthogonality would cancel
    (|ρ s_u − ρ s_v| = |s_u − s_v|). The transpose on the target
    endpoint makes the energy depend on the relative transport ρ².

    Attributes:
        stalk_dim: Dimension of sheaf stalks.
    """

    def __init__(self, stalk_dim: int) -> None:
        """Initialize coboundary operator.

        Args:
            stalk_dim: Dimension of sheaf stalks.
        """
        super().__init__()
        self.stalk_dim = stalk_dim
        self.family = "orthogonal"

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        rho: Tensor,
    ) -> Tensor:
        """Compute coboundary norm from node features and maps.

        Args:
            x: Node features in stalk space [num_nodes, stalk_dim].
            edge_index: Edge indices [2, num_edges].
            rho: Restriction maps [num_edge_types, d, d].

        Returns:
            Scalar coboundary norm (mean over edges).
        """
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        x_dst = x[dst]

        rho_mat = rho[0]
        mapped_src = torch.matmul(x_src, rho_mat)
        if self.family == "unconstrained":
            # GL energy: s(u) ρ − s(v). Do not use ρᵀ; that assumes SO(d).
            delta = mapped_src - x_dst
        else:
            # Relative transport: source uses ρ, target uses ρᵀ (not ρ, ρ).
            mapped_dst = torch.matmul(x_dst, rho_mat.T)
            delta = mapped_src - mapped_dst
        return delta.norm(dim=-1).mean()


class JerkDetector(nn.Module):
    """Detects regime changes via 2nd derivative of coboundary norms.

    Monitors jerk = d²/dt² of coboundary norms using EMA smoothing.
    Triggers when relative jerk exceeds threshold.

    Attributes:
        ema_alpha: Exponential moving average smoothing factor.
        threshold: Detection threshold for relative jerk signal.
    """

    def __init__(
        self,
        ema_alpha: float = DEFAULT_JERK_EMA_ALPHA,
        threshold: float = DEFAULT_JERK_THRESHOLD,
    ) -> None:
        """Initialize jerk detector.

        Args:
            ema_alpha: EMA smoothing factor (higher = more reactive).
            threshold: Jerk detection threshold.
        """
        super().__init__()
        self.ema_alpha = ema_alpha
        self.threshold = threshold
        self.register_buffer("ema_norm", torch.tensor(0.0))
        self.register_buffer("prev_delta", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, coboundary_norm: Tensor) -> Tensor:
        """Compute jerk signal from current coboundary norm.

        The jerk is the second derivative (rate of change of the
        rate of change) of the coboundary norm time series.

        Args:
            coboundary_norm: Scalar norm of coboundary output.

        Returns:
            Binary tensor (1.0 if jump detected, else 0.0).
        """
        norm_val = coboundary_norm.detach()

        if not self.initialized:
            self.ema_norm.fill_(norm_val.item())
            self.initialized.fill_(True)
            return torch.tensor(0.0, device=coboundary_norm.device)

        # First derivative: change from EMA baseline
        delta = norm_val - self.ema_norm

        # Second derivative: jerk
        jerk = (delta - self.prev_delta).abs()

        # Update EMA buffers in-place (never keep autograd history)
        new_ema = (
            self.ema_alpha * norm_val
            + (1.0 - self.ema_alpha) * self.ema_norm
        )
        self.ema_norm.copy_(new_ema)
        self.prev_delta.copy_(delta)

        # Relative jerk normalized by EMA magnitude
        relative_jerk = jerk / (self.ema_norm.abs() + COBOUNDARY_EPS)
        detected = (relative_jerk > self.threshold).float().detach()

        return detected

    def reset(self) -> None:
        """Reset detector state for new sequence."""
        self.ema_norm.fill_(0.0)
        self.prev_delta.fill_(0.0)
        self.initialized.fill_(False)


class LinearDecayRecovery(nn.Module):
    """Recovery window with linear decay from 1.0 to 0.0.

    After jump detection, forces the gate open for N steps with
    linearly decaying strength:
        strength = max(0, 1 - step / window)

    Attributes:
        max_steps: Maximum recovery window length (default 6).
    """

    def __init__(
        self,
        max_steps: int = DEFAULT_RECOVERY_STEPS,
    ) -> None:
        """Initialize recovery controller.

        Args:
            max_steps: Number of steps in recovery window.
        """
        super().__init__()
        self.max_steps = max_steps
        self.register_buffer("steps_remaining", torch.tensor(0))

    def trigger(self) -> None:
        """Trigger recovery window (called on jump detection)."""
        self.steps_remaining.fill_(self.max_steps)

    def step(self) -> Tensor:
        """Get current recovery strength and advance state.

        Strength formula: max(0, 1 - elapsed / window)

        Returns:
            Gate override value in [0.0, 1.0].
        """
        if self.steps_remaining <= 0:
            return torch.tensor(0.0, device=self.steps_remaining.device)

        elapsed = self.max_steps - self.steps_remaining.item()
        strength = max(0.0, 1.0 - elapsed / self.max_steps)
        self.steps_remaining -= 1

        return torch.tensor(strength, device=self.steps_remaining.device)

    def reset(self) -> None:
        """Reset recovery state for new sequence."""
        self.steps_remaining.fill_(0)


class AdaptiveGate(nn.Module):
    """Adaptive update gate with bias initialization and floor.

    The gate controls how much of the new restriction map update
    is applied. Initialized with positive bias (+2.0) so it starts
    open, with a minimum floor of 0.1 to prevent dead gates.

    Attributes:
        gate_net: Two-layer network computing gate activation.
        floor: Minimum gate activation value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        bias_init: float = DEFAULT_GATE_BIAS_INIT,
        floor: float = DEFAULT_GATE_FLOOR,
    ) -> None:
        """Initialize adaptive gate.

        Args:
            input_dim: Dimension of input features.
            hidden_dim: Hidden layer dimension.
            bias_init: Initial bias (+2.0 means gate starts open).
            floor: Minimum gate value (0.1 prevents dead gates).
        """
        super().__init__()
        self.floor = floor
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize final bias so gate starts open
        nn.init.constant_(self.gate_net[-1].bias, bias_init)

    def forward(self, features: Tensor) -> Tensor:
        """Compute gate activation.

        Args:
            features: Input features [batch, input_dim].

        Returns:
            Gate value in [floor, 1.0], shape [batch, 1].
        """
        raw = torch.sigmoid(self.gate_net(features))
        return self.floor + (1.0 - self.floor) * raw


class RhoEvolver(nn.Module):
    """Evolves restriction maps temporally via GRU + Cayley map.

    Architecture:
        1. Projects coboundary statistics to GRU input dimension
        2. 2-layer GRU (hidden=128) processes temporal signal
        3. Projects GRU output to skew-symmetric parameters
        4. Cayley map produces orthogonal delta
        5. Residual update: rho_new = rho_old + gate * delta_rho

    Attributes:
        config: Sheaf configuration.
        input_proj: Projects coboundary stats to GRU input size.
        gru: Recurrent unit for temporal evolution.
        skew_proj: Projects hidden state to Cayley parameters.
    """

    def __init__(self, config: SheafConfig) -> None:
        """Initialize rho evolver.

        Args:
            config: Sheaf network configuration.
        """
        super().__init__()
        self.config = config
        d = config.stalk_dim

        # Coboundary statistics: [norm, delta_norm, jerk] = 3 dims
        cb_stats_dim = 3
        gru_input_size = d * d + cb_stats_dim

        # Input projection to handle dimension mismatch
        self.input_proj = nn.Linear(gru_input_size, config.hidden_dim)

        # 2-layer GRU with hidden_dim=128
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
        )

        self.family = getattr(config, "connection_family", "orthogonal")
        n_skew_params = d * (d - 1) // 2
        if self.family == "unconstrained":
            self.dense_proj = nn.Linear(config.hidden_dim, d * d)
            nn.init.zeros_(self.dense_proj.weight)
            nn.init.zeros_(self.dense_proj.bias)
            self.skew_proj = None
        else:
            self.dense_proj = None
            self.skew_proj = nn.Linear(config.hidden_dim, n_skew_params)

        self.h_state: Optional[Tensor] = None

    def forward(
        self,
        current_rho: Tensor,
        cb_stats: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute new restriction map candidate via Cayley map.

        Args:
            current_rho: Current restriction maps
                [num_types, d, d].
            cb_stats: Coboundary statistics [3] containing
                [norm, delta_norm, jerk].

        Returns:
            Tuple of (delta_rho [d, d], gru_hidden [hidden_dim]).
        """
        d = self.config.stalk_dim

        # Flatten current rho and concatenate with cb_stats
        # Detach carried rho so prior timesteps aren't in this graph
        rho_flat = current_rho[0].detach().reshape(1, -1)  # [1, d*d]
        cb_expanded = cb_stats.detach().reshape(1, -1)  # [1, 3]
        gru_input = torch.cat([rho_flat, cb_expanded], dim=-1)

        # Project to GRU input dimension
        gru_input = self.input_proj(gru_input)  # [1, hidden_dim]
        gru_input = gru_input.unsqueeze(0)  # [1, 1, hidden_dim]

        # GRU forward pass — carry only a detached hidden state
        h_in = None if self.h_state is None else self.h_state.detach()
        gru_out, h_new = self.gru(gru_input, h_in)
        self.h_state = h_new.detach()

        hidden = gru_out.squeeze(0).squeeze(0)  # [hidden_dim]

        if self.family == "unconstrained":
            delta_rho = self.dense_proj(hidden).view(d, d)
        else:
            skew_params = self.skew_proj(hidden)
            delta_rho = self._cayley_map(skew_params)

        return delta_rho, hidden

    def _cayley_map(self, skew_params: Tensor) -> Tensor:
        """Apply Cayley map: R = (I - A)(I + A)^{-1}.

        Args:
            skew_params: Parameters for skew-symmetric matrix,
                shape [d*(d-1)//2].

        Returns:
            Orthogonal matrix [d, d].
        """
        d = self.config.stalk_dim
        device = skew_params.device
        dtype = skew_params.dtype

        # Build skew-symmetric matrix A
        A = torch.zeros(d, d, device=device, dtype=dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                A[i, j] = skew_params[idx]
                A[j, i] = -skew_params[idx]
                idx += 1

        # Cayley transform: R = (I - A)(I + A)^{-1}
        eye = torch.eye(d, device=device, dtype=dtype)
        R = torch.linalg.solve(eye + A, eye - A)

        return R

    def reset(self) -> None:
        """Reset GRU hidden state for new sequence."""
        self.h_state = None

    def set_hidden(self, h0: Tensor) -> None:
        """Set initial hidden state (for warm-start).

        Args:
            h0: Initial hidden state
                [gru_layers, 1, hidden_dim].
        """
        self.h_state = h0.detach()


class FastRotationEncoder(nn.Module):
    """2-layer GNN estimating ρ_fast from the current snapshot (G_t, X_t).

    Immune to stale GRU history: maps graph embedding → so(d) → Cayley → SO(d)
    per edge type.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stalk_dim: int,
        num_edge_types: int,
    ) -> None:
        """Initialize fast rotation encoder.

        Args:
            input_dim: Node feature dimension.
            hidden_dim: Message-passing hidden size.
            stalk_dim: Sheaf stalk dimension d.
            num_edge_types: Number of edge types E.
        """
        super().__init__()
        self.stalk_dim = stalk_dim
        self.num_edge_types = num_edge_types
        self.lie_dim = stalk_dim * (stalk_dim - 1) // 2

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.lie_dim * num_edge_types)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Estimate ρ_fast and a graph embedding from one snapshot.

        Args:
            features: Node features [N, input_dim].
            edge_index: Edges [2, E_edges].
            edge_types: Edge type ids [E_edges] (unused; global pool).

        Returns:
            (rho_fast [num_types, d, d], graph_embedding [hidden]).
        """
        del edge_types  # global pool; types share one embedding
        device = features.device
        num_nodes = features.size(0)

        h = F.relu(self.lin1(features))
        src, dst = edge_index[0], edge_index[1]
        if src.numel() > 0:
            msg = h[src]
            agg = torch.zeros(num_nodes, h.size(1), device=device)
            counts = torch.zeros(num_nodes, 1, device=device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            counts.scatter_add_(
                0,
                dst.unsqueeze(1),
                torch.ones(src.size(0), 1, device=device),
            )
            agg = agg / counts.clamp(min=1.0)
        else:
            agg = torch.zeros_like(h)

        h2 = F.relu(self.lin2(torch.cat([h, agg], dim=-1)))
        graph_emb = h2.mean(dim=0)
        lie_coeffs = self.out_proj(graph_emb).reshape(
            self.num_edge_types, self.lie_dim
        )
        rho_fast = self._lie_to_rotation(lie_coeffs, device)
        return rho_fast, graph_emb

    def _lie_to_rotation(self, coeffs: Tensor, device: torch.device) -> Tensor:
        """Map so(d) coefficients to SO(d) via Cayley.

        Args:
            coeffs: Lie coefficients [E, lie_dim].
            device: Torch device.

        Returns:
            Rotation matrices [E, d, d].
        """
        d = self.stalk_dim
        e = self.num_edge_types
        skew = torch.zeros(e, d, d, device=device, dtype=coeffs.dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                skew[:, i, j] = coeffs[:, idx]
                skew[:, j, i] = -coeffs[:, idx]
                idx += 1
        skew = skew * 0.1
        eye = torch.eye(d, device=device, dtype=coeffs.dtype).unsqueeze(0).expand(
            e, -1, -1
        )
        return torch.linalg.solve(eye - skew, eye + skew)


class ChangeDetector(nn.Module):
    """Cosine-distance gate between current and EMA graph embeddings.

    g≈0 trusts ρ_slow; g≈1 trusts ρ_fast after a regime shift.
    """

    def __init__(
        self,
        embedding_dim: int,
        ema_alpha: float = DEFAULT_GATE_EMA_ALPHA,
    ) -> None:
        """Initialize change detector.

        Args:
            embedding_dim: Graph embedding dimension.
            ema_alpha: EMA weight on previous embedding.
        """
        super().__init__()
        self.ema_alpha = ema_alpha
        self.gate_proj = nn.Linear(1, 1)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.constant_(self.gate_proj.weight, 5.0)
        self.prev_embedding: Optional[Tensor] = None

    def reset(self) -> None:
        """Clear embedding history for a new sequence."""
        self.prev_embedding = None

    def forward(self, current_embedding: Tensor) -> Tensor:
        """Compute fusion gate in [0, 1].

        Args:
            current_embedding: Current graph embedding [hidden].

        Returns:
            Scalar gate tensor.
        """
        device = current_embedding.device
        if self.prev_embedding is None:
            self.prev_embedding = current_embedding.detach().clone()
            return torch.zeros((), device=device, dtype=current_embedding.dtype)

        cos_sim = F.cosine_similarity(
            current_embedding.unsqueeze(0),
            self.prev_embedding.to(device).unsqueeze(0),
            dim=1,
        )
        cos_dist = (1.0 - cos_sim).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_proj(cos_dist)).squeeze()
        self.prev_embedding = (
            self.ema_alpha * self.prev_embedding
            + (1.0 - self.ema_alpha) * current_embedding.detach()
        )
        return gate


class SheafDiffusionLayer(nn.Module):
    """Sheaf diffusion layer with skip connection.

    Implements: X^{l+1} = LayerNorm(sigma(W * (I - L_F)X^l) + X^l)

    Attributes:
        linear: Learnable weight matrix.
        norm: Layer normalization.
        dropout: Dropout layer.
    """

    def __init__(
        self,
        stalk_dim: int,
        dropout: float = DEFAULT_DROPOUT,
        family: str = "orthogonal",
    ) -> None:
        """Initialize diffusion layer.

        Args:
            stalk_dim: Stalk dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.family = family
        self.linear = nn.Linear(stalk_dim, stalk_dim)
        self.norm = nn.LayerNorm(stalk_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        rho: Tensor,
    ) -> Tensor:
        """Apply one step of sheaf diffusion with skip connection.

        Args:
            x: Node features [num_nodes, stalk_dim].
            edge_index: Edge indices [2, num_edges].
            rho: Current restriction maps [num_types, d, d].

        Returns:
            Updated node features [num_nodes, stalk_dim].
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Message passing via restriction maps
        rho_mat = rho[0]
        messages_src = torch.matmul(x[src], rho_mat)
        if self.family == "unconstrained":
            messages_dst = x[dst]
        else:
            messages_dst = torch.matmul(x[dst], rho_mat.T)

        # Aggregate messages at each node
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages_src)
        agg.index_add_(0, src, messages_dst)

        # Degree normalization
        deg = torch.zeros(num_nodes, device=x.device)
        ones = torch.ones(src.size(0), device=x.device)
        deg.index_add_(0, src, ones)
        deg.index_add_(0, dst, ones)
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        # Diffused signal: (I - L_F)x
        diffused = x - agg / deg

        # Linear + activation + dropout + skip connection
        out = self.linear(diffused)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.norm(out + x)

        return out


class TemporalSheafV35(nn.Module):
    """Temporal Sheaf Network V3.5 with full temporal machinery.

    Architecture overview:
        1. Input projection: node features -> stalk space
        2. Coboundary computation for current state
        3. Jerk detection on coboundary norm time series
        4. GRU-based restriction map evolution (Cayley param) → ρ_slow
        5. Adaptive gating with recovery window (residual slow update)
        6. FastRotationEncoder → ρ_fast from (G_t, X_t)
        7. ChangeDetector gate + geodesic fusion on SO(d)
        8. Sheaf diffusion stack
        9. Edge-level anomaly scoring MLP

    Attributes:
        config: Network configuration.
        input_proj: Projects node features to stalk dimension.
        coboundary: Coboundary operator for norm computation.
        rho_evolver: GRU + Cayley map for rho evolution.
        gate: Adaptive update gate.
        jerk_detector: 2nd-derivative jump detector.
        recovery: Linear-decay recovery controller.
        fast_encoder: Snapshot ρ estimator (optional).
        change_detector: Cosine-distance fusion gate (optional).
        diffusion_layers: Stack of sheaf diffusion layers.
        anomaly_head: MLP for edge anomaly scoring.
    """

    def __init__(self, config: SheafConfig) -> None:
        """Initialize temporal sheaf V3.5 model.

        Args:
            config: Sheaf network configuration.
        """
        super().__init__()
        self.config = config
        d = config.stalk_dim

        # Input projection
        self.input_proj = nn.Linear(config.node_feat_dim, d)

        # Coboundary operator
        self.coboundary = CoboundaryOperator(d)
        self.coboundary.family = getattr(config, "connection_family", "orthogonal")

        self.static_skew: Optional[nn.Parameter] = None
        self.static_matrix: Optional[nn.Parameter] = None
        self.rho_evolver: Optional[RhoEvolver] = None
        if config.static_connection:
            if config.connection_family == "unconstrained":
                self.static_matrix = nn.Parameter(torch.eye(d))
            else:
                self.static_skew = nn.Parameter(
                    torch.zeros(d * (d - 1) // 2)
                )
        else:
            self.rho_evolver = RhoEvolver(config)

        # Adaptive gate (bias=+2.0, floor=0.1)
        gate_input_dim = config.hidden_dim + 1  # hidden + cb_norm
        self.gate = AdaptiveGate(
            input_dim=gate_input_dim,
            hidden_dim=64,
            bias_init=config.gate_bias_init,
            floor=config.gate_floor,
        )

        # Jump detection and recovery
        self.jerk_detector = JerkDetector(
            ema_alpha=config.jerk_ema_alpha,
            threshold=config.jerk_threshold,
        )
        self.recovery = LinearDecayRecovery(
            max_steps=config.recovery_steps,
        )

        # Snapshot ρ_fast path (expert Point #7)
        self.fast_encoder: Optional[FastRotationEncoder] = None
        self.change_detector: Optional[ChangeDetector] = None
        if config.use_fast_encoder:
            self.fast_encoder = FastRotationEncoder(
                input_dim=config.node_feat_dim,
                hidden_dim=config.fast_encoder_hidden,
                stalk_dim=d,
                num_edge_types=config.num_edge_types,
            )
            self.change_detector = ChangeDetector(
                embedding_dim=config.fast_encoder_hidden,
                ema_alpha=config.gate_ema_alpha,
            )

        # Sheaf diffusion layers
        family = getattr(config, "connection_family", "orthogonal")
        self.diffusion_layers = nn.ModuleList(
            [
                SheafDiffusionLayer(d, config.dropout, family=family)
                for _ in range(config.num_diffusion_layers)
            ]
        )

        # Edge anomaly scoring head (optional pair residual from IT-GLLS)
        extra = max(int(config.pair_residual_dim), 0)
        self.anomaly_head = nn.Sequential(
            nn.Linear(2 * d + extra, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Restriction map state buffer
        self.register_buffer(
            "current_rho",
            torch.eye(d).unsqueeze(0).repeat(config.num_edge_types, 1, 1),
        )

        # Track previous coboundary norm for stats
        self._prev_cb_norm: float = 0.0
        self._prev_delta: float = 0.0
        self._last_fusion_gate: Optional[Tensor] = None
        self._last_rho_fast: Optional[Tensor] = None
        self._last_fast_cb: Optional[Tensor] = None

    def reset_state(self) -> None:
        """Reset all temporal state for a new sequence."""
        d = self.config.stalk_dim
        device = self.current_rho.device

        self.current_rho = (
            torch.eye(d, device=device)
            .unsqueeze(0)
            .repeat(self.config.num_edge_types, 1, 1)
        )
        if self.rho_evolver is not None:
            self.rho_evolver.reset()

    def _materialize_static_rho(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """One learned map, identical at every snapshot."""
        d = self.config.stalk_dim
        if self.static_matrix is not None:
            return self.static_matrix.to(device=device, dtype=dtype).unsqueeze(0)
        skew = self.static_skew.to(device=device, dtype=dtype)
        A = torch.zeros(d, d, device=device, dtype=dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                A[i, j] = skew[idx]
                A[j, i] = -skew[idx]
                idx += 1
        eye = torch.eye(d, device=device, dtype=dtype)
        rho = torch.linalg.solve(eye + A, eye - A)
        return rho.unsqueeze(0)

    def _score_after_rho(
        self,
        x_stalk: Tensor,
        edge_index: Tensor,
        rho_for_diff: Tensor,
        pair_residual: Optional[Tensor],
        diagnostics: Dict[str, float],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Diffusion + edge head given a connection for this snapshot."""
        device = x_stalk.device
        x_out = x_stalk
        for layer in self.diffusion_layers:
            x_out = layer(x_out, edge_index, rho_for_diff)
        src, dst = edge_index[0], edge_index[1]
        parts = [x_out[src], x_out[dst]]
        extra = max(int(self.config.pair_residual_dim), 0)
        if extra > 0:
            n_e = edge_index.size(1)
            if pair_residual is None:
                residual = torch.zeros(n_e, extra, device=device, dtype=x_out.dtype)
            else:
                residual = pair_residual.to(device=device, dtype=x_out.dtype)
                if residual.dim() == 1:
                    residual = residual.unsqueeze(-1)
                if residual.size(0) != n_e:
                    residual = residual[:n_e]
                if residual.size(-1) < extra:
                    pad = torch.zeros(
                        n_e, extra - residual.size(-1),
                        device=device, dtype=x_out.dtype,
                    )
                    residual = torch.cat([residual, pad], dim=-1)
                residual = residual[:, :extra]
            parts.append(residual)
            diagnostics["pair_residual_mean"] = float(residual.mean().item())
        edge_repr = torch.cat(parts, dim=-1)
        scores = self.anomaly_head(edge_repr).squeeze(-1)
        return scores, diagnostics
        self.jerk_detector.reset()
        self.recovery.reset()
        if self.change_detector is not None:
            self.change_detector.reset()
        self._prev_cb_norm = 0.0
        self._prev_delta = 0.0
        self._last_fusion_gate = None
        self._last_rho_fast = None
        self._last_fast_cb = None

    def _reinit_slow_from(self, rho: Tensor) -> None:
        """Hard handoff: carry ρ_fast and clear GRU contamination.

        Args:
            rho: Restriction maps [E, d, d] (typically ρ_fast).
        """
        rho = rho.detach()
        if rho.dim() == 2:
            rho = rho.unsqueeze(0)
        if rho.shape[0] == 1 and self.config.num_edge_types > 1:
            rho = rho.expand(self.config.num_edge_types, -1, -1).contiguous()
        elif rho.shape != self.current_rho.shape:
            n = min(rho.shape[0], self.current_rho.shape[0])
            d = min(rho.shape[-1], self.current_rho.shape[-1])
            carried = torch.eye(
                self.config.stalk_dim,
                device=rho.device,
                dtype=rho.dtype,
            ).unsqueeze(0).repeat(self.config.num_edge_types, 1, 1)
            carried[:n, :d, :d] = rho[:n, :d, :d]
            rho = carried
        self.current_rho = rho.contiguous()

        if self.rho_evolver is None:
            self._prev_cb_norm = 0.0
            self._prev_delta = 0.0
            self.jerk_detector.reset()
            self.recovery.reset()
            return
        self.rho_evolver.reset()
        d = self.config.stalk_dim
        flat_rho = self.current_rho[0].reshape(1, -1)
        cb_stats_dim = 3
        padded = torch.zeros(
            1,
            d * d + cb_stats_dim,
            device=flat_rho.device,
            dtype=flat_rho.dtype,
        )
        padded[:, : d * d] = flat_rho
        with torch.no_grad():
            projected = self.rho_evolver.input_proj(padded)
            h0 = projected.unsqueeze(0).repeat(self.config.gru_layers, 1, 1)
        self.rho_evolver.set_hidden(h0)
        self._prev_cb_norm = 0.0
        self._prev_delta = 0.0
        self.jerk_detector.reset()
        self.recovery.reset()

    def _cayley_log(self, rotations: Tensor) -> Tensor:
        """Cayley logarithm: SO(d) → so(d).

        Args:
            rotations: Orthogonal maps [E, d, d].

        Returns:
            Skew-symmetric tangent [E, d, d].
        """
        d = rotations.size(-1)
        e = rotations.size(0)
        eye = torch.eye(d, device=rotations.device, dtype=rotations.dtype)
        eye = eye.unsqueeze(0).expand(e, -1, -1)
        skew = torch.linalg.solve(rotations + eye + CAYLEY_EPS * eye, rotations - eye)
        return 0.5 * (skew - skew.transpose(-2, -1))

    def _cayley_exp(self, skew: Tensor) -> Tensor:
        """Cayley exponential: so(d) → SO(d).

        Args:
            skew: Skew-symmetric maps [E, d, d].

        Returns:
            Orthogonal maps [E, d, d].
        """
        d = skew.size(-1)
        e = skew.size(0)
        eye = torch.eye(d, device=skew.device, dtype=skew.dtype)
        eye = eye.unsqueeze(0).expand(e, -1, -1)
        return torch.linalg.solve(eye - skew, eye + skew)

    def _geodesic_fuse(
        self,
        rho_slow: Tensor,
        rho_fast: Tensor,
        gate: Tensor,
    ) -> Tensor:
        """Blend ρ_slow and ρ_fast along the SO(d) geodesic.

        ρ = ρ_slow · Cayley(g · log(ρ_slow^T ρ_fast))
        g=0 → ρ_slow; g=1 → ≈ ρ_fast.

        Args:
            rho_slow: Slow GRU maps [E, d, d].
            rho_fast: Fast encoder maps [E, d, d].
            gate: Scalar fusion gate in [0, 1].

        Returns:
            Fused restriction maps [E, d, d].
        """
        relative = torch.bmm(rho_slow.transpose(-2, -1), rho_fast)
        delta = self._cayley_log(relative)
        correction = self._cayley_exp(gate * delta)
        return torch.bmm(rho_slow, correction)

    def warm_start(self, static_rho: Tensor) -> None:
        """Initialize from pretrained static restriction maps.

        Projects static rho into GRU hidden state h0.

        Args:
            static_rho: Pretrained maps [num_types, d, d] or [d, d].
        """
        rho = static_rho.detach()
        if rho.dim() == 2:
            rho = rho.unsqueeze(0)
        if rho.shape[0] == 1 and self.config.num_edge_types > 1:
            rho = rho.repeat(self.config.num_edge_types, 1, 1)
        if rho.shape != self.current_rho.shape:
            # Best-effort: take leading types / stalk dims that fit
            n = min(rho.shape[0], self.current_rho.shape[0])
            d = min(rho.shape[-1], self.current_rho.shape[-1])
            self.current_rho.zero_()
            self.current_rho[:n, :d, :d] = rho[:n, :d, :d]
        else:
            self.current_rho.copy_(rho)

        if self.rho_evolver is None:
            return
        # RhoEvolver.forward uses current_rho[0] only → flatten one map
        d = self.config.stalk_dim
        flat_rho = self.current_rho[0].reshape(1, -1)  # [1, d*d]

        cb_stats_dim = 3
        gru_input_size = d * d + cb_stats_dim
        padded = torch.zeros(
            1, gru_input_size, device=flat_rho.device, dtype=flat_rho.dtype
        )
        padded[:, : d * d] = flat_rho

        with torch.no_grad():
            projected = self.rho_evolver.input_proj(padded)
            h0 = projected.unsqueeze(0).repeat(
                self.config.gru_layers, 1, 1
            )
        self.rho_evolver.set_hidden(h0)

    def forward_timestep(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_types: Optional[Tensor] = None,
        force_gate_open: bool = False,
        force_fusion_open: bool = False,
        jump_active: bool = False,
        hard_handoff: bool = False,
        external_rho: Optional[Tensor] = None,
        pair_residual: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Process one timestep and return edge anomaly scores.

        Args:
            x: Node features [num_nodes, node_feat_dim].
            edge_index: Edge indices [2, num_edges].
            edge_types: Edge type indices [num_edges] (optional).
            force_gate_open: Force residual AdaptiveGate=1.0.
            force_fusion_open: Force fusion gate=1.0 (use ρ_fast).
            jump_active: Ground-truth jump pulse at this step.
            hard_handoff: If True with jump_active, ρ←ρ_fast and reset GRU.
            external_rho: Optional [E, d, d] map that replaces ρ_fast and
                is hard-handoff'd (algebraic snapshot estimator).
            pair_residual: Optional per-edge GL(d) residual [num_edges].

        Returns:
            Tuple of (edge_scores [num_edges], diagnostics dict).
        """
        e_types = self.config.num_edge_types
        device = x.device
        diagnostics: Dict[str, float] = {}

        # Default edge types if not provided
        if edge_types is None:
            edge_types = torch.zeros(
                edge_index.size(1),
                dtype=torch.long,
                device=device,
            )

        # Project input to stalk space
        x_stalk = self.input_proj(x)

        if self.config.static_connection:
            rho_for_diff = self._materialize_static_rho(
                device=device, dtype=x.dtype
            )
            if rho_for_diff.shape[0] == 1 and e_types > 1:
                rho_for_diff = rho_for_diff.expand(e_types, -1, -1).contiguous()
            cb_norm = self.coboundary(x_stalk, edge_index, rho_for_diff)
            diagnostics["coboundary_norm"] = float(cb_norm.detach().item())
            diagnostics["fusion_gate"] = 0.0
            diagnostics["hard_handoff"] = 0.0
            diagnostics["gate_effective"] = 0.0
            diagnostics["jump_detected"] = 0.0
            diagnostics["recovery_strength"] = 0.0
            self._last_fusion_gate = None
            self._last_rho_fast = None
            self._last_fast_cb = None
            self._last_rho_live = rho_for_diff
            self.current_rho = rho_for_diff.detach().contiguous()
            return self._score_after_rho(
                x_stalk, edge_index, rho_for_diff, pair_residual, diagnostics
            )

        # Carried rho must be a leaf for truncated BPTT
        rho_prev = self.current_rho.detach()

        # Compute coboundary norm
        cb_norm = self.coboundary(x_stalk, edge_index, rho_prev)
        cb_norm_val = float(cb_norm.detach().item())
        diagnostics["coboundary_norm"] = cb_norm_val

        # Compute coboundary statistics for GRU input
        delta_norm = cb_norm_val - self._prev_cb_norm
        jerk_val = abs(delta_norm - self._prev_delta)
        self._prev_delta = delta_norm
        self._prev_cb_norm = cb_norm_val

        cb_stats = torch.tensor(
            [cb_norm_val, delta_norm, jerk_val],
            device=device,
            dtype=x.dtype,
        )

        # Jerk detection
        jump_detected = self.jerk_detector(cb_norm.detach())
        diagnostics["jump_detected"] = float(jump_detected.item())

        if float(jump_detected.item()) > 0.5:
            self.recovery.trigger()

        # Recovery override strength
        recovery_strength = self.recovery.step()
        diagnostics["recovery_strength"] = float(recovery_strength.item())

        # Evolve restriction maps via GRU + Cayley (or GL increment)
        delta_rho, gru_hidden = self.rho_evolver(rho_prev, cb_stats)

        # Compute adaptive gate value
        gate_input = torch.cat(
            [gru_hidden, cb_norm.detach().unsqueeze(0)], dim=-1
        )
        gate_val = self.gate(gate_input.unsqueeze(0)).squeeze()
        diagnostics["gate_raw"] = float(gate_val.detach().item())

        # Override residual gate during curriculum or recovery
        if force_gate_open:
            gate_val = torch.ones_like(gate_val)
        elif float(recovery_strength.item()) > 0.0:
            gate_val = torch.max(
                gate_val,
                recovery_strength.to(dtype=gate_val.dtype)
                * torch.ones_like(gate_val),
            )

        diagnostics["gate_effective"] = float(gate_val.detach().item())

        family = getattr(self.config, "connection_family", "orthogonal")
        if family == "unconstrained":
            eye = torch.eye(
                self.config.stalk_dim, device=device, dtype=x.dtype
            )
            slow_step = eye + gate_val * delta_rho
            slow_step = slow_step.unsqueeze(0).expand(e_types, -1, -1)
            rho_slow = torch.bmm(rho_prev, slow_step)
        else:
            # Lie-group slow update: ρ ← ρ · Cayley(g · log(Δρ))
            delta_batch = delta_rho.unsqueeze(0)
            delta_lie = self._cayley_log(delta_batch)
            slow_step = self._cayley_exp(gate_val * delta_lie)
            slow_step = slow_step.expand(e_types, -1, -1)
            rho_slow = torch.bmm(rho_prev, slow_step)

        # Snapshot ρ_fast + change-detector fusion (rescue at jumps)
        fusion_gate = torch.zeros((), device=device, dtype=x.dtype)
        rho_fast: Optional[Tensor] = None
        did_handoff = False
        if external_rho is not None:
            rho_fast = external_rho.to(device=device, dtype=x.dtype)
            if rho_fast.dim() == 2:
                rho_fast = rho_fast.unsqueeze(0)
            if rho_fast.shape[0] == 1 and e_types > 1:
                rho_fast = rho_fast.expand(e_types, -1, -1).contiguous()
            fusion_gate = torch.ones((), device=device, dtype=x.dtype)
            rho_for_diff = rho_fast
            did_handoff = True
            self._last_rho_fast = rho_fast
            self._last_fast_cb = self.coboundary(x_stalk, edge_index, rho_fast)
        elif (
            self.fast_encoder is not None
            and self.change_detector is not None
        ):
            rho_fast, graph_emb = self.fast_encoder(x, edge_index, edge_types)
            # Unsupervised consistency of ρ_fast w.r.t. current features
            self._last_rho_fast = rho_fast
            self._last_fast_cb = self.coboundary(x_stalk, edge_index, rho_fast)

            fusion_gate = self.change_detector(graph_emb)
            if force_fusion_open or force_gate_open or jump_active:
                fusion_gate = torch.ones_like(fusion_gate)
            elif float(recovery_strength.item()) > 0.0:
                fusion_gate = torch.max(
                    fusion_gate,
                    recovery_strength.to(dtype=fusion_gate.dtype),
                )

            if hard_handoff and jump_active:
                # Hard switch: use ρ_fast and clear stale GRU state
                rho_for_diff = rho_fast
                did_handoff = True
            else:
                rho_for_diff = self._geodesic_fuse(
                    rho_slow, rho_fast, fusion_gate
                )
        else:
            rho_for_diff = rho_slow
            self._last_rho_fast = None
            self._last_fast_cb = None

        self._last_fusion_gate = fusion_gate
        diagnostics["fusion_gate"] = float(fusion_gate.detach().item())
        diagnostics["hard_handoff"] = float(did_handoff)
        diagnostics["jump_active"] = float(jump_active)

        # Live rho for this step's regularization / scoring
        self._last_rho_live = rho_for_diff
        if did_handoff and rho_fast is not None:
            self._reinit_slow_from(rho_fast)
        else:
            self.current_rho = rho_for_diff.detach().contiguous()

        return self._score_after_rho(
            x_stalk, edge_index, rho_for_diff, pair_residual, diagnostics
        )

    def score_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_types: Optional[Tensor] = None,
        force_fusion_open: bool = False,
    ) -> Tensor:
        """Score edges for anomaly detection (inference mode).

        Args:
            x: Node features [num_nodes, node_feat_dim].
            edge_index: Edge indices [2, num_edges].
            edge_types: Edge type indices (optional).
            force_fusion_open: Force fusion gate=1 (ρ_fast ceiling).

        Returns:
            Anomaly probabilities for each edge.
        """
        scores, _ = self.forward_timestep(
            x,
            edge_index,
            edge_types,
            force_fusion_open=force_fusion_open,
        )
        return torch.sigmoid(scores)

    def get_current_rho(self) -> Tensor:
        """Return current restriction maps (for regularization).

        Prefers the live (differentiable) maps from the latest forward
        when available so orthogonality/smoothness losses can train.

        Returns:
            Current rho tensor [num_types, d, d].
        """
        live = getattr(self, "_last_rho_live", None)
        if live is not None:
            return live
        return self.current_rho

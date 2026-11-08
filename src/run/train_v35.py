"""Training procedure for Temporal Sheaf V3.5 (routing curriculum).

Three-phase schedule to break fast-encoder / gate collapse:
    Phase 1: Freeze GRU; force fusion g=1; train ρ_fast (+ head).
    Phase 2: Freeze trunks; BCE(gate, jump_indicator).
    Phase 3: End-to-end fine-tune at low LR.

Loss components:
    - BCE with pos_weight=9.0 for class imbalance
    - Orthogonality penalty on restriction maps
    - Temporal smoothness penalty on rho changes
    - Jump-gate BCE (not L1 sparsity — that fights spikes)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from temporal_sheaf_v35 import (
    SheafConfig,
    TrainConfig,
    TemporalSheafV35,
)


PRINT_EVERY = 20


def compute_orthogonality_penalty(rho: Tensor) -> Tensor:
    """Compute penalty for deviation from orthogonality.

    Penalty = ||R^T R - I||_F^2 averaged over edge types.
    """
    eye = torch.eye(rho.size(-1), device=rho.device, dtype=rho.dtype)
    rtR = torch.matmul(rho.transpose(-2, -1), rho)
    return (rtR - eye).pow(2).sum(dim=(-2, -1)).mean()


def compute_smoothness_penalty(
    rho_current: Tensor,
    rho_previous: Optional[Tensor],
) -> Tensor:
    """Temporal smoothness penalty ||rho_t - rho_{t-1}||_F^2."""
    if rho_previous is None:
        return torch.tensor(0.0, device=rho_current.device)
    diff = rho_current - rho_previous
    return diff.pow(2).sum(dim=(-2, -1)).mean()


def extract_static_rho(static_model: nn.Module) -> Tensor:
    """Extract restriction maps from a trained static model."""
    for attr_name in [
        "current_rho",
        "rho",
        "restriction_maps",
        "sheaf_maps",
    ]:
        if hasattr(static_model, attr_name):
            rho = getattr(static_model, attr_name)
            if isinstance(rho, Tensor):
                return rho.detach().clone()
    raise AttributeError(
        "Could not find restriction maps in static model. "
        "Expected one of: current_rho, rho, restriction_maps"
    )


def build_jump_targets(
    num_timesteps: int,
    jump_times: Sequence[float],
    window_after: int = 1,
) -> List[float]:
    """Binary gate targets from normalized jump times in [0, 1].

    Marks the first timestep at/after each jump, plus ``window_after``
    following steps (so the gate learns a short open window).
    """
    targets = [0.0] * num_timesteps
    if num_timesteps <= 0 or not jump_times:
        return targets

    denom = max(num_timesteps - 1, 1)
    for i in range(num_timesteps):
        t = i / denom
        t_prev = (i - 1) / denom if i > 0 else -1.0
        for jt in jump_times:
            crossed = t_prev < float(jt) <= t
            exact = abs(t - float(jt)) < 1e-8
            if crossed or exact:
                for k in range(i, min(i + 1 + window_after, num_timesteps)):
                    targets[k] = 1.0
    return targets


def _set_requires_grad(params: Iterable[nn.Parameter], enabled: bool) -> None:
    for p in params:
        p.requires_grad = enabled


def _module_params(module: Optional[nn.Module]) -> List[nn.Parameter]:
    if module is None:
        return []
    return list(module.parameters())


def _configure_phase_trainable(
    model: TemporalSheafV35,
    phase: int,
) -> None:
    """Freeze / unfreeze modules for the routing curriculum phase."""
    all_params = list(model.parameters())
    _set_requires_grad(all_params, False)

    if phase == 1:
        # Pre-train ρ_fast path (+ scoring); GRU frozen
        trainable = (
            _module_params(model.fast_encoder)
            + _module_params(model.input_proj)
            + _module_params(model.diffusion_layers)
            + _module_params(model.anomaly_head)
        )
        _set_requires_grad(trainable, True)
    elif phase == 2:
        # Gate only
        trainable = (
            _module_params(model.change_detector)
            + _module_params(model.gate)
            + _module_params(model.jerk_detector)
        )
        _set_requires_grad(trainable, True)
    else:
        # End-to-end
        _set_requires_grad(all_params, True)


def _build_optimizer(
    model: TemporalSheafV35,
    config: TrainConfig,
    phase: int,
) -> torch.optim.Optimizer:
    gate_params = (
        list(model.gate.parameters())
        + list(model.jerk_detector.parameters())
        + list(model.recovery.parameters())
        + _module_params(model.change_detector)
    )
    gate_ids = {id(p) for p in gate_params}
    main_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids
    ]
    gate_trainable = [p for p in gate_params if p.requires_grad]

    lr = (
        config.finetune_learning_rate
        if phase == 3
        else config.learning_rate
    )
    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": lr})
    if gate_trainable:
        g_lr = config.gate_learning_rate if phase == 2 else lr
        param_groups.append({"params": gate_trainable, "lr": g_lr})
    if not param_groups:
        # Fallback (should not happen)
        param_groups = [{"params": list(model.parameters()), "lr": lr}]
    return torch.optim.Adam(param_groups)


def train_temporal_v35(
    model: "TemporalSheafV35",
    static_model: Optional[nn.Module],
    dataset: List[List[Dict[str, Tensor]]],
    config: "TrainConfig",
    sheaf_config: "SheafConfig",
    train_timesteps: List[int],
    jump_times: Optional[Sequence[float]] = None,
) -> "TemporalSheafV35":
    """Train temporal sheaf V3.5 with routing curriculum.

    Args:
        model: Temporal sheaf V3.5 model to train.
        static_model: Pretrained static model for warm-start.
        dataset: List of snapshot sequences.
        config: Training configuration.
        sheaf_config: Sheaf architecture configuration.
        train_timesteps: Holdout train indices (t < 0.5).
        jump_times: Normalized jump times in [0, 1] for gate BCE.

    Returns:
        Trained model.
    """
    device = next(model.parameters()).device
    jump_times = list(jump_times or [])

    if static_model is not None:
        try:
            static_rho = extract_static_rho(static_model).to(device)
            model.warm_start(static_rho)
            print("  [warm-start] Initialized from static rho")
        except AttributeError as e:
            print(f"  [warm-start] Skipped: {e}")

    pos_weight = torch.tensor(sheaf_config.pos_weight, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    if config.use_routing_curriculum:
        phase_epochs = [
            (1, config.epochs_phase1),
            (2, config.epochs_phase2),
            (3, config.epochs_phase3),
        ]
    else:
        # Legacy: force fusion open then normal train
        phase_epochs = [
            (1, config.epochs_phase1),
            (3, config.epochs_phase2),
        ]

    for phase, n_epochs in phase_epochs:
        # Phase 2 trains only the jump gate. With no jump supervision it would
        # produce no differentiable loss, so skip it explicitly.
        if phase == 2 and config.jump_gate_bce_weight <= 0.0:
            continue
        if n_epochs <= 0:
            continue

        _configure_phase_trainable(model, phase)
        optimizer = _build_optimizer(model, config, phase)
        scheduler = None
        if phase == 3 and n_epochs > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs
            )

        # Multi-jump: oversample in Phase 1 so later regimes are seen more
        phase_dataset = list(dataset)
        if phase == 1 and len(jump_times) >= 2:
            phase_dataset = phase_dataset + list(dataset) + list(dataset)

        print(
            f"  [curriculum] Phase {phase}: {n_epochs} epochs "
            f"(jump_times={jump_times}, "
            f"window={config.jump_gate_window}, "
            f"seqs={len(phase_dataset)})"
        )

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for sequence in phase_dataset:
                model.reset_state()
                rho_previous: Optional[Tensor] = None
                jump_targets = build_jump_targets(
                    len(sequence),
                    jump_times,
                    window_after=config.jump_gate_window,
                )

                # Full rollout every phase (phase-1 uses jump labels for
                # ρ_fast; phase-3 task loss is masked to train_timesteps).
                t_iter = (
                    list(train_timesteps)
                    if config.train_window_only
                    else list(range(len(sequence)))
                )
                for t in t_iter:
                    snap = sequence[t]
                    x = snap["x"].to(device)
                    edge_index = snap["edge_index"].to(device)
                    edge_types = snap.get("edge_types")
                    if edge_types is not None:
                        edge_types = edge_types.to(device)
                    labels = snap["labels"].to(device)

                    jump_active = (
                        jump_targets[t] > 0.5
                        and config.jump_gate_bce_weight > 0.0
                    )
                    force_fusion = phase == 1
                    pair_residual = snap.get("pair_residual")
                    if pair_residual is not None:
                        pair_residual = pair_residual.to(device)
                    scores, _ = model.forward_timestep(
                        x,
                        edge_index,
                        edge_types,
                        force_gate_open=False,
                        force_fusion_open=force_fusion,
                        jump_active=jump_active,
                        hard_handoff=(
                            config.hard_handoff_on_jump and jump_active
                        ),
                        pair_residual=pair_residual,
                    )

                    total_loss = torch.zeros((), device=device)
                    # Task loss never uses post-split snapshots unless
                    # train_window_only is False (oracle-routing diagnostic).
                    use_task = t in train_timesteps or (
                        phase == 1 and not config.train_window_only
                    )
                    if use_task:
                        total_loss = total_loss + loss_fn(scores, labels)

                    rho_current = model.get_current_rho()
                    if use_task:
                        if config.ortho_penalty > 0:
                            total_loss = total_loss + (
                                config.ortho_penalty
                                * compute_orthogonality_penalty(rho_current)
                            )
                        if config.smooth_penalty > 0:
                            total_loss = total_loss + (
                                config.smooth_penalty
                                * compute_smoothness_penalty(
                                    rho_current, rho_previous
                                )
                            )
                    rho_previous = rho_current.detach().clone()

                    # Unsupervised ρ_fast consistency (phases 1 & 3)
                    if (
                        phase in (1, 3)
                        and config.fast_cb_penalty > 0.0
                        and model._last_fast_cb is not None
                    ):
                        total_loss = total_loss + (
                            config.fast_cb_penalty * model._last_fast_cb
                        )

                    # Jump-gate BCE (phases 2–3)
                    if (
                        phase >= 2
                        and config.jump_gate_bce_weight > 0.0
                        and model._last_fusion_gate is not None
                        and model.change_detector is not None
                    ):
                        target = torch.tensor(
                            jump_targets[t],
                            device=device,
                            dtype=model._last_fusion_gate.dtype,
                        )
                        # Jumps are rare — upweight positives
                        gate_bce = F.binary_cross_entropy(
                            model._last_fusion_gate.clamp(1e-4, 1 - 1e-4),
                            target,
                            weight=(
                                torch.tensor(5.0, device=device)
                                if jump_active
                                else None
                            ),
                        )
                        total_loss = (
                            total_loss
                            + config.jump_gate_bce_weight * gate_bce
                        )

                    if (
                        phase == 3
                        and config.fusion_gate_penalty > 0.0
                        and model._last_fusion_gate is not None
                        and not jump_active
                    ):
                        total_loss = total_loss + (
                            config.fusion_gate_penalty
                            * model._last_fusion_gate.mean()
                        )

                    if not total_loss.requires_grad:
                        continue

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.grad_clip_norm,
                    )
                    optimizer.step()

                    epoch_loss += float(total_loss.detach().item())
                    num_batches += 1

            if scheduler is not None:
                scheduler.step()

            if epoch % PRINT_EVERY == 0 or epoch == 1:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(
                    f"    [P{phase}] Epoch {epoch:3d}/{n_epochs}, "
                    f"Loss: {avg_loss:.4f}"
                )

    # Ensure all params trainable for any further use
    _set_requires_grad(list(model.parameters()), True)
    return model


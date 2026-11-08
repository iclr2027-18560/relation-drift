import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Try importing from Week 4 files; if unavailable, use stubs
try:
    from graph_generator import GraphConfig, GraphGenerator
except ImportError:
    pass

try:
    from restriction_map_generator import MapConfig, RestrictionMapGenerator
except ImportError:
    pass

try:
    from temporal_evolution import JumpProcess, TemporalEvolution
except ImportError:
    pass


# ============================================================================
# FILE 1: noise_injector.py
#
# Purpose: Adds realistic noise and variable supervision to the clean
# synthetic data from Week 4.
# ============================================================================


@dataclass
class NoiseConfig:
    """Configuration for noise injection parameters.

    Attributes:
        label_noise_rate: Probability epsilon of flipping a label
            (0 = no noise).
        label_noise_type: Either "symmetric" (equal flip rates) or
            "asymmetric" (false negatives 3x more likely than false
            positives).
        feature_noise_snr: Signal-to-noise ratio in dB; inf means no
            noise added.
        supervision_fraction: Fraction of labels visible during training
            (1.0 = all).
        seed: Random seed for reproducibility.
    """

    label_noise_rate: float = 0.0
    label_noise_type: str = "symmetric"
    feature_noise_snr: float = float("inf")
    supervision_fraction: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        assert 0.0 <= self.label_noise_rate <= 0.5, (
            f"label_noise_rate must be in [0, 0.5], got {self.label_noise_rate}"
        )
        assert self.label_noise_type in ("symmetric", "asymmetric"), (
            f"label_noise_type must be 'symmetric' or 'asymmetric', "
            f"got {self.label_noise_type}"
        )
        assert self.feature_noise_snr > 0, (
            f"feature_noise_snr must be positive, got {self.feature_noise_snr}"
        )
        assert 0.0 < self.supervision_fraction <= 1.0, (
            f"supervision_fraction must be in (0, 1], "
            f"got {self.supervision_fraction}"
        )


class NoiseInjector:
    """Applies realistic noise to labels and features for benchmarking.

    Supports three noise modalities:
      1. Label noise (symmetric or asymmetric flipping)
      2. Feature noise (additive Gaussian calibrated by SNR)
      3. Supervision masking (hiding a fraction of labels)
    """

    def __init__(self, config: NoiseConfig) -> None:
        """Initialize the noise injector with given configuration.

        Args:
            config: NoiseConfig specifying noise parameters.
        """
        self.config = config
        self.rng = torch.Generator()
        self.rng.manual_seed(config.seed)

    def apply_label_noise(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply label noise by flipping labels with probability epsilon.

        Symmetric noise: both 0->1 and 1->0 flip with probability epsilon.
        Asymmetric noise: anomalous->normal flip rate = 3*epsilon,
                          normal->anomalous flip rate = epsilon.

        Args:
            labels: Binary tensor of shape (N,) with values in {0, 1}.
                Convention: 0 = normal, 1 = anomalous.

        Returns:
            Tuple of (noisy_labels, noise_mask) where noise_mask is True
            for positions that were flipped.
        """
        epsilon = self.config.label_noise_rate
        noisy_labels = labels.clone()
        noise_mask = torch.zeros_like(labels, dtype=torch.bool)

        if epsilon == 0.0:
            return noisy_labels, noise_mask

        n = labels.shape[0]
        flip_probs = torch.rand(n, generator=self.rng)

        if self.config.label_noise_type == "symmetric":
            # Flip each label with uniform probability epsilon
            flip_indices = flip_probs < epsilon
            noise_mask = flip_indices
            noisy_labels[flip_indices] = 1 - noisy_labels[flip_indices]

        elif self.config.label_noise_type == "asymmetric":
            # Anomalous -> Normal: flip with probability 3*epsilon (capped)
            # Normal -> Anomalous: flip with probability epsilon
            anomalous_mask = labels == 1
            normal_mask = labels == 0

            # False negatives (anomalous -> normal): 3x more likely
            fn_rate = min(3.0 * epsilon, 1.0)
            fn_flips = anomalous_mask & (flip_probs < fn_rate)

            # False positives (normal -> anomalous): base rate
            fp_flips = normal_mask & (flip_probs < epsilon)

            noise_mask = fn_flips | fp_flips
            noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]

        return noisy_labels, noise_mask

    def apply_feature_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features calibrated by signal-to-noise ratio.

        Computes noise standard deviation as:
            sigma = sqrt(P_signal / 10^(SNR/10))
        where P_signal = mean(features^2).

        Args:
            features: Tensor of shape (N, D) containing node features.

        Returns:
            Noisy features tensor of the same shape.
        """
        if self.config.feature_noise_snr == float("inf"):
            return features.clone()

        # Compute signal power
        p_signal = torch.mean(features ** 2)

        # Compute noise standard deviation from SNR in dB
        snr_linear = 10.0 ** (self.config.feature_noise_snr / 10.0)
        sigma = torch.sqrt(p_signal / snr_linear)

        # Generate and add Gaussian noise
        noise = torch.randn(features.shape, generator=self.rng) * sigma
        noisy_features = features + noise

        return noisy_features

    def apply_supervision_mask(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask a fraction of labels as unobserved.

        Args:
            labels: Tensor of shape (N,) with label values.

        Returns:
            Tuple of (masked_labels, supervision_mask) where:
                - masked_labels has -1 for unobserved positions
                - supervision_mask is True for observed positions
        """
        n = labels.shape[0]
        # Determine which labels are observed
        observe_probs = torch.rand(n, generator=self.rng)
        supervision_mask = observe_probs < self.config.supervision_fraction

        # Create masked labels
        masked_labels = labels.clone()
        masked_labels[~supervision_mask] = -1

        return masked_labels, supervision_mask

    def apply_all(
        self, labels: torch.Tensor, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Apply all noise modalities in sequence.

        Args:
            labels: Binary tensor of shape (N,) with values in {0, 1}.
            features: Tensor of shape (N, D) containing node features.

        Returns:
            Dictionary containing:
                - 'noisy_labels': labels after label noise
                - 'noise_mask': which labels were flipped
                - 'noisy_features': features after Gaussian noise
                - 'masked_labels': labels after supervision masking (with -1)
                - 'supervision_mask': which labels are observed
                - 'clean_labels': original labels for reference
                - 'clean_features': original features for reference
        """
        # Apply label noise
        noisy_labels, noise_mask = self.apply_label_noise(labels)

        # Apply feature noise
        noisy_features = self.apply_feature_noise(features)

        # Apply supervision mask to the noisy labels
        masked_labels, supervision_mask = self.apply_supervision_mask(
            noisy_labels
        )

        return {
            "noisy_labels": noisy_labels,
            "noise_mask": noise_mask,
            "noisy_features": noisy_features,
            "masked_labels": masked_labels,
            "supervision_mask": supervision_mask,
            "clean_labels": labels.clone(),
            "clean_features": features.clone(),
        }


# ----------------------------------------------------------------------------
# Minimal stubs for standalone testing of noise_injector.py
# ----------------------------------------------------------------------------


def _generate_synthetic_data_for_noise_test(
    num_nodes: int = 1000,
    feature_dim: int = 16,
    anomaly_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple synthetic labels and features for noise testing.

    Normal nodes have features drawn from N(0, 1).
    Anomalous nodes have features drawn from N(2, 1) (shifted mean).

    Args:
        num_nodes: Total number of nodes.
        feature_dim: Dimensionality of features.
        anomaly_fraction: Fraction of anomalous nodes.
        seed: Random seed.

    Returns:
        Tuple of (labels, features).
    """
    torch.manual_seed(seed)
    num_anomalies = int(num_nodes * anomaly_fraction)
    num_normal = num_nodes - num_anomalies

    # Labels: 0 = normal, 1 = anomalous
    labels = torch.cat([
        torch.zeros(num_normal, dtype=torch.long),
        torch.ones(num_anomalies, dtype=torch.long),
    ])

    # Features: normal from N(0,1), anomalous from N(2,1)
    normal_features = torch.randn(num_normal, feature_dim)
    anomalous_features = torch.randn(num_anomalies, feature_dim) + 2.0
    features = torch.cat([normal_features, anomalous_features], dim=0)

    # Shuffle
    perm = torch.randperm(num_nodes)
    labels = labels[perm]
    features = features[perm]

    return labels, features


def _compute_violation_separation(
    features: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute a simple violation separation ratio.

    Uses L2 norm of features as a proxy for violation score.
    Returns ratio of mean anomalous score to mean normal score.

    Args:
        features: Node feature tensor of shape (N, D).
        labels: Binary label tensor of shape (N,).

    Returns:
        Separation ratio as a float.
    """
    scores = torch.norm(features, dim=1)
    normal_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]

    if len(normal_scores) == 0 or len(anomalous_scores) == 0:
        return 0.0

    return (anomalous_scores.mean() / normal_scores.mean()).item()


# ----------------------------------------------------------------------------
# Validation for noise_injector.py
# ----------------------------------------------------------------------------


def _validate_noise_injector() -> None:
    """Run validation checks for the NoiseInjector module."""
    print("=" * 70)
    print("VALIDATION: noise_injector.py")
    print("=" * 70)

    labels, features = _generate_synthetic_data_for_noise_test(
        num_nodes=10000, feature_dim=16, anomaly_fraction=0.1, seed=42
    )

    # --- Test 1: Label noise at various epsilon values ---
    print("\n[Test 1] Label noise rate validation:")
    for epsilon in [0.05, 0.1, 0.2, 0.3]:
        config = NoiseConfig(
            label_noise_rate=epsilon, label_noise_type="symmetric", seed=42
        )
        injector = NoiseInjector(config)
        noisy_labels, noise_mask = injector.apply_label_noise(labels)
        actual_flip_rate = noise_mask.float().mean().item()
        print(
            f"  epsilon={epsilon:.2f} -> actual flip rate: "
            f"{actual_flip_rate:.4f} (expected ~{epsilon:.4f})"
        )

    # --- Test 1b: Asymmetric noise ---
    print("\n[Test 1b] Asymmetric label noise validation:")
    config = NoiseConfig(
        label_noise_rate=0.1, label_noise_type="asymmetric", seed=42
    )
    injector = NoiseInjector(config)
    noisy_labels, noise_mask = injector.apply_label_noise(labels)
    anomalous_mask = labels == 1
    normal_mask = labels == 0
    fn_rate = noise_mask[anomalous_mask].float().mean().item()
    fp_rate = noise_mask[normal_mask].float().mean().item()
    print(
        f"  epsilon=0.1, asymmetric -> FN rate: {fn_rate:.4f} "
        f"(expected ~0.30), FP rate: {fp_rate:.4f} (expected ~0.10)"
    )

    # --- Test 2: Feature noise degrades violation separation ---
    print("\n[Test 2] Feature noise impact on violation separation:")
    clean_ratio = _compute_violation_separation(features, labels)
    print(f"  Clean (SNR=inf): separation ratio = {clean_ratio:.4f}")

    for snr_db in [20.0, 10.0, 5.0]:
        config = NoiseConfig(feature_noise_snr=snr_db, seed=42)
        injector = NoiseInjector(config)
        noisy_features = injector.apply_feature_noise(features)
        noisy_ratio = _compute_violation_separation(noisy_features, labels)
        print(
            f"  SNR={snr_db:>4.0f} dB: separation ratio = {noisy_ratio:.4f} "
            f"(degradation: {(1 - noisy_ratio / clean_ratio) * 100:.1f}%)"
        )

    # --- Test 3: Supervision mask hides correct fraction ---
    print("\n[Test 3] Supervision mask validation:")
    for frac in [1.0, 0.5, 0.1, 0.05, 0.01]:
        config = NoiseConfig(supervision_fraction=frac, seed=42)
        injector = NoiseInjector(config)
        masked_labels, sup_mask = injector.apply_supervision_mask(labels)
        actual_frac = sup_mask.float().mean().item()
        num_hidden = (masked_labels == -1).sum().item()
        print(
            f"  fraction={frac:.2f} -> observed: {actual_frac:.4f} "
            f"(expected ~{frac:.4f}), hidden: {num_hidden}/{len(labels)}"
        )

    print("\n" + "=" * 70)
    print("noise_injector.py validation COMPLETE")
    print("=" * 70)

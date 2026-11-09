"""PPG-realistic data augmentations, applied to already per-window-normalized
windows (mean 0, std 1 -- see data/windowing.py). Because windows are already
standardized, an amplitude expressed as "a fraction of window std" is simply
that literal value in this space (e.g. noise_std=0.03 means 3% of one unit of
standardized amplitude), with no extra rescaling needed.

Each function operates on a batch of windows, shape (batch, window_len), and
takes an explicit torch.Generator so augmentation randomness is reproducible
given a fixed seed, independent of the DataLoader's own shuffling generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch


def add_gaussian_noise(x: torch.Tensor, std: float, generator: torch.Generator) -> torch.Tensor:
    """Additive moderate noise: independent Gaussian noise per sample."""
    if std <= 0:
        return x
    noise = torch.empty_like(x)
    noise.normal_(mean=0.0, std=std, generator=generator)
    return x + noise


def scale_amplitude(x: torch.Tensor, scale_range: Tuple[float, float], generator: torch.Generator) -> torch.Tensor:
    """Small amplitude scaling: one random scale factor per window (not per
    sample), simulating e.g. varying sensor-skin contact pressure.
    """
    low, high = scale_range
    if low == high:
        return x * low
    factors = torch.empty(x.shape[0], 1, device=x.device, dtype=x.dtype)
    factors.uniform_(low, high, generator=generator)
    return x * factors


def add_baseline_offset(x: torch.Tensor, offset_range: Tuple[float, float], generator: torch.Generator) -> torch.Tensor:
    """Baseline (DC) offset: one random constant per window, simulating slow
    baseline wander.
    """
    low, high = offset_range
    if low == high and low == 0.0:
        return x
    offsets = torch.empty(x.shape[0], 1, device=x.device, dtype=x.dtype)
    offsets.uniform_(low, high, generator=generator)
    return x + offsets


def apply_dropout(
    x: torch.Tensor, dropout_fraction: float, generator: torch.Generator, fill_value: float = 0.0
) -> torch.Tensor:
    """Sample dropout: zero out (or set to `fill_value`) a random fraction of
    samples per window, independently per window, simulating brief sensor
    dropout / motion-artifact gaps.
    """
    if dropout_fraction <= 0:
        return x
    keep_prob = 1.0 - dropout_fraction
    mask = torch.empty_like(x)
    mask.bernoulli_(p=keep_prob, generator=generator)
    return x * mask + fill_value * (1.0 - mask)


@dataclass
class AugmentationConfig:
    """Parameters for one augmented view. Defaults match PLAN-stage defaults
    (moderate noise, small scale/offset perturbation, a handful of dropped
    samples), expressed directly in the per-window-normalized (std=1) space.
    """

    enabled: bool = False
    noise_std: float = 0.03
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    offset_range: Tuple[float, float] = (-0.1, 0.1)
    dropout_fraction: float = 0.07
    dropout_value: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "AugmentationConfig":
        if not cfg:
            return cls(enabled=False)
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            noise_std=float(cfg.get("noise_std", 0.03)),
            amplitude_scale_range=tuple(cfg.get("amplitude_scale_range", (0.9, 1.1))),
            offset_range=tuple(cfg.get("offset_range", (-0.1, 0.1))),
            dropout_fraction=float(cfg.get("dropout_fraction", 0.07)),
            dropout_value=float(cfg.get("dropout_value", 0.0)),
        )


def augment_batch(x: torch.Tensor, cfg: AugmentationConfig, generator: torch.Generator) -> torch.Tensor:
    """Apply the full augmentation pipeline (noise -> scale -> offset ->
    dropout, in that order) to a batch of windows.

    Order rationale: dropout is applied last so it still zeroes out samples of
    the otherwise-augmented signal (closer to how a real sensor dropout would
    interact with existing noise/motion), rather than being immediately
    overwritten by a later step.

    The heart-rate target is never touched by this function -- augmentations
    operate on the input signal only, consistent with the "target must remain
    unchanged" requirement for paired-consistency training.
    """
    out = x
    out = add_gaussian_noise(out, cfg.noise_std, generator)
    out = scale_amplitude(out, cfg.amplitude_scale_range, generator)
    out = add_baseline_offset(out, cfg.offset_range, generator)
    out = apply_dropout(out, cfg.dropout_fraction, generator, cfg.dropout_value)
    return out

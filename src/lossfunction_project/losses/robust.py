"""Custom robust (Huber-style) regression loss, implemented from scratch.

Standard squared-error loss (MSE) weights every residual quadratically, so a
single large-residual sample -- e.g. a PPG window corrupted by a motion artifact,
or a heart-rate label computed from an unusually noisy ECG segment -- can
dominate the training gradient and pull the model away from a good fit on the
rest of the data. Physiological signals routinely contain this kind of atypical
sample without it being a labeling error: artifacts are an expected property of
ambulatory PPG recordings, not something that can simply be filtered out ahead of
time with full confidence.

A Huber-style loss addresses this by keeping the usual quadratic penalty for
small residuals (so the loss stays smooth and well-behaved near zero, and small
errors are still penalized proportionally to their square, as with MSE), but
switches to a linear penalty once the residual exceeds a threshold `delta`. This
caps the influence any single large-residual sample can have on the gradient: the
gradient magnitude saturates at `delta` instead of growing without bound, which is
exactly the robustness-to-atypical-samples property this loss is meant to provide.
"""
from __future__ import annotations

import torch
from torch import nn


class RobustHuberLoss(nn.Module):
    """Huber-style robust loss, written directly from its definition (not via
    ``nn.HuberLoss``/``nn.SmoothL1Loss``), so the quadratic/linear switch and the
    resulting gradient-capping behaviour are explicit and easy to audit.

    For a residual ``e = pred - target``:

        loss(e) = 0.5 * e^2                        if |e| <= delta
                = delta * (|e| - 0.5 * delta)       if |e| >  delta

    The two branches agree in value and in first derivative at ``|e| = delta``
    (the function is continuously differentiable there), which is what keeps
    training stable across the switch point.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        abs_residual = residual.abs()
        quadratic = 0.5 * residual**2
        linear = self.delta * (abs_residual - 0.5 * self.delta)
        # Elementwise switch: use the quadratic branch below delta, the linear
        # branch above it. abs_residual is always >= 0, so this correctly picks
        # one branch per element, not just per batch.
        elementwise_loss = torch.where(abs_residual <= self.delta, quadratic, linear)

        if self.reduction == "mean":
            return elementwise_loss.mean()
        if self.reduction == "sum":
            return elementwise_loss.sum()
        return elementwise_loss

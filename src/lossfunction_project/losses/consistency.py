"""Custom consistency loss for paired training, implemented from scratch.

Given a model prediction `pred` on an original window and `pred_aug` on an
augmented view of the *same* window (same heart-rate target), the consistency
loss penalizes any difference between the two predictions. This is a
regression setting, so "agreement" is measured directly in prediction space
(a continuous value) rather than via a classification-style agreement measure
(e.g. KL divergence between class probabilities).

Encouraging pred ~ pred_aug (regardless of the true target) is what makes this
a *consistency* term rather than a second supervised loss term: it does not
use the label `y` at all, only the two predictions. Combined with a
supervised regression loss on the original window, it nudges the model toward
representations that are invariant to the augmentations used -- which is the
property this training scheme is meant to test for a measurable effect.
"""
from __future__ import annotations

import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """Mean squared difference between two prediction tensors.

    Written directly rather than by calling `nn.MSELoss(pred, pred_aug)`, so
    the fact that this loss operates on a *pair of predictions* -- not a
    prediction and a label -- is visible at the call site and not easily
    confused with the supervised regression loss used elsewhere in the
    training step.
    """

    def forward(self, pred: torch.Tensor, pred_aug: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - pred_aug) ** 2)

"""Shared 1D CNN backbone used, unchanged, by every experiment config.

Keeping exactly one model definition for baseline, robust-loss, consistency, and
full-method runs is what lets the ablation studies attribute any observed
difference to the loss function / training procedure rather than to a different
model architecture or capacity.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class CNN1D(nn.Module):
    """Small 1D CNN regressor: raw PPG window -> scalar heart-rate prediction.

    Input shape: (batch, 1, window_len). Output shape: (batch,) -- a single scalar
    per window, in the (optionally standardized) target space; see
    evaluation/metrics.py and training/train_loop.py for conversion back to bpm.
    """

    def __init__(
        self,
        conv_channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")
        padding = kernel_size // 2

        layers = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, window_len) -> (batch, 1, window_len)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        return self.head(x).squeeze(-1)  # (batch,)


def build_model(model_cfg: dict) -> CNN1D:
    """Build the shared CNN1D model from a config dict (see configs/*.yaml)."""
    return CNN1D(
        conv_channels=tuple(model_cfg.get("conv_channels", (16, 32, 64))),
        kernel_size=model_cfg.get("kernel_size", 7),
        dropout=model_cfg.get("dropout", 0.1),
    )

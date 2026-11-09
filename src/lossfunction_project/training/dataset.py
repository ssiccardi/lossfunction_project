"""Torch Dataset wrapper around ArrayBundle, plus target standardization helpers.

Target standardization (z-scoring the heart-rate label) is fit on the *training*
split only and applied unchanged to validation/test, to avoid leaking any
information about the held-out subjects' label distribution into preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lossfunction_project.data.splits import ArrayBundle


@dataclass
class TargetScaler:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def fit_target_scaler(train_bundle: ArrayBundle, eps: float = 1e-8) -> TargetScaler:
    mean = float(train_bundle.y.mean())
    std = float(train_bundle.y.std())
    return TargetScaler(mean=mean, std=max(std, eps))


class WindowDataset(Dataset):
    """Wraps an ArrayBundle's (X, y) pair as a torch Dataset; y is optionally
    standardized with a scaler fit on the training split.
    """

    def __init__(self, bundle: ArrayBundle, scaler: TargetScaler | None = None):
        self.X = torch.from_numpy(bundle.X.astype(np.float32))
        y = bundle.y.astype(np.float32)
        if scaler is not None:
            y = scaler.transform(y).astype(np.float32)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(
    bundle: ArrayBundle,
    scaler: TargetScaler | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a DataLoader with a seeded generator, so shuffling is reproducible."""
    dataset = WindowDataset(bundle, scaler)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )

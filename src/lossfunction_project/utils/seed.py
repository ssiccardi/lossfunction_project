"""Reproducibility helpers: seed Python/NumPy/PyTorch random state in one place."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's random, NumPy, and PyTorch (CPU) RNGs.

    This project runs CPU-only, so GPU/cuDNN determinism flags are not relevant.
    Full bit-for-bit determinism for every PyTorch op is not guaranteed (some ops
    have no deterministic CPU implementation), but with a fixed seed and the
    small, simple architecture used here, repeated runs are expected to match
    closely; exact reproduction should always be checked empirically, not assumed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

"""Pairing strategies: how to build the "augmented view" x' paired with each
training window x.

Three strategies are implemented:

- "true" (`TRUE_PAIRING`): x' is an augmented view of x itself -- the only
  strategy used by the core deliverable and its ablations (configs B-E). This
  is what makes the consistency term meaningful: the model is asked to agree
  with itself on two views of the *same* underlying signal and heart rate.

- "random" (`RANDOM_PAIRING`): x' is an augmented view of a *different*,
  randomly chosen window from the same batch -- implemented as a per-batch
  cyclic shift of the batch by a random amount in [1, batch_size-1] (drawn
  fresh for each batch via the supplied generator), so every window is
  guaranteed to be paired with a genuinely different one (no accidental
  self-pairing), while the random choice still varies across batches/runs.
  A negative control: if the consistency term's benefit depends on the
  pairing being correct (same underlying signal/label), pairing with an
  unrelated window should make the "gain" shrink or reverse.

- "wrong" (`WRONG_PAIRING`): the same cyclic-shift mechanism, but with a
  *fixed* shift (not re-drawn per batch), making the mismatch deterministic
  and reproducible rather than randomized -- a stricter/more adversarial
  negative-control variant.

Known edge case: with a batch size of 1, there is no other window to pair
with, so both "random" and "wrong" degrade to self-pairing (shift=0) for that
batch -- a documented limitation, irrelevant in practice since every config in
this project uses batch_size=64.
"""
from __future__ import annotations

import torch

from lossfunction_project.data.augmentations import AugmentationConfig, augment_batch

TRUE_PAIRING = "true"
RANDOM_PAIRING = "random"
WRONG_PAIRING = "wrong"
_KNOWN_STRATEGIES = {TRUE_PAIRING, RANDOM_PAIRING, WRONG_PAIRING}


def _cyclic_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Pair each window in the batch with the one `shift` positions away
    (wrapping around), along the batch dimension. shift=0 means no shift
    (self-pairing); any shift in [1, batch_size-1] guarantees every element
    is paired with a *different* one.
    """
    return torch.roll(x, shifts=shift, dims=0)


def pair_batch(
    x: torch.Tensor,
    aug_cfg: AugmentationConfig,
    strategy: str,
    generator: torch.Generator,
    wrong_shift: int = 1,
) -> torch.Tensor:
    """Build the paired batch x' for a batch of windows x, according to `strategy`.

    `wrong_shift` is only used by strategy="wrong"; ignored otherwise.
    """
    n = x.shape[0]
    if strategy == TRUE_PAIRING:
        return augment_batch(x, aug_cfg, generator)
    if strategy == RANDOM_PAIRING:
        shift = 0
        if n > 1:
            shift = int(torch.randint(1, n, (1,), generator=generator).item())
        return augment_batch(_cyclic_shift(x, shift), aug_cfg, generator)
    if strategy == WRONG_PAIRING:
        shift = (wrong_shift % n) if n > 1 else 0
        return augment_batch(_cyclic_shift(x, shift), aug_cfg, generator)
    raise NotImplementedError(
        f"Pairing strategy {strategy!r} is not recognized. Supported strategies: "
        f"{sorted(_KNOWN_STRATEGIES)}."
    )


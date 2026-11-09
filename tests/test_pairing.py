"""Unit tests for pairing strategies (true, random, wrong -- negative controls)."""
import pytest
import torch

from lossfunction_project.data.augmentations import AugmentationConfig
from lossfunction_project.training.pairing import (
    RANDOM_PAIRING,
    TRUE_PAIRING,
    WRONG_PAIRING,
    _cyclic_shift,
    pair_batch,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _no_op_aug_cfg():
    """An AugmentationConfig whose transforms are all true no-ops, so tests
    can isolate the pairing (shift) mechanism from augmentation noise. Note
    `enabled` itself has no effect on augment_batch/pair_batch -- it is only
    consulted upstream in train_loop.py to decide whether to use the paired
    training path at all -- so it must not be relied on to suppress noise.
    """
    return AugmentationConfig(
        noise_std=0.0,
        amplitude_scale_range=(1.0, 1.0),
        offset_range=(0.0, 0.0),
        dropout_fraction=0.0,
    )


def test_true_pairing_returns_augmented_view_same_shape():
    cfg = AugmentationConfig(enabled=True, noise_std=0.05, dropout_fraction=0.1)
    x = torch.randn(6, 512)
    x_aug = pair_batch(x, cfg, strategy=TRUE_PAIRING, generator=_gen())
    assert x_aug.shape == x.shape
    assert not torch.allclose(x_aug, x)


def test_unknown_strategy_raises_not_implemented():
    cfg = AugmentationConfig(enabled=True)
    x = torch.randn(4, 512)
    with pytest.raises(NotImplementedError):
        pair_batch(x, cfg, strategy="not_a_real_strategy", generator=_gen())


def test_cyclic_shift_zero_is_identity():
    x = torch.arange(20).reshape(4, 5)
    out = _cyclic_shift(x, shift=0)
    torch.testing.assert_close(out, x)


def test_cyclic_shift_nonzero_pairs_every_element_with_a_different_one():
    """For any shift in [1, n-1], no row should end up paired with its own
    original row -- this is the core guarantee that makes "random"/"wrong"
    pairing genuine negative controls (never accidentally identical to true
    pairing)."""
    n = 8
    x = torch.arange(n * 3).reshape(n, 3)
    for shift in range(1, n):
        out = _cyclic_shift(x, shift=shift)
        for i in range(n):
            assert not torch.equal(out[i], x[i])


def test_random_pairing_shape_and_differs_from_input():
    cfg = AugmentationConfig(enabled=True, noise_std=0.02)
    x = torch.randn(10, 512)
    x_aug = pair_batch(x, cfg, strategy=RANDOM_PAIRING, generator=_gen())
    assert x_aug.shape == x.shape
    assert not torch.allclose(x_aug, x)


def test_random_pairing_varies_with_generator_state():
    """Different generator seeds should (very likely) pick a different shift,
    producing a different pairing -- unlike "wrong" pairing, whose shift is
    fixed regardless of generator state."""
    cfg = _no_op_aug_cfg()  # isolate the shift mechanism from augmentation noise
    x = torch.arange(50).reshape(10, 5).float()
    out_a = pair_batch(x, cfg, strategy=RANDOM_PAIRING, generator=_gen(1))
    out_b = pair_batch(x, cfg, strategy=RANDOM_PAIRING, generator=_gen(2))
    assert not torch.equal(out_a, out_b)


def test_wrong_pairing_is_deterministic_regardless_of_generator_seed():
    """The "wrong" strategy's shift is fixed (wrong_shift), not drawn from the
    generator -- so the resulting pairing must be identical across different
    generator seeds."""
    cfg = _no_op_aug_cfg()
    x = torch.arange(50).reshape(10, 5).float()
    out_a = pair_batch(x, cfg, strategy=WRONG_PAIRING, generator=_gen(1), wrong_shift=3)
    out_b = pair_batch(x, cfg, strategy=WRONG_PAIRING, generator=_gen(2), wrong_shift=3)
    torch.testing.assert_close(out_a, out_b)


def test_wrong_pairing_respects_custom_shift():
    cfg = _no_op_aug_cfg()
    x = torch.arange(50).reshape(10, 5).float()
    expected = _cyclic_shift(x, shift=4)
    out = pair_batch(x, cfg, strategy=WRONG_PAIRING, generator=_gen(), wrong_shift=4)
    torch.testing.assert_close(out, expected)


def test_random_and_wrong_handle_batch_size_one_gracefully():
    """Documented edge case: with a single-element batch there is no other
    window to pair with, so both strategies degrade to self-pairing (shift=0)
    rather than raising an error."""
    cfg = _no_op_aug_cfg()
    x = torch.randn(1, 512)
    out_random = pair_batch(x, cfg, strategy=RANDOM_PAIRING, generator=_gen())
    out_wrong = pair_batch(x, cfg, strategy=WRONG_PAIRING, generator=_gen(), wrong_shift=5)
    torch.testing.assert_close(out_random, x)
    torch.testing.assert_close(out_wrong, x)

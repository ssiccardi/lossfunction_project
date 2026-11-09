"""Unit tests for PPG augmentation primitives."""
import torch

from lossfunction_project.data.augmentations import (
    AugmentationConfig,
    add_baseline_offset,
    add_gaussian_noise,
    apply_dropout,
    augment_batch,
    scale_amplitude,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_add_gaussian_noise_changes_values_and_is_zero_mean_ish():
    x = torch.zeros(1000, 512)
    out = add_gaussian_noise(x, std=0.5, generator=_gen())
    assert not torch.allclose(out, x)
    assert abs(out.mean().item()) < 0.05  # large sample, should be close to 0


def test_add_gaussian_noise_zero_std_is_noop():
    x = torch.randn(4, 512)
    out = add_gaussian_noise(x, std=0.0, generator=_gen())
    torch.testing.assert_close(out, x)


def test_scale_amplitude_applies_one_factor_per_window():
    x = torch.ones(5, 10)
    out = scale_amplitude(x, scale_range=(2.0, 2.0), generator=_gen())
    torch.testing.assert_close(out, torch.full((5, 10), 2.0))


def test_scale_amplitude_varies_across_windows_not_within():
    x = torch.ones(4, 100)
    out = scale_amplitude(x, scale_range=(0.5, 1.5), generator=_gen())
    # Within each window, every sample must share the same scale factor.
    for row in out:
        assert torch.allclose(row, row[0].expand_as(row))
    # Across windows, factors should generally differ (not asserting strict
    # inequality to avoid any flakiness, but variance should be > 0).
    assert out[:, 0].std().item() > 0


def test_add_baseline_offset_zero_range_is_noop():
    x = torch.randn(4, 512)
    out = add_baseline_offset(x, offset_range=(0.0, 0.0), generator=_gen())
    torch.testing.assert_close(out, x)


def test_add_baseline_offset_shifts_whole_window_by_constant():
    x = torch.zeros(3, 10)
    out = add_baseline_offset(x, offset_range=(5.0, 5.0), generator=_gen())
    torch.testing.assert_close(out, torch.full((3, 10), 5.0))


def test_apply_dropout_zero_fraction_is_noop():
    x = torch.randn(4, 512)
    out = apply_dropout(x, dropout_fraction=0.0, generator=_gen())
    torch.testing.assert_close(out, x)


def test_apply_dropout_sets_approximately_expected_fraction_to_fill_value():
    x = torch.ones(1, 10000)
    out = apply_dropout(x, dropout_fraction=0.1, generator=_gen(), fill_value=0.0)
    fraction_zeroed = (out == 0.0).float().mean().item()
    assert abs(fraction_zeroed - 0.1) < 0.02  # large sample, should be close


def test_apply_dropout_uses_custom_fill_value():
    x = torch.ones(1, 10000)
    out = apply_dropout(x, dropout_fraction=1.0, generator=_gen(), fill_value=-7.0)
    # dropout_fraction=1.0 -> keep_prob=0 -> every sample replaced by fill_value
    assert torch.allclose(out, torch.full_like(out, -7.0))


def test_augmentation_config_from_dict_defaults():
    cfg = AugmentationConfig.from_dict(None)
    assert cfg.enabled is False

    cfg2 = AugmentationConfig.from_dict({"enabled": True, "noise_std": 0.1})
    assert cfg2.enabled is True
    assert cfg2.noise_std == 0.1
    assert cfg2.amplitude_scale_range == (0.9, 1.1)  # default preserved


def test_augment_batch_runs_full_pipeline_and_changes_input():
    cfg = AugmentationConfig(
        enabled=True,
        noise_std=0.05,
        amplitude_scale_range=(0.9, 1.1),
        offset_range=(-0.1, 0.1),
        dropout_fraction=0.1,
    )
    x = torch.randn(8, 512)
    out = augment_batch(x, cfg, generator=_gen())
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_augment_batch_reproducible_given_same_generator_seed():
    cfg = AugmentationConfig(enabled=True, noise_std=0.05, dropout_fraction=0.1)
    x = torch.randn(8, 512)
    out_a = augment_batch(x, cfg, generator=_gen(123))
    out_b = augment_batch(x, cfg, generator=_gen(123))
    torch.testing.assert_close(out_a, out_b)

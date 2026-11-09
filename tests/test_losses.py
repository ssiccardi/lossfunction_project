"""Unit tests for the custom robust (Huber-style) loss."""
import torch
import torch.nn.functional as F
import pytest

from lossfunction_project.losses.robust import RobustHuberLoss


def test_matches_pytorch_huber_reference():
    """Validates the from-scratch implementation against PyTorch's own
    F.huber_loss (used here only as a correctness oracle for testing, never as
    part of the production loss path).
    """
    torch.manual_seed(0)
    pred = torch.randn(100) * 3
    target = torch.randn(100) * 3
    for delta in (0.5, 1.0, 2.5):
        ours = RobustHuberLoss(delta=delta, reduction="mean")(pred, target)
        reference = F.huber_loss(pred, target, delta=delta, reduction="mean")
        assert torch.allclose(ours, reference, atol=1e-6), f"mismatch at delta={delta}"


def test_quadratic_region_matches_half_squared_error():
    loss_fn = RobustHuberLoss(delta=2.0, reduction="none")
    pred = torch.tensor([1.0])
    target = torch.tensor([2.0])  # residual = -1.0, |residual| < delta=2.0
    out = loss_fn(pred, target)
    assert torch.allclose(out, torch.tensor([0.5 * 1.0**2]))


def test_linear_region_matches_delta_scaled_formula():
    loss_fn = RobustHuberLoss(delta=1.0, reduction="none")
    pred = torch.tensor([5.0])
    target = torch.tensor([0.0])  # residual = 5.0, |residual| > delta=1.0
    out = loss_fn(pred, target)
    expected = 1.0 * (5.0 - 0.5 * 1.0)
    assert torch.allclose(out, torch.tensor([expected]))


def test_continuous_at_switch_point():
    """Quadratic and linear branches must agree exactly at |residual| == delta."""
    delta = 1.5
    loss_fn = RobustHuberLoss(delta=delta, reduction="none")
    pred = torch.tensor([delta])
    target = torch.tensor([0.0])
    out = loss_fn(pred, target)
    assert torch.allclose(out, torch.tensor([0.5 * delta**2]), atol=1e-6)


def test_gradient_is_capped_beyond_delta():
    """The whole point of the robust loss: gradient magnitude must not keep
    growing for residuals far beyond delta (unlike plain MSE)."""
    delta = 1.0
    loss_fn = RobustHuberLoss(delta=delta, reduction="mean")

    pred_large = torch.tensor([10.0], requires_grad=True)
    target = torch.tensor([0.0])
    loss_fn(pred_large, target).backward()
    grad_large = pred_large.grad.abs().item()

    pred_huge = torch.tensor([1000.0], requires_grad=True)
    loss_fn(pred_huge, target).backward()
    grad_huge = pred_huge.grad.abs().item()

    # Gradient saturates at `delta` in the linear region; it must not grow
    # further even though the residual grew by 100x.
    assert grad_large == pytest.approx(delta, abs=1e-5)
    assert grad_huge == pytest.approx(delta, abs=1e-5)


def test_mse_gradient_would_not_be_capped_for_contrast():
    """Sanity check contrasting with plain MSE, whose gradient grows linearly
    with the residual -- this is exactly the behaviour the robust loss avoids.
    """
    pred_large = torch.tensor([10.0], requires_grad=True)
    target = torch.tensor([0.0])
    (0.5 * (pred_large - target) ** 2).backward()
    grad_large = pred_large.grad.abs().item()

    pred_huge = torch.tensor([1000.0], requires_grad=True)
    (0.5 * (pred_huge - target) ** 2).backward()
    grad_huge = pred_huge.grad.abs().item()

    assert grad_huge > grad_large  # unbounded growth, unlike the robust loss above


def test_elementwise_branch_selection_within_one_batch():
    """A batch mixing small and large residuals must apply the correct branch
    to each element independently, not pick one branch for the whole batch."""
    delta = 1.0
    loss_fn = RobustHuberLoss(delta=delta, reduction="none")
    pred = torch.tensor([0.5, 10.0])
    target = torch.tensor([0.0, 0.0])
    out = loss_fn(pred, target)
    expected_small = 0.5 * 0.5**2  # quadratic branch
    expected_large = delta * (10.0 - 0.5 * delta)  # linear branch
    assert torch.allclose(out, torch.tensor([expected_small, expected_large]))


def test_reduction_modes():
    loss_fn_none = RobustHuberLoss(delta=1.0, reduction="none")
    loss_fn_sum = RobustHuberLoss(delta=1.0, reduction="sum")
    loss_fn_mean = RobustHuberLoss(delta=1.0, reduction="mean")
    pred = torch.tensor([0.5, 10.0])
    target = torch.tensor([0.0, 0.0])

    elementwise = loss_fn_none(pred, target)
    assert torch.allclose(loss_fn_sum(pred, target), elementwise.sum())
    assert torch.allclose(loss_fn_mean(pred, target), elementwise.mean())


def test_rejects_invalid_delta():
    with pytest.raises(ValueError):
        RobustHuberLoss(delta=0.0)
    with pytest.raises(ValueError):
        RobustHuberLoss(delta=-1.0)


def test_rejects_invalid_reduction():
    with pytest.raises(ValueError):
        RobustHuberLoss(delta=1.0, reduction="invalid")

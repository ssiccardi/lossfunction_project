"""Unit tests for the consistency loss."""
import torch

from lossfunction_project.losses.consistency import ConsistencyLoss


def test_zero_for_identical_predictions():
    loss_fn = ConsistencyLoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn(pred, pred.clone())
    assert torch.allclose(loss, torch.tensor(0.0))


def test_matches_manual_mean_squared_difference():
    loss_fn = ConsistencyLoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    pred_aug = torch.tensor([1.5, 2.0, 4.0])
    expected = torch.mean((pred - pred_aug) ** 2)
    loss = loss_fn(pred, pred_aug)
    assert torch.allclose(loss, expected)


def test_ignores_label_only_depends_on_two_predictions():
    """The consistency loss must not take a label argument at all -- this test
    just documents/enforces the 2-argument signature."""
    loss_fn = ConsistencyLoss()
    pred = torch.tensor([5.0])
    pred_aug = torch.tensor([5.0])
    loss = loss_fn(pred, pred_aug)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_is_symmetric():
    loss_fn = ConsistencyLoss()
    a = torch.tensor([1.0, 5.0, -2.0])
    b = torch.tensor([3.0, 1.0, 0.0])
    assert torch.allclose(loss_fn(a, b), loss_fn(b, a))


def test_gradient_flows_to_both_inputs():
    loss_fn = ConsistencyLoss()
    pred = torch.tensor([2.0], requires_grad=True)
    pred_aug = torch.tensor([5.0], requires_grad=True)
    loss = loss_fn(pred, pred_aug)
    loss.backward()
    assert pred.grad is not None
    assert pred_aug.grad is not None
    assert pred.grad.item() != 0.0

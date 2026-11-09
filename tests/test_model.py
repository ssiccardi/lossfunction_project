"""Unit tests for the shared CNN1D model."""
import torch

from lossfunction_project.models.cnn1d import CNN1D, build_model


def test_forward_output_shape():
    model = CNN1D(conv_channels=(8, 16), kernel_size=5)
    x = torch.randn(4, 512)
    out = model(x)
    assert out.shape == (4,)


def test_forward_accepts_channel_dim():
    model = CNN1D(conv_channels=(8, 16), kernel_size=5)
    x = torch.randn(4, 1, 512)
    out = model(x)
    assert out.shape == (4,)


def test_rejects_even_kernel_size():
    import pytest

    with pytest.raises(ValueError):
        CNN1D(kernel_size=6)


def test_build_model_from_config_dict():
    model = build_model({"conv_channels": [8, 16, 32], "kernel_size": 5, "dropout": 0.2})
    x = torch.randn(2, 512)
    out = model(x)
    assert out.shape == (2,)


def test_handles_arbitrary_window_length():
    """AdaptiveAvgPool1d means the model is not hardcoded to window_len=512."""
    model = CNN1D(conv_channels=(8, 16))
    x = torch.randn(3, 256)
    out = model(x)
    assert out.shape == (3,)

"""Unit tests for the dataset/target-scaler helpers."""
import numpy as np

from lossfunction_project.data.splits import ArrayBundle
from lossfunction_project.training.dataset import fit_target_scaler, make_loader


def _make_bundle(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return ArrayBundle(
        X=rng.standard_normal((n, 512)).astype(np.float32),
        y=rng.uniform(60, 100, n).astype(np.float32),
        subject_id=np.full(n, "S1", dtype=object),
        window_start_s=np.arange(n, dtype=np.float32) * 2.0,
        label_index=np.arange(n, dtype=np.int64),
    )


def test_scaler_fit_and_inverse_roundtrip():
    bundle = _make_bundle()
    scaler = fit_target_scaler(bundle)
    transformed = scaler.transform(bundle.y)
    restored = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(restored, bundle.y, atol=1e-4)
    np.testing.assert_allclose(transformed.mean(), 0.0, atol=1e-4)
    np.testing.assert_allclose(transformed.std(), 1.0, atol=1e-3)


def test_scaler_handles_zero_variance_target():
    bundle = _make_bundle()
    bundle.y[:] = 75.0  # degenerate case: constant target
    scaler = fit_target_scaler(bundle)
    assert scaler.std > 0  # eps floor avoids division by zero


def test_make_loader_reproducible_shuffle():
    bundle = _make_bundle(n=37)
    scaler = fit_target_scaler(bundle)
    loader_a = make_loader(bundle, scaler, batch_size=8, shuffle=True, seed=123)
    loader_b = make_loader(bundle, scaler, batch_size=8, shuffle=True, seed=123)
    ys_a = np.concatenate([y.numpy() for _X, y in loader_a])
    ys_b = np.concatenate([y.numpy() for _X, y in loader_b])
    np.testing.assert_array_equal(ys_a, ys_b)


def test_make_loader_covers_all_samples():
    bundle = _make_bundle(n=37)
    scaler = fit_target_scaler(bundle)
    loader = make_loader(bundle, scaler, batch_size=8, shuffle=False, seed=1)
    ys = np.concatenate([y.numpy() for _X, y in loader])
    assert len(ys) == 37

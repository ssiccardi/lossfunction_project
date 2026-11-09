"""Unit tests for label-aligned PPG windowing."""
import numpy as np

from lossfunction_project.data.extract import SubjectArrays
from lossfunction_project.data.windowing import build_subject_windows, window_length_samples


def _make_synthetic_arrays(n_labels=5, fs_bvp=64.0, label_window_s=8.0, label_shift_s=2.0):
    # Enough BVP samples to cover the last label's window with some margin.
    t_end_last = label_window_s + (n_labels - 1) * label_shift_s
    n_bvp = int(t_end_last * fs_bvp) + 100
    bvp = np.arange(n_bvp, dtype=np.float32)  # ramp signal, easy to check slicing
    label = np.linspace(60, 80, n_labels, dtype=np.float32)
    activity = np.zeros(10, dtype=np.float32)
    return SubjectArrays(
        subject="S_test",
        bvp=bvp,
        label=label,
        activity=activity,
        fs_bvp=fs_bvp,
        label_window_s=label_window_s,
        label_shift_s=label_shift_s,
    )


def test_window_length_samples():
    arrays = _make_synthetic_arrays()
    assert window_length_samples(arrays) == int(8.0 * 64.0)


def test_window_count_and_shape():
    arrays = _make_synthetic_arrays(n_labels=5)
    windows = build_subject_windows(arrays, normalize=False)
    assert windows.X.shape == (5, 512)
    assert windows.y.shape == (5,)
    np.testing.assert_allclose(windows.y, arrays.label)


def test_window_alignment_no_future_leakage():
    """Each window must end exactly at t_i and contain no samples beyond it."""
    arrays = _make_synthetic_arrays(n_labels=5)
    windows = build_subject_windows(arrays, normalize=False)
    fs = arrays.fs_bvp
    for i in range(len(windows.y)):
        t_end_s = arrays.label_window_s + i * arrays.label_shift_s
        end_sample = int(round(t_end_s * fs))
        # window content (ramp signal) must equal bvp[end_sample-512:end_sample]
        expected = arrays.bvp[end_sample - 512 : end_sample]
        np.testing.assert_array_equal(windows.X[i], expected)
        assert windows.X[i].max() < end_sample  # nothing from t_end_s onward


def test_drops_window_before_recording_start():
    arrays = _make_synthetic_arrays(n_labels=3)
    # Truncate the BVP signal so the first label's window cannot be fully built.
    arrays.bvp = arrays.bvp[50:]
    windows = build_subject_windows(arrays, normalize=False)
    assert len(windows.y) <= 3  # at least the too-early window(s) should be dropped


def test_normalize_zero_mean_unit_var():
    arrays = _make_synthetic_arrays(n_labels=3)
    windows = build_subject_windows(arrays, normalize=True)
    means = windows.X.mean(axis=1)
    stds = windows.X.std(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-3)

"""Unit tests for regression metrics."""
import numpy as np

from lossfunction_project.evaluation.metrics import (
    mae,
    per_subject_metrics,
    pearson_r,
    regression_metrics,
    residual_coverage,
    rmse,
)


def test_mae_rmse_basic():
    y_true = np.array([60.0, 70.0, 80.0])
    y_pred = np.array([62.0, 68.0, 85.0])
    assert mae(y_true, y_pred) == np.mean([2, 2, 5])
    assert rmse(y_true, y_pred) == np.sqrt(np.mean([4, 4, 25]))


def test_pearson_r_perfect_correlation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])
    assert abs(pearson_r(y_true, y_pred) - 1.0) < 1e-9


def test_pearson_r_nan_on_zero_variance():
    y_true = np.array([70.0, 70.0, 70.0])
    y_pred = np.array([60.0, 65.0, 72.0])
    assert np.isnan(pearson_r(y_true, y_pred))


def test_regression_metrics_keys():
    y_true = np.array([60.0, 70.0, 80.0])
    y_pred = np.array([61.0, 69.0, 82.0])
    m = regression_metrics(y_true, y_pred)
    assert set(m.keys()) == {"n", "mae", "rmse", "pearson_r"}
    assert m["n"] == 3


def test_per_subject_metrics_splits_correctly():
    y_true = np.array([60.0, 70.0, 80.0, 90.0])
    y_pred = np.array([61.0, 69.0, 82.0, 88.0])
    subject_id = np.array(["S1", "S1", "S2", "S2"], dtype=object)
    result = per_subject_metrics(y_true, y_pred, subject_id)
    assert set(result.keys()) == {"S1", "S2"}
    assert result["S1"]["n"] == 2
    assert result["S2"]["n"] == 2


def test_residual_coverage_basic():
    y_true_scaled = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred_scaled = np.array([0.2, 0.5, 1.5, 3.0])  # abs residuals: 0.2, 0.5, 1.5, 3.0
    cov = residual_coverage(y_true_scaled, y_pred_scaled, delta=1.0)
    assert cov["n"] == 4
    assert cov["n_exceeding"] == 2  # 1.5 and 3.0 exceed delta=1.0
    assert cov["fraction_exceeding"] == 0.5
    assert cov["max_abs_residual_scaled"] == 3.0
    assert cov["mean_abs_residual_scaled"] == np.mean([0.2, 0.5, 1.5, 3.0])


def test_residual_coverage_zero_exceeding():
    y_true_scaled = np.zeros(5)
    y_pred_scaled = np.full(5, 0.1)
    cov = residual_coverage(y_true_scaled, y_pred_scaled, delta=1.0)
    assert cov["n_exceeding"] == 0
    assert cov["fraction_exceeding"] == 0.0


def test_residual_coverage_empty_input():
    cov = residual_coverage(np.array([]), np.array([]), delta=1.0)
    assert cov["n"] == 0
    assert np.isnan(cov["fraction_exceeding"])

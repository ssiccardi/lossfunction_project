"""Regression metrics for heart-rate predictions: overall and per-subject breakdown."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation; returns NaN if either array has zero variance (e.g. a
    degenerate split too small to decide, or a negative-control run that collapses
    to a constant prediction), rather than raising or silently returning 0.
    """
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Core metric set: MAE, RMSE, Pearson r."""
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
    }


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_id: np.ndarray,
    subjects: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-subject metric breakdown, since pooled metrics can hide subject-specific
    behaviour, especially with only a handful of subjects playing each role in a
    split.
    """
    if subjects is None:
        subjects = sorted(set(subject_id.tolist()))
    result = {}
    for s in subjects:
        mask = subject_id == s
        result[s] = regression_metrics(y_true[mask], y_pred[mask])
    return result


def residual_coverage(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, delta: float) -> Dict[str, float]:
    """Diagnostic for interpreting a robust (Huber-style) loss: what fraction of
    residuals actually exceed a given threshold `delta`, in the same standardized
    (z-scored) target space the loss itself operates on.

    A robust loss only changes training behaviour for samples with
    |residual| > delta (its linear region); if that fraction is small, most
    training signal still comes from the same quadratic penalty as plain MSE, and
    any difference from a standard-loss run should be attributed to how the
    (few) large-residual samples are handled, not to a wholesale change in the
    loss shape. This function is generic -- it can be computed for any run
    (standard-loss or robust-loss) and at any reference `delta`, so results stay
    directly comparable across configs.
    """
    residuals = y_pred_scaled - y_true_scaled
    abs_residuals = np.abs(residuals)
    n = int(len(residuals))
    n_exceeding = int(np.sum(abs_residuals > delta))
    return {
        "delta": float(delta),
        "n": n,
        "n_exceeding": n_exceeding,
        "fraction_exceeding": float(n_exceeding / n) if n > 0 else float("nan"),
        "mean_abs_residual_scaled": float(np.mean(abs_residuals)) if n > 0 else float("nan"),
        "max_abs_residual_scaled": float(np.max(abs_residuals)) if n > 0 else float("nan"),
    }

"""Unit test for the figure-generation script's CSV-reading logic."""
import csv

from scripts.make_figures import read_pooled_test_metric


def _write_metrics_csv(path, rows):
    fieldnames = ["split", "subject", "n", "mae", "rmse", "pearson_r"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_pooled_test_metric_picks_test_all_row(tmp_path):
    _write_metrics_csv(
        tmp_path / "demo_metrics.csv",
        [
            {"split": "train", "subject": "ALL", "n": 100, "mae": 1.0, "rmse": 1.5, "pearson_r": 0.9},
            {"split": "test", "subject": "S1", "n": 40, "mae": 5.0, "rmse": 6.0, "pearson_r": 0.5},
            {"split": "test", "subject": "ALL", "n": 80, "mae": 3.5, "rmse": 4.5, "pearson_r": 0.7},
        ],
    )
    mae = read_pooled_test_metric("demo", "mae", tmp_path)
    r = read_pooled_test_metric("demo", "pearson_r", tmp_path)
    assert mae == 3.5
    assert r == 0.7


def test_read_pooled_test_metric_raises_if_missing(tmp_path):
    _write_metrics_csv(
        tmp_path / "nodata_metrics.csv",
        [{"split": "train", "subject": "ALL", "n": 100, "mae": 1.0, "rmse": 1.5, "pearson_r": 0.9}],
    )
    import pytest

    with pytest.raises(ValueError):
        read_pooled_test_metric("nodata", "mae", tmp_path)

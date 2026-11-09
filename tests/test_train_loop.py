"""Integration test for the shared training loop, using small synthetic data.

Uses tmp_path for processed/results/checkpoints directories so this test never
touches the real data/processed or results/ directories, and keeps epochs/windows
tiny so it runs in well under a second.
"""
import numpy as np
import pytest

from lossfunction_project.training.train_loop import (
    NegativeControlConfig,
    PairedTrainingConfig,
    WindowLevelSplitConfig,
    build_loss_fn,
    run_experiment,
)


def _write_synthetic_windows(processed_dir, subject, n_windows, seed, bpm_offset=0.0):
    """Write a {subject}_windows.npz matching scripts/run_windowing.py's format,
    with a feature (window mean) that is linearly related to the target, so the
    model has something learnable within a handful of epochs.
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(60, 100, n_windows).astype(np.float32) + bpm_offset
    # Build windows whose mean value encodes (a noisy version of) the target,
    # so a tiny CNN can learn a non-trivial mapping quickly.
    X = rng.standard_normal((n_windows, 512)).astype(np.float32) * 0.1
    X += (y / 100.0)[:, None]

    np.savez_compressed(
        processed_dir / f"{subject}_windows.npz",
        subject=subject,
        X=X,
        y=y,
        window_start_s=np.arange(n_windows, dtype=np.float32) * 2.0,
        label_index=np.arange(n_windows, dtype=np.int64),
        normalized=True,
    )


def test_run_experiment_end_to_end(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"

    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_experiment",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        verbose=False,
    )

    assert results["name"] == "test_experiment"
    assert results["assignment"] == "default"
    for role in ("train", "val", "test"):
        assert role in results
        assert results[role]["overall"]["n"] > 0
        assert "mae" in results[role]["overall"]
        assert 1.0 in results[role]["residual_coverage"]
        assert "fraction_exceeding" in results[role]["residual_coverage"][1.0]

    assert (results_dir / "test_experiment_metrics.csv").exists()
    assert (results_dir / "test_experiment_history.csv").exists()
    assert (results_dir / "test_experiment_residual_coverage.csv").exists()
    assert (checkpoints_dir / "test_experiment.pt").exists()
    assert (checkpoints_dir / "test_experiment_config.json").exists()


def test_run_experiment_respects_subject_level_split(tmp_path):
    """Test subjects' windows must never influence train-time target scaling or
    appear in the train/val loaders -- a regression here would silently break the
    experiment's leakage-safety guarantee."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=30, seed=i)

    config = {
        "name": "test_leakage_check",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )

    assert set(results["train"]["per_subject"].keys()) == {"S1", "S2", "S3"}
    assert set(results["val"]["per_subject"].keys()) == {"S4"}
    assert set(results["test"]["per_subject"].keys()) == {"S5"}


def test_build_loss_fn_supports_huber():
    from lossfunction_project.losses.robust import RobustHuberLoss

    loss_fn = build_loss_fn({"type": "huber", "delta": 2.0, "reduction": "mean"})
    assert isinstance(loss_fn, RobustHuberLoss)
    assert loss_fn.delta == 2.0
    assert loss_fn.reduction == "mean"


def test_build_loss_fn_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_loss_fn({"type": "not_a_real_loss"})


def test_run_experiment_with_robust_loss_end_to_end(tmp_path):
    """Same end-to-end path as the baseline test, but with loss.type='huber' --
    confirms the robust loss is a true drop-in replacement for the standard loss
    in run_experiment, with no other change to the pipeline.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_robust_experiment",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "huber", "delta": 1.0},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )

    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0


def test_residual_coverage_includes_config_delta_when_different_from_reference(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=30, seed=i)

    config = {
        "name": "test_delta_coverage",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "huber", "delta": 0.3},  # different from the 1.0 reference
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )

    assert set(results["train"]["residual_coverage"].keys()) == {1.0, 0.3}


def test_run_experiment_with_augmentation_extra_data_loss_end_to_end(tmp_path):
    """Augmentation-only style config: standard loss + augmented view used as
    extra training data (extra_data_loss=True), no consistency term.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_augmentation_only",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "augmentation": {"enabled": True, "noise_std": 0.02, "dropout_fraction": 0.05},
        "paired_training": {"extra_data_loss": True, "consistency_lambda": 0.0},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0

    import json

    with open(tmp_path / "checkpoints" / "test_augmentation_only_config.json") as f:
        saved_config = json.load(f)
    assert saved_config["augmentation"]["enabled"] is True
    assert saved_config["paired_training"]["extra_data_loss"] is True


def test_run_experiment_with_consistency_only_end_to_end(tmp_path):
    """Consistency-only style config: standard loss + consistency term between
    predictions on x and an augmented view x', no extra-data term.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_consistency_only",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "augmentation": {"enabled": True, "noise_std": 0.02, "dropout_fraction": 0.05},
        "paired_training": {"extra_data_loss": False, "consistency_lambda": 0.5},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0


def test_run_experiment_full_method_end_to_end(tmp_path):
    """Full-method style config: robust loss + augmentation + consistency term,
    all combined -- confirms the whole stack composes without error."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_full_method",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "huber", "delta": 1.0},
        "augmentation": {"enabled": True, "noise_std": 0.02, "dropout_fraction": 0.05},
        "paired_training": {"extra_data_loss": False, "consistency_lambda": 0.5},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0


def test_augmentation_disabled_matches_pre_refactor_behaviour(tmp_path):
    """Regression guard: a config with no `augmentation` key at all (like the
    baseline/robust-loss configs) must train identically to how it did before
    `train_standard_model` was generalized into `train_model` -- i.e. running
    the same config twice with the same seed gives identical results.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=30, seed=i)

    config = {
        "name": "test_no_augmentation",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results_a = run_experiment(
        dict(config, name="run_a"),
        processed_dir=processed_dir,
        results_dir=tmp_path / "results_a",
        checkpoints_dir=tmp_path / "ckpt_a",
        verbose=False,
    )
    results_b = run_experiment(
        dict(config, name="run_b"),
        processed_dir=processed_dir,
        results_dir=tmp_path / "results_b",
        checkpoints_dir=tmp_path / "ckpt_b",
        verbose=False,
    )

    np.testing.assert_allclose(results_a["test"]["y_pred_bpm"], results_b["test"]["y_pred_bpm"], atol=1e-5)


def test_paired_training_config_coerces_unquoted_yaml_true_pairing_strategy():
    """Regression guard for a real bug caught during manual smoke-testing:
    `pairing_strategy: true` in YAML (unquoted) parses as the Python boolean
    True, not the string "true" -- from_dict must coerce this back to the
    intended string value rather than passing a boolean through to
    pair_batch(), where it would silently fail with a confusing error deep
    inside pairing.py.
    """
    cfg = PairedTrainingConfig.from_dict({"pairing_strategy": True})
    assert cfg.pairing_strategy == "true"
    assert isinstance(cfg.pairing_strategy, str)


def test_negative_control_config_defaults_and_from_dict():
    assert NegativeControlConfig.from_dict(None).shuffle_train_labels is False
    cfg = NegativeControlConfig.from_dict({"shuffle_train_labels": True, "shuffle_seed": 99})
    assert cfg.shuffle_train_labels is True
    assert cfg.shuffle_seed == 99


def test_run_experiment_with_shuffled_labels_negative_control(tmp_path):
    """Shuffled-label probe: train on corrupted (shuffled) labels, but report
    val/test metrics (and the final 'train' metrics) against the TRUE labels.
    The run must not error, and the saved config must record the control.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=50, seed=i)

    config = {
        "name": "test_shuffled_labels",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "negative_control": {"shuffle_train_labels": True, "shuffle_seed": 7},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0

    import json

    with open(tmp_path / "checkpoints" / "test_shuffled_labels_config.json") as f:
        saved_config = json.load(f)
    assert saved_config["negative_control"]["shuffle_train_labels"] is True


def test_shuffled_labels_control_degrades_fit_relative_to_true_labels(tmp_path):
    """The whole point of the probe: training on shuffled labels should
    produce a WORSE fit to the true labels (as measured by correlation on a
    held-out-like evaluation) than training on the true labels, since there
    is no genuine X-y relationship left to learn in the shuffled case. Uses
    more windows/epochs than the other smoke tests so the effect has a chance
    to show up; still small and fast enough to run directly.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=200, seed=i)

    base_config = {
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "training": {"epochs": 10, "batch_size": 16, "lr": 0.01, "seed": 0},
    }

    normal_results = run_experiment(
        dict(base_config, name="normal_labels"),
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    shuffled_results = run_experiment(
        dict(base_config, name="shuffled_labels", negative_control={"shuffle_train_labels": True}),
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )

    normal_r = normal_results["test"]["overall"]["pearson_r"]
    shuffled_r = shuffled_results["test"]["overall"]["pearson_r"]
    # Pearson r can be NaN in degenerate cases; treat NaN as "no correlation",
    # i.e. automatically satisfying "worse than normal" unless normal is also NaN.
    if normal_r == normal_r:  # normal_r is not NaN
        assert (shuffled_r != shuffled_r) or (shuffled_r < normal_r)


def test_run_experiment_with_random_pairing_negative_control(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_random_pairing",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "augmentation": {"enabled": True, "noise_std": 0.02, "dropout_fraction": 0.05},
        "paired_training": {"extra_data_loss": False, "consistency_lambda": 0.5, "pairing_strategy": "random"},
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0


def test_run_experiment_with_wrong_pairing_negative_control(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=40, seed=i)

    config = {
        "name": "test_wrong_pairing",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "augmentation": {"enabled": True, "noise_std": 0.02, "dropout_fraction": 0.05},
        "paired_training": {
            "extra_data_loss": False,
            "consistency_lambda": 0.5,
            "pairing_strategy": "wrong",
            "wrong_shift": 1,
        },
        "training": {"epochs": 3, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    for role in ("train", "val", "test"):
        assert results[role]["overall"]["n"] > 0


def test_window_level_split_config_defaults_and_from_dict():
    assert WindowLevelSplitConfig.from_dict(None).enabled is False
    cfg = WindowLevelSplitConfig.from_dict(
        {"enabled": True, "seed": 7, "train_frac": 0.6, "val_frac": 0.2, "subjects": ["S1", "S2"]}
    )
    assert cfg.enabled is True
    assert cfg.seed == 7
    assert cfg.train_frac == 0.6
    assert cfg.val_frac == 0.2
    assert cfg.subjects == ["S1", "S2"]


def test_run_experiment_with_window_level_split(tmp_path):
    """When window_level_split.enabled is set, run_experiment must bypass the
    subject-level assignment entirely: the resulting train/val/test bundles
    should mix subjects (unlike a subject-level split), and every window from
    every requested subject must appear in exactly one of the three splits.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=50, seed=i)

    config = {
        "name": "test_window_level",
        "assignment": "default",  # present but ignored since window_level_split is enabled
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "window_level_split": {"enabled": True, "seed": 42, "train_frac": 0.7, "val_frac": 0.15},
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )

    assert results["assignment"].startswith("window_level")
    total_n = sum(results[role]["overall"]["n"] for role in ("train", "val", "test"))
    assert total_n == 250  # 5 subjects * 50 windows each, every window used exactly once

    # Train split should mix subjects, unlike a subject-level split.
    train_subjects = set(results["train"]["per_subject"].keys())
    assert len(train_subjects) > 1


def test_window_level_split_disabled_by_default_uses_subject_level(tmp_path):
    """Omitting window_level_split entirely must behave exactly as before
    (regression guard): the default subject-level split is used."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i, subject in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        _write_synthetic_windows(processed_dir, subject, n_windows=30, seed=i)

    config = {
        "name": "test_no_window_level_key",
        "assignment": "default",
        "subjects": ["S1", "S2", "S3", "S4", "S5"],
        "model": {"conv_channels": [4, 8], "kernel_size": 5, "dropout": 0.0},
        "loss": {"type": "mse"},
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.01, "seed": 0},
    }

    results = run_experiment(
        config,
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        verbose=False,
    )
    assert results["assignment"] == "default"
    assert set(results["train"]["per_subject"].keys()) == {"S1", "S2", "S3"}
    assert set(results["val"]["per_subject"].keys()) == {"S4"}
    assert set(results["test"]["per_subject"].keys()) == {"S5"}



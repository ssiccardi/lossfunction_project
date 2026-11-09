"""Shared, config-driven training loop.

This module is deliberately generic: every experiment config (baseline, robust-loss,
augmentation-only, consistency-only, full method, negative controls) goes through
the same `run_experiment` entry point, differing only in which loss function and
batch construction it plugs in via its config:

- `loss`: which supervised regression loss to use (standard mse/mae, or the robust
  Huber-style loss).
- `augmentation` (optional, default disabled): whether to build a paired augmented
  view x' of each training window, and with what parameters.
- `paired_training` (optional, default a no-op): how to use x' once built --
  `extra_data_loss` adds a second supervised loss term on x' (treating it as
  ordinary extra training data), `consistency_lambda` adds a consistency penalty
  between the model's predictions on x and x' (not using the label at all). Both
  can in principle be combined, though the experiment configs in this project each
  use only one, to keep every config attributable to a single named mechanism.
  `pairing_strategy` additionally selects negative-control pairing ("random"/
  "wrong") instead of the real ("true") pairing -- see training/pairing.py.
- `negative_control` (optional, default disabled): `shuffle_train_labels`
  randomly permutes the training split's labels before training (see
  data/splits.shuffle_bundle_labels), independent of the pairing mechanism.

Augmentation and pairing never touch validation/test evaluation or the final
reported metrics -- they only affect what happens inside the training loop.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from lossfunction_project.data.augmentations import AugmentationConfig
from lossfunction_project.data.splits import (
    ArrayBundle,
    apply_subject_level_split,
    apply_window_level_split,
    bundle_summary,
    find_assignment_config,
    load_all_cached_windows,
    load_assignment,
    shuffle_bundle_labels,
)
from lossfunction_project.evaluation.metrics import per_subject_metrics, regression_metrics, residual_coverage
from lossfunction_project.losses.consistency import ConsistencyLoss
from lossfunction_project.losses.robust import RobustHuberLoss
from lossfunction_project.models.cnn1d import build_model
from lossfunction_project.training.dataset import TargetScaler, fit_target_scaler, make_loader
from lossfunction_project.training.pairing import TRUE_PAIRING, pair_batch
from lossfunction_project.utils.config import PROJECT_ROOT
from lossfunction_project.utils.seed import seed_everything

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def build_loss_fn(loss_cfg: dict) -> nn.Module:
    """Build the training loss from a config dict.

    Standard losses (mse/mae) are the baseline comparison point. The robust
    Huber-style loss (losses/robust.py) is a drop-in replacement operating on the
    same (standardized) prediction/target space -- so switching `loss.type` is
    the *only* change relative to the baseline config, which is what lets the
    resulting comparison attribute any difference to the loss function itself.
    Note: `delta` for the robust loss is interpreted in standardized-target units
    (the training-set z-score space), not raw bpm -- see training/dataset.py.
    """
    loss_type = loss_cfg.get("type", "mse")
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "huber":
        return RobustHuberLoss(
            delta=loss_cfg.get("delta", 1.0),
            reduction=loss_cfg.get("reduction", "mean"),
        )
    raise ValueError(f"Unknown loss type: {loss_type!r} (expected 'mse', 'mae', or 'huber')")


def build_optimizer(model: nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    lr = training_cfg.get("lr", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 0.0)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@dataclass
class PairedTrainingConfig:
    """How to use the augmented view x' once built (see `augmentation` config).

    `extra_data_loss`: add a second supervised regression loss term on x'
    (treats the augmented view as ordinary extra training data -- used by the
    augmentation-only ablation, which isolates "more/varied data" from any
    consistency effect). Only meaningful when `pairing_strategy="true"` --
    combining it with "random"/"wrong" pairing would supervise the model
    toward an unrelated window's own label using the *original* window's
    label, which is a different (and not currently used) kind of corruption;
    no experiment config in this project combines the two.
    `consistency_lambda`: weight of the consistency penalty between the
    model's predictions on x and x' (0 disables it). Does not use the label.
    `pairing_strategy`: which pairing strategy builds x' from x (see
    training/pairing.py): "true" (the real deliverable), or the negative
    controls "random"/"wrong" (deliberately mismatched pairing, used to test
    whether the consistency term's effect depends on the pairing being
    correct).
    `wrong_shift`: fixed batch-cyclic shift used only by the "wrong" pairing
    strategy (ignored otherwise).
    """

    extra_data_loss: bool = False
    consistency_lambda: float = 0.0
    pairing_strategy: str = TRUE_PAIRING
    wrong_shift: int = 1

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "PairedTrainingConfig":
        if not cfg:
            return cls()
        pairing_strategy = cfg.get("pairing_strategy", TRUE_PAIRING)
        if pairing_strategy is True:
            # Common YAML footgun: an unquoted `pairing_strategy: true` parses
            # as the boolean True, not the string "true" -- coerce it to the
            # intended string value rather than failing deep inside pairing.py.
            pairing_strategy = TRUE_PAIRING
        return cls(
            extra_data_loss=bool(cfg.get("extra_data_loss", False)),
            consistency_lambda=float(cfg.get("consistency_lambda", 0.0)),
            pairing_strategy=pairing_strategy,
            wrong_shift=int(cfg.get("wrong_shift", 1)),
        )

    @property
    def active(self) -> bool:
        return self.extra_data_loss or self.consistency_lambda > 0


@dataclass
class NegativeControlConfig:
    """Negative controls unrelated to pairing (see training/pairing.py for the
    random/wrong pairing controls).

    `shuffle_train_labels`: if True, the training split's heart-rate labels
    are randomly permuted (see data/splits.shuffle_bundle_labels) before
    training, breaking the X-y correspondence. Validation/test always use
    the true, unshuffled labels, so this tests whether a model can still
    achieve low error on real data despite having no genuine signal to learn
    from during training -- it should not, and if it does, that would point
    to a leakage or shortcut elsewhere in the pipeline.
    `shuffle_seed`: seed for the label permutation; defaults to the
    experiment's main training seed if not given.
    """

    shuffle_train_labels: bool = False
    shuffle_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "NegativeControlConfig":
        if not cfg:
            return cls()
        shuffle_seed = cfg.get("shuffle_seed")
        return cls(
            shuffle_train_labels=bool(cfg.get("shuffle_train_labels", False)),
            shuffle_seed=int(shuffle_seed) if shuffle_seed is not None else None,
        )


@dataclass
class WindowLevelSplitConfig:
    """Opt-in override: use the pooled, shuffled WINDOW-LEVEL split (see
    data/splits.apply_window_level_split) instead of the normal subject-level
    split for this experiment, to illustrate the leakage risk of ignoring
    subject identity in physiological-signal datasets. Included only as a
    secondary, leakage-illustration comparison -- not a recommended
    evaluation protocol.

    When `enabled`, this bypasses `config["assignment"]` entirely for this
    run; `train_frac`/`val_frac` set the split sizes (the remainder goes to
    test), and `subjects` (if omitted, all of `config["subjects"]`) controls
    which subjects' windows are pooled.
    """

    enabled: bool = False
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    subjects: Optional[list] = None

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "WindowLevelSplitConfig":
        if not cfg:
            return cls()
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            seed=int(cfg.get("seed", 42)),
            train_frac=float(cfg.get("train_frac", 0.70)),
            val_frac=float(cfg.get("val_frac", 0.15)),
            subjects=cfg.get("subjects"),
        )


def run_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """One pass over `loader`; trains if `optimizer` is given, else evaluates.

    Used, unchanged, for every validation/test pass, and for training when no
    augmentation/paired-consistency scheme is configured (configs A and B).
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            if is_train:
                optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n_samples += len(y)
    return total_loss / max(n_samples, 1)


def run_paired_training_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    aug_cfg: AugmentationConfig,
    paired_cfg: PairedTrainingConfig,
    consistency_fn: Optional[ConsistencyLoss],
    generator: torch.Generator,
) -> float:
    """One training epoch using a paired augmented view x' of each batch.

    For each batch x: builds x' via `pair_batch` (currently only "true"
    pairing -- an augmented view of x itself, same order, same label),
    computes predictions on both, and combines:

        loss = loss_fn(pred, y)
               [+ loss_fn(pred_aug, y)              if paired_cfg.extra_data_loss]
               [+ paired_cfg.consistency_lambda * consistency_fn(pred, pred_aug)
                                                     if consistency_lambda > 0]

    This mirrors the `L = L_reg(f(x), y) [+ L_reg(f(x'), y)] + lambda *
    L_consistency(f(x), f(x'))` formula: the extra-data term and the
    consistency term are independent, config-controlled additions to the same
    base supervised loss, so a single config can isolate either effect (or, in
    principle, both at once).

    Never used for validation/test -- those always use the plain `run_epoch`,
    so reported metrics are never affected by augmentation.
    """
    model.train(True)
    total_loss = 0.0
    n_samples = 0
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        X_aug = pair_batch(
            X,
            aug_cfg,
            strategy=paired_cfg.pairing_strategy,
            generator=generator,
            wrong_shift=paired_cfg.wrong_shift,
        )
        pred_aug = model(X_aug)

        if paired_cfg.extra_data_loss:
            loss = loss + loss_fn(pred_aug, y)
        if paired_cfg.consistency_lambda > 0:
            assert consistency_fn is not None, "consistency_lambda > 0 requires a consistency_fn"
            loss = loss + paired_cfg.consistency_lambda * consistency_fn(pred, pred_aug)

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        n_samples += len(y)
    return total_loss / max(n_samples, 1)


def predict(model: nn.Module, loader) -> np.ndarray:
    """Run the model over `loader` and return predictions in scaled (training-time)
    target space, in the same order as the loader's underlying dataset.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _y in loader:
            preds.append(model(X).numpy())
    return np.concatenate(preds, axis=0)


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_mae_bpm: list = field(default_factory=list)
    best_epoch: int = 0


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scaler: TargetScaler,
    aug_cfg: Optional[AugmentationConfig] = None,
    paired_cfg: Optional[PairedTrainingConfig] = None,
    consistency_fn: Optional[ConsistencyLoss] = None,
    aug_generator: Optional[torch.Generator] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Trains for a fixed number of epochs, tracking val MAE (in bpm, inverse-
    transformed) each epoch, and restores the model weights from the epoch with
    the lowest val MAE at the end (best-checkpoint selection) -- a reasonable
    safeguard given how small this dataset is and how noisy single-subject
    validation can be.

    If `aug_cfg` is enabled, training batches go through
    `run_paired_training_epoch` (augmented view x' + extra-data/consistency
    terms per `paired_cfg`); otherwise training uses the plain `run_epoch`,
    identical to configs with no augmentation at all. Validation always uses
    the plain `run_epoch` with the base `loss_fn` only, regardless of
    `aug_cfg`, so val/test metrics stay comparable across every config.
    """
    use_paired = aug_cfg is not None and aug_cfg.enabled
    if use_paired:
        assert paired_cfg is not None and aug_generator is not None, (
            "aug_cfg.enabled requires paired_cfg and aug_generator to be provided"
        )

    history = TrainingHistory()
    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        if use_paired:
            train_loss = run_paired_training_epoch(
                model, train_loader, loss_fn, optimizer, aug_cfg, paired_cfg, consistency_fn, aug_generator
            )
        else:
            train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None)

        val_pred_scaled = predict(model, val_loader)
        val_true_scaled = np.concatenate([y.numpy() for _X, y in val_loader])
        val_pred_bpm = scaler.inverse_transform(val_pred_scaled)
        val_true_bpm = scaler.inverse_transform(val_true_scaled)
        val_mae_bpm = float(np.mean(np.abs(val_true_bpm - val_pred_bpm)))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_mae_bpm.append(val_mae_bpm)

        if val_mae_bpm < best_val_mae:
            best_val_mae = val_mae_bpm
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_mae_bpm={val_mae_bpm:.2f}"
                f"{'  <- best so far' if epoch == history.best_epoch else ''}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _evaluate_split(
    model: nn.Module,
    bundle: ArrayBundle,
    scaler: TargetScaler,
    batch_size: int,
    seed: int,
    residual_deltas: tuple = (1.0,),
) -> dict:
    if len(bundle) == 0:
        return {
            "overall": regression_metrics(np.array([]), np.array([])),
            "per_subject": {},
            "residual_coverage": {d: residual_coverage(np.array([]), np.array([]), d) for d in residual_deltas},
        }
    loader = make_loader(bundle, scaler, batch_size=batch_size, shuffle=False, seed=seed)
    pred_scaled = predict(model, loader)
    pred_bpm = scaler.inverse_transform(pred_scaled)
    true_bpm = bundle.y
    true_scaled = scaler.transform(true_bpm)
    overall = regression_metrics(true_bpm, pred_bpm)
    per_subject = per_subject_metrics(true_bpm, pred_bpm, bundle.subject_id)
    residual_diag = {d: residual_coverage(true_scaled, pred_scaled, d) for d in residual_deltas}
    return {
        "overall": overall,
        "per_subject": per_subject,
        "residual_coverage": residual_diag,
        "y_pred_bpm": pred_bpm,
        "y_true_bpm": true_bpm,
    }


def run_experiment(
    config: dict,
    processed_dir=None,
    results_dir=None,
    checkpoints_dir=None,
    verbose: bool = True,
) -> dict:
    """End-to-end: load assignment + windows -> build model/loss/optimizer ->
    train -> evaluate on train/val/test (overall + per-subject) -> save a metrics
    table and a training-history log -> return the full results dict.
    """
    name = config["name"]
    seed = config["training"].get("seed", 42)
    seed_everything(seed)

    all_windows = load_all_cached_windows(config["subjects"], processed_dir=processed_dir)

    window_level_cfg = WindowLevelSplitConfig.from_dict(config.get("window_level_split"))
    if window_level_cfg.enabled:
        # Secondary, leakage-illustration protocol: bypass subject-level
        # assignment entirely and pool+shuffle windows by position instead.
        wl_subjects = window_level_cfg.subjects if window_level_cfg.subjects is not None else config["subjects"]
        wl_split = apply_window_level_split(
            all_windows,
            subjects=wl_subjects,
            train_frac=window_level_cfg.train_frac,
            val_frac=window_level_cfg.val_frac,
            seed=window_level_cfg.seed,
        )
        split = wl_split
        assignment_name = f"window_level(seed={window_level_cfg.seed})"
    else:
        assignment_path = find_assignment_config(config["assignment"])
        assignment = load_assignment(assignment_path)
        split = apply_subject_level_split(all_windows, assignment)
        assignment_name = assignment.name

    if verbose:
        print(f"[{name}] assignment={assignment_name}  seed={seed}")
        for role, bundle in (("train", split.train), ("val", split.val), ("test", split.test)):
            s = bundle_summary(bundle)
            print(f"  {role:5} n_windows={s['n_windows']:>6}  subjects={s['subjects']}")

    scaler = fit_target_scaler(split.train)
    if verbose:
        print(f"  target scaler fit on train only: mean={scaler.mean:.2f} std={scaler.std:.2f}")

    # Negative control: shuffle training labels only, AFTER fitting the scaler
    # on the true (unshuffled) distribution -- the scaler's mean/std would be
    # identical either way (shuffling preserves the marginal distribution of
    # y), but fitting it first makes explicit that nothing about the scaler
    # itself is part of this negative control; only the X-y pairing fed to
    # the training loader is corrupted. Validation/test bundles are never
    # touched, so reported val/test metrics always reflect true labels.
    neg_cfg = NegativeControlConfig.from_dict(config.get("negative_control"))
    train_bundle = split.train
    if neg_cfg.shuffle_train_labels:
        shuffle_seed = neg_cfg.shuffle_seed if neg_cfg.shuffle_seed is not None else seed
        train_bundle = shuffle_bundle_labels(split.train, seed=shuffle_seed)
        if verbose:
            print(f"  NEGATIVE CONTROL: training labels shuffled (shuffle_seed={shuffle_seed})")

    batch_size = config["training"].get("batch_size", 64)
    train_loader = make_loader(train_bundle, scaler, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(split.val, scaler, batch_size=batch_size, shuffle=False, seed=seed)

    model = build_model(config["model"])
    loss_fn = build_loss_fn(config["loss"])
    optimizer = build_optimizer(model, config["training"])
    epochs = config["training"].get("epochs", 30)

    aug_cfg = AugmentationConfig.from_dict(config.get("augmentation"))
    paired_cfg = PairedTrainingConfig.from_dict(config.get("paired_training"))
    consistency_fn = ConsistencyLoss() if paired_cfg.consistency_lambda > 0 else None
    aug_generator = None
    if aug_cfg.enabled:
        aug_generator = torch.Generator()
        aug_generator.manual_seed(seed)

    if verbose:
        print(f"  training for {epochs} epochs (loss={config['loss'].get('type', 'mse')}) ...")
        if aug_cfg.enabled:
            print(
                f"  augmentation enabled (pairing={paired_cfg.pairing_strategy}): "
                f"noise_std={aug_cfg.noise_std}, scale_range={aug_cfg.amplitude_scale_range}, "
                f"offset_range={aug_cfg.offset_range}, dropout_fraction={aug_cfg.dropout_fraction}"
            )
            print(
                f"  paired training: extra_data_loss={paired_cfg.extra_data_loss}, "
                f"consistency_lambda={paired_cfg.consistency_lambda}"
            )
    history = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        epochs,
        scaler,
        aug_cfg=aug_cfg,
        paired_cfg=paired_cfg,
        consistency_fn=consistency_fn,
        aug_generator=aug_generator,
        verbose=verbose,
    )

    results = {"name": name, "assignment": assignment_name, "seed": seed, "best_epoch": history.best_epoch}

    # Always compute residual coverage at a fixed reference delta (1.0, in
    # standardized-target units) so every config -- robust-loss or not -- is
    # directly comparable on "how many residuals would a Huber loss with this
    # delta actually treat differently from MSE". If this config's own loss is
    # huber with a different delta, also report coverage at that exact value.
    residual_deltas = {1.0}
    if config["loss"].get("type") == "huber":
        residual_deltas.add(float(config["loss"].get("delta", 1.0)))

    for role, bundle in (("train", split.train), ("val", split.val), ("test", split.test)):
        results[role] = _evaluate_split(model, bundle, scaler, batch_size, seed, residual_deltas=tuple(sorted(residual_deltas)))

    if verbose:
        for role in ("train", "val", "test"):
            for delta, cov in results[role]["residual_coverage"].items():
                print(
                    f"  [{role}] residual coverage @ delta={delta}: "
                    f"{cov['fraction_exceeding']:.1%} of residuals exceed delta "
                    f"({cov['n_exceeding']}/{cov['n']}), "
                    f"mean|residual|={cov['mean_abs_residual_scaled']:.3f}, "
                    f"max|residual|={cov['max_abs_residual_scaled']:.3f} (standardized units)"
                )

    _save_outputs(name, config, history, results, model, results_dir=results_dir, checkpoints_dir=checkpoints_dir)
    return results


def _save_outputs(
    name: str,
    config: dict,
    history: TrainingHistory,
    results: dict,
    model: nn.Module,
    results_dir=None,
    checkpoints_dir=None,
) -> None:
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_TABLES_DIR
    checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for role in ("train", "val", "test"):
        overall = results[role]["overall"]
        metrics_rows.append({"split": role, "subject": "ALL", **overall})
        for subject, m in results[role]["per_subject"].items():
            metrics_rows.append({"split": role, "subject": subject, **m})

    metrics_path = results_dir / f"{name}_metrics.csv"
    _write_csv(metrics_path, metrics_rows)

    history_path = results_dir / f"{name}_history.csv"
    history_rows = [
        {"epoch": i + 1, "train_loss": tl, "val_loss": vl, "val_mae_bpm": vm}
        for i, (tl, vl, vm) in enumerate(zip(history.train_loss, history.val_loss, history.val_mae_bpm))
    ]
    _write_csv(history_path, history_rows)

    residual_rows = []
    for role in ("train", "val", "test"):
        for delta, cov in results[role]["residual_coverage"].items():
            residual_rows.append({"split": role, **cov})
    residual_path = results_dir / f"{name}_residual_coverage.csv"
    _write_csv(residual_path, residual_rows)

    checkpoint_path = checkpoints_dir / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    meta_path = checkpoints_dir / f"{name}_config.json"
    with open(meta_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  saved: {metrics_path}")
    print(f"  saved: {history_path}")
    print(f"  saved: {residual_path}")
    print(f"  saved: {checkpoint_path} (gitignored, local only)")


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

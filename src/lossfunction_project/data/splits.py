"""Subject-level and window-level train/val/test splitting.

Two parallel protocols are produced from the same window pool (see windowing.py):

- **Subject-level (primary)**: entire subjects are assigned to train/val/test, so no
  window from a validation or test subject's recording ever appears in training. The
  default assignment (S1,S2,S3=train, S4=val, S5=test) is reused unchanged by every
  experiment config, so configs are directly comparable. Two additional role-reshuffle
  assignments (same 5 subjects, see configs/subject_assignments/) exist only for a
  sensitivity check -- they are not extra "folds" to average over, but a way to check
  whether conclusions depend on which subject plays which role.

- **Window-level (secondary, leakage-illustration only)**: all windows from all
  subjects are pooled and split by position after shuffling, ignoring subject
  identity. Because consecutive windows overlap 75% (label_shift_s=2s vs
  label_window_s=8s), near-duplicate windows can end up on both sides of this split.
  This protocol exists specifically to demonstrate that risk, not as a method to use
  for real evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml

from lossfunction_project.data.windowing import SubjectWindows, load_cached_windows
from lossfunction_project.utils.config import PROJECT_ROOT

CONFIGS_DIR = PROJECT_ROOT / "configs" / "subject_assignments"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


@dataclass
class SubjectAssignment:
    name: str
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]

    def all_subjects(self) -> List[str]:
        return self.train_subjects + self.val_subjects + self.test_subjects

    def validate(self) -> None:
        all_subj = self.all_subjects()
        if len(all_subj) != len(set(all_subj)):
            raise ValueError(
                f"Assignment '{self.name}' assigns a subject to more than one role: {all_subj}"
            )


def load_assignment(path: str | Path) -> SubjectAssignment:
    """Load one subject-assignment YAML config (see configs/subject_assignments/)."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    assignment = SubjectAssignment(
        name=raw["name"],
        train_subjects=list(raw["train_subjects"]),
        val_subjects=list(raw["val_subjects"]),
        test_subjects=list(raw["test_subjects"]),
    )
    assignment.validate()
    return assignment


def find_assignment_config(name: str, configs_dir: Path | None = None) -> Path:
    """Locate the assignment YAML whose internal `name:` field matches `name`.

    File stems under configs/subject_assignments/ (e.g. "assignment_default.yaml")
    are not assumed to match the `name` an experiment config refers to (e.g.
    "default"); this looks it up by content instead of guessing a filename.
    """
    for path in list_assignment_configs(configs_dir).values():
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        if raw.get("name") == name:
            return path
    raise FileNotFoundError(
        f"No assignment config with name='{name}' found under "
        f"{Path(configs_dir) if configs_dir is not None else CONFIGS_DIR}"
    )


def list_assignment_configs(configs_dir: Path | None = None) -> Dict[str, Path]:
    """Discover assignment YAML files under configs/subject_assignments/ (by stem)."""
    base = Path(configs_dir) if configs_dir is not None else CONFIGS_DIR
    return {p.stem: p for p in sorted(base.glob("*.yaml"))}


def load_all_cached_windows(
    subjects: Sequence[str], processed_dir=None
) -> Dict[str, SubjectWindows]:
    """Load the windowed-data cache (scripts/run_windowing.py output) for several subjects."""
    return {s: load_cached_windows(s, processed_dir) for s in subjects}


@dataclass
class ArrayBundle:
    """Concatenated window arrays for one split (train, val, or test)."""

    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray  # dtype=object (strings), kept per-window for traceability
    window_start_s: np.ndarray
    label_index: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _concat_subjects(all_windows: Dict[str, SubjectWindows], subjects: Sequence[str]) -> ArrayBundle:
    if not subjects:
        return ArrayBundle(
            X=np.empty((0, 0), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            subject_id=np.empty((0,), dtype=object),
            window_start_s=np.empty((0,), dtype=np.float32),
            label_index=np.empty((0,), dtype=np.int64),
        )
    Xs, ys, subs, starts, idxs = [], [], [], [], []
    for s in subjects:
        w = all_windows[s]
        Xs.append(w.X)
        ys.append(w.y)
        subs.append(np.full(len(w.y), s, dtype=object))
        starts.append(w.window_start_s)
        idxs.append(w.label_index)
    return ArrayBundle(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        subject_id=np.concatenate(subs, axis=0),
        window_start_s=np.concatenate(starts, axis=0),
        label_index=np.concatenate(idxs, axis=0),
    )


@dataclass
class SubjectLevelSplitArrays:
    assignment_name: str
    train: ArrayBundle
    val: ArrayBundle
    test: ArrayBundle


def apply_subject_level_split(
    all_windows: Dict[str, SubjectWindows],
    assignment: SubjectAssignment,
) -> SubjectLevelSplitArrays:
    """Assemble concatenated train/val/test arrays for a given subject assignment."""
    missing = [s for s in assignment.all_subjects() if s not in all_windows]
    if missing:
        raise KeyError(
            f"Assignment '{assignment.name}' references subjects not present "
            f"in all_windows: {missing}"
        )
    return SubjectLevelSplitArrays(
        assignment_name=assignment.name,
        train=_concat_subjects(all_windows, assignment.train_subjects),
        val=_concat_subjects(all_windows, assignment.val_subjects),
        test=_concat_subjects(all_windows, assignment.test_subjects),
    )


@dataclass
class WindowLevelSplitArrays:
    seed: int
    train: ArrayBundle
    val: ArrayBundle
    test: ArrayBundle


def apply_window_level_split(
    all_windows: Dict[str, SubjectWindows],
    subjects: Sequence[str] | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> WindowLevelSplitArrays:
    """Pool windows from `subjects` (default: all available), shuffle, and split by
    position -- ignoring subject identity and the 2 s shift between overlapping
    windows. Included only to illustrate the leakage risk of window-level splitting
    in physiological-signal datasets; not recommended for real evaluation.
    """
    if subjects is None:
        subjects = sorted(all_windows.keys())
    pooled = _concat_subjects(all_windows, subjects)
    n = len(pooled)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def _select(idx: np.ndarray) -> ArrayBundle:
        return ArrayBundle(
            X=pooled.X[idx],
            y=pooled.y[idx],
            subject_id=pooled.subject_id[idx],
            window_start_s=pooled.window_start_s[idx],
            label_index=pooled.label_index[idx],
        )

    return WindowLevelSplitArrays(
        seed=seed, train=_select(train_idx), val=_select(val_idx), test=_select(test_idx)
    )


def save_subject_level_split_index(assignment: SubjectAssignment, out_dir: Path | None = None) -> Path:
    """Persist a subject-level split as a small JSON index (subject -> role)."""
    out_dir = Path(out_dir) if out_dir is not None else SPLITS_DIR / "subject_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{assignment.name}.json"
    payload = {
        "protocol": "subject_level",
        "assignment_name": assignment.name,
        "train_subjects": assignment.train_subjects,
        "val_subjects": assignment.val_subjects,
        "test_subjects": assignment.test_subjects,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def save_window_level_split_index(split: WindowLevelSplitArrays, out_dir: Path | None = None) -> Path:
    """Persist a window-level split as a JSON index of (subject, label_index) pairs."""
    out_dir = Path(out_dir) if out_dir is not None else SPLITS_DIR / "window_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pooled_seed{split.seed}.json"

    def _pairs(bundle: ArrayBundle):
        return [[str(s), int(i)] for s, i in zip(bundle.subject_id, bundle.label_index)]

    payload = {
        "protocol": "window_level",
        "seed": split.seed,
        "train": _pairs(split.train),
        "val": _pairs(split.val),
        "test": _pairs(split.test),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def bundle_summary(bundle: ArrayBundle) -> dict:
    """Small sanity-check summary for a single split bundle."""
    if len(bundle) == 0:
        return {"n_windows": 0, "n_subjects": 0, "subjects": [], "y_bpm_mean": None}
    subjects = sorted(set(bundle.subject_id.tolist()))
    return {
        "n_windows": len(bundle),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "y_bpm_mean": round(float(bundle.y.mean()), 1),
    }


def shuffle_bundle_labels(bundle: ArrayBundle, seed: int) -> ArrayBundle:
    """Negative control: randomly permute `y` only, keeping X (and subject_id,
    window_start_s, label_index) in their original order.

    This deliberately breaks the X-y correspondence for training: each window
    ends up paired with a different, randomly chosen window's heart-rate
    label. If a model trained this way still achieves low error on a true
    (unshuffled) validation/test set, that would indicate some leakage or
    shortcut unrelated to the actual PPG-to-HR relationship is inflating
    performance; the expected, "healthy" outcome is that error collapses
    toward predicting the training-label mean.

    A plain random permutation (not a strict derangement) is used -- with
    thousands of training windows, the handful of samples that might by
    chance keep their own label is negligible and does not meaningfully
    weaken the probe. The permutation preserves the marginal distribution of
    y exactly (same values, reassigned to different windows), so a target
    scaler fit before vs. after this shuffle has identical mean/std; by
    convention, the scaler is always fit on the unshuffled bundle first (see
    training/train_loop.py), and shuffling is applied only to the copy fed to
    the training DataLoader.
    """
    rng = np.random.default_rng(seed)
    permuted_y = rng.permutation(bundle.y)
    return ArrayBundle(
        X=bundle.X,
        y=permuted_y,
        subject_id=bundle.subject_id,
        window_start_s=bundle.window_start_s,
        label_index=bundle.label_index,
    )

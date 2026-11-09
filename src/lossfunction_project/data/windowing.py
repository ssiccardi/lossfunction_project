"""Builds fixed-length PPG windows aligned to each heart-rate label.

For PPG-DaLiA, label i (0-indexed) is the heart rate computed by the dataset
authors from an 8 s window of chest ECG ending at time
``t_i = label_window_s + i * label_shift_s`` seconds into the recording. This
module reconstructs the matching 8 s window of *wrist PPG* (BVP) ending at the
same ``t_i``, so that each window/label pair reflects "the last 8 s of signal
up to the moment the ground-truth heart rate was measured" — no information
from after ``t_i`` leaks into the window.

Consecutive windows overlap heavily (label_shift_s=2s vs label_window_s=8s, i.e.
75% overlap): this is an intrinsic property of the PPG-DaLiA labeling protocol,
not a choice made here. It is exactly why the window-level pooled/shuffled split
(see splits.py) is expected to leak information between train and test, which is
the point of including it as a leakage-illustration comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lossfunction_project.data.extract import SubjectArrays


@dataclass
class SubjectWindows:
    """Windowed PPG data for one subject."""

    subject: str
    X: np.ndarray  # (n_windows, window_len), float32
    y: np.ndarray  # (n_windows,), float32, heart rate in bpm
    window_start_s: np.ndarray  # (n_windows,), float32, window start time in seconds
    label_index: np.ndarray  # (n_windows,), int64, original index into arrays.label


def window_length_samples(arrays: SubjectArrays) -> int:
    """Number of BVP samples in one label window (8 s * 64 Hz = 512 by default)."""
    length = arrays.label_window_s * arrays.fs_bvp
    if not float(length).is_integer():
        raise ValueError(
            f"label_window_s ({arrays.label_window_s}) * fs_bvp ({arrays.fs_bvp}) "
            f"is not an integer number of samples ({length}); windowing assumes it is."
        )
    return int(length)


def build_subject_windows(
    arrays: SubjectArrays,
    normalize: bool = True,
    eps: float = 1e-8,
) -> SubjectWindows:
    """Build all valid label-aligned PPG windows for one subject.

    Windows that would extend before the start of the recording (only possible
    for the very first label, if ever) are dropped rather than zero-padded, to
    avoid introducing artificial flat segments into the training data.

    Parameters
    ----------
    normalize:
        If True, each window is independently z-score normalized (zero mean,
        unit variance) using only its own samples. This is a per-window
        operation, computed after windowing and *before* any augmentation, so
        it never uses information from outside the window (no leakage) and is
        applied identically regardless of which split a window ends up in.
    """
    win_len = window_length_samples(arrays)
    fs = arrays.fs_bvp
    n_labels = len(arrays.label)

    X_list = []
    y_list = []
    start_list = []
    idx_list = []

    for i in range(n_labels):
        t_end_s = arrays.label_window_s + i * arrays.label_shift_s
        end_sample = int(round(t_end_s * fs))
        start_sample = end_sample - win_len

        if start_sample < 0:
            continue  # window would start before the recording began
        if end_sample > len(arrays.bvp):
            continue  # window would run past the end of the recorded signal

        window = arrays.bvp[start_sample:end_sample].astype(np.float32)
        if len(window) != win_len:
            continue  # defensive: should not happen given the checks above

        if normalize:
            mean = window.mean()
            std = window.std()
            window = (window - mean) / (std + eps)

        X_list.append(window)
        y_list.append(arrays.label[i])
        start_list.append(start_sample / fs)
        idx_list.append(i)

    if not X_list:
        raise ValueError(f"No valid windows could be built for subject {arrays.subject}.")

    return SubjectWindows(
        subject=arrays.subject,
        X=np.stack(X_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        window_start_s=np.asarray(start_list, dtype=np.float32),
        label_index=np.asarray(idx_list, dtype=np.int64),
    )


def windows_summary(windows: SubjectWindows) -> dict:
    """Small sanity-check summary, surfaced to the CLI rather than silently assumed."""
    return {
        "subject": windows.subject,
        "n_windows": int(windows.X.shape[0]),
        "window_len": int(windows.X.shape[1]),
        "y_bpm_min": round(float(windows.y.min()), 1),
        "y_bpm_mean": round(float(windows.y.mean()), 1),
        "y_bpm_max": round(float(windows.y.max()), 1),
    }


def load_and_window_subject(
    subject: str,
    processed_dir=None,
    normalize: bool = True,
) -> SubjectWindows:
    """Convenience wrapper: load a subject's cached .npz and build its windows."""
    from lossfunction_project.data.extract import load_subject_npz

    arrays = load_subject_npz(subject, processed_dir)
    return build_subject_windows(arrays, normalize=normalize)


def load_cached_windows(subject: str, processed_dir=None) -> SubjectWindows:
    """Load a subject's windows from the cache written by scripts/run_windowing.py.

    Faster than re-windowing from the raw BVP signal every time (used by splits.py
    and training code downstream).
    """
    from lossfunction_project.utils.config import PROCESSED_DIR

    base = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    in_path = base / f"{subject}_windows.npz"
    with np.load(in_path, allow_pickle=False) as npz:
        return SubjectWindows(
            subject=str(npz["subject"]),
            X=npz["X"],
            y=npz["y"],
            window_start_s=npz["window_start_s"],
            label_index=npz["label_index"],
        )


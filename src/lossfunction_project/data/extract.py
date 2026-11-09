"""Extracts only the fields this project needs from a PPG-DaLiA subject .pkl file.

The raw ``S{n}.pkl`` files are ~1.3 GB each (they include 700 Hz chest ECG/ACC/Resp
that this project does not use). This module reads a file once and writes a much
smaller ``.npz`` cache containing only:

- ``bvp``: wrist PPG signal (Empatica E4 BVP channel), sampled at 64 Hz.
- ``label``: ground-truth heart rate (bpm), one value per 2 s, each computed by
  PPG-DaLiA from an 8 s window of the chest RespiBAN ECG (see PPG-DaLiA
  documentation; window/shift values are stored as metadata, not hardcoded
  downstream).
- ``activity``: coarse activity-type signal at 4 Hz (kept for possible future use,
  not required by the core regression task).

Everything downstream (windowing, splits, training) reads only the cached
``.npz`` files, never the raw ``.pkl`` directly.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lossfunction_project.utils.config import processed_npz_path, subject_pkl_path

# PPG-DaLiA protocol constants (confirmed against the raw files during planning).
BVP_SAMPLE_RATE_HZ = 64.0
ACTIVITY_SAMPLE_RATE_HZ = 4.0
LABEL_WINDOW_S = 8.0
LABEL_SHIFT_S = 2.0


@dataclass
class SubjectArrays:
    """Minimal in-memory representation of one subject's extracted data."""

    subject: str
    bvp: np.ndarray
    label: np.ndarray
    activity: np.ndarray
    fs_bvp: float = BVP_SAMPLE_RATE_HZ
    fs_activity: float = ACTIVITY_SAMPLE_RATE_HZ
    label_window_s: float = LABEL_WINDOW_S
    label_shift_s: float = LABEL_SHIFT_S


def load_subject_pkl(subject: str, raw_dir: str | Path | None = None) -> SubjectArrays:
    """Load one subject's raw .pkl and keep only the fields this project needs."""
    pkl_path = subject_pkl_path(subject, raw_dir)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float32).reshape(-1)
    label = np.asarray(data["label"], dtype=np.float32).reshape(-1)
    activity = np.asarray(data["activity"], dtype=np.float32).reshape(-1)

    return SubjectArrays(subject=subject, bvp=bvp, label=label, activity=activity)


def save_subject_npz(arrays: SubjectArrays, processed_dir: str | Path | None = None) -> Path:
    """Write a subject's extracted arrays to a compact .npz cache."""
    out_path = processed_npz_path(arrays.subject, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        subject=arrays.subject,
        bvp=arrays.bvp,
        label=arrays.label,
        activity=arrays.activity,
        fs_bvp=arrays.fs_bvp,
        fs_activity=arrays.fs_activity,
        label_window_s=arrays.label_window_s,
        label_shift_s=arrays.label_shift_s,
    )
    return out_path


def load_subject_npz(subject: str, processed_dir: str | Path | None = None) -> SubjectArrays:
    """Load a subject's cached .npz (fast path for everything downstream of extraction)."""
    in_path = processed_npz_path(subject, processed_dir)
    with np.load(in_path, allow_pickle=False) as npz:
        return SubjectArrays(
            subject=str(npz["subject"]),
            bvp=npz["bvp"],
            label=npz["label"],
            activity=npz["activity"],
            fs_bvp=float(npz["fs_bvp"]),
            fs_activity=float(npz["fs_activity"]),
            label_window_s=float(npz["label_window_s"]),
            label_shift_s=float(npz["label_shift_s"]),
        )


def sanity_summary(arrays: SubjectArrays) -> dict:
    """Basic consistency checks, surfaced to the CLI rather than silently assumed.

    Cross-checks the recording duration implied by the BVP signal against the
    duration implied by the number of labels, given the known window/shift
    convention. A large mismatch would indicate a wrong assumption about the
    label timing and should be investigated before windowing.
    """
    bvp_duration_s = len(arrays.bvp) / arrays.fs_bvp
    # n-th label (0-indexed) is derived from the window ending at
    # label_window_s + n * label_shift_s seconds into the recording.
    label_duration_s = arrays.label_window_s + (len(arrays.label) - 1) * arrays.label_shift_s
    return {
        "subject": arrays.subject,
        "n_bvp_samples": int(len(arrays.bvp)),
        "n_labels": int(len(arrays.label)),
        "n_activity_samples": int(len(arrays.activity)),
        "bvp_duration_s": round(bvp_duration_s, 1),
        "label_implied_duration_s": round(label_duration_s, 1),
        "duration_mismatch_s": round(bvp_duration_s - label_duration_s, 1),
        "label_bpm_min": round(float(np.min(arrays.label)), 1),
        "label_bpm_max": round(float(np.max(arrays.label)), 1),
        "label_bpm_mean": round(float(np.mean(arrays.label)), 1),
    }


def extract_subject(subject: str, raw_dir=None, processed_dir=None) -> dict:
    """Full pipeline step: raw .pkl -> cached .npz, returning a sanity summary."""
    arrays = load_subject_pkl(subject, raw_dir)
    save_subject_npz(arrays, processed_dir)
    return sanity_summary(arrays)

"""Project-wide path and config helpers.

Raw PPG-DaLiA data lives outside the repository. Its location is resolved,
in order of precedence, from:
1. an explicit function argument,
2. the ``PPGDALIA_RAW_DIR`` environment variable,
3. a hardcoded local default (convenient for this machine only; never relied
   upon by anything meant to run elsewhere).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Local convenience default; irrelevant on any other machine, where the
# PPGDALIA_RAW_DIR environment variable should be set instead.
_LOCAL_DEFAULT_RAW_DIR = r"E:\fitmi\ppgdalia\PPG_FieldStudy"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_raw_data_dir(override: str | os.PathLike | None = None) -> Path:
    """Return the directory containing per-subject PPG-DaLiA folders (S1, S2, ...)."""
    if override is not None:
        raw_dir = Path(override)
    else:
        raw_dir = Path(os.environ.get("PPGDALIA_RAW_DIR", _LOCAL_DEFAULT_RAW_DIR))
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"PPG-DaLiA raw data directory not found: {raw_dir}. "
            "Set the PPGDALIA_RAW_DIR environment variable, or pass an explicit path, "
            "to the folder containing S1/, S2/, ... subject subfolders."
        )
    return raw_dir


def subject_pkl_path(subject: str, raw_dir: str | os.PathLike | None = None) -> Path:
    """Path to a given subject's raw .pkl file, e.g. subject='S1'."""
    base = get_raw_data_dir(raw_dir)
    return base / subject / f"{subject}.pkl"


def processed_npz_path(subject: str, processed_dir: str | os.PathLike | None = None) -> Path:
    """Path to a given subject's cached .npz file, e.g. subject='S1'."""
    base = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    return base / f"{subject}.npz"


def load_experiment_config(path: str | os.PathLike) -> dict:
    """Load an experiment config YAML (e.g. configs/baseline.yaml) as a plain dict."""
    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    required = {"name", "assignment", "subjects", "model", "loss", "training"}
    missing = required - config.keys()
    if missing:
        raise KeyError(f"Experiment config {path} is missing required keys: {sorted(missing)}")
    return config

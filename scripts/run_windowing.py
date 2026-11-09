#!/usr/bin/env python
"""CLI: build label-aligned PPG windows for the given subjects and cache them.

Usage:
    python scripts/run_windowing.py --subjects S1 S2 S3 S4 S5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lossfunction_project.data.windowing import load_and_window_subject, windows_summary
from lossfunction_project.utils.config import PROCESSED_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S1", "S2", "S3", "S4", "S5"],
        help="Subject IDs to window (must already be extracted; see run_extraction.py).",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory with per-subject .npz caches and where window caches are written "
        "(default: data/processed).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-window z-score normalization (default: normalization is on).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    processed_dir = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR
    normalize = not args.no_normalize

    rows = []
    for subject in args.subjects:
        print(f"[{subject}] windowing (normalize={normalize}) ...", flush=True)
        windows = load_and_window_subject(subject, processed_dir=processed_dir, normalize=normalize)
        summary = windows_summary(windows)
        rows.append(summary)

        out_path = processed_dir / f"{subject}_windows.npz"
        np.savez_compressed(
            out_path,
            subject=windows.subject,
            X=windows.X,
            y=windows.y,
            window_start_s=windows.window_start_s,
            label_index=windows.label_index,
            normalized=normalize,
        )
        print(
            f"[{subject}] n_windows={summary['n_windows']} window_len={summary['window_len']} "
            f"bpm[min/mean/max]={summary['y_bpm_min']}/{summary['y_bpm_mean']}/{summary['y_bpm_max']} "
            f"-> {out_path.name}"
        )

    print("\nSummary:")
    header = f"{'subject':8} {'n_windows':>10} {'window_len':>11} {'bpm_mean':>9}"
    print(header)
    for row in rows:
        print(f"{row['subject']:8} {row['n_windows']:>10} {row['window_len']:>11} {row['y_bpm_mean']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

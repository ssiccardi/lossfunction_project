#!/usr/bin/env python
"""CLI: build compact .npz caches for the given subjects.

Usage:
    python scripts/run_extraction.py --subjects S1 S2 S3 S4 S5
    python scripts/run_extraction.py --subjects S1 S2 S3 S4 S5 --raw-dir "E:\\path\\to\\PPG_FieldStudy"
"""
from __future__ import annotations

import argparse
import sys

from lossfunction_project.data.extract import extract_subject


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S1", "S2", "S3", "S4", "S5"],
        help="Subject IDs to extract (default: project's default 5 subjects).",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override the PPGDALIA_RAW_DIR directory for this run.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override the output directory for .npz caches (default: data/processed).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = []
    for subject in args.subjects:
        print(f"[{subject}] extracting ...", flush=True)
        summary = extract_subject(subject, raw_dir=args.raw_dir, processed_dir=args.processed_dir)
        rows.append(summary)
        mismatch = summary["duration_mismatch_s"]
        flag = "  <-- CHECK: large mismatch" if abs(mismatch) > 30 else ""
        print(
            f"[{subject}] bvp_samples={summary['n_bvp_samples']} "
            f"n_labels={summary['n_labels']} "
            f"bvp_duration_s={summary['bvp_duration_s']} "
            f"label_implied_duration_s={summary['label_implied_duration_s']} "
            f"mismatch_s={mismatch} "
            f"bpm[min/mean/max]={summary['label_bpm_min']}/"
            f"{summary['label_bpm_mean']}/{summary['label_bpm_max']}{flag}"
        )

    print("\nSummary:")
    header = f"{'subject':8} {'n_bvp':>10} {'n_labels':>9} {'dur_s':>8} {'mismatch_s':>11} {'bpm_mean':>9}"
    print(header)
    for row in rows:
        print(
            f"{row['subject']:8} {row['n_bvp_samples']:>10} {row['n_labels']:>9} "
            f"{row['bvp_duration_s']:>8} {row['duration_mismatch_s']:>11} {row['label_bpm_mean']:>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

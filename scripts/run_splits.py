#!/usr/bin/env python
"""CLI: build and persist subject-level and window-level splits.

Reads the per-subject window caches produced by scripts/run_windowing.py, applies:
  - the fixed default subject assignment plus the 2 role-reshuffle variants
    (see configs/subject_assignments/) -- the primary, leakage-safe protocol, and
  - the pooled, shuffled window-level split -- the secondary, leakage-illustration
    protocol,
and writes a small JSON split-index file for each, under
data/processed/splits/{subject_level,window_level}/.

Usage:
    python scripts/run_splits.py
    python scripts/run_splits.py --subjects S1 S2 S3 S4 S5 --window-level-seed 42
"""
from __future__ import annotations

import argparse
import sys

from lossfunction_project.data.splits import (
    apply_subject_level_split,
    apply_window_level_split,
    bundle_summary,
    list_assignment_configs,
    load_all_cached_windows,
    load_assignment,
    save_subject_level_split_index,
    save_window_level_split_index,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S1", "S2", "S3", "S4", "S5"],
        help="Subjects to load windows for (must already be windowed; see run_windowing.py).",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory with per-subject *_windows.npz caches (default: data/processed).",
    )
    parser.add_argument(
        "--window-level-seed",
        type=int,
        default=42,
        help="Random seed for the pooled window-level split (default: 42).",
    )
    parser.add_argument(
        "--window-level-train-frac",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--window-level-val-frac",
        type=float,
        default=0.15,
    )
    return parser.parse_args(argv)


def _print_bundle_line(role: str, bundle) -> None:
    s = bundle_summary(bundle)
    subjects_str = ",".join(s["subjects"]) if s["subjects"] else "(none)"
    print(
        f"    {role:5} n_windows={s['n_windows']:>6}  n_subjects={s['n_subjects']}  "
        f"subjects=[{subjects_str}]  bpm_mean={s['y_bpm_mean']}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Loading cached windows for subjects: {args.subjects}")
    all_windows = load_all_cached_windows(args.subjects, processed_dir=args.processed_dir)
    for s, w in all_windows.items():
        print(f"  {s}: {len(w.y)} windows")

    print("\n=== Subject-level splits (primary protocol) ===")
    assignment_configs = list_assignment_configs()
    if not assignment_configs:
        print("  WARNING: no assignment YAML files found under configs/subject_assignments/")
    for name, path in assignment_configs.items():
        assignment = load_assignment(path)
        split = apply_subject_level_split(all_windows, assignment)

        overlap_train_val = set(split.train.subject_id.tolist()) & set(split.val.subject_id.tolist())
        overlap_train_test = set(split.train.subject_id.tolist()) & set(split.test.subject_id.tolist())
        overlap_val_test = set(split.val.subject_id.tolist()) & set(split.test.subject_id.tolist())
        assert not overlap_train_val and not overlap_train_test and not overlap_val_test, (
            f"Subject overlap detected in assignment '{name}' -- this should be impossible "
            "and indicates a bug in splits.py."
        )

        print(f"\n  Assignment: {name}  (source: {path.name})")
        _print_bundle_line("train", split.train)
        _print_bundle_line("val", split.val)
        _print_bundle_line("test", split.test)

        out_path = save_subject_level_split_index(assignment)
        print(f"    -> saved split index: {out_path}")

    print("\n=== Window-level pooled split (secondary, leakage-illustration only) ===")
    window_split = apply_window_level_split(
        all_windows,
        train_frac=args.window_level_train_frac,
        val_frac=args.window_level_val_frac,
        seed=args.window_level_seed,
    )
    print(f"  seed={window_split.seed}  train_frac={args.window_level_train_frac}  "
          f"val_frac={args.window_level_val_frac}")
    _print_bundle_line("train", window_split.train)
    _print_bundle_line("val", window_split.val)
    _print_bundle_line("test", window_split.test)
    print(
        "  NOTE: unlike the subject-level splits above, each split here is expected to "
        "mix multiple subjects, and near-duplicate overlapping windows can end up on "
        "both sides of the split -- this is the leakage risk the protocol is meant to "
        "illustrate, not an evaluation method to trust."
    )

    out_path = save_window_level_split_index(window_split)
    print(f"  -> saved split index: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

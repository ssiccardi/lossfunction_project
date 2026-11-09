#!/usr/bin/env python
"""CLI: run one experiment config end-to-end (load data -> train -> evaluate -> save).

Usage:
    python scripts/run_experiment.py --config configs/baseline.yaml
    python scripts/run_experiment.py --config configs/baseline.yaml --seed 123
    python scripts/run_experiment.py --config configs/baseline_expanded.yaml --assignment expanded_reshuffle_1
"""
from __future__ import annotations

import argparse
import sys

from lossfunction_project.training.train_loop import run_experiment
from lossfunction_project.utils.config import load_experiment_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment config YAML.")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory with per-subject *_windows.npz caches (default: data/processed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config's training.seed, for a seed-variability check. "
        "Results are saved under a '{name}_seed{N}' name, so the original config's "
        "results are never overwritten.",
    )
    parser.add_argument(
        "--assignment",
        default=None,
        help="Override the config's subject assignment (by its internal 'name:' field, "
        "e.g. 'expanded_reshuffle_1'), for a subject-assignment sensitivity check. "
        "Results are saved under a '{name}_{assignment}' name, so the original "
        "config's results are never overwritten.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)

    if args.seed is not None:
        config = dict(config, training=dict(config["training"], seed=args.seed))
        config["name"] = f"{config['name']}_seed{args.seed}"

    if args.assignment is not None:
        config = dict(config, assignment=args.assignment)
        config["name"] = f"{config['name']}_{args.assignment}"

    print(f"=== Running experiment: {config['name']} (config: {args.config}) ===\n")
    results = run_experiment(config, processed_dir=args.processed_dir, verbose=True)

    print(f"\n=== Final metrics: {config['name']} (best epoch {results['best_epoch']}) ===")
    header = f"{'split':6} {'n':>6} {'mae':>8} {'rmse':>8} {'pearson_r':>10}"
    print(header)
    for role in ("train", "val", "test"):
        m = results[role]["overall"]
        pr = m["pearson_r"]
        pr_str = f"{pr:.3f}" if pr == pr else "nan"  # NaN check without importing math/np here
        print(f"{role:6} {m['n']:>6} {m['mae']:>8.2f} {m['rmse']:>8.2f} {pr_str:>10}")

        for subject, sm in results[role]["per_subject"].items():
            spr = sm["pearson_r"]
            spr_str = f"{spr:.3f}" if spr == spr else "nan"
            print(f"  {subject:4} {sm['n']:>6} {sm['mae']:>8.2f} {sm['rmse']:>8.2f} {spr_str:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

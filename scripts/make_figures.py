#!/usr/bin/env python
"""CLI: generate the summary comparison figures referenced by REPORT.md.

Reads the per-run metrics CSVs already saved under results/tables/ (one run =
one scripts/run_experiment.py invocation) and produces two bar charts
comparing pooled TEST MAE and pooled TEST Pearson r across the primary
8-subject ablation configs and the 3 negative controls. Training-curve plots
are intentionally not produced here.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; this script only writes PNG files
import matplotlib.pyplot as plt

from lossfunction_project.utils.config import PROJECT_ROOT

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# (config name as saved by run_experiment.py, display label, is_negative_control)
CONFIGS = [
    ("baseline_expanded", "Baseline", False),
    ("robust_loss_only_expanded", "Robust loss", False),
    ("augmentation_only_expanded", "Augmentation", False),
    ("consistency_only_expanded", "Consistency", False),
    ("full_method_expanded", "Full method", False),
    ("negative_control_shuffled_labels", "NC: shuffled\nlabels", True),
    ("negative_control_random_pairing", "NC: random\npairing", True),
    ("negative_control_wrong_pairing", "NC: wrong\npairing", True),
]

METHOD_COLOR = "#4C72B0"
NEGATIVE_CONTROL_COLOR = "#C44E52"


def read_pooled_test_metric(config_name: str, metric: str, tables_dir: Path) -> float:
    """Read the pooled ("ALL") test-split value of `metric` for one run."""
    path = tables_dir / f"{config_name}_metrics.csv"
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["subject"] == "ALL":
                return float(row[metric])
    raise ValueError(f"No pooled test row found in {path}")


def make_bar_chart(
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    tables_dir: Path,
    higher_is_better: bool,
) -> None:
    labels = [label for _, label, _ in CONFIGS]
    values = [read_pooled_test_metric(name, metric, tables_dir) for name, _, _ in CONFIGS]
    colors = [NEGATIVE_CONTROL_COLOR if is_nc else METHOD_COLOR for _, _, is_nc in CONFIGS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors)

    # Reference line at the baseline's own value, so every other bar can be
    # read as "better/worse than plain standard-loss training" at a glance.
    # Label placed just outside the right edge of the axes (clip_on=False) so
    # it never overlaps a tall bar, regardless of which bars cross the line.
    baseline_value = values[0]
    ax.axhline(baseline_value, color="gray", linestyle="--", linewidth=1)
    ax.text(
        1.01,
        baseline_value,
        "baseline",
        color="gray",
        va="center",
        ha="left",
        fontsize=8,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    arrow = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel(f"Pooled test set, 8-subject default assignment ({arrow})")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables/ location.")
    parser.add_argument("--figures-dir", default=None, help="Override results/figures/ location.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tables_dir = Path(args.tables_dir) if args.tables_dir else RESULTS_TABLES_DIR
    figures_dir = Path(args.figures_dir) if args.figures_dir else RESULTS_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    make_bar_chart(
        metric="mae",
        ylabel="MAE (bpm)",
        title="Pooled test MAE across configurations",
        out_path=figures_dir / "test_mae_comparison.png",
        tables_dir=tables_dir,
        higher_is_better=False,
    )
    make_bar_chart(
        metric="pearson_r",
        ylabel="Pearson r",
        title="Pooled test Pearson r across configurations",
        out_path=figures_dir / "test_pearson_r_comparison.png",
        tables_dir=tables_dir,
        higher_is_better=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

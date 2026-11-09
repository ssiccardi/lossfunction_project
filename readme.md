# PPG Heart-Rate Estimation: Custom Training Methodologies

This project builds a small, reproducible pipeline that estimates heart rate from
wrist PPG (photoplethysmography) signals. The point of the project is **not** to
produce the most accurate heart-rate estimator. It is to implement, and rigorously
evaluate, a set of custom training-time mechanisms — a hand-written robust loss
function, data augmentation, and a paired-consistency training scheme — and to
show, through ablations and negative controls, whether any observed improvement
genuinely comes from these mechanisms rather than from extra computation, data
leakage, or chance.

In other words: the reusable asset here is the training methodology and the
evidence around it, not any single trained model.

See [REPORT.md](REPORT.md) for the full write-up of methods, results, and
limitations. This README covers what the project does and how to reproduce it.

## Dataset

[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia), 8 subjects by
default (S1-S8). Input: wrist PPG (`signal.wrist.BVP`, 64 Hz). Target: heart rate
(`label`), provided by the dataset as one value every 2 s, computed from an 8 s
chest-ECG window.

Raw data is **not included** in this repository. See
[data/raw/README.md](data/raw/README.md) for where to download it and how to
point the pipeline at it.

## Method overview

| Config | Standard loss | Robust loss | Augmentation | Consistency term | Role |
|---|---|---|---|---|---|
| Baseline | Yes | | | | Reference point |
| Robust-loss only | | Yes | | | Ablation: full method minus augmentation/consistency |
| Augmentation-only | Yes | | Yes | | Ablation: augmented view used as extra training data, no consistency term |
| Consistency-only | Yes | | Yes | Yes | Ablation: full method minus the robust loss |
| Full method | | Yes | Yes | Yes | Core deliverable |

Three negative controls probe whether the above effects are genuine:

| Control | What it does |
|---|---|
| Shuffled-label probe | Trains on randomly permuted training labels; evaluation always uses true labels |
| Random pairing | Pairs each window with an augmented view of a *different*, randomly chosen window |
| Wrong pairing | Same idea, but with a fixed (not randomized) mismatch between windows |

Every configuration above shares the same model, data windows, and evaluation code;
only the loss function and/or batch construction differ between configs (see
`src/lossfunction_project/training/train_loop.py`). This is what lets any observed
difference in results be attributed to the training method itself.

Splitting is **subject-level** by default (whole subjects assigned to train/val/test,
so no window from a held-out subject's recording ever appears in training), with a
secondary **window-level** (pooled and shuffled) split used only to illustrate the
leakage risk of ignoring subject identity in this kind of data.

## Repository structure

```
configs/                   Experiment configs (YAML) and subject-assignment configs
data/
  raw/                     Not committed -- see data/raw/README.md
  processed/                Not committed -- per-subject .npz caches, built locally
src/lossfunction_project/
  data/                     Extraction, windowing, splitting, augmentations
  models/                   Shared 1D CNN backbone
  losses/                   Custom robust (Huber-style) and consistency losses
  training/                 Shared training loop, pairing strategies, dataset utils
  evaluation/               Regression metrics, residual-coverage diagnostics
  utils/                    Config/path helpers, seeding
scripts/
  run_extraction.py         Raw .pkl -> compact .npz cache, per subject
  run_windowing.py          Builds label-aligned PPG windows from the cache
  run_splits.py             Builds and inspects subject-level / window-level splits
  run_experiment.py         Runs one experiment config end-to-end
tests/                      Unit and integration tests (pytest)
results/
  tables/                   Per-run metrics, training history, residual coverage (CSV)
  figures/                  Summary plots referenced by REPORT.md
```

## Setup

Requires Python 3.9+.

```
python -m venv .venv
.venv\Scripts\activate          # on Windows; use source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
```

This installs the pinned dependencies (PyTorch, NumPy, SciPy, Matplotlib, PyYAML)
and the project package itself in editable mode.

## Running the pipeline

1. **Get the data.** Download PPG-DaLiA and set `PPGDALIA_RAW_DIR` to the folder
   containing the `S1/`, `S2/`, ... subfolders (see
   [data/raw/README.md](data/raw/README.md)).

2. **Extract and window the subjects you need:**

   ```
   python scripts/run_extraction.py --subjects S1 S2 S3 S4 S5 S6 S7 S8
   python scripts/run_windowing.py --subjects S1 S2 S3 S4 S5 S6 S7 S8
   ```

3. **(Optional) Inspect the splits:**

   ```
   python scripts/run_splits.py
   ```

4. **Run an experiment:**

   ```
   python scripts/run_experiment.py --config configs/baseline_expanded.yaml
   python scripts/run_experiment.py --config configs/full_method_expanded.yaml
   ```

   Every config in `configs/` can be run the same way. `run_experiment.py` also
   supports `--seed` (seed-variability check) and `--assignment` (subject-assignment
   sensitivity check) overrides without needing to create new config files; see
   `python scripts/run_experiment.py --help`.

Each run prints per-epoch training progress and final train/val/test metrics
(overall and per-subject), and saves a metrics table, a training-history log, and a
residual-coverage diagnostic under `results/tables/`, plus a model checkpoint
(locally only; not committed) under `checkpoints/`.

## Running the tests

```
pytest tests/
```

## Results

See [REPORT.md](REPORT.md) for the full set of results (ablations, negative
controls, subject-assignment sensitivity, window-level leakage comparison) and an
honest discussion of the method's limitations.

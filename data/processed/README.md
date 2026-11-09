# Processed data cache (not committed)

Populated by `scripts/run_extraction.py`. Contains one `S{n}.npz` per subject
with only the arrays needed for this project (wrist BVP signal, HR label,
activity, and sampling-rate metadata) — not the full PPG-DaLiA record, so
these files are small (a few MB) compared to the ~1.3 GB raw `.pkl` files.

Regenerate locally with:

```
python scripts/run_extraction.py --subjects S1 S2 S3 S4 S5 S6 S7 S8
```

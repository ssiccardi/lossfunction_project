# Raw data (not committed)

This project uses the **PPG-DaLiA** dataset:
https://archive.ics.uci.edu/dataset/495/ppg+dalia

The dataset is distributed under its own license/usage terms (see the page
above); this project's [LICENSE](../../LICENSE) covers only the code in this
repository, not the dataset itself.

Download `PPG_FieldStudy` and point `PPGDALIA_RAW_DIR` (see
`src/lossfunction_project/utils/config.py`) at the folder containing
`S1/`, `S2/`, ... subject subfolders, e.g.:

```
PPGDALIA_RAW_DIR=/path/to/PPG_FieldStudy
```

This project only uses 8 subjects by default (S1-S8) and only the wrist PPG
(`signal.wrist.BVP`) and heart-rate label (`label`) fields from each subject's
`.pkl` file; the chest/ECG/EDA/TEMP signals are not read.

Data is never committed to this repository.

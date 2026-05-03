# SCATS Preprocessing Outputs

## How to run

```bash
.\.venv\Scripts\python src/data/preprocess_scats.py
```

Optional flags:
- `--input "Info/Scats Data October 2006.xls"`
- `--output-dir "data/processed"`
- `--train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`
- `--sequence-length 8`
- `--skip-sequences`

## Generated files
- `cleaned_scats_long.csv`: cleaned long-format table (one SCATS site per timestamp row).
- `train.csv`, `validation.csv`, `test.csv`: chronological splits (no shuffle).
- `scats_sequences.npz`: optional LSTM/GRU-ready arrays.
- `metadata.json`: cleaning summary, split details, and output paths.

## Column meanings
- `timestamp`: combined datetime for each 15-minute interval.
- `date`: date portion from timestamp.
- `time`: time portion from timestamp.
- `scats_site`: SCATS site ID.
- `flow_15min`: observed traffic flow for the 15-minute interval.
- `flow_hour_simple`: `flow_15min * 4`.
- `flow_hour_rolling`: rolling 1-hour flow over current and previous 3 intervals per site.
- `day_of_week`: Monday=0 ... Sunday=6.
- `hour`: hour from timestamp.
- `minute`: minute from timestamp.
- `interval_index`: index from 0 to 95 across the day.

## Why chronological split
Time-series models must learn from past data to predict future data. Chronological split avoids leakage that can happen with random shuffling.

## Loading examples

```python
import pandas as pd
import numpy as np

train_df = pd.read_csv("data/processed/train.csv", parse_dates=["timestamp"])
seq = np.load("data/processed/scats_sequences.npz")
X_train, y_train = seq["X_train"], seq["y_train"]
```

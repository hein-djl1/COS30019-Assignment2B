from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    "flow_15min",
    "flow_hour_simple",
    "flow_hour_rolling",
    "day_of_week",
    "hour",
    "minute",
    "interval_index",
]


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and preprocess SCATS October 2006 traffic data into ML-ready files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Info") / "Scats Data October 2006.xls",
        help="Path to input .xls file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Output directory for processed files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip generation of scats_sequences.npz",
    )
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {total:.6f} "
            f"({train_ratio}, {val_ratio}, {test_ratio})."
        )
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be > 0.")


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Place 'Scats Data October 2006.xls' in Info/ or pass --input."
        )

    try:
        data = pd.read_excel(input_path, sheet_name="Data", header=1)
    except ValueError as exc:
        raise ValueError(
            "Workbook does not contain a 'Data' sheet. Please verify input format."
        ) from exc

    required_key_columns = {"SCATS Number", "Date"}
    missing = required_key_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Data sheet: {sorted(missing)}")

    return data


def to_long_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    interval_columns = [c for c in raw_df.columns if isinstance(c, str) and c.startswith("V") and c[1:].isdigit()]
    interval_columns = sorted(interval_columns, key=lambda x: int(x[1:]))

    if len(interval_columns) != 96:
        raise ValueError(
            "Expected 96 interval columns (V00..V95). "
            f"Found {len(interval_columns)}: {interval_columns[:5]}..."
        )

    keep_columns = ["SCATS Number", "Date"]
    id_frame = raw_df[keep_columns].rename(columns={"SCATS Number": "scats_site", "Date": "date_raw"})
    work = pd.concat([id_frame, raw_df[interval_columns]], axis=1)

    long_df = work.melt(
        id_vars=["scats_site", "date_raw"],
        value_vars=interval_columns,
        var_name="interval_col",
        value_name="flow_15min",
    )

    long_df["interval_index"] = long_df["interval_col"].str[1:].astype(int)
    return long_df.drop(columns=["interval_col"])


def clean_and_engineer(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows_before_cleaning = int(len(long_df))

    df = long_df.copy()
    df["scats_site"] = pd.to_numeric(df["scats_site"], errors="coerce")
    df["date_raw"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["flow_15min"] = pd.to_numeric(df["flow_15min"], errors="coerce")

    missing_before = {
        "scats_site": int(df["scats_site"].isna().sum()),
        "date_raw": int(df["date_raw"].isna().sum()),
        "flow_15min": int(df["flow_15min"].isna().sum()),
    }

    df = df.dropna(subset=["scats_site", "date_raw"])

    negative_rows = int((df["flow_15min"] < 0).sum())
    df = df[df["flow_15min"].isna() | (df["flow_15min"] >= 0)]

    df["date"] = df["date_raw"].dt.floor("D")
    df["timestamp"] = df["date"] + pd.to_timedelta(df["interval_index"] * 15, unit="minute")

    duplicate_count = int(df.duplicated(subset=["scats_site", "timestamp"]).sum())
    df = df.drop_duplicates(subset=["scats_site", "timestamp"], keep="first")

    df = df.sort_values(["scats_site", "timestamp"]).reset_index(drop=True)

    df["flow_hour_simple"] = df["flow_15min"] * 4
    df["flow_hour_rolling"] = (
        df.groupby("scats_site", group_keys=False)["flow_15min"]
        .rolling(window=4, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["scats_site"] = df["scats_site"].astype(int)

    ordered_columns = [
        "timestamp",
        "date",
        "time",
        "scats_site",
        "flow_15min",
        "flow_hour_simple",
        "flow_hour_rolling",
        "day_of_week",
        "hour",
        "minute",
        "interval_index",
    ]

    cleaned = df[ordered_columns].copy()
    cleaned = cleaned.sort_values(["scats_site", "timestamp"]).reset_index(drop=True)

    rows_after_cleaning = int(len(cleaned))
    missing_after = {col: int(cleaned[col].isna().sum()) for col in cleaned.columns}

    stats = {
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": rows_after_cleaning,
        "missing_before_cleaning": missing_before,
        "missing_after_cleaning": missing_after,
        "negative_flow_rows_removed": negative_rows,
        "duplicates_removed": duplicate_count,
    }
    return cleaned, stats


def chronological_split(
    cleaned: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> SplitResult:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    unique_ts = np.array(sorted(cleaned["timestamp"].unique()))
    n = len(unique_ts)
    if n < 3:
        raise ValueError("Not enough timestamps to create train/validation/test splits.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= 0:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= n:
        val_end = n - 1

    train_ts = set(unique_ts[:train_end])
    val_ts = set(unique_ts[train_end:val_end])

    train_df = cleaned[cleaned["timestamp"].isin(train_ts)].copy()
    val_df = cleaned[cleaned["timestamp"].isin(val_ts)].copy()
    test_df = cleaned[~cleaned["timestamp"].isin(train_ts | val_ts)].copy()

    return SplitResult(train=train_df, validation=val_df, test=test_df)


def make_sequences(df: pd.DataFrame, seq_len: int, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for _, group in df.groupby("scats_site"):
        g = group.sort_values("timestamp")
        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g["flow_15min"].to_numpy(dtype=np.float32)

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            x_window = features[i : i + seq_len]
            y_target = targets[i + seq_len]

            if np.isnan(x_window).any() or np.isnan(y_target):
                continue

            x_parts.append(x_window)
            y_parts.append(np.array(y_target, dtype=np.float32))

    if not x_parts:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    x = np.stack(x_parts)
    y = np.array(y_parts, dtype=np.float32)
    return x, y


def write_processed_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    content = """# SCATS Preprocessing Outputs

## How to run

```bash
.\\.venv\\Scripts\\python src/data/preprocess_scats.py
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
"""
    readme.write_text(content, encoding="utf-8")


def build_metadata(
    *,
    input_path: Path,
    cleaned: pd.DataFrame,
    split: SplitResult,
    cleaning_stats: dict[str, Any],
    output_paths: dict[str, str],
    sequence_stats: dict[str, Any],
) -> dict[str, Any]:
    timestamp_min = cleaned["timestamp"].min()
    timestamp_max = cleaned["timestamp"].max()

    metadata = {
        "input_file": str(input_path),
        "rows_before_cleaning": cleaning_stats["rows_before_cleaning"],
        "rows_after_cleaning": cleaning_stats["rows_after_cleaning"],
        "scats_sites_count": int(cleaned["scats_site"].nunique()),
        "date_range": {
            "start": str(timestamp_min),
            "end": str(timestamp_max),
        },
        "missing_before_cleaning": cleaning_stats["missing_before_cleaning"],
        "missing_after_cleaning": cleaning_stats["missing_after_cleaning"],
        "duplicates_removed": cleaning_stats["duplicates_removed"],
        "negative_flow_rows_removed": cleaning_stats["negative_flow_rows_removed"],
        "split_config": {
            "train_ratio": None,
            "validation_ratio": None,
            "test_ratio": None,
            "chronological": True,
        },
        "split_rows": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "output_files": output_paths,
        "sequence_info": sequence_stats,
    }
    return metadata


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(input_path)
    long_df = to_long_format(raw)
    cleaned, cleaning_stats = clean_and_engineer(long_df)

    splits = chronological_split(cleaned, args.train_ratio, args.val_ratio, args.test_ratio)

    cleaned_path = output_dir / "cleaned_scats_long.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "metadata.json"
    sequences_path = output_dir / "scats_sequences.npz"

    cleaned.to_csv(cleaned_path, index=False)
    splits.train.to_csv(train_path, index=False)
    splits.validation.to_csv(val_path, index=False)
    splits.test.to_csv(test_path, index=False)

    output_paths = {
        "cleaned_scats_long_csv": str(cleaned_path),
        "train_csv": str(train_path),
        "validation_csv": str(val_path),
        "test_csv": str(test_path),
    }

    sequence_stats: dict[str, Any] = {
        "generated": False,
        "sequence_length": args.sequence_length,
        "features": DEFAULT_FEATURE_COLUMNS,
    }

    if not args.skip_sequences:
        x_train, y_train = make_sequences(splits.train, args.sequence_length, DEFAULT_FEATURE_COLUMNS)
        x_val, y_val = make_sequences(splits.validation, args.sequence_length, DEFAULT_FEATURE_COLUMNS)
        x_test, y_test = make_sequences(splits.test, args.sequence_length, DEFAULT_FEATURE_COLUMNS)

        np.savez(
            sequences_path,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
            feature_names=np.array(DEFAULT_FEATURE_COLUMNS, dtype=object),
            sequence_length=np.array([args.sequence_length], dtype=np.int32),
        )

        output_paths["scats_sequences_npz"] = str(sequences_path)
        sequence_stats.update(
            {
                "generated": True,
                "train_shape": list(x_train.shape),
                "validation_shape": list(x_val.shape),
                "test_shape": list(x_test.shape),
            }
        )

    metadata = build_metadata(
        input_path=input_path,
        cleaned=cleaned,
        split=splits,
        cleaning_stats=cleaning_stats,
        output_paths=output_paths,
        sequence_stats=sequence_stats,
    )
    metadata["split_config"]["train_ratio"] = args.train_ratio
    metadata["split_config"]["validation_ratio"] = args.val_ratio
    metadata["split_config"]["test_ratio"] = args.test_ratio

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_processed_readme(output_dir)

    print("Preprocessing complete.")
    print(f"Rows before cleaning: {cleaning_stats['rows_before_cleaning']}")
    print(f"Rows after cleaning:  {cleaning_stats['rows_after_cleaning']}")
    print(f"Output directory:     {output_dir}")


if __name__ == "__main__":
    main()

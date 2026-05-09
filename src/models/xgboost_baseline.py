from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


DEFAULT_BASE_FEATURES = [
    "flow_15min",
    "flow_hour_simple",
    "flow_hour_rolling",
    "day_of_week",
    "hour",
    "minute",
    "interval_index",
    "latitude",
    "longitude",
]
DEFAULT_LAGS = [1, 2, 4, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost baseline model for SCATS flow prediction."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing train.csv, validation.csv, test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "models" / "xgboost_baseline",
        help="Directory to save model and metrics",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=1,
        help="Prediction horizon in 15-minute steps. 1 means next interval.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=DEFAULT_LAGS,
        help="Lag steps for flow_15min per SCATS site.",
    )

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"

    missing = [p for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split files: " + ", ".join(str(p) for p in missing)
        )

    train_df = pd.read_csv(train_path, parse_dates=["timestamp"])
    val_df = pd.read_csv(val_path, parse_dates=["timestamp"])
    test_df = pd.read_csv(test_path, parse_dates=["timestamp"])

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        required = {"timestamp", "scats_site", "flow_15min"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{name}.csv missing required columns: {sorted(required - set(df.columns))}"
            )
        for optional_location_feature in ["latitude", "longitude"]:
            if optional_location_feature not in df.columns:
                df[optional_location_feature] = 0.0

    return train_df, val_df, test_df


def add_temporal_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lags: list[int],
    target_horizon: int,
) -> pd.DataFrame:
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    full = pd.concat([train, val, test], ignore_index=True)
    full = full.sort_values(["scats_site", "timestamp"]).reset_index(drop=True)

    full["target_flow_15min"] = (
        full.groupby("scats_site")["flow_15min"].shift(-target_horizon)
    )

    for lag in sorted(set(lags)):
        full[f"flow_15min_lag_{lag}"] = full.groupby("scats_site")["flow_15min"].shift(lag)

    full["is_weekend"] = (full["day_of_week"] >= 5).astype(int)
    for col in ["latitude", "longitude"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)

    return full


def build_model_frame(
    feature_df: pd.DataFrame,
    base_features: list[str],
    lags: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    lag_features = [f"flow_15min_lag_{lag}" for lag in sorted(set(lags))]
    numeric_features = base_features + lag_features + ["is_weekend"]

    missing_features = [c for c in numeric_features if c not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features after feature engineering: {missing_features}")

    model_df = feature_df.dropna(subset=numeric_features + ["target_flow_15min"]).copy()

    site_dummies = pd.get_dummies(model_df["scats_site"].astype(str), prefix="site", dtype=np.int8)
    x = pd.concat([model_df[numeric_features].astype(np.float32), site_dummies], axis=1)
    y = model_df["target_flow_15min"].astype(np.float32)

    model_df = model_df.reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return model_df, x, y


def split_xy(model_frame: pd.DataFrame, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, ...]:
    train_mask = model_frame["split"] == "train"
    val_mask = model_frame["split"] == "validation"
    test_mask = model_frame["split"] == "test"

    x_train = x.loc[train_mask]
    y_train = y.loc[train_mask]
    x_val = x.loc[val_mask]
    y_val = y.loc[val_mask]
    x_test = x.loc[test_mask]
    y_test = y.loc[test_mask]

    x_cols = x.columns.tolist()
    return x_train, y_train, x_val, y_val, x_test, y_test, x_cols


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = read_splits(args.processed_dir)

    feature_df = add_temporal_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        lags=args.lags,
        target_horizon=args.target_horizon,
    )

    model_frame, x_all, y_all = build_model_frame(
        feature_df=feature_df,
        base_features=DEFAULT_BASE_FEATURES,
        lags=args.lags,
    )

    x_train, y_train, x_val, y_val, x_test, y_test, feature_names = split_xy(
        model_frame, x_all, y_all
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = {
        "validation": evaluate(y_val.to_numpy(), val_pred),
        "test": evaluate(y_test.to_numpy(), test_pred),
        "config": {
            "target_horizon": args.target_horizon,
            "lags": args.lags,
            "base_features": DEFAULT_BASE_FEATURES,
        },
        "row_counts": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "feature_count": len(feature_names),
    }

    model_path = args.output_dir / "xgboost_model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    features_path = args.output_dir / "feature_names.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

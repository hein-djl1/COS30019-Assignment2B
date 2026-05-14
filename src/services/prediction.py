from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from tensorflow.keras.models import load_model

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_scats_long.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "xgboost_baseline"
DEFAULT_LSTM_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "lstm_model.h5"


@dataclass(frozen=True)
class PredictionRequest:
    scats_site: int
    timestamp: pd.Timestamp
    horizon_steps: int = 1


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    scats_site: int
    timestamp: pd.Timestamp
    predicted_flow_15min: float
    predicted_flow_hour: float


class TrafficModel(Protocol):
    model_id: str
    label: str

    def predict(self, request: PredictionRequest) -> PredictionResult:
        ...


class XGBoostGlobalModel:
    model_id = "xgboost_global"
    label = "Global XGBoost"

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        processed_path: Path = DEFAULT_PROCESSED_PATH,
    ) -> None:
        self.model = joblib.load(model_dir / "xgboost_model.joblib")
        self.feature_names = pd.read_json(model_dir / "feature_names.json", typ="series").tolist()
        self.history = pd.read_csv(processed_path, parse_dates=["timestamp"])
        self.history = self.history.sort_values(["scats_site", "timestamp"]).reset_index(drop=True)
        self.site_history = {
            int(site): group.sort_values("timestamp").reset_index(drop=True)
            for site, group in self.history.groupby("scats_site")
        }

    def predict(self, request: PredictionRequest) -> PredictionResult:
        if request.scats_site not in self.site_history:
            raise ValueError(f"Unknown SCATS site: {request.scats_site}")

        row = self._build_feature_row(request)
        prediction = float(self.model.predict(row[self.feature_names])[0])
        prediction = max(0.0, prediction)
        return PredictionResult(
            model_id=self.model_id,
            scats_site=request.scats_site,
            timestamp=request.timestamp,
            predicted_flow_15min=prediction,
            predicted_flow_hour=prediction * 4.0,
        )

    def _build_feature_row(self, request: PredictionRequest) -> pd.DataFrame:
        site_df = self.site_history[request.scats_site]
        ts = pd.Timestamp(request.timestamp)
        prior = site_df[site_df["timestamp"] <= ts]
        if prior.empty:
            prior = site_df.iloc[[0]]

        latest = prior.iloc[-1]
        feature_values: dict[str, float] = {
            "flow_15min": float(latest["flow_15min"]),
            "flow_hour_simple": float(latest["flow_hour_simple"]),
            "flow_hour_rolling": float(latest["flow_hour_rolling"]),
            "day_of_week": float(ts.dayofweek),
            "hour": float(ts.hour),
            "minute": float(ts.minute),
            "interval_index": float(ts.hour * 4 + ts.minute // 15),
            "latitude": float(latest.get("latitude", 0.0)),
            "longitude": float(latest.get("longitude", 0.0)),
            "is_weekend": float(ts.dayofweek >= 5),
        }

        for lag in [1, 2, 4, 8]:
            if len(prior) >= lag:
                feature_values[f"flow_15min_lag_{lag}"] = float(prior.iloc[-lag]["flow_15min"])
            else:
                feature_values[f"flow_15min_lag_{lag}"] = float(latest["flow_15min"])

        site_feature = f"site_{request.scats_site}"
        for feature_name in self.feature_names:
            if feature_name.startswith("site_"):
                feature_values[feature_name] = 1.0 if feature_name == site_feature else 0.0

        row = pd.DataFrame([{name: feature_values.get(name, 0.0) for name in self.feature_names}])
        return row.astype(np.float32)

class LSTMGlobalModel:
    model_id = "lstm_global"
    label = "Global LSTM"

    def __init__(
        self,
        model_path: Path = DEFAULT_LSTM_MODEL_PATH,
        processed_path: Path = DEFAULT_PROCESSED_PATH,
    ) -> None:
        # Load trained LSTM model
        self.model = load_model(
            model_path,
            compile=False
        )

        # Load processed SCATS history
        self.history = pd.read_csv(
            processed_path,
            parse_dates=["timestamp"]
        )

        # Sort by SCATS site and timestamp
        self.history = self.history.sort_values(
            ["scats_site", "timestamp"]
        ).reset_index(drop=True)

        # Group traffic history by SCATS site
        self.site_history = {
            int(site): group.sort_values("timestamp").reset_index(drop=True)
            for site, group in self.history.groupby("scats_site")
        }

    def predict(self, request: PredictionRequest) -> PredictionResult:

        # Check whether SCATS site exists
        if request.scats_site not in self.site_history:
            raise ValueError(
                f"Unknown SCATS site: {request.scats_site}"
            )

        # Get traffic history for selected SCATS site
        site_df = self.site_history[request.scats_site]

        # Convert request timestamp
        ts = pd.Timestamp(request.timestamp)

        # Get historical records before timestamp
        prior = site_df[site_df["timestamp"] <= ts]

        # Ensure enough sequence history exists
        if len(prior) < 8:
            raise ValueError(
                f"Not enough history for SCATS site {request.scats_site}"
            )

        # Get latest 8 timesteps
        sequence = prior.iloc[-8:]

        # Build feature matrix
        features = sequence[
            [
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
        ].values.astype(np.float32)

        # Reshape into LSTM input format
        features = np.expand_dims(features, axis=0)

        # Predict traffic flow
        prediction = float(
            self.model.predict(features, verbose=0)[0][0]
        )

        # Prevent negative predictions
        prediction = max(0.0, prediction)

        # Return standardized prediction result
        return PredictionResult(
            model_id=self.model_id,
            scats_site=request.scats_site,
            timestamp=request.timestamp,
            predicted_flow_15min=prediction,
            predicted_flow_hour=prediction * 4.0,
        )

class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], TrafficModel]] = {
            XGBoostGlobalModel.model_id: XGBoostGlobalModel,
            LSTMGlobalModel.model_id: LSTMGlobalModel,
        }
        self._instances: dict[str, TrafficModel] = {}

    def register(self, model_id: str, factory: Callable[[], TrafficModel]) -> None:
        self._factories[model_id] = factory
        self._instances.pop(model_id, None)

    def list_models(self) -> list[tuple[str, str]]:
        labels = []
        for model_id, factory in self._factories.items():
            label = getattr(factory, "label", model_id)
            labels.append((model_id, label))
        return labels

    def get(self, model_id: str) -> TrafficModel:
        if model_id not in self._factories:
            raise ValueError(f"Model '{model_id}' is not registered.")
        if model_id not in self._instances:
            self._instances[model_id] = self._factories[model_id]()
        return self._instances[model_id]


registry = ModelRegistry()


def predict_traffic_flow(
    scats_site: int,
    timestamp: pd.Timestamp,
    model_id: str = XGBoostGlobalModel.model_id,
    horizon_steps: int = 1,
) -> PredictionResult:
    model = registry.get(model_id)
    return model.predict(
        PredictionRequest(
            scats_site=int(scats_site),
            timestamp=pd.Timestamp(timestamp),
            horizon_steps=horizon_steps,
        )
    )

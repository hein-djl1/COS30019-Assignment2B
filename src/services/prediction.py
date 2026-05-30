from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from tensorflow.keras.models import load_model

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCATION_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_location" / "cleaned_scats_long.csv"
DEFAULT_SITE_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_scats_long.csv"
DEFAULT_XGBOOST_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "xgboost"
DEFAULT_XGBOOST_SITE_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "xgboost_site"
DEFAULT_LSTM_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "lstm_model.h5"
DEFAULT_LSTM_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "lstm_model"
DEFAULT_LSTM_SITE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "lstm_model_site.h5"
DEFAULT_LSTM_SITE_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "lstm_model_site"
DEFAULT_GRU_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "gru_model_simple"
DEFAULT_GRU_SITE_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "gru_model_site"
SEQUENCE_FEATURE_COLUMNS = [
    "flow_15min",
    "flow_hour_simple",
    "flow_hour_rolling",
    "day_of_week",
    "hour",
    "minute",
    "interval_index",
    "latitude",
    "longitude",
    "location_encoded",
]


def _ensure_location_encoded_column(history: pd.DataFrame) -> pd.DataFrame:
    if "location_encoded" in history.columns:
        return history

    if "location" not in history.columns:
        history["location_encoded"] = 0.0
        return history

    history["location"] = history["location"].fillna("").astype(str)
    location_values = sorted(history["location"].unique())
    location_denominator = max(1, len(location_values) - 1)
    location_codes = {
        location: index / location_denominator
        for index, location in enumerate(location_values)
    }
    history["location_encoded"] = history["location"].map(location_codes).astype(float)
    return history


def _resolve_sequence_feature_columns(model, history: pd.DataFrame) -> list[str]:
    expected_features = int(model.input_shape[-1])
    if expected_features == len(SEQUENCE_FEATURE_COLUMNS):
        return SEQUENCE_FEATURE_COLUMNS
    if expected_features == len(SEQUENCE_FEATURE_COLUMNS) - 1:
        return [column for column in SEQUENCE_FEATURE_COLUMNS if column != "location_encoded"]

    available = [column for column in SEQUENCE_FEATURE_COLUMNS if column in history.columns]
    if len(available) == expected_features:
        return available

    raise ValueError(
        f"Model expects {expected_features} input features, but no supported feature mapping was found."
    )


def _prepare_location_history(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame], dict[int, list[str]]]:
    if "location" not in history.columns:
        history["location"] = ""
    history["location"] = history["location"].fillna("").astype(str)
    history = history.sort_values(["scats_site", "location", "timestamp"]).reset_index(drop=True)

    location_history = {
        (int(site), str(location)): group.sort_values("timestamp").reset_index(drop=True)
        for (site, location), group in history.groupby(["scats_site", "location"])
    }
    locations_by_site = {
        int(site): sorted(group["location"].drop_duplicates().astype(str).tolist())
        for site, group in history.groupby("scats_site")
    }
    return history, location_history, locations_by_site


def _resolve_location_request(
    request: "PredictionRequest",
    model_label: str,
    locations_by_site: dict[int, list[str]],
    strict_location: bool = True,
) -> str:
    locations = locations_by_site.get(request.scats_site)
    if not locations:
        raise ValueError(f"Unknown SCATS site: {request.scats_site}")
    if request.location is not None:
        location = str(request.location)
        if location in locations:
            return location
        if strict_location:
            raise ValueError(
                f"Unknown location for SCATS site {request.scats_site}: {location}"
            )
        return locations[0]
    if len(locations) == 1:
        return locations[0]
    if not strict_location:
        return locations[0]
    raise ValueError(
        f"{model_label} requires a location for SCATS site {request.scats_site}."
    )


@dataclass(frozen=True)
class PredictionRequest:
    scats_site: int
    timestamp: pd.Timestamp
    location: str | None = None
    horizon_steps: int = 1


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    scats_site: int
    timestamp: pd.Timestamp
    predicted_flow_15min: float
    predicted_flow_hour: float
    location: str | None = None


class TrafficModel(Protocol):
    model_id: str
    label: str

    def predict(self, request: PredictionRequest) -> PredictionResult:
        ...


class XGBoostModel:
    model_id = "xgboost"
    label = "XGBoost"

    def __init__(
        self,
        model_dir: Path = DEFAULT_XGBOOST_MODEL_DIR,
        processed_path: Path = DEFAULT_LOCATION_PROCESSED_PATH,
        strict_location: bool = True,
    ) -> None:
        self.model = joblib.load(model_dir / "xgboost_model.joblib")
        self.feature_names = pd.read_json(model_dir / "feature_names.json", typ="series").tolist()
        self.history = pd.read_csv(processed_path, parse_dates=["timestamp"])
        self.strict_location = strict_location
        self.history, self.location_history, self.locations_by_site = _prepare_location_history(
            self.history
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        location = self._resolve_location(request)
        key = (request.scats_site, location)
        if key not in self.location_history:
            raise ValueError(
                f"Unknown SCATS location: {request.scats_site} / {location}"
            )

        row = self._build_feature_row(request, location)
        prediction = float(self.model.predict(row[self.feature_names])[0])
        prediction = max(0.0, prediction)
        return PredictionResult(
            model_id=self.model_id,
            scats_site=request.scats_site,
            timestamp=request.timestamp,
            predicted_flow_15min=prediction,
            predicted_flow_hour=prediction * 4.0,
            location=location,
        )

    def _resolve_location(self, request: PredictionRequest) -> str:
        return _resolve_location_request(
            request,
            self.label,
            self.locations_by_site,
            strict_location=self.strict_location,
        )

    def _build_feature_row(self, request: PredictionRequest, location: str) -> pd.DataFrame:
        site_df = self.location_history[(request.scats_site, location)]
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
        location_feature = f"location_{location}"
        # Match the one-hot feature columns saved by the XGBoost trainer.
        for feature_name in self.feature_names:
            if feature_name.startswith("site_"):
                feature_values[feature_name] = 1.0 if feature_name == site_feature else 0.0
            elif feature_name.startswith("location_"):
                feature_values[feature_name] = 1.0 if feature_name == location_feature else 0.0

        row = pd.DataFrame([{name: feature_values.get(name, 0.0) for name in self.feature_names}])
        return row.astype(np.float32)

class LSTMModel:
    model_id = "lstm"
    label = "LSTM"

    def __init__(
        self,
        model_path: Path = DEFAULT_LSTM_MODEL_PATH,
        model_dir: Path = DEFAULT_LSTM_MODEL_DIR,
        processed_path: Path = DEFAULT_LOCATION_PROCESSED_PATH,
        strict_location: bool = True,
    ) -> None:
        # Load trained LSTM model
        self.model = load_model(
            model_path,
            compile=False
        )
        scaler_path = model_dir / "y_scaler.pkl"
        self.y_scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # Load processed SCATS history
        self.history = pd.read_csv(
            processed_path,
            parse_dates=["timestamp"]
        )
        self.strict_location = strict_location
        self.history = _ensure_location_encoded_column(self.history)
        self.history, self.location_history, self.locations_by_site = _prepare_location_history(
            self.history
        )

        self.sequence_feature_columns = _resolve_sequence_feature_columns(self.model, self.history)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        location = _resolve_location_request(
            request,
            self.label,
            self.locations_by_site,
            strict_location=self.strict_location,
        )
        key = (request.scats_site, location)
        if key not in self.location_history:
            raise ValueError(
                f"Unknown SCATS location: {request.scats_site} / {location}"
            )

        site_df = self.location_history[key]

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
        features = sequence[self.sequence_feature_columns].values.astype(np.float32)

        # Reshape into LSTM input format
        features = np.expand_dims(features, axis=0)

        # Predict traffic flow
        prediction = float(
            self.model.predict(features, verbose=0)[0][0]
        )
        if self.y_scaler is not None:
            prediction = float(
                self.y_scaler.inverse_transform(np.array([[prediction]])).flatten()[0]
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
            location=location,
        )

class GRUModel:
    model_id = "gru"
    label = "GRU"

    def __init__(
        self,
        model_dir: Path = DEFAULT_GRU_MODEL_DIR,
        processed_path: Path = DEFAULT_LOCATION_PROCESSED_PATH,
        strict_location: bool = True,
    ) -> None:
        self.model = load_model(model_dir / "gru_model.keras", compile=False)
        self.X_scaler = joblib.load(model_dir / "X_scaler.pkl")
        self.y_scaler = joblib.load(model_dir / "y_scaler.pkl")

        self.strict_location = strict_location
        self.history = pd.read_csv(processed_path, parse_dates=["timestamp"])
        self.history = _ensure_location_encoded_column(self.history)
        self.history, self.location_history, self.locations_by_site = _prepare_location_history(
            self.history
        )
        self.sequence_feature_columns = _resolve_sequence_feature_columns(self.model, self.history)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        location = _resolve_location_request(
            request,
            self.label,
            self.locations_by_site,
            strict_location=self.strict_location,
        )
        key = (request.scats_site, location)
        if key not in self.location_history:
            raise ValueError(
                f"Unknown SCATS location: {request.scats_site} / {location}"
            )
        site_df = self.location_history[key]
        ts = pd.Timestamp(request.timestamp)
        prior = site_df[site_df["timestamp"] <= ts]
        if len(prior) < 8:
            raise ValueError(
                f"Not enough history for SCATS site {request.scats_site} at location {location}"
            )

        sequence = prior.iloc[-8:]
        features = sequence[self.sequence_feature_columns].values.astype(np.float32)
        features = self.X_scaler.transform(features)
        features = np.expand_dims(features, axis=0)

        prediction_scaled = float(self.model.predict(features, verbose=0)[0][0])
        prediction = float(
            self.y_scaler.inverse_transform(np.array([[prediction_scaled]])).flatten()[0]
        )
        prediction = max(0.0, prediction)

        return PredictionResult(
            model_id=self.model_id,
            scats_site=request.scats_site,
            timestamp=request.timestamp,
            predicted_flow_15min=prediction,
            predicted_flow_hour=prediction * 4.0,
            location=location,
        )


class XGBoostLocationModel(XGBoostModel):
    model_id = "xgboost_location"
    label = "XGBoost (Location-level)"

    def __init__(self) -> None:
        super().__init__(
            model_dir=DEFAULT_XGBOOST_MODEL_DIR,
            processed_path=DEFAULT_LOCATION_PROCESSED_PATH,
            strict_location=True,
        )


class XGBoostSiteModel(XGBoostModel):
    model_id = "xgboost_site"
    label = "XGBoost (Site-level)"

    def __init__(self) -> None:
        super().__init__(
            model_dir=DEFAULT_XGBOOST_SITE_MODEL_DIR,
            processed_path=DEFAULT_SITE_PROCESSED_PATH,
            strict_location=False,
        )


class LSTMLocationModel(LSTMModel):
    model_id = "lstm_location"
    label = "LSTM (Location-level)"

    def __init__(self) -> None:
        super().__init__(
            model_path=DEFAULT_LSTM_MODEL_PATH,
            model_dir=DEFAULT_LSTM_MODEL_DIR,
            processed_path=DEFAULT_LOCATION_PROCESSED_PATH,
            strict_location=True,
        )


class LSTMSiteModel(LSTMModel):
    model_id = "lstm_site"
    label = "LSTM (Site-level)"

    def __init__(self) -> None:
        super().__init__(
            model_path=DEFAULT_LSTM_SITE_MODEL_PATH,
            model_dir=DEFAULT_LSTM_SITE_MODEL_DIR,
            processed_path=DEFAULT_SITE_PROCESSED_PATH,
            strict_location=False,
        )


class GRULocationModel(GRUModel):
    model_id = "gru_location"
    label = "GRU (Location-level)"

    def __init__(self) -> None:
        super().__init__(
            model_dir=DEFAULT_GRU_MODEL_DIR,
            processed_path=DEFAULT_LOCATION_PROCESSED_PATH,
            strict_location=True,
        )


class GRUSiteModel(GRUModel):
    model_id = "gru_site"
    label = "GRU (Site-level)"

    def __init__(self) -> None:
        super().__init__(
            model_dir=DEFAULT_GRU_SITE_MODEL_DIR,
            processed_path=DEFAULT_SITE_PROCESSED_PATH,
            strict_location=False,
        )


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], TrafficModel]] = {
            XGBoostLocationModel.model_id: XGBoostLocationModel,
            XGBoostSiteModel.model_id: XGBoostSiteModel,
            GRULocationModel.model_id: GRULocationModel,
            GRUSiteModel.model_id: GRUSiteModel,
            LSTMLocationModel.model_id: LSTMLocationModel,
            LSTMSiteModel.model_id: LSTMSiteModel,
        }
        self._legacy_aliases: dict[str, str] = {
            "xgboost": XGBoostLocationModel.model_id,
            "gru": GRULocationModel.model_id,
            "lstm": LSTMLocationModel.model_id,
        }
        self._instances: dict[str, TrafficModel] = {}

    def list_models(self) -> list[tuple[str, str]]:
        labels = []
        for model_id, factory in self._factories.items():
            label = getattr(factory, "label", model_id)
            labels.append((model_id, label))
        return labels

    def get(self, model_id: str) -> TrafficModel:
        model_id = self._legacy_aliases.get(model_id, model_id)
        if model_id not in self._factories:
            raise ValueError(f"Model '{model_id}' is not registered.")
        if model_id not in self._instances:
            self._instances[model_id] = self._factories[model_id]()
        return self._instances[model_id]


registry = ModelRegistry()


def predict_traffic_flow(
    scats_site: int,
    timestamp: pd.Timestamp,
    model_id: str = XGBoostLocationModel.model_id,
    location: str | None = None,
    horizon_steps: int = 1,
) -> PredictionResult:
    model = registry.get(model_id)
    return model.predict(
        PredictionRequest(
            scats_site=int(scats_site),
            timestamp=pd.Timestamp(timestamp),
            location=location,
            horizon_steps=horizon_steps,
        )
    )

"""
gru_model.py
Train and evaluate a simple GRU model for traffic flow prediction using pre-made sequences.
Usage:
    python gru_model.py --epochs 20 --batch-size 32 --units 64
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

try:
    from loss_curve import save_loss_curve
except ModuleNotFoundError:
    from src.models.loss_curve import save_loss_curve


def parse_args():
    """Parse command-line arguments / use defaults"""
    parser = argparse.ArgumentParser(description="Train GRU for traffic prediction")
    parser.add_argument("--data", type=Path, default="data/processed_location/scats_sequences.npz",
                        help="Path to .npz sequence file")
    parser.add_argument("--output", type=Path, default="data/models/gru_model_simple",
                        help="Output directory for model and metrics")
    parser.add_argument("--units", type=int, default=64, help="Number of GRU units")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train for all epochs instead of stopping when validation loss stops improving",
    )

    return parser.parse_args()


def load_sequences(path):
    """Load pre-made train/val/test sequences from .npz file"""
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    data = np.load(path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in .npz: {missing}")
    return (data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"])


def normalize_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """Fit scaler on training data only, transform all splits"""
    n_train, timesteps, features = X_train.shape

    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train.reshape(-1, features)).reshape(n_train, timesteps, features)
    X_val   = X_scaler.transform(X_val.reshape(-1, features)).reshape(X_val.shape)
    X_test  = X_scaler.transform(X_test.reshape(-1, features)).reshape(X_test.shape)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    joblib.dump(X_scaler, output_dir / "X_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")

    return X_train, y_train, X_val, y_val, X_test, y_test, y_scaler


def build_model(input_shape, units, learning_rate=0.001):
    """Create and compile a simple one-layer GRU model."""
    model = Sequential()
    model.add(GRU(units, input_shape=input_shape))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    patience,
    use_early_stopping=True,
):
    """Train with validation monitoring and optional early stopping."""
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
    ]
    if use_early_stopping:
        callbacks.insert(
            0,
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
        )

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)
    return history


def evaluate_model(model, X_test, y_test, y_scaler):
    y_pred_scaled = model.predict(X_test).flatten()

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range else np.nan
    return {"rmse": float(rmse), "mae": float(mae), "nrmse": float(nrmse)}


def get_best_epoch(history):
    """Return the 1-based epoch with the lowest validation loss."""
    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        return None, None

    best_index = int(np.argmin(val_losses))
    return best_index + 1, float(val_losses[best_index])


def save_results(model, metrics, args, history, output_dir):
    """Save model, metrics.json, and training history in CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "gru_model.keras")

    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    config["feature_scaling"] = "minmax_X_train"
    config["target_scaling"] = "minmax_y_train"

    best_epoch, best_val_loss = get_best_epoch(history)
    loss_curve_path = output_dir / "loss_curve.svg"
    full_metrics = {
        "validation": metrics["validation"],
        "test": metrics["test"],
        "test_metrics": metrics["test"],
        "config": config,
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopping_enabled": not args.no_early_stopping,
        "loss_curve": str(loss_curve_path),
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "history.csv", index=False)
    save_loss_curve(history_df, loss_curve_path, "GRU Loss Curve")
    print(f"Saved model and results to {output_dir}")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences(args.data)

    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = normalize_data(
        X_train, y_train, X_val, y_val, X_test, y_test, args.output
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, args.units, args.learning_rate)
    model.summary()

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        args.epochs,
        args.batch_size,
        args.patience,
        use_early_stopping=not args.no_early_stopping,
    )

    metrics = {
        "validation": evaluate_model(model, X_val, y_val, y_scaler),
        "test": evaluate_model(model, X_test, y_test, y_scaler),
    }
    print(
        f"Validation RMSE: {metrics['validation']['rmse']:.4f}, "
        f"MAE: {metrics['validation']['mae']:.4f}, "
        f"NRMSE: {metrics['validation']['nrmse']:.4f}"
    )
    print(
        f"Test RMSE: {metrics['test']['rmse']:.4f}, "
        f"MAE: {metrics['test']['mae']:.4f}, "
        f"NRMSE: {metrics['test']['nrmse']:.4f}"
    )

    save_results(model, metrics, args, history, args.output)


if __name__ == "__main__":
    main()

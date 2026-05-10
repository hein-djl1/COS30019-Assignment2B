"""
gru_model.py
Train and evaluate a GRU model for traffic flow prediction using pre-made sequences.
Usage:
    python gru_model.py --epochs 100 --batch-size 32 --units 64 --dropout 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
import joblib


def parse_args():
    """Parse command-line arguments / use defaults"""
    parser = argparse.ArgumentParser(description="Train GRU for traffic prediction")
    parser.add_argument("--data", type=Path, default="data/processed/scats_sequences.npz",
                        help="Path to .npz sequence file")
    parser.add_argument("--output", type=Path, default="data/models/gru_model",
                        help="Output directory for model and metrics")
    parser.add_argument("--units", type=int, default=64, help="Number of GRU units")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

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


def build_model(input_shape, units, dropout, learning_rate=0.001):
    """Create and compile GRU model with two layers and batch norm"""
    model = Sequential()
    # First GRU layer (returns sequences for stacking)
    model.add(GRU(units, input_shape=input_shape, dropout=dropout,
                  recurrent_dropout=dropout, return_sequences=True))
    model.add(BatchNormalization())
    # Second GRU layer
    model.add(GRU(units // 2, dropout=dropout, recurrent_dropout=dropout))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
    """Train with early stopping and learning rate reduction"""
    early_stop = EarlyStopping(monitor="val_loss", patience=patience,
                               restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                  min_lr=1e-6, verbose=1)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr],
                        verbose=1)
    return history


def evaluate_model(model, X_test, y_test, y_scaler):
    y_pred_scaled = model.predict(X_test).flatten()

    # Back to original traffic flow units
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae)}


def save_results(model, metrics, args, history, output_dir):
    """Save model, metrics.json, and training history in CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model.save(output_dir / "gru_model.keras")

    # Save metrics and config
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)

    full_metrics = {
        "test_metrics": metrics,
        "config": config,
        "training_samples": len(history.history["loss"])
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    # Save training history as CSV
    import pandas as pd
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    print(f"Saved model and results to {output_dir}")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences(args.data)

    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = normalize_data(
        X_train, y_train, X_val, y_val, X_test, y_test, args.output
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, args.units, args.dropout, args.learning_rate)
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val,
                          args.epochs, args.batch_size, args.patience)

    test_metrics = evaluate_model(model, X_test, y_test, y_scaler)
    print(f"Test RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")

    save_results(model, test_metrics, args, history, args.output)


if __name__ == "__main__":
    main()

"""
lstm_model.py
This script sets up the dependencies and loads preprocessed sequence data 
for building and evaluating an LSTM-based neural network model.
"""

# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

# Core data manipulation and file path handling
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Metrics for evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Keras components for building the LSTM neural network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

try:
    from loss_curve import save_loss_curve
except ModuleNotFoundError:
    from src.models.loss_curve import save_loss_curve


def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range else np.nan
    return float(rmse), float(mae), float(nrmse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM for traffic prediction")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed_location/scats_sequences.npz"),
        help="Path to .npz sequence file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/lstm_model"),
        help="Directory for metrics/history/scaler/loss curve",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/lstm_model.h5"),
        help="Path to save trained .h5 model",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--units", type=int, default=64, help="Number of LSTM units")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---------------------------------------------------
    # LOAD PREPROCESSED DATA
    # ---------------------------------------------------

    # Define the path to the preprocessed sequence data
    data_path = args.data

    # Load the numpy archive dataset into memory
    data = np.load(data_path)

    # ---------------------------------------------------
    # EXTRACT TRAIN / VALIDATION / TEST DATA
    # ---------------------------------------------------

    # Training data
    X_train = data["X_train"]
    y_train = data["y_train"]

    # Validation data
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Testing data
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Keep original targets for final metrics, but train on normalized targets.
    y_train_original = y_train.copy()
    y_val_original = y_val.copy()
    y_test_original = y_test.copy()

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ---------------------------------------------------
    # CHECK DATA SHAPES
    # ---------------------------------------------------

    # These checks are VERY important.
    # LSTM expects input in the form:
    # (samples, timesteps, features)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # ---------------------------------------------------
    # BUILD LSTM MODEL
    # ---------------------------------------------------

    # Sequential means layers are added one after another
    model = Sequential()

    # Add the LSTM layer
    # 64 = number of memory units (neurons)
    # input_shape = (timesteps, features)
    model.add(
        LSTM(
        args.units,
        input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )

    # ---------------------------------------------------
    # COMPILE MODEL
    # ---------------------------------------------------

    # Dense output layer
    # Outputs one predicted traffic value
    model.add(Dense(1))

    # Adam optimizer is commonly used for deep learning
    # MSE is used because this is a regression problem
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # Print the model architecture summary
    model.summary()

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------

    # Train the LSTM model
    history = model.fit(

        # Training input data
        X_train,

        # Correct target values
        y_train,

        # Validation dataset
        validation_data=(X_val, y_val),

        # Number of full passes through dataset
        epochs=args.epochs,

        # Number of samples processed at once
        batch_size=args.batch_size
    )

    # ---------------------------------------------------
    # MAKE PREDICTIONS
    # ---------------------------------------------------

    # Predict traffic flow using the validation and test datasets
    val_predictions_scaled = model.predict(X_val)
    predictions_scaled = model.predict(X_test)

    # Convert predictions into flat 1D arrays
    val_predictions_scaled = val_predictions_scaled.flatten()
    predictions_scaled = predictions_scaled.flatten()

    val_predictions = y_scaler.inverse_transform(
        val_predictions_scaled.reshape(-1, 1)
    ).flatten()
    predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()

    # ---------------------------------------------------
    # EVALUATE MODEL PERFORMANCE
    # ---------------------------------------------------

    val_rmse, val_mae, val_nrmse = evaluate_predictions(
        y_val_original, val_predictions
    )
    rmse, mae, nrmse = evaluate_predictions(
        y_test_original, predictions
    )

    print("Validation RMSE:", val_rmse)
    print("Validation MAE:", val_mae)
    print("Validation NRMSE:", val_nrmse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("NRMSE:", nrmse)

    # ---------------------------------------------------
    # SAVE TRAINED MODEL
    # ---------------------------------------------------

    # Save the trained LSTM model
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)

    # Save metrics and training history for model comparison
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")

    metrics = {
        "validation": {
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "nrmse": float(val_nrmse),
        },
        "test": {
            "rmse": float(rmse),
            "mae": float(mae),
            "nrmse": float(nrmse),
        },
        "config": {
            "data": str(data_path),
            "model_path": str(args.model_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "units": args.units,
            "target_scaling": "minmax_y_train",
        },
        "epochs_trained": len(history.history["loss"]),
        "loss_curve": str(output_dir / "loss_curve.svg"),
    }

    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.csv"
    loss_curve_path = output_dir / "loss_curve.svg"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False)
    save_loss_curve(history_df, loss_curve_path, "LSTM Loss Curve")

    print("Model saved successfully.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"History saved to: {history_path}")
    print(f"Loss curve saved to: {loss_curve_path}")

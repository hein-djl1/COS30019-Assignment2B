"""
lstm_model.py
This script sets up the dependencies and loads preprocessed sequence data 
for building and evaluating an LSTM-based neural network model.
"""

# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

# Core data manipulation and file path handling
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Metrics for evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Keras components for building the LSTM neural network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

if __name__ == "__main__":

    # ---------------------------------------------------
    # LOAD PREPROCESSED DATA
    # ---------------------------------------------------

    # Define the path to the preprocessed sequence data
    data_path = Path("data/processed/scats_sequences.npz")

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
        64,
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
        epochs=20,

        # Number of samples processed at once
        batch_size=32
    )

    # ---------------------------------------------------
    # MAKE PREDICTIONS
    # ---------------------------------------------------

    # Predict traffic flow using the validation and test datasets
    val_predictions = model.predict(X_val)
    predictions = model.predict(X_test)

    # Convert predictions into flat 1D arrays
    val_predictions = val_predictions.flatten()
    predictions = predictions.flatten()

    # ---------------------------------------------------
    # EVALUATE MODEL PERFORMANCE
    # ---------------------------------------------------

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(
        y_test, predictions
    )

    # Print evaluation results
    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_predictions)
    )

    val_mae = mean_absolute_error(
        y_val, val_predictions
    )

    print("Validation RMSE:", val_rmse)
    print("Validation MAE:", val_mae)
    print("RMSE:", rmse)
    print("MAE:", mae)

    # ---------------------------------------------------
    # SAVE TRAINED MODEL
    # ---------------------------------------------------

    # Save the trained LSTM model
    model.save("data/models/lstm_model.h5")

    # Save metrics and training history for model comparison
    output_dir = Path("data/models/lstm_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation": {
            "rmse": float(val_rmse),
            "mae": float(val_mae),
        },
        "test": {
            "rmse": float(rmse),
            "mae": float(mae),
        },
        "config": {
            "data": str(data_path),
            "model_path": "data/models/lstm_model.h5",
            "epochs": 20,
            "batch_size": 32,
            "units": 64,
        },
        "epochs_trained": len(history.history["loss"]),
    }

    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False)

    print("Model saved successfully.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"History saved to: {history_path}")

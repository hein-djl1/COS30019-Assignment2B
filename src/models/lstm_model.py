"""
lstm_model.py
This script sets up the dependencies and loads preprocessed sequence data 
for building and evaluating an LSTM-based neural network model.
"""

# Core data manipulation and file path handling
import numpy as np
from pathlib import Path

# Metrics for evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Keras components for building the LSTM neural network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the path to the preprocessed sequence data
data_path = Path("data/processed/scats_sequences.npz")

# Load the numpy archive dataset into memory
data = np.load(data_path)

# Training data
X_train = data["X_train"]
y_train = data["y_train"]

# Validation data
X_val = data["X_val"]
y_val = data["y_val"]

# Testing data
X_test = data["X_test"]
y_test = data["y_test"]

# These checks are VERY important.
# LSTM expects input in the form:
# (samples, timesteps, features)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Sequential means layers are added one after another
model = Sequential()

# Add the LSTM layer
# 64 = number of memory units (neurons)
# input_shape = (timesteps, features)
model.add(
    LSTM(
        64,
        input_shape=(8, 7)
    )
)

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
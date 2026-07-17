"""
Smart Rainfall Prediction - Deep Learning Model Module
======================================================
Defines LSTM, GRU, and CNN-LSTM architectures for rainfall prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, Input, Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import yaml


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_lstm_model(input_shape, config=None):
    """
    Build a stacked LSTM model for rainfall prediction.
    
    Args:
        input_shape: (sequence_length, n_features)
        config: Model configuration dict
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = load_config()["model"]
    
    model = Sequential([
        # First LSTM layer with return sequences
        Bidirectional(
            LSTM(config["lstm_units"], return_sequences=True, input_shape=input_shape),
        ),
        BatchNormalization(),
        Dropout(config["dropout"]),
        
        # Second LSTM layer
        Bidirectional(
            LSTM(config["lstm_units"] // 2, return_sequences=False)
        ),
        BatchNormalization(),
        Dropout(config["dropout"]),
        
        # Dense layers
        Dense(config["dense_units"], activation="relu"),
        Dropout(config["dropout"] / 2),
        Dense(config["dense_units"] // 2, activation="relu"),
        
        # Output layer (regression)
        Dense(1, activation="linear")
    ])
    
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    
    return model


def build_gru_model(input_shape, config=None):
    """
    Build a GRU model for rainfall prediction.
    """
    if config is None:
        config = load_config()["model"]
    
    model = Sequential([
        GRU(config["lstm_units"], return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(config["dropout"]),
        
        GRU(config["lstm_units"] // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(config["dropout"]),
        
        Dense(config["dense_units"], activation="relu"),
        Dropout(config["dropout"] / 2),
        Dense(1, activation="linear")
    ])
    
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    return model


def build_cnn_lstm_model(input_shape, config=None):
    """
    Build a CNN-LSTM hybrid model for rainfall prediction.
    CNN extracts local patterns, LSTM captures temporal dependencies.
    """
    if config is None:
        config = load_config()["model"]
    
    model = Sequential([
        # CNN feature extraction
        Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(config["dropout"]),
        
        # LSTM temporal modeling
        LSTM(config["lstm_units"], return_sequences=False),
        BatchNormalization(),
        Dropout(config["dropout"]),
        
        # Dense output
        Dense(config["dense_units"], activation="relu"),
        Dropout(config["dropout"] / 2),
        Dense(1, activation="linear")
    ])
    
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    return model


def get_model(model_type, input_shape, config=None):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: 'lstm', 'gru', or 'cnn_lstm'
        input_shape: (sequence_length, n_features)
        config: Optional config dict
    
    Returns:
        Compiled Keras model
    """
    builders = {
        "lstm": build_lstm_model,
        "gru": build_gru_model,
        "cnn_lstm": build_cnn_lstm_model,
    }
    
    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(builders.keys())}")
    
    return builders[model_type](input_shape, config)


def get_callbacks(config=None):
    """
    Get training callbacks for early stopping, learning rate reduction, 
    and model checkpointing.
    """
    if config is None:
        config = load_config()["model"]
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    config = load_config()
    seq_length = config["data"]["sequence_length"]
    n_features = len(config["model"]["input_features"])
    input_shape = (seq_length, n_features)
    
    print("Building models...\n")
    
    for model_type in ["lstm", "gru", "cnn_lstm"]:
        print(f"\n{'='*50}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*50}")
        model = get_model(model_type, input_shape)
        # Build memory by passing a dummy batch to enable .summary()
        model(tf.zeros((1,) + input_shape))
        model.summary()

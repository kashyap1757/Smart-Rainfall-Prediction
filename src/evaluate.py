"""
Smart Rainfall Prediction - Evaluation Module
==============================================
Comprehensive model evaluation with metrics and visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import joblib
import yaml
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model_path=None, config_path="configs/config.yaml"):
    """
    Evaluate the trained model on the test set.
    
    Returns:
        Dictionary of evaluation metrics
    """
    import tensorflow as tf
    
    config = load_config(config_path)
    
    if model_path is None:
        model_path = config["api"]["model_path"]
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    # Load scalers for inverse transform
    target_scaler = joblib.load("models/target_scaler.pkl")
    
    # Predictions
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform to get actual rainfall values
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred_actual = np.maximum(y_pred_actual, 0)  # Rainfall can't be negative
    
    # Calculate metrics
    metrics = {
        "mse": float(mean_squared_error(y_test_actual, y_pred_actual)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))),
        "mae": float(mean_absolute_error(y_test_actual, y_pred_actual)),
        "r2_score": float(r2_score(y_test_actual, y_pred_actual)),
        "explained_variance": float(explained_variance_score(y_test_actual, y_pred_actual)),
        "mape": float(np.mean(np.abs((y_test_actual - y_pred_actual) / 
                     (y_test_actual + 1e-8))) * 100),  # avoid div by zero
    }
    
    # Print evaluation report
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"  MSE:                {metrics['mse']:.4f}")
    print(f"  RMSE:               {metrics['rmse']:.4f}")
    print(f"  MAE:                {metrics['mae']:.4f}")
    print(f"  R² Score:           {metrics['r2_score']:.4f}")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"  MAPE:               {metrics['mape']:.2f}%")
    print("=" * 50)
    
    # Save metrics
    os.makedirs("models", exist_ok=True)
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to models/evaluation_metrics.json")
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()

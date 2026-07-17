"""
Smart Rainfall Prediction - Training Module
============================================
Trains the deep learning model with MLflow experiment tracking.
Supports multiple model architectures and hyperparameter logging.
"""

import os
import sys
import numpy as np
import yaml
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import prepare_data, load_config
from src.model import get_model, get_callbacks


def train_model(config_path="configs/config.yaml"):
    """
    Train the rainfall prediction model with MLflow tracking.
    
    Steps:
    1. Prepare data (generate/load, scale, create sequences)
    2. Build model
    3. Train with MLflow experiment tracking
    4. Save model and artifacts
    """
    config = load_config(config_path)
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    
    # =========================================
    # 1. Data Preparation
    # =========================================
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test, full_df = prepare_data(config)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    
    # =========================================
    # 2. MLflow Setup
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 2: MLflow Experiment Setup")
    print("=" * 60)
    
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # =========================================
    # 3. Model Training with MLflow Tracking
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 3: Model Training")
    print("=" * 60)
    
    with mlflow.start_run(run_name=f"{model_config['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param("lstm_units", model_config["lstm_units"])
        mlflow.log_param("dropout", model_config["dropout"])
        mlflow.log_param("dense_units", model_config["dense_units"])
        mlflow.log_param("learning_rate", model_config["learning_rate"])
        mlflow.log_param("epochs", model_config["epochs"])
        mlflow.log_param("batch_size", model_config["batch_size"])
        mlflow.log_param("sequence_length", config["data"]["sequence_length"])
        mlflow.log_param("n_features", len(model_config["input_features"]))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Build model
        model = get_model(model_config["type"], input_shape, model_config)
        import tensorflow as tf
        model(tf.zeros((1,) + input_shape))
        model.summary()
        
        # Get callbacks
        callbacks = get_callbacks(model_config)
        
        # Train
        print(f"\nTraining {model_config['type'].upper()} model...")
        print(f"  Epochs: {model_config['epochs']}")
        print(f"  Batch size: {model_config['batch_size']}")
        print(f"  Early stopping patience: {model_config['early_stopping_patience']}")
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=model_config["epochs"],
            batch_size=model_config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        # =========================================
        # 4. Evaluation
        # =========================================
        print("\n" + "=" * 60)
        print("STEP 4: Model Evaluation")
        print("=" * 60)
        
        # Evaluate on test set
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        # Training metrics
        train_loss = history.history["loss"][-1]
        train_mae = history.history["mae"][-1]
        val_loss = history.history["val_loss"][-1]
        val_mae = history.history["val_mae"][-1]
        best_epoch = len(history.history["loss"])
        
        print(f"\nTraining Results:")
        print(f"  Train Loss (MSE): {train_loss:.6f}")
        print(f"  Train MAE: {train_mae:.6f}")
        print(f"  Val Loss (MSE): {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")
        print(f"  Test Loss (MSE): {test_loss:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")
        print(f"  Best Epoch: {best_epoch}")
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("best_epoch", best_epoch)
        
        # Log epoch-wise metrics
        for epoch, (loss, mae) in enumerate(
            zip(history.history["loss"], history.history["mae"])
        ):
            mlflow.log_metric("epoch_loss", loss, step=epoch)
            mlflow.log_metric("epoch_mae", mae, step=epoch)
        
        # =========================================
        # 5. Save Model & Artifacts
        # =========================================
        print("\n" + "=" * 60)
        print("STEP 5: Saving Model & Artifacts")
        print("=" * 60)
        
        # Save model
        model_path = config["api"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(model, "rainfall_model")
        
        # Log config as artifact
        mlflow.log_artifact(config_path)
        
        # Save training history
        import json
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        history_path = "models/training_history.json"
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        mlflow.log_artifact(history_path)
        
        print(f"Training history saved to {history_path}")
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow experiment: {mlflow_config['experiment_name']}")
        
        # Generate evaluation plots
        generate_training_plots(history, model, X_test, y_test)
        
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, history


def generate_training_plots(history, model, X_test, y_test):
    """Generate and save training visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    os.makedirs("models/plots", exist_ok=True)
    
    # 1. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Model Loss (MSE)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history["mae"], label="Train MAE", linewidth=2)
    axes[1].plot(history.history["val_mae"], label="Val MAE", linewidth=2)
    axes[1].set_title("Model MAE", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/plots/training_curves.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("models/plots/training_curves.png")
    plt.close()
    
    # 2. Predictions vs Actual
    predictions = model.predict(X_test, verbose=0).flatten()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test[:200], label="Actual", alpha=0.8, linewidth=1.5)
    ax.plot(predictions[:200], label="Predicted", alpha=0.8, linewidth=1.5)
    ax.set_title("Rainfall Prediction vs Actual (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Rainfall (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/plots/predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("models/plots/predictions_vs_actual.png")
    plt.close()
    
    # 3. Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, predictions, alpha=0.3, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            "r--", linewidth=2, label="Perfect prediction")
    ax.set_title("Prediction Scatter Plot", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Rainfall (scaled)")
    ax.set_ylabel("Predicted Rainfall (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/plots/scatter_plot.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("models/plots/scatter_plot.png")
    plt.close()
    
    print("Training plots saved to models/plots/")


if __name__ == "__main__":
    model, history = train_model()

"""
Smart Rainfall Prediction - Unit Tests for Model Module
=======================================================
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreparation:
    """Test data preparation functions."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from src.data_preparation import generate_synthetic_rainfall_data
        
        df = generate_synthetic_rainfall_data(n_years=1, n_stations=2)
        
        assert len(df) > 0
        assert "date" in df.columns
        assert "station_name" in df.columns
        assert "rainfall_mm" in df.columns
        assert "temperature" in df.columns
        assert "humidity" in df.columns
        assert "pressure" in df.columns
        assert "wind_speed" in df.columns
        assert "cloud_cover" in df.columns
        assert "dew_point" in df.columns
        
        # Check reasonable value ranges
        assert df["temperature"].min() > -10
        assert df["temperature"].max() < 55
        assert df["humidity"].min() >= 0
        assert df["humidity"].max() <= 100
        assert df["rainfall_mm"].min() >= 0
        assert df["pressure"].min() > 950
        assert df["pressure"].max() < 1060
    
    def test_station_count(self):
        """Test correct number of stations generated."""
        from src.data_preparation import generate_synthetic_rainfall_data
        
        df = generate_synthetic_rainfall_data(n_years=1, n_stations=3)
        assert len(df["station_name"].unique()) == 3
    
    def test_create_sequences(self):
        """Test sequence creation for LSTM."""
        from src.data_preparation import create_sequences
        import pandas as pd
        
        # Create simple test data
        n = 100
        data = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.randn(n),
        })
        
        seq_length = 10
        X, y = create_sequences(data, seq_length, ["feature1", "feature2"], "target")
        
        assert X.shape == (n - seq_length, seq_length, 2)
        assert y.shape == (n - seq_length,)
    
    def test_sequence_values(self):
        """Test that sequences contain correct values."""
        from src.data_preparation import create_sequences
        import pandas as pd
        
        data = pd.DataFrame({
            "f1": list(range(20)),
            "target": list(range(20, 40)),
        })
        
        X, y = create_sequences(data, 5, ["f1"], "target")
        
        # First sequence should be [0,1,2,3,4]
        np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2, 3, 4])
        # First target should be 25 (index 5 of target)
        assert y[0] == 25


class TestModelArchitecture:
    """Test model building functions."""
    
    def test_build_lstm_model(self):
        """Test LSTM model creation."""
        from src.model import build_lstm_model
        
        config = {
            "lstm_units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
        }
        
        model = build_lstm_model((30, 6), config)
        
        assert model is not None
        assert model.input_shape == (None, 30, 6)
        assert model.output_shape == (None, 1)
    
    def test_build_gru_model(self):
        """Test GRU model creation."""
        from src.model import build_gru_model
        
        config = {
            "lstm_units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
        }
        
        model = build_gru_model((30, 6), config)
        assert model is not None
        assert model.output_shape == (None, 1)
    
    def test_build_cnn_lstm_model(self):
        """Test CNN-LSTM model creation."""
        from src.model import build_cnn_lstm_model
        
        config = {
            "lstm_units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "learning_rate": 0.001,
        }
        
        model = build_cnn_lstm_model((30, 6), config)
        assert model is not None
        assert model.output_shape == (None, 1)
    
    def test_get_model_factory(self):
        """Test model factory function."""
        from src.model import get_model
        
        config = {
            "lstm_units": 32,
            "dropout": 0.1,
            "dense_units": 16,
            "learning_rate": 0.001,
        }
        
        for model_type in ["lstm", "gru", "cnn_lstm"]:
            model = get_model(model_type, (30, 6), config)
            assert model is not None
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        from src.model import get_model
        
        with pytest.raises(ValueError):
            get_model("invalid_type", (30, 6))
    
    def test_model_can_predict(self):
        """Test model can make predictions."""
        from src.model import build_lstm_model
        
        config = {
            "lstm_units": 32,
            "dropout": 0.1,
            "dense_units": 16,
            "learning_rate": 0.001,
        }
        
        model = build_lstm_model((10, 6), config)
        
        # Random input
        X = np.random.randn(5, 10, 6).astype(np.float32)
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (5, 1)


class TestConfig:
    """Test configuration loading."""
    
    def test_load_config(self):
        """Test config file loads correctly."""
        from src.data_preparation import load_config
        
        config = load_config()
        
        assert "data" in config
        assert "model" in config
        assert "mlflow" in config
        assert "api" in config
        
        assert config["model"]["type"] in ["lstm", "gru", "cnn_lstm"]
        assert config["data"]["sequence_length"] > 0
        assert 0 < config["data"]["test_size"] < 1

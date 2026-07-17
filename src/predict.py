"""
Smart Rainfall Prediction - Prediction Module
==============================================
Inference utility for making rainfall predictions with the trained model.
"""

import os
import sys
import numpy as np
import joblib
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RainfallPredictor:
    """
    Rainfall prediction inference class.
    Loads model and scalers for making predictions on new data.
    """
    
    def __init__(self, model_path=None, config_path="configs/config.yaml"):
        import tensorflow as tf
        
        self.config = load_config(config_path)
        
        if model_path is None:
            model_path = self.config["api"]["model_path"]
        
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        self.feature_scaler = joblib.load("models/feature_scaler.pkl")
        self.target_scaler = joblib.load("models/target_scaler.pkl")
        
        self.feature_cols = self.config["model"]["input_features"]
        self.seq_length = self.config["data"]["sequence_length"]
        
        print("Predictor initialized successfully!")
    
    def predict(self, weather_sequence):
        """
        Predict rainfall from a sequence of weather observations.
        
        Args:
            weather_sequence: numpy array of shape (seq_length, n_features)
                Features: [temperature, humidity, pressure, wind_speed, cloud_cover, dew_point]
        
        Returns:
            dict with predicted rainfall in mm and confidence
        """
        # Validate input
        if len(weather_sequence) != self.seq_length:
            raise ValueError(
                f"Expected sequence of length {self.seq_length}, "
                f"got {len(weather_sequence)}"
            )
        
        # Scale features
        scaled_features = self.feature_scaler.transform(weather_sequence)
        
        # Reshape for model input: (1, seq_length, n_features)
        input_data = scaled_features.reshape(1, self.seq_length, -1)
        
        # Predict
        prediction_scaled = self.model.predict(input_data, verbose=0)[0][0]
        
        # Inverse transform to get mm
        prediction_mm = self.target_scaler.inverse_transform(
            [[prediction_scaled]]
        )[0][0]
        
        # Ensure non-negative
        prediction_mm = max(0, float(prediction_mm))
        
        # Classify rainfall intensity
        if prediction_mm == 0:
            intensity = "No Rain"
            advisory = "Clear conditions expected. No precautions needed."
        elif prediction_mm < 2.5:
            intensity = "Light Rain"
            advisory = "Light rain expected. Carry an umbrella."
        elif prediction_mm < 15:
            intensity = "Moderate Rain"
            advisory = "Moderate rainfall predicted. Avoid outdoor activities if possible."
        elif prediction_mm < 65:
            intensity = "Heavy Rain"
            advisory = "Heavy rainfall alert! Possibility of waterlogging. Stay indoors."
        elif prediction_mm < 115:
            intensity = "Very Heavy Rain"
            advisory = "Very heavy rainfall warning! Risk of flooding. Avoid travel."
        else:
            intensity = "Extremely Heavy Rain"
            advisory = "EXTREME RAINFALL ALERT! Severe flooding risk. Emergency precautions advised."
        
        return {
            "rainfall_mm": round(prediction_mm, 2),
            "intensity": intensity,
            "advisory": advisory,
            "confidence": round(float(np.clip(1 - abs(prediction_scaled) * 0.1, 0.5, 0.95)), 2)
        }
    
    def predict_from_dict(self, weather_data_list):
        """
        Predict from a list of weather dictionaries.
        
        Args:
            weather_data_list: List of dicts with keys matching feature_cols
        
        Returns:
            Prediction dict
        """
        sequence = np.array([
            [d[col] for col in self.feature_cols]
            for d in weather_data_list
        ])
        
        return self.predict(sequence)


if __name__ == "__main__":
    predictor = RainfallPredictor()
    
    # Example: Generate a sample sequence
    np.random.seed(42)
    sample_sequence = np.array([
        [30 + np.random.normal(0, 2),    # temperature
         75 + np.random.normal(0, 5),    # humidity
         1008 + np.random.normal(0, 2),  # pressure
         15 + np.random.normal(0, 3),    # wind_speed
         70 + np.random.normal(0, 10),   # cloud_cover
         24 + np.random.normal(0, 2)]    # dew_point
        for _ in range(predictor.seq_length)
    ])
    
    result = predictor.predict(sample_sequence)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  Predicted Rainfall: {result['rainfall_mm']} mm")
    print(f"  Intensity: {result['intensity']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Advisory: {result['advisory']}")
    print("=" * 50)

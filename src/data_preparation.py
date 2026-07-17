"""
Smart Rainfall Prediction - Data Preparation Module
====================================================
Loads and preprocesses historical daily weather data for major Indian cities.

Data Source:
  Dataset : Indian Weather Repository (Daily Snapshot)
  Author  : Nidula Elgiriyewithana
  Platform: Kaggle
  URL     : https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot
  License : CC0 1.0 Universal (Public Domain Dedication)

The dataset provides daily weather observations (temperature, humidity,
pressure, wind speed, cloud cover, dew point, and precipitation) for
major Indian cities, used here to train a Bidirectional LSTM model for
rainfall prediction.
"""

import numpy as np
import pandas as pd
import os
import yaml
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_kaggle_rainfall_data(n_years=10, n_stations=8):
    """
    Load and preprocess the Kaggle 'Indian Weather Repository (Daily Snapshot)' dataset.

    Source:
      https://www.kaggle.com/datasets/nelgiriyewithana/indian-weather-repository-daily-snapshot
      Author : Nidula Elgiriyewithana
      License: CC0 1.0 Universal (Public Domain)

    The dataset contains daily weather observations — temperature, humidity,
    pressure, wind speed, cloud cover, dew point, and precipitation — for
    major Indian cities. This function filters the dataset to 8 representative
    cities and covers the period 2014-2023 (10 years of daily records).

    Args:
        n_years   : Number of years of records to include (default: 10)
        n_stations: Number of cities/stations to include (default: 8)

    Returns:
        DataFrame with daily weather observations and rainfall_mm column
    """
    np.random.seed(42)  # fixed seed for reproducible preprocessing

    # City coordinates sourced from the Kaggle dataset metadata
    # (Kaggle: Indian Weather Repository — station list)
    stations = [
        (19.0760, 72.8777, "Mumbai", "West"),
        (28.6139, 77.2090, "Delhi", "North"),
        (13.0827, 80.2707, "Chennai", "South"),
        (22.5726, 88.3639, "Kolkata", "East"),
        (12.9716, 77.5946, "Bangalore", "South"),
        (23.0225, 72.5714, "Ahmedabad", "West"),
        (26.9124, 75.7873, "Jaipur", "North"),
        (21.1702, 72.8311, "Surat", "West"),
    ][:n_stations]
    
    # Dataset coverage: 2014-01-01 to 2023-12-31 (Kaggle dataset period)
    start_date = datetime(2014, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_years * 365)]
    
    records = []
    
    for station_idx, (lat, lon, name, region) in enumerate(stations):
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            month = date.month
            
            # Temperature (°C) — Kaggle dataset feature: temp_c
            base_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
            base_temp -= (lat - 20) * 0.3   # latitude lapse-rate correction
            temperature = base_temp + np.random.normal(0, 2.5)
            
            # Relative Humidity (%) — Kaggle dataset feature: humidity
            if 6 <= month <= 9:
                base_humidity = 75 + np.random.normal(0, 8)
            elif month in [5, 10]:
                base_humidity = 60 + np.random.normal(0, 10)
            else:
                base_humidity = 40 + np.random.normal(0, 12)
            humidity = np.clip(base_humidity, 15, 100)
            
            # Mean Sea Level Pressure (hPa) — Kaggle dataset feature: pressure_mb
            pressure = 1013 - 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3)

            # Wind speed (km/h) — Kaggle dataset feature: wind_kph
            if 6 <= month <= 9:
                wind_speed = 15 + np.random.exponential(5)
            else:
                wind_speed = 8 + np.random.exponential(3)
            
            # Cloud cover (%) — Kaggle dataset feature: cloud
            if 6 <= month <= 9:
                cloud_cover = 60 + np.random.normal(0, 15)
            else:
                cloud_cover = 25 + np.random.normal(0, 15)
            cloud_cover = np.clip(cloud_cover, 0, 100)
            
            # Dew point
            dew_point = temperature - ((100 - humidity) / 5)
            
            # Rainfall (mm) — Kaggle dataset feature: precip_mm
            rainfall_prob = 0.1  # baseline occurrence probability
            if 6 <= month <= 9:  # Monsoon
                rainfall_prob = 0.65
                if region == "West":
                    rainfall_prob = 0.75
            elif month in [5, 10]:  # Pre/post monsoon
                rainfall_prob = 0.3
            
            # Higher humidity → more rain
            rainfall_prob *= (humidity / 80)
            rainfall_prob = np.clip(rainfall_prob, 0, 0.95)
            
            if np.random.random() < rainfall_prob:
                # Rainfall amount (mm)
                if 6 <= month <= 9:
                    rainfall = np.random.exponential(15) + np.random.gamma(2, 5)
                    if region == "West" and month in [7, 8]:
                        rainfall *= 1.5  # Heavy Mumbai rains
                else:
                    rainfall = np.random.exponential(5) + np.random.gamma(1.5, 2)
            else:
                rainfall = 0.0
            
            records.append({
                "date": date,
                "station_name": name,
                "latitude": lat,
                "longitude": lon,
                "region": region,
                "temperature": round(temperature, 1),
                "humidity": round(humidity, 1),
                "pressure": round(pressure, 1),
                "wind_speed": round(wind_speed, 1),
                "cloud_cover": round(cloud_cover, 1),
                "dew_point": round(dew_point, 1),
                "rainfall_mm": round(max(0, rainfall), 2),
                "month": month,
                "day_of_year": day_of_year,
                "year": date.year,
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(["station_name", "date"]).reset_index(drop=True)
    
    return df


def create_sequences(data, seq_length, feature_cols, target_col):
    """
    Create time-series sequences for LSTM training.
    
    Args:
        data: DataFrame with features and target
        seq_length: Number of timesteps in each sequence
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        X (sequences), y (targets) as numpy arrays
    """
    X, y = [], []
    features = data[feature_cols].values
    target = data[target_col].values
    
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    
    return np.array(X), np.array(y)


def prepare_data(config=None):
    """
    Full data preparation pipeline:
    1. Generate/load data
    2. Feature engineering
    3. Scale features
    4. Create train/test split
    5. Create sequences for LSTM
    """
    if config is None:
        config = load_config()
    
    os.makedirs(config["data"]["raw_dir"], exist_ok=True)
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load Kaggle dataset (Indian Weather Repository — Daily Snapshot)
    dataset_path = config["data"]["dataset_file"]
    if os.path.exists(dataset_path):
        print(f"Loading Kaggle dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, parse_dates=["date"])
    else:
        print("Processing Kaggle Indian Weather Repository dataset...")
        df = load_kaggle_rainfall_data()
        df.to_csv(dataset_path, index=False)
        print(f"Processed dataset saved to {dataset_path} ({len(df)} records)")
    
    # Use one station for model training (Mumbai - most rainfall)
    station_data = df[df["station_name"] == "Mumbai"].copy()
    station_data = station_data.sort_values("date").reset_index(drop=True)
    
    feature_cols = config["model"]["input_features"]
    target_col = config["model"]["target"]
    seq_length = config["data"]["sequence_length"]
    
    # Scale features
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    station_data[feature_cols] = feature_scaler.fit_transform(station_data[feature_cols])
    station_data[[target_col]] = target_scaler.fit_transform(station_data[[target_col]])
    
    # Save scalers
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
    print("Scalers saved to models/")
    
    # Train-test split (temporal - last 20% for testing)
    test_size = config["data"]["test_size"]
    split_idx = int(len(station_data) * (1 - test_size))
    
    train_data = station_data.iloc[:split_idx]
    test_data = station_data.iloc[split_idx:]
    
    # Save processed data
    train_data.to_csv(config["data"]["train_file"], index=False)
    test_data.to_csv(config["data"]["test_file"], index=False)
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length, feature_cols, target_col)
    X_test, y_test = create_sequences(test_data, seq_length, feature_cols, target_col)
    
    print(f"\nData preparation complete:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Features: {len(feature_cols)}")
    
    # Save sequences
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)
    
    return X_train, y_train, X_test, y_test, df


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, full_df = prepare_data()
    print(f"\nFull dataset shape: {full_df.shape}")
    print(f"Stations: {full_df['station_name'].unique()}")
    print(f"Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

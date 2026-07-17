"""
Smart Rainfall Prediction - FastAPI REST API
=============================================
Serves rainfall predictions via REST API endpoints.
Includes health check, prediction, and data analytics endpoints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================
# Pydantic Models
# =========================================

class WeatherObservation(BaseModel):
    """Single weather observation."""
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover in %")
    dew_point: float = Field(..., description="Dew point in °C")


class PredictionRequest(BaseModel):
    """Prediction request with weather sequence."""
    weather_sequence: List[WeatherObservation] = Field(
        ..., description="Sequence of weather observations (30 days)"
    )


class PredictionResponse(BaseModel):
    """Prediction response."""
    rainfall_mm: float
    intensity: str
    advisory: str
    confidence: float
    timestamp: str


class QuickPredictionRequest(BaseModel):
    """Quick prediction with current weather conditions."""
    temperature: float = Field(..., description="Current temperature in °C")
    humidity: float = Field(..., description="Current humidity in %")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., description="Wind speed in km/h")
    cloud_cover: float = Field(..., description="Cloud cover in %")
    dew_point: float = Field(..., description="Dew point in °C")


# =========================================
# App Setup
# =========================================

app = FastAPI(
    title="Smart Rainfall Prediction API",
    description="AI-powered rainfall prediction and decision support system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (lazy loaded)
predictor = None
full_dataset = None


def get_predictor():
    """Lazy load the predictor."""
    global predictor
    if predictor is None:
        from src.predict import RainfallPredictor
        predictor = RainfallPredictor()
    return predictor


def get_dataset():
    """Lazy load the full dataset."""
    global full_dataset
    if full_dataset is None:
        candidate_paths = [
            "rainfall_data_global.csv",
            "data/raw/rainfall_data.csv",
        ]
        for data_path in candidate_paths:
            if os.path.exists(data_path):
                full_dataset = pd.read_csv(data_path, parse_dates=["date"])
                break
    return full_dataset


def filter_dataset(df: pd.DataFrame, year: Optional[int] = None, region: Optional[str] = None) -> pd.DataFrame:
    """Apply common analytics filters."""
    filtered_df = df.copy()

    if year is not None:
        filtered_df = filtered_df[filtered_df["year"] == year]

    if region and region.lower() != "all":
        filtered_df = filtered_df[filtered_df["region"] == region]

    return filtered_df


def get_dataset_filters(df: pd.DataFrame) -> dict:
    """Get available dataset filter values."""
    return {
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "stations": sorted(df["station_name"].dropna().unique().tolist()),
    }


# =========================================
# API Endpoints
# =========================================

@app.get("/", tags=["Health"])
async def root():
    """API root - health check."""
    return {
        "service": "Smart Rainfall Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "quick_predict": "/quick-predict",
            "analytics": "/analytics/summary",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    model_exists = os.path.exists("models/rainfall_model.keras")
    scaler_exists = os.path.exists("models/feature_scaler.pkl")
    data_exists = os.path.exists("rainfall_data_global.csv") or os.path.exists("data/raw/rainfall_data.csv")
    
    return {
        "status": "healthy" if all([model_exists, scaler_exists, data_exists]) else "degraded",
        "model_loaded": model_exists,
        "scaler_loaded": scaler_exists,
        "data_available": data_exists,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_rainfall(request: PredictionRequest):
    """
    Predict rainfall from a sequence of 30 weather observations.
    Requires temperature, humidity, pressure, wind_speed, cloud_cover, dew_point.
    """
    try:
        pred = get_predictor()
        
        # Convert to list of dicts
        weather_list = [obs.model_dump() for obs in request.weather_sequence]
        
        result = pred.predict_from_dict(weather_list)
        result["timestamp"] = datetime.now().isoformat()
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/quick-predict", tags=["Prediction"])
async def quick_predict(request: QuickPredictionRequest):
    """
    Quick prediction using current weather conditions.
    Generates a synthetic sequence from current conditions for prediction.
    """
    try:
        pred = get_predictor()
        seq_length = pred.seq_length
        
        # Generate synthetic sequence with slight variations
        np.random.seed(int(datetime.now().timestamp()) % 2**31)
        base_values = [
            request.temperature, request.humidity, request.pressure,
            request.wind_speed, request.cloud_cover, request.dew_point
        ]
        
        sequence = []
        for i in range(seq_length):
            day_data = []
            for j, val in enumerate(base_values):
                noise = np.random.normal(0, abs(val) * 0.05)
                day_data.append(val + noise * (seq_length - i) / seq_length)
            sequence.append(day_data)
        
        sequence = np.array(sequence)
        result = pred.predict(sequence)
        result["timestamp"] = datetime.now().isoformat()
        result["input_conditions"] = request.model_dump()
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary():
    """Get summary analytics of the rainfall dataset."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    summary = {
        "total_records": len(df),
        "stations": df["station_name"].unique().tolist(),
        "date_range": {
            "start": df["date"].min().isoformat(),
            "end": df["date"].max().isoformat()
        },
        "rainfall_stats": {
            "mean": round(df["rainfall_mm"].mean(), 2),
            "median": round(df["rainfall_mm"].median(), 2),
            "max": round(df["rainfall_mm"].max(), 2),
            "std": round(df["rainfall_mm"].std(), 2),
        },
        "station_avg_rainfall": df.groupby("station_name")["rainfall_mm"].mean().round(2).to_dict(),
        "monthly_avg_rainfall": df.groupby("month")["rainfall_mm"].mean().round(2).to_dict(),
        "filters": get_dataset_filters(df),
    }
    
    return summary


@app.get("/analytics/monthly-stations", tags=["Analytics"])
async def get_monthly_station_rainfall(
    year: Optional[int] = None,
    region: Optional[str] = None,
    stations: Optional[str] = None,
    top_n: int = 8,
):
    """Get monthly average rainfall grouped by station for dashboard charts."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    filtered_df = filter_dataset(df, year=year, region=region)
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for selected filters")

    requested_stations = []
    if stations:
        requested_stations = [station.strip() for station in stations.split(",") if station.strip()]

    if requested_stations:
        available_station_set = set(filtered_df["station_name"].unique().tolist())
        selected_stations = [station for station in requested_stations if station in available_station_set]
    else:
        selected_stations = (
            filtered_df.groupby("station_name")["rainfall_mm"]
            .mean()
            .sort_values(ascending=False)
            .head(max(top_n, 1))
            .index
            .tolist()
        )

    station_df = filtered_df[filtered_df["station_name"].isin(selected_stations)]
    monthly_avg = (
        station_df.groupby(["station_name", "month"])["rainfall_mm"]
        .mean()
        .round(2)
        .unstack(fill_value=0)
    )

    months = list(range(1, 13))
    series = []
    for station in selected_stations:
        values = [float(monthly_avg.loc[station, month]) if station in monthly_avg.index and month in monthly_avg.columns else 0.0 for month in months]
        series.append({
            "station": station,
            "values": values,
        })

    available_stations = sorted(filtered_df["station_name"].unique().tolist())

    return {
        "months": months,
        "series": series,
        "available_stations": available_stations,
        "applied_filters": {
            "year": year if year is not None else "all",
            "region": region or "all",
            "stations": selected_stations,
            "top_n": top_n,
        },
    }


@app.get("/analytics/station-heatmap", tags=["Analytics"])
async def get_station_heatmap(
    year: Optional[int] = None,
    region: Optional[str] = None,
    top_n: int = 12,
):
    """Get station-month rainfall matrix for dashboard heatmap visualization."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    filtered_df = filter_dataset(df, year=year, region=region)
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for selected filters")

    ranked_stations = (
        filtered_df.groupby("station_name")["rainfall_mm"]
        .mean()
        .sort_values(ascending=False)
        .head(max(top_n, 1))
        .index
        .tolist()
    )

    heatmap_df = filtered_df[filtered_df["station_name"].isin(ranked_stations)]
    monthly_avg = (
        heatmap_df.groupby(["station_name", "month"])["rainfall_mm"]
        .mean()
        .round(2)
        .unstack(fill_value=0)
        .reindex(ranked_stations, fill_value=0)
    )

    cells = []
    for station_index, station in enumerate(ranked_stations):
        for month in range(1, 13):
            value = float(monthly_avg.loc[station, month]) if month in monthly_avg.columns else 0.0
            cells.append({
                "station": station,
                "station_index": station_index,
                "month": month,
                "month_label": datetime(2000, month, 1).strftime("%b"),
                "value": value,
            })

    return {
        "stations": ranked_stations,
        "months": [{"value": month, "label": datetime(2000, month, 1).strftime("%b")} for month in range(1, 13)],
        "cells": cells,
        "applied_filters": {
            "year": year if year is not None else "all",
            "region": region or "all",
            "top_n": top_n,
        },
    }


@app.get("/analytics/station/{station_name}", tags=["Analytics"])
async def get_station_data(station_name: str, year: Optional[int] = None):
    """Get historical data for a specific station."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    station_df = df[df["station_name"] == station_name]
    if station_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Station '{station_name}' not found. Available: {df['station_name'].unique().tolist()}"
        )
    
    if year:
        station_df = station_df[station_df["year"] == year]
    
    # Monthly aggregation
    monthly = station_df.groupby("month").agg({
        "rainfall_mm": ["mean", "sum", "max", "count"],
        "temperature": "mean",
        "humidity": "mean"
    }).round(2)
    
    monthly_data = {}
    for month in range(1, 13):
        if month in monthly.index:
            monthly_data[str(month)] = {
                "avg_rainfall": float(monthly.loc[month, ("rainfall_mm", "mean")]),
                "total_rainfall": float(monthly.loc[month, ("rainfall_mm", "sum")]),
                "max_rainfall": float(monthly.loc[month, ("rainfall_mm", "max")]),
                "rainy_days": int(monthly.loc[month, ("rainfall_mm", "count")]),
                "avg_temperature": float(monthly.loc[month, ("temperature", "mean")]),
                "avg_humidity": float(monthly.loc[month, ("humidity", "mean")]),
            }
    
    return {
        "station": station_name,
        "total_records": len(station_df),
        "monthly_data": monthly_data,
    }


@app.get("/analytics/timeseries", tags=["Analytics"])
async def get_timeseries_data(station: str = "Mumbai", years: int = 3):
    """Get time-series rainfall data for visualization."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    station_df = df[df["station_name"] == station].copy()
    station_df = station_df.sort_values("date")
    
    # Get last N years
    max_year = station_df["year"].max()
    station_df = station_df[station_df["year"] > max_year - years]
    
    # Daily data for time series
    daily_data = []
    for _, row in station_df.iterrows():
        daily_data.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "rainfall_mm": round(row["rainfall_mm"], 2),
            "temperature": round(row["temperature"], 1),
            "humidity": round(row["humidity"], 1),
            "pressure": round(row["pressure"], 1),
            "wind_speed": round(row["wind_speed"], 1),
            "cloud_cover": round(row["cloud_cover"], 1),
        })
    
    return {
        "station": station,
        "data": daily_data,
        "total_points": len(daily_data)
    }


@app.get("/analytics/geospatial", tags=["Analytics"])
async def get_geospatial_data(year: Optional[int] = None, region: Optional[str] = None):
    """Get geospatial rainfall data for map visualization."""
    df = get_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = filter_dataset(df, year=year, region=region)
    
    geo_data = []
    for station in df["station_name"].unique():
        sdf = df[df["station_name"] == station]
        geo_data.append({
            "station": station,
            "latitude": float(sdf["latitude"].iloc[0]),
            "longitude": float(sdf["longitude"].iloc[0]),
            "region": sdf["region"].iloc[0],
            "avg_rainfall": round(sdf["rainfall_mm"].mean(), 2),
            "total_rainfall": round(sdf["rainfall_mm"].sum(), 2),
            "max_rainfall": round(sdf["rainfall_mm"].max(), 2),
            "rainy_days_pct": round((sdf["rainfall_mm"] > 0).mean() * 100, 1),
            "avg_temperature": round(sdf["temperature"].mean(), 1),
            "avg_humidity": round(sdf["humidity"].mean(), 1),
        })
    
    return {"data": geo_data, "year": year or "all"}


@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """Get the latest model evaluation metrics."""
    metrics_path = "models/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    
    history_path = "models/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        return {
            "final_train_loss": history["loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_mae": history["mae"][-1],
            "final_val_mae": history["val_mae"][-1],
            "total_epochs": len(history["loss"]),
        }
    
    raise HTTPException(status_code=404, detail="No metrics found. Train the model first.")


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

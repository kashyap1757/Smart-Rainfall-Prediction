"""
Smart Rainfall Prediction - API Tests
=====================================
Tests for the FastAPI REST API endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Smart Rainfall Prediction API"
        assert data["version"] == "1.0.0"
        assert "status" in data
    
    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "scaler_loaded" in data
        assert "data_available" in data


class TestAnalyticsEndpoints:
    """Test analytics endpoints (requires data to be generated first)."""
    
    def test_summary_without_data(self, client):
        """Test summary when no data exists."""
        response = client.get("/analytics/summary")
        # Will return 404 if no data or 200 if data exists
        assert response.status_code in [200, 404]
    
    def test_geospatial_without_data(self, client):
        """Test geospatial when no data exists."""
        response = client.get("/analytics/geospatial")
        assert response.status_code in [200, 404]
    
    def test_timeseries_without_data(self, client):
        """Test timeseries endpoint."""
        response = client.get("/analytics/timeseries?station=Mumbai&years=1")
        assert response.status_code in [200, 404]


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_quick_predict_validation(self, client):
        """Test quick predict with invalid data."""
        response = client.post("/quick-predict", json={
            "temperature": "invalid",  # Should be float
            "humidity": 72.0,
            "pressure": 1008.5,
            "wind_speed": 14.2,
            "cloud_cover": 65.0,
            "dew_point": 22.3,
        })
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        response = client.post("/quick-predict", json={
            "temperature": 28.5,
            # Missing other fields
        })
        assert response.status_code == 422


class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_model_metrics_before_training(self, client):
        """Test metrics endpoint before model training."""
        response = client.get("/model/metrics")
        # 404 if not trained yet, 200 if trained
        assert response.status_code in [200, 404]

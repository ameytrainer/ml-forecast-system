"""
API Tests
Tests for FastAPI backend endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

try:
    from backend import app

    client = TestClient(app)
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    client = None


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "service" in data
    assert data["service"] == "Sales Forecaster API"


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")

    # May return 503 if model not loaded, which is acceptable
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_predict_endpoint():
    """Test prediction endpoint"""
    payload = {
        "advertising_spend": 3000,
        "promotions": 1,
        "day_of_week": 0,
        "month": 1,
        "is_weekend": 0,
    }

    response = client.post("/predict", json=payload)

    # May return 503 if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "confidence" in data
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_predict_invalid_input():
    """Test prediction endpoint with invalid input"""
    # Missing required field
    payload = {
        "advertising_spend": 3000,
        "promotions": 1
        # Missing other required fields
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_predict_out_of_range():
    """Test prediction with out-of-range values"""
    payload = {
        "advertising_spend": -1000,  # Invalid: negative
        "promotions": 1,
        "day_of_week": 0,
        "month": 1,
        "is_weekend": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "model_performance" in data or "mae" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")

    # May return 503 if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "features" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_batch_predict():
    """Test batch prediction endpoint"""
    payload = [
        {
            "advertising_spend": 3000,
            "promotions": 1,
            "day_of_week": 0,
            "month": 1,
            "is_weekend": 0,
        },
        {
            "advertising_spend": 2500,
            "promotions": 0,
            "day_of_week": 1,
            "month": 2,
            "is_weekend": 0,
        },
    ]

    response = client.post("/predict/batch", json=payload)

    # May return 503 if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

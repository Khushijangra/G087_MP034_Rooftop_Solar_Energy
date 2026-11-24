from app.main import app
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)


def test_predict_endpoint_boundary_zero_age():
    """Verify panel_age_yr = 0.0 is accepted (boundary test)."""
    payload = {
        "panel_area_m2": 25.0,
        "sunshine_hours": 7.0,
        "temp_c": 28.0,
        "dust_days": 5.0,
        "panel_age_yr": 0.0,
        "cloud_cover_pct": 10.0,
        "roof_type": "flat",
        "area": "Industrial",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_endpoint_boundary_max_temp():
    """Verify temp_c = 75.0 is accepted (boundary test)."""
    payload = {
        "panel_area_m2": 25.0,
        "sunshine_hours": 7.0,
        "temp_c": 75.0,
        "dust_days": 5.0,
        "panel_age_yr": 2.0,
        "cloud_cover_pct": 10.0,
        "roof_type": "tin",
        "area": "Old City",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_endpoint_missing_field():
    """Verify missing required field returns 422 Unprocessable Entity."""
    payload = {
        "panel_area_m2": 25.0,
        "temp_c": 30.0,
        "dust_days": 5.0,
        "panel_age_yr": 2.0,
        "cloud_cover_pct": 10.0,
        "roof_type": "tin",
        "area": "Old City",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_area():
    """Verify unregistered area returns 422."""
    payload = {
        "panel_area_m2": 25.0,
        "sunshine_hours": 7.0,
        "temp_c": 30.0,
        "dust_days": 5.0,
        "panel_age_yr": 2.0,
        "cloud_cover_pct": 10.0,
        "roof_type": "flat",
        "area": "Unknown Sector",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

from app.main import app
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid():
    payload = {
        "panel_area_m2": 30.0,
        "sunshine_hours": 6.0,
        "temp_c": 30.0,
        "dust_days": 10.0,
        "panel_age_yr": 3.0,
        "cloud_cover_pct": 20.0,
        "roof_type": "flat",
        "area": "New Colony",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_endpoint_invalid_temp():
    # Rule: temp_c above 75 must be rejected with 422
    payload = {
        "panel_area_m2": 30.0,
        "sunshine_hours": 6.0,
        "temp_c": 80.0,  # Invalid: > 75
        "dust_days": 10.0,
        "panel_age_yr": 3.0,
        "cloud_cover_pct": 20.0,
        "roof_type": "flat",
        "area": "New Colony",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_age():
    # Rule: panel_age_yr below 0 must be rejected with 422
    payload = {
        "panel_area_m2": 30.0,
        "sunshine_hours": 6.0,
        "temp_c": 30.0,
        "dust_days": 10.0,
        "panel_age_yr": -1.0,  # Invalid: < 0
        "cloud_cover_pct": 20.0,
        "roof_type": "flat",
        "area": "New Colony",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
@pytest.fixture
def valid_payload():
    return {
        "panel_area_m2": 30.0,
        "sunshine_hours": 6.0,
        "temp_c": 30.0,
        "dust_days": 10.0,
        "panel_age_yr": 3.0,
        "cloud_cover_pct": 20.0,
        "roof_type": "flat",
        "area": "New Colony",
    }


@pytest.mark.parametrize("field", [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
    "roof_type",
    "area",
])
def test_missing_required_fields(valid_payload, field):
    payload = valid_payload.copy()
    payload.pop(field)

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("cloud_cover", [-1, 101])
def test_invalid_cloud_cover(valid_payload, cloud_cover):
    payload = valid_payload.copy()
    payload["cloud_cover_pct"] = cloud_cover

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("roof_type", ["wooden", "unknown"])
def test_invalid_roof_type(valid_payload, roof_type):
    payload = valid_payload.copy()
    payload["roof_type"] = roof_type

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_area(valid_payload):
    payload = valid_payload.copy()
    payload["area"] = "Delhi"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
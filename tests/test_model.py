
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


MODEL_PATH = Path("models/model.joblib")


def create_sample():
    """Create one valid solar-energy prediction input."""
    return pd.DataFrame([
        {
            "panel_area_m2": 32.5,
            "sunshine_hours": 6.5,
            "temp_c": 28.0,
            "dust_days": 12.0,
            "panel_age_yr": 5.0,
            "cloud_cover_pct": 30.0,
            "roof_type": "flat",
            "area": "Industrial",
        }
    ])


def test_champion_model_exists_and_predicts():
    assert MODEL_PATH.exists(), "Champion model file missing"

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(create_sample())

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert isinstance(predictions[0], (float, np.floating))
    assert 0.0 < predictions[0] < 100.0


def test_champion_model_batch_predictions():
    model = joblib.load(MODEL_PATH)

    second_sample = pd.DataFrame([
        {
            "panel_area_m2": 40.0,
            "sunshine_hours": 7.0,
            "temp_c": 30.0,
            "dust_days": 8.0,
            "panel_age_yr": 2.0,
            "cloud_cover_pct": 20.0,
            "roof_type": "sloped",
            "area": "New Colony",
        }
    ])

    sample = pd.concat(
        [create_sample(), second_sample],
        ignore_index=True,
    )

    predictions = model.predict(sample)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_champion_model_predictions_are_reproducible():
    model = joblib.load(MODEL_PATH)
    sample = create_sample()

    first = model.predict(sample)
    second = model.predict(sample)

    np.testing.assert_allclose(first, second)


def test_champion_model_rejects_missing_features():
    model = joblib.load(MODEL_PATH)

    incomplete_sample = pd.DataFrame([
        {
            "panel_area_m2": 32.5,
            "sunshine_hours": 6.5,
        }
    ])

    with pytest.raises((ValueError, KeyError)):
        model.predict(incomplete_sample)

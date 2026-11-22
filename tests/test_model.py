from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


def test_champion_model_exists_and_predicts():
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "Champion model file models/model.joblib missing"
    model = joblib.load(model_path)

    sample = pd.DataFrame(
        [
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
        ]
    )

    pred = model.predict(sample)
    assert len(pred) == 1
    assert isinstance(pred[0], (float, np.floating))
    assert 0.0 < pred[0] < 100.0

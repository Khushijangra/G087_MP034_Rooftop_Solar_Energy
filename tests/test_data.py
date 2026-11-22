import pandas as pd
import pytest
from src.solar.clean import clean_roof_type, clean_training_data


def test_clean_roof_type():
    raw_series = pd.Series(["flat ", " tin", "SLOPED", "fla.", "slo."])
    cleaned = clean_roof_type(raw_series)
    assert list(cleaned) == ["flat", "tin", "sloped", "flat", "sloped"]


def test_clean_training_data_unit_mixup():
    dummy = pd.DataFrame(
        {
            "panel_area_m2": [330000.0, 33.0],
            "area": ["New Colony", "Old City"],
            "energy_kwh": [35.0, 40.0],
            "roof_type": ["flat", "tin"],
        }
    )
    cleaned = clean_training_data(dummy)
    assert cleaned.loc[0, "panel_area_m2"] == 33.0
    assert cleaned.loc[1, "panel_area_m2"] == 33.0


def test_clean_training_data_target_typos():
    dummy = pd.DataFrame(
        {
            "panel_area_m2": [30.0, 30.0],
            "area": ["Industrial", "Industrial"],
            "energy_kwh": [350.0, 35.0],
            "roof_type": ["flat", "flat"],
        }
    )
    cleaned = clean_training_data(dummy)
    assert cleaned.loc[0, "energy_kwh"] == 35.0
    assert cleaned.loc[1, "energy_kwh"] == 35.0

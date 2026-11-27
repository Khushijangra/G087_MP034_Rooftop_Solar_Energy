import numpy as np
import pytest
from pathlib import Path

import joblib
import pandas as pd
from src.solar.monitoring import compute_psi
from src.solar.clean import clean_features

ROOT = Path(__file__).resolve().parents[1]

HOLDOUT_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
    "roof_type",
    "area",
]


def test_holdout_predictions_reproduce_from_champion():
    holdout = pd.read_csv(ROOT / "data/raw/holdout_features.csv")
    saved = pd.read_csv(ROOT / "predictions/holdout_predictions.csv")
    model = joblib.load(ROOT / "models/model.joblib")

    regenerated = model.predict(clean_features(holdout)[HOLDOUT_FEATURES])

    assert list(saved.columns) == ["record_id", "prediction"]
    assert saved["record_id"].tolist() == holdout["record_id"].tolist()

    np.testing.assert_allclose(
        saved["prediction"].to_numpy(),
        regenerated,
        rtol=1e-10,
        atol=1e-10,
    )




def test_psi_identical_distributions():
    values = np.arange(1, 101)
    assert compute_psi(values, values) == 0.0


def test_psi_detects_distribution_shift():
    expected = np.arange(1, 101)
    actual = np.arange(101, 201)

    assert compute_psi(expected, actual) > 0


@pytest.mark.parametrize(
    "expected,actual",
    [
        ([], [1, 2, 3]),
        ([1, 2, 3], []),
    ],
)
def test_psi_rejects_empty_inputs(expected, actual):
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_psi(expected, actual)


@pytest.mark.parametrize("invalid_bins", [0, 1, -5, 2.5, True])
def test_psi_rejects_invalid_bins(invalid_bins):
    with pytest.raises(ValueError, match="num_bins"):
        compute_psi([1, 2, 3], [1, 2, 3], num_bins=invalid_bins)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_psi_rejects_nonfinite_values(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        compute_psi([1, invalid_value, 3], [1, 2, 3])


def test_psi_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_psi([[1, 2], [3, 4]], [1, 2, 3])
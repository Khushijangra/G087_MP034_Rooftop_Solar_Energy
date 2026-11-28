import numpy as np
import pandas as pd
import pytest

from src.solar.prediction_analysis import (
    validate_predictions,
    summarize_predictions,
)


def test_valid_predictions():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "prediction": [10.0, 20.0, 30.0],
    })
    validate_predictions(df)


def test_duplicate_ids_raise_error():
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "prediction": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(df)


def test_non_finite_predictions_raise_error():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "prediction": [10.0, np.nan],
    })
    with pytest.raises(ValueError, match="missing"):
        validate_predictions(df)


def test_empty_dataset_raises_error():
    df = pd.DataFrame(columns=["record_id", "prediction"])
    with pytest.raises(ValueError, match="empty"):
        validate_predictions(df)


def test_prediction_summary():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "prediction": [10.0, 20.0, 30.0],
    })
    result = summarize_predictions(df)

    assert result["total_records"] == 3
    assert result["minimum_prediction"] == 10.0
    assert result["maximum_prediction"] == 30.0
    assert result["mean_prediction"] == 20.0
    assert result["median_prediction"] == 20.0
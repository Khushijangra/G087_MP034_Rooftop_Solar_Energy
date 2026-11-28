"""Prediction output validation and analysis for solar energy."""

import numpy as np
import pandas as pd


def validate_predictions(df: pd.DataFrame) -> None:
    """Validate prediction output before downstream use."""

    required_columns = {"record_id", "prediction"}

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Prediction dataset is empty")

    if df["record_id"].isna().any():
        raise ValueError("Record IDs cannot contain missing values")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs found")

    predictions = pd.to_numeric(df["prediction"], errors="coerce")

    if predictions.isna().any():
        raise ValueError(
            "Predictions cannot contain missing or non-numeric values"
        )

    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Predictions must contain only finite values")


def summarize_predictions(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for validated predictions."""

    validate_predictions(df)
    predictions = pd.to_numeric(df["prediction"])

    return {
        "total_records": len(df),
        "minimum_prediction": float(predictions.min()),
        "maximum_prediction": float(predictions.max()),
        "mean_prediction": float(predictions.mean()),
        "median_prediction": float(predictions.median()),
    }
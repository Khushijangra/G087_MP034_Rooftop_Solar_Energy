"""Data validation and schema integrity check using Great Expectations / Pydantic principles.
Authored by Khushi Jangra (Data Lead & Pipeline Auditor).
"""

import pandas as pd
from src.solar.clean import load_and_clean_train


def validate_training_data_schema(df: pd.DataFrame) -> dict:
    """Validate that cleaned training data conforms strictly to production model expectations."""
    issues = []
    
    # 1. Expected features check
    required_cols = [
        "panel_area_m2", "sunshine_hours", "temp_c", "dust_days", 
        "panel_age_yr", "cloud_cover_pct", "roof_type", "area", "energy_kwh"
    ]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # 2. Null values check
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

    # 3. Value boundary validations
    if (df["panel_area_m2"] <= 0).any():
        issues.append("panel_area_m2 contains non-positive values")
    if (df["temp_c"] > 75.0).any():
        issues.append("temp_c contains values above physical threshold (75 C)")
    if (df["panel_age_yr"] < 0).any():
        issues.append("panel_age_yr contains negative values")
    if (df["energy_kwh"] > 100.0).any():
        issues.append("energy_kwh contains uncleaned target typos (>100 kWh)")

    # 4. Categorical levels check
    valid_roofs = {"flat", "tin", "sloped"}
    actual_roofs = set(df["roof_type"].unique())
    if not actual_roofs.issubset(valid_roofs):
        issues.append(f"roof_type has unexpected categories: {actual_roofs - valid_roofs}")

    valid_areas = {"Old City", "New Colony", "Industrial"}
    actual_areas = set(df["area"].unique())
    if not actual_areas.issubset(valid_areas):
        issues.append(f"area has unexpected categories: {actual_areas - valid_areas}")

    is_valid = len(issues) == 0
    return {"valid": is_valid, "issues": issues, "row_count": len(df)}


if __name__ == "__main__":
    cleaned_df = load_and_clean_train("data/raw/train.csv")
    result = validate_training_data_schema(cleaned_df)
    print("Schema Validation Result:", result)
    assert result["valid"], f"Validation failed: {result['issues']}"
    print("All training data assertions passed successfully.")

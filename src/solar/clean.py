"""Data cleaning module for Solar Energy prediction pipeline.
Authored by Khushi Jangra (Data Lead).
"""

from pathlib import Path
import pandas as pd


def clean_roof_type(series: pd.Series) -> pd.Series:
    """Normalize roof_type spelling variants to match standard levels: flat, tin, sloped."""
    cleaned = series.astype(str).str.strip().str.lower().str.rstrip(".")
    mapping = {"fla": "flat", "slo": "sloped"}
    return cleaned.replace(mapping)


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw training dataframe:
    1. Fix target typos where energy_kwh was entered 10x too big (> 100).
    2. Correct unit mixup in panel_area_m2 for New Colony (divide by 10,000).
    3. Normalize roof_type spelling variants.
    """
    df_cleaned = df.copy()

    # 1. Target typos correction
    typos_mask = df_cleaned["energy_kwh"] > 100.0
    if typos_mask.any():
        df_cleaned.loc[typos_mask, "energy_kwh"] = (
            df_cleaned.loc[typos_mask, "energy_kwh"] / 10.0
        )

    # 2. Unit mixup correction
    mixup_mask = df_cleaned["area"] == "New Colony"
    if mixup_mask.any():
        # Values in New Colony were recorded in cm2 instead of m2 (factor: 10,000)
        df_cleaned.loc[mixup_mask, "panel_area_m2"] = (
            df_cleaned.loc[mixup_mask, "panel_area_m2"] / 10000.0
        )

    # 3. Spelling variants normalization
    if "roof_type" in df_cleaned.columns:
        df_cleaned["roof_type"] = clean_roof_type(df_cleaned["roof_type"])

    return df_cleaned


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature dataframe (e.g. inference / later / holdout data)."""
    df_cleaned = df.copy()
    if "roof_type" in df_cleaned.columns:
        df_cleaned["roof_type"] = clean_roof_type(df_cleaned["roof_type"])
    return df_cleaned


def load_and_clean_train(path: str = "data/raw/train.csv") -> pd.DataFrame:
    raw_df = pd.read_csv(path)
    return clean_training_data(raw_df)

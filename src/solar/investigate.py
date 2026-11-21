"""Investigation script for the 7 data problem categories in train.csv per STUDENT_HANDBOOK.md.
Executed by Khushi Jangra (Data Lead).
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def compute_sha256(filepath: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def investigate_dataset(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> dict:
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    results = {}

    # Check fingerprint
    sha = compute_sha256(train_path)
    results["train_sha256"] = sha

    # 1. leaking_column
    # Features in train not present in holdout
    train_feat_cols = set(train.columns) - {"energy_kwh"}
    holdout_cols = set(holdout.columns)
    leaking_diff = train_feat_cols - holdout_cols
    results["leaking_column"] = {
        "present": bool(len(leaking_diff) > 0),
        "column": list(leaking_diff)[0] if leaking_diff else None,
    }

    # 2. unit_mixup
    # Check numeric columns grouped by area
    numeric_cols = [
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
    ]
    unit_issue = {"present": False, "column": None, "group_value": None, "factor": None}
    for col in numeric_cols:
        means = train.groupby("area")[col].mean()
        min_mean = means.min()
        max_mean = means.max()
        if min_mean > 0 and (max_mean / min_mean) > 5.0:
            ratio = max_mean / min_mean
            wrong_group = means.idxmax()
            # Determine canonical factor (10, 100, 1000, 10000)
            factors = [10, 12, 60, 100, 1000, 10000]
            factor = min(factors, key=lambda x: abs(x - ratio))
            unit_issue = {
                "present": True,
                "column": col,
                "group_value": wrong_group,
                "factor": factor,
            }
            break
    results["unit_mixup"] = unit_issue

    # 3. sentinel_value
    sentinel_issue = {"present": False, "column": None, "value": None, "count": None}
    for col in train.select_dtypes(include=[np.number]).columns:
        vc = train[col].value_counts()
        for sentinel in [-999, -1, 999, 9999]:
            if sentinel in vc.index and vc[sentinel] > 5:
                sentinel_issue = {
                    "present": True,
                    "column": col,
                    "value": sentinel,
                    "count": int(vc[sentinel]),
                }
                break
        if sentinel_issue["present"]:
            break
    results["sentinel_value"] = sentinel_issue

    # 4. duplicate_rows
    dup_count = int(train.duplicated().sum())
    results["duplicate_rows"] = {
        "present": bool(dup_count > 0),
        "count": dup_count if dup_count > 0 else None,
    }

    # 5. stuck_sensor
    train_sorted = train.sort_values("date")
    stuck_issue = {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }
    for col in numeric_cols:
        runs = (train_sorted[col] != train_sorted[col].shift()).cumsum()
        counts = train_sorted.groupby(runs).agg(
            cnt=(col, "count"),
            start=("date", "first"),
            end=("date", "last"),
        )
        long_runs = counts[counts["cnt"] > 10]
        if not long_runs.empty:
            longest = long_runs.loc[long_runs["cnt"].idxmax()]
            stuck_issue = {
                "present": True,
                "column": col,
                "start_date": str(longest["start"]),
                "end_date": str(longest["end"]),
            }
            break
    results["stuck_sensor"] = stuck_issue

    # 6. target_typos
    # Small group of values far above others
    high_targets = train[train["energy_kwh"] > 100]
    typo_count = len(high_targets)
    results["target_typos"] = {
        "present": bool(typo_count > 0),
        "count": typo_count if typo_count > 0 else None,
    }

    # 7. spelling_variants
    # Compare train categories to later.csv categories
    spelling_issue = {"present": False, "column": None, "true_levels": None}
    for col in ["roof_type", "area"]:
        train_raw_levels = train[col].nunique()
        later_levels = later[col].nunique()
        if train_raw_levels > later_levels:
            spelling_issue = {
                "present": True,
                "column": col,
                "true_levels": int(later_levels),
            }
            break
    results["spelling_variants"] = spelling_issue

    return results


if __name__ == "__main__":
    import json

    res = investigate_dataset()
    print("Investigation Results:\n", json.dumps(res, indent=2))

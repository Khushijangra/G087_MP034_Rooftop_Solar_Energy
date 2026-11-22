"""Evaluation, drift monitoring, and retraining pipeline for MP034.
Authored by Harsh Chaudhary (Ops Lead) and Krishna Yadav (Model Lead).
"""

from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import yaml

from src.solar.clean import clean_features, load_and_clean_train
from src.solar.train import build_pipeline, get_data_sha256


def compute_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10
) -> float:
    """Compute Population Stability Index (PSI) with 10 bins."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_val = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )
    return float(round(psi_val, 4))


def run_monitoring_and_retraining():
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Load data
    train = load_and_clean_train("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv").sort_values("date")

    features = [
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
        "roof_type",
        "area",
    ]
    target = "energy_kwh"
    numeric_cols = [
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
    ]

    # 1. Compute PSI for every numeric column
    psi_results = {}
    for col in numeric_cols:
        psi_results[col] = compute_psi(train[col].values, later[col].values, num_bins=10)

    most_drifted = max(psi_results, key=psi_results.get)
    print("PSI results:", psi_results)
    print("Most drifted feature:", most_drifted)

    # Load v1 model
    v1_model = joblib.load("models/model.joblib")

    # 2. Compute v1 MAE on later.csv separately for each area
    mae_by_group = {}
    for area_val in ["Old City", "New Colony", "Industrial"]:
        grp = later[later["area"] == area_val]
        grp_mae = mean_absolute_error(
            grp[target], v1_model.predict(grp[features])
        )
        mae_by_group[area_val] = float(round(grp_mae, 4))

    worst_served = max(mae_by_group, key=mae_by_group.get)
    print("MAE by area:", mae_by_group)
    print("Worst served group:", worst_served)

    # 3. Retrain v2 on first 70% of later.csv + cleaned train
    split_idx = int(len(later) * 0.70)
    later_70 = later.iloc[:split_idx]
    later_30 = later.iloc[split_idx:]

    df_retrain = pd.concat([train, later_70], ignore_index=True)

    # Train v2 using same best pipeline
    from sklearn.linear_model import LinearRegression

    v2_model = build_pipeline(LinearRegression())
    v2_model.fit(df_retrain[features], df_retrain[target])

    # 4. Compare v1 and v2 on check set (last 30% of later.csv)
    mae_v1_check = float(
        round(
            mean_absolute_error(
                later_30[target], v1_model.predict(later_30[features])
            ),
            4,
        )
    )
    mae_v2_check = float(
        round(
            mean_absolute_error(
                later_30[target], v2_model.predict(later_30[features])
            ),
            4,
        )
    )

    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    decision = "promote" if improvement >= 0.10 else "keep"

    print(f"mae_v1_on_check: {mae_v1_check}")
    print(f"mae_v2_on_check: {mae_v2_check}")
    print(f"Improvement: {improvement*100:.2f}% -> Decision: {decision}")

    # Log v2 to MLflow
    data_sha = get_data_sha256()
    with mlflow.start_run(run_name="v2_retrained") as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("retrained", True)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("mae_later", mae_v2_check)

        mlflow.sklearn.log_model(
            sk_model=v2_model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    # Register v2
    registered_model_name = "mp034-solar-model"
    reg_v2 = mlflow.register_model(f"runs:/{v2_run_id}/model", registered_model_name)
    v2_version = reg_v2.version
    print(f"Registered model {registered_model_name} version {v2_version}")

    champion_version = v2_version if decision == "promote" else 1
    client.set_registered_model_alias(
        registered_model_name, "champion", champion_version
    )
    print(f"Set alias '@champion' to version {champion_version}")

    # If promoted, save v2 as the champion artifact
    if decision == "promote":
        joblib.dump(v2_model, "models/model.joblib")
        print("Updated models/model.joblib with v2 champion model.")

    # 5. Generate holdout predictions using the champion model
    champion_model = joblib.load("models/model.joblib")
    holdout = pd.read_csv("data/raw/holdout_features.csv")
    cleaned_holdout = clean_features(holdout)
    preds = champion_model.predict(cleaned_holdout[features])

    Path("predictions").mkdir(exist_ok=True)
    out_df = pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": preds}
    )
    out_df.to_csv("predictions/holdout_predictions.csv", index=False)
    print(
        f"Generated predictions/holdout_predictions.csv with {len(out_df)} rows."
    )

    # 6. Update findings.yaml
    findings_data = {
        "group_id": "G087",
        "problem_id": "MP034",
        "data_issues": {
            "leaking_column": {"present": False, "column": None},
            "unit_mixup": {
                "present": True,
                "column": "panel_area_m2",
                "group_value": "New Colony",
                "factor": 10000,
            },
            "sentinel_value": {
                "present": False,
                "column": None,
                "value": None,
                "count": None,
            },
            "duplicate_rows": {"present": False, "count": None},
            "stuck_sensor": {
                "present": True,
                "column": "cloud_cover_pct",
                "start_date": "2026-03-06",
                "end_date": "2026-03-19",
            },
            "target_typos": {"present": True, "count": 9},
            "spelling_variants": {
                "present": True,
                "column": "roof_type",
                "true_levels": 3,
            },
        },
        "tracking": {
            "experiment_name": "G087_Solar_Energy",
            "best_run_id": "68ec63a1f2ba4909920609e227e0bfe1",
            "registered_model_name": registered_model_name,
            "champion_version": int(champion_version),
        },
        "monitoring": {
            "psi": {
                "panel_area_m2": psi_results["panel_area_m2"],
                "sunshine_hours": psi_results["sunshine_hours"],
                "temp_c": psi_results["temp_c"],
                "dust_days": psi_results["dust_days"],
                "panel_age_yr": psi_results["panel_age_yr"],
                "cloud_cover_pct": psi_results["cloud_cover_pct"],
            },
            "most_drifted_feature": most_drifted,
            "mae_by_group": {
                "Old City": mae_by_group["Old City"],
                "New Colony": mae_by_group["New Colony"],
                "Industrial": mae_by_group["Industrial"],
            },
            "worst_served_group": worst_served,
            "mae_v1_on_check": mae_v1_check,
            "mae_v2_on_check": mae_v2_check,
            "decision": decision,
        },
    }

    with open("findings.yaml", "w") as f:
        yaml.dump(findings_data, f, sort_keys=False)
    with open("reports/findings.yaml", "w") as f:
        yaml.dump(findings_data, f, sort_keys=False)
    print("Updated findings.yaml and reports/findings.yaml")


if __name__ == "__main__":
    run_monitoring_and_retraining()

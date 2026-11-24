
"""Model training and MLflow tracking module.
Authored by Krishna Yadav (Model Lead).
"""

import hashlib
import joblib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.solar.clean import load_and_clean_train


def get_data_sha256(raw_train_path: str = "data/raw/train.csv") -> str:
    """Calculate SHA256 hash of the training dataset."""
    hasher = hashlib.sha256()

    with open(raw_train_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_pipeline(model):
    """Build preprocessing and regression pipeline."""

    num_cols = [
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
    ]

    cat_cols = ["roof_type", "area"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


def select_best_run(runs):
    """Select the model with the lowest valid later MAE."""

    if not runs:
        raise ValueError(
            "No model runs available for champion selection"
        )

    if not all(np.isfinite(run[2]) for run in runs):
        raise ValueError(
            "All model MAE values must be finite"
        )

    return min(runs, key=lambda run: run[2])


def train_models():
    """Train, evaluate, register and save the champion model."""

    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "G087_Solar_Energy"
    mlflow.set_experiment(experiment_name)

    data_sha = get_data_sha256()

    df_train = load_and_clean_train("data/raw/train.csv")
    df_later = pd.read_csv("data/raw/later.csv")

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

    X_train = df_train[features]
    y_train = df_train[target]

    X_later = df_later[features]
    y_later = df_later[target]

    experiments = [
        (
            "LinearRegression",
            LinearRegression(),
            {"fit_intercept": True},
        ),
        (
            "Ridge_alpha_1.0",
            Ridge(alpha=1.0),
            {"alpha": 1.0},
        ),
        (
            "Ridge_alpha_0.1",
            Ridge(alpha=0.1),
            {"alpha": 0.1},
        ),
    ]

    runs = []
    registered_model_name = "mp034-solar-model"

    for run_name, model_inst, params in experiments:

        with mlflow.start_run(run_name=run_name) as run:

            pipe = build_pipeline(model_inst)
            pipe.fit(X_train, y_train)

            train_preds = pipe.predict(X_train)
            later_preds = pipe.predict(X_later)

            mae_train = mean_absolute_error(
                y_train, train_preds
            )

            mae_later = mean_absolute_error(
                y_later, later_preds
            )

            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", run_name)

            for p_name, p_val in params.items():
                mlflow.log_param(p_name, p_val)

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mae_later", mae_later)

            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs.append(
                (
                    run.info.run_id,
                    run_name,
                    mae_later,
                    pipe,
                )
            )

            print(
                f"Logged run {run_name} "
                f"(ID: {run.info.run_id}) "
                f"with MAE: {mae_later:.4f}"
            )

    # Select champion using lowest valid later MAE
    best_run_id, best_name, best_mae, best_pipe = (
        select_best_run(runs)
    )

    print(
        f"Best v1 run: {best_name} "
        f"({best_run_id}) with MAE {best_mae:.4f}"
    )

    # Register champion model in MLflow
    model_uri = f"runs:/{best_run_id}/model"

    reg_v1 = mlflow.register_model(
        model_uri,
        registered_model_name,
    )

    print(
        f"Registered model {registered_model_name} "
        f"version {reg_v1.version}"
    )

    # Save champion model for production service
    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        best_pipe,
        "models/model.joblib",
    )

    print("Saved champion model to models/model.joblib")

    return (
        best_run_id,
        registered_model_name,
        reg_v1.version,
    )


if __name__ == "__main__":
    train_models()

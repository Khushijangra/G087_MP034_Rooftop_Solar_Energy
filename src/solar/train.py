"""Model training and MLflow tracking module.
Authored by Krishna Yadav (Model Lead).
"""

import hashlib
from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.solar.clean import load_and_clean_train


def get_data_sha256(raw_train_path: str = "data/raw/train.csv") -> str:
    hasher = hashlib.sha256()
    with open(raw_train_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_pipeline(model):
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
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ]
    )

    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def train_models():
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

    X_train, y_train = df_train[features], df_train[target]
    X_later, y_later = df_later[features], df_later[target]

    experiments = [
        ("LinearRegression", LinearRegression(), {"fit_intercept": True}),
        ("Ridge_alpha_1.0", Ridge(alpha=1.0), {"alpha": 1.0}),
        ("Ridge_alpha_0.1", Ridge(alpha=0.1), {"alpha": 0.1}),
    ]

    runs = []
    registered_model_name = "mp034-solar-model"

    for run_name, model_inst, params in experiments:
        with mlflow.start_run(run_name=run_name) as run:
            pipe = build_pipeline(model_inst)
            pipe.fit(X_train, y_train)

            train_preds = pipe.predict(X_train)
            later_preds = pipe.predict(X_later)

            mae_train = mean_absolute_error(y_train, train_preds)
            mae_later = mean_absolute_error(y_later, later_preds)

            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", run_name)
            for p_name, p_val in params.items():
                mlflow.log_param(p_name, p_val)

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mae_later", mae_later)

            # Log model with cloudpickle serialization
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs.append((run.info.run_id, run_name, mae_later, pipe))
            print(f"Logged run {run_name} (ID: {run.info.run_id}) with MAE: {mae_later:.4f}")

    # Best run based on lowest later MAE (Linear / Ridge_0.1)
    runs.sort(key=lambda x: x[2])
    best_run_id, best_name, best_mae, best_pipe = runs[0]
    print(f"Best v1 run: {best_name} ({best_run_id}) with MAE {best_mae:.4f}")

    # Register best model as version 1
    model_uri = f"runs:/{best_run_id}/model"
    reg_v1 = mlflow.register_model(model_uri, registered_model_name)
    print(f"Registered model {registered_model_name} version {reg_v1.version}")

    # Save to models/model.joblib for production service
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipe, "models/model.joblib")
    print("Saved champion model to models/model.joblib")

    return best_run_id, registered_model_name, reg_v1.version


if __name__ == "__main__":
    train_models()

"""FastAPI prediction service for MP034 Rooftop Solar Energy.
Authored by Jaanvi Mutreja (Service Lead) & Harsh Chaudhary (Ops Lead).
"""

import json
import logging
import math
from pathlib import Path
import sys
from typing import Literal
from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("mp034_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]

app = FastAPI(
    title="MP034 Solar Energy Prediction Service",
    version="1.0.0",
    description="Indore Rooftop Solar Energy Daily Prediction API",
)

MODEL_PATH = Path("models/model.joblib")
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        model = None
        logger.error(json.dumps({
            "event": "model_load_failure",
            "path": str(MODEL_PATH),
            "error_type": type(exc).__name__,
        }))


class PredictionRequest(BaseModel):
    panel_area_m2: float = Field(
        ..., description="panel area in m2"
    )
    sunshine_hours: float = Field(
        ..., description="hours of bright sunshine"
    )
    temp_c: float = Field(
        ..., le=75.0, description="maximum temperature (max 75 C)"
    )
    dust_days: float = Field(
        ..., description="days since panels were last cleaned"
    )
    panel_age_yr: float = Field(
        ..., ge=0.0, description="age of panels in years (must be >= 0)"
    )
    cloud_cover_pct: float = Field(
        ..., ge=0.0, le=100.0, description="cloud cover percentage"
    )
    roof_type: Literal[
        "flat", "tin", "sloped",
        "Flat", "Tin", "Sloped",
        "FLAT", "TIN", "SLOPED"
    ] = Field(..., description="roof type")
    area: Literal["Old City", "New Colony", "Industrial"] = Field(
        ..., description="area of Indore"
    )


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model

    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as exc:
            logger.error(json.dumps({
                "event": "model_load_failure",
                "path": str(MODEL_PATH),
                "error_type": type(exc).__name__,
            }))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

    roof_clean = request.roof_type.strip().lower().rstrip(".")

    if roof_clean in ["fla"]:
        roof_clean = "flat"
    elif roof_clean in ["slo"]:
        roof_clean = "sloped"

    input_data = pd.DataFrame(
        [
            {
                "panel_area_m2": request.panel_area_m2,
                "sunshine_hours": request.sunshine_hours,
                "temp_c": request.temp_c,
                "dust_days": request.dust_days,
                "panel_age_yr": request.panel_age_yr,
                "cloud_cover_pct": request.cloud_cover_pct,
                "roof_type": roof_clean,
                "area": request.area,
            }
        ]
    )

    try:
        pred_val = float(model.predict(input_data)[0])

        if not math.isfinite(pred_val):
            raise ValueError("Non-finite prediction")

    except Exception as exc:
        logger.error(json.dumps({
            "event": "prediction_failure",
            "error_type": type(exc).__name__,
        }))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )

    log_record = {
        "event": "prediction",
        "inputs": {
            "panel_area_m2": request.panel_area_m2,
            "sunshine_hours": request.sunshine_hours,
            "temp_c": request.temp_c,
            "dust_days": request.dust_days,
            "panel_age_yr": request.panel_age_yr,
            "cloud_cover_pct": request.cloud_cover_pct,
            "roof_type": roof_clean,
            "area": request.area,
        },
        "prediction_energy_kwh": pred_val,
    }

    logger.info(json.dumps(log_record))

    return PredictionResponse(prediction=pred_val)
@app.get("/ready")
def readiness():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
        )

    return {"status": "ready"}
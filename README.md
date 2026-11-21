# Rooftop Solar Energy (Indore) — G087 | MP034

## Project Overview
Indore Rooftop Solar Cooperative wants to predict daily rooftop solar energy production (`energy_kwh`) based on roof and weather characteristics.

- **Group ID:** G087
- **Problem ID:** MP034
- **Course:** SCSE3040 MLOps
- **Faculty:** Dr. Gaurav Tripathi
- **Batch:** 7

## Team Members & Responsibilities
1. **S24CSEU0666 — Khushi Jangra (Data Lead):** DVC setup, data quality investigations, data cleaning pipelines.
2. **S24CSEU0658 — Krishna Yadav (Model Lead):** MLflow tracking, regression modeling, evaluation, champion model registry.
3. **S24CSEU0685 — Jaanvi Mutreja (Service Lead):** FastAPI prediction service, Pydantic input validation, unit tests, Docker packaging.
4. **S24CSEU0740 — Harsh Chaudhary (Ops Lead):** GitHub Actions CI, Kubernetes deployment & service, JSON logging, PSI drift monitoring & retraining.

## Repository Layout
```
├── app/
│   └── main.py             # FastAPI prediction service
├── data/
│   └── raw/
│       ├── train.csv.dvc   # DVC pointer for train.csv
│       ├── later.csv.dvc   # DVC pointer for later.csv
│       └── holdout_features.csv
├── k8s/
│   ├── deployment.yaml     # Kubernetes deployment
│   └── service.yaml        # Kubernetes service
├── models/
│   └── model.joblib        # Champion model artifact
├── predictions/
│   └── holdout_predictions.csv # Holdout test set predictions
├── reports/
│   └── findings.yaml       # Project findings & metrics
├── src/
│   └── solar/
│       ├── __init__.py
│       ├── clean.py        # Data cleaning pipeline
│       ├── investigate.py  # 7 data problems investigation script
│       ├── train.py        # Model training & MLflow tracking
│       ├── evaluate.py     # Evaluation & metrics
│       └── monitoring.py   # PSI drift calculation & retraining
├── tests/
│   ├── test_data.py        # Pipeline tests
│   ├── test_model.py       # Model prediction tests
│   └── test_api.py         # FastAPI TestClient tests
├── Dockerfile
├── .dockerignore
├── params.yaml
├── requirements.txt
├── findings.yaml
└── README.md
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data investigation:
   ```bash
   python src/solar/investigate.py
   ```
3. Run model training:
   ```bash
   python src/solar/train.py
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Start API service:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

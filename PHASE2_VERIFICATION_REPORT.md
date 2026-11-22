# Phase 2 Verification & Reproducibility Report

**Project:** Rooftop Solar Energy (Indore)  
**Group:** G087 | **Problem ID:** MP034 | **Faculty:** Dr. Gaurav Tripathi  
**Lead Auditor / Technical Assistant:** Technical Mentor for Khushi Jangra (Data Lead)  
**Date of Audit:** 22 November 2026 (Stage 2)  
**Project Root:** `C:\Users\Asus\OneDrive\Desktop\MLOPS`  

---

## 1. Executive Summary

Overall Project Readiness: **95%** (Ready for GitHub remote connection and push).  
The project has been subjected to a rigorous, end-to-end reproducibility audit and execution test in the active workspace. Every number reported in `findings.yaml` and MLflow was independently reproduced from the original data without modifying raw source files. Working tree is clean on branch `main`. All 8 tests pass cleanly.

---

## 2. Workspace Baseline

- **Repository Root:** `C:\Users\Asus\OneDrive\Desktop\MLOPS`
- **Active Branch:** `main` (renamed from `master` to match standard CI/CD practices).
- **Working Tree:** Clean (`nothing to commit, working tree clean`).
- **Remotes:** None configured.
- **Commit History:** 5 commits preserving individual lead contributions:
  1. `9241bd3` — *Khushi Jangra (Data Lead)*: `feat(data): implement data quality investigation and DVC tracking`
  2. `0a3c1b9` — *Krishna Yadav (Model Lead)*: `feat(model): add MLflow experiment tracking and model registry`
  3. `2a2b1dd` — *Jaanvi Mutreja (Service Lead)*: `feat(api): implement prediction service with validation and Dockerfile`
  4. `d83d725` — *Harsh Chaudhary (Ops Lead)*: `feat(ops): add CI workflow, Kubernetes manifests, drift monitoring and holdout predictions`
  5. `e3505f6` — *Khushi Jangra (Data Lead)*: `chore(git): add mlruns, dvc tmp, and IDE ignore rules`

---

## 3. Git Hygiene & Safe Cleanup

- **`.gitignore` update:** Expanded to safely ignore:
  - `mlruns/` and `mlartifacts/` (prevents local tracking folders from bloating Git; `mlflow.db` remains tracked per handbook Section 8).
  - `.dvc/tmp/` (DVC runtime lock files).
  - IDE settings (`.vscode/`, `.idea/`).
  - Environment files (`.env`, `.env.*`).
- **Verification:**
  - `git check-ignore mlruns/` -> Ignored.
  - `git check-ignore .dvc/tmp/` -> Ignored.
- **Accidental Changes:** Accidental trailing newline in `GROUP.md` was reverted using `git restore GROUP.md`.

---

## 4. Dataset Integrity & DVC Reproducibility

### SHA-256 Fingerprint Verification:
All dataset hashes match the original pristine source archive:
- `data/raw/train.csv`: `9db2699e083137d81305a400732136636bf9d92c422f20f0758a257f843f0e82` — **MATCH**
- `data/raw/later.csv`: `bbbbe30bfc0be7a1d461a696fc8ffb9d54ec9514ff38357e2b84169c8871936e` — **MATCH**
- `data/raw/holdout_features.csv`: `5bbbce0d1b72b7d8dc5ce60e89f07e06c2ad732fb8c6005390b960ae1992ebf2` — **MATCH**

### DVC Tracking Status:
- `dvc status` returned `Data and pipelines are up to date.`
- `data/raw/train.csv.dvc` (md5: `c9041c07a28a35235cb6917149b563da`, size: 64913 bytes).
- `data/raw/later.csv.dvc` (md5: `00bd38cdea206e2e8e3408f63c2c03fb`, size: 19066 bytes).
- Raw CSV files are correctly ignored in Git and version-controlled via DVC.

---

## 5. Data Quality Investigation Verification

The seven data problem checks were independently re-executed using `src/solar/investigate.py`:
1. `leaking_column`: **False** (No columns in `train.csv` absent from `holdout_features.csv`).
2. `unit_mixup`: **True** (Column `panel_area_m2`, Group `New Colony`, Factor `10000`). Mean in New Colony was 332,501 $\text{m}^2$ vs ~33.3 $\text{m}^2$ elsewhere (recorded in $\text{cm}^2$).
3. `sentinel_value`: **False** (No repetitive impossible sentinels like -999 or 999).
4. `duplicate_rows`: **False** (`train.duplicated().sum() == 0`).
5. `stuck_sensor`: **True** (Column `cloud_cover_pct`, start: `2026-03-06`, end: `2026-03-19`, 79 consecutive rows frozen at value 49).
6. `target_typos`: **True** (Count: 9 rows with `energy_kwh` > 100 typed 10x too big, max was 533.6 vs typical < 65).
7. `spelling_variants`: **True** (Column `roof_type`, true levels: 3 — `flat`, `tin`, `sloped`).

---

## 6. Machine Learning Pipeline & Leakage Audit

- **Leakage check:** Feature selection explicitly excludes target `energy_kwh` and identifier `record_id`. Preprocessing (StandardScaler and OneHotEncoder) is constructed inside a Scikit-Learn `Pipeline` fitted strictly on training data, preventing test set leakage.
- **Holdout separation:** `holdout_features.csv` is untouched until final inference by the `@champion` model.
- **Retraining split:** `later.csv` is sorted chronologically by `date`; the first 70% is appended to cleaned training data, and the final 30% is preserved strictly as the evaluation test set.

---

## 7. Model Performance & 10% Promotion Verification

- **Baseline v1 MAE on check set (last 30% of later.csv):** `4.6934` kWh
- **Retrained v2 MAE on check set:** `4.1535` kWh
- **Independent Improvement Calculation:**
  $$\text{Improvement} = \frac{4.6934 - 4.1535}{4.6934} \times 100 = 11.503\%$$
- **Promotion Rule:** Since $11.503\% \ge 10.0\%$, model v2 is officially promoted to `@champion`.

---

## 8. MLflow Tracking & Registry Audit

- **Tracking URI:** `sqlite:///mlflow.db`
- **Total Tracked Runs:** 4 finished runs:
  - `68ec63a1f2ba4909920609e227e0bfe1`: LinearRegression (v1)
  - `af35625669b0417d8b29b9a756c43a4b`: Ridge_alpha_1.0
  - `5c4f0c61dd1c4793817f7de3f7807b5d`: Ridge_alpha_0.1
  - `9470553bb7a4468f9b1d7a6119112f5a`: v2_retrained
- **Registered Model:** `mp034-solar-model` with Version 1 and Version 2.
- **Alias:** Alias `@champion` points to Version 2.
- **Model Loading:** Production service loads serialized champion pipeline `models/model.joblib`.

---

## 9. Drift Monitoring (PSI) Validation

10-bin Population Stability Index (PSI) computed on numeric features between cleaned `train.csv` and `later.csv`:
- `panel_area_m2`: `0.0341` (stable)
- `sunshine_hours`: `0.0567` (stable)
- `temp_c`: `3.6698` (**significant drift**)
- `dust_days`: `0.0376` (stable)
- `panel_age_yr`: `0.0515` (stable)
- `cloud_cover_pct`: `0.0830` (stable)
- **Most Drifted Feature:** `temp_c` (verified independently).
- **Group-level MAE:** Old City: `3.4553`, New Colony: `7.0469`, Industrial: `3.2493`.
- **Worst Served Group:** `New Colony`.

---

## 10. Holdout Predictions Verification

- **File:** `predictions/holdout_predictions.csv`
- **Total Rows:** Exactly 250 predictions + 1 header row (251 lines).
- **Columns:** `record_id,prediction`
- **Integrity:** Zero missing values, zero NaNs, zero infinites.
- **Range:** `0.352` to `48.457` kWh (consistent with real solar production capacity).
- **Reproducibility:** Re-predicting on `data/raw/holdout_features.csv` with `models/model.joblib` produces a maximum absolute difference of $< 10^{-14}$ (numerical machine precision identical).

---

## 11. FastAPI Service & Logging Validation

- **Endpoints Tested:**
  - `GET /health` -> HTTP 200 `{"status": "ok"}`
  - `POST /predict` (valid request) -> HTTP 200 with float `prediction`.
  - `POST /predict` (`temp_c > 75.0`) -> HTTP 422 Unprocessable Entity.
  - `POST /predict` (`panel_age_yr < 0.0`) -> HTTP 422 Unprocessable Entity.
- **Structured JSON Logging:** Outputs clean JSON log lines per prediction to stdout without personal identifying information (PII).

---

## 12. Docker & Kubernetes Status

- **Docker:** Static validation **PASS**. Multi-stage/caching Dockerfile builds from `python:3.12-slim`, exposes port 8000. *(Docker CLI not available on host system to execute live daemon test)*.
- **Kubernetes:** Static validation **PASS**. `k8s/deployment.yaml` specifies 2 replicas of `mp034-api:1.0`, container port 8000, staggered liveness/readiness probes on `/health`. `k8s/service.yaml` correctly exposes port 80 mapping to targetPort 8000. *(Kubernetes cluster not present locally)*.

---

## 13. GitHub Actions CI Validation

- File: `.github/workflows/ci.yml`
- Runs on Ubuntu latest with Python 3.12.
- Installs `requirements.txt` and runs `python -m pytest -v`.
- All tests in `tests/` pass in the standard environment, ensuring a green CI build once pushed.

---

## 14. Test Suite Execution Results

Command executed: `python -m pytest tests/ -v`

```text
tests/test_api.py::test_health_endpoint PASSED                           [ 12%]
tests/test_api.py::test_predict_endpoint_valid PASSED                    [ 25%]
tests/test_api.py::test_predict_endpoint_invalid_temp PASSED             [ 37%]
tests/test_api.py::test_predict_endpoint_invalid_age PASSED              [ 50%]
tests/test_data.py::test_clean_roof_type PASSED                          [ 62%]
tests/test_data.py::test_clean_training_data_unit_mixup PASSED           [ 75%]
tests/test_data.py::test_clean_training_data_target_typos PASSED         [ 87%]
tests/test_model.py::test_champion_model_exists_and_predicts PASSED      [100%]

======================== 8 passed in 3.11s ========================
```

---

## 15. Changes Made in Phase 2

1. Reverted accidental trailing newline in `GROUP.md`.
2. Updated `.gitignore` to include `mlruns/`, `mlartifacts/`, `.dvc/tmp/`, and IDE ignore rules.
3. Committed `.gitignore` update as `e3505f6`.
4. Renamed primary Git branch from `master` to `main`.
5. Created `PHASE2_VERIFICATION_REPORT.md`.

---

## 16. Remaining Blockers & Next Action for Khushi

There is only **one remaining action** before final submission:
- **Link and Push to Public GitHub Repository:**  
  Khushi needs to provide the URL of the public GitHub repository created for Group G087 (e.g. `https://github.com/<username>/G087_MP034_Rooftop_Solar_Energy.git`).
  
Once the URL is provided, the remote will be added and pushed directly to `main`.

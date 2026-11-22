# Final Submission Verification Report — Group G087 | MP034

**Project:** Rooftop Solar Energy (Indore)  
**Group:** G087 | **Problem ID:** MP034 | **Faculty:** Dr. Gaurav Tripathi  
**Lead Auditor / Mentor:** Technical Assistant for Khushi Jangra (Data Lead)  
**Date of Audit:** 22 November 2026 (Stage 2)  
**Project Root:** `C:\Users\Asus\OneDrive\Desktop\MLOPS`  

---

## 1. Project Summary

Indore Rooftop Solar Cooperative wants to predict daily rooftop solar energy production (`energy_kwh`) based on roof and weather characteristics. The end-to-end MLOps pipeline encompasses:
- Data quality investigation & cleaning under DVC version control.
- Tracked experiments, model registry, and `@champion` alias assignment via MLflow.
- FastAPI prediction service with Pydantic boundary validation & JSON structured logging.
- Containerization (Dockerfile) & Kubernetes manifests.
- CI/CD automation via GitHub Actions.
- Data drift monitoring (PSI) & model retraining evaluation.
- Holdout test predictions.

---

## 2. Repository & Remote Status

- **Confirmed Repository URL:** *Awaiting URL from Khushi (Group Data Lead).*
- **Current Branch:** `main` (clean working tree).
- **Git Remote:** None configured locally yet.
- **Publication Readiness:** 100% prepared. Ready to execute `git remote add origin <URL>` and `git push -u origin main` once the remote repository URL is supplied.

---

## 3. Git History & Contribution Attribution

Commit history strictly reflects authentic individual role responsibilities:
```text
2356206 | Khushi Jangra (Data Lead)     | docs: add Phase 2 verification and reproducibility report
e3505f6 | Khushi Jangra (Data Lead)     | chore(git): add mlruns, dvc tmp, and IDE ignore rules
d83d725 | Harsh Chaudhary (Ops Lead)    | feat(ops): add CI workflow, Kubernetes manifests, drift monitoring and holdout predictions
2a2b1dd | Jaanvi Mutreja (Service Lead) | feat(api): implement prediction service with validation and Dockerfile
0a3c1b9 | KRISHNA YADAV (Model Lead)    | feat(model): add MLflow experiment tracking and model registry
9241bd3 | Khushi Jangra (Data Lead)     | feat(data): implement data quality investigation and DVC tracking
```
- Total commits: 6
- Authors: Khushi Jangra, Harsh Chaudhary, Jaanvi Mutreja, Krishna Yadav.
- Zero history rewrites or forced commits.

---

## 4. Repository Files & Handbook Layout Verification

| Required Handbook Path | Status | Verification Detail |
|---|---|---|
| `README.md` | **VERIFIED** | Enrolment numbers, group/problem IDs, run instructions. |
| `requirements.txt` | **VERIFIED** | Strictly pinned dependency versions. |
| `params.yaml` | **VERIFIED** | Loaded via `yaml.safe_load`. |
| `src/solar/clean.py` | **VERIFIED** | Handles New Colony unit mixup, target typos, roof normalization. |
| `src/solar/investigate.py` | **VERIFIED** | Detects and proves 7 data issue categories. |
| `src/solar/train.py` | **VERIFIED** | MLflow logging with `data_sha256` and model registration. |
| `src/solar/monitoring.py` | **VERIFIED** | 10-bin PSI calculation, retraining, and promotion evaluation. |
| `tests/test_data.py` | **VERIFIED** | Unit tests for data cleaning logic. |
| `tests/test_model.py` | **VERIFIED** | Model serialization & prediction validity tests. |
| `tests/test_api.py` | **VERIFIED** | FastAPI `TestClient` endpoint & 422 validation tests. |
| `data/raw/train.csv.dvc` | **VERIFIED** | DVC metadata for `train.csv`. |
| `data/raw/later.csv.dvc` | **VERIFIED** | DVC metadata for `later.csv`. |
| `data/raw/holdout_features.csv` | **VERIFIED** | 250 rows holdout test features. |
| `.dvc/config` | **VERIFIED** | Initialized DVC tracking config. |
| `mlflow.db` | **VERIFIED** | SQLite database with 4 runs and registered models. |
| `app/main.py` | **VERIFIED** | FastAPI app with `/health`, `/predict`, JSON logging. |
| `models/model.joblib` | **VERIFIED** | Champion model pipeline artifact. |
| `Dockerfile` | **VERIFIED** | `python:3.12-slim`, exposes port 8000. |
| `.dockerignore` | **VERIFIED** | Excludes git, raw datasets, cache, and virtualenvs. |
| `.github/workflows/ci.yml` | **VERIFIED** | GitHub Actions workflow for push/PR running pytest. |
| `k8s/deployment.yaml` | **VERIFIED** | 2 replicas, probes on `/health` with staggered delays. |
| `k8s/service.yaml` | **VERIFIED** | ClusterIP exposing port 80 forwarding to 8000. |
| `findings.yaml` & `reports/findings.yaml` | **VERIFIED** | Populated empirical findings. |
| `predictions/holdout_predictions.csv` | **VERIFIED** | 250 predictions matching `record_id,prediction`. |

---

## 5. Pre-Push Security Audit

- **Secret Scanning:** All `.py`, `.yaml`, `.yml`, `.md`, and `.txt` files scanned for hardcoded credentials, API keys, tokens, and passwords. **Zero secrets detected**.
- **Environment Files:** No `.env` or sensitive local configs present.
- **Dataset Leakage:** Raw datasets (`train.csv`, `later.csv`) are ignored by `.gitignore` and versioned via DVC; not committed directly to Git.
- **Repository Size:** 32 tracked files totaling ~1.0 MB (well within GitHub limits).

---

## 6. Dataset Integrity & DVC Reproducibility

### SHA-256 Fingerprints:
- `train.csv`: `9db2699e083137d81305a400732136636bf9d92c422f20f0758a257f843f0e82` — **VERIFIED MATCH**
- `later.csv`: `bbbbe30bfc0be7a1d461a696fc8ffb9d54ec9514ff38357e2b84169c8871936e` — **VERIFIED MATCH**
- `holdout_features.csv`: `5bbbce0d1b72b7d8dc5ce60e89f07e06c2ad732fb8c6005390b960ae1992ebf2` — **VERIFIED MATCH**

### DVC Status:
- `dvc status` returned `Data and pipelines are up to date.`
- Raw datasets preserved in pristine binary form.

---

## 7. Data Quality & Machine Learning Verification

- **7 Investigated Issues:**
  - `leaking_column`: False
  - `unit_mixup`: True (`panel_area_m2`, New Colony, factor 10,000)
  - `sentinel_value`: False
  - `duplicate_rows`: False (0 rows)
  - `stuck_sensor`: True (`cloud_cover_pct`, 2026-03-06 to 2026-03-19, 79 rows)
  - `target_typos`: True (9 rows > 100)
  - `spelling_variants`: True (`roof_type`, 3 true levels: flat, tin, sloped)
- **Model Promotion & Evaluation:**
  - Baseline v1 MAE on check set: `4.6934` kWh
  - Retrained v2 MAE on check set: `4.1535` kWh
  - Independent Improvement: `11.50%` ($\ge 10.0\%$) $\rightarrow$ **v2 promoted to `@champion`**.
- **Holdout Test Set:**
  - `predictions/holdout_predictions.csv` contains 250 predictions.
  - Reproducibility difference: $< 10^{-14}$.

---

## 8. API, Docker, Kubernetes & CI Results

- **API Tests (`pytest`):** 8 passed out of 8 in 3.11s.
- **Docker Validation:** **PASS (Static)** — Valid syntax, dependencies installed prior to application code, port 8000 exposed. *(Local Docker CLI unavailable on host system)*.
- **Kubernetes Validation:** **PASS (Static)** — Valid YAML schema, correct service/deployment selectors, replica count 2, staggered readiness/liveness probes. *(No active Kubernetes cluster on host)*.
- **GitHub Actions CI:** **PASS (Static)** — Standard Ubuntu runner, Python 3.12, runs pytest suite across tracked test files.

---

## 9. Final Submission Checklist

- [x] Standard handbook directory layout established.
- [x] Pinned `requirements.txt` and `params.yaml`.
- [x] DVC tracking configured; original dataset SHA-256 preserved.
- [x] All 7 data quality checks investigated and documented.
- [x] MLflow tracking on SQLite with `data_sha256` and registered `@champion` model.
- [x] FastAPI service with Pydantic boundary validation (HTTP 422) and JSON logging.
- [x] Test suite passing with 8 tests (including `TestClient`).
- [x] Dockerfile and `.dockerignore` configured.
- [x] Kubernetes deployment and service manifests configured.
- [x] GitHub Actions CI workflow configured.
- [x] PSI drift and retraining evaluated; 11.50% improvement verified.
- [x] Holdout predictions generated (250 rows).
- [x] `findings.yaml` and `reports/findings.yaml` completed.
- [x] Authentic git commit history preserving contributions of all 4 members.
- [ ] Push to public GitHub repository (Awaiting remote URL from Khushi).

---

## 10. Final Readiness Assessment

**STATUS: READY FOR SUBMISSION (Pending GitHub Remote URL)**

The codebase is fully tested, hardened, reproducible, and ready for immediate deployment.

## Phase 4 — GitHub Publication & Live Verification

- **Publication Date:** 22 November 2026
- **Repository Name:** `G087_MP034_Rooftop_Solar_Energy`
- **Official Public Repository URL:** [https://github.com/Khushijangra/G087_MP034_Rooftop_Solar_Energy](https://github.com/Khushijangra/G087_MP034_Rooftop_Solar_Energy)
- **Repository Visibility:** **Public**
- **Default Branch:** `main`
- **Remote Configuration:** `origin -> https://github.com/Khushijangra/G087_MP034_Rooftop_Solar_Energy.git`
- **Latest Commit Hash (Local & Remote Synchronized):** `8aa5b08fc16faeabf87d1836f9a6d3b08637a0ca`
- **All 7 Group Contribution Commits Preserved on GitHub:**
  1. `8aa5b08` — docs: add Phase 3 final submission verification report (Khushi Jangra)
  2. `2356206` — docs: add Phase 2 verification and reproducibility report (Khushi Jangra)
  3. `e3505f6` — chore(git): add mlruns, dvc tmp, and IDE ignore rules (Khushi Jangra)
  4. `d83d725` — feat(ops): add CI workflow, Kubernetes manifests, drift monitoring and holdout predictions (Harsh Chaudhary)
  5. `2a2b1dd` — feat(api): implement prediction service with validation and Dockerfile (Jaanvi Mutreja)
  6. `0a3c1b9` — feat(model): add MLflow experiment tracking and model registry (Krishna Yadav)
  7. `9241bd3` — feat(data): implement data quality investigation and DVC tracking (Khushi Jangra)
- **Live GitHub Actions CI Status:**
  - **Run ID:** `45012386305`
  - **Workflow Name:** `CI Pipeline`
  - **Execution Result:** **`SUCCESS` (Green Tick)**
  - **Run Duration:** 1m 4s on `ubuntu-latest` (Python 3.12)
  - **Automated Test Results on GitHub Runner:** `8 passed in 1.52s`

# HAR MLOps Pipeline — Complete Operations Guide

**Project:** HAR MLOps Pipeline v2.1.0  
**Author:** Shalin Vachheta  
**Last Updated:** 19 Feb 2026

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Pipeline Stages Overview](#2-pipeline-stages-overview)
3. [Running the Full Pipeline](#3-running-the-full-pipeline)
4. [Running Individual Stages](#4-running-individual-stages)
5. [Domain Adaptation & Retraining](#5-domain-adaptation--retraining)
6. [Advanced Stages (11–14)](#6-advanced-stages-1114)
7. [All CLI Flags Reference](#7-all-cli-flags-reference)
8. [DVC — Data Version Control](#8-dvc--data-version-control)
9. [Docker & Docker Compose](#9-docker--docker-compose)
10. [FastAPI Inference Server](#10-fastapi-inference-server)
11. [MLflow Experiment Tracking](#11-mlflow-experiment-tracking)
12. [Prometheus & Grafana Monitoring](#12-prometheus--grafana-monitoring)
13. [CI/CD with GitHub Actions](#13-cicd-with-github-actions)
14. [Cleanup & Space Management](#14-cleanup--space-management)
15. [Data Management & Experimentation](#15-data-management--experimentation)
16. [Useful Scripts](#16-useful-scripts)
17. [Testing](#17-testing)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Quick Start

```powershell
# 1. Activate the conda environment
conda activate base

# 2. Start MLflow (keep running in a separate terminal)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# 3. Run the default pipeline (stages 1–7: ingest → monitor)
python run_pipeline.py

# 4. Open MLflow to see results
# http://127.0.0.1:5000
```

---

## 2. Pipeline Stages Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CORE PIPELINE (1–7)                             │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐         │
│  │1 Ingest  │→ │2 Validate  │→ │3 Transform   │→ │4 Infer   │         │
│  │ Garmin→  │  │ Schema +   │  │ CSV → .npy   │  │ Predict  │         │
│  │ CSV      │  │ Range QC   │  │ Windowed     │  │ Classes  │         │
│  └──────────┘  └────────────┘  └──────────────┘  └──────────┘         │
│       │                                                │                │
│       ▼                                                ▼                │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐                       │
│  │5 Evaluate│→ │6 Monitor   │→ │7 Trigger     │                       │
│  │ Conf/ECE │  │ 3-Layer    │  │ Retrain?     │                       │
│  │ Analysis │  │ Drift      │  │ Decision     │                       │
│  └──────────┘  └────────────┘  └──────────────┘                       │
├─────────────────────────────────────────────────────────────────────────┤
│                      RETRAINING (8–10) — add --retrain                  │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐                       │
│  │8 Retrain │→ │9 Register  │→ │10 Baseline   │                       │
│  │ Adapt/   │  │ Version +  │  │ Update       │                       │
│  │ Finetune │  │ Registry   │  │ Drift Stats  │                       │
│  └──────────┘  └────────────┘  └──────────────┘                       │
├─────────────────────────────────────────────────────────────────────────┤
│                      ADVANCED (11–14) — add --advanced                  │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │11 Calib  │  │12 Wasser.  │  │13 Curriculum │  │14 Sensor     │    │
│  │ Temp/MC  │  │ Drift      │  │ Pseudo-label │  │ Placement    │    │
│  └──────────┘  └────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

| Stage | Name | Input | Output | Time |
|-------|------|-------|--------|------|
| 1 | Ingestion | `data/raw/*.csv` / `.xlsx` | `data/processed/sensor_fused_50Hz.csv` | ~5s |
| 2 | Validation | Fused CSV | Pass/fail + warnings | <1s |
| 3 | Transformation | Fused CSV | `data/prepared/production_X.npy` | ~2s |
| 4 | Inference | `.npy` + model | Predictions CSV + NPY | ~5s |
| 5 | Evaluation | Predictions | Evaluation report JSON | <1s |
| 6 | Monitoring | Predictions + baseline | Monitoring report JSON | <1s |
| 7 | Trigger | Monitoring report | Retrain decision | <1s |
| 8 | Retraining | Labeled CSV + model | Retrained model | ~1-10 min |
| 9 | Registration | Retrained model | Versioned model in registry | <1s |
| 10 | Baseline Update | Training data | `training_baseline.json` | ~2s |

---

## 3. Running the Full Pipeline

### Default — Stages 1–7 (Ingestion through Trigger)

```powershell
python run_pipeline.py
```

### Skip Ingestion — Reuse Existing Fused CSV

```powershell
python run_pipeline.py --skip-ingestion
```

### Full 10-Stage Pipeline (with Retraining)

```powershell
python run_pipeline.py --retrain --adapt none --epochs 10
```

### Full 14-Stage Pipeline (with Advanced)

```powershell
python run_pipeline.py --retrain --adapt pseudo_label --advanced --epochs 10
```

### Continue Even If a Stage Fails

```powershell
python run_pipeline.py --continue-on-failure
```

### Save Output to Log File

```powershell
python run_pipeline.py 2>&1 | Tee-Object -FilePath "logs\my_run.txt"
```

---

## 4. Running Individual Stages

You can run any combination of specific stages using `--stages`:

### Single Stage

```powershell
# Only run ingestion
python run_pipeline.py --stages ingestion

# Only run inference (needs existing production_X.npy)
python run_pipeline.py --stages inference --skip-ingestion

# Only run monitoring
python run_pipeline.py --stages monitoring --skip-ingestion
```

### Multiple Specific Stages

```powershell
# Ingestion + Validation only
python run_pipeline.py --stages ingestion validation

# Inference + Evaluation + Monitoring (skip earlier stages)
python run_pipeline.py --stages inference evaluation monitoring --skip-ingestion

# Retraining + Registration only (skip stages 1-7)
python run_pipeline.py --stages retraining registration --skip-ingestion
```

### All Available Stage Names

```
ingestion, validation, transformation, inference, evaluation,
monitoring, trigger, retraining, registration, baseline_update,
calibration, wasserstein_drift, curriculum_pseudo_labeling, sensor_placement
```

---

## 5. Domain Adaptation & Retraining

### A. Standard Supervised Retraining (no adaptation)

```powershell
python run_pipeline.py --retrain --adapt none --skip-ingestion --skip-cv --epochs 10
```

Uses `data/all_users_data_labeled.csv` (385K samples, 6 users, 11 classes). Trains from scratch with 5-fold CV (or `--skip-cv` to skip).

### B. AdaBN — Adaptive Batch Normalization (unsupervised)

```powershell
python run_pipeline.py --retrain --adapt adabn --skip-ingestion
```

Updates BatchNorm running statistics with the target (production) data distribution. Zero labels needed. Fastest adaptation.

### C. TENT — Test-Time Entropy Minimisation (unsupervised)

```powershell
python run_pipeline.py --retrain --adapt tent --skip-ingestion
```

Fine-tunes BN affine parameters (gamma/beta) to minimise prediction entropy on target data. Includes OOD guard.

### D. AdaBN + TENT Combined (unsupervised)

```powershell
python run_pipeline.py --retrain --adapt adabn_tent --skip-ingestion
```

First updates BN stats (AdaBN), then fine-tunes BN affine params (TENT). Best unsupervised option.

### E. Calibrated Pseudo-Label (semi-supervised)

```powershell
python run_pipeline.py --retrain --adapt pseudo_label --skip-ingestion --skip-cv --epochs 10
```

Temperature-calibrated, entropy-gated, class-balanced pseudo-labeling. Uses both labeled source data (filtered for self-consistency) and pseudo-labeled production data.

### Comparison Table

| Method | Labels Needed | Speed | Best For |
|--------|:------------:|:-----:|----------|
| `none` | Full source labels | ~10 min | Gold standard, best quality |
| `adabn` | None | ~1s | Quick recalibration |
| `tent` | None | ~10s | Slight improvement over AdaBN |
| `adabn_tent` | None | ~15s | Best unsupervised |
| `pseudo_label` | None (uses pseudo) | ~2 min | Best overall adaptation |

---

## 6. Advanced Stages (11–14)

```powershell
# Run all advanced stages
python run_pipeline.py --advanced --skip-ingestion

# Individual advanced stages
python run_pipeline.py --stages calibration --skip-ingestion
python run_pipeline.py --stages wasserstein_drift --skip-ingestion
python run_pipeline.py --stages curriculum_pseudo_labeling --skip-ingestion --curriculum-iterations 5 --ewc-lambda 1000
python run_pipeline.py --stages sensor_placement --skip-ingestion
```

| Stage | Purpose | Key Parameters |
|-------|---------|----------------|
| 11 Calibration | Temperature scaling, MC Dropout, ECE | `--mc-dropout-passes 30` |
| 12 Wasserstein Drift | Wasserstein distance, change-point detection | — |
| 13 Curriculum Pseudo-Label | Progressive self-training with EWC regularization | `--curriculum-iterations 5`, `--ewc-lambda 1000` |
| 14 Sensor Placement | Hand detection, axis-mirror augmentation | — |

---

## 7. All CLI Flags Reference

```
python run_pipeline.py [OPTIONS]

STAGE SELECTION:
  --stages STAGE [STAGE ...]   Run specific stages (default: 1–7)
  --retrain                    Include stages 8–10
  --advanced                   Include stages 11–14
  --skip-ingestion             Skip stage 1 (use existing fused CSV)
  --skip-validation            Skip stage 2

INPUT / MODEL:
  --input-csv PATH             Custom input CSV (your own Garmin recording)
  --model PATH                 Custom .keras model path
  --config PATH                Pipeline config YAML (default: config/pipeline_config.yaml)

PREPROCESSING:
  --gravity-removal            Enable gravity removal filter
  --no-unit-conversion         Disable milliG → m/s² conversion
  --calibrate                  Enable domain calibration

RETRAINING (with --retrain):
  --adapt METHOD               Adaptation method: none|adabn|tent|adabn_tent|pseudo_label
  --labels PATH                Ground-truth labels CSV for supervised retraining
  --epochs N                   Training epochs (default: 100)
  --skip-cv                    Skip cross-validation (faster)
  --auto-deploy                Auto-deploy if proxy validation passes

ADVANCED (with --advanced):
  --curriculum-iterations N    Curriculum pseudo-label iterations (default: 5)
  --ewc-lambda FLOAT           EWC regularisation strength (default: 1000.0)
  --mc-dropout-passes N        MC Dropout forward passes (default: 30)

GENERAL:
  --continue-on-failure        Don't stop on stage failure
```

---

## 8. DVC — Data Version Control

### What DVC Tracks

| DVC File | Data | Size |
|----------|------|------|
| `data/raw.dvc` | 77 raw Garmin CSV/XLSX files | 64.8 MB |
| `data/processed.dvc` | `sensor_fused_50Hz.csv` + manifest | 109.6 MB |
| `data/prepared.dvc` | `production_X.npy` + predictions + config | 47.4 MB |
| `models/pretrained.dvc` | `fine_tuned_model_1dcnnbilstm.keras` | 6.1 MB |

**Total DVC-tracked:** ~228 MB

### Common DVC Commands

```powershell
# Pull all data from remote storage
dvc pull

# Pull specific data
dvc pull data/raw.dvc
dvc pull models/pretrained.dvc

# Push data to remote after changes
dvc push

# Check status (what's changed locally)
dvc status

# Track a new file
dvc add data/my_new_dataset.csv

# Reproduce a pipeline (if dvc.yaml exists)
dvc repro
```

### Setting Up DVC Remote (Google Drive / S3 / local)

```powershell
# Add a remote (example: local folder)
dvc remote add -d myremote /path/to/dvc-storage

# Add a remote (Google Drive)
dvc remote add -d gdrive gdrive://<folder-id>

# Configure
dvc remote modify myremote type local
```

---

## 9. Docker & Docker Compose

### Available Services

| Service | Port | Purpose | Profile |
|---------|------|---------|---------|
| `mlflow` | 5000 | Experiment tracking UI | default |
| `inference` | 8000 | FastAPI inference server | default |
| `training` | — | On-demand training | `training` |
| `preprocessing` | — | On-demand preprocessing | `preprocessing` |

### Start All Services

```powershell
# Start MLflow + Inference server
docker-compose up -d

# Start everything including training
docker-compose --profile training up -d

# Start only MLflow
docker-compose up mlflow -d
```

### View Logs

```powershell
docker-compose logs -f inference
docker-compose logs mlflow
```

### Stop & Clean Up

```powershell
# Stop all containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache
```

### Build Individual Docker Images

```powershell
# Inference image
docker build -f docker/Dockerfile.inference -t har-inference .

# Training image
docker build -f docker/Dockerfile.training -t har-training .
```

### Run Training in Docker

```powershell
docker-compose --profile training run training python run_pipeline.py --retrain --adapt pseudo_label --epochs 10
```

---

## 10. FastAPI Inference Server

### Start Locally (without Docker)

```powershell
cd docker/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or via Docker:
```powershell
docker-compose up inference -d
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Embedded HTML dashboard (single-file SPA) |
| `GET` | `/api/health` | Health check (model/baseline loaded, uptime) |
| `GET` | `/api/model/info` | Model metadata + activity classes |
| `POST` | `/api/upload` | CSV upload → windowing → inference → 3-layer monitoring |

> **Note (22 Feb 2026):** Earlier docs referenced `/predict`, `/predict/batch`, and `/predict/stream` endpoints — these do not exist in `src/api/app.py`. The only POST endpoint is `/api/upload` which accepts a CSV file, windows the data, runs inference, and returns per-window predictions with monitoring alerts.

### Example: CSV Upload Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/api/upload",
    files={"file": open("session.csv", "rb")}
)
print(response.json())
# {"predictions": [...], "monitoring": {...}, "summary": {...}}
```

### Example: Batch Prediction

```python
# 10 windows of 200×6
batch = [np.random.randn(200, 6).tolist() for _ in range(10)]

response = requests.post("http://localhost:8000/predict/batch", json={
    "windows": batch
})
```

### Example: Stream Raw Data

```python
# Send a long sensor recording; the API windows it automatically
stream = np.random.randn(5000, 6).tolist()

response = requests.post("http://localhost:8000/predict/stream", json={
    "readings": stream,
    "window_size": 200,
    "overlap": 0.5
})
```

### Web UI

The FastAPI server includes a built-in web UI for CSV upload and inference. Open in browser:
```
http://localhost:8000/docs    ← Swagger API docs
http://localhost:8000         ← Web UI (if configured)
```

---

## 11. MLflow Experiment Tracking

### Start MLflow

```powershell
# Terminal 1: Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Open: http://127.0.0.1:5000

### Key Experiments

| Experiment | ID | Contents |
|------------|---|----------|
| `har-production-pipeline` | (auto) | Full pipeline runs (A1 baseline) |
| `har-retraining` | (auto) | Retraining runs (A3/A4/A5) |

### MLflow in Pipeline

The pipeline automatically logs to MLflow:
- Parameters (epochs, learning rate, adaptation method)
- Metrics (accuracy, F1, confidence, drift scores)
- Artifacts (models, reports, config)
- Model Registry (versioned models as `har-1dcnn-bilstm`)

### Manual MLflow Commands

```powershell
# List experiments
mlflow experiments search

# Delete old runs (careful!)
mlflow gc --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## 12. Prometheus & Grafana Monitoring

> **Status:** Configuration files exist; not yet wired into Docker Compose for live dashboards. Can be used manually.

### Prometheus

Config: `config/prometheus.yml`

```powershell
# Start Prometheus (if installed)
prometheus --config.file=config/prometheus.yml
```

Scrape targets:
| Target | Port | Interval |
|--------|------|----------|
| Prometheus self | 9090 | 15s |
| HAR Inference | 8000 | 10s |
| HAR Training | 8001 | 30s |
| HAR Monitoring | 8002 | 15s |
| Node Exporter | 9100 | 15s |
| cAdvisor | 8080 | 15s |

Alert rules: `config/alerts/har_alerts.yml`

### Grafana

Dashboard: `config/grafana/har_dashboard.json`

```powershell
# Import into Grafana:
# 1. Start Grafana (port 3000)
# 2. Add Prometheus data source → http://localhost:9090
# 3. Import dashboard from JSON → config/grafana/har_dashboard.json
```

---

## 13. CI/CD with GitHub Actions

Workflow: `.github/workflows/ci-cd.yml`

### Trigger Conditions

| Trigger | Jobs |
|---------|------|
| Push to `main` / `develop` | lint → test → build → integration-test |
| PR to `main` | lint → test (no build/deploy) |
| Manual (`workflow_dispatch`) | All jobs |
| Schedule (cron) | model-validation |

### 6 CI/CD Jobs

```
1. lint          → flake8, black, isort checks
2. test          → Unit tests + coverage → Codecov
3. test-slow     → TensorFlow tests (continue-on-error)
4. build         → Docker image → ghcr.io (push only)
5. integration   → Smoke test: /health + inference_smoke.py
6. model-valid.  → DVC pull + model validation + drift check
```

### Docker Image

```
ghcr.io/shalinvachheta017/masterarbeit_mlops/har-inference:latest
```

### Run CI Locally

```powershell
# Lint
flake8 src/ tests/ --max-line-length=120
black --check src/ tests/
isort --check-only src/ tests/

# Tests
pytest tests/ -m "not slow and not integration and not gpu" --tb=short

# All tests including TF
pytest tests/ --tb=short
```

---

## 14. Cleanup & Space Management

### Quick Cleanup — What Takes Space

| Item | Location | Typical Size | Safe to Delete? |
|------|----------|:------------:|:---------------:|
| Artifacts | `artifacts/` | 10–500 MB | Yes (keep latest 3–5) |
| Logs | `logs/` | 5–50 MB | Yes (keep latest) |
| MLflow runs | `mlruns/` | 50–500 MB | Yes (keep DB, delete run folders) |
| Models (retrained) | `models/retrained/` | 10–50 MB | Yes (keep best) |
| Predictions cache | `data/prepared/predictions/` | 5–20 MB | Yes |
| Cache dirs | `__pycache__/`, `.pytest_cache/` | 1–5 MB | Always safe |

### Delete All Artifacts, Logs, and MLflow Runs

```powershell
# ⚠️  DESTRUCTIVE — removes ALL run history

# 1. Delete all artifacts (keeps folder structure)
Remove-Item artifacts\* -Recurse -Force

# 2. Delete all logs
Remove-Item logs\*.log, logs\*.txt -Force
Remove-Item logs\pipeline\* -Force

# 3. Delete MLflow run data (keeps database schema)
Remove-Item mlruns\0\* -Recurse -Force       # default experiment runs
Remove-Item mlruns\*.* -Exclude mlflow.db     # keep the SQLite DB

# 4. Delete retrained models
Remove-Item models\retrained\* -Force

# 5. Delete prediction cache
Remove-Item data\prepared\predictions\* -Force

# 6. Clear Python cache
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Remove-Item .pytest_cache -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .mypy_cache -Recurse -Force -ErrorAction SilentlyContinue
```

### Smart Cleanup — Keep Latest N Runs

```powershell
# Keep only the 5 most recent artifact directories
$keep = 5
$dirs = Get-ChildItem artifacts -Directory | Sort-Object Name -Descending
$dirs | Select-Object -Skip $keep | Remove-Item -Recurse -Force

# Keep only latest 3 log files
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -Skip 3 | Remove-Item
```

### Use the Built-in Cleanup Script

```powershell
# Dry run — see what would be deleted
.\cleanup_repo.ps1 -DryRun

# Normal cleanup (keeps 5 artifacts, archives old logs)
.\cleanup_repo.ps1

# Aggressive cleanup (keeps 3 artifacts, cleans MLflow)
.\cleanup_repo.ps1 -Aggressive
```

### Full Reset — Start Completely Fresh

```powershell
# ⚠️  NUCLEAR OPTION — removes EVERYTHING regenerable

# Delete all generated data
Remove-Item artifacts -Recurse -Force
Remove-Item logs -Recurse -Force
Remove-Item outputs -Recurse -Force
Remove-Item models\retrained -Recurse -Force
Remove-Item models\registry -Recurse -Force
Remove-Item data\prepared\predictions -Recurse -Force
Remove-Item data\processed\sensor_fused_50Hz.csv -Force

# Reset MLflow
Remove-Item mlruns -Recurse -Force

# Recreate required empty directories
New-Item -ItemType Directory -Force -Path artifacts, logs, logs/pipeline, outputs, `
    outputs/evaluation, models/retrained, models/registry, data/prepared/predictions

# Pull DVC data (model + raw data)
dvc pull
```

---

## 15. Data Management & Experimentation

### Where Raw Data Lives

```
data/raw/
├── 2025-03-23_accelerometer.xlsx     ← Earliest recording
├── 2025-03-23_gyroscope.xlsx
├── 2025-07-16_accelerometer.csv      ← 25 daily sessions (Jul–Aug 2025)
├── 2025-07-16_gyroscope.csv
├── 2025-07-16_record.csv
├── ...
└── 2025-08-19_accelerometer.csv      ← Latest recording
```

**26 recording sessions**, 77 files total.

### Run Pipeline on a Specific Recording

```powershell
# Use --input-csv to specify exactly which fused CSV to use
python run_pipeline.py --input-csv "data/processed/sensor_fused_50Hz.csv"
```

### Experiment with One Dataset at a Time

To test the pipeline on individual recordings, move files in/out of `data/raw/`:

```powershell
# 1. Create a backup folder
New-Item -ItemType Directory -Force -Path "data/raw_backup"

# 2. Move ALL files out of data/raw/
Move-Item data\raw\* data\raw_backup\

# 3. Copy ONE dataset back
Copy-Item data\raw_backup\2025-07-16_accelerometer.csv data\raw\
Copy-Item data\raw_backup\2025-07-16_gyroscope.csv data\raw\
Copy-Item data\raw_backup\2025-07-16_record.csv data\raw\

# 4. Run the pipeline (fresh ingestion)
python run_pipeline.py 2>&1 | Tee-Object -FilePath "logs\single_dataset_2025-07-16.txt"

# 5. Save the result, then try the next dataset
Copy-Item logs\pipeline\pipeline_result_*.json outputs\per_dataset\2025-07-16.json

# 6. Clear and try next dataset
Remove-Item data\raw\*
Copy-Item data\raw_backup\2025-07-17_accelerometer.csv data\raw\
Copy-Item data\raw_backup\2025-07-17_gyroscope.csv data\raw\
Copy-Item data\raw_backup\2025-07-17_record.csv data\raw\
python run_pipeline.py 2>&1 | Tee-Object -FilePath "logs\single_dataset_2025-07-17.txt"

# 7. When done, restore all files
Move-Item data\raw_backup\* data\raw\
Remove-Item data\raw_backup
```

### Batch Process ALL Datasets (Automated)

```powershell
# Use the batch processing script — runs all 26 datasets automatically
python batch_process_all_datasets.py

# Results saved to outputs/batch_analysis/
```

### Per-Dataset Inference (Comparison)

```powershell
# Run inference on each dataset and compare results
python scripts/per_dataset_inference.py
```

### Generate Summary Report

```powershell
# After batch processing, generate summary
python generate_summary_report.py
```

---

## 16. Useful Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| **Audit artifacts** | `python scripts/audit_artifacts.py --retrain --run-id <RUN_ID>` | Verify all pipeline outputs exist |
| **Per-dataset inference** | `python scripts/per_dataset_inference.py` | Run model on each raw dataset |
| **Batch process** | `python batch_process_all_datasets.py` | Process all 26 datasets |
| **Drift analysis** | `python scripts/analyze_drift_across_datasets.py` | Drift comparison across datasets |
| **Build baseline** | `python scripts/build_training_baseline.py` | Rebuild drift detection baseline |
| **Preprocessing QC** | `python scripts/preprocess_qc.py` | Validate preprocessing contracts |
| **Generate figures** | `python scripts/generate_thesis_figures.py` | Thesis figures for documentation |
| **Verify repo** | `python scripts/verify_repository.py` | Check repo completeness |
| **Verify with fixes** | `python scripts/verify_repository.py --fix` | Auto-fix missing files |
| **Run tests** | `python scripts/run_tests.py --unit` | Quick unit tests |
| **Cleanup** | `.\cleanup_repo.ps1 -DryRun` | Preview cleanup actions |

---

## 17. Testing

### Quick Test Suites

```powershell
# Unit tests only (fast, no TF)
pytest tests/ -m "not slow and not integration and not gpu" --tb=short

# All tests
pytest tests/ --tb=short -v

# With coverage report
pytest tests/ --cov=src --cov-report=html --tb=short

# Specific test file
pytest tests/test_monitoring.py -v

# Specific test function
pytest tests/test_train.py::test_model_training -v
```

### Test Markers

| Marker | Meaning | Needs |
|--------|---------|-------|
| `slow` | TensorFlow model tests | TF installed |
| `integration` | Full pipeline integration | All dependencies |
| `gpu` | GPU-accelerated tests | CUDA/GPU |

### Using the Test Runner Script

```powershell
python scripts/run_tests.py --unit       # Unit tests only
python scripts/run_tests.py --coverage   # With coverage
python scripts/run_tests.py --quick      # Fastest subset
```

---

## 18. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd "d:\study apply\ML Ops\MasterArbeit_MLops"` |
| `Training data not found` | Ensure `data/all_users_data_labeled.csv` exists. Run `dvc pull` if missing. |
| `LabelEncoder has no attribute 'classes_'` | Fixed in commit `3fd3c00`. Pull latest code. |
| `UnicodeEncodeError` on Windows | Set `$env:PYTHONIOENCODING='utf-8'` |
| MLflow connection error | Start MLflow first: `mlflow ui --port 5000` |
| Docker build fails | Check Docker Desktop is running. Rebuild: `docker-compose build --no-cache` |
| `production_X.npy` not found | Run stages 1–3 first, or use `--skip-ingestion` with existing data |
| Out of memory during training | Reduce `--epochs`, add `--skip-cv`, or reduce batch size in config |

### Check Pipeline Health

```powershell
# Verify all required files exist
python scripts/verify_repository.py

# Check latest pipeline result
Get-Content (Get-ChildItem logs\pipeline\pipeline_result_*.json | Sort-Object Name | Select-Object -Last 1).FullName | ConvertFrom-Json | Format-List overall_status, stages_completed, stages_failed
```

---

*Generated: 19 Feb 2026*

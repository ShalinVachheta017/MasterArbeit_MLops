# Quick Run Guide (Raw → Inference)
**Date:** 2025-12-11  
**Goal:** Minimal steps to go from raw Garmin data to predictions, with/without Docker, plus DVC tracking.

## 1) Prereqs (local)
- Activate env: `conda activate thesis-mlops` (or your venv)
- Install deps: `pip install -r config/requirements.txt`
- Ensure MLflow available: `pip show mlflow` (or use Docker service)

## 2) Run Pipeline (3 steps)

### Step 1: Raw → Fused CSV (auto-discover latest accel/gyro in data/raw)
```powershell
python src/sensor_data_pipeline.py
```
Output: `data/preprocessed/sensor_fused_50Hz.csv`

### Step 2: Fused CSV → Inference Windows (.npy)
**Without gravity removal:**
```powershell
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv
```

**With gravity removal (recommended if training data had gravity removed):**
```powershell
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --gravity-removal
```
Output: `data/prepared/production_X.npy`, `production_metadata.json`

### Step 3: Inference
```powershell
python src/run_inference.py --input data/prepared/production_X.npy --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras --output data/prepared/predictions
```
Output: `data/prepared/predictions/` (CSV, metadata, probs)

## 3) Compare gravity on vs off
Run steps 1-3 twice:
1. Without `--gravity-removal` → check predictions
2. With `--gravity-removal` → check predictions
Compare distribution of predicted classes.

## 6) Track artifacts with DVC
```powershell
# Track new preprocessing + prepared artifacts
dvc add data/preprocessed data/prepared
# Push to remote storage
dvc push
# Commit dvc files
git add data/preprocessed.dvc data/prepared.dvc data/preprocessed/.gitignore data/prepared/.gitignore
git commit -m "track new preprocessing run"
```

## 7) Docker path (services + on-demand jobs)
- Start services (MLflow + inference API):
```powershell
docker-compose up -d mlflow inference
```
- Preprocess raw → fused CSV inside container:
```powershell
docker-compose --profile preprocessing run --rm preprocessing python src/sensor_data_pipeline.py
```
- Create windows:
```powershell
docker-compose --profile preprocessing run --rm preprocessing python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv
```
- Inference options:
  - API: Swagger at http://localhost:8000/docs (POST /predict)
  - Script inside container:
```powershell
docker-compose run --rm inference python src/run_inference.py --input data/prepared/production_X.npy --output data/prepared/predictions
```

## 8) MLflow UI
- Local: `mlflow ui --port 5000` (blocks terminal; use new tab)
- Docker: `docker-compose up -d mlflow` then open http://localhost:5000

## 9) Run twice (gravity on/off)
1) Set `enable_gravity_removal: true`, run steps 3–5 (and DVC if desired).
2) Set `enable_gravity_removal: false`, rerun steps 3–5.
3) Compare outputs in `data/prepared/predictions/` or via MLflow if logged during evaluation.

## 10) Minimal command set (local)
```powershell
# 1) Raw -> fused
python src/sensor_data_pipeline.py
# 2) Fused -> windows
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv
# 3) Inference
python src/run_inference.py --input data/prepared/production_X.npy --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras --output data/prepared/predictions
```

---
For reference, a fuller runbook is at docs/PIPELINE_RUNBOOK.md.

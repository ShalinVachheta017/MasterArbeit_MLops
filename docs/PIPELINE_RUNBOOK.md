# Pipeline Runbook (Local + Docker)
**Date:** 2025-12-11
**Thesis Progress:** ~50% complete (data versioning, tracking, containerization done; CI/CD + monitoring pending)

## End-to-End Flow (Happy Path)
1) Pull data + models: `dvc pull`
2) Ingest raw Garmin Excel → fused CSV: `sensor_data_pipeline.py`
3) Create model-ready windows: `preprocess_data.py --input pre_processed_data/sensor_fused_50Hz.csv`
4) Run inference: `run_inference.py --input data/prepared/production_X.npy`
5) Track experiments (optional training): MLflow UI at http://localhost:5000
6) Version new outputs: `dvc add ... && dvc push`

## What to Run (Local)
- **Prereqs:** activate env, `pip install -r config/requirements.txt`; pull data `dvc pull`
- **1) Raw → fused 50Hz CSV** (accelerometer + gyroscope Excel)
  - Place files under `data/` (e.g., `data/accel.xlsx`, `data/gyro.xlsx`)
  - Run (PowerShell):
    ```powershell
    python - <<'PY'
    from pathlib import Path
    from src.sensor_data_pipeline import SensorDataPipeline
    base = Path(__file__).resolve().parent
    pipeline = SensorDataPipeline(base)
    pipeline.process_sensor_files(base/'data'/'accel.xlsx', base/'data'/'gyro.xlsx')
    PY
    ```
  - Outputs: `pre_processed_data/sensor_fused_50Hz.csv`, `sensor_merged_native_rate.csv`, `sensor_fused_meta.json`
- **2) Fused CSV → windows (.npy)**
  - Command: `python src/preprocess_data.py --input pre_processed_data/sensor_fused_50Hz.csv`
  - Does: unit auto-detect/convert (milliG→m/s²), normalization (uses `data/prepared/config.json` scaler), sliding windows (200x6, 50% overlap)
  - Outputs: `data/prepared/production_X.npy`, `production_metadata.json`
- **3) Inference**
  - Command: `python src/run_inference.py --input data/prepared/production_X.npy --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras --output data/prepared/predictions`
  - Outputs: `predictions/predictions.csv`, `predictions_summary.json`, log in `logs/inference/`
- **4) Version artifacts**
  - `dvc add pre_processed_data data/prepared` (and any new model) 
  - `dvc push`
  - `git add *.dvc pre_processed_data/.gitignore data/prepared/.gitignore`
  - `git commit -m "updated read me" && git push`

## What to Run (Docker)
- Start tracking + inference: `docker-compose up -d mlflow inference`
  - MLflow UI: http://localhost:5000
  - API: http://localhost:8000/docs
- On-demand preprocessing: `docker-compose --profile preprocessing run --rm preprocessing python src/sensor_data_pipeline.py`
- On-demand training (if needed later): `docker-compose --profile training run --rm training python src/train.py`
- Logs: `docker-compose logs -f inference`

## Behind the Scenes
- `sensor_data_pipeline.py`: parses Excel list cells, validates schema, explodes samples, aligns accel/gyro timestamps, resamples to 50Hz, optional gravity removal (config/pipeline_config.yaml), writes fused CSV + metadata.
- `preprocess_data.py`: detects units, converts to m/s² if needed, loads saved scaler, normalizes, builds sliding windows (200x6), saves NumPy + metadata. Labeled training is purposely blocked here.
- `run_inference.py`: loads `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`, runs batch or realtime inference, flags low-confidence predictions, writes predictions + logs.
- `config.py`: central paths (data/, models/, logs/, config/, etc.), window size 200, overlap 0.5.
- Docker: `docker-compose.yml` wires MLflow (port 5000), FastAPI inference (port 8000, mounts models/config/logs), training/preprocessing profiles reuse training image.

## File Hand-offs
- Raw Excel → **pre_processed_data/sensor_fused_50Hz.csv** → **data/prepared/production_X.npy** → **data/prepared/predictions/**
- Metadata: `sensor_fused_meta.json`, `production_metadata.json`, inference logs in `logs/inference/`

## Quick Checklist for New Data
- [ ] Place accel/gyro Excel in `data/`
- [ ] Run sensor pipeline (step 1)
- [ ] Run preprocessing to .npy (step 2)
- [ ] Run inference (step 3) or API
- [ ] `dvc add` + `dvc push`
- [ ] `git commit -m "updated read me" && git push`

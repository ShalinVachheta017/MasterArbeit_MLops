# Tooling Roles: DVC · MLflow · Docker · FastAPI

**Thesis:** Developing a MLOps Pipeline for Continuous Mental Health Monitoring Using Wearable Sensor Data  
**Author:** Shalin Vachheta  
**Date:** February 2026  
**Purpose:** Explains what each major infrastructure tool does in this pipeline — first as a plain-English summary, then mapped to every relevant pipeline stage (Stages 0–13). For use in the thesis architecture chapter and in viva preparation.

---

## Quick Reference — One-Sentence Roles

| Tool | One-Sentence Role |
|---|---|
| **DVC** | Tracks and versions every data file, processed artefact, and model so that any past pipeline run can be reproduced exactly. |
| **MLflow** | Records every experiment — parameters, metrics, artefacts, and model files — and maintains a model registry that controls what is in production. |
| **Docker** | Packages the inference API and its dependencies into a portable container so the model can be served identically on any machine or cloud environment. |
| **FastAPI** | Provides the HTTP REST interface through which clients upload sensor CSV files and receive activity predictions and monitoring results. |

---

## 1. DVC (Data Version Control)

### What DVC is

DVC is a version control system for data and machine learning pipelines, designed to work alongside Git. Git tracks code; DVC tracks large binary files (CSV datasets, trained models, preprocessed artefacts) that cannot be committed to Git directly.

### How DVC works in this project

- `.dvc` pointer files are committed to Git. Each pointer stores the MD5 hash and storage path of the real file it tracks.
- The real data files live in `.dvc/cache/` (local) or in a remote storage location defined in `.dvc/config`.
- `dvc repro` re-executes only the pipeline stages whose upstream inputs have changed, giving reproducible and incremental re-runs.
- `dvc push` / `dvc pull` synchronise data between the local cache and the remote store, enabling other developers or the CI runner to work with the same dataset.

### DVC role at each stage

| Stage | Stage Name | DVC Role |
|---|---|---|
| **Stage 0** | Data Ingestion | The raw sensor CSV (`data/raw/sensor_fused_50Hz.csv`) is tracked as a DVC artefact. Pulling this file with `dvc pull` guarantees other runs use byte-identical raw data. |
| **Stage 1** | Data Validation | Validated data schema report saved to `artifacts/{timestamp}/` — tracked by DVC so the validation result for a specific data version is stored and linked to it. |
| **Stage 2** | Preprocessing | `data/processed/sensor_fused_50Hz_converted.csv` and `data/prepared/config.json` (the fitted StandardScaler parameters) are DVC-tracked outputs. Changing the raw data automatically marks these as stale and triggers re-preprocessing. |
| **Stage 3** | Feature Engineering | Windowed feature files (`data/prepared/X_train.npy`, `y_train.npy`, etc.) are DVC outputs. The window size (200 samples) and overlap (50%) are recorded in the stage's `dvc.yaml` params section. |
| **Stage 4** | Model Ingestion | `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` is DVC-tracked. Any consumer downstream sees the exact model weights used — even months later. |
| **Stage 5** | Model Evaluation | Evaluation reports (`artifacts/{timestamp}/evaluation_report.json`) are tracked artefacts. The model version and data version that produced them are linked implicitly through DVC's dependency graph. |
| **Stage 6** | Experiment Tracking | DVC records which data version was used as input to each training run. `src/model_rollback.py` stores `data_version` (the DVC hash) as part of the rollback metadata, so a rollback can restore both model AND data. |
| **Stage 7** | Training | Trained model weights saved to `artifacts/{timestamp}/model.keras` — DVC-tracked output. `dvc repro` will re-train only if the training data or config has changed. |
| **Stage 9** | Baseline Construction | `models/training_baseline.json` and `models/normalized_baseline.json` are DVC-tracked. These reference distributions must match the exact training data version; the DVC link makes this traceable. |
| **Stages 10–11** | Drift Detection, Trigger Policy | Per-session post-inference CSVs written by `batch_process_all_datasets.py` are DVC-trackable. The drift scores they produce are reproducible from a fixed data pointer. |
| **Stage 13** | Monitoring | Audit artefacts (`artifacts/{timestamp}/audit_results.json`) are DVC outputs. The audit can be re-run against any combination of frozen model + frozen data by checking out the relevant Git commit and running `dvc pull`. |

### What DVC does NOT do here

DVC does not serve models, run training, or provide a UI. It is purely a versioning and reproducibility layer.

---

## 2. MLflow

### What MLflow is

MLflow is an open-source platform for the machine learning lifecycle. It provides four main components used here: Tracking (logging experiment results), Models (saving model artefacts in a standard format), Model Registry (versioning deployed models), and Projects (packaging code — not used in this thesis).

### How MLflow works in this project

- A local MLflow tracking server runs at `http://localhost:5000` (via the `har-mlflow` Docker service in `docker-compose.yml`).
- All experiment data is stored in `mlruns/` (SQLite backend) and model artefacts in `mlruns/<experiment_id>/<run_id>/artifacts/`.
- `src/mlflow_tracking.py` provides a `MLflowTracker` class that wraps all MLflow calls across the pipeline.
- The MLflow Model Registry (`models/registry/model_registry.json`) is the source of truth for which model is currently promoted to Production.

### MLflow role at each stage

| Stage | Stage Name | MLflow Role |
|---|---|---|
| **Stage 0** | Data Ingestion | Dataset metadata (row count, column count, timestamp) logged as MLflow params under experiment `har-data-ingestion`. |
| **Stage 2** | Preprocessing | StandardScaler parameters (mean, std per channel) logged as params. Unit conversion flag (`enable_unit_conversion`) logged so the preprocessing decision is reproducible. |
| **Stage 3** | Feature Engineering | Window count, window size, overlap fraction logged. Shape of `X_train` / `X_test` logged as metrics. |
| **Stage 5** | Model Evaluation | Per-class F1 score, overall accuracy, macro average — all logged as MLflow metrics. The classification report saved as an artefact. |
| **Stage 6** | Experiment Tracking (primary) | This is MLflow's main stage. Every training run creates a named MLflow run under `har-training`. Logged items: all `TrainingConfig` hyperparameters as params; epoch-by-epoch `val_loss` and `val_accuracy` as metrics; final trained model via `mlflow.keras.log_model(model, name="model")`. The `input_example` and model signature are also stored so the model can be loaded and called without knowing the input schema. |
| **Stage 7** | Training | Nested runs used for cross-validation folds. Each fold's `val_accuracy` is logged; the best fold's model is promoted to the parent run. |
| **Stage 9** | Baseline Construction | Baseline statistics (per-channel mean, std, drift thresholds) logged as artefacts. Commit hash of `models/normalized_baseline.json` logged as a param for traceability. |
| **Stage 10** | Drift Detection | PSI scores per channel logged as MLflow metrics in `har-monitoring`. `drift_detected: 0/1` logged as a param. |
| **Stage 11** | Trigger Policy | Trigger decision (`forced_retrain`, `retrain_triggered`, `no_trigger`) logged as an MLflow param under `har-trigger`. Retrain reason (PSI, z-score, schedule) logged as a text artefact. |
| **Stage 12** | Safe Retraining | Dedicated experiment `har-retraining`. Logged items: adaptation method (`adabn_tent`, `pseudo_label`); pre-/post-adaptation mean confidence; `tent_rollback: 0/1`; `entropy_delta`; `confidence_improvement`; model artefact. Canary evaluation results (window-level accuracy on held-out users) stored as an artefact. |
| **Stage 13** | Monitoring, Canary, Registry | MLflow Model Registry records: model version, `current_version` field in `models/registry/model_registry.json`, promotion status (`Staging`, `Production`, `Archived`). When a model passes the 5% canary tolerance gate, it is registered and tagged `Production`. When it fails (e.g., confidence dropped 7.6% in the AdaBN+TENT run on 2026-02-23), it is tagged `Archived`. `current_version: null` means no model has passed the canary gate — the original pretrained model continues to serve. |

### What MLflow does NOT do here

MLflow does not run containers, schedule pipeline stages, or serve models via HTTP (the FastAPI / Docker stack does that). It is purely an observability and registry layer.

---

## 3. Docker

### What Docker is

Docker packages an application and all its dependencies (Python, TensorFlow, system libraries) into a self-contained image. When the image is run, it creates a container — an isolated process that behaves the same on any host.

### How Docker works in this project

Two Docker services are defined in `docker-compose.yml`:

1. **`har-mlflow`** — Runs the MLflow tracking server (`mlflow server`) on port 5000. Mounts `./mlruns` and `./models` so data persists across container restarts.
2. **`har-inference`** — Runs the FastAPI inference API (`uvicorn src.api.app:app`) on port 8000. Built from `docker/Dockerfile.inference`. Mounts `./models` read-only so the container uses the current production model.

The CI/CD workflow (`.github/workflows/ci-cd.yml`) builds the inference image and pushes it to GitHub Container Registry (`ghcr.io/shalinvachheta017/masterarbeit_mlops/har-inference`).

### Docker role at each stage

| Stage | Stage Name | Docker Role |
|---|---|---|
| **Stage 0–7** | Data to Training | No direct Docker role. These stages run in the host Python environment (or CI runner). DVC and MLflow are language-level libraries, not Docker services. |
| **Stage 6** | Experiment Tracking | The MLflow tracking server must be running (`docker-compose up -d mlflow`) before any training or monitoring stage can log to `http://localhost:5000`. |
| **Stage 8** | Deployment and Inference | The `har-inference` container is the deployment artefact. The container: (1) loads Python 3.11-slim as base; (2) installs requirements from `requirements.txt`; (3) copies `src/`, `models/`, `config/` into the image; (4) sets `PYTHONPATH=/app:/app/src`; (5) runs `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`. The model is baked in via a volume mount (not into the image), so updating `models/pretrained/` takes effect on next container restart without rebuilding the image. |
| **Stage 10–11** | Monitoring, Trigger | Monitoring runs against the live API through its HTTP interface. The container must be running for post-inference monitoring to collect responses. |
| **Stage 13** | Canary Deployment | The CI/CD workflow builds a candidate image from the retrained model and runs the integration smoke test against it. If the smoke test passes, the image is tagged `latest` and pushed. If not, the `Archived` tag is applied in the model registry instead. |

### Key Dockerfile decisions (from commit history)

- `docker/api/` directory renamed to `/app/docker_api` inside the container to prevent Python import shadowing of `src/api/app.py` (fix: commit `7f892d8`).
- `PYTHONPATH=/app:/app/src` set explicitly so `from src.api.app import app` resolves correctly (fix: commit `7f892d8`).
- Image name is all-lowercase (`har-inference`) because GHCR rejects uppercase names (fix: commit `8b4dab7`).
- Health check endpoint is `/api/health`, not `/health` — corrected in both the Dockerfile `HEALTHCHECK` instruction and the CI workflow (fix: commit `edbc399`).

---

## 4. FastAPI

### What FastAPI is

FastAPI is a modern Python web framework for building HTTP APIs. It uses Python type annotations to automatically generate request validation, serialisation, and interactive API documentation (Swagger UI at `/docs`). It runs under `uvicorn`, an async HTTP server.

### How FastAPI works in this project

The entire inference API lives in `src/api/app.py`. On startup (via the `@asynccontextmanager` lifespan hook), the application:
1. Loads `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` into memory.
2. Loads `models/normalized_baseline.json` (reference distribution for drift detection).
3. Reads monitoring thresholds from `src/entity/config_entity.py :: PostInferenceMonitoringConfig`.

The `ACTIVITY_CLASSES` dictionary (11 anxiety-behaviour classes) and the preprocessing pipeline (StandardScaler parameters from `data/prepared/config.json`) are applied within the API request handler before inference.

### FastAPI role at each stage

| Stage | Stage Name | FastAPI Role |
|---|---|---|
| **Stage 0–7** | Data to Training | No direct role. FastAPI is only active during serving (Stage 8 onwards). |
| **Stage 8** | Deployment and Inference | FastAPI provides two primary endpoints: (1) `POST /api/predict` — accepts a multipart CSV file upload; returns predicted activity class, confidence score, and per-window probabilities. (2) `GET /api/health` — returns `{"status": "healthy"}`, the container health check endpoint. The preprocessing pipeline (unit detection → StandardScaler → windowing at 200 samples, 50% overlap) runs inside the request handler before the model is called. |
| **Stage 9** | Post-Inference Monitoring (Layer 1 — Confidence) | After each batch prediction, the API runs Layer 1 monitoring: computes mean confidence across windows; compares against `confidence_warn_threshold = 0.60` and `uncertain_pct_threshold = 30%`. Warnings are returned in the response JSON alongside the predictions so the caller knows immediately if quality is suspect. |
| **Stage 9** | Post-Inference Monitoring (Layer 2 — Temporal) | Layer 2 computes the activity transition rate (how often the predicted class changes between consecutive windows). If `transition_rate > 50%` the prediction is flagged as erratic — a sign of low-quality or misaligned sensor data. |
| **Stage 10** | Post-Inference Monitoring (Layer 3 — Drift) | Layer 3 computes per-channel z-scores of the incoming data relative to `models/normalized_baseline.json`. If `z_score > drift_zscore_threshold = 2.0` on any channel, the response includes a `drift_warning` flag and specifies which channels are drifting. This is the online, per-request drift check; the offline PSI-based batch drift check (Stage 10 trigger) runs separately via `scripts/post_inference_monitoring.py`. |
| **Stage 11–13** | Trigger, Retraining, Monitoring | The inference API's `/api/predict` response includes a `monitoring_summary` field that `scripts/post_inference_monitoring.py` can read to aggregate drift flags across a session. These aggregated flags feed the trigger policy (Stage 11). The API itself does not trigger retraining — it only reports observations. |

### FastAPI endpoint summary

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/api/predict` | POST | CSV file (multipart) | `predictions`, `activity_counts`, `mean_confidence`, `uncertain_windows_pct`, `transition_rate`, `drift_warning`, `monitoring_summary` |
| `/api/health` | GET | — | `{"status": "healthy", "model": "loaded", "timestamp": "..."}` |
| `/docs` | GET | — | Swagger interactive documentation (auto-generated by FastAPI) |
| `/` | GET | — | HTML dashboard with upload form and visualisation |

---

## How the Four Tools Interact

```
┌─────────────────────────────────────────────────────────────────┐
│                       Developer / CI                            │
│                                                                 │
│  git commit → DVC tracks data hash                              │
│             → MLflow logs training run                          │
│             → Docker builds inference image                     │
│             → FastAPI serves predictions                        │
└─────────────────────────────────────────────────────────────────┘

Data Layer        DVC
                  ├── data/raw/ (tracked)
                  ├── data/processed/ (tracked)
                  ├── data/prepared/ (tracked, includes scaler config)
                  └── models/pretrained/ (tracked)

Experiment Layer  MLflow
                  ├── mlruns/ (all run history)
                  ├── Experiment: har-training
                  ├── Experiment: har-retraining
                  ├── Experiment: har-monitoring
                  └── Model Registry → Production / Archived

Serving Layer     Docker
                  ├── har-mlflow container (port 5000)
                  └── har-inference container (port 8000)
                        │
                        └── FastAPI
                              ├── POST /api/predict  (inference + 3-layer monitoring)
                              └── GET  /api/health   (health check)

Retraining Loop   (all four tools participate)
  DVC     → provides data version for audit trail
  MLflow  → logs retrain experiment, canary results, registry update
  Docker  → builds candidate image for canary integration test
  FastAPI → serves retrained model after container restart
```

### Tool interaction at the retraining decision point (Stage 12 → 13)

1. Drift detected (Stage 10 PSI > 0.75) → trigger fires (Stage 11).
2. `run_pipeline.py --retrain --adapt adabn_tent` executes.
3. **DVC** ensures the training dataset used is the same version that produced the current production baseline.
4. **MLflow** logs the adaptation run: pre/post confidence, `tent_rollback`, `entropy_delta`, model artefact.
5. If canary passes: **Docker** image is rebuilt with updated model volume and pushed to GHCR.
6. **FastAPI** container is restarted, loads the new model weights from the updated volume mount.
7. **MLflow** Model Registry entry updated: previous model → `Archived`, new model → `Production`.

---

## Why Each Tool Was Chosen (Thesis Justification Paragraph)

**DVC** was chosen over alternatives (e.g., Git LFS, Neptune.ai data versioning) because it integrates directly with Git without a cloud dependency, supports local caching, and allows stage-level incremental reproduction via `dvc repro`. For a thesis project with large sensor CSV files, it provides reproducibility without requiring a cloud storage account.

**MLflow** was chosen over Weights & Biases or TensorBoard because it is entirely self-hosted (no API keys, no data leaves the machine), supports both experiment tracking and a model registry in a single tool, and has a stable Python API that handles the full lifecycle from `log_params` through model promotion and archiving.

**Docker** was chosen because it is the industry-standard containerisation tool, is natively supported by GitHub Actions (the CI/CD provider), and allows the exact inference environment — including the Python and TensorFlow version — to be frozen and reproduced on any host. The `docker-compose.yml` configuration also allows the MLflow server and the inference API to be started together with a single command.

**FastAPI** was chosen over Flask or Django because its async architecture handles concurrent sensor data uploads efficiently, its type-annotation-based validation eliminates a class of input error bugs, and its automatic Swagger UI (`/docs`) provides immediate interactive documentation with no extra work — useful during thesis demonstration.

---

*All endpoint paths, threshold values, and file paths in this document are taken directly from the source code as of commit `b92ae0a` (2026-02-23). Verify with `git show <hash> -- <file>` if in doubt.*

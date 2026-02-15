# 🧠 MLOps Pipeline for Mental Health Monitoring

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://tensorflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.50+-purple.svg)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-green.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-225%20Passing-brightgreen.svg)](tests/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](.github/workflows/ci-cd.yml)

**Master's Thesis Project** | January 2026 - May 2026  
**Last Updated:** February 15, 2026  
**Progress:** 95% complete

---

## 📊 Current Status

> **🎯 MAIN DOCUMENT:** See [docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md](docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md) for complete thesis status, achievements, and remaining work.

**Latest Achievements (Feb 15, 2026):**
- ✅ **CI/CD Pipeline** fully operational with GitHub Actions
- ✅ **All 225 tests passing** (full test coverage)
- ✅ **FastAPI Web UI** with CSV upload & interactive dashboard
- ✅ **3-layer monitoring** integrated (confidence, temporal, drift)
- ✅ **Production optimizations** (971x model caching, vectorized windowing)
- ✅ **Docker images** automatically built and pushed to ghcr.io

**Quick Links:**
- 🚀 [Run the FastAPI](#-quick-start): `python -m src.api.app`
- 📖 [Thesis Plan](Thesis_Plan.md): Original 6-month timeline
- 🧪 [Run Tests](#-testing): `pytest tests/`
- 🔧 [Pipeline Guide](docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md): Full pipeline docs

---

## 📚 Key Documentation

| Document | Purpose |
|----------|---------|
| [Thesis Status & Plan](docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md) | **Main document** - Current status, achievements, timeline |
| [What's Remaining](docs/WHATS_REMAINING.md) | **Quick view** - What's left to complete (5% remaining) |
| [CI/CD Beginner's Guide](docs/GITHUB_ACTIONS_CICD_BEGINNER_GUIDE.md) | Complete GitHub Actions tutorial from scratch |
| [Thesis Plan (Original)](Thesis_Plan.md) | Original 6-month roadmap (Oct 2025 - Apr 2026) |
| [Pipeline Operations](docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md) | Complete pipeline documentation & architecture |
| [API Documentation](docs/DATA_INGESTION_AND_INFERENCE_GUIDE.md) | FastAPI endpoints and usage |
| [Research Papers Analysis](docs/HAR_MLOps_QnA_With_Papers.md) | Insights from 77+ research papers |

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture & Pipeline Flow](#-architecture--pipeline-flow)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [📖 Complete Documentation](#-complete-documentation)
6. [DVC - Data Version Control](#-dvc---data-version-control)
7. [MLflow - Experiment Tracking](#-mlflow---experiment-tracking)
8. [Docker - Containerization](#-docker---containerization)
9. [Pipeline Stages](#-pipeline-stages)
10. [Adding New Datasets](#-adding-new-datasets)
11. [API Reference](#-api-reference)
12. [Configuration](#-configuration)
13. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

An end-to-end MLOps pipeline for **Human Activity Recognition (HAR)** using wearable sensor data. The system recognizes 11 anxiety-related activities from accelerometer and gyroscope data collected via Garmin smartwatches.

### Key Features

| Feature | Technology | Status |
|---------|------------|--------|
| Data Versioning | DVC | ✅ Complete |
| Experiment Tracking | MLflow | ✅ Complete |
| Containerization | Docker | ✅ Complete |
| Model Serving API | FastAPI | ✅ Complete |
| Domain Calibration | Distribution Alignment | ✅ Complete |
| Gravity Removal | Butterworth Filter | ✅ Complete |
| CI/CD Pipeline | GitHub Actions | ✅ Complete |
| Monitoring | Prometheus/Grafana | ⏳ Planned |

### Model Details

- **Architecture:** 1D-CNN-BiLSTM (1.5M parameters)
- **Input:** 200 timesteps × 6 sensors (4 seconds @ 50Hz)
- **Output:** 11 activity classes
- **Sensors:** Ax, Ay, Az (accelerometer) + Gx, Gy, Gz (gyroscope)

### Activity Classes

```
0: ear_rubbing      4: hand_tapping     8: sitting
1: forehead_rubbing 5: knuckles_cracking 9: smoking
2: hair_pulling     6: nail_biting      10: standing
3: hand_scratching  7: nape_rubbing
```

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  New Dataset │
                              │  (Garmin CSV)│
                              └──────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. DATA INGESTION                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐                │
│  │ Raw Data    │───▶│ Validation   │───▶│ DVC Tracking    │                │
│  │ (data/raw/) │    │ (data_       │    │ (dvc add)       │                │
│  │             │    │  validator)  │    │                 │                │
│  └─────────────┘    └──────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2. PREPROCESSING                                    │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────────┐   │
│  │ Sensor      │───▶│ Unit         │───▶│ Domain       │──▶│ Windowing│   │
│  │ Fusion      │    │ Conversion   │    │ Calibration  │   │ (200x6)  │   │
│  │ (50Hz)      │    │ (milliG→m/s²)│    │ (Align Dist) │   │ 50% Olap │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   └──────────┘   │
│         │                   │                    │                │         │
│         ▼                   ▼                    ▼                ▼         │
│  sensor_fused.csv    Az: -9.83 m/s²     offset: -6.30      1815 windows   │
│  (181,699 samples)   (raw gravity)      → -3.53 m/s²       prepared/*.npy │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3. TRAINING (Optional)                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐                │
│  │ Load Data   │───▶│ Train Model  │───▶│ Log to MLflow   │                │
│  │ (DVC pull)  │    │ (1D-CNN-     │    │ (metrics,       │                │
│  │             │    │  BiLSTM)     │    │  artifacts)     │                │
│  └─────────────┘    └──────────────┘    └─────────────────┘                │
│                            │                    │                           │
│                            ▼                    ▼                           │
│                     models/trained/       mlruns/                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         4. INFERENCE (Production)                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐                │
│  │ Docker      │───▶│ FastAPI      │───▶│ Predictions     │                │
│  │ Container   │    │ /predict     │    │ + Confidence    │                │
│  │             │    │ endpoint     │    │                 │                │
│  └─────────────┘    └──────────────┘    └─────────────────┘                │
│         │                                                                   │
│         ▼                                                                   │
│  localhost:8000/docs  (Swagger UI)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         5. MONITORING (Operational)                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐                │
│  │ Layer 1:    │    │ Layer 2:     │    │ Layer 3:        │                │
│  │ Confidence  │───▶│ Temporal     │───▶│ Drift           │                │
│  │ Analysis    │    │ Patterns     │    │ Detection       │                │
│  └─────────────┘    └──────────────┘    └─────────────────┘                │
│         │                   │                    │                          │
│         └───────────────────┴────────────────────┘                          │
│                             │                                                │
│                             ▼                                                │
│                     Trigger Evaluation                                       │
│                     (PASS/WARNING/FAIL)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Git

### 1. Clone & Setup

```powershell
# Clone repository
git clone https://github.com/ShalinVachheta017/MasterArbeit_MLops-.git
cd MasterArbeit_MLops

# Create conda environment
conda create -n thesis-mlops python=3.11 -y
conda activate thesis-mlops

# Install dependencies
pip install -r config/requirements.txt
```

### 2. Pull Data with DVC

```powershell
# Pull all versioned data from DVC storage
dvc pull

# Verify data
ls data/prepared/
ls models/pretrained/
```

### 3. Run FastAPI Web Application (Recommended)

```powershell
# Start the FastAPI server with web UI
python -m src.api.app

# Open browser to http://127.0.0.1:8000
# - Drag & drop CSV file with sensor data
# - Auto-detects columns (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
# - Full pipeline: windowing → inference → 3-layer monitoring
# - Interactive dashboard with results
```

**API Endpoints:**
- `GET /` - Web dashboard (interactive UI)
- `POST /api/upload` - Upload CSV for inference & monitoring
- `GET /api/health` - System health check
- `GET /api/model/info` - Model information

### 4. OR: Start Services with Docker

```powershell
# Start MLflow + Inference API
docker-compose up -d mlflow inference

# Check status
docker-compose ps

# View logs
docker-compose logs -f inference
```

### 4. Test the API

```powershell
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Open Swagger UI in browser
start http://localhost:8000/docs
```

### 5. View Experiments in MLflow

```powershell
# Open MLflow UI
start http://localhost:5000
```

---

## 🧪 Testing

**Run full test suite** (225 tests across all components):

```powershell
pytest tests/ -v
```

**Quick test run** (essential tests only):

```powershell
pytest tests/ -m "not slow"
```

**Test coverage**:
- Pipeline stages (preprocessing, training, evaluation)
- API endpoints and request handling
- Feature engineering (temporal, statistical, spectral)
- Data validation and error handling
- Monitoring layers (confidence, temporal, drift)

---

## 📚 Complete Documentation

### 📚 Main Documentation Files

| Document | Purpose |
|----------|---------|
| **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** | Complete folder/file structure with visuals |
| **[docs/PIPELINE_RERUN_GUIDE.md](docs/PIPELINE_RERUN_GUIDE.md)** | Step-by-step pipeline execution |
| **[docs/SRC_FOLDER_ANALYSIS.md](docs/SRC_FOLDER_ANALYSIS.md)** | Detailed source code analysis |
| **[docs/CONCEPTS_EXPLAINED.md](docs/CONCEPTS_EXPLAINED.md)** | Technical concepts & formulas |
| **[docs/RESEARCH_PAPERS_ANALYSIS.md](docs/RESEARCH_PAPERS_ANALYSIS.md)** | Reference papers & summaries |
| **[docs/FILE_ORGANIZATION_SUMMARY.md](docs/FILE_ORGANIZATION_SUMMARY.md)** | Doc organization & archive info |

### 📦 Archived Documentation

Old/outdated docs moved to [docs/archived/](docs/archived/):
- **DELETE_\*.md** (26 files) - Can be safely deleted
- **KEEP_LATER_\*.md** (3 files) - Reference for future

---

## �📁 Project Structure

```
MasterArbeit_MLops/
│
├── 📂 config/                      # Configuration files
│   ├── pipeline_config.yaml        # Preprocessing settings (gravity removal toggle)
│   ├── mlflow_config.yaml          # MLflow experiment settings
│   ├── requirements.txt            # Python dependencies
│   └── .pylintrc                   # Code quality settings
│
├── 📂 data/                        # Data files (tracked by DVC)
│   ├── raw/                        # Original sensor data
│   │   └── *.xlsx                  # Garmin accelerometer/gyroscope exports
│   ├── processed/                  # Preprocessed data
│   │   └── sensor_fused_50Hz.csv   # Fused & resampled sensor data
│   ├── prepared/                   # ML-ready data
│   │   ├── train_X.npy, train_y.npy
│   │   ├── val_X.npy, val_y.npy
│   │   ├── test_X.npy, test_y.npy
│   │   ├── production_X.npy        # Unlabeled production data
│   │   └── config.json             # Scaler parameters
│   ├── prepared.dvc                # DVC tracking file
│   ├── processed.dvc
│   └── raw.dvc
│
├── 📂 models/                      # Model artifacts (tracked by DVC)
│   ├── pretrained/                 # Pre-trained model
│   │   └── fine_tuned_model_1dcnnbilstm.keras
│   ├── trained/                    # New trained models
│   └── pretrained.dvc              # DVC tracking file
│
├── 📂 src/                         # Source code
│   ├── config.py                   # Path configurations
│   ├── sensor_data_pipeline.py     # Main preprocessing pipeline
│   ├── data_validator.py           # Input data validation
│   ├── mlflow_tracking.py          # MLflow integration
│   ├── run_inference.py            # Batch inference script
│   ├── evaluate_predictions.py     # Model evaluation
│   └── preprocess_data.py          # Data preparation
│
├── 📂 docker/                      # Docker configurations
│   ├── Dockerfile.training         # Training container
│   ├── Dockerfile.inference        # Inference API container
│   └── api/
│       ├── __init__.py
│       └── main.py                 # FastAPI application
│
├── 📂 notebooks/                   # Jupyter notebooks
│   └── exploration/
│       └── gravity_removal_demo.ipynb  # Gravity removal analysis
│
├── 📂 research_papers/             # Reference datasets & papers (DVC tracked)
│   ├── anxiety_dataset.csv         # Anxiety study dataset
│   └── all_users_data_labeled.csv  # Labeled training data
│
├── 📂 mlruns/                      # MLflow tracking data (git-ignored)
├── 📂 logs/                        # Application logs
├── 📂 tests/                       # Unit tests (TODO)
├── 📂 docs/                        # Documentation
│
├── 📄 docker-compose.yml           # Service orchestration
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .dockerignore                # Docker build exclusions
├── 📄 .dvcignore                   # DVC ignore rules
└── 📄 README.md                    # This file
```

---

## 📦 DVC - Data Version Control

DVC tracks large data files and models, keeping Git history clean while enabling full reproducibility.

### What's Tracked by DVC

| Directory | Contents | Size |
|-----------|----------|------|
| `data/raw/` | Original Garmin exports | ~60MB |
| `data/processed/` | Fused sensor CSVs | ~110MB |
| `data/prepared/` | Windowed .npy arrays | ~50MB |
| `models/pretrained/` | Keras model | ~18MB |
| `research_papers/*.csv` | Reference datasets | ~120MB |

### DVC Commands

```powershell
# ═══════════════════════════════════════════════════════════════
# PULLING DATA (After cloning or when data updates)
# ═══════════════════════════════════════════════════════════════

# Pull all tracked data
dvc pull

# Pull specific directory
dvc pull data/prepared.dvc

# Pull specific file
dvc pull models/pretrained.dvc


# ═══════════════════════════════════════════════════════════════
# ADDING NEW DATA (When you have new datasets)
# ═══════════════════════════════════════════════════════════════

# 1. Add new file/folder to DVC tracking
dvc add data/raw/new_dataset.csv

# 2. Push to DVC remote storage
dvc push

# 3. Commit the .dvc file to Git
git add data/raw/new_dataset.csv.dvc data/raw/.gitignore
git commit -m "Add new dataset"
git push


# ═══════════════════════════════════════════════════════════════
# CHECKING STATUS
# ═══════════════════════════════════════════════════════════════

# See what's changed
dvc status

# See what's tracked
dvc list . --dvc-only

# Check remote storage
dvc remote list


# ═══════════════════════════════════════════════════════════════
# SWITCHING BETWEEN DATA VERSIONS
# ═══════════════════════════════════════════════════════════════

# Checkout specific Git commit (data version follows)
git checkout <commit-hash>
dvc checkout

# Go back to latest
git checkout main
dvc checkout
```

### DVC Remote Storage

Currently using local storage. To switch to cloud:

```powershell
# Add Google Drive remote
dvc remote add gdrive gdrive://<folder-id>
dvc remote default gdrive

# Add S3 remote
dvc remote add s3 s3://my-bucket/dvc-storage
dvc remote default s3
```

---

## 📊 MLflow - Experiment Tracking

MLflow tracks experiments, parameters, metrics, and model artifacts.

### Starting MLflow UI

```powershell
# Option 1: Via Docker Compose (recommended)
docker-compose up -d mlflow
start http://localhost:5000

# Option 2: Standalone
mlflow ui --port 5000
```

### Using MLflow in Code

```python
from src.mlflow_tracking import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name="anxiety-activity-recognition")

# Log a training run
with tracker.start_run(run_name="training_v1") as run:
    # Log parameters
    run.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "window_size": 200,
        "gravity_removal": True
    })
    
    # Train your model...
    history = model.fit(X_train, y_train, ...)
    
    # Log training history (metrics per epoch)
    run.log_training_history(history)
    
    # Log final metrics
    run.log_metrics({
        "accuracy": 0.95,
        "f1_macro": 0.93,
        "loss": 0.12
    })
    
    # Log confusion matrix
    run.log_confusion_matrix(y_true, y_pred, class_names=ACTIVITY_CLASSES)
    
    # Log the model
    run.log_keras_model(
        model,
        artifact_path="har_model",
        registered_model_name="har-1dcnn-bilstm"
    )

# Find best run
best_run = tracker.get_best_run(metric="accuracy")
print(f"Best accuracy: {best_run['metrics.accuracy']}")
```

### MLflow CLI Commands

```powershell
# List experiments
python src/mlflow_tracking.py --list-experiments

# List runs for an experiment
python src/mlflow_tracking.py --list-runs "anxiety-activity-recognition"

# Start UI
python src/mlflow_tracking.py --ui
```

---

## 🐳 Docker - Containerization

Docker ensures reproducible environments across development, testing, and production.

### Available Images

| Image | Purpose | Port |
|-------|---------|------|
| `har-inference` | FastAPI model serving | 8000 |
| `har-training` | Model training environment | - |
| MLflow (via compose) | Experiment tracking | 5000 |

### Docker Commands

```powershell
# ═══════════════════════════════════════════════════════════════
# BUILDING IMAGES
# ═══════════════════════════════════════════════════════════════

# Build inference API image
docker build -t har-inference -f docker/Dockerfile.inference .

# Build training image
docker build -t har-training -f docker/Dockerfile.training .


# ═══════════════════════════════════════════════════════════════
# RUNNING WITH DOCKER COMPOSE (Recommended)
# ═══════════════════════════════════════════════════════════════

# Start all services (MLflow + Inference)
docker-compose up -d

# Start specific service
docker-compose up -d inference

# View logs
docker-compose logs -f inference

# Stop all
docker-compose down

# Run training (on-demand)
docker-compose --profile training run training python src/train.py


# ═══════════════════════════════════════════════════════════════
# RUNNING STANDALONE CONTAINERS
# ═══════════════════════════════════════════════════════════════

# Run inference API
docker run -d \
  --name har-api \
  -p 8000:8000 \
  -v ${PWD}/models:/app/models:ro \
  har-inference

# Run training with mounted volumes
docker run -it \
  --name har-train \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/mlruns:/app/mlruns \
  har-training \
  python src/train.py


# ═══════════════════════════════════════════════════════════════
# USEFUL COMMANDS
# ═══════════════════════════════════════════════════════════════

# Check running containers
docker ps

# Check container logs
docker logs har-inference-test

# Shell into container
docker exec -it har-inference-test /bin/bash

# Clean up
docker system prune -a
```

### Docker Compose Services

```yaml
# docker-compose.yml structure:
services:
  mlflow:      # MLflow tracking server (port 5000)
  inference:   # FastAPI model serving (port 8000)
  training:    # Training environment (on-demand, profile: training)
  preprocessing:  # Data preprocessing (on-demand, profile: preprocessing)
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion

```powershell
# Place new raw data in data/raw/
cp new_accelerometer.xlsx data/raw/
cp new_gyroscope.xlsx data/raw/

# Validate the data
python -c "
from src.data_validator import DataValidator
import pandas as pd

df = pd.read_excel('data/raw/new_accelerometer.xlsx')
validator = DataValidator()
result = validator.validate(df)
print(f'Valid: {result.is_valid}')
print(f'Errors: {result.errors}')
"
```

### Stage 2: Preprocessing

```powershell
# Run the preprocessing pipeline
python src/sensor_data_pipeline.py

# Or with Docker
docker-compose --profile preprocessing run preprocessing
```

**Pipeline Steps (sensor_data_pipeline.py):**
1. Load accelerometer & gyroscope data
2. Merge sensors on timestamp
3. Handle missing values
4. Resample to 50Hz
5. Fuse sensor streams
6. Convert units (milliG → m/s²)
7. Remove duplicate timestamps
8. Apply temporal sorting
9. Validate data quality
10. Save to `data/preprocessed/`

### Stage 3: Data Preparation

```powershell
# Create ML-ready windows
python src/preprocess_data.py
```

**Steps:**
1. Load preprocessed data
2. **Domain Calibration** (--calibrate flag)
   - Align production distribution to training distribution
   - Offset = production_mean - training_mean
   - Example: Az offset = -9.83 - (-3.53) = -6.30 m/s²
3. Apply StandardScaler normalization (with saved scaler from training)
4. Create sliding windows (200 samples, 50% overlap)
5. Save as .npy arrays to `data/prepared/`

**Recommended Flags:**
- `--calibrate`: For production data (domain adaptation)
- `--gravity-removal`: For research/analysis only (not recommended for production)

### Stage 4: Training (Optional)

```powershell
# Train with MLflow tracking
python src/train.py

# Or with Docker
docker-compose --profile training run training
```

### Stage 5: Inference

```powershell
# Batch inference
python src/run_inference.py

# API inference (start the service first)
docker-compose up -d inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"window": [[0.1, 0.2, -9.8, 0.01, 0.02, 0.03], ...]}'
```

---

## 📥 Adding New Datasets

When you receive new sensor data (e.g., new participant data), follow this workflow:

### Step-by-Step Process

```powershell
# ═══════════════════════════════════════════════════════════════
# STEP 1: Add Raw Data
# ═══════════════════════════════════════════════════════════════

# Copy new data to raw folder
cp new_participant_accelerometer.xlsx data/raw/
cp new_participant_gyroscope.xlsx data/raw/


# ═══════════════════════════════════════════════════════════════
# STEP 2: Validate Data Quality
# ═══════════════════════════════════════════════════════════════

python -c "
from src.data_validator import DataValidator
import pandas as pd

# Load and validate
df = pd.read_excel('data/raw/new_participant_accelerometer.xlsx')
validator = DataValidator()
result = validator.validate(df)

if result.is_valid:
    print('✅ Data validation passed')
    print(f'Stats: {result.stats}')
else:
    print('❌ Validation failed:')
    for error in result.errors:
        print(f'  - {error}')
"


# ═══════════════════════════════════════════════════════════════
# STEP 3: Run Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════

# Check gravity removal setting
cat config/pipeline_config.yaml | Select-String "enable_gravity_removal"

# Run preprocessing
python src/sensor_data_pipeline.py


# ═══════════════════════════════════════════════════════════════
# STEP 4: Create ML-Ready Data
# ═══════════════════════════════════════════════════════════════

python src/preprocess_data.py


# ═══════════════════════════════════════════════════════════════
# STEP 5: Version with DVC
# ═══════════════════════════════════════════════════════════════

# Update DVC tracking
dvc add data/raw data/processed data/prepared

# Push to storage
dvc push

# Commit to Git
git add data/*.dvc
git commit -m "Add new participant data (participant_id: XXX)"
git push


# ═══════════════════════════════════════════════════════════════
# STEP 6: Run Inference (if using existing model)
# ═══════════════════════════════════════════════════════════════

python src/run_inference.py --input data/prepared/production_X.npy


# ═══════════════════════════════════════════════════════════════
# STEP 7: (Optional) Retrain Model
# ═══════════════════════════════════════════════════════════════

# If you have labels, retrain with MLflow tracking
python src/train.py --experiment "new_data_v2"
```

### Data Flow Diagram for New Data

```
New Garmin Export (XLSX)
         │
         ▼
┌─────────────────┐
│ data/raw/       │  ← Place files here
│ new_acc.xlsx    │
│ new_gyro.xlsx   │
└────────┬────────┘
         │  dvc add data/raw
         ▼
┌─────────────────┐
│ Preprocessing   │  ← python src/sensor_data_pipeline.py
│ (fusion, 50Hz,  │
│  gravity removal│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ data/processed/ │  ← Intermediate result
│ sensor_fused.csv│
└────────┬────────┘
         │  python src/preprocess_data.py
         ▼
┌─────────────────┐
│ data/prepared/  │  ← ML-ready data
│ production_X.npy│
└────────┬────────┘
         │  dvc push + git commit
         ▼
┌─────────────────┐
│ DVC Storage     │  ← Versioned & tracked
│ (local/.dvc_    │
│  storage)       │
└─────────────────┘
```

---

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/model/info` | Model details |
| POST | `/predict` | Single window prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/predict/stream` | Stream of readings |

### Example Requests

```powershell
# Health Check
curl http://localhost:8000/health
# Response: {"status":"healthy","model_loaded":true,...}

# Model Info
curl http://localhost:8000/model/info
# Response: {"model_name":"1D-CNN-BiLSTM HAR","input_shape":[200,6],...}

# Single Prediction (200x6 window)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "window": [
      [0.1, 0.2, -9.8, 0.01, 0.02, 0.03],
      ... (200 rows total)
    ],
    "return_probabilities": true
  }'

# Batch Prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "windows": [
      [[...], [...], ...],
      [[...], [...], ...]
    ]
  }'
```

### Swagger UI

Interactive API documentation available at: http://localhost:8000/docs

---

## ⚙️ Configuration

### Pipeline Configuration (`config/pipeline_config.yaml`)

```yaml
preprocessing:
  # Toggle gravity removal (fixes domain shift)
  enable_gravity_removal: true
  
  # Filter parameters
  gravity_filter:
    cutoff_hz: 0.3    # High-pass cutoff frequency
    order: 3          # Butterworth filter order
  
  sampling_frequency_hz: 50

validation:
  enabled: true
  thresholds:
    max_missing_ratio: 0.05
    max_acceleration_ms2: 50.0
```

### MLflow Configuration (`config/mlflow_config.yaml`)

```yaml
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "anxiety-activity-recognition"
  
  registry:
    model_name: "har-1dcnn-bilstm"

run_defaults:
  tags:
    project: "MasterArbeit_MLops"
    model_type: "1D-CNN-BiLSTM"
```

---

## 🔧 Troubleshooting

### DVC Issues

```powershell
# "Unable to find DVC remote"
dvc remote list  # Check remotes
dvc remote add -d local_storage .dvc_storage  # Add local remote

# "Checkout failed"
dvc fetch  # Download from remote first
dvc checkout  # Then checkout

# "File already tracked by Git"
git rm -r --cached data/folder
dvc add data/folder
```

### Docker Issues

```powershell
# "Port already in use"
docker-compose down
docker stop $(docker ps -q)

# "Model not found"
# Ensure model is mounted correctly
docker run -v ${PWD}/models:/app/models:ro har-inference

# "Out of memory"
docker system prune -a  # Clean up unused images
```

### MLflow Issues

```powershell
# "Experiment not found"
python -c "import mlflow; mlflow.set_experiment('anxiety-activity-recognition')"

# "Cannot connect to tracking server"
docker-compose up -d mlflow
```

### Gravity Removal

```powershell
# Check if gravity removal is enabled
cat config/pipeline_config.yaml | Select-String "enable_gravity"

# Toggle in config
# enable_gravity_removal: true  → removes gravity
# enable_gravity_removal: false → keeps gravity
```

---

## 📈 Current Progress

| Phase | Task | Status |
|-------|------|--------|
| **Month 1** | Data ingestion & preprocessing | ✅ Complete |
| **Month 2** | Model versioning (DVC, MLflow) | ✅ Complete |
| **Month 2** | Docker containerization | ✅ Complete |
| **Month 3** | CI/CD pipeline (GitHub Actions) | ⏳ Next |
| **Month 3** | FastAPI deployment | ✅ Complete |
| **Month 4** | Monitoring & drift detection | ⏳ Planned |
| **Month 5** | Refinement & documentation | ⏳ Planned |
| **Month 6** | Thesis writing | ⏳ Planned |

---

## 📚 Key Findings

### Domain Shift Issue (Resolved)

**Problem:** Model predicted 95% "hand_tapping" on all production data.

**Root Cause:** 
- Training data: Gravity **removed** (Az ≈ -3.42 m/s²)
- Production data: Gravity **present** (Az ≈ -9.83 m/s²)

**Solution:** Butterworth high-pass filter (0.3 Hz) to remove gravity component.

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Az mean | -9.83 m/s² | ~0 m/s² |
| hand_tapping % | 95.4% | 4.2% |
| Unique classes | 4/11 | 7/11 |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Pull data: `dvc pull`
4. Make changes
5. Run tests: `pytest tests/`
6. Push data: `dvc push`
7. Commit: `git commit -m "Add new feature"`
8. Push: `git push origin feature/new-feature`
9. Open Pull Request

---

## 📄 License

This project is part of a Master's Thesis at [University Name].

---

## 📞 Contact

- **Author:** [Your Name]
- **Email:** [your.email@university.edu]
- **GitHub:** [@ShalinVachheta017](https://github.com/ShalinVachheta017)

---

**Last Updated:** December 11, 2025  
**Version:** 2.0.0

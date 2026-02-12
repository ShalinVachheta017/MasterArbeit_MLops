# 🔬 PIPELINE DEEP DIVE: End-to-End HAR MLOps System

> **Document Version:** 1.0  
> **Generated:** January 17, 2026  
> **Repository:** MA_MLops-  
> **Author:** Master Thesis MLOps Project

---

## 📋 Table of Contents

0. [Executive Summary](#0-executive-summary)
1. [Repo Map (Folder-by-Folder)](#1-repo-map-folder-by-folder)
2. [Data Flow (Start → End)](#2-data-flow-start--end)
3. [Confidence in Unlabeled Production](#3-confidence-in-unlabeled-production)
4. [Normalization vs Unit Conversion vs Gravity Removal](#4-normalization-vs-unit-conversion-vs-gravity-removal)
5. [DVC in This Repo](#5-dvc-in-this-repo)
6. [MLflow in This Repo](#6-mlflow-in-this-repo)
7. [Drift Detection + Retraining Triggers](#7-drift-detection--retraining-triggers)
8. [Logging & Exceptions (Production Readiness)](#8-logging--exceptions-production-readiness)
9. [What to Monitor WITHOUT Labels](#9-what-to-monitor-without-labels)
10. [Next Improvements Checklist](#10-next-improvements-checklist)
11. [Q&A Section](#11-qa-section)
12. [TL;DR Pipeline Execution Commands](#12-tldr-pipeline-execution-commands)
13. [Citations](#13-citations)

---

## 0. Executive Summary

This MLOps pipeline performs **Human Activity Recognition (HAR)** on wearable sensor data to detect **11 anxiety-related activities** from Garmin smartwatch accelerometer and gyroscope signals.

**What it does in 10 lines:**
1. **Ingests** raw Excel/CSV files from Garmin smartwatch (accelerometer + gyroscope)
2. **Validates** data quality (missing values, sampling rate, outliers)
3. **Fuses** sensor streams with timestamp alignment → resamples to 50Hz
4. **Converts** units (milliG → m/s²) and optionally removes gravity component
5. **Normalizes** using StandardScaler fitted on TRAINING data only (avoids leakage)
6. **Windows** data into 200-sample segments (4 seconds @ 50Hz, 50% overlap)
7. **Infers** activity using pre-trained 1D-CNN-BiLSTM model (499K parameters)
8. **Returns** predicted activity + confidence score + probability distribution
9. **Logs** everything to MLflow (experiments) and DVC (data/model versions)
10. **Enables** drift detection and retraining triggers via distribution monitoring

---

## 1. Repo Map (Folder-by-Folder)

### 📁 `config/`
| File | What it Contains | Why it Exists | When Used | Outputs |
|------|-----------------|---------------|-----------|---------|
| `pipeline_config.yaml` | Preprocessing toggles, gravity filter params, validation thresholds | Centralized configuration without code changes | Every pipeline run | N/A (read-only) |
| `mlflow_config.yaml` | MLflow tracking URI, experiment name, logging settings | Experiment tracking configuration | Training, evaluation | N/A (read-only) |
| `requirements.txt` | Python dependencies | Reproducible environments | Setup, CI/CD | N/A |

**Evidence:** [config/pipeline_config.yaml](../config/pipeline_config.yaml) lines 1-70

### 📁 `data/`
| Subfolder | What it Contains | Why it Exists | When Used | DVC Tracked? |
|-----------|-----------------|---------------|-----------|--------------|
| `raw/` | Original Excel files from Garmin | Immutable raw data source | Stage 1: Sensor fusion | ✅ `raw.dvc` |
| `processed/` | `sensor_fused_50Hz.csv` | Resampled, aligned sensor data | Stage 2: Preprocessing | ✅ `processed.dvc` |
| `prepared/` | `.npy` arrays + `config.json` (scaler params) | ML-ready windowed data | Stage 3: Inference | ✅ `prepared.dvc` |
| `prepared/predictions/` | Inference outputs (CSV, JSON, NPY) | Prediction results | After inference | ❌ (generated) |

**Key files in `data/prepared/`:**
- `production_X.npy` — Unlabeled production windows `(N, 200, 6)`
- `config.json` — Scaler mean/scale from training, activity mappings
- `baseline_stats.json` — Reference statistics for drift detection

**Evidence:** [data/prepared/config.json](../data/prepared/config.json) lines 1-72

### 📁 `src/`
| File | Purpose | Entry Point? | Key Functions |
|------|---------|--------------|---------------|
| `config.py` | Path constants, model hyperparameters | No | `PROJECT_ROOT`, `WINDOW_SIZE=200`, `OVERLAP=0.5` |
| `data_validator.py` | Validate raw data quality | No | `validate_file()`, `check_sampling_rate()` |
| `sensor_data_pipeline.py` | Raw → Fused CSV | **Yes** | `SensorDataLoader`, `SensorFusion`, `DataProcessor` |
| `preprocess_data.py` | Fused CSV → Windowed NPY | **Yes** | `UnitDetector`, `GravityRemover`, `DomainCalibrator`, `UnifiedPreprocessor` |
| `run_inference.py` | NPY → Predictions | **Yes** | `InferencePipeline`, `InferenceEngine`, `ResultsExporter` |
| `mlflow_tracking.py` | Experiment logging wrapper | No | `MLflowTracker.log_params()`, `.log_metrics()`, `.log_keras_model()` |
| `evaluate_predictions.py` | Post-inference analysis | **Yes** | Calibration analysis, confidence distributions |

**Evidence:** [src/config.py](../src/config.py) lines 1-80

### 📁 `docker/`
| File | Purpose | When Used |
|------|---------|-----------|
| `api/main.py` | FastAPI inference server | Production serving |
| `Dockerfile.inference` | Inference container image | Docker deployment |
| `Dockerfile.training` | Training container image | Model retraining |

**Endpoints defined in `docker/api/main.py`:**
- `GET /health` — Health check
- `GET /model/info` — Model metadata
- `POST /predict` — Single window prediction
- `POST /predict/batch` — Batch prediction

**Evidence:** [docker/api/main.py](../docker/api/main.py) lines 1-150

### 📁 `models/`
| Subfolder | What it Contains | DVC Tracked? |
|-----------|-----------------|--------------|
| `pretrained/` | `fine_tuned_model_1dcnnbilstm.keras`, `model_info.json` | ✅ `pretrained.dvc` |
| `trained/` | Output from new training runs | ❌ (temporary) |
| `archived_experiments/` | Old model versions | ❌ |

### 📁 `logs/`
| Subfolder | What it Contains | Retention |
|-----------|-----------------|-----------|
| `preprocessing/` | `preprocessing_*.log` files | Rotating (2MB, 5 backups) |
| `inference/` | `inference_*.log` files | Rotating |
| `training/` | Training run logs | Per-experiment |
| `evaluation/` | Evaluation logs | Per-evaluation |

### 📁 `notebooks/`
Used for exploratory data analysis and debugging. Not part of production pipeline.

### 📁 `docs/`
| Subfolder | Purpose |
|-----------|---------|
| `research/` | Paper insights (`RESEARCH_PAPER_INSIGHTS.md`) |
| `technical/` | Technical specifications |
| `thesis/` | Thesis-related documentation |

### 📁 `mlruns/`
MLflow local tracking storage. Contains experiment runs, metrics, artifacts. **Git-ignored** but crucial for reproducibility.

---

## 2. Data Flow (Start → End)

### 🔄 Pipeline Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW: RAW → PREDICTIONS                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │ Garmin Watch │
  │ Raw Export   │
  │ (.xlsx/.csv) │
  └──────┬───────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────────────────────────────────────┐
│   data/raw/        │     │ STAGE 1: Sensor Data Pipeline                   │
│ ├─ *accelerometer* │────▶│ Script: src/sensor_data_pipeline.py             │
│ └─ *gyroscope*     │     │ Actions:                                        │
└────────────────────┘     │  • Parse Excel (explode list columns)           │
        │                  │  • Create timestamps (base_time + offset)       │
        │                  │  • Merge accel+gyro (1ms tolerance)             │
        │                  │  • Resample to 50Hz (interpolate)               │
        │                  │  • Validate quality (missingness, duplicates)   │
        │                  └─────────────────────────────────────────────────┘
        │                                    │
        │                                    ▼
        │                  ┌─────────────────────────────────────────────────┐
        │                  │   data/processed/sensor_fused_50Hz.csv          │
        │                  │   Columns: timestamp_iso, Ax, Ay, Az, Gx, Gy, Gz│
        │                  └─────────────────────────────────────────────────┘
        │                                    │
        │                                    ▼
        │                  ┌─────────────────────────────────────────────────┐
        │                  │ STAGE 2: Preprocessing Pipeline                 │
        │                  │ Script: src/preprocess_data.py                  │
        │                  │ Actions:                                        │
        │                  │  • Detect units (milliG vs m/s²)                │
        │                  │  • Convert milliG → m/s² (×0.00981)             │
        │                  │  • [Optional] Remove gravity (high-pass 0.3Hz)  │
        │                  │  • [Optional] Domain calibration (mean offset)  │
        │                  │  • Normalize using TRAINING scaler (μ, σ)       │
        │                  │  • Create sliding windows (200 samples, 50% OL) │
        │                  └─────────────────────────────────────────────────┘
        │                                    │
        │                                    ▼
        │                  ┌─────────────────────────────────────────────────┐
        │                  │   data/prepared/                                │
        │                  │   ├─ production_X.npy    (N, 200, 6) float32    │
        │                  │   ├─ config.json         (scaler params)        │
        │                  │   └─ production_metadata.json                   │
        │                  └─────────────────────────────────────────────────┘
        │                                    │
        │            ┌───────────────────────┴───────────────────────┐
        │            ▼                                               ▼
        │  ┌─────────────────────┐                    ┌─────────────────────┐
        │  │ STAGE 3A: Batch     │                    │ STAGE 3B: FastAPI   │
        │  │ Inference           │                    │ Inference           │
        │  │ src/run_inference.py│                    │ docker/api/main.py  │
        │  │                     │                    │                     │
        │  │ • Load model        │                    │ • POST /predict     │
        │  │ • Load production_X │                    │ • Single window     │
        │  │ • Batch predict     │                    │ • Real-time         │
        │  │ • Export CSV/JSON   │                    │ • JSON response     │
        │  └─────────┬───────────┘                    └─────────┬───────────┘
        │            │                                          │
        │            ▼                                          ▼
        │  ┌─────────────────────────────────────────────────────────────────┐
        │  │                     OUTPUTS                                     │
        │  │  ├─ predictions_YYYYMMDD_HHMMSS.csv     (human-readable)        │
        │  │  ├─ predictions_*_metadata.json         (run metadata)          │
        │  │  ├─ predictions_*_probs.npy             (raw probabilities)     │
        │  │  └─ API JSON response:                                          │
        │  │       { "activity": "hand_tapping", "confidence": 0.94, ... }   │
        │  └─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING (Future/Planned)                              │
│  • Confidence distribution drift                                           │
│  • Feature distribution drift (KS-test, PSI)                               │
│  • Prediction class histogram drift                                        │
│  • Retraining trigger when drift threshold exceeded                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### 📊 Pipeline Stages Table

| Stage | Input | Script/Entry-Point | Output | DVC? | MLflow? |
|-------|-------|-------------------|--------|------|---------|
| **1. Sensor Fusion** | `data/raw/*.xlsx` | `src/sensor_data_pipeline.py` | `data/processed/sensor_fused_50Hz.csv` | ✅ | ❌ |
| **2. Preprocessing** | `data/processed/*.csv` | `src/preprocess_data.py` | `data/prepared/production_X.npy` + `config.json` | ✅ | ❌ |
| **3a. Batch Inference** | `data/prepared/production_X.npy` | `src/run_inference.py` | `data/prepared/predictions/*.csv` | ❌ | ✅ (optional) |
| **3b. API Inference** | JSON payload `(200, 6)` | `docker/api/main.py` (FastAPI) | JSON response | ❌ | ❌ |
| **4. Evaluation** | Predictions + ground truth | `src/evaluate_predictions.py` | Metrics, confusion matrix | ❌ | ✅ |
| **5. Training** | `data/prepared/{train,val}_*.npy` | Archived training script | `models/trained/*.keras` | ✅ | ✅ |

---

## 3. Confidence in Unlabeled Production

### 3.1 Where Confidence is Computed

**Batch Inference:** [src/run_inference.py](../src/run_inference.py) lines 435-448
```python
# Get raw probabilities from softmax layer
probabilities = self.model.predict(data, batch_size=self.config.batch_size, verbose=0)

# Get predicted class (highest probability)
predictions = np.argmax(probabilities, axis=1)

# Calculate confidences (max probability for each window)
confidences = np.max(probabilities, axis=1)
```

**FastAPI Inference:** [docker/api/main.py](../docker/api/main.py) — returns `confidence` field in `PredictionResponse`

### 3.2 How Confidence is Returned

The output CSV contains:
- `confidence`: Float 0-1 (softmax max)
- `confidence_pct`: Human-readable percentage
- `confidence_level`: Categorical (HIGH/MODERATE/LOW/UNCERTAIN)
- `is_uncertain`: Boolean flag for `conf < 0.50`
- `prob_{activity}`: Per-class probabilities (11 columns)

**Threshold Categories:** (from [run_inference.py](../src/run_inference.py) lines 520-530)
| Confidence | Level | Recommendation |
|------------|-------|----------------|
| > 90% | HIGH | Trust prediction |
| 70-90% | MODERATE | Likely correct |
| 50-70% | LOW | Review manually |
| < 50% | UNCERTAIN | Flag for review |

### 3.3 Limitations of Softmax Confidence

**Problem: Overconfidence**
Neural networks can produce high softmax probabilities even when wrong. A model might output 95% confidence but only be correct 80% of the time — this is called **miscalibration**.

**Evidence:** [run_inference.py](../src/run_inference.py) lines 374-384 explicitly documents this:
> "The softmax output gives us P(class|input), but this is NOT calibrated probability."

**Why this matters for production:**
- High confidence ≠ correctness guarantee
- Under **domain shift** (lab vs real-world), overconfidence worsens
- Manual review of low-confidence predictions is recommended

### 3.4 Better Alternatives to Top-1 Probability

| Metric | Formula | What it Measures | Requires Labels? | Recommended? |
|--------|---------|------------------|------------------|--------------|
| **Entropy** | `H(p) = -Σ pₖ log(pₖ)` | Overall uncertainty (higher = less sure) | ❌ | ✅ YES |
| **Margin** | `p_top1 - p_top2` | Ambiguity between top classes | ❌ | ✅ YES |
| **Temperature Scaling** | `p'ₖ = softmax(zₖ/T)` | Post-hoc calibration (tune T on val set) | ✅ (once) | ✅ YES |
| **Conformal Prediction Sets** | Coverage guarantee | Returns set of plausible classes | ✅ (cal set) | ⚠️ Advanced |
| **MC Dropout Variance** | Multiple forward passes | Epistemic uncertainty | ❌ | ⚠️ Slow |
| **Deep Ensembles** | Multiple models | Ensemble disagreement | ❌ | ⚠️ Expensive |
| **Energy Score** | `E = -T·log Σ exp(zₖ/T)` | OOD detection | ❌ | ✅ YES |
| **Embedding Distance** | Distance to training centroids | OOD detection | ❌ | ⚠️ Advanced |

### 3.5 Recommended Metrics for This Pipeline (Top 3)

**1. Entropy (Uncertainty)**
```python
def compute_entropy(probs):
    """Higher entropy = model is unsure"""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)
```
- **Why:** Direct measure of prediction uncertainty
- **Threshold:** `entropy > 1.5` → flag as uncertain

**2. Margin (Ambiguity)**
```python
def compute_margin(probs):
    """Small margin = ambiguous between top 2 classes"""
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]
```
- **Why:** Detects when model is torn between two activities
- **Threshold:** `margin < 0.20` → flag as ambiguous

**3. Temperature Scaling (Calibration)**
```python
# Fit T on validation set (once), then apply in production
def calibrated_probs(logits, T):
    return softmax(logits / T)
```
- **Why:** Makes confidence = actual accuracy
- **Requirement:** Fit `T` offline using `evaluate_predictions.py`

### 3.6 Proposed Abstention Policy

```python
def should_abstain(conf, entropy, margin):
    """
    Returns: (should_abstain: bool, reason: str)
    """
    if conf < 0.50:
        return True, "low_confidence"
    if entropy > 1.5:
        return True, "high_uncertainty"
    if margin < 0.20:
        return True, "ambiguous_top2"
    return False, "accepted"
```

---

## 4. Normalization vs Unit Conversion vs Gravity Removal

### 4.1 Why StandardScaler Must Fit on TRAINING ONLY

**The Problem: Data Leakage**
If you fit the scaler on production data, you're using information from the future/test set to transform the data. This leads to:
- Overly optimistic performance estimates
- Train-serve skew (model sees different distributions)
- Unreproducible results

**The Solution:** [preprocess_data.py](../src/preprocess_data.py) lines 473-510
```python
# Load scaler config from TRAINING run
config_path = DATA_PREPARED / "config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

self.scaler.mean_ = np.array(config['scaler_mean'])
self.scaler.scale_ = np.array(config['scaler_scale'])

# Transform production data using TRAINING stats
df_normalized[sensor_cols] = self.scaler.transform(df_normalized[sensor_cols])
```

**Saved Scaler Parameters:** (from [config.json](../data/prepared/config.json))
```json
"scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
"scaler_scale": [6.57, 4.35, 3.24, 49.93, 14.81, 14.17]
```

### 4.2 Can Unit Conversion Replace Normalization? **NO**

| Step | What it Does | Why Needed |
|------|--------------|------------|
| **Unit Conversion** | milliG → m/s² (×0.00981) | Physical consistency |
| **Normalization** | (x - μ_train) / σ_train | Scale alignment to model expectations |

**Unit conversion** makes units physically meaningful.
**Normalization** makes values numerically suitable for the neural network.

**Evidence:** [preprocess_data.py](../src/preprocess_data.py) lines 102-165 (`UnitDetector` class)
- Conversion factor: `CONVERSION_FACTOR = 0.00981`
- After conversion, Az ≈ -9.8 m/s² (gravity)

### 4.3 Can Gravity Removal Replace Normalization? **NO**

| Step | What it Does | Why Needed |
|------|--------------|------------|
| **Gravity Removal** | High-pass filter (0.3Hz) on accelerometer | Removes DC offset from orientation |
| **Normalization** | StandardScaler transform | Scale + variance alignment |

**Gravity removal** addresses **orientation differences** between sensors.
**Normalization** addresses **numeric scale** differences between training and inference.

**Evidence:** [preprocess_data.py](../src/preprocess_data.py) lines 280-330 (`GravityRemover` class)
```python
# Butterworth high-pass filter
nyquist = self.sampling_freq / 2
normalized_cutoff = self.cutoff_hz / nyquist
b, a = butter(self.order, normalized_cutoff, btype='high')
filtered = filtfilt(b, a, values)
```

**Configuration:** [pipeline_config.yaml](../config/pipeline_config.yaml) lines 7-16
```yaml
preprocessing:
  enable_gravity_removal: false  # Toggle
  gravity_filter:
    cutoff_hz: 0.3
    order: 3
```

### 4.4 Summary: All Three Are Different

| Operation | Purpose | Replaces Others? | Config Location |
|-----------|---------|------------------|-----------------|
| Unit Conversion | Physical units (milliG → m/s²) | ❌ | `UnitDetector` class |
| Gravity Removal | Remove DC offset | ❌ | `enable_gravity_removal` in YAML |
| Normalization | Scale to training distribution | ❌ | `config.json` scaler params |

**Order of operations in `preprocess_data.py`:**
1. Unit detection & conversion (if needed)
2. Gravity removal (if enabled)
3. Domain calibration (if enabled)
4. **Normalization (always required)**
5. Windowing

---

## 5. DVC in This Repo

### 5.1 What is Tracked

```
data/
├── raw.dvc          # Raw Excel files (64MB, 3 files)
├── processed.dvc    # sensor_fused_50Hz.csv
└── prepared.dvc     # production_X.npy, config.json, etc.

models/
└── pretrained.dvc   # fine_tuned_model_1dcnnbilstm.keras
```

**Evidence:** `.dvc` files at [data/raw.dvc](../data/raw.dvc)
```yaml
outs:
- md5: a1df11782807ac51484f9e9747bc68f2.dir
  size: 64817199
  nfiles: 3
  hash: md5
  path: raw
```

### 5.2 DVC Configuration

**Storage Location:** [.dvc/config](../.dvc/config)
```
[core]
    remote = local_storage
['remote "local_storage"']
    url = ../.dvc_storage
```

The DVC remote is a local folder `.dvc_storage/` (sibling to project root).

### 5.3 Common DVC Commands

```bash
# After cloning: fetch all data/models
dvc pull

# Check what's changed
dvc status

# Version new data
dvc add data/raw/new_file.xlsx
git add data/raw/new_file.xlsx.dvc data/raw/.gitignore
git commit -m "Add new raw data"
dvc push

# Switch to a different data version
git checkout <commit-hash>
dvc checkout

# Compare data between versions
dvc diff HEAD~1
```

### 5.4 How to Version a New Dataset

```bash
# 1. Place new files in appropriate folder
cp ~/Downloads/new_accelerometer.xlsx data/raw/

# 2. Add to DVC tracking
dvc add data/raw/

# 3. Commit pointer file to Git
git add data/raw.dvc data/raw/.gitignore
git commit -m "Add new Garmin data from week 2026-01-17"

# 4. Push data to remote storage
dvc push

# 5. Push Git changes
git push origin main
```

---

## 6. MLflow in This Repo

### 6.1 Configuration

**Location:** [config/mlflow_config.yaml](../config/mlflow_config.yaml)
```yaml
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "anxiety-activity-recognition"
  registry:
    model_name: "har-1dcnn-bilstm"
  artifact_location: "mlruns/artifacts"
```

### 6.2 Where Runs Are Created

**Training:** Uses `MLflowTracker` wrapper from [src/mlflow_tracking.py](../src/mlflow_tracking.py)
```python
tracker = MLflowTracker()
with tracker.start_run(run_name="training_v1"):
    tracker.log_params({"learning_rate": 0.001, "epochs": 50})
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.12})
    tracker.log_keras_model(model, "har_model")
```

### 6.3 What Gets Logged

| Category | Items | Code Location |
|----------|-------|---------------|
| **Parameters** | learning_rate, batch_size, epochs, window_size, stride, optimizer, dropout_rate | `log_params()` |
| **Metrics** | accuracy, loss, val_accuracy, val_loss, f1_score, precision, recall | `log_metrics()` |
| **Artifacts** | model_weights, confusion_matrix, classification_report, training_history, preprocessing_config | `log_artifact()` |
| **Models** | Keras model with signature | `log_keras_model()` |

**Evidence:** [mlflow_config.yaml](../config/mlflow_config.yaml) lines 30-55

### 6.4 How to Reproduce a Run

```bash
# 1. Start MLflow UI
mlflow ui --port 5000

# 2. Find run ID in UI or via CLI
mlflow runs list --experiment-id 0

# 3. Get run details
mlflow runs describe --run-id <run_id>

# 4. Download artifacts
mlflow artifacts download --run-id <run_id> --dst-path ./artifacts

# 5. Checkout matching data version (requires logging DVC revision)
git checkout <git_commit_from_run_tags>
dvc checkout
```

### 6.5 Finding the Best Run

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("anxiety-activity-recognition")

# Get best run by metric
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="",
    order_by=["metrics.val_accuracy DESC"],
    max_results=1
)
best_run = runs[0]
print(f"Best run: {best_run.info.run_id}, val_accuracy: {best_run.data.metrics['val_accuracy']}")
```

### 6.6 Linking DVC to MLflow (Recommended)

**Currently Missing:** DVC revision is not automatically logged to MLflow.

**Proposed Addition:**
```python
import subprocess

# In training script, before mlflow.start_run():
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
dvc_status = subprocess.check_output(['dvc', 'status']).decode().strip()

with tracker.start_run(run_name="training_v1"):
    tracker.log_params({
        "git_commit": git_commit,
        "dvc_data_hash": "from data/prepared.dvc",  # Parse this
    })
```

---

## 7. Drift Detection + Retraining Triggers

### 7.1 Types of Drift

| Drift Type | Definition | Detectable Without Labels? | Detection Method |
|------------|------------|---------------------------|------------------|
| **Covariate Drift** | P(X) changes (input distribution) | ✅ YES | KS-test, PSI on features |
| **Prior Drift** | P(Y) changes (class distribution) | ⚠️ Indirect (via predictions) | Prediction histogram |
| **Concept Drift** | P(Y\|X) changes (relationship) | ❌ NO (needs labels) | Monitor labeled samples |
| **Prediction Drift** | Model output distribution changes | ✅ YES | Prediction histogram |
| **Confidence Drift** | Confidence distribution changes | ✅ YES | Mean/std of confidence |

### 7.2 Proposed Drift Detection Methods

#### 7.2.1 Population Stability Index (PSI)

```python
def calculate_psi(reference, current, bins=10):
    """
    PSI measures distribution shift between reference and current data.
    
    Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 ≤ PSI < 0.2: Moderate shift (investigate)
    - PSI ≥ 0.2: Significant shift (retrain)
    """
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    
    # Normalize to proportions
    ref_pct = (ref_hist + 1) / (ref_hist.sum() + bins)  # Laplace smoothing
    cur_pct = (cur_hist + 1) / (cur_hist.sum() + bins)
    
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi
```

#### 7.2.2 Kolmogorov-Smirnov Test

```python
from scipy.stats import ks_2samp

def detect_feature_drift(train_data, prod_data, threshold=0.05):
    """
    KS test per feature channel.
    
    Returns: dict with p-values and drift flags
    """
    channels = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    drift_report = {}
    
    for i, ch in enumerate(channels):
        train_flat = train_data[:, :, i].flatten()
        prod_flat = prod_data[:, :, i].flatten()
        
        stat, p_value = ks_2samp(train_flat, prod_flat)
        
        drift_report[ch] = {
            'ks_statistic': stat,
            'p_value': p_value,
            'drifted': p_value < threshold
        }
    
    return drift_report
```

**Evidence:** Drift detection recommended in [RESEARCH_PAPER_INSIGHTS.md](research/RESEARCH_PAPER_INSIGHTS.md) lines 160-190

### 7.3 Concrete Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI per feature | < 0.1 | OK |
| PSI per feature | 0.1 - 0.2 | Warning, investigate |
| PSI per feature | ≥ 0.2 | Alert, consider retrain |
| KS p-value | < 0.01 (Bonferroni: 0.01/6 ≈ 0.0017) | Feature drifted |
| Drift share | > 50% features drifted | Trigger retrain |
| Confidence mean shift | > 10% from baseline | Warning |
| Entropy mean shift | > 20% from baseline | Warning |

### 7.4 Robust Trigger Rule

```python
def should_trigger_retrain(drift_report, confidence_stats, baseline_stats):
    """
    Multi-signal retraining trigger with persistence check.
    
    Returns: (trigger: bool, reason: str)
    """
    # 1. Feature drift: >50% channels have PSI > 0.2
    high_psi_count = sum(1 for ch in drift_report if drift_report[ch]['psi'] > 0.2)
    feature_drift = high_psi_count > 3  # >50% of 6 channels
    
    # 2. Confidence drift: mean shifted >10%
    conf_shift = abs(confidence_stats['mean'] - baseline_stats['confidence_mean']) 
    conf_drift = conf_shift > 0.10 * baseline_stats['confidence_mean']
    
    # 3. Entropy drift: mean increased >20%
    entropy_shift = confidence_stats['entropy_mean'] - baseline_stats['entropy_mean']
    entropy_drift = entropy_shift > 0.20 * baseline_stats['entropy_mean']
    
    # Trigger logic: feature drift OR (confidence + entropy drift)
    if feature_drift:
        return True, "feature_distribution_drift"
    if conf_drift and entropy_drift:
        return True, "prediction_uncertainty_drift"
    
    return False, "no_drift_detected"
```

### 7.5 Calibrating Thresholds

**Important:** Thresholds are data-dependent. Calibrate using a **reference window**:

1. Split historical production data into reference (e.g., week 1) and current (week 2)
2. Compute PSI/KS between them when NO drift is expected
3. Set threshold at 95th percentile of "normal" variation
4. Monitor and adjust based on false positive rate

---

## 8. Logging & Exceptions (Production Readiness)

### 8.1 Current Logging Structure

**Log Directory:** `logs/`
```
logs/
├── preprocessing/    # preprocessing_YYYYMMDD_HHMMSS.log
├── inference/        # inference_YYYYMMDD_HHMMSS.log
├── training/         # training logs
└── evaluation/       # evaluation logs
```

**Evidence:** [run_inference.py](../src/run_inference.py) lines 130-175 (`InferenceLogger` class)

**Current Format:**
```
2026-01-17 14:30:22 | INFO     | predict_batch:435 | Processing 1500 windows...
```

### 8.2 Proposed Structured Logging (JSON Lines)

```python
# src/structured_logger.py
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, log_file="logs/inference.jsonl"):
        self.log_file = log_file
    
    def log_inference_batch(self, batch_id, predictions, confidences, 
                           entropies, margins, metadata):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'inference_batch',
            'batch_id': batch_id,
            'request_id': metadata.get('request_id'),
            'git_commit': metadata.get('git_commit'),
            'dvc_revision': metadata.get('dvc_rev'),
            'model_version': metadata.get('model_ver'),
            'stats': {
                'n_windows': len(predictions),
                'avg_confidence': float(confidences.mean()),
                'avg_entropy': float(entropies.mean()),
                'avg_margin': float(margins.mean()),
                'pct_high_conf': float((confidences > 0.9).sum() / len(confidences)),
                'pct_abstained': float((confidences < 0.5).sum() / len(confidences))
            },
            'class_distribution': {
                str(k): int(v) for k, v in 
                zip(*np.unique(predictions, return_counts=True))
            }
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

### 8.3 Proposed Exception Hierarchy

```python
# src/exceptions.py

class PipelineError(Exception):
    """Base exception for all pipeline errors"""
    pass

class DataValidationError(PipelineError):
    """Raised when data fails validation checks"""
    pass

class ResamplingError(PipelineError):
    """Raised when resampling to 50Hz fails"""
    pass

class ScalerNotFoundError(PipelineError):
    """Raised when config.json with scaler params is missing"""
    pass

class ShapeMismatchError(PipelineError):
    """Raised when input shape != expected (N, 200, 6)"""
    pass

class ModelLoadError(PipelineError):
    """Raised when model file can't be loaded"""
    pass

class UnitDetectionError(PipelineError):
    """Raised when accelerometer units are ambiguous"""
    pass

class DriftDetectedError(PipelineError):
    """Raised when significant drift triggers alert"""
    pass
```

**Usage in pipeline:**
```python
try:
    data = load_data(path)
except FileNotFoundError as e:
    raise DataValidationError(f"Input file not found: {path}") from e

if data.shape[1:] != (200, 6):
    raise ShapeMismatchError(f"Expected (N, 200, 6), got {data.shape}")
```

---

## 9. What to Monitor WITHOUT Labels

### 9.1 Confidence Distribution Monitoring

| Metric | How to Compute | Alert Threshold |
|--------|----------------|-----------------|
| Mean confidence | `confidences.mean()` | Shift > 10% from baseline |
| Std confidence | `confidences.std()` | Shift > 20% from baseline |
| % high confidence (>90%) | Count/total | Significant change |
| % uncertain (<50%) | Count/total | Increase > 2x baseline |

### 9.2 Entropy Distribution Monitoring

```python
entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
```

| Metric | Alert Threshold |
|--------|-----------------|
| Mean entropy | Increase > 20% |
| Max entropy | Consistently > 2.0 |

### 9.3 Prediction Class Histogram Drift

Compare prediction distribution over time:
```python
def prediction_histogram_drift(baseline_dist, current_dist):
    """Chi-square test for prediction distribution shift"""
    from scipy.stats import chisquare
    stat, p_value = chisquare(current_dist, f_exp=baseline_dist)
    return p_value < 0.05  # Drift detected
```

### 9.4 Sensor Quality Checks

| Check | Method | Alert Condition |
|-------|--------|-----------------|
| Missingness | `df.isna().sum()` | > 5% per channel |
| Spikes | `abs(diff) > 10 * std` | > 0.1% samples |
| Variance collapse | `std < 0.01` | Any channel |
| Sampling rate | `1 / median(diff(timestamps))` | Deviation > 10% from 50Hz |

### 9.5 Temporal Consistency (Flip Rate)

```python
def compute_flip_rate(predictions):
    """
    Measure how often prediction changes between consecutive windows.
    High flip rate = model instability or noisy data.
    """
    flips = np.sum(predictions[1:] != predictions[:-1])
    return flips / (len(predictions) - 1)
```

**Normal range:** 5-15% flip rate (activities change occasionally)
**Alert:** > 30% flip rate (possible noise or drift)

### 9.6 OOD Detection via Embedding Distance

**Currently Not Implemented.** Proposed approach:

1. Extract embeddings from penultimate layer during training
2. Compute centroid per class
3. At inference, compute distance to nearest centroid
4. Flag as OOD if distance > threshold

---

## 10. Next Improvements Checklist

### 🔴 Priority 1 (Critical)

- [ ] **Add entropy + margin metrics to inference output**
  - Location: `src/run_inference.py` after line 436
  - Effort: 1 hour
  
- [ ] **Implement structured JSON logging**
  - Create: `src/structured_logger.py`
  - Effort: 2 hours

- [ ] **Add exception hierarchy**
  - Create: `src/exceptions.py`
  - Effort: 30 minutes

- [ ] **Link DVC revision to MLflow runs**
  - Location: Training script, `log_params()`
  - Effort: 1 hour

### 🟡 Priority 2 (Important)

- [ ] **Implement drift detection module**
  - Create: `src/drift_detector.py` with PSI + KS tests
  - Effort: 3 hours

- [ ] **Add temperature scaling calibration**
  - Location: `src/evaluate_predictions.py`
  - Effort: 2 hours

- [ ] **Create baseline statistics file**
  - Output: `data/prepared/baseline_stats.json`
  - Effort: 1 hour

- [ ] **Add abstention policy to inference**
  - Location: `src/run_inference.py`
  - Effort: 1 hour

### 🟢 Priority 3 (Nice to Have)

- [ ] **Implement embedding-based OOD detection**
  - Effort: 4 hours

- [ ] **Add Prometheus metrics endpoint to FastAPI**
  - Effort: 2 hours

- [ ] **Set up GitHub Actions CI/CD**
  - Effort: 3 hours

- [ ] **Add conformal prediction sets**
  - Effort: 4 hours

---

## 11. Q&A Section

### Q1: What is confidence here? How is it computed and what are better alternatives?

**Answer:**
Confidence in this pipeline is the **maximum softmax probability** from the model's output layer. It's computed as:
```python
confidences = np.max(probabilities, axis=1)
```

**Location:** [run_inference.py](../src/run_inference.py) line 436

**Limitations:**
- Softmax confidence is often **miscalibrated** (overconfident)
- High confidence doesn't guarantee correctness
- Under domain shift, overconfidence worsens

**Better alternatives:**
1. **Entropy:** `H(p) = -Σ pₖ log(pₖ)` — measures overall uncertainty
2. **Margin:** `p_top1 - p_top2` — measures ambiguity between top classes
3. **Temperature Scaling:** Post-hoc calibration using validation set
4. **Energy Score:** `-T·log Σ exp(zₖ/T)` — good for OOD detection

---

### Q2: Why do we normalize with training dataset stats? Can we skip if we convert units/remove gravity?

**Answer:**
**No, you cannot skip normalization.** Each step serves a different purpose:

| Step | Purpose |
|------|---------|
| Unit Conversion | Physical consistency (milliG → m/s²) |
| Gravity Removal | Remove DC offset from orientation |
| **Normalization** | Align numeric scale to what model learned |

The model was trained expecting inputs transformed as:
```
x_normalized = (x - μ_train) / σ_train
```

If you skip normalization in production:
- Model sees different numeric distribution
- May collapse to one class or produce garbage
- This is called **train-serve skew**

**Evidence:** Scaler params saved in [config.json](../data/prepared/config.json):
```json
"scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
"scaler_scale": [6.57, 4.35, 3.24, 49.93, 14.81, 14.17]
```

---

### Q3: How does DVC work here (what's tracked, how versions switch)?

**Answer:**
**What's tracked:**
- `data/raw.dvc` — Raw Excel files
- `data/processed.dvc` — Fused CSV
- `data/prepared.dvc` — Windowed NPY files
- `models/pretrained.dvc` — Trained model

**How Git + DVC work together:**
- Git tracks `.dvc` files (pointers with MD5 hash)
- DVC tracks actual large files in `.dvc_storage/`

**Switching versions:**
```bash
git checkout <old-commit>  # Gets old .dvc pointer
dvc checkout               # Fetches matching data from storage
```

---

### Q4: How does MLflow work here (what gets logged, where runs live)?

**Answer:**
**What gets logged:**
- Parameters: learning_rate, batch_size, epochs, etc.
- Metrics: accuracy, loss, f1_score, etc.
- Artifacts: model weights, confusion matrix, training history
- Models: Keras model with signature

**Where runs live:**
- Local: `mlruns/` directory (git-ignored)
- Configured in [mlflow_config.yaml](../config/mlflow_config.yaml)

**How to use:**
```python
from src.mlflow_tracking import MLflowTracker

tracker = MLflowTracker()
with tracker.start_run("my_experiment"):
    tracker.log_params({"lr": 0.001})
    tracker.log_metrics({"accuracy": 0.95})
```

---

### Q5: Explain inference path end-to-end (batch + FastAPI).

**Answer:**

**Batch Inference Path:**
```
data/prepared/production_X.npy
    ↓
src/run_inference.py
    ├── InferenceConfig (load settings)
    ├── ModelLoader (load .keras model)
    ├── DataLoader (load .npy, validate shape)
    ├── InferenceEngine.predict_batch()
    │   ├── model.predict(data, batch_size=32)
    │   ├── np.argmax(probs, axis=1) → predictions
    │   └── np.max(probs, axis=1) → confidences
    └── ResultsExporter
        ├── predictions_*.csv
        ├── predictions_*_metadata.json
        └── predictions_*_probs.npy
```

**FastAPI Inference Path:**
```
POST /predict
    Body: { "window": [[Ax,Ay,Az,Gx,Gy,Gz], ...] }  # 200×6
    ↓
docker/api/main.py
    ├── Validate input (Pydantic: 200 timesteps, 6 values)
    ├── Convert to numpy array
    ├── model.predict() → probabilities
    ├── argmax → predicted_class
    └── Return JSON:
        {
          "activity": "hand_tapping",
          "activity_id": 4,
          "confidence": 0.94,
          "timestamp": "2026-01-17T14:30:22",
          "model_version": "1.0.0"
        }
```

---

### Q6: What else can we measure in production without labels?

**Answer:**
1. **Confidence distribution:** mean, std, % high/low
2. **Entropy distribution:** uncertainty over time
3. **Margin distribution:** ambiguity between top-2 classes
4. **Prediction histogram:** class balance over time
5. **Feature drift:** PSI, KS-test per sensor channel
6. **Sensor QC:** missingness, spikes, variance collapse
7. **Temporal consistency:** flip rate between windows
8. **Abstention rate:** % predictions below threshold

---

### Q7: What is drift detection and what threshold should trigger retraining?

**Answer:**

**Drift Types:**
- **Covariate drift:** Input distribution P(X) changes
- **Prior drift:** Class distribution P(Y) changes
- **Concept drift:** Relationship P(Y|X) changes (needs labels)

**Detection Methods:**
- **PSI:** Population Stability Index per feature
- **KS-test:** Kolmogorov-Smirnov test per feature

**Recommended Thresholds:**
| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI | < 0.1 | OK |
| PSI | 0.1 - 0.2 | Investigate |
| PSI | ≥ 0.2 | Retrain |
| KS p-value | < 0.05/6 (Bonferroni) | Feature drifted |
| >50% features drifted | — | Trigger retrain |

**Important:** Calibrate thresholds on your data. These are starting points.

---

## 12. TL;DR Pipeline Execution Commands

```bash
# ============================================================================
# SETUP (one-time)
# ============================================================================
# Clone and setup
git clone https://github.com/ShalinVachheta017/MA_MLops-.git
cd MA_MLops-
pip install -r config/requirements.txt

# Pull data and models via DVC
dvc pull

# ============================================================================
# STAGE 1: Sensor Data Pipeline (Raw → Fused CSV)
# ============================================================================
python src/sensor_data_pipeline.py \
    --accel data/raw/accelerometer.xlsx \
    --gyro data/raw/gyroscope.xlsx \
    --output data/processed/sensor_fused_50Hz.csv

# ============================================================================
# STAGE 2: Preprocessing (Fused CSV → Windowed NPY)
# ============================================================================
python src/preprocess_data.py \
    --input data/processed/sensor_fused_50Hz.csv \
    --output data/prepared/ \
    --calibrate  # optional: apply domain calibration

# ============================================================================
# STAGE 3a: Batch Inference
# ============================================================================
python src/run_inference.py \
    --input data/prepared/production_X.npy \
    --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras \
    --output data/prepared/predictions/

# ============================================================================
# STAGE 3b: API Inference (Docker)
# ============================================================================
docker compose up inference
# Then: POST http://localhost:8000/predict with JSON body

# ============================================================================
# EVALUATION (if labels available)
# ============================================================================
python src/evaluate_predictions.py \
    --predictions data/prepared/predictions/predictions_*.csv \
    --labels data/prepared/test_y.npy

# ============================================================================
# VERSIONING
# ============================================================================
# Track new data
dvc add data/raw/
git add data/raw.dvc
git commit -m "Add new raw data"
dvc push
git push

# Track experiment
mlflow ui --port 5000  # View at http://localhost:5000
```

---

## 13. Citations

### Papers Referenced in This Document

| Paper | Location in Repo | Key Insight (≤25 words) |
|-------|-----------------|------------------------|
| **EHB_2025_71** | Previously in `papers needs to read/` | RAG-enhanced multi-stage pipeline: HAR → bout analysis → LLM report generation |
| **ICTH_16** | Previously in `papers needs to read/` | Domain adaptation addresses lab-to-life gap: 49% → 87% accuracy via fine-tuning |
| **ADAM-sense (Khan et al., 2021)** | Referenced in docs | Foundational dataset for 11 anxiety-related activities used in training |
| **MLOps: A Survey** | [RESEARCH_PAPER_INSIGHTS.md](research/RESEARCH_PAPER_INSIGHTS.md) | Drift detection (KS-test/PSI) and auto-retraining triggers recommended |
| **Deep CNN-LSTM With Self-Attention** | [RESEARCH_PAPER_INSIGHTS.md](research/RESEARCH_PAPER_INSIGHTS.md) | Self-attention improves temporal modeling in HAR |

### Code References

| Component | File | Line Numbers |
|-----------|------|--------------|
| Confidence computation | `src/run_inference.py` | 435-448 |
| Scaler loading | `src/preprocess_data.py` | 473-510 |
| Unit detection | `src/preprocess_data.py` | 100-165 |
| Gravity removal | `src/preprocess_data.py` | 280-330 |
| Domain calibration | `src/preprocess_data.py` | 333-390 |
| MLflow tracking | `src/mlflow_tracking.py` | 50-200 |
| FastAPI endpoints | `docker/api/main.py` | 150-250 |
| DVC configuration | `.dvc/config` | 1-5 |
| Pipeline config | `config/pipeline_config.yaml` | 1-70 |

---

*Document generated by analyzing the MA_MLops- repository structure, source code, and documentation.*

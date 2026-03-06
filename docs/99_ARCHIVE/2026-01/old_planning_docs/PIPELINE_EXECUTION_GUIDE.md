# HAR MLOps Pipeline Execution Guide

> **Complete guide to running the Human Activity Recognition MLOps pipeline**  
> Last Updated: January 30, 2026

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage-by-Stage Execution](#4-stage-by-stage-execution)
5. [Full Pipeline Orchestration](#5-full-pipeline-orchestration)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Retraining & Rollback](#7-retraining--rollback)
8. [Deployment](#8-deployment)
9. [Testing](#9-testing)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### 1.1 Environment Setup

```powershell
# Clone repository (if not already done)
cd "D:\study apply\ML Ops"
git clone <repository-url> MasterArbeit_MLops
cd MasterArbeit_MLops

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r config/requirements.txt
```

### 1.2 Required Dependencies

```powershell
# Core ML dependencies
pip install tensorflow==2.14.0
pip install scikit-learn==1.3.0
pip install pandas numpy

# MLOps tools
pip install mlflow==2.8.0
pip install dvc==3.30.0

# Monitoring
pip install prometheus-client

# Testing
pip install pytest pytest-cov
```

### 1.3 Verify Installation

```powershell
# Test all imports
cd "D:\study apply\ML Ops\MasterArbeit_MLops"
python -c "
import sys
sys.path.insert(0, 'src')
from config import PipelineConfig
from train import HARTrainer, TrainingConfig
from trigger_policy import TriggerPolicyEngine
from model_rollback import ModelRegistry
from ood_detection import EnsembleOODDetector
from active_learning_export import ActiveLearningExporter
from prometheus_metrics import MetricsExporter
print('✓ All modules imported successfully!')
"
```

---

## 2. Quick Start

### Run Full Pipeline (One Command)

```powershell
cd "D:\study apply\ML Ops\MasterArbeit_MLops"
python src/pipeline_orchestrator.py run-full
```

### Run Individual Stages

```powershell
# Preprocessing
python src/pipeline_orchestrator.py preprocess --input data/raw

# Training
python src/pipeline_orchestrator.py train

# Inference
python src/pipeline_orchestrator.py infer --input data/preprocessed

# Monitoring
python src/pipeline_orchestrator.py monitor

# Evaluate Retraining Trigger
python src/pipeline_orchestrator.py evaluate-trigger
```

---

## 3. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HAR MLOps Pipeline Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌───────────┐
    │   Raw    │────▶│ Preprocessing│────▶│ Training │────▶│  Model    │
    │   Data   │     │              │     │  (5-CV)  │     │ Registry  │
    └──────────┘     └──────────────┘     └──────────┘     └───────────┘
                                                                  │
         ┌────────────────────────────────────────────────────────┘
         ▼
    ┌───────────┐     ┌────────────┐     ┌─────────────┐     ┌──────────┐
    │ Inference │────▶│ Monitoring │────▶│  Trigger    │────▶│ Retrain/ │
    │           │     │ (Proxy)    │     │  Policy     │     │ Rollback │
    └───────────┘     └────────────┘     └─────────────┘     └──────────┘
         │                  │                   │
         ▼                  ▼                   ▼
    ┌───────────┐     ┌────────────┐     ┌─────────────┐
    │Predictions│     │ Prometheus │     │   Alerts    │
    │           │     │  Metrics   │     │             │
    └───────────┘     └────────────┘     └─────────────┘
```

### Pipeline Modules

| Module | File | Description |
|--------|------|-------------|
| Configuration | `src/config.py` | Central configuration management |
| Preprocessing | `src/preprocess_data.py` | Data cleaning, normalization, windowing |
| Training | `src/train.py` | 1D-CNN-BiLSTM model, 5-fold CV, MLflow |
| Inference | `src/run_inference.py` | Batch prediction generation |
| Monitoring | `src/post_inference_monitoring.py` | Proxy metrics, drift detection |
| OOD Detection | `src/ood_detection.py` | Energy-based out-of-distribution detection |
| Trigger Policy | `src/trigger_policy.py` | 2-of-3 voting retraining decision |
| Model Registry | `src/model_rollback.py` | Version management, safe rollback |
| Active Learning | `src/active_learning_export.py` | Sample selection for labeling |
| Metrics Export | `src/prometheus_metrics.py` | Prometheus metrics server |
| Deployment | `src/deployment_manager.py` | Docker deployment strategies |
| Orchestrator | `src/pipeline_orchestrator.py` | End-to-end pipeline coordination |

---

## 4. Stage-by-Stage Execution

### 4.1 Data Preprocessing

```powershell
# Option 1: Using orchestrator
python src/pipeline_orchestrator.py preprocess --input data/raw

# Option 2: Direct module execution
python -c "
import sys
sys.path.insert(0, 'src')
from preprocess_data import preprocess_pipeline
result = preprocess_pipeline(
    input_dir='data/raw',
    output_dir='data/preprocessed'
)
print(f'Preprocessed: {result}')
"

# Option 3: Using the notebook
# Open notebooks/production_preprocessing.ipynb in Jupyter
```

**Input:** `data/raw/` - Raw sensor CSV files  
**Output:** `data/preprocessed/` - Normalized, windowed data

### 4.2 Model Training

```powershell
# Option 1: Using orchestrator
python src/pipeline_orchestrator.py train

# Option 2: Direct training with custom config
python -c "
import sys
sys.path.insert(0, 'src')
from train import HARTrainer, TrainingConfig, DataLoader

# Configure training
config = TrainingConfig(
    experiment_name='har_training_run',
    n_folds=5,
    epochs=100,
    batch_size=64,
    learning_rate=0.001
)

# Load data
loader = DataLoader(config)
X, y = loader.load_from_directory('data/preprocessed')
print(f'Loaded: X={X.shape}, y={y.shape}')

# Train with cross-validation
trainer = HARTrainer(config)
results = trainer.train_with_cv(X, y)
print(f'CV F1: {results[\"cv_f1_mean\"]:.3f} ± {results[\"cv_f1_std\"]:.3f}')
"

# Option 3: Train with MLflow tracking UI
mlflow ui --port 5000
# Then run training and view at http://localhost:5000
```

**Input:** `data/preprocessed/` - Preprocessed data  
**Output:** `models/` - Trained model, `mlruns/` - MLflow artifacts

### 4.3 Model Inference

```powershell
# Option 1: Using orchestrator
python src/pipeline_orchestrator.py infer --input data/preprocessed

# Option 2: Direct inference
python -c "
import sys
sys.path.insert(0, 'src')
from run_inference import run_inference_pipeline

result = run_inference_pipeline(
    input_path='data/preprocessed',
    model_path='models/pretrained',
    output_path='outputs/predictions.csv'
)
print(f'Predictions saved: {result}')
"

# Option 3: Using Docker container
docker run -v $(pwd)/data:/data -v $(pwd)/models:/models har-inference:latest
```

**Input:** `data/preprocessed/`, `models/pretrained/`  
**Output:** `outputs/predictions.csv`

### 4.4 Post-Inference Monitoring

```powershell
# Option 1: Using orchestrator
python src/pipeline_orchestrator.py monitor

# Option 2: Direct monitoring execution
python -c "
import sys
sys.path.insert(0, 'src')
from post_inference_monitoring import MonitoringPipeline

pipeline = MonitoringPipeline()
report = pipeline.run_full_monitoring(
    predictions_path='outputs/predictions.csv'
)

print('=== MONITORING REPORT ===')
print(f'Mean Confidence: {report[\"proxy_metrics\"][\"mean_confidence\"]:.3f}')
print(f'Mean Entropy: {report[\"proxy_metrics\"][\"mean_entropy\"]:.3f}')
print(f'Flip Rate: {report[\"proxy_metrics\"][\"flip_rate\"]:.3f}')
print(f'Drift Detected: {report[\"drift_detection\"][\"drift_detected\"]}')
"
```

**Input:** `outputs/predictions.csv`  
**Output:** Monitoring report (JSON), metrics to Prometheus

### 4.5 OOD Detection

```powershell
# Run OOD detection on predictions
python -c "
import sys
import numpy as np
sys.path.insert(0, 'src')
from ood_detection import EnsembleOODDetector
import pandas as pd

# Load predictions with probabilities
predictions = pd.read_csv('outputs/predictions.csv')

# Extract probability columns (if available)
prob_cols = [c for c in predictions.columns if c.startswith('prob_')]
if prob_cols:
    probs = predictions[prob_cols].values
else:
    # Simulate from confidence
    probs = np.random.dirichlet(np.ones(11) * 5, size=len(predictions))

# Run OOD detection
detector = EnsembleOODDetector()
results = detector.detect(probs)

print('=== OOD DETECTION ===')
print(f'Total Samples: {results[\"n_samples\"]}')
print(f'OOD Samples: {results[\"n_ood\"]} ({results[\"ood_ratio\"]:.1%})')
print(f'Energy Breakdown:')
print(f'  Normal: {results[\"energy_breakdown\"][\"n_normal\"]}')
print(f'  Warning: {results[\"energy_breakdown\"][\"n_warning\"]}')
print(f'  Critical: {results[\"energy_breakdown\"][\"n_critical\"]}')
"

# Demo mode
python src/ood_detection.py --demo
```

### 4.6 Trigger Policy Evaluation

```powershell
# Option 1: Using orchestrator
python src/pipeline_orchestrator.py evaluate-trigger

# Option 2: Direct evaluation
python -c "
import sys
sys.path.insert(0, 'src')
from trigger_policy import TriggerPolicyEngine

# Create mock monitoring report (or load real one)
monitoring_report = {
    'proxy_metrics': {
        'mean_confidence': 0.72,  # Below 0.75 threshold
        'mean_entropy': 1.6,      # Above 1.5 threshold
        'flip_rate': 0.08,
        'low_confidence_ratio': 0.25
    },
    'drift_detection': {
        'psi_scores': {'acc_x': 0.12, 'acc_y': 0.08},
        'drift_detected': False
    },
    'temporal_metrics': {
        'confidence_trend': -0.02
    }
}

engine = TriggerPolicyEngine()
decision = engine.evaluate(monitoring_report)

print('=== TRIGGER EVALUATION ===')
print(f'Alert Level: {decision[\"alert_level\"]}')
print(f'Should Retrain: {decision[\"should_retrain\"]}')
print(f'Confidence Vote: {decision[\"votes\"][\"confidence\"]}')
print(f'Entropy Vote: {decision[\"votes\"][\"entropy\"]}')
print(f'Drift Vote: {decision[\"votes\"][\"drift\"]}')
if decision['reasons']:
    print(f'Reasons: {decision[\"reasons\"]}')
"
```

**Output:** Retraining decision (True/False), Alert level

---

## 5. Full Pipeline Orchestration

### 5.1 Run Complete Pipeline

```powershell
# Standard full pipeline
python src/pipeline_orchestrator.py run-full

# With specific stages
python src/pipeline_orchestrator.py run-full --stages preprocess train infer monitor

# Continue on failures
python src/pipeline_orchestrator.py run-full --continue-on-failure
```

### 5.2 Pipeline Status

```powershell
# Check pipeline status
python src/pipeline_orchestrator.py status

# Check specific run
python src/pipeline_orchestrator.py status --run-id 20260130_120000
```

### 5.3 Automated Scheduling (Cron/Task Scheduler)

**Linux/Mac Cron:**
```bash
# Run full pipeline daily at 2 AM
0 2 * * * cd /path/to/MasterArbeit_MLops && python src/pipeline_orchestrator.py run-full >> logs/cron.log 2>&1

# Run monitoring every hour
0 * * * * cd /path/to/MasterArbeit_MLops && python src/pipeline_orchestrator.py monitor >> logs/monitoring.log 2>&1
```

**Windows Task Scheduler (PowerShell):**
```powershell
# Create scheduled task for daily pipeline run
$action = New-ScheduledTaskAction -Execute "python" -Argument "src/pipeline_orchestrator.py run-full" -WorkingDirectory "D:\study apply\ML Ops\MasterArbeit_MLops"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "HAR_Pipeline_Daily" -Description "Daily HAR MLOps Pipeline"
```

---

## 6. Monitoring & Observability

### 6.1 Start Prometheus Metrics Server

```powershell
# Start metrics server (background)
python src/prometheus_metrics.py --serve --port 8000

# Or with demo data
python src/prometheus_metrics.py --demo --serve --port 8000
```

**Endpoints:**
- `http://localhost:8000/metrics` - Prometheus format
- `http://localhost:8000/metrics/json` - JSON format
- `http://localhost:8000/health` - Health check

### 6.2 Start Prometheus

```powershell
# Using Docker
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/config/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Access at http://localhost:9090
```

### 6.3 Start Grafana

```powershell
# Using Docker
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana

# Access at http://localhost:3000 (admin/admin)

# Import dashboard
# 1. Go to Dashboards > Import
# 2. Upload config/grafana/har_dashboard.json
```

### 6.4 Using Docker Compose (All Services)

```powershell
# Start all monitoring services
docker-compose up -d prometheus grafana

# View logs
docker-compose logs -f
```

### 6.5 Key Metrics to Watch

| Metric | Threshold | Action |
|--------|-----------|--------|
| `har_confidence_mean` | < 0.75 | Warning: Check for drift |
| `har_entropy_mean` | > 1.5 | Warning: Model uncertainty high |
| `har_flip_rate` | > 0.15 | Critical: Model unstable |
| `har_drift_psi` | > 0.25 | Critical: Significant drift |
| `har_ood_ratio` | > 0.20 | Warning: Many OOD samples |
| `har_trigger_state` | = 2 | Retraining triggered |

---

## 7. Retraining & Rollback

### 7.1 Manual Retraining

```powershell
# Trigger retraining manually
python -c "
import sys
sys.path.insert(0, 'src')
from train import HARTrainer, TrainingConfig, DataLoader

config = TrainingConfig(
    experiment_name='manual_retrain',
    n_folds=5,
    epochs=100
)

loader = DataLoader(config)
X, y = loader.load_from_directory('data/preprocessed')

trainer = HARTrainer(config)
results = trainer.train_with_cv(X, y)
print(f'Retraining complete: F1={results[\"cv_f1_mean\"]:.3f}')
"
```

### 7.2 Domain Adaptation Retraining (No Labels)

```powershell
python -c "
import sys
sys.path.insert(0, 'src')
from train import DomainAdaptationTrainer

trainer = DomainAdaptationTrainer()
result = trainer.retrain_with_pseudo_labels(
    old_data_path='data/preprocessed',
    new_data_path='data/new_unlabeled'
)
print(f'Domain adaptation: {result}')
"
```

### 7.3 Model Registry Operations

```powershell
# Register a new model
python src/model_rollback.py register --model-path models/new_model --metrics '{"f1": 0.91}'

# List all versions
python src/model_rollback.py list

# Set active version
python src/model_rollback.py set-active --version v20260130_120000

# Rollback to previous version
python src/model_rollback.py rollback

# Rollback to specific version
python src/model_rollback.py rollback --version v20260129_100000
```

### 7.4 Active Learning for Labeling

```powershell
# Export uncertain samples for human labeling
python src/active_learning_export.py \
  --predictions outputs/predictions.csv \
  --data data/preprocessed/features.csv \
  --top-k 100 \
  --strategy hybrid

# Import labeled data after human annotation
python src/active_learning_export.py \
  --import-labels data/active_learning/batch_20260130_120000

# Demo mode
python src/active_learning_export.py --demo
```

---

## 8. Deployment

### 8.1 Build Docker Image

```powershell
# Build inference image
python src/deployment_manager.py build --version v1.0.0

# Build with custom model path
python src/deployment_manager.py build --version v1.0.0 --model models/pretrained
```

### 8.2 Deploy Model

```powershell
# Rolling deployment (default)
python src/deployment_manager.py deploy --version v1.0.0

# Blue-green deployment
python src/deployment_manager.py deploy --version v1.0.0 --strategy blue_green

# Canary deployment (10% traffic)
python src/deployment_manager.py deploy --version v1.0.0 --strategy canary
```

### 8.3 Promote Canary

```powershell
# After validating canary, promote to full deployment
python src/deployment_manager.py promote-canary
```

### 8.4 Rollback Deployment

```powershell
# Rollback to previous version
python src/deployment_manager.py rollback

# Rollback to specific version
python src/deployment_manager.py rollback --to v0.9.0
```

### 8.5 Deployment Status

```powershell
python src/deployment_manager.py status
```

### 8.6 Using Docker Compose

```powershell
# Start full stack
docker-compose up -d

# Scale inference service
docker-compose up -d --scale inference=3

# View logs
docker-compose logs -f inference

# Stop all services
docker-compose down
```

---

## 9. Testing

### 9.1 Run All Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
start htmlcov/index.html
```

### 9.2 Run Specific Test Modules

```powershell
# Test trigger policy
pytest tests/test_trigger_policy.py -v

# Test drift detection
pytest tests/test_drift_detection.py -v

# Test OOD detection
pytest tests/test_ood_detection.py -v

# Test model rollback
pytest tests/test_model_rollback.py -v

# Test active learning
pytest tests/test_active_learning.py -v

# Test metrics
pytest tests/test_prometheus_metrics.py -v
```

### 9.3 Run Specific Test

```powershell
pytest tests/test_trigger_policy.py::TestTriggerVoting::test_two_of_three_voting -v
```

### 9.4 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs automatically on push:

1. **Lint** - Code style checks (flake8, black, isort)
2. **Test** - Unit tests with pytest
3. **Build** - Docker image build
4. **Integration** - End-to-end tests

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Import Errors
```powershell
# Ensure you're in the right directory
cd "D:\study apply\ML Ops\MasterArbeit_MLops"

# Add src to Python path
$env:PYTHONPATH = "src"

# Or use sys.path in scripts
python -c "import sys; sys.path.insert(0, 'src'); from config import PipelineConfig"
```

#### TensorFlow GPU Issues
```powershell
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU mode
$env:CUDA_VISIBLE_DEVICES = "-1"
```

#### MLflow Tracking Issues
```powershell
# Set tracking URI
$env:MLFLOW_TRACKING_URI = "file:./mlruns"

# Or programmatically
python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns')"
```

#### Docker Build Failures
```powershell
# Check Docker is running
docker info

# Clear Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t har-inference:latest -f docker/Dockerfile.inference .
```

### 10.2 Log Locations

| Component | Log Path |
|-----------|----------|
| Pipeline Orchestrator | `logs/orchestrator/` |
| Training | `logs/training/` |
| Inference | `logs/inference/` |
| Monitoring | `logs/evaluation/` |
| Preprocessing | `logs/preprocessing/` |

### 10.3 Getting Help

```powershell
# Module help
python src/pipeline_orchestrator.py --help
python src/train.py --help
python src/deployment_manager.py --help

# Run in verbose mode
python src/pipeline_orchestrator.py run-full 2>&1 | Tee-Object -FilePath debug.log
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         HAR PIPELINE QUICK REFERENCE                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  FULL PIPELINE:     python src/pipeline_orchestrator.py run-full           │
│                                                                            │
│  PREPROCESS:        python src/pipeline_orchestrator.py preprocess         │
│  TRAIN:             python src/pipeline_orchestrator.py train              │
│  INFER:             python src/pipeline_orchestrator.py infer --input X    │
│  MONITOR:           python src/pipeline_orchestrator.py monitor            │
│  TRIGGER:           python src/pipeline_orchestrator.py evaluate-trigger   │
│                                                                            │
│  METRICS SERVER:    python src/prometheus_metrics.py --serve --port 8000   │
│  ROLLBACK:          python src/model_rollback.py rollback                  │
│  DEPLOY:            python src/deployment_manager.py deploy --version X    │
│                                                                            │
│  RUN TESTS:         pytest tests/ -v                                       │
│  DOCKER UP:         docker-compose up -d                                   │
│                                                                            │
│  MONITORING URLs:                                                          │
│    Metrics:    http://localhost:8000/metrics                               │
│    Prometheus: http://localhost:9090                                       │
│    Grafana:    http://localhost:3000                                       │
│    MLflow:     http://localhost:5000                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

**Author:** HAR MLOps Pipeline  
**Version:** 1.0.0  
**License:** MIT

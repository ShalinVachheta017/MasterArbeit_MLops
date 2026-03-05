# 🔧 Repository Restructure & Pipeline Packages Plan

> **📝 Purpose:** This document outlines the complete restructuring plan to transform MA_MLops from a research-style repo into a production-like MLOps pipeline structure (offline-friendly, thesis-optimized).

> **📅 Created:** January 9, 2026 (Month 4, Week 13)

> **⏱️ Estimated Effort:** 3-4 days across Week 13-14

---

## 📊 Current State vs Target State

### Current Structure (Research-Style)
```
MA_MLops/
├─ src/                          # Scripts, not a package
│   ├─ config.py
│   ├─ data_validator.py
│   ├─ sensor_data_pipeline.py
│   ├─ preprocess_data.py
│   ├─ mlflow_tracking.py
│   ├─ run_inference.py
│   └─ evaluate_predictions.py
├─ docker/
│   ├─ api/main.py
│   ├─ Dockerfile.inference
│   └─ Dockerfile.training
├─ config/
├─ data/
├─ models/
├─ notebooks/
├─ docs/
├─ tests/
├─ *.md files at root           # Mixed docs/planning
└─ *.pdf files at root          # Research papers
```

### Target Structure (Production-Like)
```
MA_MLops/
├─ README.md
├─ pyproject.toml                  # Package definition + deps
├─ Makefile                        # One-command workflows
├─ dvc.yaml                        # Pipeline stages
├─ params.yaml                     # Pipeline parameters
├─ docker-compose.yml              # Local services orchestration
│
├─ .github/workflows/
│   ├─ ci.yml                      # Lint + tests + smoke test
│   └─ train.yml                   # Manual training trigger
│
├─ configs/
│   ├─ pipeline_config.yaml
│   ├─ mlflow_config.yaml
│   ├─ schema.yaml                 # Sensor data schema
│   └─ logging.yaml
│
├─ data/                           # DVC-tracked
│   ├─ raw/
│   ├─ interim/
│   ├─ processed/
│   └─ prepared/
│
├─ models/
│   ├─ pretrained/
│   ├─ trained/
│   └─ registry/                   # Blessed models for serving
│
├─ reports/
│   ├─ evaluation/
│   ├─ drift_reports/
│   └─ figures/
│
├─ services/
│   ├─ inference_api/
│   │   ├─ app/main.py
│   │   └─ Dockerfile
│   ├─ training_job/
│   │   ├─ entrypoint.sh
│   │   └─ Dockerfile
│   └─ monitoring/
│       ├─ prometheus.yml
│       └─ grafana/
│
├─ src/ma_mlops/                   # REAL Python package
│   ├─ __init__.py
│   ├─ pipelines/
│   │   ├─ __init__.py
│   │   ├─ train_pipeline.py
│   │   ├─ inference_pipeline.py
│   │   └─ monitoring_pipeline.py
│   ├─ stages/
│   │   ├─ __init__.py
│   │   ├─ ingest.py
│   │   ├─ validate.py
│   │   ├─ preprocess.py
│   │   ├─ train.py
│   │   ├─ evaluate.py
│   │   └─ register.py
│   ├─ monitoring/
│   │   ├─ __init__.py
│   │   ├─ drift_detection.py
│   │   ├─ data_quality.py
│   │   └─ alerts.py
│   ├─ config/
│   │   ├─ __init__.py
│   │   └─ settings.py
│   └─ utils/
│       ├─ __init__.py
│       ├─ io.py
│       ├─ logging.py
│       └─ time.py
│
├─ tests/
│   ├─ __init__.py
│   ├─ test_data_validation.py
│   ├─ test_preprocessing.py
│   └─ test_inference_smoke.py
│
├─ notebooks/                      # Exploration only
│   └─ exploration/
│
└─ docs/
    ├─ architecture.md
    ├─ pipeline_rerun_guide.md
    ├─ papers/                     # Research PDFs
    └─ planning/                   # Planning docs
```

---

## 📁 File Migration Map

### Source Code Migration

| Current Location | New Location | Notes |
|------------------|--------------|-------|
| `src/config.py` | `src/ma_mlops/config/settings.py` | Config loading + validation |
| `src/data_validator.py` | `src/ma_mlops/monitoring/data_quality.py` | Data validation checks |
| `src/sensor_data_pipeline.py` | `src/ma_mlops/stages/ingest.py` | Garmin data ingestion |
| `src/preprocess_data.py` | `src/ma_mlops/stages/preprocess.py` | Preprocessing logic |
| `src/mlflow_tracking.py` | `src/ma_mlops/stages/register.py` | Model registry helpers |
| `src/run_inference.py` | `src/ma_mlops/stages/predict.py` | Inference logic |
| `src/evaluate_predictions.py` | `src/ma_mlops/stages/evaluate.py` | Evaluation metrics |

### Docker/Services Migration

| Current Location | New Location |
|------------------|--------------|
| `docker/api/main.py` | `services/inference_api/app/main.py` |
| `docker/Dockerfile.inference` | `services/inference_api/Dockerfile` |
| `docker/Dockerfile.training` | `services/training_job/Dockerfile` |
| `docker-compose.yml` | `docker-compose.yml` (update paths) |

### Config Migration

| Current Location | New Location |
|------------------|--------------|
| `config/pipeline_config.yaml` | `configs/pipeline_config.yaml` |
| `config/mlflow_config.yaml` | `configs/mlflow_config.yaml` |
| `config/requirements.txt` | Root `pyproject.toml` or `requirements.txt` |

### Documentation Cleanup

| Current Location | New Location |
|------------------|--------------|
| Root `*.pdf` files | `docs/papers/` |
| Root planning `*.md` files | `docs/planning/` |
| `KEEP_*.md` files | `docs/planning/` |
| Keep only `README.md` at root | - |

---

## 🐍 Python Package Structure

### `pyproject.toml` (New File)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ma_mlops"
version = "0.1.0"
description = "MLOps Pipeline for Mental Health Monitoring using Wearable Sensor Data"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Shalin Vachheta"}
]

dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "tensorflow>=2.13.0",
    "scikit-learn>=1.3.0",
    "mlflow>=2.8.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "evidently>=0.4.0",  # For drift detection
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.6.0",
]

[project.scripts]
ma-train = "ma_mlops.pipelines.train_pipeline:main"
ma-infer = "ma_mlops.pipelines.inference_pipeline:main"
ma-monitor = "ma_mlops.pipelines.monitoring_pipeline:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

---

## 📦 DVC Pipeline Configuration

### `dvc.yaml` (New File)

```yaml
stages:
  ingest:
    cmd: python -m ma_mlops.stages.ingest
    deps:
      - data/raw/
      - src/ma_mlops/stages/ingest.py
    outs:
      - data/interim/sensor_merged.csv

  validate:
    cmd: python -m ma_mlops.stages.validate
    deps:
      - data/interim/sensor_merged.csv
      - src/ma_mlops/stages/validate.py
      - configs/schema.yaml
    outs:
      - reports/validation_report.json

  preprocess:
    cmd: python -m ma_mlops.stages.preprocess
    deps:
      - data/interim/sensor_merged.csv
      - src/ma_mlops/stages/preprocess.py
    params:
      - preprocessing.window_size
      - preprocessing.sample_rate
      - preprocessing.overlap
    outs:
      - data/processed/X_train.npy
      - data/processed/y_train.npy
      - data/processed/X_val.npy
      - data/processed/y_val.npy

  train:
    cmd: python -m ma_mlops.stages.train
    deps:
      - data/processed/X_train.npy
      - data/processed/y_train.npy
      - data/processed/X_val.npy
      - data/processed/y_val.npy
      - src/ma_mlops/stages/train.py
    params:
      - model.architecture
      - model.epochs
      - model.batch_size
      - model.learning_rate
    outs:
      - models/trained/model.keras
    metrics:
      - reports/evaluation/train_metrics.json:
          cache: false

  evaluate:
    cmd: python -m ma_mlops.stages.evaluate
    deps:
      - models/trained/model.keras
      - data/processed/X_val.npy
      - data/processed/y_val.npy
      - src/ma_mlops/stages/evaluate.py
    metrics:
      - reports/evaluation/eval_metrics.json:
          cache: false
    plots:
      - reports/evaluation/confusion_matrix.csv:
          x: predicted
          y: actual

  register:
    cmd: python -m ma_mlops.stages.register
    deps:
      - models/trained/model.keras
      - reports/evaluation/eval_metrics.json
    outs:
      - models/registry/production_model.keras
```

### `params.yaml` (New File)

```yaml
# Data preprocessing parameters
preprocessing:
  window_size: 128        # samples per window (2.56s at 50Hz)
  sample_rate: 50         # Hz
  overlap: 0.5            # 50% overlap
  sensors:
    - accelerometer
    - gyroscope

# Model architecture parameters
model:
  architecture: "1d_cnn_bilstm"
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10

# Monitoring thresholds
monitoring:
  drift_threshold: 0.1    # PSI threshold
  quality_min_samples: 100
  retrain_trigger: true

# MLflow configuration
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "anxiety_activity_recognition"
```

---

## 🔄 Makefile (New File)

```makefile
.PHONY: install test lint train serve monitor clean

# Install package in development mode
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=src/ma_mlops --cov-report=html

# Lint code
lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Run full DVC pipeline
train:
	dvc repro

# Run specific stage
train-only:
	dvc repro train

# Start inference API
serve:
	uvicorn services.inference_api.app.main:app --reload --port 8000

# Start all services (Docker Compose)
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Run monitoring checks
monitor:
	python -m ma_mlops.pipelines.monitoring_pipeline

# Generate reports
report:
	python -m ma_mlops.stages.evaluate --generate-report

# Clean artifacts
clean:
	rm -rf reports/evaluation/*
	rm -rf models/trained/*
	rm -rf __pycache__ .pytest_cache .mypy_cache
```

---

## 🐳 Docker Compose (Updated)

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./models:/mlflow/models
    command: >
      mlflow server 
      --backend-store-uri sqlite:///mlflow/mlruns/mlflow.db 
      --default-artifact-root /mlflow/models 
      --host 0.0.0.0
    networks:
      - mlops-network

  # Inference API
  inference_api:
    build:
      context: .
      dockerfile: services/inference_api/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models/registry:/app/models
      - ./configs:/app/configs
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    networks:
      - mlops-network
    deploy:
      replicas: 1  # Scale to 3 for demo: docker-compose up --scale inference_api=3

  # Prometheus (metrics collection)
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./services/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - mlops-network

  # Grafana (visualization)
  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    volumes:
      - ./services/monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./services/monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    networks:
      - mlops-network

networks:
  mlops-network:
    driver: bridge
```

---

## 🏗️ Pipeline Stage Templates

### `src/ma_mlops/stages/ingest.py` (Template)

```python
"""
Data Ingestion Stage
Loads raw Garmin IMU data and merges sensor streams.
"""
import logging
from pathlib import Path

import pandas as pd

from ma_mlops.config.settings import get_settings
from ma_mlops.utils.io import save_dataframe

logger = logging.getLogger(__name__)


def ingest_garmin_data(raw_dir: Path) -> pd.DataFrame:
    """Load and merge raw Garmin accelerometer/gyroscope data."""
    settings = get_settings()
    
    # Load accelerometer
    acc_path = raw_dir / "accelerometer.csv"
    acc_df = pd.read_csv(acc_path)
    logger.info(f"Loaded accelerometer: {len(acc_df)} samples")
    
    # Load gyroscope
    gyro_path = raw_dir / "gyroscope.csv"
    gyro_df = pd.read_csv(gyro_path)
    logger.info(f"Loaded gyroscope: {len(gyro_df)} samples")
    
    # Merge on timestamp
    merged = pd.merge_asof(
        acc_df.sort_values("timestamp"),
        gyro_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("20ms")  # 50Hz tolerance
    )
    
    logger.info(f"Merged dataset: {len(merged)} samples")
    return merged


def main():
    settings = get_settings()
    raw_dir = Path(settings.data.raw_dir)
    output_path = Path(settings.data.interim_dir) / "sensor_merged.csv"
    
    df = ingest_garmin_data(raw_dir)
    save_dataframe(df, output_path)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

### `src/ma_mlops/monitoring/drift_detection.py` (Template)

```python
"""
Drift Detection Module
Implements data drift and prediction drift detection for sensor data.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection results."""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    is_drift_detected: bool
    threshold: float


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for drift detection.
    
    PSI < 0.1: No significant drift
    PSI 0.1-0.25: Moderate drift (monitor)
    PSI > 0.25: Significant drift (action needed)
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    # Calculate proportions
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)
    
    # Avoid division by zero
    ref_props = np.clip(ref_props, 1e-10, 1)
    cur_props = np.clip(cur_props, 1e-10, 1)
    
    # PSI calculation
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi


def detect_sensor_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list[str],
    threshold: float = 0.1
) -> list[DriftReport]:
    """
    Detect drift in sensor features between reference and current data.
    
    Args:
        reference_data: Baseline/training data distribution
        current_data: New incoming data
        features: List of feature columns to check
        threshold: PSI threshold for drift detection
        
    Returns:
        List of DriftReport for each feature
    """
    reports = []
    
    for feature in features:
        if feature not in reference_data.columns or feature not in current_data.columns:
            logger.warning(f"Feature {feature} not found in data")
            continue
            
        ref_values = reference_data[feature].dropna().values
        cur_values = current_data[feature].dropna().values
        
        # Calculate PSI
        psi_score = calculate_psi(ref_values, cur_values)
        
        # KS test for additional validation
        ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
        
        is_drift = psi_score > threshold or ks_pvalue < 0.05
        
        report = DriftReport(
            feature_name=feature,
            psi_score=psi_score,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            is_drift_detected=is_drift,
            threshold=threshold
        )
        reports.append(report)
        
        if is_drift:
            logger.warning(f"DRIFT DETECTED in {feature}: PSI={psi_score:.4f}")
        else:
            logger.info(f"No drift in {feature}: PSI={psi_score:.4f}")
    
    return reports


def save_drift_report(reports: list[DriftReport], output_path: Path) -> None:
    """Save drift reports to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "features": [
            {
                "name": r.feature_name,
                "psi_score": r.psi_score,
                "ks_statistic": r.ks_statistic,
                "ks_pvalue": r.ks_pvalue,
                "is_drift_detected": r.is_drift_detected,
                "threshold": r.threshold
            }
            for r in reports
        ],
        "any_drift_detected": any(r.is_drift_detected for r in reports)
    }
    
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Drift report saved to {output_path}")
```

---

## ✅ Migration Checklist

### Phase 1: Package Structure (Day 1-2)
- [ ] Create `src/ma_mlops/` directory with `__init__.py`
- [ ] Create sub-packages: `stages/`, `pipelines/`, `monitoring/`, `config/`, `utils/`
- [ ] Create `pyproject.toml`
- [ ] Move existing source files to new locations
- [ ] Update all imports
- [ ] Run `pip install -e .` to verify package works

### Phase 2: DVC Pipeline (Day 2-3)
- [ ] Create `dvc.yaml` with all stages
- [ ] Create `params.yaml` with parameters
- [ ] Test `dvc repro` runs successfully
- [ ] Verify outputs are tracked correctly

### Phase 3: Services (Day 3)
- [ ] Create `services/inference_api/` structure
- [ ] Create `services/training_job/` structure
- [ ] Create `services/monitoring/` with Prometheus/Grafana configs
- [ ] Update `docker-compose.yml`
- [ ] Test `docker-compose up`

### Phase 4: CI/CD (Day 4)
- [ ] Create `.github/workflows/ci.yml`
- [ ] Add lint + test + smoke test jobs
- [ ] Create `Makefile` with standard commands
- [ ] Verify CI runs on push

### Phase 5: Cleanup (Day 4)
- [ ] Move PDFs to `docs/papers/`
- [ ] Move planning docs to `docs/planning/`
- [ ] Create `reports/` directory structure
- [ ] Update `README.md` with new structure
- [ ] Archive old `src/` files (or delete after verification)

---

## 🎯 Success Criteria

After restructuring, you should be able to:

1. **Install as package:** `pip install -e .` ✓
2. **Run full pipeline:** `dvc repro` ✓
3. **Run tests:** `make test` ✓
4. **Start services:** `docker-compose up` ✓
5. **Access MLflow:** `http://localhost:5000` ✓
6. **Access API:** `http://localhost:8000/docs` ✓
7. **Access Grafana:** `http://localhost:3000` ✓

---

## 🚫 Kubernetes Decision

**Decision: NOT using Kubernetes for this thesis.**

**Rationale:**
- Single-machine, offline deployment
- Docker Compose provides sufficient orchestration
- Can demonstrate scalability via `--scale` flag
- Lower complexity, higher thesis ROI

**If needed later:** Add `infra/k8s/` with manifests (deployment.yaml, service.yaml) as optional "future work" appendix.

---

## 📚 References

- Vehicle-Insurance-DataPipeline-MLops (modular stages pattern)
- YT-Capstone-Project (cookiecutter + DVC pipeline pattern)
- Cookiecutter Data Science template
- MLflow documentation
- DVC pipeline documentation

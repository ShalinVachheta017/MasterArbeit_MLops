# 🎯 WHAT TO DO NEXT - Actionable Roadmap

> **Last Updated:** December 17, 2025  
> **Current Completion:** 58%  
> **Time Remaining:** ~3.5 months (until April 2026)  
> **Priority:** Focus on CI/CD, Testing, and Drift Detection

---

## 📊 Current State Summary

### ✅ What We Have Completed (58%)

| Component | Status | Location | Evidence |
|-----------|--------|----------|----------|
| **Data Ingestion** | ✅ 100% | `src/sensor_data_pipeline.py` | Garmin data processing pipeline |
| **Preprocessing** | ✅ 100% | `src/preprocess_data.py` | Unit conversion, windowing, calibration |
| **Model Integration** | ✅ 95% | `models/pretrained/` | 1D-CNN-BiLSTM (499K params) |
| **MLflow Tracking** | ✅ 95% | `src/mlflow_tracking.py` | 654 lines, full experiment tracking |
| **DVC Versioning** | ✅ 100% | `.dvc/`, `data/*.dvc` | Data and model versioning |
| **Docker Containers** | ✅ 85% | `docker/` | Training + Inference Dockerfiles |
| **FastAPI Serving** | ✅ 80% | `docker/api/main.py` | `/predict` endpoint working |
| **Documentation** | ✅ 90% | `docs/`, `*.md` | 15+ markdown files |
| **Research Analysis** | ✅ 100% | `research_papers/` | 77+ papers analyzed |

### ❌ What's Missing (42%)

| Component | Status | Priority | Effort |
|-----------|--------|----------|--------|
| **CI/CD Pipeline** | ❌ 10% | 🔴 Critical | 2-3 days |
| **Unit Tests** | ❌ 0% | 🔴 Critical | 2-3 days |
| **Drift Detection** | ❌ 15% | 🔴 Critical | 3-4 days |
| **Monitoring Dashboard** | ❌ 20% | 🟡 Important | 4-5 days |
| **Retraining Triggers** | ❌ 5% | 🟡 Important | 3-4 days |
| **Thesis Writing** | ❌ 5% | 🟡 Important | 4-6 weeks |

---

## 🗓️ Recommended Action Plan

### 📅 Week 1: December 17-23, 2025 - CI/CD & Testing

#### Task 1.1: Create GitHub Actions CI/CD Pipeline
**Why:** Required for automated testing, building, and deployment per thesis plan Month 3  
**Research Support:** 
- ["MLOps: A Survey"] - CI/CD is core MLOps lifecycle component
- ["Enabling End-To-End Machine Learning Replicability"] - Docker + CI/CD for reproducibility
- ["Reproducible workflow for online AI in digital health"] - Healthcare-specific automation

**How to Implement:**

```bash
# Step 1: Create workflows directory
mkdir -p .github/workflows
```

**Create file: `.github/workflows/ci.yml`**
```yaml
name: MLOps CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pylint flake8
          pip install -r config/requirements.txt
      - name: Run linting
        run: pylint src/ --rcfile=config/.pylintrc --fail-under=7.0

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r config/requirements.txt pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Build inference image
        run: docker build -f docker/Dockerfile.inference -t har-inference:${{ github.sha }} .
      - name: Build training image
        run: docker build -f docker/Dockerfile.training -t har-training:${{ github.sha }} .
```

**Deliverables:**
- [ ] `.github/workflows/ci.yml` - Main CI pipeline
- [ ] `.github/workflows/docker.yml` - Docker build/push (optional)
- [ ] Badge in README.md showing build status

---

#### Task 1.2: Create Unit Tests
**Why:** Empty `tests/` folder - need automated testing for reliability  
**Research Support:**
- ["Toward Reusable Science with Readable Code"] - Code quality best practices
- ["DevOps-Driven Real-Time Health Analytics"] - Testing for health pipelines

**How to Implement:**

**Create file: `tests/test_preprocessing.py`**
```python
"""Unit tests for preprocessing pipeline."""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from sensor_data_pipeline import SensorDataPipeline


class TestUnitConversion:
    """Tests for accelerometer unit conversion."""
    
    def test_millig_to_ms2_conversion(self):
        """Test milliG to m/s² conversion factor."""
        milli_g = 1000  # 1G in milliG
        conversion_factor = 0.00981
        m_s2 = milli_g * conversion_factor
        assert abs(m_s2 - 9.81) < 0.01, "Conversion factor should produce ~9.81 m/s²"
    
    def test_gravity_component(self):
        """Test Az gravity component after conversion."""
        # Raw Garmin Az in milliG (pointing down)
        raw_az_millig = -1000  # -1G
        converted = raw_az_millig * 0.00981
        assert abs(converted - (-9.81)) < 0.1, "Gravity should be ~-9.81 m/s²"


class TestWindowing:
    """Tests for sliding window creation."""
    
    def test_window_shape(self):
        """Test output window dimensions."""
        data = np.random.randn(1000, 6)  # 1000 samples, 6 sensors
        window_size = 200
        overlap = 0.5
        step = int(window_size * (1 - overlap))
        
        n_windows = (len(data) - window_size) // step + 1
        windows = np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)])
        
        assert windows.shape == (n_windows, 200, 6), f"Expected shape (n, 200, 6), got {windows.shape}"
    
    def test_50_percent_overlap(self):
        """Test 50% overlap produces correct number of windows."""
        data = np.random.randn(500, 6)
        window_size = 200
        step = 100  # 50% overlap
        
        expected_windows = (500 - 200) // 100 + 1  # = 4
        assert expected_windows == 4, "500 samples with 200 window, 50% overlap = 4 windows"


class TestDomainCalibration:
    """Tests for domain calibration offset."""
    
    def test_az_offset_application(self):
        """Test -6.295 m/s² offset for Garmin Az."""
        offset = -6.295
        raw_az = -9.81  # Gravity in m/s²
        calibrated = raw_az + offset
        
        # Should match training data distribution (mean ~-3.5)
        assert -5.0 < calibrated < -2.0, f"Calibrated Az should be around -3.5, got {calibrated}"


class TestSensorFusion:
    """Tests for accelerometer + gyroscope fusion."""
    
    def test_six_channel_output(self):
        """Test that output has 6 channels (Ax, Ay, Az, Gx, Gy, Gz)."""
        acc_data = np.random.randn(100, 3)  # 3 accelerometer channels
        gyro_data = np.random.randn(100, 3)  # 3 gyroscope channels
        
        fused = np.hstack([acc_data, gyro_data])
        assert fused.shape == (100, 6), "Fused data should have 6 channels"
```

**Create file: `tests/test_inference.py`**
```python
"""Unit tests for inference pipeline."""
import pytest
import numpy as np


class TestModelInput:
    """Tests for model input validation."""
    
    def test_input_shape(self):
        """Test model expects (batch, 200, 6) input."""
        expected_shape = (1, 200, 6)
        test_input = np.random.randn(*expected_shape)
        assert test_input.shape == expected_shape
    
    def test_batch_inference(self):
        """Test batch input shape."""
        batch_size = 32
        expected_shape = (batch_size, 200, 6)
        test_input = np.random.randn(*expected_shape)
        assert test_input.shape == expected_shape


class TestModelOutput:
    """Tests for model output validation."""
    
    def test_output_classes(self):
        """Test model outputs 11 activity classes."""
        n_classes = 11
        mock_output = np.random.rand(1, n_classes)
        assert mock_output.shape[1] == 11, "Model should output 11 classes"
    
    def test_softmax_probabilities(self):
        """Test output probabilities sum to 1."""
        mock_probs = np.array([0.1, 0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05])
        assert abs(mock_probs.sum() - 1.0) < 0.01, "Probabilities should sum to 1"


class TestActivityLabels:
    """Tests for activity label mapping."""
    
    def test_activity_count(self):
        """Test 11 anxiety-related activities."""
        activities = [
            "forehead_rubbing", "nail_biting", "talking", "yawning",
            "shivering", "stretching", "hair_pulling", "scratching",
            "normal_hand_use", "fidgeting", "face_touching"
        ]
        assert len(activities) == 11, "Should have 11 activity classes"
```

**Create file: `tests/conftest.py`**
```python
"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    return np.random.randn(1000, 6)


@pytest.fixture
def sample_window():
    """Generate a single 200-timestep window."""
    np.random.seed(42)
    return np.random.randn(1, 200, 6)


@pytest.fixture
def activity_labels():
    """Return list of 11 activity classes."""
    return [
        "forehead_rubbing", "nail_biting", "talking", "yawning",
        "shivering", "stretching", "hair_pulling", "scratching",
        "normal_hand_use", "fidgeting", "face_touching"
    ]
```

**Run tests:**
```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

**Deliverables:**
- [ ] `tests/test_preprocessing.py` - Preprocessing tests
- [ ] `tests/test_inference.py` - Inference tests
- [ ] `tests/conftest.py` - Pytest fixtures
- [ ] `tests/test_data_validator.py` - Validation tests (optional)

---

### 📅 Week 2: December 24-30, 2025 - Drift Detection

#### Task 2.1: Implement Data Drift Detection
**Why:** Critical for monitoring model performance in production (Month 4 requirement)  
**Research Support:**
- ["Domain Adaptation for IMU-based HAR: A Survey"] - 40%+ accuracy drop due to domain shift
- ["ICTH_16"] - Lab-to-life gap demonstration (49% → 87% accuracy)
- ["Are Anxiety Detection Models Generalizable"] - Cross-device generalization challenges
- ["Resilience of ML Models in Anxiety Detection"] - Sensor noise impact

**How to Implement:**

**Update file: `src/data_validator.py`** (add these functions)
```python
"""
Data Drift Detection Module
Based on: Domain Adaptation for IMU-based HAR (Survey)
         ICTH_16 - Lab-to-life gap analysis
"""
from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class DriftDetector:
    """
    Detect distribution drift between reference (training) and production data.
    
    Uses Kolmogorov-Smirnov test and Wasserstein distance for drift detection.
    Research basis: "Domain Adaptation for IMU-based HAR: A Survey"
    """
    
    SENSOR_NAMES = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    def __init__(self, reference_stats_path: str = None):
        """
        Initialize drift detector with reference statistics.
        
        Args:
            reference_stats_path: Path to JSON file with training data statistics
        """
        self.reference_stats = None
        if reference_stats_path:
            self.load_reference_stats(reference_stats_path)
    
    def load_reference_stats(self, path: str):
        """Load reference statistics from training data."""
        with open(path, 'r') as f:
            self.reference_stats = json.load(f)
    
    def compute_reference_stats(self, reference_data: np.ndarray) -> dict:
        """
        Compute and save reference statistics from training data.
        
        Args:
            reference_data: Training data array (n_windows, 200, 6)
            
        Returns:
            dict: Statistics per sensor channel
        """
        stats = {}
        for idx, sensor in enumerate(self.SENSOR_NAMES):
            channel_data = reference_data[:, :, idx].flatten()
            stats[sensor] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'percentile_25': float(np.percentile(channel_data, 25)),
                'percentile_75': float(np.percentile(channel_data, 75)),
                'sample_size': len(channel_data)
            }
        
        self.reference_stats = stats
        return stats
    
    def detect_drift(self, production_data: np.ndarray, 
                     ks_threshold: float = 0.05,
                     wasserstein_threshold: float = 0.5) -> dict:
        """
        Detect data drift using statistical tests.
        
        Based on: "Are Anxiety Detection Models Generalizable" - drift detection methods
        
        Args:
            production_data: New production data (n_windows, 200, 6)
            ks_threshold: P-value threshold for KS test (default 0.05)
            wasserstein_threshold: Distance threshold for Wasserstein (default 0.5)
            
        Returns:
            dict: Drift detection results per sensor
        """
        if self.reference_stats is None:
            raise ValueError("Reference statistics not loaded. Call compute_reference_stats first.")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'sensors': {},
            'summary': {}
        }
        
        drift_count = 0
        
        for idx, sensor in enumerate(self.SENSOR_NAMES):
            prod_data = production_data[:, :, idx].flatten()
            
            # Generate reference sample from stored statistics
            ref_mean = self.reference_stats[sensor]['mean']
            ref_std = self.reference_stats[sensor]['std']
            ref_sample = np.random.normal(ref_mean, ref_std, len(prod_data))
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(ref_sample, prod_data)
            
            # Wasserstein distance (Earth Mover's Distance)
            w_distance = wasserstein_distance(ref_sample, prod_data)
            
            # Determine drift
            ks_drift = ks_pvalue < ks_threshold
            w_drift = w_distance > wasserstein_threshold
            
            sensor_drift = ks_drift or w_drift
            if sensor_drift:
                drift_count += 1
            
            results['sensors'][sensor] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'ks_drift_detected': ks_drift,
                'wasserstein_distance': float(w_distance),
                'wasserstein_drift_detected': w_drift,
                'overall_sensor_drift': sensor_drift,
                'production_mean': float(np.mean(prod_data)),
                'production_std': float(np.std(prod_data)),
                'reference_mean': ref_mean,
                'reference_std': ref_std
            }
        
        # Overall drift: if 2+ sensors show drift
        results['overall_drift_detected'] = drift_count >= 2
        results['summary'] = {
            'total_sensors': len(self.SENSOR_NAMES),
            'sensors_with_drift': drift_count,
            'drift_percentage': (drift_count / len(self.SENSOR_NAMES)) * 100
        }
        
        return results
    
    def save_drift_report(self, results: dict, output_path: str):
        """Save drift detection results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_file


def detect_prediction_drift(reference_predictions: np.ndarray,
                           production_predictions: np.ndarray,
                           threshold: float = 0.1) -> dict:
    """
    Detect prediction distribution drift.
    
    Based on: "MLOps: A Survey" - prediction monitoring best practices
    
    Args:
        reference_predictions: Historical prediction distribution
        production_predictions: Recent prediction distribution
        threshold: Threshold for distribution change
        
    Returns:
        dict: Prediction drift results
    """
    # Class distribution comparison
    ref_dist = np.bincount(reference_predictions, minlength=11) / len(reference_predictions)
    prod_dist = np.bincount(production_predictions, minlength=11) / len(production_predictions)
    
    # Jensen-Shannon divergence approximation
    distribution_diff = np.abs(ref_dist - prod_dist).mean()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'distribution_difference': float(distribution_diff),
        'drift_detected': distribution_diff > threshold,
        'reference_distribution': ref_dist.tolist(),
        'production_distribution': prod_dist.tolist()
    }
```

**Create file: `src/monitor_drift.py`** (standalone monitoring script)
```python
#!/usr/bin/env python3
"""
Drift Monitoring Script
Run periodically to check for data and prediction drift.

Usage:
    python src/monitor_drift.py --data data/production/latest.csv --reference data/prepared/reference_stats.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from data_validator import DriftDetector, detect_prediction_drift


def main():
    parser = argparse.ArgumentParser(description='Monitor for data and prediction drift')
    parser.add_argument('--data', required=True, help='Path to production data CSV')
    parser.add_argument('--reference', required=True, help='Path to reference statistics JSON')
    parser.add_argument('--output', default='logs/drift/', help='Output directory for reports')
    parser.add_argument('--alert-threshold', type=float, default=0.05, help='KS test threshold')
    
    args = parser.parse_args()
    
    # Load production data
    print(f"Loading production data from {args.data}...")
    prod_df = pd.read_csv(args.data)
    
    # Reshape to windows (assuming preprocessed format)
    # Adjust based on your actual data format
    sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    prod_data = prod_df[sensor_cols].values
    
    # Create windows
    window_size = 200
    step = 100
    windows = []
    for i in range(0, len(prod_data) - window_size + 1, step):
        windows.append(prod_data[i:i+window_size])
    prod_windows = np.array(windows)
    
    print(f"Created {len(windows)} windows from production data")
    
    # Initialize drift detector
    detector = DriftDetector(args.reference)
    
    # Detect drift
    print("Running drift detection...")
    results = detector.detect_drift(prod_windows, ks_threshold=args.alert_threshold)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output) / f'drift_report_{timestamp}.json'
    detector.save_drift_report(results, str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("DRIFT DETECTION REPORT")
    print("="*60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Drift Detected: {'⚠️ YES' if results['overall_drift_detected'] else '✅ NO'}")
    print(f"Sensors with Drift: {results['summary']['sensors_with_drift']}/{results['summary']['total_sensors']}")
    print()
    
    for sensor, data in results['sensors'].items():
        status = "⚠️ DRIFT" if data['overall_sensor_drift'] else "✅ OK"
        print(f"  {sensor}: {status}")
        print(f"    KS p-value: {data['ks_pvalue']:.4f}")
        print(f"    Wasserstein: {data['wasserstein_distance']:.4f}")
    
    print(f"\nReport saved to: {output_path}")
    
    # Return exit code based on drift detection
    return 1 if results['overall_drift_detected'] else 0


if __name__ == '__main__':
    sys.exit(main())
```

**Deliverables:**
- [ ] Update `src/data_validator.py` with `DriftDetector` class
- [ ] Create `src/monitor_drift.py` monitoring script
- [ ] Create `data/prepared/reference_stats.json` from training data
- [ ] Add drift monitoring to CI/CD pipeline (optional)

---

### 📅 Week 3-4: January 1-14, 2026 - Monitoring & Evaluation

#### Task 3.1: Create Monitoring Dashboard
**Why:** Month 4 requirement - visualize model performance and drift  
**Research Support:**
- ["Designing a Clinician-Centered Wearable Data Dashboard (CarePortal)"] - Dashboard design
- ["DevOps-Driven Real-Time Health Analytics"] - Real-time monitoring

**Option A: Simple Matplotlib Dashboard** (Recommended for thesis)
```python
# src/generate_monitoring_report.py
import matplotlib.pyplot as plt
import json
from pathlib import Path

def generate_report(drift_results: dict, predictions_log: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sensor drift status
    # Plot 2: Prediction distribution
    # Plot 3: Accuracy over time
    # Plot 4: Confidence distribution
    
    plt.savefig('outputs/monitoring_report.png')
```

**Option B: Streamlit Dashboard** (More interactive)
```python
# dashboard/app.py
import streamlit as st
import pandas as pd

st.title("HAR Pipeline Monitoring Dashboard")
# Add interactive visualizations
```

---

#### Task 3.2: Implement Model Evaluation Improvements
**Why:** Current evaluation is basic, need robust metrics  
**Research Support:**
- ["ICTH_16"] - 5-fold stratified cross-validation methodology
- ["A Close Look into Human Activity Recognition Models"] - Comprehensive evaluation

**Add to `src/evaluate_predictions.py`:**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

def cross_validate(model, X, y, n_splits=5):
    """5-fold stratified cross-validation (per ICTH_16 methodology)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics = {'fold': [], 'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        y_pred = model.predict(X_val).argmax(axis=1)
        
        metrics['fold'].append(fold)
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['f1_macro'].append(f1_score(y_val, y_pred, average='macro'))
        metrics['f1_weighted'].append(f1_score(y_val, y_pred, average='weighted'))
    
    return pd.DataFrame(metrics)
```

---

### 📅 Week 5-8: January 15 - February 14, 2026 - Refinement

#### Task 4.1: Implement Retraining Trigger (Prototype)
**Why:** Month 5 requirement - automated retraining based on drift  
**Research Support:**
- ["MLOps: A Survey"] - Retraining strategies
- ["MLHOps: Machine Learning for Healthcare Operations"] - Healthcare-specific retraining

```python
# src/retraining_trigger.py
def check_retraining_needed(drift_report: dict, performance_metrics: dict) -> bool:
    """
    Determine if model retraining should be triggered.
    
    Triggers retraining if:
    1. Data drift detected in 3+ sensors
    2. F1 score drops below 0.80
    3. Prediction distribution shift > 15%
    """
    triggers = []
    
    # Check data drift
    if drift_report['summary']['sensors_with_drift'] >= 3:
        triggers.append('data_drift')
    
    # Check performance
    if performance_metrics.get('f1_macro', 1.0) < 0.80:
        triggers.append('performance_degradation')
    
    return len(triggers) > 0, triggers
```

---

#### Task 4.2: Data Augmentation (Optional Enhancement)
**Why:** Improve model robustness  
**Research Support:**
- ["Deep learning for sensor-based activity recognition: A survey"] - Standard augmentation
- ["Masked Video and Body-worn IMU Autoencoder"] - Advanced augmentation

```python
# src/data_augmentation.py
def augment_window(window, jitter=0.05, scale_range=(0.9, 1.1)):
    """Apply augmentation to a sensor window."""
    augmented = window.copy()
    
    # Jittering
    noise = np.random.normal(0, jitter, window.shape)
    augmented += noise
    
    # Scaling
    scale = np.random.uniform(*scale_range)
    augmented *= scale
    
    return augmented
```

---

### 📅 Week 9-16: February 15 - April 2026 - Thesis Writing

#### Task 5.1: Thesis Document Structure
**Why:** Month 6 requirement  

**Recommended Thesis Outline:**
```
1. Introduction
   - Problem Statement: Mental health monitoring using wearables
   - Research Objectives
   - Thesis Structure

2. Literature Review (from research_papers/)
   - HAR & Deep Learning
   - Mental Health Monitoring
   - MLOps Best Practices
   - Domain Adaptation

3. Methodology
   - Data Pipeline (sensor_data_pipeline.py)
   - Model Architecture (1D-CNN-BiLSTM)
   - MLOps Infrastructure (MLflow, DVC, Docker)

4. Implementation
   - System Architecture
   - CI/CD Pipeline
   - Drift Detection
   - API Serving

5. Results & Evaluation
   - Model Performance
   - Domain Adaptation Results
   - MLOps Metrics

6. Discussion & Future Work
   - Limitations
   - Prognosis Model Integration
   - RAG-Enhanced Reports (from EHB_2025_71)

7. Conclusion
```

---

## 📊 Priority Matrix

```
                    HIGH IMPACT
                        │
    ┌───────────────────┼───────────────────┐
    │   CI/CD Pipeline  │  Drift Detection  │
    │   Unit Tests      │  Monitoring       │
    │                   │                   │
LOW ├───────────────────┼───────────────────┤ HIGH
EFFORT                  │                   │  EFFORT
    │   README Updates  │  Self-Attention   │
    │   Code Cleanup    │  RAG Reports      │
    │                   │  Prognosis Model  │
    └───────────────────┼───────────────────┘
                        │
                    LOW IMPACT
```

---

## 📋 Immediate Checklist (This Week)

### Day 1 (Today - Dec 17)
- [ ] Create `.github/workflows/` directory
- [ ] Write `ci.yml` workflow file
- [ ] Push and verify GitHub Actions runs

### Day 2 (Dec 18)
- [ ] Create `tests/conftest.py` with fixtures
- [ ] Write `tests/test_preprocessing.py`
- [ ] Run `pytest tests/ -v` locally

### Day 3 (Dec 19)
- [ ] Write `tests/test_inference.py`
- [ ] Add pytest-cov for coverage
- [ ] Update CI to run tests

### Day 4-5 (Dec 20-21)
- [ ] Start `DriftDetector` class in `data_validator.py`
- [ ] Create reference statistics from training data
- [ ] Test drift detection locally

### Weekend (Dec 22-23)
- [ ] Review and refine
- [ ] Update documentation
- [ ] Commit all changes

---

## 🔬 Research Papers to Reference

For each implementation task, refer to these papers:

| Task | Primary Papers |
|------|----------------|
| CI/CD | "MLOps: A Survey", "Enabling End-To-End ML Replicability" |
| Testing | "Toward Reusable Science with Readable Code", "DevOps-Driven Real-Time Health Analytics" |
| Drift Detection | "Domain Adaptation for IMU-based HAR", "ICTH_16", "Are Anxiety Models Generalizable" |
| Monitoring | "CarePortal Dashboard Design", "MLHOps for Healthcare" |
| Evaluation | "ICTH_16" (5-fold CV), "A Close Look into HAR Models" |
| Augmentation | "Deep learning for HAR Survey", "Masked IMU Autoencoder" |
| Self-Attention | "Deep CNN-LSTM With Self-Attention", "CNNs/RNNs/Transformers Survey" |

---

## 📈 Expected Thesis Completion After Tasks

| After Completing | New Completion % |
|------------------|------------------|
| CI/CD + Tests | 65% |
| Drift Detection | 72% |
| Monitoring | 78% |
| Retraining Trigger | 85% |
| Thesis Writing | 100% |

---

*Document Generated: December 17, 2025*  
*Next Review: December 24, 2025*  
*Full Paper References: `research_papers/COMPREHENSIVE_RESEARCH_PAPERS_SUMMARY.md`*

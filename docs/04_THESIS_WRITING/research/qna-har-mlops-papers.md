# HAR MLOps Pipeline: Complete Q&A Reference Guide

**Document Version:** 1.0  
**Generated:** January 28, 2026  
**Thesis Focus:** Wearable IMU-based Human Activity Recognition for Anxiety Detection with MLOps  
**Sensor Configuration:** AX, AY, AZ, GX, GY, GZ (6-channel window-based)

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Time Alignment & ISO Timestamps](#2-time-alignment--iso-timestamps)
3. [Sensor Fusion Per Dataset](#3-sensor-fusion-per-dataset)
4. [Naming Convention & Ingestion Rules](#4-naming-convention--ingestion-rules)
5. [Preprocessing Picks Newest Fused Output](#5-preprocessing-picks-newest-fused-output)
6. [Processing-State Tracking](#6-processing-state-tracking)
7. [Production vs Training Preprocessing](#7-production-vs-training-preprocessing)
8. [Data QC & Validation](#8-data-qc--validation)
9. [Baseline & Reference Datasets](#9-baseline--reference-datasets)
10. [Gravity Removal Across Production Datasets](#10-gravity-removal-across-production-datasets)
11. [MLflow Logging Plan](#11-mlflow-logging-plan)
12. [Separate Confidence Monitoring Component](#12-separate-confidence-monitoring-component)
13. [Smoke Tests](#13-smoke-tests)
14. [Drift Thresholds (PSI Ranges)](#14-drift-thresholds-psi-ranges)
15. [ABCD Cases & Dominant-Hand Detection](#15-abcd-cases--dominant-hand-detection)
16. [Uncertainty Quantification & OOD Detection](#16-uncertainty-quantification--ood-detection)
17. [Operating Without Labels](#17-operating-without-labels)
18. [Retraining When Labels Never Exist](#18-retraining-when-labels-never-exist)
19. [Audit Meaning & Traceability](#19-audit-meaning--traceability)
20. [Prometheus/Grafana Integration Plan](#20-prometheusgrafana-integration-plan)

---

## Pipeline ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         HAR MLOPS PIPELINE - END-TO-END VIEW                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  RAW DATA   │    │   PREPROCESS    │    │   INFERENCE     │    │   MONITORING    │  │
│  │   LAYER     │───▶│     LAYER       │───▶│     LAYER       │───▶│     LAYER       │  │
│  ├─────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤  │
│  │• Garmin FIT │    │• Resample 50Hz  │    │• 1DCNN-BiLSTM   │    │• Confidence     │  │
│  │• Accel+Gyro │    │• Fuse sensors   │    │• Window (200,6) │    │• Drift metrics  │  │
│  │• Timestamp  │    │• Normalize      │    │• Activity pred  │    │• Temporal       │  │
│  │  alignment  │    │• Gravity filter │    │• Probabilities  │    │  consistency    │  │
│  └─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│         │                   │                      │                      │            │
│         ▼                   ▼                      ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              MLOPS INFRASTRUCTURE                                │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │  • DVC (Data Versioning)              • Docker (Containerization)               │   │
│  │  • MLflow (Experiment Tracking)       • FastAPI (Model Serving)                 │   │
│  │  • GitHub Actions (CI/CD)             • Prometheus/Grafana (Metrics)            │   │
│  │  • Audit Logs & Model Registry        • Baseline Stats (Drift Reference)        │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Pipeline Overview

### What This Pipeline Does

This pipeline transforms raw 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) from wearable devices into activity predictions for anxiety-related behavior detection. It follows MLOps best practices for reproducibility, monitoring, and continuous improvement.

### Key Takeaway

The pipeline implements a **two-stage domain adaptation** approach: pre-train on public datasets (ADAMSense), then fine-tune on target device data (Garmin Venu 3).

### Evidence

**[Recognition of Anxiety-Related Activities using 1DCNN-BiLSTM on Sensor Data from a Commercial Wearable Device, Oleh & Obermaisser, 2025 | PDF: papers/papers needs to read/ICTH_16.pdf]**

- Models trained in controlled environments fail in production without domain adaptation (89% → 49%)
- Fine-tuning with small custom dataset recovers accuracy (49% → 87%)
- Window size: 200 timesteps at 50Hz (4 seconds, 50% overlap)

**[MACHINE LEARNING OPERATIONS: A SURVEY ON MLOPS, Hewage & Meedeniya, 2022 | PDF: papers/mlops_production/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf]**

- MLOps enables reproducibility through data versioning, experiment tracking, and containerization
- Continuous monitoring is essential for detecting model degradation in production

---

## 2. Time Alignment & ISO Timestamps

### Question

How do we handle time alignment after accelerometer/gyroscope shifting, and ensure ISO timestamp format?

### Answer

The pipeline performs the following steps:

1. **Parse timestamps** from raw sensor files (Excel/CSV) into pandas datetime
2. **Convert to ISO 8601 format** with timezone awareness when available
3. **Align accelerometer and gyroscope streams** using nearest-neighbor interpolation within tolerance (±1ms)
4. **Resample to uniform 50Hz** (20ms intervals) using mean aggregation

### What to Do Checklist

- [ ] Ensure raw files have consistent timestamp format
- [ ] Apply timezone normalization (convert to UTC internally)
- [ ] Use tolerance-based alignment (±1ms for 50Hz target)
- [ ] Log alignment quality metrics (# matched, # interpolated, # dropped)
- [ ] Store fusion metadata in JSON alongside fused CSV

### Evidence

**[A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare, 2023 | PDF: papers/papers needs to read/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf]**

- Timestamp synchronization is critical for multi-sensor fusion
- Recommends explicit logging of alignment parameters for reproducibility

**[Combining Accelerometer and Gyroscope Data in Smartphone-Based Activity Recognition | PDF: papers/research_papers/76 papers/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf]**

- Multi-sensor fusion improves HAR accuracy over single-sensor approaches
- Temporal alignment within 10ms is sufficient for 50Hz data

---

## 3. Sensor Fusion Per Dataset

### Question

How do we fuse accelerometer and gyroscope data while maintaining schema consistency and batch identity?

### Answer

Each dataset batch produces a fused output with:

- **Schema:** `timestamp, Ax, Ay, Az, Gx, Gy, Gz` (6 sensor columns)
- **Batch identity:** Filename includes source date and processing timestamp
- **Metadata file:** JSON sidecar with fusion parameters and source file hashes

### Fusion Process

```
RAW ACCEL (.xlsx)  ─┐
                    ├──▶  sensor_fused_50Hz.csv + sensor_fused_meta.json
RAW GYRO  (.xlsx)  ─┘
```

### What to Do Checklist

- [ ] Validate column names match expected schema before fusion
- [ ] Compute and store file hashes for provenance tracking
- [ ] Generate metadata JSON with: source files, target_hz, tolerance_ms, rows_native, rows_resampled
- [ ] Use deterministic filename pattern: `sensor_fused_{target_hz}Hz_{batch_id}.csv`

### Evidence

**[Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition | PDF: papers/research_papers/76 papers/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf]**

- Proposes FusionActNet architecture using both sensors
- 6-feature approach (3-axis accel + 3-axis gyro) is standard practice

**[An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors | PDF: papers/research_papers/76 papers/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf]**

- End-to-end pipelines should include explicit data versioning
- Batch identity tracking prevents mixing data from different collection sessions

---

## 4. Naming Convention & Ingestion Rules

### Question

What naming conventions and ingestion renaming rules should we use for CI/automation-friendly data management?

### Answer

Use hierarchical naming with ISO date prefix for automatic chronological sorting:

```
PATTERN: {YYYY-MM-DD}_{source}_{datatype}_{version}.{ext}

EXAMPLES:
  2025-03-23_garmin_accelerometer_raw.xlsx
  2025-03-23_garmin_gyroscope_raw.xlsx
  2025-03-23_garmin_fused_50Hz_v1.csv
  2026-01-15_production_windows_v2.npy
```

### Ingestion Renaming Rules

1. **Strip spaces** → replace with underscores
2. **Lowercase** all filenames
3. **Add ISO date prefix** if missing (from file modification time)
4. **Version suffix** for reprocessed files (v1, v2, etc.)

### What to Do Checklist

- [ ] Create `scripts/rename_ingested_files.py` for CI automation
- [ ] Validate filename pattern before ingestion
- [ ] Log original → renamed mapping in manifest
- [ ] Reject files that don't match expected naming after transformation

### Evidence

**[MLDEV: DATA SCIENCE EXPERIMENT AUTOMATION AND REPRODUCIBILITY | PDF: papers/research_papers/76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf]**

- Consistent naming conventions are essential for automated pipelines
- Recommends ISO date prefixes for natural sorting

**[Toward Reusable Science with Readable Code | PDF: papers/research_papers/76 papers/Toward Reusable Science with Readable Code and.pdf]**

- Clear file naming supports reproducibility and collaboration
- Version suffixes prevent overwriting without explicit intent

---

## 5. Preprocessing Picks Newest Fused Output

### Question

How does preprocessing always pick the newest fused output (manifest/versioning)?

### Answer

The pipeline uses a **manifest-based approach**:

1. **Manifest file** (`data/prepared/manifest.json`) tracks all processed datasets
2. **Automatic newest selection** scans for highest version or latest timestamp
3. **Explicit override** available via command-line flag for reproducibility

![Figure 1: Dataset Timeline & Newest Fused File Logic](figures/fig1_dataset_timeline.png)

*Figure 1 shows the chronological ordering of datasets and how the manifest selects the newest fused file for preprocessing. The arrow indicates the selection logic flow.*

### Manifest Structure

```json
{
  "latest_fused": "2025-03-23_garmin_fused_50Hz_v1.csv",
  "datasets": [
    {"name": "...", "created": "2025-03-23T15:23:10", "hash": "abc123..."},
    {"name": "...", "created": "2026-01-15T12:58:26", "hash": "def456..."}
  ]
}
```

### What to Do Checklist

- [ ] Create manifest.json on first data ingestion
- [ ] Update manifest after each preprocessing run
- [ ] Log which dataset was selected and why
- [ ] Provide `--use-specific` flag to override newest selection

### Evidence

**[Enabling End-To-End Machine Learning Replicability | PDF: papers/research_papers/76 papers/Enabling End-To-End Machine Learning.pdf]**

- Manifest files enable reproducibility by recording exact data versions used
- Automatic selection with override capability balances convenience and control

---

## 6. Processing-State Tracking

### Question

How do we track processing state to avoid reprocessing production batches?

### Answer

Each batch has a **processing state** tracked in the manifest:

| State | Description |
|-------|-------------|
| `raw` | Ingested but not preprocessed |
| `fused` | Accelerometer + gyroscope merged |
| `prepared` | Normalized, windowed, ready for inference |
| `predicted` | Inference complete, results stored |
| `monitored` | Post-inference monitoring complete |

### State Transition Rules

- Only transition forward (no skipping states)
- Store transition timestamp and operator/script name
- Reprocessing requires explicit `--force` flag

### What to Do Checklist

- [ ] Add `processing_state` field to manifest entries
- [ ] Log state transitions with timestamps
- [ ] Implement `--force` flag for intentional reprocessing
- [ ] CI pipeline should skip already-processed batches

### Evidence

**[Reproducible workflow for online AI in digital health | PDF: papers/research_papers/76 papers/Reproducible workflow for online AI in digital health.pdf]**

- State tracking prevents redundant computation and ensures consistency
- Explicit force flags protect against accidental reprocessing

---

## 7. Production vs Training Preprocessing

### Question

What are the differences between production preprocessing (unlabeled) and training preprocessing (labeled)?

### Answer

| Aspect | Training Preprocessing | Production Preprocessing |
|--------|------------------------|-------------------------|
| **Labels** | Present in data | Not available |
| **Output** | `X_train.npy`, `y_train.npy` | `production_X.npy` only |
| **Split** | Train/val/test stratified | No splitting needed |
| **Augmentation** | Optional (noise, rotation) | Never applied |
| **Baseline stats** | Computed and saved | Loaded from training |
| **Normalization** | Fit StandardScaler | Transform only (use saved scaler) |

### Key Principle

**The scaler fitted during training MUST be reused for production data.** This is the most common source of distribution mismatch.

### What to Do Checklist

- [ ] Save `scaler.pkl` during training preprocessing
- [ ] Load same scaler for production preprocessing
- [ ] Never fit a new scaler on production data
- [ ] Log scaler source file hash in production metadata

### Evidence

**[Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey | PDF: papers/domain_adaptation/Domain Adaptation for Inertial Measurement Unit-based Human.pdf]**

- Normalization mismatch is a primary cause of domain shift
- Training statistics must be frozen for production inference

**[Recognition of Anxiety-Related Activities using 1DCNN-BiLSTM, 2025 | PDF: papers/papers needs to read/ICTH_16.pdf]**

- Uses StandardScaler fitted on training data for all inference
- Window size (200 samples) and overlap (50%) must match training exactly

---

## 8. Data QC & Validation

### Question

What must be checked in data QC & validation, including time-gap, sensor range, and fusion checks?

### Answer

The QC pipeline performs three layers of validation:

### Layer 1: Raw Data Checks

- **Timestamp monotonicity** — no backward jumps
- **Time gaps** — flag if delta > 2× expected (40ms for 50Hz)
- **Sensor range** — accelerometer typically ±16g, gyroscope ±2000°/s
- **NaN/Inf detection** — reject if >1% missing

### Layer 2: Fusion Checks

- **Row count match** — accel and gyro should have similar counts after resampling
- **Column schema** — exactly 7 columns (timestamp + 6 sensors)
- **Dtype validation** — timestamp as datetime, sensors as float32

### Layer 3: Prepared Data Checks

- **Window shape** — must be (N, 200, 6)
- **Normalized mean** — should be ≈0 per channel
- **Normalized std** — should be ≈1 per channel
- **Variance collapse detection** — flag if std < 0.1

![Figure 2: Sampling Rate / Time-Gap QC](figures/fig2_sampling_rate_qc.png)

*Figure 2 shows (A) the distribution of time deltas between samples, with expected 20ms at 50Hz, and (B) detection of gaps per window for quality control.*

### What to Do Checklist

- [ ] Run `scripts/preprocess_qc.py` on every new dataset
- [ ] Fail pipeline if critical checks fail (shape, dtype)
- [ ] Warn but continue for moderate issues (small time gaps)
- [ ] Store QC report JSON in `reports/preprocess_qc/`

### Evidence

**[Comparative Study on the Effects of Noise in HAR | PDF: papers/papers needs to read/Comparative Study on the Effects of Noise in.pdf]**

- Sensor noise significantly impacts HAR model performance
- QC checks for range violations detect sensor failures early

**[Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets | PDF: papers/research_papers/76 papers/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf]**

- Schema validation prevents downstream errors
- Automated QC enables scaling to many datasets

---

## 9. Baseline & Reference Datasets

### Question

What is the role of baseline/reference datasets (frozen training baseline + rolling production reference)?

### Answer

We maintain **two types of reference datasets**:

### Frozen Training Baseline

- **Purpose:** Ground truth distribution for drift detection
- **Contents:** Per-channel statistics (mean, std, percentiles, histogram)
- **Update policy:** Only changes when model is retrained
- **Storage:** `data/prepared/baseline_stats.json`

### Rolling Production Reference

- **Purpose:** Track recent production data trends
- **Contents:** Last N batches of production statistics
- **Update policy:** Rolling window, updated after each inference batch
- **Storage:** `reports/monitoring/rolling_reference.json`

### Usage

| Comparison | Detects |
|------------|---------|
| Production vs Training Baseline | Domain shift from training distribution |
| Production vs Rolling Reference | Sudden changes in recent production |

### What to Do Checklist

- [ ] Generate baseline stats during training data preparation
- [ ] Never modify baseline stats without explicit retraining decision
- [ ] Update rolling reference after each monitored batch
- [ ] Alert if production differs significantly from both references

### Evidence

**[From Development to Deployment: An Approach to MLOps Monitoring for Machine Learning Model Operationalization, 2023 | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**

- Baseline datasets are essential for drift detection
- Rolling references capture natural data evolution

---

## 10. Gravity Removal Across Production Datasets

### Question

When and why do we apply gravity removal, and what should we compare across production datasets?

### Answer

### When to Apply

- **Training data with gravity removed:** If the training dataset (e.g., ADAMSense) had gravity removed, production data MUST also have it removed
- **Detection method:** Check if Az mean is near 0 (gravity removed) or near 9.81 m/s² (gravity present)

### How to Apply

Use a **Butterworth high-pass filter** with cutoff at 0.3 Hz:
- Gravity is a constant (DC) component at ~9.81 m/s²
- Human movement is above 0.3 Hz
- Filter removes low-frequency gravity while preserving motion

### What to Compare

| Metric | Before Gravity Removal | After Gravity Removal |
|--------|------------------------|-----------------------|
| Az mean | ~9.81 m/s² | ~0 m/s² |
| Az std | Similar | Similar |
| Magnitude | High baseline | Lower baseline |

![Figure 3: Gravity Removal Impact](figures/fig3_gravity_removal_impact.png)

*Figure 3 shows the before/after distributions for (A) X-axis acceleration, (B) Z-axis with gravity bias, and (C) total acceleration magnitude. Note the shift in Az from ~9.81 to ~0 after filtering.*

### What to Do Checklist

- [ ] Check training data gravity status (inspect Az mean)
- [ ] Apply matching preprocessing to production data
- [ ] Log gravity removal decision in preprocessing metadata
- [ ] Compare Az distributions before/after in QC report

### Evidence

**[Deep learning for sensor-based activity recognition: A survey | PDF: papers/research_papers/76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf]**

- Gravity removal is common preprocessing for accelerometer-based HAR
- Mismatch between training and inference preprocessing causes model failure

**[Recognition of Anxiety-Related Activities using 1DCNN-BiLSTM, 2025 | PDF: papers/papers needs to read/ICTH_16.pdf]**

- ADAMSense dataset uses gravity-removed accelerometer data
- Consumer devices (Garmin) report raw accelerometer with gravity

---

## 11. MLflow Logging Plan

### Question

What is the MLflow logging plan for proxy metrics, drift metrics, and distinguishing training vs monitoring runs?

### Answer

### Run Types

| Run Type | Experiment Name | Purpose |
|----------|-----------------|---------|
| Training | `har_training` | Model training, hyperparameter search |
| Inference | `har_inference` | Production predictions |
| Monitoring | `har_monitoring` | Drift and confidence metrics |

### Metrics to Log

**Training Runs:**
- `accuracy`, `f1_macro`, `loss`
- Per-class precision/recall
- Confusion matrix artifact

**Inference Runs:**
- `n_windows`, `inference_time_sec`
- `mean_confidence`, `uncertain_ratio`
- Predictions CSV artifact

**Monitoring Runs:**
- `drift/ks_max`, `drift/wasserstein_max`
- `confidence/mean`, `confidence/uncertain_ratio`
- `temporal/flip_rate`, `temporal/mean_dwell_time`
- Drift and confidence reports as artifacts

### What to Do Checklist

- [ ] Set `MLFLOW_TRACKING_URI` in environment
- [ ] Use nested runs for multi-stage pipelines
- [ ] Tag runs with `run_type`, `batch_id`, `model_version`
- [ ] Store artifacts (reports, predictions) in run context

### Evidence

**[MLOps: A Step Forward to Enterprise Machine Learning, 2023 | PDF: papers/mlops_production/MLOps A Step Forward to Enterprise Machine Learning 2023.pdf]**

- MLflow provides unified experiment tracking across training and inference
- Tagging enables filtering and comparison across run types

**[Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools, 2021 | PDF: papers/mlops_production/Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf]**

- MLflow + DVC combination is recommended for comprehensive MLOps
- Artifacts should include both metrics and raw data for audit

---

## 12. Separate Confidence Monitoring Component

### Question

Do we need a separate confidence monitoring component? What is the architecture argument?

### Answer

**Yes, a separate component is recommended** for the following reasons:

### Architecture Arguments

1. **Separation of concerns** — Inference should be fast; monitoring can be asynchronous
2. **Different update cadence** — Inference runs per-batch; monitoring aggregates across batches
3. **Alert system integration** — Monitoring feeds into alerting (Prometheus/Grafana)
4. **Independent scaling** — Monitoring compute can scale separately from inference

### Proposed Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Inference     │────▶│  Predictions Queue   │────▶│   Monitoring    │
│   (Fast)        │     │  (async buffer)      │     │   (Batch)       │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  Prometheus     │
                                                     │  (Alerts)       │
                                                     └─────────────────┘
```

### What to Do Checklist

- [ ] Create `scripts/post_inference_monitoring.py` (already exists)
- [ ] Run monitoring as separate process after inference completes
- [ ] Export metrics to Prometheus format
- [ ] Configure alert thresholds in monitoring config

### Evidence

**[MLHOps: Machine Learning for Healthcare Operations | PDF: papers/research_papers/76 papers/MLHOps Machine Learning for Healthcare Operations.pdf]**

- Healthcare ML requires robust monitoring separate from inference
- Asynchronous monitoring prevents inference latency impact

---

## 13. Smoke Tests

### Question

What smoke tests should we implement for data, model loading, and logging?

### Answer

### Data Smoke Test

```
CHECK: Can we load the expected data files?
  ✓ production_X.npy exists and loads
  ✓ Shape is (N, 200, 6)
  ✓ Dtype is float32
  ✓ No NaN/Inf values
```

### Model Load Smoke Test

```
CHECK: Can we load the model and make predictions?
  ✓ Model file exists at expected path
  ✓ Model loads without errors
  ✓ Input shape matches (None, 200, 6)
  ✓ Output shape is (None, 11)
  ✓ Dummy prediction succeeds
```

### Logging Smoke Test

```
CHECK: Is logging infrastructure working?
  ✓ MLflow tracking URI is reachable
  ✓ Can create a test run
  ✓ Can log metrics and artifacts
  ✓ Log files are written to expected directory
```

### What to Do Checklist

- [ ] Run smoke tests in CI before deployment
- [ ] Run smoke tests on startup in production
- [ ] Fail fast with clear error messages
- [ ] Store smoke test results in `reports/inference_smoke/`

### Evidence

**[Practical MLOps: Operationalizing Machine Learning Models | PDF: papers/mlops_production/Practical-mlops-operationalizing-machine-learning-models.pdf]**

- Smoke tests are essential for production deployment
- Fast failure prevents silent errors in production

---

## 14. Drift Thresholds (PSI Ranges)

### Question

What PSI ranges indicate drift, and are these thresholds supported by local papers?

### Answer

### Standard PSI Interpretation

| PSI Range | Interpretation | Action |
|-----------|----------------|--------|
| PSI < 0.10 | No/minimal shift | PASS — continue normal operation |
| 0.10 ≤ PSI < 0.25 | Moderate shift | WARN — investigate, monitor closely |
| PSI ≥ 0.25 | Major shift | ACTION — likely drift, consider retraining |

### Additional Drift Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| KS Statistic | > 0.2 | Significant distribution difference |
| Wasserstein Distance | > 0.5 | Large effect size |
| Normalized Mean Shift | > 0.5σ | Mean has shifted substantially |

![Figure 5: Drift Metrics Overview](figures/fig5_drift_metrics_overview.png)

*Figure 5 shows drift metrics (KS statistic, Wasserstein distance, mean shift) per sensor channel. Red bars indicate channels exceeding thresholds, suggesting drift.*

### Evidence

**[From Development to Deployment: An Approach to MLOps Monitoring, 2023 | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**

- PSI thresholds (0.10, 0.25) are industry standard from credit risk modeling
- Applied successfully to ML model monitoring

**[Resilience-aware MLOps for AI-based medical diagnostic system, 2024 | PDF: papers/mlops_production/Resilience-aware MLOps for AI-based medical diagnostic system  2024.pdf]**

- Healthcare ML requires conservative drift thresholds
- Multiple metrics (KS, PSI, Wasserstein) provide robust detection

**Evidence missing in local papers — needs citation:**
- Specific PSI thresholds for wearable HAR applications not found in local papers
- Thresholds above are adapted from general MLOps literature

---

## 15. ABCD Cases & Dominant-Hand Detection

### Question

How do we handle ABCD cases (dominant/non-dominant hand wearing) and detect low-observability situations?

### Answer

### The Four Cases

| Case | Watch Wrist | Activity Hand | Signal Quality |
|------|-------------|---------------|----------------|
| A | Dominant | Dominant | **BEST** — strong motion signal |
| B | Non-dominant | Dominant | **WEAKEST** — watch misses activity |
| C | Dominant | Non-dominant | GOOD — decent signal |
| D | Non-dominant | Non-dominant | MODERATE — reasonable signal |

### Detection Heuristics

**Low observability indicators (Case B pattern):**
- Low motion variance (< 0.5 of training baseline)
- Low mean confidence (< 0.70)
- High flip rate (> 0.40)
- High idle activity percentage (> 60%)

### Adaptive Thresholds

When low observability is detected, use **relaxed thresholds**:

| Metric | Normal Threshold | Relaxed Threshold |
|--------|------------------|-------------------|
| Confidence | 0.50 | 0.35 |
| Entropy | 2.0 | 2.5 |
| Flip rate | 0.30 | 0.45 |

![Figure 7: ABCD Cases Comparison](figures/fig7_abcd_cases_comparison.png)

*Figure 7 shows signal variance and SNR distributions for the four wrist-activity configurations. Case B (non-dominant watch, dominant activity) shows significantly lower signal quality.*

### What to Do Checklist

- [ ] Compute motion variance during first N windows
- [ ] Compare to training baseline variance
- [ ] If variance < 50% baseline, flag low observability
- [ ] Apply relaxed thresholds for the session
- [ ] Log observability status in monitoring report

### Evidence

**[Are Anxiety Detection Models Generalizable? A Cross-Activity and Cross-Population Study Using Wearables | PDF: papers/anxiety_detection/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf]**

- Cross-activity generalization is a major challenge
- Wrist placement affects signal observability significantly

**[Resilience of Machine Learning Models in Anxiety Detection: Assessing the Impact of Gaussian Noise on Wearable Sensors | PDF: papers/papers needs to read/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf]**

- Model performance degrades with reduced signal quality
- Adaptive thresholds can compensate for low-observability scenarios

---

## 16. Uncertainty Quantification & OOD Detection

### Question

How do we implement uncertainty quantification, OOD detection, temporal consistency, and conformal prediction for no-label monitoring?

### Answer

### Uncertainty Quantification Methods

**Proxy metrics (no retraining required):**
- **Max probability (confidence):** Lower values indicate uncertainty
- **Entropy:** Higher values indicate uncertain predictions
- **Margin:** Difference between top-2 class probabilities

**Advanced methods (requires model changes):**
- **MC Dropout:** Enable dropout at inference, average predictions
- **Deep ensembles:** Train multiple models, measure disagreement

### OOD Detection

**Approaches for detecting out-of-distribution inputs:**

1. **Feature-space distance:** Compare input embedding to training cluster centers
2. **Energy-based detection:** Compute energy score from logits
3. **Reconstruction error:** Use autoencoder, high error = OOD

### Temporal Consistency

**Checks for plausible activity sequences:**
- **Flip rate:** Proportion of windows where prediction changes
- **Dwell time:** Duration of each activity bout
- **Transition matrix:** Compare to expected transitions from training

### Conformal Prediction

**Provides calibrated prediction sets:**
- Instead of single prediction, output set of plausible classes
- Set size indicates uncertainty (larger = more uncertain)
- Requires calibration set (small labeled sample)

![Figure 4: Proxy Metrics Distributions](figures/fig4_proxy_metrics_distributions.png)

*Figure 4 shows confidence, entropy, and margin distributions with paper-backed thresholds. The shaded regions indicate uncertain/ambiguous zones.*

### What to Do Checklist

- [ ] Compute confidence, entropy, margin for every prediction
- [ ] Flag windows below confidence threshold (0.50)
- [ ] Track flip rate and dwell times for temporal consistency
- [ ] Consider conformal prediction if calibration data available

### Evidence

**[A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models | PDF: papers/research_papers/76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf]**

- Two-stage anomaly detection separates point and contextual anomalies
- Combining methods improves OOD detection

**[NeurIPS 2020: Energy-based Out-of-Distribution Detection | PDF: papers/new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf]**

- Energy scores from neural network logits detect OOD effectively
- Simple to implement, requires no model changes

**[NeurIPS 2021: Adaptive Conformal Inference Under Distribution Shift | PDF: papers/new paper/NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf]**

- Conformal prediction provides calibrated uncertainty estimates
- Adaptive methods handle distribution shift without retraining

---

## 17. Operating Without Labels

### Question

How do we operate the pipeline when labels are not available (monitor → alert → sample → optional label acquisition → retrain)?

### Answer

### The No-Label Monitoring Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NO-LABEL MONITORING LOOP                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │ Monitor │───▶│  Alert  │───▶│ Sample  │───▶│ Optional Label  │  │
│  │ Proxy   │    │ if      │    │ Low-    │    │ Acquisition     │  │
│  │ Metrics │    │ Degraded│    │ Conf    │    │                 │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────────────┘  │
│       │                                              │              │
│       │              ┌──────────────────────────────▶│              │
│       │              │                               ▼              │
│       │         ┌────┴────┐                   ┌─────────────────┐  │
│       └────────▶│ Continue│                   │ Retrain if      │  │
│                 │ Normal  │                   │ Labels Available│  │
│                 │ Operation                   └─────────────────┘  │
│                 └─────────┘                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Alert Conditions

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| Mean confidence | < 0.75 | < 0.60 |
| Uncertain ratio | > 0.15 | > 0.30 |
| Drift (any channel) | KS > 0.15 | KS > 0.25 |
| Flip rate | > 0.35 | > 0.50 |

### Sampling Strategy for Label Acquisition

1. **Uncertainty sampling:** Select windows with lowest confidence
2. **Diversity sampling:** Select windows from different predicted classes
3. **Boundary sampling:** Select windows near decision boundaries (low margin)

### What to Do Checklist

- [ ] Monitor proxy metrics continuously
- [ ] Configure alert thresholds in config file
- [ ] Export low-confidence windows for optional review
- [ ] Track alert history for trend analysis
- [ ] If labels acquired, fine-tune model and update baseline

### Evidence

**[Passive Sensing for Mental Health Monitoring Using Machine Learning With Wearables and Smartphones | PDF: papers/papers needs to read/Passive Sensing for Mental Health Monitoring Using Machine.pdf]**

- Passive monitoring is the norm for mental health applications
- Active label acquisition should be minimally burdensome

**[Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years | PDF: papers/papers needs to read/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf]**

- Long-term monitoring without continuous labeling is essential
- Model updates should be triggered by detected degradation

---

## 18. Retraining When Labels Never Exist

### Question

How do we safely adapt/retrain when ground-truth labels are never available?

### Answer

### Safe Adaptation Strategies

1. **Self-training with confidence gating:**
   - Use high-confidence predictions as pseudo-labels
   - Only include predictions with confidence > 0.90
   - Retrain on pseudo-labeled data + original training data

2. **Consistency regularization:**
   - Apply augmentations to production data
   - Enforce consistent predictions across augmentations
   - No explicit labels required

3. **Feature alignment:**
   - Align production feature distributions to training
   - Use domain adaptation techniques (MMD, adversarial)
   - Preserves class boundaries without labels

### Safeguards

| Safeguard | Purpose |
|-----------|---------|
| Confidence threshold | Only use highly confident pseudo-labels |
| Holdout validation | Reserve some original labeled data for validation |
| Rollback capability | Keep previous model version for comparison |
| Gradual deployment | A/B test new model before full replacement |
| Performance bounds | Reject update if validation accuracy drops > 5% |

### What to Do Checklist

- [ ] Implement pseudo-labeling pipeline with confidence gate
- [ ] Always validate on held-out labeled data before deployment
- [ ] Version all models for rollback
- [ ] Monitor new model closely after deployment
- [ ] Document adaptation rationale in MLflow

### Evidence

**[Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey | PDF: papers/domain_adaptation/Domain Adaptation for Inertial Measurement Unit-based Human.pdf]**

- Self-training is effective for HAR domain adaptation
- Confidence gating reduces error propagation

**[Transfer Learning in Human Activity Recognition: A Survey | PDF: papers/domain_adaptation/Transfer Learning in Human Activity Recognition  A Survey.pdf]**

- Feature alignment preserves learned representations
- Gradual adaptation prevents catastrophic forgetting

**[Self-supervised learning for fast and scalable time series hyper-parameter tuning | PDF: papers/research_papers/76 papers/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf]**

- Self-supervised methods reduce labeling requirements
- Can be combined with supervised fine-tuning when labels available

---

## 19. Audit Meaning & Traceability

### Question

What does "audit" mean in this context, and what must we store for traceability and governance?

### Answer

### Audit Definition

An **audit trail** ensures that any prediction or model decision can be traced back to:
- The exact input data used
- The model version and parameters
- The preprocessing steps applied
- The timestamp and operator

### What to Store

| Category | Items | Storage Location |
|----------|-------|------------------|
| **Data Provenance** | Source file hashes, preprocessing params, QC results | `data/prepared/*.json` |
| **Model Versioning** | Model weights, architecture, training hyperparams | MLflow Model Registry |
| **Prediction Logs** | Input hash, prediction, confidence, timestamp | `data/prepared/predictions/` |
| **Monitoring Reports** | Drift metrics, alerts, thresholds used | `reports/monitoring/` |
| **Configuration** | Pipeline config, feature flags, thresholds | `config/*.yaml`, Git |

### Governance Requirements

1. **Immutability:** Once logged, records should not be modified
2. **Completeness:** Every inference should be traceable
3. **Accessibility:** Authorized users can retrieve any historical state
4. **Retention:** Define retention policy (e.g., 2 years for healthcare)

### What to Do Checklist

- [ ] Generate unique batch IDs for each inference run
- [ ] Hash input data and store in prediction metadata
- [ ] Log model version/run ID with every prediction
- [ ] Store all configs in version control
- [ ] Implement retention policy for old logs

### Evidence

**[The Role of MLOps in Healthcare: Enhancing Predictive Analytics and Patient Outcomes | PDF: papers/mlops_production/The Role of MLOps in Healthcare Enhancing Predictive Analytics and Patient Outcomes.pdf]**

- Healthcare ML requires comprehensive audit trails
- Regulatory compliance (HIPAA, GDPR) mandates traceability

**[Roadmap for a Scalable MLOps Pipeline in Mental Health Monitoring | PDF: papers/mlops_production/Roadmap for a Scalable MLOps Pipeline in Mental Health Monitoring (Master's Thesis).pdf]**

- Mental health applications require special attention to data governance
- Audit logs support clinical accountability

---

## 20. Prometheus/Grafana Integration Plan

### Question

What metrics should we export to Prometheus/Grafana, and how does it complement MLflow?

### Answer

### Metrics to Export

**Real-time (per-batch):**
- `har_inference_latency_seconds` — Time to process batch
- `har_predictions_total` — Counter of predictions made
- `har_uncertain_predictions_total` — Counter of low-confidence predictions
- `har_confidence_mean` — Gauge of mean confidence

**Drift monitoring:**
- `har_drift_ks_max` — Maximum KS statistic across channels
- `har_drift_alert` — Binary (0/1) drift alert status

**System health:**
- `har_model_loaded` — Binary (0/1) model status
- `har_data_available` — Binary (0/1) data availability

### Complementing MLflow

| Aspect | MLflow | Prometheus/Grafana |
|--------|--------|-------------------|
| **Focus** | Experiment tracking, model versioning | Real-time monitoring, alerting |
| **Granularity** | Per-run metrics | Per-batch/per-minute metrics |
| **Visualization** | Experiment comparison | Live dashboards |
| **Alerting** | None built-in | Alertmanager integration |
| **Retention** | Long-term (experiments) | Short-term (rolling windows) |

### Grafana Dashboard Panels

1. **Inference Health:** Latency, throughput, error rate
2. **Confidence Trends:** Mean confidence over time, uncertain ratio
3. **Drift Status:** Per-channel drift metrics, alert status
4. **Activity Distribution:** Prediction class distribution pie chart

![Figure 6: Drift Over Time](figures/fig6_drift_over_time.png)

*Figure 6 shows how drift metrics trend over batches, indicating when retraining triggers should fire. This view would be replicated in Grafana for real-time monitoring.*

### What to Do Checklist

- [ ] Add `prometheus_client` to inference service
- [ ] Expose `/metrics` endpoint
- [ ] Create Grafana dashboard with key panels
- [ ] Configure Alertmanager for drift and confidence alerts
- [ ] Link Grafana alerts to on-call rotation

### Evidence

**[DevOps-Driven Real-Time Health Analytics | PDF: papers/research_papers/76 papers/DevOps-Driven Real-Time Health Analytics.pdf]**

- Prometheus/Grafana is industry standard for real-time monitoring
- Complements experiment tracking tools like MLflow

**[Essential MLOps: Data Science Horizons 2023 | PDF: papers/mlops_production/Essential_MLOps_Data_Science_Horizons_2023_Data_Science_Horizons_Final_2023.pdf]**

- Multi-layer monitoring (MLflow + Prometheus) provides comprehensive observability
- Real-time alerting enables rapid response to production issues

---

## Summary: Key Takeaways

1. **Domain adaptation is essential** — Models trained in lab conditions fail in production without fine-tuning on target device data

2. **Preprocessing must match exactly** — Same scaler, same window size, same overlap, same gravity handling

3. **Monitor without labels** — Use proxy metrics (confidence, entropy, margin) and drift detection

4. **Multiple baselines** — Frozen training baseline + rolling production reference

5. **ABCD cases matter** — Wrist placement affects signal observability; use adaptive thresholds

6. **Audit everything** — Data provenance, model versions, predictions, and configs must be traceable

7. **Multi-layer monitoring** — MLflow for experiments, Prometheus/Grafana for real-time alerts

---

## Document Metadata

- **Generated by:** HAR MLOps Pipeline Documentation Script
- **Source papers:** 77+ research papers from local repository
- **Figures:** 7 figures generated and stored in `docs/figures/`
- **Related document:** [Bibliography_From_Local_PDFs.md](Bibliography_From_Local_PDFs.md)

---

*This document is part of the Master's Thesis on MLOps-Enhanced Human Activity Recognition for Anxiety Detection. For questions or updates, refer to the thesis repository.*

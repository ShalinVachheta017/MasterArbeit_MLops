# Thesis-Ready Plan: Unlabeled Data Evaluation for HAR MLOps

> **Master Thesis MLOps Project**  
> **Date:** January 15, 2026  
> **Topic:** Scientifically Defensible Evaluation of Unlabeled Production Data

---

## 📋 Table of Contents

1. [Repo Findings](#1-repo-findings)
2. [Paper Findings](#2-paper-findings)
3. [Answer to Doubt: Labels vs No-Label Evaluation](#3-answer-to-doubt)
4. [Unlabeled Evaluation Framework](#4-unlabeled-evaluation-framework)
5. [Estimated Accuracy Plan](#5-estimated-accuracy-plan)
6. [Implementation Plan](#6-implementation-plan)
7. [Checklist](#7-checklist)

---

## 1. Repo Findings

### 1.1 Repository Structure Overview

| Category | Files Found | Purpose |
|----------|-------------|---------|
| **Preprocessing** | `src/preprocess_data.py`, `src/sensor_data_pipeline.py` | CSV → cleaned/resampled/normalized/windowed data |
| **Inference** | `src/run_inference.py` | NumPy → model → probabilities (softmax) |
| **Evaluation** | `src/evaluate_predictions.py` | Prediction analysis, confidence metrics |
| **QC/Smoke Tests** | `scripts/preprocess_qc.py`, `scripts/inference_smoke.py` | Variance collapse, idle detection, unit checks |
| **MLflow** | `src/mlflow_tracking.py`, `config/mlflow_config.yaml` | Experiment tracking, metrics logging |
| **Data Validation** | `src/data_validator.py` | Schema, missing values, value ranges |
| **Configuration** | `src/config.py`, `config/pipeline_config.yaml` | Paths, window size, overlap, thresholds |

### 1.2 Key Files Analyzed

#### Preprocessing Pipeline
- **[src/preprocess_data.py](src/preprocess_data.py)** (794 lines)
  - Unit detection (milliG vs m/s²) and automatic conversion
  - Butterworth filtering, sliding window creation
  - Comprehensive logging with rotation
  
- **[src/sensor_data_pipeline.py](src/sensor_data_pipeline.py)** (1182 lines)
  - Raw Excel/CSV → time-aligned sensor fusion
  - Resampling to 50Hz, interpolation
  - Accelerometer + gyroscope merging

#### Inference Pipeline
- **[src/run_inference.py](src/run_inference.py)** (896 lines)
  - Batch/realtime inference modes
  - Softmax probability output (11 classes)
  - Confidence thresholds documented (>90% HIGH, 70-90% MODERATE, etc.)
  - Already mentions calibration analysis

#### Evaluation (Partial)
- **[src/evaluate_predictions.py](src/evaluate_predictions.py)** (766 lines)
  - `PredictionAnalyzer` for unlabeled data
  - Distribution analysis, confidence analysis, temporal patterns
  - ECE (Expected Calibration Error) mentioned but only valid on labeled data

#### QC Scripts
- **[scripts/inference_smoke.py](scripts/inference_smoke.py)** (638 lines)
  - Model loading validation
  - Shape compatibility checks
  - Uniform prediction detection
  - **Input variance check (idle data detection)**
  - Determinism check

- **[scripts/preprocess_qc.py](scripts/preprocess_qc.py)** (802 lines)
  - Schema validation
  - Missingness checks
  - Sampling rate verification
  - **Unit detection (milliG vs m/s²)**
  - Variance collapse detection

### 1.3 Existing Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Scaler config (mean/std) | `data/prepared/config.json` | ✅ Exists |
| Activity mapping | `data/prepared/config.json` | ✅ Exists (11 classes) |
| Window config | `src/config.py` | ✅ 200 samples, 50% overlap |
| Pretrained model | `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` | ✅ Exists |
| Model info | `models/pretrained/model_info.json` | ✅ Exists |
| **Training baseline stats** | `data/prepared/baseline_stats.json` | ❌ **Missing** → Created |
| **Monitoring framework** | `scripts/post_inference_monitoring.py` | ❌ **Missing** → Created |

### 1.4 What's Missing (Now Addressed)

1. ❌ **No drift detection** comparing production to training distribution
2. ❌ **No baseline statistics** from training data for drift comparison
3. ❌ **No temporal plausibility analysis** (flip rate, dwell time)
4. ❌ **No gating logic** to block pipeline on anomalies
5. ❌ **No documentation** explaining unlabeled evaluation scientifically

---

## 2. Paper Findings

### 2.1 Local Papers Analyzed

#### Core Thesis Papers

| Paper | Location | Key Contribution |
|-------|----------|------------------|
| **ICTH_16.pdf** | `root` | 1D-CNN-BiLSTM architecture, domain adaptation (49%→87% accuracy via fine-tuning) |
| **EHB_2025_71.pdf** | `root` | RAG-enhanced pipeline, bout analysis methodology |

#### OOD/Drift Detection Papers

| Paper | Location | Method | How We Use It |
|-------|----------|--------|---------------|
| **NeurIPS-2020-energy-based-out-of-distribution-detection** | `new paper/` | Energy Score: $E(x) = -\log \sum \exp(f_i(x))$ | Layer 3 optional OOD score |
| **NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift** | `new paper/` | Adaptive conformal prediction | Future extension for prediction sets |

#### Calibration Papers

| Paper | Location | Method | How We Use It |
|-------|----------|--------|---------------|
| **When Does Optimizing a Proper Loss Yield Calibration** | `papers needs to read/76 papers/` | Calibration theory | ECE metric on labeled validation only |

#### HAR & Domain Adaptation Papers

| Paper | Location | Method | How We Use It |
|-------|----------|--------|---------------|
| **Domain Adaptation for IMU-based HAR** | `papers needs to read/76 papers/` | Cross-device drift | Feature drift detection motivation |
| **Are Anxiety Detection Models Generalizable** | `papers needs to read/76 papers/` | Cross-population study | Justify monitoring for generalization |
| **Resilience of ML Models - Gaussian Noise** | `papers needs to read/76 papers/` | Noise robustness | Sensor integrity checks |

#### MLOps Papers

| Paper | Location | Method | How We Use It |
|-------|----------|--------|---------------|
| **MACHINE LEARNING OPERATIONS A SURVEY** | `papers needs to read/76 papers/` | MLOps best practices | Framework structure |
| **MLHOps: ML for Healthcare Operations** | `papers needs to read/76 papers/` | Healthcare MLOps | Gating and auditability |
| **MLDEV: Data Science Experiment Automation** | `papers needs to read/76 papers/` | DVC + MLflow | Already implemented |

### 2.2 Paper → Method → Application Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PAPER → METHOD → APPLICATION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  NeurIPS-2020 Energy OOD         →  Energy Score                    │
│    └─→ Optional Layer 3 metric for OOD detection                    │
│                                                                      │
│  Calibration Theory Paper        →  ECE (Expected Calibration Error)│
│    └─→ Only on labeled validation, not production                   │
│                                                                      │
│  Domain Adaptation Survey        →  KS test, feature drift          │
│    └─→ Layer 3 drift detection per sensor channel                   │
│                                                                      │
│  ICTH_16 (Core Paper)            →  Window structure, activities    │
│    └─→ 200 samples @ 50Hz, 11 anxiety-related classes               │
│                                                                      │
│  EHB_2025_71 (Core Paper)        →  Bout analysis, temporal windows │
│    └─→ Layer 2 temporal plausibility (gap thresholds)               │
│                                                                      │
│  MLOps Survey                    →  Monitoring, drift triggers      │
│    └─→ Overall framework structure, gating decisions                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Web Search Queries (If Papers Missing)

If you need additional references, search for:

1. `"expected calibration error" ECE deep learning confidence`
2. `"energy score" out-of-distribution detection neural network`
3. `"Kolmogorov-Smirnov test" covariate shift machine learning`
4. `"conformal prediction" time series classification uncertainty`
5. `"human activity recognition" domain adaptation wearable IMU`
6. `"uncertainty quantification" softmax neural network`
7. `"flip rate" temporal consistency activity recognition`
8. `"minimal labeling" active learning deployment estimation`
9. `"MLOps monitoring" drift detection production`
10. `"sensor data quality" accelerometer gyroscope validation`

---

## 3. Answer to Doubt

### 3.1 Why We Cannot Compute Accuracy Without Labels

**Short Answer (for thesis):**

> Accuracy is defined as the ratio of correct predictions to total predictions. To compute this, we must compare predicted labels to true labels. Without ground truth labels (which are unavailable for unlabeled production data), we cannot determine whether any individual prediction is correct or incorrect. Therefore, **accuracy, precision, recall, F1-score, and confusion matrices are mathematically undefined for unlabeled data**. Any system claiming to compute these metrics on unlabeled data is either using assumptions (pseudo-labels), historical metrics from labeled validation sets, or producing scientifically invalid results.

### 3.2 What We CAN Evaluate (Label-Free)

#### A. Confidence & Uncertainty Metrics
- **Source:** Model's softmax output probabilities
- **Metrics:** Max probability, entropy, margin (top1 - top2)
- **What they prove:** Model's internal certainty about prediction
- **What they don't prove:** Actual correctness of prediction

#### B. Temporal Plausibility Metrics
- **Source:** Sequence of predicted classes
- **Metrics:** Flip rate, dwell time, transition matrix
- **What they prove:** Predictions follow realistic activity patterns
- **What they don't prove:** Individual predictions are correct

#### C. Drift/OOD Metrics
- **Source:** Comparison of production features to training baselines
- **Metrics:** KS test, Wasserstein distance, mean/std shift
- **What they prove:** Data distribution is similar to training
- **What they don't prove:** Model generalizes correctly

#### D. Signal Quality Metrics
- **Source:** Raw sensor data
- **Metrics:** Sampling rate, missingness, clipping, gravity check
- **What they prove:** Data collection was successful
- **What they don't prove:** Anything about model performance

### 3.3 Critical Scientific Statement

```
┌─────────────────────────────────────────────────────────────────────┐
│  WHAT METRICS PROVE vs DON'T PROVE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ High confidence + plausible sequences + no drift                │
│     → Model is LIKELY performing reasonably                         │
│     → Data LOOKS similar to training                                │
│     → Predictions APPEAR temporally consistent                      │
│                                                                      │
│  ❌ These DO NOT prove:                                             │
│     → Predictions are CORRECT                                       │
│     → Accuracy is HIGH                                              │
│     → Model has not degraded                                        │
│                                                                      │
│  📊 To PROVE accuracy:                                              │
│     → Must label some production data                               │
│     → Even a small random sample gives confidence interval          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Unlabeled Evaluation Framework

### 4.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    3-LAYER MONITORING FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 1: Per-Window Confidence (from model output)                 │
│  ├── max_probability (softmax confidence)                           │
│  ├── entropy (uncertainty measure)                                  │
│  ├── margin (top1 - top2 difference)                               │
│  └── Flag: confidence < 0.50 → "uncertain"                         │
│                                                                      │
│  LAYER 2: Temporal Plausibility (from prediction sequence)          │
│  ├── flip_rate (% of consecutive class changes)                    │
│  ├── mean_dwell_time (average activity duration)                   │
│  ├── transition_matrix (class → class counts)                      │
│  └── Flag: flip_rate > 30% → "unstable"                            │
│                                                                      │
│  LAYER 3: Batch Drift Detection (vs training baseline)              │
│  ├── KS test per channel (p-value < 0.01 = drift)                  │
│  ├── Wasserstein distance (> threshold = drift)                    │
│  ├── Variance collapse (std < 0.1 = idle/failure)                  │
│  └── Flag: multiple channels drift → "block"                       │
│                                                                      │
│  GATING LOGIC:                                                      │
│  ├── All PASS → Continue pipeline                                  │
│  ├── Any WARN → Continue + flag for review                         │
│  └── Any BLOCK → Stop pipeline, investigate                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Layer 1: Confidence/Uncertainty

**Metrics:**

| Metric | Formula | Threshold | Interpretation |
|--------|---------|-----------|----------------|
| `max_prob` | $\max_i p_i$ | < 0.50 | Uncertain prediction |
| `entropy` | $-\sum p_i \log p_i$ | > 2.0 | High uncertainty |
| `margin` | $p_{top1} - p_{top2}$ | < 0.10 | Ambiguous (two classes) |

**Output:**
- Per-window flags (uncertain/confident)
- Summary: % uncertain, mean confidence
- Visualization: confidence histogram

### 4.3 Layer 2: Temporal Plausibility

**Metrics:**

| Metric | Definition | Threshold | Interpretation |
|--------|------------|-----------|----------------|
| `flip_rate` | transitions / (windows - 1) | > 0.30 | Unstable predictions |
| `mean_dwell` | avg seconds per activity bout | < 2.0s | Unrealistically short |
| `n_bouts` | count of continuous segments | Context | Activity diversity |

**Domain Knowledge (HAR-specific):**
- Anxiety activities typically last 5-30 seconds
- Sitting/standing bouts last minutes
- Rapid oscillation (< 1s) is unrealistic

### 4.4 Layer 3: Drift Detection

**Metrics:**

| Metric | Method | Threshold | Interpretation |
|--------|--------|-----------|----------------|
| `ks_pvalue` | Kolmogorov-Smirnov test | < 0.01 | Significant drift |
| `wasserstein` | Earth mover's distance | > 0.5 | Distribution shifted |
| `mean_shift` | \|prod_mean - train_mean\| / train_std | > 0.5 | Systematic bias |
| `variance` | std of channel | < 0.1 | Variance collapse |

**Baseline Required:**
- `data/prepared/baseline_stats.json` (created by `build_training_baseline.py`)
- Contains per-channel mean, std, percentiles from training data

---

## 5. Estimated Accuracy Plan

### 5.1 Random Sampling Protocol

**Goal:** Estimate true accuracy with confidence interval

**Protocol:**
1. From each batch (e.g., weekly), randomly sample **50-200 windows**
2. Have human annotator label these windows (ground truth)
3. Compute accuracy on sample: $\hat{a} = \text{correct} / \text{total}$
4. Calculate 95% confidence interval:

$$\hat{a} \pm 1.96 \sqrt{\frac{\hat{a}(1-\hat{a})}{n}}$$

**Example:**
- Sample 100 windows, 82 correct → $\hat{a} = 82\%$
- 95% CI: $82\% \pm 7.5\%$ → True accuracy likely 74.5% - 89.5%

**MLflow Integration:**
```python
mlflow.log_metric("estimated_accuracy", 0.82)
mlflow.log_metric("accuracy_ci_lower", 0.745)
mlflow.log_metric("accuracy_ci_upper", 0.895)
mlflow.log_param("labeled_sample_size", 100)
```

### 5.2 Active Sampling Protocol

**Goal:** Focus labeling effort on uncertain predictions

**Protocol:**
1. Sort windows by uncertainty (lowest confidence first)
2. Label top 20-50 most uncertain windows
3. This gives **worst-case accuracy** estimate
4. Also label 20 high-confidence windows to check calibration

**Rationale:**
- If uncertain predictions are mostly correct → model is under-confident (good)
- If uncertain predictions are mostly wrong → model correctly identifies uncertainty

### 5.3 Sentinel Session Protocol

**Goal:** Direct accuracy measurement in deployment conditions

**Protocol:**
1. Weekly: Record a **controlled 5-minute session**
2. Perform scripted activities with known labels:
   - 60s sitting
   - 60s standing
   - 30s nail_biting
   - 30s ear_rubbing
   - 60s transitions
3. Run inference on sentinel data
4. Compute exact accuracy (ground truth available)
5. Track accuracy trend over weeks

**Benefits:**
- Controls for device/wearer variability
- Detects model degradation over time
- No randomness - exact measurement

---

## 6. Implementation Plan

### 6.1 New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `docs/UNLABELED_EVALUATION.md` | Scientific documentation | ~400 |
| `scripts/post_inference_monitoring.py` | Main monitoring script | ~650 |
| `scripts/build_training_baseline.py` | Baseline builder | ~350 |

### 6.2 File: `scripts/post_inference_monitoring.py`

**Classes:**
- `MonitoringConfig` - Thresholds and configuration
- `ConfidenceAnalyzer` - Layer 1 metrics
- `TemporalAnalyzer` - Layer 2 metrics
- `DriftDetector` - Layer 3 metrics
- `PostInferenceMonitor` - Orchestrator

**Usage:**
```bash
# Basic usage
python scripts/post_inference_monitoring.py \
    --predictions data/prepared/predictions/latest.csv

# With drift detection
python scripts/post_inference_monitoring.py \
    --predictions data/prepared/predictions/latest.csv \
    --data data/prepared/production_X.npy \
    --baseline data/prepared/baseline_stats.json

# With MLflow logging
python scripts/post_inference_monitoring.py \
    --predictions data/prepared/predictions/latest.csv \
    --mlflow
```

**Output:**
```
reports/monitoring/2026-01-15_12-30-00_batch001/
├── confidence_report.json
├── temporal_report.json
├── drift_report.json
└── summary.json
```

### 6.3 File: `scripts/build_training_baseline.py`

**Usage:**
```bash
# From labeled CSV
python scripts/build_training_baseline.py \
    --input data/raw/all_users_data_labeled.csv

# With embedding baseline
python scripts/build_training_baseline.py \
    --input data/raw/all_users_data_labeled.csv \
    --embeddings \
    --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras
```

**Output:**
```
data/prepared/
├── baseline_stats.json      # Feature statistics
└── baseline_embeddings.npz  # Model embeddings (optional)
```

### 6.4 Integration Points

**After inference (`run_inference.py`):**
```python
# At end of inference script
from scripts.post_inference_monitoring import PostInferenceMonitor

monitor = PostInferenceMonitor()
report = monitor.run(
    predictions_path=predictions_csv_path,
    production_data_path=production_npy_path,
    baseline_path=baseline_json_path,
    output_dir=reports_dir
)

if report.gating_decision == "BLOCK":
    raise PipelineError("Monitoring failed - see reports")
```

**In MLflow tracking:**
```python
# Log monitoring as part of inference run
mlflow.log_metric("monitoring/mean_confidence", report.layer1_confidence["metrics"]["mean_confidence"])
mlflow.log_metric("monitoring/flip_rate", report.layer2_temporal["metrics"]["flip_rate"])
mlflow.set_tag("monitoring_status", report.overall_status)
mlflow.set_tag("needs_review", str(report.needs_review))
```

### 6.5 Gating Logic

```python
# In pipeline orchestration
def should_continue_pipeline(report: MonitoringReport) -> bool:
    if report.gating_decision == "BLOCK":
        logger.error("Pipeline BLOCKED due to monitoring failures")
        # Alert / stop pipeline
        return False
    
    if report.gating_decision == "PASS_WITH_REVIEW":
        logger.warning("Pipeline continuing but needs human review")
        mlflow.set_tag("needs_review", "true")
        # Send alert / create ticket
    
    return True
```

---

## 7. Checklist

### 7.1 Immediate Actions (Do Now)

- [x] Create `docs/UNLABELED_EVALUATION.md` - Scientific documentation
- [x] Create `scripts/post_inference_monitoring.py` - Main monitoring script
- [x] Create `scripts/build_training_baseline.py` - Baseline builder
- [ ] **Run baseline builder on training data:**
  ```bash
  python scripts/build_training_baseline.py
  ```
- [ ] **Test monitoring on existing predictions:**
  ```bash
  python scripts/post_inference_monitoring.py --predictions data/prepared/predictions/latest.csv
  ```

### 7.2 Integration Actions (This Week)

- [ ] Add monitoring call to end of `run_inference.py`
- [ ] Add gating logic to pipeline orchestration
- [ ] Configure MLflow to log monitoring metrics
- [ ] Create dashboard/alert for WARN/BLOCK status

### 7.3 Validation Actions (Before Deployment)

- [ ] Run monitoring on 3-5 different production batches
- [ ] Verify drift detection catches simulated drift
- [ ] Verify variance collapse detection works
- [ ] Label 50-100 windows to estimate initial accuracy
- [ ] Document sentinel session protocol

### 7.4 Thesis Writing Actions

- [ ] Write "Unlabeled Evaluation" section using `UNLABELED_EVALUATION.md`
- [ ] Include 3-layer framework diagram in thesis
- [ ] Cite relevant papers (see Section 2)
- [ ] Present estimated accuracy with confidence intervals
- [ ] Discuss limitations (what metrics don't prove)

### 7.5 Future Extensions

- [ ] Energy score (requires logit output instead of softmax)
- [ ] Conformal prediction sets
- [ ] Embedding drift detection
- [ ] Automated retraining triggers
- [ ] Web dashboard for monitoring visualization

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ❌ CANNOT compute on unlabeled data:                               │
│     Accuracy, Precision, Recall, F1, Confusion Matrix               │
│                                                                      │
│  ✅ CAN compute on unlabeled data:                                  │
│     Confidence, Entropy, Margin, Flip Rate, Drift Score             │
│                                                                      │
│  📊 To estimate accuracy:                                           │
│     Label 50-200 random windows → compute accuracy + CI             │
│                                                                      │
│  🚨 Gating rules:                                                   │
│     Drift detected → BLOCK                                          │
│     >30% uncertain → WARN                                           │
│     Variance collapse → BLOCK (sensor issue)                        │
│     All pass → PASS (still label samples!)                          │
│                                                                      │
│  📁 Key new files:                                                  │
│     docs/UNLABELED_EVALUATION.md                                    │
│     scripts/post_inference_monitoring.py                            │
│     scripts/build_training_baseline.py                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

**Document Generated:** January 15, 2026  
**For:** Master Thesis MLOps Project - HAR Pipeline

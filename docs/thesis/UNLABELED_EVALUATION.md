# Unlabeled Data Evaluation Framework for HAR MLOps

> **📝 Summary:** This document provides a scientifically defensible framework for evaluating model predictions on **unlabeled deployment data**. It explains why accuracy cannot be computed without labels, what metrics CAN be computed, and how to obtain estimated accuracy through minimal labeling.

**Version:** 1.0  
**Date:** January 15, 2026  
**Author:** Master Thesis MLOps Project

---

## Table of Contents
1. [Why We Cannot Compute Accuracy Without Labels](#1-why-we-cannot-compute-accuracy-without-labels)
2. [What We CAN Evaluate on Unlabeled Data](#2-what-we-can-evaluate-on-unlabeled-data)
3. [Three-Layer Monitoring Framework](#3-three-layer-monitoring-framework)
4. [Metrics Interpretation Guide](#4-metrics-interpretation-guide)
5. [Estimated Accuracy via Minimal Labeling](#5-estimated-accuracy-via-minimal-labeling)
6. [Implementation Guide](#6-implementation-guide)
7. [MLflow Integration](#7-mlflow-integration)
8. [References](#8-references)

---

## 1. Why We Cannot Compute Accuracy Without Labels

### The Fundamental Problem

**Accuracy** is defined as:

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

To compute this, we need to know:
1. **Predicted labels** (from the model) ✅ Available
2. **True labels** (ground truth) ❌ **NOT Available** for deployment data

Without ground truth labels, we **cannot know** if a prediction is correct or incorrect. Therefore:

> **Scientific Fact:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrices are **undefined** for unlabeled data. Any system that reports these metrics on unlabeled data is either:
> 1. Using pseudo-labels (which are assumptions, not ground truth)
> 2. Reporting metrics from a validation set (not the current data)
> 3. Faking the metrics (scientifically invalid)

### What This Means for This Pipeline

| Data Type | Location | Labels | Can Compute Accuracy? |
|-----------|----------|--------|----------------------|
| Training Data | `data/raw/all_users_data_labeled.csv` | ✅ Yes | ✅ Yes |
| Validation Data | Split from training | ✅ Yes | ✅ Yes |
| Production Data | `data/prepared/production_X.npy` | ❌ No | ❌ **No** |
| New Garmin CSVs | `decoded_csv_files/` | ❌ No | ❌ **No** |

---

## 2. What We CAN Evaluate on Unlabeled Data

Although we cannot compute accuracy, we CAN compute **label-free quality metrics** that indicate:
- **Model confidence and uncertainty**
- **Temporal plausibility of predictions**
- **Distribution shift from training data**
- **Signal quality and sensor integrity**

### 2.1 Confidence & Uncertainty Metrics

These metrics come from the **model's output probabilities** (softmax outputs):

| Metric | Formula | What It Tells Us |
|--------|---------|------------------|
| **Max Probability** | $p_{max} = \max_i p_i$ | How confident the model is in its top prediction |
| **Entropy** | $H = -\sum p_i \log p_i$ | Overall uncertainty (higher = more uncertain) |
| **Margin** | $m = p_{top1} - p_{top2}$ | Difference between top two predictions |
| **Energy Score** | $E = -\log \sum \exp(z_i)$ | OOD detection score (requires logits, not probs) |

**Interpretation:**
- High confidence ($p_{max} > 0.9$) → Model is certain (but not necessarily correct!)
- Low confidence ($p_{max} < 0.5$) → Model is uncertain → Flag for review
- Near-zero margin → Model is confused between two classes

**Critical Note:** High confidence does NOT guarantee correctness. Neural networks are often **overconfident**. This is why we need calibration metrics (see Section 2.4).

### 2.2 Temporal Plausibility Metrics

These metrics analyze **sequences of predictions** to detect unrealistic behavior:

| Metric | Definition | Threshold |
|--------|------------|-----------|
| **Flip Rate** | % of consecutive windows with different predictions | < 30% is typical |
| **Dwell Time** | Average duration of each predicted activity | > 4 seconds expected |
| **Transition Violations** | Count of "impossible" transitions (e.g., sitting → nail_biting → sitting in 1 second) | Should be 0 |
| **Activity Entropy** | Diversity of predicted activities per session | Context-dependent |

**Why This Matters:**
- Human activities have **temporal structure** (you don't switch from sitting to standing 10 times per second)
- High flip rate suggests the model is unstable/uncertain
- Impossible transitions suggest OOD data or model confusion

### 2.3 Distribution Drift Metrics

These metrics compare **production data** to **training data baselines**:

| Metric | Method | What It Detects |
|--------|--------|-----------------|
| **Feature Drift** | KS test, Wasserstein distance on raw features | Sensor characteristics changed |
| **Mean/Std Shift** | Compare μ and σ per channel | Systematic bias or variability change |
| **Embedding Drift** | Compare model embeddings | Semantic shift in data |
| **Variance Collapse** | Std deviation near zero | Idle/stationary data or sensor failure |

**Interpretation:**
- KS test p-value < 0.05 → Significant drift detected
- Wasserstein distance > threshold → Feature distribution changed
- Variance collapse → Data is idle or sensor is malfunctioning

### 2.4 Signal Quality Metrics (Sensor Integrity)

| Metric | Check | Implication |
|--------|-------|-------------|
| **Sampling Rate** | Actual Hz vs expected 50 Hz | Resampling issues |
| **Missing Values** | % NaN per channel | Data integrity |
| **Clipping** | Values at min/max sensor limits | Sensor saturation |
| **Gravity Check** | Az mean ≈ -9.8 m/s² | Unit conversion correct |
| **Noise Floor** | Minimum variance | Sensor working |

---

## 3. Three-Layer Monitoring Framework

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POST-INFERENCE MONITORING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 1: Per-Window Confidence/Uncertainty                         │
│  ├── Max softmax probability                                        │
│  ├── Entropy of probabilities                                       │
│  ├── Margin (top1 - top2)                                          │
│  └── Flag uncertain windows (confidence < threshold)                │
│                                                                      │
│  LAYER 2: Sequence Temporal Plausibility                            │
│  ├── Flip rate (class changes / total windows)                      │
│  ├── Dwell time per activity                                        │
│  ├── Transition matrix validation                                   │
│  └── Smoothing recommendations (majority voting, HMM)               │
│                                                                      │
│  LAYER 3: Batch-Level Drift vs Training Baseline                    │
│  ├── Feature drift (KS test, Wasserstein)                          │
│  ├── Embedding drift (cosine similarity)                            │
│  ├── Sensor integrity (sampling rate, missingness, clipping)        │
│  └── Gating decision (PASS / WARN / BLOCK)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Per-Window Confidence/Uncertainty

**Purpose:** Flag individual predictions that may be unreliable.

**Inputs:**
- Softmax probabilities: shape `(n_windows, 11)`
- (Optional) Logits for energy score

**Outputs:**
- `confidence_scores.csv`: Per-window confidence metrics
- `uncertain_windows.csv`: Windows flagged for review
- Summary statistics in MLflow

**Thresholds (configurable):**
| Metric | Threshold | Action |
|--------|-----------|--------|
| `confidence < 0.50` | Uncertain | Flag for review |
| `confidence < 0.30` | Very uncertain | Consider excluding |
| `entropy > 2.0` | High uncertainty | Flag for review |
| `margin < 0.10` | Ambiguous | Two classes equally likely |

### Layer 2: Sequence Temporal Plausibility

**Purpose:** Detect unrealistic prediction sequences.

**Inputs:**
- Predicted class sequence: shape `(n_windows,)`
- Timestamps or window IDs

**Outputs:**
- `temporal_analysis.json`: Flip rate, dwell times, transition counts
- Warnings for unrealistic patterns

**Metrics:**
```python
flip_rate = (n_class_changes) / (n_windows - 1)
mean_dwell_time = mean([duration for each continuous activity bout])
transition_matrix[i][j] = count of transitions from class i to class j
```

**Example Impossible Transitions (HAR-specific):**
- `sitting` → `standing` → `sitting` in < 2 seconds (unrealistic)
- Any anxiety activity lasting < 1 second (too short)

### Layer 3: Batch-Level Drift/OOD Detection

**Purpose:** Detect if production data distribution differs from training.

**Inputs:**
- Production data: `(n_windows, 200, 6)` NumPy array
- Training baseline: `baseline_stats.json` (mean, std, percentiles)
- (Optional) Training embeddings: `baseline_embeddings.npz`

**Outputs:**
- `drift_report.json`: Per-channel drift scores
- Gating decision: PASS / WARN / BLOCK

**Metrics:**
```python
# Feature-level drift (per channel)
for channel in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
    ks_stat, ks_pvalue = scipy.stats.ks_2samp(train_channel, prod_channel)
    wasserstein = scipy.stats.wasserstein_distance(train_channel, prod_channel)
    mean_diff = abs(train_mean - prod_mean)
    std_ratio = prod_std / train_std

# Gating logic
if any(ks_pvalue < 0.01) or wasserstein > threshold:
    decision = "WARN" or "BLOCK"
```

---

## 4. Metrics Interpretation Guide

### What These Metrics PROVE

| Metric Category | What It Proves | Scientific Basis |
|-----------------|----------------|------------------|
| Confidence/Uncertainty | Model's internal certainty about prediction | Information theory (entropy) |
| Temporal Plausibility | Predictions follow realistic activity patterns | Domain knowledge (HAR) |
| Drift Detection | Data distribution differs from training | Statistical testing (KS, Wasserstein) |
| Sensor Integrity | Data collection was successful | Physics (gravity, units) |

### What These Metrics DO NOT PROVE

| Claim | Why It's Invalid |
|-------|------------------|
| "High confidence = correct prediction" | Neural networks are often overconfident |
| "No drift = high accuracy" | Model could still be systematically wrong |
| "Plausible sequence = correct predictions" | Wrong predictions can still be temporally plausible |
| "All checks pass = deploy safely" | Still need minimal labeling to estimate accuracy |

### Decision Matrix

| Layer 1 | Layer 2 | Layer 3 | Overall Decision |
|---------|---------|---------|------------------|
| ✅ High confidence | ✅ Plausible | ✅ No drift | PASS (but still get labeled samples!) |
| ⚠️ Some uncertain | ✅ Plausible | ✅ No drift | WARN - Review uncertain windows |
| ⚠️ Some uncertain | ⚠️ Unstable | ✅ No drift | WARN - Model may be confused |
| Any | Any | ❌ Drift detected | BLOCK - Investigate data pipeline |
| ❌ Low confidence | ❌ Unstable | Any | BLOCK - Serious issues |

---

## 5. Estimated Accuracy via Minimal Labeling

### Why Minimal Labeling?

Without labels, we cannot know true accuracy. But we can **estimate** accuracy by labeling a **small random sample** of deployment data.

### Strategy 1: Random Sampling

**Protocol:**
1. From each batch of N windows, randomly sample k windows (e.g., k=50)
2. Have a human label these windows (ground truth)
3. Compute accuracy on this sample
4. Calculate confidence interval

**Statistical Basis:**
- Sample accuracy $\hat{a}$ is an unbiased estimator of true accuracy
- 95% confidence interval: $\hat{a} \pm 1.96 \sqrt{\frac{\hat{a}(1-\hat{a})}{k}}$

**Example:**
- Sample 50 windows, 42 correct → $\hat{a} = 84\%$
- 95% CI: $84\% \pm 10.2\%$ → True accuracy likely 74-94%
- Sample 200 windows for tighter bounds

### Strategy 2: Active Sampling (Uncertainty-Based)

**Protocol:**
1. Sort windows by uncertainty (low confidence first)
2. Label the most uncertain windows
3. This gives **worst-case accuracy** estimate
4. Also label some high-confidence windows to check calibration

**Rationale:**
- Uncertain predictions are more likely to be wrong
- Labeling these first gives faster insight into model failures
- Reveals confusion patterns between specific classes

### Strategy 3: Sentinel Protocol (Controlled Sessions)

**Protocol:**
1. Weekly: Record a **controlled session** (5-10 minutes)
2. Perform known activities in a scripted sequence
3. These sessions have **ground truth labels**
4. Run inference and compute accuracy on sentinel data

**Benefits:**
- Direct accuracy measurement on real deployment conditions
- Detects degradation over time
- Controls for device/wearer variability

**Implementation:**
```yaml
sentinel_session:
  duration_minutes: 5
  activities:
    - name: sitting
      duration_seconds: 60
    - name: standing
      duration_seconds: 60
    - name: nail_biting
      duration_seconds: 30
    - name: ear_rubbing
      duration_seconds: 30
    - name: walking (transition)
      duration_seconds: 60
  frequency: weekly
```

---

## 6. Implementation Guide

### File Structure

```
scripts/
├── post_inference_monitoring.py    # Main monitoring script (NEW)
├── build_training_baseline.py      # Build baseline from training data (NEW)
├── inference_smoke.py              # Existing smoke test
└── preprocess_qc.py                # Existing QC

src/
├── monitoring/                     # NEW monitoring module
│   ├── __init__.py
│   ├── confidence_metrics.py       # Layer 1 metrics
│   ├── temporal_metrics.py         # Layer 2 metrics
│   ├── drift_metrics.py            # Layer 3 metrics
│   └── gating.py                   # Decision logic

data/prepared/
├── baseline_stats.json             # Training feature statistics (NEW)
├── baseline_embeddings.npz         # Training embeddings (optional, NEW)
└── config.json                     # Existing scaler config

reports/
└── monitoring/                     # NEW output directory
    ├── 2026-01-15_batch001/
    │   ├── confidence_report.json
    │   ├── temporal_report.json
    │   ├── drift_report.json
    │   └── summary.json
    └── dashboards/
        └── monitoring_dashboard.html
```

### Gating Rules

**In MLflow Run:**
```python
# Tag run with monitoring status
if drift_score > DRIFT_THRESHOLD:
    mlflow.set_tag("monitoring_status", "DRIFT_DETECTED")
    mlflow.set_tag("needs_review", "true")
    
if uncertain_ratio > 0.3:  # >30% uncertain windows
    mlflow.set_tag("monitoring_status", "HIGH_UNCERTAINTY")
    mlflow.set_tag("needs_review", "true")
```

**In Pipeline:**
```python
if monitoring_result.decision == "BLOCK":
    raise PipelineGatingError("Monitoring checks failed - see drift_report.json")
```

---

## 7. MLflow Integration

### Metrics to Log

| Metric | Type | Key |
|--------|------|-----|
| Mean confidence | Metric | `monitoring/mean_confidence` |
| Uncertain ratio | Metric | `monitoring/uncertain_ratio` |
| Flip rate | Metric | `monitoring/flip_rate` |
| KS test p-value (min) | Metric | `monitoring/ks_pvalue_min` |
| Drift detected | Tag | `drift_detected` |
| Monitoring status | Tag | `monitoring_status` |

### Artifacts to Log

| Artifact | Description |
|----------|-------------|
| `confidence_report.json` | Per-window confidence scores |
| `temporal_report.json` | Sequence analysis |
| `drift_report.json` | Drift detection results |
| `confidence_histogram.png` | Visualization |

---

## 8. References

### Papers Used in This Framework

| Paper | Method | Applied Here |
|-------|--------|--------------|
| `NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf` | Energy Score for OOD | Layer 3 (optional) |
| `NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf` | Conformal prediction under shift | Future extension |
| `When Does Optimizing a Proper Loss Yield Calibration.pdf` | Calibration theory | ECE metric |
| `MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | MLOps monitoring best practices | Overall framework |
| `Are Anxiety Detection Models Generalizable-A Cross-Activity...pdf` | Cross-domain HAR | Drift detection motivation |
| `Resilience of Machine Learning Models in Anxiety Detection...pdf` | Noise robustness in HAR | Sensor integrity checks |
| `Domain Adaptation for Inertial Measurement Unit-based Human.pdf` | IMU domain shift | Feature drift metrics |

### Google Scholar Queries for Further Reading

If you need additional papers, search for:

1. `"expected calibration error" neural network confidence`
2. `"energy score" out-of-distribution detection`
3. `"drift detection" machine learning production`
4. `"conformal prediction" time series classification`
5. `"human activity recognition" domain adaptation wearable`
6. `"uncertainty quantification" deep learning classification`
7. `"temporal consistency" activity recognition prediction`
8. `"Kolmogorov-Smirnov test" distribution shift detection`
9. `"minimal labeling" active learning deployment`
10. `"MLOps monitoring" model performance production`

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNLABELED EVALUATION QUICK REF                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ❌ CANNOT compute on unlabeled data:                               │
│     • Accuracy, Precision, Recall, F1-Score                         │
│     • Confusion Matrix                                              │
│     • Per-class accuracy                                            │
│                                                                      │
│  ✅ CAN compute on unlabeled data:                                  │
│     • Confidence scores (max prob, entropy, margin)                 │
│     • Temporal plausibility (flip rate, dwell time)                 │
│     • Drift metrics (KS test, Wasserstein, mean shift)              │
│     • Sensor quality (sampling rate, missingness)                   │
│                                                                      │
│  📊 To get ESTIMATED ACCURACY:                                      │
│     1. Random sample: Label 50-200 windows → compute accuracy       │
│     2. Active sample: Label uncertain windows → worst-case          │
│     3. Sentinel: Weekly controlled session → direct measurement     │
│                                                                      │
│  🚨 Gating Rules:                                                   │
│     • Drift detected (KS p < 0.01) → WARN/BLOCK                     │
│     • >30% uncertain windows → WARN                                 │
│     • Variance collapse → BLOCK (sensor issue)                      │
│     • All pass → PASS (still need labeled samples!)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

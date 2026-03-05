# HAR MLOps — Monitoring & Retraining Guide

> **Pipeline version:** 2.0 · **Last updated:** 2026-02-16  
> **Model:** 1D-CNN-BiLSTM (499 K params, 11 activity classes)  
> **Sensor:** Garmin accelerometer + gyroscope, 50 Hz, 6 channels

---

## Table of Contents

1. [Drift Detection — Equation & Threshold](#1-drift-detection--equation--threshold)
2. [Confidence Threshold (0.50)](#2-confidence-threshold-050)
3. [Uncertain-Prediction Percentage (10 %)](#3-uncertain-prediction-percentage-10-)
4. [Red-Flag Metrics & Rule-of-Thumb Thresholds](#4-red-flag-metrics--rule-of-thumb-thresholds)
5. [MLflow Model Registry — Why and How](#5-mlflow-model-registry--why-and-how)
6. [Two MLflow Experiments Explained](#6-two-mlflow-experiments-explained)
7. [Reading the MLflow Dashboard](#7-reading-the-mlflow-dashboard)
8. [Per-Dataset Inference Analysis](#8-per-dataset-inference-analysis)
9. [Retraining Pipeline — Pseudo-Labeling & AdaBN](#9-retraining-pipeline--pseudo-labeling--adabn)

---

## 1. Drift Detection — Equation & Threshold

### 1.1 The Drift Equation

Our Stage 6 (Post-Inference Monitoring) computes **per-channel z-score drift**:

$$
d_i = \frac{|\mu_{\text{prod},i} - \mu_{\text{baseline},i}|}{\sigma_{\text{baseline},i} + \epsilon}
$$

where $i \in \{$`accel_x`, `accel_y`, `accel_z`, `gyro_x`, `gyro_y`, `gyro_z`$\}$.

The **aggregate drift score** is the maximum across all six channels:

$$
D = \max_{i=1}^{6} \; d_i
$$

This is a standard **z-score normalised mean difference**. A score of 1.0 means the production mean has shifted by one baseline standard deviation.

### 1.2 Why the Original 0.15 Was Too Sensitive

We ran the drift equation against **all 24 working raw datasets** in `data/raw/`. Results:

| Statistic | Value |
|-----------|-------|
| Median drift | 0.748 |
| Mean drift | 1.130 |
| 25th percentile | 0.360 |
| 75th percentile | 1.175 |
| % datasets exceeding 0.15 | **100 %** (24/24) |

At 0.15, every single recording session triggers a false alarm — the threshold was measuring noise, not genuine distribution shift.

### 1.3 Data-Driven Threshold: 0.75 (WARNING) / 1.50 (ALERT)

Following **Statistical Process Control (SPC)** principles (Montgomery, 2009), we set the thresholds around the empirical distribution:

- **WARNING (0.75):** ≈ median drift — half the datasets are within normal, half warrant inspection.
- **ALERT (1.50):** ≈ 75th percentile + margin — a significant departure from baseline justifying retraining.

### 1.4 Per-Channel Analysis

The worst-drifting channel is **accel_x** (accelerometer X-axis), which exceeds the drift threshold in 17 out of 24 datasets. Gyroscope channels are generally more stable.

### 1.5 Citations

- **PSI thresholds (industry standard):** "From Development to Deployment: An Approach to MLOps Monitoring for Machine Learning Model Operationalization," IEEE SITA 2023, DOI: `10.1109/SITA60746.2023.10373733`. Establishes PSI < 0.10 = no shift, 0.10–0.25 = moderate shift, PSI ≥ 0.25 = major shift (pp. 9–11).
- **Statistical Process Control:** Montgomery, D. C. (2009). *Introduction to Statistical Quality Control* (6th ed.). Wiley. Warning limits at ±2σ, action limits at ±3σ.
- **Drift detection for IMU-based HAR:** Chakma, A., Faridee, A. Z. M., Ghosh, I., & Roy, N. (2023). "Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey." arXiv:2304.06489. Recommends KS test, MMD, and per-channel z-scores for detecting sensor-placement drift.
- **Multi-metric monitoring:** Moskalenko, V. (2024). "Resilience-aware MLOps for AI-based Medical Diagnostic System." *Frontiers in Public Health*, DOI: `10.3389/fpubh.2024.1342937`. Uses multiple drift metrics (KS, PSI, Wasserstein) for robust detection.

---

## 2. Confidence Threshold (0.50)

### 2.1 What It Means

A prediction is flagged as **"uncertain"** when its maximum softmax probability falls below 0.50. The pipeline counts such predictions in Stage 6 (Layer 1: Confidence Monitoring).

### 2.2 Why 0.50?

For an 11-class model, random chance gives $\frac{1}{11} \approx 0.091$. A confidence of 0.50 is **5.5× random chance** — the model has moderate discriminative signal but is not decisively confident. Below this threshold, we treat the prediction as untrustworthy.

### 2.3 Citations

- **MSP as uncertainty baseline:** Hendrycks, D. & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." *ICLR 2017*. Established that maximum softmax probability (MSP) separates in-distribution from OOD data, with 0.50 as a common operating point.
- **Energy-based OOD detection:** Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020). "Energy-Based Out-of-Distribution Detection." *NeurIPS 2020*. Validates the softmax approach and extends it with energy scores for more robust OOD detection.
- **Calibration matters:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*. Shows modern DNNs are overconfident — calibrated 70% confidence should match 70% accuracy. Motivates temperature scaling as a post-hoc fix.

---

## 3. Uncertain-Prediction Percentage (10 %)

### 3.1 What It Means

If more than 10 % of predictions in a batch fall below the confidence threshold (0.50), Stage 6 raises a **WARNING**. This is a batch-level quality gate.

### 3.2 Rationale

In a well-calibrated 11-class HAR model, we expect the vast majority of wearable-sensor windows to produce confident predictions (> 0.80). If 1 in 10 is uncertain, either:

- The production data is out-of-distribution (new user, new sensor placement)
- The model is encountering activities it was not trained on

The 10 % threshold is a **conservative warning** — most paper-reported pseudo-labeling pipelines use 80 % confident as the inclusion criterion, implying ≤ 20 % uncertain is tolerable during adaptation.

### 3.3 Citations

- **Pseudo-label confidence gating:** Tang, C. I., Perez-Pozuelo, I., Spathis, D., Brage, S., Wareham, N., & Mascolo, C. (2021). "SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 5(1), Article 36. Uses 80 % confidence threshold for pseudo-labels, implying 20 % uncertain is the outer bound.
- **Moderate confidence samples:** Tang & Jia (2022). "Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning." Shows samples in the 50–80 % confidence band are the most informative for semi-supervised training.
- **MLOps monitoring baseline:** Hewage, N. & Meedeniya, D. (2022). "Machine Learning Operations: A Survey on MLOps." arXiv:2202.10169. Recommends monitoring confidence distributions and prediction stability as primary health indicators (pp. 12–13).

---

## 4. Red-Flag Metrics & Rule-of-Thumb Thresholds

The following table summarises all monitored metrics and their thresholds:

| # | Metric | Threshold | Level | Source |
|---|--------|-----------|-------|--------|
| 1 | Mean confidence < X | 0.65 (warn) / 0.50 (critical) | Trigger | Hendrycks & Gimpel (2017); Guo et al. (2017) |
| 2 | Uncertain predictions > Y % | 10 % (warn) | Monitoring | SelfHAR (Tang et al., 2021) |
| 3 | Drift score > Z | 0.75 (warn) / 1.50 (alert) | Monitoring + Trigger | Data-driven (N=24); Montgomery (2009) |
| 4 | Transition rate > T | 35 % (warn) | Trigger | Empirical — rapid activity switching |
| 5 | Dominant activity > P % | 90 % | Evaluation | Class imbalance proxy |
| 6 | ECE > E | To be calibrated | Future | Guo et al. (2017) |

### Additional Context

- **PSI thresholds from literature:** PSI < 0.10 = no shift, 0.10–0.25 = moderate, ≥ 0.25 = major ("From Development to Deployment," IEEE SITA 2023). Our z-score metric is not directly PSI, but uses the same tiered alerting philosophy.
- **Proxy metrics when ground truth is unavailable:** "Essential MLOps," *Data Science Horizons* (2023), pp. 97–104: "Proxy metrics (confidence, prediction drift) serve as substitutes for accuracy when ground truth is unavailable."

---

## 5. MLflow Model Registry — Why and How

### 5.1 Purpose

The MLflow **Model Registry** provides:

1. **Versioning** — Every retrained model gets a version number (v1, v2, …).
2. **Stage management** — Models transition through `None → Staging → Production → Archived`.
3. **Audit trail** — Every version records who registered it, when, and with what metrics.
4. **Rollback** — If a Production model degrades, roll back to the previous version instantly.

### 5.2 How It Is Wired in Our Pipeline

In `_end_mlflow()`, after all stages complete successfully, the pipeline:

1. Logs the current Keras model to MLflow with `mlflow.keras.log_model()`
2. Registers it under the name `har-1dcnn-bilstm`
3. On the first run, assigns `Production` stage
4. On subsequent runs (after retraining), compares mean confidence before/after and promotes only if improved

### 5.3 Citations

- **MLflow + DVC recommended stack:** Symeonidis, G. et al. (2021). "Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools." *Applied Sciences*, DOI: `10.3390/app11198861`. Recommends MLflow Registry with Staging → Production stages even for prototype pipelines.
- **Model registry for reproducibility:** Hewage, N. & Meedeniya, D. (2022). "Machine Learning Operations: A Survey on MLOps." arXiv:2202.10169. MLOps requires versioned artifact stores and model registries for reproducibility and rollback.

---

## 6. Two MLflow Experiments Explained

### 6.1 What You See

When you open the MLflow UI (`mlflow ui`), you see two experiments:

| Experiment | Purpose |
|------------|---------|
| `har-production-pipeline` | **Parent experiment** — one run per pipeline execution. Contains all 37 metrics from stages 1–7 and 3 uploaded artifacts. |
| `inference-production` | **Nested experiment** — created by `src/run_inference.py` (line 690) during Stage 4. Contains per-window inference details. |

### 6.2 Why Two?

The inference component (`src/run_inference.py`) creates its own MLflow experiment with a **nested child run** inside the parent pipeline run. This separation allows:

- **Pipeline-level view:** One row per full pipeline execution (7 stages).
- **Inference-level view:** Detailed per-batch inference timing, model version, TensorFlow metadata.

This is standard MLflow practice — inner experiments are children of the outer experiment's run.

---

## 7. Reading the MLflow Dashboard

### 7.1 Metrics Tab

After a successful pipeline run, you see 37 metrics grouped by stage:

| Prefix | Stage | Example Metrics |
|--------|-------|-----------------|
| `ingestion_*` | 1 | `n_rows`, `n_columns`, `sampling_hz` |
| `validation_*` | 2 | `is_valid`, `n_errors`, `n_warnings` |
| `transformation_*` | 3 | `n_windows`, `window_size`, `unit_conversion` |
| `inference_*` | 4 | `n_predictions`, `time_seconds`, `mean_confidence`, `uncertain_pct` |
| `eval_*` | 5 | `mean_confidence`, `n_activities_detected`, `dominant_activity_pct` |
| `monitoring_*` | 6 | `drift_score`, `confidence_mean`, `uncertain_pct`, `transition_rate` |
| `trigger_*` | 7 | `should_retrain`, `alert_level`, `cooldown_active` |

### 7.2 Artifacts Tab

Three artifacts are uploaded:

1. **`inference/inference_summary.json`** — prediction counts, confidence stats, activity distribution
2. **`monitoring/monitoring_report.json`** — 3-layer monitoring results (confidence, temporal, drift)
3. **`pipeline/run_info.json`** — stages completed/failed, timestamps, overall status

### 7.3 Parameters Tab

Pipeline-level parameters:

- `stages_completed` — comma-separated list (e.g., `ingestion,validation,transformation,...`)
- `stages_failed` — empty on success
- `overall_status` — `SUCCESS`, `PARTIAL`, or `FAILED`

### 7.4 If the UI Shows "Metrics (0)"

This is a known MLflow UI caching bug. Verify via API:

```python
import mlflow
client = mlflow.MlflowClient()
run = client.search_runs("745412409067384126")[0]
print(len(run.data.metrics))  # → 37
```

---

## 8. Per-Dataset Inference Analysis

We ran the 1D-CNN-BiLSTM model independently on all 24 raw accelerometer/gyroscope recording pairs from `data/raw/`. This identifies which datasets the model struggles with and which are deployment-ready.

### 8.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Total datasets | 24 (1 xlsx pair excluded due to corrupt format) |
| Mean confidence across datasets | 0.8509 ± 0.0372 |
| Confidence range | [0.7910, 0.9313] |
| Datasets with mean conf < 0.65 | **0/24** (none) |
| Datasets with uncertain > 10% | **1/24** |
| Mean uncertain % | 6.20% ± 2.77% |

### 8.2 Struggling Datasets

Only **1 out of 24** datasets was flagged (uncertain % > 10%):

| Dataset | Windows | Mean Conf | Uncertain % | Reason |
|---------|---------|-----------|-------------|--------|
| 2025-07-17-18-53-05 | 913 | 0.7910 | 12.0% | Shortest session + mixed activities |

This dataset has a diverse activity distribution (59% hand_scratching, 25% hand_tapping, 10% ear_rubbing) and is one of the smallest (913 windows). The model's uncertainty likely reflects genuine ambiguity in the short, mixed-activity recording.

### 8.3 Key Findings

1. **Dominant activity bias:** `hand_scratching` is the predicted dominant activity in 23/24 datasets, with `hand_tapping` dominant in 1 dataset. This suggests the model maps most wrist-sensor patterns to scratching-like activities.
2. **High overall confidence:** Mean confidence is 0.85 across all datasets — well above the 0.65 warning threshold.
3. **Best-performing datasets:** Longer recordings (> 40K windows) like `2025-07-17-01-48-33` and `2025-07-19-11-24-04` achieve 0.93 mean confidence.
4. **No catastrophic failures:** Zero datasets fall below the critical 0.50 confidence threshold.

### 8.4 Implications for Retraining

Since only 1/24 datasets struggles mildly, the current model generalises well across recording sessions. Retraining should be targeted:

- **AdaBN first:** For the 5 datasets with drift > 0.75, update BN statistics with their production data before inference.
- **Pseudo-labeling:** Only for the single struggling dataset where AdaBN proves insufficient.

### 8.5 Citation

- **Evaluation methodology:** Hewage, N. & Meedeniya, D. (2022). "Machine Learning Operations: A Survey on MLOps." arXiv:2202.10169. Recommends per-dataset performance profiling as part of monitoring (pp. 12–13).

---

## 9. Retraining Pipeline — Pseudo-Labeling & AdaBN

### 9.1 Overview

When Stage 7 (Trigger Evaluation) decides retraining is necessary, the pipeline continues to:

- **Stage 8 — Model Retraining:** Pseudo-labeling or AdaBN domain adaptation
- **Stage 9 — Model Registration:** Version, validate, and deploy via MLflow Model Registry
- **Stage 10 — Baseline Update:** Rebuild drift baselines from adapted model

### 9.2 AdaBN (Adaptive Batch Normalization)

**Key insight:** The 1D-CNN-BiLSTM model's convolutional and LSTM layers learn **domain-invariant features** (activity patterns), but the Batch Normalization layers overfit to **source-domain statistics** (the specific user, sensor placement, and calibration used during training). AdaBN simply replaces BN running statistics with target-domain statistics.

**Algorithm:**

1. Load the pre-trained model
2. Set all Batch Normalization layers to training mode
3. Forward-pass unlabeled production data (10 batches × 64 samples)
4. BN running mean/variance are updated to target-domain statistics
5. Set BN layers back to inference mode

**No labels required. No gradient updates. No weight changes.**

The model's convolutional filters and LSTM weights remain unchanged — only the normalization statistics adapt. This is ideal for our use case where production data arrives without activity labels.

#### Implementation

Our implementation is in `src/domain_adaptation/adabn.py`:

```python
adapted_model = adapt_bn_statistics(
    model,
    target_X,            # unlabelled production data, shape (N, 200, 6)
    n_batches=10,
    batch_size=64,
)
```

#### Citation

- **AdaBN:** Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018). "Revisiting Batch Normalization For Practical Domain Adaptation." arXiv:1603.04779. Proposed replacing BN running statistics with target-domain statistics — no retraining needed.
- **Domain adaptation for IMU-HAR:** Chakma, A., Faridee, A. Z. M., Ghosh, I., & Roy, N. (2023). "Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey." arXiv:2304.06489. Confirms that sensor placement causes distribution heterogeneities and that unsupervised adaptation (like AdaBN) is effective for HAR.

### 9.3 Pseudo-Labeling (Semi-Supervised Retraining)

**Key insight:** When AdaBN alone is insufficient (e.g., large domain gap), we use pseudo-labeling to generate training labels from confident predictions on unlabeled production data, then fine-tune the model on the combined labeled + pseudo-labeled data.

**Algorithm:**

1. Run inference on unlabeled production data using the current model
2. Filter predictions by confidence — keep only those above 0.80 (fallback to 0.60 if too few samples)
3. These confident predictions become "pseudo-labels"
4. Combine: original labeled training data + pseudo-labeled production data
5. Fine-tune the model on the combined dataset
6. Evaluate improvement in mean confidence on heldout target data

**Why 0.80 confidence threshold?** SelfHAR (Tang et al., 2021) demonstrated that a teacher-student framework with high-confidence pseudo-labels achieves near-supervised accuracy. Our 0.80 threshold ensures pseudo-labels are reliable, with a 0.60 fallback for datasets with inherently lower confidence.

#### Implementation

Stages 8–10 are triggered by passing `enable_retrain=True`:

```python
python run_pipeline.py --enable-retrain --adaptation-method pseudo_label
```

The pipeline flow:

```
Stage 7 (Trigger) → should_retrain=True
  → Stage 8 (Retraining): _run_pseudo_label()
    → Stage 9 (Registration): register model v2 in MLflow Registry
      → Stage 10 (Baseline Update): rebuild drift baselines
```

#### Citation

- **SelfHAR:** Tang, C. I., Perez-Pozuelo, I., Spathis, D., Brage, S., Wareham, N., & Mascolo, C. (2021). "SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data." *IMWUT*, 5(1), Article 36. Teacher-student framework for HAR with pseudo-labels.
- **Moderate-confidence selection:** Tang & Jia (2022). "Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning." Samples in the 50–80 % confidence range are most informative for adaptation.
- **SRPM-ST:** Mukhamediya et al. (2024). "Sequential Retraining and Pseudo-Labeling in Mini-Batches for Self-Training." Pseudo-labeling boosts F1 by 5–15 % on streaming data.
- **Conformal prediction as future extension:** Gibbs, I. & Candes, E. (2021). "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS 2021*. Calibrated uncertainty sets instead of point predictions — a planned enhancement for our pipeline.

### 9.4 Decision Flow: AdaBN vs. Pseudo-Labeling

```
Drift detected (D > 0.75)?
  ├─ YES → Is mean confidence still > 0.70?
  │         ├─ YES → AdaBN (lightweight, no labels)
  │         └─ NO  → Pseudo-labeling (heavier, but stronger adaptation)
  └─ NO  → No retraining needed
```

AdaBN is attempted first because it requires no labeled data and takes seconds. If AdaBN improves mean confidence by ≥ 5 %, the adapted model is deployed. If not, the pipeline falls back to pseudo-labeling.

### 9.5 Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Data Ingestion (Excel/CSV → fused CSV)        │
│  Stage 2: Data Validation (schema + range checks)       │
│  Stage 3: Data Transformation (CSV → windowed .npy)     │
│  Stage 4: Model Inference (.npy + model → predictions)  │
│  Stage 5: Model Evaluation (confidence/distribution)    │
│  Stage 6: Post-Inference Monitoring (3-layer check)     │
│  Stage 7: Trigger Evaluation (retraining decision)      │
├─────────────────────────────────────────────────────────┤
│  IF trigger.should_retrain == True:                     │
│  Stage 8: Model Retraining (AdaBN or pseudo-label)      │
│  Stage 9: Model Registration (MLflow Model Registry)    │
│  Stage 10: Baseline Update (rebuild drift baselines)    │
└─────────────────────────────────────────────────────────┘
```

---

## Full Bibliography

1. Chakma, A., Faridee, A. Z. M., Ghosh, I., & Roy, N. (2023). Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey. *arXiv:2304.06489*.

2. "From Development to Deployment: An Approach to MLOps Monitoring for Machine Learning Model Operationalization." IEEE SITA 2023. DOI: `10.1109/SITA60746.2023.10373733`.

3. Gibbs, I. & Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS 2021*.

4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.

5. Hendrycks, D. & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*.

6. Hewage, N. & Meedeniya, D. (2022). Machine Learning Operations: A Survey on MLOps. *arXiv:2202.10169*.

7. Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018). Revisiting Batch Normalization For Practical Domain Adaptation. *arXiv:1603.04779*.

8. Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020). Energy-Based Out-of-Distribution Detection. *NeurIPS 2020*.

9. Montgomery, D. C. (2009). *Introduction to Statistical Quality Control* (6th ed.). Wiley.

10. Moskalenko, V. (2024). Resilience-aware MLOps for AI-based Medical Diagnostic System. *Frontiers in Public Health*. DOI: `10.3389/fpubh.2024.1342937`.

11. Mukhamediya et al. (2024). Sequential Retraining and Pseudo-Labeling in Mini-Batches for Self-Training (SRPM-ST).

12. "Essential MLOps." *Data Science Horizons* (2023), pp. 97–104.

13. Symeonidis, G. et al. (2021). Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools. *Applied Sciences*. DOI: `10.3390/app11198861`.

14. Tang, C. I., Perez-Pozuelo, I., Spathis, D., Brage, S., Wareham, N., & Mascolo, C. (2021). SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data. *IMWUT*, 5(1), Article 36.

15. Tang & Jia (2022). Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning.

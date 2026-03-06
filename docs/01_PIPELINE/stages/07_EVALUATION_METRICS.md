# Stage 6: Evaluation & Metrics Logging

**Pipeline Stage:** Compute & log all metrics for training and monitoring  
**Input:** Model predictions, ground truth (if available), production outputs  
**Output:** MLflow metrics, evaluation reports, baseline comparisons

---

## Complete Metrics Inventory

### Training Metrics (Require Labels)

| Metric | When Used | MLflow Key | Purpose |
|--------|-----------|------------|---------|
| **Accuracy** | Training/Eval | `accuracy` | Overall correctness |
| **Macro Precision** | Training/Eval | `macro_precision` | Per-class precision averaged |
| **Macro Recall** | Training/Eval | `macro_recall` | Per-class recall averaged |
| **Macro F1** | Training/Eval | `macro_f1` | Balanced precision-recall |
| **Weighted F1** | Training/Eval | `weighted_f1` | F1 weighted by class support |
| **Cohen's Kappa** | Training/Eval | `cohen_kappa` | Agreement corrected for chance |
| **Per-Class F1** | Training/Eval | `f1_class_{N}` | Individual class performance |

---

### Monitoring Metrics (NO Labels Required)

| Metric | When Used | MLflow Key | Purpose |
|--------|-----------|------------|---------|
| **Mean Confidence** | Inference | `mean_confidence` | Model certainty |
| **Std Confidence** | Inference | `std_confidence` | Certainty variance |
| **Uncertain Ratio** | Inference | `uncertain_ratio` | % windows below threshold |
| **Mean Entropy** | Inference | `mean_entropy` | Prediction uncertainty |
| **Mean Margin** | Inference | `mean_margin` | Gap top-1 vs top-2 |
| **Flip Rate** | Inference | `flip_rate` | Prediction stability |
| **Mean Dwell Time** | Inference | `mean_dwell_time` | Activity duration |
| **KS Statistic (per channel)** | Monitoring | `ks_{channel}` | Distribution shift |
| **PSI (per channel)** | Monitoring | `psi_{channel}` | Population stability |
| **Wasserstein (per channel)** | Monitoring | `wasserstein_{channel}` | Earth mover's distance |

---

### Data Quality Metrics

| Metric | When Used | MLflow Key | Purpose |
|--------|-----------|------------|---------|
| **NaN Ratio** | Preprocessing | `nan_ratio` | Missing data % |
| **Variance Collapse Count** | QC | `variance_collapse_count` | Sensor failures |
| **Range Violation Count** | QC | `range_violations` | Out-of-range values |
| **Total Windows** | All stages | `n_windows` | Data volume |
| **Class Distribution** | Training/Eval | `class_dist_{N}` | Balance check |

---

## MLflow Logging Strategy

### Experiment Organization

```
mlruns/
├── training/                    # Experiment: model training runs
│   ├── run_001/
│   │   ├── params/              # Hyperparameters
│   │   ├── metrics/             # Training metrics
│   │   ├── artifacts/           # Model, config, plots
│   │   └── tags/                # Metadata
│   └── run_002/
│
├── monitoring/                  # Experiment: production monitoring
│   ├── batch_2026-01-15/
│   │   ├── metrics/             # Confidence, drift, temporal
│   │   └── artifacts/           # Reports, alerts
│   └── batch_2026-01-16/
│
└── evaluation/                  # Experiment: when labels available
    └── eval_weekly/
        ├── metrics/             # Full classification metrics
        └── artifacts/           # Confusion matrix, reports
```

---

### What to Log Where

**Training Experiment:**
```python
with mlflow.start_run(experiment_id=training_exp_id):
    # Parameters
    mlflow.log_param("model_type", "1DCNN_BiLSTM")
    mlflow.log_param("window_size", 200)
    mlflow.log_param("n_classes", 11)
    
    # Training metrics (per epoch)
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("val_accuracy", acc, step=epoch)
    
    # Final metrics
    mlflow.log_metrics({
        "test_accuracy": 0.934,
        "macro_f1": 0.918,
        "weighted_f1": 0.932,
        "cohen_kappa": 0.927
    })
    
    # Artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("model.h5")
```

**Monitoring Experiment:**
```python
with mlflow.start_run(experiment_id=monitoring_exp_id, run_name=batch_id):
    # Confidence metrics
    mlflow.log_metrics({
        "mean_confidence": 0.862,
        "uncertain_ratio": 0.079,
        "mean_entropy": 0.407
    })
    
    # Drift metrics (per channel)
    for channel in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
        mlflow.log_metrics({
            f"ks_{channel}": ks_values[channel],
            f"psi_{channel}": psi_values[channel]
        })
    
    # Temporal metrics
    mlflow.log_metrics({
        "flip_rate": 0.193,
        "mean_dwell_time": 10.33
    })
    
    # Reports
    mlflow.log_artifact("monitoring_report.json")
```

---

## When Do We Need Labels?

| Stage | Labels Needed? | Why? |
|-------|----------------|------|
| QC Checks | NO | Checking data quality, not correctness |
| Inference | NO | Just generating predictions |
| Confidence Monitoring | NO | Using prediction probabilities |
| Drift Detection | NO | Comparing input distributions |
| Temporal Analysis | NO | Analyzing prediction sequences |
| Accuracy/F1 Evaluation | **YES** | Need truth to measure error |
| Confusion Matrix | **YES** | Need truth for class-level analysis |
| Retrain Validation | **YES** | Need truth to verify improvement |

**Key Insight:** For production monitoring, we can detect PROBLEMS without labels, but we cannot measure ACCURACY without labels.

---

## Metrics Comparison Structure

### Compare Against Training Baseline

```json
{
  "comparison": {
    "baseline_accuracy": 0.934,
    "baseline_macro_f1": 0.918,
    "current_mean_confidence": 0.862,
    "confidence_vs_accuracy_gap": 0.072,
    "interpretation": "Confidence lower than training accuracy - possible OOD data"
  }
}
```

### Compare Against Recent Rolling Reference

```json
{
  "rolling_comparison": {
    "reference_period": "2026-01-08 to 2026-01-14",
    "reference_mean_confidence": 0.891,
    "current_mean_confidence": 0.862,
    "change_percent": -3.3,
    "interpretation": "Slight decline in confidence over past week"
  }
}
```

---

## What to Do Checklist

- [ ] Log all training metrics to MLflow training experiment
- [ ] Log all monitoring metrics to MLflow monitoring experiment
- [ ] Set up separate experiment for labeled evaluation (when available)
- [ ] Create metric comparison functions (vs baseline, vs rolling)
- [ ] Configure automatic metric export to Prometheus
- [ ] Generate weekly metrics summary report
- [ ] Set up metric trend visualization in Grafana

---

## Evidence from Papers

**[ICTH 2025: Wearable IMU HAR MLOps | PDF: papers/new paper/ICTH_2025_Oleh_Paper_MLOps_Summary.md]**
- 93.4% accuracy on ADAMSense
- F1 scores per class logged to MLflow

**[From Development to Deployment: MLOps Monitoring | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**
- PSI thresholds established for drift
- Metric logging to experiment tracker required

**[MLOps: A Taxonomy and a Methodology, 2022 | PDF: papers/mlops_production/MLOps_A_Taxonomy_and_a_Methodology 2022.pdf]**
- Comprehensive metric taxonomy for ML systems
- Separation of training vs operational metrics

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add calibration metrics when labels available | Medium | Better confidence interpretation |
| **HIGH** | Create metric trend dashboard in Grafana | Medium | Visual anomaly detection |
| **MEDIUM** | Add metric comparison automation | Low | Faster degradation detection |
| **MEDIUM** | Implement metric anomaly alerting | Medium | Proactive monitoring |
| **LOW** | Add per-user metric breakdown | High | Personalization insights |

---

**Previous Stage:** [06_MONITORING_DRIFT.md](06_MONITORING_DRIFT.md)  
**Next Stage:** [08_ALERTING_RETRAINING.md](08_ALERTING_RETRAINING.md)

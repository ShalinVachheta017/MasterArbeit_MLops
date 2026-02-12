# Stage 5: Monitoring & Drift Detection

**Pipeline Stage:** Monitor model behavior WITHOUT ground-truth labels  
**Input:** Predictions, probabilities, production data statistics  
**Output:** Monitoring reports, drift alerts, degradation signals

---

## The Core Question: How Do We Know the Model is Degraded WITHOUT Labels?

**Answer:** Use **proxy metrics** that indicate problems without needing ground truth.

### The Key Insight

When a model degrades, it shows symptoms even without labels:
- **Less confident** in its predictions
- **More uncertain** (higher entropy)
- **Inconsistent** over time (high flip rate)
- **Input data has shifted** from what it learned

---

## Complete List of Proxy Metrics

### Category 1: Confidence Metrics (Per-Window)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Max Probability (Confidence)** | `max(softmax_output)` | How sure the model is |
| **Entropy** | `-Σ p_i × log(p_i)` | Uncertainty in distribution |
| **Margin** | `p_top1 - p_top2` | Gap between top two classes |

**Thresholds:**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Mean Confidence | > 0.80 | 0.70 - 0.80 | < 0.70 |
| Uncertain Ratio | < 0.10 | 0.10 - 0.20 | > 0.20 |
| Mean Entropy | < 1.0 | 1.0 - 1.5 | > 1.5 |
| Mean Margin | > 0.30 | 0.15 - 0.30 | < 0.15 |

---

### Category 2: Drift Metrics (Distribution Comparison)

| Metric | What it Measures | Formula/Method |
|--------|------------------|----------------|
| **KS Statistic** | Max difference between CDFs | `scipy.stats.ks_2samp()` |
| **Wasserstein Distance** | "Earth mover's" distance | `scipy.stats.wasserstein_distance()` |
| **PSI** | Population Stability Index | `Σ (actual% - expected%) × ln(actual%/expected%)` |
| **Mean Shift** | Change in mean (normalized) | `|μ_prod - μ_baseline| / σ_baseline` |
| **Variance Collapse** | Std much lower than expected | `σ_prod / σ_baseline < 0.1` |

**Thresholds:**

| Metric | No Drift | Moderate | Significant |
|--------|----------|----------|-------------|
| KS Statistic | < 0.10 | 0.10 - 0.20 | > 0.20 |
| Wasserstein | < 0.25 | 0.25 - 0.50 | > 0.50 |
| PSI | < 0.10 | 0.10 - 0.25 | > 0.25 |
| Mean Shift | < 0.25σ | 0.25σ - 0.50σ | > 0.50σ |

---

### Category 3: Temporal Metrics (Sequence Analysis)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Flip Rate** | `n_transitions / n_windows` | How often prediction changes |
| **Mean Dwell Time** | `avg(bout_duration)` | How long each activity lasts |
| **Min Dwell Time** | `min(bout_duration)` | Shortest activity bout |
| **Transition Count** | Raw count of class changes | Activity stability |

**Thresholds:**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Flip Rate | < 0.20 | 0.20 - 0.35 | > 0.35 |
| Min Dwell Time | > 4s | 2s - 4s | < 2s |
| Mean Dwell Time | > 10s | 5s - 10s | < 5s |

---

### Category 4: Data Quality Metrics

| Metric | What it Detects |
|--------|-----------------|
| Variance Collapse | Sensor failure, wrong scaler, static data |
| Range Violations | Sensor malfunction, unit conversion error |
| NaN Ratio | Data pipeline issues |
| Constant Channels | Sensor disconnection |

---

## How These Metrics Detect Degradation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEGRADATION DETECTION FLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Production predictions + data stats                             │
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │ Check Confidence│───▶ Low? ───▶ Model is uncertain                  │
│  │ Metrics         │              (may be seeing OOD data)             │
│  └─────────────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │ Check Drift     │───▶ High? ───▶ Input distribution has shifted    │
│  │ Metrics         │              (model may not generalize)           │
│  └─────────────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │ Check Temporal  │───▶ High flip? ───▶ Model is unstable            │
│  │ Metrics         │                    (predictions not plausible)    │
│  └─────────────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  AGGREGATE: Multiple bad signals = HIGH confidence of degradation      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring Report Structure

```json
{
  "timestamp": "2026-01-15T13:13:21Z",
  "batch_id": "2026-01-15_batch_001",
  "overall_status": "WARNING",
  
  "confidence_metrics": {
    "mean_confidence": 0.862,
    "std_confidence": 0.187,
    "uncertain_ratio": 0.079,
    "mean_entropy": 0.407,
    "mean_margin": 0.781,
    "status": "PASS"
  },
  
  "drift_metrics": {
    "n_channels_with_drift": 3,
    "drift_channels": ["Ax", "Ay", "Az"],
    "max_ks_statistic": 0.514,
    "max_wasserstein": 1.039,
    "status": "WARNING"
  },
  
  "temporal_metrics": {
    "flip_rate": 0.193,
    "mean_dwell_time": 10.33,
    "min_dwell_time": 2.0,
    "n_transitions": 504,
    "status": "PASS"
  },
  
  "alerts": [
    {
      "type": "DRIFT_WARNING",
      "message": "3 channels show distribution shift (KS > 0.10)",
      "severity": "WARNING",
      "recommended_action": "Investigate data source; consider retraining"
    }
  ]
}
```

---

## What to Do Checklist

- [ ] Compute all confidence metrics for each batch
- [ ] Compare input distributions to frozen baseline
- [ ] Analyze temporal consistency of predictions
- [ ] Generate monitoring report JSON
- [ ] Export metrics to MLflow (monitoring experiment)
- [ ] Configure alert thresholds in config file
- [ ] Set up Prometheus export for dashboarding

---

## Evidence from Papers

**[NeurIPS 2020: Energy-based OOD Detection | PDF: papers/new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf]**
- Confidence and entropy are effective proxy metrics for OOD
- No labels required for detection

**[From Development to Deployment: MLOps Monitoring, 2023 | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**
- PSI thresholds (0.10, 0.25) are industry standard
- Multiple metrics provide robust detection

**[A Two-Stage Anomaly Detection Framework | PDF: papers/research_papers/76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf]**
- Combining multiple detection methods improves accuracy
- Temporal consistency catches point anomaly detectors miss

---

## Did We Miss Anything in MLOps Papers?

Based on review of local papers, here are metrics we **SHOULD** consider adding:

| Metric | Source Paper | Status |
|--------|--------------|--------|
| Energy Score (OOD) | NeurIPS 2020 | **Not implemented** - could add |
| Reconstruction Error (autoencoder) | Various | **Not implemented** - requires extra model |
| Feature-space distance | Domain adaptation papers | **Partially** - using Wasserstein |
| Calibration (ECE, Brier) | When Does Optimizing Loss Yield Calibration | **Not implemented** - needs calibration set |
| Conformal prediction sets | NeurIPS 2021 Adaptive Conformal | **Not implemented** - needs calibration set |

**Recommendation:** Energy score is easy to add (just logits manipulation). Others require model changes or calibration data.

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add energy-based OOD score | Low | Better OOD detection |
| **HIGH** | Multi-channel drift aggregation | Low | Reduce false positives |
| **MEDIUM** | Add calibration metrics (if labels available) | Medium | Better confidence interpretation |
| **MEDIUM** | Trend analysis over batches | Medium | Early warning system |
| **LOW** | Feature-space OOD (requires embedding) | High | More sophisticated detection |

---

**Previous Stage:** [05_INFERENCE.md](05_INFERENCE.md)  
**Next Stage:** [07_EVALUATION_METRICS.md](07_EVALUATION_METRICS.md)

# Stage 3: Training & Baseline Creation

> **⚠️ STATUS (2026-01-30): PARTIAL - Baseline exists, but NO training script!**
> 
> **TODO:**
> - [ ] Create `src/train.py` with 5-fold CV
> - [ ] Add MLflow experiment tracking
> - [ ] Create `src/retrain_pseudo.py` for curriculum pseudo-labeling
> - [ ] Implement model restart between retraining cycles (prevents confirmation bias)

**Pipeline Stage:** Train model and create reference baselines  
**Input:** Labeled training data  
**Output:** Trained model, scaler, frozen baseline statistics

---

## Key Question: Baseline vs Reference — Are They the Same?

**NO, they serve DIFFERENT purposes:**

| Aspect | Frozen Training Baseline | Rolling Production Reference |
|--------|--------------------------|------------------------------|
| **What it is** | Statistics from training data | Statistics from recent production |
| **Purpose** | Detect drift from what model learned | Detect sudden changes in production |
| **When created** | Once, during training | Updated after each batch |
| **When updated** | Only on model retraining | Continuously (rolling window) |
| **Stored in** | `baseline_stats.json` | `rolling_reference.json` |
| **Contains** | Per-channel: mean, std, percentiles, histogram | Last N batches: mean, std, trends |

---

## Why We Need Both

### Scenario 1: Gradual Drift
```
Training baseline: mean = 0.0
Week 1 production:  mean = 0.1  → Small drift from baseline
Week 2 production:  mean = 0.2  → More drift from baseline
Week 3 production:  mean = 0.3  → Significant drift from baseline ⚠️

Rolling reference tracks: 0.1 → 0.2 → 0.3 (gradual change, no sudden jump)
```
**Baseline catches this** — production is drifting from training distribution.

### Scenario 2: Sudden Change
```
Training baseline: mean = 0.0
Week 1-5 production: mean ≈ 0.2 (stable, small drift)
Week 6 production:   mean = 1.5  → Sudden spike! ⚠️

Baseline comparison: 0.2 vs 0.0 = moderate drift (may not alert)
Rolling reference: 0.2 vs 1.5 = HUGE change → immediate alert!
```
**Rolling reference catches this** — sudden anomaly in production.

---

## Baseline Statistics Structure

```json
{
  "created_at": "2026-01-15T12:58:26Z",
  "source_file": "data/raw/all_users_data_labeled.csv",
  "n_samples": 385326,
  "sensor_columns": ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"],
  "per_channel": {
    "Ax": {
      "mean": 3.22,
      "std": 6.57,
      "min": -26.85,
      "max": 31.95,
      "percentile_5": -9.42,
      "percentile_25": -1.53,
      "percentile_50": 6.38,
      "percentile_75": 8.76,
      "percentile_95": 9.53,
      "histogram": {
        "counts": [1, 0, 0, ...],
        "bin_edges": [-26.85, -25.67, ...]
      }
    },
    // ... other channels
  },
  "scaler_path": "models/scaler.pkl",
  "scaler_hash": "sha256:abc123...",
  "model_version": "1dcnn_bilstm_v1.0"
}
```

---

## Rolling Reference Structure

```json
{
  "last_updated": "2026-01-28T10:30:00Z",
  "window_size": 5,
  "batches": [
    {
      "batch_id": "2026-01-24_batch_001",
      "timestamp": "2026-01-24T14:00:00Z",
      "per_channel_mean": [0.12, -0.05, 0.03, 0.01, -0.02, 0.01],
      "per_channel_std": [0.95, 1.02, 0.98, 1.01, 0.99, 1.00]
    },
    // ... last N batches
  ],
  "aggregated": {
    "rolling_mean": [0.11, -0.04, 0.02, 0.01, -0.01, 0.01],
    "rolling_std": [0.02, 0.03, 0.02, 0.01, 0.02, 0.01],
    "trend": "stable"
  }
}
```

---

## Comparison Logic

```python
def check_drift(production_stats, baseline, rolling_ref):
    """
    Check drift against both references.
    
    Returns:
        - "normal": No significant drift
        - "gradual_drift": Drifting from baseline (may need retraining)
        - "sudden_change": Anomaly vs recent production (investigate immediately)
    """
    # Compare to frozen baseline
    baseline_drift = compute_ks_statistic(production_stats, baseline)
    
    # Compare to rolling reference
    rolling_diff = compute_difference(production_stats, rolling_ref)
    
    if rolling_diff > SUDDEN_CHANGE_THRESHOLD:
        return "sudden_change"  # Alert immediately!
    elif baseline_drift > DRIFT_THRESHOLD:
        return "gradual_drift"  # Consider retraining
    else:
        return "normal"
```

---

## When to Update Each

| Event | Baseline | Rolling Reference |
|-------|----------|-------------------|
| New production batch | ❌ No change | ✅ Add to rolling window |
| Model retrained | ✅ Regenerate from new training data | ✅ Reset to empty |
| Data issue detected | ❌ No change | ❌ Exclude bad batch |
| Regular schedule | ❌ No change | ✅ Auto-update |

---

## What to Do Checklist

- [ ] Generate baseline_stats.json during training
- [ ] Include scaler hash in baseline for validation
- [ ] Implement rolling reference update after each batch
- [ ] Set rolling window size (e.g., last 5 batches)
- [ ] Compare production to BOTH references
- [ ] Alert on sudden changes (rolling ref)
- [ ] Track gradual drift (baseline comparison)

---

## Evidence from Papers

**[From Development to Deployment: MLOps Monitoring, 2023 | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**
- Baseline datasets are essential for drift detection
- Both fixed and rolling references recommended

**[Resilience-aware MLOps for AI-based medical diagnostic system, 2024 | PDF: papers/mlops_production/Resilience-aware MLOps for AI-based medical diagnostic system  2024.pdf]**
- Healthcare ML requires multiple reference comparisons
- Sudden change detection prevents cascading errors

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Auto-generate baseline on training completion | Low | Ensure baseline always exists |
| **HIGH** | Add baseline validation (schema, completeness) | Low | Prevent corrupt baselines |
| **MEDIUM** | Configurable rolling window size | Low | Tune sensitivity |
| **MEDIUM** | Baseline visualization dashboard | Medium | Better understanding of training data |
| **LOW** | Multi-model baseline support | Medium | Support model versioning |

---

**Previous Stage:** [03_QC_VALIDATION.md](03_QC_VALIDATION.md)  
**Next Stage:** [05_INFERENCE.md](05_INFERENCE.md)

# Stage 9: Improvements Roadmap

**Purpose:** Identify gaps in current pipeline and prioritize improvements  
**Scope:** Based on local papers, existing implementation, and MLOps best practices  
**Target:** Complete production-ready HAR MLOps system

---

## Current State Assessment

### What We Have ✅

| Component | Status | Location |
|-----------|--------|----------|
| Data versioning (DVC) | ✅ Implemented | `data/*.dvc` |
| Preprocessing pipeline | ✅ Implemented | `src/preprocess_data.py` |
| QC checks | ✅ Implemented | `scripts/preprocess_qc.py` |
| Training pipeline | ✅ Implemented | `src/train_model.py` |
| Inference pipeline | ✅ Implemented | `src/run_inference.py` |
| Baseline generation | ✅ Implemented | `scripts/create_normalized_baseline.py` |
| MLflow tracking | ✅ Implemented | `src/mlflow_tracking.py` |
| Confidence monitoring | ✅ Implemented | `scripts/post_inference_monitoring.py` |
| Drift detection | ✅ Implemented | `scripts/post_inference_monitoring.py` |
| Temporal analysis | ✅ Implemented | `scripts/post_inference_monitoring.py` |
| Docker containers | ✅ Partial | `docker/Dockerfile.*` |
| Documentation | ✅ Implemented | `docs/` |

### What's Missing ❌

| Component | Status | Priority |
|-----------|--------|----------|
| Domain adaptation training | ❌ Not implemented | HIGH |
| Energy-based OOD detection | ❌ Not implemented | HIGH |
| Automatic retraining trigger | ❌ Not implemented | HIGH |
| Prometheus metrics export | ❌ Not implemented | MEDIUM |
| Grafana dashboard | ❌ Not implemented | MEDIUM |
| Calibration metrics | ❌ Not implemented | MEDIUM |
| A/B testing infrastructure | ❌ Not implemented | MEDIUM |
| Automatic rollback | ❌ Not implemented | LOW |
| Canary deployments | ❌ Not implemented | LOW |

---

## Priority 1: Must Have (Before Thesis Submission)

### 1.1 Energy-Based OOD Detection

**What:** Add energy score computation to monitoring

**Why:** Better OOD detection than softmax confidence alone (NeurIPS 2020)

**Effort:** LOW (2-4 hours)

**Implementation:**
```python
def compute_energy_score(logits, temperature=1.0):
    """
    Energy score: -T * log(sum(exp(logits/T)))
    Lower energy = more in-distribution
    """
    return -temperature * np.log(np.sum(np.exp(logits / temperature), axis=-1))
```

**Files to modify:**
- `scripts/post_inference_monitoring.py` - Add energy computation
- Monitoring reports - Add energy metrics

---

### 1.2 Gravity Removal Experiment

**What:** Run 4-6 production datasets with and without gravity removal

**Why:** Garmin data doesn't have gravity removed; need to determine impact

**Effort:** MEDIUM (1-2 days)

**Experiment Design:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│              GRAVITY REMOVAL EXPERIMENT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Datasets to test:                                                      │
│  1. ADAMSense (original training data)                                 │
│  2. MobiAct (public, multi-activity)                                   │
│  3. UCI HAR (public, smartphone)                                       │
│  4. Garmin data (your production target)                               │
│  5. Custom test set (if available)                                     │
│                                                                         │
│  For each dataset:                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dataset X                                                        │   │
│  │                                                                  │   │
│  │  ├── Preprocess WITH gravity removal                            │   │
│  │  │   └── Train model A                                          │   │
│  │  │   └── Evaluate on all datasets                               │   │
│  │  │                                                               │   │
│  │  └── Preprocess WITHOUT gravity removal                         │   │
│  │      └── Train model B                                          │   │
│  │      └── Evaluate on all datasets                               │   │
│  │                                                                  │   │
│  │  Compare: Accuracy, F1, cross-dataset generalization            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Key questions:                                                         │
│  - Does gravity removal help or hurt cross-dataset transfer?           │
│  - Which is better for Garmin deployment?                              │
│  - Is there a hybrid approach?                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Hypothesis:** Model trained WITHOUT gravity removal may generalize better to Garmin data.

---

### 1.3 Automatic Retrain Trigger

**What:** Script that checks monitoring metrics and triggers retrain

**Why:** Currently manual; should be automated for production

**Effort:** MEDIUM (1 day)

**Implementation outline:**
```python
def check_retrain_triggers(monitoring_reports, config):
    """
    Check if retraining should be triggered based on monitoring metrics.
    """
    triggers = []
    
    # Check consecutive warnings
    recent_reports = monitoring_reports[-config["lookback_batches"]:]
    
    warning_count = sum(1 for r in recent_reports 
                       if r["overall_status"] == "WARNING")
    
    if warning_count >= config["warning_threshold"]:
        triggers.append({
            "type": "CONSECUTIVE_WARNINGS",
            "count": warning_count,
            "action": "RETRAIN"
        })
    
    # Check critical metrics
    latest = recent_reports[-1]
    if latest["confidence_metrics"]["mean_confidence"] < config["critical_confidence"]:
        triggers.append({
            "type": "LOW_CONFIDENCE",
            "value": latest["confidence_metrics"]["mean_confidence"],
            "action": "RETRAIN"
        })
    
    return triggers
```

---

## Priority 2: Should Have (Production Quality)

### 2.1 Prometheus Metrics Export

**What:** Export real-time metrics for Grafana dashboarding

**Effort:** MEDIUM (4-8 hours)

**Files to create:**
- `src/metrics_exporter.py` - Prometheus client
- `docker/prometheus.yml` - Prometheus config
- `docker/grafana/` - Dashboard JSON

---

### 2.2 Calibration Metrics (When Labels Available)

**What:** Compute ECE, Brier score when periodic labels are available

**Effort:** LOW (2-4 hours)

**Implementation:**
```python
def expected_calibration_error(confidences, predictions, labels, n_bins=10):
    """
    ECE: Expected Calibration Error
    Measures how well confidence matches accuracy
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_accuracy = (predictions[mask] == labels[mask]).mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    
    return ece / len(confidences)
```

---

### 2.3 Domain Adaptation Training

**What:** Implement MMD loss and/or DANN for unsupervised adaptation

**Effort:** HIGH (3-5 days)

**Key papers to reference:**
- papers/domain_adaptation/ folder
- ICTH 2025 paper methodology

**Implementation outline:**
```python
def mmd_loss(source_features, target_features):
    """
    Maximum Mean Discrepancy loss for domain adaptation
    """
    source_mean = tf.reduce_mean(source_features, axis=0)
    target_mean = tf.reduce_mean(target_features, axis=0)
    return tf.reduce_sum(tf.square(source_mean - target_mean))
```

---

## Priority 3: Nice to Have (Future Work)

### 3.1 A/B Testing Infrastructure

**What:** Run multiple model versions in parallel, compare metrics

**Effort:** HIGH (1 week)

---

### 3.2 Automatic Rollback

**What:** Automatically revert to previous model on critical alerts

**Effort:** HIGH (3-5 days)

---

### 3.3 Conformal Prediction

**What:** Provide prediction sets with coverage guarantees

**Effort:** HIGH (1 week)

**Paper reference:** NeurIPS 2021 - Adaptive Conformal Inference

---

## Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUGGESTED IMPLEMENTATION TIMELINE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Week 1-2: Priority 1 (Must Have)                                       │
│  ├── Day 1-2: Energy-based OOD detection                               │
│  ├── Day 3-5: Gravity removal experiment (setup)                       │
│  ├── Day 6-8: Gravity removal experiment (run + analyze)               │
│  └── Day 9-10: Automatic retrain trigger                               │
│                                                                         │
│  Week 3-4: Priority 2 (Should Have)                                     │
│  ├── Day 1-2: Prometheus metrics export                                │
│  ├── Day 3-4: Grafana dashboard                                        │
│  ├── Day 5: Calibration metrics                                        │
│  └── Day 6-10: Domain adaptation (start)                               │
│                                                                         │
│  Week 5+: Priority 3 (As time permits)                                  │
│  └── Domain adaptation completion, A/B testing, etc.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Gap Analysis: Papers vs Implementation

### Techniques in Papers But NOT Implemented

| Technique | Paper Source | Gap Level |
|-----------|--------------|-----------|
| Energy-based OOD | NeurIPS 2020 | **EASY TO ADD** |
| Conformal prediction | NeurIPS 2021 | HARD |
| Domain-adversarial (DANN) | Domain adaptation papers | MEDIUM |
| Self-training pseudo-labels | Various | MEDIUM |
| Attention mechanisms | HAR attention papers | HARD (model change) |
| Multi-task learning | MTL papers | HARD (model change) |
| Transfer learning fine-tune | Various | MEDIUM |

### Our Strengths vs Papers

| Our Implementation | Paper Comparison |
|--------------------|------------------|
| Full MLOps pipeline | Most papers focus on model only |
| DVC versioning | Many papers don't version data |
| Comprehensive monitoring | Rare in academic papers |
| Production-ready QC | Unique to this thesis |
| Complete drift detection | More thorough than most papers |

---

## Thesis-Specific Recommendations

### For Writing

1. **Emphasize the full pipeline** - Most papers only show model training
2. **Document monitoring metrics** - Novel contribution
3. **Show drift detection in action** - Real production value
4. **Discuss without-labels problem** - Practical industry challenge

### For Experiments

1. **Gravity removal comparison** - Clear contribution
2. **Cross-dataset evaluation** - Show generalization
3. **Domain shift analysis** - ADAMSense vs Garmin
4. **Monitoring effectiveness** - Does drift detection work?

### For Defense

1. Prepare answers for "how do you know model degraded without labels?"
2. Be ready to explain DVC vs JSON difference
3. Show monitoring dashboard screenshots
4. Demonstrate rollback procedure

---

## Quick Reference: What to Implement Next

| If you have... | Implement... |
|----------------|--------------|
| 2 hours | Energy-based OOD score |
| 4 hours | Prometheus metrics export |
| 1 day | Automatic retrain trigger |
| 2 days | Gravity removal experiment |
| 1 week | Domain adaptation training |

---

**Previous Stage:** [09_DEPLOYMENT_AUDIT.md](09_DEPLOYMENT_AUDIT.md)  
**Back to Index:** [00_STAGE_INDEX.md](00_STAGE_INDEX.md)

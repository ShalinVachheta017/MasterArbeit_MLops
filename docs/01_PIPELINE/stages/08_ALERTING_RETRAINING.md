# Stage 7: Alerting & Retraining Triggers

> **❌ STATUS (2026-01-30): NOT IMPLEMENTED - Conceptual design only!**
> 
> **TODO:**
> - [ ] Create `src/trigger_policy.py` with tiered trigger logic
> - [ ] Implement 2-of-3 voting (KS + PSI + entropy)
> - [ ] Add cooldown periods between triggers
> - [ ] Connect to retraining pipeline
> - [ ] Log all trigger decisions to MLflow

**Pipeline Stage:** Decide when to alert and when to retrain  
**Input:** Monitoring metrics, drift reports, threshold configuration  
**Output:** Alerts, retraining triggers, escalation actions

---

## Alert Threshold Configuration

### Confidence Alerts

| Metric | INFO | WARNING | CRITICAL |
|--------|------|---------|----------|
| Mean Confidence | < 0.85 | < 0.75 | < 0.65 |
| Uncertain Ratio | > 0.05 | > 0.15 | > 0.25 |
| Mean Entropy | > 0.8 | > 1.2 | > 1.8 |
| Mean Margin | < 0.40 | < 0.25 | < 0.15 |

### Drift Alerts

| Metric | INFO | WARNING | CRITICAL |
|--------|------|---------|----------|
| KS Statistic | > 0.08 | > 0.15 | > 0.25 |
| PSI | > 0.05 | > 0.10 | > 0.25 |
| Wasserstein | > 0.15 | > 0.30 | > 0.50 |
| Channels Drifted | 1-2 | 3-4 | 5-6 |

### Temporal Alerts

| Metric | INFO | WARNING | CRITICAL |
|--------|------|---------|----------|
| Flip Rate | > 0.15 | > 0.25 | > 0.40 |
| Min Dwell Time | < 5s | < 3s | < 1.5s |
| Mean Dwell Time | < 12s | < 8s | < 4s |

---

## Alert Decision Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ALERT DECISION FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Monitoring Metrics Computed                                            │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────┐                           │
│  │ Check each metric against thresholds    │                           │
│  └─────────────────────────────────────────┘                           │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────┐                           │
│  │ Any CRITICAL threshold exceeded?        │                           │
│  └─────────────────────────────────────────┘                           │
│           │                                                             │
│     YES   │   NO                                                        │
│           ▼   │                                                         │
│   ┌───────────┴──────┐                                                 │
│   │ CRITICAL ALERT   │                                                 │
│   │ - Page on-call   │                                                 │
│   │ - Block pipeline │                                                 │
│   │ - Log incident   │                                                 │
│   └──────────────────┘                                                 │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────┐                           │
│  │ Multiple WARNING thresholds exceeded?   │                           │
│  └─────────────────────────────────────────┘                           │
│           │                                                             │
│     YES   │   NO                                                        │
│           ▼   │                                                         │
│   ┌───────────┴──────┐     ┌───────────────┐                           │
│   │ WARNING ALERT    │     │ INFO/PASS     │                           │
│   │ - Notify team    │     │ - Log metrics │                           │
│   │ - Queue retrain  │     │ - Continue    │                           │
│   │ - Monitor trend  │     └───────────────┘                           │
│   └──────────────────┘                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Retraining Trigger Conditions

### Automatic Retrain Triggers

| Condition | Trigger? | Safe Without Labels? |
|-----------|----------|----------------------|
| PSI > 0.25 for 3+ consecutive batches | YES | YES (trigger only) |
| Mean confidence < 0.70 for 5+ batches | YES | YES (trigger only) |
| Flip rate > 0.35 for 3+ batches | YES | YES (trigger only) |
| 5+ channels with KS > 0.20 | YES | YES (trigger only) |
| Manual operator request | YES | YES |

### Do NOT Automatically Retrain When

| Condition | Why? |
|-----------|------|
| Single batch with bad metrics | Could be transient |
| Only 1-2 channels drifted | May be sensor noise |
| High confidence but low margin | Model may still be correct |
| No labeled data available for validation | Cannot verify improvement |

---

## Safe Retraining Strategy (Without Labels)

### The Challenge

> "How do we retrain without labeled production data?"

### The Solution: Domain Adaptation + Proxy Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│              SAFE RETRAINING WITHOUT PRODUCTION LABELS                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. TRIGGER RETRAIN (proxy metrics exceed threshold)                    │
│                                                                         │
│  2. COLLECT PRODUCTION DATA (no labels needed)                          │
│     └── Store raw windows from recent batches                           │
│                                                                         │
│  3. DOMAIN ADAPTATION TRAINING                                          │
│     ├── Use labeled source domain (existing training data)              │
│     ├── Use unlabeled target domain (production data)                   │
│     └── Apply techniques:                                               │
│         ├── MMD (Maximum Mean Discrepancy) loss                         │
│         ├── Domain-adversarial training (DANN)                          │
│         └── Self-training with pseudo-labels                            │
│                                                                         │
│  4. PROXY VALIDATION (no labels needed)                                 │
│     ├── Compare confidence on production data: new vs old model         │
│     ├── Compare entropy: should decrease                                │
│     ├── Compare flip rate: should decrease or stay same                 │
│     ├── Check temporal consistency: should improve                      │
│     └── Verify drift metrics: should improve                            │
│                                                                         │
│  5. DECISION                                                            │
│     ├── IF all proxy metrics improve → DEPLOY new model                 │
│     ├── IF mixed results → HOLD, request human review                   │
│     └── IF metrics worse → REJECT, keep old model                       │
│                                                                         │
│  6. A/B TEST (optional, if possible)                                    │
│     └── Run both models on subset, compare proxy metrics                │
│                                                                         │
│  7. FULL DEPLOY (only if proxy metrics clearly better)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Retraining Validation Without Labels

### Proxy Metrics to Compare (Old Model vs New Model)

| Metric | Better If... | Example |
|--------|--------------|---------|
| Mean Confidence | Higher | 0.85 → 0.91 ✓ |
| Uncertain Ratio | Lower | 0.15 → 0.08 ✓ |
| Mean Entropy | Lower | 1.2 → 0.8 ✓ |
| Flip Rate | Lower | 0.25 → 0.18 ✓ |
| Mean Margin | Higher | 0.20 → 0.35 ✓ |

### Minimum Improvement Threshold

```python
def should_deploy_new_model(old_metrics, new_metrics):
    improvements = {
        "confidence": new_metrics["mean_confidence"] > old_metrics["mean_confidence"] + 0.02,
        "uncertain": new_metrics["uncertain_ratio"] < old_metrics["uncertain_ratio"] - 0.02,
        "entropy": new_metrics["mean_entropy"] < old_metrics["mean_entropy"],
        "flip_rate": new_metrics["flip_rate"] <= old_metrics["flip_rate"],
    }
    
    # Require at least 3/4 improvements, no metric significantly worse
    n_improved = sum(improvements.values())
    return n_improved >= 3 and not any_significantly_worse(old_metrics, new_metrics)
```

---

## Alert Message Templates

### CRITICAL Alert

```
🚨 CRITICAL: Model degradation detected

Batch: 2026-01-15_batch_003
Time: 2026-01-15T14:30:00Z

Triggered Conditions:
- Mean confidence: 0.62 (threshold: < 0.65 CRITICAL)
- PSI on Ax: 0.31 (threshold: > 0.25 CRITICAL)
- 5/6 channels showing drift

Recommended Actions:
1. Pause production predictions
2. Investigate data source
3. Review recent sensor changes
4. Consider emergency retrain

Dashboard: https://monitoring.example.com/batch/2026-01-15_batch_003
```

### WARNING Alert

```
⚠️ WARNING: Elevated drift metrics

Batch: 2026-01-15_batch_002
Time: 2026-01-15T12:00:00Z

Triggered Conditions:
- Mean confidence: 0.74 (threshold: < 0.75 WARNING)
- 3 channels with KS > 0.10

This is WARNING #2 in sequence. 
CRITICAL threshold: 3 consecutive warnings triggers retrain queue.

Monitoring: Will continue to track in next batch.
```

---

## What to Do Checklist

- [ ] Define threshold values in config file
- [ ] Implement alert generation in monitoring script
- [ ] Set up notification channels (email, Slack, PagerDuty)
- [ ] Create retrain trigger logic
- [ ] Implement proxy validation for new models
- [ ] Set up A/B testing infrastructure (if possible)
- [ ] Document escalation procedures
- [ ] Create runbook for common alert scenarios

---

## Evidence from Papers

**[Domain Adaptation Papers | PDF: papers/domain_adaptation/]**
- MMD loss for unsupervised adaptation
- DANN for domain-adversarial training
- Self-training with pseudo-labels

**[ICTH 2025: Wearable IMU HAR MLOps | PDF: papers/new paper/ICTH_2025_Oleh_Paper_MLOps_Summary.md]**
- Domain shift between ADAMSense and production
- Importance of monitoring for adaptation decisions

**[NeurIPS 2020: Energy OOD | PDF: papers/new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf]**
- Using energy/confidence for OOD detection
- Applicable to retraining validation

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Implement domain adaptation training | High | Enable safe retraining |
| **HIGH** | Add proxy validation comparison | Medium | Safe model updates |
| **MEDIUM** | Create A/B testing infrastructure | Medium | Gradual rollout |
| **MEDIUM** | Add alert escalation with timeouts | Low | Better incident response |
| **LOW** | Implement automatic rollback | High | Safety net for bad deploys |

---

**Previous Stage:** [07_EVALUATION_METRICS.md](07_EVALUATION_METRICS.md)  
**Next Stage:** [09_DEPLOYMENT_AUDIT.md](09_DEPLOYMENT_AUDIT.md)

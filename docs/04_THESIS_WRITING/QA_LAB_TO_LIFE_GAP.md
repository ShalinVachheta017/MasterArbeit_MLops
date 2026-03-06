# Q&A: Lab-to-Life Gap and CV in MLOps Pipeline

*Date: January 6, 2026*
*Context: Thesis MLOps project discussion*

---

## Question 1: Why does ICTH_16 paper show 49% but we get 14-15%?

**Answer:**

The difference is because of what the "production data" contains:

| Scenario | Training Data | Production Data | Result |
|----------|---------------|-----------------|--------|
| **ICTH_16 Paper** | Lab anxiety activities (ADAMSense) | **SAME activities** on Garmin device | 49% |
| **Our Project** | Lab anxiety activities (6 users) | **DIFFERENT activities** (random daily Garmin wear) | 14-15% |

**Key insight**: ICTH_16's production data was people performing the SAME 11 anxiety activities but wearing a different device (Garmin instead of ADAMSense). Our production data is random daily wear - walking, sitting at desk, cooking, etc. - NOT the 11 anxiety activities.

The model is looking for patterns it learned (hand_tapping, deep_breathing, etc.) in data that doesn't contain those activities at all.

---

## Question 2: Where should 5-fold Cross-Validation be implemented?

**Answer:**

**CV belongs in TRAINING phase, NOT inference.**

```
TRAINING PHASE (where CV goes):
┌─────────────────────────────────────────────────────────────┐
│  Training Data (all_users_data_labeled.csv)                 │
│  ├── Fold 1: Train on 80%, Validate on 20%                  │
│  ├── Fold 2: Train on 80%, Validate on 20%                  │
│  ├── Fold 3: Train on 80%, Validate on 20%                  │
│  ├── Fold 4: Train on 80%, Validate on 20%                  │
│  └── Fold 5: Train on 80%, Validate on 20%                  │
│  → Mean Accuracy: 94.0% (± 0.7%)                            │
└─────────────────────────────────────────────────────────────┘

INFERENCE PHASE (NO CV here):
┌─────────────────────────────────────────────────────────────┐
│  Production Data (sensor_fused_50Hz.csv)                    │
│  └── Model predicts → Output predictions                    │
│  → No ground truth = Cannot calculate accuracy              │
└─────────────────────────────────────────────────────────────┘
```

**Why no CV in inference?**
- CV requires labels to measure accuracy
- Production data is unlabeled
- Inference is single-pass prediction

---

## Question 3: What is Unsupervised Domain Adaptation?

**Answer:**

Domain adaptation is a technique to bridge the gap between:
- **Source domain**: Training data (lab conditions)
- **Target domain**: Production data (real-world)

**Types:**
1. **Supervised**: Requires labels in target domain
2. **Unsupervised**: No labels needed in target domain

**Common UDA methods:**
- CORAL (Correlation Alignment)
- Maximum Mean Discrepancy (MMD)
- Adversarial training (DANN)

**For this thesis**: UDA is research-level complexity. Focus on documenting the gap, not solving it.

---

## Question 4: If pipeline predicts wrong every time, what's the value?

**Answer:**

The MLOps pipeline provides value beyond just accuracy:

| Component | Value |
|-----------|-------|
| **Data Pipeline** | Consistent preprocessing, validation, versioning |
| **Model Versioning** | DVC tracks data, MLflow tracks experiments |
| **Reproducibility** | Any training run can be reproduced exactly |
| **Deployment** | Docker containerization, CI/CD automation |
| **Future-Ready** | Pipeline is ready for retraining when labeled data is available |

**The limitation is the MODEL (lab-to-life gap), not the PIPELINE.**

Per ICTH_16: "Without fine-tuning, accuracy of only 48.7%"

This is a documented research limitation, not a bug.

---

## Question 5: Can we add CV during RETRAINING?

**Answer:**

**YES!** This is the correct place for CV.

When implementing a retraining pipeline, CV should be included:

```python
# Retraining Pipeline (when new labeled data is available)
def retrain_model(new_labeled_data):
    # 1. Combine with existing training data (optional)
    combined_data = merge(existing_data, new_labeled_data)
    
    # 2. Run 5-fold CV to validate model performance
    cv_results = run_cross_validation(combined_data, n_folds=5)
    
    # 3. If CV accuracy meets threshold, train final model
    if cv_results['mean_accuracy'] > THRESHOLD:
        final_model = train_on_all_data(combined_data)
        deploy(final_model)
    else:
        alert("Model accuracy below threshold")
```

**When to use CV in retraining:**
- When collecting new labeled production data
- When fine-tuning on domain-specific data
- When validating model before deployment

---

## Summary

| Question | Key Answer |
|----------|------------|
| Why 49% vs 14-15%? | ICTH_16 had same activities on different device; we have different activities entirely |
| Where does CV go? | Training phase, not inference |
| What is UDA? | Research technique to adapt models across domains (complex) |
| Pipeline value? | Data versioning, reproducibility, deployment automation |
| CV in retraining? | YES - implement CV when retraining with new labeled data |

---

## References

1. **ICTH_16**: Lab-to-life gap documentation (89% → 49%)
2. **EHB_2025_71**: Multi-stage pipeline (HAR → Temporal Bout → RAG-LLM)
3. **76 Papers Summary**: Domain adaptation methods overview

# HAR MLOps Pipeline - Stage-Based Documentation Index

**Purpose:** Quick navigation to specific pipeline stages with current implementation status.

**Last Updated:** January 30, 2026  
**Current Completion:** ~58%  
**Next Priority:** train.py → trigger_policy.py → unit tests → CI/CD

---

## 🎯 Pipeline Status Dashboard (Updated: 2026-01-30)

| Stage | Document | Status | What's Done | What's TODO |
|-------|----------|--------|-------------|-------------|
| **Stage 0** | [Data Ingestion](01_DATA_INGESTION.md) | ✅ DONE | `sensor_data_pipeline.py` (1,182 lines) | Add schema validation |
| **Stage 1** | [Preprocessing & Fusion](02_PREPROCESSING_FUSION.md) | ✅ DONE | `preprocess_data.py` (794 lines), DVC | Add augmentation module |
| **Stage 2** | [Quality Control & Validation](03_QC_VALIDATION.md) | ✅ DONE | `preprocess_qc.py` (802 lines) | Integrate into CI/CD |
| **Stage 3** | [Training & Baseline](04_TRAINING_BASELINE.md) | ⚠️ PARTIAL | `baseline_stats.json` exists | ❌ **CREATE `src/train.py`** |
| **Stage 4** | [Inference](05_INFERENCE.md) | ✅ DONE | `run_inference.py` (896 lines) | Add MC Dropout, AdaBN |
| **Stage 5** | [Monitoring & Drift](06_MONITORING_DRIFT.md) | ✅ DONE | `post_inference_monitoring.py` (1,590 lines) | Add Grafana dashboard |
| **Stage 6** | [Evaluation & Metrics](07_EVALUATION_METRICS.md) | ⚠️ PARTIAL | `evaluate_predictions.py` (766 lines) | Add proxy metrics for unlabeled |
| **Stage 7** | [Alerting & Retraining](08_ALERTING_RETRAINING.md) | ❌ TODO | Conceptual design only | ❌ **CREATE `src/trigger_policy.py`** |
| **Stage 8** | [Deployment & Audit](09_DEPLOYMENT_AUDIT.md) | ⚠️ PARTIAL | Docker files exist | ❌ **CREATE `.github/workflows/`** |
| **Stage 9** | [Improvements & Roadmap](10_IMPROVEMENTS_ROADMAP.md) | 📋 PLANNING | This document | - |

### Legend
- ✅ DONE = Working and tested
- ⚠️ PARTIAL = Exists but needs enhancements
- ❌ TODO = Must create from scratch
- 📋 PLANNING = Documentation only

---

## 🚨 Critical Blockers (as of 2026-01-30)

| # | Blocker | Impact | Priority | Effort |
|---|---------|--------|----------|--------|
| 1 | **No `src/train.py`** | Cannot retrain model | 🔴 CRITICAL | 3 days |
| 2 | **No `src/trigger_policy.py`** | Cannot automate retraining decisions | 🔴 CRITICAL | 2 days |
| 3 | **Empty `tests/` folder** | 0% test coverage | 🔴 CRITICAL | 3 days |
| 4 | **No `.github/workflows/`** | No CI/CD pipeline | 🔴 CRITICAL | 2 days |
| 5 | **No proxy metrics validated** | Cannot evaluate in production | 🟠 HIGH | 2 days |
| 6 | **Thesis writing** | 0% written | 🔴 CRITICAL | 6-8 weeks |

---

## 📅 Implementation Timeline (Starting 2026-01-30)

### Week 1-2: Core Training & Trigger
```
Priority 0: MUST COMPLETE
├── src/train.py (5-fold CV, MLflow logging)
├── src/trigger_policy.py (tiered triggers)
├── tests/test_data_validator.py
├── tests/test_inference.py
└── .github/workflows/mlops.yml
```

### Week 3-4: Adaptation & Retraining
```
Priority 1: SHOULD COMPLETE
├── src/retrain_pseudo.py (curriculum pseudo-labeling)
├── src/domain_adaptation/adabn.py
├── Add MC Dropout to run_inference.py
└── tests/test_drift_detector.py
```

### Week 5-6: Experiments & Dashboard
```
Priority 2: NICE TO HAVE
├── Run ablation experiments
├── Grafana dashboard setup
├── Active learning export
└── Generate thesis figures
```

### Week 7-14: Thesis Writing
```
Priority 0: MUST COMPLETE
├── Chapter 3: Methodology
├── Chapter 4: Implementation
├── Chapter 5: Evaluation
├── Chapter 1-2: Introduction, Related Work
└── Chapter 6: Conclusion
```

---

## Pipeline Stages

| Stage | Document | Key Topics |
|-------|----------|------------|
| **Stage 0** | [Data Ingestion](01_DATA_INGESTION.md) | Raw data handling, naming conventions, DVC vs metadata JSON |
| **Stage 1** | [Preprocessing & Fusion](02_PREPROCESSING_FUSION.md) | Sensor fusion, resampling, gravity removal, normalization |
| **Stage 2** | [Quality Control & Validation](03_QC_VALIDATION.md) | QC checks vs unit tests, data validation layers |
| **Stage 3** | [Training & Baseline](04_TRAINING_BASELINE.md) | Baseline creation, reference datasets, frozen vs rolling |
| **Stage 4** | [Inference](05_INFERENCE.md) | Production inference, model loading, batch processing |
| **Stage 5** | [Monitoring & Drift Detection](06_MONITORING_DRIFT.md) | Proxy metrics, drift thresholds, degradation detection WITHOUT labels |
| **Stage 6** | [Evaluation & Metrics](07_EVALUATION_METRICS.md) | Complete metrics list, what to track, MLflow logging |
| **Stage 7** | [Alerting & Retraining](08_ALERTING_RETRAINING.md) | Alert thresholds, retraining triggers, safe adaptation |
| **Stage 8** | [Deployment & Audit](09_DEPLOYMENT_AUDIT.md) | Containerization, audit trails, Prometheus/Grafana |
| **Stage 9** | [Improvements & Roadmap](10_IMPROVEMENTS_ROADMAP.md) | Current gaps, suggested improvements, implementation priority |

---

## Quick Answers to Common Questions

### Q: How do we know the model is degraded WITHOUT labels?

**Answer:** Use **proxy metrics** that don't require ground truth:

1. **Confidence drop** — Mean confidence decreasing over batches
2. **Entropy increase** — Predictions becoming more uncertain
3. **Drift detection** — Input distribution shifting from training baseline
4. **Temporal inconsistency** — High flip rate, short dwell times
5. **Variance collapse** — Production data has much lower variance than expected

➡️ See [Stage 5: Monitoring & Drift Detection](06_MONITORING_DRIFT.md) for details.

---

### Q: DVC vs Metadata JSON — Do we need both?

**Answer:** **YES, both serve different purposes:**

| Aspect | DVC | Metadata JSON |
|--------|-----|---------------|
| **Purpose** | Version control for large files | Human-readable processing metadata |
| **What it tracks** | File hashes, storage location | Processing parameters, QC results |
| **When updated** | On `dvc add/push` | On every preprocessing run |
| **Queryable** | No (need `dvc pull` first) | Yes (just read JSON) |
| **CI/CD friendly** | Yes (hash comparison) | Yes (parameter validation) |

**Recommendation:** Keep both:
- DVC for file versioning and storage
- JSON for quick inspection and parameter tracking

➡️ See [Stage 0: Data Ingestion](01_DATA_INGESTION.md) for details.

---

### Q: QC Checks vs Unit Tests — Same or Different?

**Answer:** **DIFFERENT purposes:**

| Aspect | QC Checks | Unit Tests |
|--------|-----------|------------|
| **What they test** | Data quality | Code correctness |
| **When they run** | On new data arrival | On code changes (CI) |
| **Failure means** | Bad data, reject batch | Bug in code, fix code |
| **Examples** | NaN detection, range validation | Function returns expected output |
| **Framework** | Custom scripts | pytest |

➡️ See [Stage 2: QC & Validation](03_QC_VALIDATION.md) for details.

---

### Q: Baseline vs Reference — Same or Different?

**Answer:** **DIFFERENT purposes:**

| Aspect | Frozen Training Baseline | Rolling Production Reference |
|--------|--------------------------|------------------------------|
| **Purpose** | Detect drift from training distribution | Detect sudden changes in production |
| **Updated** | Only on retraining | After each batch |
| **Contains** | Training data statistics | Recent N batches statistics |
| **Comparison** | Production vs "what model learned" | Production vs "recent normal" |

➡️ See [Stage 3: Training & Baseline](04_TRAINING_BASELINE.md) for details.

---

### Q: Complete List of Proxy Metrics for Drift Detection

| Category | Metric | Threshold | What it Detects |
|----------|--------|-----------|-----------------|
| **Confidence** | Mean confidence | < 0.70 warn, < 0.50 critical | Model uncertainty |
| **Confidence** | Uncertain ratio | > 0.15 warn, > 0.30 critical | Too many low-confidence predictions |
| **Entropy** | Mean entropy | > 1.5 warn, > 2.0 critical | Prediction uncertainty |
| **Margin** | Mean margin | < 0.15 warn, < 0.10 critical | Ambiguous predictions |
| **Drift** | KS statistic | > 0.10 warn, > 0.20 critical | Distribution shift |
| **Drift** | Wasserstein distance | > 0.25 warn, > 0.50 critical | Distribution shift (effect size) |
| **Drift** | PSI | > 0.10 warn, > 0.25 critical | Population stability |
| **Drift** | Mean shift | > 0.25σ warn, > 0.50σ critical | Mean has moved |
| **Temporal** | Flip rate | > 0.30 warn, > 0.50 critical | Unstable predictions |
| **Temporal** | Min dwell time | < 2s warn, < 1s critical | Too-short activity bouts |
| **Data** | Variance collapse | std < 0.1 | Sensor failure or wrong preprocessing |

➡️ See [Stage 5: Monitoring & Drift Detection](06_MONITORING_DRIFT.md) for complete details.

---

## Navigation

- **Full Q&A Document:** [HAR_MLOps_QnA_With_Papers.md](../HAR_MLOps_QnA_With_Papers.md)
- **Bibliography:** [Bibliography_From_Local_PDFs.md](../Bibliography_From_Local_PDFs.md)
- **Figures:** [figures/](../figures/)

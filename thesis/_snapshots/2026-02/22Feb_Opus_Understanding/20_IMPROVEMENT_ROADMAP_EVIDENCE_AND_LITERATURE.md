# 20 — Improvement Roadmap: Evidence-Based and Literature-Informed

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22  
> **Sources:** All Phase 1-2 files, `docs/research/appendix-paper-index.md`, `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md`, `Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv`

---

## 1 Master Improvement Table

| ID | Area | What to Improve | Why | Complexity | Thesis Impact | Priority | Finding Ref |
|:--:|------|----------------|-----|:----------:|:------------:|:--------:|:-----------:|
| IMP-01 | Pipeline Integration | Wire stages 11-14 into `production_pipeline.py` orchestrator | 4 implemented modules (~1,610 lines) are never executed | M | **H** | **Now** | D-1, CD-6 |
| IMP-02 | Trigger Safety | Replace 4 placeholder zeros in `trigger_evaluation.py` | 4 of ~8 trigger inputs are 0.0 — 2-of-3 voting is undermined | L | **H** | **Now** | T-1 |
| IMP-03 | Model Governance | Replace `is_better=True` with real proxy metric comparison | All adapted models auto-promoted without validation | M | **H** | **Now** | T-2, D-4 |
| IMP-04 | Drift Detection | Fix `wasserstein_drift.py` component field mismatch bug | `calibration_warnings` field on wrong artifact type → runtime error | L | M | **Now** | File 03 F5 |
| IMP-05 | CI/CD | Create `scripts/inference_smoke.py` + add `on.schedule` | CI integration test references missing file; model-validation unreachable | L | M | **Now** | A-1, A-4 |
| IMP-06 | CI/CD | Replace 3 echo stubs in model-validation job | Placeholder steps weaken MLOps maturity claims | L | M | **Now** | A-2 |
| IMP-07 | Calibration | Integrate temperature scaling into pipeline (post-inference) | Confidence ≠ correctness — uncalibrated probs degrade monitoring L1 + pseudo-label quality | M | **H** | **Now** | D-1 |
| IMP-08 | Monitoring | Unify API vs pipeline monitoring thresholds | Divergent thresholds (L1: 10% vs 30%; L3: 0.75 vs 2.0) — inconsistent alerting | L | M | **Next** | M-1 |
| IMP-09 | Monitoring | Add baseline staleness guard (timestamp check) | Baseline may be months old after repeated adaptation — drift L3 becomes meaningless | L | M | **Next** | M-5 |
| IMP-10 | Drift | Bridge offline Z-score thresholds → pipeline W₁ thresholds | Two independent drift metrics not reconciled — threshold recommendations don't transfer | M | M | **Next** | CD-1, CD-3 |
| IMP-11 | Cross-Dataset | Create unified drift+confidence per-session report | Batch infrastructure (confidence) and drift analysis (Z-score) never joined | M | **H** | **Next** | CD-4 |
| IMP-12 | Adaptation | Remove or properly implement DANN/MMD | Silently redirect to pseudo-labeling — misleading capability claim | L | M | **Next** | TR-2, D-5 |
| IMP-13 | Prometheus | Fix thesis chapter reference to non-existent Grafana dashboard | `CH4_IMPLEMENTATION.md` cites `config/grafana/har_dashboard.json` — file missing | L | M | **Next** | PG-3 |
| IMP-14 | Prometheus | Wire `MetricsExporter` into `app.py` `/predict` handler | 623-line module with 0 imports in production code | L | L | **Next** | PG-1 |
| IMP-15 | Monitoring | Upgrade Layer 3 drift to Wasserstein option | Currently uses Z-score of mean shift — W₁ captures full distribution shape | M | M | **Optional** | M-2 |
| IMP-16 | Monitoring | Add energy-based OOD score as monitoring signal | Complements confidence-based OOD detection (Liu et al. 2020, ICLR) | M | M | **Optional** | — |
| IMP-17 | Monitoring | Add pattern memory for recurring benign drift | Reduces false alarms on known-safe distribution patterns (LIFEWATCH-inspired) | H | L | **Optional** | — |
| IMP-18 | Adaptation | Conformal prediction for risk-aware monitoring | Provides distribution-free coverage guarantees without labels | H | M | **Optional** | — |
| IMP-19 | Adaptation | Active learning query pipeline for efficient labeling | Selects most informative samples for human annotation budget | H | L | **Optional** | — |
| IMP-20 | Reproducibility | Add DVC for raw data + intermediate artifact versioning | Currently no data versioning — only code is version-controlled | M | M | **Optional** | I-6 |

---

## 2 Detailed Improvement Cards

### IMP-01: Wire Stages 11-14 into Orchestrator

**What:** Add calibration_uncertainty, wasserstein_drift, curriculum_pseudo_labeling, sensor_placement to `ProductionPipeline.ALL_STAGES` and implement `elif` clauses in `run()`.

**Why it matters:**
- 4 modules with ~1,610 combined code lines are implemented but never executed
- Thesis cannot claim a "14-stage pipeline" when only 10 run
- [CODE: `src/pipeline/production_pipeline.py:L53`] — `ALL_STAGES` = stages 1-10 only

**Implementation sketch:**
```python
# production_pipeline.py
ALL_STAGES = [...existing 10..., "calibration", "wasserstein_drift", 
              "curriculum_pseudo_labeling", "sensor_placement"]
# Add elif clauses in run() for each; gate behind --advanced flag
```

**Literature support:** Sculley et al. (2015) — "Hidden Technical Debt in ML Systems" stresses end-to-end pipeline completeness.

**Complexity:** M (code exists — integration + testing) | **Thesis Impact:** H | **Priority:** Now

---

### IMP-02: Replace Trigger Placeholder Zeros

**What:** Map real monitoring metrics to 4 zeroed trigger inputs: `mean_entropy`, `mean_dwell_time_seconds`, `short_dwell_ratio`, `n_drifted_channels`.

**Why it matters:**
- [CODE: `src/components/trigger_evaluation.py:L73-L82`] — 4 values hardcoded to 0
- TriggerPolicyEngine's 2-of-3 voting is undermined — drift & temporal signals always read zero
- Makes trigger decisions confidence-only

**Implementation sketch:**
```python
# In trigger_evaluation.py, replace:
"mean_entropy": 0.0,        → monitoring_report.layer1.mean_entropy
"mean_dwell_time_seconds": 0 → monitoring_report.layer2.mean_dwell_time
"short_dwell_ratio": 0.0    → monitoring_report.layer2.short_dwell_ratio
"n_drifted_channels": 0     → monitoring_report.layer3.n_drifted_channels
```

**Complexity:** L (mapping only) | **Thesis Impact:** H | **Priority:** Now

---

### IMP-03: Replace `is_better=True` with Real Proxy Validation

**What:** Wire `ProxyModelValidator` (already in `trigger_policy.py`) into `model_registration.py` to compare adapted model against baseline.

**Why it matters:**
- [CODE: `src/components/model_registration.py:L69-L75`] — all models auto-promoted
- Undermines governance/rollback safety claims
- `ProxyModelValidator` exists [CODE: `src/trigger_policy.py`] but is not wired

**Implementation sketch:**
```python
# model_registration.py
validator = ProxyModelValidator(thresholds)
is_better = validator.compare(new_metrics, baseline_metrics)
if not is_better:
    registry.rollback_to_previous()
```

**Literature support:** Google ML Test Score (Breck et al. 2017) — model validation before promotion is a Level 2 requirement.

**Complexity:** M | **Thesis Impact:** H | **Priority:** Now

---

### IMP-07: Integrate Temperature Scaling into Pipeline

**What:** Add post-inference temperature scaling using `src/calibration.py:TemperatureScaler`.

**Why it matters:**
- Guo et al. (2017) "On Calibration of Modern Neural Networks" — DNNs are often overconfident
- Monitoring Layer 1 relies on confidence thresholds — uncalibrated probs make thresholds unreliable
- Pseudo-label quality depends on confidence gating — uncalibrated confidence misselects samples
- [CODE: `src/calibration.py`] — module exists with TemperatureScaler + CalibrationEvaluator

**Implementation sketch:**
```python
# Post-inference, before monitoring:
scaler = TemperatureScaler()
scaler.fit(val_logits, val_labels)  # requires small labeled holdout
calibrated_probs = scaler.transform(probs)
# Feed calibrated_probs to monitoring + pseudo-labeling
```

**Literature:**
- Guo et al. (2017) "On Calibration of Modern Neural Networks" — temperature scaling
- [Citation TODO: Naeini et al. 2015 — ECE metric definition]

**Complexity:** M (needs labeled holdout set) | **Thesis Impact:** H | **Priority:** Now

---

### IMP-11: Unified Cross-Dataset Drift + Confidence Report

**What:** Join output of `analyze_drift_across_datasets.py` (per-session Z-score) with `batch_process_all_datasets.py` (per-session confidence) keyed by session timestamp.

**Why it matters:**
- [FINDING CD-4] — the two most valuable analysis scripts are never combined
- Thesis needs: "does drift predict confidence degradation?" — requires joined data
- A scatter plot of drift_score vs mean_confidence across 26 sessions would be a key thesis figure

**Implementation sketch:**
```python
# New script: analyze_drift_vs_confidence.py
drift_df = run_drift_analysis()     # per-session drift scores
batch_df = load_batch_comparison()  # per-session confidence
merged = drift_df.merge(batch_df, on='session_id')
correlation = merged['max_drift'].corr(merged['mean_confidence'])
# Save merged table + scatter plot
```

**Complexity:** M | **Thesis Impact:** H | **Priority:** Next

---

## 3 Literature-Informed Enhancement Opportunities

### 3.1 Papers Already Referenced in Repository

The repo includes `Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv` (37 papers across 7 themes) and `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md` (88-paper synthesis).

| Theme | Papers | Key Technique | Repo Status | Gap |
|-------|-------:|--------------|-------------|-----|
| Sensor placement drift | 5 | Multi-sensor fusion, cross-hand comparison | `sensor_placement.py` exists | Not orchestrated |
| Adaptive preprocessing | 4 | Feature normalization, SALIENCE | `preprocess_data.py` has DomainCalibrator | Integrated ✓ |
| BiLSTM-based drift detection | 3 | CNN-BiLSTM hybrids, DWT | `train.py` uses 1D-CNN-BiLSTM | Architecture matches ✓ |
| Unsupervised domain adaptation | 3 | GAN-based transfer, SF-Adapter | DANN/MMD listed but placeholder | Not truly implemented |
| Data drift in wearable HAR | 3 | Continual learning, DRNNs | `wasserstein_drift.py` + monitoring | Partially integrated |
| Pseudo-labeling for unlabeled data | 3 | Sequential retraining, InfoGNN | `curriculum_pseudo_labeling.py` | Exists but not orchestrated |
| Confidence filtering in SSL | 3 | TEIF, FlexCon, classwise filtering | `train.py` has confidence gating | Integrated ✓ |

### 3.2 Key External Literature for Improvements

| Paper/Source | Relevance to Improvement | Citation Status |
|-------------|-------------------------|-----------------|
| Guo et al. (2017) — "On Calibration of Modern Neural Networks" | IMP-07: temperature scaling rationale | [Citation TODO: verify exact publication venue — ICML 2017?] |
| Sculley et al. (2015) — "Hidden Technical Debt in ML Systems" | IMP-01: pipeline completeness | [Citation TODO: NeurIPS 2015] |
| Breck et al. (2017) — "ML Test Score" | IMP-03: model validation before promotion | [Citation TODO: verify exact title + venue] |
| Wang et al. (2020) — TENT: Test Entropy Minimization | Existing TENT implementation validation | [Citation TODO: ICLR 2021 — verify year] |
| Li et al. (2016) — AdaBN: Revisiting Batch Normalization | Existing AdaBN implementation validation | [Citation TODO: Pattern Recognition 2018?] |
| Liu et al. (2020) — Energy-based OOD Detection | IMP-16: energy score monitoring | [Citation TODO: NeurIPS 2020] |
| Rabanser et al. (2019) — "Failing Loudly" | IMP-10: drift metric comparison | [Citation TODO: NeurIPS 2019] |
| Vovk et al. (2005) — Conformal Prediction | IMP-18: distribution-free coverage | [Citation TODO: Springer monograph] |

---

## 4 Priority Execution Order

### Phase A — Now (Before Experiments, ~1 week)

| Order | IMP | Task | Effort | Unlocks |
|:-----:|:---:|------|:------:|---------|
| 1 | IMP-04 | Fix Wasserstein field mismatch bug | 0.5 day | Clean artifact handoff |
| 2 | IMP-02 | Wire 4 trigger placeholder inputs | 0.5 day | Real multi-signal trigger |
| 3 | IMP-01 | Orchestrate stages 11-14 | 2-3 days | Full 14-stage pipeline |
| 4 | IMP-03 | Wire ProxyModelValidator | 1 day | Real governance gate |
| 5 | IMP-05 | Create inference_smoke.py + on.schedule | 0.5 day | Clean CI run |
| 6 | IMP-06 | Replace echo stubs | 0.5 day | Complete CI |

### Phase B — Next (During Experiments, ~1-2 weeks)

| Order | IMP | Task | Effort |
|:-----:|:---:|------|:------:|
| 7 | IMP-07 | Temperature scaling integration | 1-2 days |
| 8 | IMP-08 | Unify monitoring thresholds | 0.5 day |
| 9 | IMP-11 | Unified cross-dataset report | 1-2 days |
| 10 | IMP-12 | Remove/fix DANN/MMD | 0.5 day |
| 11 | IMP-13 | Fix Grafana thesis reference | 0.5 day |

### Phase C — Optional (If Time Permits)

| IMP | Task | Effort |
|:---:|------|:------:|
| IMP-15 | W₁ Layer 3 upgrade | 2 days |
| IMP-16 | Energy OOD score | 2-3 days |
| IMP-20 | DVC integration | 2 days |

---

## 5 Improvement-to-Finding Traceability

| Improvement | Addresses Finding(s) | From File(s) |
|:-----------:|---------------------|:------------:|
| IMP-01 | D-1, CD-6, T-8 | 13, 16, 14 |
| IMP-02 | T-1 | 14 |
| IMP-03 | T-2, T-3, D-4 | 14, 13 |
| IMP-04 | F5 (File 03) | 03 |
| IMP-05 | A-1, A-4 | 15 |
| IMP-06 | A-2 | 15 |
| IMP-07 | D-1 | 13 |
| IMP-08 | M-1 | 12 |
| IMP-09 | M-5 | 12 |
| IMP-10 | CD-1, CD-3 | 16 |
| IMP-11 | CD-4 | 16 |
| IMP-12 | TR-2, D-5 | 11, 13 |
| IMP-13 | PG-3 | 17 |
| IMP-14 | PG-1 | 17 |
- What the papers recommend vs what is implemented
- Opportunities for thesis contribution claims

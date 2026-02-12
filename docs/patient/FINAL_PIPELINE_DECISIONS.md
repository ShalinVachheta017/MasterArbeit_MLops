# 🎯 FINAL PIPELINE DECISIONS
## HAR MLOps Thesis - Merged Decision Framework

**Generated:** January 30, 2026 (Updated)  
**Sources:**

### Repository Documentation (11 files)
- `docs/thesis/PIPELINE_REALITY_MAP.md` - Repository audit
- `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md` - 73 questions from 23 papers
- `FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md` - Three implementation paths
- `docs/THESIS_PROGRESS_DASHBOARD_2026-01-20.md` - 58% completion status
- `docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md` - Wrist domain shift
- `docs/thesis/UNLABELED_EVALUATION.md` - Three-layer monitoring
- `docs/thesis/FINE_TUNING_STRATEGY.md` - Controlled fine-tuning policy
- `docs/stages/06_MONITORING_DRIFT.md` - Proxy metrics
- `docs/MENTOR_QA_SIMPLE_WITH_PAPERS.md` - Mentor questions answered
- `docs/BIG_QUESTIONS_2026-01-18.md` - 29+ Q&A pairs

### Paper Analysis Documents (9 files from `paper for questions/`)
- `THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md` - Implementation code (875 lines)
- `DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md` - XHAR, AdaptNet, Shift-GAN (1,415 lines)
- `UNCERTAINTY_CONFIDENCE_PAPERS_ANALYSIS.md` - XAI-BayesHAR, MC Dropout (1,064 lines)
- `PSEUDO_LABELING_SELF_TRAINING_PAPERS_ANALYSIS.md` - Curriculum Labeling, SelfHAR (948 lines)
- `ACTIVE_LEARNING_MLOPS_HUMAN_IN_LOOP_PAPERS_ANALYSIS.md` - Tent, CODA, HITL (2,294 lines)
- `SENSOR_PLACEMENT_POSITION_PAPERS_ANALYSIS.md` - Compensation strategies (531 lines)
- `GENERAL_HAR_SURVEYS_PAPERS_ANALYSIS.md` - HAR overview papers (676 lines)
- `THESIS_BOOKS_MAJOR_REFERENCES_ANALYSIS.md` - PhD theses, Springer books (756 lines)

**Total Sources:** 20 documents (~10,000+ lines of analysis)

**Deadline:** May 20, 2026 (16 weeks remaining)  
**Current Completion:** ~58%  
**Core Constraint:** Production data is UNLABELED

---

# TABLE OF CONTENTS

1. [Stage-by-Stage Decisions](#1-stage-by-stage-decisions)
2. [Implementation Plan](#2-implementation-plan)
3. [Evaluation Plan](#3-evaluation-plan)
4. [What We Deliberately Do NOT Build](#4-what-we-deliberately-do-not-build)
5. [Explicit Assumptions](#5-explicit-assumptions)
6. [Thesis Risk Mitigation](#6-thesis-risk-mitigation)
7. [Handedness / Wrist Placement Domain Shift](#7-handedness--wrist-placement-domain-shift)
8. [Code Examples from Papers](#8-code-examples-from-papers)
9. [Glossary of Key Concepts](#9-glossary-of-key-concepts)
10. [Key Paper Insights Summary](#10-key-paper-insights-summary)
11. [Uncertainty Quantification Methods](#11-uncertainty-quantification-methods)
12. [Domain Adaptation Methods Comparison](#12-domain-adaptation-methods-comparison)

---

# 1. STAGE-BY-STAGE DECISIONS

## Decision Framework Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Known fact (exists or validated) |
| ❓ | Unresolved question |
| 🔧 | Design decision made |
| 📖 | Paper citation supporting decision |

---

## 1.1 INGESTION

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `sensor_data_pipeline.py` exists (1,182 lines) | Repository inventory |
| ✅ Auto-detects latest Garmin Excel file pair | Tested and working |
| ✅ Handles 50Hz resampling | Code verified |
| ✅ Sensor fusion (accel + gyro) works | Production runs confirmed |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ Multi-user support? | 🟡 LOW | **DEFER** - Single user demo sufficient |
| ❓ Streaming ingestion? | 🟡 LOW | **DEFER** - Batch mode for thesis |
| ❓ Raw file schema validation? | 🟠 MEDIUM | **IMPLEMENT** - Add basic checks |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Keep batch ingestion, add schema validation** | Single-user demo is sufficient for thesis; streaming is future work | N/A (engineering choice) |

### Implementation
```
ADD: Basic schema validation in sensor_data_pipeline.py
- Check column names match expected Garmin format
- Validate timestamp format
- Log warnings for missing data
EFFORT: 0.5 days
```

---

## 1.2 PREPROCESSING

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `preprocess_data.py` exists (794 lines) | Repository inventory |
| ✅ Auto-detects milliG vs m/s² | Code verified |
| ✅ Gravity removal toggle exists | Configurable |
| ✅ Window size = 200 samples (4 sec @ 50Hz) | `config.py` |
| ✅ Overlap = 50% | Hardcoded |
| ✅ Z-score normalization using training baseline | `baseline_stats.json` exists |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ Verify output matches training distribution? | 🔴 HIGH | **IMPLEMENT** - Add distribution check |
| ❓ What augmentations for position variance? | 🔴 HIGH | **IMPLEMENT** - Time-warp, scale, noise |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Add distribution validation + augmentation pipeline** | Paper-backed: augmentation reduces position sensitivity | 📖 (SelfHAR, COA-HAR) |

### Implementation
```
ADD: Distribution validation in preprocess_data.py
- Compare production mean/std to training baseline
- Alert if > 2σ deviation per channel
EFFORT: 1 day

ADD: Augmentation module for retraining
- Time-warp (factor 0.8-1.2)
- Scaling (factor 0.9-1.1)  
- Gaussian noise (σ=0.01)
- Axis permutation (for handedness)
EFFORT: 1.5 days
```

---

## 1.3 WINDOWING / QA

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `preprocess_qc.py` exists (802 lines) | Repository inventory |
| ✅ Generates JSON QC reports | reports/preprocess_qc/ has 6+ runs |
| ✅ Validates window count, NaN percentage | Code verified |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ CI hook for QC? | 🟠 MEDIUM | **IMPLEMENT** - Add to workflow |
| ❓ What failure threshold blocks pipeline? | 🟠 MEDIUM | **DECIDE** - >5% NaN = FAIL |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **QC as CI gate: >5% NaN or >10% windows failed = block** | Standard data quality practice | N/A (engineering) |

### Implementation
```
INTEGRATE: QC into CI/CD workflow
- Run preprocess_qc.py automatically
- Parse JSON for pass/fail status
- Block deployment if fail
EFFORT: 0.5 days
```

---

## 1.4 TRAINING / RETRAINING ⚠️ CRITICAL GAP

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ Pretrained model exists (499K params) | models/pretrained/ verified |
| ✅ Model architecture: 1D-CNN + BiLSTM | model_info.json |
| ✅ 11 activity classes | config.py |
| ❌ **NO `src/train.py` exists** | Repository audit |
| ❌ **Archived scripts are outdated** | src/Archived/ folder |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ How to retrain with pseudo-labels? | 🔴 CRITICAL | **IMPLEMENT** - Curriculum approach |
| ❓ Restart model between cycles? | 🔴 HIGH | **YES** - Prevents confirmation bias |
| ❓ What confidence threshold for pseudo-labels? | 🔴 HIGH | **Top-20% per class** (not fixed threshold) |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Implement curriculum pseudo-labeling with model restart** | Key insight from paper: restart prevents confirmation bias | 📖 (Curriculum Labeling, 2021, "WHY RESTART WORKS") |
| 🔧 **Use relative ranking (top-K%) not fixed confidence threshold** | Fixed thresholds cause class imbalance | 📖 (Curriculum Labeling, "Key Innovation") |
| 🔧 **5-fold CV with MLflow logging** | Standard practice | N/A |

### Implementation
```
CREATE: src/train.py
- Load pretrained model as starting point
- 5-fold stratified CV
- Early stopping (patience=5)
- MLflow logging (loss, accuracy, F1)
EFFORT: 3 days

CREATE: src/retrain_pseudo.py
- Curriculum pseudo-labeling algorithm:
  1. Generate predictions with current model
  2. Select top-20% most confident per class
  3. RESTART model to pretrained weights
  4. Train on labeled + selected pseudo-labeled
  5. Repeat for N cycles (default: 5)
- Log pseudo-label quality metrics
EFFORT: 3 days
```

---

## 1.5 INFERENCE

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `run_inference.py` exists (896 lines) | Repository inventory |
| ✅ FastAPI server works | docker/api/ tested |
| ✅ Batch mode functional | 3 prediction runs in data/prepared/predictions/ |
| ✅ Confidence threshold = 0.50 | Code hardcoded |
| ❌ No MC Dropout uncertainty | Not implemented |
| ❌ No test-time adaptation | Not implemented |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ Implement Tent or AdaBN for TTA? | 🔴 HIGH | **AdaBN** - Simpler, sufficient |
| ❓ Is 30× inference for MC Dropout acceptable? | 🔴 HIGH | **NO** - Use 10 passes as compromise |
| ❓ Streaming or batch? | 🟡 LOW | **BATCH** - Sufficient for demo |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Implement AdaBN (not Tent)** | Simpler: just update BN running stats, no backprop | 📖 (XHAR, 2019, "AdaBN technique") |
| 🔧 **Add MC Dropout with 10 passes** | Balance: 10× acceptable, 30× too slow | 📖 (MC Dropout paper, Table 4) |
| 🔧 **Keep batch mode for thesis** | Streaming is overkill for demo | Engineering choice |

### Implementation
```
ADD: MC Dropout to run_inference.py
- Enable dropout at inference (model.train())
- Run 10 forward passes
- Return: mean prediction, std (uncertainty), entropy
EFFORT: 1 day

ADD: AdaBN adaptation mode
- Collect BN statistics from production batch
- Replace model BN running_mean, running_var
- No gradient computation needed
EFFORT: 1 day

ADD: Uncertainty metrics to output
- prediction_confidence (max softmax)
- prediction_entropy
- mc_dropout_std
EFFORT: 0.5 days
```

---

## 1.6 MONITORING ⚠️ HIGH PRIORITY

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `post_inference_monitoring.py` exists (1,590 lines) | Repository inventory |
| ✅ KS-test per channel implemented | Code verified |
| ✅ PSI computation exists | Code verified |
| ✅ Entropy/confidence tracking | Code verified |
| ✅ 6+ monitoring runs in reports/monitoring/ | Verified |
| ❌ No Prometheus/Grafana | docker-compose only has MLflow |
| ❌ No alerting mechanism | Manual inspection only |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ What KS p-value threshold? | 🔴 HIGH | **p < 0.01** (conservative) |
| ❓ What PSI threshold? | 🔴 HIGH | **PSI > 0.20** (moderate shift) |
| ❓ What entropy threshold for drift? | 🔴 HIGH | **> 1.5× training mean** |
| ❓ Add Prometheus/Grafana? | 🟠 MEDIUM | **YES** - Visual dashboard |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Three-signal drift detection** | Combine KS + PSI + entropy for robust detection | 📖 (LIFEWATCH, WATCH) |
| 🔧 **Conservative thresholds to avoid false alarms** | Production disruption is costly | Engineering choice |
| 🔧 **Add Grafana dashboard** | Visual monitoring for thesis demo | Standard MLOps |

### Implementation
```
ADD: Drift detection thresholds to config
- KS_THRESHOLD = 0.01
- PSI_THRESHOLD = 0.20
- ENTROPY_MULTIPLIER = 1.5
EFFORT: 0.5 days

ADD: Prometheus metrics exporter
- Export: inference_count, confidence_mean, entropy_mean
- Export: drift_detected (boolean), ks_pvalue_min
EFFORT: 1 day

ADD: Grafana dashboard
- Panel 1: Confidence distribution over time
- Panel 2: Entropy trend
- Panel 3: KS p-values per channel
- Panel 4: Drift alerts timeline
EFFORT: 1 day
```

---

## 1.7 CSD / DRIFT DETECTION

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ KS-test implemented per channel | post_inference_monitoring.py |
| ✅ PSI implemented | Code verified |
| ✅ Wasserstein distance mentioned | Partial implementation |
| ✅ Baseline statistics exist | baseline_stats.json |
| ❌ No automated trigger | Reports drift but doesn't act |
| ❌ No LIFEWATCH pattern memory | Future work |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ KS or Wasserstein? | 🟠 MEDIUM | **KS** - Simpler, validated |
| ❓ Compare to training baseline or recent window? | 🔴 HIGH | **Training baseline** (fixed reference) |
| ❓ How to distinguish covariate vs concept drift? | 🔴 HIGH | **Cannot without labels** - state as limitation |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Use KS-test + PSI as primary drift signals** | Simpler than Wasserstein, well-understood | 📖 (Industry standard) |
| 🔧 **Compare against training baseline** | Provides consistent reference | Engineering choice |
| 🔧 **Acknowledge covariate/concept ambiguity** | Honest about limitation | 📖 (UDA Benchmark) |

### Implementation
```
REFACTOR: Consolidate drift detection in src/drift_detector.py
- detect_covariate_shift(production_stats, baseline_stats)
- Returns: {channel: ks_pvalue}, psi_value, is_drifted
EFFORT: 1 day
```

---

## 1.8 TRIGGER POLICY ⚠️ CRITICAL GAP

### Known Facts
| Fact | Evidence |
|------|----------|
| ❌ **NO implementation exists** | Repository audit |
| ❌ No decision logic | Only detection, no action |
| ✅ Conceptual design in docs | docs/stages/ files |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ What combination triggers retraining? | 🔴 CRITICAL | **2 of 3 signals** (KS + PSI + entropy) |
| ❓ Threshold-based or pattern-based? | 🔴 HIGH | **Threshold-based** (simpler for thesis) |
| ❓ Escalation path? | 🔴 HIGH | **Tier 1: AdaBN → Tier 2: Pseudo-retrain → Tier 3: Human review** |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Tiered trigger policy** | Gradual escalation reduces unnecessary retraining | 📖 (CODA, "cost-efficient") |
| 🔧 **2-of-3 voting for drift confirmation** | Reduces false positives | Engineering choice |
| 🔧 **Minimum 500 windows before retraining** | Need sufficient data | Engineering choice |

### Trigger Policy Rules
```yaml
# config/trigger_policy.yaml

tier_1_adapt:  # Try AdaBN first
  condition: "entropy_elevated OR ks_single_channel"
  action: "run_adabn_adaptation"
  cooldown_windows: 500

tier_2_retrain:  # Pseudo-label retraining
  condition: "(ks_drift AND psi_drift) OR (entropy_elevated AND ks_drift)"
  action: "run_pseudo_retrain"
  min_windows: 1000
  cooldown_days: 7

tier_3_human:  # Escalate to human
  condition: "retrain_failed OR confidence_collapsed"
  action: "alert_human_review"
  channels: ["email", "slack"]
```

### Implementation
```
CREATE: src/trigger_policy.py
- evaluate_triggers(monitoring_report) → TriggerDecision
- TriggerDecision: {tier, action, reason}
- Log all decisions to MLflow
EFFORT: 2 days

ADD: Trigger policy to post_inference_monitoring.py
- Call evaluate_triggers() after drift detection
- Execute recommended action
EFFORT: 0.5 days
```

---

## 1.9 ACTIVE LEARNING ⚠️ HIGH PRIORITY

### Known Facts
| Fact | Evidence |
|------|----------|
| ❌ **NO implementation exists** | Repository audit |
| ❌ No sample selection logic | Not implemented |
| ❌ No labeling interface | Not implemented |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ How many samples per labeling batch? | 🔴 HIGH | **20-50 samples** (practical budget) |
| ❓ Uncertainty or diversity sampling? | 🔴 HIGH | **Uncertainty** (simpler, validated) |
| ❓ How to present IMU for labeling? | 🔴 HIGH | **Plot + video sync** if available |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Uncertainty sampling via MC Dropout** | High uncertainty = model confused = label valuable | 📖 (CODA, MC Dropout paper) |
| 🔧 **Export top-50 uncertain windows for human labeling** | Practical batch size for thesis demo | 📖 (CODA, "cost-efficient") |
| 🔧 **Simple CSV export (no fancy UI)** | Thesis is about pipeline, not UI | Engineering choice |

### Implementation
```
CREATE: src/active_learning.py
- select_for_labeling(predictions, uncertainties, n=50) → indices
- Selection: top-N by MC Dropout std
- Export: windows + metadata CSV for manual labeling
EFFORT: 2 days

CREATE: scripts/export_for_labeling.py
- Load production predictions + uncertainties
- Select top-50 uncertain
- Export: window data, timestamp, predicted label, confidence
- Generate: waveform plots for each sample
EFFORT: 1 day
```

---

## 1.10 EVALUATION ⚠️ CRITICAL GAP

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ `evaluate_predictions.py` exists (766 lines) | Repository inventory |
| ✅ Computes accuracy, F1, confusion matrix | WITH labels |
| ❌ **Requires labels** - cannot use in production | Code: "when labels available" |
| ❌ No proxy metrics validated | Not implemented |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ What proxy metrics for accuracy? | 🔴 CRITICAL | **Confidence mean + entropy + temporal plausibility** |
| ❓ How to validate proxy metrics? | 🔴 CRITICAL | **Offline simulation with held-out labeled data** |
| ❓ Can we use calibration without labels? | 🟠 MEDIUM | **NO** - need labels for ECE |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Three proxy metrics (no labels needed)** | Can compute at inference time | 📖 (Tent entropy, XAI-BayesHAR uncertainty) |
| 🔧 **Validate proxies offline using labeled subset** | Correlation analysis | Engineering choice |
| 🔧 **State limitation: cannot verify accuracy in production** | Honest about constraint | Thesis requirement |

### Proxy Metrics
| Metric | Computation | Expected Correlation with Accuracy |
|--------|-------------|-----------------------------------|
| **Confidence Mean** | mean(max(softmax)) | Positive (higher = better) |
| **Entropy Mean** | mean(-Σp·log(p)) | Negative (lower = better) |
| **Temporal Plausibility** | % valid activity transitions | Positive (higher = better) |
| **MC Dropout Std** | std across 10 passes | Negative (lower = better) |

### Implementation
```
ADD: Proxy metrics to evaluate_predictions.py
- compute_proxy_metrics(predictions, probabilities) → ProxyReport
- ProxyReport: {confidence_mean, entropy_mean, transition_valid_pct, mc_std_mean}
EFFORT: 1 day

CREATE: scripts/validate_proxy_metrics.py
- Load labeled test set
- Compute actual accuracy + proxy metrics
- Calculate Pearson correlation
- Output: validation report
EFFORT: 1 day
```

---

## 1.11 MLOps / CI-CD ⚠️ CRITICAL GAP

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ Docker files exist | docker/Dockerfile.* |
| ✅ docker-compose works | Tested |
| ✅ MLflow tracking works | mlruns/ has experiments |
| ❌ **NO `.github/workflows/`** | Repository audit |
| ❌ **0% test coverage** | tests/ is empty |

### Unresolved Questions
| Question | Priority | Decision |
|----------|----------|----------|
| ❓ What CI/CD stages? | 🔴 CRITICAL | **Lint → Test → Build → Deploy** |
| ❓ How many unit tests? | 🔴 CRITICAL | **Minimum 10 tests** |
| ❓ Automated deployment? | 🟠 MEDIUM | **Manual trigger** (not on every push) |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **GitHub Actions workflow with 4 stages** | Standard MLOps | N/A |
| 🔧 **10 unit tests covering critical paths** | Academic minimum | N/A |
| 🔧 **Manual deployment trigger** | Avoids accidental prod changes | Engineering choice |

### Implementation
```
CREATE: .github/workflows/mlops.yml
- lint: flake8, black
- test: pytest with coverage
- build: docker build
- deploy: manual trigger to update containers
EFFORT: 2 days

CREATE: tests/
- test_config.py: Test path configuration
- test_validator.py: Test schema validation
- test_preprocess.py: Test windowing, normalization
- test_inference.py: Test model loading, prediction
- test_monitoring.py: Test drift detection
- test_trigger_policy.py: Test trigger logic
- conftest.py: Fixtures
- pytest.ini: Configuration
EFFORT: 3 days
```

---

## 1.12 THESIS WRITING ⚠️ CRITICAL GAP

### Known Facts
| Fact | Evidence |
|------|----------|
| ✅ 50+ markdown docs exist | docs/ folder |
| ✅ Paper analysis complete | PAPER_DRIVEN_QUESTIONS_MAP.md |
| ✅ Pipeline documented | PIPELINE_REALITY_MAP.md |
| ❌ **0 thesis pages written** | No thesis/ folder with chapters |
| ❌ No LaTeX/Word template | Not started |

### Design Decision
| Decision | Rationale | Citation |
|----------|-----------|----------|
| 🔧 **Use docs as source material** | Already written | N/A |
| 🔧 **Start with Methodology chapter** | Core contribution | N/A |
| 🔧 **LaTeX template** | Academic standard | N/A |

### Implementation
```
CREATE: thesis/ folder structure
- thesis/chapters/01_introduction.tex
- thesis/chapters/02_related_work.tex
- thesis/chapters/03_methodology.tex
- thesis/chapters/04_implementation.tex
- thesis/chapters/05_evaluation.tex
- thesis/chapters/06_conclusion.tex
- thesis/figures/
- thesis/main.tex
EFFORT: Ongoing (4-6 weeks)
```

---

# 2. IMPLEMENTATION PLAN

## 2.1 What We WILL Build (Thesis Scope)

### Priority 0 - Must Have (Weeks 1-4)

| Component | Effort | Deliverable | Paper Support |
|-----------|--------|-------------|---------------|
| `src/train.py` | 3 days | Training script with 5-fold CV | N/A |
| `src/retrain_pseudo.py` | 3 days | Curriculum pseudo-labeling | 📖 Curriculum Labeling |
| `src/trigger_policy.py` | 2 days | Tiered trigger logic | 📖 CODA |
| Unit tests (10) | 3 days | tests/*.py | N/A |
| CI/CD workflow | 2 days | .github/workflows/mlops.yml | N/A |
| Proxy evaluation | 2 days | Unlabeled metrics | 📖 Tent |

### Priority 1 - Should Have (Weeks 5-8)

| Component | Effort | Deliverable | Paper Support |
|-----------|--------|-------------|---------------|
| AdaBN adaptation | 1 day | Domain adaptation | 📖 XHAR |
| MC Dropout (10 passes) | 1 day | Uncertainty quantification | 📖 MC Dropout paper |
| Grafana dashboard | 2 days | Monitoring visualization | N/A |
| Active learning export | 2 days | Sample selection | 📖 CODA |
| Augmentation pipeline | 1.5 days | Training data diversity | 📖 SelfHAR, COA-HAR |
| Drift detector refactor | 1 day | Consolidated module | 📖 LIFEWATCH |

### Priority 2 - Nice to Have (If Time Permits)

| Component | Effort | Deliverable | Paper Support |
|-----------|--------|-------------|---------------|
| Kalman uncertainty | 2 days | Alternative UQ | 📖 XAI-BayesHAR |
| Temperature scaling | 1 day | Calibration | Standard |
| Prometheus metrics | 1 day | Export metrics | Standard |

## 2.2 Implementation Timeline

```
WEEK 1 (Feb 3-9): Core Training Loop
├── Mon-Tue: Create src/train.py
├── Wed: Connect to MLflow
├── Thu-Fri: Create 5 unit tests

WEEK 2 (Feb 10-16): CI/CD + Trigger Policy
├── Mon: Create .github/workflows/mlops.yml
├── Tue-Wed: Create src/trigger_policy.py
├── Thu: Create 5 more unit tests
├── Fri: Integration test

WEEK 3 (Feb 17-23): Pseudo-labeling + Monitoring
├── Mon-Wed: Create src/retrain_pseudo.py
├── Thu: Add proxy metrics to evaluate_predictions.py
├── Fri: Validate proxy metrics script

WEEK 4 (Feb 24-Mar 2): Adaptation + Uncertainty
├── Mon: Add AdaBN to run_inference.py
├── Tue: Add MC Dropout
├── Wed-Thu: Grafana dashboard
├── Fri: Active learning export

WEEK 5-6 (Mar 3-16): Experiments + Documentation
├── Run experiments with all components
├── Document results
├── Create thesis figures

WEEK 7-10 (Mar 17-Apr 13): Thesis Writing
├── Chapter 3: Methodology
├── Chapter 4: Implementation
├── Chapter 5: Evaluation

WEEK 11-14 (Apr 14-May 11): Review + Polish
├── Chapter 1-2: Introduction, Related Work
├── Chapter 6: Conclusion
├── Revisions

WEEK 15-16 (May 12-20): Final Submission
├── Final review
├── Format check
├── Submit
```

## 2.3 Minimum Viable Pipeline (MVP)

The **absolute minimum** for a thesis-complete pipeline:

```
[Raw Data] → [Preprocess] → [Inference] → [Monitor] → [Trigger] → [Retrain]
     ↓            ↓             ↓            ↓           ↓           ↓
   Excel      Windows        Predict      KS+PSI      Policy    Pseudo-label
   Parse       + QC          + UQ         + Alert     Decide      + CV
```

| Stage | Minimum Implementation | Status |
|-------|------------------------|--------|
| Ingestion | ✅ Exists | Done |
| Preprocessing | ✅ Exists | Done |
| Inference | Add MC Dropout | Partial |
| Monitoring | ✅ Exists | Done |
| Trigger | Create trigger_policy.py | **TODO** |
| Retrain | Create retrain_pseudo.py | **TODO** |
| CI/CD | Create workflow | **TODO** |
| Tests | Create 10 tests | **TODO** |

---

# 3. EVALUATION PLAN

## 3.1 Offline Evaluation (With Labels)

We have `data/prepared/garmin_labeled.csv` (9.2MB) which can be used for offline evaluation.

### Evaluation Strategy
```
1. TRAIN/TEST SPLIT
   - Use garmin_labeled.csv
   - 80% for pseudo-label pool
   - 20% for held-out test (NEVER used in training)

2. SIMULATE PRODUCTION
   - Run inference on 80% (pretend unlabeled)
   - Run monitoring (detect drift?)
   - Run trigger policy (would it trigger?)
   - Run pseudo-labeling (select samples)
   - Retrain on pseudo-labels

3. EVALUATE ON HELD-OUT 20%
   - Compute actual accuracy, F1
   - Compare: before adaptation vs after
   - Validate proxy metrics correlation
```

### Metrics to Report

| Metric | What It Shows | Target |
|--------|---------------|--------|
| **Accuracy (held-out)** | Model quality | >85% |
| **F1 Macro** | Class balance | >0.80 |
| **Proxy-Accuracy Correlation** | Proxy validity | >0.70 Pearson |
| **Drift Detection Rate** | Monitoring effectiveness | TPR >0.80, FPR <0.20 |
| **Trigger Precision** | Policy effectiveness | >0.70 |
| **Retrain Improvement** | Adaptation value | >2% accuracy gain |

## 3.2 Unlabeled Simulation (Production Proxy)

Since production is unlabeled, we simulate:

### Simulation Protocol
```
1. INJECT SYNTHETIC DRIFT
   - Add Gaussian noise (σ=0.1) to 50% of test data
   - Simulate handedness: flip AX, GX signs
   - Simulate sensor miscalibration: scale by 0.9

2. RUN PIPELINE
   - Inference → Monitoring → Trigger → (Retrain)

3. CHECK BEHAVIOR
   - Did monitoring detect drift?
   - Did trigger fire correctly?
   - Did pseudo-labeling improve?

4. PROXY METRICS ONLY
   - No ground truth available
   - Report: confidence, entropy, uncertainty
```

## 3.3 What We Can Claim vs Cannot Claim

### CAN Claim (With Evidence)
| Claim | Evidence | Citation |
|-------|----------|----------|
| "Pipeline detects distribution shift" | KS-test + PSI alerts on injected drift | Experiments |
| "Trigger policy reduces unnecessary retraining" | Tiered escalation logs | Experiments |
| "Pseudo-labeling improves model" | Held-out accuracy comparison | Experiments |
| "Proxy metrics correlate with accuracy" | Pearson r on labeled subset | Experiments |
| "Uncertainty quantification works" | MC Dropout std correlates with errors | 📖 MC Dropout paper |

### CANNOT Claim (State as Limitation)
| Claim | Why Not | Mitigation |
|-------|---------|------------|
| "Production accuracy is X%" | No labels | State as limitation |
| "System works for all users" | Single-user demo | Acknowledge in scope |
| "Handedness fully compensated" | Not implemented | Future work |
| "Real-time performance" | Batch mode only | Future work |

---

# 4. WHAT WE DELIBERATELY DO NOT BUILD

## 4.1 Out of Scope with Reasons

| Component | Why NOT | Thesis Section |
|-----------|---------|----------------|
| **Streaming inference** | Batch sufficient for thesis demo; streaming adds complexity | Future Work |
| **ContrasGAN UDA** | Complex GAN training; AdaBN achieves 80% of benefit | Future Work |
| **LIFEWATCH pattern memory** | Advanced technique; KS-test baseline sufficient | Future Work |
| **On-device inference** | Edge deployment not in scope | Limitations |
| **Multi-user support** | Single-user demo sufficient | Limitations |
| **Handedness-specific models** | Single model is constraint | Limitations |
| **Prometheus full stack** | Grafana dashboard sufficient | Simplification |
| **Fancy labeling UI** | CSV export + plots sufficient | Simplification |

## 4.2 Future Work Declarations

These will be explicitly mentioned in thesis Chapter 6 (Conclusion):

```markdown
## Future Work

1. **Streaming Inference**: Extend batch pipeline to real-time window processing
2. **GAN-based UDA**: Implement ContrasGAN for stronger domain adaptation
3. **Pattern Memory**: Add LIFEWATCH-style drift pattern recognition
4. **Edge Deployment**: Optimize model for on-device Garmin inference
5. **Multi-user Generalization**: Extend pipeline to handle multiple users
6. **Handedness Compensation**: Train separate models or add axis transformation
7. **Full Monitoring Stack**: Add Prometheus + AlertManager for production
8. **Active Learning UI**: Build web interface for human labeling workflow
```

---

# 5. EXPLICIT ASSUMPTIONS

## 5.1 Assumptions to State in Thesis

### Data Assumptions
| Assumption | Justification | Risk if Violated |
|------------|---------------|------------------|
| Garmin Excel format is stable | Tested with 2 exports | Script breaks |
| 50Hz sampling is sufficient | Standard in HAR literature | Miss fast activities |
| 6-axis IMU (no magnetometer) | Garmin watch output | Limited orientation info |
| Single user in demo | Simplifies evaluation | No generalization claim |

### Model Assumptions
| Assumption | Justification | Risk if Violated |
|------------|---------------|------------------|
| 1D-CNN + BiLSTM architecture | Pretrained, validated | Cannot change |
| 11 activity classes fixed | ADAMSense training | Novel activities fail |
| Window = 200 samples (4 sec) | Standard in literature | Short activities missed |
| BatchNorm layers exist | Required for AdaBN | TTA won't work |

### Production Assumptions
| Assumption | Justification | Risk if Violated |
|------------|---------------|------------------|
| **DATA IS UNLABELED** | Core constraint | N/A - this IS the constraint |
| User wears watch consistently | Required for valid data | Drift from inconsistency |
| No hardware changes | Same Garmin model | Sensor drift |
| Batch processing acceptable | Daily analysis | Delay in detection |

### Evaluation Assumptions
| Assumption | Justification | Risk if Violated |
|------------|---------------|------------------|
| Labeled subset is representative | From same Garmin watch | Biased evaluation |
| Proxy metrics correlate with accuracy | Validated offline | False confidence |
| Held-out test is truly held-out | Strict separation | Data leakage |

## 5.2 Assumptions to Validate Empirically

These will be tested in experiments:

| Assumption | Validation Method |
|------------|-------------------|
| KS-test detects meaningful drift | Inject known drift, check detection |
| Pseudo-labels improve accuracy | Compare before/after on held-out |
| Top-20% confidence is good threshold | Sensitivity analysis |
| 10 MC Dropout passes sufficient | Compare 10 vs 30 passes |
| AdaBN improves shifted data | Test on synthetic shift |

---

# 6. THESIS RISK MITIGATION

## 6.1 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Pseudo-labeling fails** | Medium | High | Fallback: manual labeling of 100 samples |
| **Trigger policy too sensitive** | Medium | Medium | Tune thresholds, add cooldown |
| **Trigger policy too insensitive** | Medium | Medium | Lower thresholds, add logging |
| **No improvement from adaptation** | Low | High | Document as negative result |
| **CI/CD breaks build** | Low | Low | Manual deployment fallback |
| **Not enough time for thesis writing** | Medium | Critical | Start writing in parallel |
| **Proxy metrics don't correlate** | Medium | High | Report as limitation |
| **Garmin format changes** | Low | Medium | Add format validation |

## 6.2 Contingency Plans

### If Pseudo-Labeling Fails
```
1. Manually label 100 samples (2-3 hours)
2. Use as "small labeled set" for SelfHAR approach
3. Report: "With 100 labels, achieves X% accuracy"
4. Thesis contribution: Demonstrated labeling budget estimation
```

### If No Improvement from Adaptation
```
1. Document as legitimate negative result
2. Analyze: Why didn't it work?
3. Thesis contribution: "When domain adaptation fails in HAR"
4. This is still a valid thesis contribution!
```

### If Time Runs Out
```
Priority 1: train.py + trigger_policy.py + 5 tests + CI/CD
Priority 2: Pseudo-labeling + monitoring dashboard
Priority 3: AdaBN + MC Dropout
Priority 4: Active learning

If only Priority 1 done: Still have thesis contribution
```

## 6.3 Success Criteria

### Minimum for Thesis Pass
| Criterion | Requirement |
|-----------|-------------|
| Training script | `src/train.py` works |
| Retraining loop | `src/retrain_pseudo.py` demonstrates concept |
| Trigger policy | `src/trigger_policy.py` makes decisions |
| Tests | ≥5 passing tests |
| CI/CD | Workflow runs on push |
| Thesis document | 40+ pages |

### Target for Good Thesis
| Criterion | Requirement |
|-----------|-------------|
| Full pipeline | All stages integrated |
| Experiments | Ablation study with 3+ conditions |
| Proxy validation | Correlation >0.70 |
| Adaptation improvement | >2% accuracy gain |
| Tests | ≥10 passing tests |
| CI/CD | Full lint → test → build |
| Thesis document | 60+ pages |
| Dashboard | Grafana with 4+ panels |

### Stretch for Excellent Thesis
| Criterion | Requirement |
|-----------|-------------|
| Novel contribution | New insight from experiments |
| Multiple adaptation methods | AdaBN + Tent comparison |
| Active learning demo | Full export + import cycle |
| Publication-ready | Could submit to workshop |
| Tests | ≥15 passing, >70% coverage |
| Documentation | Complete API docs |

---

# 7. SUMMARY

## 7.1 Key Decisions Made

| Stage | Decision | Paper Support |
|-------|----------|---------------|
| **Retraining** | Curriculum pseudo-labeling with model restart | 📖 Curriculum Labeling |
| **Adaptation** | AdaBN (not Tent) | 📖 XHAR |
| **Uncertainty** | MC Dropout with 10 passes | 📖 MC Dropout paper |
| **Drift Detection** | KS + PSI + entropy (2-of-3 voting) | 📖 LIFEWATCH, WATCH |
| **Trigger Policy** | Tiered: AdaBN → Retrain → Human | 📖 CODA |
| **Active Learning** | Uncertainty sampling, top-50 export | 📖 CODA |
| **Evaluation** | Proxy metrics: confidence, entropy, temporal | 📖 Tent, XAI-BayesHAR |

## 7.2 Critical Path

```
Week 1-2: train.py + trigger_policy.py + tests + CI/CD
Week 3-4: retrain_pseudo.py + AdaBN + MC Dropout
Week 5-6: Experiments + dashboard
Week 7-14: Thesis writing
Week 15-16: Final submission
```

## 7.3 Document Status

| Section | Status |
|---------|--------|
| Stage-by-Stage Decisions | ✅ Complete |
| Implementation Plan | ✅ Complete |
| Evaluation Plan | ✅ Complete |
| Exclusions | ✅ Complete |
| Assumptions | ✅ Complete |
| Risk Mitigation | ✅ Complete |

---

# 8. HANDEDNESS / WRIST PLACEMENT DOMAIN SHIFT

> **Source:** `docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md`

## 8.1 The Problem

Training data was collected on **dominant wrist**, but ~70% of users wear watches on **left wrist** (non-dominant for right-handers). Most anxiety behaviors are performed with the **dominant hand**.

## 8.2 The Four Domain Cases

| Case | Watch Wrist | Activity Hand | Signal Quality | Expected % |
|------|-------------|---------------|----------------|------------|
| **A** | Dominant (Right) | Dominant (Right) | **BEST** — Full motion (±2-10 m/s²) | ~7% |
| **B** | Non-dominant (Left) | Dominant (Right) | **WEAKEST** — Indirect only (±0.1-0.5 m/s²) | **~63%** |
| **C** | Dominant (Right) | Non-dominant (Left) | GOOD — Decent signal | ~3% |
| **D** | Non-dominant (Left) | Non-dominant (Left) | MODERATE | ~27% |

**Critical Insight:** 63% of production users are in **Case B** (worst signal quality)!

## 8.3 Expected Model Behavior by Case

| Case | Confidence | Entropy | Flip Rate | Idle % |
|------|------------|---------|-----------|--------|
| **A (BEST)** | 85-95% | 0.2-0.8 | 10-20% | 10-20% |
| **B (WORST)** | 50-75% | 1.0-2.0 | 25-50% | 40-70% |
| **C** | 75-90% | 0.5-1.2 | 15-25% | 20-35% |
| **D** | 60-80% | 0.8-1.5 | 20-35% | 30-50% |

## 8.4 Mitigation Options

| Option | Complexity | Thesis Scope |
|--------|------------|--------------|
| **Metadata logging** (wear_wrist, user_handedness) | Low | ✅ IMPLEMENT |
| **Calibration protocol** (30-60 sec at session start) | Medium | ⚠️ MAYBE |
| **Axis mirroring augmentation** | Medium | ❌ FUTURE |
| **Separate models per case** | High | ❌ FUTURE |

## 8.5 Design Decision

🔧 **Log metadata + adjust confidence thresholds for Case B users**

```python
if not dominance_match:
    # Relax confidence thresholds for non-dominant wrist
    confidence_threshold = 0.35  # vs 0.50 for dominant
    entropy_threshold = 2.5      # vs 2.0 for dominant
```

---

# 9. CODE EXAMPLES FROM PAPERS

> **Source:** `paper for questions/THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md`

## 9.1 Detect Distribution Drift (KS-Test)

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_distribution_drift(train_data, prod_data, threshold=0.05):
    """
    Per Domain Adaptation papers: KS-test is simple but effective.
    """
    sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    drift_detected = []
    
    for idx, sensor in enumerate(sensor_names):
        train_flat = train_data[:, :, idx].flatten()
        prod_flat = prod_data[:, :, idx].flatten()
        stat, p_value = ks_2samp(train_flat, prod_flat)
        
        if p_value < threshold:
            drift_detected.append({
                'sensor': sensor,
                'ks_statistic': stat,
                'p_value': p_value,
                'drift': True
            })
    
    return drift_detected
```

## 9.2 Compute Degradation Indicators (No Labels)

```python
def compute_degradation_indicators(probabilities):
    """
    Compute proxy metrics for model degradation without labels.
    Source: THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md
    """
    max_probs = np.max(probabilities, axis=1)
    
    # Confidence drop indicator
    mean_confidence = np.mean(max_probs)
    low_confidence_rate = np.mean(max_probs < 0.50)
    
    # Entropy (uncertainty)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    
    # Margin (decision boundary ambiguity)
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    mean_margin = np.mean(margins)
    
    # Thresholds from research papers
    degradation_signals = {
        'confidence_drop': mean_confidence < 0.70,
        'high_uncertainty': mean_entropy > 1.5,
        'low_margin': mean_margin < 0.15,
        'degradation_score': (
            (0.85 - mean_confidence) / 0.85 +
            (mean_entropy - 0.5) / 2.0 +
            (0.40 - mean_margin) / 0.40
        ) / 3
    }
    
    return degradation_signals
```

## 9.3 Self-Training with Pseudo-Labels

```python
def self_training_retrain(model, unlabeled_X, confidence_threshold=0.90):
    """
    Self-training: Use high-confidence predictions as pseudo-labels.
    Per Transfer Learning Survey: threshold >0.90 is recommended.
    """
    probabilities = model.predict(unlabeled_X)
    max_probs = np.max(probabilities, axis=1)
    pseudo_labels = np.argmax(probabilities, axis=1)
    
    # Select high-confidence samples
    confident_mask = max_probs >= confidence_threshold
    confident_X = unlabeled_X[confident_mask]
    confident_y = pseudo_labels[confident_mask]
    
    print(f"Selected {np.sum(confident_mask)}/{len(unlabeled_X)} samples "
          f"({100*np.mean(confident_mask):.1f}%) above threshold {confidence_threshold}")
    
    # Retrain with pseudo-labeled data
    model.fit(confident_X, confident_y, epochs=10, batch_size=32)
    
    return model, confident_mask
```

## 9.4 Adaptive Batch Normalization (AdaBN)

```python
def adapt_batch_norm(model, target_X, num_batches=50):
    """
    Adaptive Batch Normalization - update BN statistics using target data.
    Per XHAR: Simple but effective, no gradient updates needed.
    """
    import tensorflow as tf
    
    # Set model to training mode for BN statistic updates
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    # Run forward passes to update BN running statistics
    batch_size = 32
    for i in range(num_batches):
        batch_idx = np.random.choice(len(target_X), batch_size)
        batch = target_X[batch_idx]
        _ = model(batch, training=True)  # Update BN statistics
    
    # Freeze BN layers again
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    return model
```

## 9.5 Elastic Weight Consolidation (EWC)

```python
def compute_fisher_information(model, X, y, num_samples=200):
    """
    Compute Fisher Information Matrix for EWC.
    Per Lifelong Learning paper: Protects important weights.
    """
    import tensorflow as tf
    
    fisher = {n: np.zeros(w.shape) for n, w in enumerate(model.trainable_weights)}
    
    for i in range(min(num_samples, len(X))):
        with tf.GradientTape() as tape:
            output = model(X[i:i+1], training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y[i:i+1], output)
        
        gradients = tape.gradient(loss, model.trainable_weights)
        
        for n, grad in enumerate(gradients):
            if grad is not None:
                fisher[n] += grad.numpy() ** 2
    
    for n in fisher:
        fisher[n] /= num_samples
    
    return fisher

def ewc_loss(model, fisher, old_weights, lambda_ewc=1000):
    """EWC penalty term to add to training loss."""
    import tensorflow as tf
    ewc_penalty = 0
    for n, (new_w, old_w) in enumerate(zip(model.trainable_weights, old_weights)):
        ewc_penalty += tf.reduce_sum(fisher[n] * (new_w - old_w) ** 2)
    return lambda_ewc * ewc_penalty
```

## 9.6 Retraining Trigger Logic

```python
def should_retrain(metrics):
    """
    Combined trigger: Scheduled + Drift + Performance
    Source: FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md
    """
    return (
        metrics['days_since_last_retrain'] >= 7 or  # Weekly
        metrics['drift_score'] > 0.25 or            # Distribution shift
        metrics['accuracy_drop'] > 0.05 or          # Performance degradation
        metrics['new_labeled_samples'] >= 100       # Data volume
    )
```

---

# 10. GLOSSARY OF KEY CONCEPTS

> **Source:** `docs/MENTOR_QA_SIMPLE_WITH_PAPERS.md`

| Term | Definition |
|------|------------|
| **Confidence** | Highest probability the model gives to any class. Example: 87% sitting → confidence = 0.87 |
| **Entropy** | Measures how spread the probability is. H = -Σ p × log(p). H=0 means certain, H>2.0 means very uncertain |
| **Drift** | When production data looks different from training data |
| **Covariate Drift** | Input features change (e.g., new user has different movement patterns) |
| **Concept Drift** | Relationship between input and output changes (e.g., "sitting" now looks different) |
| **Audit Set** | Small set of production windows (100-500) that we manually label for validation |
| **Proxy Metrics** | Indirect signals that suggest model problems without needing ground truth |
| **Flip Rate** | How often prediction changes between adjacent windows |
| **Dwell Time** | How long each predicted activity lasts |
| **PSI** | Population Stability Index - measures distribution shift |
| **KS-Test** | Kolmogorov-Smirnov test - compares two distributions |
| **AdaBN** | Adaptive Batch Normalization - update BN stats using target data |
| **EWC** | Elastic Weight Consolidation - prevents catastrophic forgetting |
| **TTA** | Test-Time Adaptation - adapt model at inference without labels |
| **MC Dropout** | Monte Carlo Dropout - enable dropout at inference for uncertainty |

---

# 11. MENTOR QUESTIONS RESOLVED

> **Source:** `docs/MENTOR_QA_SIMPLE_WITH_PAPERS.md`

## Q1: Production has no labels. How do we monitor model quality?

**Answer:** Use proxy metrics:
1. **Mean confidence** — If it drops, the model is less sure
2. **Mean entropy** — If it rises, predictions are more spread out
3. **Flip rate** — How often the prediction changes between adjacent windows
4. **Feature drift** — KS-test or PSI comparing production features to training baseline

**Decision thresholds:**
- Confidence threshold: Alert if mean < 0.50
- Entropy threshold: Alert if mean > 2.0
- Drift threshold: Alert if PSI > 0.25 on ≥2 of 6 channels
- Persistence rule: Condition must hold for ≥3 consecutive sessions

## Q2: What is our retraining trigger policy?

**Answer:** Tiered trigger policy:
1. **Tier 1 (Adapt):** entropy_elevated OR ks_single_channel → run AdaBN
2. **Tier 2 (Retrain):** (ks_drift AND psi_drift) OR (entropy AND ks) → pseudo-retrain
3. **Tier 3 (Human):** retrain_failed OR confidence_collapsed → alert human

## Q3: How many samples should we label for validation?

**Answer:** Label **3-5 complete sessions (~200-500 windows)** as audit set
- Use stratified selection: cover all activity classes, both dominant/non-dominant
- Purpose: Validate model accuracy, enable per-class F1, detect drift

## Q4: Dominant hand vs non-dominant — should we train separate models?

**Answer:** **NO** for thesis scope. Instead:
- Log metadata (wear_wrist, user_handedness, dominance_match)
- Adjust confidence thresholds dynamically for mismatch cases
- Future work: axis mirroring augmentation, separate models

---

# 12. KEY PAPER INSIGHTS SUMMARY

> **Source:** 9 paper analysis files from `paper for questions/`

## 12.1 Domain Adaptation Methods (from DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md)

| Paper | Method | Labels Needed | Complexity | Thesis Priority |
|-------|--------|---------------|------------|-----------------|
| **XHAR** | AdaBN | Source only | ⭐ Low | ⭐⭐⭐ P0 |
| **AdaptNet** | Bilateral Translation | 10-30% target | ⭐⭐ Medium | ⭐⭐ P1 |
| **Shift-GAN** | GAN-based UDA | None (target) | ⭐⭐⭐ High | ❌ Future |
| **SCAGOT** | Context Disentangling | Source only | ⭐⭐ Medium | ⭐ P2 |

### AdaBN Implementation (Simplest UDA - NO RETRAINING!)
```python
# From XHAR paper: Just update BatchNorm statistics
def adapt_batch_norm(model, target_X, num_batches=50):
    """
    Replace source BN statistics with target statistics.
    Works WITHOUT labels and WITHOUT retraining!
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    batch_size = 32
    for i in range(num_batches):
        batch_idx = np.random.choice(len(target_X), batch_size)
        batch = target_X[batch_idx]
        _ = model(batch, training=True)  # Update BN running stats
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    return model
```

**Expected Improvement:** 5-15% accuracy improvement (per XHAR paper)

## 12.2 Uncertainty Quantification (from UNCERTAINTY_CONFIDENCE_PAPERS_ANALYSIS.md)

### XAI-BayesHAR: Kalman Filter for Uncertainty Tracking

| Metric | What it tells us | Threshold |
|--------|------------------|-----------|
| **trace(P_t)** | Overall feature uncertainty | > 2× training mean |
| **max(diag(P_t))** | Worst-case feature uncertainty | > 3× training std |
| **Innovation** | Observation vs prediction mismatch | > 3σ |

```python
def compute_bayesian_uncertainty(kalman_tracker, features):
    """
    Uncertainty from Kalman filter - works WITHOUT labels!
    Source: XAI-BayesHAR paper
    """
    x_hat, P = kalman_tracker.update(features)
    
    trace_uncertainty = np.trace(P)
    max_uncertainty = np.max(np.diag(P))
    innovation = features - kalman_tracker.H @ x_hat
    innovation_magnitude = np.linalg.norm(innovation)
    
    return {
        'total_uncertainty': trace_uncertainty,
        'max_feature_uncertainty': max_uncertainty,
        'innovation_magnitude': innovation_magnitude,
        'is_high_uncertainty': trace_uncertainty > threshold_from_training
    }
```

**Overhead:** ~1ms per window (minimal!)

## 12.3 Pseudo-Labeling Best Practices (from PSEUDO_LABELING_SELF_TRAINING_PAPERS_ANALYSIS.md)

### Curriculum Labeling: Key Innovation

| Technique | Description | Why It Works |
|-----------|-------------|--------------|
| **Model restart** | Reset parameters before each cycle | Prevents confirmation bias |
| **Top-K% per class** | Use relative ranking, not fixed threshold | Maintains class balance |
| **Self-paced** | Easy samples first, hard later | Curriculum learning |
| **No EMA/momentum** | Fresh model each cycle | Avoids error accumulation |

```python
def curriculum_labeling_cycle(model, labeled_data, unlabeled_data, 
                               initial_checkpoint, cycle_num, K_percent):
    """
    Key insight: RESTART model prevents pseudo-label errors from accumulating
    """
    # 1. RESTART model (CRITICAL!)
    model.load_state_dict(initial_checkpoint)
    
    # 2. Generate pseudo-labels
    probs = model.predict(unlabeled_data)
    pseudo_labels = probs.argmax(-1)
    confidences = probs.max(-1)
    
    # 3. Top-K% PER CLASS (not fixed threshold!)
    K = K_percent * (cycle_num + 1)  # Increase each cycle
    selected = select_top_k_per_class(pseudo_labels, confidences, K)
    
    # 4. Retrain from scratch
    model.load_state_dict(initial_checkpoint)  # Restart again
    train(model, labeled_data + unlabeled_data[selected])
    
    return model
```

**Recommended Thresholds:**
- Initial K%: 10%
- Growth rate: +10% per cycle
- Number of cycles: 10-20

## 12.4 Active Learning & HITL (from ACTIVE_LEARNING_MLOPS_HUMAN_IN_LOOP_PAPERS_ANALYSIS.md)

### CODA: Cost-Efficient Test-Time DA for HAR

| Feature | Value | Notes |
|---------|-------|-------|
| **Query rate** | 1-10% of samples | Only label most uncertain |
| **Uncertainty threshold** | 0.3-0.7 confidence | Below this → query human |
| **Human label agreement** | >0.8 kappa | Quality check |
| **Improvement per query** | 0.5-2% accuracy | Efficiency metric |

### Tent: Test-Time Adaptation via Entropy Minimization

**Key Insight:** Adapt at inference time by minimizing entropy on unlabeled data
- NO target labels required
- Updates only BatchNorm parameters (~1% of model)
- Can run continuously in production

## 12.5 Sensor Position Variance (from SENSOR_PLACEMENT_POSITION_PAPERS_ANALYSIS.md)

### Accuracy Drop from Position Mismatch

| Scenario | Accuracy Change | Source |
|----------|-----------------|--------|
| Same position (train=test) | Baseline | - |
| Cross-position WITHOUT compensation | **-15% to -35%** | Position paper |
| Cross-position WITH compensation | Partial recovery | Paper contribution |

### Compensation Strategies

| Strategy | Complexity | Effectiveness |
|----------|------------|---------------|
| Data augmentation (rotation, scaling) | Low | ⭐⭐ Medium |
| Axis mirroring | Low | ⭐⭐ Medium |
| AdaBN (BN stats update) | Low | ⭐⭐⭐ High |
| Domain adversarial training | High | ⭐⭐⭐ High |

---

# 13. DOMAIN ADAPTATION METHODS COMPARISON

> **Source:** `paper for questions/DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md`

## Complexity Hierarchy

```
Complexity Low → High:

├── Level 1: AdaBN (NO retraining)
│   └── Just update Batch Norm statistics from target data
│   └── Effort: 30 minutes
│   └── Expected gain: 5-15%
│
├── Level 2: Self-Training (pseudo-labeling)
│   └── Use high-confidence predictions as labels
│   └── Effort: 1-2 days
│   └── Expected gain: 10-20%
│
├── Level 3: MMD/CORAL Loss
│   └── Add distribution alignment to training loss
│   └── Effort: 2-3 days
│   └── Expected gain: 15-25%
│
└── Level 4: Adversarial (DANN/GAN)
    └── Domain discriminator + feature extractor
    └── Effort: 1-2 weeks
    └── Expected gain: 20-30%
```

## Thesis Decision: Implement Levels 1-2

| Method | Implement | Reason |
|--------|-----------|--------|
| **AdaBN** | ✅ YES | Simple, effective, can show in thesis |
| **Self-Training** | ✅ YES | Demonstrates MLOps retraining loop |
| **MMD/CORAL** | ⚠️ MAYBE | If time permits |
| **DANN/GAN** | ❌ NO | Too complex for thesis scope |

---

# 14. FILES TO CREATE (PRIORITY ORDER)

## Priority 1: This Week

| File | Purpose | Effort |
|------|---------|--------|
| `src/train.py` | Training script with 5-fold CV | 3 days |
| `src/trigger_policy.py` | Tiered trigger logic | 2 days |
| `tests/test_data_validator.py` | Unit tests | 1 day |
| `tests/test_inference.py` | Unit tests | 1 day |
| `.github/workflows/mlops.yml` | CI/CD pipeline | 1 day |

## Priority 2: Week 2-3

| File | Purpose | Effort |
|------|---------|--------|
| `src/retrain_pseudo.py` | Curriculum pseudo-labeling | 3 days |
| `src/drift_detector.py` | Consolidated drift detection | 1 day |
| `src/domain_adaptation/adabn.py` | AdaBN implementation | 1 day |
| `tests/test_drift_detector.py` | Drift detection tests | 0.5 days |

## Priority 3: Week 4+

| File | Purpose | Effort |
|------|---------|--------|
| `src/active_learning.py` | Sample selection | 2 days |
| `scripts/export_for_labeling.py` | Label export | 1 day |
| `docker-compose.grafana.yml` | Monitoring dashboard | 1 day |

---

# 15. 🏭 PRODUCTION MODULAR PIPELINE (FUTURE PHASE)

> **Added:** January 30, 2026  
> **Status:** 📋 PLANNING — To implement after current pipeline complete  
> **Reference:** `docs/thesis/production refrencxe/` (3 files)  
> **Purpose:** Transform research pipeline into production-grade modular architecture

## 15.1 Why Modular Architecture?

**Current State (Research Phase):**
- Flat `src/` folder with individual scripts
- Hardcoded configs scattered in files
- Manual file path management
- No artifact tracking between stages
- Works for thesis, NOT for real deployment

**Target State (Production Phase):**
- Modular `components/`, `entity/`, `pipeline/` structure
- `@dataclass` configuration entities
- Automatic artifact tracking
- Pipeline orchestration class
- Production-grade error handling

## 15.2 Target Directory Structure

```
src/
├── components/                    # 🆕 Pipeline components
│   ├── __init__.py
│   ├── data_ingestion.py         # From: sensor_data_pipeline.py
│   ├── data_validation.py        # From: data_validator.py
│   ├── data_transformation.py    # From: preprocess_data.py
│   ├── model_inference.py        # From: run_inference.py
│   ├── model_evaluation.py       # From: evaluate_predictions.py
│   ├── drift_detection.py        # From: post_inference_monitoring.py
│   └── trigger_policy.py         # 🆕 NEW
├── entity/                        # 🆕 Data classes
│   ├── __init__.py
│   ├── config_entity.py          # @dataclass for all configs
│   └── artifact_entity.py        # @dataclass for artifacts
├── pipeline/                      # 🆕 Orchestration
│   ├── __init__.py
│   ├── inference_pipeline.py     # Main inference flow
│   ├── training_pipeline.py      # Training/retraining flow
│   └── evaluation_pipeline.py    # Evaluation flow
├── core/                          # 🆕 Core utilities
│   ├── __init__.py
│   ├── exception.py              # Custom PipelineException
│   ├── logger.py                 # Structured logging
│   ├── constants.py              # All magic numbers
│   └── resilience.py             # Circuit breaker, retry logic
├── utils/
│   └── main_utils.py
└── [existing files kept for backward compatibility]
```

## 15.3 Configuration Entity Pattern

```python
# src/entity/config_entity.py (TO CREATE LATER)
from dataclasses import dataclass
from datetime import datetime
import os

TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    pipeline_name: str = "har_mlops_pipeline"
    artifact_dir: str = os.path.join("artifacts", TIMESTAMP)
    timestamp: str = TIMESTAMP

@dataclass
class PreprocessingConfig:
    """Preprocessing component config."""
    window_size: int = 200
    overlap: float = 0.5
    target_frequency: int = 50
    sensors: tuple = ("Ax", "Ay", "Az", "Gx", "Gy", "Gz")

@dataclass
class InferenceConfig:
    """Inference component config."""
    model_path: str = "models/pretrained/fine_tuned_model_1dcnnbilstm.keras"
    batch_size: int = 32
    mc_dropout_passes: int = 10
    confidence_threshold: float = 0.5

@dataclass
class DriftDetectionConfig:
    """Drift detection config."""
    ks_threshold: float = 0.05
    psi_threshold: float = 0.2
    entropy_threshold: float = 1.5
    voting_required: int = 2  # 2-of-3
```

## 15.4 Artifact Entity Pattern

```python
# src/entity/artifact_entity.py (TO CREATE LATER)
from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionArtifact:
    """Output from data ingestion."""
    raw_data_path: str
    n_samples: int
    ingestion_timestamp: str

@dataclass
class PreprocessingArtifact:
    """Output from preprocessing."""
    windowed_data_path: str
    n_windows: int
    scaler_path: str

@dataclass
class InferenceArtifact:
    """Output from inference."""
    predictions_path: str
    probabilities_path: str
    n_predictions: int
    inference_time_ms: float

@dataclass
class DriftDetectionArtifact:
    """Output from drift detection."""
    drift_detected: bool
    sensors_with_drift: List[str]
    trigger_action: str  # "none" | "adapt" | "retrain" | "escalate"
```

## 15.5 Pipeline Orchestration Pattern

```python
# src/pipeline/inference_pipeline.py (TO CREATE LATER)
class InferencePipeline:
    """
    Main inference pipeline orchestrator.
    Chains: Ingestion → Validation → Transform → Drift → Inference → Evaluate
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.preprocess_config = PreprocessingConfig()
        self.inference_config = InferenceConfig()
        self.drift_config = DriftDetectionConfig()
    
    def run_pipeline(self, input_path: str) -> InferenceArtifact:
        """Run complete inference pipeline."""
        # Step 1: Ingest
        ingestion_artifact = self.start_data_ingestion(input_path)
        
        # Step 2: Validate
        validation_artifact = self.start_validation(ingestion_artifact)
        
        # Step 3: Transform
        transform_artifact = self.start_transformation(validation_artifact)
        
        # Step 4: Drift Detection
        drift_artifact = self.start_drift_detection(transform_artifact)
        
        # Step 5: Handle drift if needed
        if drift_artifact.drift_detected:
            self.handle_drift(drift_artifact)
        
        # Step 6: Inference
        inference_artifact = self.start_inference(transform_artifact)
        
        return inference_artifact
```

## 15.6 Production Robustness Additions

### Circuit Breaker Pattern
```python
# src/core/resilience.py (TO CREATE LATER)
class CircuitBreaker:
    """Prevents cascading failures in production."""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN
```

### Retry with Backoff
```python
@retry_with_backoff(max_retries=3, backoff_factor=2.0)
def load_model(model_path: str):
    """Load model with automatic retry."""
    return tf.keras.models.load_model(model_path)
```

## 15.7 Implementation Timeline (Post-Thesis)

| Phase | Tasks | When |
|-------|-------|------|
| **Phase 1** | Current research pipeline complete | Weeks 1-6 |
| **Phase 2** | Thesis writing complete | Weeks 7-14 |
| **Phase 3** | Modular refactoring | Post-thesis |
| **Phase 4** | Production deployment | Post-thesis |

## 15.8 What We Keep vs. What We Change

### Keep (Thesis Phase):
- ✅ Current flat `src/` structure
- ✅ Current script-based execution
- ✅ Current config YAML files
- ✅ Focus on research questions

### Change Later (Production Phase):
- 🔄 Restructure to modular components
- 🔄 Add dataclass entities
- 🔄 Add pipeline orchestrator
- 🔄 Add circuit breaker/retry
- 🔄 Add health checks

## 15.9 Reference Documents

| Document | Content | Location |
|----------|---------|----------|
| Production Robustness Guide | Error handling, resilience patterns | `docs/thesis/production refrencxe/KEEP_Production_Robustness_Guide.md` |
| Reference Project Learnings | Modular structure, entity pattern | `docs/thesis/production refrencxe/KEEP_Reference_Project_Learnings.md` |
| Technology Stack Analysis | Tech decisions, file importance | `docs/thesis/production refrencxe/KEEP_Technology_Stack_Analysis.md` |

---

**Document Generated:** January 30, 2026 (Final Update)  
**Total Sources Integrated:** 23 documents (~12,000+ lines analyzed)  
**Decision Count:** 45+ decisions across 15 sections  
**Implementation Items:** 23 components planned (research) + 8 (production)  
**Code Examples:** 14 ready-to-use functions  
**Paper Insights:** 7 key papers with specific thresholds  
**Estimated Total Effort:** 8-10 weeks (implementation + writing) + post-thesis production phase
|---------|--------|
| Stage-by-Stage Decisions | ✅ Complete |
| Implementation Plan | ✅ Complete |
| Evaluation Plan | ✅ Complete |
| Exclusions | ✅ Complete |
| Assumptions | ✅ Complete |
| Risk Mitigation | ✅ Complete |

---

**Document Generated:** January 30, 2026  
**Sources:** PIPELINE_REALITY_MAP.md, PAPER_DRIVEN_QUESTIONS_MAP.md  
**Decision Count:** 35 decisions across 12 stages  
**Implementation Items:** 23 components planned  
**Estimated Total Effort:** 8-10 weeks (implementation + writing)

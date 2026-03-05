# 🔍 PIPELINE REALITY MAP - Engineering Reality Check
## HAR Wearable IMU MLOps Thesis Project

**Generated:** January 30, 2026  
**Purpose:** Brutally honest assessment of current pipeline state  
**Constraint:** Production data is UNLABELED — no online evaluation

---

# 1. Repository Inventory (Tree Summary)

## 1.1 Root Level Files

| File | Exists | Purpose | Last Modified |
|------|--------|---------|---------------|
| `README.md` | ✅ | Project overview | Active |
| `docker-compose.yml` | ✅ | MLflow + Inference services | Dec 2025 |
| `Thesis_Plan.md` | ✅ | 6-month timeline | Outdated |
| `PROJECT_GUIDE.md` | ✅ | Folder/file reference | Dec 2025 |
| `MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md` | ✅ | Action items | Jan 2026 |
| `FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md` | ✅ | Strategy options | Jan 2026 |
| `.github/workflows/` | ❌ **MISSING** | CI/CD | N/A |

## 1.2 Source Code (`src/`)

| File | Lines | Status | Function |
|------|-------|--------|----------|
| `config.py` | 83 | ✅ Active | Path configuration |
| `sensor_data_pipeline.py` | 1,182 | ✅ Active | Raw Excel → Fused CSV |
| `preprocess_data.py` | 794 | ✅ Active | CSV → Normalized NPY |
| `data_validator.py` | 322 | ✅ Active | Schema validation |
| `run_inference.py` | 896 | ✅ Active | Model inference |
| `evaluate_predictions.py` | 766 | ⚠️ Partial | Evaluation (needs labels) |
| `mlflow_tracking.py` | ~650 | ✅ Active | Experiment tracking |
| `diagnostic_pipeline_check.py` | ? | ❓ Unknown | Diagnostics |
| `train.py` | ❌ **MISSING** | Critical Gap | Retraining script |
| `Archived(...)/` | Multiple | 📦 Archived | Old training scripts |

## 1.3 Scripts (`scripts/`)

| Script | Lines | Status | Function |
|--------|-------|--------|----------|
| `preprocess_qc.py` | 802 | ✅ Active | QC validation |
| `post_inference_monitoring.py` | 1,590 | ✅ Active | Drift detection |
| `inference_smoke.py` | ? | ✅ Active | Smoke tests |
| `build_training_baseline.py` | ? | ✅ Active | Baseline stats |
| `create_normalized_baseline.py` | ? | ⚠️ Uses absolute paths | Baseline |
| `generate_thesis_figures.py` | ? | ⚠️ Partial | Visualization |

## 1.4 Data (`data/`)

| Path | Exists | Size | Content |
|------|--------|------|---------|
| `data/raw/*.xlsx` | ✅ | ~64MB | Garmin exports (2 files) |
| `data/raw/all_users_data_labeled.csv` | ✅ | Large | Training data |
| `data/processed/sensor_fused_50Hz.csv` | ❓ **Unverified** | ? | Fused sensor data |
| `data/prepared/production_X.npy` | ✅ | 12.5MB | 5,217 windows |
| `data/prepared/baseline_stats.json` | ✅ | 1.7MB | Training baseline |
| `data/prepared/config.json` | ✅ | 1.5KB | Scaler params |
| `data/prepared/garmin_labeled.csv` | ✅ | 9.2MB | Labeled subset |
| `data/prepared/predictions/` | ✅ | 3 runs | Inference outputs |

## 1.5 Models (`models/`)

| Path | Exists | Content |
|------|--------|---------|
| `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` | ✅ | 1D-CNN-BiLSTM (499K params) |
| `models/pretrained/model_info.json` | ✅ | Model metadata |
| `models/normalized_baseline.json` | ✅ | Normalization reference |

## 1.6 Tests (`tests/`)

| Content | Status |
|---------|--------|
| Test files | ❌ **EMPTY FOLDER** |
| pytest.ini | ❌ MISSING |
| conftest.py | ❌ MISSING |

## 1.7 Docker (`docker/`)

| File | Exists | Status |
|------|--------|--------|
| `Dockerfile.inference` | ✅ | 71 lines, ready |
| `Dockerfile.training` | ✅ | Exists |
| `api/` | ✅ | FastAPI server |

## 1.8 Reports (`reports/`)

| Path | Exists | Content |
|------|--------|---------|
| `reports/preprocess_qc/` | ✅ | 6+ QC reports |
| `reports/monitoring/` | ✅ | 6+ monitoring runs |
| `reports/inference_smoke/` | ✅ | Smoke test results |

## 1.9 Documentation (`docs/`)

| Folder | Count | Status |
|--------|-------|--------|
| `docs/stages/` | 11 files | ✅ Stage documentation |
| `docs/thesis/` | 10+ files | ✅ Thesis prep docs |
| `docs/technical/` | 10+ files | ✅ Technical docs |
| `docs/research/` | 3 files | ✅ Paper analysis |

---

# 2. Stage-by-Stage Reality Check

## Legend
- ✅ **Exists & Works** — Tested, produces expected output
- ⚠️ **Partial** — Exists but incomplete or untested
- ❌ **Missing** — Not implemented
- 🔴 **CRITICAL** — Blocks thesis completion
- 🟠 **HIGH** — Should fix before defense
- 🟡 **MEDIUM** — Nice to have
- 🟢 **LOW** — Future work

---

| Stage | What Exists | Assumptions Made | What's Missing | Risk Level |
|-------|-------------|------------------|----------------|------------|
| **Ingestion** | `sensor_data_pipeline.py` (1,182 lines), auto-detects latest file pair | Garmin Excel format stable; single user | No multi-user handling; no streaming; no schema validation on raw files | 🟡 MEDIUM |
| **Preprocessing** | `preprocess_data.py` (794 lines), unit detection, gravity removal toggle | milliG vs m/s² detection works; scaler config exists | No automated verification that output matches training distribution | 🟠 HIGH |
| **Windowing/QA** | `preprocess_qc.py` (802 lines), generates JSON reports | Window size=200, overlap=50% hardcoded; validation thresholds reasonable | No CI hook; QC is manual run only | 🟡 MEDIUM |
| **Training** | `Archived/train_with_cv.py` (archived), pretrained model exists | Model trained externally (ADAMSense); no retraining needed | **NO `src/train.py`**; no active training script; cannot retrain | 🔴 **CRITICAL** |
| **Inference** | `run_inference.py` (896 lines), FastAPI server, batch mode | Model loads correctly; confidence threshold=0.50 | No streaming/real-time mode; no MC Dropout uncertainty | 🟡 MEDIUM |
| **Monitoring** | `post_inference_monitoring.py` (1,590 lines), 3-layer monitoring | KS-test, PSI, entropy thresholds reasonable | No Prometheus/Grafana; manual runs only; no alerting | 🟠 HIGH |
| **CSD/Drift** | KS-test per channel, PSI computation, Wasserstein distance | Thresholds from papers; baseline exists | No automated trigger; no pattern memory (LIFEWATCH); no online detection | 🟠 HIGH |
| **Trigger Policy** | Conceptual in docs only | "If drift detected, retrain" | **NO IMPLEMENTATION**; no decision logic; no escalation | 🔴 **CRITICAL** |
| **Retraining** | Archived scripts only; EWC mentioned in docs | Would use CV; would log to MLflow | **NO SCRIPT EXISTS**; cannot update model with new data | 🔴 **CRITICAL** |
| **Active Learning** | Not implemented | N/A | **COMPLETELY MISSING**; no sample selection; no human-in-loop | 🟠 HIGH |
| **Evaluation** | `evaluate_predictions.py` (766 lines), confidence analysis | Works only WITH labels; production is unlabeled | **Cannot evaluate without labels**; no proxy metrics validated | 🔴 **CRITICAL** |
| **MLOps/CI-CD** | Docker files exist; docker-compose works | Manual deployment | **NO GitHub Actions**; no automated tests; **0% test coverage** | 🔴 **CRITICAL** |
| **Reporting/Thesis** | 50+ markdown docs; research foundation | Documentation exists | **NO THESIS CHAPTERS WRITTEN**; 0% thesis completion | 🔴 **CRITICAL** |

---

# 3. Top 10 Gaps Blocking Thesis-Complete Pipeline

## 🔴 CRITICAL BLOCKERS (Must Fix)

### 1. **No Retraining Script (`src/train.py`)**
- **What:** Cannot retrain model with new data
- **Impact:** Defeats entire MLOps purpose — thesis is about retraining loop
- **Evidence:** `src/train.py` does not exist; archived scripts are outdated
- **Fix Effort:** 3-5 days
- **Fix:** Create `src/train.py` with 5-fold CV, MLflow logging

### 2. **Empty Test Suite (`tests/` is empty)**
- **What:** Zero unit tests, zero integration tests
- **Impact:** 0% code coverage; any change could break pipeline
- **Evidence:** `tests/` folder exists but contains no files
- **Fix Effort:** 3-4 days for basic coverage
- **Fix:** Create `test_validator.py`, `test_preprocess.py`, `test_inference.py`

### 3. **No CI/CD Pipeline (`.github/workflows/` missing)**
- **What:** No automated testing, linting, or deployment
- **Impact:** Cannot demonstrate MLOps automation
- **Evidence:** `file_search(".github/workflows/*.yml")` returns no files
- **Fix Effort:** 2-3 days
- **Fix:** Create `mlops.yml` with lint → test → build → deploy stages

### 4. **No Trigger Policy Implementation**
- **What:** Drift detection exists, but no decision logic for "what to do when drift detected"
- **Impact:** Pipeline doesn't close the loop
- **Evidence:** `post_inference_monitoring.py` reports drift but doesn't act
- **Fix Effort:** 1-2 days
- **Fix:** Add `src/trigger_policy.py` with threshold-based retraining triggers

### 5. **Cannot Evaluate Without Labels**
- **What:** Production data is UNLABELED, but `evaluate_predictions.py` requires labels
- **Impact:** Cannot measure accuracy in production; thesis claims are unverifiable
- **Evidence:** `evaluate_predictions.py` line 8: "EVALUATION (when labels available)"
- **Fix Effort:** 2-3 days
- **Fix:** Implement proxy metrics (entropy correlation, confidence tracking)

### 6. **Thesis Chapters Not Written**
- **What:** 0% thesis document completion
- **Impact:** Nothing to submit
- **Evidence:** No `thesis/chapters/` folder; no LaTeX/Word files
- **Fix Effort:** 4-6 weeks
- **Fix:** Start writing immediately; use docs as source material

---

## 🟠 HIGH PRIORITY GAPS

### 7. **No Prometheus/Grafana Monitoring**
- **What:** Monitoring script exists but no visualization dashboard
- **Impact:** Cannot demonstrate real-time monitoring
- **Evidence:** `docker-compose.yml` mentions MLflow but not Prometheus
- **Fix Effort:** 2-3 days
- **Fix:** Add Prometheus + Grafana to docker-compose; create dashboards

### 8. **No Active Learning Implementation**
- **What:** No mechanism to select samples for labeling
- **Impact:** Cannot demonstrate human-in-loop workflow
- **Evidence:** No `active_learning.py` or similar
- **Fix Effort:** 3-4 days
- **Fix:** Implement uncertainty sampling from papers (CODA style)

### 9. **Training Baseline Has Absolute Paths**
- **What:** `scripts/create_normalized_baseline.py` uses hardcoded absolute paths
- **Impact:** Won't work on other machines; breaks reproducibility
- **Evidence:** Grep shows absolute path usage
- **Fix Effort:** 1 hour
- **Fix:** Refactor to use `config.py` paths

### 10. **No Streaming/Real-Time Inference**
- **What:** Only batch inference supported
- **Impact:** Cannot demo real-time recognition
- **Evidence:** `run_inference.py` line 47: "BATCH MODE: Process all windows at once"
- **Fix Effort:** 2-3 days
- **Fix:** Add streaming mode to FastAPI; implement window buffer

---

# 4. What Must Be Implemented vs Future Work

## 4.1 MUST IMPLEMENT (For Thesis Submission)

| Component | Priority | Effort | Rationale |
|-----------|----------|--------|-----------|
| `src/train.py` | 🔴 P0 | 3-5 days | Core thesis contribution |
| Unit tests (5-10 tests) | 🔴 P0 | 3-4 days | Academic requirement |
| CI/CD workflow | 🔴 P0 | 2-3 days | MLOps thesis, must have automation |
| Trigger policy logic | 🔴 P0 | 1-2 days | Closes retraining loop |
| Proxy evaluation metrics | 🔴 P0 | 2-3 days | Cannot claim accuracy without |
| Thesis Chapter 1-3 | 🔴 P0 | 3 weeks | Nothing to submit |
| Temperature scaling | 🟠 P1 | 1-2 days | Calibration is standard |
| AdaBN implementation | 🟠 P1 | 2-3 days | Domain adaptation for lab→production |

## 4.2 SHOULD IMPLEMENT (Strengthens Thesis)

| Component | Priority | Effort | Rationale |
|-----------|----------|--------|-----------|
| Prometheus + Grafana | 🟠 P1 | 2-3 days | Visual monitoring |
| Uncertainty quantification (MC Dropout) | 🟠 P1 | 2 days | Standard for ML |
| Active learning | 🟠 P1 | 3-4 days | Practical labeling strategy |
| Thesis figures/diagrams | 🟠 P1 | 2-3 days | Visualization |
| API documentation | 🟡 P2 | 1 day | OpenAPI spec |

## 4.3 FUTURE WORK (Acknowledge in Thesis)

| Component | Why Future Work | Thesis Section |
|-----------|-----------------|----------------|
| Streaming inference | Beyond scope, batch is sufficient | Future Work |
| GAN-based UDA (ContrasGAN) | Complex, AdaBN is sufficient | Future Work |
| Multi-user support | Demo is single-user | Limitations |
| On-device inference | Edge deployment not in scope | Future Work |
| LIFEWATCH pattern memory | Advanced, KS-test is baseline | Future Work |
| Handedness-specific models | Single model is constraint | Limitations |

---

# 5. Technical Debt Inventory

## 5.1 Code Quality Issues

| Issue | Location | Severity | Fix |
|-------|----------|----------|-----|
| Absolute paths in baseline script | `scripts/create_normalized_baseline.py` | 🟠 HIGH | Use `config.py` |
| Hardcoded window size | Multiple files | 🟡 LOW | Centralize to config |
| No type hints in some functions | `sensor_data_pipeline.py` | 🟢 LOW | Add type hints |
| Duplicate activity class definitions | `run_inference.py`, `evaluate_predictions.py`, `config.py` | 🟡 MEDIUM | Single source of truth |

## 5.2 Missing Configurations

| Config | Impact | Location Needed |
|--------|--------|-----------------|
| Drift thresholds | Manual tuning required | `config/drift_thresholds.yaml` |
| Retraining triggers | No automation | `config/trigger_policy.yaml` |
| Alert escalation rules | No alerting | `config/alerting.yaml` |
| Test configuration | No pytest.ini | `pytest.ini` |

## 5.3 Documentation Gaps

| Gap | Impact | Fix |
|-----|--------|-----|
| No API documentation | Hard to use endpoints | Generate OpenAPI |
| No runbook for failures | Manual recovery | Create `docs/RUNBOOK.md` |
| Outdated Thesis_Plan.md | Misleading timeline | Update or remove |

---

# 6. Recommended Next 4 Weeks

## Week 1 (Jan 27 - Feb 2): Core Pipeline

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon-Tue | Create `src/train.py` with 5-fold CV | Working training script |
| Wed | Connect training to MLflow | Logged experiments |
| Thu | Create 5 unit tests | `tests/test_*.py` |
| Fri | Create CI/CD workflow | `.github/workflows/mlops.yml` |

## Week 2 (Feb 3 - Feb 9): Monitoring & Triggers

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon | Implement trigger policy | `src/trigger_policy.py` |
| Tue | Add proxy evaluation metrics | Updated `evaluate_predictions.py` |
| Wed-Thu | Add Prometheus + Grafana | Working dashboards |
| Fri | Integration test | End-to-end smoke test |

## Week 3 (Feb 10 - Feb 16): Domain Adaptation

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon-Tue | Implement AdaBN | `src/domain_adaptation/adabn.py` |
| Wed | Add temperature scaling | Calibrated model |
| Thu-Fri | Run experiments | Documented results |

## Week 4 (Feb 17 - Feb 23): Thesis Writing Start

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon-Tue | Chapter 1: Introduction | 5-page draft |
| Wed-Thu | Chapter 2: Related Work outline | Section structure |
| Fri | Chapter 3: Methodology outline | Pipeline diagrams |

---

# 7. Summary Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Pipeline stages implemented | 7/13 | 13/13 | 6 missing |
| Test coverage | 0% | 70% | 70% |
| CI/CD workflows | 0 | 1 | 1 |
| Thesis pages written | 0 | 60+ | 60+ |
| Critical blockers | 6 | 0 | -6 |
| Weeks until deadline | 16 | - | - |

---

# 8. Conclusion

**Bottom Line:** The inference pipeline works, but the project is **NOT a complete MLOps pipeline**. It's a demo-level inference system with good documentation but missing:

1. ❌ Retraining capability (core thesis claim)
2. ❌ Automated testing (academic standard)
3. ❌ CI/CD automation (MLOps requirement)
4. ❌ Evaluation without labels (production reality)
5. ❌ Written thesis (submission requirement)

**Risk Assessment:** HIGH — 16 weeks remain, but 6 critical blockers must be resolved. Immediate action required on `src/train.py` and test suite.

---

**Document Status:** Complete  
**Next Review:** February 7, 2026  
**Owner:** Thesis Student

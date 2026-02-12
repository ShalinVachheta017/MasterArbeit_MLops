# COMPREHENSIVE FILE INVENTORY & CATEGORIZATION
## MasterArbeit_MLops Repository — Full Audit
**Date:** February 12, 2026 | **Auditor:** Copilot deep-scan

---

## LEGEND

| Tag | Meaning |
|-----|---------|
| **KEEP_ESSENTIAL** | Critical for pipeline execution or thesis defense |
| **KEEP_REFERENCE** | Useful reference but not required to run pipeline |
| **ARCHIVE** | Can be moved to an `_archive/` folder |
| **DELETE** | Safe to remove — redundant, empty, or misplaced |
| **OUTDATED** | Superseded by newer files; keep only if no replacement exists |

---

# 1. ROOT-LEVEL FILES

| # | File | Lines | Purpose | Status | Verdict |
|---|------|-------|---------|--------|---------|
| 1 | `README.md` | 1045 | Main project overview, architecture, setup, API reference | Active, last ref Dec 2025 | **KEEP_ESSENTIAL** |
| 2 | `PROJECT_GUIDE.md` | 553 | Complete folder/file reference with visual diagrams | Active | **KEEP_ESSENTIAL** |
| 3 | `Thesis_Plan.md` | ~80 | Original 6-month timeline (Oct 2025 – Apr 2026) | Still the master roadmap | **KEEP_ESSENTIAL** |
| 4 | `HOW_TO_READ_THIS_REPOSITORY.md` | 209 | Meta-guide: reading order for all markdown files | Useful but duplicates `PROJECT_GUIDE.md` — consider merging | **KEEP_REFERENCE** |
| 5 | `MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md` | 355 | Jan 6 2026 file categorization (KEEP/DELETE) + next steps | Partially outdated (referenced files like `WHAT_TO_DO_NEXT.md` already deleted) | **OUTDATED** |
| 6 | `FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md` | 414 | Three thesis-completion strategies (research vs practical vs hybrid) | Valuable for decision-making — the chosen path drives implementation | **KEEP_ESSENTIAL** |
| 7 | `FEBRUARY_2026_ACTION_PLAN.md` | 658 | Detailed week-by-week plan for Feb 2026 | **Active NOW** — this is the current action plan | **KEEP_ESSENTIAL** |
| 8 | `PIPELINE_EXECUTION_GUIDE.md` | 794 | Full pipeline execution instructions (prerequisites, stage-by-stage) | Comprehensive; overlaps with `docs/technical/PIPELINE_RERUN_GUIDE.md` but more detailed | **KEEP_ESSENTIAL** |
| 9 | `run_pipeline.py` | 141 | **Single entry point** to run full pipeline (all 7 stages) | Core pipeline driver — actively used | **KEEP_ESSENTIAL** |
| 10 | `docker-compose.yml` | 143 | Orchestrates MLflow + Inference + Training services | Essential for Docker deployment | **KEEP_ESSENTIAL** |
| 11 | `pytest.ini` | 45 | Pytest configuration: test paths, markers, output settings | Essential for test suite | **KEEP_ESSENTIAL** |
| 12 | `Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv` | CSV | Research paper summary table | Reference for thesis writing | **KEEP_REFERENCE** |
| 13 | `Summary_of_7_Research_Themes_in_HAR.csv` | CSV | Thematic summary of HAR research | Reference for related work chapter | **KEEP_REFERENCE** |
| 14 | `EHB_2025_71.pdf` | PDF | Foundational thesis paper (RAG pipeline for mental health) | **Duplicate** — already in `papers/` | **DELETE** |
| 15 | `ICTH_16.pdf` | PDF | Foundational thesis paper (1DCNN-BiLSTM anxiety recognition) | **Duplicate** — already in `papers/` | **DELETE** |
| 16 | `unnamed.jpg` | Image | Random image in root | Misplaced in root | **DELETE** |
| 17 | `New Microsoft Word Document (AutoRecovered).docx` | Docx | Empty/temp auto-recovered Word file | Accidental file | **DELETE** |
| 18 | `.gitignore` | — | Git ignore rules | Essential | **KEEP_ESSENTIAL** |
| 19 | `.dvcignore` | — | DVC ignore rules | Essential | **KEEP_ESSENTIAL** |
| 20 | `.dockerignore` | — | Docker ignore rules | Essential | **KEEP_ESSENTIAL** |

---

# 2. `docs/` FOLDER

## 2.1 docs/ Top-Level Files

| # | File | Lines | Purpose | Verdict |
|---|------|-------|---------|---------|
| 1 | `README.md` | 248 | Docs index with quick-start for thesis writing, folder structure | **KEEP_ESSENTIAL** |
| 2 | `HOW_TO_READ_DOCS.md` | 118 | Reading guide for docs/ folder — goals-based navigation | **KEEP_REFERENCE** (overlaps with `README.md`) |
| 3 | `ORGANIZATION_MAP.md` | 251 | Structured map of all docs and papers — thesis chapter mapping | **KEEP_ESSENTIAL** |
| 4 | `PIPELINE_DEEP_DIVE_opus.md` | 1239 | Deep technical pipeline analysis: data flow, DVC, MLflow, monitoring | **KEEP_ESSENTIAL** — best single technical reference |
| 5 | `PIPELINE_STAGE_PROGRESS_DASHBOARD.md` | 1266 | 10-stage pipeline with rubric-based %, Mermaid diagram, appendices | **KEEP_ESSENTIAL** — most current stage tracking (Jan 26) |
| 6 | `THESIS_MASTER_PROGRESS_2026-01-31.md` | 596 | Latest progress dashboard (68% complete, 15 weeks remaining) | **KEEP_ESSENTIAL** — most recent progress (Jan 31) |
| 7 | `THESIS_PROGRESS_DASHBOARD_2026-01-20.md` | 405 | Earlier progress snapshot (58%, Jan 20) | **OUTDATED** — superseded by Jan 31 version |
| 8 | `BIG_QUESTIONS_2026-01-18.md` | 1469 | 29+ answered Q&A with research citations | **KEEP_ESSENTIAL** — thesis-critical design decisions |
| 9 | `BIG_QUESTIONS_RISK_PAPERS_2026-01-18.md` | 1672 | Same Q&A with IEEE paper citations only | **KEEP_ESSENTIAL** — thesis writing material |
| 10 | `HAR_MLOps_QnA_With_Papers.md` | 1120 | Complete Q&A with ASCII pipeline diagram, 20 sections | **KEEP_ESSENTIAL** — comprehensive thesis reference |
| 11 | `MENTOR_QA_SIMPLE_WITH_PAPERS.md` | 932 | Simple-language mentor Q&A with evidence | **KEEP_REFERENCE** — useful for mentor meetings |
| 12 | `output_1801_2026-01-18.md` | 9193 | Massive detailed research output (all 29 questions expanded) | **KEEP_REFERENCE** — raw research material, very large |
| 13 | `Bibliography_From_Local_PDFs.md` | 550 | Formatted bibliography from local PDFs with citations | **KEEP_ESSENTIAL** — needed for thesis references chapter |
| 14 | `APPENDIX_FILE_INVENTORY.md` | 784 | File inventory grouped by macro-stage (machine-generated) | **KEEP_REFERENCE** — partially superseded by this audit |
| 15 | `APPENDIX_PAPER_INDEX.md` | 997 | Paper index with metadata, tags, paths | **KEEP_ESSENTIAL** — needed for thesis appendix |
| 16 | `REPOSITORY_CLEANUP_ANALYSIS_2026-01-18.md` | 508 | Jan 18 cleanup analysis with categorization | **OUTDATED** — superseded by this audit |

## 2.2 docs/archive/ (4 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `extranotes.md` | 467 | Online vs offline pipeline comparison table | **ARCHIVE** — interesting reference but not active |
| `LATER_Offline_MLOps_Guide.md` | 1172 | Comprehensive offline/edge MLOps guide | **ARCHIVE** — future reference for production deployment |
| `Mondaymeet.md` | 200 | Prep notes for Jan 12 meeting (pipeline audit findings) | **DELETE** — one-time meeting prep, outdated |
| `RESTRUCTURE_PIPELINE_PACKAGES.md` | 766 | Restructuring plan (current → production structure) | **KEEP_REFERENCE** — restructuring blueprint still relevant |

## 2.3 docs/patient/ (2 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `FINAL_PIPELINE_DECISIONS.md` | 1744 | **Merged decision framework** from 20 documents (10K+ lines analyzed) | **KEEP_ESSENTIAL** — master decision document |
| `PAPER_DRIVEN_QUESTIONS_MAP.md` | 779 | Question extraction from 88 papers, grouped by pipeline stage | **KEEP_ESSENTIAL** — drives thesis pipeline design |

## 2.4 docs/research/ (3 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `RESEARCH_PAPERS_ANALYSIS.md` | 495 | Deep analysis of ICTH_16 & EHB_2025_71 (foundational papers) | **KEEP_ESSENTIAL** — core thesis methodology |
| `RESEARCH_PAPER_INSIGHTS.md` | 571 | Actionable insights from 76+ papers | **KEEP_ESSENTIAL** — improvement recommendations |
| `KEEP_Research_QA_From_Papers.md` | 205 | Q&A on UDA, domain adaptation, retraining from papers | **KEEP_ESSENTIAL** — thesis background material |

## 2.5 docs/stages/ (11 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `00_STAGE_INDEX.md` | 198 | Navigation index for all stages, blocker list | **KEEP_ESSENTIAL** |
| `01_DATA_INGESTION.md` | 152 | Stage 0: naming conventions, DVC decisions | **KEEP_ESSENTIAL** |
| `02_PREPROCESSING_FUSION.md` | 141 | Stage 1: sensor fusion, gravity removal decisions | **KEEP_ESSENTIAL** |
| `03_QC_VALIDATION.md` | 186 | Stage 2: QC vs unit tests distinction | **KEEP_ESSENTIAL** |
| `04_TRAINING_BASELINE.md` | 194 | Stage 3: training/baseline creation, TODOs | **KEEP_ESSENTIAL** |
| `05_INFERENCE.md` | 178 | Stage 4: inference flow, enhancement TODOs | **KEEP_ESSENTIAL** |
| `06_MONITORING_DRIFT.md` | 233 | Stage 5: proxy metrics for unlabeled monitoring | **KEEP_ESSENTIAL** |
| `07_EVALUATION_METRICS.md` | 229 | Stage 6: complete metrics inventory | **KEEP_ESSENTIAL** |
| `08_ALERTING_RETRAINING.md` | 284 | Stage 7: trigger thresholds, retraining design | **KEEP_ESSENTIAL** |
| `09_DEPLOYMENT_AUDIT.md` | 359 | Stage 8: deployment architecture, CI/CD TODOs | **KEEP_ESSENTIAL** |
| `10_IMPROVEMENTS_ROADMAP.md` | 347 | Stage 9: gap analysis and priorities | **KEEP_ESSENTIAL** |

## 2.6 docs/technical/ (10 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `README.md` | 59 | Technical docs reading guide | **KEEP_ESSENTIAL** |
| `PIPELINE_RERUN_GUIDE.md` | 914 | Step-by-step pipeline commands (raw → inference) | **KEEP_ESSENTIAL** — primary pipeline execution ref |
| `PIPELINE_VISUALIZATION_CURRENT.md` | 542 | ASCII architecture diagrams of current pipeline state | **KEEP_ESSENTIAL** — visual reference for thesis |
| `PIPELINE_TEST_RESULTS.md` | 109 | Test execution logs from Jan 15 pipeline run | **KEEP_REFERENCE** |
| `pipeline_audit_map.md` | 266 | Visual data flow audit from raw → predictions | **KEEP_REFERENCE** |
| `evaluation_audit.md` | 188 | Audit of evaluate_predictions.py correctness | **KEEP_REFERENCE** |
| `tracking_audit.md` | 270 | MLflow + DVC tracking audit | **KEEP_REFERENCE** |
| `QC_EXECUTION_SUMMARY.md` | 248 | QC results: all checks passed, root cause = idle data | **KEEP_REFERENCE** |
| `root_cause_low_accuracy.md` | 276 | Root cause: 14-15% accuracy from stationary data | **KEEP_REFERENCE** |
| `FRESH_START_CLEANUP_GUIDE.md` | 500 | Commands to delete all outputs and start fresh | **KEEP_REFERENCE** |

## 2.7 docs/thesis/ (11 files + 1 subfolder)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `README.md` | 77 | Thesis docs reading guide | **KEEP_ESSENTIAL** |
| `FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md` | 1935 | **Master thesis plan** — comprehensive timeline, architecture | **KEEP_ESSENTIAL** |
| `THESIS_STRUCTURE_OUTLINE.md` | 518 | Chapter-by-chapter outline (60-80 pages) | **KEEP_ESSENTIAL** — thesis writing template |
| `CONCEPTS_EXPLAINED.md` | 591 | Background concepts: units, normalization, windowing | **KEEP_ESSENTIAL** — Chapter 2 material |
| `UNLABELED_EVALUATION.md` | 470 | 3-layer monitoring framework for unlabeled data | **KEEP_ESSENTIAL** — Chapter 3 methodology |
| `THESIS_READY_UNLABELED_EVALUATION_PLAN.md` | 587 | Ready-to-use evaluation text + experimental setup | **KEEP_ESSENTIAL** — Chapter 4-5 material |
| `HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md` | 418 | Domain shift analysis for left/right wrist | **KEEP_ESSENTIAL** — Chapter 5 discussion |
| `QA_LAB_TO_LIFE_GAP.md` | 150 | Lab-to-production accuracy gap explanation | **KEEP_ESSENTIAL** — discussion material |
| `FINE_TUNING_STRATEGY.md` | 401 | When/how to fine-tune models in production | **KEEP_ESSENTIAL** |
| `PIPELINE_REALITY_MAP.md` | 353 | Honest assessment of current pipeline state | **KEEP_REFERENCE** |
| `PAPER_DRIVEN_QUESTIONS_MAP.md` | 820 | Questions from 88 papers mapped to pipeline stages | **KEEP_ESSENTIAL** |

### docs/thesis/production refrencxe/ (3 files — note: folder has a typo)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `KEEP_Production_Robustness_Guide.md` | 1264 | Production robustness improvements + research backing | **KEEP_REFERENCE** — Chapter 6 (future work) |
| `KEEP_Reference_Project_Learnings.md` | 956 | Learnings from reference MLOps project (vikashishere) | **KEEP_REFERENCE** — informed architecture decisions |
| `KEEP_Technology_Stack_Analysis.md` | 758 | All technologies analyzed with importance ratings | **KEEP_REFERENCE** — Chapter 3 justification |

## 2.8 docs/figures/ (7 PNG files)

| File | Purpose | Verdict |
|------|---------|---------|
| `fig1_dataset_timeline.png` | Thesis figure: dataset timeline | **KEEP_ESSENTIAL** |
| `fig2_sampling_rate_qc.png` | Thesis figure: sampling rate QC | **KEEP_ESSENTIAL** |
| `fig3_gravity_removal_impact.png` | Thesis figure: gravity removal | **KEEP_ESSENTIAL** |
| `fig4_proxy_metrics_distributions.png` | Thesis figure: monitoring metrics | **KEEP_ESSENTIAL** |
| `fig5_drift_metrics_overview.png` | Thesis figure: drift metrics | **KEEP_ESSENTIAL** |
| `fig6_drift_over_time.png` | Thesis figure: drift temporal | **KEEP_ESSENTIAL** |
| `fig7_abcd_cases_comparison.png` | Thesis figure: ABCD cases | **KEEP_ESSENTIAL** |

---

# 3. `ai helps/` FOLDER (4 files)

| File | Type | Purpose | Verdict |
|------|------|---------|---------|
| `MLOps Lifecycle Framework and LLMOps Integration – Repository Analysis.pdf` | PDF | AI-generated repo analysis | **ARCHIVE** — one-time analysis, low ongoing value |
| `image.png` | Image | Screenshot/diagram | **ARCHIVE** |
| `image-1.png` | Image | Screenshot/diagram | **ARCHIVE** |
| `image-2.png` | Image | Screenshot/diagram | **ARCHIVE** |

**Recommendation:** This entire folder can be archived. The valuable plan file (`FINAL_Thesis_Status_and_Plan`) has been moved to `docs/thesis/`.

---

# 4. `cheat sheet/` FOLDER (2 files)

| File | Type | Purpose | Verdict |
|------|------|---------|---------|
| `1697955590966.pdf` | PDF | Generic cheat sheet (MLOps/ML related) | **ARCHIVE** — personal reference |
| `DVC_cheatsheet.pdf` | PDF | DVC commands cheat sheet | **KEEP_REFERENCE** — useful while using DVC |

---

# 5. `config/` FOLDER (6 files + 2 subfolders)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `pipeline_config.yaml` | ~70 | **Main config**: preprocessing, validation, inference settings | **KEEP_ESSENTIAL** |
| `mlflow_config.yaml` | 64 | MLflow tracking URI, experiment name, registry, tags | **KEEP_ESSENTIAL** |
| `prometheus.yml` | 70 | Prometheus scrape config for inference + MLflow | **KEEP_ESSENTIAL** |
| `requirements.txt` | 59 | All Python dependencies | **KEEP_ESSENTIAL** |
| `.pylintrc` | 648 | Pylint linting rules (default + customizations) | **KEEP_REFERENCE** — useful for code quality |
| `alerts/har_alerts.yml` | 191 | Prometheus alert rules (confidence, entropy, drift, flip rate) | **KEEP_ESSENTIAL** |
| `grafana/har_dashboard.json` | 540 | Grafana dashboard JSON for monitoring | **KEEP_ESSENTIAL** |

---

# 6. `docker/` FOLDER (4 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `Dockerfile.inference` | 71 | Docker image for FastAPI inference API | **KEEP_ESSENTIAL** |
| `Dockerfile.training` | 54 | Docker image for training environment | **KEEP_ESSENTIAL** |
| `api/main.py` | 447 | **FastAPI server**: /health, /model/info, /predict, /predict/batch | **KEEP_ESSENTIAL** |
| `api/__init__.py` | 1 | Package init | **KEEP_ESSENTIAL** |

---

# 7. `scripts/` FOLDER (7 files + `__pycache__/`)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `build_training_baseline.py` | 461 | Compute baseline stats from labeled training data for drift detection | **KEEP_ESSENTIAL** |
| `create_normalized_baseline.py` | 97 | Create normalized baseline (⚠️ uses absolute paths!) | **KEEP_ESSENTIAL** — fix absolute paths |
| `generate_thesis_figures.py` | 535 | Generate all thesis figures (7 PNGs in `docs/figures/`) | **KEEP_ESSENTIAL** |
| `inference_smoke.py` | 638 | Smoke test for inference pipeline (12 checks) | **KEEP_ESSENTIAL** |
| `post_inference_monitoring.py` | 1590 | **3-layer monitoring**: confidence, temporal, drift detection | **KEEP_ESSENTIAL** — core monitoring |
| `preprocess_qc.py` | 802 | QC validation for preprocessing (9 checks) | **KEEP_ESSENTIAL** |
| `run_tests.py` | 124 | Convenience script to run pytest suite with options | **KEEP_ESSENTIAL** |

---

# 8. `tests/` FOLDER (9 files)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `__init__.py` | — | Package init | **KEEP_ESSENTIAL** |
| `conftest.py` | 283 | Shared pytest fixtures (project paths, sample data) | **KEEP_ESSENTIAL** |
| `test_preprocessing.py` | 272 | Tests: gravity removal, normalization, windowing | **KEEP_ESSENTIAL** |
| `test_data_validation.py` | 134 | Tests: sensor columns, missing values, range checks | **KEEP_ESSENTIAL** |
| `test_drift_detection.py` | 266 | Tests: KS-test, PSI, drift thresholds | **KEEP_ESSENTIAL** |
| `test_model_rollback.py` | 270 | Tests: model versioning, registry, rollback | **KEEP_ESSENTIAL** |
| `test_ood_detection.py` | 190 | Tests: energy-based OOD, ensemble OOD | **KEEP_ESSENTIAL** |
| `test_active_learning.py` | 231 | Tests: uncertainty sampling, export | **KEEP_ESSENTIAL** |
| `test_prometheus_metrics.py` | 237 | Tests: Prometheus metric recording | **KEEP_ESSENTIAL** |
| `test_trigger_policy.py` | 293 | Tests: trigger thresholds, policy engine, alert levels | **KEEP_ESSENTIAL** |

**NOTE:** Previous dashboards said tests/ was empty. This is now **fully populated with 8 test files** covering key modules.

---

# 9. `src/` FOLDER

## 9.1 Top-Level src/ Python Files

| File | Purpose | Verdict |
|------|---------|---------|
| `README.md` (804 lines) | Source code inventory, logging structure, pipeline flow | **KEEP_ESSENTIAL** |
| `__init__.py` | Package init | **KEEP_ESSENTIAL** |
| `config.py` | Central path configuration (data, models, outputs) | **KEEP_ESSENTIAL** |
| `sensor_data_pipeline.py` | Raw Excel/CSV → fused `sensor_fused_50Hz.csv` (Stage A) | **KEEP_ESSENTIAL** |
| `preprocess_data.py` | CSV → normalized, windowed `.npy` arrays (Stage B) | **KEEP_ESSENTIAL** |
| `data_validator.py` | Schema & range validation for sensor data (Stage C) | **KEEP_ESSENTIAL** |
| `run_inference.py` | Model inference: `.npy` → predictions CSV (Stage G) | **KEEP_ESSENTIAL** |
| `evaluate_predictions.py` | Prediction analysis, confidence, distribution (Stage F) | **KEEP_ESSENTIAL** |
| `mlflow_tracking.py` | MLflow experiment tracking integration (Stage E) | **KEEP_ESSENTIAL** |
| `train.py` | Training script with K-fold CV (Stage E) | **KEEP_ESSENTIAL** |
| `trigger_policy.py` | Automated retraining trigger logic (Stage H) | **KEEP_ESSENTIAL** |
| `model_rollback.py` | Model versioning and rollback (Stage I) | **KEEP_ESSENTIAL** |
| `ood_detection.py` | Energy-based OOD detection (Stage H) | **KEEP_ESSENTIAL** |
| `active_learning_export.py` | Uncertainty-based sample selection (Stage H) | **KEEP_ESSENTIAL** |
| `prometheus_metrics.py` | Prometheus metric recording/export (Stage H) | **KEEP_ESSENTIAL** |
| `deployment_manager.py` | Deployment management (Stage I) | **KEEP_ESSENTIAL** |
| `pipeline_orchestrator.py` | Pipeline orchestration class | **KEEP_ESSENTIAL** |
| `diagnostic_pipeline_check.py` | Diagnostic checks for pipeline health | **KEEP_REFERENCE** |

## 9.2 src/core/ (3 files)

| File | Purpose | Verdict |
|------|---------|---------|
| `__init__.py` | Package init | **KEEP_ESSENTIAL** |
| `exception.py` | Custom exception classes | **KEEP_ESSENTIAL** |
| `logger.py` | Structured logging module | **KEEP_ESSENTIAL** |

## 9.3 src/entity/ (3 files)

| File | Purpose | Verdict |
|------|---------|---------|
| `__init__.py` | Package init | **KEEP_ESSENTIAL** |
| `config_entity.py` | `@dataclass` config entities (`PipelineConfig`, etc.) | **KEEP_ESSENTIAL** |
| `artifact_entity.py` | `@dataclass` artifact tracking entities | **KEEP_ESSENTIAL** |

## 9.4 src/pipeline/ (2 files)

| File | Purpose | Verdict |
|------|---------|---------|
| `__init__.py` | Package init | **KEEP_ESSENTIAL** |
| `inference_pipeline.py` | Main `InferencePipeline` class — orchestrates all stages | **KEEP_ESSENTIAL** |

## 9.5 src/Archived(prepare traning- production- conversion)/ (6 files)

| File | Purpose | Verdict |
|------|---------|---------|
| `Archived files.txt` | Note explaining archived status | **ARCHIVE** |
| `convert_production_units.py` | Old unit conversion script | **ARCHIVE** — superseded by `preprocess_data.py` |
| `prepare_production_data.py` | Old production prep | **ARCHIVE** — superseded |
| `prepare_training_data.py` | Old training prep | **ARCHIVE** — superseded |
| `validate_garmin_data.py` | Old Garmin validation | **ARCHIVE** — superseded by `data_validator.py` |
| `validate_model_and_diagnose.py` | Old diagnostic script | **ARCHIVE** |
| `training_cv_experiment/` | Old training experiments folder | **ARCHIVE** — historical reference only |

---

# 10. `data/` FOLDER

| Item | Type | Purpose | Verdict |
|------|------|---------|---------|
| `raw/` | Directory | Original raw Garmin sensor data (.xlsx, .csv) | **KEEP_ESSENTIAL** — DVC tracked |
| `preprocessed/` | Directory | Intermediate fused sensor data | **KEEP_ESSENTIAL** |
| `processed/` | Directory | DVC tracked processed data | **KEEP_ESSENTIAL** |
| `prepared/` | Directory | Model-ready `.npy` windows + predictions | **KEEP_ESSENTIAL** |
| `raw.dvc` | DVC | DVC pointer for raw data (64MB) | **KEEP_ESSENTIAL** |
| `processed.dvc` | DVC | DVC pointer for processed data | **KEEP_ESSENTIAL** |
| `prepared.dvc` | DVC | DVC pointer for prepared data (47MB) | **KEEP_ESSENTIAL** |
| `all_users_data_labeled.csv` | CSV | ADAMSense labeled training data | **KEEP_ESSENTIAL** |
| `anxiety_dataset.csv` | CSV | Additional anxiety dataset | **KEEP_REFERENCE** |
| `samples_2005 dataset/` | Directory | Contains `f_data_50hz.csv` — reference dataset at 50Hz | **KEEP_REFERENCE** |
| `.gitignore` | — | Data-specific git rules | **KEEP_ESSENTIAL** |

---

# 11. `reports/` FOLDER

| Subfolder | Contents | Purpose | Verdict |
|-----------|----------|---------|---------|
| `inference_smoke/` | 3 JSON files (Jan 9 smoke test results) | Inference validation results | **KEEP_REFERENCE** |
| `monitoring/` | 6 subdirectories with confidence/drift/temporal reports | Monitoring output from pipeline runs | **KEEP_REFERENCE** |
| `preprocess_qc/` | 6 JSON files (QC check results) | Preprocessing validation results | **KEEP_REFERENCE** |

---

# 12. `.github/workflows/` (1 file)

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `ci-cd.yml` | 282 | CI/CD: lint → test → build Docker → deploy | **KEEP_ESSENTIAL** |

---

# 13. `notebooks/` FOLDER

| File | Purpose | Verdict |
|------|---------|---------|
| `README.md` | Notebook overview guide | **KEEP_REFERENCE** |
| `data_preprocessing_step1.ipynb` | Original preprocessing notebook | **KEEP_REFERENCE** — historical |
| `production_preprocessing.ipynb` | Production preprocessing workflow | **KEEP_REFERENCE** |
| `from_guide_processing.ipynb` | Experimental notebook | **ARCHIVE** |

---

# 14. `images/` FOLDER (15 files)

| File | Type | Purpose | Verdict |
|------|------|---------|---------|
| `1D CNN-3BiLSTm.png` | PNG | Model architecture diagram | **KEEP_ESSENTIAL** — thesis figure |
| `1D CNN-BiLSTM.png` | PNG | Model architecture diagram | **KEEP_ESSENTIAL** — thesis figure |
| `1DCNN-BiLSTM_for_Wearable_Anxiety_Detection.pdf` | PDF | Paper duplicate | **DELETE** — exists in `papers/` |
| `83f2361f.png` | PNG | Unknown image | **ARCHIVE** |
| `Data structure of garmin.png` | PNG | Garmin data structure diagram | **KEEP_REFERENCE** |
| `garmin data -Motion_Data_Intelligence.pdf` | PDF | Paper duplicate | **DELETE** — reference material |
| `Gemini_Generated_Image_kk7036kk7036kk70.png` | PNG | AI-generated image | **DELETE** |
| `LM01.png`, `LM02.png` | PNG | Learning material screenshots | **ARCHIVE** |
| `mlflow.png` | PNG | MLflow screenshot | **KEEP_REFERENCE** |
| `MLOps_Production_System_Blueprint.pdf` | PDF | MLOps blueprint | **ARCHIVE** |
| `PIPELINE LM1.png` | PNG | Pipeline diagram | **KEEP_REFERENCE** |
| `Prognosis model.png` | PNG | Prognosis model diagram | **KEEP_REFERENCE** |
| `Prognosis_Models_Building_Predictive_Foresight.pdf` | PDF | Reference document | **ARCHIVE** |
| `unnamed.jpg` | JPG | Unknown image | **DELETE** |

---

# 15. `models/` FOLDER

| Item | Purpose | Verdict |
|------|---------|---------|
| `pretrained/` | Contains `fine_tuned_model_1dcnnbilstm.keras` model weights | **KEEP_ESSENTIAL** |
| `pretrained.dvc` | DVC pointer for model | **KEEP_ESSENTIAL** |
| `normalized_baseline.json` | Normalized baseline for drift detection | **KEEP_ESSENTIAL** |
| `.gitignore` | Model-specific git rules | **KEEP_ESSENTIAL** |

---

# 16. `outputs/` FOLDER

| File | Purpose | Verdict |
|------|---------|---------|
| `predictions_20260212_105050.csv` | Latest prediction results | **KEEP_REFERENCE** — pipeline output |
| `predictions_20260212_105050_metadata.json` | Prediction metadata | **KEEP_REFERENCE** |
| `predictions_20260212_105050_probs.npy` | Prediction probabilities | **KEEP_REFERENCE** |
| `predictions_fresh.csv` | Fresh prediction results | **KEEP_REFERENCE** |
| `production_labels_fresh.npy` | Fresh production labels | **KEEP_REFERENCE** |
| `production_predictions_fresh.npy` | Fresh production predictions | **KEEP_REFERENCE** |
| `gravity_removal_comparison.png` | Gravity removal comparison figure | **KEEP_REFERENCE** |
| `evaluation/` | Contains evaluation_20251212 JSON + TXT | **KEEP_REFERENCE** |

---

# 17. `logs/` FOLDER

| Subfolder | Contents | Verdict |
|-----------|----------|---------|
| `evaluation/` | Evaluation logs | **KEEP_REFERENCE** — auto-generated |
| `inference/` | Inference logs | **KEEP_REFERENCE** — auto-generated |
| `pipeline/` | Pipeline execution logs + result JSON | **KEEP_REFERENCE** — auto-generated |
| `preprocessing/` | Preprocessing logs | **KEEP_REFERENCE** — auto-generated |
| `training/` | Training logs | **KEEP_REFERENCE** — auto-generated |

---

# SUMMARY STATISTICS

## By Category

| Category | Count | Action |
|----------|-------|--------|
| **KEEP_ESSENTIAL** | ~95 files | Do not touch |
| **KEEP_REFERENCE** | ~40 files | Keep, low priority |
| **ARCHIVE** | ~20 files | Move to `_archive/` |
| **DELETE** | ~8 files | Remove immediately |
| **OUTDATED** | ~3 files | Update or merge into current versions |

## Immediate DELETE Candidates (Safe)

| File | Reason |
|------|--------|
| `EHB_2025_71.pdf` (root) | Duplicate — exists in `papers/` |
| `ICTH_16.pdf` (root) | Duplicate — exists in `papers/` |
| `unnamed.jpg` (root) | Random image, no purpose |
| `New Microsoft Word Document (AutoRecovered).docx` | Empty temp file |
| `images/1DCNN-BiLSTM_for_Wearable_Anxiety_Detection.pdf` | Duplicate — exists in `papers/` |
| `images/Gemini_Generated_Image_kk7036kk7036kk70.png` | AI-generated, no use |
| `images/unnamed.jpg` | Duplicate of root unnamed.jpg |
| `docs/archive/Mondaymeet.md` | One-time meeting prep, obsolete |

## ARCHIVE Candidates (Move to `_archive/`)

| Item | Reason |
|------|--------|
| `ai helps/` (entire folder) | Valuable content already migrated to `docs/thesis/` |
| `cheat sheet/1697955590966.pdf` | Personal reference |
| `src/Archived(...)/*` | Already marked as archived old code |
| `docs/archive/extranotes.md` | Reference, not active |
| `docs/archive/LATER_Offline_MLOps_Guide.md` | Future reference |
| `images/MLOps_Production_System_Blueprint.pdf` | Reference material |
| `images/Prognosis_Models_Building_Predictive_Foresight.pdf` | Reference material |
| `notebooks/from_guide_processing.ipynb` | Experimental |

## OUTDATED Files (Merge or Update)

| File | Superseded By |
|------|---------------|
| `MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md` | This inventory + `FEBRUARY_2026_ACTION_PLAN.md` |
| `docs/THESIS_PROGRESS_DASHBOARD_2026-01-20.md` | `docs/THESIS_MASTER_PROGRESS_2026-01-31.md` |
| `docs/REPOSITORY_CLEANUP_ANALYSIS_2026-01-18.md` | This inventory |

## Key Issues Found

1. **Folder typo:** `docs/thesis/production refrencxe/` should be `production_reference/`
2. **Absolute paths:** `scripts/create_normalized_baseline.py` uses hardcoded `D:\study apply\...` paths
3. **Duplicate PDFs:** 3 PDFs in root/images that exist in `papers/`
4. **Documentation sprawl:** ~15 markdown files serve overlapping purposes (progress dashboards, Q&A files, pipeline guides). Consider consolidating
5. **`docs/stages/00_STAGE_INDEX.md`** lists `train.py` and `trigger_policy.py` as missing but they now exist in `src/`

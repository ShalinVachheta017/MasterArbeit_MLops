# 02 — File/Folder Inventory and Role Map

> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`
> **Audit Date:** 2026-02-22

---

## 1. Source Code — `src/` (Core Modules)

| File | Role / Purpose | Status | Thesis Contribution | Evidence |
|------|----------------|--------|--------------------:|----------|
| `src/train.py` | HARTrainer, DomainAdaptationTrainer, HARModelBuilder, pseudo-label retraining | **Implemented + Integrated** | Core — model architecture + all training modes | [CODE: src/train.py:L1-L1219] ~850 code lines |
| `src/trigger_policy.py` | TriggerPolicyEngine: 2-of-3 voting, cooldown, escalation, proxy validation | **Implemented + Integrated** | Core — retraining decision logic | [CODE: src/trigger_policy.py:L1-L822] ~550 code lines |
| `src/calibration.py` | TemperatureScaler, MCDropoutEstimator, CalibrationEvaluator, ECE/Brier/MCE | **Implemented** (not orchestrated) | Important — confidence reliability | [CODE: src/calibration.py:L1-L544] ~370 code lines |
| `src/wasserstein_drift.py` | WassersteinDriftDetector, ChangePointDetector, multi-resolution analysis, integrated PSI+KS+Wasserstein | **Implemented** (not orchestrated) | Important — drift detection depth | [CODE: src/wasserstein_drift.py:L1-L460] ~320 code lines |
| `src/curriculum_pseudo_labeling.py` | CurriculumTrainer, PseudoLabelSelector, EWCRegularizer, teacher-student EMA | **Implemented** (not orchestrated) | Important — advanced retraining | [CODE: src/curriculum_pseudo_labeling.py:L1-L460] ~320 code lines |
| `src/sensor_placement.py` | AxisMirrorAugmenter, HandDetector, HandPerformanceReporter | **Implemented** (not orchestrated) | Supporting — placement robustness | [CODE: src/sensor_placement.py:L1-L370] ~260 code lines |
| `src/model_rollback.py` | ModelRegistry: register/deploy/rollback, SHA256 hashing, inference validation | **Implemented + Integrated** | Core — model governance | [CODE: src/model_rollback.py:L1-L532] ~350 code lines |
| `src/prometheus_metrics.py` | MetricsExporter (singleton), Prometheus text format, HTTP server, JSON export | **Implemented** (not in default path) | Supporting — operational monitoring | [CODE: src/prometheus_metrics.py:L1-L623] ~420 code lines |
| `src/config.py` | Project paths, constants, model paths | **Implemented + Integrated** | Infrastructure | [CODE: src/config.py] |
| `src/data_validator.py` | DataValidator: schema, range, missing, sampling checks | **Implemented + Integrated** | Core — data quality | [CODE: src/data_validator.py] |
| `src/preprocess_data.py` | UnifiedPreprocessor, UnitDetector, GravityRemover, DomainCalibrator | **Implemented + Integrated** | Core — preprocessing | [CODE: src/preprocess_data.py] |
| `src/run_inference.py` | InferencePipeline: batch inference, confidence stats | **Implemented + Integrated** | Core — model inference | [CODE: src/run_inference.py] |
| `src/evaluate_predictions.py` | EvaluationPipeline: labeled + unlabeled paths | **Implemented + Integrated** | Core — evaluation | [CODE: src/evaluate_predictions.py] |
| `src/mlflow_tracking.py` | MLflowTracker: experiment tracking integration | **Implemented + Integrated** | Supporting — experiment tracking | [CODE: src/mlflow_tracking.py] |
| `src/ood_detection.py` | Out-of-distribution detection utilities | **Implemented** (partial use) | Supporting — OOD signals | [CODE: src/ood_detection.py] |
| `src/robustness.py` | Robustness testing utilities | **Implemented** | Supporting — robustness analysis | [CODE: src/robustness.py] |
| `src/active_learning_export.py` | Active learning sample export | **Implemented** | Future work | [CODE: src/active_learning_export.py] |
| `src/deployment_manager.py` | Deployment management utilities | **Implemented** | Supporting — deployment | [CODE: src/deployment_manager.py] |
| `src/diagnostic_pipeline_check.py` | Pipeline diagnostic checks | **Implemented** | Debugging | [CODE: src/diagnostic_pipeline_check.py] |
| `src/sensor_data_pipeline.py` | Sensor data processing pipeline | **Implemented** | Supporting | [CODE: src/sensor_data_pipeline.py] |

---

## 2. Source Code — `src/components/` (Pipeline Stage Wrappers)

| File | Stage | Wraps | Status | Evidence |
|------|-------|-------|--------|----------|
| `data_ingestion.py` | Stage 1 | Self-contained (full sensor fusion) | **Integrated + Validated** | [CODE: src/components/data_ingestion.py:L1-L607] ~400 lines; [LOG: pipeline results show ingestion] |
| `data_validation.py` | Stage 2 | `src.data_validator.DataValidator` | **Integrated + Validated** | [CODE: src/components/data_validation.py:L1-L60]; [LOG: validation stage in results] |
| `data_transformation.py` | Stage 3 | `src.preprocess_data.UnifiedPreprocessor` + sub-components | **Integrated + Validated** | [CODE: src/components/data_transformation.py:L1-L130] |
| `model_inference.py` | Stage 4 | `src.run_inference.InferencePipeline` | **Integrated + Validated** | [CODE: src/components/model_inference.py:L1-L130] |
| `model_evaluation.py` | Stage 5 | `src.evaluate_predictions.EvaluationPipeline` | **Integrated + Validated** | [CODE: src/components/model_evaluation.py:L1-L70] |
| `post_inference_monitoring.py` | Stage 6 | `scripts.post_inference_monitoring.PostInferenceMonitor` | **Integrated + Validated** | [CODE: src/components/post_inference_monitoring.py:L1-L105] |
| `trigger_evaluation.py` | Stage 7 | `src.trigger_policy.TriggerPolicyEngine` | **Integrated** (placeholder zeros) | [CODE: src/components/trigger_evaluation.py:L73-L82] — 4 hardcoded zeros |
| `model_retraining.py` | Stage 8 | `src.domain_adaptation.*`, `src.train.*` | **Integrated + Validated** | [CODE: src/components/model_retraining.py:L1-L310] 5 methods |
| `model_registration.py` | Stage 9 | `src.model_rollback.ModelRegistry` | **Integrated** (placeholder `is_better`) | [CODE: src/components/model_registration.py:L69-L75] — hardcoded `True` |
| `baseline_update.py` | Stage 10 | `scripts.build_training_baseline.BaselineBuilder` | **Integrated + Validated** | [CODE: src/components/baseline_update.py:L1-L140] governance-aware |
| `calibration_uncertainty.py` | Stage 11 | `src.calibration.*` | **Implemented** (not orchestrated) | [CODE: src/components/calibration_uncertainty.py:L1-L140] |
| `wasserstein_drift.py` | Stage 12 | `src.wasserstein_drift.*` | **Implemented** (not orchestrated, field mismatch) | [CODE: src/components/wasserstein_drift.py:L83] — `calibration_warnings` field bug |
| `curriculum_pseudo_labeling.py` | Stage 13 | `src.curriculum_pseudo_labeling.*` | **Implemented** (not orchestrated) | [CODE: src/components/curriculum_pseudo_labeling.py:L1-L140] |
| `sensor_placement.py` | Stage 14 | `src.sensor_placement.*` | **Implemented** (not orchestrated) | [CODE: src/components/sensor_placement.py:L1-L110] detection only |

---

## 3. Source Code — `src/domain_adaptation/`

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `adabn.py` | Adaptive Batch Normalization — BN stats recalibration | **Implemented + Integrated** (via Stage 8) | [CODE: src/domain_adaptation/adabn.py:L1-L155] ~100 code lines |
| `tent.py` | Test-time Entropy Minimization — BN affine update with OOD guard + rollback | **Implemented + Integrated** (via Stage 8) | [CODE: src/domain_adaptation/tent.py:L1-L255] ~180 code lines; documented bug fix |

---

## 4. Source Code — `src/pipeline/`

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `production_pipeline.py` | Main orchestrator: runs stages 1-10, artifact passing, MLflow logging | **Implemented + Validated** | [CODE: src/pipeline/production_pipeline.py:L1-L689] ~450 code lines; 60 pipeline results |
| `inference_pipeline.py` | Lightweight inference-only pipeline | **Implemented** | [CODE: src/pipeline/inference_pipeline.py] |

---

## 5. Source Code — `src/entity/`

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `artifact_entity.py` | 14 stage artifact dataclasses + PipelineResult | **Implemented + Integrated** | [CODE: src/entity/artifact_entity.py:L1-L271] — all 14 stages defined |
| `config_entity.py` | Stage configuration dataclasses | **Implemented + Integrated** | [CODE: src/entity/config_entity.py] |

---

## 6. Entry Points

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `run_pipeline.py` | CLI entry — 20+ args, YAML config, stages 1-14 in argparse | **Implemented** (stages 11-14 silently ignored) | [CODE: run_pipeline.py:L1-L510] ~340 code lines |
| `src/api/app.py` | FastAPI REST API with embedded dashboard | **Implemented** | [CODE: src/api/app.py:L1-L775] ~500 code lines |
| `docker/api/main.py` | Docker-specific API entry | **Implemented** | [CODE: docker/api/main.py] |

---

## 7. Tests — `tests/`

| File | # Tests | Markers | Real Models? | Modules Covered |
|------|---------|---------|-------------|-----------------|
| `test_pipeline_integration.py` | 7 | `integration` | No (mocked) | ProductionPipeline, PipelineResult |
| `test_adabn.py` | 7 | `slow` | Yes (Keras) | adabn.adapt_bn_statistics, BN layer detection |
| `test_trigger_policy.py` | 14 | — | No | TriggerPolicyEngine, thresholds, escalation, proxy |
| `test_calibration.py` | 11 | — | No | TemperatureScaler, CalibrationEvaluator |
| `test_wasserstein_drift.py` | 10 | — | No | WassersteinDriftDetector, ChangePointDetector |
| `test_retraining.py` | 3 | `slow` | Yes (Keras) | ModelRetraining component (AdaBN path) |
| `test_data_validation.py` | ~15 | — | No | DataValidator |
| `test_preprocessing.py` | ~12 | — | No | UnifiedPreprocessor, windowing |
| `test_model_rollback.py` | ~12 | — | No | ModelRegistry, rollback |
| `test_drift_detection.py` | ~10 | — | No | Drift detection utilities |
| `test_ood_detection.py` | ~8 | — | No | OOD detection |
| `test_sensor_placement.py` | ~10 | — | No | HandDetector, AxisMirrorAugmenter |
| `test_curriculum_pseudo_labeling.py` | ~10 | — | No | CurriculumTrainer |
| `test_baseline_update.py` | ~8 | — | No | BaselineUpdate, governance |
| `test_robustness.py` | ~10 | — | No | Robustness checks |
| `test_prometheus_metrics.py` | ~8 | — | No | MetricsExporter |
| `test_active_learning.py` | ~5 | — | No | ActiveLearningExport |
| `test_progress_dashboard.py` | ~5 | — | No | Progress dashboard |
| `conftest.py` | 12 fixtures | — | No | Shared test data |
| **TOTAL** | **~215** | | | |

**FACT:** 215 test functions verified via grep. Confidence: **High**
[TEST: tests/test_*.py | grep "def test_" count = 215]

---

## 8. Scripts — `scripts/`

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `audit_artifacts.py` | Artifact completeness/consistency checker | **Implemented + Validated** | [CODE: scripts/audit_artifacts.py]; 12/12 pass documented |
| `analyze_drift_across_datasets.py` | Cross-dataset drift statistics | **Implemented** | [CODE: scripts/analyze_drift_across_datasets.py] |
| `build_training_baseline.py` | Baseline statistics builder | **Implemented + Integrated** | [CODE: scripts/build_training_baseline.py]; used by Stage 10 |
| `build_normalized_baseline.py` | Normalized baseline builder | **Implemented** | [CODE: scripts/build_normalized_baseline.py] |
| `post_inference_monitoring.py` | 3-layer monitoring implementation | **Implemented + Integrated** | [CODE: scripts/post_inference_monitoring.py]; used by Stage 6 |
| `verify_repository.py` | Repository integrity checker | **Implemented** | [CODE: scripts/verify_repository.py] |
| `export_mlflow_runs.py` | MLflow experiment data export | **Implemented** | [CODE: scripts/export_mlflow_runs.py] |
| `generate_thesis_figures.py` | Thesis figure generation | **Implemented** | [CODE: scripts/generate_thesis_figures.py] |
| `per_dataset_inference.py` | Per-dataset inference runner | **Implemented** | [CODE: scripts/per_dataset_inference.py] |
| `preprocess.py` | Standalone preprocessing script | **Implemented** | [CODE: scripts/preprocess.py] |
| `preprocess_qc.py` | Preprocessing quality checks | **Implemented** | [CODE: scripts/preprocess_qc.py] |
| `update_progress_dashboard.py` | Progress dashboard update | **Implemented** | [CODE: scripts/update_progress_dashboard.py] |
| `run_tests.py` | Test runner script | **Implemented** | [CODE: scripts/run_tests.py] |
| `train.py` | Standalone training script | **Implemented** | [CODE: scripts/train.py] |

**RISK:** `scripts/inference_smoke.py` is referenced by CI/CD but does **NOT exist**. Confidence: **High**
[CFG: .github/workflows/ci-cd.yml | integration-test job references inference_smoke.py]

---

## 9. Configuration Files

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `config/pipeline_config.yaml` | Preprocessing, validation, inference params | **Implemented + Integrated** | [CFG: config/pipeline_config.yaml] |
| `config/mlflow_config.yaml` | MLflow tracking setup | **Implemented** | [CFG: config/mlflow_config.yaml] |
| `config/prometheus.yml` | Prometheus scrape config (6 targets) | **Implemented** (not proven running) | [CFG: config/prometheus.yml] |
| `config/alerts/har_alerts.yml` | Alert rules (4 conditions) | **Implemented** (not proven active) | [CFG: config/alerts/har_alerts.yml] |
| `config/grafana/har_dashboard.json` | Grafana dashboard | **Implemented** (not proven deployed) | [CFG: config/grafana/har_dashboard.json] |
| `pyproject.toml` | Package metadata, deps, markers | **Implemented** | [CFG: pyproject.toml] v2.1.0 |
| `pytest.ini` | Test configuration | **Implemented** | [CFG: pytest.ini] strict markers |

---

## 10. Docker & Deployment

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `docker/Dockerfile.inference` | Inference container (python:3.11-slim, uvicorn) | **Implemented** | [CODE: docker/Dockerfile.inference] port 8000, healthcheck |
| `docker/Dockerfile.training` | Training container | **Implemented** | [CODE: docker/Dockerfile.training] |
| `docker/api/main.py` | Docker API entry | **Implemented** | [CODE: docker/api/main.py] |
| `docker-compose.yml` | 4 services: mlflow, inference, training, preprocessing | **Implemented** | [CFG: docker-compose.yml] profiles for on-demand services |

---

## 11. CI/CD

| File | Purpose | Status | Evidence |
|------|---------|--------|----------|
| `.github/workflows/ci-cd.yml` | 7-job pipeline (lint, test, test-slow, build, integration, model-validation, notify) | **Partial** | [CFG: .github/workflows/ci-cd.yml] — missing `on.schedule`, placeholder echo steps, missing smoke script |

---

## 12. Logs, Artifacts, and Run Evidence

| Path | Content | Count | Status |
|------|---------|------:|--------|
| `logs/pipeline/pipeline_result_*.json` | Pipeline run results | 60 | **Strong evidence** of repeated execution |
| `artifacts/` | Timestamped artifact snapshots | 32 | **Strong evidence** of artifact generation |
| `models/registry/` | Registered model + registry metadata | 2 files | **Evidence** of model registration |
| `mlruns/` | MLflow experiment data | Present | **Evidence** of experiment tracking |
| `logs/` (root) | 39+ log files + subdirectories | Multiple | **Evidence** of operational logging |

---

## 13. Documentation — `docs/`

| Folder | # Files | Purpose | Trust Level |
|--------|--------:|---------|-------------|
| `docs/19_Feb/` | 4 | Previous work cycle documentation | Reference only |
| `docs/archive/` | 23 | Historical/deprecated docs (OLD_* prefixed) | Reference only |
| `docs/figures/` | 7 | Pipeline and analysis figures (PNG) | Useful for thesis |
| `docs/research/` | 7 | Paper index, bibliography, QnA analysis | Reference for literature |
| `docs/stages/` | 11 | Per-stage documentation | Reference only |
| `docs/technical/` | 10 | CI/CD guides, operations, monitoring | Reference only |
| `docs/thesis/` | 16+ | Thesis structure, chapters, plans | Partially drafted |

**RISK:** Existing docs claim "95% complete" and "225 tests passing" — these numbers are NOT independently verified and should not be cited. Confidence: **High**
[DOC: README.md — claims unverified]

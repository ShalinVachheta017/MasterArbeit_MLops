# Pipeline Factsheet — HAR MLOps Production Pipeline
**Extracted from codebase on: 2026-02-26**
**Version: 3.1.0 (current branch: main)**

---

## Entrypoints

| Entrypoint | File | Class / Callable | Port / Mode |
|---|---|---|---|
| **Batch CLI** | `run_pipeline.py:1` | `parse_args()` → `ProductionPipeline.run()` | CLI |
| **REST API** | `src/api/app.py:865` | FastAPI `app`, `if __name__ == "__main__"` → uvicorn on `0.0.0.0:8000` | HTTP port 8000 |
| **Training (direct)** | `src/train.py:1287` | `HARTrainer(config)` | CLI (`python src/train.py`) |
| **Pipeline Orchestrator** | `src/pipeline/production_pipeline.py:79` | `ProductionPipeline` class | internal |

---

## Ordered Stage Registry
Defined in `src/pipeline/production_pipeline.py:54–68`:
```
ALL_STAGES = ["ingestion","validation","transformation","inference","evaluation",
              "monitoring","trigger","retraining","registration","baseline_update",
              "calibration","wasserstein_drift","curriculum_pseudo_labeling","sensor_placement"]
RETRAIN_STAGES  = {"retraining","registration","baseline_update"}   # line 69
ADVANCED_STAGES = {"calibration","wasserstein_drift","curriculum_pseudo_labeling","sensor_placement"}  # line 72
```
Default run (no flags): stages 1–7.
`--retrain` adds stages 8–10; `--advanced` adds stages 11–14.

---

## Stage-by-Stage Detail

---

### Stage 1 — Data Ingestion

| Field | Value |
|---|---|
| **Purpose** | Merge raw Garmin accelerometer + gyroscope Excel/CSV files into a single time-aligned, 50 Hz fused CSV. |
| **Module** | `src/components/data_ingestion.py` |
| **Main class** | `DataIngestion` (line 1); method `initiate_data_ingestion()` |
| **Underlying module** | `discover_sensor_files()` at `data_ingestion.py:80`; `DataIngestion` merges via `pd.merge_asof` internally |
| **Input paths** | `data/raw/` (auto-discover accel/gyro pair) OR `--input-csv` override (`DataIngestionConfig.input_csv`) |
| **Output artifact** | `DataIngestionArtifact` (`artifact_entity.py:24`) |
| **Output file** | `data/processed/sensor_fused_50Hz.csv` (`data_ingestion.py:26, 361`) |
| **Skip cache** | `ingestion_manifest.json` in `data/processed/` prevents reprocessing already-fused pairs (`data_ingestion.py:71`) |
| **Config class** | `DataIngestionConfig` (`config_entity.py:83`) |
| **Config keys** | `target_hz=50`, `merge_tolerance_ms=1`, `accel_file=None` (auto), `gyro_file=None` (auto), `input_csv=None` |
| **Failure modes** | (a) No accel/gyro file pair found → `FileNotFoundError`; (b) timestamp mismatch > `merge_tolerance_ms` → row drop / sparse CSV; (c) corrupt Excel → `openpyxl.BadZipFile`; (d) column rename mismatch (`ACCEL_RENAME/GYRO_RENAME` dictionaries at `data_ingestion.py:54-68`) → `KeyError` in downstream normalization |
| **Safeguards** | Manifest-based skip; 3 discovery paths (root → Decoded/ → sub-folders, `data_ingestion.py:80+`); unit-detection fallback |
| **Tests** | NOT VERIFIED (no `test_data_ingestion.py` found in `tests/`) |

---

### Stage 2 — Data Validation

| Field | Value |
|---|---|
| **Purpose** | Validate the fused CSV for schema correctness, sensor-range plausibility, and acceptable missing-data ratio before windowing. |
| **Module** | `src/components/data_validation.py` |
| **Main class** | `DataValidation` (line 1); method `initiate_data_validation()` |
| **Underlying module** | `src/data_validator.py → DataValidator.validate()` |
| **Input** | `DataIngestionArtifact.fused_csv_path` |
| **Output artifact** | `DataValidationArtifact` (`artifact_entity.py:36`) — fields: `is_valid`, `errors`, `warnings`, `stats` |
| **Output file** | `artifacts/<timestamp>/validation/validation_report.json` (`production_pipeline.py:258`) |
| **Config class** | `DataValidationConfig` (`config_entity.py:94`) |
| **Config keys** | `sensor_columns=["Ax","Ay","Az","Gx","Gy","Gz"]`, `expected_frequency_hz=50.0`, `max_acceleration_ms2=50.0`, `max_gyroscope_dps=500.0`, `max_missing_ratio=0.05` |
| **Failure modes** | (a) `is_valid=False` → pipeline breaks unless `--continue-on-failure` is set (`production_pipeline.py:271-275`); (b) **GAP**: `--continue-on-failure` bypasses the break, passing invalid data downstream |
| **Safeguards** | `break` on `is_valid=False` when `continue_on_failure=False` (`production_pipeline.py:273-275`) |
| **Tests** | `tests/test_data_validation.py` (file present) |

---

### Stage 3 — Data Transformation

| Field | Value |
|---|---|
| **Purpose** | Preprocess the fused CSV into normalised, sliding-window `production_X.npy` arrays for inference. |
| **Module** | `src/components/data_transformation.py` |
| **Main class** | `DataTransformation` (line 1); method `initiate_data_transformation()` |
| **Underlying module** | `src/preprocess_data.py → UnifiedPreprocessor`, `UnitDetector`, `GravityRemover`, `DomainCalibrator` |
| **Input** | `DataIngestionArtifact.fused_csv_path` (or `DataTransformationConfig.input_csv` override) |
| **Output artifact** | `DataTransformationArtifact` (`artifact_entity.py:52`) |
| **Output files** | `data/prepared/production_X.npy`, `data/prepared/preprocessing_metadata.json` |
| **Config class** | `DataTransformationConfig` (`config_entity.py:103`) |
| **Config keys** | `window_size=200` (4 s @ 50 Hz), `overlap=0.5`, `enable_unit_conversion=True` (milliG→m/s²), `enable_gravity_removal=False` (must match training), `enable_calibration=False` |
| **Training-consistency note** | `enable_gravity_removal=False` is intentional: training data retained gravity; changing to `True` would introduce a train/inference distribution shift. Comment at `config_entity.py:110`. |
| **Failure modes** | (a) Wrong column names → `KeyError` in `detect_data_format()`; (b) Mismatched `enable_unit_conversion` vs. training → silent scale error |
| **Safeguards** | `data_format, sensor_cols = preprocessor.detect_data_format(df)` auto-detection (`data_transformation.py:68`); metadata saved alongside npy |
| **Tests** | `tests/test_preprocessing.py` |

---

### Stage 4 — Model Inference

| Field | Value |
|---|---|
| **Purpose** | Run batch inference with the pretrained 1D-CNN-BiLSTM model on windows, producing class predictions and per-window probability arrays. |
| **Module** | `src/components/model_inference.py` |
| **Main class** | `ModelInference` (line 1); method `initiate_model_inference()` |
| **Underlying module** | `src/run_inference.py → InferencePipeline.run()` |
| **Input** | `DataTransformationArtifact.production_X_path` (default: `data/prepared/production_X.npy`) |
| **Model file** | `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` (default; override via `ModelInferenceConfig.model_path`) |
| **Output artifact** | `ModelInferenceArtifact` (`artifact_entity.py:69`) |
| **Output files** | `data/prepared/predictions/predictions_fresh.csv`, `data/prepared/predictions/production_predictions_fresh.npy`, optional `probabilities_<timestamp>.npy` |
| **Config class** | `ModelInferenceConfig` (`config_entity.py:118`) |
| **Config keys** | `batch_size=32`, `confidence_threshold=0.50`, `mode="batch"`, `model_path=None` (auto) |
| **Failure modes** | (a) Model file missing → `FileNotFoundError`; (b) Input npy shape mismatch vs. model input shape → TF error; (c) No model hash/checksum check — any `.keras` file at path is loaded silently |
| **Safeguards** | Timing captured (`inference_time`); artifact manager saves output files |
| **Tests** | NOT VERIFIED (no `test_model_inference.py` in `tests/`) |

---

### Stage 5 — Model Evaluation

| Field | Value |
|---|---|
| **Purpose** | Compute confidence distribution, expected calibration error (ECE), and (if labels available) per-class F1/precision/recall on the inference output. |
| **Module** | `src/components/model_evaluation.py` |
| **Main class** | `ModelEvaluation` (line 1); method `initiate_model_evaluation()` |
| **Underlying module** | `src/evaluate_predictions.py → EvaluationPipeline.run()` |
| **Input** | `ModelInferenceArtifact.predictions_csv_path` |
| **Output artifact** | `ModelEvaluationArtifact` (`artifact_entity.py:86`) |
| **Output files** | `outputs/evaluation/evaluation_report.json`, `outputs/evaluation/evaluation_report.txt` |
| **Config class** | `ModelEvaluationConfig` (`config_entity.py:130`) |
| **Config keys** | `labels_path=None` (unlabeled mode), `confidence_bins=10` |
| **Failure modes** | (a) `predictions_csv` missing → `FileNotFoundError`; (b) No labels → `has_labels=False`; `classification_metrics=None` — metrics cannot be compared to SLA |
| **Safeguards** | Graceful degraded mode when no labels (`model_evaluation.py:35`, `EvaluationPipeline.run()`) |
| **Tests** | NOT VERIFIED (no `test_model_evaluation.py` in `tests/`) |

---

### Stage 6 — Post-Inference Monitoring

| Field | Value |
|---|---|
| **Purpose** | Run 3-layer monitoring (Layer 1: confidence, Layer 2: temporal transitions, Layer 3: z-score drift vs. baseline) on each inference batch. |
| **Module** | `src/components/post_inference_monitoring.py` |
| **Main class** | `PostInferenceMonitoring` (line 1); method `initiate_post_inference_monitoring()` |
| **Underlying module** | `scripts/post_inference_monitoring.py → PostInferenceMonitor.run()` |
| **Input** | `ModelInferenceArtifact.predictions_csv_path`, `DataTransformationArtifact.production_X_path`, `models/normalized_baseline.json` |
| **Calibration handoff** | Loads `outputs/calibration/temperature.json` if it exists from prior Stage 11 run (`post_inference_monitoring.py:54-65`) |
| **Baseline guard** | Logs WARNING if `baseline_json` is `> max_baseline_age_days=90` days old (`post_inference_monitoring.py:90-100`) |
| **Output artifact** | `PostInferenceMonitoringArtifact` (`artifact_entity.py:103`) — `overall_status`: HEALTHY/WARNING/CRITICAL |
| **Output file** | `outputs/evaluation/monitoring_report.json` |
| **Config class** | `PostInferenceMonitoringConfig` (`config_entity.py:143`) |
| **Config keys (thresholds)** | `confidence_warn_threshold=0.60`, `uncertain_pct_threshold=30.0`, `uncertain_window_threshold=0.50`, `transition_rate_threshold=50.0`, `drift_zscore_threshold=2.0`, `calibration_temperature=1.0`, `max_baseline_age_days=90` |
| **Failure modes** | (a) `baseline_stats_json` missing → Layer 3 silently skipped with WARNING log (not CRITICAL); (b) stale baseline logs WARNING only — no metric published, no automated trigger; (c) `is_training_session=True` can mask drift silently |
| **Safeguards** | Baseline missing → Layer 3 skip with log (`post_inference_monitoring.py:83-90`); baseline age check at `post_inference_monitoring.py:90-100` |
| **Tests** | NOT VERIFIED — no dedicated `test_post_inference_monitoring.py` in `tests/`; Prometheus metrics tested in `tests/test_prometheus_metrics.py` |

---

### Stage 7 — Trigger Evaluation

| Field | Value |
|---|---|
| **Purpose** | Decide whether to queue/trigger automated retraining based on monitoring metrics, with a cooldown gate to prevent over-training. |
| **Module** | `src/components/trigger_evaluation.py` |
| **Main class** | `TriggerEvaluation` (line 1); method `initiate_trigger_evaluation()` |
| **Underlying module** | `src/trigger_policy.py → TriggerPolicyEngine`, `TriggerThresholds`, `CooldownConfig` |
| **Input** | `PostInferenceMonitoringArtifact` (layer1/2/3 results) |
| **Output artifact** | `TriggerEvaluationArtifact` (`artifact_entity.py:116`) — `should_retrain`, `action` (NONE/MONITOR/QUEUE_RETRAIN/TRIGGER_RETRAIN/ROLLBACK), `alert_level` (INFO/WARNING/CRITICAL) |
| **State file** | `logs/trigger/trigger_state.json` (persists cooldown state across runs) |
| **Config class** | `TriggerEvaluationConfig` (`config_entity.py:200`) |
| **Config keys** | `confidence_warn=0.65`, `confidence_critical=0.50`, `drift_zscore_warn=2.0`, `drift_zscore_critical=3.0`, `temporal_flip_warn=0.35`, `temporal_flip_critical=0.50`, `cooldown_hours=24` |
| **Threshold note** | `confidence_warn=0.65` here is intentionally higher than `PostInferenceMonitoringConfig.confidence_warn_threshold=0.60` (tiered alerting: monitoring alerts first, trigger fires later) |
| **Failure modes** | (a) Corruption of `trigger_state.json` → engine resets to defaults; (b) cooldown can suppress legitimate cascading drift |
| **Safeguards** | State persistence with JSON; `consecutive_warnings` counter escalates from MONITOR → TRIGGER_RETRAIN |
| **Tests** | `tests/test_trigger_policy.py` (14+ tests including `test_degraded_metrics_trigger`, `test_state_persistence`, `test_consecutive_warnings_escalate`) |

---

### Stage 8 — Model Retraining

| Field | Value |
|---|---|
| **Purpose** | Retrain the 1D-CNN-BiLSTM model using one of four adaptation strategies (standard, AdaBN, TENT, adabn_tent, pseudo-label). |
| **Module** | `src/components/model_retraining.py` |
| **Main class** | `ModelRetraining` (line 1); method `initiate_model_retraining()` → delegates to `_run_adabn()`, `_run_tent()`, `_run_adabn_then_tent()`, `_run_pseudo_label()`, or `_run_standard()` |
| **Underlying modules** | `src/train.py → HARTrainer` (line 375), `DomainAdaptationTrainer` (line 710); `src/domain_adaptation/adabn.py → adapt_bn_statistics` |
| **NOT implemented** | `mmd`, `dann` — raises `NotImplementedError` (`model_retraining.py:68-74`) |
| **Input** | `TriggerEvaluationArtifact`, `DataTransformationArtifact.production_X_path` (unlabeled target) |
| **Output artifact** | `ModelRetrainingArtifact` (`artifact_entity.py:128`) — `retrained_model_path`, `metrics` dict, `adaptation_method` |
| **Output directory** | `models/retrained/` |
| **Config class** | `ModelRetrainingConfig` (`config_entity.py:219`) |
| **Config keys** | `epochs=100`, `batch_size=64`, `learning_rate=0.001`, `n_folds=5`, `enable_adaptation=False`, `adaptation_method="adabn"`, `adabn_n_batches=10` |
| **CLI flag** | `--adapt adabn_tent` → sets `adaptation_method="adabn_tent"` (`run_pipeline.py:~95`) |
| **Failure modes** | (a) `NotImplementedError` for `mmd`/`dann`; (b) AdaBN/TENT produce no `val_accuracy` (unsupervised) → Stage 9 has no metric to compare; (c) TENT divergence → rollback guard in `_run_adabn_then_tent()` |
| **Safeguards** | NotImplementedError guard; rollback if entropy increases in adabn_tent (`model_retraining.py` — rollback block in `_run_adabn_then_tent`) |
| **Tests** | `tests/test_retraining.py`, `tests/test_adabn.py` |

---

### Stage 9 — Model Registration

| Field | Value |
|---|---|
| **Purpose** | Version and register the retrained model in the local registry; optionally deploy if proxy validation passes the tolerance gate. |
| **Module** | `src/components/model_registration.py` |
| **Main class** | `ModelRegistration` (line 1); method `initiate_model_registration()` |
| **Underlying module** | `src/model_rollback.py → ModelRegistry.register_model()` |
| **Input** | `ModelRetrainingArtifact.retrained_model_path`, `ModelRetrainingArtifact.metrics` |
| **Output artifact** | `ModelRegistrationArtifact` (`artifact_entity.py:154`) — `registered_version`, `is_deployed`, `is_better_than_current` |
| **Registry path** | `models/registry/` |
| **Config class** | `ModelRegistrationConfig` (`config_entity.py:245`) |
| **Config keys** | `auto_deploy=False`, `proxy_validation=True`, `degradation_tolerance=0.005`, `block_if_no_metrics=False` |
| **Comparison logic** | `is_better = float(new_acc) >= float(cur_acc) - degradation_tolerance` (`model_registration.py:74`); if `new_acc` or `cur_acc` is `None` and `block_if_no_metrics=False` → `is_better=True` (register but do not auto-deploy) |
| **Failure modes** | (a) No prior registered version → `is_better=True` by default (first registration always passes); (b) AdaBN/TENT `val_accuracy=None` → gate deferred to manual deploy gate by default |
| **Safeguards** | REGRESSION log at `model_registration.py:79`; `degradation_tolerance=0.005` threshold; `block_if_no_metrics` flag |
| **Tests** | `tests/test_model_rollback.py`, `tests/test_model_registration_gate.py` (7 tests) |

---

### Stage 10 — Baseline Update

| Field | Value |
|---|---|
| **Purpose** | Rebuild the drift-detection baseline statistics (mean/std per channel per class) from labeled training data after successful retraining. |
| **Module** | `src/components/baseline_update.py` |
| **Main class** | `BaselineUpdate` (line 1); method `initiate_baseline_update()` |
| **Underlying module** | `scripts/build_training_baseline.py → BaselineBuilder.build_from_csv()` |
| **Input** | Labeled CSV (default: `data/all_users_data_labeled.csv`) |
| **Output artifact** | `BaselineUpdateArtifact` |
| **Output files (conditional)** | If `promote_to_shared=True` (`--update-baseline` CLI): `models/training_baseline.json`, `models/normalized_baseline.json` + versioned archive in `models/baseline_versions/`; else: artifact dir only |
| **Config class** | `BaselineUpdateConfig` (`config_entity.py:268`) |
| **Config keys** | `promote_to_shared=False` (governance default), `rebuild_embeddings=False` |
| **Critical governance note** | `promote_to_shared=False` by default means Stage 6 Layer 3 drift comparison always uses the ORIGINAL baseline unless `--update-baseline` is passed explicitly. After retraining, the drift baseline becomes stale indefinitely (`baseline_update.py:103`) |
| **Failure modes** | (a) Labeled CSV not found → `FileNotFoundError`; (b) Baseline NOT promoted (default) → monitoring drifts from stale reference; (c) MLflow artifact logging silently skipped if no active run |
| **Safeguards** | Versioned archive on promote; MLflow artifact logging; `promote_to_shared` governance flag |
| **Tests** | `tests/test_baseline_update.py` |

---

### Stage 11 — Calibration & Uncertainty Quantification

| Field | Value |
|---|---|
| **Purpose** | Apply post-hoc temperature scaling and MC Dropout to calibrate model confidence and quantify epistemic/aleatoric uncertainty. |
| **Module** | `src/components/calibration_uncertainty.py` |
| **Main class** | `CalibrationUncertainty` (line 1); method `initiate_calibration()` |
| **Underlying module** | `src/calibration.py → TemperatureScaler`, `CalibrationEvaluator`, `MCDropoutEstimator`, `UnlabeledCalibrationAnalyzer` |
| **Input** | `ModelInferenceArtifact.probabilities_npy_path` (falls back to `predictions_csv`) |
| **Output files** | `outputs/calibration/temperature.json`, `outputs/calibration/reliability_diagram.png` |
| **Output artifact** | `CalibrationUncertaintyArtifact` |
| **Config class** | `CalibrationUncertaintyConfig` (`config_entity.py:284`) |
| **Config keys** | `initial_temperature=1.5`, `temp_lr=0.01`, `temp_max_iter=100`, `mc_forward_passes=30`, `mc_dropout_rate=0.2`, `confidence_warn_threshold=0.65`, `entropy_warn_threshold=1.5`, `ece_warn_threshold=0.10`, `n_bins=15` |
| **Handoff to Stage 6** | `temperature.json` is auto-loaded by Stage 6 on subsequent runs (`post_inference_monitoring.py:54-65`) |
| **Failure modes** | (a) `probabilities_npy_path` missing → approximate reconstruction from CSV (`confidence` column only) with loss of class-level calibration; (b) No labels → temperature scaling uses unsupervised proxy |
| **Safeguards** | Fallback from `.npy` to `.csv`; `overall_status=WARN` artifact returned on missing data |
| **Tests** | `tests/test_calibration.py` |

---

### Stage 12 — Wasserstein Drift Detection

| Field | Value |
|---|---|
| **Purpose** | Compute per-channel Earth Mover's Distance between production data and a labeled baseline, with change-point detection on the drift time series. |
| **Module** | `src/components/wasserstein_drift.py` |
| **Main class** | `WassersteinDrift` (line 1); method `initiate_wasserstein_drift()` |
| **Underlying module** | `src/wasserstein_drift.py → WassersteinDriftDetector`, `WassersteinChangePointDetector`, `compute_integrated_drift_report` |
| **Input** | `DataTransformationArtifact.production_X_path`, baseline file `data/prepared/baseline_X.npy` (default) |
| **Output files** | `outputs/wasserstein_drift/wasserstein_report.json` |
| **Output artifact** | `WassersteinDriftArtifact` |
| **Config class** | `WassersteinDriftConfig` (`config_entity.py:305`) |
| **Config keys** | `warn_threshold=0.3`, `critical_threshold=0.5`, `min_drifted_channels_warn=2`, `cpd_window_size=50`, `cpd_threshold=2.0`, `enable_multi_resolution=True` |
| **Failure modes** | (a) `baseline_X.npy` missing → returns `WassersteinDriftArtifact(overall_status="NO_BASELINE")` — no error blocking; (b) No auto-provisioning of `baseline_X.npy` — must be created manually or by Stage 10 |
| **Safeguards** | Graceful `NO_BASELINE` status; explicit error log if `prod_X_path` missing |
| **Tests** | `tests/test_wasserstein_drift.py` (9 tests) |

---

### Stage 13 — Curriculum Pseudo-Labeling

| Field | Value |
|---|---|
| **Purpose** | Iterative self-training: progressively assign pseudo-labels to high-confidence unlabeled production windows, with EWC regularisation to prevent catastrophic forgetting. |
| **Module** | `src/components/curriculum_pseudo_labeling.py` |
| **Main class** | `CurriculumPseudoLabeling` (line 1); method `initiate_curriculum_training()` |
| **Underlying module** | `src/curriculum_pseudo_labeling.py → CurriculumTrainer` |
| **Input** | `data/prepared/train_X.npy`, `data/prepared/train_y.npy` (labeled), `DataTransformationArtifact.production_X_path` (unlabeled) |
| **Output files** | `outputs/curriculum_training/` |
| **Output artifact** | `CurriculumPseudoLabelingArtifact` |
| **Config class** | `CurriculumPseudoLabelingConfig` (`config_entity.py:322`) |
| **Config keys** | `initial_confidence_threshold=0.95`, `final_confidence_threshold=0.80`, `n_iterations=5`, `threshold_decay="linear"`, `use_ewc=True`, `ewc_lambda=1000.0`, `ewc_n_samples=200`, `use_teacher_student=True`, `ema_decay=0.999` |
| **Failure modes** | (a) `train_X.npy` or `train_y.npy` missing → empty artifact returned (`curriculum_pseudo_labeling.py:52-56`); (b) `ewc_lambda=1000.0` is hardcoded — no ablation in codebase; (c) Teacher-student requires TF model at `models/pretrained/model.h5` (fallback from `.keras`) |
| **Safeguards** | Guard on labeled data existence before training; returns empty artifact on failure |
| **Tests** | `tests/test_curriculum_pseudo_labeling.py` |

---

### Stage 14 — Sensor Placement Robustness

| Field | Value |
|---|---|
| **Purpose** | Detect dominant hand, apply axis-mirror augmentation to simulate opposite-hand placement, and report per-hand classification performance. |
| **Module** | `src/components/sensor_placement.py` |
| **Main class** | `SensorPlacement` (line 1); method `initiate_sensor_placement()` |
| **Underlying module** | `src/sensor_placement.py → HandDetector`, `AxisMirrorAugmenter`, `HandPerformanceReporter` |
| **Input** | `DataTransformationArtifact.production_X_path` |
| **Output files** | `outputs/sensor_placement/` |
| **Output artifact** | `SensorPlacementArtifact` |
| **Config class** | `SensorPlacementConfig` (`config_entity.py:356`) |
| **Config keys** | `mirror_axes=[1,2,4,5]`, `mirror_probability=0.5`, `dominant_accel_threshold=1.2`, `accel_indices=[0,1,2]`, `gyro_indices=[3,4,5]` |
| **Failure modes** | (a) Production data not found → empty artifact returned (`sensor_placement.py:51-53`); (b) `mirror_axes` indices out of range for non-6-channel data |
| **Safeguards** | Guard on `prod_X_path.exists()` before load |
| **Tests** | `tests/test_sensor_placement.py` (9 tests) |

---

## Test Coverage Summary

| Test File | Stage(s) Covered |
|---|---|
| `tests/test_adabn.py` | Stage 8 (AdaBN) |
| `tests/test_baseline_update.py` | Stage 10 |
| `tests/test_calibration.py` | Stage 11 |
| `tests/test_curriculum_pseudo_labeling.py` | Stage 13 |
| `tests/test_data_validation.py` | Stage 2 |
| `tests/test_drift_detection.py` | Stage 6 Layer 3 |
| `tests/test_model_registration_gate.py` | Stage 9 (gate logic, 7 tests) |
| `tests/test_model_rollback.py` | Stage 9 (rollback) |
| `tests/test_pipeline_integration.py` | Multi-stage integration |
| `tests/test_preprocessing.py` | Stage 3 |
| `tests/test_prometheus_metrics.py` | Observability |
| `tests/test_retraining.py` | Stage 8 |
| `tests/test_robustness.py` | Stage 3 (augmentation) |
| `tests/test_sensor_placement.py` | Stage 14 |
| `tests/test_threshold_consistency.py` | Threshold governance (5 tests) |
| `tests/test_trigger_policy.py` | Stage 7 |
| `tests/test_wasserstein_drift.py` | Stage 12 |
| `tests/test_active_learning.py` | NOT directly mapped to stage |
| `tests/test_ood_detection.py` | NOT directly mapped to stage |
| `tests/test_progress_dashboard.py` | NOT directly mapped to stage |
| **No test found** | Stage 1, Stage 4, Stage 5, Stage 6 (component-level) |

**Total test files**: 20 (as of 2026-02-26)

---

## Config Cross-Reference

| Config Key | Class | File:Line | Default | Used by |
|---|---|---|---|---|
| `confidence_warn_threshold` | `PostInferenceMonitoringConfig` | `config_entity.py:157` | 0.60 | Stage 6 Layer 1, `src/api/app.py` |
| `confidence_warn` | `TriggerEvaluationConfig` | `config_entity.py:202` | 0.65 | Stage 7 |
| `drift_zscore_threshold` | `PostInferenceMonitoringConfig` | `config_entity.py:162` | 2.0 | Stage 6 Layer 3 |
| `drift_zscore_warn` | `TriggerEvaluationConfig` | `config_entity.py:206` | 2.0 | Stage 7 |
| `uncertain_window_threshold` | `PostInferenceMonitoringConfig` | `config_entity.py:159` | 0.50 | `src/api/app.py:201` |
| `degradation_tolerance` | `ModelRegistrationConfig` | `config_entity.py:253` | 0.005 | Stage 9 |
| `block_if_no_metrics` | `ModelRegistrationConfig` | `config_entity.py:256` | False | Stage 9 |
| `promote_to_shared` | `BaselineUpdateConfig` | `config_entity.py:277` | False | Stage 10 |
| `max_baseline_age_days` | `PostInferenceMonitoringConfig` | `config_entity.py:166` | 90 | Stage 6 staleness guard |
| `ewc_lambda` | `CurriculumPseudoLabelingConfig` | `config_entity.py:330` | 1000.0 | Stage 13 |
| `entropy_warn_threshold` | `CalibrationUncertaintyConfig` | `config_entity.py:294` | 1.5 | Stage 11 |
| `window_size` | `DataTransformationConfig` | `config_entity.py:108` | 200 | Stage 3, inference must match |

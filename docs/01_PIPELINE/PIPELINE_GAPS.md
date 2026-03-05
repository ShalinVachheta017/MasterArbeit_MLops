# Pipeline Gaps Report
**Extracted from codebase audit — 2026-02-26**
**Scope: verified inconsistencies and missing evidence only. No opinions.**

Hard rule: every statement cites file path + line number.
If code could not be verified, the entry is marked **NOT VERIFIED**.

---

## Gap Index

| ID | Stage(s) | Severity | Title |
|---|---|---|---|
| G-01 | Stage 2 | HIGH | Invalid data flows downstream when `--continue-on-failure` active |
| G-02 | All | HIGH | No runtime YAML/env config override — thresholds baked into Python dataclasses |
| G-03 | Stage 10 | HIGH | `promote_to_shared=False` default — drift baseline is never updated automatically after retraining |
| G-04 | Stage 6 | MEDIUM | Stale baseline emits WARNING log only — no Prometheus gauge, no retrain trigger |
| G-05 | Stage 4 | MEDIUM | No model integrity check (hash/checksum) on pretrained model load |
| G-06 | Stage 11 vs. Stage 7 | MEDIUM | `CalibrationUncertaintyConfig.entropy_warn_threshold=1.5` vs. `TriggerThresholds.entropy_warn` (not compared in threshold tests) |
| G-07 | Stage 13 | MEDIUM | Expensive EWC training starts before verifying labeled data existence; early return only after `np.load` attempt |
| G-08 | Stage 13 | MEDIUM | `ewc_lambda=1000.0` hardcoded without ablation study |
| G-09 | Stage 6 | MEDIUM | Layer 3 drift silently downgraded to `NO_BASELINE` WARNING — not escalated to CRITICAL |
| G-10 | Stage 6 | MEDIUM | `is_training_session=True` can silently suppress ALL Layer 3 drift without audit log entry at CRITICAL level |
| G-11 | Stage 12 | MEDIUM | `baseline_X.npy` has no provisioning script — Stage 12 silently returns `NO_BASELINE` until manually created |
| G-12 | Stage 5 | LOW | No labeled ground truth in production → `has_labels=False` always; ECE and F1 metrics are never computed for live data |
| G-13 | Stage 1 | LOW | No `test_data_ingestion.py` — ingestion error paths (malformed Excel, timestamp gaps) untested |
| G-14 | Stage 4 | LOW | No `test_model_inference.py` — model loading and shape mismatch error paths untested |
| G-15 | All | LOW | `test_pipeline_integration.py` exists but full 14-stage execution not verified |
| G-16 | Config | LOW | `config/monitoring_thresholds.yaml` is NOT loaded by any pipeline code — dead reference file |
| G-17 | Observability | LOW | `_prom_latency_ms` Histogram defined but no alert rule in `config/alerts/har_alerts.yml` |

---

## Detailed Gap Descriptions

---

### G-01 — Invalid data flows downstream with `--continue-on-failure`

**File**: `src/pipeline/production_pipeline.py`
**Lines**: 271–275
**Observed code**:
```python
if not validation_art.is_valid:
    logger.warning("Validation FAILED — errors: %s", validation_art.errors)
    if not continue_on_failure:          # <── guard only fires when flag=False
        result.stages_failed.append("validation")
        break
```
**Gap**: When the user passes `--continue-on-failure` (`run_pipeline.py:~200`, `continue_on_failure=True`), the `if not continue_on_failure` guard is skipped. Stage 3 (`DataTransformation`) then processes data that failed schema/range validation. No warning is promoted to CRITICAL. The word `break` is used, not `raise`, so the loop simply continues to the next stage.

**Evidence of reachable path**: `run_pipeline.py` exposes `--continue-on-failure` flag (`run_pipeline.py:116`):
```python
parser.add_argument("--continue-on-failure", action="store_true", ...)
```
This flag is passed through to `ProductionPipeline.run(continue_on_failure=True)`.

**Missing safeguard**: No `DataValidationError` exception class exists; bad data can contaminate windows at Stage 3.

---

### G-02 — No runtime YAML/env config override

**Files**: `src/entity/config_entity.py` (entire file, lines 1–404)
**Gap**: All 14 stage configs are Python dataclasses with hardcoded defaults. There is no YAML, TOML, `.env`, or environment-variable loader that overrides them at runtime. The file `config/monitoring_thresholds.yaml` (created in previous session) documents thresholds but **is never imported or loaded by any pipeline code** (verified by `grep_search` returning zero matches for `monitoring_thresholds.yaml` in `.py` files).

**Impact**: An operator who wants to change `confidence_warn_threshold` from `0.60` to `0.55` must modify `config_entity.py:157` and re-deploy the service.

**Affected thresholds**: 17 threshold fields across 6 config classes (verified in `config_entity.py:157–404`).

---

### G-03 — Drift baseline never auto-updated after retraining

**File**: `src/components/baseline_update.py:103`
**Line**: `src/entity/config_entity.py:277`
**Observed code**:
```python
# config_entity.py:277
promote_to_shared: bool = False
```
```python
# baseline_update.py:103
} else:
    # Governance: write ONLY to the artifact dir — NEVER touch models/
    builder.save(str(save_baseline_path))   # saves to artifact dir only
```
**Gap**: After a full retrain cycle (Stages 8→9→10), `Stage 10` saves the new baseline ONLY to the timestamped `artifacts/<run_id>/models/` directory. The live monitoring path `models/normalized_baseline.json` (read by Stage 6, line `post_inference_monitoring.py:82`) is **never updated** unless `--update-baseline` is passed explicitly. This means after retraining, Stage 6 Layer 3 continues comparing production data to the original pre-retraining baseline indefinitely.

**No automated governance**: There is no code that checks whether the live baseline is older than the most recently registered model version.

---

### G-04 — Stale baseline: WARNING log only, no automated action

**File**: `src/components/post_inference_monitoring.py`
**Lines**: 90–100
**Observed code**:
```python
age_days = (_time.time() - Path(baseline_json).stat().st_mtime) / 86400
if age_days > self.config.max_baseline_age_days:
    logger.warning(
        "Baseline is %.0f days old (configured limit: %d days) — "
        "drift scores may not reflect current sensor characteristics. "
        "Consider running 'baseline_update' stage.",
        ...
    )
```
**Gap**: The staleness check fires a `logger.warning` only. It does NOT:
- Set a Prometheus gauge `har_baseline_age_days` (no such metric in `src/api/app.py:40–68`)
- Write to `monitoring_report.overall_status = "WARNING"` for staleness
- Trigger Stage 7 with an extra signal
- Appear in `config/alerts/har_alerts.yml` (file has 3 alert rules, none for baseline age)

---

### G-05 — No model integrity check at Stage 4

**File**: `src/components/model_inference.py`
**Lines**: 50–60
**Observed code**:
```python
model_path = self.config.model_path or (
    self.pipeline_config.models_pretrained_dir / "fine_tuned_model_1dcnnbilstm.keras"
)
inf_cfg = _InferenceConfig(model_path=Path(model_path), ...)
pipe = _InferencePipeline(config=inf_cfg)
result = pipe.run()
```
**Gap**: No SHA-256 or MD5 checksum is compared before loading the `.keras` model. A replaced or corrupted model file at `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` would be loaded silently without detection. There is no `model_hash` field in `ModelInferenceConfig` (`config_entity.py:118–126`) and no hash manifest file in `models/pretrained/`.

---

### G-06 — Entropy threshold inconsistency between Stage 11 and trigger policy

**File 1**: `src/entity/config_entity.py:294` — `CalibrationUncertaintyConfig.entropy_warn_threshold = 1.5`
**File 2**: `src/trigger_policy.py` — `TriggerThresholds.entropy_warn` default (NOT VERIFIED — line not read; was 1.8 in previous session's `config/alerts/har_alerts.yml` (now fixed to 1.8))
**File 3**: `tests/test_threshold_consistency.py` — covers `drift_zscore` and `confidence_warn` gaps, but does NOT compare `entropy_warn_threshold` across `CalibrationUncertaintyConfig` vs. `TriggerThresholds`

**Gap**: Stage 11's entropy warn threshold (1.5) and the trigger policy's entropy threshold (1.8, from `har_alerts.yml` post-fix) are different values. `test_threshold_consistency.py` has 5 tests but does not include an `entropy_warn` consistency assertion.

---

### G-07 — Stage 13 labeled-data guard fires after `np.load`, not before

**File**: `src/components/curriculum_pseudo_labeling.py`
**Lines**: 52–60
**Observed code**:
```python
if not Path(source_path).exists() or not Path(labels_path).exists():
    logger.error("Source labeled data not found: %s", source_path)
    return CurriculumPseudoLabelingArtifact()
```
**Observed code** (model load, lines ~80+):
```python
model_path = self.pipeline_config.models_pretrained_dir / "model.h5"
if not model_path.exists():
    ...
```
**Gap**: The guard on labeled data at line 52 is correct. However, the check for `unlabeled_path` at line 63-66 calls `np.load(unlabeled_path)` AFTER checking existence — this is correct. BUT the model load at line ~82 (`model.h5`) does NOT have an early existence check — `TF.keras.models.load_model()` will throw an exception instead of returning an empty artifact. The inconsistency is that some failure paths return `CurriculumPseudoLabelingArtifact()` (empty) while others raise exceptions, making error handling non-uniform.

---

### G-08 — `ewc_lambda=1000.0` without ablation

**File**: `src/entity/config_entity.py:330`
**Observed**: `ewc_lambda: float = 1000.0`
**Gap**: This is a critical hyperparameter controlling forgetting prevention vs. plasticity in EWC (Elastic Weight Consolidation). No ablation study code (e.g., a grid-search script or MLflow experiment comparing λ ∈ {10, 100, 1000, 10000}) was found in `scripts/`, `notebooks/`, or `tests/`. The value 1000.0 is cited in literature (Kirkpatrick et al.) for some tasks but has not been verified for this HAR dataset.

---

### G-09 — Layer 3 drift skipped on missing baseline without CRITICAL escalation

**File**: `src/components/post_inference_monitoring.py`
**Lines**: 83–90
**Observed code**:
```python
baseline_json = self.config.baseline_stats_json or (
    self.pipeline_config.models_dir / "normalized_baseline.json"
)
if not Path(baseline_json).exists():
    logger.warning(
        "Baseline file not found: %s — drift analysis will skip baseline comparison",
        baseline_json,
    )
    baseline_json = None
```
**Gap**: A missing baseline is a `logger.warning`. The overall `monitoring_report.overall_status` is not elevated to CRITICAL when the baseline is absent — Layer 3 is simply excluded from the report. An examiner reviewing `monitoring_report.json` would see no drift results and might conclude the system is healthy, when in fact drift detection is disabled.

---

### G-10 — `is_training_session=True` silently disables Layer 3

**File**: `src/components/post_inference_monitoring.py`
**Lines**: 103–109 (approximate from `post_inference_monitoring.py:~103`)
**Observed code**:
```python
if self.config.is_training_session and baseline_json is not None:
    logger.warning(
        "is_training_session=True: skipping Layer 3 drift comparison ..."
    )
    baseline_json = None
```
**Gap**: `is_training_session` is a `bool = False` field in `PostInferenceMonitoringConfig` (`config_entity.py:174`). If accidentally set to `True` in production (e.g., via a mis-configured environment or CLI override), ALL Layer 3 drift detection is silently disabled with a `logger.warning`. There is no test asserting that `is_training_session=True` in a production context raises an alert or is rejected.

---

### G-11 — `baseline_X.npy` for Stage 12 has no provisioning script

**File**: `src/components/wasserstein_drift.py:61`
**Observed code**:
```python
if baseline_path is None:
    baseline_path = self.pipeline_config.data_prepared_dir / "baseline_X.npy"
if not Path(baseline_path).exists():
    logger.warning("No baseline data at %s — skipping Wasserstein drift detection.", ...)
    return WassersteinDriftArtifact(overall_status="NO_BASELINE")
```
**Gap**: `data/prepared/baseline_X.npy` is the windowed numpy array of the training distribution. No script in `scripts/` automatically creates this file. `scripts/build_training_baseline.py` builds `training_baseline.json` (statistical baseline for Stage 6), NOT `baseline_X.npy`. Stage 12 will always return `NO_BASELINE` until the file is manually created.

---

### G-12 — Stage 5 never computes labeled metrics in production

**File**: `src/components/model_evaluation.py:53`; `src/entity/config_entity.py:130`
**Observed**: `ModelEvaluationConfig.labels_path = None` (default)
**Gap**: In production mode, ground-truth labels are never available in real-time. `ModelEvaluationArtifact.has_labels` will always be `False`, and `classification_metrics` will always be `None` (`artifact_entity.py:93`). ECE (which requires labels) cannot be computed. The evaluation stage effectively only produces confidence/distribution statistics — not model accuracy metrics. Thesis claims about "evaluation stage" need to clarify this.

---

### G-13 — No test for Stage 1 (Data Ingestion)

**File**: `tests/` directory listing
**Gap**: No `test_data_ingestion.py` was found. The ingestion component has complex logic: 3-path file discovery (`data_ingestion.py:80+`), `pd.merge_asof` with millisecond tolerance, column rename dictionaries (`ACCEL_RENAME/GYRO_RENAME` at lines 54-68), and manifest-based deduplication. Failure paths (missing files, timestamp gaps, corrupt Excel) have no automated test coverage.

---

### G-14 — No test for Stage 4 (Model Inference)

**File**: `tests/` directory listing
**Gap**: No `test_model_inference.py` was found. The inference component wraps `src/run_inference.py → InferencePipeline`. Failure paths (model file missing, input shape mismatch, batch_size edge cases) are untested.

---

### G-15 — No verified full 14-stage integration test

**File**: `tests/test_pipeline_integration.py` (present, but content NOT VERIFIED)
**Gap**: `test_pipeline_integration.py` exists. However, based on the stage definitions and the fact that Stages 11–14 (`ADVANCED_STAGES`) require explicit `--advanced` flag, it is NOT VERIFIED whether the integration test exercises all 14 stages. The integration test may only cover stages 1–7.

---

### G-16 — `config/monitoring_thresholds.yaml` never loaded by code

**File**: `config/monitoring_thresholds.yaml` (created 2026-02-26)
**Evidence of non-loading**: `grep_search` for `"monitoring_thresholds.yaml"` in `*.py` returned zero results.
**Gap**: The canonical threshold reference file is a documentation artifact only. Any threshold value change in this file has NO effect on runtime behaviour. There is no loader function in `src/entity/config_entity.py` or `run_pipeline.py` that reads it.

---

### G-17 — `har_inference_latency_ms` metric has no alert rule

**File**: `config/alerts/har_alerts.yml` (3 alert rules for: `HARLowConfidence`, `HARHighEntropy`, `HARHighFlipRate`)
**File**: `src/api/app.py:66–69` — `_prom_latency_ms = Histogram("har_inference_latency_ms", ...)`
**Gap**: The latency histogram is defined and exposed at `/metrics` but has no corresponding alert rule in `har_alerts.yml`. If inference latency exceeds an SLA (e.g., p95 > 500ms), no alert fires. No SLA value is defined anywhere in the codebase for inference latency.

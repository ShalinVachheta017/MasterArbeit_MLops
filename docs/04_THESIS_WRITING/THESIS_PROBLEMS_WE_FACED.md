# Thesis Engineering Log: Problems We Faced

**Thesis Title:** Developing a MLOps Pipeline for Continuous Mental Health Monitoring Using Wearable Sensor Data  
**Author:** Shalin Vachheta  
**Date:** February 2026  
**Purpose:** A structured record of the engineering problems encountered during development of the HAR MLOps pipeline. Each entry documents the symptom, root cause, fix, and lesson learned. This file is intended as supplementary material for the thesis discussion chapter and as a practical reference for future maintenance.

**How to use in thesis:**  
> Reference individual entries in Chapter 5 (Implementation Challenges) or Appendix B (Engineering Log). Each entry contains a commit hash that links the textual claim to concrete code evidence in the repository.

---

## Top Recurring Themes

1. **Training-production distribution mismatch** — the single most common class of bug; caused by unit differences, scaling mismatches, or preprocessing toggles that differed between training and inference.
2. **Test-time adaptation safety** — TENT and AdaBN required multiple rounds of hardening to prevent each from silently corrupting the other's internal state.
3. **Baseline governance** — shared artefact paths (`models/training_baseline.json`) were written by non-promoting runs, risking overwrite of the production reference baseline.
4. **Configuration key drift** — renaming config keys in one file without updating all consumers caused silent crashes downstream.
5. **Docker and CI/CD environment fragility** — image name casing, Python import path shadowing, and wrong health-check endpoints together caused three separate CI failures on the same day.
6. **Incorrect class labels propagated through the codebase** — the wrong dataset's label list was copied into multiple files and tests without being caught for several weeks.
7. **Model architecture mismatch** — the retraining script silently built a different network topology than the pretrained model, meaning retrained models could not reliably be compared with or replace the original.
8. **MLflow API versioning** — positional function arguments that worked in the development environment broke in the CI environment due to a different MLflow minor version.
9. **Pseudo-label pipeline data leakage** — source data was not scaled with the production scaler before being fed into pseudo-label retraining, introducing an artificial domain shift.
10. **CI test suite execution time** — TensorFlow-dependent tests running inside the standard unit-test job caused the CI pipeline to time out.

---

## Timeline of Major Problems

| Date | Problem Title | Stage / Module | Symptom | Root Cause | Fix Summary | Evidence |
|---|---|---|---|---|---|---|
| 2025-11-05 | Accelerometer unit mismatch | Preprocessing (Stage 2) | Cross-user accuracy 14.5% — near-random guessing | Production data in milliG; model trained on m/s² | Auto-detect unit from magnitude; multiply by 9.80665/1000 | `c17f3dd`, `docs/DATASET_DIFFERENCE_SUMMARY.md` |
| 2025-12-06 | Unit conversion solution confirmed | Preprocessing (Stage 2) | Converted CSV produced correct magnitude | Median magnitude detection logic verified | `data/processed/sensor_fused_50Hz_converted.csv` created; conversion ratio documented | `1c403f8`, `docs/UNIT_CONVERSION_SOLUTION.md` |
| 2026-02-19 | Missing `build_training_baseline.py` | Monitoring (Stage 9) | CLI could not build reference baseline; AdaBN/TENT refused to run | Script never created, referenced but absent | Created `scripts/build_training_baseline.py`; exposed `tent`/`adabn_tent` in CLI | `700381a` |
| 2026-02-19 | PSI threshold wrong (textbook vs pipeline reality) | Drift Detection (Stage 10) | Test assertions fail on valid data; trigger fires too early | Standard PSI thresholds (0.10 / 0.25) apply to single-channel distributions; pipeline aggregates 24 channels | Updated thresholds to 0.75 / 1.50 based on observed noise floor | `bd8dc1e`, `tests/test_trigger_policy.py` |
| 2026-02-19 | TENT corrupts AdaBN running statistics | Retraining (Stage 12) | After AdaBN + TENT, confidence dropped instead of recovering | TENT's training-mode forward passes overwrote the BN running statistics just set by AdaBN | Snapshot BN running stats before each `apply_gradients` step; restore immediately after | `f47a48b`, `src/domain_adaptation/tent.py` |
| 2026-02-19 | TENT no rollback gate | Retraining (Stage 12) | TENT silently made the model worse in some batches | No guard existed; adaptation applied unconditionally | Added rollback: restore gamma/beta if entropy increases beyond threshold | `f47a48b`, `e2bc784` |
| 2026-02-19 | Baseline governance — shared path overwritten by non-promoting runs | Retraining / Monitoring | Production baseline overwritten during test or non-canary runs | `baseline_update.py` wrote to `models/` unconditionally | Gate all shared-path writes behind `promote_to_shared` flag (default `False`) | `f47a48b`, `e2bc784`, `src/components/baseline_update.py` |
| 2026-02-19 | Pseudo-label source data not scaled with production scaler | Retraining (Stage 12) | Pseudo-label model trained on mismatched scale; validation accuracy inflated | `source_X` passed raw (not via production `config.json` scaler) | Scale `source_X` with `config.json` before pseudo-label training; add self-consistency filter | `3fd3c00`, `src/train.py` |
| 2026-02-19 | Keras `deepcopy` breaks model copy in pseudo-label loop | Retraining (Stage 12) | `model.predict()` on copied model raises runtime error | `copy.deepcopy()` does not clone Keras model weights correctly | Replace `deepcopy` with `model.save()` + `keras.models.load_model()` | `3fd3c00`, `src/train.py` |
| 2026-02-19 | `tf.function` retracing warnings from TENT | Retraining (Stage 12) | Excessive retracing warnings flooding logs; TENT runs 10× slower | `model.predict()` inside a gradient loop triggers graph retracing on each call | Replace `model.predict()` with `model(tf.constant(batch), training=False)` | `e2bc784`, `src/domain_adaptation/tent.py` |
| 2026-02-19 | MLflow `log_model` positional argument breaks on older versions | MLOps Tooling (Stage 6) | CI MLflow logging fails; model artefact not saved | `mlflow.keras.log_model(model, "model")` — positional arg not accepted in all versions | Change to `name="model"` keyword argument; add `try/except` fallback | `e2bc784`, `src/mlflow_tracking.py` |
| 2026-02-19 | CI test suite timeout — TF tests in unit job | CI/CD | Unit test CI job exceeded time limit | TensorFlow integration tests were not separated from fast unit tests | Split CI into `unit` (fast, no TF) and `slow` (TF-dependent, separate job); mark slow tests with `@pytest.mark.slow` | `1ae27cc`, `.github/workflows/ci-cd.yml` |
| 2026-02-15 | Docker image name uppercase breaks GHCR | CI/CD / Deployment | `docker push` step fails; CI cannot publish the image | GitHub Container Registry (GHCR) requires all-lowercase image names; workflow used `MasterArbeit_MLops` | Changed workflow env var to `shalinvachheta017/masterarbeit_mlops/har-inference` (all lowercase) | `8b4dab7`, `.github/workflows/ci-cd.yml` |
| 2026-02-22 | Docker container loads wrong `app` module (import shadowing) | Deployment (Stage 8) | Container starts but serves incorrect endpoints; `ImportError` in some builds | `docker/api/` folder inside the container image shadowed `src/api/app.py` on the Python import path | Copy `docker/api` to `/app/docker_api`; set `PYTHONPATH=/app:/app/src`; run `src.api.app:app` explicitly | `7f892d8`, `docker/Dockerfile.inference` |
| 2026-02-22 | CI health check hits wrong endpoint | CI/CD | Integration test always fails; CI never green | Health check URL was `/health`; actual FastAPI endpoint is `/api/health` | Corrected URL; replaced `sleep 10` with a poll loop that retries every second up to 30 s | `edbc399`, `.github/workflows/ci-cd.yml` |
| 2026-02-22 | Wrong class labels across codebase (PAMAP2 instead of anxiety classes) | Training / Active Learning | Active learning export and test fixtures contained PAMAP2 activity names | PAMAP2 label list copied from an example at an early stage; never corrected | Replaced with correct 11 anxiety-behaviour classes in `active_learning_export.py` and `tests/conftest.py` | `cf9c036`, `c98b45c` |
| 2026-02-22 | `train.py` built wrong architecture (~850 K params vs 499 K pretrained) | Training (Stage 7) | Retrained model had different capacity; layer-by-layer comparison with pretrained fails | `create_1dcnn_bilstm()` in training code was a newer experimental version; pretrained model was an earlier topology | Add `_build_v1()` / `_build_v2()` dispatch; `TrainingConfig.model_version = "v1"` defaults to pretrained topology | `5c38f65`, `src/train.py` |
| 2026-02-22 | Stage 11 crash — missing attribute on artifact entity | Trigger Policy (Stage 11) | `AttributeError` in `calibration_uncertainty.py` causes trigger evaluation to crash | Artifact entity class missing a required field; `model_registration.py` did not guard against missing TTA output | Add missing field to `artifact_entity.py`; add guard + documentation in `model_registration.py` | `e336770`, `src/components/model_registration.py` |
| 2026-02-22 | `black` formatting check fails immediately after architecture fix | CI/CD | Lint job (black) fails on `train.py` within one commit of the architecture change | Architecture changes introduced indentation style not compliant with `black` | Reformatted `train.py` to pass black | `13d1c35`, `src/train.py` |
| 2026-02-22 | Config key rename not propagated to all consumers (`psi_warn` → `drift_zscore_warn`) | Trigger Policy (Stage 11) | Stage 11 crash at runtime; key `drift_psi_warn` not found | `trigger_policy.py` and `config_entity.py` renamed the key; `trigger_evaluation.py` still read old name | Updated `trigger_evaluation.py` to read `drift_zscore_warn` | `c4b4994`, `b92ae0a`, `src/components/trigger_evaluation.py` |

---

## Problems by Category

---

### A) Data Ingestion and Schema Issues

---

#### Accelerometer Unit Mismatch: milliG vs m/s²  (2025-11-05)

**Area:** Data Ingestion / Preprocessing | **Stage:** Stage 2 — Preprocessing

**Symptom:** Cross-user evaluation reported 14.52% accuracy across six users (6–19% range). Random guessing on 11 classes produces ~9.09%. The model was confidently wrong — User 2 showed 92.36% confidence alongside 8.49% accuracy.

**Impact:** Completely blocked production deployment. No reliable predictions could be made from any user's data.

**Root Cause:** The pretrained model was trained on accelerometer data expressed in m/s². Production data exported from Garmin Connect was in milliG (1 milliG ≈ 0.0098 m/s²). After the training StandardScaler was applied, production values appeared approximately 100× smaller than expected. The StandardScaler then amplified noise rather than signal. The gyroscope channels were consistent across both datasets; the accelerometer channels were not.

Evidence from `docs/DATASET_DIFFERENCE_SUMMARY.md` (archived, retrievable from commit `c17f3dd`):
```
Training  Ax/Ay/Az means ≈ [3.2,  1.3, -3.5], std ≈ [6.6,  4.4, 3.2]
Production Ax/Ay/Az means ≈ [-16.2, -19.0, -1001.6], std ≈ [11.3, 31.0, 19.9]
```

**Fix Implemented:**  
`src/preprocess_data.py` and `src/sensor_data_pipeline.py` — added automatic unit detection:  
- Compute `median(sqrt(Ax²+Ay²+Az²))` on the incoming batch.  
- If the result exceeds 100, classify as milliG and multiply by `9.80665 / 1000`.  
- Log the detected unit and the conversion applied.  
Config key: `config/pipeline_config.yaml :: preprocessing.enable_unit_conversion: true`

**Verification:** Produced `data/processed/sensor_fused_50Hz_converted.csv`. Confirmed the converted accelerometer magnitude was ~9.81 m/s² at rest (1g), matching training data statistics. Documented in `docs/UNIT_CONVERSION_SOLUTION.md`.

**Evidence:** Commits `c17f3dd` (problem identified), `3a07e4f` (distribution analysis), `1c403f8` (conversion solution confirmed). Archived docs: `docs/CRITICAL_MODEL_ISSUE.md`, `docs/DATASET_DIFFERENCE_SUMMARY.md`.

**Lesson Learned (thesis-ready):** Unit consistency between training and production is the most consequential and easiest-to-miss preprocessing requirement. A model operating on data that is scaled 100× incorrectly will exhibit high-confidence, near-random predictions — a failure mode that is difficult to diagnose without explicit unit verification at ingestion. Automated unit detection based on physical plausibility (gravity ~9.81 m/s²) is a practical and lightweight safeguard.

**Future Prevention:**  
- Hard assertion at Stage 0: after unit conversion, `assert 8.0 < median_accel_magnitude < 12.0` (plausible gravity range).  
- Alert if gyroscope and accelerometer units appear inconsistent with each other.

---

### B) Preprocessing and Sensor Fusion Issues

---

#### Wrong PSI Thresholds for Multi-Channel Aggregated Pipeline  (2026-02-19)

**Area:** Preprocessing / Drift Detection | **Stage:** Stage 10 — Drift Detection

**Symptom:** Unit tests for the trigger policy were asserting `psi_warn = 0.10` and `psi_critical = 0.25`. These tests were passing, but the trigger was firing on stable data in integration runs — indicating the thresholds were far too sensitive for the actual pipeline output.

**Impact:** False-positive drift alarms in every integration test; unpredictable trigger behaviour when running end-to-end.

**Root Cause:** Standard textbook PSI interpretation (PSI < 0.10 = no drift, PSI 0.10–0.25 = moderate) applies to a single continuous distribution, binned into a histogram. This pipeline computes PSI per channel and then aggregates across 24 channel-resolution buckets. The aggregated score is structurally larger than a single-channel PSI and cannot be compared against textbook thresholds.

**Fix Implemented:**  
`tests/test_trigger_policy.py` — updated expected defaults to `psi_warn = 0.75`, `psi_critical = 1.50`.  
`src/trigger_policy.py :: TriggerThresholds` — updated default values.  
Comment added explaining the distinction between single-distribution and multi-channel PSI.

**Verification:** Tests pass. Integration runs on known-stable data no longer trigger false alarms.

**Evidence:** Commit `bd8dc1e` — `tests/test_trigger_policy.py` diff shows threshold change with explanatory docstring.

**Lesson Learned (thesis-ready):** Statistical thresholds derived from academic literature must be re-calibrated for the specific aggregation scheme used in production. Textbook PSI thresholds were designed for single-feature analysis; a multi-channel aggregated pipeline requires empirical calibration against a stable reference period before the thresholds become meaningful.

**Future Prevention:** Add a calibration script that computes the 95th-percentile PSI on a known-stable validation window and proposes a warning threshold automatically.

---

### C) Training and Evaluation Issues

---

#### Train Script Builds Wrong Model Architecture  (2026-02-22)

**Area:** Training | **Stage:** Stage 7 — Training and Evaluation

**Symptom:** `train.py` built a 1D-CNN-BiLSTM model with approximately 850 K parameters. The pretrained model in `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` has 499,131 parameters arranged across 17 layers. Any model retrained by the pipeline could not be directly compared with the original, and would have a different capacity.

**Impact:** Silent architecture mismatch. Comparisons between the pretrained and retrained model metrics would be meaningless because the models had different capacities. Canary evaluation and rollback metrics would be unreliable.

**Root Cause:** The training script had been updated to an experimental v2 architecture during development, but the pretrained model was a v1. The discrepancy was not caught until a layer-by-layer parameter count comparison was performed.

**Fix Implemented:**  
`src/train.py` — `create_1dcnn_bilstm()` now dispatches to `_build_v1()` (default; matches pretrained, 499 K params, 17 layers) or `_build_v2()` (experimental, kept for ablation).  
`TrainingConfig.model_version = "v1"` is the default.  
`TrainingConfig.dropout_cnn` split into `dropout_cnn_1 = 0.1` and `dropout_cnn_2 = 0.2` to match pretrained topology exactly.

**Verification:** Layer-by-layer comparison confirmed: 17 layers, 499,131 parameters, matching the pretrained model exactly. All 218 tests pass.

**Evidence:** Commit `5c38f65` — `src/train.py` diff, `docs/22Feb_Opus_Understanding/11_STAGE_TRAINING_EVALUATION_INFERENCE.md` updated with correct architecture.

**Lesson Learned (thesis-ready):** Model architecture specifications should be version-controlled explicitly (e.g., as a frozen config or a test that counts parameters), not inferred from code. A mismatch between the training script and the deployed model silently invalidates comparative evaluation results.

**Future Prevention:**  
- Add a pytest assertion: `assert model.count_params() == EXPECTED_PARAMS` in `tests/test_training.py`.  
- Store the expected parameter count in `src/config.py` as a named constant.

---

#### Wrong Class Label List Propagated Through Codebase  (2026-02-22)

**Area:** Training / Active Learning | **Stage:** Stage 7 / Stage 12

**Symptom:** `src/active_learning_export.py` exported unlabelled samples using a PAMAP2 dataset label list (e.g., `walking`, `cycling`, `rope_jumping`). The model recognises 11 anxiety-related behaviours (`ear_rubbing`, `forehead_rubbing`, `hair_pulling`, etc.). Active learning export files contained entirely wrong class names. The test fixture (`tests/conftest.py`) also contained the PAMAP2 list.

**Impact:** Active learning exports were unusable. Any test that compared predicted class names against the fixture label list would produce false passes or misleading failures.

**Root Cause:** The PAMAP2 label list was copied from a reference implementation or tutorial at an early stage of development and was never updated when the dataset was changed to the 11 anxiety-behaviour classes from Oleh and Obermaisser (2025).

**Fix Implemented:**  
`src/active_learning_export.py` — replaced PAMAP2 class list with the 11 anxiety-behaviour classes.  
`tests/conftest.py` — fixture updated to use the same 11-class list.  
`README.md`, `docs/01_REPO_SNAPSHOT_AND_SCOPE.md` — corrected model parameter count (~499 K, not ~850 K or ~1.5 M).

**Verification:** All 215 tests pass after the change.

**Evidence:** Commit `cf9c036` — `src/active_learning_export.py` and `tests/conftest.py` diffs.

**Lesson Learned (thesis-ready):** Dataset-specific constants (class names, class counts, label mappings) should be defined in a single authoritative location (e.g., `src/config.py`) and imported everywhere else. Copying these values into multiple files creates a maintenance hazard where partial updates introduce subtle inconsistencies.

**Future Prevention:**  
- Define `ACTIVITY_CLASSES: Dict[int, str]` in `src/config.py`; import it in all consumers.  
- Add a test that asserts `len(ACTIVITY_CLASSES) == 11` and that all expected class names are present.

---

#### Pseudo-Label Training Uses Source Data at Wrong Scale  (2026-02-19)

**Area:** Training / Retraining | **Stage:** Stage 12 — Safe Retraining

**Symptom:** Pseudo-label retraining reported high validation accuracy (~96.9%) in isolation, but post-deployment confidence dropped. Source data and production data were effectively on different scales inside the retraining loop.

**Impact:** The pseudo-label model learned on a mixture of correctly and incorrectly scaled inputs, inflating in-loop validation accuracy while not generalising to production.

**Root Cause:** Inside `train.py :: _retrain_pseudo_labeling()`, `source_X` (the original labelled data) was passed directly without being scaled via the production `data/prepared/config.json` scaler. Production pseudo-labels were created from data that was already scaled. The two sets were then mixed in the same training batch.

Additional issue: `copy.deepcopy(model)` does not correctly clone Keras model weights in TensorFlow 2.x, causing `model.predict()` on the copy to use wrong weights.

**Fix Implemented:**  
`src/train.py` — scale `source_X` with `config.json` before mixing with production data.  
Self-consistency filter added: keep only source windows where the pre-trained model agrees with the true label (14.5% of source windows retained; boundary-crossing windows discarded).  
`copy.deepcopy(model)` replaced with `model.save(tmp_path)` + `keras.models.load_model(tmp_path)`.  
`src/components/model_retraining.py` — share `data_loader` (including `LabelEncoder`) between trainer and retrainer to avoid label encoding mismatch.

**Verification:** Audit result `A5 pseudo_label: 20260219_134233 (9 stages, val_acc = 0.969)` — 12/12 audit checks pass (`scripts/audit_artifacts.py`).

**Evidence:** Commit `3fd3c00` — `src/train.py` and `src/components/model_retraining.py` diffs.

**Lesson Learned (thesis-ready):** In a semi-supervised retraining loop, all data paths — labelled and pseudo-labelled — must pass through the same preprocessing and scaling pipeline. Mixing data from different scaling regimes creates an invisible and difficult-to-diagnose domain shift within the training minibatch.

**Future Prevention:**  
- Add an assertion in the retraining loop: `assert abs(source_X.mean()) < 1.0` (confirms scaled data has near-zero mean).  
- Unit test that runs a mock pseudo-label loop and verifies source and production data have the same per-channel statistics.

---

### D) Inference and Serving Issues

---

#### Docker Container Loads Wrong FastAPI Module  (2026-02-22)

**Area:** Deployment | **Stage:** Stage 8 — Deployment and Inference

**Symptom:** Docker container started successfully but served the wrong endpoints. Some builds raised `ImportError` for modules that were clearly present on disk.

**Impact:** Production container could not serve correct predictions. CI integration smoke test failed.

**Root Cause:** The `docker/api/` subdirectory inside the container image was being resolved before `src/api/app.py`. Python's import path placed `/app/docker` (where `docker/` was copied) ahead of `/app/src`. When the entrypoint ran `api.app:app`, it loaded `docker/api/app.py` instead of `src/api/app.py`.

**Fix Implemented:**  
`docker/Dockerfile.inference` — copy `docker/api` to `/app/docker_api` (renamed directory, no longer on import path); set `PYTHONPATH=/app:/app/src`; run entrypoint as `src.api.app:app` (fully qualified module path).

**Verification:** Container smoke test (`scripts/inference_smoke.py`) passes; CI integration test reaches the correct endpoint.

**Evidence:** Commit `7f892d8` — `docker/Dockerfile.inference` diff (8 lines changed).

**Lesson Learned (thesis-ready):** Python import paths in Dockerised applications must be controlled explicitly. A directory with the same name as a source module will shadow the intended module silently, and the failure may not be obvious from container startup logs. Using fully qualified module names in the `CMD` or `uvicorn` invocation eliminates ambiguity.

**Future Prevention:** Add a container startup assertion (`assert app.__file__.endswith("src/api/app.py")`) as part of the `/health` response or a startup log line.

---

#### CI Integration Test Hits Wrong Health-Check Endpoint  (2026-02-22)

**Area:** CI/CD / Deployment | **Stage:** Stage 8

**Symptom:** The CI integration test step always failed with a connection timeout or 404, even after the container started correctly.

**Impact:** CI pipeline showed the integration test as permanently failing, masking real failures.

**Root Cause (1):** The CI health-check URL was `GET /health`. The actual FastAPI endpoint was `GET /api/health`.  
**Root Cause (2):** The CI workflow used `sleep 10` to wait for the container to start. On slow CI runners, 10 seconds was sometimes insufficient; the test would hit the service before it was ready.

**Fix Implemented:**  
`.github/workflows/ci-cd.yml` — corrected health-check URL to `/api/health`.  
Replaced `sleep 10` with a poll loop that checks the endpoint every second for up to 30 seconds and exits cleanly once the service responds 200.

**Verification:** CI integration test passes consistently across multiple runs.

**Evidence:** Commit `edbc399` — `.github/workflows/ci-cd.yml` diff (15 lines changed).

**Lesson Learned (thesis-ready):** CI smoke tests must be kept synchronised with the actual API contract. A health-check URL hardcoded in the workflow file becomes a silent failure point whenever the API route structure changes. Polling with a timeout is more robust than a fixed sleep, and makes CI approximately 5–15 seconds faster on fast runners.

**Future Prevention:**  
- Define the health-check URL in a single config variable shared between the workflow and the smoke test script.  
- Add the endpoint path to the test contract in `tests/integration/test_api.py`.

---

### E) Monitoring and Drift Issues

---

#### Missing Reference Baseline Blocks AdaBN, TENT, and Drift Detection  (2026-02-19)

**Area:** Monitoring / Retraining | **Stage:** Stage 9 / Stage 12

**Symptom:** Running `python run_pipeline.py --mode adabn_tent` or `--mode retrain` raised a `FileNotFoundError` for `models/normalized_baseline.json`. The adaptation and drift detection stages could not start.

**Impact:** The entire adaptation pipeline (AdaBN, TENT, pseudo-labelling) was blocked. This delayed end-to-end testing by approximately one day.

**Root Cause:** `scripts/build_training_baseline.py` did not exist; it was referenced in the CLI and docs but had never been created. `models/normalized_baseline.json` therefore never existed in the repo.

**Fix Implemented:**  
`scripts/build_training_baseline.py` (new file, 283 lines) — computes per-channel statistics from the training dataset and saves `models/training_baseline.json` and `models/normalized_baseline.json`.  
`src/components/baseline_update.py` — added save call.  
`run_pipeline.py` — exposed `--update-baseline` CLI flag.

**Verification:** `700381a` — file created and committed. Subsequent runs of `--mode adabn_tent` succeed.

**Evidence:** Commit `700381a` — `scripts/build_training_baseline.py` created (300 lines added).

**Lesson Learned (thesis-ready):** Scripts that are referenced in documentation and CLI help text must exist before those references are committed. A missing prerequisite script is effectively a broken dependency. Dependency existence checks (e.g., an assertion at CLI startup that the baseline file exists) catch this class of error early.

**Future Prevention:** Add a startup guard in `run_pipeline.py`: if `models/normalized_baseline.json` does not exist, print a clear error message with the exact command needed to create it (`python scripts/build_training_baseline.py`).

---

### F) Retraining and Adaptation Safety Issues

---

#### TENT Gradient Loop Corrupts AdaBN Running Statistics  (2026-02-19)

**Area:** Retraining / Domain Adaptation | **Stage:** Stage 12 — Safe Retraining

**Symptom:** After running AdaBN followed by TENT, the post-adaptation mean confidence was lower than the pre-AdaBN baseline. AdaBN should have improved calibration; TENT should have maintained or improved it further. The combined result was worse than neither method alone.

**Impact:** The adaptation pipeline was unreliable. In the worst case, adapting the model made it less accurate than the unadapted original, defeating the purpose of the entire Stage 12 mechanism.

**Root Cause:** AdaBN works by running forward passes in training mode to update the BatchNorm running mean and running variance. TENT also runs the BN layers in training mode during its gradient loop. Each `apply_gradients` step (in TENT) triggered a new training-mode forward pass that updated the BN running statistics, overwriting the statistics that AdaBN had carefully computed. By the end of TENT, the running statistics reflected only the last few TENT minibatches, not the full target-domain distribution.

**Fix Implemented:**  
`src/domain_adaptation/tent.py`:
- Before the gradient loop begins, snapshot all BN layer `running_mean` and `running_var` tensors.
- After each `apply_gradients` step, restore the running statistics from the snapshot.
- This ensures TENT updates only `gamma` (scale) and `beta` (shift) — the affine parameters — without touching the distribution statistics set by AdaBN.

**Verification:** After the fix, the AdaBN → TENT sequence shows: AdaBN sets stable running statistics; TENT refines gamma/beta without disturbing them. Archived model `models/retrained/adabn_tent_adapted_model.keras` (created `2026-02-23 00:31`) was produced by the fixed pipeline.

**Evidence:** Commit `f47a48b` — `src/domain_adaptation/tent.py` diff (107 lines changed); commit `e2bc784` — further TENT hardening (55 lines).

**Lesson Learned (thesis-ready):** When two adaptation methods are chained, their interaction with shared mutable state (BatchNorm running statistics) must be analysed explicitly. AdaBN and TENT both affect BatchNorm layers but through different parameters; TENT's training-mode forward passes are an implicit side effect that silently invalidates AdaBN's output. Snapshotting and restoring shared state between adaptation steps is the correct pattern for composable test-time adaptation.

**Future Prevention:**  
- Add a test that checks BN running statistics before and after TENT: they should be identical (only gamma/beta should change).  
- Document the adaptation order and state contract in `src/domain_adaptation/__init__.py`.

---

#### TENT Has No Rollback Gate — Can Make Model Worse  (2026-02-19)

**Area:** Retraining / Safety | **Stage:** Stage 12

**Symptom:** On some production batches, TENT reduced mean confidence. There was no mechanism to detect this or revert to the pre-TENT model.

**Impact:** TENT might be applied to OOD data (data far outside the training distribution), where entropy minimisation pushes the model toward spurious low-entropy predictions — harmful adaptation.

**Root Cause:** The original TENT implementation lacked any post-adaptation quality check. It applied gradient updates unconditionally.

**Fix Implemented (two rounds):**  
`src/domain_adaptation/tent.py` (commit `f47a48b`):  
- OOD safety gate: if initial mean normalised entropy > `ood_entropy_threshold = 0.85`, skip TENT entirely and return the original model unchanged.

`src/domain_adaptation/tent.py` (commit `e2bc784`):  
- Confidence-drop rollback gate: if mean confidence drops by more than 1 percentage point after TENT completes, restore all BN gamma/beta to pre-TENT values.  
- All rollback reasons are combined and logged in `tent_meta`.  
- `src/components/model_retraining.py` unpacks `tent_meta` and logs `tent_rollback` and `entropy_delta` to MLflow.

**Verification:** Model registry (`models/registry/model_registry.json`) now records `tent_rollback: 0/1` per run. The adaptation run on 2026-02-23 shows `tent_rollback: 0` (not triggered) but records `confidence_improvement: -0.076` — this exceeded the promotion threshold (5%), causing archiving rather than production promotion. The safety chain worked as designed.

**Evidence:** Commits `f47a48b`, `e2bc784` — `src/domain_adaptation/tent.py` and `src/components/model_retraining.py` diffs.

**Lesson Learned (thesis-ready):** Test-time adaptation methods must include explicit quality gates and rollback mechanisms as standard components, not optional additions. An adaptation method that degrades with a probability, even if small, will eventually cause harm in a continuously operating system. Combining an OOD input gate with a post-adaptation quality check provides defence in depth.

**Future Prevention:** Add a test that passes deliberately OOD data (random noise) to `tent_adapt()` and asserts the function returns the original model unchanged.

---

#### Baseline Update Writes to Shared Production Path During Non-Promoting Runs  (2026-02-19)

**Area:** Retraining / Safety | **Stage:** Stage 12 / Stage 13

**Symptom:** Running the pipeline in `--mode inference` or any non-canary mode was overwriting `models/training_baseline.json` and `models/normalized_baseline.json`. These files are the reference distributions used by all drift detection and monitoring code. A failed test run could corrupt the production monitoring baseline silently.

**Impact:** Drift detection thresholds were based on a corrupted reference; false positives and false negatives in drift reporting could both result.

**Root Cause:** `src/components/baseline_update.py` wrote to the shared `models/` paths unconditionally, regardless of whether the current run was a canary promotion or just an exploratory re-run.

**Fix Implemented:**  
`src/entity/config_entity.py` — added `BaselineUpdateConfig.promote_to_shared` field, defaulting to `False`.  
`src/components/baseline_update.py` — all writes to `models/training_baseline.json` and `models/normalized_baseline.json` are gated behind `if promote_to_shared: ...`. Non-promoting runs save only to the local `artifacts/{timestamp}/` directory.  
`run_pipeline.py` — added `--update-baseline` CLI flag; only when this flag is passed does `promote_to_shared = True`.

**Verification:** Running the pipeline without `--update-baseline` no longer modifies any file in `models/`. A dedicated test verifies that a mock run without the flag does not write to `models/training_baseline.json`.

**Evidence:** Commits `f47a48b`, `e2bc784` — `src/components/baseline_update.py` and `src/entity/config_entity.py` diffs.

**Lesson Learned (thesis-ready):** Shared production artefacts (baselines, normalisation statistics) must be protected by an explicit promotion gate. The default behaviour of any pipeline run should be read-only with respect to shared paths. Writes to shared paths should require a deliberate, named action — a command-line flag or a registry promotion step — not an implicit side effect of running the pipeline.

**Future Prevention:** Add a write-protection assertion in `baseline_update.py`: if `promote_to_shared` is `False` and the function is about to write to `models/`, raise `PermissionError`.

---

### G) MLOps Tooling Issues

---

#### MLflow `log_model` Positional Argument Incompatibility  (2026-02-19)

**Area:** MLOps Tooling | **Stage:** Stage 6 — Experiment Tracking

**Symptom:** Model artefact logging failed in the CI environment (`mlflow.exceptions.MlflowException`). The model was not saved to the MLflow run. The CI environment used a slightly older MLflow release than the development laptop.

**Impact:** Model artefacts were absent from some MLflow runs. Reproducibility was broken for those runs — given the run ID, the model could not be restored.

**Root Cause:** `mlflow.keras.log_model(model, "model")` — the second argument was passed positionally. In the newer MLflow API, the parameter name changed, making the positional call incompatible with older versions.

**Fix Implemented:**  
`src/mlflow_tracking.py` — changed to `mlflow.keras.log_model(model, name="model")`.  
Added `try/except` fallback for older MLflow versions.  
`src/pipeline/production_pipeline.py` — added auto-generated `input_example` derived from `model.input_shape` so MLflow can infer the model signature.

**Verification:** CI MLflow logging succeeds across versions. MLflow UI shows model artefacts attached to all runs.

**Evidence:** Commit `e2bc784` — `src/mlflow_tracking.py` diff.

**Lesson Learned (thesis-ready):** External library APIs should be called with explicit keyword arguments rather than positional arguments. Positional calls are brittle to parameter order changes across library versions. Keyword-argument calls are also self-documenting.

**Future Prevention:** Pin MLflow version in `requirements.txt` and `requirements-lock.txt`. Add a CI step that verifies the installed MLflow version matches the pinned version.

---

#### `tf.function` Retracing Warnings Slow TENT by 10×  (2026-02-19)

**Area:** MLOps Tooling / Retraining | **Stage:** Stage 12

**Symptom:** TENT adaptation emitted hundreds of `WARNING: tf.function retracing` messages. Each gradient step took approximately 3–5 seconds instead of the expected <0.1 second. A 10-step TENT run took several minutes.

**Impact:** TENT was impractically slow. In a production pipeline with a monitoring cycle of minutes, this delay was unacceptable.

**Root Cause:** Inside the TENT gradient loop, `model.predict(batch)` was called. `model.predict()` uses a different TensorFlow graph than the `@tf.function`-compiled forward pass. Each call with a new tensor shape caused TensorFlow to retrace and recompile the computation graph.

**Fix Implemented:**  
`src/domain_adaptation/tent.py` — replaced `model.predict(batch)` with `model(tf.constant(batch), training=False)`. This calls the model directly through the existing compiled graph, bypassing `predict`'s internal batching logic.

**Verification:** Retracing warnings eliminated. TENT loop completes 10 steps in under 2 seconds on CPU.

**Evidence:** Commit `e2bc784` — `src/domain_adaptation/tent.py` diff.

**Lesson Learned (thesis-ready):** `model.predict()` is designed for end-user inference on large datasets; it is not suitable for use inside gradient loops or adaptation loops where tight control over the computation graph is required. Direct model calls (`model(x, training=False)`) are the correct pattern inside custom training loops.

---

### H) CI/CD Issues

---

#### Docker Image Name Uppercase Breaks GHCR Push  (2026-02-15)

**Area:** CI/CD / Deployment | **Stage:** Stage 8

**Symptom:** The `docker push` step in the CI/CD workflow failed with a 400 error from GitHub Container Registry.

**Impact:** No Docker image was published. The deployment stage of CI was permanently broken until the fix.

**Root Cause:** GitHub Container Registry (`ghcr.io`) requires image names to be entirely lowercase. The workflow `env` variable contained `MasterArbeit_MLops` with uppercase letters.

**Fix Implemented:**  
`.github/workflows/ci-cd.yml` — changed `IMAGE_NAME` to `shalinvachheta017/masterarbeit_mlops/har-inference` (all lowercase).

**Verification:** Docker push step completes with exit code 0. Image appears in GHCR under the repository packages.

**Evidence:** Commit `8b4dab7` — `.github/workflows/ci-cd.yml` diff (2 lines changed).

**Lesson Learned (thesis-ready):** Container registry naming conventions should be validated before the CI pipeline is first run, not discovered through a failing push. Registry-specific constraints (lowercase names, path depth limits, character restrictions) are not enforced locally by `docker build` and only surface at push time.

**Future Prevention:** Add a CI step that validates the image name with a regex (`^[a-z0-9/_\-:]+$`) before the build step.

---

#### CI Test Suite Times Out — TF Tests in Wrong Job  (2026-02-19)

**Area:** CI/CD | **Stage:** All (CI infrastructure)

**Symptom:** The unit test CI job exceeded its time limit regularly. All tests nominally passed but the job timed out before the result was recorded.

**Impact:** CI green/red status was unreliable. Developers could not trust the CI result.

**Root Cause:** TensorFlow-dependent tests (AdaBN, TENT, pipeline integration) were running inside the standard unit test job alongside fast pure-Python tests. TF model loading and forward passes on CPU can take 30–120 seconds per test; with many such tests, the job exceeded the configured time limit.

**Fix Implemented:**  
`.github/workflows/ci-cd.yml` — split into two jobs:  
- `test-fast` — runs `pytest -m "not slow"` (pure Python, no TF model loading).  
- `test-slow` — runs `pytest -m slow` (TF-dependent tests, longer timeout, non-blocking for PR merge gates).  
`tests/test_adabn.py`, `tests/test_pipeline_integration.py`, `tests/test_retraining.py` — marked with `@pytest.mark.slow`.  
`src/components/baseline_update.py` — added artifact copy guard to prevent concurrent test runs from overwriting each other's outputs.

**Verification:** CI fast job completes in under 2 minutes. Slow job runs in parallel and separately.

**Evidence:** Commit `1ae27cc` — `.github/workflows/ci-cd.yml` diff (56 lines), `tests/` diffs.

**Lesson Learned (thesis-ready):** Test suites should be stratified by execution time and resource requirements from the beginning of a project, not retrofitted after timeouts are encountered. Marking tests with speed categories (`slow`, `fast`, `integration`) and running them in separate CI jobs enables fast developer feedback on small changes while still running the full test suite on significant changes.

**Future Prevention:** Enforce the `@pytest.mark.slow` rule: add a CI check that runs `pytest --collect-only -m slow` and fails if any test imports TensorFlow without the slow marker.

---

#### Config Key Rename Not Propagated — Stage 11 Crashes  (2026-02-22 / 2026-02-23)

**Area:** CI/CD / Trigger Policy | **Stage:** Stage 11 — Trigger Policy

**Symptom:** After committing the PSI threshold rename (`psi_warn` → `drift_zscore_warn`), Stage 11 raised a `KeyError` at runtime. The pipeline could not evaluate any trigger decision.

**Impact:** The trigger policy was completely non-functional for approximately 55 minutes between the two commits. Any production pipeline run during this window would have crashed at Stage 11.

**Root Cause:** `src/trigger_policy.py`, `src/entity/config_entity.py`, `scripts/post_inference_monitoring.py`, and the tests were all updated to use `drift_zscore_warn`. However, `src/components/trigger_evaluation.py` — a wrapper that reads the config — still referenced the old key name `drift_psi_warn`.

**Fix Implemented:**  
`src/components/trigger_evaluation.py` — updated two references from `drift_psi_warn` to `drift_zscore_warn`.

**Verification:** Stage 11 completes without error. Integration test passes.

**Evidence:** Commit `c4b4994` (rename), commit `b92ae0a` (follow-up fix) — `src/components/trigger_evaluation.py` diff.

**Lesson Learned (thesis-ready):** Configuration key renames are a high-risk change because the failure is a runtime `KeyError`, not a compile-time error. A single rename must be applied atomically across all consumers in the same commit. Running the full end-to-end integration test — not just unit tests — after a config rename is the most reliable way to catch consumers that were missed.

**Future Prevention:**  
- Define all config key names as string constants in `src/config.py` and import them everywhere, rather than using raw string literals. A rename then causes a Python `NameError` at import time, caught by any test.  
- Add an integration test that explicitly exercises Stage 11 from end to end.

---

## Most Important Fixes (For Thesis Highlights)

The following fixes had the greatest impact on pipeline reliability and scientific validity. They are recommended as primary references for the thesis discussion chapter on implementation challenges.

1. **Accelerometer unit mismatch fix** — increased cross-user accuracy from ~14% (near-random) to functional levels. The most consequential fix in the entire project. (`c17f3dd` → `1c403f8`)

2. **TENT running-statistics freeze** — enabled the AdaBN + TENT adaptation sequence to function correctly as a composed pipeline. Without this fix, TENT silently invalidated AdaBN's output. (`f47a48b`)

3. **TENT + AdaBN rollback gates** — added the safety layer that prevents adaptation from degrading a well-functioning model. The two-gate design (OOD input gate + post-adaptation quality gate) is a novel contribution to pipeline safety. (`f47a48b`, `e2bc784`)

4. **Baseline governance flag** — eliminated the risk of a test run corrupting the production monitoring reference baseline. This is a property-level guarantee, not a reactive fix. (`f47a48b`, `e2bc784`)

5. **Pseudo-label scaler alignment + self-consistency filter** — ensured that labelled source data and unlabelled production data are on the same scale inside the retraining loop. The self-consistency filter (retaining only the 14.5% of source windows where the model agrees with the true label) is a pragmatic data-quality measure. (`3fd3c00`)

6. **PSI multi-channel threshold recalibration** — demonstrated that statistical thresholds from academic literature must be empirically re-calibrated for the specific aggregation scheme in use. (`bd8dc1e`)

7. **train.py architecture versioning** — ensured that any model produced by the retraining pipeline is architecturally identical to the pretrained model, making comparative evaluation valid. (`5c38f65`)

8. **Config key rename as atomic operation** — the two-commit failure (`c4b4994`, `b92ae0a`) provided the concrete lesson that config key renames require end-to-end testing, leading to the recommendation to use named string constants.

---

*End of document.  
All entries are supported by commit history evidence. Evidence hashes are verifiable with `git show <hash> --stat` in the repository root.*

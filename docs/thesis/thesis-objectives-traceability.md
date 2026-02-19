# Thesis Objectives Traceability Matrix

**Purpose:** For every thesis claim, show the pipeline stage, the code, the command, and the artifact.  
**Thesis:** "Developing a MLOps Pipeline for Continuous Mental Health Monitoring using Wearable Sensor Data"  
**Last Updated:** Feb 19, 2026

---

## 1. Pipeline Objectives → Stage → Code → Artifact

| # | Objective | Stage | Code Entrypoint | Validation Command | Expected Artifact |
|---|-----------|-------|-----------------|--------------------|--------------------|
| O1 | Raw Garmin data → fused 50 Hz CSV | 1 Ingestion | `src/components/data_ingestion.py` | `python run_pipeline.py` | `data/processed/sensor_fused_50Hz.csv` (>0 rows) |
| O2 | Schema + value-range validation | 2 Validation | `src/components/data_validation.py` | (same) | Validation pass/fail in pipeline result JSON (`stages_completed` contains `"validation"`) |
| O3 | CSV → normalised windowed `.npy` | 3 Transformation | `src/components/data_transformation.py` | (same) | `data/prepared/production_X.npy` shape `(N, 200, 6)` |
| O4 | Pretrained model inference | 4 Inference | `src/components/model_inference.py` | (same) | `data/prepared/predictions/predictions_*.csv` + `.npy` |
| O5 | Confidence / entropy / ECE evaluation | 5 Evaluation | `src/components/model_evaluation.py` | (same) | `reports/evaluation_report*.json` with confidence_summary |
| O6 | 3-layer monitoring (confidence, temporal, drift) | 6 Monitoring | `scripts/post_inference_monitoring.py` | (same) | `reports/monitoring_report*.json` with layer1/2/3 |
| O7 | Automated trigger decision | 7 Trigger | `src/components/trigger_evaluation.py` + `src/trigger_policy.py` | (same) | `TriggerDecision` in pipeline result JSON (action + reasons + drift_score) |
| O8 | Domain adaptation retraining | 8 Retraining | `src/components/model_retraining.py` → `src/train.py` | `python run_pipeline.py --retrain --adapt adabn_tent` | `models/retrained/*.keras` + metrics (`final_accuracy`, `val_accuracy`) in result JSON |
| O9 | Model versioning + registry | 9 Registration | `src/components/model_registration.py` | (same with `--retrain`) | `models/registry/` version entry + MLflow model version |
| O10 | Baseline rebuild + traceability | 10 Baseline Update | `src/components/baseline_update.py` + `scripts/build_training_baseline.py` | (same with `--retrain`) | `models/training_baseline.json` + `models/normalized_baseline.json` + copy in `artifacts/<run_id>/models/` |
| O11 | CI/CD automation | — | `.github/workflows/ci-cd.yml` | Push to `main` | Green CI badge (206 unit tests pass, slow tests non-blocking) |
| O12 | Experiment tracking | — | `src/mlflow_tracking.py` | `mlflow ui` | `mlruns/` experiment runs with metrics + artifacts |
| O13 | Reproducibility | — | `data/*.dvc`, `docker/Dockerfile.*`, `docker-compose.yml` | `dvc pull` / `docker-compose up` | Data versioned via DVC, environment containerised |

---

## 2. Research Questions → Evidence Mapping

From `docs/thesis/THESIS_STRUCTURE_OUTLINE.md`:

### RQ1: How can we detect model degradation without ground truth labels?

| Evidence | Source | Location |
|----------|--------|----------|
| PSI-based drift detection per channel | Stage 6 monitoring layer 3 | `reports/monitoring_report*.json` → `layer3_drift` |
| Confidence + entropy proxy metrics | Stage 5 evaluation + Stage 6 layer 1 | `reports/evaluation_report*.json` → `confidence_summary` |
| Baseline schema guard (catches invalid baselines) | Stage 6 pre-check | `scripts/post_inference_monitoring.py` lines 260–270 |
| Audit run **A1** demonstrates detection on real production data | Pipeline result JSON | `logs/pipeline/pipeline_result_*.json` |

### RQ2: What proxy metrics correlate with actual accuracy in HAR?

| Evidence | Source | Location |
|----------|--------|----------|
| Mean confidence, std confidence, uncertain % | Stage 5 evaluation | `reports/evaluation_report*.json` |
| ECE (Expected Calibration Error) | Stage 5 evaluation | Same report |
| Normalised entropy distribution | Stage 6 monitoring layer 1 | `reports/monitoring_report*.json` → `layer1_confidence` |
| Per-activity confidence stratification | Stage 5 evaluation | Logged per predicted class |
| Compare A1 (no retrain) vs A4/A5 (adapted) confidence | Ablation table | Phase 3 §3.4 of ACTION_PLAN |

### RQ3: When should we trigger model adaptation or retraining?

| Evidence | Source | Location |
|----------|--------|----------|
| Tiered trigger policy (DRIFT_PASS / DRIFT_WARN / DRIFT_ALERT) | Stage 7 | `src/trigger_policy.py` — `TriggerThresholds` dataclass |
| PSI thresholds: 0.75 warn / 1.50 critical (aggregated multi-channel) | Stage 7 config | `src/trigger_policy.py` lines 88–91 |
| Cooldown logic (prevent trigger storms) | Stage 7 | `src/trigger_policy.py` — `cooldown_hours` field |
| Audit run **A1** shows trigger decision on real drift | Pipeline result JSON | `logs/pipeline/pipeline_result_*.json` → `trigger` block |

### RQ4: How effective is pseudo-labeling for HAR model updates?

| Evidence | Source | Location |
|----------|--------|----------|
| Calibrated pseudo-labeling (temperature scaling + entropy gate + class-balanced top-k) | Stage 8 | `src/train.py` → `_retrain_pseudo_labeling()` |
| Temperature calibration value (T) | A5 metrics | `calibration_temperature` in pipeline result |
| Pseudo-labeled sample count + class balance | A5 metrics | `pseudo_labeled_samples`, per-class counts |
| Val accuracy after pseudo-label fine-tuning | A5 metrics | `val_accuracy` in pipeline result |
| Comparison: A1 baseline vs A4 (unsupervised) vs A5 (pseudo-label) | Ablation table | Phase 3 §3.4 of ACTION_PLAN |
| OOD guard preventing adaptation on extreme drift | TENT module | `src/domain_adaptation/tent.py` → `ood_entropy_threshold` |
| Soft targets with label smoothing (ε=0.1) | Stage 8 | `src/train.py` — pseudo-label fine-tune loop |

---

## 3. How to Reproduce (Audit Runs)

```bash
# A1 — Inference only (stages 1–7)
python run_pipeline.py

# A3 — Supervised retrain, no adaptation (stages 1–10)
python run_pipeline.py --retrain --adapt none

# A4 — AdaBN + TENT unsupervised adaptation (stages 1–10)
python run_pipeline.py --retrain --adapt adabn_tent

# A5 — Calibrated pseudo-labeling (stages 1–10)
python run_pipeline.py --retrain --adapt pseudo_label

# Verify artifacts after each run
python scripts/audit_artifacts.py                     # auto-pick latest run
python scripts/audit_artifacts.py --retrain           # include stages 8–10 checks
```

---

## 4. Adaptation Methods Implemented

| Method | Code | Labels needed? | Thesis claim |
|--------|------|----------------|--------------|
| AdaBN | `src/domain_adaptation/adabn.py` | No | Updates BN running stats from target data — fast, zero-label baseline |
| TENT | `src/domain_adaptation/tent.py` | No | Entropy-minimisation on BN affine params, OOD-guarded — state-of-art unsupervised TTA |
| AdaBN+TENT | dispatched in `src/components/model_retraining.py` | No | Two-stage: AdaBN updates stats, then TENT fine-tunes affine — strongest unsupervised |
| Pseudo-label (calibrated) | `src/train.py` → `_retrain_pseudo_labeling()` | No (self-generated) | Temperature-scaled, entropy-gated, class-balanced, soft-target fine-tuning |

---

## 5. Key References

| Topic | File |
|-------|------|
| Full pipeline architecture | `docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md` |
| Monitoring & retraining design | `docs/MONITORING_AND_RETRAINING_GUIDE.md` |
| Thesis chapter outline | `docs/thesis/THESIS_STRUCTURE_OUTLINE.md` |
| Preprocessing decisions | `docs/PREPROCESSING_COMPARISON_AND_ADAPTATION.md` |
| Baseline governance rules | `docs/ACTION_PLAN_18_20_FEB_2026.md` → Phase 3 §3.3 |
| Training recipe (pretrain vs finetune) | `docs/TRAINING_RECIPE_MATRIX.md` |
| CI/CD pipeline | `.github/workflows/ci-cd.yml` |

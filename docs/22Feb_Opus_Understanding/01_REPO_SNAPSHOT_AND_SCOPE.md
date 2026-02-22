# 01 — Repository Snapshot and Scope

> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`
> **Branch:** `main` | **Commit Date:** 2026-02-19 18:04:31 +0100
> **Commit Message:** `docs: reorganise into archive/technical/research/thesis, rename guide-*/OLD_*, update README`
> **Audit Date:** 2026-02-22

---

## 1. Snapshot Identifier

| Property | Value |
|----------|-------|
| Full SHA | `168c05bb222b03e699acb7de7d41982e886c8b25` |
| Short SHA | `168c05b` |
| Branch | `main` |
| Remote | `origin/main`, `origin/HEAD` |
| Commit time | 2026-02-19 18:04:31 +0100 |
| Prior commits inspected | `b84e583` (docs reorg), `e2bc784` (TENT/baseline/GPU fixes), `f47a48b` (TENT rollback/safety), `3fd3c00` (thesis closure/audit) |

**FACT:** All analysis in this documentation pack refers to this snapshot. Confidence: **High**

**ASSUMPTION:** Uncommitted changes exist (3 modified files in `models/registry/`, plus many untracked temp/hash-named directories). These are not included in the code audit scope.

---

## 2. Repo Scan Scope

### Included in Analysis

| Path | What Was Analyzed |
|------|-------------------|
| `src/` | All 56 Python source files (excluding `__pycache__`, `.mypy_cache`, `Archived/`) |
| `src/components/` | All 15 pipeline stage wrappers |
| `src/domain_adaptation/` | `adabn.py`, `tent.py` |
| `src/pipeline/` | `production_pipeline.py`, `inference_pipeline.py` |
| `src/api/` | `app.py` (FastAPI) |
| `src/entity/` | `artifact_entity.py`, `config_entity.py` |
| `src/utils/` | `artifacts_manager.py`, `common.py`, `main_utils.py`, `production_optimizations.py` |
| `tests/` | 19 test files (215 test functions), `conftest.py` |
| `scripts/` | 12 utility scripts |
| `.github/workflows/` | `ci-cd.yml` (7 CI/CD jobs) |
| `docker/` | `Dockerfile.inference`, `Dockerfile.training`, `docker/api/main.py` |
| `config/` | 7 config files (pipeline, MLflow, Prometheus, Grafana, alerts, requirements, pylint) |
| `logs/pipeline/` | 60 pipeline result JSON files |
| `artifacts/` | 32 timestamped artifact directories |
| `models/registry/` | `har_model_vauto.keras`, `model_registry.json` |
| `docs/` | 70+ markdown files (referenced but not trusted for completion %) |
| `run_pipeline.py` | Main entry point |
| `docker-compose.yml` | 4-service compose file |
| `pyproject.toml`, `pytest.ini`, `setup.py` | Build/test configuration |

### Excluded from Analysis

| Path | Reason |
|------|--------|
| `.git/` | Internal git data |
| `__pycache__/`, `.mypy_cache/` | Build artifacts |
| `venv/`, `.venv/` | Virtual environments |
| `src/Archived(...)` | Legacy/deprecated files (explicitly archived) |
| `mlruns/` | MLflow tracking data (too large for static audit) |
| `data/` | Raw/processed data files (not code) |
| Untracked hash-named directories (e.g., `08fs4e4x`, `0d7l9o3w`, etc.) | Appear to be temporary/scratch files, not part of the codebase |

---

## 3. Key Entry Points Found

| Entry Point | Type | Location | Purpose |
|-------------|------|----------|---------|
| `run_pipeline.py` | CLI | Root | Main pipeline orchestrator entry — 20+ arguments, YAML config support |
| `src/pipeline/production_pipeline.py` | Class | `src/pipeline/` | `ProductionPipeline.run()` — executes stages 1-10 sequentially |
| `src/api/app.py` | FastAPI | `src/api/` | REST API: `/api/upload`, `/api/health`, `/api/model/info`, embedded dashboard |
| `docker/api/main.py` | FastAPI | `docker/api/` | Docker-specific API entry (separate from `src/api/app.py`) |
| `pyproject.toml` → `har-pipeline` | CLI entry | `src.cli:main` | Package entry point (ASSUMPTION: `src/cli.py` may not exist) |

**RISK:** Two separate API entry points (`src/api/app.py` and `docker/api/main.py`) may diverge. Confidence: **Medium**

---

## 4. High-Level Architecture Clues

### Pipeline Architecture (14-stage design, 10 currently orchestrated)

```
[FACT: Verified from production_pipeline.py ALL_STAGES list]

Stages 1-10 (ORCHESTRATED — run by ProductionPipeline.run()):
  1. Data Ingestion → 2. Data Validation → 3. Data Transformation →
  4. Model Inference → 5. Evaluation → 6. Post-Inference Monitoring →
  7. Trigger Evaluation → 8. Model Retraining → 9. Model Registration →
  10. Baseline Update

Stages 11-14 (IMPLEMENTED — not wired into orchestrator):
  11. Calibration & Uncertainty
  12. Wasserstein Drift Detection
  13. Curriculum Pseudo-Labeling
  14. Sensor Placement Robustness
```

[CODE: src/pipeline/production_pipeline.py:L53 | ALL_STAGES list]

### Component Pattern

Each stage follows a consistent wrapper pattern:
- **Config entity** (`src/entity/config_entity.py`) defines stage parameters
- **Component class** (`src/components/<stage>.py`) is the wrapper
- **Core module** (`src/<module>.py` or `scripts/<module>.py`) does the real work
- **Artifact entity** (`src/entity/artifact_entity.py`) defines stage output schema

[CODE: src/entity/artifact_entity.py:L1-L271 | 14 artifact dataclasses + PipelineResult]

### Model Architecture

**FACT:** 1D-CNN-BiLSTM, ~1.5M parameters, 200 timesteps × 6 sensors, 11 activity classes.
[CODE: src/train.py | symbol:HARModelBuilder]

### Adaptation Methods Available

| Method | Module | Status |
|--------|--------|--------|
| AdaBN | `src/domain_adaptation/adabn.py` | Implemented, integrated via retraining stage |
| TENT | `src/domain_adaptation/tent.py` | Implemented, integrated via retraining stage |
| AdaBN+TENT | Combined in `src/components/model_retraining.py` | Integrated |
| Pseudo-labeling | `src/train.py` → `DomainAdaptationTrainer` | Integrated via retraining stage |
| Curriculum pseudo-labeling | `src/curriculum_pseudo_labeling.py` | Implemented, NOT orchestrated |

---

## 5. Human-Readable Repository Tree Summary

```
MasterArbeit_MLops/
├── run_pipeline.py                    # Main CLI entry (14-stage pipeline)
├── docker-compose.yml                 # 4-service compose (mlflow, inference, training, preprocessing)
├── pyproject.toml                     # Package config (v2.1.0, Python >=3.10)
├── pytest.ini                         # Test config (strict markers)
├── setup.py                           # Setuptools wrapper
│
├── src/                               # Core source (56 .py files)
│   ├── components/                    # 15 pipeline stage wrappers (Stages 1-14 + __init__)
│   ├── domain_adaptation/             # AdaBN + TENT implementations
│   ├── pipeline/                      # ProductionPipeline orchestrator + InferencePipeline
│   ├── api/                           # FastAPI application
│   ├── entity/                        # Config + Artifact dataclass schemas
│   ├── utils/                         # ArtifactsManager, helpers, production optimizations
│   ├── core/                          # Logger, exception handler
│   ├── train.py                       # HARTrainer + DomainAdaptationTrainer (~1200 lines)
│   ├── trigger_policy.py              # TriggerPolicyEngine (~820 lines)
│   ├── calibration.py                 # Temperature scaling, MC Dropout, ECE (~540 lines)
│   ├── wasserstein_drift.py           # Wasserstein + change-point drift (~460 lines)
│   ├── curriculum_pseudo_labeling.py  # Teacher-student EMA + EWC (~460 lines)
│   ├── sensor_placement.py            # Hand detection + axis mirroring (~370 lines)
│   ├── model_rollback.py              # ModelRegistry + rollback (~530 lines)
│   ├── prometheus_metrics.py          # Metrics exporter (~620 lines)
│   └── [other modules]               # data_validator, preprocess_data, evaluate_predictions, etc.
│
├── tests/                             # 19 test files, 215 test functions
│   ├── conftest.py                    # 12 shared fixtures (synthetic data, baselines)
│   ├── test_pipeline_integration.py   # 7 tests (mocked, marker: integration)
│   ├── test_adabn.py                  # 7 tests (real Keras models, marker: slow)
│   ├── test_trigger_policy.py         # 14 tests (policy engine, thresholds, escalation)
│   ├── test_calibration.py            # 11 tests (temperature scaling, ECE)
│   ├── test_wasserstein_drift.py      # 10 tests (drift detection, change points)
│   └── [13 more test files]
│
├── scripts/                           # 12 utility/operations scripts
│   ├── audit_artifacts.py             # Artifact completeness checker
│   ├── analyze_drift_across_datasets.py  # Cross-dataset drift comparison
│   ├── build_training_baseline.py     # Baseline statistics builder
│   └── [others]
│
├── docker/                            # Docker build context
│   ├── Dockerfile.inference           # python:3.11-slim, uvicorn, port 8000
│   ├── Dockerfile.training            # python:3.11-slim, requirements.txt
│   └── api/main.py                    # Docker-specific API entry
│
├── .github/workflows/ci-cd.yml       # 7-job CI/CD pipeline
│
├── config/                            # Configuration files
│   ├── pipeline_config.yaml           # Preprocessing/validation/inference params
│   ├── mlflow_config.yaml             # Experiment tracking config
│   ├── prometheus.yml                 # Scrape config (6 targets)
│   ├── alerts/har_alerts.yml          # 4 alerting rules
│   └── grafana/har_dashboard.json     # Dashboard definition
│
├── docs/                              # 70+ documentation files
│   ├── 19_Feb/                        # Previous work cycle docs
│   ├── stages/                        # Per-stage documentation (11 files)
│   ├── thesis/                        # Thesis structure, chapters, plans
│   ├── research/                      # Paper index, bibliography, QnA
│   └── technical/                     # Guides, operations, comparisons
│
├── logs/pipeline/                     # 60 pipeline result JSON files
├── artifacts/                         # 32 timestamped artifact snapshots
├── models/registry/                   # Registered model + registry JSON
├── mlruns/                            # MLflow experiment tracking data
└── data/                              # Raw and processed datasets
```

---

## 6. Notable Observations

1. **FACT:** Total estimated source code: ~7,500 non-blank, non-comment lines across 27 key files. Confidence: **High**
2. **FACT:** 215 test functions across 19 test files. Confidence: **High**
3. **FACT:** 60 pipeline run results in `logs/pipeline/`, 32 artifact snapshots. Confidence: **High**
4. **RISK:** Stages 11-14 are fully implemented but not orchestrated — this is the single largest integration gap. Confidence: **High**
5. **RISK:** `scripts/inference_smoke.py` is referenced by CI/CD but does not exist. Confidence: **High**
6. **INFERENCE:** The codebase represents genuine engineering effort (not boilerplate/template), based on domain-specific logic in adaptation, monitoring, and trigger modules. Confidence: **High**

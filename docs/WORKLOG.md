# Project Worklog

## Purpose
Daily running log of major repo changes, progress, decisions, and next steps.
Auto-generated from git commit history and supplemented manually.

---

## 2026-03-06
### Done
- Refactored `src/` into domain-based subpackages: `data/`, `training/`, `inference/`, `monitoring/`, `deployment/`
- Removed reversed dependencies from `scripts/` into `src/`
- Added backward-compatibility shim modules
- Verified `python -m compileall src -q` — clean
- Verified `pytest -q` — 258/258 passed
- Restructured `docs/` into organized subdirectories: `01_PIPELINE/`, `02_TECH/`, `03_EXPERIMENTS/`, `04_THESIS_WRITING/`, `99_ARCHIVE/`
- Created pipeline stage documentation (stages 01–10)
- Created thesis writing guides, chapter drafts, research Q&A, mentor questions
- Moved archived docs to `99_ARCHIVE/` and `thesis/_snapshots/`
- Improved CD workflow: added debug endpoints step, increased smoke-test wait to 60 iterations (~120s)
- Created paper catalog (`archive/_index/PAPERS_CATALOG.md`)

### Files / Areas Affected
- `src/data/`, `src/training/`, `src/inference/`, `src/monitoring/`, `src/deployment/` — new domain subpackages
- `src/pipeline/inference_pipeline.py` — updated imports
- `src/components/baseline_update.py`, `src/components/post_inference_monitoring.py` — updated imports
- `tests/test_baseline_update.py` — updated imports
- `.github/workflows/cd.yml` — debug step + 60-iteration wait
- `docs/01_PIPELINE/`, `docs/02_TECH/`, `docs/03_EXPERIMENTS/`, `docs/04_THESIS_WRITING/` — new doc structure
- `docs/99_ARCHIVE/` — archived old docs
- `thesis/_snapshots/` — historical snapshots preserved
- `archive/_index/PAPERS_CATALOG.md` — paper index

### Why
- Flat `src/` had 30+ modules — hard to navigate and explain in thesis
- Reversed dependencies (scripts importing src, src importing scripts) made the codebase fragile
- Docs were scattered with no clear organization
- CD smoke test was flaky — insufficient wait time and no debug visibility

### Status
- Refactor complete and tests passing (258/258)
- CD workflow improved but not yet validated (needs tag push)
- Docs restructured into 4 clear categories
- Paper catalog created

### Next
- Run full pipeline locally end-to-end
- Push tag to trigger CD and validate debug output
- Copy PDFs into `thesis/refs/papers_all/` organized library
- Start paper library index

---

## 2026-03-05
### Done
- Updated March TODO planning docs
- Fixed CI workflow issues
- Split combined CI/CD into separate `ci.yml` and `cd.yml` workflows
- Documented CI/CD triggers and structure
- Fixed CD workflow and inference container configuration

### Files / Areas Affected
- `.github/workflows/ci.yml` — new standalone CI
- `.github/workflows/cd.yml` — new standalone CD
- `docker/Dockerfile.inference` — inference container fixes

### Why
- Combined workflow was hard to debug and had conflicting triggers
- CI and CD have different trigger conditions (push/PR vs tags)

### Status
- CI green
- CD not yet green (smoke test failing — route/startup issue)

### Next
- Debug actual container routes with enhanced smoke test
- Validate refactored `src/` structure

---

## 2026-02-25 – 2026-02-26
### Done
- MiKTeX installation for LaTeX thesis compilation
- README update

### Files / Areas Affected
- `auto-install=yes/` — MiKTeX package files
- `README.md`

### Status
- LaTeX toolchain available for thesis compilation

---

## 2026-02-22 – 2026-02-23
### Done
- Major bug fix sprint: aligned code to 11 anxiety behavior classes (was using wrong PAMAP2 class list)
- Fixed train.py architecture to match pretrained model (~499K params, not ~1.5M)
- Fixed Docker container: resolved `api.app` module shadowing, set correct PYTHONPATH
- Fixed CI/CD: switched Docker to `src/api/app.py`, poll loop replaces `sleep 10`, corrected `/health` → `/api/health`
- Fixed Stage 11 crash — TTA governance registration fallback
- Renamed `psi_warn` → `drift_zscore_warn` (threshold 2.0σ / 3.0σ) with thesis parameter citations
- Fixed Stage 7 — reads `drift_zscore_warn` after rename
- Comprehensive cross-code audit of all markdown files
- Reformatted `train.py` for black compliance
- Config updates
- Moved thesis writing docs to `Thesis_report/` (local-only, gitignored)

### Files / Areas Affected
- `src/active_learning_export.py` — fixed class list
- `tests/conftest.py` — fixture uses correct 11 classes
- `src/train.py` — architecture alignment + black formatting
- `docker/Dockerfile.inference` — PYTHONPATH fix
- `.github/workflows/` — CI/CD route fixes
- `README.md` — corrected params, endpoints, progress
- Multiple docs — cross-code audit corrections
- `config/pipeline_config.yaml` — threshold rename

### Why
- Wrong class list caused silent data integrity issues
- Model param count mismatch between docs and code
- Docker shadowing caused container startup failures
- Threshold naming was inconsistent across codebase

### Status
- 215 tests passing after fixes
- CI green
- CD still failing on smoke test

---

## 2026-02-19
### Done
- Fixed baseline + metric loss bugs
- Added missing `build_training_baseline.py`, saved normalized baseline, exposed tent/adabn_tent in CLI
- Hardened PSI test, schema guard in Stage 6, schema_version, artifact baseline copy
- Split unit/slow tests, fixed artifact copy guard, marked TF tests as slow
- Thesis closure: traceability docs, audit script, fixed pseudo-label pipeline
- Fixed TENT running-stats freeze + rollback, baseline governance, GPU detection
- Fixed baseline governance (no `models/` write when not promoting), TENT confidence-drop rollback
- Fixed `tf.function` retracing, GPU diagnostic message, MLflow `name=` fix
- Reorganized Feb 19 docs into `docs/19_Feb/`
- Reorganized docs into archive/technical/research/thesis structure

### Files / Areas Affected
- `src/build_training_baseline.py` — new
- `src/tent_adaptation.py`, `src/adabn_adaptation.py` — TENT fixes
- `tests/` — PSI test fix, slow test markers
- `config/pipeline_config.yaml` — schema_version added
- Multiple doc reorganization

### Why
- TENT freeze/rollback was unsafe — could corrupt model on bad batches
- Schema validation was missing in Stage 6
- Test suite needed separation of fast/slow tests for CI efficiency
- Pseudo-label pipeline had safety issues

### Status
- All tests passing
- Pipeline hardened
- Docs reorganized

---

## 2026-02-15 – 2026-02-16
### Done
- Added FastAPI with web UI for CSV upload, inference, and monitoring
- Consolidated progress tracking into single source of truth
- Added production-ready CI/CD pipeline (GitHub Actions)
- Fixed Docker image name for ghcr.io compliance (lowercase)
- Monitoring docs completed
- GitHub Actions settings configured

### Files / Areas Affected
- `src/api/app.py` — new FastAPI application
- `.github/workflows/` — CI/CD workflows
- `docker/Dockerfile.inference` — Docker build
- Multiple docs — progress consolidation

### Why
- Needed serving layer for inference API
- Needed automated CI/CD for thesis demonstration
- Docker image naming rules require lowercase

### Status
- FastAPI running locally with CSV upload
- CI/CD pipeline created
- 95% progress on thesis pipeline

---

## 2026-02-14
### Done
- Fixed all tests: 225/225 passing
- Integrated monitoring pipeline
- Added production optimizations

### Files / Areas Affected
- `src/` — monitoring integration
- `tests/` — test fixes

### Status
- Full test suite green

---

## 2026-02-12
### Done
- Completed 10-stage MLOps pipeline with AdaBN domain adaptation
- Removed large image and model files from git tracking (moved to .gitignore)

### Files / Areas Affected
- `src/` — full pipeline implementation
- `.gitignore` — added model/image exclusions
- `run_pipeline.py` — pipeline orchestration

### Why
- Large binary files were bloating repo
- AdaBN domain adaptation is a core thesis contribution

### Status
- Full 10-stage pipeline implemented
- Models tracked via DVC, not git

---

## 2026-01-09 – 2026-01-17
### Done
- Documentation updates
- Updated .gitignore

### Status
- Maintenance period

---

## 2025-12-21
### Done
- Documentation updates

---

## 2025-12-16
### Done
- Updated analysis, progress, and stack documentation
- Updated pipeline and source files
- Removed obsolete and log files

---

## 2025-12-11 – 2025-12-12
### Done
- Reorganized documentation — added PATH_COMPARISON_ANALYSIS, archived outdated files
- Added complete MLOps infrastructure: DVC, MLflow, Docker
- README updates
- Repo reform / restructuring

### Files / Areas Affected
- `config/` — DVC and MLflow configs
- `docker/` — Dockerfile
- `.dvc/` — DVC initialization
- `docs/` — documentation structure

### Why
- Needed MLOps tooling foundation: data versioning (DVC), experiment tracking (MLflow), containerization (Docker)

---

## 2025-12-06 – 2025-12-08
### Done
- Resolved conversion ratio/solution for sensor data
- Phase 2 complete: model inference & evaluation pipeline

### Files / Areas Affected
- `src/run_inference.py`, `src/evaluate_predictions.py`

### Status
- Inference pipeline functional

---

## 2025-11-05 – 2025-11-28
### Done
- Identified accelerometer data error between prod and main
- Fixed data distribution error in acceleration values
- Main data added to .gitignore

### Why
- Sensor data had inconsistencies in acceleration units/ranges

---

## 2025-10-23 – 2025-10-25
### Done
- Initial commit with structured files and folders
- Data preprocessing pipeline created
- README added
- Excluded documentation from Git (keep progress private)

### Files / Areas Affected
- Full repo scaffolding
- `src/preprocess_data.py` — initial preprocessing
- `.gitignore` — docs exclusion

### Status
- Project initialized

---

## Current Repo Status
- **Branch:** `chore/docs-restructure`
- **Tests:** 258/258 passing
- **CI:** Green
- **CD:** Smoke test failing (debug step added, needs re-trigger)
- **Refactor:** `src/` domain subpackages complete
- **Docs:** Restructured into 4 categories
- **Paper catalog:** Created at `archive/_index/PAPERS_CATALOG.md`

## Open Issues
1. CD smoke test failing — container route mismatch or slow startup
2. Full pipeline not yet validated end-to-end after refactor
3. Paper library not yet consolidated into thesis/refs/papers_all/

## Recommended Next 3 Actions
1. Run full local pipeline once to validate refactor
2. Push tag to trigger CD, inspect debug output, fix route
3. Build thesis paper library with organized subfolders + index

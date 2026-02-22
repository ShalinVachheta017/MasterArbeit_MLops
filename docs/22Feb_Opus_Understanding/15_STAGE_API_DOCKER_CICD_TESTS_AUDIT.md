# 15 — Stage Deep-Dive: API, Docker, CI/CD, Tests, and Audit

> Part of [Opus Understanding Audit Pack](00_README.md) | Phase 2 — Technical Stage Deep-Dives
> **Commit:** `168c05bb` | **Audit Date:** 2026-02-22

---

## 1. FastAPI Application

### 1.1 Overview

**FACT:** Implemented in `src/api/app.py` (~775 lines, including embedded HTML dashboard).
[CODE: src/api/app.py]

- **Framework:** FastAPI 2.0.0 with Pydantic models
- **CORS:** Fully open (`allow_origins=["*"]`)
- **Lifespan:** Model and baseline loaded at startup via `lifespan` context manager

### 1.2 Endpoints

| Method | Path | Purpose | Authentication |
|--------|------|---------|---------------|
| `GET` | `/api/health` | Health check: model loaded, baseline loaded, uptime | None |
| `GET` | `/api/model/info` | Model metadata: name, version, input_shape, params, classes | None |
| `POST` | `/api/upload` | CSV upload → preprocessing → inference → 3-layer monitoring → results | None |
| `GET` | `/` | HTML dashboard with drag-and-drop upload UI | None |

### 1.3 Upload Pipeline

The `/api/upload` endpoint runs a **mini-pipeline** inline:
1. Read CSV from upload
2. Auto-detect sensor columns (case-insensitive pattern matching)
3. Create sliding windows (200 samples, 50% overlap) using `stride_tricks.as_strided`
4. Run model inference (`model.predict()`)
5. Run inline 3-layer monitoring (see doc 12, Section 7 for threshold divergence)
6. Return JSON with predictions, activity summary, monitoring, confidence stats

**FACT:** Uses `numpy.lib.stride_tricks.as_strided` for zero-copy windowing — efficient for large CSVs.
[CODE: src/api/app.py | function:_create_windows]

### 1.4 Embedded Dashboard

The API serves a full **single-page HTML dashboard** (~300 lines of HTML/CSS/JS) at `/`:
- Drag-and-drop CSV upload
- Status indicators (model loaded, baseline loaded, API health)
- Real-time inference results: activity distribution bars, confidence statistics, monitoring layer pills
- Window-level prediction table (first 200 windows)

**INFERENCE:** This is a thesis demonstration tool — not a production-grade frontend. CORS is wide open and there is no authentication.

---

## 2. Docker Setup

### 2.1 Dockerfiles

| File | Base Image | Purpose | Size Estimate |
|------|-----------|---------|---------------|
| `docker/Dockerfile.inference` | `python:3.11-slim` | FastAPI inference API | ~2GB (TF) |
| `docker/Dockerfile.training` | `python:3.11-slim` | Training environment | ~2.5GB (TF + extras) |

**FACT:** No `docker/Dockerfile` or `docker/Dockerfile.ci` exists. Only `.inference` and `.training`.
[CODE: docker/Dockerfile.inference]
[CODE: docker/Dockerfile.training]

**Dockerfile.inference highlights:**
- Installs FastAPI, uvicorn, TensorFlow, numpy, pandas, scipy
- Copies `src/`, `config/`, `docker/api/`
- Exposes port 8000
- Healthcheck: `curl -f http://localhost:8000/health`
- Entry: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

**RISK:** Entry point references `api.main:app` but the actual app is at `src.api.app:app`. This is a potential startup failure unless `docker/api/main.py` re-exports it.

**Dockerfile.training highlights:**
- Installs build-essential, git + full requirements
- Entry: `python -c "print('HAR Training Container Ready')"` — requires manual command override

### 2.2 Docker Compose (4 Services)

**FACT:** `docker-compose.yml` defines 4 services.
[CODE: docker-compose.yml]

| Service | Image / Build | Ports | Depends On |
|---------|--------------|-------|------------|
| `mlflow` | `python:3.11-slim` (inline pip) | 5000 | — |
| `inference` | `docker/Dockerfile.inference` | 8000 | mlflow |
| `training` | `docker/Dockerfile.training` | — | mlflow |
| `preprocessing` | `docker/Dockerfile.training` | — | — |

**Volume mounts:** `./data`, `./models`, `./mlruns`, `./logs` shared across services.

---

## 3. CI/CD Pipeline

### 3.1 Overview

**FACT:** `.github/workflows/ci-cd.yml` — 7 jobs, triggered on push to `main`/`develop` and PRs to `main`.
[CODE: .github/workflows/ci-cd.yml]

### 3.2 Job-by-Job Analysis

| Job | Name | Runs On | Depends On | Status |
|-----|------|---------|------------|--------|
| 1 | `lint` | ubuntu-latest | — | **Functional** — flake8, black, isort |
| 2 | `test` | ubuntu-latest | lint | **Functional** — pytest with coverage, excludes slow/integration/gpu |
| 3 | `test-slow` | ubuntu-latest | lint | **Functional** — TF tests, `continue-on-error: true` |
| 4 | `build` | ubuntu-latest | test | **Functional** — Docker build+push to GHCR |
| 5 | `integration-test` | ubuntu-latest | build | **Partially broken** — references `scripts/inference_smoke.py` (missing) |
| 6 | `model-validation` | ubuntu-latest | — | **Placeholder** — 3 echo statements |
| 7 | `notify` | ubuntu-latest | lint, test, build | **Placeholder** — echo only |

### 3.3 Gap Analysis

| Gap | Severity | Detail |
|-----|----------|--------|
| `scripts/inference_smoke.py` missing | **HIGH** | Integration test job references it, would fail at runtime |
| `on.schedule` not configured | **MEDIUM** | `model-validation` job runs only on `workflow_dispatch`, not scheduled |
| Model validation is 3 echo stubs | **HIGH** | No actual model download, validation, or drift check |
| Notify is echo-only | **LOW** | No Slack/email integration |
| Docker registry | **INFO** | Pushes to `ghcr.io/shalinvachheta017/masterarbeit_mlops/har-inference` |

**FACT:** [CODE: .github/workflows/ci-cd.yml | job:integration-test, step:"Run smoke tests"]

### 3.4 Test Filtering Strategy

```yaml
# Fast tests (no TF):
pytest tests/ -m "not slow and not integration and not gpu"

# Slow tests (TF, non-blocking):
pytest tests/ -m "slow" --continue-on-error
```

---

## 4. Test Suite

### 4.1 Overview

**FACT:** 215 test functions across 19 test files (from Phase 1 audit).
[ART: tests/ directory scan]

### 4.2 Test Pyramid

| Tier | Marker | Count (est.) | What They Test |
|------|--------|-------------|----------------|
| **Unit** | default (no marker) | ~150 | Config, entities, validators, utility functions |
| **Integration** | `integration` | ~30 | Component-to-component data flow |
| **Slow** | `slow` | ~35 | TensorFlow model loading, training, inference |
| **GPU** | `gpu` | ~5 | GPU-specific tests (excluded in CI) |

### 4.3 Coverage Highlights

| Module | Covered By |
|--------|-----------|
| Data ingestion | `tests/test_data_ingestion.py` |
| Data validation | `tests/test_data_validation.py` |
| Data transformation | `tests/test_data_transformation.py` |
| Monitoring (3-layer) | `tests/test_monitoring.py` |
| Trigger policy | `tests/test_trigger.py` |
| Domain adaptation | `tests/test_adaptation.py` |
| Model rollback | `tests/test_rollback.py` |
| Calibration | `tests/test_calibration.py` |
| API endpoints | `tests/test_api.py` |
| Pipeline orchestration | `tests/test_pipeline.py` |

### 4.4 pytest Configuration

**FACT:** `pytest.ini` configures markers and test discovery.
[CFG: pytest.ini]

---

## 5. Audit and Reproducibility Scripts

### 5.1 Pipeline Run Evidence

**FACT:** 60 pipeline result JSON files in `logs/pipeline/` and 32 artifact snapshots in `artifacts/`.
[ART: logs/pipeline/pipeline_result_*.json]

### 5.2 Repository Verification

Scripts in `scripts/` provide audit capabilities:
- `scripts/audit_artifacts.py` — 12-check artifact integrity verification
- `scripts/verify_repository.py` — Repository structure validation

---

## 6. Critical Findings

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| A-1 | `inference_smoke.py` missing — CI integration test will fail | **HIGH** | [CODE: .github/workflows/ci-cd.yml:integration-test] |
| A-2 | Model validation job is 3 echo stubs | **HIGH** | [CODE: .github/workflows/ci-cd.yml:model-validation] |
| A-3 | Docker entry point may mismatch (`api.main` vs `src.api.app`) | **MEDIUM** | [CODE: docker/Dockerfile.inference:CMD] |
| A-4 | No `on.schedule` for automated drift checks | **MEDIUM** | CI/CD only triggers on push/PR/manual |
| A-5 | 215 tests across 19 files — substantial coverage | **STRENGTH** | [ART: tests/] |
| A-6 | CORS wide open (`*`) in API | **LOW** | Acceptable for thesis demo |
| A-7 | Embedded HTML dashboard is thesis-ready demo | **STRENGTH** | [CODE: src/api/app.py] |

---

## 7. Recommendations for Thesis

1. **Create `inference_smoke.py`**: Simple script that hits `/api/health` and `/api/upload` with a test CSV
2. **Add `on.schedule`**: Weekly or daily cron for model validation job
3. **Implement model validation**: Replace echo stubs with actual model loading + baseline drift check
4. **Docker entry point**: Verify `docker/api/main.py` imports from `src.api.app` correctly
5. **Test coverage report**: Generate and include coverage table in thesis appendix

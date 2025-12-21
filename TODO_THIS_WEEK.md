# TODO — This Week (Dec 17–23, 2025)

Goal: finish CI/tests and baseline drift detection so pipeline is automatable and monitored.

---

## Top priorities

- [ ] Add GitHub Actions CI workflow
  - Path: `.github/workflows/ci.yml`
  - What: lint (pylint), install deps, run `pytest tests/`, build `docker/Dockerfile.inference`
  - Why: automated CI for reproducible deployments (see `WHAT_TO_DO_NEXT.md`)
  - Acceptance: workflow file exists, committed, and triggers on push to `main`.

- [ ] Add unit tests
  - Files: `tests/conftest.py`, `tests/test_preprocessing.py`, `tests/test_inference.py`
  - What: tests for unit conversion (milliG -> m/s²), windowing, sensor fusion, model IO shapes
  - Why: ensure pipeline correctness and enable CI
  - Acceptance: `pytest tests/` finds and runs tests without syntax errors

- [ ] Run tests locally & integrate into CI
  - Commands:
    ```powershell
    pip install -r config/requirements.txt pytest pytest-cov
    pytest tests/ -v --cov=src --cov-report=xml
    ```
  - Acceptance: tests execute and produce coverage report; CI uses same command

---

## Secondary priorities

- [ ] Implement `DriftDetector` and `monitor_drift.py`
  - Files: `src/data_validator.py`, `src/monitor_drift.py`
  - What: KS-test + Wasserstein distance; save `logs/drift/drift_report_<ts>.json`
  - Why: detect domain shift (ICTH_16, Domain Adaptation survey)
  - Acceptance: running `python src/monitor_drift.py --data data/production/latest.csv --reference data/prepared/reference_stats.json` writes JSON report

- [ ] Compute and save `data/prepared/reference_stats.json`
  - What: compute mean/std/min/max/percentiles for each sensor channel from training windows
  - Why: reference distribution for drift detection
  - Acceptance: JSON file contains statistics for `Ax,Ay,Az,Gx,Gy,Gz`

- [ ] Add structured logging & exceptions
  - Files: `src/core/logger.py`, `src/core/exception.py`
  - What: rotating file logs + console, `PipelineException` with file/line info
  - Why: better observability and debugging
  - Acceptance: logs created at `logs/pipeline/*.log` and exceptions show file/line

---

## Nice-to-have (if time)

- [ ] Start refactor: `src/components/model_inference.py` and reduce `run_inference.py` to an orchestrator
- [ ] Add CI badge to `README.md`

---

## Papers to consult this week

- "Domain Adaptation for IMU-based HAR: A Survey" — drift detection techniques
- ICTH_16 (Recognition of Anxiety-Related Activities) — lab-to-life gap, fine-tuning
- "MLOps: A Survey" — CI/CD and retraining strategy
- "Enabling End-To-End Machine Learning Replicability" — Docker + CI best practices

---

## Quick commands (Windows PowerShell)

```powershell
# Run tests
pip install -r config/requirements.txt pytest pytest-cov
pytest tests/ -v --cov=src

# Run drift monitor (example)
python src/monitor_drift.py --data data/processed/sensor_fused_50Hz.csv --reference data/prepared/reference_stats.json --output logs/drift/
```

---

*Generated: December 17, 2025*

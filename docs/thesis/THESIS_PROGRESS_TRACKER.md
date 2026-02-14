# Thesis Progress Tracker

> **Master Thesis:** MLOps-Enhanced Human Activity Recognition for Anxiety Detection Using Wearable Sensors  
> **Author:** Shalin Vachheta  
> **Snapshot Date:** February 12, 2026  
> **Deadline:** May 20, 2026 (14 weeks remaining)

---

## Headline Scores

| Metric | Score | Description |
|--------|-------|-------------|
| **Engineering Readiness Score (ERS)** | **83%** | Weighted average of system infrastructure and experiment readiness |
| **Thesis Readiness Score (TRS)** | **37%** | Weighted composite reflecting submission readiness |

### TRS Equation

$$\text{TRS} = 0.45 \times W + 0.35 \times E + 0.20 \times S$$

Where:

| Variable | Meaning | Current Value |
|----------|---------|---------------|
| $W$ | **Writing** — thesis chapters 1–6 drafted and reviewed | 13% |
| $E$ | **Experiments** — formal results generated, plots created, MLflow exports saved | 45% |
| $S$ | **System** — pipeline runs end-to-end, monitoring deployed, CI green | 83% |

$$\text{TRS} = 0.45 \times 0.13 + 0.35 \times 0.45 + 0.20 \times 0.83 = 0.0585 + 0.1575 + 0.166 \approx 0.38$$

### ERS Breakdown

| Component | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Pipeline (14 stages implemented) | 0.30 | 90% | 27.0% |
| CI/CD (GitHub Actions, Docker) | 0.20 | 78% | 15.6% |
| Monitoring (Prometheus, Grafana, drift) | 0.20 | 75% | 15.0% |
| Testing (18 test files, pytest) | 0.15 | 85% | 12.8% |
| Documentation (guides, architecture) | 0.15 | 90% | 13.5% |
| **Total ERS** | **1.00** | — | **83.9%** |

---

## Definition of Done — CRITICAL Tasks

Every item below must be checked before thesis submission. This is the acceptance criteria.

### Writing (Chapters 1–6)

- [ ] **Ch 1 — Introduction:** Problem statement, objectives, scope, contributions (≥8 pages)
- [ ] **Ch 2 — Literature Review:** HAR survey, MLOps landscape, domain adaptation, UQ (≥20 pages)
- [ ] **Ch 3 — Methodology:** Pipeline design, 14-stage architecture, data flow diagrams (≥15 pages)
- [ ] **Ch 4 — Implementation:** Code walkthrough, config, deployment, CI/CD, Docker (≥20 pages)
- [ ] **Ch 5 — Results & Evaluation:** MLflow metrics tables, confusion matrices, calibration plots, drift analysis, robustness results (≥15 pages)
- [ ] **Ch 6 — Discussion & Future Work:** Limitations, comparison with related work, future directions (≥10 pages)
- [ ] **Abstract** written (≤300 words)
- [ ] **Bibliography** complete and formatted (BibTeX)

### Experiments & Results

- [ ] Full pipeline executed on all 26 Decoded sessions
- [ ] MLflow experiment results exported (metrics CSV + screenshots)
- [ ] Confusion matrix plots generated for all folds
- [ ] Calibration reliability diagrams generated (ECE, temperature scaling)
- [ ] Wasserstein drift detection plots saved
- [ ] Robustness test results (noise injection, missing data) tabulated
- [ ] Curriculum pseudo-labeling convergence plots
- [ ] Sensor placement axis-mirroring accuracy comparison
- [ ] Training curves (loss + accuracy per epoch) saved
- [ ] Per-class F1 scores table with confidence intervals

### System & Infrastructure

- [ ] Prometheus + Grafana deployed locally and screenshots captured
- [ ] `docker-compose up` runs successfully end-to-end
- [ ] GitHub Actions CI passes on push to main
- [ ] All 18 test files pass (`pytest tests/ -v`)
- [ ] Repository cleaned — research files separated from production
- [ ] `config/requirements.txt` merge conflicts resolved
- [ ] DVC data pulled and verified
- [ ] MLflow UI accessible with experiment history

### Final Deliverables

- [ ] Thesis PDF compiled (LaTeX or Word)
- [ ] Presentation slides prepared (≥20 slides)
- [ ] Code repository tagged as `v3.0-submission`
- [ ] Reproducibility README updated with exact steps

---

## Month-by-Month Progress Summary

| Month | Focus | Planned | Actual | % |
|-------|-------|---------|--------|---|
| 1 (Oct 2025) | Data Ingestion & Preprocessing | Ingestion, preprocessing, pipeline setup | sensor_data_pipeline.py, preprocess_data.py, DVC | 95% |
| 2 (Nov 2025) | Training & Versioning | Training loop, MLflow, model registry | train.py, mlflow_tracking.py, model_rollback.py | 82% |
| 3 (Dec 2025) | CI/CD & Deployment | GitHub Actions, Docker, FastAPI | ci-cd.yml, Dockerfiles, api/main.py | 71% |
| 4 (Jan 2026) | Monitoring & Integration | Drift detection, Prometheus, Grafana, alerts | wasserstein_drift.py, prometheus_metrics.py, configs | 74% |
| 5 (Feb 2026) | Refinement & Retraining | Calibration, pseudo-labeling, robustness | calibration.py, curriculum_pseudo_labeling.py, robustness.py | 87% |
| 6 (Mar–Apr 2026) | Thesis Writing | Chapters 1–6, results, presentation | Structure outline only | 15% |

---

## Remaining Tasks — Ranked by Priority

### CRITICAL

| # | Task | Effort | Guide |
|---|------|--------|-------|
| 1 | Write thesis chapters 1–6 | 6–8 weeks | [THESIS_STRUCTURE_OUTLINE.md](THESIS_STRUCTURE_OUTLINE.md) |
| 2 | Generate formal results and figures | 1 week | `scripts/generate_thesis_figures.py` |
| 3 | Final presentation slides | 1 week | — |

### HIGH

| # | Task | Effort | Guide |
|---|------|--------|-------|
| 4 | Deploy Prometheus + Grafana, capture screenshots | 2–3 hrs | `config/prometheus.yml`, `docker-compose.yml` |
| 5 | Run all 26 Decoded sessions through pipeline | 2–3 hrs | [DATA_INGESTION_AND_INFERENCE_GUIDE.md](../DATA_INGESTION_AND_INFERENCE_GUIDE.md) |
| 6 | Verify GitHub Actions CI passes | 1 hr | [GITHUB_ACTIONS_CICD_GUIDE.md](../GITHUB_ACTIONS_CICD_GUIDE.md) |
| 7 | Repository cleanup (research vs production) | 2–3 hrs | [PIPELINE_OPERATIONS_AND_ARCHITECTURE.md](../PIPELINE_OPERATIONS_AND_ARCHITECTURE.md) §7 |
| 8 | Resolve `config/requirements.txt` conflicts | 15 min | Delete conflicting sections; `pyproject.toml` is source of truth |

### MEDIUM

| # | Task | Effort | Guide |
|---|------|--------|-------|
| 9 | Automated hyperparameter search (Optuna) | 1 day | — |
| 10 | Self-hosted GitHub Actions runner | 2–3 hrs | [GITHUB_ACTIONS_CICD_GUIDE.md](../GITHUB_ACTIONS_CICD_GUIDE.md) §13 |
| 11 | CI pipeline smoke test job | 1 hr | CI workflow |
| 12 | Model validation job (replace placeholder) | 1 hr | CI workflow line 230 |
| 13 | File watcher for auto-ingestion | 1 hr | [DATA_INGESTION_AND_INFERENCE_GUIDE.md](../DATA_INGESTION_AND_INFERENCE_GUIDE.md) §4 |
| 14 | Processed-session registry (JSON) | 2 hrs | [PIPELINE_OPERATIONS_AND_ARCHITECTURE.md](../PIPELINE_OPERATIONS_AND_ARCHITECTURE.md) §1.4 |

---

## 14-Week Plan (Feb 12 → May 20)

| Weeks | Focus | Deliverables |
|-------|-------|-------------|
| 1–2 (Feb 12–25) | Pipeline execution & results | Run 26 sessions; deploy monitoring; capture screenshots; resolve conflicts |
| 3–4 (Feb 26 – Mar 11) | Thesis Ch 1–2 | Introduction + Literature Review |
| 5–6 (Mar 12–25) | Thesis Ch 3 | Methodology (pipeline design, architecture diagrams) |
| 7–8 (Mar 26 – Apr 8) | Thesis Ch 4 | Implementation (code walkthrough, deployment) |
| 9–10 (Apr 9–22) | Thesis Ch 5 | Results & Evaluation (metrics, plots, analysis) |
| 11–12 (Apr 23 – May 6) | Thesis Ch 6 + cleanup | Discussion, future work, repo cleanup |
| 13–14 (May 7–20) | Final review | Proofread, format, slides, submit |

---

## Automated Dashboard

> This file is the **reference source** for progress tracking.  
> Run `python scripts/update_progress_dashboard.py` to generate a compact dashboard from `docs/thesis/progress.yaml`.  
> The generated dashboard is saved to `docs/thesis/THESIS_PROGRESS_DASHBOARD.md`.

---

*This document is part of the Master Thesis: "MLOps-Enhanced Human Activity Recognition for Anxiety Detection Using Wearable Sensors."*

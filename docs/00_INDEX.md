# Documentation Index
> Single entrypoint for all project documentation.  
> Reorganized: 2026-03-05

---

## 01 — Pipeline

Architecture, stages, and flow documentation for the HAR MLOps pipeline.

| Document | Description |
|----------|-------------|
| [stages/00_STAGE_INDEX.md](01_PIPELINE/stages/00_STAGE_INDEX.md) | Stage-by-stage index (Stages 1–10) |
| [thesis-pipeline-why-what-how.md](01_PIPELINE/thesis-pipeline-why-what-how.md) | Pipeline rationale |
| [PIPELINE_FACTSHEET.md](01_PIPELINE/PIPELINE_FACTSHEET.md) | Quick-reference pipeline summary |
| [PIPELINE_GAPS.md](01_PIPELINE/PIPELINE_GAPS.md) | Known gaps and missing pieces |
| [PIPELINE_CTO_REVIEW.md](01_PIPELINE/PIPELINE_CTO_REVIEW.md) | CTO-style code review |
| [solution.md](01_PIPELINE/solution.md) | Solution overview |
| [WHY_BY_STAGE.md](01_PIPELINE/WHY_BY_STAGE.md) | Why each stage exists |
| [WHY_BY_FILE.md](01_PIPELINE/WHY_BY_FILE.md) | File-level justifications |

**Stage docs:** See [stages/](01_PIPELINE/stages/) for per-stage details (01–10).

---

## 02 — Technology

Tool-specific explanations: Docker, DVC, MLflow, FastAPI, CI/CD, monitoring.

| Document | Description |
|----------|-------------|
| [DOCKER_EXPLAINED.md](02_TECH/DOCKER_EXPLAINED.md) | Docker setup and rationale |
| [DVC_EXPLAINED.md](02_TECH/DVC_EXPLAINED.md) | DVC data versioning |
| [MLFLOW_EXPLAINED.md](02_TECH/MLFLOW_EXPLAINED.md) | MLflow experiment tracking |
| [FASTAPI_EXPLAINED.md](02_TECH/FASTAPI_EXPLAINED.md) | FastAPI inference server |
| [CICD_EXPLAINED.md](02_TECH/CICD_EXPLAINED.md) | CI/CD with GitHub Actions |
| [PROMETHEUS_EXPLAINED.md](02_TECH/PROMETHEUS_EXPLAINED.md) | Prometheus monitoring |
| [GRAFANA_EXPLAINED.md](02_TECH/GRAFANA_EXPLAINED.md) | Grafana dashboards |
| [MONITORING_EXPLAINED.md](02_TECH/MONITORING_EXPLAINED.md) | Monitoring overview |
| [MODEL_REGISTRY_EXPLAINED.md](02_TECH/MODEL_REGISTRY_EXPLAINED.md) | Model registry |
| [EXPERIMENT_TRACKING_EXPLAINED.md](02_TECH/EXPERIMENT_TRACKING_EXPLAINED.md) | Experiment tracking design |
| [tech.md](02_TECH/tech.md) | Technology stack summary |

**Guides:** See [guides/](02_TECH/guides/) for step-by-step technical guides.  
**Cheat sheets:** See [cheat_sheet/](02_TECH/cheat_sheet/) for quick-reference PDFs.

---

## 03 — Experiments

Results, metrics, calibration, and evidence.

| Document | Description |
|----------|-------------|
| [EVIDENCE_LEDGER.md](03_EXPERIMENTS/EVIDENCE_LEDGER.md) | Evidence tracking ledger |
| [EVIDENCE_PACK_INDEX.md](03_EXPERIMENTS/EVIDENCE_PACK_INDEX.md) | Evidence pack index |
| [normalization_analysis.md](03_EXPERIMENTS/normalization_analysis.md) | Normalization strategy analysis |
| [THRESHOLD_CALIBRATION_SUMMARY.md](03_EXPERIMENTS/THRESHOLD_CALIBRATION_SUMMARY.md) | Threshold calibration results |
| [TRIGGER_POLICY_EVAL.md](03_EXPERIMENTS/TRIGGER_POLICY_EVAL.md) | Trigger policy comparison |
| [WINDOWING_JUSTIFICATION.md](03_EXPERIMENTS/WINDOWING_JUSTIFICATION.md) | Window size justification |

**Raw data:** CSV/PNG/JSON experiment outputs remain in `reports/`.

---

## 04 — Thesis Writing

Chapter drafts, research notes, mentor questions, and defense preparation.

| Document | Description |
|----------|-------------|
| [THESIS_STRUCTURE_OUTLINE.md](04_THESIS_WRITING/THESIS_STRUCTURE_OUTLINE.md) | Chapter outline |
| [chapter_drafts/](04_THESIS_WRITING/chapter_drafts/) | MD drafts of chapters 1, 3, 4 |
| [research/](04_THESIS_WRITING/research/) | Paper analyses and bibliography |
| [DEFENSE_QA.md](04_THESIS_WRITING/DEFENSE_QA.md) | Defense Q&A preparation |
| [mentor_questions_explained.md](04_THESIS_WRITING/mentor_questions_explained.md) | Mentor questions with answers |
| [THESIS_PARAMETER_CITATIONS.md](04_THESIS_WRITING/THESIS_PARAMETER_CITATIONS.md) | Parameter choices with citations |
| [THESIS_PROBLEMS_WE_FACED.md](04_THESIS_WRITING/THESIS_PROBLEMS_WE_FACED.md) | Problems encountered |
| [todo/](04_THESIS_WRITING/todo/) | Remaining work and questions |

**LaTeX source:** See `thesis/` for actual .tex files.

---

## 99 — Archive

Old date-based docs, progress logs, and superseded content. Nothing deleted.

| Folder | Content |
|--------|---------|
| [2026-01/](99_ARCHIVE/2026-01/) | Jan 2026 planning docs and paper analyses |
| [2026-02/](99_ARCHIVE/2026-02/) | Feb 2026 progress dashboards and docs |
| [2026-03/](99_ARCHIVE/2026-03/) | Mar 2026 "Fourth March" snapshot (duplicates) |
| [thesis_archive/](99_ARCHIVE/thesis_archive/) | Old thesis progress and audit docs |

---

## Other Key Locations

| Path | Description |
|------|-------------|
| `thesis/` | LaTeX source (chapters, appendices, frontmatter, figures, refs) |
| `src/` | Python library modules |
| `scripts/` | CLI entrypoint scripts |
| `reports/repo_audit/` | Audit reports (SRC_SCRIPTS_AUDIT, MOVE_LOG, BROKEN_LINKS) |
| `README.md` | Project overview (root) |

# Documentation Index — What to Read and When

**Project:** HAR MLOps Pipeline v2.1.0  
**Last Updated:** 19 Feb 2026  
**Total docs:** ~70 markdown files across `docs/` + project root

---

## How to Use This Index

Instead of opening every file, use this guide to find exactly what you need:

| I need to… | Read this |
|------------|-----------|
| **Run the pipeline** | [docs/PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md) |
| **Understand what's done / not done** | [docs/REMAINING_WORK_FEB_TO_MAY_2026.md](REMAINING_WORK_FEB_TO_MAY_2026.md) |
| **Write a thesis chapter** | [docs/thesis/THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) |
| **Cite a paper** | [docs/Bibliography_From_Local_PDFs.md](Bibliography_From_Local_PDFs.md) |
| **Debug the pipeline** | [docs/technical/pipeline_audit_map.md](technical/pipeline_audit_map.md) |
| **Understand a specific stage** | [docs/stages/00_STAGE_INDEX.md](stages/00_STAGE_INDEX.md) |
| **Set up CI/CD** | [docs/GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) |
| **Answer a mentor question** | [docs/MENTOR_QA_SIMPLE_WITH_PAPERS.md](MENTOR_QA_SIMPLE_WITH_PAPERS.md) |
| **Clean up disk space** | [docs/REPOSITORY_CLEANUP_RECOMMENDATIONS.md](REPOSITORY_CLEANUP_RECOMMENDATIONS.md) |
| **Delete everything and start fresh** | [docs/technical/FRESH_START_CLEANUP_GUIDE.md](technical/FRESH_START_CLEANUP_GUIDE.md) |
| **See today's work** | [docs/WORK_DONE_19_FEB.md](WORK_DONE_19_FEB.md) |

---

## Project Root Files

| File | Purpose |
|------|---------|
| [README.md](../README.md) | Main project overview — status (95%), quick start, architecture diagram, component links |
| [Thesis_Plan.md](../Thesis_Plan.md) | 6-month thesis timeline (Oct 2025 – Apr 2026), monthly milestones |
| [pyproject.toml](../pyproject.toml) | Project metadata, dependencies, tool config (v2.1.0) |
| [docker-compose.yml](../docker-compose.yml) | Docker services: MLflow, inference, training, preprocessing |
| [run_pipeline.py](../run_pipeline.py) | Main pipeline entry point — 14 stages, 20+ CLI flags |
| [batch_process_all_datasets.py](../batch_process_all_datasets.py) | Batch-process all 26 recording sessions |
| [generate_summary_report.py](../generate_summary_report.py) | Summary report generator after batch runs |
| [cleanup_repo.ps1](../cleanup_repo.ps1) | PowerShell cleanup script (`-DryRun` / `-Aggressive`) |

---

## `docs/` — Operational & Architecture

These are the main reference documents for running and understanding the pipeline.

| File | What It Covers | Read When… |
|------|---------------|------------|
| [PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md) | **Master runbook** — 18 sections: CLI flags, Docker, FastAPI, DVC, MLflow, cleanup, data management, testing, troubleshooting | You want to **run anything** |
| [PIPELINE_OPERATIONS_AND_ARCHITECTURE.md](PIPELINE_OPERATIONS_AND_ARCHITECTURE.md) | Production pipeline ops — auto-detection of new files, adaptation strategies, model versioning, environment setup | You want to understand **how the pipeline works internally** |
| [DATA_INGESTION_AND_INFERENCE_GUIDE.md](DATA_INGESTION_AND_INFERENCE_GUIDE.md) | Step-by-step: ingest Garmin data → run inference → view results in MLflow | You have **new sensor data** to process |
| [MONITORING_AND_RETRAINING_GUIDE.md](MONITORING_AND_RETRAINING_GUIDE.md) | Drift detection equations, confidence thresholds, ECE, pseudo-labeling/AdaBN retraining | You want to understand **monitoring & retraining logic** |
| [PREPROCESSING_COMPARISON_AND_ADAPTATION.md](PREPROCESSING_COMPARISON_AND_ADAPTATION.md) | Compares runs with/without gravity removal, explains AdaBN & pseudo-label adaptation | You want to **compare preprocessing options** |
| [TRAINING_RECIPE_MATRIX.md](TRAINING_RECIPE_MATRIX.md) | Two-dataset training story (ADAMSense → wrist), 4 experiment recipes (T1–T4) | You want to understand **training experiments** |
| [MANUAL_PREPROCESSING_TEST.md](MANUAL_PREPROCESSING_TEST.md) | Manual test of preprocessing on 10 sessions with/without gravity removal | You want to **manually test preprocessing** |

---

## `docs/` — CI/CD & Infrastructure

| File | What It Covers | Read When… |
|------|---------------|------------|
| [GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) | Technical reference: 6 CI/CD jobs, trigger rules, secrets, customization | You need to **modify CI/CD** |
| [GITHUB_ACTIONS_CICD_BEGINNER_GUIDE.md](GITHUB_ACTIONS_CICD_BEGINNER_GUIDE.md) | Beginner tutorial: what is GitHub Actions, step-by-step walkthrough | You are **new to GitHub Actions** |
| [REPOSITORY_CLEANUP_RECOMMENDATIONS.md](REPOSITORY_CLEANUP_RECOMMENDATIONS.md) | Repo size analysis (~12 GB), 2.8 GB reclaimable, priority cleanup commands | Your disk is **running out of space** |

---

## `docs/` — Planning & Status

| File | What It Covers | Read When… |
|------|---------------|------------|
| [REMAINING_WORK_FEB_TO_MAY_2026.md](REMAINING_WORK_FEB_TO_MAY_2026.md) | Month-by-month TODO (Feb–May), 95% pipeline done, thesis writing timeline | You want to **plan the next steps** |
| [WHATS_REMAINING.md](WHATS_REMAINING.md) | High-level checklist: completed (95%), remaining (5%) — audit, ablation, writing | You want a **quick overview** of what's left |
| [ACTION_PLAN_18_20_FEB_2026.md](ACTION_PLAN_18_20_FEB_2026.md) | 5 pipeline issues found Feb 18, step-by-step fixes applied Feb 19 | You want to see **what bugs were fixed** this week |
| [WORK_DONE_19_FEB.md](WORK_DONE_19_FEB.md) | Session log: 4 audit runs (A1/A3/A4/A5), bug fixes, ablation table, git commits | You want to see **exactly what was done on Feb 19** |

---

## `docs/` — Q&A & Mentor Prep

| File | What It Covers | Read When… |
|------|---------------|------------|
| [HAR_MLOps_QnA_With_Papers.md](HAR_MLOps_QnA_With_Papers.md) | 19 Q&A topics: sensor fusion, QC, drift, uncertainty, pseudo-labeling, audit trail | You need **detailed technical answers** |
| [MENTOR_QA_SIMPLE_WITH_PAPERS.md](MENTOR_QA_SIMPLE_WITH_PAPERS.md) | Plain-English answers to thesis questions with paper citations + glossary | You need to **explain things simply to a mentor** |

---

## `docs/` — References & Bibliography

| File | What It Covers | Read When… |
|------|---------------|------------|
| [Bibliography_From_Local_PDFs.md](Bibliography_From_Local_PDFs.md) | 35+ formatted citations organized by topic (HAR, MLOps, domain adaptation, uncertainty) | You are **writing a thesis chapter and need citations** |
| [APPENDIX_PAPER_INDEX.md](APPENDIX_PAPER_INDEX.md) | 30+ papers table with IDs, DOIs, tags, file paths + stage mapping | You need to **find a specific paper** |
| [THESIS_OBJECTIVES_TRACEABILITY.md](THESIS_OBJECTIVES_TRACEABILITY.md) | 13 thesis objectives mapped to pipeline stages, code entry-points, and artifacts | You need to **prove traceability** in the thesis |
| [THESIS_PIPELINE_WHY_WHAT_HOW_WHEN_WHERE.md](THESIS_PIPELINE_WHY_WHAT_HOW_WHEN_WHERE.md) | Comprehensive rationale for 13 stages with architecture diagrams and references | You need to **justify design decisions** |

---

## `docs/stages/` — Per-Stage Documentation

Detailed design docs for each pipeline stage. Start with the index:

| File | Stage | Key Content |
|------|:-----:|-------------|
| [00_STAGE_INDEX.md](stages/00_STAGE_INDEX.md) | — | **Dashboard** — all stages with status and TODO |
| [01_DATA_INGESTION.md](stages/01_DATA_INGESTION.md) | 0 | Naming conventions, DVC vs metadata JSON, raw data org |
| [02_PREPROCESSING_FUSION.md](stages/02_PREPROCESSING_FUSION.md) | 1 | Accel+gyro fusion, 50 Hz resampling, gravity removal, windowing |
| [03_QC_VALIDATION.md](stages/03_QC_VALIDATION.md) | 2 | QC layers (raw/preprocessed/window), thresholds, fail actions |
| [04_TRAINING_BASELINE.md](stages/04_TRAINING_BASELINE.md) | 3 | Frozen training baseline vs rolling reference, 5-fold CV |
| [05_INFERENCE.md](stages/05_INFERENCE.md) | 4 | Inference flow, confidence computation, planned MC Dropout |
| [06_MONITORING_DRIFT.md](stages/06_MONITORING_DRIFT.md) | 5 | Proxy metrics: confidence, entropy, flip rate, KS, PSI |
| [07_EVALUATION_METRICS.md](stages/07_EVALUATION_METRICS.md) | 6 | Full metrics inventory with MLflow keys |
| [08_ALERTING_RETRAINING.md](stages/08_ALERTING_RETRAINING.md) | 7 | Alert tiers (INFO/WARNING/CRITICAL), 2-of-3 voting trigger |
| [09_DEPLOYMENT_AUDIT.md](stages/09_DEPLOYMENT_AUDIT.md) | 8 | Deployment architecture, CI/CD design, audit trail |
| [10_IMPROVEMENTS_ROADMAP.md](stages/10_IMPROVEMENTS_ROADMAP.md) | 9 | Gap analysis and prioritized improvements |

---

## `docs/technical/` — Debug & Audit

| File | What It Covers | Read When… |
|------|---------------|------------|
| [README.md](technical/README.md) | Reading guide with priority ratings (Critical/High/Reference) | Start here for technical docs |
| [pipeline_audit_map.md](technical/pipeline_audit_map.md) | ASCII data-flow diagram: raw → fused → preprocessed → inferred → evaluated | You need a **visual overview** |
| [PIPELINE_VISUALIZATION_CURRENT.md](technical/PIPELINE_VISUALIZATION_CURRENT.md) | High-level ASCII art of the pipeline | Quick **visual reference** |
| [evaluation_audit.md](technical/evaluation_audit.md) | Audits `evaluate_predictions.py` correctness | You suspect **evaluation bugs** |
| [tracking_audit.md](technical/tracking_audit.md) | MLflow + DVC tracking config audit | You suspect **tracking issues** |
| [PIPELINE_TEST_RESULTS.md](technical/PIPELINE_TEST_RESULTS.md) | Test run results from Jan 15, 2026 — data shapes, model outputs | Historical **test results** reference |
| [QC_EXECUTION_SUMMARY.md](technical/QC_EXECUTION_SUMMARY.md) | QC check results confirming correctness (idle data, not bugs) | You want to verify **QC is working** |
| [root_cause_low_accuracy.md](technical/root_cause_low_accuracy.md) | Why 14-15% accuracy? Idle/stationary data, not a bug | You're confused about **low accuracy** |
| [FRESH_START_CLEANUP_GUIDE.md](technical/FRESH_START_CLEANUP_GUIDE.md) | PowerShell commands to **delete everything** and start fresh | You want a **nuclear reset** |
| [PIPELINE_RERUN_GUIDE.md](technical/PIPELINE_RERUN_GUIDE.md) | End-to-end checklist: DVC pull → process → train → infer → evaluate | You want to **rerun everything from scratch** |

---

## `docs/thesis/` — Thesis Writing

| File | What It Covers | Read When… |
|------|---------------|------------|
| [README.md](thesis/README.md) | Reading guide with priority order | Start here for thesis docs |
| [THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) | Chapter outline (6 chapters, 60-80 pages), page budgets, writing timeline | You're **planning** the thesis document |
| [FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md](thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md) | **Master planning doc** (1995 lines) — 95% progress, week-by-week plan | You need the **most comprehensive status** |
| [CONCEPTS_EXPLAINED.md](thesis/CONCEPTS_EXPLAINED.md) | Foundational concepts: milliG, sensor fusion, windowing, domain shift | You're writing **background sections** |
| [FINE_TUNING_STRATEGY.md](thesis/FINE_TUNING_STRATEGY.md) | Fine-tuning policy: problem diagnosis and retraining decisions | You're writing about **training strategy** |
| [HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md](thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md) | Signal asymmetry, axis mirroring, dominant/non-dominant wrist | You're writing about **sensor placement** |
| [UNLABELED_EVALUATION.md](thesis/UNLABELED_EVALUATION.md) | Formal framework for evaluating unlabeled production data | You need to **justify no-label evaluation** |
| [THESIS_READY_UNLABELED_EVALUATION_PLAN.md](thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md) | Scientifically defensible plan for proxy metric evaluation | You need a **thesis-ready evaluation plan** |
| [QA_LAB_TO_LIFE_GAP.md](thesis/QA_LAB_TO_LIFE_GAP.md) | Why 49% in ICTH_16 paper but 14% in practice — lab-to-life gap | You need to **explain accuracy differences** |
| [PIPELINE_REALITY_MAP.md](thesis/PIPELINE_REALITY_MAP.md) | Brutally honest file-by-file inventory with line counts | You want to **verify what actually exists** |

### Thesis Chapter Drafts

| File | Chapter | Status |
|------|---------|--------|
| [thesis/chapters/CH1_INTRODUCTION.md](thesis/chapters/CH1_INTRODUCTION.md) | Chapter 1 — Introduction | Draft: problem statement, 4 research objectives, MLOps framing |
| [thesis/chapters/CH3_METHODOLOGY.md](thesis/chapters/CH3_METHODOLOGY.md) | Chapter 3 — Methodology | Draft: 14-stage pipeline, Inference/Retraining/Advanced cycles |
| [thesis/chapters/CH4_IMPLEMENTATION.md](thesis/chapters/CH4_IMPLEMENTATION.md) | Chapter 4 — Implementation | Draft: layered repo structure, module mapping for 14 stages |

### Copilot Prompt Templates

| File | Purpose |
|------|---------|
| [thesis/COPILOT_EVALUATION_PROMPTS.md](thesis/COPILOT_EVALUATION_PROMPTS.md) | Reusable prompts for validating monitoring equations, EWC, evaluation |
| [thesis/COPILOT_PIPELINE_OPERATIONS_PROMPTS.md](thesis/COPILOT_PIPELINE_OPERATIONS_PROMPTS.md) | Reusable prompts for pipeline operations: batch processing, retraining |

### Production & Technology Reference

| File | Purpose |
|------|---------|
| [thesis/KEEP_Production_Robustness_Guide.md](thesis/KEEP_Production_Robustness_Guide.md) | Production improvements: error handling, retries, health checks, security |
| [thesis/KEEP_Reference_Project_Learnings.md](thesis/KEEP_Reference_Project_Learnings.md) | Comparison with a reference MLOps project, refactoring recommendations |
| [thesis/KEEP_Technology_Stack_Analysis.md](thesis/KEEP_Technology_Stack_Analysis.md) | Full technology analysis with importance ratings |

> **Note:** The 3 files in `docs/thesis/production_reference/` are duplicates of the above and can be safely deleted.

---

## `docs/research/` — Paper Analysis

| File | What It Covers |
|------|---------------|
| [RESEARCH_PAPERS_ANALYSIS.md](research/RESEARCH_PAPERS_ANALYSIS.md) | Deep dive into ADAMSense + RAG papers (the 2 foundational papers) |
| [RESEARCH_PAPER_INSIGHTS.md](research/RESEARCH_PAPER_INSIGHTS.md) | Actionable insights from 76+ papers, key-papers table |
| [KEEP_Research_QA_From_Papers.md](research/KEEP_Research_QA_From_Papers.md) | UDA concepts: MMD, adversarial, AdaBN, autoencoders for HAR |

---

## `docs/patient/` — Decision Framework

| File | What It Covers |
|------|---------------|
| [patient/FINAL_PIPELINE_DECISIONS.md](patient/FINAL_PIPELINE_DECISIONS.md) | Merged decision framework from 20 source docs (~10K lines) |
| [patient/PAPER_DRIVEN_QUESTIONS_MAP.md](patient/PAPER_DRIVEN_QUESTIONS_MAP.md) | Questions from 88 papers grouped by stage, answered vs. needs mentor input |

---

## Cleanup Candidates

These files are duplicated or superseded:

| File | Reason |
|------|--------|
| `docs/thesis/production_reference/KEEP_Production_Robustness_Guide.md` | Duplicate of `thesis/KEEP_Production_Robustness_Guide.md` |
| `docs/thesis/production_reference/KEEP_Reference_Project_Learnings.md` | Duplicate of `thesis/KEEP_Reference_Project_Learnings.md` |
| `docs/thesis/production_reference/KEEP_Technology_Stack_Analysis.md` | Duplicate of `thesis/KEEP_Technology_Stack_Analysis.md` |
| `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md` | Duplicate of `patient/PAPER_DRIVEN_QUESTIONS_MAP.md` |

---

*Generated: 19 Feb 2026*

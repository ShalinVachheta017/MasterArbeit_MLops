# Docs — Navigation Index

**Last updated:** 19 Feb 2026  
**Start here:** [WHATS_NEXT.md](WHATS_NEXT.md) — current priorities and next actions

---

## Folder Structure

```
docs/
├── 19_Feb/      Session records — 19 Feb 2026 (all four audit runs, full day)
├── archive/     Old / superseded / completed plans (prefix OLD_ = safe to ignore)
├── figures/     Diagrams and images for thesis
├── research/    Literature: paper analyses, bibliographies, Q&A from papers
├── stages/      Per-pipeline-stage reference (stage 01 → 10)
├── technical/   How-to guides and operational references (prefix guide-)
└── thesis/      Thesis writing material, chapter drafts, experiments
```

---

## Active files (root)

| File | Purpose |
|------|---------|
| [WHATS_NEXT.md](WHATS_NEXT.md) | **Current priorities** — do this next |

---

## 19_Feb/ — Session records

Everything produced on 19 Feb 2026 (7 commits, both sessions).

| File | Purpose |
|------|---------|
| [WORK_DONE_19_FEB.md](19_Feb/WORK_DONE_19_FEB.md) | Full day log — all 7 commits, all bugs found, lessons |
| [PIPELINE_RUNBOOK.md](19_Feb/PIPELINE_RUNBOOK.md) | 18-section operations guide for the pipeline |
| [DOCUMENTATION_INDEX.md](19_Feb/DOCUMENTATION_INDEX.md) | Index of all 70+ markdown files in the repo |
| [REMAINING_WORK_FEB_TO_MAY_2026.md](19_Feb/REMAINING_WORK_FEB_TO_MAY_2026.md) | Feb–May 2026 thesis timeline |

---

## thesis/ — Thesis writing material

### Write your thesis using these

| File | Use for |
|------|---------|
| [THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) | Chapter structure and outline |
| [thesis-objectives-traceability.md](thesis/thesis-objectives-traceability.md) | RQ → code → commit → artifact traceability |
| [thesis-training-recipe-matrix.md](thesis/thesis-training-recipe-matrix.md) | Ablation table: A1/A3/A4/A5 results |
| [thesis-pipeline-why-what-how.md](thesis/thesis-pipeline-why-what-how.md) | Full why/what/how/when/where for Chapter 3 |
| [UNLABELED_EVALUATION.md](thesis/UNLABELED_EVALUATION.md) | 4-layer monitoring framework — Chapter 4 methodology |
| [THESIS_READY_UNLABELED_EVALUATION_PLAN.md](thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md) | Ready evaluation text for Chapter 4 |
| [HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md](thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md) | Domain shift + limitations — Chapter 5 |
| [QA_LAB_TO_LIFE_GAP.md](thesis/QA_LAB_TO_LIFE_GAP.md) | Lab-to-production gap discussion |
| [CONCEPTS_EXPLAINED.md](thesis/CONCEPTS_EXPLAINED.md) | Background concepts — Chapter 2 |
| [FINE_TUNING_STRATEGY.md](thesis/FINE_TUNING_STRATEGY.md) | Model adaptation approach |
| [PIPELINE_REALITY_MAP.md](thesis/PIPELINE_REALITY_MAP.md) | What actually works vs. planned |
| [PAPER_DRIVEN_QUESTIONS_MAP.md](thesis/PAPER_DRIVEN_QUESTIONS_MAP.md) | Research questions from literature |

### Reference / supporting (KEEP_*)

| File | Use for |
|------|---------|
| [KEEP_Technology_Stack_Analysis.md](thesis/KEEP_Technology_Stack_Analysis.md) | Tool selection rationale |
| [KEEP_Production_Robustness_Guide.md](thesis/KEEP_Production_Robustness_Guide.md) | Production best practices |
| [KEEP_Reference_Project_Learnings.md](thesis/KEEP_Reference_Project_Learnings.md) | Lessons from reference projects |

### Chapter drafts (thesis/chapters/)

| File | Status |
|------|--------|
| [CH1_INTRODUCTION.md](thesis/chapters/CH1_INTRODUCTION.md) | Draft |
| [CH3_METHODOLOGY.md](thesis/chapters/CH3_METHODOLOGY.md) | Draft |
| [CH4_IMPLEMENTATION.md](thesis/chapters/CH4_IMPLEMENTATION.md) | Draft |

---

## research/ — Literature

| File | Content |
|------|---------|
| [RESEARCH_PAPERS_ANALYSIS.md](research/RESEARCH_PAPERS_ANALYSIS.md) | Deep analysis: ICTH_16, EHB_2025_71, ADAMSense |
| [RESEARCH_PAPER_INSIGHTS.md](research/RESEARCH_PAPER_INSIGHTS.md) | Cross-paper synthesis |
| [KEEP_Research_QA_From_Papers.md](research/KEEP_Research_QA_From_Papers.md) | UDA/retraining/MLOps Q&A from literature |
| [qna-har-mlops-papers.md](research/qna-har-mlops-papers.md) | HAR MLOps Q&A with paper references |
| [qna-mentor-simple-papers.md](research/qna-mentor-simple-papers.md) | Mentor Q&A — simple answers with evidence |
| [appendix-paper-index.md](research/appendix-paper-index.md) | Paper index, stage-wise paper map |
| [bibliography-local-pdfs.md](research/bibliography-local-pdfs.md) | Bibliography from local PDFs |

---

## technical/ — How-to guides

All current guides use the `guide-*` naming convention.

| File | Content |
|------|---------|
| [guide-pipeline-operations-architecture.md](technical/guide-pipeline-operations-architecture.md) | Pipeline architecture overview |
| [guide-data-ingestion-inference.md](technical/guide-data-ingestion-inference.md) | Ingest sensor data → run inference → view in MLflow |
| [guide-monitoring-retraining.md](technical/guide-monitoring-retraining.md) | Monitoring & retraining operations |
| [guide-preprocessing-comparison-adaptation.md](technical/guide-preprocessing-comparison-adaptation.md) | Preprocessing comparison and domain adaptation |
| [guide-cicd-github-actions.md](technical/guide-cicd-github-actions.md) | GitHub Actions CI/CD — step by step |
| [guide-cicd-beginner.md](technical/guide-cicd-beginner.md) | CI/CD beginner guide |
| [guide-manual-preprocessing-test.md](technical/guide-manual-preprocessing-test.md) | Manual preprocessing test procedure |
| [guide-pipeline-rerun.md](technical/guide-pipeline-rerun.md) | Pipeline reproduction steps |
| [guide-pipeline-final-decisions.md](technical/guide-pipeline-final-decisions.md) | Key design decisions and rationale |

---

## stages/ — Per-stage reference

One file per pipeline stage (stage 00 = index):

```
00_STAGE_INDEX.md          Stage overview and dependencies
01_DATA_INGESTION.md       Stage 1: raw Garmin CSV → fused CSV
02_PREPROCESSING_FUSION.md Stage 2: windowing, unit conversion, gravity removal
03_QC_VALIDATION.md        Stage 3: schema checks, range validation
04_TRAINING_BASELINE.md    Stage 4: training baseline computation
05_INFERENCE.md            Stage 5: 1D-CNN-BiLSTM predictions + confidence
06_MONITORING_DRIFT.md     Stage 6: 3-layer monitoring (confidence, drift, ECE)
07_EVALUATION_METRICS.md   Stage 7: accuracy, F1, confusion matrix
08_ALERTING_RETRAINING.md  Stage 8–10: trigger → retrain → register
09_DEPLOYMENT_AUDIT.md     Stage 11–14: calibration, Wasserstein, curriculum, sensor
10_IMPROVEMENTS_ROADMAP.md Future improvements
```

---

## archive/ — Old / superseded

Files prefixed `OLD_` are safe to ignore — they document completed work or superseded plans. Other files (no prefix) are historical reference that may still have useful background context.

| Prefix | Meaning |
|--------|---------|
| `OLD_*` | Superseded or completed; do not use as source of truth |
| (no prefix) | Historical; may have useful background context |

---

## Quick chapter mapping

| Thesis chapter | Primary source |
|----------------|---------------|
| Ch 1 — Introduction | [thesis/chapters/CH1_INTRODUCTION.md](thesis/chapters/CH1_INTRODUCTION.md) |
| Ch 2 — Background | [thesis/CONCEPTS_EXPLAINED.md](thesis/CONCEPTS_EXPLAINED.md) + [research/RESEARCH_PAPERS_ANALYSIS.md](research/RESEARCH_PAPERS_ANALYSIS.md) |
| Ch 3 — Methodology | [thesis/thesis-pipeline-why-what-how.md](thesis/thesis-pipeline-why-what-how.md) + [stages/](stages/00_STAGE_INDEX.md) |
| Ch 4 — Experiments | [thesis/thesis-training-recipe-matrix.md](thesis/thesis-training-recipe-matrix.md) + [thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md](thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md) |
| Ch 5 — Discussion | [thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md](thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md) + [thesis/QA_LAB_TO_LIFE_GAP.md](thesis/QA_LAB_TO_LIFE_GAP.md) |
| Appendix | [thesis/thesis-objectives-traceability.md](thesis/thesis-objectives-traceability.md) + [research/appendix-paper-index.md](research/appendix-paper-index.md) |

# Remaining Work — February to May 2026

**Project:** HAR MLOps Pipeline (Master Thesis)  
**Author:** Shalin Vachheta  
**Created:** 19 Feb 2026  
**Pipeline Completion:** ~80–82% *(revised 22 Feb 2026 after cross-code audit; earlier estimate of 95% was over-stated)*  
**Thesis Writing Completion:** ~5%

---

## Status Summary — What Is Done

### Pipeline & Code (✅ ~80–82% Complete)

| Component | Status | Notes |
|-----------|:------:|-------|
| 14-stage pipeline (`run_pipeline.py`) | ✅ Done | Stages 1–14 all working |
| Data ingestion (Garmin → fused CSV) | ✅ Done | 26 sessions, 77 files |
| Schema validation & QC | ✅ Done | Range checks, missing ratio |
| Preprocessing & windowing | ✅ Done | 200-sample windows, 50% overlap |
| 1D-CNN-BiLSTM model (499K params) | ✅ Done | 11 activity classes |
| Inference pipeline | ✅ Done | Predictions + confidence |
| 3-layer monitoring (conf + drift + ECE) | ✅ Done | Baseline comparison |
| Retraining trigger logic | ✅ Done | Auto & manual triggers |
| Supervised retraining | ✅ Done | 5-fold CV, val_acc 0.969 |
| AdaBN adaptation | ✅ Done | BN stats update |
| TENT adaptation | ✅ Done | Entropy minimization + OOD guard |
| AdaBN+TENT combined | ✅ Done | Conf 0.90→0.76, working |
| Pseudo-label (calibrated) | ✅ Done | Temp-scaled, entropy-gated, class-balanced |
| Calibration (temperature + MC Dropout) | ✅ Done | Stage 11 |
| Wasserstein drift detection | ✅ Done | Stage 12, change-point detection |
| Curriculum pseudo-labeling + EWC | ✅ Done | Stage 13 |
| Sensor placement analysis | ✅ Done | Stage 14, axis-mirror augmentation |
| MLflow integration | ✅ Done | Logging, model registry |
| DVC data tracking | ✅ Done | 4 .dvc files, ~228 MB |
| FastAPI inference API | ✅ Done | 4 endpoints (`/`, `/api/health`, `/api/model/info`, `/api/upload`) |
| Docker Compose (4 services) | ✅ Done | MLflow, inference, training, preprocessing |
| GitHub Actions CI/CD | ✅ Done | 7 jobs (lint, test, test-slow, build, integration-test, model-validation, notify), push/PR/cron triggers |
| Unit + integration tests | ✅ Done | pytest with markers |
| Artifact audit system | ✅ Done | 4 audit runs, all PASS |

### Audit Runs Completed (19 Feb 2026)

| Audit | Config | Stages | Result | Key Metric |
|-------|--------|:------:|:------:|------------|
| A1 Inference | `--stages inference` | 7 | ✅ 8/8 | 1027 predictions |
| A3 Supervised | `--retrain --adapt none` | 9 | ✅ 12/12 | F1 = 0.814 |
| A4 AdaBN+TENT | `--retrain --adapt adabn_tent` | 9 | ✅ 12/12 | Conf 0.90→0.76 |
| A5 Pseudo-label | `--retrain --adapt pseudo_label` | 9 | ✅ 12/12 | val_acc = 0.969 |

### What Is NOT Done (5% Remaining)

| Component | Status | Priority |
|-----------|:------:|:--------:|
| Thesis document (writing) | ❌ 5% | 🔴 Critical |
| Prometheus live integration | ❌ 0% | 🟡 Optional |
| Grafana live dashboards | ❌ 0% | 🟡 Optional |
| A2 audit (comparison run) | ❌ 0% | 🟢 Nice-to-have |
| Cross-dataset comparison report | ❌ 0% | 🟡 Medium |
| Summary figures for thesis | ⚠️ Partial | 🔴 Critical |

---

## February 2026 — Remaining (19–28 Feb)

### Week of 19–23 Feb

- [x] ~~All pipeline bugs fixed (Phases 1–3)~~
- [x] ~~4 audit runs (A1/A3/A4/A5) — all PASS~~
- [x] ~~WORK_DONE_19_FEB.md created~~
- [x] ~~Pipeline Runbook created~~
- [ ] **Cross-dataset comparison run**: Use `batch_process_all_datasets.py` on all 26 sessions
- [ ] **Drift analysis across datasets**: `python scripts/analyze_drift_across_datasets.py`
- [ ] **Generate thesis figures**: `python scripts/generate_thesis_figures.py`
- [ ] **A2 Audit Run** (optional): Baseline vs. retrained model comparison
- [ ] **Start Chapter 3 outline**: Methodology / pipeline architecture

### Week of 24–28 Feb

- [ ] **Chapter 3 draft**: System Architecture & Pipeline Design
  - 14-stage pipeline description
  - Data flow diagram (ingest → predict → monitor → retrain)
  - Technology stack table
- [ ] **Chapter 4 draft**: Implementation Details
  - Model architecture (1D-CNN-BiLSTM)
  - Domain adaptation methods (AdaBN, TENT, pseudo-label)
  - MLOps infrastructure (DVC, MLflow, Docker, CI/CD)
- [ ] **Review existing docs** for thesis-reusable content (look at `docs/thesis/`)
- [ ] **Prometheus + Grafana**: Decide if wiring them into docker-compose is worth the effort

---

## March 2026 — Core Thesis Writing

### Week of 3–7 Mar

- [ ] **Chapter 5 draft**: Experimental Evaluation
  - Ablation table (all 4 audit runs)
  - Per-dataset inference results (26 sessions)
  - Drift detection effectiveness
  - Domain adaptation comparison table
- [ ] **Chapter 5 figures**: Confusion matrices, confidence distributions, drift plots
- [ ] **Run final experiments** if any gaps exist

### Week of 10–14 Mar

- [ ] **Chapter 1 draft**: Introduction
  - Motivation (wearable HAR in clinical + daily life)
  - Problem statement (domain shift, continuous learning)
  - Research questions
  - Thesis outline
- [ ] **Chapter 2 draft**: Related Work / Background
  - HAR overview (existing methods)
  - MLOps principles
  - Domain adaptation literature
  - Integrate content from `docs/research/*.md` and `papers/`

### Week of 17–21 Mar

- [ ] **Chapter 6 draft**: Discussion
  - Interpret results from Chapter 5
  - Limitations
  - Comparison with literature
  - Lessons learned
- [ ] **Review all chapters** for consistency

### Week of 24–28 Mar

- [ ] **Chapter 7 draft**: Conclusion & Future Work
  - Summary of contributions
  - Future work: real-time deployment, more sensors, online learning
- [ ] **Abstract** and **Zusammenfassung** (German abstract)

---

## April 2026 — Polish & Integration

### Week of 31 Mar – 4 Apr

- [ ] **Integrate all chapters** into single LaTeX/Word document
- [ ] **Bibliography**: Ensure all references from `docs/Bibliography_From_Local_PDFs.md` are properly cited
- [ ] **Add appendices**:
  - Appendix A: Full pipeline configuration (`config/pipeline_config.yaml`)
  - Appendix B: API specification (FastAPI endpoints)
  - Appendix C: CI/CD workflow diagram
  - Appendix D: Dataset overview (26 sessions, data collection protocol)

### Week of 7–11 Apr

- [ ] **Figures & tables review**: All figures in correct format (vector if possible)
- [ ] **Grammar & language review** (English academic style)
- [ ] **Prometheus/Grafana** (optional): If included, add 1 page to Chapter 4 + screenshots

### Week of 14–18 Apr

- [ ] **Self-review**: Read entire thesis end-to-end
- [ ] **Peer review**: Send to a colleague or friend for feedback
- [ ] **Mentor check-in**: Submit draft to thesis advisor for feedback

### Week of 21–25 Apr

- [ ] **Incorporate feedback** from advisor
- [ ] **Final figures** and formatting

---

## May 2026 — Final Submission

### Week of 28 Apr – 2 May

- [ ] **Final proofreading**
- [ ] **Check formatting requirements** (university template, margins, font)
- [ ] **Generate PDF**: Compile final version
- [ ] **Print** (if physical copy required)

### Week of 5–9 May

- [ ] **Submit thesis** 🎓
- [ ] **Prepare presentation** slides if defense is required
- [ ] **Clean up repo**: Archive personal notes, tag final release `v3.0.0`

### Week of 12–16 May

- [ ] **Thesis defense / presentation** (if scheduled)
- [ ] **Push final code release** with DOI if applicable

---

## Monitoring & Comparison — Analysis TODO

These are specific analysis tasks that feed into Chapter 5:

### Cross-Dataset Comparison

```powershell
# Run all 26 datasets through the pipeline
python batch_process_all_datasets.py

# Analyze drift patterns across datasets
python scripts/analyze_drift_across_datasets.py

# Generate summary comparison report
python generate_summary_report.py
```

Expected outputs:
- [ ] Per-dataset prediction confidence table
- [ ] Drift score comparison (which sessions drift most?)
- [ ] Activity distribution comparison across sessions
- [ ] Temporal drift trend (March recordings vs August recordings)

### Adaptation Method Comparison

| Metric | Baseline (A1) | Supervised (A3) | AdaBN+TENT (A4) | Pseudo-label (A5) |
|--------|:------------:|:--------------:|:---------------:|:-----------------:|
| F1 Score | — | 0.814 | — | — |
| Accuracy | — | — | — | 0.969 (val) |
| Confidence | ~0.70 | — | 0.76 | — |
| ECE | — | — | — | — |
| Cohen's κ | — | 0.795 | — | — |

- [ ] Fill in ALL cells from MLflow and artifact JSONs
- [ ] Add per-class F1 breakdown
- [ ] Create bar chart / radar chart for visual comparison
- [ ] Statistical significance tests (if applicable)

### Monitoring Analysis

- [ ] Baseline drift statistics (from `training_baseline.json`)
- [ ] Per-feature drift (accelerometer X/Y/Z, gyroscope X/Y/Z)
- [ ] Wasserstein distance time series (Stage 12 outputs)
- [ ] Confidence calibration curves (Stage 11 outputs)
- [ ] ECE before/after temperature scaling

---

## Quick Reference — Time Estimates

| Task | Estimated Time | Month |
|------|:--------------:|:-----:|
| Cross-dataset analysis | 1–2 days | Feb |
| Chapter 3 (Architecture) | 3–5 days | Feb/Mar |
| Chapter 4 (Implementation) | 3–5 days | Feb/Mar |
| Chapter 5 (Experiments) | 5–7 days | Mar |
| Chapter 1 (Introduction) | 2–3 days | Mar |
| Chapter 2 (Related Work) | 3–5 days | Mar |
| Chapter 6 (Discussion) | 2–3 days | Mar |
| Chapter 7 (Conclusion) | 1–2 days | Mar |
| Integration & polish | 5–7 days | Apr |
| Review cycle & feedback | 5–7 days | Apr |
| Final submission prep | 3–5 days | May |
| **Total writing effort** | **~35–50 days** | **Feb–May** |

---

## Notes

- **Thesis writing is THE critical path.** All code/pipeline work is essentially done.
- The ~80–82% code completion (revised 22 Feb 2026) means core pipeline works end-to-end; remaining items include Prometheus/Grafana live dashboards, Chapter 5 thesis writing, and some doc clean-up.
- Existing documentation in `docs/thesis/` contains outlines, evaluation plans, and concept notes that should be reused.
- `docs/thesis/chapters/` has draft chapter files that can be starting points.
- The `FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md` (1995 lines) has the most detailed planning.

---

*Generated: 19 Feb 2026*

# 21 — Thesis Report Blueprint and Chapter Plan

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22

---

## 1 Thesis Title (Suggested)

> **Monitoring, Drift Detection, and Unsupervised Adaptation for Wearable IMU-Based Human Activity Recognition: An MLOps Pipeline Approach**

---

## 2 Chapter-by-Chapter Blueprint

### Chapter 1: Introduction (~8-10 pages)

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| 1.1 Motivation | HAR in anxiety-detection wearables; real-world deployment challenges; why MLOps matters | — |
| 1.2 Problem Statement | Unlabeled production data; sensor drift; model degradation without labels | — |
| 1.3 Research Questions | RQ1: Can 3-layer monitoring detect degradation without labels? RQ2: Which adaptation method (AdaBN/TENT/pseudo-label) best recovers? RQ3: Does a trigger policy reduce unnecessary retraining? | — |
| 1.4 Contributions | (1) 14-stage MLOps pipeline, (2) 3-layer label-free monitoring, (3) tiered adaptation with safety gates, (4) governance + rollback | — |
| 1.5 Thesis Organization | Chapter roadmap | — |

**Writing order:** Write LAST (after all technical chapters). **Evidence:** [DOC: docs/thesis/chapters/CH1_INTRODUCTION_PROBLEM.md]

---

### Chapter 2: Background and Related Work (~15-20 pages)

| Section | Content | Key References |
|---------|---------|----------------|
| 2.1 Wearable IMU-Based HAR | Sensor modalities, windowing, feature extraction, DL architectures | [Citation TODO: Bulling et al. 2014, Ordóñez & Roggen 2016] |
| 2.2 Domain Shift in HAR | User/device/placement/temporal drift types | Repo CSV: 37 papers across 7 HAR themes |
| 2.3 Test-Time Adaptation | AdaBN (Li et al. 2018), TENT (Wang et al. 2021) | `src/domain_adaptation/adabn.py`, `tent.py` |
| 2.4 Pseudo-Labeling & Curriculum Self-Training | Confidence gating, class-balanced selection, EMA teacher-student | `src/curriculum_pseudo_labeling.py` |
| 2.5 Uncertainty & Calibration | Temperature scaling (Guo et al. 2017), MC Dropout, ECE | `src/calibration.py` |
| 2.6 Drift Detection & Monitoring | KS, PSI, Wasserstein, Z-score, multi-signal monitoring | `src/wasserstein_drift.py`, `scripts/post_inference_monitoring.py` |
| 2.7 MLOps for Healthcare/Wearable Systems | Pipeline orchestration, model governance, CI/CD | Sculley et al. 2015, Breck et al. 2017 |
| 2.8 Research Gap Summary | Table: what exists vs what this thesis adds | — |

**Writing order:** Write in parallel with Ch3. **Evidence:** [DOC: docs/research/appendix-paper-index.md], [DOC: docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md]

---

### Chapter 3: Methodology (~20-25 pages) — CORE CHAPTER

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| 3.1 System Overview | 14-stage pipeline architecture, design rationale | **Fig: End-to-end pipeline** (Mermaid → PDF) |
| 3.2 Data Pipeline | Ingestion (3 paths), validation (5 checks), preprocessing (4 steps), windowing | **Fig: Data flow diagram** |
| 3.3 Model Architecture | 1D-CNN-BiLSTM (v1): Conv1D(16)→Conv1D(32)→BiLSTM(64)→BiLSTM(32)→Dense(32)→Dense(11); ~499K params; 200×6 input | **Fig: Architecture diagram**, **Table: Layer shapes + params** |
| 3.4 3-Layer Monitoring Framework | L1 confidence, L2 temporal, L3 drift; orthogonal design; combined status | **Fig: 3-layer monitoring** (from File 12 Mermaid) |
| 3.5 Trigger Policy | 2-of-3 voting, 5 action levels, cooldown + escalation | **Fig: Trigger state machine** (from File 14 Mermaid) |
| 3.6 Adaptation Methods | AdaBN algorithm, TENT with 3 safety gates, AdaBN+TENT combined, calibrated pseudo-labeling with 7 stages | **Fig: Adaptation decision tree** (from File 13 Mermaid), **Table: Method comparison** |
| 3.7 Model Governance & Rollback | Registry, version management, SHA256 fingerprint, rollback validation | **Fig: Model promotion lifecycle** |
| 3.8 Calibration & Uncertainty | Temperature scaling, MC Dropout, ECE/Brier | — |
| 3.9 Observability Strategy | MLflow (primary) vs Prometheus/Grafana (production-ready config) | — |

**Writing order:** Write FIRST (directly backed by code). **Evidence:** All `src/` files documented in Files 10-14.

---

### Chapter 4: Implementation (~12-15 pages)

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| 4.1 Repository & Component Architecture | File structure, module responsibilities, ~7,500 lines across 27 key files | **Table: Module inventory** (from File 02) |
| 4.2 Pipeline Orchestrator | `production_pipeline.py`, stage interface, `run_pipeline.py` CLI | **Fig: Orchestrator sequence diagram** |
| 4.3 API & Containerization | FastAPI (3 endpoints + dashboard), 2 Dockerfiles, 4-service compose | **Fig: Docker architecture** |
| 4.4 CI/CD Workflow | 7-job GitHub Actions, lint → test → build → integration → notify | **Fig: CI/CD pipeline** |
| 4.5 Test Strategy | 215 tests, 19 files, unit/integration/slow markers, 12 fixtures | **Table: Test coverage matrix** |
| 4.6 Audit & Artifact Verification | `audit_artifacts.py` (12/12 pass), `verify_repository.py` | — |
| 4.7 Implementation Gaps & Engineering Tradeoffs | Stages 11-14 status, CI gaps, known limitations | **Table: Gap summary** (from File 03) |

**Writing order:** Write second (after Ch3). **Evidence:** [DOC: docs/thesis/chapters/CH4_IMPLEMENTATION.md] (existing draft)

---

### Chapter 5: Experimental Evaluation (~20-25 pages) — RESULTS CHAPTER

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| 5.1 Experimental Setup | Datasets (26 sessions), hardware, software versions, random seeds | **Table: Dataset characteristics** |
| 5.2 Baseline Performance | 5-fold CV results on training data; per-fold accuracy, F1, Kappa | **Table: CV results**, **Fig: Confusion matrix** |
| 5.3 Cross-Dataset Degradation | Drift scores across 26 sessions; confidence drop on production data | **Fig: Drift vs confidence scatter**, **Table: Per-session drift** |
| 5.4 Monitoring Evaluation | 3-layer behavior across sessions; which layers fire when | **Fig: Monitoring layer activations**, **Table: Alert frequencies** |
| 5.5 Adaptation Comparison | No-adapt vs AdaBN vs TENT vs AdaBN+TENT vs pseudo-label | **Table: Adaptation results** (Δconfidence, Δentropy), **Fig: Bar chart comparison** |
| 5.6 Trigger Policy Analysis | How often trigger fires; false positives; cooldown behavior | **Table: Trigger decision distribution** |
| 5.7 Proxy vs Labeled Audit Correlation | Confidence vs true accuracy on small labeled subset | **Fig: Scatter plot + Pearson r**, **Table: Proxy reliability** |
| 5.8 Ablation Studies | (a) 1 vs 2 vs 3 monitoring layers, (b) With vs without calibration, (c) Pseudo-label threshold sensitivity | **Tables: Ablation results** |
| 5.9 Runtime & Operational Cost | Per-stage timing, total pipeline, inference throughput | **Table: Runtime breakdown** |

**Writing order:** Write after experiments (Week 3-4). **Evidence:** Requires running experiments first — currently no results.

---

### Chapter 6: Discussion, Limitations, and Future Work (~8-10 pages)

| Section | Content |
|---------|---------|
| 6.1 Key Findings | Answer each RQ; summarize what worked and what didn't |
| 6.2 Practical Deployment Considerations | Offline vs service mode; Prometheus/Grafana decision; thesis vs production maturity |
| 6.3 Challenges Faced | Windows encoding issues; unlabeled evaluation difficulty; threshold tuning; integration complexity |
| 6.4 Lessons Learned | Multi-signal monitoring > single metric; safety gates essential for adaptation; governance often overlooked |
| 6.5 Threats to Validity | Internal: no labeled production data; External: single sensor modality; Construct: proxy ≠ accuracy |
| 6.6 Limitations | 26 sessions limited diversity; single participant(?); GPU dependency for full experiments |
| 6.7 Future Work | Energy OOD, LIFEWATCH memory, active learning, conformal monitoring, multi-sensor fusion |

**Writing order:** Write after Ch5.

---

### Chapter 7: Conclusion (~3-4 pages)

| Section | Content |
|---------|---------|
| 7.1 Summary of Contributions | Restate 4 contributions with evidence |
| 7.2 Answers to Research Questions | Concise RQ1-RQ3 answers with section references |
| 7.3 Practical Implications | What this means for wearable HAR deployment |
| 7.4 Closing Remarks | Final reflection |

**Writing order:** Write LAST.

---

### Appendices

| Appendix | Content |
|----------|---------|
| A | Pipeline configuration YAML reference |
| B | CI/CD workflow YAML + sample run output |
| C | Test suite summary table (215 tests × 19 files) |
| D | Artifact schema reference (JSON structures) |
| E | Reproducibility checklist (from File 27) |
| F | Full confusion matrices + per-class metrics |

---

## 3 Existing Draft Inventory

| File | Status | Reuse Potential |
|------|--------|----------------|
| `docs/thesis/chapters/CH1_INTRODUCTION_PROBLEM.md` | Partial draft | Medium — needs RQ sharpening |
| `docs/thesis/chapters/CH3_METHODOLOGY.md` | Partial draft | High — code-backed, needs expansion |
| `docs/thesis/chapters/CH4_IMPLEMENTATION.md` | Partial draft | High — needs Grafana ref fix (PG-3) |
| `docs/thesis/THESIS_STRUCTURE_OUTLINE.md` | Detailed outline | Reference only |
| `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md` | 88-paper synthesis | Reference for Ch2 |
| `22 feb codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS.md` | Independent analysis | Cross-reference |

---

## 4 Experiments and Ablations Checklist

| # | Experiment | Required Data | Status | Chapter |
|--:|-----------|--------------|--------|:-------:|
| E-1 | 5-fold CV baseline performance | Training data + labels | Likely completed (logs suggest past runs) | 5.2 |
| E-2 | Cross-dataset degradation (26 sessions) | All raw sessions + trained model | Partially done (batch_process script exists) | 5.3 |
| E-3 | Drift analysis (Z-score across sessions) | All raw sessions + baseline | Script exists, needs combined run | 5.3 |
| E-4 | Monitoring layer behavior analysis | Pipeline runs with 3-layer output | Partially available (60 pipeline results) | 5.4 |
| E-5 | Adaptation comparison (5 methods) | Target data + source replay | **NOT YET RUN** — critical gap | 5.5 |
| E-6 | Trigger policy behavior analysis | Extended pipeline runs with varied data | **NOT YET RUN** | 5.6 |
| E-7 | Proxy vs labeled audit correlation | Small labeled subset | **NOT YET AVAILABLE** — needs annotation | 5.7 |
| E-8 | Monitoring layer ablation (1/2/3 layers) | Same pipeline runs | **NOT YET RUN** | 5.8 |
| E-9 | Calibration impact (with/without T-scaling) | Labeled holdout | **NOT YET RUN** | 5.8 |
| E-10 | Runtime profiling | Any pipeline run | Easy to add timing | 5.9 |

---

## 5 Error Analysis and Limitations Preparation

### What to Analyze in Error Analysis
- Per-class confusion: which activities are confused with which?
- High-drift sessions: does the model fail on specific motion patterns?
- Adaptation failure modes: when does TENT make things worse?
- False trigger analysis: how often does the policy fire unnecessarily?

### Threats to Validity

| Threat Type | Specific Threat | Mitigation |
|-------------|----------------|------------|
| **Internal** | No ground-truth labels for production data | Proxy metrics + small labeled audit subset |
| **Internal** | Proxy ≠ accuracy (confidence can be miscalibrated) | Temperature scaling; ECE evaluation |
| **External** | Single wearable device (Garmin smartwatch) | Acknowledge; propose multi-device future work |
| **External** | Single deployment environment (wrist-worn) | Acknowledge; sensor_placement module as partial mitigation |
| **Construct** | Monitoring thresholds are manually set | Cross-dataset analysis + sensitivity study to justify |
| **Conclusion** | 26 sessions may be insufficient for statistical power | Report confidence intervals; acknowledge |

---

## 6 Writing Order Recommendation

```
WEEK 1: Chapter 3 (Methodology) — directly backed by code
WEEK 2: Chapter 4 (Implementation) — directly backed by code/config
WEEK 3: Chapter 5 (Results) — after experiments complete
WEEK 4: Chapter 2 (Background) — literature review
WEEK 5: Chapter 6 (Discussion) + Chapter 1 (Introduction)
WEEK 6: Chapter 7 (Conclusion) + Appendices + Abstract
WEEK 7: Review cycle + figure finalization
WEEK 8: Final polish + submission preparation
```

**Rationale:** Code-backed chapters first (3, 4) → forces experiment execution → results chapter (5) → literature framing (2) → bookend chapters (1, 6, 7).

# HAR MLOps Thesis — Everything Remaining (ChatGPT Handoff File 3 of 3)

> **Purpose:** Complete roadmap of what still needs to be done before thesis submission. Ordered by priority and dependency. Steps 1-6 already done (see File 2). Steps 7-10 + optional items are here.
> **Repository:** `d:\study apply\ML Ops\MasterArbeit_MLops`, branch `main`
> **Status:** ~75-78% complete. ~164-230 required hours remaining (5-6 weeks full-time).
> **Urgent risk:** Chapter 5 (Results) is completely empty — the thesis cannot be submitted without running experiments first.

---

## Priority Order

```
STEP 7 (Experiments)  →  STEP 8 (Figures)  →  STEP 9 (Thesis Writing)  →  STEP 10 (Polish + Submit)
      ↓ data needed           ↓ Ch 5 data           ↓ Ch 3 first                     ↓ last
```

**Do NOT start thesis writing before experiments. Chapter 5 cannot be written without experiment data.**

---

## STEP 7 — Run Experiments (20-30 hours, GPU recommended)

> **WHY:** Chapter 5 (Results & Evaluation) is completely empty. The three core research questions (RQ1, RQ2, RQ3) cannot be answered without measured data. All 26 sessions need to pass through the full 14-stage pipeline.

### Required experiments

- [ ] **7a. Batch all 26 sessions through the pipeline**
  ```bash
  python batch_process_all_datasets.py
  ```
  - Produces per-session inference CSVs + monitoring reports + drift scores
  - **Output:** 26 × monitoring artifacts, 26 × trigger decisions
  - ⏱ 8-12 hours (depends on hardware)

- [ ] **7b. Run cross-dataset drift analysis**
  ```bash
  python scripts/analyze_drift_across_datasets.py
  ```
  - Computes PSI and Wasserstein W₁ across all 26 sessions
  - **Output:** Drift trend table — needed for Ch 5 §monitoring effectiveness
  - ⏱ 2 hours

- [ ] **7c. 5-fold CV baseline** — confirm/update accuracy, F1, Cohen's κ
  ```bash
  python run_pipeline.py --retrain --adapt none
  ```
  - **Target:** acc > 0.92, F1 > 0.90 on source domain
  - **Output:** E-1 results table — baseline performance
  - ⏱ 2-4 hours

- [ ] **7d. Adaptation comparison** — 5 representative sessions × 5 methods
  ```bash
  # Run for each session with each --adapt flag:
  python run_pipeline.py --retrain --adapt none       --skip-cv
  python run_pipeline.py --retrain --adapt adabn      --skip-cv
  python run_pipeline.py --retrain --adapt tent       --skip-cv
  python run_pipeline.py --retrain --adapt adabn_tent --skip-cv
  python run_pipeline.py --retrain --adapt pseudo_label --skip-cv --epochs 10
  ```
  - **Output:** Ablation table — answers RQ2 (which adapter wins?)
  - ⏱ 6-8 hours (GPU)

- [ ] **7e. Monitoring ablation** — 1-layer vs 2-layer vs 3-layer detection rate
  - Run pipeline with each monitoring configuration variant
  - Measure: true positive rate (detected degradation), false positive rate (unnecessary triggers)
  - **Output:** E-8 detection rate table — answers RQ3
  - ⏱ 3-4 hours

- [ ] **7f. Trigger policy analysis** — decision distribution across 26 sessions
  - Count RETRAIN / ADAPT_ONLY / NO_ACTION decisions per session
  - **Output:** Trigger decision histogram — answers RQ1
  - ⏱ 2-3 hours (aggregation of 7a output)

- [ ] **7g. A2 Audit Run** — AdaBN only (fills ablation table gap)
  ```bash
  python run_pipeline.py --retrain --adapt adabn --skip-ingestion
  ```
  - Needed to complete the 5-method ablation table
  - ⏱ 1 hour

### Re-run validation (from Step 5)
- [ ] **5a. A4 re-run** — verify TENT confidence-drop rollback with new −0.01 threshold
  ```bash
  python run_pipeline.py --retrain --adapt adabn_tent --skip-ingestion
  ```
  - Previous A4: Δconf = −0.079 — should now trigger rollback
  - ⏱ 30 min

- [ ] **5b. Full 14-stage end-to-end run** (first time with all 14 stages active)
  ```bash
  python run_pipeline.py --retrain --adapt pseudo_label --advanced --epochs 10 --skip-ingestion
  ```
  - All 14 stages should complete. Generate artifacts for Stages 11-14.
  - ⏱ 1-2 hours

---

## STEP 8 — Generate All Thesis Figures & Tables (8-12 hours)

> **WHY:** The thesis requires specific figures and tables to illustrate results. Scripts exist but have not been run with full experiment data.

- [ ] **8a. Generate thesis figures** from scripts
  ```bash
  python scripts/generate_thesis_figures.py
  ```
  - **Produces:** confusion matrix, confidence distributions per session, drift over time, adaptation comparison bar charts
  - **Needs:** Step 7 data first
  - ⏱ 3-4 hours

- [ ] **8b. Render all 10 Mermaid diagrams to SVG/PDF** (D-1 through D-10)
  - Source: `docs/22Feb_Opus_Understanding/22_FIGURES_AND_DIAGRAMS_MERMAID_PACK.md`
  - Use: VS Code Mermaid Preview extension → export to SVG or PDF
  - **Diagrams include:** pipeline architecture overview, monitoring framework diagram, trigger policy logic, adaptation method comparison, ML lifecycle, data flow diagram, CI/CD workflow
  - ⏱ 2-3 hours

- [ ] **8c. Create 1D-CNN-BiLSTM architecture figure**
  - Detailed layer diagram with parameter counts: Conv(64, k=3) → BN → MaxPool → Dropout → Conv(128) → BN → MaxPool → BiLSTM(128) → Dense(64) → Dense(6)
  - Use draw.io, matplotlib, or similar
  - ⏱ 2-3 hours

- [ ] **8d. Fill ablation comparison table** — needs real numbers from Step 7
  - **Partial table** exists in `docs/TRAINING_RECIPE_MATRIX.md`
  - Add: no-adapt vs AdaBN vs TENT vs AdaBN+TENT vs pseudo-label — val_acc, F1, entropy_before, entropy_after, rollback_rate
  - ⏱ 1-2 hours (after Step 7d)

- [ ] **8e. Fill 20-figure + 17-table backlog** as needed
  - Source: `docs/22Feb_Opus_Understanding/26_THESIS_FIGURES_AND_TABLES_BACKLOG.md`
  - Generate only what gets referenced in the thesis text
  - ⏱ 2-3 hours (spread across writing)

---

## STEP 9 — Thesis Writing (100-130 hours)

> **Write chapters in this order. Each builds on the previous.**
> Existing drafts location: `docs/thesis/chapters/`
> **~30% already done** — outlines + 3 partial drafts

### Writing order (recommended)

#### 9a. Chapter 3 — Methodology (WRITE FIRST) (~20-25 pages)
> Directly code-backed. Most concrete. Doesn't wait for results.
- 3.1 Framework design rationale
- 3.2 Data pipeline (window segmentation, normalization, splits)
- 3.3 Model architecture (1D-CNN-BiLSTM with mathematical notation)
- 3.4 Three-layer monitoring framework (each layer with math definition)
  - Layer 1: mean confidence $\bar{c}$, entropy $H = -\sum p_i \log p_i$, class KL-divergence
  - Layer 2: mean dwell time $\bar{d}$, short-dwell ratio $r_{sd}$
  - Layer 3: PSI $= \sum (A_i - E_i) \ln(A_i/E_i)$, W₁ distance
- 3.5 Trigger policy (composite logic, threshold params, RETRAIN/ADAPT_ONLY/NO_ACTION)
- 3.6 Domain adaptation methods (AdaBN, TENT with BN fix, pseudo-label with rollback)
  - Temperature scaling calibration: $\hat{p}_i = \exp(z_i/T) / \sum_j \exp(z_j/T)$
  - TENT objective: $\mathcal{L} = -\sum_i \hat{p}_i \log \hat{p}_i$
  - AdaBN: replace $\mu_{BN}$, $\sigma^2_{BN}$ with batch statistics
- 3.7 Baseline governance and registry comparison
- **Existing draft:** `docs/thesis/chapters/CH3_METHODOLOGY.md`
- **Estimated:** 25-30 hours

#### 9b. Chapter 4 — Implementation (~10-14 pages)
> After Ch3. Describe the "how" rather than the "what".
- 4.1 Repository structure and design principles
- 4.2 Pipeline orchestration (14-stage, YAML config, entity pattern)
- 4.3 FastAPI inference service (`/health`, `/upload`, `/predict`, `/monitoring`)
- 4.4 CI/CD automation (7 jobs, weekly schedule, smoke test)
- 4.5 Testing strategy (225 tests, unit/integration/slow markers)
- 4.6 Reproducibility measures (lock file, SHA256, MLflow, commit hash)
- 4.7 Known limitations (no labeled production data, Prometheus/Grafana config-only)
- **Existing draft:** `docs/thesis/chapters/CH4_IMPLEMENTATION.md`
- **Estimated:** 15-20 hours

#### 9c. Chapter 5 — Results & Evaluation (~15-20 pages)
> Cannot write without experiment data from Step 7.
- 5.1 Baseline model performance (E-1: 5-fold CV results)
- 5.2 Model degradation and monitoring effectiveness (E-2, E-3)
- 5.3 Trigger policy precision (E-4, E-6: FP rate, RETRAIN/ADAPT_ONLY/NO_ACTION distribution)
- 5.4 Adaptation method comparison (E-5: ablation table — head-to-head results)
- 5.5 Calibration analysis (E-7: ECE before/after temperature scaling)
- 5.6 Monitoring ablation (E-8: 3-layer vs 2-layer vs 1-layer detection rate)
- 5.7 Proxy metric validation (E-10: confidence drop ↔ accuracy drop correlation)
- **Estimated:** 20-25 hours

#### 9d. Chapter 2 — Background & Literature Review (~15-20 pages)
- 2.1 Human Activity Recognition with IMU sensors
- 2.2 Domain shift in wearable systems (user variability, placement)
- 2.3 Domain adaptation methods (AdaBN, TENT, pseudo-labeling, survey)
- 2.4 MLOps fundamentals (Sculley 2015 hidden debt, Breck 2017 test score)
- 2.5 Production monitoring (concept drift, PSI, Wasserstein)
- 2.6 Model calibration (Guo 2017 temperature scaling)
- 2.7 Research gap and thesis positioning
- **Literature base:** 88 papers in `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md`; 37-paper CSV in `docs/`
- **Estimated:** 20-22 hours

#### 9e. Chapter 6 — Discussion (~8-10 pages)
- 6.1 Answer RQ1: does monitoring detect degradation?
- 6.2 Answer RQ2: which adapter wins cross-subject?
- 6.3 Answer RQ3: does 3-layer outperform simpler baselines?
- 6.4 Limitations and threats to validity
- 6.5 Comparison with related work (COA-HAR, OFTTA, SHOT)
- **Estimated:** 10 hours

#### 9f. Chapter 1 — Introduction (WRITE LAST) (~8-10 pages)
- Problem statement (HAR degradation in deployment)
- Research gap (no MLOps framework for wearable HAR without labels)
- Research questions (RQ1, RQ2, RQ3 — sharpen these based on actual results)
- Thesis contributions (3-5 bullet points)
- Thesis outline
- **Existing draft:** `docs/thesis/chapters/CH1_INTRODUCTION.md`
- **Estimated:** 8 hours

#### 9g. Chapter 7 — Conclusion + Abstract (~3-4 pages)
- Summary of contributions
- Future work (active learning, Prometheus live wiring, multi-dataset benchmarks)
- Abstract in English (~250 words)
- Zusammenfassung in German (~250 words)
- **Estimated:** 4 hours

#### 9h. Appendices A-F (~5-8 pages)
- A: Full YAML config reference (key parameters for reproducibility)
- B: Test matrix (225 tests by category, coverage map)
- C: Reproducibility checklist (62 items from Opus File 27)
- D: FastAPI endpoint specification
- E: Dataset overview (26 sessions, per-class window counts)
- F: CI/CD workflow diagram (from Mermaid diagrams in Step 8b)
- **Estimated:** 6 hours

#### 9i. Resolve 12 Citation TODOs
- CT-1 through CT-12: 12 in-code BibTeX references are incomplete
- Source: `docs/22Feb_Opus_Understanding/23_REFERENCES_AND_CITATION_LEDGER.md`
- Check each against Google Scholar / Semantic Scholar
- **Estimated:** 3-4 hours

---

## STEP 10 — Polish & Submit (15-25 hours)

- [ ] **10a. Full self-review** — read entire thesis end-to-end
  - Check: transitions between sections, consistent notation, figure references
  - ⏱ 6-8 hours

- [ ] **10b. Supervisor review cycle** — send complete draft
  - Send → receive feedback → incorporate
  - Build in 1-2 week buffer
  - ⏱ 5-10 hours (active editing time)

- [ ] **10c. Final formatting and bibliography**
  - Check: LaTeX/Word formatting, figure captions, table numbering
  - Verify all 12 Citation TODOs are resolved
  - Check bibliography against every in-text citation
  - ⏱ 4-6 hours

- [ ] **10d. Submit** 🎓

---

## Defense Preparation (after thesis is drafted)

| # | Task | Time | Notes |
|---|------|:----:|-------|
| D-1 | Prepare 15-20 slide presentation | 6-8h | Use Mermaid diagrams (Step 8b) as figures; pipeline overview + results |
| D-2 | Live demo: full 14-stage pipeline run (1 session) | 2-3h | Record screencast as backup; requires `python run_pipeline.py --advanced` |
| D-3 | Demo: adaptation before/after comparison | 2h | Show entropy reduction with TENT; show 43 pseudo-labeled samples |
| D-4 | Rehearse 7 examiner questions | 2h | Answers drafted in `CHATGPT_1_THESIS_KNOWLEDGE_BASE.md §11` |
| D-5 | Print rubric with self-assessment | 1h | Rubric in `CHATGPT_1_THESIS_KNOWLEDGE_BASE.md §12` |

---

## Optional Tasks (Cut First if Deadline Tight)

| # | Task | ⏱ | Why Optional | Source |
|---|------|:-:|-------------|--------|
| O-1 | Energy-based OOD score in monitoring | 2-4h | Better Ch5 story but thesis is valid without | IMP-16 |
| O-2 | Wasserstein option for Layer 3 (replace/augment PSI) | 3-4h | Stage 12 exists; just not plumbed into monitoring comparison | IMP-15 |
| O-3 | Wire `MetricsExporter` into `app.py /predict` | 2h | Prometheus metrics from API; nice for demo | IMP-14 |
| O-4 | Unified drift + confidence per-session report | 4-6h | Better visualization for thesis; manual combination works too | IMP-11 |
| O-5 | Bridge offline Z-score ↔ pipeline W₁ thresholds | 3-4h | Thesis consistency; can note as future work instead | IMP-10 |
| O-6 | Pattern memory for recurring benign drift | 8-12h | Research extension; not needed for RQ answers | IMP-17 |
| O-7 | Conformal prediction monitoring | 8-12h | Novel but expensive; future work | IMP-18 |
| O-8 | Active learning query pipeline | 8-12h | Future work | IMP-19 |
| O-9 | **Wire Prometheus/Grafana into `docker-compose`** | 4-6h | Currently config-only; enables live dashboard demo for defense | Stages Roadmap §2.1 |
| O-10 | Online TTA benchmarking (COA-HAR/OFTTA) | 10-15h | Literature comparison; future work if time tight | Codex §20 P2-2 |
| O-11 | Gravity removal experiment | 1-2 days | Preprocessing variant; interesting but not core | Stages Roadmap §1.2 |
| O-12 | A/B testing infrastructure | 1 week | Production-only feature; future work | Stages Roadmap §3.1 |
| O-13 | Create labeled audit subset for proxy-metric correlation | 4-8h | Would strengthen E-10 (Q2 examiner answer) | Opus W-5 |

**Recommended optional tasks if time allows:** O-3 (2h, easy) → O-9 (enables Grafana demo) → O-4 (better figures) → O-13 (strengthens W-5 weakness)

---

## Open Research Questions (Still Unanswered)

These are open questions from Opus File 24 that the experiments or thesis writing should answer:

| ID | Question | How to answer | Chapter |
|----|---------|--------------|:-------:|
| Q-1 | Does confidence drop predict accuracy drop significantly on this dataset? | E-10: correlation analysis across 26 sessions | Ch 5.7 |
| Q-2 | Does 3-layer monitoring outperform 1-layer meaningfully? | E-8: ablation — detection rate at each layer count | Ch 5.6 |
| Q-3 | Which adapter wins cross-subject HAR — AdaBN, TENT, or pseudo-label? | E-5: head-to-head across 5 sessions × 5 methods | Ch 5.4 |
| Q-4 | At what drift magnitude does the trigger fire correctly vs too early? | E-4: trigger precision analysis | Ch 5.3 |
| Q-5 | Does temperature scaling actually improve calibration (ECE) on this model? | E-7: ECE before/after Stage 11 | Ch 5.5 |
| Q-6 | Is pseudo-label rollback triggered often? What fraction of sessions need it? | A5 extended + Step 7a aggregate | Ch 5.4 note |
| Q-7 | Does TENT still improve after AdaBN (composition vs standalone)? | E-5 ablation — AdaBN alone vs AdaBN+TENT | Ch 5.4 |

---

## Week-by-Week Schedule (Recommended)

| Week | Priority Steps | Key Deliverable | Risk if Skipped |
|:----:|---------------|----------------|-----------------|
| **1** | Step 7 (all experiments) | All experiment data collected; ablation table filled | Ch 5 stays empty |
| **2** | Steps 5a/5b + Step 8 | All figures/diagrams generated; Mermaid SVGs rendered | Ch 5 has no figures |
| **3** | Step 9a (Ch 3) | Methodology chapter complete (~25 pages) | Weakest section in rubric |
| **4** | Steps 9b + 9c | Ch 4 Implementation + Ch 5 Results (~30 pages) | Most examiner-scrutinized |
| **5** | Steps 9d + 9e | Ch 2 Literature + Ch 6 Discussion (~25 pages) | Background thin |
| **6** | Steps 9f + 9g + 9h + 9i | Ch 1 + Ch 7 + Appendices + Citations (~20 pages) | Framing missing |
| **7** | Step 10a + 10b | Full draft to supervisor | Buffer for revision |
| **8** | Steps 10b + 10c + 10d | Revisions → Final submission | — |

---

## Time Budget

| Step | Hours | Status |
|------|:-----:|:------:|
| 1-6. Code fixes + improvements | ~29-38h | ✅ **DONE** |
| 7. Experiments | 20-30h | ❌ Not started |
| 8. Figures & tables | 8-12h | ❌ Not started |
| 9. Thesis writing | 100-130h | ~30% done (outlines + 3 partial drafts) |
| 10. Polish & submit | 15-25h | ❌ Not started |
| Optional (if time) | 60-100h | Cut if tight |
| Defense preparation | 13-16h | Q&A answers drafted only |
| **REQUIRED REMAINING (Steps 7-10)** | **~143-197h** | **~4-5 weeks full-time** |

---

## Context for ChatGPT

If you're helping write or improve any part of this thesis, here's what you should know:

1. **The pipeline is done** — 14 stages, 225/225 tests, all bugs fixed. Don't suggest pipeline changes unless critical.

2. **The 3 research questions** the thesis must answer: (1) Can the 3-layer monitoring framework detect HAR model degradation? (2) Which domain adaptation method works best without labels? (3) Does composite trigger logic reduce false-positive retraining?

3. **The biggest gap** is Chapter 5 — no experimental results yet. Every other chapter can be written, but Ch 5 needs Step 7 data first.

4. **The proxy metric problem** — there are no labeled production samples. All evaluation uses: (a) training set holdout for accuracy, (b) proxy metrics (confidence, entropy, PSI) for production monitoring quality. This is a known limitation — not a flaw, but must be stated clearly.

5. **Adaptation safety rails** are in place — TENT rolls back on entropy increase, pseudo-label rolls back on accuracy drop > 10pp, baseline requires explicit `--update-baseline` flag. These are thesis-worthy contributions.

6. **Language note** — thesis may be in German or English (or both). Abstract needs both (Zusammenfassung + Abstract).

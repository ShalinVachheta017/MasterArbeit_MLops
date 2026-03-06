# 25 — Execution Plan: 4 to 8 Weeks

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22

---

## 1 Top 10 Strengths (What's Already Strong)

| # | Strength | Evidence | Impact |
|--:|----------|----------|--------|
| S-1 | 14-stage pipeline design with clear stage interface | `production_pipeline.py` — `PipelineStage` + orchestrator | Architecture is publication-ready |
| S-2 | 3-layer orthogonal monitoring (confidence / temporal / drift) | `post_inference_monitoring.py` — 3 independent layers | Novel contribution; rare in HAR MLOps |
| S-3 | Sophisticated trigger policy (2-of-3 voting, cooldown, escalation) | `trigger_policy.py` — 822 lines, 17 configurable params | Demonstrates production-grade thinking |
| S-4 | TENT adaptation with 3 safety gates | `tent.py` — OOD guard, rollback check, stat restoration | Safety-first design; differentiates from naive TTA |
| S-5 | Comprehensive test suite (215 tests, 19 files) | `tests/` directory, `pytest.ini` with 3 markers | Solid engineering foundation |
| S-6 | Model governance with SHA256 fingerprint + rollback | `model_rollback.py` — registry, validation, audit trail | Production MLOps best practice |
| S-7 | CI/CD with 7-job GitHub Actions workflow | `.github/workflows/ci-cd.yml` — lint → test → build → notify | Continuous integration operational |
| S-8 | Docker setup (2 Dockerfiles + 4-service compose) | `docker/`, `docker-compose.yml` | Deployment-ready infrastructure |
| S-9 | Calibrated pseudo-labeling with 7-stage pipeline | `curriculum_pseudo_labeling.py` — temperature, entropy gate, class balance | Methodologically sound |
| S-10 | Cross-dataset analysis framework (26 sessions, batch processing) | `batch_process_all_datasets.py`, `analyze_drift_across_datasets.py` | Produces thesis evaluation data |

---

## 2 Top 10 Tasks (Ordered by Impact)

| # | Task | Priority | Effort | Blocks |
|--:|------|:--------:|:------:|--------|
| T-1 | Wire stages 11-14 into orchestrator | **P0** | 4-6h | Ch 5 experiments |
| T-2 | Fix 4 placeholder zeros in trigger_evaluation.py | **P0** | 2-3h | Trigger experiments |
| T-3 | Replace `is_better=True` with proxy comparison | **P0** | 4-6h | Governance claim |
| T-4 | Run full 26-session adaptation experiments | **P0** | 8-16h (GPU) | Ch 5.5, 5.6 results |
| T-5 | Write Chapter 3 (Methodology) | **P1** | 20-30h | Thesis core |
| T-6 | Write Chapter 4 (Implementation) | **P1** | 15-20h | Thesis core |
| T-7 | Unify monitoring thresholds (API vs pipeline) | **P1** | 2-3h | System consistency |
| T-8 | Write Chapter 5 (Results) | **P1** | 20-25h | Thesis core |
| T-9 | Write Chapters 2, 6, 1, 7 (remaining) | **P2** | 30-40h | Thesis completion |
| T-10 | Generate all thesis figures from Mermaid + experiment data | **P2** | 8-12h | Visual quality |

---

## 3 Full 8-Week Plan

### Week 1: Critical Bug Fixes + Integration (Code Focus)

| Day | Task | Deliverable | Hours |
|-----|------|-------------|:-----:|
| Mon | Wire stages 11-14 into `ALL_STAGES` (IMP-01) | Stages callable via `run_pipeline.py` | 4-6 |
| Tue | Fix placeholder zeros in `trigger_evaluation.py` (IMP-02) | Real monitoring signals flow to trigger | 2-3 |
| Wed | Implement proxy model comparison in `model_registration.py` (IMP-03) | `is_better` uses confidence delta | 4-6 |
| Thu | Unify monitoring thresholds (IMP-11); fix Grafana phantom ref (PG-3) | Single source of truth for thresholds | 3-4 |
| Fri | Test all fixes — full pipeline run (1 session) + pytest | Green CI + 1 clean pipeline log | 4-6 |

**Week 1 exit criteria:** All critical bugs fixed; stages 11-14 orchestrated; full pipeline runs end-to-end.

---

### Week 2: Experiment Execution (Data Focus)

| Day | Task | Deliverable | Hours |
|-----|------|-------------|:-----:|
| Mon | Run 5-fold CV baseline (verify/update E-1) | `results/baseline_cv.json` with accuracy, F1, Kappa | 4 |
| Tue-Wed | Run 26-session batch with full pipeline (E-2, E-3) | Per-session monitoring reports + drift scores | 8-12 |
| Thu | Run 5-method adaptation comparison — 5 representative sessions (E-5) | `results/adaptation_comparison.csv` | 6-8 |
| Fri | Run trigger policy analysis (E-6) + monitoring ablation (E-8) | Trigger decision distribution + ablation tables | 6 |

**Week 2 exit criteria:** All experiment data collected; results in structured format ready for analysis.

---

### Week 3: Chapter 3 — Methodology (Writing Focus)

| Section | Content Source | Pages | Hours |
|---------|--------------|:-----:|:-----:|
| 3.1 System Overview | File 01, Diagram D-1 | 2-3 | 4 |
| 3.2 Data Pipeline | File 10, Diagram D-10 | 2-3 | 4 |
| 3.3 Model Architecture | File 11, `train.py` | 2-3 | 3 |
| 3.4 Monitoring Framework | File 12, Diagrams D-2/D-3 | 3-4 | 5 |
| 3.5 Trigger Policy | File 14, Diagrams D-5/D-6 | 2-3 | 4 |
| 3.6 Adaptation Methods | File 13, Diagram D-4 | 3-4 | 5 |
| 3.7-3.9 Governance, Calibration, Observability | Files 14, 13, 17 | 3-4 | 5 |

**Week 3 exit criteria:** Chapter 3 complete first draft (~20-25 pages).

---

### Week 4: Chapter 4 — Implementation + Chapter 5 Start

| Task | Content Source | Pages | Hours |
|------|--------------|:-----:|:-----:|
| Ch 4.1-4.3 (Repo, Pipeline, API/Docker) | Files 02, 15, Diagrams D-8/D-9 | 6-8 | 10 |
| Ch 4.4-4.7 (CI/CD, Tests, Audit, Gaps) | Files 03, 15 | 4-6 | 8 |
| Ch 5.1-5.3 (Setup, Baseline, Degradation) | Experiment data from Week 2 | 4-6 | 8 |
| Generate key experiment figures | Scripts + matplotlib/seaborn | — | 4 |

**Week 4 exit criteria:** Chapter 4 complete; Chapter 5 sections 5.1-5.3 drafted.

---

### Week 5: Chapter 5 — Results + Chapter 2 Start

| Task | Content Source | Pages | Hours |
|------|--------------|:-----:|:-----:|
| Ch 5.4-5.6 (Monitoring, Adaptation, Trigger) | Experiment data | 6-8 | 12 |
| Ch 5.7-5.9 (Proxy, Ablation, Runtime) | Experiment data + File 21 checklist | 4-6 | 8 |
| Ch 2 outline + Section 2.1-2.3 draft | CSV paper collection (37 papers) | 6-8 | 10 |

**Week 5 exit criteria:** Chapter 5 complete first draft; Chapter 2 half-drafted.

---

### Week 6: Chapters 2 + 6 + 1

| Task | Content Source | Pages | Hours |
|------|--------------|:-----:|:-----:|
| Ch 2.4-2.8 (complete literature review) | Paper collection + Citation TODOs | 8-10 | 12 |
| Ch 6 Discussion, Limitations, Future Work | File 21 §6 sections | 8-10 | 10 |
| Ch 1 Introduction | File 21 §1 — write last | 8-10 | 8 |

**Week 6 exit criteria:** All 6 content chapters drafted (~80-100 pages total).

---

### Week 7: Chapter 7 + Figures + First Review

| Task | Deliverable | Hours |
|------|-------------|:-----:|
| Ch 7 Conclusion + Abstract | 3-4 pages | 4 |
| Appendices (A-F) | Configuration refs, test tables, reproducibility checklist | 6 |
| Finalize all figures (render Mermaid → PDF, generate plots) | ~20 figures thesis-ready | 8 |
| Full self-review pass | Marked-up draft for revision | 10 |
| Send to supervisor for review | PDF + source | 2 |

**Week 7 exit criteria:** Complete first draft sent to supervisor.

---

### Week 8: Revision + Polish + Submission

| Task | Deliverable | Hours |
|------|-------------|:-----:|
| Address supervisor feedback | Revised sections | 12 |
| Proofread all chapters | Clean text | 6 |
| Verify all citations (resolve CT-1 through CT-12) | Complete bibliography | 4 |
| Format check (page limits, margins, ToC, LoF, LoT) | Submission-ready PDF | 4 |
| Final artifact check (commit hash in appendix) | Reproducibility verified | 2 |
| Submit | 🎓 | 1 |

**Week 8 exit criteria:** Thesis submitted.

---

## 4 Compressed 4-Week Plan (Tight Deadline)

If only 4 weeks are available, cut scope as follows:

| Week | Focus | Cut |
|:----:|-------|-----|
| 1 | Bug fixes (T-1 through T-3) + Run subset of experiments (5 sessions, not 26) | Skip full 26-session sweep; use 5 representative sessions |
| 2 | Write Ch 3 + Ch 4 + key Ch 5 results | Skip ablation studies (5.8); minimal runtime analysis (5.9) |
| 3 | Write Ch 2 (abbreviated) + Ch 5 remaining + Ch 6 | Shorten literature review to 8-10 pages; skip deep DANN/MMD discussion |
| 4 | Ch 1 + Ch 7 + figures + review + submit | Minimal revision cycle; risk: no supervisor feedback loop |

**4-week risk:** No buffer for supervisor feedback; reduced experimental evidence; literature review less comprehensive.

---

## 5 Resource Requirements

| Resource | Needed For | Availability |
|----------|-----------|:------------:|
| GPU (8GB+ VRAM) | Experiments E-4, E-5, E-9 | **Check** |
| 26 session datasets | Full cross-dataset analysis | Available in repo |
| Trained base model | All adaptation experiments | `models/fine_tuned_model_1dcnnbilstm.keras` |
| MLflow server | Experiment tracking | Docker compose available |
| LaTeX environment | Thesis formatting | **Check** |
| Supervisor availability | Review in Week 7-8 | **Check** |

---

## 6 Milestone Checkpoints

| Milestone | Target Date | Verification |
|-----------|:----------:|-------------|
| M1: All critical bugs fixed | End Week 1 | Green CI + clean pipeline run |
| M2: All experiments complete | End Week 2 | Results CSVs/JSONs in `results/` |
| M3: Chapters 3+4 drafted | End Week 4 | ~35 pages |
| M4: All chapters drafted | End Week 6 | ~80-100 pages |
| M5: First complete draft | End Week 7 | PDF sent to supervisor |
| M6: Submission | End Week 8 | Thesis submitted |

# 28 — Reviewer/Examiner Rubric Alignment

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22  
> **Note:** Rubric criteria are based on typical European CS/ML master's thesis standards. Adjust to your university's specific rubric if available.

---

## 1 Rubric Alignment Matrix

| # | Criterion | Weight | Repo Evidence | Confidence | Gap/Risk |
|--:|----------|:------:|---------------|:----------:|----------|
| R-1 | **Problem definition & research questions** | 10% | `THESIS_STRUCTURE_OUTLINE.md` has RQ sketch; audit suggests 3 RQs (monitoring, adaptation, trigger) | 🟡 Medium | RQs need sharpening; no formal hypothesis statements yet |
| R-2 | **Literature review depth & currency** | 15% | 37-paper CSV + 7-theme summary; `PAPER_DRIVEN_QUESTIONS_MAP.md` with 88 papers | 🟢 High | Strong paper base; needs structured write-up (Ch 2) |
| R-3 | **Methodology rigor & justification** | 20% | 14-stage pipeline, 3-layer monitoring, 4 adaptation methods, trigger policy with voting — all in code | 🟢 High | Code-first methodology; needs algorithm pseudocode in thesis |
| R-4 | **Implementation quality & completeness** | 15% | ~7,500 LOC, 27 key files, 215 tests, Docker, CI/CD | 🟢 High | Stages 11-14 not orchestrated (D-1); 2 critical placeholders (T-1, T-2) |
| R-5 | **Experimental design & statistical validity** | 15% | Batch processing framework for 26 sessions; 5-fold CV code | 🟡 Medium | **Core gap:** Experiments E-5 through E-9 not yet run; no results |
| R-6 | **Results presentation & interpretation** | 10% | No results chapter content exists; thesis writing at 30% | 🔴 Low | Entirely dependent on running experiments first |
| R-7 | **Discussion of limitations** | 5% | Audit identifies extensive limitations (File 24); code acknowledges some gaps | 🟡 Medium | Needs honest write-up; examiner will probe unlabeled evaluation |
| R-8 | **Contribution significance** | 5% | 3-layer monitoring + tiered adaptation + governance is novel for HAR MLOps | 🟢 High | Strong if experiments validate the design |
| R-9 | **Writing quality & structure** | 5% | Existing chapter drafts partial; no complete chapter | 🔴 Low | Writing is the largest remaining effort |

---

## 2 Strengths to Highlight in Defense

| # | Strength | Evidence | Examiner Appeal |
|--:|----------|---------|----------------|
| 1 | **Novel 3-layer orthogonal monitoring** — rare in HAR MLOps | `post_inference_monitoring.py` — confidence, temporal, drift layers operate independently | "No single-point monitoring failure; each layer catches different degradation modes" |
| 2 | **Safety-gated adaptation** — TENT with 3 rollback conditions | `tent.py` — OOD guard, entropy rollback, stat restoration | "Production safety; adaptation cannot make model worse" |
| 3 | **Sophisticated trigger policy** — multi-signal voting with cooldown | `trigger_policy.py` — 822 lines, 5 action levels, escalation override | "Prevents alert fatigue; balances reactivity vs stability" |
| 4 | **Full MLOps governance chain** — SHA256 fingerprint, audit trail, rollback | `model_rollback.py` — registry, validation, append-only history | "Model provenance; can trace any deployed model to its training data" |
| 5 | **Comprehensive test suite** — 215 tests across 19 files | `tests/` — unit, integration, slow markers; 12 fixtures | "Engineering rigor beyond typical thesis work" |
| 6 | **Docker-ready deployment** — 2 Dockerfiles + 4-service compose | `docker/`, `docker-compose.yml` | "Demonstrates production deployment awareness" |
| 7 | **Cross-dataset evaluation framework** — 26 sessions | `batch_process_all_datasets.py` | "Beyond single-dataset evaluation; evaluates generalization" |
| 8 | **Calibration-aware pseudo-labeling** — temperature scaling + entropy gating | `curriculum_pseudo_labeling.py` — 7-stage pipeline | "Addresses miscalibration in pseudo-labels; methodologically sound" |

---

## 3 Weaknesses to Address Before Submission

| # | Weakness | Severity | Mitigation | Effort |
|--:|----------|:--------:|-----------|:------:|
| W-1 | **No experiment results exist** — Ch 5 is empty | Critical | Run experiments Week 2 (File 25) | 20-30h |
| W-2 | **Stages 11-14 not orchestrated** — claims vs reality gap | Critical | Wire into `ALL_STAGES` (IMP-01) | 4-6h |
| W-3 | **`is_better=True` placeholder** — governance claim undermined | Critical | Implement proxy comparison (IMP-03) | 4-6h |
| W-4 | **4 placeholder zeros** — trigger never gets real signals | Critical | Wire real monitoring data (IMP-02) | 2-3h |
| W-5 | **No labeled production data** — proxy metrics unverifiable | High | Create small labeled audit subset OR acknowledge as limitation | 4-8h |
| W-6 | **DANN/MMD redirect to pseudo-label** — misleading code | Medium | Remove or document as intentional design decision | 1-2h |
| W-7 | **Prometheus/Grafana config-only** — never integrated | Medium | Either integrate or frame as "deployment-ready configuration" | 4-8h |
| W-8 | **Thesis writing at ~30%** — most chapters unwritten | High | Follow 8-week writing plan (File 25) | 80+h |

---

## 4 Likely Examiner Questions & Prepared Responses

### Q1: "Why can't you evaluate adaptation methods with labeled accuracy?"

> **Prepared response:** "In real wearable deployment, labels are unavailable post-deployment. Our proxy metrics (mean confidence, normalized entropy, transition rate) aim to detect degradation without labels. We validate the proxy-accuracy correlation on a small labeled audit subset [if available] and report Pearson r. Temperature scaling (Guo et al. 2017) improves the reliability of confidence as a proxy."
>
> **Evidence:** `src/calibration.py:TemperatureScaler`, experiment E-7 (File 21)

### Q2: "How do you know the trigger policy thresholds are appropriate?"

> **Prepared response:** "The 17 trigger parameters were set based on cross-dataset analysis of 26 sessions. We analyze the distribution of drift scores and confidence drops to select thresholds that separate normal from degraded sessions. The sensitivity analysis (ablation E-8) evaluates how threshold changes affect trigger behavior."
>
> **Evidence:** `src/trigger_policy.py:L50-150`, `scripts/analyze_drift_across_datasets.py`

### Q3: "Why not use standard MLOps platforms (Kubeflow, Vertex AI)?"

> **Prepared response:** "Our pipeline is designed for wearable health research where (1) data stays local for privacy, (2) monitoring requires domain-specific signal interpretation (not just standard ML metrics), and (3) adaptation methods (AdaBN, TENT) are specific to distribution shift in sensor data, not supported by generic platforms. The custom orchestrator gives us stage-level control needed for our monitoring → trigger → adaptation → governance loop."
>
> **Evidence:** 14-stage pipeline design, `production_pipeline.py`

### Q4: "What happens if TENT makes the model worse?"

> **Prepared response:** "TENT has 3 safety gates: (1) OOD guard rejects data with normalized entropy > 0.85, (2) per-step rollback if ΔH > 0.05 or Δconfidence < -0.01, (3) BN running stats are restored after each optimization step. If all gates fail, the original model snapshot is restored."
>
> **Evidence:** `src/domain_adaptation/tent.py` — complete safety gate implementation

### Q5: "Your model registration always accepts the new model (`is_better=True`). How is this governance?"

> **Prepared response:** "This is a known placeholder (Finding T-2, Priority CRITICAL). The implementation plan (IMP-03) replaces this with proxy comparison using confidence delta on a holdout buffer. The governance infrastructure (SHA256 fingerprint, version registry, rollback path) is fully operational — only the comparison gate needs wiring."
>
> **Evidence:** `src/components/model_registration.py:L50+`, File 20 IMP-03

### Q6: "Why 3 monitoring layers and not 2 or 4?"

> **Prepared response:** "The 3 layers capture orthogonal degradation signals: (1) confidence measures model uncertainty, (2) temporal analysis catches erratic prediction patterns, (3) drift detects feature distribution shift. Our ablation study (E-8) compares 1-layer, 2-layer, and 3-layer configurations to quantify the marginal detection improvement."
>
> **Evidence:** File 12, `post_inference_monitoring.py`, experiment E-8

### Q7: "How reproducible are your results?"

> **Prepared response:** "The entire pipeline is version-controlled (commit `168c05b`), Docker-containerized, and CI/CD validated (215 tests). All experiment parameters are logged via MLflow. The reproducibility checklist (Appendix E) covers 62 criteria with 69% fully verified, 15% partially met. The 2 gaps (no dependency lock file, no data checksums) are noted and addressable."
>
> **Evidence:** File 27 — reproducibility scorecard

---

## 5 Defense Preparation Checklist

| # | Preparation Item | Status | Notes |
|--:|-----------------|:------:|-------|
| 5.1 | Prepare 15-20 slide presentation | ❌ | Use File 22 Mermaid diagrams as slide figures |
| 5.2 | Live demo: full pipeline run (1 session) | ❌ | Requires bugs fixed (W-2, W-3, W-4) |
| 5.3 | Demo: monitoring dashboard (FastAPI `/` endpoint) | 🔶 | Dashboard exists; ensure it shows real data |
| 5.4 | Demo: adaptation before/after comparison | ❌ | Requires experiment results |
| 5.5 | Prepare answers for Q1-Q7 above | ✅ | Answers drafted in this file |
| 5.6 | Know exact locations of all key code files | ✅ | File 02 inventory + File 23 citations |
| 5.7 | Print/prepare rubric with self-assessment | ❌ | Use Section 1 of this file |
| 5.8 | Review all 47 findings from audit | ✅ | File 24 compiles all findings |

---

## 6 Improvement Plan for Examiner Readiness

| Priority | Action | Timeline | Impact on Rubric |
|:--------:|--------|:--------:|:----------------:|
| **P0** | Fix W-1 through W-4 (critical weaknesses) | Week 1-2 | R-4, R-5 jump from 🟡/🔴 to 🟢 |
| **P1** | Run all experiments (E-1 through E-10) | Week 2-3 | R-5, R-6 jump from 🔴 to 🟢 |
| **P2** | Complete all thesis chapters | Week 3-7 | R-1, R-6, R-7, R-9 all improve |
| **P3** | Polish figures, bibliography, formatting | Week 7-8 | R-9 writing quality |

**Expected rubric after 8 weeks:** All criteria at 🟡 or 🟢 | Current: 3🟢, 3🟡, 2🟡, 1🔴

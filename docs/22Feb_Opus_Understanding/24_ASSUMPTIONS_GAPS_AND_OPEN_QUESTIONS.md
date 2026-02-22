# 24 — Assumptions, Gaps, and Open Questions

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22

---

## 1 Assumptions Made in This Audit

### 1.1 General Audit Assumptions

| ID | Assumption | Impact if Wrong |
|----|-----------|-----------------|
| A-00-1 | All conclusions refer to commit `168c05b`. Uncommitted local changes are out of scope. | Findings may miss recent fixes |
| A-00-2 | Test pass/fail counts are based on code inspection, not a live pytest run. | Actual pass rate may differ |
| A-00-3 | External services (MLflow, Prometheus, Grafana) were not validated as running. | Integration claims cannot be confirmed |
| A-00-4 | Files marked "Implemented" have code present; "Integrated" means wired into orchestrator; "Validated" means tests/artifacts confirm. | Terminology is consistently applied |

### 1.2 Phase 2 Assumptions

| ID | File | Assumption | Impact if Wrong |
|----|------|-----------|-----------------|
| A-10-1 | 10 | The 20ms merge_asof tolerance is tuned for 50Hz data (20ms = 1 sample gap). | Larger tolerance could introduce sensor alignment errors |
| A-10-2 | 10 | Gravity removal via Butterworth uses standard human-biomechanics cutoff. | Non-standard cutoff could affect feature extraction |
| A-11-1 | 11 | The 5-fold stratified CV does not leak data across folds (windowing before split is a risk). | Data leakage would inflate reported accuracy |
| A-12-1 | 12 | API monitoring thresholds were intentionally set differently from pipeline thresholds. | Could be accidental divergence (likely a bug) |
| A-13-1 | 13 | DANN/MMD redirect to pseudo-labeling is intentional, not incomplete. | If unintentional, 2 of 5 adaptation methods are stubs |
| A-14-1 | 14 | The 4 placeholder zeros in trigger_evaluation.py are temporary test scaffolding. | If permanent, trigger policy never receives real signals for 4 of 7 inputs |
| A-16-1 | 16 | The training session is not excluded from drift analysis — all 26 sessions analyzed against one baseline. | Self-comparison inflates low-drift count by 1 |

---

## 2 Known Gaps (Compiled from All Phase 2 Files)

### 2.1 Critical Gaps (Block Thesis Claims)

| ID | Gap | Severity | Source File | Recommendation |
|----|-----|----------|------------|----------------|
| T-1 | 4 placeholder zeros in trigger_evaluation.py | **Critical** | File 14 | Wire real monitoring signals |
| T-2 | model_registration.py `is_better=True` hardcoded | **Critical** | File 14 | Implement proxy comparison |
| D-1 | Calibration stage not orchestrated (stages 11-14 not in ALL_STAGES) | **High** | File 13 | Add stages 11-14 to orchestrator |
| CD-6 | No automated adaptation experiment results | **High** | File 16 | Orchestrate stages 11-14 in batch mode |
| PG-1 | Prometheus/Grafana fully specified but never integrated | **High** | File 17 | Wire or honestly present as config-only |
| PG-3 | Thesis references non-existent `config/grafana/har_dashboard.json` | **High** | File 17 | Create file or remove reference |

### 2.2 Medium Gaps (Affect Quality)

| ID | Gap | Severity | Source File | Recommendation |
|----|-----|----------|------------|----------------|
| M-1 | API vs pipeline monitoring thresholds diverge | **Medium** | File 12 | Unify or document rationale |
| M-5 | Baseline staleness after adaptation | **Medium** | File 12 | Auto-recompute baseline post-adaptation |
| D-4 | is_better=True in model_registration — no real comparison | **Medium** | File 13 | Integrate ProxyModelValidator |
| D-5 | DANN/MMD silently redirect to pseudo-labeling | **Medium** | File 13 | Remove or implement properly |
| CD-1 | Offline drift metric (Z-score) ≠ pipeline drift metric (W₁) | **Medium** | File 16 | Reconcile or validate correlation |
| CD-4 | Batch infrastructure and drift analysis never joined | **Medium** | File 16 | Create unified per-session report |
| PG-4 | Alert rules use PSI/KS but pipeline uses Wasserstein | **Medium** | File 17 | Align drift metric across stack |
| A-1 | `scripts/inference_smoke.py` referenced in CI but missing | **Medium** | File 15 | Create file or remove CI step |

### 2.3 Low Gaps (Polish Items)

| ID | Gap | Severity | Source File |
|----|-----|----------|------------|
| A-2 | CI model validation job is echo stubs | Low | File 15 |
| CD-2 | `generate_summary_report.py` hardcodes CSV path | Low | File 16 |
| CD-5 | Training session not excluded from drift analysis | Low | File 16 |
| I-1 | No automated data quality gate (validation warnings don't halt) | Low | File 10 |

---

## 3 All FINDING IDs Across Phase 2

| Finding ID | File | Summary |
|------------|------|---------|
| I-1 through I-6 | 10 | Ingestion/preprocessing findings |
| TR-1 through TR-5 | 11 | Training/evaluation findings |
| M-1 through M-5 | 12 | Monitoring findings |
| D-1 through D-6 | 13 | Drift/calibration/adaptation findings |
| T-1 through T-8 | 14 | Trigger/governance/rollback findings |
| A-1 through A-7 | 15 | API/Docker/CI/CD/tests findings |
| CD-1 through CD-6 | 16 | Cross-dataset findings |
| PG-1 through PG-5 | 17 | Prometheus/Grafana findings |

**Total unique findings:** ~47 across 8 files

---

## 4 Open Questions for Thesis Author

| # | Question | Why It Matters |
|---|----------|---------------|
| Q-1 | Which adaptation method (AdaBN / TENT / pseudo-label) will be **primary** for thesis experiments? | Determines evaluation depth |
| Q-2 | Is there a labeled audit subset for proxy metric correlation (confidence vs true accuracy)? | Without labels, proxy claims are unverifiable |
| Q-3 | What is the thesis submission deadline? | Drives scope decisions for remaining work |
| Q-4 | Is GPU access available for full 26-dataset adaptation experiments? | CPU-only would take days |
| Q-5 | What level of Prometheus/Grafana integration does the supervisor expect? | Determines effort on PG-1/PG-3 |
| Q-6 | Should DANN/MMD be removed or properly implemented? | D-5: currently misleading |
| Q-7 | Are stages 11-14 expected to work in the orchestrator before submission? | D-1/CD-6: core thesis gap |

---

## 5 Technical Debt Summary

| Priority | Count | Key Items |
|----------|-------|-----------|
| **Critical** | 3 | Placeholder zeros (T-1), is_better=True (T-2), stages 11-14 not orchestrated (D-1) |
| **High** | 3 | Prometheus never integrated (PG-1), phantom Grafana file (PG-3), no adaptation results (CD-6) |
| **Medium** | 8 | Threshold divergence, drift metric mismatch, DANN stub, missing smoke test, etc. |
| **Low** | 4 | Hardcoded paths, training session self-comparison, validation gate, CI stubs |
| **Total** | **18** | Across 8 Phase 2 deep-dive files |

---

## TODO: Phase 3 Additions

- ~~Add assumptions/gaps from Phase 3 files (20-28) when written~~
- ~~Cross-reference technical debt with improvement roadmap (File 20)~~
- ~~Map open questions to thesis chapter plan (File 21)~~

---

## 6 Phase 3 Assumptions (From Files 20-28)

| ID | File | Assumption | Impact if Wrong |
|----|------|-----------|-----------------|
| A-20-1 | 20 | Literature references (Guo 2017, TENT 2021, etc.) are accurately cited; full bibliographic details in File 23 Citation TODOs | Improvement justifications lose academic grounding |
| A-20-2 | 20 | Complexity estimates (hours per improvement) assume single developer with existing codebase familiarity | Estimates may 2-3× for unfamiliar contributor |
| A-21-1 | 21 | Thesis follows standard 7-chapter CS/ML master's structure | Supervisor may require different chapter layout |
| A-21-2 | 21 | Experiments E-1 through E-10 are feasible with available hardware/data | GPU access or dataset licensing issues could block Ch 5 |
| A-22-1 | 22 | Mermaid diagrams render correctly in standard Mermaid renderers (v10+) | Syntax differences across renderers may require adjustments |
| A-25-1 | 25 | 8-week timeline assumes full-time effort (~40h/week) | Part-time effort would require 12-16 weeks |
| A-27-1 | 27 | Reproducibility assessed from code inspection, not actual execution | Some items marked ✅ may fail on clean environment |
| A-28-1 | 28 | Rubric criteria are generic European CS master's standards | Actual university rubric may differ significantly |

---

## 7 Phase 3 Additional Gaps

| ID | Gap | Severity | Source File |
|----|-----|----------|------------|
| P3-1 | No labeled production test set exists for proxy metric validation | **High** | File 21 (E-7) |
| P3-2 | Experiments E-5 through E-9 have never been run | **High** | File 21 (exp checklist) |
| P3-3 | No thesis abstract or introduction draft exists | **Medium** | File 21 (Ch 1, Ch 7) |
| P3-4 | Grafana dashboard file referenced in thesis chapter drafts does not exist | **Medium** | File 17 / File 20 IMP-20 |
| P3-5 | Runtime profiling data for pipeline stages not collected | **Low** | File 21 (E-10) |

---

## 8 Cross-Reference: Gaps → Improvements → Chapters

| Gap ID | Improvement ID (File 20) | Thesis Chapter (File 21) |
|--------|--------------------------|--------------------------|
| T-1 | IMP-02 | Ch 3.5, Ch 4.7 |
| T-2 | IMP-03 | Ch 3.7, Ch 4.7 |
| D-1 | IMP-01 | Ch 3.1, Ch 4.2, Ch 5.5 |
| CD-6 | IMP-01 | Ch 5.5, Ch 5.6 |
| PG-1 | IMP-20 | Ch 3.9, Ch 6.2 |
| PG-3 | IMP-20 | Ch 4.3 |
| M-1 | IMP-11 | Ch 3.4, Ch 6.3 |
| P3-1 | — (data collection needed) | Ch 5.7 |
| P3-2 | — (experiment execution) | Ch 5 (all sections) |

---

## 9 Open Questions → Chapter Mapping

| Question | Relevant Chapter | Impact on Writing |
|----------|:---------------:|-------------------|
| Q-1 (primary adaptation method) | Ch 3.6, Ch 5.5 | Determines depth of method comparison |
| Q-2 (labeled audit subset) | Ch 5.7 | Without this, proxy correlation section is theoretical |
| Q-3 (submission deadline) | All | Drives scope for compressed vs full plan (File 25) |
| Q-4 (GPU access) | Ch 5 | CPU-only: limit to AdaBN + small pseudo-label experiments |
| Q-5 (Prometheus/Grafana expectation) | Ch 3.9, Ch 4.3 | Determines integration effort priority |
| Q-6 (DANN/MMD disposition) | Ch 2.3, Ch 3.6 | Remove = cleaner thesis; implement = more comparisons |
| Q-7 (stages 11-14 before submission) | Ch 3.1, Ch 5 | Core experiment gap if not wired |

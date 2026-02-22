# Questions & Doubts — Master Thesis HAR MLOps

> **Compiled from:** Opus 22-file audit pack + Codex independent analysis  
> **Repository:** commit `168c05bb`, branch `main`  
> **Date:** 2026-02-22

---

## How to Use This File

Every item below is something that **needs a decision, answer, or justification** before thesis submission. Items marked 🔴 may be asked by an examiner. Items marked 🟡 are design doubts worth resolving. Items marked 🟢 are clarifications that improve the thesis but won't block it.

---

## 1 🔴 Questions Your Examiner Will Likely Ask

These are the 7 most probable examiner questions, with draft answers already prepared in [File 28 §4](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md).

| # | Question | Why They'll Ask | Prepared? | Source |
|--:|----------|----------------|:---------:|--------|
| EQ-1 | **"Why can't you evaluate adaptation with labeled accuracy?"** | Proxy metrics are your substitute for ground truth — examiner will probe validity | ✅ Draft | [File 28 Q1](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| EQ-2 | **"How do you know your trigger thresholds are appropriate?"** | 17 tunable parameters look arbitrary without sensitivity analysis | ✅ Draft | [File 28 Q2](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| EQ-3 | **"Why build a custom pipeline instead of using Kubeflow/Vertex AI?"** | Standard challenge for any custom MLOps solution | ✅ Draft | [File 28 Q3](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| EQ-4 | **"What if TENT makes the model worse?"** | Safety-critical concern for any test-time adaptation | ✅ Draft | [File 28 Q4](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| EQ-5 | **"`is_better=True` — how is this governance?"** | Directly undermines your model governance claim | ✅ Draft (but code fix needed first) | [File 28 Q5](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| EQ-6 | **"Why exactly 3 monitoring layers, not 2 or 4?"** | Core design decision needs empirical justification | ✅ Draft (ablation experiment needed) | [File 28 Q6](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md), [Codex §7](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| EQ-7 | **"How reproducible are your results?"** | Standard thesis quality criterion | ✅ Draft | [File 28 Q7](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md), [File 27](../docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md) |

**Action:** Review draft answers in File 28. After running experiments, update with actual numbers.

---

## 2 🔴 Open Decisions You Must Make

These questions from [File 24 §4](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) need **your answer** — the audit cannot decide for you.

| # | Decision | Why It Matters | Impact | Source |
|--:|----------|---------------|--------|--------|
| Q-1 | **Which adaptation method is primary for thesis experiments?** (AdaBN / TENT / pseudo-labeling) | Determines depth of evaluation — one primary + others as comparison, or all equal? | Scopes Ch 3.6 + Ch 5.5 | [File 24 Q-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| Q-2 | **Do you have a labeled audit subset for proxy metric validation?** | Without labels, the proxy-accuracy correlation (Ch 5.7) is theoretical only. Consider setting aside 2-3 labeled sessions. | Scopes Ch 5.7 entirely | [File 24 Q-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 28 W-5](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| Q-3 | **What is your thesis submission deadline?** | 8 weeks → full plan feasible. 4 weeks → compressed plan with cuts. | Scopes everything | [File 24 Q-3](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 25 §4](../docs/22Feb_Opus_Understanding/25_EXECUTION_PLAN_4_TO_8_WEEKS.md) |
| Q-4 | **Do you have GPU access (8GB+ VRAM)?** | Full 26-session sweep + TENT optimization on CPU would take days. 5-session subset is fallback. | Scopes experiment depth | [File 24 Q-4](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 25 §5](../docs/22Feb_Opus_Understanding/25_EXECUTION_PLAN_4_TO_8_WEEKS.md) |
| Q-5 | **What level of Prometheus/Grafana integration does your supervisor expect?** | Full integration = 4-8h extra. Config-only = 0h but weaker MLOps claim. "Deployment-ready config" framing is a middle ground. | Scopes PG-1, PG-3 effort | [File 24 Q-5](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [Codex §16](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| Q-6 | **Should DANN/MMD be removed or properly implemented?** | Currently, selecting `dann` or `mmd` silently runs pseudo-labeling instead. Remove → cleaner thesis. Implement → more comparisons but 10-15h work. | Scopes Ch 3.6, D-5 | [File 24 Q-6](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 20 IMP-12](../docs/22Feb_Opus_Understanding/20_IMPROVEMENT_ROADMAP_EVIDENCE_AND_LITERATURE.md) |
| Q-7 | **Must stages 11-14 work in the orchestrator before submission?** | If yes → IMP-01 is P0 (4-6h). If not → describe as "implemented modules" in thesis and frame integration as future work. | Scopes D-1, CD-6 | [File 24 Q-7](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

---

## 3 🟡 Design Doubts — Things That Might Be Wrong

These are assumptions the audit made that **could be incorrect**. If any are wrong, the impact is noted.

### 3.1 Data & Preprocessing

| # | Doubt | If Wrong... | Source |
|--:|-------|------------|--------|
| D-1 | The 20ms `merge_asof` tolerance is tuned for 50Hz data (1 sample gap) | Larger tolerance could introduce sensor alignment errors, inflating accuracy | [File 24 A-10-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-2 | Gravity removal Butterworth cutoff uses standard biomechanics frequency | Non-standard cutoff → features may contain gravity component → model learns wrong patterns | [File 24 A-10-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-3 | 5-fold stratified CV doesn't leak data across folds (windowing before split) | **If windowing happens before fold split → data leakage → inflated accuracy.** This is the highest-risk assumption. | [File 24 A-11-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

### 3.2 Monitoring & Adaptation

| # | Doubt | If Wrong... | Source |
|--:|-------|------------|--------|
| D-4 | API monitoring thresholds (10%, 30%, 0.75) were intentionally different from pipeline thresholds | If accidental → inconsistent alerting is a bug, not a design choice | [File 24 A-12-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-5 | DANN/MMD redirect to pseudo-labeling is intentional | If unintentional → 2 of 5 adaptation methods are stubs, not features | [File 24 A-13-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-6 | The 4 placeholder zeros in trigger evaluation are temporary scaffolding | **If permanent → trigger policy never receives real signals for 4 of 7 inputs; 2-of-3 voting is confidence-only** | [File 24 A-14-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-7 | Training session is not excluded from drift analysis (all 26 sessions analyzed) | Self-comparison inflates the "low-drift" count by 1 | [File 24 A-16-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

### 3.3 Thesis & Meta Assumptions

| # | Doubt | If Wrong... | Source |
|--:|-------|------------|--------|
| D-8 | Thesis follows standard 7-chapter CS/ML master's structure | Supervisor may require different layout → chapter plan needs revision | [File 24 A-21-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-9 | Experiments E-1 to E-10 are feasible with available hardware/data | GPU or dataset issues → Ch 5 blocked entirely | [File 24 A-21-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-10 | 8-week timeline assumes full-time 40h/week | Part-time → need 12-16 weeks instead | [File 24 A-25-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-11 | Reproducibility assessed from code inspection, not actual execution | Items marked ✅ in File 27 may fail on clean environment | [File 24 A-27-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-12 | Rubric criteria are generic European CS master's standards | Actual university rubric may differ significantly | [File 24 A-28-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

---

## 4 🟡 Architectural Concerns Worth Justifying in Thesis

These aren't necessarily wrong — but an examiner or reviewer will want a justification.

| # | Concern | Justification Needed | Source |
|--:|---------|---------------------|--------|
| AC-1 | **Why 3 monitoring layers specifically?** Not 2. Not 4. | Ablation experiment (E-8) should quantify marginal detection improvement per layer. Until then, the "3" is unjustified. | [Codex §7](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md), [File 28 Q6](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| AC-2 | **Why these specific trigger thresholds?** 17 parameters look arbitrary. | Sensitivity analysis + cross-dataset calibration needed. Document derivation process. | [Codex §8-9](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md), [File 28 Q2](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| AC-3 | **Why custom pipeline over Kubeflow/Vertex AI?** | Domain-specific monitoring + local privacy + sensor-specific adaptation. Write this explicitly in Ch 6 Discussion. | [File 28 Q3](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md), [Codex §6 Stage 6](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| AC-4 | **Why proxy metrics instead of labeled evaluation?** | "In real wearable deployment, labels are unavailable post-deployment." Needs to be stated upfront in Ch 3. | [File 28 Q1](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| AC-5 | **Why 1D-CNN-BiLSTM over Transformer/other architectures?** | Thesis should justify model choice with ablation or literature comparison. | [File 11 deep-dive](../docs/22Feb_Opus_Understanding/11_DEEP_DIVE_TRAINING_EVALUATION.md) |

---

## 5 🟡 Claims vs Reality Gaps

These are things the code/thesis currently **claims** but the evidence doesn't fully support.

| # | Claim | Reality | Risk | Source |
|--:|-------|---------|------|--------|
| CR-1 | "14-stage pipeline" | Only 10 stages orchestrated; 4 exist as modules | Examiner checks pipeline code → mismatch | [File 04 §6](../docs/22Feb_Opus_Understanding/04_OVERALL_PROGRESS_RISK_AND_CRITICAL_PATH.md), [File 24 D-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| CR-2 | "Model governance with validation" | `is_better=True` auto-promotes every model | Examiner reads code → governance is fake | [File 24 T-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 28 Q5](../docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md) |
| CR-3 | "2-of-3 voting trigger" | 4 of 7 inputs are zeros → voting is confidence-only | Trigger policy is undermined | [File 24 T-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| CR-4 | "5 adaptation methods" | DANN/MMD silently run pseudo-labeling | Only 3 methods actually work | [File 24 D-5](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 20 IMP-12](../docs/22Feb_Opus_Understanding/20_IMPROVEMENT_ROADMAP_EVIDENCE_AND_LITERATURE.md) |
| CR-5 | "Prometheus/Grafana observability" | Config files exist but never integrated into running services | Observable in theory, not practice | [File 24 PG-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [Codex §16](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| CR-6 | "CI/CD model validation" | 3 steps are echo stubs + missing smoke script | CI doesn't actually validate models | [File 24 A-1, A-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

**Strategy:** Either fix each gap (see [01_REMAINING_WORK.md](01_REMAINING_WORK.md) P0 tasks) **or** reword the thesis claim to accurately describe what exists.

---

## 6 🟢 Minor Clarifications

| # | Item | Note | Source |
|--:|------|------|--------|
| MC-1 | `generate_summary_report.py` hardcodes CSV path | Low priority but breaks portability | [File 24 CD-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-2 | No automated data quality gate — validation warnings don't halt pipeline | Decide if this is intentional (log-only) or should block | [File 24 I-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-3 | Some paper-summary docs cite page numbers that need manual verification | Page-level citations may be inaccurate | [Codex §17](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| MC-4 | Runtime profiling data not collected for any pipeline stage | E-10 experiment asks for this but no baseline exists | [File 24 P3-5](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-5 | No dependency lock file (`requirements.txt` exists but no pinned versions / `poetry.lock`) | Reproducibility gap — File 27 notes this | [File 27](../docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md) |
| MC-6 | No data checksums for the 26 session datasets | Can't verify data integrity across machines | [File 27](../docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md) |

---

## 7 Priority Action: Top 5 Things to Resolve First

1. **Answer Q-3 (deadline)** → this scopes everything else
2. **Answer Q-7 (stages 11-14 requirement)** → largest code task depends on this
3. **Fix CR-2 (`is_better=True`)** → most embarrassing examiner finding
4. **Fix CR-3 (placeholder zeros)** → second-most embarrassing finding
5. **Answer Q-1 (primary adaptation method)** → scopes experiment design

---

## 8 Source Files Index

| Source | What It Contains | Path |
|--------|-----------------|------|
| Opus File 24 | 18 tech debt items, 7 open questions, assumptions, cross-reference matrix | `docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md` |
| Opus File 28 | 9-criterion rubric, 8 weaknesses, 7 examiner Q&As with draft answers | `docs/22Feb_Opus_Understanding/28_REVIEWER_EXAMINER_RUBRIC_ALIGNMENT.md` |
| Opus File 27 | 62-item reproducibility checklist, 69% verified scorecard | `docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md` |
| Opus File 04 | Overall progress, 6 blockers, risk matrix, final verdict | `docs/22Feb_Opus_Understanding/04_OVERALL_PROGRESS_RISK_AND_CRITICAL_PATH.md` |
| Opus File 20 | 20 improvements with literature references | `docs/22Feb_Opus_Understanding/20_IMPROVEMENT_ROADMAP_EVIDENCE_AND_LITERATURE.md` |
| Codex Analysis | Per-stage pseudocode, §7 monitoring rationale, §8-9 trigger/adaptation design, §16 Prometheus decision, §20 remaining work | `22 feb codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md` |

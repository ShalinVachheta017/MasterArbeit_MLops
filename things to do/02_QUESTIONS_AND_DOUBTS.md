# Questions & Doubts — Master Thesis HAR MLOps

> **Compiled from:** Opus 22-file audit pack + Codex independent analysis  
> **Repository:** commit `7f892d8`, branch `main` — **CI GREEN ✅**  
> **Date:** 2026-02-22 (updated after all Steps 1-6 + Docker fix)

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
| Q-6 | ~~**Should DANN/MMD be removed or properly implemented?**~~ | ✅ **RESOLVED** — Step 6d: `dann`/`mmd` now raise `NotImplementedError`; thesis claims 3 methods. No silent fallback. | Scopes Ch 3.6, D-5 | [File 24 Q-6](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md), [File 20 IMP-12](../docs/22Feb_Opus_Understanding/20_IMPROVEMENT_ROADMAP_EVIDENCE_AND_LITERATURE.md) |
| Q-7 | ~~**Must stages 11-14 work in the orchestrator before submission?**~~ | ✅ **RESOLVED** — Step 3: All 14 stages orchestrated. `python run_pipeline.py --advanced` executes all 14. | Scopes D-1, CD-6 | [File 24 Q-7](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

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
| D-4 | ~~API monitoring thresholds (10%, 30%, 0.75) were intentionally different from pipeline thresholds~~ | ✅ **RESOLVED** — Step 6a unified thresholds: API now imports `PostInferenceMonitoringConfig` and reads `confidence_warn_threshold=0.60`, `uncertain_pct_threshold=30.0`, `transition_rate_threshold=50.0`, `drift_zscore_threshold=2.0` | [File 24 A-12-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-5 | ~~DANN/MMD redirect to pseudo-labeling is intentional~~ | ✅ **RESOLVED** — Step 6d: DANN/MMD now raise `NotImplementedError`; thesis claims 3 methods (AdaBN, TENT, pseudo-label) | [File 24 A-13-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-6 | ~~The 4 placeholder zeros in trigger evaluation are temporary scaffolding~~ | ✅ **RESOLVED** — Step 2a: All 4 trigger inputs now wired to real monitoring output (mean_entropy, mean_dwell_time_seconds, short_dwell_ratio, n_drifted_channels) | [File 24 A-14-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| D-7 | ~~Training session is not excluded from drift analysis~~ | ✅ **RESOLVED** — Step 6g: `is_training_session=True` flag skips baseline comparison, logs `TRAINING_SESSION` reason | [File 24 A-16-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |

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
| CR-1 | "14-stage pipeline" | ✅ **FIXED** — All 14 stages now orchestrated in `production_pipeline.py` | Resolved — `python run_pipeline.py --advanced` runs all 14 end-to-end | Step 3 |
| CR-2 | "Model governance with validation" | ✅ **FIXED** — `model_registration.py` now calls `registry.list_versions()` and gates on 99% F1 threshold | Resolved | Step 2b |
| CR-3 | "2-of-3 voting trigger" | ✅ **FIXED** — All 4 trigger inputs (entropy, dwell, short_dwell_ratio, n_drifted_channels) now read from real monitoring output | Resolved | Step 2a |
| CR-4 | "5 adaptation methods" | ✅ **FIXED** — DANN/MMD removed from docs; now raise `NotImplementedError`; thesis claims 3 methods | Resolved | Step 6d |
| CR-5 | "Prometheus/Grafana observability" | ⚠️ **STILL OPEN** — Config files exist but never integrated into running services | Candidate for Optional Task O-9; must be framed in thesis as "deployment-ready config" | [File 24 PG-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| CR-6 | "CI/CD model validation" | ✅ **FIXED** — Smoke script created (Step 1b); Docker api.app shadowing resolved; CI confirmed GREEN (commit `7f892d8`) | Resolved | Steps 1b, 4, Docker fix |
| CR-7 | **README and docs described dataset as "6 activities (walking, jogging, sitting, standing, stairs up/down)"** but `data/all_users_data_labeled.csv` has **11 anxiety behavior classes** (ear_rubbing, forehead_rubbing, hair_pulling, hand_scratching, hand_tapping, knuckles_cracking, nail_biting, nape_rubbing, sitting, smoking, standing). `src/api/app.py` was correct all along. | ✅ **FIXED** — README project overview, model details, and activity classes block now say 11 anxiety behavior classes. `CHATGPT_1` dataset section and model architecture (`Dense(11)`) updated to match. | Resolved — 22 Feb 2026 |
| CR-8 | README and knowledge-base docs called Layer 3 monitoring "**PSI drift**" but the actual implementation uses **z-score** (`\|mean_prod − mean_base\| / std_base`) | ✅ **FIXED** — README, CHATGPT_1 Layer 3 table, and threshold field names now all say "z-score drift vs baseline". PSI/W₁ are correctly attributed to Stage 12 (advanced `--advanced` flag only) | Resolved — 22 Feb 2026 |

**Strategy:** Either fix each gap (see [01_REMAINING_WORK.md](01_REMAINING_WORK.md) P0 tasks) **or** reword the thesis claim to accurately describe what exists.

---

## 6 🟢 Minor Clarifications

| # | Item | Note | Source |
|--:|------|------|--------|
| MC-1 | `generate_summary_report.py` hardcodes CSV path | Low priority but breaks portability | [File 24 CD-2](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-2 | No automated data quality gate — validation warnings don't halt pipeline | Decide if this is intentional (log-only) or should block | [File 24 I-1](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-3 | Some paper-summary docs cite page numbers that need manual verification | Page-level citations may be inaccurate | [Codex §17](../22%20feb%20codex/THESIS_COMPLETION_AND_PIPELINE_ANALYSIS_22_FEB_2026.md) |
| MC-4 | Runtime profiling data not collected for any pipeline stage | E-10 experiment asks for this but no baseline exists | [File 24 P3-5](../docs/22Feb_Opus_Understanding/24_ASSUMPTIONS_GAPS_AND_OPEN_QUESTIONS.md) |
| MC-5 | ~~No dependency lock file~~ | ✅ **RESOLVED** — `config/requirements-lock.txt` created (578 pinned packages, Step 6f) | [File 27](../docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md) |
| MC-6 | No data checksums for the 26 session datasets | Can't verify data integrity across machines | [File 27](../docs/22Feb_Opus_Understanding/27_REPRODUCIBILITY_AND_AUDIT_CHECKLIST.md) |

---

## 7 Priority Action: Current State (Updated 22 Feb 2026)

The critical blockers are now all resolved. Here is the updated priority order:

**Already DONE (Steps 1-6 + CI/Docker fix):**
- ~~Fix CR-2 (`is_better=True`)~~ ✅ Real registry comparison live
- ~~Fix CR-3 (placeholder zeros)~~ ✅ All 4 trigger inputs wired to real monitoring
- ~~Fix CR-1 (14 stages)~~ ✅ All 14 stages orchestrated
- ~~Fix CR-6 (CI stubs)~~ ✅ CI is GREEN (commit `7f892d8`)

**Still to do — in this order:**
1. **Answer Q-3 (submission deadline)** → scopes everything below
2. **Answer Q-4 (GPU access)** → determines experiment depth
3. **Run Step 7 experiments** → Chapter 5 cannot be written without this
4. **Answer Q-1 (primary adaptation method)** → focus ablation comparisons
5. **Address CR-5 (Prometheus/Grafana)** → frame as "deployment-ready config" in thesis, or wire it (Optional O-9)

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

---

## 9 🔴 Questions to Ask Your Supervisor (Mentor Meeting)

> These questions cannot be resolved from the codebase alone — they require your supervisor's decision or your university's rules. Bring these to your next meeting. Answers directly change the thesis scope.

### 9.1 Deadline and Scope

| # | Question | Why It Matters | Impact If Not Clarified |
|--:|---------|--------------|------------------------|
| **M-1** | **What is the exact submission deadline?** | 8 weeks → full experiment suite feasible. 6 weeks → compressed. 4 weeks → cut optional tasks immediately. | Cannot plan weeks 1-8 without this |
| **M-2** | **How many pages is the thesis expected to be?** (German CS master's: typically 60-100 pages) | Current plan writes ~90-100 pages across 7 chapters. If limit is 60 pages, cut Chapter 2 depth. | Affects chapter length targets |
| **M-3** | **Is the abstract required in both German (Zusammenfassung) and English?** | Adds ~250 words + translation effort. | Affects Chapter 7 scope |

### 9.2 Data and Experiments

| # | Question | Why It Matters | Impact If Not Clarified |
|--:|---------|--------------|------------------------|
| **M-4** | **Does the supervisor expect experiments on the full 26 sessions, or is a subset acceptable?** | Full sweep needs ~8-12h GPU time. If 5 sessions are sufficient, experiments can be done on CPU. | Scopes Step 7 dramatically |
| **M-5** | **Is there access to university GPU cluster / cloud GPU (Colab Pro, Kaggle, Azure ML)?** | Full 5-method ablation on 26 sessions × 5 adaptations takes days on CPU. | Without GPU, only subset experiments are realistic |
| **M-6** | **Which public dataset is this? (WISDM / PAMAP2 / UCI HAR?)** | Dataset attribution is needed for citations and reproducibility section. The audit calls it "26-session IMU benchmark" but never names it explicitly. | Missing citation in Chapter 2 |

### 9.3 Methodology and Claims

| # | Question | Why It Matters | Impact If Not Clarified |
|--:|---------|--------------|------------------------|
| **M-7** | **Is the proxy metric evaluation approach (confidence drop ↔ accuracy drop) acceptable without labeled production data?** | This is the core thesis claim. The examiner will probe it (EQ-1). If supervisor requires labeled data, need to set aside 2-3 labeled sessions now. | Scopes experiment E-10 and Chapter 5.7 |
| **M-8** | **Should DANN/MMD be "removed" (framed as future work) or does the thesis need to compare 5 adaptation methods?** | DANN/MMD now raise `NotImplementedError`. Thesis claims 3 methods. If the rubric requires 5 comparisons, this becomes a 10-15h implementation task. | Changes thesis contribution claim |
| **M-9** | **Which adaptation method should be the primary thesis contribution?** (AdaBN / TENT / pseudo-label, or the AdaBN+TENT composition) | Determines which method gets most depth in Ch 3 and most ablation cells in Ch 5. | Focuses experiment design at Step 7d |

### 9.4 Observability and Infrastructure

| # | Question | Why It Matters | Impact If Not Clarified |
|--:|---------|--------------|------------------------|
| **M-10** | **What level of Prometheus/Grafana integration is expected?** Three options:  (a) Config exists = acceptable (current state)  (b) Live dashboard demo required for defense  (c) Just mention as future work | Currently CR-5: configs exist but never wired into running services. Option (b) = 4-8h extra. | Determines if Optional O-9 must become mandatory |
| **M-11** | **Does the defense require a live running demo, or is a video recording acceptable?** | If live demo: the Docker container must be tested end-to-end on defense hardware. If recording: lower risk. | Affects defense preparation (D-2) |

### 9.5 Data Leakage — High-Priority Verification

| # | Question | Why It Matters | Recommended Action |
|--:|---------|--------------|-------------------|
| **M-12** | **Was windowing done BEFORE or AFTER the train/val/test split?** If windowing happens before split, adjacent windows from the same session appear in both train and val folds → **data leakage → inflated accuracy**. | Our val_acc is 0.969 — suspicious if there is leakage. If leakage is confirmed, the entire accuracy claim needs to be recalculated. | Run `grep -n "window\|split\|fold" src/components/data_transformation.py` and check line order. Verify with supervisor before relying on accuracy numbers in Chapter 5. |

---

## 10 ⚠️ Genuine Concerns About the Thesis (Honest Assessment)

> These are things that, from a pure thesis-quality perspective, are risks worth addressing NOW rather than at the defense.

| # | Concern | Severity | Recommended Action |
|--:|---------|:--------:|-------------------|
| **W-1** | **Chapter 5 (Results) is completely empty.** No experiments at scale have been run. A1/A3/A4/A5 are single-session runs, not the full 26-session suite. | 🔴 HIGH | Start Step 7 immediately. This is the biggest blocking risk. |
| **W-2** | **Data leakage risk (D-3).** If windowing happens before CV fold split, `val_acc 0.969` may be inflated. | 🔴 HIGH | Verify in `src/components/data_transformation.py` before writing Chapter 5. |
| **W-3** | **F1 0.814 vs val_acc 0.969.** A gap this large (15pp) suggests class imbalance — one or more classes are being mis-classified. An examiner will ask about this immediately. | 🟡 MEDIUM | Run per-class confusion matrix. Identify which class has low recall. Add a brief explanation in Chapter 5. |
| **W-4** | **A2 (AdaBN-only) audit run has never been done.** The ablation table has a missing cell. Cannot compare all 3 methods head-to-head. | 🟡 MEDIUM | Run `python run_pipeline.py --retrain --adapt adabn --skip-ingestion` (1 hour). |
| **W-5** | **Prometheus/Grafana (CR-5) is config-only.** If the defense requires a live monitoring dashboard, it cannot currently be demonstrated. | 🟡 MEDIUM | Either wire it (Optional O-9, 4-8h) or explicitly frame it as "deployment-ready configuration" in Chapter 4.7 Known Limitations. |
| **W-6** | **The thesis title references "continuous monitoring" but the actual monitoring is batch-triggered, not continuous.** An examiner may probe the word "continuous". | 🟡 MEDIUM | Clarify in Ch 3.1: "continuous" means every inference session is monitored; it does not mean real-time stream processing. |
| **W-7** | **12 Citation TODOs unresolved.** BibTeX entries incomplete for 12 in-code references. | 🟢 LOW | Resolve during thesis writing (Step 9i). 3-4 hours. |
| **W-8** | **Runtime profiling data (E-10) has no baseline.** The proxy-metric correlation experiment requires per-session timing data that was never collected. | 🟢 LOW | Either collect during Step 7a batch run, or remove E-10 from experiment plan and frame as future work. |

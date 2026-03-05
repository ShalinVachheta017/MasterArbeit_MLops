# Questions for Mentor — HAR MLOps Thesis
## February 2026 — Shalin Vachheta

**Thesis Title:** Developing a MLOps Pipeline for Continuous Mental Health Monitoring Using Wearable Sensor Data  
**Prepared:** 24 February 2026  
**Pipeline Version:** 2.1.0 | **Stage:** 14-stage orchestrated pipeline (1D-CNN-BiLSTM)  
**Status at time of writing:** ~97% code complete — thesis writing phase beginning

> **How to use this document:**  
> Questions are grouped by topic and ordered Basic → Intermediate → Advanced within each section.  
> Each question is grounded in a concrete finding from the codebase, CTO review, or thesis engineering log.  
> Please answer, redirect, or confirm scope for each one.

---

## Table of Contents

1. [Threshold Setting & Monitoring Calibration](#1-threshold-setting--monitoring-calibration)
2. [Scalability — What It Means and How to Prove It Offline](#2-scalability--what-it-means-and-how-to-prove-it-offline)
3. [Prognosis Model — Scope, Design & Integration](#3-prognosis-model--scope-design--integration)
4. [Online Monitoring — Definitions, Validity & Design Choices](#4-online-monitoring--definitions-validity--design-choices)
5. [Model Architecture, Training & Evaluation Integrity](#5-model-architecture-training--evaluation-integrity)
6. [Data, Preprocessing & Domain Shift](#6-data-preprocessing--domain-shift)
7. [Adaptation Methods — AdaBN, TENT & Pseudo-Labeling](#7-adaptation-methods--adabn-tent--pseudo-labeling)
8. [CI/CD, Deployment Maturity & Automation](#8-cicd-deployment-maturity--automation)
9. [Thesis Scope, Results & Defense Preparation](#9-thesis-scope-results--defense-preparation)
10. [Future Work — Clarifications Before Writing](#10-future-work--clarifications-before-writing)

---

## 1. Threshold Setting & Monitoring Calibration

> **Context:** The 3-layer post-inference monitor uses several thresholds.
> Some are hardcoded defaults; some are literature-derived; one (z-score drift = 0.75) was
> computed data-driven from N=24 training sessions. There are also two parallel monitor
> implementations (pipeline script vs. API inline) with different values.

**Q1.1 (Basic)**
The confidence warning threshold is currently set to **0.60** in the API monitor and **0.50** in the pipeline monitor. Neither value was validated against this model's actual calibration curve (ECE). Is a literature-cited default acceptable for the thesis, or must I run a calibration experiment and choose the threshold from the resulting reliability diagram?

**Q1.2 (Basic)**
The temporal transition-rate threshold is hardcoded at **50%** (i.e., FLAG if more than half of consecutive predictions change activity). This value was not derived from any data or paper — it was chosen as a "permissive" default. Is this acceptable for a thesis prototype, or is there a published baseline for reasonable activity-switching rates in anxiety-behavior HAR that I should cite?

**Q1.3 (Intermediate)**
The z-score drift threshold of **2.0σ** follows the standard 95th-percentile null-distribution argument (Gama et al. 2014, Page 1954 CUSUM). However, the pipeline aggregates drift across **6 sensor channels simultaneously**, meaning the aggregate z-score is structurally larger than a single-channel value. We already discovered this problem for PSI (textbook 0.10 → recalibrated 0.75 after N=24 sessions). Should I run the same empirical recalibration for z-score drift and report it in the thesis, or is the 2σ rule defensible as a conservative bound even after aggregation?

**Q1.4 (Intermediate)**
We have two parallel implementations of the 3-layer monitor with *different* threshold values:

| Layer | Pipeline Script | API Inline |
|---|---|---|
| Confidence warn | 0.50 | 0.60 |
| Uncertain % | 10% | 30% |
| Z-score drift | 0.75 | 2.0 |
| Transition rate | 50% | 50% |

This inconsistency was flagged as a risk in the CTO review. For the thesis, which should be the **canonical** set of thresholds? Should I unify them to a single `PostInferenceMonitoringConfig` and document why those specific values were chosen?

**Q1.5 (Advanced)**
Is there a principled, experiment-driven method — for example, injecting synthetic drift of known magnitude into held-out sessions, varying the threshold, and measuring (false trigger rate, detection delay) — that would make the threshold choices a **publishable contribution** rather than a heuristic? Or is that level of threshold sensitivity analysis out of scope for this thesis?

**Q1.6 (Advanced)**
Temperature scaling (Stage 11) rescales softmax probabilities before monitoring. If the calibrated temperature $T \neq 1$, the confidence thresholds derived from raw softmax outputs are no longer directly applicable. Should all threshold derivations be re-done on post-calibration probabilities, and should the thesis explicitly state this dependency?

---

## 2. Scalability — What It Means and How to Prove It Offline

> **Context:** The CTO review classifies the system as **not yet "Scalable"** under the
> Amazon MLO    ps maturity model. The stack runs on a single machine via Docker Compose;
> no Kubernetes, no horizontal scaling, and no benchmarked latency exists.
> The p95 ≤ 250 ms latency figure is a design *target*, not a measured SLO.

**Q2.1 (Basic)**
What exactly does "scalability" mean in the evaluation context of this master's thesis? Are the examiners expecting evidence of:  
(a) dataset size scalability — the pipeline handles more sessions without code changes;  
(b) computational scalability — wall-clock time grows predictably with data volume;  
(c) architectural scalability — the design supports horizontal scaling even if not deployed;  
(d) a combination of the above?

**Q2.2 (Basic)**
We currently have **26 lab sessions** from Garmin wearables. The batch processing script runs inference on
 all 26 in sequence. Is testing performance on these 26 sessions — and reporting mean inference time per session — sufficient to make a scalability claim? If not, how many sessions would be "enough"?

**Q2.3 (Intermediate)**
Since Kubernetes is explicitly out of scope (stated in the thesis plan and CTO review), what is the **minimum viable scalability evidence** acceptable in an academic MLOps thesis? For example:  
- a timing benchmark (batch=1 vs batch=8 vs batch=32 at inference);  
- a linearity plot (inference time vs number of windows);  
- a theoretical architecture diagram showing where horizontal scale points would be inserted?

**Q2.4 (Intermediate)**
The latency target is p95 ≤ 250 ms per inference window at batch=1 on CPU. There is **no benchmarked measurement** of this in the repository. Should I run a formal timing experiment (e.g., 1000 inference calls, report p50/p90/p95/p99 with hardware specs) and add it to the thesis? If yes, is this for the discussion chapter or the results chapter?

**Q2.5 (Intermediate)**
The DVC remote is local (`../.dvc_storage`). For the thesis to claim "reproducible at scale," is it sufficient to state *"the pipeline supports cloud-backed DVC (S3/GCS/Azure) by changing one config line"* without actually running it, or must I demonstrate a live cloud sync?

**Q2.6 (Advanced)**
For the offline retraining path — AdaBN, TENT, and pseudo-labeling — how should I measure and report **computational overhead** per additional subject? Is there a standard metric from the continual learning / incremental learning literature (e.g., memory footprint growth, wall-clock time per new domain) that I should follow? Or is a simple "retraining time vs number of sessions" plot sufficient?

**Q2.7 (Advanced)**
The trigger-policy state (`logs/trigger/trigger_state.json`) is a local file. For a scaled deployment where multiple nodes could run inference simultaneously, the trigger state would need to be stored externally (S3, database, etc.). Should I discuss this architectural gap explicitly in the thesis as a known limitation, or implement a minimal proof-of-concept (e.g., serialize trigger state to an S3 mock using `moto`) to demonstrate the pattern?

---

## 3. Prognosis Model — Scope, Design & Integration

> **Context:** The original Thesis Plan (Month 4, Weeks 13–14) specifies a **second downstream
> "prognosis model"** that receives aggregated activity outputs from the HAR model as input.
> No prognosis model is currently implemented in the codebase. The relationship between the
> HAR model and any prognosis or clinical-outcome model is completely unspecified in code.

**Q3.1 (Basic)**
What exactly is the prognosis model supposed to predict? A clinical risk score, a diagnostic label category, a trend alert (e.g., "anxiety behaviors increasing over 7 days"), or something else? Where does the ground truth for this model come from — clinician labels, self-report, or something else?

**Q3.2 (Basic)**
Is the prognosis model **within the mandatory scope** of this thesis at this stage, or has it become a future-work item? The current codebase has no implementation at all. Should it at minimum be a placeholder pipeline stage with a documented interface, or can Chapter 6 (Discussion) mention it only as future work?

**Q3.3 (Basic)**
What is the intended **input format** to the prognosis model? Options include:  
(a) a raw per-window activity label sequence from a session;  
(b) an aggregated feature vector — e.g., percentage of time spent in each of the 11 activity classes per session;  
(c) temporal patterns — e.g., frequency transitions, sequential bursts of the same behavior;  
(d) a combination. Which approach was discussed in your thesis specification?

**Q3.4 (Intermediate)**
If the prognosis model is a **rule-based system** (e.g., "if nail-biting > 30% of session → elevated anxiety risk"), does it still require MLOps infrastructure (versioning, monitoring, retraining policy) — or is it outside the MLOps scope because there are no learnable parameters to adapt?

**Q3.5 (Intermediate)**
The thesis plan says the data flow from HAR → prognosis model "might involve storing intermediate results in a simple database or file system." Has a specific storage strategy been agreed — CSV files, SQLite, a REST API call? This choice affects the pipeline architecture diagram in Chapter 3.

**Q3.6 (Intermediate)**
If the prognosis model IS a learned model (not rule-based), how should its monitoring be designed? The HAR model can use proxy signals (confidence, transition rate, drift) without ground truth. Prognosis labels (clinical outcomes) are even rarer and delayed. What monitoring strategy is recommended, and is there literature I should cite?

**Q3.7 (Advanced)**
You mentioned **"coral"** or a "coral prognosis model" in our discussion. Could you clarify whether this refers to:  
(a) **CORAL** — CORrelation ALignment, a domain adaptation technique (Sun & Saenko 2016) that aligns feature covariance between source and target domains;  
(b) a specific clinical tool or scoring system called "coral" used in your research group;  
(c) something else entirely?  
If it is CORAL (domain adaptation), should it be applied to the HAR model, the prognosis model, or both — and how should I evaluate whether CORAL alignment improves generalization given the limited labeled outcome data?

**Q3.8 (Advanced)**
If the HAR model produces activity predictions that feed a prognosis model, should **monitoring of the prognosis model** be coupled to monitoring of the HAR model (i.e., if HAR drift is detected, assume prognosis inputs are also corrupted), or should they have independent monitoring trigger policies?

---

## 4. Online Monitoring — Definitions, Validity & Design Choices

> **Context:** The pipeline currently operates in **batch/offline mode** — raw CSV files are uploaded
> through the FastAPI endpoint or processed by the pipeline script. No live BLE streaming
> from Garmin devices is implemented. The 3-layer monitoring runs after each batch upload.

**Q4.1 (Basic)**
What is the definition of "online monitoring" in this thesis? Does it mean:  
(a) **real-time per-window inference** — each 4-second window is classified and checked immediately;  
(b) **batch-level online monitoring** — each uploaded file is classified and checked before returning results;  
(c) **continuous system monitoring** — Prometheus/Grafana scraping metrics from a live service;  
or a combination?

**Q4.2 (Basic)**
The current pipeline detects drift and flags a WARNING, but **does not automatically retrain** — the `--retrain` flag must be run manually. Is a manually triggered retraining loop sufficient for a master's thesis, or is a closed-loop automated retrain-and-deploy cycle expected by the examiners?

**Q4.3 (Basic)**
The Prometheus metrics are **defined** in `src/prometheus_metrics.py` but are **not being scraped** — there is no live Prometheus service in the Docker Compose stack. The Grafana dashboard JSON exists but is not deployed. Does the thesis need these to be live and demonstrated, or is the architecture design + code evidence sufficient for the thesis claims?

**Q4.4 (Intermediate)**
For the label-free monitoring claim — tracking mean confidence, uncertain prediction %, transition rate, and z-score drift without ground truth — is adding **PSI (Population Stability Index)** and/or **MMD (Maximum Mean Discrepancy)** a requirement, or are the four current signals sufficient to justify "label-free operability"? We already have Wasserstein distance implemented in Stage 12 (advanced flag).

**Q4.5 (Intermediate)**
Is it valid in the thesis to **simulate online monitoring** by replaying the 26 held-out session files sequentially through the API endpoint — one at a time — as a substitute for a real live data stream? What conditions make this simulation scientifically defensible, and how should I describe the simulation protocol in Chapter 4?

**Q4.6 (Intermediate)**
The 3-layer monitoring uses a **"2-of-3 vote" trigger policy**: at least 2 of the 3 layers must fire simultaneously before escalating to a CRITICAL retrain trigger. Is there a theoretical justification for majority voting over a single-layer trigger (e.g., fault-tolerant systems literature, ensemble anomaly detection), or should I empirically demonstrate the false-positive reduction it achieves on the simulated session replay?

**Q4.7 (Advanced)**
Layer 2 monitoring (temporal consistency / transition rate) is **HAR-domain-specific**: it exploits the physical constraint that humans do not switch activities every 4 seconds. Is there published work specifically on domain-informed monitoring signals for wearable-sensor HAR systems that I should cite to justify this design choice over a generic statistical drift detector like **ADWIN** or **DDM**?

**Q4.8 (Advanced)**
The **cooldown period** in the trigger policy (after a retrain is fired, the policy waits before allowing another trigger) is implemented but the cooldown duration is a hardcoded default. What is the recommended way to choose this value — empirically from drift event inter-arrival times in the training data, or from a SLO argument (e.g., "retraining takes X hours, so cooldown must be at least X + buffer")?

**Q4.9 (Advanced)**
The CTO review flags **Risk R13**: proxy metrics (confidence/entropy/drift) can fire false alarms when the model is actually performing correctly (or fail to fire when it is degrading). For the thesis, should I include a **labeled audit validation** — where I periodically apply the monitoring proxy signals to sessions with known ground-truth labels and measure precision/recall of the monitoring alarms — to validate the proxy signal framework? If yes, which sessions in our dataset have labels suitable for this?

---

## 5. Model Architecture, Training & Evaluation Integrity

> **Context:** Training uses 5-fold `StratifiedKFold` on shuffled windows — NOT subject-wise.
> Windows from the same user can appear in both train and validation folds, potentially
> inflating accuracy figures. The current mean 5-fold val_accuracy ≈ 0.938, f1_macro ≈ 0.939.
> An earlier document cited 0.969 / 0.814 but these figures are not traceable to any MLflow run.

**Q5.1 (Basic)**
The training currently uses `StratifiedKFold` on shuffled **windows** — not on subjects. This means windows from the same user can appear in both train and validation folds. For a wearable HAR thesis, is this split strategy acceptable for the main results, or must I switch to **GroupKFold (subject-wise)** before submission?

**Q5.2 (Basic)**
The figures "val_acc 0.969 / F1 0.814" from an older planning document are **not present in any MLflow run** and therefore cannot be cited as evidence. The MLflow-traceable results are val_accuracy ≈ 0.938, f1_macro ≈ 0.939. Should these be the only performance figures in the thesis, with a clear note that earlier planning documents are superseded?

**Q5.3 (Intermediate)**
The model has two architecture versions: **v1** (499,131 parameters, 17 layers — matching the pretrained model) and **v2** (experimental, ~850K parameters). Should the ablation study in Chapter 5 compare v1 vs v2 alongside the adaptation strategies (AdaBN, TENT, pseudo-label), or is v2 out of scope for the evaluation?

**Q5.4 (Intermediate)**
The thesis recognizes **class imbalance** among the 11 activity classes (rare behaviors like nail-biting vs. ambient activities like sitting/standing). We use macro F1 as the primary metric, which treats all classes equally. Should I also report **per-class recall** for each of the 11 classes separately in Chapter 5, and set a minimum per-class recall threshold (e.g., ≥ 0.60) as part of the acceptance criteria?

**Q5.5 (Intermediate)**
We have **4 adaptation runs** documented in the WHATS_NEXT guide (A1 = baseline inference, A3 = supervised retrain, A4 = AdaBN+TENT, A5 = pseudo-label). Should these 4 runs form the **main ablation table** in Chapter 5? If the A4 TENT rollback experiment is not yet re-validated (it was flagged as needing a re-run), is it safe to include it in the thesis results, or should it be marked provisional?

**Q5.6 (Advanced)**
The `is_better` check during model registration **falls back to `True` for TTA runs** (AdaBN/TENT) because there is no labeled holdout to compare against. This means any adaptation can promote itself to current model without verified improvement. For the thesis evaluation, how should I validate that an adapted model is genuinely better — using post-deployment confidence stability, a small labeled audit set, or another proxy?

**Q5.7 (Advanced)**
The architecture achieves 93.8% validation accuracy using a window-level split. If I switch to **subject-wise (GroupKFold or LOSO)** evaluation, I expect the accuracy to drop (potentially significantly). How large a drop is acceptable for the thesis? Is there a minimum performance threshold we agreed on — and if the subject-wise result falls below it, does that require additional work (more data, better architecture) before submission?

---

## 6. Data, Preprocessing & Domain Shift

> **Context:** The most critical engineering fix in the project was discovering that production
> Garmin data was in **milliG** while the model was trained on **m/s²** — causing near-random
> 14.5% accuracy before the unit conversion fix. The preprocessing pipeline now auto-detects units.

**Q6.1 (Basic)**
The unit auto-detection uses a magnitude heuristic: if `median(√(Ax²+Ay²+Az²)) > 100`, classify as milliG and convert. This is validated on 26 sessions but not tested against edge cases (e.g., highly dynamic motion where the magnitude threshold may not hold). Should I add a hard assertion (e.g., `assert 8.0 < post_conversion_magnitude < 12.0 m/s²`) as a safeguard, and describe this assertion in the thesis as a data quality gate?

**Q6.2 (Basic)**
All 26 training sessions come from **controlled lab recordings** at a single site. The thesis acknowledges this but does it qualify as a "real-world" or "production" dataset? How should I characterize the generalizability of the results in Chapter 6 — what language is appropriate when the training and test data come from the same lab protocol?

**Q6.3 (Intermediate)**
The pseudo-label retraining fix introduced a **self-consistency filter** that retains only source windows where the pre-trained model agrees with the true label (14.5% of source windows retained). This is a significant reduction. Is this filter too conservative — am I discarding too much labeled training data? Is there a standard practice for calibrating this kind of confidence-based data selection threshold?

**Q6.4 (Intermediate)**
The data validation stage checks schema, range, and missing values. However, there is **no automated alert if no new data arrives for > N hours** (flagged as a P1 missing monitor in the CTO review). For the thesis discussion of real-world deployment, should I at minimum describe what a data freshness monitor would look like, even if it is not implemented?

**Q6.5 (Advanced)**
The PSI threshold recalibration (from textbook 0.10 → empirical 0.75 for multi-channel aggregated PSI) was discovered through a production bug. This suggests that **all statistical thresholds for this pipeline must be empirically calibrated** rather than taken from literature. Should this finding be a standalone contribution in the thesis methodology — described as a "multi-channel threshold recalibration protocol" — or is it better placed in the implementation challenges section?

**Q6.6 (Advanced)**
The dataset from Oleh & Obermaisser (2025) provides the 11-class activity labels. We rely completely on this dataset for training. Should the thesis include a **data contribution statement** acknowledging the dependency on this specific dataset, and address what would happen to the pipeline if the dataset's activity definitions or label boundaries changed in a future version?

---

## 7. Adaptation Methods — AdaBN, TENT & Pseudo-Labeling

> **Context:** Three test-time adaptation strategies are implemented and compared (A3–A5 audit runs).
> A critical bug was fixed: TENT was overwriting AdaBN's batch-norm running statistics,
> causing confidence to drop instead of improve. The fix snapshots and restores BN stats.

**Q7.1 (Basic)**
For the thesis comparison of AdaBN, TENT, and pseudo-labeling — what is the **baseline** that all three should be compared against? Is it:  
(a) the original pretrained model with no adaptation (A1);  
(b) a supervised fine-tune on the same data with labels (A3);  
(c) both?

**Q7.2 (Basic)**
The TENT rollback fires when **confidence drops more than 0.01** after adaptation (the gate threshold is `Δconf < −0.01`). This was chosen empirically. Is there a cited paper or principled argument for this specific rollback threshold, or should I describe it as an empirically tuned safeguard and explain how I would derive a better threshold with more data?

**Q7.3 (Intermediate)**
The TENT corruption of AdaBN batch-norm statistics was a **novel engineering failure** we encountered and fixed (commit `f47a48b`). The fix "snapshots BN running stats before each gradient step and restores immediately after." Is this interaction between AdaBN and TENT documented in any published work, or is this an original finding that should be stated as a contribution in the thesis?

**Q7.4 (Intermediate)**
**Elastic Weight Consolidation (EWC)** is referenced as a regularizer in Stage 13 (Curriculum Pseudo-Labeling). EWC requires computing a Fisher information matrix over a reference dataset. Is the EWC implementation in our pipeline using the correct Fisher computation, and should I include EWC vs. no-EWC as a separate ablation condition?

**Q7.5 (Intermediate)**
The pseudo-labeling loop only includes **high-confidence windows** (curriculum strategy). However, for rare activity classes (e.g., nail-biting, knuckles-cracking) the model may never generate high-confidence predictions, meaning those classes are never included in pseudo-label retraining. Does this create a class-collapse risk, and is there a strategy (e.g., class-balanced pseudo-label sampling) I should implement or at least discuss?

**Q7.6 (Advanced)**
The **self-consistency filter** in pseudo-labeling retains only 14.5% of labeled source windows. This means 85.5% of available labeled data is discarded before retraining. In a dataset as small as ours (26 sessions), this is a severe reduction. Should I run an ablation comparing pseudo-labeling with and without the self-consistency filter, to justify the filter's effect on final performance vs. data efficiency?

**Q7.7 (Advanced)**
The TENT implementation was found to be **10× slower than expected** due to `model.predict()` being called inside a gradient loop (causing TensorFlow graph retracing). The fix replaced `model.predict()` with `model(tf.constant(batch), training=False)`. Is this implementation detail thesis-worthy as a "TF-specific implementation challenge in test-time adaptation," or is it too implementation-specific and better placed only in the engineering log appendix?

---

## 8. CI/CD, Deployment Maturity & Automation

> **Context:** The CTO review assesses current maturity as **Google MLOps Level 1 / Microsoft Level 2**.
> Key gaps before Level 2/3: no automated retrain CI loop, no staging→production approval gate,
> Prometheus/Grafana not live, DVC remote local-only, and no MLflow Model Registry integration.

**Q8.1 (Basic)**
The thesis plan originally targeted deployment and CI/CD as Month 3 deliverables. The current status: Docker Compose and GitHub Actions CI are complete; but the blue-green/canary logic exists only in code — it is not wired to CI. For the thesis submission, is the **existence of this code** sufficient, or does a deployment workflow need to be demonstrated end-to-end?

**Q8.2 (Basic)**
The model registry `current_version` field is currently **null** — no model has ever been formally "deployed" into the registry. Should I run a promotion sequence (train → evaluate → register → promote) and document the output as part of Chapter 4 experimental evidence, or is a screenshot / log output OK?

**Q8.3 (Intermediate)**
The retraining loop is **manually triggered** by running `python run_pipeline.py --retrain`. The CTO review proposes an `auto-retrain.yml` GitHub Actions workflow that reads `trigger_state.json` and fires retraining automatically on CRITICAL. The review also identifies a blocker: `trigger_state.json` is a local file unavailable to CI runners. Should I:  
(a) implement a minimal solution (persist state to a DVC-tracked file or GitHub artifact);  
(b) describe the architecture of the fully automated loop in the thesis without implementing it;  
(c) implement it fully (with whatever external state store is feasible)?

**Q8.4 (Intermediate)**
The thesis documents **19 engineering bugs** in the THESIS_PROBLEMS_WE_FACED log, each with symptom, root cause, fix, commit hash, and thesis-ready lesson. Should all 19 be described in Chapter 4 (Implementation), or should only the top 8 "most impactful" fixes be featured, with the full log moved to an appendix?

**Q8.5 (Advanced)**
The DANN and MMD domain adaptation methods are referenced in the code but raise `NotImplementedError`. Should I:  
(a) implement at least DANN as an additional comparison in the ablation study;  
(b) remove them from the codebase to avoid examiner questions about incomplete code;  
(c) leave the stubs in place with a clear docstring that they are "planned future work"?

**Q8.6 (Advanced)**
The `is_better` check for TTA model promotion **defaults to `True`** when no labeled holdout exists. For a thesis that claims automated model governance, this is a significant assumption. What is the minimum proxy-validation strategy that would make this promotion gate scientifically defensible — e.g., requiring confidence stability + entropy reduction + holdout-set performance above a floor threshold before setting `is_better = True`?

---

## 9. Thesis Scope, Results & Defense Preparation

> **Context:** The CTO review flags R10 (evaluation incomplete) and R11 (data leakage / split
> integrity) as the two highest-priority risks before thesis submission. Chapter 5 tables
> and plots are not yet written. The thesis writing phase begins now (Feb–Mar 2026).

**Q9.1 (Basic)**
The CTO review identifies several NFRs as **"targets not yet measured"**: p95 latency ≤ 250 ms, availability ≥ 99.5%, rollback ≤ 5 min, false trigger ≤ 1/week. For the final thesis, which of these must become **measured experimental results** with hardware specs and reported numbers, and which can remain as **justified design targets** supported by architecture evidence?

**Q9.2 (Basic)**
What level of **statistical rigor** is expected for the experiments? Specifically:  
(a) Is a single 5-fold cross-validation run on one machine sufficient, or must I report mean ± std over multiple random seeds?  
(b) Are hypothesis tests (e.g., Wilcoxon signed-rank) required to compare adaptation strategies, or are descriptive statistics and a table of results OK?

**Q9.3 (Intermediate)**
The **security and privacy** section is entirely not implemented — no encryption at rest, no access control, no GDPR compliance. This is mental health / behavioral sensor data (health data). Does the thesis need to at minimum include a chapter section describing what a compliant implementation would require (GDPR Article 9 sensitive data, encryption at rest, access logs)? Or is "out of scope" with a single sentence acceptable?

**Q9.4 (Intermediate)**
The pipeline currently stores **model artifacts as local `.keras` files** in a JSON registry. The CTO review recommends MLflow Model Registry integration (staging → production stages, approval gates). Is the local JSON registry sufficient for the thesis, or is MLflow Model Registry integration a required deliverable before submission?

**Q9.5 (Intermediate)**
The README states "~70% of chapters remain" to be written. Given the February 24, 2026 date and the planned April 2026 submission, what is the **minimum viable thesis structure** that satisfies the examination requirements? Which chapters are most critical to complete first?

**Q9.6 (Advanced)**
The validation accuracy figures (0.938 window-level) may drop significantly when recomputed as **subject-wise (GroupKFold)**. If the subject-wise accuracy is, say, below 0.85 or macro F1 below 0.75, does this affect the thesis pass/fail decision? Is there a minimum performance threshold defined in the thesis specification, and what is the fallback strategy if results fall short?

**Q9.7 (Advanced)**
The thesis claims contributions at three levels: (1) the 14-stage pipeline architecture, (2) the 3-layer label-free monitoring framework, (3) the AdaBN+TENT composition fix. For the **thesis defense**, which of these is expected to be the primary original contribution — and should the other two be framed as implementation or engineering contributions rather than scientific novelty?

**Q9.8 (Advanced)**
All `file.py#L<line>` evidence links in the CTO review use current line numbers, which will drift as code evolves. Before the final submitted thesis references the repository, should I freeze a **submission commit SHA** and replace all file links with commit-pinned GitHub permalinks (e.g., `github.com/.../blob/<SHA>/src/train.py#L397`)? When should this freeze happen?

---

## 10. Future Work — Clarifications Before Writing

> **Context:** These are items from the CTO roadmap, WHATS_NEXT guide, and engineering log
> that are currently out of scope but may appear in Chapter 6 (Discussion / Future Work).
> Clarification is needed to know whether any of them must be moved to "current work."

**Q10.1 (Basic)**
The roadmap item **"Add subject-wise evaluation (GroupKFold)"** is marked P0 Thesis Critical. Does this mean it must be a completed experiment in Chapter 5, or can it appear only as a limitation and future-work recommendation in Chapter 6?

**Q10.2 (Basic)**
The roadmap lists **"Add labeling queue for unlabeled production (human-in-the-loop audit)"** as P1. Should I implement this as a working pipeline stage with code, or is describing the architecture sufficient for the thesis? Is the labeling queue part of the submitted deliverables?

**Q10.3 (Intermediate)**
The Wasserstein drift detection (Stage 12) and Curriculum Pseudo-Labeling (Stage 13) are gated behind an `--advanced` flag and **not included in default CI runs**. Should the thesis evaluation include results with `--advanced` on, or only the default 10-stage pipeline? If `--advanced` is omitted, how do I frame the Wasserstein and pseudo-labeling work — as implemented but unevaluated features?

**Q10.4 (Intermediate)**
The **OOD (Out-of-Distribution) detection** module (`src/ood_detection.py`, energy-based scoring) is implemented but not explicitly evaluated in any of the current audit runs (A1–A5). Should an OOD evaluation be added to the thesis results, or is it legitimate to describe OOD detection as "implemented and unit-tested, full evaluation deferred to future work"?

**Q10.5 (Advanced)**
The **sensor-placement analysis** (Stage 14 — wrist placement robustness, mirrored features) is behind the `--advanced` flag. Given that the Garmin is always worn on the wrist, is Stage 14 still scientifically relevant for this particular dataset? Should it remain in the pipeline with documentation, or should it be removed to reduce examiner questions about features that cannot be evaluated with the available data?

**Q10.6 (Advanced)**
The CTO review identifies that the pipeline is at **Google MLOps Level 1 / approaching Level 2**, with Level 2 requiring fully automated retraining triggered by drift signals and automated deployment gates. Is it acceptable for the thesis to **describe and design** the Level 2 architecture (CI workflow, trigger-state persistence, staging gates) without fully implementing it — presenting it as an evaluated design contribution rather than a running system?

---

## Summary Checklist for Mentor Response

Please confirm or redirect the following before the next meeting:

| # | Item | Needed Response |
|---|------|----------------|
| ✦ | Subject-wise GroupKFold evaluation | → Must do now / Can defer to future work |
| ✦ | Threshold calibration method | → Literature citation OK / Must run sensitivity analysis |
| ✦ | Prognosis model scope | → Must implement stub / Future work only |
| ✦ | "Coral" method clarification | → Name of method / tool / other |
| ✦ | Scalability evidence minimum | → Timing benchmark / Architecture diagram / Other |
| ✦ | Prometheus/Grafana live requirement | → Must be live / Architecture evidence sufficient |
| ✦ | Automated retrain loop requirement | → Must implement / Design description sufficient |
| ✦ | Minimum performance threshold | → Confirmed threshold for subject-wise F1 |
| ✦ | Security/privacy section length | → One sentence / Dedicated section required |
| ✦ | DANN/MMD stubs | → Implement / Remove / Leave as future work |

---

*Document prepared by GitHub Copilot from codebase analysis — 24 February 2026.*  
*All questions grounded in: `docs/PRODUCT_OWNER_CTO_REVIEW.md`, `Thesis_report/docs/THESIS_PROBLEMS_WE_FACED.md`, `Thesis_report/docs/WHATS_NEXT.md`, `Thesis_report/docs/thesis/chapters/CH3_METHODOLOGY.md`, `Thesis_report/Thesis_Plan.md`, and pipeline source code.*

---

---

# Simple English Version — Questions for Mentor (February 2026)

> **Note:** Same questions as above, written in simple and easy language.  
> Each topic is combined into one question so it is quick to read and easy to discuss.  
> Please give your feedback on each point.

---

### Q1 — About the Warning Numbers (Thresholds) in our Monitoring System

Our system uses numbers to decide when something is going wrong — for example, "if the model's confidence drops below 0.60, show a warning." But we have two places in the code with different numbers (one says 0.50, the other says 0.60), and most of these numbers were just copied from papers or chosen as a guess — they were never properly tested on our actual model.

**Can you help us understand:**
- Is it okay to use numbers from papers without testing them on our own data, or do we need to run experiments to find the right numbers?
- Our system checks 6 sensor channels at the same time, which makes the drift score bigger than usual. Do we need to recalculate a proper warning number for that, or is the standard value (2σ) still fine?
- Which set of numbers should we use as the "official" ones — the pipeline version or the API version — and do we need to make them the same everywhere?

---

### Q2 — About Scalability: What Does It Mean and How Do We Show It?

Our whole system runs on one computer using Docker. We have never tested it with more than 26 sessions, and we have never measured how fast it actually is (our target is under 250ms per prediction, but we have not checked this yet).

**Our questions are:**
- What exactly does "scalable" mean for a master's thesis like ours? Does it mean the system works with more data, or runs faster if we add more machines, or just that the design *can* be scaled up later?
- Do we need to actually run a speed test and write down the numbers (like "99% of predictions finished in under 250ms"), or is showing the design on a diagram enough?
- Since we only have 26 sessions, is that enough to say the pipeline is scalable — and if not, what is the minimum we should test?
- For the retraining part (AdaBN, TENT, pseudo-labeling), how do we measure and show that it can handle more users over time?

---

### Q3 — About the Prognosis Model (Second Model)

The original thesis plan mentions a second model that takes the output of our activity recognition model and uses it to make a health prognosis (some kind of risk or outcome prediction). Right now, this model does not exist anywhere in the code.

**We need clarity on:**
- What is the prognosis model supposed to predict exactly — a risk level, a trend (like "anxiety is getting worse over 7 days"), or something else? And where does the training data (labels/outcomes) come from for this model?
- Is implementing this model still required for the thesis, or has it become a "future work" item? If it is required, do we at least need to build a basic placeholder stage with a clear input/output definition?
- You mentioned something called **"coral"** — can you please clarify what this means? Is it the CORAL method (a technique that aligns data from different users to reduce differences), a clinical scoring system, or something else? And if it is a method, where in the pipeline should it be applied?
- If the prognosis model gets bad HAR predictions as input (because the HAR model is drifting), should they share one monitoring system or have separate ones?

---

### Q4 — About Online Monitoring

Currently our system is not truly "online" — data is uploaded as files through our API, not streamed live from the Garmin device. Monitoring runs after each file upload. Prometheus and Grafana (the real-time dashboards) are set up in code but not actually running.

**Our questions are:**
- When you say "online monitoring" in the thesis context, do you mean: (a) checking each 4-second window instantly as it is collected, (b) checking each uploaded file before returning results, or (c) a live dashboard showing system health in real time? What level do we need to demonstrate?
- Right now, if our system detects drift, it shows a warning — but retraining still has to be started manually by a person. Is a manual trigger enough for the thesis, or do we need to show a fully automatic "detect drift → retrain → deploy" loop?
- Does Prometheus/Grafana need to actually be running with live data, or is it enough that the code and config are ready and we explain how it would work?
- We use 4 signals to detect problems without labels (confidence, prediction stability, activity-switching rate, and drift score). Are these 4 enough to claim "label-free monitoring," or do we need to add more methods like PSI or MMD?
- Is it acceptable to simulate the online scenario by sending our 26 saved session files through the API one by one (as if they were coming in live)? What conditions make this simulation valid?

---

### Q5 — About Model Training, Results and Fair Evaluation

We trained the model using 5-fold cross-validation, but the split was done on individual windows — not on users. This means data from the same person appears in both training and testing, which makes the accuracy look better than it really is. Our current best result is about 93.8% accuracy and 93.9% F1 score.

**Our questions are:**
- Do we need to redo the evaluation using a proper user-based split (GroupKFold or Leave-One-Subject-Out), where no user's data appears in both train and test? If yes, must this be done before submission?
- If the accuracy drops a lot when we do the user-based split (for example, below 85%), does that affect whether the thesis passes? Is there a minimum performance level we agreed on?
- We have an older document that says 96.9% accuracy and 81.4% F1 — but these numbers are not in any experiment log and cannot be verified. Should we only use the verified numbers (93.8% / 93.9%) in the thesis and clearly say the older numbers are outdated?
- Some activity classes are rare (like nail-biting). Should we report the accuracy for each of the 11 classes separately so we can see if the model is bad at recognizing rare behaviors?
- We ran 4 comparison experiments (A1: no adaptation, A3: supervised retrain, A4: AdaBN+TENT, A5: pseudo-label). Can these 4 be the main results table in Chapter 5, and do we need to add statistical tests (like Wilcoxon) to compare them, or is a simple table enough?

---

### Q6 — About Data Quality and the Unit Bug We Fixed

We discovered that the Garmin sensor data was in a different unit than what the model was trained on (milliG vs m/s²). This caused the accuracy to drop to 14.5% — nearly random guessing. We fixed it by automatically detecting and converting the unit, which brought accuracy back to normal. This was the most important bug we fixed.

**Our questions are:**
- Should we add a hard safety check after the unit conversion (like "assert gravity magnitude is between 8 and 12 m/s² at rest") so that future wrong-unit data is caught immediately? And should this be described in the thesis as a data quality gate?
- All 26 sessions come from one controlled lab. How should we describe the limits of our results — can we say it works "in general," or should we specifically say it was only tested in lab conditions and may not work in the real world?
- We found that standard textbook drift thresholds (PSI = 0.10) were wrong for our multi-channel pipeline and had to be recalibrated (to PSI = 0.75). Should this finding be a proper contribution in the thesis, or just mentioned in the "problems we faced" section?

---

### Q7 — About the Adaptation Methods (AdaBN, TENT, Pseudo-Labeling)

We implemented three ways to adapt the model to new users without labeled data: AdaBN (adjusts normalization statistics), TENT (minimizes prediction entropy at test time), and Pseudo-Labeling (uses the model's own high-confidence predictions as training data). We found and fixed a serious bug: TENT was accidentally overwriting AdaBN's settings, making things worse instead of better.

**Our questions are:**
- The fix we made — where TENT snapshots and restores AdaBN's batch normalization statistics before each gradient step — is not described in any paper we know of. Should we present this as an original finding/contribution in the thesis?
- The pseudo-labeling only keeps windows where the model is very confident — but for rare classes like nail-biting, the model may never be confident enough, so those classes never appear in retraining. Is there a risk the model gradually forgets rare classes (class collapse)? What should we do about this?
- We only kept 14.5% of the labeled source data after our self-consistency filter. This is a big loss of training data on a small dataset. Should we run a test comparing pseudo-labeling with and without this filter to show whether the filter actually helps?
- The TENT method was 10x slower than expected because of a TensorFlow graph retracing issue inside the gradient loop. We fixed it. Is this kind of TensorFlow-specific implementation detail worth mentioning in the thesis, or only in the appendix?

---

### Q8 — About Automation, Deployment and How Complete the System Is

The system can detect drift, decide to retrain, and register a new model — but the step of automatically triggering retraining through a CI/CD pipeline is not yet wired up. The deployment logic (blue-green / canary) exists in code but is not connected to the CI workflow. The model registry shows `current_version = null` — no model has ever been formally deployed through the registry.

**Our questions are:**
- Do we need a fully working automatic retrain-and-deploy loop for the thesis, or is it enough to show the design (workflow diagram + working code pieces) and explain what is missing and why?
- Should we run the model promotion sequence at least once (train → evaluate → register → promote to `current_version`) and include the log as evidence in Chapter 4?
- We have two domain adaptation methods (`DANN` and `MMD`) in the code that are not implemented — they raise `NotImplementedError`. Should we implement one of them, remove them, or leave them as clearly labeled "planned future work"?
- We documented 19 engineering bugs with root cause, fix, and commit evidence. How many of these should appear in the thesis — all 19 in the appendix, or only the most important ones (like the unit bug, TENT corruption fix, and PSI threshold recalibration) in the main text?

---

### Q9 — About the Thesis Itself: What Is Still Needed Before Submission

Chapter writing has not started yet. The model's performance on subject-wise evaluation is unknown. Several performance targets (latency, availability, rollback time) are design goals that have never been measured.

**Our questions are:**
- Which of our performance targets must become real measured numbers before submission (for example, running a proper speed test for the 250ms latency claim), and which can remain as design targets with architecture diagrams as evidence?
- What is the minimum structure the thesis needs to pass the examination — which chapters are most important to finish first given the April 2026 deadline?
- Our thesis handles sensitive mental health and behavioral sensor data. Does Chapter 6 need a proper section explaining what GDPR compliance and data security would require in a real deployment, or is one sentence saying "security is out of scope" sufficient?
- The three main contributions of this thesis are: (1) the 14-stage pipeline design, (2) the 3-layer label-free monitoring framework, (3) the AdaBN+TENT composition fix. Which of these is the **primary scientific contribution** that the thesis defense should focus on? Are the other two engineering contributions or also scientific novelty?
- Before submitting the thesis PDF, should we freeze the code at a specific Git commit and replace all file references in the thesis with commit-pinned links (so they never break)? When is the right time to do this?

---

### Q10 — About Future Work: What Must Be Done Now vs. Later

Some items in our roadmap are marked as important but are not yet done. We need to know whether they must be completed now or can be moved to "future work" in Ch. 6.

**Our questions are:**
- **Subject-wise evaluation (GroupKFold):** This is marked as P0 (most critical). Must this be a completed experiment in Chapter 5 with real numbers, or can it be a limitation and recommendation in Chapter 6?
- **Wasserstein drift (Stage 12) and Curriculum Pseudo-Labeling (Stage 13):** These only run with `--advanced` flag. Should the thesis evaluation include results with this flag on, or is it fine to describe them as "implemented and tested, full evaluation is future work"?
- **OOD (Out-of-Distribution) detection:** We built it but never evaluated it in any experiment. Can we say "implemented but evaluation deferred," or must we include at least one OOD evaluation run?
- **Sensor placement analysis (Stage 14):** The Garmin is always worn on the wrist. Is this stage still useful to evaluate with our dataset, or should we remove it and explain why in the thesis?
- **MLOps maturity level:** Our system is currently at Google Level 1 / Microsoft Level 2. Is it acceptable to *design and describe* the next level (automated retraining loop, staging gates) without fully building it — presenting it as an evaluated design contribution?

---

*End of simplified question list — February 2026.*

# HAR MLOps Thesis — Key Decisions Needed

**Date:** February 24, 2026  
**Status:** Pre-submission phase, thesis writing in progress  
**Final Submission Deadline:** June 30, 2026

---

## Executive Summary

This document lists the key decisions I need for my master's thesis. There are 34 questions in 10 topics, plus a short email draft you can answer first.

**How to use this document:**
- The **email template (Section 11)** contains 9 grouped decisions — this is what I plan to send
- Sections 1-10 contain the **detailed 34-question list** for deeper discussion if needed
- Each question includes my proposed approach so you can quickly confirm/correct

---

## 1. Monitoring Thresholds

### Q1: Minimum requirement
**Question:** For the monitoring thresholds (confidence, drift, transition rate), is it enough for the thesis if I (a) pick one consistent set of threshold values, and (b) justify them with a short experiment on our data, or do you expect a full sensitivity analysis (injecting synthetic drift etc.)?

**My proposal:** Unified config + simple verification experiments (e.g., synthetic drift injection on the 26 sessions)

---

### Q2: Canonical config
**Question:** Which version should be the official one for the thesis – the pipeline thresholds or the API thresholds – or should I unify them into a single config and document that as canonical?

**My proposal:** Unify into one canonical config file

---

## 2. Scalability and Performance

### Q3: What 'scalable' means here
**Question:** For this master's thesis, what counts as enough evidence of scalability: (a) timing benchmarks and simple plots from our 26 sessions, or (b) also an architecture discussion about how it would scale to multiple machines?

**My proposal:** Timing benchmarks + scaling curves + architecture discussion (without actual Kubernetes deployment)

---

### Q4: Latency target
**Question:** Do I need to actually measure latency (e.g., p50/p95 over many calls on my hardware) and report the numbers in the thesis, or can the 250 ms figure stay as a design target only?

**My proposal:** Run timing experiments and report p50/p90/p95 with hardware specs

---

## 3. Prognosis Model ⚠️ **CRITICAL - Need Your Guidance**

### Q5: What is the prognosis model?
**Question:** I realize I don't have a clear understanding of what the prognosis model should do in this thesis. Could you please explain: (a) What exactly should this model predict? (anxiety levels, mental health risk scores, or something else?), (b) What inputs should it take from the HAR model? (activity percentages, sequences, transitions?), and (c) What output format do you expect?

**My proposal:** I need your guidance on the scope and requirements

---

### Q6: Main thesis focus: MLOps + Prognosis ⚠️ **CRITICAL**
**Question:** I understand the main focus should be **MLOps + Prognosis**, with the technical implementation including HAR pipeline, monitoring, and test-time adaptation as supporting components. However, since the prognosis model is not yet implemented, how should I balance this in the thesis: (a) Should the prognosis model be a **co-equal main contribution** alongside MLOps infrastructure, or (b) Can it be documented as designed-but-not-implemented with clear future work?

**My proposal:** Position thesis as "MLOps pipeline for HAR with prognosis integration design" — emphasize the complete MLOps infrastructure (14-stage pipeline, 3-layer monitoring, CI/CD) and document the prognosis model interface/design clearly for future implementation

---

### Q7: Prognosis implementation priority ⚠️ **CRITICAL**
**Question:** Given the June 30, 2026 deadline and that I need to complete subject-wise evaluation, ablation studies, and thesis writing, what is the **priority level** for the prognosis model: (a) Must be fully implemented with results before thesis defense, (b) Design + clear interface is sufficient for submission, or (c) Can be documented as future work with architectural diagrams?

**My proposal:** Need your explicit guidance on whether prognosis implementation is required for thesis completion

---

### Q8: 'Coral' clarification
**Question:** When you mentioned 'coral', did you mean the CORAL domain adaptation method, a clinical scale, or something else — and where in the pipeline should it conceptually sit?

**My proposal:** Need your clarification to document this correctly

---

## 4. Online Monitoring & Automation

### Q9: Definition of 'online' for this thesis
**Question:** For this project, is it enough to simulate 'online' by replaying our 26 files one by one through the API and monitoring them, or do you expect true live streaming from the device?

**My proposal:** File-by-file replay through API as simulation of online monitoring

---

### Q10: Manual vs automatic retrain
**Question:** Is a manually triggered retraining loop (I run a command after a warning) acceptable for the thesis, or do you expect a fully automatic 'detect drift → retrain → deploy' CI/CD cycle?

**My proposal:** Manual triggering + thorough documentation of how full automation would work

---

### Q11: Prometheus/Grafana vs MLflow only
**Question:** Since MLflow already tracks experiments, model versions, and metrics, can all monitoring and tracking needs be satisfied with MLflow alone, or do you expect Prometheus/Grafana to be implemented as well? If Prometheus/Grafana are needed, should they be running live during evaluation, or is it enough to show the metric definitions and dashboard configs?

**My proposal:** Use MLflow for all tracking and monitoring; treat Prometheus/Grafana as optional extensions for production deployments (not required for thesis)

---

## 5. Evaluation Integrity and Metrics

### Q12: Evaluation approach and data splits
**Question:** For the evaluation, I understand from our previous discussion that strict subject-wise evaluation (ensuring no person appears in both train and test) is **not required** for this thesis. Should I continue with window-level K-fold cross-validation with labeled data for retraining experiments, and clearly document the limitations of this approach in the thesis?

**My proposal:** Use window-level StratifiedKFold for main experiments with labeled data; clearly state that subject-wise generalization is a limitation and future work

---

### Q13: Minimum acceptable performance
**Question:** With our current window-level evaluation approach (~93.8% accuracy, 0.939 macro F1 on labeled data), is there a minimum performance threshold you would consider necessary for the thesis to pass, especially after applying adaptation methods?

**My proposal:** Need your guidance on minimum acceptable performance levels

---

### Q14: Which results to report
**Question:** Can I ignore the older untraceable numbers (96.9% / 81.4%) and only report the MLflow-traceable results (~93.8% acc / 93.9% macro F1) in the thesis?

**My proposal:** Use only MLflow-traceable results, mark older numbers as obsolete

---

### Q15: Per-class metrics and statistics
**Question:** Do you expect per-class recall for all 11 classes and basic descriptive stats (mean ± std), or should I also run hypothesis tests (e.g., Wilcoxon) between adaptation methods?

**My proposal:** Per-class recall + descriptive stats; hypothesis tests only if you deem necessary

---

### Q16: Hyperparameter optimization requirement
**Question:** The original thesis plan (Month 2, Week 5-6) explicitly mentions "integrate automated hyperparameter optimization within the training loop." Right now, I mainly use a fixed configuration with experiment tracking in MLflow. Do you expect a **full HPO setup** (e.g., sweeps/grid search/Optuna) as a thesis requirement, or is a well-justified fixed configuration + reproducible runs sufficient?

**My proposal:** Use well-justified fixed configuration + document rationale; treat full HPO as future work

---

## 6. Data Quality & Generalization

### Q17: Unit bug and data checks
**Question:** Is it good practice (and thesis-relevant) to add a hard check after unit conversion (e.g., gravity between 8–12 m/s²) and describe it as a data quality gate?

**My proposal:** Implement safety check and document as data quality gate

---

### Q18: How to talk about generalization
**Question:** Since all 26 sessions are from one lab, how strongly should I (or should I not) claim real-world generalization in the thesis?

**My proposal:** Be explicit about single-lab limitation, frame generalization cautiously

---

### Q19: Drift detection method choice (PSI vs Z-score)
**Question:** For drift detection on unlabeled data, we are currently using **Z-score** to detect distribution shifts in streaming data. Should I compare this with PSI (Population Stability Index) and justify the choice, or is using Z-score alone with proper threshold calibration sufficient for the thesis?

**My proposal:** Use Z-score for unlabeled data drift detection; document threshold calibration for multi-channel sensor data as methodological contribution

---

## 7. Adaptation Methods

### Q20: AdaBN and TENT - Use separately or together?
**Question:** We discovered that using AdaBN and TENT **together** creates a critical bug where TENT's backward pass resets the batch normalization statistics that AdaBN just updated, effectively making AdaBN's work zero and resulting in **worse performance**. Should the thesis: (a) Present AdaBN and TENT as **separate alternative approaches** (use one OR the other, not both), or (b) Document the bug fix (snapshot and restore BN stats) as a contribution? Given the bug makes combined use worse, I lean toward presenting them as alternatives.

**My proposal:** Present AdaBN and TENT as two separate test-time adaptation methods to be used independently, NOT combined; document the interaction issue as a cautionary finding

---

### Q21: Self-consistency filter & class imbalance
**Question:** The self-consistency filter for pseudo-labeling retains only about 14.5% of samples. Additionally, **we don't have all 11 activity classes in every single dataset** — getting all 11 classes in one session is difficult. Furthermore, our inference results show **very high prevalence of hand-tapping activity** in most datasets, possibly because when someone is working at a desk, their hands are constantly on the table which gets detected as hand-tapping. Should I: (a) run ablation with/without filter, (b) analyze class distribution across datasets, or (c) discuss these data characteristics qualitatively?

**My proposal:** Run ablation study (with/without filter) and include analysis of class distribution across the 26 sessions, noting hand-tapping prevalence and class availability limitations

---

### Q22: Implementation details (TENT slowdown)
**Question:** Should TensorFlow-specific engineering issues (like the TENT slowdown and graph retracing fix) appear in the main chapters, or only in an appendix / log?

**My proposal:** Brief mention in main text, details in appendix

---

## 8. CI/CD & Code Completeness

### Q23: How complete the CI/CD must be
**Question:** For the thesis, is it enough to **design and partially implement** the automatic retrain‑and‑deploy loop, or do you expect a fully working end‑to‑end demonstration?

**My proposal:** Design + partial implementation + thorough documentation

---

### Q24: Model registry evidence
**Question:** Should I run exactly one full 'train → evaluate → register → promote' sequence and include logs/screenshots as evidence, or is the design description enough?

**My proposal:** Run at least one full sequence with concrete evidence (logs/screenshots)

---

### Q25: DANN and MMD - Remove or keep as future work?
**Question:** DANN and MMD are currently in the code as `NotImplementedError` stubs. Since we are not implementing them, should I: (a) **Remove them entirely** from the codebase to keep it clean, or (b) Keep the stubs and clearly mark as future work in both code comments and thesis? If they stay as "future work," should they remain in the code or only be mentioned in the thesis Future Work chapter?

**My proposal:** **Remove DANN and MMD from the codebase** entirely; mention them only in the Related Work and Future Work sections of the thesis as potential alternative approaches

---

### Q26: How many bugs to show
**Question:** Is it better if I show only the top few impactful bugs (unit mismatch, PSI recalibration, AdaBN+TENT) in the main thesis and move the full list of 19 bugs to an appendix?

**My proposal:** Top 3-4 bugs in main text, full list in appendix

---

### Q27: Proof-of-concept vs production-grade expectations
**Question:** The original thesis plan explicitly states the emphasis is on **"proof-of-concept and scalability principles, rather than a full production-grade system."** Based on my current implementation (Dockerized pipeline, API, CI tests, model registry, 3-layer monitoring, manual retrain trigger), do you consider this **sufficient to meet the original plan**, with Level-2 style automation and additional adaptation methods treated as 'nice-to-have' extensions rather than mandatory?

**My proposal:** Current PoC implementation is sufficient; fully automated CI/CD and additional methods (DANN/MMD) are extensions, not requirements

---

## 9. Thesis Expectations and Contributions

### Q29: Which targets must be measured
**Question:** Of these targets (latency 250 ms, availability, rollback time, false triggers), which **must** have real measured results in the thesis, and which can remain as design targets backed by architecture arguments only?

**My proposal:** Need your guidance on which require empirical measurement

---

### Q30: Thesis structure priority ⚠️ **CRITICAL**
**Question:** Given the time until **June 30, 2026** (though I aim to finish earlier), which chapters do you consider absolutely essential to complete first (e.g., Methods, Results, Discussion), and which can be lighter or moved to appendices?

**My proposal:** Prioritize Methods, Results (including all ablation studies), and Discussion/Future Work; keep Introduction and Related Work focused and concise

---

### Q32: Primary scientific contribution ⚠️ **CRITICAL**
**Question:** Given that this is a **technical thesis** with main focus on **MLOps + Prognosis**, which contributions should I emphasize as the main scientific contributions: (1) Complete 14-stage MLOps pipeline for HAR, (2) 3-layer label-free monitoring framework (needs literature citations to present as contribution), (3) Test-time adaptation findings (AdaBN/TENT as alternatives), (4) Prognosis model integration design? Test-time adaptation is more of a technical implementation detail rather than core focus.

**My proposal:** Main contributions = (1) Complete MLOps pipeline with model registry and CI/CD, (2) 3-layer monitoring framework with multi-channel threshold calibration (need to find supporting citations), (3) Prognosis model architectural design and integration approach; test-time adaptation as technical enhancement

---

### Q33: Freezing the code
**Question:** Before submission, should I freeze the repository at a specific commit and use commit‑pinned links in the thesis? If yes, when do you recommend doing this?

**My proposal:** Freeze right before final proofreading, use commit-pinned links

---

## 10. What Must Be Finished vs Future Work

### Q34: Scope of additional features
**Question:** For the following implemented features beyond the core pipeline: (a) Wasserstein distance drift detection, (b) curriculum pseudo-labeling, (c) OOD detection, (d) sensor placement analysis — should any of these be evaluated with results in the thesis, or can they all be documented as "implemented capabilities" with evaluation left for future work?

**My proposal:** Focus evaluation effort on core pipeline, monitoring, and adaptation methods; document additional features as available capabilities with brief description, mark comprehensive evaluation as future work

---

---

# 11. Email Template for Mentor (Simple English)

**Subject:** Key decisions for HAR MLOps thesis (Feb 2026)

---

Dear [Mentor Name],

I hope you are doing well. I am now starting the thesis‑writing phase and I want to be sure that I match the original plan and your expectations.

Below I list the **most important decisions** I need from you. For each point I also write **what I propose**. It would help me a lot if you could say whether you **agree or disagree**, and if anything important is missing.

---

## 1. Monitoring thresholds (confidence, drift, transition rate)

**What I understand:**  
I should use **one set of thresholds** across the whole system (pipeline + API) and not different values in different places. I should **check these values on our data** instead of just copying them from papers.

**My proposal:**

- Use one shared config file for all thresholds.
- Run small experiments (e.g. replaying the 26 sessions, simple drift injection) to choose reasonable values.
- Clearly explain in the thesis how these thresholds were chosen.

**Question:**  
Is this level of calibration enough for the thesis, or do you expect a deeper / more formal threshold study?

---

## 2. Scalability, latency, and hyperparameter optimization

**Context:**

- The system now runs on one machine with 26 sessions.
- We have a **design target** of p95 ≤ 250 ms latency, but no timing experiment yet.
- The original plan mentioned **automatic hyperparameter optimization (HPO)**. Right now I mainly use a fixed configuration with MLflow tracking.

**My proposal:**

- Measure latency properly (many repeated inferences) and report p50/p90/p95 with hardware details.
- Make simple plots that show how inference time grows with the number of windows / sessions.
- Add a short architecture section that explains **how** this design could scale to more machines, even if we do not build a full cluster.
- For HPO: keep a **well‑justified fixed configuration** and clearly explain the choices; treat full auto‑HPO (Optuna/grid search) as future work.

**Questions:**

1. Is this enough to say that the system is "scalable" for a master's thesis?
2. Do you expect a real automatic HPO setup, or is a fixed, well‑explained configuration acceptable?

---

## 3. Scope: MLOps + prognosis model (NEED YOUR GUIDANCE)

**Context:**

- I understand the main thesis focus should be **MLOps + Prognosis**.
- However, I realize I **don't have a clear understanding** of what the prognosis model should do.
- My current work has focused on the HAR pipeline, 3-layer monitoring framework, model registry, CI/CD, and test-time adaptation methods.

**My proposal:**

- I need your guidance on the prognosis model:
  - What should it predict? (anxiety level, mental health risk scores, clinical outcomes, etc.?)
  - What inputs from HAR? (activity percentages, sequences, transitions?)
  - What is the expected output format?
- For the thesis, I can:
  - Design and document the full **data flow from HAR → prognosis**
  - Create architectural diagrams showing where/how prognosis model would integrate
  - Implement a placeholder interface with clear input/output specification
  - Mark full prognosis model training and evaluation as **future work**
- Emphasize the complete **MLOps infrastructure** (14-stage pipeline, 3-layer monitoring, model registry, CI/CD) as the technical foundation that supports both HAR and future prognosis models.

**Questions:**

1. Can you explain what the prognosis model should do in this thesis?
2. Should it be a **co-equal main contribution** with implemented results, or can it be **designed-but-not-implemented** with clear future work?
3. Given the June 30 deadline, what is the **priority level** — must it be fully implemented, or is the design + interface sufficient?

---

## 4. "Online monitoring" and automation

**Context:**

- We do **not** have live BLE streaming. Data arrives as files through the API.
- For monitoring tools: **MLflow** is fully implemented and tracks all experiments, models, and metrics.
- **Prometheus and Grafana** are configured in code but not running. **Question: Are they necessary if MLflow handles all tracking needs?**
- Retraining is started **manually** (I run a command) after a warning.

**My proposal:**

- For the thesis, define "online" as **processing each uploaded file through the API one‑by‑one**, and monitor them as if they arrived in order.
- Use a **replay experiment** of the 26 sessions to simulate continuous operation.
- **Use MLflow for all monitoring and tracking**; treat Prometheus/Grafana as optional for production deployments (not required for thesis).
- Keep retraining as a **manual step**, but clearly describe how a fully automatic "detect drift → retrain → deploy" loop would work.

**Questions:**

1. Can all monitoring needs be satisfied with MLflow alone, or do you expect Prometheus/Grafana implementation?
2. For a master's thesis, is manual retraining with clear documentation of automation design sufficient?

---

## 5. Evaluation approach and performance targets

**Context:**

- Current results use **window‑level StratifiedKFold**, which mixes windows from the same person across train and validation.
- From our previous discussion, I understand that **strict subject-wise evaluation is NOT required** for this thesis.
- K-fold cross-validation will be used when we have labeled data for retraining experiments.
- Current MLflow-tracked results: ~93.8% accuracy, 0.939 macro F1
- Some older numbers (96.9% / 81.4%) are not found in MLflow and cannot be traced.

**My proposal:**

- Use **window-level StratifiedKFold** for main experiments with labeled data.
- Clearly document this as a limitation — explain that subject-wise generalization (ensuring no person in both train/test) is important for real deployment and mark it as **future work**.
- Only use **MLflow‑traceable results** in the thesis (current: ~93.8% acc, 0.939 macro F1).
- Report **per‑class recall** for the 11 classes and use macro F1 as the main metric.
- Use descriptive statistics (mean ± std) for ablation comparisons.

**Questions:**

1. Do you confirm that window-level evaluation with clear limitations documented is acceptable?
2. Is there a minimum macro F1 you would see as necessary for a passing thesis?

---

## 6. Data quality and drift detection

**Context:**

- We fixed the milliG vs m/s² error with automatic unit detection.
- All 26 sessions are from one lab environment.
- For drift detection on **unlabeled data**, we are using **Z-score** to monitor distribution shifts in the streaming data.

**My proposal:**

- Add a **hard safety check** after unit conversion (gravity magnitude must be in realistic range) and describe this as a **data quality gate**.
- Be very clear that all data comes from one lab, and be careful with claims about real‑world generalization.
- Document the **Z-score threshold calibration for multi-channel sensor data** as a methodological contribution.

**Question:**  
Is this level of detail about data quality, Z-score drift detection, and limited generalization what you expect in the thesis?

---

## 7. Test-time adaptation methods

**Context:**

- We implemented AdaBN (Adaptive Batch Normalization) and TENT (Test Entropy Minimization) as separate test-time adaptation methods.
- **Critical finding**: Using AdaBN + TENT **together causes a severe bug** — TENT's backward pass resets the batch-norm statistics that AdaBN just updated, making **performance worse** than using either method alone.
- Therefore, these should be presented as **alternative approaches** (use one OR the other), not combined.
- The self-consistency filter for pseudo-labeling keeps only about 14.5% of data.
- **Data characteristics**: We don't have all 11 activity classes in every single dataset. Also, inference results show **very high hand-tapping prevalence** — possibly because when people work at desks, hands constantly on table get detected as hand-tapping.

**My proposal:**

- Present **AdaBN and TENT as separate alternative methods** for test-time adaptation, NOT to be combined.
- Document the interaction bug as a **cautionary finding** about combining these methods.
- Run **ablation experiments**: (a) No adaptation, (b) AdaBN only, (c) TENT only, (d) Pseudo-labeling with self-consistency filter, (e) Pseudo-labeling without filter.
- Include **analysis of class distribution** across the 26 sessions, noting hand-tapping prevalence and the challenge that not all 11 classes appear in every dataset.
- Mention TensorFlow‑specific issues briefly in main text, details in appendix.

**Question:**  
Is this approach to presenting adaptation methods and data characteristics appropriate for the thesis?

---

## 8. CI/CD and code organization

**Context:**

- We have Docker, CI tests, a model registry, drift‑triggered retrain logic, and some deployment code, but the CI/CD retrain loop is not fully wired.
- **DANN and MMD** exist in the code only as `NotImplementedError` stubs and are not being used.
- The original plan says the goal is a **proof‑of‑concept**, not a full production system.

**My proposal:**

- Treat the current system (Dockerized pipeline, API, CI tests, model registry, 3‑layer monitoring, manual retrain trigger) as a **complete proof‑of‑concept** for the thesis.
- Describe and partly implement the automatic retrain‑and‑deploy loop, but do not try to make it fully production‑ready.
- Run at least **one full run** of "train → evaluate → register → promote" and include logs/screenshots as evidence.
- For **DANN/MMD**: **Remove them entirely from the codebase** (since they're not implemented); mention them only in Related Work and Future Work sections of thesis as potential alternative approaches.
- Show only the **most important bugs** (unit mismatch, Z-score calibration, AdaBN/TENT interaction) in the main text; move the full bug list to an appendix.

**Questions:**

1. Do you agree that this level of CI/CD is enough for a proof‑of‑concept thesis?
2. Should I remove DANN/MMD from the code, or keep them as stubs marked as future work?

---

## 9. Thesis contributions and priorities

**Context:**

- **Submission deadline: June 30, 2026** (though I aim to finish earlier)
- This is a **technical thesis** with main focus on **MLOps + Prognosis**
- The thesis uses sensitive mental‑health sensor data.
- The code is still evolving.

**My proposal:**

- Focus first on **Methods**, **Results** (including all ablation studies), and **Discussion/Future Work**. Keep Introduction and Related Work focused.
- Include a short but **concrete** section on data protection and GDPR (especially Article 9).
- Treat the main contributions as:
  1. **Complete 14-stage MLOps pipeline** for HAR with model registry, CI/CD, deployment design
  2. **3-layer label-free monitoring framework** with multi-channel threshold calibration (need to find supporting citations for this)
  3. **Prognosis model architectural design** and integration approach
  4. **Test-time adaptation findings** (AdaBN/TENT as alternatives, cautionary note about combining them) as technical enhancement
- Before final submission, **freeze the code** at a specific commit and use commit‑pinned links in the thesis.

**Questions:**

1. Do you agree with this choice of main contributions and chapter priorities?
2. For the 3-layer monitoring framework, can you suggest any literature I should cite to position this as a contribution?
3. When do you recommend freezing the repository?

---

If you prefer, I can also bring the longer 34‑question document to our next meeting and walk through it with you. The points above are the main decisions I need to safely finish the thesis on time.

Thank you very much for your time and support.

Best regards,  
Shalin

---

## Notes
- **Removed duplicate question** (old Q29 about subject-wise evaluation was same as Q12)
- **4 new questions added** based on Thesis Plan alignment check:
  - Q6 & Q7: Focus shift and prognosis minimum deliverable (CRITICAL)
  - Q16: HPO requirement from original plan
  - Q27: PoC vs production-grade expectations
- **Critical decision points** are marked with ⚠️ (now includes 5 critical questions: Q6, Q7, Q12, Q30, Q32)
- **Total questions: 34** (was 30, removed 1 duplicate, added 4 new)
- The email template can be sent standalone; detailed questions available for deeper discussion

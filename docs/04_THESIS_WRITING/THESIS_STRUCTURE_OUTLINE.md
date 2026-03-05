# 📚 MASTER THESIS STRUCTURE OUTLINE
## MLOps Pipeline for Human Activity Recognition with Wearable Sensors

**Created:** January 30, 2026  
**Deadline:** May 20, 2026 (16 weeks remaining)  
**Expected Length:** 60-80 pages (without appendices)  
**Writing Start:** Week 7 (after implementation complete)

---

# TABLE OF CONTENTS

1. [Chapter Overview](#chapter-overview)
2. [Chapter 1: Introduction](#chapter-1-introduction-8-10-pages)
3. [Chapter 2: Background & Related Work](#chapter-2-background--related-work-15-18-pages)
4. [Chapter 3: Methodology](#chapter-3-methodology-15-18-pages)
5. [Chapter 4: Implementation](#chapter-4-implementation-12-15-pages)
6. [Chapter 5: Evaluation](#chapter-5-evaluation-12-15-pages)
7. [Chapter 6: Discussion & Conclusion](#chapter-6-discussion--conclusion-8-10-pages)
8. [References & Appendices](#references--appendices)
9. [Writing Timeline](#writing-timeline)

---

# CHAPTER OVERVIEW

| Chapter | Title | Pages | Status | Priority |
|---------|-------|-------|--------|----------|
| 1 | Introduction | 8-10 | ❌ Not started | Week 11 |
| 2 | Background & Related Work | 15-18 | 📋 Research done | Week 12 |
| 3 | Methodology | 15-18 | ❌ Not started | Week 7-8 |
| 4 | Implementation | 12-15 | ❌ Not started | Week 9 |
| 5 | Evaluation | 12-15 | ❌ Not started | Week 10 |
| 6 | Discussion & Conclusion | 8-10 | ❌ Not started | Week 13-14 |
| - | References | 3-5 | 📋 Papers collected | Ongoing |
| - | Appendices | 10-20 | ❌ Not started | Week 14 |

**Recommended Writing Order:** 3 → 4 → 5 → 2 → 1 → 6 (Methodology first!)

---

# CHAPTER 1: INTRODUCTION (8-10 pages)

## Purpose
Set the stage for the reader. Explain WHY this thesis exists and WHAT it contributes.

## Sections

### 1.1 Motivation (2 pages)
**Content:**
- Growing importance of wearable health monitoring
- Anxiety affects 300+ million people globally
- Wearable IMU sensors can detect anxiety-related micro-behaviors
- GAP: Most HAR systems are trained once and deployed, ignoring model degradation

**Why needed:** Reader must understand the real-world problem and care about it.

**Source:** General statistics, WHO reports, existing HAR surveys

### 1.2 Problem Statement (1-2 pages)
**Content:**
- Trained HAR models degrade over time (domain shift)
- Production data is UNLABELED (cannot compute accuracy)
- Need automated monitoring, drift detection, and adaptation
- Need CI/CD pipeline for continuous improvement

**Why needed:** Crystalize the specific technical challenge.

**Key Sentence:** "How can we maintain HAR model performance in production when ground truth labels are unavailable?"

### 1.3 Research Questions (1 page)
**Content:**
| # | Research Question |
|---|-------------------|
| RQ1 | How can we detect model degradation without ground truth labels? |
| RQ2 | What proxy metrics correlate with actual accuracy in HAR? |
| RQ3 | When should we trigger model adaptation or retraining? |
| RQ4 | How effective is pseudo-labeling for HAR model updates? |

**Why needed:** Give structure to the entire thesis. Reader knows what to expect.

### 1.4 Contributions (1-2 pages)
**Content:**
1. **MLOps pipeline** for HAR with 10 stages (ingestion → deployment)
2. **Proxy metrics framework** for unlabeled production monitoring
3. **Tiered trigger policy** for adaptation decisions
4. **Curriculum pseudo-labeling** implementation for HAR
5. **Empirical evaluation** comparing adaptation methods

**Why needed:** State clearly what is NEW in this thesis.

### 1.5 Thesis Structure (1 page)
**Content:** Brief overview of each chapter (1 paragraph each).

**Why needed:** Navigation guide for the reader.

---

# CHAPTER 2: BACKGROUND & RELATED WORK (15-18 pages)

## Purpose
Establish theoretical foundation and position this work in the literature.

## Sections

### 2.1 Human Activity Recognition (4 pages)
**Content:**
- Definition and taxonomy of HAR
- IMU sensors: accelerometer, gyroscope
- Deep learning for HAR: CNN, LSTM, Transformer
- Benchmark datasets: UCI-HAR, PAMAP2, Opportunity

**Why needed:** Reader must understand the domain.

**Sources:** `GENERAL_HAR_SURVEYS_PAPERS_ANALYSIS.md`, `A_Survey_on_Human_Activity_Recognition.pdf`

### 2.2 Domain Adaptation in HAR (4 pages)
**Content:**
- Cross-person, cross-device, cross-position heterogeneity
- Unsupervised Domain Adaptation (UDA) methods
- AdaBN, DANN, Contrastive learning
- Test-Time Adaptation (TTA): Tent, entropy minimization

**Why needed:** Core technique for handling production drift.

**Sources:** `DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md`, XHAR, AdaptNet, Shift-GAN papers

### 2.3 MLOps for Machine Learning (3 pages)
**Content:**
- Definition of MLOps lifecycle
- CI/CD for ML: training, testing, deployment
- Model monitoring in production
- Drift detection: data drift, concept drift, prediction drift

**Why needed:** Frame this thesis as an MLOps contribution, not just HAR.

**Sources:** `ACTIVE_LEARNING_MLOPS_HUMAN_IN_LOOP_PAPERS_ANALYSIS.md`, MLOps survey papers

### 2.4 Uncertainty Quantification (3 pages)
**Content:**
- Aleatoric vs Epistemic uncertainty
- MC Dropout, Bayesian neural networks
- Calibration: Expected Calibration Error (ECE)
- Uncertainty for OOD detection

**Why needed:** Uncertainty is our primary proxy for accuracy.

**Sources:** `UNCERTAINTY_CONFIDENCE_PAPERS_ANALYSIS.md`, XAI-BayesHAR, MC Dropout papers

### 2.5 Semi-Supervised Learning (2 pages)
**Content:**
- Self-training and pseudo-labeling
- Curriculum learning
- Confirmation bias and how to avoid it
- Teacher-student frameworks

**Why needed:** Our retraining strategy uses pseudo-labels.

**Sources:** `PSEUDO_LABELING_SELF_TRAINING_PAPERS_ANALYSIS.md`, Curriculum Labeling, SelfHAR papers

### 2.6 Related Work Summary (1-2 pages)
**Content:**
- Gap analysis: What existing work does NOT do
- Position our contribution

**Why needed:** Show the novelty of this thesis.

---

# CHAPTER 3: METHODOLOGY (15-18 pages)

## Purpose
Describe HOW we solve the problem. This is the CORE of the thesis.

## Sections

### 3.1 System Overview (2 pages)
**Content:**
- High-level architecture diagram
- 10-stage pipeline overview
- Data flow from Garmin watch to predictions

**Why needed:** Give reader the big picture before details.

**Figure:** Pipeline architecture diagram

### 3.2 Data Pipeline (3 pages)
**Content:**
- Garmin Excel format parsing
- Sensor fusion (accelerometer + gyroscope)
- Resampling to 50Hz
- Windowing: 200 samples (4 sec) with 50% overlap
- Normalization using training baseline

**Why needed:** Reproducibility - reader can replicate.

**Source:** `src/sensor_data_pipeline.py`, `src/preprocess_data.py`

### 3.3 Monitoring Framework (4 pages)
**Content:**
- **Layer 1: Confidence/Uncertainty metrics**
  - Max probability, entropy, margin
  - MC Dropout uncertainty
- **Layer 2: Temporal plausibility**
  - Flip rate, dwell time, transition violations
- **Layer 3: Distribution drift**
  - KS-test, PSI, Wasserstein distance
  - Per-channel analysis

**Why needed:** Novel contribution - how to monitor without labels.

**Source:** `scripts/post_inference_monitoring.py`, `06_MONITORING_DRIFT.md`

### 3.4 Trigger Policy (2 pages)
**Content:**
- Tiered decision framework:
  - Tier 1: Adapt (AdaBN)
  - Tier 2: Retrain (pseudo-labeling)
  - Tier 3: Escalate (human review)
- 2-of-3 voting for drift confirmation
- Cooldown periods and minimum data requirements

**Why needed:** Automated decision-making.

**Source:** `docs/patient/FINAL_PIPELINE_DECISIONS.md` Section 1.8

### 3.5 Adaptation Methods (4 pages)
**Content:**
- **AdaBN:** Update BatchNorm statistics without labels
- **MC Dropout:** Uncertainty quantification (10 passes)
- **Curriculum Pseudo-labeling:**
  - Model restart between cycles
  - Top-K% per class selection
  - 10-20 cycles until convergence

**Why needed:** Core techniques we implement.

**Source:** XHAR paper, Curriculum Labeling paper, `DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md`

### 3.6 Evaluation Methodology (2 pages)
**Content:**
- Offline evaluation with held-out labeled data
- Proxy metric validation (correlation analysis)
- Ablation study design
- Metrics: Accuracy, F1, ECE, proxy correlation

**Why needed:** How we will measure success.

---

# CHAPTER 4: IMPLEMENTATION (12-15 pages)

## Purpose
Describe WHAT we built. Code-level details for reproducibility.

## Sections

### 4.1 Technology Stack (2 pages)
**Content:**
- Python 3.11, TensorFlow/Keras
- MLflow for experiment tracking
- DVC for data versioning
- Docker for containerization
- GitHub Actions for CI/CD

**Why needed:** Reproducibility.

**Table:** Dependencies with versions

### 4.2 Repository Structure (1 page)
**Content:**
- Directory layout
- Key files and their purposes

**Why needed:** Navigation for readers who access code.

### 4.3 Data Processing Implementation (3 pages)
**Content:**
- `sensor_data_pipeline.py`: Key functions, design decisions
- `preprocess_data.py`: Windowing algorithm, normalization
- QC checks and validation

**Why needed:** Implementation details.

**Code snippets:** Key algorithms

### 4.4 Inference Pipeline (3 pages)
**Content:**
- Model loading and batching
- MC Dropout implementation
- AdaBN adaptation code
- Output format and logging

**Why needed:** Core production code.

### 4.5 Monitoring & Alerting (2 pages)
**Content:**
- Drift detection implementation
- Alert thresholds configuration
- MLflow metric logging
- Grafana dashboard design

**Why needed:** Monitoring system details.

### 4.6 CI/CD Pipeline (2 pages)
**Content:**
- GitHub Actions workflow
- Lint → Test → Build → Deploy stages
- Test coverage and quality gates

**Why needed:** MLOps automation.

---

# CHAPTER 5: EVALUATION (12-15 pages)

## Purpose
Present RESULTS and answer research questions.

## Sections

### 5.1 Experimental Setup (2 pages)
**Content:**
- Hardware: Garmin Venu 2 watch specs
- Dataset: X windows from Y sessions
- Train/test split strategy
- Baseline model: 1D-CNN + BiLSTM (499K params)

**Why needed:** Reproducibility.

### 5.2 RQ1: Degradation Detection (3 pages)
**Content:**
- Experiment: Inject synthetic drift, measure detection
- Results: Detection rate vs false positive rate
- Finding: KS + PSI + entropy achieves X% detection

**Why needed:** Answer RQ1.

**Figures:** ROC curves, detection latency

### 5.3 RQ2: Proxy Metric Validation (3 pages)
**Content:**
- Experiment: Correlate proxy metrics with actual accuracy
- Results: Pearson correlation coefficients
- Finding: Confidence mean has r=X with accuracy

**Why needed:** Answer RQ2.

**Figures:** Scatter plots, correlation matrix

### 5.4 RQ3: Trigger Policy Effectiveness (2 pages)
**Content:**
- Experiment: Compare trigger policies
- Results: Precision/recall of triggers
- Finding: Tiered policy reduces false triggers by X%

**Why needed:** Answer RQ3.

### 5.5 RQ4: Adaptation Effectiveness (3 pages)
**Content:**
- Experiment: Compare AdaBN vs pseudo-labeling vs no adaptation
- Results: Accuracy before/after adaptation
- Finding: Curriculum pseudo-labeling improves by X%

**Why needed:** Answer RQ4.

**Figures:** Bar charts, learning curves

### 5.6 Ablation Study (2 pages)
**Content:**
- What if we remove MC Dropout?
- What if we don't restart model in pseudo-labeling?
- What if we use fixed threshold vs top-K%?

**Why needed:** Understand component contributions.

---

# CHAPTER 6: DISCUSSION & CONCLUSION (8-10 pages)

## Purpose
Reflect on results, acknowledge limitations, suggest future work.

## Sections

### 6.1 Summary of Findings (2 pages)
**Content:**
- Restate research questions and answers
- Key numerical results

**Why needed:** Remind reader what we learned.

### 6.2 Discussion (3 pages)
**Content:**
- Why do proxy metrics work?
- When does adaptation fail?
- Comparison to related work
- Practical implications

**Why needed:** Interpret results.

### 6.3 Limitations (2 pages)
**Content:**
- Single-user demo (not multi-user validated)
- Same 11 activity classes (no novel activities)
- Batch mode only (not streaming)
- Cannot verify actual production accuracy

**Why needed:** Honest about scope.

### 6.4 Future Work (2 pages)
**Content:**
- Streaming inference
- GAN-based UDA (ContrasGAN)
- LIFEWATCH pattern memory
- Edge deployment on Garmin
- Multi-user generalization
- Handedness-specific models

**Why needed:** Guide future research.

### 6.5 Conclusion (1 page)
**Content:**
- Final summary paragraph
- Main contribution in one sentence

**Why needed:** Closure.

---

# REFERENCES & APPENDICES

## References (3-5 pages)
- ~50-80 citations expected
- Use BibTeX for management
- IEEE or ACM format

**Source files:**
- `docs/Bibliography_From_Local_PDFs.md`
- `docs/APPENDIX_PAPER_INDEX.md`
- All analysis files in `paper for questions/`

## Appendix A: Full Pipeline Configuration (3-5 pages)
- `config/pipeline_config.yaml` full listing
- Threshold values and their justifications

## Appendix B: Code Listings (5-10 pages)
- Key algorithms with comments
- `train.py`, `trigger_policy.py`, `adabn.py`

## Appendix C: Additional Figures (3-5 pages)
- Confusion matrices
- Per-class performance
- Extended ablation results

## Appendix D: User Study Protocol (if applicable) (2-3 pages)
- Labeling instructions
- Consent form template

---

# WRITING TIMELINE

## Phase 1: Core Chapters (Weeks 7-10)

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 7 | Chapter 3.1-3.3 | Methodology draft (data + monitoring) |
| Week 8 | Chapter 3.4-3.6 | Methodology complete |
| Week 9 | Chapter 4 | Implementation chapter |
| Week 10 | Chapter 5 | Evaluation chapter |

## Phase 2: Framing Chapters (Weeks 11-13)

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 11 | Chapter 1 | Introduction |
| Week 12 | Chapter 2 | Related Work |
| Week 13 | Chapter 6 | Discussion & Conclusion |

## Phase 3: Polish (Weeks 14-16)

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 14 | Appendices | All appendices complete |
| Week 15 | Review | Self-review, fix issues |
| Week 16 | Submit | Final formatting, submit |

---

# CHECKLIST BEFORE SUBMISSION

## Content
- [ ] All research questions answered
- [ ] All figures have captions and are referenced in text
- [ ] All tables are formatted consistently
- [ ] All citations are complete (no missing references)
- [ ] Abstract written (250-300 words)
- [ ] Acknowledgments written

## Format
- [ ] Page numbers correct
- [ ] Table of contents updated
- [ ] List of figures/tables complete
- [ ] Consistent font and spacing
- [ ] Margins per university guidelines

## Quality
- [ ] Spell check passed
- [ ] Grammar check passed
- [ ] Supervisor review complete
- [ ] At least one friend/colleague proofread

---

**Document Status:** THESIS OUTLINE COMPLETE  
**Next Step:** Start implementation (Week 1-6), then begin Chapter 3 writing (Week 7)

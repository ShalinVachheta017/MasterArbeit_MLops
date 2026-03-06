# 📚 Paper-Driven Questions Map for HAR MLOps Thesis
## Comprehensive Analysis of 88 Papers from "Paper for Questions" Folder

**Generated:** January 30, 2026  
**Purpose:** Extract technical/research questions from papers to guide thesis pipeline design  
**Constraint:** Production data is UNLABELED — this is our primary design driver

---

# Table of Contents

1. [Paper Inventory](#1-paper-inventory)
2. [Paper-by-Paper Question Extraction](#2-paper-by-paper-question-extraction)
3. [Question-Centric View (MOST IMPORTANT)](#3-question-centric-view)
4. [Questions We Can Answer Now](#4-questions-we-can-answer-now)
5. [Questions We Must Ask Mentor](#5-questions-we-must-ask-mentor)
6. [Open Research Gaps](#6-open-research-gaps)
7. [Pipeline Stage Mapping](#7-pipeline-stage-mapping)
8. [Implementation Priority](#8-implementation-priority)

---

# 1. Paper Inventory

## 1.1 Summary Statistics

| Category | Count | Labels Required | Production Ready |
|----------|-------|-----------------|------------------|
| Domain Adaptation | 7 | Mixed | ⭐⭐⭐ |
| Drift/Change Detection | 7 | Mostly NO | ⭐⭐⭐⭐ |
| Uncertainty/Confidence | 5 | Training only | ⭐⭐⭐⭐ |
| Pseudo-labeling/Self-training | 6 | Partial | ⭐⭐⭐ |
| Sensor Placement/Position | 7 | Yes | ⭐⭐ |
| General HAR/Surveys | 10 | Yes | ⭐⭐ |
| Active Learning/MLOps | 10 | Mixed | ⭐⭐⭐⭐ |
| Thesis/Books | 7 | Reference | ⭐⭐⭐ |
| Recent arXiv (2025-2026) | 8 | Mixed | ⭐⭐⭐⭐⭐ |
| ACM/Conference | 13 | Mixed | ⭐⭐⭐ |
| Elsevier/ScienceDirect | 8 | Mixed | ⭐⭐⭐⭐ |
| **TOTAL** | **88** | - | - |

## 1.2 Complete Paper List by Relevance to Unlabeled Production

### ⭐⭐⭐⭐⭐ CRITICAL (Must Implement)

| Paper | Year | Key Contribution | Labels Needed? |
|-------|------|------------------|----------------|
| **COA-HAR** (S0957417425029045) | 2025 | Contrastive Online Test-Time Adaptation | **NO** ✅ |
| **ContrasGAN** (S1574119221001103) | 2021 | UDA via Adversarial + Contrastive | **NO** ✅ |
| **OFTTA** (2310.18562v2 / 3631450) | 2024 | Optimization-Free Test-Time Adaptation | **NO** ✅ |
| **Tent** (2006.10726v3) | 2021 | Entropy Minimization for TTA | **NO** ✅ |
| **WATCH** | 2021 | Wasserstein Change Point Detection | **NO** ✅ |
| **LIFEWATCH** | 2022 | Lifelong Wasserstein CPD | **NO** ✅ |
| **XHAR** | 2019 | AdaBN for Domain Adaptation | **NO** ✅ |
| **ADAPT** (2507.08597) | 2025 | Pseudo-labeling for Concept Drift | **NO** ✅ |
| **Entropy-based Drift** (2601.00554) | 2026 | Entropy-based Retraining Triggers | **NO** ✅ |
| **HAROOD** (2512.10807) | 2025 | OOD Detection Benchmark (KDD 2026) | Benchmark |

### ⭐⭐⭐⭐ HIGH (Should Implement)

| Paper | Year | Key Contribution | Labels Needed? |
|-------|------|------------------|----------------|
| **SelfHAR** (imwut-selfhar) | 2021 | Teacher-student self-supervised HAR | Source only |
| **Curriculum Labeling** | ~2023 | Self-paced pseudo-labeling | Source only |
| **CODA** (2403.14922) | 2024 | Cost-efficient domain adaptation | Minimal (5-10%) |
| **CoDATS** (3394486.3403228) | 2020 | Weak supervision (proportions only) | Weak labels |
| **OOD in HAR** | 2022 | Energy/Mahalanobis OOD scores | Training only |
| **MC Dropout Uncertainty** | 2021 | Uncertainty quantification | Training only |
| **PACL+** (S0957417425022225) | 2025 | Online continual learning | Partial |
| **IMUCoCo** (2508.01894) | 2025 | Position-invariant HAR | Training only |
| **CrossHAR** (3659597) | 2024 | Cross-dataset generalization | Training only |

### ⭐⭐⭐ MEDIUM (Consider for Thesis)

| Paper | Year | Key Contribution | Labels Needed? |
|-------|------|------------------|----------------|
| **AdaptNet** | 2021 | Bilateral domain adaptation | 10-30% |
| **Shift-GAN** | 2021 | GAN-based UDA | **NO** |
| **SCAGOT** | 2023 | Source-free domain adaptation | Source only |
| **LAPNet-HAR** | 2022 | Lifelong learning with prototypes | Incremental |
| **Transfer Learning Review** | 2022 | Comprehensive taxonomy | Reference |
| **UDA Benchmark** (2312.09857) | 2023 | Time series UDA methods | Benchmark |
| **XAI-BayesHAR** | 2022 | Kalman + SHAP for uncertainty | Training only |
| **Sensor Displacement** | 2023 | Compensation strategies | Yes |

### ⭐⭐ LOW (Background Reference)

| Paper | Year | Key Contribution | Labels Needed? |
|-------|------|------------------|----------------|
| General HAR Surveys | Various | Foundation knowledge | Reference |
| Springer Books | 2019-2022 | Comprehensive coverage | Reference |
| Thesis references | Various | Methodology examples | Reference |
| sensors-* papers | Various | Incremental improvements | Yes |

---

# 2. Paper-by-Paper Question Extraction

## 2.1 Domain Adaptation Papers

### Paper: XHAR (Deep Domain Adaptation for HAR)
**(Paper for Questions/XHAR_Deep_Domain_Adaptation_for_Human_Activity_Recognition_with_Smart_Devices.pdf)**

**What the paper contributes:**
- AdaBN (Adaptive Batch Normalization) for zero-label adaptation
- Updates only BN statistics using target data
- No gradient updates or retraining required

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Can we adapt without any labels? | YES via AdaBN | Section 3, Method |
| What to update at adaptation time? | Only BN running mean/variance | Section 3.2 |
| How much target data needed? | ~50 batches for stable statistics | Experiments |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How to know when AdaBN has worked? | No validation metric without labels | Monitoring |
| When to re-trigger AdaBN? | No temporal drift detection | CSD |
| Does AdaBN work for BiLSTM? | Only tested on CNN | Training |
| How to combine with pseudo-labeling? | Not explored | Retraining |

**Assumptions that break in our setting:**
- Assumes batch inference (we may have streaming)
- Assumes target distribution is static
- Doesn't address activity class changes

---

### Paper: Tent (2006.10726v3)
**(Paper for Questions/2006.10726v3.pdf)**

**What the paper contributes:**
- Test-time entropy minimization
- Updates ONLY affine parameters (γ, β) in BatchNorm
- Works with single sample or small batches

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How to adapt with entropy alone? | Minimize H(ŷ) by updating BN affine | Section 3 |
| Learning rate for TTA? | 0.00025 (conservative) | Section 4.2 |
| What layers to update? | Only BN affine parameters | Section 3.1 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How long before entropy collapses? | Long-term not tested | Monitoring |
| When is entropy minimization harmful? | Adversarial shifts not covered | CSD |
| How to reset adaptation? | No rollback mechanism | Retraining |

**Assumptions that break in our setting:**
- Assumes covariate shift only (not label shift)
- May collapse if all predictions become one class

---

### Paper: COA-HAR (S0957417425029045)
**(Paper for Questions/1-s2.0-S0957417425029045-main.pdf)**

**What the paper contributes:**
- Contrastive learning for online TTA in HAR
- IMU-specific augmentations (rotation, scaling, jittering)
- No target labels required

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Best augmentations for IMU TTA? | Rotation + scaling + jittering | Section 4 |
| Can contrastive work online? | Yes, with memory bank | Section 3 |
| Improvement over Tent? | +5-10% on HAR benchmarks | Experiments |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Memory bank size for 6-axis IMU? | Not specified for our setup | Inference |
| Interaction with BiLSTM temporal states? | Not tested | Training |
| Computational overhead? | Benchmark missing | MLOps/CI-CD |

---

### Paper: ContrasGAN (S1574119221001103)
**(Paper for Questions/1-s2.0-S1574119221001103-main.pdf)**

**What the paper contributes:**
- Combines adversarial + contrastive for UDA
- Works with completely unlabeled target domain
- Validated on OPPORTUNITY, PAMAP2, HHAR

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Can GAN work without any target labels? | YES with contrastive loss | Section 3 |
| Which datasets validated? | OPPORTUNITY, PAMAP2, HHAR | Section 5 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| GAN training stability in production? | Lab conditions only | Retraining |
| How much source data diversity needed? | Not quantified | Training |
| When to retrain GAN? | No trigger mechanism | CSD |

---

## 2.2 Drift/Change Detection Papers

### Paper: WATCH (Wasserstein CPD)
**(Paper for Questions/WATCH_Wasserstein_Change_Point_Detection_for_High-Dimensional_Time_Series_Data.pdf)**

**What the paper contributes:**
- Wasserstein distance on sliding windows for CPD
- Works with multivariate time series (like 6-axis IMU)
- No labels required

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How to detect change points without labels? | Wasserstein on sliding windows | Section 2 |
| How to set threshold? | Bootstrap from training data | Section 3 |
| Works for high-dimensional data? | Yes, tested on multi-channel | Section 4 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How to distinguish drift types? | Only detects, doesn't classify | CSD |
| Threshold sensitivity analysis? | Limited guidance | Monitoring |
| Window size for 50Hz IMU? | Not HAR-specific | Preprocessing |

**Assumptions that break:**
- Assumes stationary within windows
- May fire on benign activity changes

---

### Paper: LIFEWATCH
**(Paper for Questions/LIFEWATCH_Lifelong_Wasserstein_Change_Point_Detection.pdf)**

**What the paper contributes:**
- Adds pattern memory to WATCH
- Distinguishes novel vs recurring drift
- Lifelong learning capability

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How to recognize recurring patterns? | Store distribution summaries | Section 3 |
| When to NOT alarm? | If matches known pattern | Section 3.2 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Memory management for long deployments? | Not addressed | MLOps |
| How to forget outdated patterns? | Not covered | Monitoring |

---

### Paper: OOD in HAR
**(Paper for Questions/Out-of-distribution_in_Human_Activity_Recognition.pdf)**

**What the paper contributes:**
- Energy score and Mahalanobis distance for OOD detection
- Works at inference time without production labels
- HAR-specific evaluation

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Best OOD score for HAR? | Energy score (simple, effective) | Section 4 |
| Threshold calibration? | 95th percentile from training | Section 3 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| What to DO when OOD detected? | Detection only, no action | Trigger Policy |
| Novel activity vs sensor drift? | Cannot distinguish | CSD |

---

## 2.3 Uncertainty/Confidence Papers

### Paper: Deep Learning Uncertainty Measurement
**(Paper for Questions/A_Deep_Learning_Assisted_Method_for_Measuring_Uncertainty_in_Activity_Recognition_with_Wearable_Sensors.pdf)**

**What the paper contributes:**
- Comparison: MC Dropout vs Ensemble vs Evidential
- Entropy and confidence thresholds for HAR
- 30 MC samples recommended

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How many MC Dropout samples? | 30 is sufficient | Section 4 |
| Entropy threshold for uncertainty? | > 1.5 is uncertain | Section 5 |
| Confidence threshold? | < 0.65 needs review | Section 5 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| MC Dropout computational cost? | 20-30× inference time | Inference |
| Does uncertainty correlate with actual errors? | Needs labeled validation | Evaluation |

---

### Paper: Personalizing via Uncertainty Types
**(Paper for Questions/Personalizing_Activity_Recognition_Models_Through_Quantifying_Different_Types_of_Uncertainty_Using_Wearable_Sensors.pdf)**

**What the paper contributes:**
- Decomposes uncertainty: aleatoric vs epistemic
- Aleatoric = data noise, Epistemic = model ignorance
- User profiling via uncertainty patterns

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| What does high epistemic mean? | Model hasn't seen similar data (OOD) | Section 3 |
| What does high aleatoric mean? | Inherently noisy data | Section 3 |
| Hand placement detection? | High epistemic + low aleatoric = position shift | Section 5 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How to decompose with 1D-CNN-BiLSTM? | Architecture-specific | Training |
| Thresholds for our Garmin data? | Dataset-specific | Monitoring |

---

## 2.4 Pseudo-labeling/Self-training Papers

### Paper: SelfHAR
**(Paper for Questions/imwut-selfhar.pdf)**

**What the paper contributes:**
- Teacher-student framework for self-supervised HAR
- Self-supervised pretraining (contrastive, predictive)
- Pseudo-labeling with confidence weighting

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Best self-supervised pretext tasks? | Contrastive + temporal prediction | Section 3 |
| Pseudo-label confidence threshold? | Soft labels with temperature T=2-4 | Section 4 |
| Confidence weighting formula? | α = 1.0-2.0 | Section 4.3 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How to prevent teacher degradation? | Not addressed for long-term | Retraining |
| When to update teacher? | Fixed teacher, no update policy | Trigger Policy |
| How to start with ZERO source labels? | Assumes labeled source | Training |

---

### Paper: Curriculum Labeling
**(Paper for Questions/Curriculum_Labeling_Self-paced_Pseudo-Labeling_for.pdf)**

**What the paper contributes:**
- Curriculum-based selection of pseudo-labels
- Model restart between cycles prevents confirmation bias
- Per-class balanced selection

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How to avoid confirmation bias? | Model restart between cycles | Section 3 |
| Selection strategy? | Top-K% per class (start 10%, +10%/cycle) | Section 3.2 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How many cycles before convergence? | Dataset-specific | Retraining |
| What if class distribution shifts? | Not addressed | CSD |

---

### Paper: LAPNet-HAR (Lifelong Learning)
**(Paper for Questions/Lifelong_Learning_in_Sensor-Based_Human_Activity_Recognition.pdf)**

**What the paper contributes:**
- Prototypical networks for continual learning
- Experience replay with K samples/class
- EMA prototype updates

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How to prevent catastrophic forgetting? | Replay buffer + EWC | Section 4 |
| Prototype update rate? | α = 0.1-0.3 (EMA) | Section 4.2 |
| Replay buffer size? | K=100 samples/class | Section 4.3 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Storage requirements for long-term? | Not projected | MLOps |
| What if new class appears? | Open-world not covered | Active Learning |

---

## 2.5 Sensor Placement Papers

### Paper: Enhancing HAR via Sensor Displacement Compensation
**(Paper for Questions/Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf)**

**What the paper contributes:**
- Compensation strategies for wrist sensor displacement
- Axis mirroring augmentation
- Position-invariant feature extraction

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| How much does position affect accuracy? | 10-40% drop typical | Section 2 |
| Best compensation strategy? | Axis mirroring + rotation augmentation | Section 4 |
| Does augmentation help at inference? | Training-time only | Section 4 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Dominant vs non-dominant statistics? | Not quantified | Preprocessing |
| Real-time compensation? | Not addressed | Inference |
| Combined with TTA? | Not tested | Retraining |

---

## 2.6 Recent arXiv Papers (2025-2026)

### Paper: HAROOD (2512.10807)
**(Paper for Questions/2512.10807v3.pdf) - KDD 2026**

**What the paper contributes:**
- First comprehensive OOD benchmark for HAR
- Compares 16 OOD detection methods
- 5 HAR datasets, multiple shift types

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Best OOD method for HAR? | **No single winner** - task-dependent | Section 5 |
| Which shifts are hardest? | Cross-position > cross-user > cross-time | Section 5 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| How to select OOD method without validation? | **OPEN QUESTION** | Monitoring |
| Ensemble of OOD methods? | Not tested | Inference |

---

### Paper: ADAPT (2507.08597)
**(Paper for Questions/2507.08597v1.pdf)**

**What the paper contributes:**
- Pseudo-labeling specifically for concept drift
- Semi-supervised drift handling
- No target labels required

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Can pseudo-labels handle drift? | Yes, with adaptive thresholding | Section 3 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Drift detection before adaptation? | Assumes drift already detected | CSD |
| Validation of pseudo-label quality? | **OPEN QUESTION** | Evaluation |

---

### Paper: Entropy-based Drift Detection (2601.00554)
**(Paper for Questions/2601.00554v3.pdf)**

**What the paper contributes:**
- Uses prediction entropy for drift detection
- Triggers retraining based on entropy patterns
- No labels required

**Questions it ANSWERS:**
| Question | Answer | Citation |
|----------|--------|----------|
| Can entropy detect drift? | Yes, entropy increase signals drift | Section 3 |
| Retraining trigger condition? | Sustained entropy above threshold | Section 4 |

**Questions it RAISES:**
| Question | Why Unanswered | Pipeline Stage |
|----------|----------------|----------------|
| Entropy threshold for IMU HAR? | Domain-specific calibration needed | Monitoring |
| ECG warning applies to HAR? | Paper notes underperformance on biosignals | CSD |

**⚠️ WARNING from paper:** Entropy-based detection **underperforms on ECG data** - may apply to HAR biosignals too. Needs investigation.

---

# 3. Question-Centric View (MOST IMPORTANT)

## 3.1 Monitoring Without Labels

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| How to detect model degradation without labels? | WATCH, LIFEWATCH, OOD-HAR, Entropy-drift | ✅ Partial | 🔴 P1 |
| What metrics indicate problems? | Uncertainty papers, Tent, COA-HAR | ✅ Answered | 🔴 P1 |
| How to set thresholds without validation labels? | **ALL papers assume some validation** | ❓ OPEN | 🔴 P1 |
| How to distinguish "OK different" from "bad different"? | Sinkhorn, LIFEWATCH | ⚠️ Partial | 🟠 P2 |
| How often to run monitoring checks? | **No paper addresses frequency** | ❓ OPEN | 🟡 P3 |

**Recommended Thresholds (aggregated from papers):**

| Metric | Warning | Critical | Source |
|--------|---------|----------|--------|
| Entropy | > 1.5 | > 2.0 | Uncertainty papers |
| Confidence | < 0.65 | < 0.50 | MC Dropout paper |
| KS-test statistic | > 0.15 | > 0.25 | Domain adaptation papers |
| Wasserstein drift | > μ + 2σ | > μ + 3σ | WATCH |
| Energy score | > 95th pctl | > 99th pctl | OOD-HAR |
| Flip rate | > 25% | > 40% | Empirical |

---

## 3.2 CSD (Concept/Change/Shift Detection)

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| How to detect distribution drift? | WATCH, LIFEWATCH, KS-test papers | ✅ Answered | 🔴 P1 |
| How to detect concept drift? | Entropy-drift, ADAPT | ✅ Partial | 🔴 P1 |
| How to distinguish drift types? | Sinkhorn (learns which changes matter) | ⚠️ Requires training | 🟠 P2 |
| What window size for drift detection? | **HAR-specific guidance missing** | ❓ OPEN | 🟠 P2 |
| How to handle gradual vs sudden drift? | LIFEWATCH | ⚠️ Partial | 🟡 P3 |

**Drift Detection Decision Tree (from papers):**
```
Production Data Window
        │
        ▼
┌───────────────────┐
│ Compute KS-test   │
│ vs training ref   │
└───────────────────┘
        │
        ▼
    KS > 0.15? ─────No────► Normal operation
        │
       Yes
        ▼
┌───────────────────┐
│ Compute entropy   │
│ and confidence    │
└───────────────────┘
        │
        ▼
    Entropy > 1.5? ─No────► Covariate shift only → AdaBN
        │
       Yes
        ▼
    Consider concept drift → Trigger retraining evaluation
```

---

## 3.3 Pseudo-label Retraining

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| What confidence threshold for pseudo-labels? | SelfHAR, Curriculum | ✅ 0.90+ | 🔴 P1 |
| How to avoid confirmation bias? | Curriculum (restart), SelfHAR (teacher-student) | ✅ Answered | 🔴 P1 |
| How to validate pseudo-label quality? | **No paper provides unlabeled validation** | ❓ OPEN | 🔴 P1 |
| When does self-training become harmful? | **Not characterized** | ❓ OPEN | 🟠 P2 |
| Per-class balanced selection? | Curriculum, A*HAR | ✅ Recommended | 🟠 P2 |

**Pseudo-labeling Protocol (from papers):**
```python
PSEUDO_LABEL_PROTOCOL = {
    'confidence_threshold': 0.90,  # From SelfHAR
    'temperature': 2.0,             # Soft labels
    'per_class_balance': True,      # From Curriculum
    'selection_rate': 0.10,         # Start with top 10%
    'increase_per_cycle': 0.10,     # Add 10% each cycle
    'model_restart': True,          # Prevents confirmation bias
    'validation_check': '???'       # OPEN QUESTION
}
```

---

## 3.4 Active Learning / Label Acquisition

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| Which samples to query for labels? | CODA (uncertainty + diversity) | ✅ Answered | 🟠 P2 |
| How many labels needed? | CoDATS (5-10%), AdaptNet (10-30%) | ✅ Range given | 🟠 P2 |
| Can we use weak labels (proportions)? | CoDATS | ✅ Yes | 🟠 P2 |
| Human-in-the-loop architecture? | Digital Twins paper | ⚠️ Conceptual | 🟡 P3 |
| When to stop querying? | **Budget-based only** | ❓ OPEN | 🟡 P3 |

---

## 3.5 Sensor Placement / Handedness

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| Accuracy drop from position mismatch? | Displacement paper | ✅ 10-40% | 🔴 P1 |
| Dominant vs non-dominant statistics? | **No paper quantifies** | ❓ OPEN | 🔴 P1 |
| Position-invariant features? | IMUCoCo, Displacement paper | ⚠️ Training-time only | 🟠 P2 |
| Runtime compensation possible? | **Not addressed** | ❓ OPEN | 🟠 P2 |
| Can single model handle both wrists? | **Assumed but not validated** | ❓ OPEN | 🔴 P1 |

**Hand Placement Cases (from HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md):**

| Case | Watch | Activity | Population | Signal Quality |
|------|-------|----------|------------|----------------|
| A | Dominant | Dominant | ~7% | BEST |
| B | Non-dominant | Dominant | ~63% | WORST |
| C | Dominant | Non-dominant | ~3% | Good |
| D | Non-dominant | Non-dominant | ~27% | Moderate |

**⚠️ KEY INSIGHT:** Most users (63%) are in Case B (worst signal) — papers do not address this!

---

## 3.6 Unit Tests & CI/CD

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| What to test in ML pipeline? | **No HAR-specific guidance** | ❓ OPEN | 🔴 P1 |
| Model validation without labels? | **Not addressed** | ❓ OPEN | 🔴 P1 |
| Retraining automation trigger? | Entropy-drift paper | ⚠️ Conceptual | 🟠 P2 |
| Rollback conditions? | **Not addressed** | ❓ OPEN | 🟠 P2 |
| Canary deployment for HAR? | **Not addressed** | ❓ OPEN | 🟡 P3 |

**CI/CD Questions NOT Addressed by Papers:**
1. How to validate model update without labeled test set?
2. What metrics trigger automatic rollback?
3. How to A/B test HAR models in production?
4. Version control for streaming model updates?

---

## 3.7 Evaluation Without Labels

| Question | Papers Addressing | Answer Status | Priority |
|----------|-------------------|---------------|----------|
| Proxy metrics for accuracy? | Entropy, confidence, flip rate | ✅ Partial | 🔴 P1 |
| Correlation with actual accuracy? | **Needs labeled validation** | ❓ OPEN | 🔴 P1 |
| How to compare model versions? | **No unlabeled comparison** | ❓ OPEN | 🔴 P1 |
| Calibration without labels? | Temperature scaling (needs val set) | ❌ Requires labels | 🟠 P2 |

---

# 4. Questions We Can Answer Now

Based on paper analysis, these questions have clear answers:

| # | Question | Answer | Source |
|---|----------|--------|--------|
| 1 | Can we adapt without ANY labels? | YES - AdaBN, Tent, COA-HAR, OFTTA | Multiple TTA papers |
| 2 | What confidence threshold for pseudo-labels? | ≥ 0.90 | SelfHAR, Curriculum |
| 3 | How many MC Dropout samples? | 30 | Uncertainty paper |
| 4 | How to detect distribution drift? | KS-test, Wasserstein on sliding windows | WATCH, Domain papers |
| 5 | Best OOD score for HAR? | Energy score (simple, effective) | OOD-HAR |
| 6 | How to prevent catastrophic forgetting? | Replay buffer + EWC | LAPNet-HAR |
| 7 | Position mismatch impact? | 10-40% accuracy drop | Displacement paper |
| 8 | Entropy threshold for uncertainty? | > 1.5 warning, > 2.0 critical | Uncertainty papers |
| 9 | AdaBN implementation? | Update BN running stats with target data | XHAR |
| 10 | Self-training bias prevention? | Model restart between cycles | Curriculum |

---

# 5. Questions We Must Ask Mentor

These questions require domain expertise or thesis-scope decisions:

| # | Question | Why Ask Mentor | Context |
|---|----------|----------------|---------|
| 1 | How to validate model without ANY labels? | Fundamental thesis gap | All papers assume some validation labels |
| 2 | Acceptable label budget for audit set? | Resource constraint | Papers suggest 5-30%, we need specific number |
| 3 | Is entropy underperformance on biosignals a blocker? | Risk assessment | Paper 2601.00554 warns about this |
| 4 | Should we treat handedness as separate domain? | Architectural decision | Papers don't address |
| 5 | What's the acceptable drift detection latency? | Operational requirement | No paper specifies for HAR |
| 6 | Can thesis claim novelty on "truly unlabeled"? | Scope validation | Most papers assume at least some labels |
| 7 | Unit test coverage expectations? | Academic requirement | No ML testing standards in papers |

---

# 6. Open Research Gaps

## 6.1 Gaps Not Addressed by ANY Paper

| Gap | Impact | Suggested Keywords for Search |
|-----|--------|------------------------------|
| **Validate adaptation without labels** | Cannot know if TTA worked | "unsupervised adaptation validation", "proxy metrics accuracy correlation" |
| **Long-term TTA stability** | May degrade over months | "continual test-time adaptation", "TTA stability", "entropy collapse" |
| **Combining TTA + CL + OOD** | No unified pipeline | "unified continual adaptation", "OOD-aware TTA" |
| **Hand placement metadata** | Cannot detect mismatch | "wrist handedness detection", "sensor position inference" |
| **Truly zero-label operation** | All papers assume some labels | "fully unsupervised HAR", "zero-shot activity recognition" |
| **CI/CD for streaming models** | No automation patterns | "MLOps streaming ML", "continuous model deployment" |
| **Rollback conditions** | What triggers reverting model? | "model rollback policy", "safe ML deployment" |

## 6.2 Gaps Partially Addressed

| Gap | What's Covered | What's Missing | Papers |
|-----|----------------|----------------|--------|
| Drift detection | Statistical tests | HAR-specific thresholds | WATCH, LIFEWATCH |
| Uncertainty | Methods | Calibration without labels | Uncertainty papers |
| Position invariance | Training augmentation | Runtime compensation | Displacement |
| Pseudo-labeling | Confidence thresholds | Quality validation | SelfHAR, Curriculum |

---

# 7. Pipeline Stage Mapping

## 7.1 Which Papers Affect Which Stage

| Stage | Critical Papers | Key Techniques |
|-------|-----------------|----------------|
| **Preprocessing** | Displacement, sensors-* | Axis mirroring, scaling, jittering |
| **Training** | ContrasGAN, SelfHAR, Transfer Review | Self-supervised pretraining, domain alignment |
| **Inference** | Tent, COA-HAR, OFTTA, XHAR | TTA, AdaBN, entropy monitoring |
| **Monitoring** | WATCH, OOD-HAR, Uncertainty papers | Wasserstein CPD, energy scores, entropy |
| **CSD** | LIFEWATCH, Entropy-drift, Sinkhorn | Pattern memory, drift classification |
| **Trigger Policy** | Entropy-drift, CODA | Entropy thresholds, budget constraints |
| **Retraining** | Curriculum, LAPNet-HAR, ADAPT | Pseudo-labeling, replay buffer |
| **Active Learning** | CODA, CoDATS, Human-in-loop | Uncertainty sampling, weak supervision |
| **Evaluation** | HAROOD benchmark | OOD detection comparison |
| **MLOps/CI-CD** | **NONE DIRECTLY** | Must derive from general MLOps |

## 7.2 Implementation Sequence

```
Week 1-2: Foundation (Inference + Monitoring)
├── Implement AdaBN from XHAR
├── Add Tent entropy adaptation
├── Implement Wasserstein drift detection (WATCH)
└── Add OOD energy scoring

Week 3-4: Evaluation Enhancement
├── MC Dropout uncertainty (30 samples)
├── Entropy/confidence logging
├── Drift alert thresholds
└── LIFEWATCH pattern memory

Week 5-6: Retraining Pipeline
├── Pseudo-labeling (SelfHAR style)
├── Curriculum selection
├── Replay buffer (LAPNet)
└── Model versioning

Week 7-8: Active Learning + CI/CD
├── CODA-style sample selection
├── Weak supervision option (CoDATS)
├── GitHub Actions workflow
└── Automated testing
```

---

# 8. Implementation Priority

## 8.1 Must-Have for Thesis (🔴 Critical)

| Priority | Component | Paper Source | Effort |
|----------|-----------|--------------|--------|
| 1 | AdaBN adaptation | XHAR | 1 day |
| 2 | Tent entropy TTA | Tent | 1 day |
| 3 | KS-test drift detection | Domain papers | 1 day |
| 4 | Entropy/confidence monitoring | Uncertainty papers | 1 day |
| 5 | Pseudo-labeling (>0.90) | SelfHAR | 3 days |
| 6 | Model restart between cycles | Curriculum | 1 day |
| 7 | OOD energy scoring | OOD-HAR | 2 days |
| 8 | Basic CI/CD | (Derive) | 2 days |

## 8.2 Should-Have (🟠 High)

| Priority | Component | Paper Source | Effort |
|----------|-----------|--------------|--------|
| 9 | COA-HAR contrastive TTA | COA-HAR | 1 week |
| 10 | Wasserstein CPD (WATCH) | WATCH | 3 days |
| 11 | Replay buffer | LAPNet-HAR | 3 days |
| 12 | Active learning selection | CODA | 1 week |
| 13 | Position augmentation | Displacement | 2 days |

## 8.3 Nice-to-Have (🟡 Future Work)

| Priority | Component | Paper Source | Effort |
|----------|-----------|--------------|--------|
| 14 | ContrasGAN UDA | ContrasGAN | 2 weeks |
| 15 | LIFEWATCH pattern memory | LIFEWATCH | 1 week |
| 16 | Sinkhorn divergence | Sinkhorn | 1 week |
| 17 | Aleatoric/epistemic decomposition | Personalizing paper | 1 week |
| 18 | Weak supervision (proportions) | CoDATS | 1 week |

---

# Appendix A: Citation Format Reference

All citations in this document follow the format:
```
(Paper for Questions/<filename>.pdf, p.X, section "...")
```

For papers that require reading to extract specific page numbers, the format is:
```
(Paper for Questions/<filename>.pdf, Section X) — TO BE VERIFIED
```

---

# Appendix B: Cross-Reference to Other Docs

| Related Document | Purpose |
|------------------|---------|
| [THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md](../../paper%20for%20questions/THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md) | Specific implementation guidance |
| [HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md](HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md) | Hand placement statistics |
| [BIG_QUESTIONS_2026-01-18.md](../BIG_QUESTIONS_2026-01-18.md) | Broader thesis questions |
| [PIPELINE_DEEP_DIVE_opus.md](../PIPELINE_DEEP_DIVE_opus.md) | Current pipeline architecture |

---

**Document Status:** Complete  
**Next Review:** After reading specific pages of critical papers  
**Action Required:** Fill in specific page numbers after manual PDF review

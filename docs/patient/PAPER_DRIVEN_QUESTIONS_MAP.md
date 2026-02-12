# 📚 PAPER-DRIVEN QUESTIONS MAP
## HAR MLOps Thesis - Question Extraction from "Paper for Questions" Folder

**Generated:** January 30, 2026  
**Purpose:** Extract questions from papers for thesis pipeline development  
**Total Papers Analyzed:** 88 files (PDFs + analysis markdown files)

---

# 🔖 TABLE OF CONTENTS

1. [Paper Inventory](#1-paper-inventory)
2. [Paper-by-Paper Question Extraction](#2-paper-by-paper-question-extraction)
3. [Question-Centric Grouping by Pipeline Stage](#3-question-centric-grouping-by-pipeline-stage)
4. [Questions We Can Answer Now](#4-questions-we-can-answer-now)
5. [Questions for Mentor](#5-questions-for-mentor)
6. [Open Research Gaps](#6-open-research-gaps)

---

# 1. PAPER INVENTORY

## 1.1 Papers by Category

### 🔴 CRITICAL: Direct Application (No Target Labels Required)

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **COA-HAR** | 2025 | Contrastive test-time adaptation, zero labels | (Paper for Questions/1-s2.0-S0957417425029045-main.pdf) |
| **ContrasGAN** | 2021 | UDA via adversarial + contrastive, zero target labels | (Paper for Questions/1-s2.0-S1574119221001103-main.pdf) |
| **Tent** | 2021 | Entropy minimization TTA, source-free | (Paper for Questions/2006.10726v3.pdf) |
| **OFTTA** | 2023 | Optimization-free test-time adaptation | (Paper for Questions/3631450.pdf) |
| **XHAR** | 2019 | AdaBN technique for instant adaptation | (Paper for Questions/XHAR_Deep_Domain_Adaptation_for_Human_Activity_Recognition_with_Smart_Devices.pdf) |

### 🟠 HIGH: Requires Small Labeled Set or Pseudo-Labels

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **SelfHAR** | 2021 | Self-training with unlabeled data | (Paper for Questions/3448112.pdf, Paper for Questions/imwut-selfhar.pdf) |
| **Curriculum Labeling** | 2021 | Self-paced pseudo-labeling | (Paper for Questions/Curriculum_Labeling_Self-paced_Pseudo-Labeling_for.pdf) |
| **PACL+** | 2025 | Online CL with Gaussian replay | (Paper for Questions/1-s2.0-S0957417425022225-main.pdf) |
| **CODA** | 2024 | Cost-efficient active learning for HAR | (Paper for Questions/2403.14922v1.pdf) |
| **CoDATS** | 2020 | Multi-source UDA with weak supervision | (Paper for Questions/3394486.3403228.pdf) |

### 🟡 MEDIUM: Surveys and Methodological References

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **UDA Benchmark for TSC** | 2025 | Standardized UDA evaluation | (Paper for Questions/2312.09857v3.pdf) |
| **Transfer Learning Review** | 2022 | Comprehensive TL taxonomy | (Paper for Questions/Transfer_Learning_of_Human_Activities_Based_on_IMU_Sensors_A_Review.pdf) |
| **SSL for HAR Assessment** | 2022 | Self-supervised method comparison | (Paper for Questions/3550299.pdf) |
| **DAGHAR Benchmark** | 2024 | Cross-dataset HAR standardization | (Paper for Questions/41597_2024_Article_3951.pdf) |

### 🟢 Uncertainty & Confidence (Works Without Labels at Inference)

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **XAI-BayesHAR** | 2022 | Kalman filter uncertainty | (Paper for Questions/XAI-BayesHAR_A_novel_Framework_for_Human_Activity_Recognition_with_Integrated_Uncertainty_and_Shapely_Values.pdf) |
| **MC Dropout for HAR** | 2021 | Uncertainty comparison | (Paper for Questions/A_Deep_Learning_Assisted_Method_for_Measuring_Uncertainty_in_Activity_Recognition_with_Wearable_Sensors.pdf) |
| **Personalizing via Uncertainty** | 2020 | Aleatoric + epistemic separation | (Paper for Questions/Personalizing_Activity_Recognition_Models_Through_Quantifying_Different_Types_of_Uncertainty_Using_Wearable_Sensors.pdf) |

### 🔵 Change Point Detection / Drift

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **LIFEWATCH** | 2024 | Lifelong Wasserstein CPD | (Paper for Questions/LIFEWATCH_Lifelong_Wasserstein_Change_Point_Detection.pdf) |
| **WATCH** | 2023 | Wasserstein CPD for high-dim TS | (Paper for Questions/WATCH_Wasserstein_Change_Point_Detection_for_High-Dimensional_Time_Series_Data.pdf) |
| **Sinkhorn CPD** | 2022 | Optimal transport for CPD | (Paper for Questions/Learning_Sinkhorn_Divergences_for_Change_Point_Detection.pdf) |
| **Normalizing SSL for CPD** | 2023 | Self-supervised CPD | (Paper for Questions/Normalizing_Self-Supervised_Learning_for_Provably_Reliable_Change_Point_Detection.pdf) |

### 🟣 Sensor Placement / Handedness

| Paper | Year | Key Contribution | Citation |
|-------|------|------------------|----------|
| **Sensor Displacement Compensation** | 2024 | Wrist-worn position variance | (Paper for Questions/Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf) |
| **Deep UDA Survey for Sensors** | 2022 | Position as domain gap source | (Paper for Questions/sensors-22-05507-v2.pdf) |

## 1.2 Summary Statistics

| Category | Count | Labels Required |
|----------|-------|-----------------|
| Direct Application (Zero Labels) | 5 | ❌ None |
| Requires Small Labeled Set | 5 | ⚠️ Partial |
| Surveys/References | 4 | Various |
| Uncertainty Methods | 3 | ❌ None at inference |
| Drift Detection | 4 | ❌ None |
| Sensor Placement | 2 | Various |
| **Total Unique Papers** | **23 key papers** | - |

---

# 2. PAPER-BY-PAPER QUESTION EXTRACTION

## 2.1 COA-HAR (2025) - ⭐⭐⭐⭐⭐ DIRECT APPLICATION

**Source:** (Paper for Questions/1-s2.0-S0957417425029045-main.pdf)

### Summary
Contrastive Online Test-Time Adaptation for wearable sensor HAR using sensor data augmentation. Adapts at inference time with ZERO target labels using self-supervised contrastive loss.

### Assumptions
- Pre-trained on labeled source domain (ADAMSense in our case)
- Same activity classes in source and target
- Access to mini-batches of unlabeled target data
- BatchNorm layers exist in architecture

### What It Solves
✅ Adapt to unlabeled production data without ANY labels  
✅ IMU-specific augmentation strategies (time-warp, scale, rotate, noise)  
✅ Update only BN + final layers (preserves source knowledge)  
✅ 5-15% accuracy improvement on shifted domains

### What It Does NOT Solve
❌ How to detect when TTA is helping vs hurting  
❌ Handling concept drift (same activities, different behavior patterns)  
❌ Combining TTA with continual learning for long-term deployment  
❌ Computational overhead for real-time inference

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **How to monitor if contrastive TTA is improving predictions?** | Monitoring | 🔴 HIGH |
| **What batch size is optimal for our 50Hz IMU windows?** | Inference | 🟠 MEDIUM |
| **Can we combine COA-HAR with Tent entropy minimization?** | Inference | 🟡 LOW |
| **How often should adaptation occur? Per-batch or periodic?** | MLOps | 🔴 HIGH |
| **What augmentations work best for 6-axis IMU (AX,AY,AZ,GX,GY,GZ)?** | Preprocessing | 🔴 HIGH |

---

## 2.2 ContrasGAN (2021) - ⭐⭐⭐⭐⭐ DIRECT APPLICATION

**Source:** (Paper for Questions/1-s2.0-S1574119221001103-main.pdf)

### Summary
Unsupervised Domain Adaptation via adversarial (GAN) + contrastive learning. Trains on labeled source + unlabeled target, then deploys adapted model.

### Assumptions
- Labeled source domain available
- Unlabeled target domain accessible during training (not just inference)
- Same activity classes across domains
- GAN training is stable (requires hyperparameter tuning)

### What It Solves
✅ True UDA with zero target labels  
✅ 5-15% improvement over no-adaptation baseline  
✅ Combines adversarial (distribution alignment) + contrastive (class structure)  
✅ Validated on OPPORTUNITY, PAMAP2, HHAR

### What It Does NOT Solve
❌ How much unlabeled target data needed before GAN training converges  
❌ Continuous domain shift (not one-time adaptation)  
❌ What if source and target activity distributions differ significantly  
❌ Detecting GAN convergence without labels

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **How much production data needed before running ContrasGAN?** | Data | 🔴 HIGH |
| **How to detect when adversarial alignment has failed?** | Monitoring | 🔴 HIGH |
| **Can we adapt the GAN architecture for 1D-CNN-BiLSTM?** | Training | 🟠 MEDIUM |
| **How to handle periodic re-adaptation as more data arrives?** | Retraining | 🟠 MEDIUM |
| **What's the GAN training time and compute requirements?** | MLOps | 🟡 LOW |

---

## 2.3 Tent (2021) - ⭐⭐⭐⭐⭐ DIRECT APPLICATION

**Source:** (Paper for Questions/2006.10726v3.pdf)

### Summary
Fully test-time adaptation by entropy minimization. Updates ONLY BatchNorm affine parameters (γ, β) during inference using entropy loss as self-supervision signal.

### Assumptions
- BatchNorm layers exist (our 1D-CNN has BN, BiLSTM typically doesn't)
- Reasonable batch sizes (64-128 recommended)
- Model not already collapsed to trivial solution
- Source pre-training produces well-calibrated confidence

### What It Solves
✅ Source-free adaptation (no access to source data during deployment)  
✅ ~10 lines of code to implement  
✅ Updates only ~1% of parameters  
✅ Real-time batch-wise adaptation

### What It Does NOT Solve
❌ BiLSTM compatibility (LayerNorm vs BatchNorm)  
❌ Risk of entropy collapse (all predictions same class)  
❌ Long-term stability of repeated entropy minimization  
❌ Time-series specific validation (paper tested on images)

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **Does Tent work with BiLSTM's LayerNorm or only CNN's BatchNorm?** | Training | 🔴 HIGH |
| **What entropy threshold indicates domain shift vs normal variation?** | Monitoring | 🔴 HIGH |
| **How to prevent entropy collapse with our 11 activity classes?** | Inference | 🔴 HIGH |
| **Can we use entropy as drift detection signal before triggering retraining?** | Trigger Policy | 🟠 MEDIUM |
| **What learning rate for entropy minimization on HAR?** | Inference | 🟠 MEDIUM |

---

## 2.4 SelfHAR (2021) - ⭐⭐⭐⭐ HIGH RELEVANCE

**Source:** (Paper for Questions/3448112.pdf, Paper for Questions/imwut-selfhar.pdf)

### Summary
Self-training with unlabeled data via teacher-student framework + multi-task self-supervision pretext tasks. Achieves 10x reduction in labeled data needed.

### Assumptions
- Small labeled dataset available (not zero labels)
- Large unlabeled dataset for self-supervision
- Self-supervised pretext tasks transferable to HAR
- Teacher model quality is reasonable

### What It Solves
✅ Leverage massive unlabeled production data  
✅ Up to 12% F1 improvement  
✅ Self-supervised pretext tasks for IMU: reconstruction, transformation prediction  
✅ Soft pseudo-labels with confidence weighting (not hard thresholds)

### What It Does NOT Solve
❌ What if we have ZERO initial labeled data  
❌ How to update teacher model in production  
❌ Detecting when self-training produces harmful pseudo-labels  
❌ Handling distribution shift during self-training

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **What's minimum labeled data needed to bootstrap SelfHAR?** | Active Learning | 🔴 HIGH |
| **Which pretext tasks work best for 6-axis IMU?** | Training | 🔴 HIGH |
| **How to detect pseudo-label quality degradation without labels?** | Monitoring | 🔴 HIGH |
| **Can we combine SelfHAR with ContrasGAN for fully unlabeled scenario?** | Retraining | 🟠 MEDIUM |
| **What temperature (T) for soft pseudo-labels?** | Retraining | 🟡 LOW |

---

## 2.5 Curriculum Labeling (2021) - ⭐⭐⭐⭐ HIGH RELEVANCE

**Source:** (Paper for Questions/Curriculum_Labeling_Self-paced_Pseudo-Labeling_for.pdf)

### Summary
Self-paced pseudo-labeling with curriculum learning. Key insight: RESTART model parameters between cycles to avoid confirmation bias.

### Assumptions
- Small initial labeled set exists
- Batch setting (full unlabeled set available)
- Per-class sample selection possible
- Model restarts are feasible (not online)

### What It Solves
✅ Prevents confirmation bias via model restart  
✅ Uses relative ranking (top-K%) instead of fixed confidence threshold  
✅ Per-class balancing for imbalanced datasets  
✅ Clear algorithm structure (cycles of: restart → pseudo-label → select → train)

### What It Does NOT Solve
❌ Zero initial labels (needs bootstrap)  
❌ Streaming data (designed for batch)  
❌ How many cycles until convergence for HAR  
❌ BiLSTM-specific considerations

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **How to apply curriculum learning with streaming production data?** | Retraining | 🔴 HIGH |
| **What initial K% to select per class for our 11 activities?** | Retraining | 🟠 MEDIUM |
| **How to bootstrap without ANY initial labels?** | Active Learning | 🔴 HIGH |
| **How many curriculum cycles needed for HAR convergence?** | Retraining | 🟠 MEDIUM |
| **Can model restart strategy work with continual learning?** | Retraining | 🟡 LOW |

---

## 2.6 XAI-BayesHAR (2022) - ⭐⭐⭐⭐ UNCERTAINTY QUANTIFICATION

**Source:** (Paper for Questions/XAI-BayesHAR_A_novel_Framework_for_Human_Activity_Recognition_with_Integrated_Uncertainty_and_Shapely_Values.pdf)

### Summary
Kalman filter on feature embeddings to track uncertainty + SHAP values for explainability. Provides uncertainty WITHOUT requiring labels at inference.

### Assumptions
- Features approximately Gaussian
- Stationary dynamics (process noise Q known)
- Kalman filter parameters well-tuned on training data

### What It Solves
✅ Uncertainty quantification without labels  
✅ Innovation (residual) detects OOD/domain shift  
✅ Minimal overhead (~1ms per sample)  
✅ Covariance growth indicates model "doesn't know"

### What It Does NOT Solve
❌ Non-Gaussian feature distributions  
❌ Kalman parameter calibration without labeled validation set  
❌ Long-term covariance drift (may need periodic reset)  
❌ Relationship between Kalman uncertainty and actual accuracy

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **Are our 1D-CNN-BiLSTM features approximately Gaussian?** | Evaluation | 🟠 MEDIUM |
| **What innovation threshold indicates OOD for our activities?** | Monitoring | 🔴 HIGH |
| **How often to reset Kalman covariance in production?** | Monitoring | 🟠 MEDIUM |
| **Can Kalman uncertainty trigger retraining policy?** | Trigger Policy | 🔴 HIGH |
| **How to calibrate Q (process noise) without labeled validation?** | Training | 🟠 MEDIUM |

---

## 2.7 MC Dropout / Ensemble Methods (2021) - ⭐⭐⭐⭐ UNCERTAINTY QUANTIFICATION

**Source:** (Paper for Questions/A_Deep_Learning_Assisted_Method_for_Measuring_Uncertainty_in_Activity_Recognition_with_Wearable_Sensors.pdf)

### Summary
Systematic comparison of MC Dropout, Deep Ensembles, and Evidential Deep Learning for HAR uncertainty quantification.

### Assumptions
- Dropout layers exist (or can be added)
- Computational budget for multiple forward passes (MC Dropout: 30×, Ensemble: 5×)
- Training data representative of deployment distribution

### What It Solves
✅ MC Dropout: Easy to add, 30 forward passes → uncertainty  
✅ Ensemble: Best OOD detection (0.86-0.93 AUROC)  
✅ Evidential: 1× inference cost, single forward pass  
✅ Comparison metrics for method selection

### What It Does NOT Solve
❌ Which method to choose for specific HAR deployment  
❌ Real-time feasibility (30× passes may be too slow)  
❌ Calibration without labeled data  
❌ BiLSTM-specific dropout considerations

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **Is 30× inference acceptable for our pipeline latency requirements?** | Inference | 🔴 HIGH |
| **Which uncertainty method (MC Dropout/Ensemble/Evidential) best detects drift?** | Monitoring | 🔴 HIGH |
| **Can we use ensemble disagreement as trigger for human labeling?** | Active Learning | 🟠 MEDIUM |
| **What's the correlation between uncertainty and actual errors?** | Evaluation | 🟠 MEDIUM |
| **How to add dropout to BiLSTM layers?** | Training | 🟡 LOW |

---

## 2.8 LIFEWATCH (2024) - ⭐⭐⭐⭐ DRIFT DETECTION

**Source:** (Paper for Questions/LIFEWATCH_Lifelong_Wasserstein_Change_Point_Detection.pdf)

### Summary
Lifelong Wasserstein change point detection that maintains a "memory" of past distribution states for pattern recognition in drift.

### Assumptions
- Sufficient historical data for pattern library
- Wasserstein distance computable for high-dimensional data
- Change points have detectable signatures

### What It Solves
✅ Detects distribution changes without labels  
✅ Recognizes recurring patterns (e.g., seasonal user behavior)  
✅ Wasserstein distance is principled metric for distribution shift  
✅ Memory enables long-term monitoring

### What It Does NOT Solve
❌ What action to take after detecting change point  
❌ Distinguishing covariate shift vs concept drift  
❌ Real-time computation for high-dimensional IMU features  
❌ Threshold selection for Wasserstein distance

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **What Wasserstein threshold indicates retraining need for HAR?** | Trigger Policy | 🔴 HIGH |
| **How much history to maintain in pattern memory?** | Monitoring | 🟠 MEDIUM |
| **Can LIFEWATCH distinguish user change vs activity change?** | Monitoring | 🔴 HIGH |
| **Is Wasserstein distance computationally feasible for 200×6 windows?** | Monitoring | 🟠 MEDIUM |
| **How to integrate LIFEWATCH with our KS-test monitoring?** | Monitoring | 🟡 LOW |

---

## 2.9 Sensor Displacement Compensation (2024) - ⭐⭐⭐⭐ HANDEDNESS

**Source:** (Paper for Questions/Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf)

### Summary
Compensation strategies for sensor displacement on wrist-worn devices. Includes data augmentation for position variance.

### Assumptions
- Sensor displacement is the primary domain gap
- Training data can include position variations
- Compensation can be learned from augmentation

### What It Solves
✅ Quantifies accuracy drop with position displacement (15-35%)  
✅ Compensation strategies specific to wrist sensors  
✅ Augmentation techniques for position variance  
✅ Single model approach for multiple positions

### What It Does NOT Solve
❌ Left vs right wrist specifically (focuses on same-wrist displacement)  
❌ Detecting position mismatch without labels  
❌ Confidence calibration under position shift  
❌ Runtime adaptation without labeled target data

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **Does left/right wrist mismatch behave like same-wrist displacement?** | Preprocessing | 🔴 HIGH |
| **What augmentations simulate dominant/non-dominant hand switch?** | Training | 🔴 HIGH |
| **Can we detect handedness mismatch from feature statistics?** | Monitoring | 🔴 HIGH |
| **What accuracy drop to expect from wrong-hand watch wearing?** | Evaluation | 🟠 MEDIUM |
| **Is axis transformation (AX↔-AX) sufficient for hand compensation?** | Preprocessing | 🟠 MEDIUM |

---

## 2.10 CODA (2024) - ⭐⭐⭐⭐ ACTIVE LEARNING

**Source:** (Paper for Questions/2403.14922v1.pdf)

### Summary
Cost-efficient test-time domain adaptation for HAR using active learning + clustering loss. Validated on watch-based HAR.

### Assumptions
- Some human labeling budget available
- Clustering structure meaningful in feature space
- Instance-level adaptation feasible
- User-induced concept drift is gradual

### What It Solves
✅ Watch-based HAR (similar to Garmin)  
✅ 3.7-17.4% accuracy improvement  
✅ Cost-efficient (minimal labels via active learning)  
✅ On-device feasibility

### What It Does NOT Solve
❌ Specific label budget quantification  
❌ Integration with CI/CD pipeline  
❌ Long-term stability (months of deployment)  
❌ Combining with Tent approach

### Questions It Creates for OUR Pipeline

| Question | Pipeline Stage | Priority |
|----------|---------------|----------|
| **How many samples need human labeling for CODA to be effective?** | Active Learning | 🔴 HIGH |
| **What importance score threshold triggers human query?** | Active Learning | 🔴 HIGH |
| **Can we use confidence + diversity for sample selection?** | Active Learning | 🟠 MEDIUM |
| **How to integrate CODA with periodic centralized retraining?** | MLOps | 🟠 MEDIUM |
| **Is CODA's clustering loss compatible with our 11 activity classes?** | Retraining | 🟡 LOW |

---

## 2.11 Additional Key Papers Summary

### PACL+ (2025) - Online Continual Learning
**Source:** (Paper for Questions/1-s2.0-S0957417425022225-main.pdf)

| Question | Pipeline Stage |
|----------|---------------|
| Can PACL+ work with pseudo-labels instead of ground truth? | Retraining |
| How sensitive is Gaussian replay to non-Gaussian HAR features? | Retraining |
| How to update proxy anchors without labels? | Retraining |

### CrossHAR (2024) - Cross-Dataset Generalization
**Source:** (Paper for Questions/3666025.3699339.pdf)

| Question | Pipeline Stage |
|----------|---------------|
| Does physics-based augmentation help our IMU data? | Training |
| How much source dataset diversity needed? | Training |
| Can hierarchical SSL features detect drift? | Monitoring |

### Hi-OSCAR (2025) - Open-Set Classification
**Source:** (Paper for Questions/3770681.pdf)

| Question | Pipeline Stage |
|----------|---------------|
| How to handle unknown activities in production? | Inference |
| Can we automatically construct activity hierarchy? | Training |
| What OOD threshold without labeled validation? | Monitoring |

### AdaShadow (2024) - Fast TTA
**Source:** (Paper for Questions/3666025.3699339.pdf)

| Question | Pipeline Stage |
|----------|---------------|
| Which layers most important to adapt for HAR? | Inference |
| What speedup achievable with selective adaptation? | Inference |
| How to balance adaptation speed vs accuracy? | Inference |

---

# 3. QUESTION-CENTRIC GROUPING BY PIPELINE STAGE

## 3.1 MONITORING WITHOUT LABELS

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | What entropy threshold indicates domain shift vs normal variation? | Tent | 🔴 HIGH |
| 2 | What innovation threshold indicates OOD for our activities? | XAI-BayesHAR | 🔴 HIGH |
| 3 | How to detect pseudo-label quality degradation without labels? | SelfHAR | 🔴 HIGH |
| 4 | Can Kalman uncertainty trigger retraining policy? | XAI-BayesHAR | 🔴 HIGH |
| 5 | What Wasserstein threshold indicates retraining need? | LIFEWATCH | 🔴 HIGH |
| 6 | Which uncertainty method best detects drift? | MC Dropout paper | 🔴 HIGH |
| 7 | Can LIFEWATCH distinguish user change vs activity change? | LIFEWATCH | 🔴 HIGH |
| 8 | Can we detect handedness mismatch from feature statistics? | Sensor Displacement | 🔴 HIGH |
| 9 | How to monitor if contrastive TTA is improving predictions? | COA-HAR | 🔴 HIGH |
| 10 | How to detect when adversarial alignment has failed? | ContrasGAN | 🔴 HIGH |

## 3.2 CSD / DRIFT DETECTION

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | What KS-test p-value threshold for per-channel drift? | WATCH | 🔴 HIGH |
| 2 | Is PSI > 0.25 appropriate for our 6-axis IMU? | Best practices | 🟠 MEDIUM |
| 3 | How often to run drift detection (every batch vs hourly)? | LIFEWATCH | 🟠 MEDIUM |
| 4 | Can we use Wasserstein distance instead of KS-test? | LIFEWATCH/WATCH | 🟠 MEDIUM |
| 5 | How much history to maintain in pattern memory? | LIFEWATCH | 🟠 MEDIUM |
| 6 | How to distinguish covariate shift from concept drift? | UDA Benchmark | 🔴 HIGH |
| 7 | What baseline statistics to compare against (training vs recent)? | General | 🟠 MEDIUM |

## 3.3 PSEUDO-LABEL RETRAINING

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | What confidence threshold for pseudo-label acceptance? | Curriculum Labeling | 🔴 HIGH |
| 2 | How to prevent confirmation bias in self-training? | Curriculum Labeling | 🔴 HIGH |
| 3 | Should we restart model weights between retraining cycles? | Curriculum Labeling | 🔴 HIGH |
| 4 | What temperature (T) for soft pseudo-labels? | SelfHAR | 🟠 MEDIUM |
| 5 | How many curriculum cycles until convergence? | Curriculum Labeling | 🟠 MEDIUM |
| 6 | Can PACL+ Gaussian replay work with pseudo-labels? | PACL+ | 🟠 MEDIUM |
| 7 | How to update teacher model in production? | SelfHAR | 🟠 MEDIUM |
| 8 | What's minimum labeled data to bootstrap pseudo-labeling? | Curriculum Labeling | 🔴 HIGH |

## 3.4 ACTIVE LEARNING

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | How many samples need human labeling per week/month? | CODA | 🔴 HIGH |
| 2 | What importance score threshold triggers human query? | CODA | 🔴 HIGH |
| 3 | Should we use uncertainty sampling or diversity sampling? | General | 🟠 MEDIUM |
| 4 | Can ensemble disagreement trigger human labeling? | MC Dropout paper | 🟠 MEDIUM |
| 5 | How to present IMU data for human labeling? | General | 🔴 HIGH |
| 6 | What's labeling interface for HAR (video sync required)? | General | 🟠 MEDIUM |
| 7 | Is margin sampling better than entropy for HAR? | CODA | 🟡 LOW |

## 3.5 SENSOR PLACEMENT / HANDEDNESS

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | Does left/right wrist behave like same-wrist displacement? | Sensor Displacement | 🔴 HIGH |
| 2 | What augmentations simulate hand switch? | Sensor Displacement | 🔴 HIGH |
| 3 | Is axis transformation (AX↔-AX) sufficient for hand compensation? | Sensor Displacement | 🟠 MEDIUM |
| 4 | What accuracy drop from wrong-hand watch wearing? | Sensor Displacement | 🟠 MEDIUM |
| 5 | Should we train separate left/right models? | General | 🟡 LOW |
| 6 | Can AdaBN handle handedness shift? | XHAR | 🟠 MEDIUM |

## 3.6 EVALUATION (WITHOUT LABELS)

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | What proxy metrics correlate with actual accuracy? | General | 🔴 HIGH |
| 2 | Is confidence mean a reliable proxy for accuracy? | Uncertainty papers | 🔴 HIGH |
| 3 | Can we use entropy distribution as quality metric? | Tent | 🟠 MEDIUM |
| 4 | What's ECE (Expected Calibration Error) without labels? | General | 🟠 MEDIUM |
| 5 | Is temporal plausibility (transition patterns) reliable proxy? | General | 🟠 MEDIUM |
| 6 | What's correlation between uncertainty and actual errors? | MC Dropout paper | 🟠 MEDIUM |

## 3.7 MLOps FEASIBILITY

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | Is 30× inference (MC Dropout) acceptable for latency? | MC Dropout paper | 🔴 HIGH |
| 2 | How often should adaptation occur (per-batch vs periodic)? | COA-HAR | 🔴 HIGH |
| 3 | What's GAN training time for ContrasGAN? | ContrasGAN | 🟠 MEDIUM |
| 4 | Can we do TTA on edge device (Garmin watch)? | AdaShadow | 🟠 MEDIUM |
| 5 | How to version adapted model weights? | General | 🟠 MEDIUM |
| 6 | What triggers centralized retraining vs local adaptation? | CODA | 🔴 HIGH |
| 7 | Is Wasserstein distance computationally feasible for 200×6? | LIFEWATCH | 🟠 MEDIUM |

## 3.8 TRIGGER POLICY

| # | Question | Source Paper | Priority |
|---|----------|--------------|----------|
| 1 | What combination of signals triggers retraining? | Multiple | 🔴 HIGH |
| 2 | Should trigger be threshold-based or pattern-based? | LIFEWATCH | 🔴 HIGH |
| 3 | How to escalate: local adapt → central retrain → human review? | General | 🔴 HIGH |
| 4 | What's minimum data for retraining to be beneficial? | General | 🟠 MEDIUM |
| 5 | How to prevent false positive retraining triggers? | General | 🟠 MEDIUM |
| 6 | Can uncertainty trend predict when retraining is needed? | XAI-BayesHAR | 🟠 MEDIUM |

---

# 4. QUESTIONS WE CAN ANSWER NOW

Based on the papers, the following questions have clear answers:

## 4.1 Monitoring Without Labels

| Question | Answer | Citation |
|----------|--------|----------|
| Can we detect domain shift without labels? | **YES** - Entropy, uncertainty, Wasserstein distance all work | (Tent, XAI-BayesHAR, LIFEWATCH) |
| Which uncertainty method is easiest to implement? | **MC Dropout** - Just enable dropout at inference, run 30 passes | (MC Dropout paper, Section 3.2) |
| Can Kalman filter provide uncertainty? | **YES** - Track covariance P, trace(P) indicates uncertainty | (XAI-BayesHAR, "Kalman Update Equations") |

## 4.2 CSD / Drift Detection

| Question | Answer | Citation |
|----------|--------|----------|
| What causes domain shift in HAR? | **Users, devices, placement, protocols, environment** | (DAGHAR Benchmark, Section 4.2) |
| Can KS-test detect per-channel drift? | **YES** - p-value < 0.05 indicates distribution change | (Our implementation validated) |
| Is PSI effective for HAR drift? | **YES** - PSI > 0.25 indicates significant shift | (Industry standard, used in monitoring) |

## 4.3 Pseudo-Label Retraining

| Question | Answer | Citation |
|----------|--------|----------|
| How to prevent confirmation bias? | **RESTART model parameters between cycles** | (Curriculum Labeling, Section "WHY RESTART WORKS") |
| Should we use hard confidence threshold? | **NO** - Use relative top-K% per class ranking | (Curriculum Labeling, "Key Innovation") |
| What augmentations work for IMU? | **Time-warp, scaling, rotation, noise, permutation** | (SelfHAR, COA-HAR) |

## 4.4 Test-Time Adaptation

| Question | Answer | Citation |
|----------|--------|----------|
| Can we adapt at inference without labels? | **YES** - Tent, COA-HAR, OFTTA all work | (Multiple papers) |
| What to update during TTA? | **BatchNorm parameters only** (γ, β) - ~1% of model | (Tent, Section "What Gets Updated") |
| What batch size for TTA? | **64-128 recommended** for stable adaptation | (Tent, "Specific Metrics") |

## 4.5 Sensor Placement

| Question | Answer | Citation |
|----------|--------|----------|
| How much accuracy drops with position displacement? | **15-35%** without compensation | (Sensor Displacement, "Accuracy Statistics") |
| Can augmentation help with position variance? | **YES** - Include position variations during training | (Sensor Displacement) |

## 4.6 Uncertainty Quantification

| Question | Answer | Citation |
|----------|--------|----------|
| MC Dropout vs Ensemble vs Evidential? | **Ensemble best for OOD (0.86-0.93 AUROC), MC Dropout easiest to add** | (MC Dropout paper, Table 4) |
| How many forward passes for MC Dropout? | **30 passes** recommended for HAR | (MC Dropout paper) |
| What's overhead for Kalman uncertainty? | **~1ms per sample** - minimal | (XAI-BayesHAR, "Computational overhead") |

---

# 5. QUESTIONS FOR MENTOR

## 5.1 High Priority - Need Mentor Input

| # | Question | Context | Why Important |
|---|----------|---------|---------------|
| 1 | **Should we implement AdaBN, Tent, or COA-HAR?** | Multiple TTA methods available, each with trade-offs | Core thesis implementation decision |
| 2 | **What's acceptable human labeling budget for thesis demo?** | CODA needs some labels, but how many? | Determines active learning scope |
| 3 | **Is 30× inference for MC Dropout acceptable?** | Or should we use single-pass Evidential? | Performance vs accuracy trade-off |
| 4 | **Should thesis focus on single adaptation method or comparison?** | Could compare Tent vs ContrasGAN vs CODA | Scope and contribution clarity |
| 5 | **What proxy metrics should we validate as accuracy substitutes?** | Confidence, entropy, uncertainty, temporal plausibility | Critical for production evaluation |
| 6 | **Is handedness compensation in scope for thesis?** | Affects 63% of production users | Major domain gap source |

## 5.2 Medium Priority - Clarification Needed

| # | Question | Context |
|---|----------|---------|
| 7 | How to present IMU data for human labeling? | Waveform plots? Video sync? |
| 8 | What retraining trigger thresholds should thesis use? | KS p-value? PSI? Entropy? Combination? |
| 9 | Should we implement LIFEWATCH pattern memory or simpler approach? | Complexity vs benefit |
| 10 | Is EWC (Elastic Weight Consolidation) needed for our continual learning? | Or is Gaussian replay sufficient? |

## 5.3 Low Priority - Future Work Candidates

| # | Question | Notes |
|---|----------|-------|
| 11 | Can we combine TTA with GAN-based UDA? | Complex, possibly future work |
| 12 | Should we build activity hierarchy for Hi-OSCAR? | Open-set classification optional |
| 13 | Is streaming inference needed or batch sufficient? | Demo vs production requirements |

---

# 6. OPEN RESEARCH GAPS

## 6.1 Gaps Identified from Papers

| Gap | Papers That Mention It | Suggested Search Keywords |
|-----|------------------------|---------------------------|
| **Zero-label bootstrapping for HAR** | Curriculum Labeling, SelfHAR | "zero-shot HAR", "unsupervised HAR initialization" |
| **Combining TTA + Continual Learning** | Tent, PACL+, CODA | "continual test-time adaptation", "lifelong TTA" |
| **Long-term TTA stability (months)** | All TTA papers | "long-term domain adaptation", "temporal degradation TTA" |
| **BiLSTM-compatible TTA** | Tent | "recurrent network TTA", "LayerNorm adaptation" |
| **Distinguishing covariate vs concept drift** | LIFEWATCH, UDA Benchmark | "drift type detection", "concept drift classification" |
| **Handedness-specific adaptation** | Sensor Displacement | "wrist handedness HAR", "bilateral sensor HAR" |
| **Automatic activity hierarchy construction** | Hi-OSCAR | "hierarchical activity recognition", "activity ontology" |
| **Proxy metric validation without labels** | General | "unsupervised model evaluation", "proxy accuracy metrics" |

## 6.2 Suggested Implementations for Thesis

| Priority | Implementation | Based On | Effort |
|----------|----------------|----------|--------|
| 🔴 P0 | AdaBN/Tent TTA | XHAR, Tent | 2-3 days |
| 🔴 P0 | MC Dropout uncertainty | MC Dropout paper | 1-2 days |
| 🔴 P0 | Entropy-based drift detection | Tent | 1 day |
| 🟠 P1 | Curriculum pseudo-labeling | Curriculum Labeling | 3-4 days |
| 🟠 P1 | Kalman uncertainty tracking | XAI-BayesHAR | 2-3 days |
| 🟠 P1 | CODA active learning | CODA | 3-4 days |
| 🟡 P2 | ContrasGAN UDA | ContrasGAN | 5-7 days |
| 🟡 P2 | LIFEWATCH pattern memory | LIFEWATCH | 3-4 days |

## 6.3 Gap-Filling Search Strategy

For open gaps, search using these keyword combinations:

```
Gap: Zero-label bootstrapping
→ "HAR" AND "unsupervised" AND ("bootstrap" OR "initialization" OR "zero-shot")
→ "wearable" AND "self-supervised" AND "no labels"

Gap: Long-term TTA
→ "test-time adaptation" AND ("long-term" OR "months" OR "temporal")
→ "continual adaptation" AND "HAR"

Gap: BiLSTM TTA
→ "recurrent" AND "test-time" AND "adaptation"
→ "LSTM" AND ("domain adaptation" OR "LayerNorm")

Gap: Handedness
→ "wrist" AND ("handedness" OR "bilateral" OR "dominant")
→ "HAR" AND ("left" OR "right") AND "wrist"

Gap: Proxy metrics
→ "unsupervised evaluation" AND "classification"
→ "proxy" AND "accuracy" AND "no labels"
```

---

# 7. SUMMARY

## 7.1 Key Takeaways

1. **Zero-label TTA is POSSIBLE** - Tent, COA-HAR, OFTTA all work without any target labels
2. **Uncertainty without labels** - MC Dropout, Kalman filter, entropy all computable at inference
3. **Restart prevents bias** - Curriculum Labeling's key insight for pseudo-labeling
4. **AdaBN is simplest** - Just update BatchNorm statistics, instant adaptation
5. **15-35% accuracy drop** from position/handedness mismatch is significant

## 7.2 Recommended Implementation Order

1. **Week 1:** Implement Tent (entropy minimization TTA)
2. **Week 1:** Add MC Dropout uncertainty
3. **Week 2:** Implement AdaBN as alternative TTA
4. **Week 2:** Add entropy-based drift detection to monitoring
5. **Week 3:** Implement curriculum pseudo-labeling
6. **Week 3:** Add Kalman uncertainty tracking
7. **Week 4:** Implement CODA active learning sampling

## 7.3 Document Status

| Section | Status |
|---------|--------|
| Paper Inventory | ✅ Complete |
| Question Extraction | ✅ Complete |
| Pipeline Stage Grouping | ✅ Complete |
| Answered Questions | ✅ Complete |
| Mentor Questions | ✅ Complete |
| Research Gaps | ✅ Complete |

---

**Document Generated:** January 30, 2026  
**Papers Analyzed:** 88 files (23 key papers in depth)  
**Questions Extracted:** 73 unique questions  
**Pipeline Stages Covered:** 8 stages

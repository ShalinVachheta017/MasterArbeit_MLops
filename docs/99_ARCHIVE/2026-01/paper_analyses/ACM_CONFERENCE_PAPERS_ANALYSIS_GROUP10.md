# ACM/Conference Papers Analysis - Group 10
## For MLOps HAR Thesis with Unlabeled Production Data

**Thesis Constraints Reminder:**
- Production data is **UNLABELED**
- No online evaluation with labels
- Deep model: 1D-CNN + BiLSTM
- Sensors: AX AY AZ GX GY GZ (6-axis IMU)

---

## Paper 1: 10325312.pdf

### Full Title & Year
**"Continual Learning for Activity Recognition"** (2020)
- Authors: Ramesh Kumar Sah, Seyed Iman Mirzadeh, Hassan Ghasemzadeh
- IEEE Conference Paper

### Problem Addressed
- **Catastrophic forgetting** in neural networks when learning sequentially from new subjects/data
- DNNs trained on wearable sensor data lose previous knowledge when fine-tuned on new data
- Challenge of deploying ML models that need to adapt to new users without forgetting old users
- Distribution shift across different subjects performing the same activities

### Labels Assumed?
⚠️ **YES - Labels are assumed for all tasks**
- Requires labeled data from each new subject (task)
- Training on sequential tasks requires labels for validation
- Multi-task/joint training explicitly uses labels from all subjects

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Why does fine-tuning on new data degrade performance on old data?**
   - Answer: Neural network parameters adapt to new data distribution, becoming unsuitable for old distributions
   
2. **What is the accuracy gap between fine-tuning vs joint training?**
   - Answer: Fine-tuning shows significant performance drop (demonstrated on PAMAP2 with 8 subjects)

3. **Can continual learning algorithms help?**
   - Answer: Yes, techniques like EWC (Elastic Weight Consolidation), LwF (Learning without Forgetting), and replay methods help but still gap exists

#### Questions RAISED but NOT Answered:
1. ❓ **How to do continual learning WITHOUT labels from new users in production?**
2. ❓ **Can self-supervised continual learning work for HAR?**
3. ❓ **How to detect when catastrophic forgetting is happening without labels?**
4. ❓ **What triggers the need for model update in production?**

### Specific Techniques Mentioned
- **Elastic Weight Consolidation (EWC)**: Penalizes changes to important weights
- **Learning without Forgetting (LwF)**: Uses knowledge distillation
- **Memory replay**: Stores examples from previous tasks
- **Gradient Episodic Memory (GEM)**
- Dataset: PAMAP2

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Training | Continual learning strategies needed |
| Model Update | How to update without forgetting |
| Monitoring | Detecting distribution shift |
| Deployment | Sequential user adaptation |

### Relevance to Unlabeled Production HAR
**MEDIUM-HIGH** - Directly addresses the problem of model degradation over time with new users, but assumes labels are available. Need to combine with unsupervised methods.

---

## Paper 2: 3267305.3274148.pdf

### Full Title & Year
**"CLAW 2018 – Chairs' Message: Fourth Workshop on Legal and Technical Issues in Cloud and Pervasive Computing (IoT)"** (2018)
- Authors: Jatinder Singh, Julia Powles, Angela Daly
- UbiComp/ISWC 2018 Workshop Proceedings

### Problem Addressed
- Workshop introduction/editorial - NOT a research paper
- Focuses on legal/technical issues in cloud computing and IoT
- GDPR compliance and data privacy

### Labels Assumed?
N/A - Not a research paper

### KEY QUESTIONS for Our Scenario
**NOT RELEVANT** - This is a workshop chairs' message, not a technical paper on HAR or MLOps.

### Relevance to Unlabeled Production HAR
**NONE** - Skip this paper for thesis.

---

## Paper 3: 3380985.pdf

### Full Title & Year
**"A Systematic Study of Unsupervised Domain Adaptation for Robust Human-Activity Recognition"** (2020)
- Authors: Youngjae Chang, Akhil Mathur, Anton Isopoussu, Junehwa Song, Fahim Kawsar
- IMWUT/UbiComp 2020, 115 citations

### Problem Addressed
- **Wearing diversity problem**: Different sensor placements (wrist, pocket, ear, etc.)
- Domain shift between training and test data due to body position changes
- How to adapt HAR models to new wearing positions using only **UNLABELED** data
- Evaluates UDA (Unsupervised Domain Adaptation) algorithms for HAR

### Labels Assumed?
🎯 **NO labels required for target domain!**
- Source domain: Labeled
- Target domain: **UNLABELED** (our exact scenario!)
- Pre-trained classifier adapted using only unlabeled target data

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Do deep learning models suffer from wearing diversity?**
   - Answer: YES, even state-of-the-art deep models degrade significantly

2. **Can UDA work without ANY target labels?**
   - Answer: YES, but with caveats and implicit assumptions

3. **Which UDA methods work for HAR?**
   - Answer: Three techniques evaluated - performance varies by scenario

4. **What are the failure modes of UDA?**
   - Answer: Paper reveals hidden assumptions that cause UDA to fail

#### Questions RAISED but NOT Answered:
1. ❓ **How to select which UDA algorithm to use without labels?**
2. ❓ **Can UDA handle continuous distribution shift (not just A→B)?**
3. ❓ **How to combine UDA with continual learning?**
4. ❓ **What if source and target activities are different?**

### Specific Techniques Mentioned
- **Maximum Mean Discrepancy (MMD)**
- **Domain Adversarial Neural Networks (DANN)**
- **Adversarial Discriminative Domain Adaptation (ADDA)**
- Datasets: OPPORTUNITY, PAMAP2, USC-HAD, HHAR

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Training | Pre-train on labeled source domain |
| Preprocessing | Feature alignment needed |
| Inference | Adaptation at deployment |
| Monitoring | Detect domain shift |

### Relevance to Unlabeled Production HAR
**VERY HIGH** ⭐⭐⭐⭐⭐ - Directly addresses our scenario of adapting to unlabeled production data! Paper warns about hidden assumptions - critical reading.

---

## Paper 4: 3394486.3403228.pdf

### Full Title & Year
**"Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data"** (2020)
- Authors: Garrett Wilson, Janardhan Rao Doppa, Diane J. Cook
- KDD 2020, 121 citations

### Problem Addressed
- Domain adaptation for **time-series sensor data** (specifically HAR)
- Using **multiple source domains** to improve adaptation
- Novel problem: **Weak supervision** using only target label distributions (not per-sample labels)
- Proposes CoDATS: Convolutional deep Domain Adaptation model for Time Series

### Labels Assumed?
🎯 **Mixed - Innovative weak supervision approach!**
- Source domains: Fully labeled
- Target domain: **UNLABELED** for standard UDA
- Novel: Can use **weak supervision** (just label proportions like "30% walking, 10% running")

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Can CNNs outperform RNNs for domain adaptation in time series?**
   - Answer: YES - CoDATS is faster and more accurate than RNN-based methods

2. **Does using multiple source domains help?**
   - Answer: YES, especially for complex datasets with high variability

3. **Can approximate label distributions (weak supervision) help?**
   - Answer: YES - even rough estimates of activity proportions improve adaptation

4. **What architecture works for time-series UDA?**
   - Answer: CoDATS with domain-invariant features

#### Questions RAISED but NOT Answered:
1. ❓ **How to estimate target label distribution without any labels?**
2. ❓ **What if activities in target differ from source?**
3. ❓ **How to handle real-time streaming data?**
4. ❓ **Can this work with BiLSTM architectures?**

### Specific Techniques Mentioned
- **CoDATS architecture** (1D CNN based)
- **Domain-invariant feature learning**
- **Gradient reversal layer**
- **Multi-source domain adaptation**
- **Weak supervision via label proportions**
- Datasets: WISDM, UCI HAR, HHAR, Sleep Stage Classification

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Data Collection | Multiple source datasets beneficial |
| Training | Multi-source adaptation |
| Deployment | Weak supervision signals |
| Monitoring | Label distribution estimation |

### Relevance to Unlabeled Production HAR
**VERY HIGH** ⭐⭐⭐⭐⭐ - The weak supervision idea is brilliant! In production, we might estimate "this user is mostly sedentary" without per-sample labels. CoDATS architecture is close to our 1D-CNN.

---

## Paper 5: 3448112.pdf

### Full Title & Year
**"SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data"** (2021)
- Authors: Chi Ian Tang, Ignacio Perez-Pozuelo, Dimitris Spathis, Soren Brage, Nick Wareham, Cecilia Mascolo
- IMWUT/UbiComp 2021

### Problem Addressed
- Leveraging **massive amounts of unlabeled wearable sensor data**
- Semi-supervised learning for HAR
- Combining **teacher-student self-training** with **multi-task self-supervision**
- Data efficiency - achieving good performance with less labeled data

### Labels Assumed?
🎯 **Semi-supervised: Small labeled + Large unlabeled**
- Requires **small labeled dataset** for initial training
- Leverages **large unlabeled dataset** for improvement
- Claims 10x reduction in labeled data needed

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Can unlabeled HAR data improve models?**
   - Answer: YES - up to 12% F1 improvement using unlabeled data

2. **How to leverage unlabeled data in HAR?**
   - Answer: Teacher-student self-training + self-supervision tasks

3. **Which self-supervision tasks work for sensor data?**
   - Answer: Predicting distorted versions, transformation prediction

4. **How much labeled data is really needed?**
   - Answer: With SelfHAR, 10x less labeled data can achieve same performance

#### Questions RAISED but NOT Answered:
1. ❓ **What if we have ZERO labeled production data?**
2. ❓ **How to handle distribution shift with self-training?**
3. ❓ **When does self-training fail or give bad pseudo-labels?**
4. ❓ **How to continuously update with streaming data?**

### Specific Techniques Mentioned
- **Teacher-student self-training**
- **Pseudo-labeling**
- **Multi-task self-supervision**
- **Data augmentation for sensor data**
- **Knowledge distillation**
- Datasets: Multiple HAR benchmarks

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Pre-training | Self-supervised pre-training on unlabeled |
| Training | Teacher-student framework |
| Data | Leverage unlabeled production data |
| Augmentation | Signal distortion techniques |

### Relevance to Unlabeled Production HAR
**HIGH** ⭐⭐⭐⭐ - Directly relevant! Shows how to use unlabeled data. But still needs SOME labeled data initially. Could combine with UDA for zero-label production.

---

## Paper 6: 3530910.pdf

### Full Title & Year
**"Resource-Efficient Continual Learning for Sensor-Based Human Activity Recognition"** (2022)
- Authors: Clayton Frederick Souza Leite, Yu Xiao
- ACM Transactions on Embedded Computing Systems

### Problem Addressed
- Continual learning for HAR on **resource-constrained devices**
- **Class-incremental scenario**: New activity classes need to be recognized
- **Style-incremental scenario**: Variations in how activities are performed
- Memory and computation efficiency for on-device training

### Labels Assumed?
⚠️ **YES - Labels required for new tasks**
- New classes/styles require labeled examples
- Focus is on resource efficiency, not label efficiency

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Can continual learning run on embedded devices?**
   - Answer: YES with compressed replay memory and efficient training

2. **How to handle both new classes AND new styles?**
   - Answer: Expandable neural network with selective rehearsal

3. **How much compression can rehearsal data tolerate?**
   - Answer: Highly compressed (downsampled, precision reduced) data still works

4. **How to select what to remember?**
   - Answer: Maximize data variability in replay buffer

#### Questions RAISED but NOT Answered:
1. ❓ **How to do on-device continual learning WITHOUT labels?**
2. ❓ **How to detect when new classes appear without labels?**
3. ❓ **Can self-supervised signals replace labeled rehearsal?**
4. ❓ **How to handle style drift in unlabeled setting?**

### Specific Techniques Mentioned
- **Gradient Episodic Memory (GEM)**
- **Replay-based continual learning**
- **Memory compression** (downsampling, precision reduction)
- **Expandable neural networks**
- Datasets: OPPORTUNITY, PAMAP2, DSADS, Skoda

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Model Architecture | Expandable networks |
| Memory Management | Compressed replay buffer |
| On-device Training | Resource constraints |
| Deployment | Edge device considerations |

### Relevance to Unlabeled Production HAR
**MEDIUM** ⭐⭐⭐ - Resource efficiency is relevant for MLOps, but assumes labels. Need to combine with unsupervised techniques.

---

## Paper 7: 3550299.pdf

### Full Title & Year
**"Assessing the State of Self-Supervised Human Activity Recognition Using Wearables"** (2022)
- Authors: Harish Haresamudram, Irfan Essa, Thomas Plötz
- IMWUT/UbiComp 2022, 86 citations

### Problem Addressed
- Comprehensive evaluation of **self-supervised learning (SSL)** methods for HAR
- "Pretrain-then-finetune" paradigm for wearable sensor data
- Leveraging **unlabeled movement data** for representation learning
- Multi-faceted assessment (~50k experiments)

### Labels Assumed?
🎯 **Pre-training: UNLABELED, Fine-tuning: Small labeled set**
- Self-supervision uses NO labels for feature learning
- Only small labeled set needed for final classifier

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Which self-supervised methods work best for HAR?**
   - Answer: Evaluated Multi-task SSL, Masked Reconstruction, CPC, SimCLR - each has strengths

2. **Do SSL methods generalize across datasets?**
   - Answer: Varied - some robust to source/target differences, some not

3. **What dataset characteristics affect SSL?**
   - Answer: Size, activity types, sensor positions all matter

4. **How good are learned features?**
   - Answer: Can match supervised with enough unlabeled data

#### Questions RAISED but NOT Answered:
1. ❓ **Which SSL to choose for specific deployment scenarios?**
2. ❓ **Can SSL features detect distribution shift?**
3. ❓ **How to do online SSL for streaming data?**
4. ❓ **Optimal amount of unlabeled pre-training data?**

### Specific Techniques Mentioned
- **Multi-task self-supervision**
- **Masked reconstruction (autoencoders)**
- **Contrastive Predictive Coding (CPC)**
- **SimCLR contrastive learning**
- **MLP classifiers on frozen features**
- Datasets: Mobiact, PAMAP2, HHAR, many others

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Pre-training | SSL on unlabeled data |
| Feature Learning | Representation quality |
| Transfer | Source-target robustness |
| Evaluation | Multi-faceted assessment framework |

### Relevance to Unlabeled Production HAR
**VERY HIGH** ⭐⭐⭐⭐⭐ - Excellent guide for which SSL method to use! Critical for leveraging unlabeled production data. Provides evaluation framework.

---

## Paper 8: 3631450.pdf

### Full Title & Year
**"Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition"** (OFTTA) (2023)
- Authors: Shuoyuan Wang, Jindong Wang, Huajun Xi, Bob Zhang, Lei Zhang, Hongxin Wei
- IMWUT 2023

### Problem Addressed
- **Test-Time Adaptation (TTA)** for HAR - adapt during inference!
- Cross-person activity recognition with distribution shift
- **Optimization-free** approach (no backpropagation at test time)
- Resource-efficient for edge devices

### Labels Assumed?
🎯 **NO labels at test time!**
- Pre-trained on labeled source data
- Adapts at test time using ONLY unlabeled test samples
- No ground truth needed for adaptation

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Can we adapt at test time without labels?**
   - Answer: YES - OFTTA does this efficiently

2. **How to adapt without backpropagation?**
   - Answer: Modify batch normalization (EDTN) + prototype-based classifier

3. **Is TTA computationally feasible for edge devices?**
   - Answer: YES - OFTTA is optimization-free, fast

4. **How to handle pseudo-label noise?**
   - Answer: Support set maintenance with reliable features

#### Questions RAISED but NOT Answered:
1. ❓ **How to handle continuous domain shift over long periods?**
2. ❓ **What if activity classes change (not just distributions)?**
3. ❓ **How to combine TTA with continual learning?**
4. ❓ **Optimal batch size for TTA in streaming scenarios?**

### Specific Techniques Mentioned
- **Exponential Decay Test-time Normalization (EDTN)**
- **Test-time Batch Normalization (TBN)**
- **Prototype-based classification**
- **Support set maintenance**
- **Pseudo-labeling**
- Datasets: UniMiB, PAMAP2, USC-HAD
- Code: https://github.com/Claydon-Wang/OFTTA

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Inference | Real-time adaptation |
| Normalization | Modified BN layers |
| Classification | Prototype-based |
| Deployment | Edge device support |

### Relevance to Unlabeled Production HAR
**EXTREMELY HIGH** ⭐⭐⭐⭐⭐ - This is EXACTLY what we need! Adapts at test time without labels. Code available. Must implement for thesis.

---

## Paper 9: 3659591.pdf

### Full Title & Year
**"M3BAT: Unsupervised Domain Adaptation for Multimodal Mobile Sensing with Multi-Branch Adversarial Training"** (2024)
- Authors: Lakmal Meegahapola, Hamza Hassoune, Daniel Gatica-Perez
- IMWUT 2024

### Problem Addressed
- **Distribution shift in multimodal mobile sensing**
- Unsupervised domain adaptation for multiple sensor modalities
- Handling different users, populations, environments
- Novel multi-branch architecture for multimodal UDA

### Labels Assumed?
🎯 **Source: Labeled, Target: UNLABELED**
- Standard UDA setting
- No labels needed for target domain deployment

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Does UDA work for multimodal sensor data?**
   - Answer: YES, DANN and M3BAT effective

2. **How to handle multiple sensor modalities in UDA?**
   - Answer: Multi-branch architecture for each modality

3. **What performance gains are achievable?**
   - Answer: Up to 12% AUC improvement, 0.13 MAE reduction

4. **Does it work for both classification and regression?**
   - Answer: YES, tested on both

#### Questions RAISED but NOT Answered:
1. ❓ **How to handle missing modalities in production?**
2. ❓ **Optimal branch architecture for IMU data?**
3. ❓ **How to select source domains for best transfer?**
4. ❓ **Computational cost for real-time deployment?**

### Specific Techniques Mentioned
- **Domain Adversarial Neural Networks (DANN)**
- **M3BAT: Multi-Branch Adversarial Training**
- **Feature alignment across modalities**
- Tasks: Activity recognition, mood inference, energy expenditure
- Datasets: StudentLife, others

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Feature Extraction | Multi-branch architecture |
| Domain Adaptation | Adversarial training |
| Multimodal Fusion | Branch-wise adaptation |
| Deployment | Cross-population transfer |

### Relevance to Unlabeled Production HAR
**HIGH** ⭐⭐⭐⭐ - Relevant for multimodal IMU (Acc + Gyro). Shows UDA works for sensor data. Architecture ideas applicable.

---

## Paper 10: 3659597.pdf

### Full Title & Year
**"CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"** (2024)
- Authors: Zhiqing Hong, Zelong Li, Shuxin Zhong, Wenjun Lyu, Haotian Wang, Yi Ding, Tian He, Desheng Zhang
- IMWUT 2024, 42 citations

### Problem Addressed
- **Cross-dataset HAR** - harder than cross-domain!
- Multiple simultaneous domain shifts (users, devices, placements, protocols)
- Self-supervised pretraining for generalization
- Improve performance on **unseen target datasets**

### Labels Assumed?
🎯 **Pre-training: UNLABELED, Fine-tuning: Small labeled source set**
- Hierarchical SSL uses no labels
- Only needs labeled data from source dataset
- Target dataset can be completely unlabeled

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **How severe is cross-dataset performance drop?**
   - Answer: 42.39% average accuracy drop! Much worse than cross-domain (6.77%)

2. **Can SSL help cross-dataset generalization?**
   - Answer: YES - CrossHAR outperforms SOTA by 10.83%

3. **What data augmentation works for sensors?**
   - Answer: Physics-based augmentation from sensor data generation principles

4. **Why hierarchical pretraining?**
   - Answer: Captures both local patterns and global activity structure

#### Questions RAISED but NOT Answered:
1. ❓ **How to handle completely new activity classes in target?**
2. ❓ **Minimum source dataset size for good transfer?**
3. ❓ **How to combine with online adaptation?**
4. ❓ **Does it work with different architectures (BiLSTM)?**

### Specific Techniques Mentioned
- **Hierarchical self-supervised pretraining**
- **Physics-based data augmentation**
- **Contrastive learning**
- **Transfer to unseen datasets**
- Datasets: UCI, HHAR, and others

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Pre-training | Hierarchical SSL |
| Augmentation | Physics-based |
| Transfer | Cross-dataset |
| Generalization | Unseen data handling |

### Relevance to Unlabeled Production HAR
**VERY HIGH** ⭐⭐⭐⭐⭐ - Addresses our exact problem of deploying to new unlabeled environments. Physics-based augmentation is valuable. 10.83% improvement is significant.

---

## Paper 11: 3666025.3699339.pdf

### Full Title & Year
**"AdaShadow: Responsive Test-time Model Adaptation in Non-stationary Mobile Environments"** (2024)
- Authors: Cheng Fang, Sicong Liu, Zimu Zhou, Bin Guo, Jiaqi Tang, Ke Ma, Zhiwen Yu
- SenSys 2024

### Problem Addressed
- **Latency of Test-Time Adaptation (TTA)** for mobile applications
- TTA's forward-backward-reforward pipeline is too slow for real-time
- Selective layer updates for speed
- Non-stationary domain shifts (continuous changes)

### Labels Assumed?
🎯 **NO labels at test time!**
- Unsupervised TTA
- Adapts using only unlabeled test data

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Is TTA too slow for real-time applications?**
   - Answer: YES - standard TTA adds significant latency

2. **How to make TTA faster?**
   - Answer: Selective layer updates (AdaShadow achieves 2-3.5x speedup)

3. **Which layers are most important to adapt?**
   - Answer: Backpropagation-free assessor identifies critical layers

4. **Can efficient TTA match full TTA accuracy?**
   - Answer: YES - comparable accuracy with ms-level latency

#### Questions RAISED but NOT Answered:
1. ❓ **How to handle gradual vs sudden domain shifts?**
2. ❓ **Optimal layer selection strategy for HAR specifically?**
3. ❓ **Memory footprint for continuous adaptation?**
4. ❓ **How long can adaptation continue before degradation?**

### Specific Techniques Mentioned
- **Selective layer adaptation**
- **Backpropagation-free layer importance estimation**
- **Unit-based runtime prediction**
- **Online scheduling**
- **Memory I/O-aware computation reuse**
- Applications: AR, autonomous driving

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Inference | Real-time adaptation |
| Layer Selection | Critical layer identification |
| Latency | 2-3.5x speedup |
| Edge Deployment | Mobile device support |

### Relevance to Unlabeled Production HAR
**HIGH** ⭐⭐⭐⭐ - Important for MLOps! Shows how to make TTA practical for real-time deployment. Latency considerations critical for production.

---

## Paper 12: 3770681.pdf

### Full Title & Year
**"Hi-OSCAR: Hierarchical Open-set Classifier for Human Activity Recognition"** (2025)
- Authors: Conor McCarthy, Loes Quirijnen, Jan Peter van Zandwijk, Zeno Geradts, Marcel Worring
- IMWUT 2025

### Problem Addressed
- **Open-set classification** for HAR - handling UNKNOWN activities
- Gap between training activities and real-world activity diversity
- **Hierarchical activity structure** (some activities overlap/encompass others)
- New dataset: NFI_FARED with 19 activities

### Labels Assumed?
⚠️ **Training requires labels, but handles unknown classes at test time**
- Labeled training set with hierarchical structure
- Test time: Can identify samples as Out-of-Distribution (OOD)

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **How to handle unknown activities in production?**
   - Answer: Hierarchical open-set classifier can reject unknowns

2. **Can we localize unknown activities?**
   - Answer: YES - identifies nearest known activity group

3. **How to structure activity hierarchies?**
   - Answer: Group similar activities (walking/walking upstairs closer than walking/punching)

4. **What happens with OOD detection in HAR?**
   - Answer: Binary known/unknown plus hierarchical localization

#### Questions RAISED but NOT Answered:
1. ❓ **How to automatically construct activity hierarchy?**
2. ❓ **Can unknown activities be learned incrementally?**
3. ❓ **How to handle OOD without labels for validation?**
4. ❓ **Threshold selection for OOD detection without labels?**

### Specific Techniques Mentioned
- **Hierarchical classification**
- **Out-of-Distribution (OOD) detection**
- **Open-set recognition**
- **Activity hierarchy construction**
- New Dataset: **NFI_FARED** (19 activities, public)
- URL: https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Classification | Hierarchical structure |
| OOD Detection | Unknown activity handling |
| Uncertainty | Know when model doesn't know |
| Safety | Avoid misclassifying unknowns |

### Relevance to Unlabeled Production HAR
**HIGH** ⭐⭐⭐⭐ - Critical for production! Must handle unknown activities. Hierarchical localization helps understand "what went wrong" even without labels.

---

## Paper 13: 41597_2024_Article_3951.pdf

### Full Title & Year
**"DAGHAR: A Benchmark for Domain Adaptation and Generalization in Smartphone-based Human Activity Recognition"** (2024)
- Authors: Otávio Napoli, Dami Duarte, Patrick Alves, et al.
- Scientific Data (Nature)

### Problem Addressed
- **Lack of standardized benchmark** for domain adaptation in HAR
- Cross-dataset evaluation is hard due to incompatible formats
- Standardizes 6 HAR datasets for fair comparison
- Domain adaptation and generalization research enablement

### Labels Assumed?
⚠️ **All datasets have labels for evaluation purposes**
- Benchmark provides standardized labeled data
- For evaluating DA/DG methods

### KEY QUESTIONS Raised for Our Scenario

#### Questions ANSWERED by Paper:
1. **Why do models fail across datasets?**
   - Answer: Incompatible formats, units, sampling rates, label encodings

2. **What factors cause distribution shift in HAR?**
   - Answer: Sensor hardware, software, placement, demographics, terrain, protocols

3. **Can standardization help cross-dataset research?**
   - Answer: YES - enables controlled evaluation

4. **Which baseline methods work?**
   - Answer: Provides ML and DL baseline metrics

#### Questions RAISED but NOT Answered:
1. ❓ **How to standardize unlabeled production data?**
2. ❓ **Optimal preprocessing for domain-invariant features?**
3. ❓ **Which domain differences matter most?**
4. ❓ **How to use benchmark for unlabeled scenario?**

### Specific Techniques Mentioned
- **Dataset standardization**: Units, sampling rate, gravity component
- **User partitioning**
- **t-SNE visualization** of domain differences
- **State-of-the-art baselines** (ML and DL)
- Datasets: RealWorld, ExtraSensory, and 4 others (6 total)

### Pipeline Stages Affected
| Stage | Impact |
|-------|--------|
| Data Preprocessing | Standardization protocols |
| Evaluation | Fair comparison framework |
| Benchmarking | Baseline metrics |
| Research | Controlled experiments |

### Relevance to Unlabeled Production HAR
**MEDIUM-HIGH** ⭐⭐⭐⭐ - Excellent for understanding domain shift sources. Use their standardization approach for preprocessing. Benchmark useful for evaluating DA methods.

---

# Summary Table: All 13 Papers

| # | Paper | Year | Labels Required? | Key Contribution | Relevance |
|---|-------|------|------------------|------------------|-----------|
| 1 | Continual Learning for AR | 2020 | Yes | Catastrophic forgetting in HAR | ⭐⭐⭐ |
| 2 | CLAW 2018 Workshop | 2018 | N/A | Workshop intro (skip) | ❌ |
| 3 | UDA for Robust HAR | 2020 | Target: No | Systematic UDA study | ⭐⭐⭐⭐⭐ |
| 4 | CoDATS (Multi-Source UDA) | 2020 | Target: No (weak OK) | Weak supervision idea | ⭐⭐⭐⭐⭐ |
| 5 | SelfHAR | 2021 | Small labeled | Self-training with unlabeled | ⭐⭐⭐⭐ |
| 6 | Resource-Efficient CL | 2022 | Yes | On-device continual learning | ⭐⭐⭐ |
| 7 | SSL for HAR Assessment | 2022 | Pre-train: No | SSL method comparison | ⭐⭐⭐⭐⭐ |
| 8 | OFTTA (Test-Time Adapt) | 2023 | Test: No | Optimization-free TTA | ⭐⭐⭐⭐⭐ |
| 9 | M3BAT (Multimodal UDA) | 2024 | Target: No | Multi-branch for modalities | ⭐⭐⭐⭐ |
| 10 | CrossHAR | 2024 | Pre-train: No | Cross-dataset generalization | ⭐⭐⭐⭐⭐ |
| 11 | AdaShadow | 2024 | Test: No | Fast TTA for mobile | ⭐⭐⭐⭐ |
| 12 | Hi-OSCAR | 2025 | Train: Yes | Open-set/unknown activities | ⭐⭐⭐⭐ |
| 13 | DAGHAR Benchmark | 2024 | Yes (benchmark) | Standardized HAR datasets | ⭐⭐⭐⭐ |

---

# Critical Questions for Thesis

## Answered by These Papers:
1. ✅ Can we adapt HAR models to unlabeled production data? **YES** (UDA, TTA methods)
2. ✅ Which self-supervised methods work? **CPC, SimCLR, Masked Reconstruction** (Paper 7)
3. ✅ Is test-time adaptation feasible? **YES, even optimization-free** (Papers 8, 11)
4. ✅ What causes domain shift in HAR? **Users, devices, placement, protocols** (Papers 3, 13)
5. ✅ Can weak supervision help? **YES, even label proportions help** (Paper 4)

## Still Unanswered (Research Gaps):
1. ❓ How to combine TTA + Continual Learning + OOD detection in one system?
2. ❓ How to select adaptation method without any labels for validation?
3. ❓ How to handle completely new activity classes in unlabeled production?
4. ❓ Optimal architecture (1D-CNN + BiLSTM) for TTA/UDA?
5. ❓ How to detect when adaptation is failing without labels?
6. ❓ Long-term continuous adaptation (weeks/months) without degradation?

---

# Recommended Pipeline for Thesis

Based on these papers, suggested approach:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Pre-training    │     │ Deployment      │     │ Production      │
│ (Source Domain) │ ──► │ (Initial)       │ ──► │ (Continuous)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
│                       │                       │
├─ SSL Pre-train        ├─ OFTTA for TTA        ├─ Monitor shift
│  (CrossHAR style)     │  (Paper 8)            │  (entropy, etc.)
│                       │                       │
├─ Multi-source DA      ├─ Open-set detect      ├─ Self-training
│  (CoDATS style)       │  (Hi-OSCAR)           │  (SelfHAR)
│                       │                       │
└─ Labeled training     └─ No labels needed     └─ No labels needed
```

---

*Analysis completed: January 30, 2026*
*For MLOps HAR Thesis with Unlabeled Production Constraints*

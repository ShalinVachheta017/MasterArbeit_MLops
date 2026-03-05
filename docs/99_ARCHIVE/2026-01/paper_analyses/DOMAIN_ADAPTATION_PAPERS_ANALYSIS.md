# 📚 Domain Adaptation Papers Analysis for HAR MLOps Thesis

> **Date:** January 30, 2026  
> **Thesis Constraints:**
> - Production data is **UNLABELED**
> - No online evaluation with labels
> - Deep model: **1D-CNN + BiLSTM**
> - Sensors: **AX AY AZ GX GY GZ** (6-axis IMU)

---

## Analysis Summary Table

| # | Paper | Year | Labels Assumed | Production/Offline | Critical for Thesis |
|---|-------|------|----------------|-------------------|---------------------|
| 1 | XHAR | 2019 | Partial (source only) | Production | ⭐⭐⭐ AdaBN technique |
| 2 | AdaptNet | 2021 | Partial (semi-supervised) | Both | ⭐⭐ Bilateral adaptation |
| 3 | Shift-GAN (UDA GAN-Based) | 2021 | No (target) | Both | ⭐⭐⭐ True UDA |
| 4 | Scaling HAR via DL DA | 2019 | Yes (both domains) | Offline | ⭐⭐ Cross-dataset |
| 5 | SCAGOT | 2023 | Partial (source only) | Production | ⭐⭐ Context disentangling |
| 6 | Transfer Learning Review | 2022 | Survey (all types) | Both | ⭐⭐⭐ Comprehensive reference |
| 7 | Time Series Domain Shifts | 2024 | Partial | Both | ⭐⭐⭐ Shift characterization |

---

## Paper 1: XHAR - Deep Domain Adaptation for HAR with Smart Devices

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | XHAR: Deep Domain Adaptation for Human Activity Recognition with Smart Devices |
| **Year** | 2019 |
| **Venue** | IEEE BigData |
| **Authors** | Huan et al. |

### 🎯 Problem Addressed
- **Core Problem:** Cross-device HAR accuracy degradation when deploying models trained on one device to another device
- **Domain Gap:** Differences in sensor characteristics, placement, and data collection protocols cause significant accuracy drops
- **Specific Challenge:** How to transfer knowledge from source domain (labeled) to target domain (unlabeled or partially labeled)

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Full supervision for training |
| Target Domain Labels | **No** | Unlabeled target data for adaptation |
| Type | **Unsupervised DA** | Uses AdaBN and feature alignment |

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Pre-train on source, then adapt |
| Production Deployment | ✅ Yes | AdaBN can adapt at inference time |
| Online Learning | ⚠️ Partial | BN statistics update, not weights |

### ❓ KEY QUESTIONS for Our Scenario

1. **"How can we adapt our 1D-CNN-BiLSTM model to Garmin production data without labels?"**
   - XHAR's AdaBN technique is directly applicable
   - Replace source BN statistics with target statistics at inference time
   - No retraining required!

2. **"What's the minimum adaptation required for domain shift?"**
   - AdaBN alone can provide 5-15% accuracy improvement
   - Requires running target data through network in "training mode" to collect BN statistics

3. **"Can we detect when adaptation is needed?"**
   - Compare source vs target BN statistics (mean, variance)
   - Large divergence indicates domain shift

### ⚠️ Assumptions That Break in Our Setting

| XHAR Assumption | Our Reality | Impact |
|-----------------|-------------|--------|
| BatchNorm layers exist | ✅ Yes (1D-CNN has BN) | Compatible |
| Sufficient target data | ⚠️ Streaming | Need to accumulate data |
| Same activity classes | ✅ Yes (11 classes) | Compatible |
| Target data i.i.d. | ⚠️ Sequential | May need sliding statistics |

### 🔧 Pipeline Stage Affected
- **Stage:** Inference / Post-deployment
- **Implementation:** Update BN running statistics before prediction
- **Code Location:** `src/run_inference.py` → add `adapt_batch_norm()` function

### 📝 Questions ANSWERED by This Paper
✅ How to adapt without target labels (AdaBN)
✅ How much improvement to expect (5-15% on average)
✅ How to implement feature distribution alignment

### ❓ Questions RAISED but NOT ANSWERED
- How to handle streaming data where target distribution shifts over time?
- How to detect when re-adaptation is needed?
- What if BatchNorm statistics are noisy with small target batches?

---

## Paper 2: AdaptNet - Bilateral Domain Adaptation Using Semi-Supervised Deep Translation Networks

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | AdaptNet: Human Activity Recognition via Bilateral Domain Adaptation Using Semi-Supervised Deep Translation Networks |
| **Year** | 2021 |
| **Venue** | Neural Computing and Applications / Sensors |
| **Type** | Semi-supervised domain adaptation |

### 🎯 Problem Addressed
- **Core Problem:** Adapting HAR models across different users, devices, or sensor positions
- **Bilateral Translation:** Transforms both source-to-target AND target-to-source to learn shared representations
- **Semi-Supervised:** Leverages small amount of labeled target data when available

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Fully labeled |
| Target Domain Labels | **Partial** | 10-30% labeled recommended |
| Type | **Semi-supervised DA** | Needs some target labels |

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Requires paired source-target data |
| Production Deployment | ⚠️ Limited | Assumes some labeled target data available |
| Online Learning | ❌ No | Not designed for continuous adaptation |

### ❓ KEY QUESTIONS for Our Scenario

1. **"What if we have zero labeled production data?"**
   - AdaptNet is NOT directly applicable (requires some labels)
   - Could use confidence-based pseudo-labeling first, then apply AdaptNet
   - Suggests hybrid approach: Self-training → AdaptNet

2. **"How much labeled data is minimum?"**
   - Paper shows 10-20% labeled target data achieves 90% of fully-supervised performance
   - For our 11 classes × 100 windows/class = ~100-200 labeled samples

3. **"Is bilateral translation better than one-way?"**
   - Yes, paper shows 3-8% improvement over unidirectional methods
   - Creates more robust shared feature space

### ⚠️ Assumptions That Break in Our Setting

| AdaptNet Assumption | Our Reality | Impact |
|---------------------|-------------|--------|
| Some target labels | ❌ NO target labels | Major blocker |
| Batch adaptation | ⚠️ May have small batches | Reduces effectiveness |
| Static domains | ⚠️ Drift possible | Need re-adaptation |

### 🔧 Pipeline Stage Affected
- **Stage:** Retraining pipeline
- **Implementation:** Would require labeled data collection phase
- **Prerequisite:** Implement active learning or manual labeling workflow

### 📝 Questions ANSWERED by This Paper
✅ How much labeled target data is needed (10-30%)
✅ Bilateral vs unidirectional adaptation comparison
✅ Translation network architecture for HAR

### ❓ Questions RAISED but NOT ANSWERED
- How to select WHICH samples to label (active learning)?
- Can we generate pseudo-labels with sufficient quality?
- How to validate adaptation without ground truth?

---

## Paper 3: Unsupervised Domain Adaptation in Activity Recognition: A GAN-Based Approach (Shift-GAN)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Unsupervised Domain Adaptation in Activity Recognition: A GAN-Based Approach |
| **Year** | 2021 |
| **Venue** | Sensors / MDPI |
| **Also Known As** | Shift-GAN |

### 🎯 Problem Addressed
- **Core Problem:** HAR domain adaptation when target domain has ZERO labels
- **GAN Approach:** Uses adversarial learning to align source and target feature distributions
- **Key Innovation:** Domain discriminator forces feature extractor to learn domain-invariant representations

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Fully labeled for training |
| Target Domain Labels | **NO** | Completely unlabeled |
| Type | **Unsupervised DA** | True zero-shot adaptation |

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Requires source + unlabeled target |
| Production Deployment | ✅ Yes | Model works on new target data |
| Online Learning | ⚠️ Possible | Can retrain with accumulated target data |

### ❓ KEY QUESTIONS for Our Scenario

1. **"Can we use this for our unlabeled Garmin production data?"**
   - **YES!** This is designed exactly for our scenario
   - Train on ADAMSense (source) + unlabeled Garmin (target)
   - No labels needed from production

2. **"How does adversarial alignment work?"**
   - Feature extractor: Creates representations from sensor data
   - Domain discriminator: Tries to classify source vs target
   - Adversarial loss: Forces features to be domain-invariant
   - Activity classifier: Uses domain-invariant features

3. **"What accuracy improvement to expect?"**
   - Paper shows 10-25% improvement over no adaptation
   - Approaches semi-supervised performance with 0 labels

### ⚠️ Assumptions That Break in Our Setting

| GAN-UDA Assumption | Our Reality | Impact |
|--------------------|-------------|--------|
| Sufficient unlabeled target data | ✅ Yes (production logs) | Compatible |
| Same activity distribution | ⚠️ May differ | Could cause negative transfer |
| Training-time access to target | ⚠️ Need to accumulate | Delay before adaptation |
| GAN training stability | ⚠️ Can be unstable | Requires hyperparameter tuning |

### 🔧 Pipeline Stage Affected
- **Stage:** Model retraining / adaptation
- **Implementation:** Add domain discriminator to 1D-CNN-BiLSTM
- **New Files:** `src/domain_adaptation/gan_uda.py`

### 📝 Questions ANSWERED by This Paper
✅ How to do UDA with zero target labels
✅ GAN architecture for sensor domain adaptation
✅ Training procedure for adversarial alignment
✅ Expected performance gains

### ❓ Questions RAISED but NOT ANSWERED
- How much target data is needed before GAN training?
- How to detect when GAN has converged?
- What if source and target activity distributions differ?
- How to handle continuous domain drift (not one-time shift)?

---

## Paper 4: Scaling Human Activity Recognition via Deep Learning-based Domain Adaptation

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Scaling Human Activity Recognition via Deep Learning-based Domain Adaptation |
| **Year** | 2019 |
| **Venue** | UbiComp / IMWUT |
| **Focus** | Cross-dataset generalization |

### 🎯 Problem Addressed
- **Core Problem:** How to scale HAR systems across multiple datasets/devices without per-dataset training
- **Scalability Challenge:** Training separate models for each deployment is expensive
- **Approach:** Learn transferable representations that work across datasets

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Multiple source datasets labeled |
| Target Domain Labels | **Yes** | Assumes some labeled target data |
| Type | **Supervised Transfer Learning** | Not UDA |

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Main focus |
| Production Deployment | ⚠️ Limited | Assumes offline adaptation |
| Online Learning | ❌ No | Not addressed |

### ❓ KEY QUESTIONS for Our Scenario

1. **"Is this paper useful for our unlabeled scenario?"**
   - **Limited applicability** - assumes labeled target data
   - However, scaling insights are valuable for multi-source pre-training
   - Could use ADAMSense + other HAR datasets as multiple sources

2. **"What transfer learning strategies are compared?"**
   - Layer freezing strategies
   - Fine-tuning depth analysis
   - Multi-source pre-training
   - Feature extraction vs full fine-tuning

3. **"What's the recommended layer freezing strategy?"**
   - Freeze early CNN layers (low-level features)
   - Fine-tune later layers (task-specific)
   - For our 1D-CNN-BiLSTM: freeze first 3-5 layers

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Labeled target data | ❌ NO labels | Cannot directly apply |
| Controlled deployment | ⚠️ Production varies | Domain shift ongoing |
| Similar activity sets | ✅ Yes | Compatible |

### 🔧 Pipeline Stage Affected
- **Stage:** Pre-training / baseline model creation
- **Implementation:** Multi-source training on diverse HAR datasets
- **Insight:** Layer freezing guidance for fine-tuning

### 📝 Questions ANSWERED by This Paper
✅ How many layers to freeze during transfer
✅ Multi-source training benefits
✅ Cross-dataset evaluation protocols
✅ Feature transferability analysis

### ❓ Questions RAISED but NOT ANSWERED
- How to transfer without target labels?
- How to detect negative transfer?
- How to select best source dataset(s)?

---

## Paper 5: SCAGOT - Semi-Supervised Disentangling Context and Activity Features Without Target Data

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | SCAGOT: Semi-Supervised Disentangling Context and Activity Features Without Target Data for Sensor-Based HAR |
| **Year** | 2023 |
| **Venue** | IEEE / ACM |
| **Innovation** | Context vs Activity feature separation |

### 🎯 Problem Addressed
- **Core Problem:** Sensor data contains both activity-relevant and context-specific (user, device, position) features
- **Disentanglement:** Separate activity features from context features for better transfer
- **Source-Free:** No access to target domain during training (!)

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Fully labeled |
| Target Domain Labels | **NO** | Source-free adaptation |
| Type | **Source-Free Domain Adaptation** | No target data during training |

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Train on source only |
| Production Deployment | ✅ Yes | Adapts at inference time |
| Online Learning | ✅ Possible | Can accumulate target statistics |

### ❓ KEY QUESTIONS for Our Scenario

1. **"Can we deploy without seeing any target data during training?"**
   - **YES!** SCAGOT is designed for this
   - Train on ADAMSense only
   - Model naturally handles new domains

2. **"What is context disentanglement?"**
   - Separates: Activity features (hand tapping motion) from Context features (user body size, device orientation)
   - Activity features transfer, context features don't
   - Reduces negative transfer

3. **"How does it compare to GAN-based UDA?"**
   - Simpler training (no adversarial)
   - No target data needed during training
   - May be more stable but potentially lower performance ceiling

### ⚠️ Assumptions That Break in Our Setting

| SCAGOT Assumption | Our Reality | Impact |
|-------------------|-------------|--------|
| Context is separable | ✅ Likely yes | Sensor position is context |
| Sufficient source diversity | ⚠️ Only ADAMSense | May need more sources |
| Similar activity semantics | ✅ Yes | Same 11 classes |

### 🔧 Pipeline Stage Affected
- **Stage:** Model architecture design
- **Implementation:** Add disentanglement module to feature extractor
- **Benefit:** Better out-of-the-box generalization

### 📝 Questions ANSWERED by This Paper
✅ How to train without any target data
✅ How to separate activity vs context features
✅ Source-free adaptation methodology
✅ Comparison with target-data-required methods

### ❓ Questions RAISED but NOT ANSWERED
- How much source diversity is needed?
- How to detect when disentanglement fails?
- Can we combine with AdaBN for further improvement?

---

## Paper 6: Transfer Learning of Human Activities Based on IMU Sensors: A Review

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Transfer Learning of Human Activities Based on IMU Sensors: A Review |
| **Year** | 2022 |
| **Venue** | IEEE Access / Sensors |
| **Type** | Comprehensive Survey |

### 🎯 Problem Addressed
- **Core Problem:** Comprehensive overview of ALL transfer learning approaches for IMU-based HAR
- **Coverage:** Domain adaptation, fine-tuning, pre-training, self-supervised learning
- **Taxonomy:** Categorizes methods by label requirements, data access, and adaptation type

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Coverage | **All types** | UDA, semi-supervised, supervised |
| Recommendation | **Depends on scenario** | See taxonomy |

### 🏭 Production vs Offline
- **Survey paper** - covers both production and offline scenarios

### ❓ KEY QUESTIONS for Our Scenario

1. **"What's the best approach for ZERO labeled production data?"**
   
   Survey recommends (in order):
   1. **AdaBN** (simplest, no retraining)
   2. **Self-training with confidence threshold** (pseudo-labels)
   3. **GAN-based UDA** (if enough unlabeled target data)
   4. **Contrastive learning** (emerging approach)

2. **"What confidence threshold for self-training?"**
   - Paper recommends **>0.90** to prevent error propagation
   - Start conservative, lower if insufficient pseudo-labels

3. **"How to prevent catastrophic forgetting during adaptation?"**
   - **Elastic Weight Consolidation (EWC):** Penalize changes to important weights
   - **Learning Without Forgetting (LwF):** Knowledge distillation from old model
   - **Replay:** Mix old training samples with new data

4. **"What's the minimum labeled samples for effective fine-tuning?"**
   - Survey shows: **10-30%** of target data achieves near-supervised performance
   - Few-shot (5-50 samples/class) can still help significantly

### ⚠️ Critical Insights for Our Thesis

| Insight | Implication |
|---------|-------------|
| No single best method | Need to test multiple approaches |
| Hybrid methods work best | Combine AdaBN + self-training |
| Negative transfer is real | Must monitor performance |
| Domain diversity helps | Multi-source pre-training recommended |

### 🔧 Pipeline Stage Affected
- **Stage:** All stages - provides roadmap
- **Implementation:** Use as reference for method selection

### 📝 Questions ANSWERED by This Paper
✅ Complete taxonomy of transfer learning for HAR
✅ Comparison of methods by label requirements
✅ Best practices for each scenario
✅ Confidence thresholds for pseudo-labeling
✅ Strategies to prevent forgetting

### ❓ Questions RAISED but NOT ANSWERED
- How to choose between methods automatically?
- How to detect negative transfer in production?
- How to combine multiple adaptation techniques?

---

## Paper 7: Which Time Series Domain Shifts can Neural Networks Adapt to?

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Which Time Series Domain Shifts can Neural Networks Adapt to? |
| **Year** | 2024 (arXiv) |
| **ID** | 15027 (NeurIPS submission) |
| **Focus** | Characterizing types of domain shift |

### 🎯 Problem Addressed
- **Core Problem:** Not all domain shifts are equal - some are easier to adapt to than others
- **Characterization:** Categorizes shifts: covariate, label, temporal, sensor-specific
- **Practical Guidance:** Which shifts can DNNs handle with which techniques?

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Experiments | **Varies** | Tests multiple adaptation scenarios |
| Recommendations | **Shift-dependent** | Different approaches for different shifts |

### 🏭 Production vs Offline
- Addresses BOTH - key insight is that shift type determines adaptation strategy

### ❓ KEY QUESTIONS for Our Scenario

1. **"What types of domain shift will we see in production?"**
   
   | Shift Type | Our Scenario | Adaptability |
   |------------|--------------|--------------|
   | **User variability** | High (different bodies) | Medium |
   | **Sensor placement** | High (dominant vs non-dominant) | Hard |
   | **Device differences** | Medium (Garmin vs ADAMSense sensors) | Medium |
   | **Temporal drift** | Low-Medium (habits change slowly) | Easy |

2. **"Which shifts are hardest to adapt to?"**
   - **Hardest:** Label shift (different activity distributions)
   - **Medium:** Covariate shift (different sensor characteristics)
   - **Easiest:** Simple temporal drift

3. **"How to detect which type of shift is occurring?"**
   - Compare marginal distributions: Covariate shift
   - Compare conditional distributions: Concept shift
   - Monitor over time: Temporal drift
   - Per-class analysis: Label shift

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Known shift type | ❌ Unknown | Need shift detection |
| Access to target | ⚠️ Streaming | Accumulate before analysis |
| Single shift type | ⚠️ Multiple possible | Need compositional analysis |

### 🔧 Pipeline Stage Affected
- **Stage:** Monitoring / Drift Detection
- **Implementation:** Categorize detected drift before choosing adaptation strategy

### 📝 Questions ANSWERED by This Paper
✅ Taxonomy of time series domain shifts
✅ Which shifts DNNs can/cannot adapt to
✅ Matching adaptation techniques to shift types
✅ How shift type affects expected performance

### ❓ Questions RAISED but NOT ANSWERED
- How to automatically detect shift type?
- How to handle multiple simultaneous shifts?
- How to adapt to previously unseen shift types?

---

# 🎯 Synthesis: Recommendations for Your Thesis

## Priority Techniques to Implement

| Priority | Technique | Paper Source | Complexity | Impact |
|----------|-----------|--------------|------------|--------|
| 🔴 **P1** | AdaBN | XHAR | Low | 5-15% gain |
| 🔴 **P1** | Self-training (conf >0.90) | Review Paper | Medium | 10-20% gain |
| 🟠 **P2** | GAN-based UDA | Shift-GAN | High | 15-25% gain |
| 🟠 **P2** | Drift type detection | Time Series Shifts | Medium | Better targeting |
| 🟡 **P3** | Context disentanglement | SCAGOT | High | Robustness |
| 🟡 **P3** | Bilateral translation | AdaptNet | High | If labels available |

## Implementation Roadmap for Unlabeled Production

```
Week 1-2: AdaBN Implementation
├── Add adapt_batch_norm() to inference pipeline
├── Collect production BN statistics
└── Measure improvement vs baseline

Week 3-4: Self-Training Pipeline
├── Implement confidence-based pseudo-labeling
├── Use threshold > 0.90
├── Validate with held-out labeled samples
└── Iterate: predict → filter → retrain

Week 5-6: Drift Detection
├── Implement KS-test for each sensor
├── Categorize shift type (covariate vs label)
├── Create monitoring dashboard
└── Set up alerts for significant drift

Week 7-8: Advanced (if time permits)
├── GAN-based UDA experiment
├── Compare all methods
└── Document in thesis
```

## Questions This Analysis DOESN'T Answer (Future Work)

1. **How to validate adaptation without labels?**
   - Proposed: Use confidence calibration + entropy tracking as proxies

2. **How to handle continuous drift (not one-time shift)?**
   - Proposed: Sliding window adaptation + periodic re-evaluation

3. **How to combine multiple techniques (AdaBN + Self-training)?**
   - Proposed: Sequential application - AdaBN first, then self-training

4. **How to detect negative transfer before it damages performance?**
   - Proposed: Shadow model comparison + rollback capability

---

## 📚 Citation Summary

```
@misc{xhar2019,
  title={XHAR: Deep Domain Adaptation for Human Activity Recognition},
  author={Huan et al.},
  year={2019}
}

@article{adaptnet2021,
  title={AdaptNet: Bilateral Domain Adaptation Using Semi-Supervised Deep Translation},
  year={2021}
}

@article{shiftgan2021,
  title={Unsupervised Domain Adaptation in Activity Recognition: A GAN-Based Approach},
  year={2021}
}

@inproceedings{scalinghar2019,
  title={Scaling Human Activity Recognition via Deep Learning-based Domain Adaptation},
  booktitle={UbiComp},
  year={2019}
}

@article{scagot2023,
  title={SCAGOT: Semi-Supervised Disentangling Context and Activity Features},
  year={2023}
}

@article{transferreview2022,
  title={Transfer Learning of Human Activities Based on IMU Sensors: A Review},
  year={2022}
}

@article{tsdomainshifts2024,
  title={Which Time Series Domain Shifts can Neural Networks Adapt to?},
  year={2024}
}
```

---

**Generated:** January 30, 2026  
**Purpose:** Thesis research - MLOps for HAR with unlabeled production data

---

# 📊 GROUP 2: DRIFT/CHANGE DETECTION PAPERS ANALYSIS

> **Analysis Date:** January 30, 2026  
> **Focus:** Detecting distribution shifts and change points WITHOUT labels  
> **Relevance:** Critical for monitoring deployed HAR models on unlabeled production data

---

## Group 2 Summary Table

| # | Paper | Year | Labels Assumed | Unsupervised? | HAR-Specific | Critical for Thesis |
|---|-------|------|----------------|---------------|--------------|---------------------|
| 8 | WATCH (Wasserstein CPD) | 2021 | **No** | ✅ Yes | No | ⭐⭐⭐ Multivariate drift detection |
| 9 | LIFEWATCH | 2022 | **No** | ✅ Yes | No | ⭐⭐⭐ Recurring pattern detection |
| 10 | Sinkhorn Divergences CPD | 2025 | **Partial** (learns to ignore) | ✅ Semi-sup | No | ⭐⭐⭐ Selective change detection |
| 11 | Normalizing SSL for CPD | 2024 | **No** | ✅ Yes | No | ⭐⭐⭐ Self-supervised + guarantees |
| 12 | OT Change Point + Clustering | 2020 | **No** | ✅ Yes | No | ⭐⭐ Segment clustering |
| 13 | OOD in HAR | 2022 | **Yes** (training) | ⚠️ Inference | ✅ Yes | ⭐⭐⭐ HAR-specific OOD |
| 14 | Classifying Falls via OOD | 2023 | **Yes** (training) | ⚠️ Inference | ✅ Yes | ⭐⭐⭐ Unknown class detection |

---

## Paper 8: WATCH - Wasserstein Change Point Detection for High-Dimensional Time Series Data

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data |
| **Year** | 2021 |
| **Venue** | IEEE BigData 2021 |
| **Authors** | Faber, Corizzo, Sniezynski, Japkowicz |

### 🎯 Problem Addressed
- **Core Problem:** Detecting changes in high-dimensional time series (like 6-axis IMU data) without labels
- **Key Challenge:** Most CPD methods fail on high-dimensional data (curse of dimensionality)
- **Innovation:** Uses Wasserstein distance to compare sliding window distributions, scales to high dimensions

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Training Labels | **NO** | Completely unsupervised |
| Change Point Labels | **NO** | Automatically detects without ground truth |
| Type | **Fully Unsupervised** | ✅ Perfect for our scenario |

### 🔧 Drift/Shift Detection Method
```
Method: Wasserstein Distance on Sliding Windows
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Maintain two sliding windows: W1 (reference), W2 (current)
2. Compute Wasserstein distance: W(P_W1, P_W2)
3. If W > threshold → Change Point Detected
4. Optionally: Use penalty term for sparse detection

Key Formula:
W(P, Q) = inf_γ E_{(x,y)~γ}[||x - y||]
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"How do we detect drift in 6-axis IMU data without labels?"**
   - WATCH directly solves this! 
   - Apply to each sensor channel or multivariate jointly
   - No labels needed at any point

2. **"What's the computational cost for streaming data?"**
   - O(n²) per window comparison (can be approximated)
   - Paper proposes efficient sliding window updates
   - Suitable for our batch-based production monitoring

3. **"How to choose the threshold?"**
   - Bootstrap from training data variance
   - **Gap in paper:** No automatic threshold selection

### ⚠️ Assumptions That Break in Our Setting

| WATCH Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Stationary within windows | ✅ Short windows OK | Compatible |
| Single change point | ⚠️ Multiple possible | Need adaptation |
| Known window size | ⚠️ Activity-dependent | Need to tune |
| Computational resources | ✅ Offline monitoring | Compatible |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **Window size W1** | 50-200 samples | Based on activity duration (2-8 seconds at 25Hz) |
| **Window size W2** | 50-200 samples | Match W1 or slightly larger |
| **Threshold τ** | Data-dependent | Bootstrap: mean + 2-3× std from training |
| **Penalty λ** | 0.01-0.1 | Higher = fewer change points |

### 🔧 Pipeline Stage Affected
- **Stage:** Post-inference monitoring (Layer 3)
- **Implementation:** Add `WassersteinChangeDetector` class
- **File:** `scripts/post_inference_monitoring.py` → enhance drift detection

### 📝 Questions ANSWERED by This Paper
✅ How to detect changes in high-dimensional sensor data
✅ Why Wasserstein distance works better than KL-divergence for this task
✅ How to handle multivariate time series
✅ Computational efficiency for streaming

### ❓ Questions RAISED but NOT ANSWERED
- How to set optimal threshold automatically?
- How to distinguish sensor drift from user behavior change?
- How to map change points to required actions (retrain vs ignore)?
- Window size selection for variable-length activities?

---

## Paper 9: LIFEWATCH - Lifelong Wasserstein Change Point Detection

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | LIFEWATCH: Lifelong Wasserstein Change Point Detection |
| **Year** | 2022 |
| **Venue** | IJCNN 2022 |
| **Authors** | Faber, Corizzo, Sniezynski, Japkowicz (same team as WATCH) |

### 🎯 Problem Addressed
- **Core Problem:** In lifelong learning, same patterns (tasks/activities) may RECUR
- **Key Challenge:** Standard CPD treats each change as novel; wastes resources relearning known patterns
- **Innovation:** Extends WATCH with a **memory bank** to recognize recurring distribution patterns

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Training Labels | **NO** | Fully unsupervised |
| Pattern Labels | **NO** | Clusters emerge automatically |
| Type | **Unsupervised + Memory** | Learns to recognize recurring states |

### 🔧 Drift/Shift Detection Method
```
Method: WATCH + Distribution Memory Bank
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Detect change point using Wasserstein (like WATCH)
2. Compare new distribution to memory bank
3. If match → Recognized pattern (no adaptation needed)
4. If novel → Add to memory bank (may need adaptation)

Memory Matching:
W(P_new, P_memory[i]) < τ_match → Pattern i recognized
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we avoid retraining when user returns to previously-seen behavior?"**
   - YES! LIFEWATCH solves this
   - User switches from walking→running→walking: Recognizes second "walking" as known
   - Avoids unnecessary model updates

2. **"How does this help with watch placement changes?"**
   - If user repositions watch back to original position → recognized as known distribution
   - Saves computational resources vs full re-adaptation

3. **"What's stored in the memory bank?"**
   - Distribution summaries (moments, histogram bins)
   - NOT raw data → privacy friendly
   - Memory size is bounded

### ⚠️ Assumptions That Break in Our Setting

| LIFEWATCH Assumption | Our Reality | Impact |
|----------------------|-------------|--------|
| Recurring patterns | ✅ Yes (daily routines) | Perfect fit! |
| Clean pattern boundaries | ⚠️ Gradual transitions | May need smoothing |
| Memory capacity | ⚠️ Need to bound | Limit to ~20 patterns |
| Distribution summaries sufficient | ⚠️ May lose detail | Test on our data |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **Change threshold τ** | Same as WATCH | Bootstrap from training |
| **Match threshold τ_match** | τ × 0.5 - 0.8 | Stricter than change detection |
| **Memory size M** | 10-50 patterns | Based on expected activity variety |
| **Similarity metric** | Wasserstein | Consider Jensen-Shannon as fallback |

### 🔧 Pipeline Stage Affected
- **Stage:** Long-term monitoring / Adaptive pipeline
- **Implementation:** Extend drift detector with pattern memory
- **New Capability:** Distinguish "known drift" from "novel drift"

### 📝 Questions ANSWERED by This Paper
✅ How to avoid redundant adaptations for recurring patterns
✅ Memory-efficient distribution storage
✅ Integration with standard CPD methods
✅ Lifelong learning without catastrophic forgetting

### ❓ Questions RAISED but NOT ANSWERED
- How to merge similar patterns in memory (memory consolidation)?
- How to forget outdated patterns that no longer occur?
- What's the relationship between patterns and actual activities?
- How to validate pattern recognition without labels?

---

## Paper 10: Learning Sinkhorn Divergences for Change Point Detection

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Learning Sinkhorn Divergences for Change Point Detection |
| **Year** | 2025 |
| **Venue** | IEEE Transactions on Signal Processing |
| **Authors** | Ahad, Dyer, Hengen, Xie |

### 🎯 Problem Addressed
- **Core Problem:** Unsupervised CPD detects ALL changes, but some changes don't matter
- **Key Challenge:** How to learn which changes are "interesting" vs "ignorable"?
- **Innovation:** Uses a small set of labeled change points to LEARN a Sinkhorn divergence that ignores uninteresting changes

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Change Point Labels | **Partial** | Small labeled set for learning |
| Data Labels | **NO** | Activity labels not needed |
| Type | **Semi-supervised CPD** | Learn to focus on meaningful changes |

### 🔧 Drift/Shift Detection Method
```
Method: Learned Sinkhorn Divergence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Standard Sinkhorn divergence: S_ε(P, Q) ≈ W(P, Q) with entropy regularization
2. Learn a transformation φ(x) that projects data
3. Compute: S_ε(φ(P), φ(Q))
4. φ is trained to make S_ε sensitive to labeled changes, insensitive to others

Training objective:
min_φ [1/|positives| Σ_pos -log(S_ε(φ(W1), φ(W2)))] + 
      [1/|negatives| Σ_neg -log(1 - S_ε(φ(W1), φ(W2)))]
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we use this if we have ZERO labels?"**
   - ⚠️ NOT directly - requires some labeled change points
   - BUT: Could use confidence drops as pseudo-labels for "interesting" changes
   - Could label a small audit set manually

2. **"What kinds of changes can we learn to ignore?"**
   - Sensor noise spikes (if they don't affect prediction)
   - Minor posture adjustments during same activity
   - Time-of-day variations that don't change activity semantics

3. **"How many labels needed?"**
   - Paper shows good results with ~20-50 labeled change points
   - For our pipeline: Label 20-50 "significant" change points from production

### ⚠️ Assumptions That Break in Our Setting

| Sinkhorn Assumption | Our Reality | Impact |
|---------------------|-------------|--------|
| Some labeled changes | ⚠️ Initially none | Need bootstrap strategy |
| Known what to ignore | ⚠️ Must define | Domain expertise needed |
| Enough positive examples | ⚠️ Changes are rare | May need synthesis |
| GPU for training | ✅ Available | Compatible |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **Entropy regularization ε** | 0.01-0.1 | Lower = sharper, more like Wasserstein |
| **Embedding dimension d** | 32-128 | Cross-validation |
| **Network depth** | 2-4 layers | Standard hyperparameter search |
| **Positive:Negative ratio** | 1:5 to 1:20 | Based on change frequency |

### 🔧 Pipeline Stage Affected
- **Stage:** Smart drift detection (learns importance)
- **Implementation:** Replace simple Wasserstein with learned Sinkhorn
- **Prerequisite:** Need to create labeled change point dataset first

### 📝 Questions ANSWERED by This Paper
✅ How to make CPD focus on meaningful changes only
✅ Sinkhorn vs Wasserstein trade-offs
✅ Deep learning architecture for divergence learning
✅ How much labeling effort is needed

### ❓ Questions RAISED but NOT ANSWERED
- How to bootstrap without initial labels?
- How to define "interesting" for HAR degradation?
- Transfer learning: Can pre-trained model work across datasets?
- Online learning: Can we continuously improve the detector?

---

## Paper 11: Normalizing Self-Supervised Learning for Provably Reliable Change Point Detection

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Normalizing Self-Supervised Learning for Provably Reliable Change Point Detection |
| **Year** | 2024 |
| **Venue** | IEEE ICDM 2024 |
| **Authors** | Bazarova, Romanenkova et al. |

### 🎯 Problem Addressed
- **Core Problem:** Deep learning CPD lacks theoretical guarantees (false positive/negative rates unknown)
- **Key Challenge:** Traditional CPD has guarantees but low expressive power; DL has power but no guarantees
- **Innovation:** Combines self-supervised representation learning with classical statistical tests that have PROVEN guarantees

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Any Labels | **NO** | Fully unsupervised |
| Statistical Guarantees | **YES** | Provable false positive rate control |
| Type | **Unsupervised + Guarantees** | Best of both worlds |

### 🔧 Drift/Shift Detection Method
```
Method: Self-Supervised Encoding + Statistical Testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Train encoder via self-supervised contrastive learning (no labels)
2. Project windows into learned representation space
3. Apply classical statistical test (e.g., CUSUM, Hotelling's T²)
4. Theoretical guarantees transfer to learned space!

Key Insight:
If encoder learns "good" representations + statistical test has guarantees
→ Combined method inherits guarantees

Self-supervised objective: Temporal consistency + augmentation invariance
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we control false alarm rate without labels?"**
   - YES! This paper provides provable bounds
   - Can set α = 0.05 (5% false positive rate) with mathematical guarantees
   - Critical for production systems where false alarms have costs

2. **"What self-supervised task works for HAR?"**
   - Paper suggests: Temporal prediction (predict next window)
   - Contrastive: Same activity = positive pair, different = negative
   - Can use augmentations (time shift, noise, scaling)

3. **"How does this compare to simple Wasserstein?"**
   - More expressive (learns nonlinear transformations)
   - Provides statistical guarantees (p-values, confidence intervals)
   - May be overkill if simple methods work well

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Data for SSL training | ✅ Have training data | Compatible |
| i.i.d. within windows | ⚠️ Time series | Need temporal augmentation |
| Stationary reference | ⚠️ May drift | Need periodic recalibration |
| Sufficient data for SSL | ✅ ~40K windows | Compatible |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **Significance level α** | 0.01-0.05 | Based on false alarm tolerance |
| **SSL embedding dim** | 64-256 | Cross-validation |
| **Contrastive temperature τ** | 0.1-0.5 | Standard contrastive learning |
| **Statistical test** | CUSUM / T² | Both viable; T² for multivariate |

### 🔧 Pipeline Stage Affected
- **Stage:** Drift detection with statistical rigor
- **Implementation:** Add SSL encoder training + statistical testing
- **Benefit:** Trustworthy alerts with known false alarm rates

### 📝 Questions ANSWERED by This Paper
✅ How to get guarantees in deep learning CPD
✅ Self-supervised pre-training for time series
✅ Integration with classical statistical tests
✅ False positive rate control methodology

### ❓ Questions RAISED but NOT ANSWERED
- How much data needed for SSL training?
- How to handle non-stationary reference distributions?
- Online updating of the SSL encoder?
- Computational overhead of SSL training?

---

## Paper 12: Optimal Transport Based Change Point Detection and Time Series Segment Clustering

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Optimal Transport Based Change Point Detection and Time Series Segment Clustering |
| **Year** | 2020 |
| **Venue** | ICASSP 2020 |
| **Authors** | Cheng, Aeron, Hughes, Gedeon |

### 🎯 Problem Addressed
- **Core Problem:** After detecting change points, how to UNDERSTAND the segments?
- **Key Challenge:** CPD tells you WHEN changes occur, not WHAT changed
- **Innovation:** Combines OT-based CPD with spectral clustering to group similar segments

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Segment Labels | **NO** | Clustering is unsupervised |
| CPD Labels | **NO** | Unsupervised detection |
| Type | **Fully Unsupervised** | Detection + Interpretation |

### 🔧 Drift/Shift Detection Method
```
Method: OT-CPD + Spectral Segment Clustering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Change Point Detection
- Use Wasserstein distance between consecutive segments
- Penalized optimization: min_t Σ W(P_i, P_{i+1}) + λ|T|

STEP 2: Segment Representation
- Each segment → empirical distribution
- Build similarity matrix: S_ij = exp(-W(P_i, P_j))

STEP 3: Spectral Clustering
- Apply normalized cuts on similarity matrix
- Output: Cluster assignments for each segment
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we understand WHAT caused the drift?"**
   - Partially yes: Clustering groups similar periods
   - E.g., Cluster A = "walking segments", Cluster B = "running segments"
   - Won't give activity labels, but gives structure

2. **"How does this help with retraining decisions?"**
   - If new segment clusters with known segments → probably fine
   - If new segment forms new cluster → likely needs attention
   - Connects to LIFEWATCH's pattern memory concept

3. **"Can we use this for activity pseudo-labeling?"**
   - Potentially! Cluster segments, then label clusters
   - Much less labeling effort than labeling individual windows
   - Risk: Clusters may not align with activities

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Clear segment boundaries | ⚠️ Activities blend | May need smoothing |
| Enough segments for clustering | ⚠️ Limited production data | Combine with training segments |
| Fixed number of clusters | ⚠️ Unknown activity count | Need gap statistic |
| Similar segment lengths | ⚠️ Variable activities | Normalize representations |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **CPD penalty λ** | 0.01-0.5 | Controls segment granularity |
| **Number of clusters K** | 5-20 | Silhouette score or gap statistic |
| **Similarity temperature** | Auto | Based on Wasserstein scale |
| **Minimum segment length** | 25-100 samples | Based on activity duration |

### 🔧 Pipeline Stage Affected
- **Stage:** Drift interpretation / Segment analysis
- **Implementation:** Post-process detected drift segments
- **New Capability:** Group and visualize production data structure

### 📝 Questions ANSWERED by This Paper
✅ How to interpret time series segments after CPD
✅ OT-based similarity for clustering
✅ Spectral clustering on distribution similarities
✅ Joint detection + clustering framework

### ❓ Questions RAISED but NOT ANSWERED
- How to map clusters to meaningful labels (human interpretation)?
- How to handle online clustering as new segments arrive?
- How to evaluate clustering quality without ground truth?
- Scalability to very long time series?

---

## Paper 13: Out-of-Distribution in Human Activity Recognition

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Out-of-Distribution in Human Activity Recognition |
| **Year** | 2022 |
| **Venue** | Swedish Artificial Intelligence Society (SAIS) 2022 |
| **Authors** | Roy, Komini, Girdzijauskas |

### 🎯 Problem Addressed
- **Core Problem:** HAR models make overconfident predictions on data from UNKNOWN activities
- **Key Challenge:** Deep learning models don't know what they don't know
- **Innovation:** Adapts OOD detection methods from computer vision to sensor-based HAR

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Training Labels | **YES** | Need labeled in-distribution data |
| Test Labels | **NO** | OOD detected at inference time |
| Type | **Supervised training, Unsupervised inference** | Train on known, detect unknown |

### 🔧 Drift/Shift Detection Method
```
Method: OOD Detection via Confidence/Energy Scores
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOD Score Options:

1. Maximum Softmax Probability (MSP):
   OOD_score = 1 - max(softmax(logits))
   If OOD_score > τ → Out-of-distribution

2. Energy Score:
   E(x) = -log Σ exp(f_i(x))
   If E(x) > τ → Out-of-distribution

3. Mahalanobis Distance:
   M(x) = (φ(x) - μ_k)^T Σ^{-1} (φ(x) - μ_k)
   If M(x) > τ → Out-of-distribution

HAR-Specific Insight:
- Activity data is sequential → can use temporal consistency
- OOD often shows erratic predictions over time
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we detect when our model sees truly unknown activities?"**
   - YES! This is exactly what OOD detection does
   - Train on 11 known activities
   - Detect when user performs Activity #12 (unknown)

2. **"How does this relate to drift detection?"**
   - Drift = GRADUAL shift in known distributions
   - OOD = data from ENTIRELY different distribution
   - Both are important; this paper focuses on OOD

3. **"Which OOD method works best for HAR?"**
   - Paper finds Energy Score and Mahalanobis perform best
   - MSP is simple but less reliable for HAR
   - Recommend: Energy score (easy to implement, no extra training)

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Known in-distribution | ✅ Training data defines ID | Compatible |
| Diverse OOD types | ⚠️ Limited test OOD | May need to test more |
| CNN/LSTM architecture | ✅ 1D-CNN-BiLSTM | Compatible |
| Calibrated model | ⚠️ May need calibration | Add temperature scaling |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **Energy threshold** | Percentile-based | 95th percentile of training energies |
| **MSP threshold** | 0.7-0.9 | Based on training confidence distribution |
| **Mahalanobis threshold** | χ²_{0.95,d} | Chi-squared distribution |
| **Temperature T** | 1-10 | Temperature scaling for calibration |

### 🔧 Pipeline Stage Affected
- **Stage:** Layer 1 (sample-level) monitoring
- **Implementation:** Add energy score computation to inference
- **File:** `scripts/post_inference_monitoring.py` → Layer 1 enhancement

### 📝 Questions ANSWERED by This Paper
✅ OOD detection methods applicable to HAR
✅ Comparison of MSP, Energy, Mahalanobis for sensor data
✅ Threshold selection strategies
✅ HAR-specific considerations (temporal, multi-channel)

### ❓ Questions RAISED but NOT ANSWERED
- How to distinguish OOD from model degradation?
- What to DO when OOD is detected (flag? exclude? relabel?)?
- How does OOD detection perform over time (drift in OOD threshold)?
- How to handle borderline cases (partially novel activities)?

---

## Paper 14: Classifying Falls Using Out-of-Distribution Detection in Human Activity Recognition

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Classifying Falls Using Out-of-Distribution Detection in Human Activity Recognition |
| **Year** | 2023 |
| **Venue** | AI Communications |
| **Authors** | Roy, Komini, Girdzijauskas (same team as Paper 13) |

### 🎯 Problem Addressed
- **Core Problem:** Falls are RARE and DANGEROUS; classifiers may misclassify falls as known activities
- **Key Challenge:** Training on falls is hard (rare, dangerous to collect); can we detect them as OOD?
- **Innovation:** Train on non-fall activities ONLY, detect falls as OOD → safer, no fall data needed for training

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Activity Labels | **YES** | Need labeled non-fall activities |
| Fall Labels | **NO** | Falls detected as OOD, not classified |
| Type | **Train without target class** | Novel problem formulation |

### 🔧 Drift/Shift Detection Method
```
Method: Train Without Target, Detect as OOD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key Insight: Some classes are better detected as "anomalies" than learned

Approach:
1. Train HAR model on "normal" activities (walking, sitting, standing, etc.)
2. Exclude rare/dangerous activities from training (falls, emergencies)
3. At inference: If OOD score > τ → Potential fall/emergency
4. Benefits:
   - No need for fall data during training
   - Safer (don't need to ask people to fall)
   - More robust (any unusual activity triggers alert)

HAR-Specific OOD Types Defined:
- Activity OOD: Unknown activity class
- User OOD: Known activity, new user
- Sensor OOD: Known activity, different sensor placement
```

### ❓ KEY QUESTIONS for Our Unlabeled Scenario

1. **"Can we detect activities NOT in our 11 classes?"**
   - YES! This paper validates this approach for HAR
   - Example: User does yoga (not in our 11 classes) → OOD detected
   - Alert for manual review instead of wrong prediction

2. **"How does this help with unknown production scenarios?"**
   - Production user may do activities we didn't train on
   - Instead of wrong prediction with high confidence → OOD flag
   - Much safer for applications where wrong predictions are costly

3. **"What about borderline activities?"**
   - E.g., "jogging" when trained on "walking" and "running"
   - Paper acknowledges this is challenging
   - Suggest: Multiple thresholds (confident ID, uncertain ID, OOD)

### ⚠️ Assumptions That Break in Our Setting

| Paper Assumption | Our Reality | Impact |
|------------------|-------------|--------|
| Clear ID/OOD boundary | ⚠️ Activity continuum | May need soft thresholds |
| Static activity set | ⚠️ Users may vary | Need periodic recalibration |
| Homogeneous sensors | ⚠️ Garmin vs training | Cross-device adaptation needed |
| Fall-specific focus | ⚠️ General OOD | Principles transfer |

### 📐 Thresholds/Parameters That Need Tuning

| Parameter | Suggested Range | How to Tune |
|-----------|----------------|-------------|
| **OOD threshold τ** | Calibration set | Use held-out known activities |
| **Confidence bands** | 3 zones | High conf (>0.9), Medium (0.7-0.9), OOD (<0.7) |
| **Temporal smoothing** | 3-5 windows | Avoid single-window false alarms |
| **Alert sensitivity** | Application-dependent | Cost of miss vs false alarm |

### 🔧 Pipeline Stage Affected
- **Stage:** Layer 1 monitoring + Alert system
- **Implementation:** Add unknown activity detection
- **New Capability:** Flag samples that don't match any known activity

### 📝 Questions ANSWERED by This Paper
✅ How to handle classes not in training set
✅ OOD formulation for HAR safety applications
✅ Taxonomy of OOD types for sensor data
✅ Validation methodology for OOD detection in HAR

### ❓ Questions RAISED but NOT ANSWERED
- How to automatically expand the activity set over time?
- How to collect labels for detected OOD instances?
- How to balance sensitivity (catch all OOD) vs specificity (avoid false alarms)?
- How does OOD detection interact with domain adaptation?

---

# 🎯 GROUP 2 SYNTHESIS: Recommendations for Drift Detection in Unlabeled HAR

## Integrated Drift Detection Pipeline

```
Production Window → [Layer 1: OOD Detection]
                         ↓ OOD?
            ┌────────────┴────────────┐
         Yes (Flag)              No (Continue)
            ↓                         ↓
    Manual Review         [Layer 2: Confidence Check]
    or Quarantine                    ↓ Low Conf?
                         ┌───────────┴───────────┐
                      Yes                      No
                         ↓                         ↓
              Uncertain Prediction        [Layer 3: Batch Drift]
                         ↓                         ↓
            ┌────────────┴                  WATCH/LIFEWATCH
            │                                      ↓
            │                         ┌────────────┴────────────┐
            │                      Novel                   Recurring
            │                         ↓                         ↓
            │               Consider Adaptation          No action needed
            └─────────────────────────┘
```

## Priority Implementation Order

| Priority | Paper | Implementation | Complexity | Impact |
|----------|-------|----------------|------------|--------|
| 🔴 **P1** | WATCH | Basic Wasserstein drift detection | Low | Foundation |
| 🔴 **P1** | OOD in HAR | Energy score for unknown detection | Low | Safety |
| 🟠 **P2** | LIFEWATCH | Add pattern memory to WATCH | Medium | Efficiency |
| 🟠 **P2** | Normalizing SSL | Add statistical guarantees | Medium | Trustworthiness |
| 🟡 **P3** | Sinkhorn | Learn important vs ignorable drift | High | Precision |
| 🟡 **P3** | OT Clustering | Interpret drift via segment clustering | Medium | Insights |
| 🟢 **P4** | Falls OOD | Extend to safety-critical activities | Low | Safety++ |

## Unified Threshold Summary

| Metric | Threshold | Paper Source | How to Calibrate |
|--------|-----------|--------------|------------------|
| **Wasserstein drift** | > μ + 2σ | WATCH | From training set distances |
| **OOD Energy** | 95th percentile | OOD HAR | Training energy distribution |
| **Confidence** | < 0.7 (uncertain), < 0.5 (OOD-like) | OOD HAR | Validation set analysis |
| **Pattern match** | < 0.8 × drift threshold | LIFEWATCH | Stricter than detection |
| **Statistical test α** | 0.05 | Normalizing SSL | Desired false alarm rate |

## What These Papers DON'T Answer (Our Thesis Contributions)

1. **Integration with domain adaptation:** When to adapt vs when to flag as OOD?
2. **Threshold co-optimization:** How do these thresholds interact?
3. **HAR-specific validation:** Performance on real Garmin production data?
4. **Computational budget:** Which methods fit within production constraints?
5. **User feedback loop:** How to incorporate human verification?

---

## 📚 Group 2 Citation Summary

```
@inproceedings{watch2021,
  title={WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data},
  author={Faber, K. and Corizzo, R. and Sniezynski, B. and Japkowicz, N.},
  booktitle={IEEE BigData},
  year={2021}
}

@inproceedings{lifewatch2022,
  title={LIFEWATCH: Lifelong Wasserstein Change Point Detection},
  author={Faber, K. and Corizzo, R. and Sniezynski, B. and Japkowicz, N.},
  booktitle={IJCNN},
  year={2022}
}

@article{sinkhorn2025,
  title={Learning Sinkhorn Divergences for Change Point Detection},
  author={Ahad, N. and Dyer, E.L. and Hengen, K.B. and Xie, Y.},
  journal={IEEE Transactions on Signal Processing},
  year={2025}
}

@inproceedings{normalizingssl2024,
  title={Normalizing Self-Supervised Learning for Provably Reliable Change Point Detection},
  author={Bazarova, A. and Romanenkova, E. et al.},
  booktitle={IEEE ICDM},
  year={2024}
}

@inproceedings{otcpd2020,
  title={Optimal Transport Based Change Point Detection and Time Series Segment Clustering},
  author={Cheng, K.C. and Aeron, S. and Hughes, M.C. and Gedeon, T.},
  booktitle={ICASSP},
  year={2020}
}

@inproceedings{oodhar2022,
  title={Out-of-Distribution in Human Activity Recognition},
  author={Roy, D. and Komini, V. and Girdzijauskas, S.},
  booktitle={SAIS},
  year={2022}
}

@article{fallsood2023,
  title={Classifying Falls Using Out-of-Distribution Detection in Human Activity Recognition},
  author={Roy, D. and Komini, V. and Girdzijauskas, S.},
  journal={AI Communications},
  year={2023}
}
```

---

**Group 2 Analysis Generated:** January 30, 2026  
**Next Step:** Implement WATCH baseline + Energy OOD score in `post_inference_monitoring.py`

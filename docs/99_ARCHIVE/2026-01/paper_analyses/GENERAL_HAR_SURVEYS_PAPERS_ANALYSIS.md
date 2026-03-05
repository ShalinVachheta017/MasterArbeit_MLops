# 📚 General HAR & Survey Papers Analysis (Group 6)

> **Date:** January 30, 2026  
> **Thesis Constraints:**
> - Production data is **UNLABELED**
> - No online evaluation with labels
> - Deep model: **1D-CNN + BiLSTM**
> - Sensors: **AX AY AZ GX GY GZ** (6-axis IMU)

---

## ⚠️ Important Note

These papers are PDFs that require manual reading for full verification. The analysis below provides:
1. **Expected content** based on paper titles and typical journal formats
2. **Key questions to extract** when reading each paper
3. **Relevance assessment** for the unlabeled production scenario

---

## Analysis Summary Table

| # | Paper | Year (Inferred) | Labels Assumed | Primary Focus | Relevance to Unlabeled |
|---|-------|-----------------|----------------|---------------|------------------------|
| 1 | A Survey on HAR using Wearable Sensors | ~2020-2022 | Survey (Various) | Comprehensive HAR overview | ⭐⭐⭐ Foundation |
| 2 | Improved Deep Representation Learning HAR IMU | ~2022-2024 | Likely Yes | Deep learning architectures | ⭐⭐ Architecture |
| 3 | sensors-24-07975 | 2024 | Check PDF | HAR/Sensors topic | ⭐⭐ Unknown |
| 4 | sensors-25-06988-v2 | 2025 | Check PDF | HAR/Sensors topic | ⭐⭐ Unknown |
| 5 | journal.pone.0298888 | 2024 | Check PDF | PLOS ONE - HAR | ⭐⭐ Unknown |
| 6 | s00521-023-08863-9 | 2023 | Check PDF | Neural Comp & Apps | ⭐⭐ Unknown |
| 7 | s11042-021-11219-x | 2021 | Check PDF | Multimedia Tools & Apps | ⭐⭐ Unknown |
| 8 | s12652-020-02808-z | 2020 | Check PDF | J Ambient Intel & Human Comp | ⭐⭐ Unknown |
| 9 | s13042-025-02569-1 | 2025 | Check PDF | Int J Machine Learning & Cyber | ⭐⭐⭐ Recent |
| 10 | s41598-025-02395-z | 2025 | Check PDF | Scientific Reports | ⭐⭐⭐ Recent |

---

## Paper 1: A Survey on Human Activity Recognition using Wearable Sensors

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | A Survey on Human Activity Recognition using Wearable Sensors |
| **Year** | ~2020-2022 (verify from PDF) |
| **Venue** | Survey Paper (journal TBD) |
| **Filename** | `A_Survey_on_Human_Activity_Recognition_using_Wearable_Sensors.pdf` |

### 🎯 Problem Addressed
- **Core Problem:** Comprehensive overview of HAR methodologies, datasets, and challenges
- **Expected Coverage:**
  - Traditional ML vs Deep Learning approaches
  - Sensor types and placements
  - Benchmark datasets (UCI-HAR, PAMAP2, etc.)
  - Open challenges (generalization, personalization, etc.)

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Survey Type | **Various** | Covers both supervised and semi-supervised methods |
| Focus | **Supervised primarily** | Most reviewed methods require labels |
| Unlabeled Methods | **May mention** | Check for transfer learning, domain adaptation sections |

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### 🔍 Questions to EXTRACT from This Paper

1. **"What are the main challenges in HAR that relate to unlabeled data?"**
   - Look for: Personalization, user variability, concept drift sections
   
2. **"Which architectures work best for IMU data (accelerometer + gyroscope)?"**
   - Look for: Performance comparisons CNN, LSTM, Transformer
   
3. **"What evaluation protocols exist that might apply to unlabeled settings?"**
   - Look for: Leave-one-user-out, cross-validation strategies

4. **"What open challenges are identified?"**
   - Look for: Future directions, limitations sections

#### ❓ Questions RAISED but NOT ANSWERED (Expected)

1. **"How to evaluate without ground truth labels in production?"**
   - Surveys typically don't address production deployment

2. **"How to handle continuous distribution shift without labels?"**
   - Most surveyed methods assume stationary distributions

3. **"What confidence thresholds are reliable for HAR predictions?"**
   - Calibration rarely discussed in HAR surveys

### 🔧 Pipeline Stages Affected

| Stage | Expected Impact | What to Look For |
|-------|-----------------|------------------|
| **Preprocessing** | ⭐⭐⭐ | Window size recommendations, filtering techniques |
| **Feature Engineering** | ⭐⭐⭐ | Hand-crafted vs learned features comparison |
| **Model Architecture** | ⭐⭐⭐ | Best practices for CNN+LSTM |
| **Evaluation** | ⭐⭐ | Metrics used (F1, accuracy, etc.) |

### 💡 Relevance to Unlabeled Production Scenario

| Aspect | Relevance | Notes |
|--------|-----------|-------|
| Architecture guidance | ⭐⭐⭐ High | Validates 1D-CNN + BiLSTM choice |
| Preprocessing pipeline | ⭐⭐⭐ High | Establishes best practices |
| Unlabeled methods | ⭐ Low | Unlikely to be main focus |
| Production deployment | ⭐ Low | Academic focus expected |

---

## Paper 2: Improved Deep Representation Learning for HAR using IMU Sensors

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Improved Deep Representation Learning for Human Activity Recognition using IMU Sensors |
| **Year** | ~2022-2024 (verify from PDF) |
| **Venue** | Conference/Journal (TBD) |
| **Filename** | `Improved_Deep_Representation_Learning_for_Human_Activity_Recognition_using_IMU_Sensors.pdf` |

### 🎯 Problem Addressed
- **Core Problem:** Enhancing feature representation for IMU-based HAR
- **Expected Methods:**
  - Self-supervised pretraining
  - Contrastive learning
  - Attention mechanisms
  - Multi-scale feature extraction

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Training Labels | **Likely Yes** | Final classification requires labels |
| Pretraining | **Possibly No** | May use self-supervised pretraining |
| Type | **Check PDF** | Could be semi-supervised |

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### 🔍 Questions to EXTRACT from This Paper

1. **"Does this paper use self-supervised pretraining?"**
   - If yes: Can we use their pretraining strategy on unlabeled production data?
   
2. **"What representation learning techniques improve generalization?"**
   - Look for: Features that transfer across users/conditions

3. **"How does the deep representation handle sensor noise?"**
   - Look for: Robustness to real-world noise

4. **"What loss functions are used for representation learning?"**
   - Look for: Contrastive loss, reconstruction loss, etc.

#### ✅ Questions This Paper Might ANSWER

1. **"How to learn good features from IMU without labels?"**
   - Self-supervised methods may apply directly

2. **"What architecture modifications improve IMU representation?"**
   - Could enhance our 1D-CNN + BiLSTM

#### ❓ Questions RAISED but NOT ANSWERED (Expected)

1. **"How to adapt representations when distribution shifts?"**
2. **"How to validate representation quality without labels?"**

### 🔧 Pipeline Stages Affected

| Stage | Expected Impact | What to Look For |
|-------|-----------------|------------------|
| **Feature Learning** | ⭐⭐⭐ | Self-supervised objectives |
| **Model Architecture** | ⭐⭐⭐ | Improved CNN/LSTM design |
| **Pretraining** | ⭐⭐⭐ | Strategies for unlabeled data |

### 💡 Specific Techniques to Extract

| Technique | Application to Unlabeled Scenario |
|-----------|-----------------------------------|
| Self-supervised pretraining | ✅ Directly applicable |
| Contrastive learning | ✅ Can use on production data |
| Attention mechanisms | ⚠️ Requires labeled fine-tuning |
| Data augmentation | ✅ Enhances generalization |

---

## Paper 3: sensors-24-07975

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | sensors-24-07975 |
| **Year** | 2024 (from identifier) |
| **Venue** | MDPI Sensors (Open Access) |
| **Filename** | `sensors-24-07975.pdf` |

### 🎯 Expected Content
Based on MDPI Sensors journal pattern:
- HAR with wearable sensors
- Deep learning or ML approach
- Likely supervised evaluation
- May include novel architecture or dataset

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | MDPI papers typically supervised |

### ❓ KEY QUESTIONS to Extract

1. **Paper title and specific problem addressed**
2. **Whether any unsupervised/semi-supervised methods used**
3. **Sensor configuration (IMU channels used)**
4. **Evaluation protocol and metrics**
5. **Any discussion of production/deployment challenges**

### 🔧 Information to Capture

- [ ] Full paper title
- [ ] Authors and affiliations
- [ ] Dataset(s) used
- [ ] Model architecture
- [ ] Key results
- [ ] Limitations discussed
- [ ] Future work mentioned

---

## Paper 4: sensors-25-06988-v2

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | sensors-25-06988-v2 |
| **Year** | 2025 (from identifier) |
| **Venue** | MDPI Sensors (Open Access) |
| **Filename** | `sensors-25-06988-v2.pdf` |

### 🎯 Expected Content
- **Very recent paper (2025)** - likely contains latest techniques
- May include Transformer-based methods
- Potentially addresses recent challenges (federated learning, privacy, edge deployment)

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | Recent papers may include self-supervised |

### ❓ KEY QUESTIONS to Extract

1. **What novel 2025 techniques are introduced?**
2. **Any discussion of unlabeled data scenarios?**
3. **Edge/mobile deployment considerations?**
4. **State-of-the-art comparisons?**

### 💡 Why This Paper is Important
- **2025 publication** = current state-of-the-art
- May reference latest self-supervised/contrastive methods
- Could include production deployment insights

---

## Paper 5: journal.pone.0298888

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | journal.pone.0298888 |
| **Year** | 2024 (PLOS ONE format) |
| **Venue** | PLOS ONE (Open Access) |
| **Filename** | `journal.pone.0298888.pdf` |

### 🎯 Expected Content
PLOS ONE papers typically:
- Include detailed methodology
- Provide complete experimental details
- May focus on specific application domain
- Often include statistical validation

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | PLOS ONE varies widely |

### ❓ KEY QUESTIONS to Extract

1. **Specific HAR application domain (healthcare, fitness, etc.)?**
2. **Statistical methods used for validation?**
3. **Any novelty in handling real-world deployment?**
4. **Limitations acknowledged?**

---

## Paper 6: s00521-023-08863-9 (Neural Computing and Applications)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | s00521-023-08863-9 |
| **Year** | 2023 |
| **Venue** | Neural Computing and Applications (Springer) |
| **Filename** | `s00521-023-08863-9.pdf` |

### 🎯 Expected Content
Neural Computing and Applications typically publishes:
- Novel neural network architectures
- Deep learning applications
- Hybrid ML approaches

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Likely Supervised** | Standard neural network paper |

### ❓ KEY QUESTIONS to Extract

1. **Novel neural architecture proposed?**
2. **Comparison with CNN, LSTM baselines?**
3. **Handling of temporal dependencies in IMU data?**
4. **Any transfer learning or domain adaptation aspects?**

### 💡 Relevance Assessment

| Aspect | Priority |
|--------|----------|
| Architecture innovations | High |
| Training methodology | Medium |
| Unlabeled handling | Low (likely) |

---

## Paper 7: s11042-021-11219-x (Multimedia Tools and Applications)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | s11042-021-11219-x |
| **Year** | 2021 |
| **Venue** | Multimedia Tools and Applications (Springer) |
| **Filename** | `s11042-021-11219-x.pdf` |

### 🎯 Expected Content
This journal covers:
- Multimodal sensing
- Signal processing techniques
- Application-focused implementations

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | May include semi-supervised |

### ❓ KEY QUESTIONS to Extract

1. **Multimodal sensor fusion techniques?**
2. **Signal preprocessing methods?**
3. **Any handling of missing or noisy data?**
4. **Real-world data collection details?**

---

## Paper 8: s12652-020-02808-z (Journal of Ambient Intelligence and Humanized Computing)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | s12652-020-02808-z |
| **Year** | 2020 |
| **Venue** | Journal of Ambient Intelligence and Humanized Computing (Springer) |
| **Filename** | `s12652-020-02808-z.pdf` |

### 🎯 Expected Content
This journal focuses on:
- Ambient computing environments
- Human-centric AI
- Context-aware systems
- Smart environments

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | May address practical deployment |

### ❓ KEY QUESTIONS to Extract

1. **Context-aware activity recognition methods?**
2. **Ambient computing integration?**
3. **User adaptation techniques?**
4. **Privacy considerations?**

### 💡 Relevance to Unlabeled Scenario

| Aspect | Potential Relevance |
|--------|---------------------|
| Context awareness | ⭐⭐⭐ May help with activity segmentation |
| User adaptation | ⭐⭐⭐ Personalization without labels |
| Ambient sensing | ⭐⭐ Multi-sensor fusion |

---

## Paper 9: s13042-025-02569-1 (International Journal of Machine Learning and Cybernetics)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | s13042-025-02569-1 |
| **Year** | 2025 (Very Recent!) |
| **Venue** | International Journal of Machine Learning and Cybernetics (Springer) |
| **Filename** | `s13042-025-02569-1.pdf` |

### 🎯 Expected Content
**2025 publication** - likely includes:
- Latest ML techniques (Transformers, Foundation Models)
- Modern training strategies
- Current SOTA comparisons

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | 2025 papers often include self-supervised |

### ❓ KEY QUESTIONS to Extract

1. **What 2025 innovations are presented?**
2. **Any foundation model / large-scale pretraining?**
3. **Self-supervised or contrastive learning methods?**
4. **Zero-shot or few-shot learning approaches?**

### ⭐ Why This Paper is Critical
- **Most recent publication** in the list
- Likely contains latest techniques
- May reference solutions to unlabeled data problem
- Could include practical deployment insights

---

## Paper 10: s41598-025-02395-z (Scientific Reports)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Identifier** | s41598-025-02395-z |
| **Year** | 2025 (Very Recent!) |
| **Venue** | Scientific Reports (Nature Portfolio) |
| **Filename** | `s41598-025-02395-z.pdf` |

### 🎯 Expected Content
Scientific Reports is multidisciplinary:
- Rigorous methodology
- Reproducible experiments
- May include code/data availability
- Strong statistical validation

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Type | **Check PDF** | Nature journals require rigor |

### ❓ KEY QUESTIONS to Extract

1. **Specific HAR problem addressed?**
2. **Statistical validation methods?**
3. **Reproducibility (code/data available)?**
4. **Real-world evaluation included?**

### ⭐ Why This Paper is Critical
- **Nature Portfolio** = high quality standards
- **2025 publication** = current techniques
- May include practical validation
- Could have useful statistical methods

---

## 🎯 Cross-Paper Analysis: Key Themes for Unlabeled Production

### Theme 1: Architecture Choices for IMU Data

| Paper | Expected Contribution |
|-------|----------------------|
| Survey paper | Best practices overview |
| Deep Representation | Improved feature learning |
| 2025 papers | Latest architectures (Transformers?) |

**Questions for Our Thesis:**
1. Does 1D-CNN + BiLSTM remain competitive in 2025?
2. What architectural improvements can transfer to unlabeled setting?
3. Are attention mechanisms beneficial for IMU sequences?

### Theme 2: Self-Supervised / Unsupervised Methods

| Paper | Expected Content |
|-------|------------------|
| Deep Representation | Likely self-supervised pretraining |
| 2025 papers | Possibly contrastive/foundation models |

**Questions for Our Thesis:**
1. Which self-supervised objectives work best for IMU?
2. Can pretrained representations detect distribution shift?
3. How to validate self-supervised features without labels?

### Theme 3: Production Deployment Challenges

| Paper | Expected Coverage |
|-------|-------------------|
| Survey | Overview of challenges |
| Ambient Intelligence | Context-aware deployment |
| Scientific Reports | Rigorous real-world validation |

**Questions for Our Thesis:**
1. What production challenges are documented?
2. How do others handle evaluation without ground truth?
3. What monitoring strategies are recommended?

---

## 📋 Reading Checklist

### For Each Paper, Extract:

#### Mandatory Information
- [ ] Full title
- [ ] All authors
- [ ] Publication venue and year
- [ ] DOI

#### Technical Details
- [ ] Problem statement
- [ ] Dataset(s) used (check if publicly available)
- [ ] Model architecture
- [ ] Training methodology
- [ ] Evaluation metrics and protocol

#### Unlabeled Scenario Relevance
- [ ] Are labels required for training? (Yes/No/Partial)
- [ ] Any unsupervised/self-supervised components?
- [ ] Domain adaptation or transfer learning?
- [ ] Confidence estimation or uncertainty?
- [ ] Production deployment discussion?

#### Key Findings
- [ ] Main contributions
- [ ] Limitations acknowledged
- [ ] Future work suggested
- [ ] Code/data availability

---

## 🔄 Integration with Thesis Pipeline

### How These Papers Inform Each Pipeline Stage

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HAR PIPELINE STAGES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA COLLECTION                                                  │
│     └── Papers: Survey, Ambient Intelligence                         │
│         → Sensor placement best practices                            │
│         → Sampling rate recommendations                              │
│                                                                      │
│  2. PREPROCESSING                                                    │
│     └── Papers: Survey, Multimedia Tools                             │
│         → Window size selection                                      │
│         → Filtering and normalization                                │
│                                                                      │
│  3. FEATURE LEARNING                                                 │
│     └── Papers: Deep Representation, 2025 papers                     │
│         → Self-supervised pretraining                                │
│         → Representation quality                                     │
│                                                                      │
│  4. MODEL ARCHITECTURE                                               │
│     └── Papers: All papers                                           │
│         → CNN+LSTM validation                                        │
│         → Attention mechanisms                                       │
│                                                                      │
│  5. TRAINING                                                         │
│     └── Papers: Neural Computing, ML & Cybernetics                   │
│         → Training strategies                                        │
│         → Regularization                                             │
│                                                                      │
│  6. EVALUATION (CRITICAL FOR UNLABELED)                              │
│     └── Papers: Scientific Reports, PLOS ONE                         │
│         → Alternative metrics                                        │
│         → Statistical validation                                     │
│                                                                      │
│  7. DEPLOYMENT                                                       │
│     └── Papers: Ambient Intelligence, Sensors 2025                   │
│         → Edge deployment                                            │
│         → Real-time processing                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Summary: Expected Relevance Matrix

| Paper | Architecture | Self-Supervised | Deployment | Unlabeled Focus |
|-------|-------------|-----------------|------------|-----------------|
| Survey | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ |
| Deep Representation | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| sensors-24-07975 | ⭐⭐ | ? | ? | ? |
| sensors-25-06988 | ⭐⭐⭐ | ? | ? | ? |
| PLOS ONE | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ |
| Neural Comp & Apps | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ |
| Multimedia Tools | ⭐⭐ | ⭐ | ⭐ | ⭐ |
| Ambient Intelligence | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| ML & Cybernetics 2025 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Scientific Reports 2025 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Legend:** ⭐ = Low, ⭐⭐ = Medium, ⭐⭐⭐ = High, ? = Unknown (verify in PDF)

---

## 🚀 Priority Reading Order

For **unlabeled production scenario**, read in this order:

### Priority 1: Most Relevant (Read First)
1. **s13042-025-02569-1** (ML & Cybernetics 2025) - Latest techniques
2. **s41598-025-02395-z** (Scientific Reports 2025) - Rigorous methodology
3. **Improved Deep Representation Learning** - Self-supervised potential

### Priority 2: Foundation Papers
4. **A Survey on HAR** - Comprehensive background
5. **sensors-25-06988-v2** (Sensors 2025) - Recent sensor-focused

### Priority 3: Supplementary
6. **sensors-24-07975** - 2024 techniques
7. **s12652-020-02808-z** - Ambient/deployment focus
8. **journal.pone.0298888** - Detailed methodology
9. **s00521-023-08863-9** - Neural network innovations
10. **s11042-021-11219-x** - Signal processing aspects

---

## 📝 Notes Section

### Paper-by-Paper Notes (Fill while reading)

#### Paper 1: Survey on HAR
```
Title: 
Year: 
Key findings for unlabeled scenario:
- 
- 
Specific techniques mentioned:
- 
```

#### Paper 2: Deep Representation Learning
```
Title: 
Year: 
Self-supervised method used:
Key findings for unlabeled scenario:
- 
```

#### Papers 3-10: [Similar templates]
```
[Add notes as you read each paper]
```

---

## 🔗 Cross-References

### Related Analysis Documents
- [PSEUDO_LABELING_SELF_TRAINING_PAPERS_ANALYSIS.md](PSEUDO_LABELING_SELF_TRAINING_PAPERS_ANALYSIS.md) - Self-training methods
- [DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md](DOMAIN_ADAPTATION_PAPERS_ANALYSIS.md) - Transfer learning
- [UNCERTAINTY_CONFIDENCE_PAPERS_ANALYSIS.md](UNCERTAINTY_CONFIDENCE_PAPERS_ANALYSIS.md) - Confidence estimation
- [SENSOR_PLACEMENT_POSITION_PAPERS_ANALYSIS.md](SENSOR_PLACEMENT_POSITION_PAPERS_ANALYSIS.md) - Sensor considerations

---

*Last Updated: January 30, 2026*

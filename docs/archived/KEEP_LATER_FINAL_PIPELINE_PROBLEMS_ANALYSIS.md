> 📦 **ARCHIVED - KEEP FOR LATER**
> 
> **Reason:** Domain shift analysis - needed when we resume that work
> 
> **Why keeping:** Contains detailed root cause analysis (gravity signature, Az = -9.83 m/s²). Will be needed when implementing domain shift fixes after MLOps pipeline is complete.
> 
> **When to use:** Week 5+ when domain shift work resumes

---

# 🔍 Final Pipeline Problems Analysis

**Date:** December 9, 2025  
**Project:** Mental Health Activity Recognition - Master's Thesis  
**Status:** Critical Issues Identified with Root Causes Confirmed

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Background: What the Paper Claims](#background-what-the-paper-claims)
3. [Our Implementation](#our-implementation)
4. [The Critical Discovery](#the-critical-discovery)
5. [Root Cause Analysis](#root-cause-analysis)
6. [Evidence and Data](#evidence-and-data)
7. [Summary of Problems](#summary-of-problems)
8. [Implications for MLOps Pipeline](#implications-for-mlops-pipeline)

---

## 📌 EXECUTIVE SUMMARY

### The Problem in One Sentence

**Our model predicts "hand_tapping" for 100% of production data because the production data's gravity signature (-9.83 m/s²) matches the training pattern for hand_tapping (-8.85 m/s²) more closely than any other activity.**

### Key Numbers

| Metric | Paper Claim | Our Reality |
|--------|-------------|-------------|
| Model Accuracy | 87.0% ± 1.2% | Unknown (no labels) |
| Prediction Diversity | 11 classes | 1 class (hand_tapping) |
| Training Az Mean | -3.53 m/s² | Same |
| Production Az Mean | N/A | -9.83 m/s² (gravity) |
| Closest Activity to -9.8 | hand_tapping (-8.85) | Model predicts this! |

---

## 📚 BACKGROUND: WHAT THE PAPER CLAIMS

### Paper: ICTH_16 - "Recognition of Anxiety-Related Activities using 1DCNNBiLSTM"

**Two-Stage Training Approach:**

1. **Stage 1: Pre-training on ADAMSense Dataset**
   - Public research-grade IMU dataset
   - Wrist + chest + pocket sensors (25 features)
   - Reduced to 6 features (wrist accelerometer + gyroscope only)
   - Achieved 89.11% accuracy on this subset

2. **Stage 2: Fine-tuning on Garmin Venu 3 Dataset**
   - Custom collected from 6 volunteers
   - Wrist-worn Garmin Venu 3 smartwatch
   - 6 features: Ax, Ay, Az (accelerometer) + Gx, Gy, Gz (gyroscope)
   - 11 anxiety-related activities

**Paper's Reported Results:**

| Scenario | Accuracy |
|----------|----------|
| Full ADAMSense dataset | 98.8% |
| ADAMSense wrist-only (6 features) | 89.11% |
| Base model on Garmin (NO fine-tuning) | **48.7%** |
| After fine-tuning (5-fold CV) | **87.0% ± 1.2%** |

### Critical Detail: How the 87% Was Achieved

The paper used **5-fold cross-validation on the same 6 volunteers**:
- Split data from 6 volunteers into 5 folds
- Train on 4 folds, test on 1 fold
- Repeat 5 times, each fold as test once
- **Mean accuracy across folds = 87%**

**This is NOT testing on new users - it's testing on held-out data from the SAME users!**

---

## 🔧 OUR IMPLEMENTATION

### Training Data: `all_users_data_labeled.csv`

This appears to be the **Garmin Venu 3 fine-tuning dataset** from the paper:

```
Source:  data/raw/all_users_data_labeled.csv
Rows:    385,326 samples
Users:   6 (matching paper's 6 volunteers)
Columns: timestamp, Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, activity, User
Classes: 11 anxiety-related activities
```

**User Distribution:**
```
User 1:  60,712 samples
User 2:  68,352 samples
User 3:  61,613 samples
User 4:  63,034 samples
User 5:  64,189 samples
User 6:  67,426 samples
```

### Our Preprocessing Configuration

From `data/prepared/config.json`:

```json
{
  "train_users": [1, 2, 3, 4],
  "val_users": [5],
  "test_users": [6],
  "window_size": 200,
  "overlap": 0.5,
  "scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
  "scaler_scale": [6.57, 4.35, 3.24, 49.93, 14.81, 14.17]
}
```

**Key Observation:** We split by USER, not by fold. This means:
- Training: Users 1-4
- Validation: User 5
- Test: User 6

### Production Data: Garmin Venu 3 (New Collection)

```
Source:  data/processed/sensor_fused_50Hz.csv (raw, milliG)
         data/processed/sensor_fused_50Hz_converted.csv (converted to m/s²)
Rows:    181,699 samples
Date:    March 24, 2025
User:    NOT one of the 6 training volunteers
```

---

## 🔬 THE CRITICAL DISCOVERY

### Discovery 1: Activity-Specific Gravity Signatures

Each activity in the training data has a **distinct mean Az value**:

| Activity | Az Mean (m/s²) | Normalized Az | Distance from Production |
|----------|----------------|---------------|-------------------------|
| **hand_tapping** | **-8.85** | **-1.646** | **0.300** ← CLOSEST! |
| sitting | -6.00 | -0.765 | 1.181 |
| knuckles_cracking | -4.81 | -0.396 | 1.550 |
| nape_rubbing | -4.72 | -0.369 | 1.577 |
| hand_scratching | -3.16 | 0.114 | 2.060 |
| standing | -2.63 | 0.277 | 2.223 |
| nail_biting | -2.09 | 0.444 | 2.390 |
| hair_pulling | -1.97 | 0.482 | 2.428 |
| ear_rubbing | -1.94 | 0.490 | 2.436 |
| forehead_rubbing | -1.89 | 0.506 | 2.451 |
| smoking | -1.30 | 0.690 | 2.636 |

**Production normalized Az = -1.946 (distance measured from this)**

### Discovery 2: Production Data Has Full Gravity

Our production data after unit conversion:

```
Az mean: -9.83 m/s² (essentially Earth's gravity)
Az std:   0.20 m/s² (very low variance)
```

### Discovery 3: The Model's "Logic"

When the model sees production data with normalized Az ≈ -1.95:

| Activity | Normalized Az | Distance from -1.95 |
|----------|---------------|---------------------|
| **hand_tapping** | -1.646 | **0.30** ← WINNER |
| sitting | -0.765 | 1.18 |
| knuckles_cracking | -0.396 | 1.55 |
| nape_rubbing | -0.369 | 1.58 |
| hand_scratching | 0.114 | 2.06 |
| ... | ... | ... |
| smoking | 0.690 | 2.64 ← FARTHEST |

**Hand_tapping is 4x closer to production data than the next closest activity!**

The model correctly identifies hand_tapping as the closest match - this is NOT a bug!

---

## 🔍 ROOT CAUSE ANALYSIS

### Root Cause #1: Different Device Orientations

**Training Data (6 Volunteers):**
- Activities performed in controlled lab setting
- Various arm positions during activities
- Az varies significantly: -1.30 to -8.85 m/s² depending on activity
- **Average Az = -3.53 m/s²** (wrist not always gravity-aligned)

**Production Data (Real-World):**
- Natural device wearing position
- User's wrist predominantly in gravity-aligned position
- **Az = -9.83 m/s²** (constant, full gravity)

### Root Cause #2: StandardScaler Amplifies the Difference

The scaler was fitted on training data (Az mean = -3.53, scale = 3.24):

```python
# How StandardScaler normalizes:
normalized_Az = (raw_Az - mean) / scale

# For training data:
normalized_Az = (-3.53 - (-3.53)) / 3.24 = 0.0  # Centered

# For production data:
normalized_Az = (-9.83 - (-3.53)) / 3.24 = -1.95  # Shifted 2 std devs!
```

**Production Az is always -1.95 standard deviations below training mean!**

### Root Cause #3: Az is the Dominant Feature

In the training data:
- **hand_tapping** has the most negative Az (-8.85 m/s²)
- When normalized, hand_tapping Az ≈ -1.64σ
- Production data normalized Az ≈ -1.95σ
- **Hand_tapping is the closest pattern!**

### Root Cause #4: No Cross-User Generalization

The paper's 87% accuracy was achieved with:
- 5-fold CV on **same 6 volunteers**
- Each person's data appears in both training and testing (different folds)

We are testing on:
- A **completely new person** (not in the 6 volunteers)
- Different movement patterns
- Different device orientation habits
- **This is cross-user generalization, which the paper didn't validate!**

---

## 📊 EVIDENCE AND DATA

### Evidence 1: Training Data Statistics

```
All Users Az: mean = -3.53 m/s², std = 3.24 m/s²

By User:
  User 1: Az mean = -4.08 m/s²
  User 2: Az mean = -2.98 m/s²
  User 3: Az mean = -3.22 m/s²
  User 4: Az mean = -4.15 m/s²
  User 5: Az mean = -4.19 m/s²
  User 6: Az mean = -2.66 m/s²
```

### Evidence 2: Production Data Statistics

**Raw (milliG):**
```
Az mean: -1001.56 milliG (≈ -1g)
Az std:  19.92 milliG
```

**After Conversion (m/s²):**
```
Az mean: -9.83 m/s² (gravity!)
Az std:  0.20 m/s²
```

**After StandardScaler Normalization:**
```
Az mean: -1.95 (normalized units)
Az std:  0.06 (very narrow!)
```

### Evidence 3: Inference Results

```
Total Windows:     1,772
Predictions:       100% hand_tapping
Mean Confidence:   93.4%
Activity Transitions: 0

This is physically impossible for 2 hours of human activity!
```

### Evidence 4: Paper's Own Warning

The paper explicitly states:
> "The challenge of cross-user scenarios, where data distribution differences exist between training and real-world usage, is also actively addressed by domain adaptation techniques."

And their baseline without fine-tuning:
> "Without any fine-tuning, the model performed poorly, achieving an accuracy of only **48.7%**"

---

## 📋 SUMMARY OF PROBLEMS

### Problem #1: Gravity Distribution Mismatch (CRITICAL)

| Aspect | Training | Production |
|--------|----------|------------|
| Az mean | -3.53 m/s² | -9.83 m/s² |
| Az range | -45 to +24 | -9.5 to -10.1 |
| Az variance | High (activities vary) | Very low (constant gravity) |
| Device orientation | Varied by activity | Predominantly gravity-aligned |

**Impact:** Production Az (-9.83) matches hand_tapping pattern (-8.85) best.

### Problem #2: Cross-User Generalization Not Tested

| Paper's Evaluation | Our Evaluation |
|--------------------|----------------|
| 5-fold CV on 6 users | New user entirely |
| Same users in train+test | Train users ≠ test user |
| 87% accuracy | Unknown (likely much lower) |

**Impact:** Model doesn't generalize to new users.

### Problem #3: StandardScaler Bias

```
Training Az normalized: mean = 0.0, std = 1.0
Production Az normalized: mean = -1.95, std = 0.06
```

**Impact:** All production windows have nearly identical Az, making them indistinguishable.

### Problem #4: Device Orientation During Collection

| Training Collection | Production Collection |
|--------------------|----------------------|
| Controlled lab | Real-world free-living |
| Prompted activities | Natural behavior |
| Varied orientations | Consistent orientation |
| 6 volunteers | 1 user |

**Impact:** Training conditions don't match deployment conditions.

### Problem #5: Model Overconfidence

```
Mean confidence: 93.4%
High confidence (>90%): 99.6% of predictions
```

The model is **extremely confident** in wrong predictions because:
- Neural networks don't know what they don't know
- Production data is out-of-distribution
- Model still outputs high softmax probabilities

---

## 🏗️ IMPLICATIONS FOR MLOPS PIPELINE

### Current Pipeline Status

```
[Data Collection] ✅ Working (Garmin → CSV)
        ↓
[Unit Conversion] ✅ Working (milliG → m/s²)
        ↓
[Normalization]   ⚠️ PROBLEM (Scaler from different distribution)
        ↓
[Windowing]       ✅ Working (200 samples, 50% overlap)
        ↓
[Model Inference] ⚠️ PROBLEM (Model sees OOD data)
        ↓
[Predictions]     ❌ FAILED (100% same class)
```

### What This Means for Your Thesis

1. **The model architecture is sound** - 1D-CNN-BiLSTM works well (87% in paper)
2. **The preprocessing is correct** - No bugs in code
3. **The problem is domain shift** - Training ≠ production distribution
4. **This is a known research problem** - Paper explicitly mentions it

### Questions to Address

1. **Is the production user wearing the watch differently?**
   - Training users had varied orientations during activities
   - Production user has watch in consistent gravity-aligned position

2. **Do we need user-specific calibration?**
   - Paper's 87% was within-user (same person in train/test)
   - Cross-user performance was never validated

3. **Should we fine-tune on production user's data?**
   - Paper shows fine-tuning is critical (48.7% → 87%)
   - Same approach could work for new users

4. **Can we adjust for gravity offset?**
   - Production Az is ~6.3 m/s² more negative than training mean
   - Could normalize/shift production data to match training distribution

---

## 🎯 CONCLUSION

### Why Paper Got 87% But We Get 100% Single Class

| Factor | Paper | Us |
|--------|-------|-----|
| Evaluation type | Within-user (5-fold CV) | Cross-user (new person) |
| Device orientation | Varied during activities | Consistent (gravity-aligned) |
| Az distribution | Varies by activity (-1 to -9) | Constant (~-9.8) |
| Domain | Same as training | Different from training |

### The Model is Not Broken

The model correctly learned:
- **hand_tapping → wrist pointing down → Az ≈ -8.85 m/s²**
- **other activities → wrist at various angles → Az varies more**

When it sees production data (Az ≈ -9.8):
- Closest match = hand_tapping
- High confidence because pattern is clear
- **This is correct behavior for the model's training!**

### The Real Problem

**Domain shift between training and production:**
1. Different users (cross-user generalization)
2. Different device orientations (lab vs real-world)
3. Different activity execution (prompted vs natural)

---

## 🆕 NEW FINDING: Pre-training vs Fine-tuning Dataset Mismatch (December 9)

### Dataset Comparison: ADAMSense vs Garmin

We compared the ADAMSense (pre-training) dataset with all_users_data_labeled (Garmin fine-tuning):

| Feature | ADAMSense (Pre-training) | Garmin (Fine-tuning) |
|---------|--------------------------|----------------------|
| Samples | 709,582 | 385,326 |
| Users | 10 (IDs: 2,3,5,6,7,8,10,14,15,16) | 6 (IDs: 1,2,3,4,5,6) |
| Device | Samsung Frontier 1 & 2 | Garmin Venu 3 |
| **Sensor Units** | **Normalized ±2g** (capped) | **Raw m/s²** (~±45) |
| Az_w mean | -0.46 g | -3.53 m/s² |
| Az_w range | [-2.0, +2.0] | [-45.2, +24.2] |

### 🔴 CRITICAL: Sensor Unit Mismatch!

**The pre-trained model was trained on NORMALIZED ±2g data, but fine-tuned on RAW m/s² data!**

This means:
1. **Pre-training:** Model learned patterns from [-2, +2] range (normalized gravity units)
2. **Fine-tuning:** Model had to adapt to [-45, +45] range (raw m/s²)
3. The scale difference is ~10x (1g ≈ 9.8 m/s²)

**This is a massive domain shift between pre-training and fine-tuning!**

### Activity Patterns Preserved

Both datasets show hand_tapping as the activity with most negative Az:
- ADAMSense: hand_tapping = -0.91g (most negative)
- Garmin: hand_tapping = -8.85 m/s² (most negative)
- Our production: Az = -9.83 m/s² → matches hand_tapping pattern

### Implications

1. The pre-trained model's learned patterns may have been "forgotten" during fine-tuning
2. The paper's 48.7% without fine-tuning (Samsung→Garmin) shows the domain shift
3. Our case is Samsung→Garmin→Production = **double domain shift**

---

## 📚 REFERENCES

1. **ICTH_16.pdf** - "Recognition of Anxiety-Related Activities using 1DCNNBiLSTM on Sensor Data from a Commercial Wearable Device"
   - Location: `research_papers/ICTH_16.pdf`
   - Key findings used in this analysis

2. **EHB_2025_71.pdf** - "RAG-Enhanced Pipeline for Mental Health Reports"
   - Location: `research_papers/EHB_2025_71.pdf`
   - Uses same HAR model

3. **Data Files Analyzed:**
   - `data/raw/all_users_data_labeled.csv` - Training data (6 users)
   - `data/processed/sensor_fused_50Hz.csv` - Production data (raw)
   - `data/processed/sensor_fused_50Hz_converted.csv` - Production data (converted)
   - `data/prepared/production_X.npy` - Preprocessed production windows
   - `data/prepared/config.json` - Scaler configuration
   - `research_papers/anxiety_dataset.csv` - ADAMSense dataset (pre-training data)

---

**Document Status:** FINAL  
**Analysis Confidence:** HIGH (based on actual data analysis and paper review)  
**Last Updated:** December 9, 2025 (Added ADAMSense comparison)

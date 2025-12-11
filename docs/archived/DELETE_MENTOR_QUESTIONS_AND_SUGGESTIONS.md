> ⚠️ **ARCHIVED - SAFE TO DELETE**
> 
> **Reason:** Questions answered by mentor
> 
> **Why not needed:** Mentor responded on Dec 11: Focus on MLOps, defer domain shift fixes.

---

# 📋 Mentor Discussion: Questions & Suggestions

**Date:** December 9, 2025  
**Context:** Critical issue discovered in HAR model inference pipeline  
**Purpose:** Guidance needed before proceeding with MLOps implementation

---

## 📌 EXECUTIVE SUMMARY FOR MENTOR

We discovered that our production inference returns 100% "hand_tapping" predictions because:
1. Production data has consistent gravity signature (Az = -9.83 m/s²)
2. In training data, hand_tapping has Az = -8.85 m/s² (closest to production)
3. The model correctly identifies the closest pattern - but this reveals a **domain shift problem**

**Core Question:** How should we proceed with the MLOps pipeline given this fundamental HAR model limitation?

---

## ❓ QUESTIONS FOR MENTOR

### Category 1: Understanding the Training Data

#### Q1.1: Is the training data (all_users_data_labeled.csv) the Garmin fine-tuning dataset from the ICTH paper?
**Why asking:** 
- The file has 6 users (matching the paper's 6 volunteers)
- Column names suggest wrist-mounted sensors (Ax_w, Ay_w, Az_w)
- Need to confirm this IS the fine-tuned Garmin dataset, not ADAMSense

**Impact on thesis:** Clarifies data provenance and helps explain the domain shift

---

#### Q1.2: Why does the training data have Az mean = -3.53 m/s² instead of -9.8 m/s² (gravity)?
**Why asking:**
- If gravity is -9.8 m/s², the training Az should be around that value
- Current training Az suggests either:
  - Activities performed with arm NOT pointing down
  - Some gravity compensation applied
  - Different device orientation during collection

**Possible explanations to verify:**
1. Volunteers performed activities with varied arm positions
2. Data collection protocol specified certain postures
3. Some preprocessing removed gravity component

**Impact on thesis:** Critical for understanding why production data doesn't match training distribution

---

#### Q1.3: During training data collection, how was the Garmin device oriented?
**Why asking:**
- Az varies significantly by activity (-1.30 to -8.85 m/s²)
- This suggests different arm positions during each activity
- Production user may be wearing device differently

**Specific concerns:**
- Was there a protocol for device placement?
- Were volunteers instructed on arm position?
- Is there video documentation of the data collection?

---

### Category 2: Model & Methodology Clarification

#### Q2.1: The paper reports 87% accuracy with 5-fold CV. Was this cross-user or within-user?
**Why asking:**
- 5-fold CV on 6 users could be:
  - **Within-user:** Each fold contains data from all 6 users (some samples held out)
  - **Cross-user:** Each fold is one user (leave-one-user-out)
- Paper text suggests within-user (same users in train and test)

**Impact on thesis:** 
- If within-user: 87% doesn't guarantee cross-user performance
- Our 14.5% accuracy (User 6 test) confirms cross-user gap exists

---

#### Q2.2: Is there a pre-trained model that was trained on ADAMSense (before Garmin fine-tuning)?
**Why asking:**
- Paper mentions two-stage training:
  1. Pre-train on ADAMSense (research-grade data)
  2. Fine-tune on Garmin data
- We only have `fine_tuned_model_1dcnnbilstm.keras`
- The base model might generalize better

**Request:** Can we access the base model (before fine-tuning)?

---

#### Q2.3: What normalization was applied during model training?
**Why asking:**
- Our `config.json` shows StandardScaler with:
  - Mean: [3.22, 1.28, -3.53, 0.60, 0.23, 0.09]
  - Scale: [6.57, 4.35, 3.24, 49.93, 14.81, 14.17]
- Was this the same scaler used in the original training?
- Was the scaler fitted on all 6 users or only train users (1-4)?

**Impact on thesis:** Scaler mismatch could be amplifying domain shift

---

### Category 3: Production Data & Deployment

#### Q3.1: Is the production data collected from the same Garmin Venu 3 device?
**Why asking:**
- Paper used Garmin Venu 3 for fine-tuning
- Our production data shows:
  - Raw Az = -1001 milliG → converted to -9.83 m/s²
  - This is correct gravity value
- Need to confirm device consistency

---

#### Q3.2: What activities was the production user performing during data collection?
**Why asking:**
- We have no labels for production data
- If user was mainly sedentary (sitting at desk), Az ≈ -9.8 makes sense
- If user was performing the 11 target activities, we'd expect more variation

**Impact on thesis:** Helps determine if prediction is reasonable or not

---

#### Q3.3: Should we collect labeled calibration data from the production user?
**Why asking:**
- Paper shows fine-tuning is essential (48.7% → 87%)
- A few minutes of labeled data from production user could enable:
  - Per-user fine-tuning
  - User-specific scaler fitting
  - Baseline establishment

---

### Category 4: MLOps Pipeline Direction

#### Q4.1: Given the domain shift issue, should we still proceed with MLOps implementation?
**Why asking:**
- MLOps (DVC, MLflow, Docker, API) is valuable regardless of model performance
- But deploying a non-functional model isn't useful
- Need to decide: Fix model first, or build MLOps around current state?

**Options to discuss:**
1. **Option A:** Proceed with MLOps using current model (demonstrate infrastructure)
2. **Option B:** Fix domain shift first, then add MLOps
3. **Option C:** Build MLOps with placeholder/dummy model, swap later

---

#### Q4.2: Is the goal to demonstrate MLOps practices, or to achieve high HAR accuracy?
**Why asking:**
- Master's thesis focus: MLOps for HAR pipeline
- The HAR model came from prior work (ICTH paper)
- Should thesis success depend on model accuracy?

**Suggested thesis scope:**
- Document the domain shift as a real-world MLOps challenge
- Show how MLOps practices can help detect/monitor such issues
- Propose solutions even if not fully implemented

---

#### Q4.3: Can we use synthetic or simulated data to demonstrate the MLOps pipeline?
**Why asking:**
- Real production data has domain shift issues
- To demonstrate MLOps tools (DVC versioning, MLflow tracking, API serving), we need:
  - Training data
  - Test data (with labels)
  - Production data

**Suggestion:** 
- Use training data (users 1-5) for training
- Use user 6 data (labeled) as "production simulation"
- This gives labeled production data for proper evaluation

---

### Category 5: Thesis & Documentation

#### Q5.1: How should we document the domain shift discovery in the thesis?
**Why asking:**
- This is a significant finding about real-world ML deployment
- Shows the "lab-to-life" gap the paper mentions
- Could be a valuable contribution to the thesis

**Suggested sections:**
1. Background: Paper's claims (87% accuracy)
2. Discovery: Production inference results (100% single class)
3. Analysis: Root cause investigation
4. Discussion: Implications for MLOps

---

#### Q5.2: Should we contact the original paper authors for clarification?
**Why asking:**
- Paper authors (Ugonna Oleh, Roman Obermaisser) are from University of Siegen
- They may have additional documentation or data
- Could clarify training data preprocessing

---

## 💡 SUGGESTIONS & POTENTIAL SOLUTIONS

### Solution 1: Gravity-Invariant Normalization

**Problem:** Production Az = -9.8 m/s² but training Az mean = -3.53 m/s²

**Proposed Solution:**
```python
# Instead of StandardScaler, use activity-relative normalization
# Remove gravity component before scaling
Az_dynamic = Az_raw - (-9.8)  # Remove gravity
# Now Az_dynamic represents only motion-induced acceleration
```

**Pros:**
- Makes model invariant to device orientation
- Aligns production data with training distribution

**Cons:**
- Requires knowing which axis points "down" (may vary)
- Training data may already have different gravity baseline

**Recommendation:** Test this approach and compare results

---

### Solution 2: Per-User Calibration Step

**Problem:** Cross-user generalization is poor

**Proposed Solution:**
```python
# During deployment, collect 1-2 minutes of calibration data
# User performs known activities (e.g., "stand still", "tap hand")
# Fit user-specific scaler on this calibration data

def calibrate_user(calibration_data):
    user_scaler = StandardScaler()
    user_scaler.fit(calibration_data)
    return user_scaler
```

**Pros:**
- Adapts to individual user's movement patterns
- Captures user-specific device orientation

**Cons:**
- Requires labeled calibration session
- Adds friction to deployment

**Recommendation:** Implement as optional calibration step in pipeline

---

### Solution 3: Feature Engineering for Orientation Invariance

**Problem:** Model learned orientation-specific patterns

**Proposed Solution:**
```python
# Compute orientation-invariant features
def compute_invariant_features(Ax, Ay, Az):
    # Magnitude (invariant to rotation)
    magnitude = np.sqrt(Ax**2 + Ay**2 + Az**2)
    
    # Jerk (derivative of acceleration)
    jerk = np.diff(magnitude)
    
    # Signal energy
    energy = np.mean(magnitude**2)
    
    return magnitude, jerk, energy
```

**Pros:**
- Magnitude is rotation-invariant
- Captures motion intensity regardless of orientation

**Cons:**
- Requires retraining the model
- May lose some discriminative information

**Recommendation:** Add as additional features, don't replace raw accelerometer

---

### Solution 4: Domain Adaptation / Transfer Learning

**Problem:** Training and production data from different distributions

**Proposed Solution:**
```python
# Fine-tune model on small amount of production data
# Use techniques from paper's domain adaptation approach

# Option 1: Few-shot learning
# Collect 10-20 labeled samples per activity from production user
# Fine-tune final layers only

# Option 2: Unsupervised domain adaptation
# Use unlabeled production data to adapt feature distributions
```

**Pros:**
- Directly addresses the domain shift
- Paper already validated this approach (48.7% → 87%)

**Cons:**
- Requires some labeled production data
- Computational cost for fine-tuning

**Recommendation:** This is the paper's recommended approach - should discuss with mentor

---

### Solution 5: MLOps-Based Monitoring & Detection

**Problem:** Domain shift went undetected until manual inspection

**Proposed Solution:**
Build MLOps infrastructure that automatically detects issues:

```python
# Data drift detection
def detect_drift(production_data, training_stats):
    prod_mean = production_data.mean(axis=0)
    train_mean = training_stats['mean']
    train_std = training_stats['std']
    
    z_scores = (prod_mean - train_mean) / train_std
    
    if any(abs(z_scores) > 2.0):
        alert("Data drift detected!")
        return True
    return False

# Prediction monitoring
def monitor_predictions(predictions):
    class_distribution = predictions.value_counts(normalize=True)
    
    if class_distribution.max() > 0.5:  # >50% single class
        alert("Prediction imbalance detected!")
        return True
    return False
```

**Pros:**
- Catches issues automatically
- Core MLOps practice
- Valuable thesis contribution

**Cons:**
- Doesn't fix the model
- Requires defining thresholds

**Recommendation:** Implement this regardless - shows MLOps value

---

### Solution 6: Use Training User 6 as Production Simulation

**Problem:** Production data has no labels, can't properly evaluate

**Proposed Solution:**
```python
# Current split: train=1-4, val=5, test=6
# Use User 6 data as "production data" simulation

# This gives us:
# - Labeled "production" data
# - Ability to compute real accuracy
# - Known ground truth for debugging
```

**Pros:**
- Immediate solution for thesis work
- Enables proper evaluation
- No new data collection needed

**Cons:**
- User 6 is from same collection session as training
- Not true "production" scenario

**Recommendation:** Good for thesis demonstration, but document as limitation

---

### Solution 7: Retrain with Augmented Data

**Problem:** Training data doesn't cover production orientation

**Proposed Solution:**
```python
# Apply rotation augmentation during training
def rotate_accelerometer(Ax, Ay, Az, angles):
    """Rotate accelerometer readings to simulate different orientations"""
    # Apply rotation matrices
    # This teaches model to handle various device orientations
    pass

# Augment training data with multiple orientations
# Include samples with Az near -9.8 (gravity-aligned)
```

**Pros:**
- Makes model robust to orientation changes
- Standard data augmentation technique

**Cons:**
- Requires retraining
- May need careful implementation

**Recommendation:** Discuss with mentor - may be out of thesis scope

---

## 📊 SUMMARY TABLE

### Questions Priority

| Priority | Question | Why Critical |
|----------|----------|--------------|
| 🔴 HIGH | Q2.1 (Cross-user vs within-user CV) | Explains accuracy gap |
| 🔴 HIGH | Q4.1 (Proceed with MLOps?) | Determines thesis direction |
| 🟠 MED | Q1.2 (Why Az = -3.53?) | Root cause clarification |
| 🟠 MED | Q3.2 (Production user activities) | Validates predictions |
| 🟡 LOW | Q2.2 (Base model available?) | Potential solution |
| 🟡 LOW | Q5.2 (Contact authors?) | Additional resources |

### Solution Priority

| Priority | Solution | Effort | Impact |
|----------|----------|--------|--------|
| 🔴 HIGH | Solution 5 (MLOps Monitoring) | Medium | High for thesis |
| 🔴 HIGH | Solution 6 (User 6 simulation) | Low | High for thesis |
| 🟠 MED | Solution 2 (Per-user calibration) | Medium | High for real-world |
| 🟠 MED | Solution 4 (Domain adaptation) | High | High for accuracy |
| 🟡 LOW | Solution 1 (Gravity normalization) | Low | Medium |
| 🟡 LOW | Solution 3 (Feature engineering) | High | Medium |

---

## 🎯 RECOMMENDED NEXT STEPS

### Immediate (Before Mentor Meeting)

1. ✅ Document the problem thoroughly (DONE - FINAL_PIPELINE_PROBLEMS_ANALYSIS.md)
2. ✅ Prepare questions and suggestions (DONE - this document)
3. ⏳ Test Solution 6 (User 6 as production simulation)
4. ⏳ Implement basic drift detection (Solution 5)

### After Mentor Guidance

1. Decide on thesis scope (model accuracy vs MLOps demonstration)
2. Choose which solutions to implement
3. Proceed with MLOps infrastructure
4. Document findings as thesis contribution

---

## 📝 MEETING AGENDA SUGGESTION

1. **Problem Overview** (5 min)
   - Show FINAL_PIPELINE_PROBLEMS_ANALYSIS.md
   - Explain 100% hand_tapping finding

2. **Root Cause Discussion** (10 min)
   - Gravity/orientation mismatch
   - Cross-user generalization issue

3. **Questions** (15 min)
   - Priority questions from this document
   - Clarification on training data

4. **Solutions Discussion** (15 min)
   - Review proposed solutions
   - Decide on thesis direction

5. **Next Steps** (5 min)
   - Action items
   - Timeline

---

## 🆕 NEW FINDING: ADAMSense vs Garmin Dataset Comparison (December 9, 2025)

### Dataset Comparison Summary

We compared the ADAMSense (pre-training) dataset with all_users_data_labeled (Garmin fine-tuning) dataset:

| Feature | ADAMSense (Pre-training) | all_users_data_labeled (Garmin Fine-tuning) |
|---------|--------------------------|---------------------------------------------|
| **Samples** | 709,582 | 385,326 |
| **Users** | 10 users (IDs: 2,3,5,6,7,8,10,14,15,16) | 6 users (IDs: 1,2,3,4,5,6) |
| **Device** | Samsung Frontier 1 & 2 smartwatches | Garmin Venu 3 smartwatch |
| **Sensor Units** | **Normalized ±2g** (gravity units, capped) | **Raw m/s²** (meters per second squared) |
| **Az_w range** | [-2.00, +2.00] g | [-45.23, +24.15] m/s² |
| **Az_w mean** | -0.46 g | -3.53 m/s² |
| **Columns** | 26 (wrist+pocket sensors, GPS, magnetometer) | 9 (wrist sensors only) |

### 🔴 CRITICAL NEW INSIGHTS

#### 1. SENSOR UNIT MISMATCH BETWEEN PRE-TRAINING AND FINE-TUNING!
- **ADAMSense:** Normalized to ±2g range (capped, in gravity units)
- **Garmin:** Raw m/s² units (not capped, ~10x larger scale)
- **Impact:** The pre-trained model learned patterns from [-2, +2] range, then had to adapt to [-45, +45] range during fine-tuning!

#### 2. DIFFERENT SENSOR HARDWARE
- **ADAMSense:** Samsung Frontier (SF1/SF2) smartwatches
- **Garmin:** Garmin Venu 3 smartwatch  
- **Impact:** Different sensor noise characteristics, sampling methods

#### 3. COMPLETELY DIFFERENT USER POOLS
- **ADAMSense:** Research subjects (numbered 2-16)
- **Garmin:** New volunteers (numbered 1-6)
- **Impact:** Zero user overlap between pre-training and fine-tuning = double domain shift

#### 4. ACTIVITY RANKING PRESERVED BUT SCALE DIFFERENT
Both datasets show **hand_tapping** as the activity with **most negative Az**:
- ADAMSense: hand_tapping = -0.91g (most negative)
- Garmin: hand_tapping = -8.85 m/s² (most negative)

This confirms the gravity-based activity signatures are consistent across devices, but the scale differs.

### 📋 NEW QUESTIONS FOR MENTOR

#### Q6.1: Was there any normalization applied when transitioning from ADAMSense to Garmin fine-tuning?
**Why asking:**
- ADAMSense uses normalized g units (±2 range)
- Garmin uses raw m/s² (~10x larger range)
- Was the Garmin data normalized to match ADAMSense scale before fine-tuning?

#### Q6.2: Why are the sensor units different between datasets?
**Why asking:**
- This could be a significant source of domain shift
- If pre-trained model expected ±2 range but got ±45 range, the learned patterns may not transfer

#### Q6.3: Should we normalize our production data to the same scale as ADAMSense?
**Why asking:**
- If original pre-training used ±2g normalized data
- And fine-tuning "forgot" those patterns due to scale mismatch
- Perhaps using ADAMSense scale could help?

### 💡 NEW SOLUTION SUGGESTION

#### Solution 8: Normalize to ADAMSense Scale

```python
# Convert Garmin/Production data to match ADAMSense scale
def normalize_to_adamsense_scale(acc_data):
    # ADAMSense uses ±2g scale (capped)
    # 1g = 9.8 m/s²
    G = 9.8
    
    # Convert m/s² to g
    acc_g = acc_data / G
    
    # Cap to ±2g (as ADAMSense does)
    acc_capped = np.clip(acc_g, -2.0, 2.0)
    
    return acc_capped
```

**Pros:**
- Aligns with original pre-training data distribution
- Simple transformation

**Cons:**
- May lose information if values exceed ±2g
- Requires testing to validate

---

**Document Created:** December 9, 2025  
**Document Updated:** December 9, 2025 (Added ADAMSense comparison)
**Author:** Shalin (with AI assistance)  
**Purpose:** Mentor meeting preparation  
**Related Documents:** 
- `docs/FINAL_PIPELINE_PROBLEMS_ANALYSIS.md`
- `research_papers/ICTH_16.pdf`
- `research_papers/EHB_2025_71.pdf`
- `research_papers/temp.ipynb` (Dataset comparison analysis)

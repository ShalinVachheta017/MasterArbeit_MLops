# 🚨 CRITICAL: Model Performance Issue Detected

**Date:** November 5, 2025  
**Status:** **BLOCKING** - Must resolve before proceeding  
**Priority:** **URGENT**

---

## THE PROBLEM

### Cross-User Evaluation Results:

```
User 1: 15.62% accuracy (608 windows)
User 2:  8.49% accuracy (683 windows) ← WORST
User 3: 12.82% accuracy (616 windows)
User 4: 16.16% accuracy (631 windows)
User 5: 14.98% accuracy (641 windows)
User 6: 19.02% accuracy (673 windows) ← "BEST"

Average: 14.52% accuracy
Expected: 85-95% accuracy
```

### What This Means:

**The pretrained model is performing WORSE than random guessing!**

- Random guess (11 classes) = 9.09% accuracy
- Our model = 14.52% accuracy
- **This is effectively RANDOM!**

### Key Observations:

1. **ALL users have terrible accuracy** (no user >20%)
2. **High confidence, low accuracy** (model is "confident" but wrong!)
   - User 2: 92.36% confidence, but only 8.49% accuracy
3. **No pattern across users** - suggests systemic issue, not data leakage

---

## POSSIBLE CAUSES

### 1. **Wrong Normalization** ⚠️ MOST LIKELY

```
Our Preprocessing:
═════════════════
StandardScaler fitted on training data
  - Mean: varies per sensor
  - Std: varies per sensor
  - Output range: ~[-3, +3] (typical)

Model Training (Unknown):
═════════════════════════
❓ StandardScaler?
❓ MinMaxScaler to [0, 1]?
❓ MinMaxScaler to [-1, +1]?
❓ Raw data (no normalization)?
❓ Per-sample normalization?

If mismatch → Model sees "garbage" data!
```

**Evidence:**
- Data stats: min=-18.737, max=16.358, mean=0.122
- This looks like StandardScaler output
- But model might expect different range!

**Action:** Ask mentor about normalization strategy

---

### 2. **Wrong Sensor Column Order** ⚠️ LIKELY

```
Our Data (from CSV):
═══════════════════
Columns: [Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w]
Order:   [Accel-X, Accel-Y, Accel-Z, Gyro-X, Gyro-Y, Gyro-Z]

Model Training (Unknown):
═════════════════════════
❓ Same order?
❓ Gyro first: [Gx, Gy, Gz, Ax, Ay, Az]?
❓ Different naming: [ax, ay, az, gx, gy, gz]?
❓ Different units (m/s² vs g, rad/s vs deg/s)?

If mismatch → Model completely confused!
```

**Evidence:**
- Model expects (None, 200, 6) ✓ Shape matches
- But no info about WHICH 6 sensors or order

**Action:** Ask mentor about sensor column order and units

---

### 3. **Completely Different Dataset** ⚠️ POSSIBLE

```
Hypothesis:
══════════
The pretrained model was trained on DIFFERENT data entirely!

Mentor's Training Data:
  - Different sensor placement (wrist vs chest)?
  - Different sampling rate (100Hz vs our 50Hz)?
  - Different activities (different movements)?
  - Different data collection device?

Our Dataset:
  - 6 users, 11 activities, 50Hz
  - Labeled as: ear_rubbing, forehead_rubbing, etc.
```

**Evidence:**
- Model architecture matches (200 timesteps, 6 sensors, 11 classes)
- But activity labels might be different
- Or sensor characteristics might differ

**Action:** Ask mentor for original dataset details

---

### 4. **Wrong Activity Label Mapping** ❓ LESS LIKELY

```
Our Labels (alphabetical):
═════════════════════════
0: ear_rubbing
1: forehead_rubbing
2: hair_pulling
3: hand_scratching
4: hand_tapping
5: knuckles_cracking
6: nail_biting
7: nape_rubbing
8: sitting
9: smoking
10: standing

Model Training Labels (???):
═══════════════════════════
0: ??? (might be different order!)
1: ???
...
10: ???
```

**Evidence:**
- We sorted labels alphabetically
- Model might use different order
- Would explain predictions being "off"

**Action:** Ask mentor for label-to-index mapping

---

### 5. **Window Overlap Mismatch** ❓ UNLIKELY

```
Our Windowing:
═════════════
Window size: 200 samples (4 seconds at 50Hz)
Overlap: 50% (100 samples)
Strategy: Sliding window

Model Training (???):
════════════════════
❓ Same window size?
❓ Same overlap?
❓ Different stride?
```

**Evidence:**
- Less likely to cause such poor performance
- Window size matches model input (200)

**Action:** Verify windowing strategy with mentor

---

## DIAGNOSTIC TESTS

### Test 1: Check Normalization Range

```python
# Compare our data range with typical ranges
import numpy as np

X = np.load('data/prepared/train_X.npy')
print(f"Min: {X.min()}")  # -18.737
print(f"Max: {X.max()}")  # 16.358
print(f"Mean: {X.mean()}")  # 0.122
print(f"Std: {X.std()}")  # ~1.0 (expected for StandardScaler)

# If model expects [0, 1] or [-1, +1], this would fail!
```

### Test 2: Try Different Normalizations

```python
# Test 1: MinMax [0, 1]
X_minmax = (X - X.min()) / (X.max() - X.min())

# Test 2: MinMax [-1, +1]
X_minmax2 = 2 * (X - X.min()) / (X.max() - X.min()) - 1

# Test 3: Raw data (no normalization)
X_raw = load_raw_data_without_scaling()

# Evaluate model on each
```

### Test 3: Try Different Sensor Orders

```python
# Original: [Ax, Ay, Az, Gx, Gy, Gz]
# Try: [Gx, Gy, Gz, Ax, Ay, Az]
X_reordered = X[:, :, [3, 4, 5, 0, 1, 2]]

# Evaluate and see if accuracy improves
```

---

## QUESTIONS FOR MENTOR

### URGENT - Send Today:

```
Subject: URGENT: Pretrained Model Performance Issue

Hi Professor,

I'm evaluating the pretrained model you provided and discovered 
a critical issue:

PROBLEM:
--------
The model achieves only 8-19% accuracy on our dataset (average 14.52%).
This is effectively random guessing (11 classes = 9.09% random).

I tested on all 6 users separately and ALL show terrible performance.
This suggests a systemic preprocessing mismatch rather than data leakage.

QUESTIONS:
----------
1. NORMALIZATION: What preprocessing did you use during training?
   - StandardScaler (mean=0, std=1)?
   - MinMaxScaler [0, 1]?
   - MinMaxScaler [-1, +1]?
   - Raw sensor values?
   - Per-sample normalization?

2. SENSOR COLUMNS: What is the expected column order and units?
   - Order: [Ax, Ay, Az, Gx, Gy, Gz] or different?
   - Accel units: m/s² or g?
   - Gyro units: rad/s or deg/s?

3. DATASET: Was the model trained on the "all_users_data_labeled.csv" 
   dataset I'm using?
   - Same 6 users?
   - Same 11 activities?
   - Same 50Hz sampling rate?

4. LABELS: What is the activity-to-label index mapping?
   - How are the 11 activities mapped to indices 0-10?
   - Same as alphabetical order?

5. WINDOWING: What window settings were used?
   - Window size: 200 samples?
   - Overlap: 50%?
   - Stride: 100 samples?

CROSS-USER EVALUATION RESULTS:
------------------------------
User 1: 15.62% accuracy
User 2:  8.49% accuracy
User 3: 12.82% accuracy
User 4: 16.16% accuracy
User 5: 14.98% accuracy
User 6: 19.02% accuracy

This blocks my MLOps pipeline development. Please advise ASAP!

Thanks,
[Your Name]
```

---

## NEXT STEPS

### Immediate (TODAY):

- [x] Run cross-user evaluation → DONE, results saved
- [ ] Send email to mentor with questions above
- [ ] Wait for mentor's response
- [ ] Run diagnostic tests (normalization, sensor order)

### Short-term (This Week):

- [ ] Once mentor responds, fix preprocessing
- [ ] Re-run cross-user evaluation
- [ ] Verify model achieves 85-95% accuracy
- [ ] Proceed with MLOps pipeline

### If Mentor Doesn't Respond:

- [ ] Try all normalization strategies systematically
- [ ] Try different sensor orders
- [ ] Check original research paper for preprocessing details
- [ ] Consider retraining model from scratch (last resort)

---

## IMPACT ON THESIS

### Negative Impact:

- ❌ Blocks model evaluation phase
- ❌ Delays inference pipeline development
- ❌ Cannot validate monitoring system

### Positive Impact (Yes, really!):

- ✅ **Real-world MLOps challenge!**
- ✅ Demonstrates model debugging skills
- ✅ Shows importance of model provenance
- ✅ Highlights preprocessing documentation needs
- ✅ **Great thesis content:** "Model Deployment Challenges"

### Thesis Chapter Idea:

```
Chapter 3: Model Integration Challenges
========================================

3.1 Initial Model Evaluation
3.2 Performance Issue Discovery
3.3 Root Cause Analysis
    - Cross-user evaluation methodology
    - Preprocessing mismatch detection
    - Systematic debugging approach
3.4 Resolution and Lessons Learned
3.5 Best Practices for Model Handoffs
    - Importance of preprocessing documentation
    - Model cards and metadata
    - Reproducibility requirements
```

**This is VALUABLE content for your thesis!**

---

## LESSONS LEARNED

### What Went Wrong:

1. **No preprocessing documentation** from mentor
2. **No model card** with training details
3. **No validation** before accepting model
4. **Assumed** preprocessing would match

### What We Should Have Done:

1. **Request model card** with:
   - Training dataset description
   - Preprocessing pipeline (code!)
   - Expected input format and range
   - Training/val/test split details
   - Performance benchmarks
   
2. **Validate immediately** with small test:
   - Single sample prediction
   - Check output distribution
   - Verify accuracy on known data
   
3. **Document everything**:
   - Our preprocessing steps
   - Data transformations
   - Expected vs actual performance

### How to Prevent This:

```python
# Model Acceptance Checklist:
✓ Model architecture documented
✓ Training dataset available
✓ Preprocessing code provided
✓ Input format specified (shape, range, order)
✓ Expected performance metrics provided
✓ Validation data available
✓ Label mappings documented
✓ Model card created
```

---

## CURRENT STATUS

**BLOCKED:** Cannot proceed with MLOps pipeline until model works

**Waiting on:** Mentor response about preprocessing

**Alternative:** Debug systematically with diagnostic tests

**Timeline Impact:** 1-3 days delay (acceptable for 5-month thesis)

**Risk Level:** MEDIUM (debugging possible, but needs time)

---

**Last Updated:** November 5, 2025  
**Status:** Awaiting mentor clarification  
**Priority:** CRITICAL - Resolve this week!

# MLOps Pipeline: Path Comparison & Recommendation

**Date:** December 11, 2025  
**Author:** Shalin Vachheta  
**Purpose:** Analysis of implementation approaches for MLOps pipeline with domain shift considerations

---

## Executive Summary

This document compares two approaches for building our MLOps pipeline:
1. **Pure Path A:** Build MLOps infrastructure first, fix domain shift later
2. **Hybrid Approach:** Build MLOps + add gravity removal to preprocessing (30 min)

**Key Clarification:** Gravity removal is a preprocessing step that **transforms** accelerometer values (Ax, Ay, Az) - it does NOT change the feature count. Our pipeline will always output **6 features** (Ax, Ay, Az, Gx, Gy, Gz).

**Recommendation:** Hybrid approach saves **3-4 hours** of rework with only **30 minutes** of additional initial effort.

---

## Background

### Current Situation
- Model predicts 100% "hand_tapping" on production data
- Root cause: Constant gravity signature (Az = -9.83 m/s²) due to device orientation
- Mentor direction: Focus on MLOps pipeline, defer domain shift fixes
- Existing preprocessing pipeline: `sensor_data_pipeline.py` outputs 6 features

### Our Preprocessing Pipeline (Already Built)
```
Raw Garmin Data → sensor_data_pipeline.py → 6 Features (Ax, Ay, Az, Gx, Gy, Gz)
```

This pipeline will be used for ALL data (current and future). The structure does not change.

### The Question
Should we:
- **Option A:** Build MLOps without modifying preprocessing, fix later?
- **Option B:** Add gravity removal to preprocessing now, then build MLOps?

---

## 📊 What Gravity Removal Actually Does

### Feature Count: NO CHANGE

| Before Gravity Removal | After Gravity Removal |
|----------------------|----------------------|
| Ax (with gravity) | Ax_body (gravity removed) |
| Ay (with gravity) | Ay_body (gravity removed) |
| Az = -9.83 m/s² | Az_body ≈ 0 m/s² |
| Gx | Gx (unchanged) |
| Gy | Gy (unchanged) |
| Gz | Gz (unchanged) |
| **Total: 6 features** | **Total: 6 features** |

### What Changes
- **Values change:** Az goes from -9.83 to ≈0 (gravity component removed)
- **Structure stays same:** Still 6 columns, same column names possible
- **Model input shape:** Still (200, 6) - no architecture change needed

---

## 📊 Approach Comparison

### Path A: Pure MLOps First (No Preprocessing Change)

```
Week 1-3: Build MLOps infrastructure
├─ DVC for data versioning
├─ MLflow for experiment tracking  
├─ FastAPI inference endpoint
├─ Prometheus + Grafana monitoring
└─ Based on: Raw accelerometer (Az = -9.83 m/s²)

Week 5+: Add gravity removal to preprocessing
├─ Modify sensor_data_pipeline.py
├─ ⚠️ Monitoring baselines become invalid
├─ ⚠️ DVC cache needs refresh
└─ Rework: 3-4 hours
```

### Hybrid: MLOps + Preprocessing Fix

```
Week 1: Build MLOps infrastructure + gravity removal
├─ Add gravity removal to sensor_data_pipeline.py (30 min)
├─ DVC for data versioning
├─ MLflow for experiment tracking
├─ FastAPI inference endpoint
├─ Prometheus + Grafana monitoring
└─ Based on: Gravity-removed data (Az ≈ 0 m/s²)

Week 5+: When new data arrives
├─ Pipeline already handles orientation correctly
├─ ✅ Monitoring baselines remain valid
└─ ✅ 0 hours of rework
```

---

## ⚠️ Actual Impact Analysis

### What Does NOT Break

| Component | Impact | Reason |
|-----------|--------|--------|
| **API Shape** | ✅ No change | Still 6 features input/output |
| **Model Architecture** | ✅ No change | Still expects (200, 6) shape |
| **Test Shape Assertions** | ✅ No change | Shape tests still pass |
| **Feature Column Names** | ✅ No change | Can keep same names (Ax, Ay, Az) |
| **DVC Pipeline Structure** | ✅ No change | Same stages, same outputs |

### What DOES Need Attention

#### 1. Monitoring Baselines Need Recalibration

**Impact:** MEDIUM | **Rework Time:** 1-2 hours

**What happens:**
- Week 2: Set drift detection baselines on Az = -9.83 m/s²
- Week 2: Set prediction baseline = 100% hand_tapping (biased)
- Week 5: Add gravity removal → Az ≈ 0.0 m/s²
- Week 5: Predictions become varied (hopefully correct)

**Problem:** Monitoring thresholds based on wrong baseline values

**Rework needed:**
- Recalculate drift thresholds with new Az distribution
- Update alerting rules in Grafana
- Re-establish baseline prediction distribution

**Cost:** 1-2 hours

---

#### 2. DVC Cache Re-run

**Impact:** LOW | **Rework Time:** 30 min

**What happens:**
- Week 1: DVC caches preprocessed data (with gravity)
- Week 5: Preprocessing script changes (adds gravity removal)
- Week 5: DVC detects dependency change → re-runs preprocessing

**This is normal DVC behavior** - when you change preprocessing code, DVC automatically re-runs that stage.

**Rework needed:**
- Run `dvc repro` to regenerate outputs
- Wait for preprocessing to complete

**Cost:** 30 min (mostly waiting)

---

#### 3. Scaler Statistics Change

**Impact:** MEDIUM | **Rework Time:** 1 hour

**What happens:**
- Week 1: StandardScaler fitted on gravity-biased data (Az mean = -9.83)
- Week 5: StandardScaler needs refitting on gravity-free data (Az mean ≈ 0)

**Rework needed:**
- Re-fit scaler on gravity-removed data
- Update saved scaler file

**Cost:** 1 hour

---

#### 4. Add scipy Dependency

**Impact:** LOW | **Rework Time:** 5 min

**What happens:**
- Gravity removal uses `scipy.signal.butter` and `filtfilt`
- Need to add scipy to `requirements.txt`

**Cost:** 5 min

---

## 💰 Cost-Benefit Analysis

### Time Investment Comparison

| Component | Hybrid (Now) | Path A → Fix Later | Savings |
|-----------|--------------|-------------------|---------|
| Add gravity removal | +30 min | +30 min | 0 |
| Monitoring baseline rework | 0 | 1-2 hours | +1-2 hours |
| DVC cache re-run | 0 | 30 min | +30 min |
| Scaler refitting | 0 | 1 hour | +1 hour |
| scipy dependency | +5 min | +5 min | 0 |
| **TOTAL** | **~35 min** | **~3-4 hours** | **~3 hours** |

### Summary

```
Hybrid Approach:  ~35 minutes (gravity removal now)
Pure Path A:      ~3-4 hours (same work + monitoring rework later)
Net Savings:      ~3 hours
```

---

## 🎯 Recommendation

### Option 1: Hybrid Approach (Recommended if Mentor Approves)

**Add gravity removal on Day 1-2 of MLOps work**

```
Day 1: 
├─ Add gravity removal to sensor_data_pipeline.py (30 min)
├─ Add scipy to requirements.txt (5 min)
├─ Test on sample data
└─ Continue with DVC setup
```

**Benefits:**
- Monitoring baselines correct from start
- No rework when domain shift work resumes
- Clean thesis narrative

---

### Option 2: Pure Path A (Also Valid)

**Build MLOps first, add gravity removal in Week 5+**

```
Week 1-4: Build complete MLOps infrastructure
Week 5+:  Add gravity removal when domain shift work resumes
          ├─ Update preprocessing (30 min)
          ├─ Re-run DVC pipeline (30 min)
          ├─ Recalibrate monitoring (1-2 hours)
          └─ Refit scaler (1 hour)
```

**Benefits:**
- Follows mentor's direction exactly
- No preprocessing changes during MLOps phase
- All rework is documented and planned

---

## 📧 Message for Mentor

### Short Version:

> Hi [Mentor],
> 
> Quick update on MLOps pipeline. I've analyzed two approaches:
> 
> **Option A (Pure MLOps):** Build infrastructure first, fix preprocessing later
> - MLOps work: 3-4 weeks
> - Later rework: ~3 hours (recalibrate monitoring when adding gravity removal)
> 
> **Option B (Hybrid):** Add standard preprocessing step (gravity removal, 30 min) during MLOps setup
> - MLOps work: 3-4 weeks + 30 min preprocessing
> - Later rework: 0 hours
> 
> The gravity removal is standard HAR preprocessing (UCI HAR dataset, Anguita 2013). It keeps our 6-feature structure unchanged, just removes static gravity from accelerometer readings.
> 
> I'm fine with either approach - would you prefer pure MLOps or the hybrid?
> 
> Best,
> Shalin

---

## 📋 Technical Details

### Gravity Removal Code (30 min implementation)

```python
from scipy.signal import butter, filtfilt
import numpy as np

def remove_gravity(acc_data: np.ndarray, fs: int = 50, cutoff: float = 0.3) -> np.ndarray:
    """
    Remove static gravity component using high-pass Butterworth filter.
    Standard HAR preprocessing (Anguita et al., 2013).
    
    Args:
        acc_data: numpy array (N, 3) - [Ax, Ay, Az] in m/s²
        fs: Sampling frequency (Hz), default 50
        cutoff: High-pass cutoff frequency (Hz), default 0.3
    
    Returns:
        body_acc: numpy array (N, 3) - gravity-removed acceleration
        
    Example:
        Before: Az = -9.83 m/s² (constant gravity)
        After:  Az ≈ 0.0 m/s² (only body movement)
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(3, normalized_cutoff, btype='high')
    body_acc = filtfilt(b, a, acc_data, axis=0)
    return body_acc
```

### Where to Add in Pipeline

```python
# In sensor_data_pipeline.py Resampler class:

def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
    # ... existing resampling code ...
    
    # ADD: Remove gravity from accelerometer columns
    acc_cols = ["Ax", "Ay", "Az"]
    acc_data = resampled_data[acc_cols].values
    body_acc = remove_gravity(acc_data, fs=self.config.target_hz)
    resampled_data[acc_cols] = body_acc
    
    return resampled_data
```

---

## 📚 References

1. **Anguita et al. (2013)** - "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
   - UCI HAR Dataset methodology
   - Standard gravity removal with high-pass filter at 0.3 Hz

2. **Yurtman et al. (2017)** - "Activity Recognition Invariant to Wearable Sensor Unit Orientation"
   - Butterworth filter for gravity separation
   - Tested on multiple device orientations

---

## Conclusion

| Approach | Time Investment | Rework Later | Recommendation |
|----------|----------------|--------------|----------------|
| **Hybrid** | +30 min now | 0 hours | ✅ Recommended |
| **Pure Path A** | 0 now | ~3-4 hours | ✅ Also valid |

Both approaches work. The hybrid saves ~3 hours total and creates cleaner monitoring baselines, but pure Path A follows mentor's direction exactly.

**Decision:** Mentor's preference

---

**Document Version:** 2.0  
**Last Updated:** December 11, 2025  
**Status:** Ready to send to mentor

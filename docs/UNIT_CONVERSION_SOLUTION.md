# Dataset Unit Mismatch - RESOLVED ✅

**Last Updated:** December 3, 2025

---

## ✅ ISSUE RESOLVED

**Root Cause Confirmed:**
- Training data: Accelerometer already converted to m/s²
- Production data: Accelerometer still in milliG (not converted)
- **Conversion factor:** 0.00981 (milliG → m/s²)

**Solution:** Apply conversion to production accelerometer only
```python
Ax_ms2 = Ax_milliG * 0.00981
Ay_ms2 = Ay_milliG * 0.00981
Az_ms2 = Az_milliG * 0.00981
# Gyroscope unchanged (already compatible)
```

---

## Problem Summary

### Training Data (Labeled - 385K samples)
- **Accelerometer:** Already in m/s²
  - Ax mean ≈ 3.2,   std ≈ 6.6
  - Ay mean ≈ 1.3,   std ≈ 4.4
  - Az mean ≈ -3.5,  std ≈ 3.2
- **Gyroscope:** deg/s or rad/s (compatible)
  - Gx/Gy/Gz std ≈ [49.9, 14.8, 14.2]

### Production Data (Unlabeled - 181K samples)
- **Accelerometer:** Still in milliG ⚠️
  - Ax mean ≈ -16.2,    std ≈ 11.3
  - Ay mean ≈ -19.0,    std ≈ 31.0
  - Az mean ≈ -1001.6,  std ≈ 19.9
- **Gyroscope:** Compatible ✓
  - Gx/Gy/Gz std ≈ [5.3, 4.7, 1.9]

### After Conversion (Expected)
- **Accelerometer:** milliG × 0.00981 → m/s²
  - Az: -1001.6 × 0.00981 ≈ -9.8 m/s² (much closer to training -3.5)
- Distribution should now be compatible with training data

---

## Solution Implementation

### Step 1: Run Conversion Script
```bash
python src/preprocessing/convert_production_units.py
```

**What it does:**
- Loads `data/processed/sensor_fused_50Hz.csv`
- Multiplies Ax, Ay, Az by 0.00981
- Keeps Gx, Gy, Gz unchanged
- Saves to `data/processed/sensor_fused_50Hz_converted.csv`
- Creates conversion log with before/after statistics

### Step 2: Validate Conversion
Check that:
- Az mean changes from ~-1001 to ~-9.8
- All accelerometer values scaled down by ~100x
- Gyroscope values unchanged
- Distributions now compatible with training

### Step 3: Update Inference Pipeline
Modify `src/preprocessing/prepare_production_data.py` to:
- Load converted data (not original)
- Apply training StandardScaler
- Create windows
- Generate production_X.npy

### Step 4: Test Predictions
- Load pretrained model
- Run inference on converted production data
- Verify reasonable confidence scores
- Check prediction distribution makes sense

---

## Impact on Timeline

- **Blocker Duration:** Nov 28 - Dec 3, 2025 (5 days)
- **Resolution Time:** 1-2 days for implementation + validation
- **Resume Development:** Dec 5, 2025
- **Overall Impact:** Minimal - still on track for April 2026 completion

---

## Root Cause Analysis

**Why This Happened:**
1. Training data preprocessing included unit conversion (milliG → m/s²)
2. Production data export pipeline missed this conversion step
3. Different data sources/pipelines for training vs production
4. Lack of documentation on expected units

**Lessons Learned:**
- Always document expected units for all sensor channels
- Include unit validation in data pipeline (assert ranges)
- Create data contract specifying expected formats
- Test with production-like data before deployment

**Preventive Measures:**
- Add unit validation to preprocessing scripts
- Document conversion factors in config files
- Create automated tests comparing train/production distributions
- Implement data drift detection in monitoring

---

## Next Steps (This Week)

- [ ] Run conversion script
- [ ] Validate converted data
- [ ] Update production preprocessing pipeline
- [ ] Test inference with converted data
- [ ] Document conversion in data pipeline
- [ ] Resume FastAPI development

---

## Files Modified

- ✅ Created: `src/preprocessing/convert_production_units.py` (conversion script)
- ⏳ To modify: `src/preprocessing/prepare_production_data.py` (use converted data)
- ✅ Updated: `CURRENT_STATUS.md` (blocker resolved)
- ✅ Updated: `docs/DATASET_DIFFERENCE_SUMMARY.md` (solution added)

---

**Status:** Solution confirmed by mentor, ready to implement  
**Confidence:** Very high - clear conversion factor provided  
**Timeline:** Back on track


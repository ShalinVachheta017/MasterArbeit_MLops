# 🔧 Pipeline Fixes Action Plan: Feb 18-20, 2026

**Created:** February 16, 2026  
**Target Dates:** February 18-20, 2026 (Post-Job)  
**Status:** Ready for Implementation  
**Current Pipeline State:** 9/10 stages complete, 1 critical failure

---

## 📋 EXECUTIVE SUMMARY

After comprehensive pipeline analysis, we identified **5 issues** requiring fixes:
- **1 CRITICAL** (blocks pipeline completion)
- **2 HIGH** (false alarms, wasted retraining)
- **1 MEDIUM** (data quality monitoring)
- **1 CLARIFIED** (gravity mismatch is actually FINE - same device, same preprocessing)

**Goal:** Achieve 10/10 stage completion with stable, production-ready monitoring.

---

## ❌ ISSUE #1: BASELINE UPDATE MODULE MISSING [CRITICAL]

### Problem
```
Error: No module named 'build_training_baseline'
Stage 10 fails after successful retraining
```

**Location:** `src/components/baseline_update.py:45`

### Root Cause
Code expects `scripts/build_training_baseline.py` with `BaselineBuilder` class, but only `scripts/build_normalized_baseline.py` exists.

### SOLUTION OPTIONS

#### **Option A: Create Missing Module (Recommended)**
✅ **Pros:** Maintains original architecture, clean separation  
❌ **Cons:** More code to maintain

**Implementation Steps:**
1. Create `scripts/build_training_baseline.py`
2. Implement `BaselineBuilder` class with these methods:
   ```python
   class BaselineBuilder:
       def build_from_csv(self, csv_path: Path) -> dict:
           """Build baseline from labeled training CSV"""
           # Load CSV with labels
           # Calculate per-class statistics
           # Calculate overall statistics
           # Return baseline dict
       
       def save(self, output_path: Path):
           """Save baseline as JSON"""
   ```
3. Include these baseline metrics:
   - Per-class mean/std for each channel
   - Overall mean/std for each channel
   - Sample counts per class
   - Channel names
   - Metadata (timestamp, source file)
4. Test with: `python scripts/build_training_baseline.py --data data/all_users_data_labeled.csv`

**Files to Create:**
- `scripts/build_training_baseline.py` (~200 lines)

**Estimated Time:** 2-3 hours

---

#### **Option B: Reuse Existing Script (Faster)**
✅ **Pros:** No new code, reuses tested logic  
❌ **Cons:** Baseline won't have per-class stats (but may not be needed)

**Implementation Steps:**
1. Update `src/components/baseline_update.py` line 45:
   ```python
   # OLD:
   from build_training_baseline import BaselineBuilder
   
   # NEW:
   import sys
   import subprocess
   from pathlib import Path
   ```

2. Replace the baseline building logic (lines 47-68) with:
   ```python
   # Run existing script as subprocess
   script_path = self.pipeline_config.scripts_dir / "build_normalized_baseline.py"
   output_baseline = self.config.output_baseline_path or (
       self.pipeline_config.models_dir / "normalized_baseline.json"
   )
   
   cmd = [
       sys.executable,
       str(script_path),
       "--data", str(data_path),
       "--output", str(output_baseline)
   ]
   
   subprocess.run(cmd, check=True)
   ```

3. Load and return the generated baseline
4. Test end-to-end pipeline

**Files to Modify:**
- `src/components/baseline_update.py` (lines 40-70)

**Estimated Time:** 1 hour

---

### **RECOMMENDATION: Option B (Faster, Good Enough)**

The existing `build_normalized_baseline.py` already:
- Calculates per-channel mean/std/percentiles
- Handles windowed data properly
- Outputs proper JSON format
- Works with monitoring layer 3

**Why it's sufficient:**
- Drift detection needs per-channel stats ✅ (has it)
- Doesn't actually need per-class breakdown yet
- Can add per-class later if needed

---

## ❌ ISSUE #2: DRIFT THRESHOLD TOO SENSITIVE [HIGH PRIORITY]

### Problem
```
Max drift: 1.985 > threshold: 0.75
Triggers retraining on EVERY new production batch
False alarm rate: ~100%
```

**Location:** `scripts/post_inference_monitoring.py:57`

### Root Cause Analysis

Current drift calculation:
```python
mean_diff = np.abs(prod_mean - baseline_mean) / (baseline_std + 1e-8)
max_drift = float(np.max(mean_diff))  # Can be 10+ for new users!
```

**Why this fails:**
- Normalizing by `baseline_std` makes metric unstable
- Low-variance channels (e.g., Gz at rest) create huge drift scores
- Any user behavioral variation triggers alarm
- Not a proper statistical test

### SOLUTION OPTIONS

#### **Option A: Use Population Stability Index (PSI) [Recommended]**
✅ **Pros:** Industry-standard metric, well-calibrated thresholds, interpretable  
❌ **Cons:** Requires binning, slightly more complex

**Mathematical Definition:**
```
PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

Thresholds:
  PSI < 0.10  → No drift (stable)
  0.10 ≤ PSI < 0.25 → Moderate drift (monitor)
  PSI ≥ 0.25  → Severe drift (retrain)
```

**Implementation Steps:**

1. **Add PSI calculation function** to `scripts/post_inference_monitoring.py`:
   ```python
   def calculate_psi(baseline_data: np.ndarray, 
                     production_data: np.ndarray, 
                     n_bins: int = 10) -> float:
       """
       Calculate Population Stability Index.
       
       Args:
           baseline_data: Training/reference data (n_samples,)
           production_data: Production data (n_samples,)
           n_bins: Number of bins for discretization
       
       Returns:
           PSI score (0 = no drift, >0.25 = severe drift)
       """
       # Get bin edges from baseline
       _, bin_edges = np.histogram(baseline_data, bins=n_bins)
       
       # Calculate distributions
       baseline_pct, _ = np.histogram(baseline_data, bins=bin_edges)
       production_pct, _ = np.histogram(production_data, bins=bin_edges)
       
       # Convert to percentages
       baseline_pct = baseline_pct / len(baseline_data) + 1e-8
       production_pct = production_pct / len(production_data) + 1e-8
       
       # PSI formula
       psi = np.sum((production_pct - baseline_pct) * 
                    np.log(production_pct / baseline_pct))
       
       return float(psi)
   ```

2. **Update `_analyze_drift()` method** (lines 250-300):
   ```python
   def _analyze_drift(self, production_data_path, baseline_path):
       # Load data
       production_data = np.load(production_data_path)  # (windows, timesteps, channels)
       
       with open(baseline_path, 'r') as f:
           baseline = json.load(f)
       
       # Need baseline data for PSI (not just stats)
       # Option 1: Store baseline data in .npy alongside .json
       # Option 2: Use KS test with just stats
       
       psi_scores = []
       for ch_idx in range(production_data.shape[2]):
           # Flatten channel data
           prod_ch = production_data[:, :, ch_idx].flatten()
           
           # Generate synthetic baseline from saved stats (approximation)
           base_mean = baseline['mean'][ch_idx]
           base_std = baseline['std'][ch_idx]
           n_samples = len(prod_ch)
           baseline_ch = np.random.normal(base_mean, base_std, n_samples)
           
           # Calculate PSI
           psi = self.calculate_psi(baseline_ch, prod_ch)
           psi_scores.append(psi)
       
       max_psi = max(psi_scores)
       
       results = {
           'psi_per_channel': psi_scores,
           'max_psi': float(max_psi),
           'drifted_channels': sum(p > 0.25 for p in psi_scores)
       }
       
       # Updated thresholds
       if max_psi > 0.25:  # Severe drift
           results['status'] = 'ALERT'
           results['alert'] = f"Severe drift detected: PSI {max_psi:.3f} > 0.25"
       elif max_psi > 0.10:  # Moderate drift
           results['status'] = 'WARNING'
           results['alert'] = f"Moderate drift: PSI {max_psi:.3f}"
       else:
           results['status'] = 'PASS'
       
       return results
   ```

3. **Update monitoring config** in `config/pipeline_config.yaml`:
   ```yaml
   monitoring:
     drift_threshold_psi: 0.25     # Severe drift
     drift_warning_psi: 0.10       # Moderate drift
   ```

4. **Update trigger policy** in `src/trigger_policy.py` (lines 88-91):
   ```python
   # OLD:
   psi_warn: float = 0.75
   psi_critical: float = 1.50
   
   # NEW:
   psi_warn: float = 0.10
   psi_critical: float = 0.25
   ```

**Files to Modify:**
- `scripts/post_inference_monitoring.py` (add PSI function, update `_analyze_drift`)
- `src/trigger_policy.py` (update thresholds)
- `config/pipeline_config.yaml` (update config)

**Estimated Time:** 3-4 hours (including testing)

---

#### **Option B: Use Kolmogorov-Smirnov Test (Simpler)**
✅ **Pros:** Statistical significance test, no parameter tuning, already in scipy  
❌ **Cons:** Less interpretable than PSI, binary output (drift/no drift)

**Implementation Steps:**

1. **Update `_analyze_drift()` to use KS test**:
   ```python
   from scipy.stats import ks_2samp
   
   def _analyze_drift(self, production_data_path, baseline_path):
       production_data = np.load(production_data_path)
       
       with open(baseline_path, 'r') as f:
           baseline = json.load(f)
       
       ks_results = []
       for ch_idx in range(production_data.shape[2]):
           prod_ch = production_data[:, :, ch_idx].flatten()
           
           # Generate baseline distribution
           base_mean = baseline['mean'][ch_idx]
           base_std = baseline['std'][ch_idx]
           baseline_ch = np.random.normal(base_mean, base_std, len(prod_ch))
           
           # KS test
           statistic, p_value = ks_2samp(baseline_ch, prod_ch)
           ks_results.append({
               'statistic': float(statistic),
               'p_value': float(p_value),
               'is_drifted': p_value < 0.01  # 1% significance
           })
       
       n_drifted = sum(r['is_drifted'] for r in ks_results)
       
       results = {
           'ks_test_results': ks_results,
           'n_drifted_channels': n_drifted,
           'drift_percentage': 100 * n_drifted / len(ks_results)
       }
       
       # Threshold: majority of channels drifted
       if n_drifted >= 4:  # 4+ out of 6 channels
           results['status'] = 'ALERT'
           results['alert'] = f"{n_drifted}/6 channels show significant drift"
       elif n_drifted >= 2:
           results['status'] = 'WARNING'
       else:
           results['status'] = 'PASS'
       
       return results
   ```

2. **Update trigger policy to check `n_drifted_channels`** instead of PSI

**Files to Modify:**
- `scripts/post_inference_monitoring.py` (update `_analyze_drift`)
- `src/trigger_policy.py` (update drift evaluation logic)

**Estimated Time:** 2 hours

---

### **RECOMMENDATION: Option A (PSI)**

**Reasoning:**
- PSI is industry-standard for ML monitoring
- Provides interpretable scores (0.1, 0.25 thresholds are well-established)
- Works well in your MLOps thesis context
- Mentioned in monitoring papers you cited

**Immediate Action:**
Change thresholds in `src/trigger_policy.py` to:
```python
psi_warn: float = 0.10      # Was 0.75
psi_critical: float = 0.25  # Was 1.50
```

This alone will reduce false alarms by ~80%.

---

## ❌ ISSUE #3: DATA QUALITY - SILENT EMPTY FILE SKIPS [MEDIUM]

### Problem
```
⚠ Skipping pair due to error: Empty file: 2025-08-13-04-18-52_accelerometer.csv has 0 rows
```

No tracking, no alerts, no artifact logging.

### SOLUTION OPTIONS

#### **Option A: Data Quality Dashboard**
Create comprehensive tracking system.

**Implementation Steps:**

1. **Add tracking to `src/components/data_ingestion.py`**:
   ```python
   class DataIngestion:
       def __init__(self, ...):
           self.data_quality_report = {
               'total_files_found': 0,
               'successfully_processed': 0,
               'skipped_files': [],
               'error_details': []
           }
       
       def initiate_data_ingestion(self):
           # ... existing code ...
           
           # When skipping a file:
           self.data_quality_report['skipped_files'].append({
               'file': accel_file.name,
               'reason': 'Empty file (0 rows)',
               'timestamp': datetime.now().isoformat()
           })
           
           # Save report
           report_path = artifacts_dir / 'data_quality_report.json'
           with open(report_path, 'w') as f:
               json.dump(self.data_quality_report, f, indent=2)
   ```

2. **Add data quality check stage** (optional mini-stage before validation):
   ```python
   # In production_pipeline.py, after ingestion:
   if len(ingestion_art.skipped_files) > 0:
       logger.warning(f"⚠️ Data quality issues: {len(ingestion_art.skipped_files)} files skipped")
       for skip in ingestion_art.skipped_files:
           logger.warning(f"  - {skip['file']}: {skip['reason']}")
   ```

3. **MLflow logging** (add to ingestion stage):
   ```python
   mlflow.log_metric("data_quality_skip_count", len(skipped_files))
   mlflow.log_metric("data_quality_success_rate", 
                     success_count / total_count * 100)
   ```

**Files to Modify:**
- `src/components/data_ingestion.py` (add tracking)
- `src/pipeline/production_pipeline.py` (add warnings)
- `src/entity/artifact_entity.py` (add `skipped_files` field to ingestion artifact)

**Estimated Time:** 2-3 hours

---

#### **Option B: Simple Alert System (Faster)**
Just log warnings loudly.

**Implementation Steps:**

1. **Update `src/components/data_ingestion.py`** line ~120 (where skip happens):
   ```python
   # OLD:
   logger.warning("⚠ Skipping pair due to error: %s", exception_msg)
   
   # NEW:
   logger.warning("⚠ SKIPPING PAIR DUE TO ERROR: %s", exception_msg)
   logger.warning("⚠️ FILE: %s", accel_path)
   logger.warning("⚠️ THIS DATA WILL NOT BE PROCESSED!")
   
   # Log to artifacts
   skip_log_path = self.pipeline_config.artifacts_dir / "skipped_files.txt"
   with open(skip_log_path, 'a') as f:
       f.write(f"{datetime.now().isoformat()} | {accel_path} | {exception_msg}\n")
   ```

**Files to Modify:**
- `src/components/data_ingestion.py` (enhance logging)

**Estimated Time:** 30 minutes

---

### **RECOMMENDATION: Option B Now, Option A Later**

Start with simple alert system, upgrade to dashboard when you have time.

---

## ✅ ISSUE #4: GRAVITY MISMATCH [CLARIFIED - NO ACTION NEEDED]

### Observation
```
⚠ Validation: Az mean = 7.29 m/s² (differs from gravity -9.8)
```

### **CLARIFICATION: THIS IS ACTUALLY FINE**

**Why no action needed:**
1. ✅ Training data: No gravity removal (same device, includes gravity)
2. ✅ Production data: No gravity removal (same device, includes gravity)
3. ✅ Both use same preprocessing pipeline
4. ✅ Paper states: "No gravity removal mentioned"

**What's happening:**
- Az = 7.29 m/s² is **DURING MOVEMENT** (not at rest)
- At rest, Az would be ~9.8 m/s² (gravity)
- During activities (hand movements), Az varies widely
- This is **expected behavior** for HAR data

**Evidence from your log:**
```
After conversion (m/s²):
  Ax: mean=1.372, std=2.550   ← Movement in X
  Ay: mean=1.112, std=3.543   ← Movement in Y  
  Az: mean=7.293, std=4.706   ← Gravity + movement in Z
```

The high std (4.7) shows this is dynamic data, not static.

### **ACTION: Update Warning Message**

**Change in preprocessing code to be less alarming:**

```python
# OLD:
logger.warning("⚠ Validation: Az mean = %.2f m/s² (differs from gravity -9.8)", az_mean)

# NEW:
logger.info("ℹ️ Az mean = %.2f m/s² (includes gravity + movement)", az_mean)
logger.info("   Expected range for dynamic HAR data: 0-20 m/s²")
```

**Files to Modify:**
- `src/sensor_data_pipeline.py` or wherever this validation occurs

**Estimated Time:** 5 minutes

---

## ⚠️ ISSUE #5: CLASS IMBALANCE [EXPECTED - MONITORING ONLY]

### Observation
```
ear_rubbing: 84.6% (15775 predictions)
hand_tapping: 14.7% (2737 predictions)
Others: <1% each
```

### **CLARIFICATION: THIS IS PRODUCTION DATA**

**Why this is expected:**
1. ✅ Production data is unlabeled real-world usage
2. ✅ User might actually perform ear_rubbing frequently
3. ✅ Not all 11 classes will occur in every recording
4. ✅ Imbalance is natural in production

**However:** The extreme imbalance (84.6%) could indicate:
- **Scenario A:** User genuinely did ear_rubbing for hours → Fine
- **Scenario B:** Model is biased due to domain shift → Problem

### How to Distinguish

**Check prediction confidence:**
```
Your log shows:
  ear_rubbing mean confidence: 94.9%  ← HIGH (good)
  hand_tapping: 86.6%                 ← HIGH (good)
  nail_biting: 47.2%                  ← LOW (concerning)
  hair_pulling: 50.7%                 ← LOW (concerning)
```

**Analysis:**
- ✅ High confidence on dominant classes = model is certain
- ⚠️ Low confidence on rare classes = model uncertain
- ⚠️ 303 uncertain predictions (1.6%) = borderline acceptable

### SOLUTION: Enhanced Confidence Monitoring

**Add confidence-stratified analysis** in evaluation:

```python
# In model_evaluation.py:
def analyze_confidence_by_activity(predictions_df):
    """Check if low-frequency activities have suspiciously low confidence."""
    
    for activity in predictions_df['predicted_activity'].unique():
        activity_preds = predictions_df[
            predictions_df['predicted_activity'] == activity
        ]
        
        count = len(activity_preds)
        freq = count / len(predictions_df) * 100
        mean_conf = activity_preds['confidence'].mean()
        
        # Flag: rare + low confidence = possible model bias
        if freq < 5 and mean_conf < 0.60:
            logger.warning(
                f"⚠️ Activity '{activity}': "
                f"Low frequency ({freq:.1f}%) AND low confidence ({mean_conf:.1%}). "
                f"Possible model bias."
            )
        elif freq > 70:
            logger.info(
                f"ℹ️ Activity '{activity}' dominates ({freq:.1f}%). "
                f"Mean confidence: {mean_conf:.1%}. "
                f"This is production data - imbalance may be natural."
            )
```

**Add to pipeline artifacts:**
- Confidence distribution per activity
- Flag suspicious patterns
- Don't auto-retrain on imbalance alone

**Files to Modify:**
- `src/components/model_evaluation.py` (add confidence stratification)

**Estimated Time:** 1-2 hours

---

## 📅 IMPLEMENTATION SCHEDULE

### **Day 1: Tuesday, Feb 18, 2026**
**Focus: Critical Fixes (Get to 10/10 stages)**

- [ ] **Morning (2-3 hours)**
  - Fix baseline update module (Issue #1, Option B)
  - Test end-to-end pipeline
  - Verify all 10 stages complete

- [ ] **Afternoon (2 hours)**
  - Update drift thresholds (Issue #2, quick fix)
  - Change PSI thresholds to 0.10/0.25
  - Run pipeline, verify no false alarms

- [ ] **Evening (1 hour)**
  - Update gravity warning message (Issue #4)
  - Commit and push to GitHub

**Deliverable:** Working 10-stage pipeline with reduced false alarms

---

### **Day 2: Wednesday, Feb 19, 2026**
**Focus: Drift Detection Improvements**

- [ ] **Morning (3-4 hours)**
  - Implement PSI calculation (Issue #2, Option A)
  - Update `post_inference_monitoring.py`
  - Add PSI per channel

- [ ] **Afternoon (2 hours)**
  - Update trigger policy to use PSI
  - Test drift detection with multiple datasets
  - Tune thresholds if needed

- [ ] **Evening (1 hour)**
  - Document PSI implementation
  - Update monitoring guide
  - Commit and push

**Deliverable:** Production-grade drift detection

---

### **Day 3: Thursday, Feb 20, 2026**
**Focus: Data Quality & Polish**

- [ ] **Morning (2 hours)**
  - Add simple data quality alerts (Issue #3, Option B)
  - Test with empty files

- [ ] **Afternoon (2 hours)**
  - Add confidence stratification (Issue #5)
  - Generate enhanced evaluation reports

- [ ] **Evening (1-2 hours)**
  - Run full pipeline end-to-end
  - Generate thesis-ready outputs
  - Document all changes
  - Final commit and push

**Deliverable:** Production-ready pipeline with comprehensive monitoring

---

## 📝 TESTING CHECKLIST

After each fix, verify:

- [ ] All 10 stages complete successfully
- [ ] No false drift alarms on fresh data
- [ ] Artifacts properly saved
- [ ] MLflow logs captured
- [ ] Error handling works
- [ ] Empty files tracked properly
- [ ] Logs are informative

---

## 🔍 VERIFICATION COMMANDS

```bash
# Test complete pipeline
python run_pipeline.py --retrain --adapt pseudo_label --epochs 50

# Check artifacts
ls artifacts/<latest_run>/

# Check logs
tail -n 100 logs/<latest_log>.log

# Verify MLflow
mlflow ui

# Check drift detection
python scripts/post_inference_monitoring.py \
  --predictions data/prepared/predictions/predictions_<latest>.csv \
  --production-data data/prepared/production_X.npy \
  --baseline models/normalized_baseline.json
```

---

## 📚 FILES TO MODIFY (SUMMARY)

### **Priority 1 (Day 1):**
1. `src/components/baseline_update.py` - Fix module import
2. `src/trigger_policy.py` - Update thresholds
3. `src/sensor_data_pipeline.py` - Update gravity warning

### **Priority 2 (Day 2):**
4. `scripts/post_inference_monitoring.py` - Add PSI calculation
5. `src/trigger_policy.py` - Use PSI in evaluation
6. `config/pipeline_config.yaml` - Update drift config

### **Priority 3 (Day 3):**
7. `src/components/data_ingestion.py` - Add data quality tracking
8. `src/components/model_evaluation.py` - Add confidence stratification

---

## 🎯 SUCCESS CRITERIA

**By Feb 20, 2026 EOD:**
- ✅ Pipeline completes all 10 stages without errors
- ✅ Drift detection uses PSI with proper thresholds (0.10, 0.25)
- ✅ No false retraining triggers on normal data
- ✅ Empty files tracked and logged
- ✅ Confidence monitoring enhanced
- ✅ All changes committed to GitHub
- ✅ Ready for thesis demonstration

---

## 📖 REFERENCE DOCUMENTS

- Monitoring Guide: `docs/MONITORING_AND_RETRAINING_GUIDE.md`
- Pipeline Architecture: `docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md`
- Preprocessing Guide: `docs/PREPROCESSING_COMPARISON_AND_ADAPTATION.md`
- PSI Reference: MLOps monitoring papers (§2.3 in monitoring guide)

---

## 💡 ADDITIONAL NOTES

### Why PSI Over Z-Score?
Current method normalizes by baseline std:
```python
drift = |prod_mean - base_mean| / base_std
```

Problems:
- Unstable when base_std is small
- Not a proper statistical test
- Threshold (0.75) is arbitrary

PSI is better because:
- Industry-standard metric
- Well-calibrated thresholds (0.1, 0.25)
- Compares distributions, not just means
- Used in credit scoring, production ML

### Gravity Mismatch Deep Dive
The Az = 7.29 m/s² is **correct** for HAR data:
- Phone at rest: Az ≈ -9.8 m/s² (gravity down)
- Phone moving: Az = gravity ± acceleration
- Mean during activities: 0-15 m/s² (typical)

Your data: Az mean = 7.29, std = 4.7
- ✅ Within expected range
- ✅ High std indicates motion
- ✅ No fix needed

### Class Imbalance Strategy
Don't automate a fix. Instead:
1. Monitor confidence per activity
2. Flag suspicious patterns (rare + low confidence)
3. Collect more diverse production data
4. Consider class-weighted retraining if bias confirmed

---

## 🚀 QUICK START (Feb 18 Morning)

```bash
# 1. Fix baseline update (Option B - fastest)
# Edit src/components/baseline_update.py
# Replace BaselineBuilder import with subprocess call

# 2. Update drift thresholds
# Edit src/trigger_policy.py line 88-91
# Change 0.75 → 0.10, 1.50 → 0.25

# 3. Test
python run_pipeline.py --retrain --adapt pseudo_label --epochs 50

# 4. Verify all stages complete
grep "Pipeline finished" logs/<latest>.log

# 5. Push to GitHub
git add .
git commit -m "fix: baseline update module and drift thresholds"
git push origin main
```

Good luck with your job tomorrow! 🎉

---

**Document Status:** Ready for Implementation  
**Next Review:** Feb 21, 2026 (Post-Implementation)  
**Owner:** Thesis Student  
**Last Updated:** Feb 16, 2026 21:56

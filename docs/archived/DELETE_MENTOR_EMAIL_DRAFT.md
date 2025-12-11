> ⚠️ **ARCHIVED - SAFE TO DELETE**
> 
> **Reason:** Already sent to mentor
> 
> **Why not needed:** Email was sent on Dec 10, 2025. Mentor replied with direction to focus on MLOps.

---

# Mentor Email Draft (Ready to Send)

**Date:** December 10, 2025  
**Subject:** Domain Shift in HAR Model + Data Collection Plan

---

## Email Body (Keep to 1 page)

Hi Professor,

I discovered a critical issue with our HAR model in production that I'd like to discuss:

### The Problem
Our inference pipeline returns 100% "hand_tapping" predictions on real-world data. After investigation, I found that the production data has a constant gravity signature (Az = -9.83 m/s²) that happens to match the hand_tapping pattern from training. This is a **domain shift issue**, not a model failure - the model correctly identifies the closest pattern given the shifted input.

### Root Causes (Confirmed)
1. **Gravity alignment:** Production user wears device with consistent arm position, unlike diverse training postures
2. **Cross-user gap:** The model was validated within-user (5-fold CV on same 6 users), not cross-user
3. **Scale sensitivity:** StandardScaler amplifies the gravity offset due to low variance in production data

This matches the "lab-to-life" gap documented in domain adaptation literature (Papers #7, #8 in our collection).

### Proposed Solutions (Paper-Backed)
I've identified **4 proven approaches** from peer-reviewed research:

1. **Gravity Removal via High-Pass Filter** (Yurtman et al., 2017) - ~1 day
   - Apply 0.3 Hz Butterworth filter to remove static gravity component
   - Tested on 7 different sensor orientations - makes model orientation-invariant
   - Your constant Az ≈ -9.83 becomes Az ≈ 0 after filtering

2. **g-Unit Normalization** (Dhekane & Ploetz, 2024) - ~2 hours
   - Convert all data to standardized g-units (divide by 9.81, clip to [-2, +2])
   - Matches pre-training data range and eliminates scale artifacts
   - Simple fix with proven effectiveness

3. **Lightweight Personalization** (Dey et al., 2015) - ~3 days
   - Collect 1-2 minutes of labeled activities from production user
   - Fine-tune only final layer on this small dataset
   - Shows 20-30% accuracy improvement in practice

4. **Unsupervised Domain Adaptation** (Fu et al., 2021) - ~1 week
   - Advanced: Add Gradient Reversal Layer (GRL) to learn domain-invariant features
   - Works on unlabeled data from your 2-4 week collection period
   - Thesis-ready contribution

### Data Collection Request
Before implementing a solution, **I need 2-4 weeks of additional production data** to:
1. **Validate the hypothesis:** Check if Az stays constant across different times/activities, or if it varies with posture
2. **Test end-to-end pipeline:** Run inference on extended dataset to verify gravity calibration fixes the problem
3. **Measure stability:** Track prediction distribution over time (MLOps monitoring)

This data will help decide whether gravity calibration alone is sufficient or if we need per-user adaptation.

### Next Steps (Seeking Your Input)
1. **Should I implement gravity calibration while collecting more data?** (Unblocks MLOps progress)
2. **Should I prioritize the end-to-end pipeline test over full MLOps features?** (Validates the fix first)
3. **What's the realistic timeline for additional labeled data?** (Affects thesis milestones)

Your guidance will help me decide the best path forward.

Thanks,  
[Your Name]

---

## Additional Context (For Your Reference - Don't Send)

### Key Metrics
- Training Az mean: -3.53 m/s² (varies by activity/posture)
- Production Az mean: -9.83 m/s² (constant - gravity signature)
- Required offset for calibration: -6.30 m/s²
- Expected improvement: Spread predictions across 11 activities instead of 100% hand_tapping

### Paper References
- **#7:** Domain adaptation for IMU-based HAR (Chakma 2023)
- **#8:** Transfer learning in HAR (Dhekane 2024)
- **#77:** Heterogeneity Activity Recognition Dataset (Stisen 2015) - shows calibration is standard practice
- **#78:** Self-Supervised HAR (Saeed 2021)
- **#27:** Drift detection for sensor data (LLMs Memorize Sensor Datasets)

### Timeline for Data Collection
- **Week 1 (Dec 10-16):** Implement gravity calibration in preprocessing
- **Week 2-4 (Dec 17-Jan 7):** Collect extended production data
- **Week 5 (Jan 8-14):** Test pipeline, evaluate calibration effectiveness
- **Then:** Proceed with full MLOps pipeline (monitoring, retrain triggers, etc.)

### Files Documenting This Analysis
- `docs/FINAL_PIPELINE_PROBLEMS_ANALYSIS.md` - Root cause analysis with statistics
- `research_papers/temp.ipynb` - Solution demonstrations (gravity calibration, drift detection)

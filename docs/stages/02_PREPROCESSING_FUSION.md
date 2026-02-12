# Stage 1: Preprocessing & Sensor Fusion

**Pipeline Stage:** Transform raw sensor data into model-ready format  
**Input:** Raw accelerometer + gyroscope files  
**Output:** Fused, resampled, normalized windows (N, 200, 6)

---

## Key Questions

### Q1: How do we fuse accelerometer and gyroscope data?

**Process:**
```
RAW ACCEL (Ax, Ay, Az) ─┐
                        ├─▶ Timestamp align ─▶ Resample 50Hz ─▶ sensor_fused_50Hz.csv
RAW GYRO  (Gx, Gy, Gz) ─┘
```

**Steps:**
1. Parse timestamps to datetime (handle timezone)
2. Align streams using nearest-neighbor within ±1ms tolerance
3. Resample to uniform 50Hz using mean aggregation
4. Output: 7 columns (timestamp + 6 sensors)

---

### Q2: Gravity Removal — When and Why?

**The Problem:**
- Training data (ADAMSense) has gravity REMOVED
- Garmin raw data has gravity PRESENT (Az ≈ 9.81 m/s²)
- This mismatch causes model failure!

**Your Plan (4-6 production datasets):**

| Experiment | Gravity Removed? | Purpose |
|------------|------------------|---------|
| Baseline | No | See raw production performance |
| Experiment 1 | Yes (high-pass 0.3Hz) | Match ADAMSense preprocessing |
| Experiment 2 | Yes (high-pass 0.5Hz) | Test different cutoff |
| Compare | Both | Document accuracy difference |

**How to detect gravity presence:**
```python
# If Az mean is near 9.81, gravity is present
az_mean = data['Az'].mean()
if 8.0 < az_mean < 11.0:
    print("Gravity PRESENT - consider removal")
else:
    print("Gravity likely REMOVED or data is normalized")
```

**High-pass filter for gravity removal:**
- Cutoff: 0.3 Hz (gravity is DC component)
- Filter type: Butterworth, order 4
- Apply to Ax, Ay, Az only (not gyroscope)

---

### Q3: Normalization — Fit vs Transform

**CRITICAL RULE:** Never fit a new scaler on production data!

| Stage | Action | Scaler |
|-------|--------|--------|
| Training | `scaler.fit_transform(X_train)` | Save `scaler.pkl` |
| Production | `scaler.transform(X_prod)` | Load saved `scaler.pkl` |

**What goes wrong if you refit:**
- Production data gets different mean/std
- Normalized values don't match what model learned
- Model performance degrades silently

---

### Q4: Windowing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 200 samples | 4 seconds at 50Hz |
| Overlap | 50% | Data augmentation + no boundary loss |
| Stride | 100 samples | 2 seconds between window starts |

**Output shape:** `(N_windows, 200, 6)`

---

## Production vs Training Preprocessing

| Aspect | Training | Production |
|--------|----------|------------|
| Labels available | Yes | No |
| Output files | X_train.npy, y_train.npy | production_X.npy only |
| Scaler | Fit and save | Load and transform |
| Augmentation | Optional | Never |
| Split | Train/val/test | No splitting |
| Baseline stats | Compute and freeze | Compare against |

---

## What to Do Checklist

- [ ] Implement timestamp alignment with tolerance check
- [ ] Resample to exactly 50Hz (not approximate)
- [ ] Save scaler.pkl during training preprocessing
- [ ] Load same scaler for production (verify hash)
- [ ] Log gravity status in metadata JSON
- [ ] Run gravity removal experiments on 4-6 datasets
- [ ] Document accuracy with/without gravity removal

---

## Evidence from Papers

**[Recognition of Anxiety-Related Activities, ICTH 2025 | PDF: papers/papers needs to read/ICTH_16.pdf]**
- Window size 200 at 50Hz (4 seconds) is optimal for anxiety activities
- 50% overlap is standard practice
- Scaler must be frozen from training

**[Deep learning for sensor-based activity recognition | PDF: papers/research_papers/76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf]**
- Gravity removal is common but must match training preprocessing
- Mismatch is a primary cause of domain shift

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add scaler hash validation | Low | Prevent silent scaler mismatch |
| **HIGH** | Auto-detect gravity presence | Low | Warn if preprocessing mismatch |
| **MEDIUM** | Support configurable window sizes | Medium | Experiment with different windows |
| **MEDIUM** | Add interpolation options for gaps | Medium | Handle missing data better |
| **LOW** | Parallel processing for large datasets | High | Speed up preprocessing |

---

**Previous Stage:** [01_DATA_INGESTION.md](01_DATA_INGESTION.md)  
**Next Stage:** [03_QC_VALIDATION.md](03_QC_VALIDATION.md)

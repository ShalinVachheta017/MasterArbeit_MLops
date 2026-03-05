# Stage 2: Quality Control & Validation

**Pipeline Stage:** Validate data quality before inference  
**Input:** Preprocessed data (windows)  
**Output:** QC report (pass/fail with details)

---

## Key Question: QC Checks vs Unit Tests — Are They the Same?

**NO, they are DIFFERENT:**

| Aspect | QC Checks (Data Quality) | Unit Tests (Code Quality) |
|--------|--------------------------|---------------------------|
| **What they validate** | Data correctness | Code correctness |
| **When they run** | On every new data batch | On every code change (CI) |
| **Trigger** | New data arrives | Git push / PR |
| **Failure means** | Reject this data batch | Bug in code, fix it |
| **Framework** | Custom scripts (`preprocess_qc.py`) | pytest |
| **Location** | `scripts/preprocess_qc.py` | `tests/` folder |
| **Examples** | NaN check, range validation | Function returns expected value |

---

## QC Checks (Data Validation)

### Layer 1: Raw Data QC

| Check | Severity | Threshold | Action on Fail |
|-------|----------|-----------|----------------|
| Timestamp monotonic | CRITICAL | No backward jumps | Reject batch |
| Time gaps | WARNING | Gap > 40ms (2× expected) | Log, continue |
| Sensor range (accel) | CRITICAL | Outside ±160 m/s² | Reject batch |
| Sensor range (gyro) | CRITICAL | Outside ±35 rad/s | Reject batch |
| NaN/Inf values | CRITICAL | > 1% missing | Reject batch |
| Row count minimum | WARNING | < 1000 rows | Log warning |

### Layer 2: Fusion QC

| Check | Severity | Threshold | Action on Fail |
|-------|----------|-----------|----------------|
| Column schema | CRITICAL | Not exactly 7 columns | Reject |
| Column names | CRITICAL | Not [timestamp, Ax, Ay, Az, Gx, Gy, Gz] | Reject |
| Dtype validation | WARNING | Sensors not float | Convert + warn |
| Accel/Gyro row match | WARNING | > 5% difference | Log warning |

### Layer 3: Prepared Data QC

| Check | Severity | Threshold | Action on Fail |
|-------|----------|-----------|----------------|
| Window shape | CRITICAL | Not (N, 200, 6) | Reject |
| Normalized mean | CRITICAL | |mean| > 2.0 per channel | Reject |
| Normalized std | CRITICAL | std < 0.1 or > 3.0 | Reject |
| Variance collapse | CRITICAL | std < 0.1 (all channels) | Reject |
| Dtype | WARNING | Not float32 | Convert + warn |

---

## Unit Tests (Code Validation)

### What to Test with pytest

```python
# tests/test_preprocessing.py

def test_windowing_shape():
    """Window function produces correct shape."""
    data = np.random.randn(1000, 6)
    windows = create_windows(data, size=200, overlap=0.5)
    assert windows.shape[1] == 200
    assert windows.shape[2] == 6

def test_scaler_load():
    """Scaler loads correctly and transforms data."""
    scaler = load_scaler("models/scaler.pkl")
    test_data = np.random.randn(100, 6)
    transformed = scaler.transform(test_data)
    assert transformed.shape == test_data.shape

def test_gravity_filter():
    """Gravity filter removes DC component."""
    # Create signal with gravity (9.81 on z-axis)
    signal = np.zeros(1000) + 9.81
    filtered = apply_gravity_filter(signal, cutoff=0.3, fs=50)
    assert abs(filtered.mean()) < 0.5  # Near zero after filtering
```

### Test Categories

| Category | What it Tests | Example |
|----------|---------------|---------|
| **Unit tests** | Individual functions | `test_windowing_shape()` |
| **Integration tests** | Pipeline stages together | `test_preprocessing_pipeline()` |
| **Smoke tests** | Basic functionality | `test_model_loads()` |
| **Regression tests** | Known outputs match | `test_known_input_output()` |

---

## When to Run What

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Code Change ───▶ pytest (unit tests) ───▶ CI Pass/Fail        │
│                                                                  │
│  New Data ───▶ QC Checks ───▶ Pass? ───▶ Continue Pipeline     │
│                     │                                            │
│                     └───▶ Fail? ───▶ Reject Batch + Alert       │
│                                                                  │
│  Deployment ───▶ Smoke Tests ───▶ Pass? ───▶ Ready to Serve    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## QC Report Structure

```json
{
  "timestamp": "2026-01-15T14:12:13Z",
  "input_file": "data/prepared/production_X.npy",
  "overall_status": "PASS",
  "checks_passed": 5,
  "checks_failed": 0,
  "checks": [
    {
      "name": "Window shape correct",
      "passed": true,
      "severity": "CRITICAL",
      "message": "Shape (2609, 200, 6) matches expected",
      "details": {"expected": [null, 200, 6], "actual": [2609, 200, 6]}
    },
    {
      "name": "Normalized mean ≈ 0",
      "passed": true,
      "severity": "CRITICAL",
      "message": "All channel means within threshold",
      "details": {"mean_per_channel": [-0.01, 0.02, -0.03, 0.01, -0.02, 0.01]}
    }
  ]
}
```

---

## What to Do Checklist

- [ ] Run `scripts/preprocess_qc.py` on every new dataset
- [ ] Store QC reports in `reports/preprocess_qc/`
- [ ] Fail pipeline if CRITICAL checks fail
- [ ] Create unit tests in `tests/` folder
- [ ] Add pytest to CI pipeline (GitHub Actions)
- [ ] Implement smoke tests for deployment

---

## Evidence from Papers

**[Comparative Study on the Effects of Noise in HAR | PDF: papers/papers needs to read/Comparative Study on the Effects of Noise in.pdf]**
- Data quality significantly impacts model performance
- Range validation catches sensor failures early

**[Building Flexible, Scalable, ML-ready Multimodal Datasets | PDF: papers/research_papers/76 papers/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf]**
- Schema validation prevents downstream errors
- Automated QC enables scaling to many datasets

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add pytest to CI workflow | Low | Catch code bugs before merge |
| **HIGH** | Auto-reject on CRITICAL QC fail | Low | Prevent bad data in pipeline |
| **MEDIUM** | Add statistical QC (distribution tests) | Medium | Detect subtle data issues |
| **MEDIUM** | QC dashboard (visualize trends) | Medium | Monitor data quality over time |
| **LOW** | ML-based anomaly detection for QC | High | Catch complex anomalies |

---

**Previous Stage:** [02_PREPROCESSING_FUSION.md](02_PREPROCESSING_FUSION.md)  
**Next Stage:** [04_TRAINING_BASELINE.md](04_TRAINING_BASELINE.md)

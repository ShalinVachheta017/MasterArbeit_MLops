# Stage 4: Inference

> **✅ STATUS (2026-01-30): DONE - Core inference working!**
> 
> **ENHANCEMENTS TODO:**
> - [ ] Add MC Dropout (10 passes) for uncertainty quantification
> - [ ] Add AdaBN for test-time adaptation
> - [ ] Export uncertainty metrics (entropy, margin, mc_std)
> - [ ] Add Kalman filter tracking (per XAI-BayesHAR paper)

**Pipeline Stage:** Run model predictions on production data  
**Input:** Preprocessed production windows  
**Output:** Predictions, probabilities, confidence scores

---

## Inference Pipeline Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Load Model    │────▶│  Load Data     │────▶│  Predict       │
│  (.keras)      │     │  (.npy)        │     │  (batch)       │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Export CSV    │◀────│  Compute       │◀────│  Get Probs     │
│  + Metadata    │     │  Confidence    │     │  (softmax)     │
└────────────────┘     └────────────────┘     └────────────────┘
```

---

## Key Operations

### Model Loading

```python
# Expected model specs
model_path = "models/pretrained/fine_tuned_model_1dcnnbilstm.keras"
input_shape = (None, 200, 6)   # (batch, timesteps, channels)
output_shape = (None, 11)      # (batch, classes)
parameters = 499_131
```

### Batch Prediction

```python
# Load and predict
X = np.load("data/prepared/production_X.npy")  # (N, 200, 6)
probabilities = model.predict(X, batch_size=32)  # (N, 11)
predictions = np.argmax(probabilities, axis=1)   # (N,)
confidence = np.max(probabilities, axis=1)       # (N,)
```

### Output Files

| File | Contents | Purpose |
|------|----------|---------|
| `predictions_{timestamp}.csv` | window_id, prediction, confidence | Main results |
| `predictions_{timestamp}_probs.npy` | Full probability matrix | Detailed analysis |
| `predictions_{timestamp}_metadata.json` | Model version, timestamps, stats | Audit trail |

---

## Inference Metadata Structure

```json
{
  "inference_timestamp": "2026-01-06T11:51:43Z",
  "model_path": "models/pretrained/fine_tuned_model_1dcnnbilstm.keras",
  "model_hash": "sha256:abc123...",
  "input_file": "data/prepared/production_X.npy",
  "input_hash": "sha256:def456...",
  "n_windows": 1815,
  "inference_time_seconds": 2.04,
  "throughput_windows_per_sec": 888.3,
  "confidence_distribution": {
    "high_90plus": 1808,
    "moderate_70_90": 5,
    "low_50_70": 2,
    "uncertain_below_50": 0
  },
  "activity_distribution": {
    "hand_tapping": 1815
  },
  "mlflow_run_id": "62435f47bef54ac9840ef3e3b413b3e9"
}
```

---

## What to Log in MLflow

| Metric | Value Type | Purpose |
|--------|------------|---------|
| `n_windows` | int | Input size |
| `inference_time_sec` | float | Performance |
| `throughput_wps` | float | Performance |
| `mean_confidence` | float | Model certainty |
| `uncertain_ratio` | float | Proportion low-confidence |
| `activity_distribution` | dict | Class balance |

**Artifacts:**
- Predictions CSV
- Probabilities NPY
- Metadata JSON

---

## Smoke Tests for Inference

Run these before every production inference:

```python
# Smoke Test 1: Model loads
def test_model_loads():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model.input_shape == (None, 200, 6)
    assert model.output_shape == (None, 11)

# Smoke Test 2: Data loads
def test_data_loads():
    X = np.load(DATA_PATH)
    assert X.ndim == 3
    assert X.shape[1] == 200
    assert X.shape[2] == 6
    assert not np.any(np.isnan(X))

# Smoke Test 3: Inference works
def test_inference_works():
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.random.randn(1, 200, 6).astype(np.float32)
    output = model.predict(dummy_input)
    assert output.shape == (1, 11)
    assert np.allclose(output.sum(), 1.0)  # Valid probability distribution
```

---

## What to Do Checklist

- [ ] Run smoke tests before inference
- [ ] Log input/model hashes for traceability
- [ ] Compute and store full probability matrix
- [ ] Calculate confidence distribution summary
- [ ] Export to MLflow with run tagging
- [ ] Store predictions with timestamps for audit

---

## Evidence from Papers

**[Recognition of Anxiety-Related Activities, ICTH 2025 | PDF: papers/papers needs to read/ICTH_16.pdf]**
- 1DCNN-BiLSTM achieves 888+ windows/sec throughput
- 11-class classification for anxiety-related activities

**[Practical MLOps: Operationalizing ML Models | PDF: papers/mlops_production/Practical-mlops-operationalizing-machine-learning-models.pdf]**
- Smoke tests are essential before production inference
- Model and data hashes enable audit trails

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add model hash validation before inference | Low | Prevent wrong model usage |
| **HIGH** | Implement automatic smoke test runner | Low | Fail fast on issues |
| **MEDIUM** | Add streaming inference mode | Medium | Support real-time use case |
| **MEDIUM** | Model warm-up for consistent latency | Low | Stable performance |
| **LOW** | GPU acceleration support | High | Faster inference |

---

**Previous Stage:** [04_TRAINING_BASELINE.md](04_TRAINING_BASELINE.md)  
**Next Stage:** [06_MONITORING_DRIFT.md](06_MONITORING_DRIFT.md)

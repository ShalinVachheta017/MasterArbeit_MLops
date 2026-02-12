<<<<<<< HEAD
  # 📚 Key Concepts Explained
=======
   # 📚 Key Concepts Explained
>>>>>>> 8632082 (Complete 10-stage MLOps pipeline with AdaBN domain adaptation)

## 1️⃣ Units: milliG vs m/s²

### What is milliG?
- **Full name:** milligravity
- **Definition:** 1 milliG = 1/1000 of Earth's gravity
- **Symbol:** mG or milliG
- **Used by:** Garmin watches, fitness trackers, consumer devices

**Example:**
```
Earth's gravity at rest = 1000 milliG (1G)
Az (vertical axis) at rest = -1000 milliG (pointing down)
```

### What is m/s²?
- **Full name:** meters per second squared
- **Definition:** Standard unit of acceleration in physics
- **Symbol:** m/s²
- **Used by:** Scientific research, standard datasets, ML models

**Example:**
```
Earth's gravity = 9.81 m/s²
Az (vertical axis) at rest = -9.81 m/s² (pointing down)
```

### Conversion Formula
```python
acceleration_ms2 = acceleration_milliG × 0.00981

# Why 0.00981?
# 1 milliG = 0.001 G
# 1 G = 9.81 m/s²
# Therefore: 1 milliG = 0.001 × 9.81 = 0.00981 m/s²
```

### Why This Matters
| Scenario | Units | Problem if Wrong |
|----------|-------|------------------|
| **Training Data** | m/s² | Model learns patterns in m/s² scale |
| **Production Data (Garmin)** | milliG | 100× larger values! |
| **If not converted** | Mixed | ❌ Model sees values 100× larger than training → wrong predictions |

**Example of mismatch:**
```
Training Az: -9.81 m/s² (gravity)
Production Az: -1000 milliG (same gravity, but 100× larger number)
Model thinks: "This is 100× stronger than gravity!" → incorrect prediction
```

---

## 2️⃣ Training Data: `all_users_data_labeled.csv`

### Location
```
data/raw/all_users_data_labeled.csv
```

### What it contains
- **Multiple users:** Combined data from several participants
- **Labeled activities:** Each row has an activity label (walking, running, sitting, etc.)
- **6 sensors:** Ax, Ay, Az, Gx, Gy, Gz
- **Units:** m/s² (standard scientific units)
- **Sampling rate:** 50Hz (50 samples per second)

### Purpose
- **Model training:** This is what the model learned from
- **Ground truth:** Activities are labeled by humans
- **Reference:** Production data must match this format

### Why we reference it
```python
# During training:
scaler = StandardScaler()
scaler.fit(training_data)  # Learn mean and std from training data
scaler.save()

# During production:
scaler = load_scaler()  # Use SAME scaler
production_normalized = scaler.transform(production_data)
```

**Critical:** Production data must be transformed using the SAME scaler that was fit on training data!

---

## 3️⃣ Windows: Breaking Time Series into Chunks

### What is a Window?

A **window** is a **fixed-size chunk of continuous sensor data** used as a single input to the model.

**Analogy:** Imagine watching a video:
- Video = continuous sensor stream (181,699 samples)
- Window = 4-second clip from the video
- Model analyzes one clip at a time to predict activity

### Window Parameters

#### Window Size = 200 samples

**What it means:**
```
200 samples × (1/50 Hz) = 4 seconds of data
```

**Why 200?**
- ✅ Long enough to capture activity patterns (walking has ~2 steps/second)
- ✅ Short enough to detect activity changes quickly
- ✅ Model architecture expects 200 timesteps
- ✅ Typical choice in activity recognition research

**What a window looks like:**
```python
Window shape: (200, 6)
200 timesteps × 6 sensors

        Ax    Ay    Az    Gx    Gy    Gz
t=0    0.5  -0.2   9.8   0.1   0.0  -0.1
t=1    0.6  -0.3   9.7   0.2   0.1  -0.2
t=2    0.4  -0.1   9.9   0.1  -0.1  -0.1
...
t=199  0.5  -0.2   9.8   0.0   0.1   0.0
```

#### Overlap = 50%

**What it means:**
- Each window shares 50% of data with the next window
- Step size = 100 samples (50% of 200)

**Visualization:**
```
Continuous data: [====================181,699 samples====================]

Window 1:  [########]
           0       199

Window 2:      [########]
              100       299

Window 3:          [########]
                  200       399
```

**Why overlap?**
- ✅ Captures transitions between activities
- ✅ More training/inference samples
- ✅ Smoother predictions over time
- ✅ Don't miss activity changes that happen between windows

**Example without overlap (BAD):**
```
Window 1: [Standing...Standing]
Window 2: [Running...Running]
Missed: The moment person started running!
```

**Example with 50% overlap (GOOD):**
```
Window 1: [Standing...Standing]
Window 2: [Standing...Starting to run]  ← Captures transition
Window 3: [Running...Running]
```

#### Step = 100 samples

**What it means:**
- Move forward 100 samples to start next window
- Step = Window_size × (1 - Overlap)
- Step = 200 × 0.5 = 100

**Calculation:**
```python
# From 181,699 samples
n_windows = (181699 - 200) // 100 + 1
n_windows = 1817 windows (approximately)
```

### Hertz (Hz) = Sampling Rate

**Definition:** Number of samples per second

**50 Hz means:**
- 50 samples every second
- 1 sample every 0.02 seconds (20 milliseconds)

**Why 50 Hz?**
- ✅ Captures human movement (walking ~2 Hz, running ~3 Hz)
- ✅ Standard in activity recognition research
- ✅ Balance between detail and data size
- ✅ Nyquist theorem: Need 2× frequency to capture signal

**Example:**
```
Human activities frequency:
- Walking: ~2 Hz (2 steps/second)
- Running: ~3 Hz (3 steps/second)
- Breathing: ~0.3 Hz

50 Hz captures all of these well (25× faster than walking)
```

### How Windows Affect the Model

#### Input Shape
```python
Model expects: (batch_size, 200, 6)
- batch_size: number of windows
- 200: timesteps per window
- 6: features (sensors)
```

#### Output Shape
```python
Model outputs: (batch_size, 11)
- batch_size: number of windows
- 11: activity classes (probabilities)

Example output for 1 window:
[0.05, 0.02, 0.01, 0.85, 0.01, 0.02, 0.01, 0.01, 0.01, 0.00, 0.01]
        ↑
    Running (85% confidence)
```

#### Why This Architecture?
- **LSTM layers** need fixed-size sequences
- **CNN layers** extract local patterns
- **200 timesteps** is the compromise between:
  - Too short: Miss activity patterns
  - Too long: Activities might change within window

---

## 4️⃣ StandardScaler: Why Same Scaler as Training?

### What is StandardScaler?

**Formula:**
```python
normalized = (value - mean) / std

# Example:
value = 5.0
mean = 3.0 (from training data)
std = 2.0 (from training data)
normalized = (5.0 - 3.0) / 2.0 = 1.0
```

### How Scaler is Created During Training

```python
# Step 1: Fit scaler on training data
training_data = load_training_data()  # all_users_data_labeled.csv
scaler = StandardScaler()
scaler.fit(training_data)

# This calculates:
scaler.mean_ = training_data.mean(axis=0)  # Mean per sensor
scaler.scale_ = training_data.std(axis=0)  # Std per sensor

# Example values:
scaler.mean_ = [0.12, -0.08, 9.81, 0.05, -0.02, 0.01]  # For Ax,Ay,Az,Gx,Gy,Gz
scaler.scale_ = [2.34, 1.98, 0.87, 15.2, 18.3, 12.1]

# Step 2: Save scaler
with open('config.json', 'w') as f:
    json.dump({
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }, f)
```

### Why Production Must Use SAME Scaler

**Scenario 1: Using SAME scaler (CORRECT) ✅**
```python
# Training
training_value = 5.0
training_mean = 3.0
training_std = 2.0
normalized_train = (5.0 - 3.0) / 2.0 = 1.0

# Production
production_value = 5.0  # Same raw value
production_mean = 3.0   # Same mean from training
production_std = 2.0    # Same std from training
normalized_prod = (5.0 - 3.0) / 2.0 = 1.0  ✅ Same normalized value!
```

**Scenario 2: Fitting NEW scaler on production (WRONG) ❌**
```python
# Training
normalized_train = (5.0 - 3.0) / 2.0 = 1.0

# Production (fitted on production data)
production_mean = 7.0   # Different! Production data has different distribution
production_std = 4.0    # Different!
normalized_prod = (5.0 - 7.0) / 4.0 = -0.5  ❌ WRONG! Model doesn't recognize this
```

### Real Example

**Training data (Az at rest):**
```python
mean = 9.81 m/s²
std = 0.5 m/s²

Az_normalized = (9.8 - 9.81) / 0.5 = -0.02  ← Model learned "standing" is around -0.02
```

**Production data (using SAME scaler):**
```python
mean = 9.81 m/s²  (from training)
std = 0.5 m/s²    (from training)

Az_normalized = (9.85 - 9.81) / 0.5 = 0.08  ← Model recognizes this as "standing"
```

**Production data (if we fit NEW scaler):**
```python
mean = 10.2 m/s²  (production might be different!)
std = 0.8 m/s²

Az_normalized = (9.85 - 10.2) / 0.8 = -0.44  ← Model thinks this is something else!
```

### Where Scaler is Stored

**Location:**
```
data/prepared/config.json
```

**Content:**
```json
{
  "scaler_mean": [0.12, -0.08, 9.81, 0.05, -0.02, 0.01],
  "scaler_scale": [2.34, 1.98, 0.87, 15.2, 18.3, 12.1]
}
```

**Loading in production:**
```python
with open('config.json', 'r') as f:
    config = json.load(f)

mean = np.array(config['scaler_mean'])
scale = np.array(config['scaler_scale'])

# Transform production data
production_normalized = (production_data - mean) / scale
```

---

## 5️⃣ Why .npy Format?

### What is .npy?

**Full name:** NumPy binary format

**Purpose:** Efficiently store NumPy arrays

### Comparison with Other Formats

| Format | Size | Load Speed | Preserves Structure | Use Case |
|--------|------|------------|---------------------|----------|
| **.npy** | 8.5 MB | ⚡ Fast (0.1s) | ✅ Yes | Production, model input |
| **.csv** | 45 MB | 🐌 Slow (2s) | ❌ No (loses shape) | Human reading, sharing |
| **.pkl** | 9 MB | ⚡ Fast (0.2s) | ✅ Yes | General Python objects |
| **.h5** | 8 MB | ⚡ Fast (0.15s) | ✅ Yes | Large datasets, complex structures |

### Why .npy for production_X?

**1. Preserves Exact Shape**
```python
# Save
X = np.array([...])  # Shape: (1772, 200, 6)
np.save('production_X.npy', X)

# Load
X_loaded = np.load('production_X.npy')
print(X_loaded.shape)  # (1772, 200, 6) ✅ Exact same shape!
```

**2. Fast Loading**
```python
# CSV (SLOW)
import pandas as pd
df = pd.read_csv('production.csv')  # 2 seconds
X = df.values.reshape(1772, 200, 6)  # Need manual reshaping

# NPY (FAST)
X = np.load('production_X.npy')  # 0.1 seconds ⚡
```

**3. Smaller File Size**
```
CSV:  45 MB (text format, lots of repeated characters)
NPY:  8.5 MB (binary, compact)
Savings: 81% smaller!
```

**4. Direct Model Input**
```python
# No conversion needed!
X = np.load('production_X.npy')
predictions = model.predict(X)  # ✅ Direct input
```

**5. Type Safety**
```python
# NPY preserves dtype
X = np.array([1.5, 2.3], dtype=np.float32)
np.save('data.npy', X)
X_loaded = np.load('data.npy')
print(X_loaded.dtype)  # float32 ✅

# CSV loses dtype
# Everything becomes string, needs conversion
```

### When to Use Each Format

| Scenario | Best Format | Why |
|----------|-------------|-----|
| Model inference | .npy | Fast, preserves shape |
| Human inspection | .csv | Readable in Excel |
| Sharing with non-Python | .csv | Universal format |
| Large datasets (>1GB) | .h5 | Compressed, chunks |
| Mixed data types | .pkl | Handles any Python object |

---

## 6️⃣ Data Drift: How to Detect

### What is Data Drift?

**Definition:** When production data distribution differs from training data distribution.

**Example:**
```
Training: Users aged 20-30, walking on flat ground
Production: Users aged 60-70, walking upstairs
→ Different patterns! Model might fail.
```

### Key Metrics for Drift Detection

#### 1. Statistical Drift

**Mean Drift:**
```python
drift_mean = |production_mean - training_mean|

# Threshold: > 0.1 means drift
if drift_mean > 0.1:
    print("⚠️ Mean has drifted!")
```

**Example:**
```
Training Az mean: 9.81 m/s²
Production Az mean: 10.2 m/s²
Drift: |10.2 - 9.81| = 0.39 > 0.1  ⚠️ DRIFT DETECTED
```

**Standard Deviation Drift:**
```python
drift_std = |production_std - training_std|

# Threshold: > 0.15 means drift
if drift_std > 0.15:
    print("⚠️ Variability has changed!")
```

#### 2. Distribution Drift (KS Test)

**Kolmogorov-Smirnov Test:**
```python
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(training_data, production_data)

if p_value < 0.05:
    print("⚠️ Distributions are significantly different!")
```

**What it tests:**
- Are two datasets from the same distribution?
- p_value < 0.05 → distributions differ

#### 3. Range Drift

**Check min/max:**
```python
# Training range
train_min, train_max = -15, 15

# Production has values outside
prod_min, prod_max = -20, 25  ⚠️ Outside training range!
```

**Why this matters:**
```
Model never saw values > 15 during training
Production has values of 25
→ Model is extrapolating (less reliable)
```

#### 4. Per-Sensor Drift

**Check each sensor separately:**
```python
sensors = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

for sensor in sensors:
    train_mean = training[sensor].mean()
    prod_mean = production[sensor].mean()
    drift = abs(prod_mean - train_mean)
    
    if drift > 0.1:
        print(f"⚠️ {sensor} drifted by {drift:.3f}")
```

**Example output:**
```
✓ Ax: drift = 0.05 (OK)
✓ Ay: drift = 0.03 (OK)
⚠️ Az: drift = 0.45 (HIGH!)
✓ Gx: drift = 0.08 (OK)
⚠️ Gy: drift = 0.52 (HIGH!)
✓ Gz: drift = 0.09 (OK)
```

### Drift Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Mean drift | > 0.1 | Significant shift |
| Std drift | > 0.15 | Variability changed |
| KS p-value | < 0.05 | Different distributions |
| Range outside | Any | Extrapolation risk |

### What to Do When Drift Detected

1. **Investigate cause:**
   - Different user demographics?
   - Different device/sensor?
   - Different environment?

2. **Check prediction quality:**
   - If predictions still good → might be OK
   - If predictions bad → need action

3. **Actions:**
   - **Retrain model** with new data
   - **Recalibrate** normalization
   - **Adjust** preprocessing pipeline
   - **Collect more** training data

### Example Drift Analysis

**Training data:**
```
Az: mean=9.81, std=0.50, range=[-12, 15]
```

**Production data:**
```
Az: mean=10.2, std=0.85, range=[-18, 25]
```

**Analysis:**
```
Mean drift: |10.2 - 9.81| = 0.39 > 0.1  ⚠️ HIGH
Std drift: |0.85 - 0.50| = 0.35 > 0.15  ⚠️ HIGH
Range: [-18, 25] exceeds [-12, 15]      ⚠️ EXTRAPOLATION

Conclusion: Significant drift detected!
Action needed: Retrain or recalibrate
```

---

**Last Updated:** December 7, 2025

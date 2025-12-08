# 📓 Notebooks Overview

This folder contains all Jupyter notebooks for the MLOps project. Each notebook serves a specific purpose in the data processing, analysis, and model workflow.

---

## 🔄 Complete Data Pipeline Flow

```
📁 data/raw/
├── accelerometer_data.xlsx  (Raw Excel from Garmin)
├── gyroscope_data.xlsx      (Raw Excel from Garmin)
    ↓
    ↓ STEP 1: Combine & Fuse Sensors
    ↓
[data_preprocessing_step1.ipynb]
    ↓
📁 data/processed/
└── sensor_fused_50Hz.csv    (Combined CSV with all 6 sensors)
    ↓
    ↓ STEP 2: Unit Detection, Windowing, Normalization
    ↓
[production_preprocessing.ipynb]
    ↓
📁 data/prepared/
├── production_X.npy         (Windowed arrays for model)
└── production_metadata.json (Pipeline info)
    ↓
    ↓ Model Inference
    ↓
🤖 Predictions (Activity Classification)
```

---

## 📋 Notebook Inventory

| # | Notebook | Purpose | Status | Order |
|---|----------|---------|--------|-------|
| 1 | `data_preprocessing_step1.ipynb` | Combine raw Excel → CSV | ✅ **STEP 1** | Run First |
| 2 | `production_preprocessing.ipynb` | CSV → Model-ready arrays | ✅ **STEP 2** | Run Second |
| 3 | `data_comparison.ipynb` | Compare training vs production data | ✅ Validation | After Step 2 |
| 4 | `from_guide_processing.ipynb` | Initial processing experiments | 📝 Reference | - |
| 5 | `scalable.ipynb` | Scalability experiments | 🔬 Experimental | - |

---

## 📖 Detailed Descriptions

### 1️⃣ `data_preprocessing_step1.ipynb` (formerly `dp.ipynb`)

**Purpose:** STEP 1 - Combine raw accelerometer and gyroscope Excel files into single CSV.

**What it does:**
- Loads raw Excel files from Garmin watch:
  - `accelerometer_data.xlsx` (Ax, Ay, Az with timestamps)
  - `gyroscope_data.xlsx` (Gx, Gy, Gz with timestamps)
- Merges both files based on timestamps
- Resamples to 50Hz (uniform sampling rate)
- Handles missing data and synchronization
- Exports combined CSV with all 6 sensors

**When to use:**
- After running `data_preprocessing_step1.ipynb`
- Before running model inference
- When you have `sensor_fused_50Hz.csv` ready

**Input:**
- `data/processed/sensor_fused_50Hz.csv` (from Step 1)

**Output:**
- `data/prepared/production_X.npy` - **Windowed arrays for model**
  - Shape: (N_windows, 200, 6)
  - N_windows ≈ 1,772 windows
  - 200 = timesteps per window (4 seconds @ 50Hz)
  - 6 = sensors (Ax, Ay, Az, Gx, Gy, Gz)
  - Normalized using training scaler
- `data/prepared/production_metadata.json` (pipeline info)

**What is production_X.npy?**
- **Windowed data:** Model expects fixed-size windows (200 timesteps)
- **Normalized:** Values scaled to match training distribution
- **Ready for inference:** Can be directly fed to model.predict()
- **Format:** NumPy array saved as .npy file

**Key Features:**
- ✅ Automatic unit detection (milliG vs m/s²)
- ✅ Sliding windows with 50% overlap
- ✅ StandardScaler normalization (from training)
- ✅ Comprehensive validation
- ✅ Production-ready

**Status:** ✅ **REQUIRED** - Run after Step 1!tion

**Status:** ✅ **REQUIRED** - Run this first!

---

### 2️⃣ `production_preprocessing.ipynb`

**Purpose:** STEP 2 - Transform CSV into model-ready windowed arrays.

**What it does:**
- Loads production CSV data (`sensor_fused_50Hz.csv`)
- **Automatic unit detection** (milliG vs m/s²)
- Converts milliG → m/s² using factor `0.00981`
- Handles NaN values (ffill + bfill)
- Normalizes using training StandardScaler
- Creates sliding windows (200 samples, 50% overlap)
- Exports `production_X.npy` for model inference

**When to use:**
- Preprocessing new production sensor data
- Before running model inference
- When you need to convert Garmin milliG data
---

### 3️⃣ `data_comparison.ipynb`50Hz.csv`

**Output:**
- `data/prepared/production_X.npy` (windowed array)
- `data/prepared/production_metadata.json` (pipeline info)

**Key Features:**
- ✅ Automatic unit detection
- ✅ Comprehensive validation
- ✅ Detailed step-by-step documentation
- ✅ Production-ready

---

### 2️⃣ `data_comparison.ipynb`

**Purpose:** Compare statistical properties of training vs production datasets.

**What it does:**
- Loads both training and production data
- Compares distributions (mean, std, min, max, percentiles)
- Detects data drift between datasets
- Validates preprocessing consistency
- Generates comparison reports

**When to use:**
- After preprocessing production data
- To validate data quality
- Before running inference (check for drift)
- Debugging prediction issues

**Input:**
- `data/prepared/train_X.npy` (training data)
- `data/prepared/production_X.npy` (production data)

**Output:**
- Comparison tables
- Drift metrics
- Distribution plots
- Validation reports

**Key Metrics:**
- Mean/std drift detection
- Per-sensor statistical comparison
- Data quality warnings

---

### 4️⃣ `from_guide_processing.ipynb`

**Purpose:** Initial preprocessing experiments based on tutorial/guide.

**What it does:**
- Contains early preprocessing experiments
- Shows initial approaches to data processing
- Documents learning process
- Reference implementation

**When to use:**
- Understanding initial preprocessing approach
- Comparing with current pipeline
- Educational reference
- Historical context

**Status:** 📝 Reference/Archive
- Early-stage experiments
- Not production-ready
- Kept for reference
- Use `production_preprocessing.ipynb` instead

---

### 5️⃣ `scalable.ipynb`

**Purpose:** Experiments with scalable processing approaches.

**What it does:**
- Tests scalability of preprocessing pipeline
- Explores batch processing
- Performance optimization experiments
- Large-scale data handling

**When to use:**
- When processing very large datasets
- Performance optimization research
- Batch processing experiments
- Scaling production pipeline

**Status:** 🔬 Experimental
- Research/development stage
- Not production-ready
- Active experimentation
- Future scalability improvements

---

## 🚀 Quick Start Guide

### For Production Use

1. **Preprocess production data:**
   ```bash
   Open: production_preprocessing.ipynb
   Run all cells
   ```

2. **Validate data quality:**
   ```bash
   Open: data_comparison.ipynb
   Run all cells
   Check drift metrics
   ```

3. **Run model inference:**
   ```bash
   Load: data/prepared/production_X.npy
   Predict with trained model
   ```

### For Exploration/Learning

- **Understanding Step 1 (fusion):** Check `data_preprocessing_step1.ipynb`
- **Understanding Step 2 (windowing):** Check `production_preprocessing.ipynb`
- **Learning initial approach:** See `from_guide_processing.ipynb`
- **Scalability research:** Explore `scalable.ipynb`

**STEP 3: Validate Data Quality**
```bash
1. Open: data_comparison.ipynb
2. Run all cells
3. Check drift metrics
4. Ensure data quality ✓
```

**STEP 4: Run Model Inference**
```python
import numpy as np
from tensorflow import keras

# Load preprocessed data
X = np.load('data/prepared/production_X.npy')
print(f"Shape: {X.shape}")  # (1772, 200, 6)

# Load trained model
model = keras.models.load_model('models/pretrained/fine_tuned_model_1dcnnbilstm.keras')

# Run inference
predictions = model.predict(X)
print(f"Predictions shape: {predictions.shape}")  # (1772, 11)
```

---

## 🔑 Key Concepts

### Unit Detection & Conversion
- **Problem:** Garmin data often in milliG, training data in m/s²
- **Solution:** Automatic detection + conversion (factor: 0.00981)
- **Validation:** Check Az ≈ -9.8 m/s² (Earth's gravity)

### Sliding Windows
- **Window size:** 200 samples (4 seconds @ 50Hz)
- **Overlap:** 50% (100 samples)
- **Purpose:** Model expects fixed-size input

### Normalization
- **Method:** StandardScaler from training
- **Critical:** Must use same scaler as training!
- **Location:** `data/prepared/config.json`

## 🔑 Key Concepts

## 📁 Related Files

| Path | Description | Created By |
|------|-------------|------------|
| `data/raw/accelerometer_data.xlsx` | Raw Garmin accelerometer | Manual export |
| `data/raw/gyroscope_data.xlsx` | Raw Garmin gyroscope | Manual export |
| `data/processed/sensor_fused_50Hz.csv` | Combined sensor CSV | Step 1 notebook |
| `data/prepared/production_X.npy` | Windowed arrays for model | Step 2 notebook |
| `data/prepared/production_metadata.json` | Pipeline metadata | Step 2 notebook |
| `data/prepared/config.json` | Training scaler parameters | Training pipeline |
| `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` | Trained model | Training pipeline |
| `src/preprocess_data.py` | Python script version (Step 2) | Manual coding |
| `src/compare_data.py` | Data comparison script | Manual coding |
### What is production_X.npy?
**Format:** NumPy array saved as binary file (.npy)

**Structure:**
```python
Shape: (N_windows, 200, 6)
- N_windows ≈ 1,772  # Number of 4-second windows
- 200 = timesteps    # 4 seconds @ 50Hz sampling
- 6 = features       # Ax, Ay, Az, Gx, Gy, Gz
```

**Why windowed?**
- Model trained on 4-second activity windows
- Sliding window with 50% overlap captures temporal patterns
- Fixed size required by LSTM layers

**Why .npy format?**
- Efficient binary storage (smaller than CSV)
- Fast loading with numpy.load()
- Preserves exact array structure and dtype

### Unit Detection & Conversion

| Path | Description |
|------|-------------|
| `data/processed/sensor_fused_50Hz.csv` | Raw production data |
| `data/prepared/production_X.npy` | Preprocessed windows |
| `data/prepared/production_metadata.json` | Pipeline metadata |
| `data/prepared/config.json` | Training scaler parameters |
| `src/preprocess_data.py` | Python script version of preprocessing |
| `src/compare_data.py` | Python script for data comparison |
## 🛠️ Maintenance

### Active Notebooks (Keep Updated)
- ✅ `data_preprocessing_step1.ipynb` - STEP 1 (sensor fusion)
- ✅ `production_preprocessing.ipynb` - STEP 2 (windowing)
- ✅ `data_comparison.ipynb` - Validation

### Reference Notebooks (Historical)
- 📝 `from_guide_processing.ipynb` - Keep for reference

### Experimental Notebooks (May Change)
- 🔬 `scalable.ipynb` - Active developmentfor reference

### Experimental Notebooks (May Change)
- 🔬 `scalable.ipynb` - Active development

---

## 📚 Additional Resources

- **Project Documentation:** `docs/`
- **Source Code:** `src/`
- **Model Files:** `models/pretrained/`
- **Configuration:** `config/`
## ⚠️ Important Notes

1. **Follow the two-step pipeline in order:**
   - ✅ STEP 1: `data_preprocessing_step1.ipynb` (Excel → CSV)
   - ✅ STEP 2: `production_preprocessing.ipynb` (CSV → .npy)
   - Don't skip Step 1!

2. **Raw data format:**
   - Must be separate Excel files (accelerometer + gyroscope)
   - With timestamps for synchronization
   - From Garmin watch export

3. **Check data comparison after preprocessing**
   - Run `data_comparison.ipynb` to validate data quality
   - Look for drift warnings

4. **Unit conversion is critical (Step 2)**
   - Garmin data often in milliG
   - Automatic detection and conversion to m/s²
   - Must match training data units

5. **Don't modify training scaler**
   - Production must use same scaler as training
   - Located in `data/prepared/config.json`

6. **production_X.npy is for model inference**
   - Direct input to model.predict()
   - Already windowed (200 samples per window)
   - Already normalized (using training scaler)
   - Don't need further preprocessing
4. **Don't modify training scaler**
   - Production must use same scaler as training
   - Located in `data/prepared/config.json`

---

**Last Updated:** December 7, 2025  
**Maintained By:** MLOps Pipeline Team

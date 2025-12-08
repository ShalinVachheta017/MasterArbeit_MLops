# 🐍 Source Code (src/) - Python Scripts

This folder contains Python scripts for the MLOps pipeline. These are **command-line versions** of the Jupyter notebooks, designed for automation and production use.

---

## 📋 File Inventory

| # | File | Purpose | Status | Logging |
|---|------|---------|--------|---------|
| 1 | `config.py` | Path configuration | ✅ Active | N/A |
| 2 | `sensor_data_pipeline.py` | Raw Excel → CSV (Step 1) | ✅ Active | ✅ `logs/preprocessing/` |
| 3 | `preprocess_data.py` | CSV → .npy windows (Step 2) | ✅ Active | ✅ `logs/preprocessing/` |
| 4 | `compare_data.py` | Data drift detection | ✅ Active | ✅ `logs/comparison/` |
| 5 | `run_inference.py` | Model inference | ✅ **NEW** | ✅ `logs/inference/` |
| 6 | `evaluate_predictions.py` | Prediction analysis | ✅ **NEW** | ✅ `logs/evaluation/` |
| 7 | Archived scripts | Old preprocessing versions | 📁 Archive | N/A |

---

## 📁 Logging Structure

All scripts generate timestamped log files:

```
logs/
├── preprocessing/     # sensor_data_pipeline.py + preprocess_data.py
│   └── preprocessing_20251208_143052.log
├── comparison/        # compare_data.py
│   └── comparison_20251208_143055.log
├── inference/         # run_inference.py
│   └── inference_20251208_143100.log
└── evaluation/        # evaluate_predictions.py
    └── evaluation_20251208_143105.log
```

**Log Format:**
```
2025-12-08 14:30:52 | INFO     | ✅ Loaded 1772 windows
2025-12-08 14:30:53 | WARNING  | ⚠️ Low confidence detected
2025-12-08 14:30:54 | ERROR    | ❌ File not found
```

---

## 🔄 Pipeline Flow

```
1. sensor_data_pipeline.py
   ├── Input: accelerometer_data.xlsx + gyroscope_data.xlsx
   └── Output: sensor_fused_50Hz.csv

2. preprocess_data.py  
   ├── Input: sensor_fused_50Hz.csv
   └── Output: production_X.npy (windowed, normalized)

3. run_inference.py  ← NEW
   ├── Input: production_X.npy + model.keras
   └── Output: predictions.csv + predictions_metadata.json

4. evaluate_predictions.py  ← NEW
   ├── Input: predictions.csv
   └── Output: evaluation report (confidence, distribution)

5. compare_data.py
   ├── Input: production_X.npy + train_X.npy
   └── Output: Drift report
```

---

## 📖 Detailed File Descriptions

### 1️⃣ `config.py`

**Purpose:** Central configuration for all file paths

**What it defines:**
```python
PROJECT_ROOT = /path/to/MasterArbeit_MLops

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_PREPARED = DATA_DIR / "prepared"

# Model paths
MODELS_PRETRAINED = MODELS_DIR / "pretrained"
PRETRAINED_MODEL = MODELS_PRETRAINED / "fine_tuned_model_1dcnnbilstm.keras"

# Config files
SCALER_CONFIG = DATA_PREPARED / "config.json"
```

**Why important:**
- ✅ Single source of truth for paths
- ✅ All scripts import from here
- ✅ Easy to update paths project-wide
- ✅ No hardcoded paths in code

**When to modify:**
- Moving project to different location
- Changing folder structure
- Adding new path constants

**Usage:**
```python
from config import DATA_PROCESSED, PRETRAINED_MODEL

# Use in your script
input_file = DATA_PROCESSED / "sensor_fused_50Hz.csv"
model = load_model(PRETRAINED_MODEL)
```

**Status:** ✅ Production-ready

---

### 2️⃣ `sensor_data_pipeline.py` (909 lines)

**Purpose:** STEP 1 - Process raw Garmin Excel files → Combined CSV

**Python equivalent of:** `data_preprocessing_step1.ipynb`

**What it does:**

1. **Load Excel files:**
   ```python
   # Load separate Excel files
   accel_df = load_excel('accelerometer_data.xlsx')
   gyro_df = load_excel('gyroscope_data.xlsx')
   ```

2. **Parse list columns:**
   ```python
   # Garmin stores [x,y,z] as strings like "[1.2, 3.4, 5.6]"
   # Parse these into separate columns
   Ax, Ay, Az = parse_xyz_columns(accel_df['xyz'])
   ```

3. **Create timestamps:**
   ```python
   # Base time + millisecond offsets
   timestamps = base_time + pd.to_timedelta(offsets, unit='ms')
   ```

4. **Merge sensors:**
   ```python
   # Align by timestamp with tolerance
   merged = merge_on_timestamps(accel_df, gyro_df, tolerance=1ms)
   ```

5. **Resample to 50Hz:**
   ```python
   # Both sensors at uniform 50Hz
   resampled = merged.resample('20ms').interpolate()
   ```

**Key Classes:**

- **`ProcessingConfig`** - Configuration parameters
- **`SensorDataLoader`** - Load and validate Excel
- **`DataProcessor`** - Parse and explode list columns
- **`SensorFusion`** - Merge accelerometer + gyroscope
- **`MetadataTracker`** - Track processing statistics

**Usage:**
```bash
# Command line
python sensor_data_pipeline.py \
  --accel data/raw/accelerometer_data.xlsx \
  --gyro data/raw/gyroscope_data.xlsx \
  --output data/processed/sensor_fused_50Hz.csv

# Or run with defaults
python sensor_data_pipeline.py
```

**Input:**
```
data/raw/
├── 2025-03-23-15-23-10-accelerometer_data.xlsx
└── 2025-03-23-15-23-10-gyroscope_data.xlsx
```

**Output:**
```
data/processed/
├── sensor_fused_50Hz.csv (181,699 rows × 7 columns)
└── sensor_fused_meta.json (processing metadata)
```

**Output CSV format:**
```
timestamp, Ax, Ay, Az, Gx, Gy, Gz
2025-03-23 15:23:10.000, -12.5, -18.3, -1001.2, 0.5, -0.2, 0.1
2025-03-23 15:23:10.020, -13.1, -17.9, -1000.8, 0.6, -0.3, 0.2
...
```

**Status:** ✅ Production-ready

**When to run:**
- When you have new raw Excel files
- First step in pipeline
- Before running preprocess_data.py

---

### 3️⃣ `preprocess_data.py` (624 lines)

**Purpose:** STEP 2 - Transform CSV → Model-ready windowed arrays

**Python equivalent of:** `production_preprocessing.ipynb`

**What it does:**

1. **Load CSV:**
   ```python
   df = pd.read_csv('sensor_fused_50Hz.csv')
   ```

2. **Detect units:**
   ```python
   # Automatic detection
   if max(abs(Ax)) > 100:
       units = 'milliG'  # Garmin format
   else:
       units = 'm/s²'    # Standard format
   ```

3. **Convert if needed:**
   ```python
   if units == 'milliG':
       Ax *= 0.00981  # Convert to m/s²
       Ay *= 0.00981
       Az *= 0.00981
   ```

4. **Handle NaN:**
   ```python
   # Forward fill + backward fill
   df = df.ffill().bfill()
   ```

5. **Normalize:**
   ```python
   # Load training scaler
   mean, std = load_scaler('config.json')
   normalized = (data - mean) / std
   ```

6. **Create windows:**
   ```python
   # 200 samples, 50% overlap
   windows = sliding_window(normalized, size=200, step=100)
   # Output: (1772, 200, 6)
   ```

7. **Save:**
   ```python
   np.save('production_X.npy', windows)
   ```

**Key Classes:**

- **`PreprocessLogger`** - Logging setup
- **`UnitDetector`** - Automatic unit detection
- **`UnifiedPreprocessor`** - Complete preprocessing pipeline

**Usage:**
```bash
# Command line
python preprocess_data.py \
  --input data/processed/sensor_fused_50Hz.csv \
  --output data/prepared/production_X.npy

# Or run with defaults
python preprocess_data.py
```

**Input:**
```
data/processed/sensor_fused_50Hz.csv (181,699 rows)
data/prepared/config.json (training scaler)
```

**Output:**
```
data/prepared/
├── production_X.npy (1,772 windows × 200 × 6)
└── production_metadata.json (pipeline info)
```

**Features:**
- ✅ Automatic unit detection (HIGH confidence)
- ✅ Conversion: milliG → m/s² (0.00981)
- ✅ Validation: Check Az ≈ -9.8
- ✅ Comprehensive logging
- ✅ Error handling

**Status:** ✅ Production-ready

**When to run:**
- After sensor_data_pipeline.py
- When you have sensor_fused_50Hz.csv
- Before model inference

---

### 4️⃣ `compare_data.py` (395 lines)

**Purpose:** Compare training vs production data distributions

**Python equivalent of:** `data_comparison.ipynb`

**What it does:**

1. **Load datasets:**
   ```python
   train_X = np.load('train_X.npy')
   prod_X = np.load('production_X.npy')
   ```

2. **Statistical comparison:**
   ```python
   # Per sensor
   for sensor in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
       train_mean = train_X[..., i].mean()
       prod_mean = prod_X[..., i].mean()
       drift = abs(prod_mean - train_mean)
   ```

3. **Drift detection:**
   ```python
   if drift > 0.1:
       print(f"⚠️ {sensor} DRIFT: {drift:.3f}")
   ```

4. **Generate report:**
   ```python
   # Markdown report with tables
   report = generate_comparison_report()
   ```

**Key Class:**

- **`DataComparator`** - Main comparison logic

**Usage:**
```bash
# Command line
python compare_data.py

# Output to file
python compare_data.py > comparison_report.txt
```

**Input:**
```
data/prepared/
├── train_X.npy (training data)
├── production_X.npy (production data)
└── config.json (scaler info)
```

**Output:**
```
Console output:
- Statistical comparison tables
- Drift warnings
- Recommendations

Optional:
- comparison_report.md
```

**Drift Metrics:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Mean drift | > 0.1 | Systematic shift |
| Std drift | > 0.15 | Variability change |
| Range outside | Any | Extrapolation risk |

**Status:** ✅ Production-ready

**When to run:**
- After preprocessing production data
- Before running inference
- When validating data quality

---

### 5️⃣ `run_inference.py` (NEW - 600+ lines)

**Purpose:** Run model inference on preprocessed production data

**What it does:**

1. **Load model:**
   ```python
   model = tf.keras.models.load_model('model.keras')
   ```

2. **Load preprocessed data:**
   ```python
   data = np.load('production_X.npy')  # (1772, 200, 6)
   ```

3. **Run inference (batch or real-time):**
   ```python
   # Softmax gives probability distribution
   probabilities = model.predict(data)  # (1772, 11)
   predictions = np.argmax(probabilities, axis=1)
   confidences = np.max(probabilities, axis=1)
   ```

4. **Export results:**
   ```python
   # CSV for human reading
   results.to_csv('predictions.csv')
   # JSON for metadata
   json.dump(metadata, 'predictions_metadata.json')
   ```

**Key Classes:**
- **`InferenceConfig`** - Configuration dataclass
- **`InferenceLogger`** - Logging to `logs/inference/`
- **`ModelLoader`** - Load and validate Keras model
- **`DataLoader`** - Load .npy files
- **`InferenceEngine`** - Batch and real-time prediction
- **`ResultsExporter`** - CSV + JSON output

**Confidence Levels:**
| Level | Confidence | Meaning |
|-------|------------|---------|
| HIGH | > 90% | Trust prediction |
| MODERATE | 70-90% | Likely correct |
| LOW | 50-70% | Review manually |
| UNCERTAIN | < 50% | Flag for review |

**Usage:**
```bash
# Default (batch mode)
python run_inference.py

# With custom input
python run_inference.py --input data/prepared/production_X.npy

# Change confidence threshold
python run_inference.py --threshold 0.6
```

**Output:**
```
data/prepared/predictions/
├── predictions_20251208_143100.csv
├── predictions_20251208_143100_metadata.json
└── predictions_20251208_143100_probs.npy
```

**Status:** ✅ NEW - Production-ready

---

### 6️⃣ `evaluate_predictions.py` (NEW - 400+ lines)

**Purpose:** Analyze predictions and confidence scores

**What it does:**

1. **Activity distribution:**
   ```python
   # Count predictions per class
   dist = df['predicted_activity'].value_counts()
   ```

2. **Confidence analysis:**
   ```python
   # Statistics, levels, per-class confidence
   mean_conf = df['confidence'].mean()
   ```

3. **Uncertainty analysis:**
   ```python
   # Which predictions are uncertain?
   uncertain = df[df['is_uncertain'] == True]
   ```

4. **Temporal patterns:**
   ```python
   # Activity transitions, sustained sequences
   transitions = count_transitions(predictions)
   ```

**Key Classes:**
- **`EvaluationConfig`** - Configuration
- **`EvaluationLogger`** - Logging to `logs/evaluation/`
- **`PredictionAnalyzer`** - Unlabeled data analysis
- **`ClassificationEvaluator`** - Labeled data (accuracy, F1, etc.)
- **`ReportGenerator`** - JSON + text reports

**Classification Metrics (when labels available):**
- Accuracy
- Precision, Recall, F1 (per-class and macro)
- Confusion Matrix
- Expected Calibration Error (ECE)

**Usage:**
```bash
# Analyze latest predictions
python evaluate_predictions.py

# Analyze specific file
python evaluate_predictions.py --input predictions_20251208.csv
```

**Output:**
```
outputs/evaluation/
├── evaluation_20251208_143105.json
└── evaluation_20251208_143105.txt
```

**Status:** ✅ NEW - Production-ready

---

## 🚀 How to Use (Step-by-Step)

### **Complete Pipeline Execution:**

```bash
# Navigate to project root
cd /path/to/MasterArbeit_MLops

# STEP 1: Process raw Excel → CSV (if you have raw files)
python src/sensor_data_pipeline.py

# STEP 2: Preprocess CSV → Windowed arrays
python src/preprocess_data.py

# STEP 3: Compare with training data (optional but recommended)
python src/compare_data.py

# STEP 4: Run inference
python src/run_inference.py

# STEP 5: Evaluate predictions
python src/evaluate_predictions.py
```

### **Individual Script Usage:**

```bash
# Run with default paths (from config.py)
python src/sensor_data_pipeline.py

# Run with custom paths
python src/preprocess_data.py \
  --input custom/path/data.csv \
  --output custom/output.npy \
  --verbose
```

---

## 📊 Progress Status

### ✅ **Completed (Production Ready)**

1. ✅ **config.py** - Path configuration
2. ✅ **sensor_data_pipeline.py** - Raw Excel processing (with logging)
3. ✅ **preprocess_data.py** - CSV preprocessing with unit detection (with logging)
4. ✅ **compare_data.py** - Drift detection (with logging)
5. ✅ **run_inference.py** - Model inference (NEW, with logging)
6. ✅ **evaluate_predictions.py** - Prediction analysis (NEW, with logging)

### ⏳ **TODO (Next Steps)**

7. ⏳ **api_server.py** - REST API (FastAPI)
   - Upload sensor data endpoint
   - Preprocessing + inference
   - Return predictions
   - Health check

8. ⏳ **train_model.py** - Model training script
   - Load training data
   - Build 1D-CNN-BiLSTM
   - Train with validation
   - Save model + metrics

---

## 🎯 Thesis Progress

### **Phase 1: Data & Preprocessing** ✅ (DONE)
- [x] Project structure
- [x] Raw data processing
- [x] Unit detection & conversion
- [x] Windowing & normalization
- [x] Data drift detection
- [x] Documentation (notebooks + scripts)

### **Phase 2: Model Inference** ✅ (DONE)
- [x] Pretrained model available
- [x] Inference script (run_inference.py)
- [x] Prediction analysis (evaluate_predictions.py)
- [x] Confidence validation
- [x] Logging for all scripts

### **Phase 3: MLOps Infrastructure** ⏳ (NEXT)
- [ ] REST API (FastAPI)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoring & logging (basic done ✅)
- [ ] Model versioning (DVC/MLflow)

### **Phase 4: Deployment & Testing** ⏳ (FUTURE)
- [ ] Cloud deployment (Azure/AWS)
- [ ] Performance testing
- [ ] User acceptance testing
- [ ] Documentation finalization

### **Phase 5: Thesis Writing** ⏳ (FUTURE)
- [ ] Introduction & background
- [ ] Methodology
- [ ] Results & analysis
- [ ] Discussion & conclusion

---

## 💡 My Questions & Suggestions

### **Questions to Clarify:**

1. **Model Inference Priority:**
   - Should we create `run_inference.py` next?
   - Do you want batch inference or real-time?
   - What output format: CSV, JSON, or database?

2. **API Requirements:**
   - REST API or gRPC?
   - Authentication needed?
   - Rate limiting?
   - Hosted where (local, Azure, AWS)?

3. **Training Pipeline:**
   - Do you need to retrain the model?
   - Or just use the pretrained one for thesis?
   - If retraining: hyperparameter tuning needed?

4. **Thesis Focus:**
   - **Option A:** Focus on MLOps (deployment, monitoring, CI/CD)
   - **Option B:** Focus on model performance (accuracy, optimization)
   - **Option C:** Balanced (both infrastructure + model)
   - Which direction for your thesis?

5. **Timeline:**
   - Submission deadline: April 2026 (4 months left)
   - How much time for each phase?
   - Any presentations/milestones before final submission?

### **Suggestions:**

1. **Immediate Next Steps (Week 1-2):**
   ```
   Priority 1: run_inference.py
   Priority 2: evaluate_predictions.py
   Priority 3: Basic API (FastAPI)
   ```

2. **Docker First:**
   - Before cloud deployment, containerize everything
   - Easier testing and reproducibility
   - Thesis can show Docker knowledge

3. **Monitoring is Key:**
   - Add Prometheus + Grafana for metrics
   - Track inference latency, drift over time
   - Impressive for thesis (shows production thinking)

4. **Version Control:**
   - Already using Git ✅
   - Add model versioning (MLflow or DVC)
   - Track experiments systematically

5. **Testing:**
   - Add unit tests (`tests/` folder exists but empty)
   - Pytest for scripts
   - Shows software engineering maturity

6. **Documentation:**
   - You're doing great with notebooks! ✅
   - Add API documentation (Swagger/OpenAPI)
   - Architecture diagrams (draw.io or mermaid)

### **My Doubts:**

1. **Production Data Source:**
   - Will there be continuous production data coming?
   - Or is this one-time analysis?
   - Affects API design (batch vs streaming)

2. **Stakeholders:**
   - Who will use this system?
   - Mental health professionals, researchers, patients?
   - Affects UI/UX decisions

3. **Privacy/Ethics:**
   - Sensitive health data → GDPR compliance?
   - Data anonymization needed?
   - Should be discussed in thesis

4. **Model Performance:**
   - Current 14% accuracy mentioned in docs
   - After unit conversion, what's the accuracy now?
   - Should we validate before proceeding?

5. **Scalability:**
   - How many users expected?
   - Real-time requirements (<100ms)?
   - Affects architecture choices

---

## 🗂️ Archived Scripts

**Location:** `src/Archived(prepare traning- production- conversion)/`

**Contents:**
- `prepare_training_data.py` - Old training preprocessing
- `prepare_production_data.py` - Old production preprocessing
- `convert_production_units.py` - Separate conversion script

**Why archived:**
- Replaced by unified `preprocess_data.py`
- Kept for historical reference
- Don't use these anymore

---

## 🔧 Development Tips

### **Before Running Scripts:**

1. **Check config.py paths:**
   ```python
   python -c "from src.config import *; print(PROJECT_ROOT)"
   ```

2. **Verify input files exist:**
   ```python
   from src.config import DATA_PROCESSED
   print((DATA_PROCESSED / "sensor_fused_50Hz.csv").exists())
   ```

3. **Check Python environment:**
   ```bash
   python --version  # Should be 3.11+
   pip list | grep numpy  # Verify dependencies
   ```

### **Common Issues:**

**Issue:** `ModuleNotFoundError: No module named 'config'`
```bash
# Solution: Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/MasterArbeit_MLops/src"
# Or run from project root
cd /path/to/MasterArbeit_MLops
python src/script.py
```

**Issue:** `FileNotFoundError: sensor_fused_50Hz.csv`
```bash
# Solution: Run sensor_data_pipeline.py first
python src/sensor_data_pipeline.py
```

**Issue:** `ValueError: Invalid units detected`
```bash
# Solution: Check data format, might need manual inspection
python -c "import pandas as pd; df = pd.read_csv('data/processed/sensor_fused_50Hz.csv'); print(df['Ax'].describe())"
```

---

## 📚 Related Documentation

- **Notebooks README:** `notebooks/README.md` - Jupyter notebook guide
- **Concepts Explained:** `docs/CONCEPTS_EXPLAINED.md` - Deep dive into concepts
- **Project Status:** `CURRENT_STATUS.md` - Overall thesis progress
- **Architecture:** `docs/ARCHITECTURE.md` - System design (TODO)

---

**Last Updated:** December 8, 2025  
**Status:** Data preprocessing complete, ready for inference implementation

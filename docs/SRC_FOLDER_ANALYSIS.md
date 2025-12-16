# SRC Folder Analysis - Current Structure

**Date:** December 12, 2025 (Updated with new MLflow tracking)
**Purpose:** Document all Python files in src/ and their roles

---

## 🆕 NEW ADDITIONS (December 12, 2025)

### MLflow Tracking Integration
- **run_inference.py:** Now includes MLflow experiment tracking
  - `mlflow.set_experiment("inference-production")`
  - `mlflow.log_params()` for configuration
  - `mlflow.log_metrics()` for inference statistics
  - `mlflow.log_artifact()` for output files
- **Result:** All experiments visible in MLflow UI at http://localhost:5000

### Fresh Pipeline Execution
- Complete pipeline executed with 1,815 windows
- Domain calibration applied (-6.295 m/s² offset)
- MLflow experiment: `63f4a91bc5924b5cafb4bcb028f69d6b`
- Results: 99.1% forehead_rubbing, 52% average confidence

---

## Current Files in `src/`

### 1. `config.py` (82 lines)
**Purpose:** Central configuration for paths, constants, and model parameters

**Key Contents:**
- **Path Definitions:** PROJECT_ROOT, DATA_DIR, MODELS_DIR, LOGS_DIR, etc.
- **Model Parameters:** WINDOW_SIZE=200, OVERLAP=0.5, NUM_SENSORS=6, NUM_CLASSES=11
- **Activity Labels:** 11 activity types (ear_rubbing, forehead_rubbing, etc.)
- **Sensor Columns:** SENSOR_COLUMNS for training data format
- **File Paths:** Quick access to common files (pretrained model, scaler config, etc.)

**Usage:** Imported by all other scripts for consistent paths and parameters

---

### 2. `sensor_data_pipeline.py` (882 lines)
**Purpose:** Process RAW sensor data from Excel files (Garmin watch format)

**Key Components:**
- **ProcessingConfig:** Configuration dataclass for processing parameters
- **LoggerSetup:** Logging infrastructure with rotating file handler
- **SensorDataLoader:** Load and validate Excel files with sensor data
  - Parse list columns (x, y, z stored as strings)
  - Normalize column names (vendor-specific → standardized)
  - Validate required columns exist
- **DataProcessor:** Transform raw data to time series
  - Explode list columns to individual rows
  - Create precise timestamps (base_time + offset)
  - Handle native sampling rates (accelerometer ≠ gyroscope)
- **SensorFusion:** Merge accelerometer + gyroscope data
  - Align timestamps with tolerance
  - Resample to target frequency (50Hz)
  - Interpolate missing values
- **MetadataTracker:** Track data lineage and statistics

**Input:** Raw Excel files from Garmin watch  
**Output:** `sensor_fused_50Hz.csv` (time-aligned, resampled sensor data)

**Status:** Used for initial raw data processing (not needed for preprocessed data)

---

### 3. `prepare_training_data.py` (345 lines)
**Purpose:** Prepare LABELED data for model training

**Key Components:**
- **DataPreparationPipeline class:**
  - `detect_data_format()`: Check if data is labeled vs unlabeled
  - `normalize_sensor_values()`: Apply StandardScaler normalization
  - `create_activity_encoding()`: Map activity names → numeric labels
  - `create_sliding_windows()`: Generate (200, 6) windows with 50% overlap
  - `split_by_user()`: Train/val/test splits ensuring user separation
  - `save_prepared_data()`: Save .npy arrays and metadata JSON

**Input:** `data/raw/all_users_data_labeled.csv` (6 users, 11 activities)  
**Output:** 
- `train_X.npy`, `train_y.npy` (2538 windows)
- `val_X.npy`, `val_y.npy` (654 windows)
- `test_X.npy`, `test_y.npy` (660 windows)
- `config.json` (scaler parameters, activity mappings)

**Sensor Columns:** Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w

---

### 4. `prepare_production_data.py` (407 lines)
**Purpose:** Prepare UNLABELED production data for inference

**Key Components:**
- `load_scaler_config()`: Load saved scaler from training
- `load_production_data()`: Load converted production CSV
- `preprocess_production_data()`: Apply same preprocessing as training
  - Map column names (Ax → Ax_w equivalent)
  - Apply saved StandardScaler (no fitting!)
  - Create sliding windows (200, 6) with 50% overlap
- `compare_with_training_data()`: Detect distribution drift
- `create_inference_readme()`: Document output files

**Input:** `data/processed/sensor_fused_50Hz_converted.csv`  
**Output:**
- `production_X.npy` (1815 windows, no labels)
- `production_metadata.json` (window timestamps, indices)

**Sensor Columns:** Ax, Ay, Az, Gx, Gy, Gz

**Critical:** Uses CONVERTED data (accelerometer already in m/s²)

---

### 5. `convert_production_units.py` (174 lines)
**Purpose:** Convert production accelerometer units from milliG to m/s²

**Key Components:**
- `convert_production_data()`: Apply conversion factor 0.00981
  - Load original production data
  - Multiply Ax, Ay, Az by 0.00981 (milliG → m/s²)
  - Leave Gx, Gy, Gz unchanged (already compatible)
  - Show before/after statistics
  - Validate against training data expectations

**Input:** `data/processed/sensor_fused_50Hz.csv` (milliG units)  
**Output:** `data/processed/sensor_fused_50Hz_converted.csv` (m/s² units)

**Conversion Factor:** 0.00981 (from mentor, Dec 3, 2025)

**Validation:** Converted Az ≈ -9.8 m/s² (Earth's gravity, physically correct)

---

## Current Workflow

```
RAW DATA (Excel files from Garmin)
    ↓
sensor_data_pipeline.py → sensor_fused_50Hz.csv (50Hz, native units)
    ↓
convert_production_units.py → sensor_fused_50Hz_converted.csv (m/s²)
    ↓
prepare_production_data.py → production_X.npy (windows for inference)

LABELED DATA (CSV from 6 users)
    ↓
prepare_training_data.py → train/val/test_X.npy (windows for training)
```

---

## Issues with Current Structure

### 1. **Redundant Unit Conversion Step**
- Production data always needs conversion, but it's a separate manual step
- Risk of using unconverted data by mistake
- Extra file created (sensor_fused_50Hz_converted.csv)

### 2. **Duplicate Preprocessing Logic**
- prepare_training_data.py and prepare_production_data.py have 70% overlapping code
- Window creation logic duplicated
- StandardScaler application duplicated
- Metadata tracking duplicated

### 3. **No Automatic Unit Detection**
- System can't detect if data needs conversion
- Manual decision required each time
- No validation that units match training data

### 4. **Limited Logging**
- sensor_data_pipeline.py has good logging
- prepare_*.py scripts have minimal logging (only print statements)
- Hard to debug preprocessing issues
- No audit trail for production inference

### 5. **Column Name Confusion**
- Training: Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w
- Production: Ax, Ay, Az, Gx, Gy, Gz
- Requires manual mapping in code

---

## Proposed Unified Structure

### New File: `preprocess_data.py`
**Single pipeline for both training and production data**

**Features:**
1. **Automatic Unit Detection**
   - Check if accelerometer values are in milliG range (-2000 to +2000)
   - Check if values are in m/s² range (-20 to +20)
   - Apply conversion only if needed
   - Log decision and reasoning

2. **Unified Preprocessing**
   - Single sliding window implementation
   - Single StandardScaler logic
   - Handles both labeled and unlabeled data
   - Consistent column mapping

3. **Comprehensive Logging**
   - Unit detection decision
   - Conversion applied (yes/no)
   - Data statistics at each step
   - Validation checks
   - Warnings for data drift

4. **Data Comparison**
   - Compare production vs training distributions
   - Flag significant differences
   - Provide recommendations

**Command Line Interface:**
```bash
# Training data
python src/preprocess_data.py --mode train --input data/raw/all_users_data_labeled.csv

# Production data (auto-detects units)
python src/preprocess_data.py --mode production --input data/processed/sensor_fused_50Hz.csv
```

---

## Files to Keep vs Archive

### **Keep (Active Use):**
1. ✅ `config.py` - Central configuration
2. ✅ `preprocess_data.py` - **NEW unified pipeline**
3. ✅ `sensor_data_pipeline.py` - For processing raw Garmin Excel files

### **Archive (Historical Reference):**
1. 📦 `prepare_training_data.py` - Logic moved to unified pipeline
2. 📦 `prepare_production_data.py` - Logic moved to unified pipeline
3. 📦 `convert_production_units.py` - Logic moved to unified pipeline

---

## Recent Additions (December 12, 2025)

### 🆕 New Tracking Module: `mlflow_tracking.py` (654 lines)
**Purpose:** Centralized MLflow integration for the MLOps pipeline

**Key Components:**
- **MLflowTracker:** Main class for experiment management
  - `set_experiment()`: Create/select experiment
  - `start_run()`: Begin a tracked run
  - `log_params()`: Log configuration parameters
  - `log_metrics()`: Log performance metrics
  - `log_model()`: Register models in registry
  - `log_artifact()`: Save output files

**Usage in Pipeline:**
- Automatically tracks all inference runs
- Logs inference speed, confidence stats, activity distribution
- Stores prediction artifacts (CSV, JSON, NPY files)
- Provides run history for experiment comparison

**Result:** All experiments visible at `http://localhost:5000`

---

## Next Steps

1. ✅ MLflow integration complete
2. ✅ Inference tracking automated
3. ✅ Pipeline reproducible and versioned
4. ⏭️ Run regularly for trend analysis
5. ⏭️ Compare runs across days/weeks
6. ⏭️ Use metrics in thesis chapters
4. ✅ Archive old files:
   - Move to `09_archive/old_src/`
   - Document what was replaced

---

## Summary

**Current:** 5 separate scripts with overlapping logic and manual steps  
**Proposed:** 3 clean scripts (config, unified preprocessor, raw data processor)

**Benefits:**
- No manual unit conversion step
- Single source of truth for preprocessing
- Better logging and debugging
- Easier to maintain and test
- Automatic validation and drift detection

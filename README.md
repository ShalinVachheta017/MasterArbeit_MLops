# MLOps Pipeline for Mental Health Monitoring

**Master's Thesis Project**  
**Duration:** 6 months (October 2025 - April 2026)  
**Last Updated:** December 6, 2025

**Current Status:** ✅ Unit conversion complete, ready for inference testing  
**Progress:** ~25% complete

---

## 📋 Project Overview

Developing an end-to-end MLOps pipeline for mental health monitoring using wearable sensor data (accelerometer + gyroscope). The system uses a pre-trained 1D-CNN-BiLSTM model to predict 11 anxiety-related activities.

### Key Components

- ✅ Data preprocessing pipeline (windowing, normalization, train/val/test splits)
- ✅ Pre-trained 1D-CNN-BiLSTM model analyzed (1.5M parameters, 11 classes)
- ✅ Prepared data: 3,852 windows from 6 users (385K samples)
- ✅ **Unit conversion resolved:** Production accelerometer converted from milliG to m/s² (factor: 0.00981)
- ✅ Converted production data: 181,699 samples now in correct units
- ⏳ **Next:** Test inference with converted production data
- ⏸️ MLOps infrastructure (API, monitoring, CI/CD) - after successful inference

---

## 📁 Project Structure

```
MasterArbeit_MLops/
│
├── data/                       # All data files
│   ├── raw/                    # Original labeled dataset (385K samples, 6 users)
│   ├── processed/              # Production unlabeled data (181K samples)
│   ├── prepared/               # Windowed train/val/test arrays + scaler config
│   └── samples/                # Sample data
│
├── src/                        # Source code
│   ├── preprocessing/          # Data pipelines (windowing, normalization)
│   ├── evaluation/             # Model evaluation scripts
│   ├── inference/              # Inference pipeline (blocked)
│   ├── monitoring/             # MLOps monitoring (future)
│   ├── training/               # Training scripts (future)
│   └── utils/                  # Helper functions
│
├── models/                     # Model artifacts
│   └── pretrained/             # 1D-CNN-BiLSTM (1.5M params, 11 classes)
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploration/            # Data exploration
│   └── experiments/            # Experiments
│
├── docs/                       # Documentation
│   ├── UNIT_CONVERSION_SOLUTION.md     # ✅ Solution to unit mismatch
│   ├── DATASET_DIFFERENCE_SUMMARY.md   # Data analysis & conversion
│   └── CRITICAL_MODEL_ISSUE.md         # Model evaluation history
│
├── research_papers/            # Research papers & references
├── images/                     # Project images & figures
├── logs/                       # Log files
├── tests/                      # Unit tests (future)
├── docker/                     # Containerization (future)
├── config/                     # Configuration files
│
├── CURRENT_STATUS.md           # 📍 START HERE - Where we are now (Dec 6, 2025)
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Read Current Status First!

```powershell
# Read this file to understand where we are and what's blocking us
cat CURRENT_STATUS.md
```

### 2. Setup Environment

```powershell
# Create conda environment
conda create -n thesis-mlops python=3.11 -y
conda activate thesis-mlops

# Install dependencies
pip install -r config/requirements.txt
```

### 3. View Prepared Data

```powershell
# Check prepared training/validation/test data
python -c "import numpy as np; X = np.load('data/prepared/train_X.npy'); print(f'Train shape: {X.shape}')"
```

### 4. Review Data Issue

```powershell
# Read about the current blocker (accelerometer unit mismatch)
cat docs/DATASET_DIFFERENCE_SUMMARY.md
cat docs/PROJECT_STATUS.md
```

---

## 📊 Current Progress (~20% Complete)

### ✅ Completed

**Data Preprocessing Pipeline**
- ✅ Built modular preprocessing system
- ✅ Created training/validation/test splits (by user, no data leakage)
- ✅ Generated 3,852 windows (200 timesteps × 6 sensors)
  - Train: 2,538 windows (users 1,2,3,4)
  - Val: 641 windows (user 5)
  - Test: 673 windows (user 6)
- ✅ Applied StandardScaler normalization
- ✅ Saved scaler parameters: `data/prepared/config.json`

**Model Analysis**
- ✅ Analyzed pre-trained 1D-CNN-BiLSTM (1.5M parameters)
- ✅ Verified architecture: Conv1D → BiLSTM → Dense
- ✅ Input: (200, 6), Output: (11 classes)
- ✅ Model info documented: `models/pretrained/model_info.json`
**Data Quality Analysis**
- ✅ Analyzed training data (385K samples, 6 users, 11 activities)
- ✅ Analyzed production data (181K samples, unlabeled)
- ✅ **Root cause identified:** Unit mismatch (training in m/s², production in milliG)
- ✅ **Solution received from mentor:** Conversion factor = 0.00981
- ✅ **Conversion completed (Dec 3, 2025):**
  - Az: -1001.6 milliG → -9.8 m/s² (Earth's gravity ✓)
  - All accelerometer channels now in correct units
  - Gyroscope channels unchanged (already compatible)

### ✅ Blocker Resolved (Dec 3, 2025)

**Production Data Unit Conversion Complete**
- Created conversion script: `src/preprocessing/convert_production_units.py`
- Converted data saved: `data/processed/sensor_fused_50Hz_converted.csv`
- Conversion log: `logs/preprocessing/unit_conversion.log`
- **Status:** Ready for inference testing
- **Documents:** 
  - `CURRENT_STATUS.md` - Current status (Dec 6, 2025)
  - `docs/UNIT_CONVERSION_SOLUTION.md` - Complete solution documentation
  - `docs/DATASET_DIFFERENCE_SUMMARY.md` - Analysis and resolution

### 🎯 Current Phase: Inference Testing

**This Week (Dec 6-13, 2025):**
1. Update `prepare_production_data.py` to use converted data
2. Apply training StandardScaler to converted production data
3. Create production windows (200 timesteps, 50% overlap)
4. Test model predictions on converted data
5. Validate confidence scores and prediction distribution

**Next Phase:**
- If inference works well → Build FastAPI serving
- If predictions poor → Investigate domain adaptation or fine-tuning
- Adds valuable thesis content on handling distribution shift

---

## 🧠 Model Information

### Architecture: 1D-CNN-BiLSTM
- **Input:** 200 timesteps × 6 sensors (4 seconds at 50Hz)
- **Sensors:** Ax, Ay, Az (accelerometer), Gx, Gy, Gz (gyroscope)
- **Output:** 11 activity classes
- **Parameters:** 1.5M total
- **Location:** `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`

### Activity Classes (11 total)
1. ear_rubbing
2. forehead_rubbing
3. hair_pulling
4. hand_scratching
5. hand_tapping
6. knuckles_cracking
7. nail_biting
8. nape_rubbing
9. sitting
10. smoking
11. standing

---

## 📖 Key Documents

### 📍 Start Here (Most Important!)
- **`CURRENT_STATUS.md`** - **READ THIS FIRST!** Complete current status, blocker, and next steps
- **`README.md`** - This file - Project overview
- **`REPO_STRUCTURE.md`** - Repository layout description

### Current Issue Documentation
- **`docs/PROJECT_STATUS.md`** - Blocker summary + mentor email template (ready to send)
- **`docs/DATASET_DIFFERENCE_SUMMARY.md`** - Statistical comparison of training vs production data
- **`docs/CRITICAL_MODEL_ISSUE.md`** - Model evaluation results showing data mismatch impact

### Data Artifacts
- **`data/prepared/README.md`** - Prepared data documentation
- **`data/prepared/config.json`** - Scaler parameters (means, stds) for production inference

---

## 🔧 Technical Stack

**Languages & Frameworks**
- Python 3.11
- TensorFlow 2.20.0
- Keras 3.11.3

**Data Processing**
- Pandas, NumPy
- scikit-learn (StandardScaler)

**MLOps Tools** (planned)
- MLflow (experiment tracking, model registry)
- FastAPI (inference API)
- Docker (containerization)
- GitHub Actions (CI/CD)

**Monitoring** (planned)
- Drift detection
- Performance monitoring
- Logging & alerting

---

## 📅 Timeline

**Original Plan:** 6 months (October 2025 - April 2026)

**Phase 1: Data Preprocessing & Analysis** (Oct-Nov) - ✅ **COMPLETE**
- ✅ Data preprocessing pipeline built
- ✅ Training/val/test splits prepared (3,852 windows)
- ✅ Model architecture analyzed
- ✅ Data quality analysis
- ✅ **Critical finding:** Accelerometer unit mismatch identified

**Phase 2: Issue Resolution** (Late Nov - Early Dec) - 🔴 **CURRENT**
- 🔴 Awaiting mentor confirmation on dataset issue
- ⏸️ Decision pending: New dataset OR conversion formula OR semi-supervised approach
**Phase 2: Issue Resolution** (Late Nov - Early Dec) - ✅ **COMPLETE**
- ✅ Identified unit mismatch (Nov 28)
- ✅ Received conversion factor from mentor (Dec 3)
- ✅ Converted production data (Dec 3)
- ✅ Validated conversion results

**Phase 3: Inference Testing** (Dec 6-13) - ⏳ **IN PROGRESS**
- ⏳ Prepare production data with converted units
- ⏳ Test model inference
- ⏳ Validate predictions

**Phase 4-6: MLOps Development** (Late Dec-Feb) - ⏸️ **UPCOMING**
- FastAPI serving
- Monitoring & drift detection
- Docker & CI/CD
**Current Progress:** ~25% complete  
**Delay Resolution:** Blocker resolved in 5 days (Nov 28 - Dec 3)
**Impact:** Minimal - back on track for April 2026 completion
**Expected Delay:** 2-3 weeks if semi-supervised approach needed  
**Impact:** Manageable - still on track for April 2026 completion

---

### Immediate (This Week - Dec 6-13, 2025)

**Inference Testing:**
1. ⏳ Update `src/preprocessing/prepare_production_data.py`
   - Load converted data: `data/processed/sensor_fused_50Hz_converted.csv`
   - Apply training StandardScaler
   - Create windows (200 timesteps, 50% overlap)
2. ⏳ Test model predictions
   - Load pretrained model
   - Run inference on production windows
   - Check confidence scores
3. ⏳ Validate results
   - Analyze prediction distribution
   - Compare with expected patterns
   - Decide: proceed with API or need fine-tuning?
4. Document approach for thesis (adds value!)
5. Proceed with MLOps infrastructure

### Long-term (Dec-Apr)
1. Complete inference pipeline
2. Build FastAPI serving
3. Implement monitoring & drift detection
4. Docker containerization & CI/CD
5. Write thesis documentation

---

## 📝 Important Notes

### Current Situation
- **Blocker:** Production accelerometer data has different units/scale than training data
- **Action:** Awaiting mentor confirmation on solution path
- **Timeline Impact:** 2-3 weeks delay if semi-supervised approach needed
- **Thesis Impact:** POSITIVE - Real-world MLOps challenge adds valuable content

### Current Situation
- **Previous Blocker:** Unit mismatch → ✅ RESOLVED (Dec 3, 2025)
- **Current Phase:** Inference testing with converted production data
- **Timeline Impact:** Minimal (5-day delay resolved)
- **Thesis Value:** Real-world data quality issue adds authentic MLOps content

### Key Files to Review
1. **`CURRENT_STATUS.md`** ← Updated Dec 6 - Where we are now
2. **`docs/UNIT_CONVERSION_SOLUTION.md`** ← How we solved the unit mismatch
3. **`docs/DATASET_DIFFERENCE_SUMMARY.md`** ← Analysis & resolution

### Project Info
- Started: October 2025
- Target completion: April 2026
- Current progress: ~25% complete
- Registration: November 1, 2025
- Conversion: `data/processed/sensor_fused_50Hz_converted.csv`

---

**Last Updated:** December 6, 2025  
**Status:** Ready for inference testing  
**Next Action:** Test model predictions on converted production data
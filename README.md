# MLOps Pipeline for Mental Health Monitoring

**Master's Thesis Project**  
**Duration:** 6 months (October 2025 - April 2026)  
**Last Updated:** November 28, 2025

**Current Status:** Data preprocessing complete, awaiting mentor confirmation on dataset issue  
**Progress:** ~20% complete

---

## 📋 Project Overview

Developing an end-to-end MLOps pipeline for mental health monitoring using wearable sensor data (accelerometer + gyroscope). The system uses a pre-trained 1D-CNN-BiLSTM model to predict 11 anxiety-related activities.

### Key Components

- ✅ Data preprocessing pipeline (windowing, normalization, train/val/test splits)
- ✅ Pre-trained 1D-CNN-BiLSTM model analyzed (1.5M parameters, 11 classes)
- ✅ Prepared data: 3,852 windows from 6 users (385K samples)
- 🔴 **Current Blocker:** Production data has different accelerometer units than training
- ⏸️ Awaiting mentor confirmation (new dataset OR conversion formula OR semi-supervised approach)
- ⏸️ Inference pipeline (blocked until data issue resolved)
- ⏸️ MLOps infrastructure (API, monitoring, CI/CD) - planned after inference works

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
│   ├── PROJECT_STATUS.md       # Current blocker + mentor email template
│   ├── DATASET_DIFFERENCE_SUMMARY.md  # Data mismatch details
│   └── CRITICAL_MODEL_ISSUE.md # Model evaluation results
│
├── logs/                       # Log files
├── tests/                      # Unit tests (future)
├── docker/                     # Containerization (future)
├── config/                     # Configuration files
│
├── CURRENT_STATUS.md           # 📍 START HERE - Complete current status
├── REPO_STRUCTURE.md           # Repository layout description
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
- ✅ **CRITICAL FINDING:** Accelerometer unit mismatch detected!
  - Production Az mean ≈ -1001.6 vs Training Az mean ≈ -3.5
  - ~50-120x scale difference (likely raw counts vs m/s² or g)
  - Gyroscope channels are compatible ✓

### 🔴 Current Blocker

**Production Data Has Different Accelerometer Units**
- Production accelerometer values are tens to hundreds of times larger than training
- Applying training StandardScaler creates out-of-distribution inputs
- Model predictions unreliable until units are aligned
- **Status:** Awaiting mentor confirmation (requesting new dataset or conversion formula)
- **Documents:** 
  - `CURRENT_STATUS.md` - Complete status and next steps
  - `docs/PROJECT_STATUS.md` - Blocker details + mentor email template
  - `docs/DATASET_DIFFERENCE_SUMMARY.md` - Statistical comparison

### ⏸️ Blocked - Awaiting Mentor Response

- Inference pipeline (needs compatible production data)
- FastAPI serving (depends on inference)
- Monitoring & drift detection (depends on inference)
- MLOps deployment (CI/CD, Docker)

### 🎯 Next Steps (After Mentor Confirmation)

**Option 1:** If mentor provides unit conversion or new dataset
- Convert production data to match training units
- Resume inference pipeline development
- Proceed with MLOps infrastructure

**Option 2:** If no conversion available
- Implement semi-supervised learning (pseudo-labeling)
- Fine-tune model on production distribution
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

**Phase 3-6: MLOps Development** (Dec-Feb) - ⏸️ **BLOCKED**
- Inference pipeline
- FastAPI serving
- Monitoring & drift detection
- Docker & CI/CD

**Phase 7: Documentation & Thesis** (Mar-Apr) - ⏸️ **FUTURE**
- Thesis writing
- Defense preparation

**Current Progress:** ~20% complete  
**Expected Delay:** 2-3 weeks if semi-supervised approach needed  
**Impact:** Manageable - still on track for April 2026 completion

---

## 🎯 Next Steps

### Immediate (This Week - Nov 28 - Dec 1)
1. 📧 **Contact mentor** about dataset issue
   - Request new dataset with correct units OR
   - Request conversion formula for accelerometer units OR
   - Confirm if we should proceed with semi-supervised approach
2. 📚 Research semi-supervised learning techniques (backup plan)
   - Pseudo-labeling / self-training
   - Active learning for smart labeling

### After Mentor Confirmation (Dec 2-8)

**Path A: If Mentor Provides Solution**
1. Apply unit conversion to production data
2. Validate distributions match training data
3. Resume inference pipeline development
4. Proceed with MLOps infrastructure

**Path B: If Semi-Supervised Approach Needed**
1. Implement pseudo-labeling or active learning
2. Fine-tune model on production distribution
3. Validate on held-out labeled data
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

### Key Files to Review
1. **`CURRENT_STATUS.md`** ← Start here for complete picture
2. **`docs/PROJECT_STATUS.md`** ← Mentor email template ready
3. **`docs/DATASET_DIFFERENCE_SUMMARY.md`** ← Data statistics

### Project Info
- Started: October 2025
- Target completion: April 2026
- Current progress: ~20% complete
- Registration: November 1, 2025

---

**Last Updated:** November 28, 2025  
**Status:** Awaiting mentor confirmation on dataset issue  
**Next Action:** Send mentor email, research semi-supervised learning as backup

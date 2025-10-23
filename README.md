# MLOps Pipeline for Mental Health Monitoring

**Master's Thesis Project**  
**Duration:** 6 months (October 2025 - April 2026)  
**Current Status:** Phase 1 Complete (25%) - Assessment & Analysis Done  
**Current Blocker:** Missing training labels (mentor communication in progress)

---

## 📋 Project Overview

Developing an end-to-end MLOps pipeline for continuous mental health monitoring using wearable sensor data (accelerometer + gyroscope). The system predicts anxiety levels using a 1D-CNN-BiLSTM deep learning model.

### Key Components
- ✅ Data preprocessing pipeline (sensor fusion, 50Hz resampling)
- ✅ Pre-trained 1D-CNN-BiLSTM model (11-class classification)
- ⏸️ Training pipeline with MLflow tracking (blocked - needs labels)
- ⏸️ Evaluation system (blocked - needs labels)
- ⏸️ MLOps infrastructure (API, monitoring, CI/CD) (blocked - needs trained model)

---

## 📁 Project Structure

```
thesis-mlops-mental-health/
│
├── 01_data/                    # All data files
│   ├── raw/                    # Original Excel sensor data (March 2025)
│   ├── processed/              # Preprocessed 50Hz CSVs
│   └── samples/                # Sample/test data
│
├── 02_src/                     # Source code
│   ├── preprocessing/          # Data preprocessing pipeline
│   ├── analysis/               # Model & data analysis scripts
│   └── training/               # Training pipeline (to be built)
│
├── 03_models/                  # Trained models
│   ├── pretrained/             # Pre-trained model from mentor
│   └── trained/                # Future trained models
│
├── 04_notebooks/               # Jupyter notebooks
│   ├── exploration/            # Data exploration notebooks
│   └── experiments/            # Experimental notebooks
│
├── 05_outputs/                 # Analysis outputs & results
│   ├── analysis/               # Data & model analysis results
│   └── reports/                # Evaluation reports (future)
│
├── 06_logs/                    # Log files
│   ├── preprocessing/          # Preprocessing logs
│   ├── training/               # Training logs (future)
│   └── evaluation/             # Evaluation logs (future)
│
├── 07_docs/                    # Documentation
│   ├── mentor_communication/   # Email and detailed request to mentor
│   ├── project_info/           # Project status and assessments
│   ├── planning/               # Roadmaps and questions
│   └── technical/              # Technical documentation
│
├── 08_config/                  # Configuration files
│   ├── requirements.txt        # Python dependencies
│   └── .pylintrc              # Linting configuration
│
├── 09_archive/                 # Old/backup files
│
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create conda environment
conda create -n thesis-mlops python=3.11 -y
conda activate thesis-mlops

# Install dependencies
pip install -r 08_config/requirements.txt
```

### 2. Run Data Preprocessing

```powershell
cd 02_src/preprocessing
python sensor_data_pipeline.py
```

### 3. Analyze Model

```powershell
cd 02_src/analysis
python inspect_model.py
```

### 4. Analyze Data

```powershell
python analyze_data.py
```

---

## 📊 Current Progress (Phase 1 Complete - 25%)

### ✅ Completed

**Data Preprocessing Pipeline**
- Built modular system with 8 specialized classes
- Processed March 2025 sensor data (181,699 samples)
- Achieved 95.1% sensor alignment accuracy
- Resampled to exact 50Hz
- Output: `01_data/processed/sensor_fused_50Hz.csv`

**Model Analysis**
- Inspected pre-trained 1D-CNN-BiLSTM architecture
- Input: (200, 6) - 200 timesteps × 6 sensors
- Output: (11) - 11-class classification
- Parameters: 1.5M total, 498K trainable
- Saved: `03_models/pretrained/model_info.json`

**Data Quality Analysis**
- Analyzed 69K + 182K samples
- Missing values: Only 0.014%
- All sensors within expected ranges
- **CRITICAL FINDING:** No training labels found!
- Outputs: `05_outputs/analysis/`

### 🔴 Current Blocker

**Missing Training Labels**
- Preprocessed data contains only sensor readings (Ax, Ay, Az, Gx, Gy, Gz)
- No label/class column present in any file
- Cannot proceed with training pipeline without ground truth
- **Status:** Mentor communication sent on October 23, 2025
- **Documents:** See `07_docs/mentor_communication/`

### ⏸️ Blocked - Awaiting Labels

- Data preparation script (sliding windows, normalization)
- Training pipeline (MLflow tracking, callbacks)
- Evaluation system (metrics, confusion matrix)
- MLOps deployment (API, monitoring)

---

## 📖 Key Documents

### Start Here
- **`07_docs/project_info/START_HERE.md`** - Project overview and current status

### Mentor Communication
- **`07_docs/mentor_communication/EMAIL_TO_MENTOR.md`** - Short email to mentor
- **`07_docs/mentor_communication/MENTOR_REQUEST_DETAILED.md`** - Detailed questions and context

### Project Status
- **`07_docs/project_info/PROJECT_ASSESSMENT.md`** - Phase 1 assessment results
- **`07_docs/project_info/QUICK_SUMMARY.md`** - Fast reference with key numbers
- **`07_docs/project_info/TERMINAL_ANALYSIS.md`** - Terminal output explained

### Planning
- **`07_docs/planning/COMPLETE_PIPELINE_ROADMAP.md`** - Full 8-phase thesis plan
- **`07_docs/planning/MENTOR_QUESTIONS.md`** - Critical questions for mentor

### Technical Documentation
- **`07_docs/technical/README_modular.md`** - Preprocessing pipeline documentation
- **`07_docs/technical/for scale .md`** - Scaling design notes

---

## 🔧 Technical Stack

**Languages & Frameworks**
- Python 3.11
- TensorFlow 2.20.0
- Keras 3.11.3

**Data Processing**
- Pandas, NumPy
- OpenPyXL (Excel reading)

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

## 📈 Model Specifications

**Architecture:** 1D-CNN-BiLSTM
- **Input:** (None, 200, 6) → 4 seconds at 50Hz, 6 sensors
- **Output:** (None, 11) → 11-class classification
- **Layers:**
  - 2× Conv1D (16, 32 filters)
  - 2× Bidirectional LSTM (64, 32 units)
  - 5× BatchNormalization
  - 5× Dropout
  - 2× Dense (32, 11 units)
- **Parameters:** 1,496,307 total
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** categorical_crossentropy

---

## 📅 Timeline

**Phase 1: Assessment** (Weeks 1-3) - ✅ **COMPLETE**
- Data preprocessing pipeline
- Model architecture analysis
- Data quality assessment

**Phase 2: Mentor Communication** (Week 4) - 🔴 **IN PROGRESS**
- Sent detailed request for labels
- Awaiting response

**Phase 3-8** (Weeks 5-24) - ⏸️ **BLOCKED**
- Data preparation
- Training pipeline
- Evaluation system
- MLOps infrastructure
- Testing & monitoring
- Documentation & thesis writing

---

## 🎯 Next Steps

### Immediate (This Week)
1. ✅ Send mentor email with detailed questions
2. ⏸️ Wait for mentor response
3. ⏸️ Review thesis registration form with mentor

### After Receiving Labels (Week 5+)
1. Build data preparation script (200-timestep sliding windows)
2. Implement training pipeline with MLflow
3. Create evaluation system
4. Deploy MLOps infrastructure
5. Write thesis documentation

### Alternative Path (If Labels Unavailable)
1. Pivot to MLOps-only focus
2. Use pre-trained model for inference
3. Focus on deployment, monitoring, versioning
4. Update thesis scope accordingly

---

## 👤 Contact

**Student:** [Your Name]  
**Student ID:** [Your ID]  
**Email:** [Your Email]  
**Thesis Supervisor:** [Mentor Name]

---

## 📝 Notes

- Project started: October 2025
- Target completion: April 2026
- Current progress: 25% (Phase 1 complete)
- Next milestone: Receive training labels from mentor
- Registration date: November 1, 2025

---

**Last Updated:** October 23, 2025

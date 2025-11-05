# MLOps Pipeline for Anxiety Detection Using Wearable Sensors# MLOps Pipeline for Mental Health Monitoring



**Master's Thesis Project** | November 2025 - April 2026  **Master's Thesis Project**  

**Focus:** Building Production-Ready MLOps Infrastructure**Duration:** 6 months (October 2025 - April 2026)  

**Current Status:** Phase 1 Complete (25%) - Assessment & Analysis Done  

---**Current Blocker:** Missing training labels (mentor communication in progress)



## 🎯 Project Overview---



This thesis demonstrates a **complete MLOps pipeline** for anxiety-related activity recognition using wearable IMU sensors. Rather than retraining an existing model, we focus on **operationalizing ML systems**: deployment, monitoring, and continuous delivery.## 📋 Project Overview



### Key Insight 💡Developing an end-to-end MLOps pipeline for continuous mental health monitoring using wearable sensor data (accelerometer + gyroscope). The system predicts anxiety levels using a 1D-CNN-BiLSTM deep learning model.

The pretrained model was already trained on our labeled dataset. Therefore, our thesis focuses on:

- ✅ **MLOps Infrastructure** (deployment, monitoring, CI/CD)### Key Components

- ❌ **NOT** Model retraining (already done)- ✅ Data preprocessing pipeline (sensor fusion, 50Hz resampling)

- ✅ Pre-trained 1D-CNN-BiLSTM model (11-class classification)

---- ⏸️ Training pipeline with MLflow tracking (blocked - needs labels)

- ⏸️ Evaluation system (blocked - needs labels)

## 📁 Simplified Project Structure- ⏸️ MLOps infrastructure (API, monitoring, CI/CD) (blocked - needs trained model)



```---

MasterArbeit_MLops/

│## 📁 Project Structure

├── data/                          # All data files

│   ├── raw/                       # Labeled dataset (385K samples)```

│   ├── processed/                 # Production data (unlabeled)thesis-mlops-mental-health/

│   └── prepared/                  # Windowed data for validation│

│├── 01_data/                    # All data files

├── models/                        # Model artifacts│   ├── raw/                    # Original Excel sensor data (March 2025)

│   └── pretrained/                # USE pretrained model AS-IS│   ├── processed/              # Preprocessed 50Hz CSVs

││   └── samples/                # Sample/test data

├── src/                           # Source code│

│   ├── preprocessing/             # Data preparation├── 02_src/                     # Source code

│   ├── inference/                 # Prediction pipeline (TODO)│   ├── preprocessing/          # Data preprocessing pipeline

│   ├── monitoring/                # Model monitoring (TODO)│   ├── analysis/               # Model & data analysis scripts

│   ├── utils/                     # Helper functions (TODO)│   └── training/               # Training pipeline (to be built)

│   └── config.py                  # Centralized configuration│

│├── 03_models/                  # Trained models

├── api/                           # FastAPI serving (TODO)│   ├── pretrained/             # Pre-trained model from mentor

├── notebooks/                     # Jupyter notebooks│   └── trained/                # Future trained models

├── tests/                         # Unit tests (TODO)│

├── docker/                        # Containerization (TODO)├── 04_notebooks/               # Jupyter notebooks

├── logs/                          # Application logs│   ├── exploration/            # Data exploration notebooks

├── docs/                          # Documentation│   └── experiments/            # Experimental notebooks

└── config/                        # Configuration files│

```├── 05_outputs/                 # Analysis outputs & results

│   ├── analysis/               # Data & model analysis results

---│   └── reports/                # Evaluation reports (future)

│

## 🧠 Model Information├── 06_logs/                    # Log files

│   ├── preprocessing/          # Preprocessing logs

### Architecture: 1D-CNN-BiLSTM│   ├── training/               # Training logs (future)

- **Input:** 200 timesteps × 6 sensors (4 seconds at 50Hz)│   └── evaluation/             # Evaluation logs (future)

- **Sensors:** Ax, Ay, Az (accelerometer), Gx, Gy, Gz (gyroscope)│

- **Output:** 11 activity classes├── 07_docs/                    # Documentation

- **Parameters:** 499,131│   ├── mentor_communication/   # Email and detailed request to mentor

- **Status:** ⚠️ **USE AS-IS - DO NOT RETRAIN**│   ├── project_info/           # Project status and assessments

│   ├── planning/               # Roadmaps and questions

### Activity Classes (11 total)│   └── technical/              # Technical documentation

1. ear_rubbing│

2. forehead_rubbing├── 08_config/                  # Configuration files

3. hair_pulling│   ├── requirements.txt        # Python dependencies

4. hand_scratching│   └── .pylintrc              # Linting configuration

5. hand_tapping│

6. knuckles_cracking├── 09_archive/                 # Old/backup files

7. nail_biting│

8. nape_rubbing└── README.md                   # This file

9. sitting```

10. smoking

11. standing---



---## 🚀 Quick Start



## 🚀 MLOps Components (Thesis Focus)### 1. Setup Environment



### ✅ Completed```powershell

- [x] Project restructuring (simplified folder names)# Create conda environment

- [x] Data preparation pipelineconda create -n thesis-mlops python=3.11 -y

- [x] Model architecture analysisconda activate thesis-mlops

- [x] Data leakage analysis

# Install dependencies

### 📋 Next Stepspip install -r 08_config/requirements.txt

```

**Phase 1: Inference Pipeline** (Month 2 - December)

- [ ] Create prediction script using pretrained model### 2. Run Data Preprocessing

- [ ] Load scaler parameters from config

- [ ] Implement sliding window inference```powershell

- [ ] Support batch predictionscd 02_src/preprocessing

python sensor_data_pipeline.py

**Phase 2: API Serving** (Month 2-3 - December-January)```

- [ ] FastAPI REST endpoint

- [ ] `/predict` endpoint (sensor data → predictions)### 3. Analyze Model

- [ ] `/health` endpoint

- [ ] Input validation```powershell

- [ ] Response formattingcd 02_src/analysis

python inspect_model.py

**Phase 3: Monitoring** (Month 3 - January)```

- [ ] Prometheus metrics

- [ ] Grafana dashboards### 4. Analyze Data

- [ ] Prediction tracking

- [ ] Data drift detection```powershell

python analyze_data.py

**Phase 4: MLflow** (Month 3 - January)```

- [ ] Model registry setup

- [ ] Log pretrained model---

- [ ] Version tracking

- [ ] Production environments## 📊 Current Progress (Phase 1 Complete - 25%)



**Phase 5: Docker** (Month 3-4 - February)### ✅ Completed

- [ ] Dockerfile for API

- [ ] Docker Compose setup**Data Preprocessing Pipeline**

- [ ] Environment configuration- Built modular system with 8 specialized classes

- Processed March 2025 sensor data (181,699 samples)

**Phase 6: CI/CD** (Month 4 - February)- Achieved 95.1% sensor alignment accuracy

- [ ] GitHub Actions workflow- Resampled to exact 50Hz

- [ ] Automated testing- Output: `01_data/processed/sensor_fused_50Hz.csv`

- [ ] Docker image building

- [ ] Deployment automation**Model Analysis**

- Inspected pre-trained 1D-CNN-BiLSTM architecture

---- Input: (200, 6) - 200 timesteps × 6 sensors

- Output: (11) - 11-class classification

## 📊 Dataset Information- Parameters: 1.5M total, 498K trainable

- Saved: `03_models/pretrained/model_info.json`

### Labeled Data (385K samples)

- **File:** `data/raw/all_users_data_labeled.csv`**Data Quality Analysis**

- **Users:** 6 users- Analyzed 69K + 182K samples

- **Sample Rate:** 50Hz- Missing values: Only 0.014%

- **Activities:** 11 classes (well-balanced)- All sensors within expected ranges

- **Usage:** ⚠️ Model validation ONLY (already trained on this)- **CRITICAL FINDING:** No training labels found!

- Outputs: `05_outputs/analysis/`

### Production Data (181K samples)

- **File:** `data/processed/sensor_fused_50Hz.csv`### 🔴 Current Blocker

- **Sample Rate:** 50Hz

- **Labels:** NONE (unlabeled)**Missing Training Labels**

- **Usage:** Production testing, drift detection- Preprocessed data contains only sensor readings (Ax, Ay, Az, Gx, Gy, Gz)

- No label/class column present in any file

---- Cannot proceed with training pipeline without ground truth

- **Status:** Mentor communication sent on October 23, 2025

## 🛠️ Setup Instructions- **Documents:** See `07_docs/mentor_communication/`



```powershell### ⏸️ Blocked - Awaiting Labels

# 1. Activate environment

conda activate thesis-mlops- Data preparation script (sliding windows, normalization)

- Training pipeline (MLflow tracking, callbacks)

# 2. Verify structure- Evaluation system (metrics, confusion matrix)

ls- MLOps deployment (API, monitoring)



# 3. Test imports---

python -c "from src.config import PRETRAINED_MODEL, WINDOW_SIZE; print('Config OK')"

```## 📖 Key Documents



---### Start Here

- **`07_docs/project_info/START_HERE.md`** - Project overview and current status

## 📈 Thesis Timeline

### Mentor Communication

- **Month 1 (Nov):** ✅ Setup, data understanding, restructuring- **`07_docs/mentor_communication/EMAIL_TO_MENTOR.md`** - Short email to mentor

- **Month 2 (Dec):** Inference pipeline + FastAPI- **`07_docs/mentor_communication/MENTOR_REQUEST_DETAILED.md`** - Detailed questions and context

- **Month 3 (Jan):** MLflow + Monitoring

- **Month 4 (Feb):** Docker + CI/CD### Project Status

- **Month 5 (Mar-Apr):** Documentation + Thesis writing- **`07_docs/project_info/PROJECT_ASSESSMENT.md`** - Phase 1 assessment results

- **`07_docs/project_info/QUICK_SUMMARY.md`** - Fast reference with key numbers

---- **`07_docs/project_info/TERMINAL_ANALYSIS.md`** - Terminal output explained



## 🎓 Why This Approach?### Planning

- **`07_docs/planning/COMPLETE_PIPELINE_ROADMAP.md`** - Full 8-phase thesis plan

### The Problem with Retraining- **`07_docs/planning/MENTOR_QUESTIONS.md`** - Critical questions for mentor

The pretrained model's architecture **exactly matches** our dataset:

- Same 11 classes### Technical Documentation

- Same 200 timesteps- **`07_docs/technical/README_modular.md`** - Preprocessing pipeline documentation

- Same 6 sensors- **`07_docs/technical/for scale .md`** - Scaling design notes



This means it was likely trained on our data. Retraining would be meaningless!---



### The Solution: Focus on MLOps## 🔧 Technical Stack

Instead of retraining, we demonstrate:

1. **Model serving** in production**Languages & Frameworks**

2. **API development** with FastAPI- Python 3.11

3. **Monitoring** with Prometheus/Grafana- TensorFlow 2.20.0

4. **CI/CD** automation- Keras 3.11.3

5. **Containerization** with Docker

6. **Model registry** with MLflow**Data Processing**

- Pandas, NumPy

**This is MORE valuable for an MLOps thesis!**- OpenPyXL (Excel reading)



---**MLOps Tools** (planned)

- MLflow (experiment tracking, model registry)

## 🔧 Technology Stack- FastAPI (inference API)

- Docker (containerization)

- **ML Framework:** TensorFlow 2.20, Keras 3.12- GitHub Actions (CI/CD)

- **API:** FastAPI

- **Monitoring:** Prometheus + Grafana**Monitoring** (planned)

- **Model Registry:** MLflow- Drift detection

- **Containerization:** Docker- Performance monitoring

- **CI/CD:** GitHub Actions- Logging & alerting

- **Data:** pandas, numpy, scikit-learn

---

---

## 📈 Model Specifications

## 📝 Key Files

**Architecture:** 1D-CNN-BiLSTM

- `src/config.py` - Centralized configuration- **Input:** (None, 200, 6) → 4 seconds at 50Hz, 6 sensors

- `src/preprocessing/prepare_training_data.py` - Data preparation- **Output:** (None, 11) → 11-class classification

- `data/prepared/config.json` - Scaler parameters- **Layers:**

- `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` - Model  - 2× Conv1D (16, 32 filters)

  - 2× Bidirectional LSTM (64, 32 units)

---  - 5× BatchNormalization

  - 5× Dropout

## ✅ Progress Tracking  - 2× Dense (32, 11 units)

- **Parameters:** 1,496,307 total

- [x] Clean project structure (Nov 4, 2025)- **Optimizer:** Adam (lr=0.0001)

- [x] Remove numbered folder prefixes- **Loss:** categorical_crossentropy

- [x] Delete temporary analysis scripts

- [x] Create standardized layout---

- [ ] Build inference pipeline

- [ ] Create FastAPI endpoint## 📅 Timeline

- [ ] Setup monitoring

- [ ] Implement CI/CD**Phase 1: Assessment** (Weeks 1-3) - ✅ **COMPLETE**

- Data preprocessing pipeline

**Last Updated:** November 4, 2025- Model architecture analysis

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

## 📝 Notes

- Project started: October 2025
- Target completion: April 2026
- Current progress: 25% (Phase 1 complete)
- Next milestone: Receive training labels from mentor
- Registration date: November 1, 2025

---

**Last Updated:** October 23, 2025

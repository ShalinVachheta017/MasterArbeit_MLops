# Project Structure - Human Activity Recognition MLOps Pipeline

This document provides a complete overview of the project folder structure, files, and their roles.

## Quick Overview

```
MasterArbeit_MLops/
├── src/                    # Main Python source code
├── data/                   # Data files (raw, processed, prepared)
├── models/                 # Pre-trained and trained models
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── docs/                   # Documentation
├── outputs/                # Evaluation results and artifacts
├── logs/                   # Training and inference logs
├── mlruns/                 # MLflow experiment tracking
├── tests/                  # Unit tests
├── docker/                 # Docker configuration
├── research_papers/        # Reference papers and materials
└── images/                 # Images for documentation
```

---

## Core Folders

### `src/` - Source Code
The main Python codebase for the ML pipeline.

| File | Description |
|------|-------------|
| `config.py` | Central configuration (paths, model settings, preprocessing params) |
| `sensor_data_pipeline.py` | Sensor data fusion and initial processing |
| `preprocess_data.py` | Data preprocessing (windowing, normalization, gravity removal) |
| `run_inference.py` | Model inference with MLflow tracking |
| `evaluate_predictions.py` | Calculate metrics (accuracy, F1, confusion matrix) |
| `compare_data.py` | Compare datasets for debugging |
| `data_validator.py` | Validate data quality and format |
| `mlflow_tracking.py` | MLflow utilities for experiment tracking |
| `README.md` | Source code documentation |
| `Archived/` | Old experimental scripts (not in use) |

### `data/` - Data Storage
All data files organized by processing stage.

| Subfolder | Description |
|-----------|-------------|
| `raw/` | Original sensor data files (DVC tracked) |
| `processed/` | Intermediate processed data (DVC tracked) |
| `prepared/` | Final prepared data for inference (DVC tracked) |
| `preprocessed/` | Alternative preprocessing outputs |
| `samples_2005 dataset/` | Sample data from 2005 dataset |

### `models/` - ML Models
| Subfolder | Description |
|-----------|-------------|
| `pretrained/` | Pre-trained 1D-CNN-BiLSTM model from mentor (DVC tracked) |
| `trained/` | Models trained during experiments |

**Model Info:**
- Architecture: 1D-CNN-BiLSTM
- Parameters: 499,131
- Input Shape: (200 timesteps, 6 features)
- Output: 11 activity classes

### `config/` - Configuration Files
| File | Description |
|------|-------------|
| `pipeline_config.yaml` | Main pipeline configuration |
| `mlflow_config.yaml` | MLflow tracking settings |
| `requirements.txt` | Python dependencies |
| `.pylintrc` | Code linting configuration |

### `outputs/` - Results and Artifacts
| Subfolder/File | Description |
|----------------|-------------|
| `evaluation/` | Confusion matrices, classification reports |
| `gravity_removal_comparison.png` | Preprocessing visualization |

### `notebooks/` - Jupyter Notebooks
| Notebook | Description |
|----------|-------------|
| `data_preprocessing_step1.ipynb` | Step-by-step preprocessing |
| `production_preprocessing.ipynb` | Production-ready preprocessing |
| `data_comparison.ipynb` | Dataset comparison analysis |
| `from_guide_processing.ipynb` | Following mentor's guide |
| `scalable.ipynb` | Scalability experiments |
| `exploration/` | Exploratory data analysis notebooks |

### `docs/` - Documentation
Essential documentation files.

| File | Description |
|------|-------------|
| `CONCEPTS_EXPLAINED.md` | Key ML/MLOps concepts explained |
| `PIPELINE_RERUN_GUIDE.md` | How to re-run the full pipeline |
| `SRC_FOLDER_ANALYSIS.md` | Detailed source code analysis |
| `RESEARCH_PAPERS_ANALYSIS.md` | Research paper summaries |
| `archived/` | Old documentation (DELETE_ and KEEP_LATER_ prefixed) |

### `mlruns/` - MLflow Tracking
MLflow experiment data stored locally.
- Contains experiment runs with metrics, parameters, artifacts
- View with: `mlflow ui --port 5001`

### `logs/` - Pipeline Logs
| Subfolder | Description |
|-----------|-------------|
| `preprocessing/` | Data preprocessing logs |
| `training/` | Model training logs |
| `evaluation/` | Evaluation logs |

### `tests/` - Unit Tests
Unit tests for pipeline components.

### `docker/` - Containerization
Docker configuration for reproducible environments.

### `research_papers/` - Reference Materials
Research papers and academic references for the project.

---

## Root Level Files

| File | Description |
|------|-------------|
| `README.md` | Main project README |
| `PROJECT_STRUCTURE.md` | This file - project overview |
| `docker-compose.yml` | Docker compose configuration |
| `.gitignore` | Git ignore patterns |
| `.dvcignore` | DVC ignore patterns |

---

## Pipeline Execution Flow

```
1. Raw Data → sensor_data_pipeline.py → Fused sensor data
2. Fused Data → preprocess_data.py → Preprocessed windows
3. Preprocessed → run_inference.py → Predictions (with MLflow tracking)
4. Predictions → evaluate_predictions.py → Metrics and reports
```

---

## Key Technical Details

- **Sampling Rate:** 50 Hz (unified from multiple sensor rates)
- **Window Size:** 200 timesteps (4 seconds)
- **Overlap:** 50%
- **Domain Calibration:** -6.295 m/s² Az offset applied
- **Activities:** 11 classes (sitting, standing, walking, etc.)

---

## Version Control

- **Git:** Source code and configuration
- **DVC:** Large data files and models (tracked via `.dvc` files)
- **MLflow:** Experiment tracking and metrics

---

## Quick Commands

```bash
# Run full pipeline
python src/preprocess_data.py
python src/run_inference.py
python src/evaluate_predictions.py

# View MLflow experiments
mlflow ui --port 5001

# Pull DVC data
dvc pull
```

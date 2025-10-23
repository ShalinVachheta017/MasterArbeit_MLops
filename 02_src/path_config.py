# Path Configuration for Restructured Project
# Import this in your scripts to use correct paths

from pathlib import Path

# Base directory is the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW = BASE_DIR / "01_data" / "raw"
DATA_PROCESSED = BASE_DIR / "01_data" / "processed"
DATA_SAMPLES = BASE_DIR / "01_data" / "samples"

# Source code directories
SRC_PREPROCESSING = BASE_DIR / "02_src" / "preprocessing"
SRC_ANALYSIS = BASE_DIR / "02_src" / "analysis"
SRC_TRAINING = BASE_DIR / "02_src" / "training"

# Model directories
MODELS_PRETRAINED = BASE_DIR / "03_models" / "pretrained"
MODELS_TRAINED = BASE_DIR / "03_models" / "trained"

# Notebooks
NOTEBOOKS_EXPLORATION = BASE_DIR / "04_notebooks" / "exploration"
NOTEBOOKS_EXPERIMENTS = BASE_DIR / "04_notebooks" / "experiments"

# Outputs
OUTPUTS_ANALYSIS = BASE_DIR / "05_outputs" / "analysis"
OUTPUTS_REPORTS = BASE_DIR / "05_outputs" / "reports"

# Logs
LOGS_DIR = BASE_DIR / "06_logs"
LOGS_PREPROCESSING = LOGS_DIR / "preprocessing"
LOGS_TRAINING = LOGS_DIR / "training"
LOGS_EVALUATION = LOGS_DIR / "evaluation"

# Docs
DOCS_DIR = BASE_DIR / "07_docs"

# Config
CONFIG_DIR = BASE_DIR / "08_config"

# Archive
ARCHIVE_DIR = BASE_DIR / "09_archive"

# Specific file paths (commonly used)
RAW_ACCEL_FILE = DATA_RAW / "2025-03-23-15-23-10-accelerometer_data.xlsx"
RAW_GYRO_FILE = DATA_RAW / "2025-03-23-15-23-10-gyroscope_data.xlsx"
PROCESSED_SENSOR_FUSED = DATA_PROCESSED / "sensor_fused_50Hz.csv"
PRETRAINED_MODEL = MODELS_PRETRAINED / "fine_tuned_model_1dcnnbilstm.keras"
MODEL_INFO_JSON = MODELS_PRETRAINED / "model_info.json"

# Usage in other scripts:
# from path_config import RAW_ACCEL_FILE, LOGS_PREPROCESSING
# pipeline = SensorDataPipeline(RAW_ACCEL_FILE, RAW_GYRO_FILE)

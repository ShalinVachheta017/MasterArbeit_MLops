"""
Path Configuration for MLOps Pipeline
Simplified structure without numbered prefixes
"""

from pathlib import Path

# Base directory is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_PREPARED = DATA_DIR / "prepared"  # Windowed data ready for training

# Source code directories
SRC_DIR = PROJECT_ROOT / "src"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_PRETRAINED = MODELS_DIR / "pretrained"

# API
API_DIR = PROJECT_ROOT / "api"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"

# Config
CONFIG_DIR = PROJECT_ROOT / "config"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"

# Docker
DOCKER_DIR = PROJECT_ROOT / "docker"

# Specific file paths (commonly used)
LABELED_DATA_FILE = DATA_RAW / "all_users_data_labeled.csv"
PROCESSED_SENSOR_FUSED = DATA_PROCESSED / "sensor_fused_50Hz.csv"
PRETRAINED_MODEL = MODELS_PRETRAINED / "fine_tuned_model_1dcnnbilstm.keras"
MODEL_INFO_JSON = MODELS_PRETRAINED / "model_info.json"
SCALER_CONFIG = DATA_PREPARED / "config.json"

# Model configuration
WINDOW_SIZE = 200  # 4 seconds at 50Hz
OVERLAP = 0.5  # 50% overlap
NUM_SENSORS = 6  # Ax, Ay, Az, Gx, Gy, Gz
NUM_CLASSES = 11  # 11 activity types

# Activity labels (from labeled dataset)
ACTIVITY_LABELS = [
    'ear_rubbing',
    'forehead_rubbing', 
    'hair_pulling',
    'hand_scratching',
    'hand_tapping',
    'knuckles_cracking',
    'nail_biting',
    'nape_rubbing',
    'sitting',
    'smoking',
    'standing'
]

# Sensor columns
SENSOR_COLUMNS = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']

# Usage:
# from src.config import PROJECT_ROOT, PRETRAINED_MODEL, WINDOW_SIZE

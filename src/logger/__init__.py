"""
Logger Module for HAR MLOps Production Pipeline

Provides centralized logging configuration for the entire pipeline.
Sets up rotating log files with timestamps and clean formatting.

Features:
- Single log file per run (timestamped)
- Rotating log files (5MB max, 3 backups)
- Clean format: [timestamp] logger - level - message
- Automatic log directory creation
- No duplicate handlers

Usage:
    from src.logger import logging
    logging.info("Your message here")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Get project root (where run_pipeline.py is)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
log_dir_path = PROJECT_ROOT / LOG_DIR
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE


def get_logger():
    """
    Configure and return the root logger.
    
    This function sets up a comprehensive logging system that:
    - Logs DEBUG level and above to rotating files
    - Logs INFO level and above to console
    - Automatically rotates log files when they reach 5MB
    - Keeps 3 backup log files
    - Uses timestamped filenames for each run
    
    Returns:
        logging.Logger: Configured root logger
    """
    logger = logging.getLogger()
    
    # Only configure if not already configured (prevent duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Define formatter for consistent log message format
        formatter = logging.Formatter(
            "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# Configure logger when module is imported
_logger = get_logger()

# Export the current log file path
CURRENT_LOG_FILE = str(log_file_path)

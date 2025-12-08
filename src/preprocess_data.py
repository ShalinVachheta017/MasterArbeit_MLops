"""
Production Data Preprocessing Pipeline (Unlabeled Only)
=======================================================
- Automatic unit detection (milliG vs m/s²)
- Automatic conversion when needed
- Comprehensive logging
- Sliding window creation for inference

Note: Training data preparation is intentionally excluded here. Use the
archived training pipeline if you need to retrain.

Author: Master Thesis MLOps Project
Date: December 7, 2025
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from config import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    DATA_PREPARED,
    LOGS_DIR,
    WINDOW_SIZE,
    OVERLAP,
    ACTIVITY_LABELS,
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class PreprocessLogger:
    """Configure comprehensive logging for preprocessing pipeline"""
    
    def __init__(self, log_name: str = "preprocessing"):
        self.log_dir = LOGS_DIR / "preprocessing"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        
        # Format: timestamp | level | message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler (INFO and above)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # File handler with rotation (2MB per file, keep 5 backups)
        log_file = self.log_dir / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("="*80)
        self.logger.info(f"Preprocessing Pipeline Started - Log: {log_file.name}")
        self.logger.info("="*80)
    
    def get_logger(self) -> logging.Logger:
        return self.logger


# ============================================================================
# UNIT DETECTION AND CONVERSION
# ============================================================================

class UnitDetector:
    """Automatically detect accelerometer units and convert if needed"""
    
    # Conversion factor from mentor (Dec 3, 2025)
    CONVERSION_FACTOR = 0.00981  # milliG → m/s²
    
    # Expected ranges for unit detection
    MILLIG_RANGE = (-2000, 2000)  # Typical milliG range
    MS2_RANGE = (-20, 20)  # Typical m/s² range (±2g)
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def detect_units(self, df: pd.DataFrame, accel_cols: List[str]) -> str:
        """
        Detect if accelerometer data is in milliG or m/s²
        
        Args:
            df: DataFrame with accelerometer data
            accel_cols: List of accelerometer column names
        
        Returns:
            'milliG', 'm/s²', or 'unknown'
        """
        self.logger.info("Detecting accelerometer units...")
        
        # Get statistics (ignore NaN values)
        accel_data = df[accel_cols].values.flatten()
        accel_data = accel_data[~np.isnan(accel_data)]  # Remove NaN
        
        if len(accel_data) == 0:
            raise ValueError("All accelerometer values are NaN!")
        
        mean_val = np.abs(accel_data).mean()
        max_val = np.abs(accel_data).max()
        min_val = accel_data.min()
        max_abs = max(abs(min_val), max_val)
        
        self.logger.info(f"  Accelerometer statistics:")
        self.logger.info(f"    Mean (abs): {mean_val:.3f}")
        self.logger.info(f"    Max (abs):  {max_abs:.3f}")
        self.logger.info(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
        
        # Decision logic
        if max_abs > 100:
            # Likely milliG (values > 100)
            unit = 'milliG'
            confidence = "HIGH" if max_abs > 500 else "MEDIUM"
            self.logger.info(f"  ✓ Detected units: {unit} (confidence: {confidence})")
            self.logger.info(f"    Reason: Max absolute value {max_abs:.1f} >> 20 (m/s² range)")
        elif max_abs < 50:
            # Likely m/s² (values < 50)
            unit = 'm/s²'
            confidence = "HIGH" if max_abs < 20 else "MEDIUM"
            self.logger.info(f"  ✓ Detected units: {unit} (confidence: {confidence})")
            self.logger.info(f"    Reason: Max absolute value {max_abs:.1f} < 50 (milliG range)")
        else:
            # Ambiguous range (50-100)
            unit = 'unknown'
            self.logger.warning(f"  ⚠ Ambiguous units detected!")
            self.logger.warning(f"    Max absolute value {max_abs:.1f} in ambiguous range [50-100]")
            self.logger.warning(f"    Manual inspection recommended")
        
        return unit
    
    def convert_to_ms2(self, df: pd.DataFrame, accel_cols: List[str]) -> pd.DataFrame:
        """
        Convert accelerometer data from milliG to m/s²
        
        Args:
            df: DataFrame with accelerometer in milliG
            accel_cols: List of accelerometer column names
        
        Returns:
            DataFrame with accelerometer in m/s²
        """
        self.logger.info("Converting accelerometer units: milliG → m/s²")
        self.logger.info(f"  Conversion factor: {self.CONVERSION_FACTOR}")
        
        df_converted = df.copy()
        
        # Show BEFORE stats
        self.logger.info("  Before conversion (milliG):")
        for col in accel_cols:
            self.logger.info(f"    {col}: mean={df[col].mean():10.3f}, std={df[col].std():10.3f}")
        
        # Apply conversion
        for col in accel_cols:
            df_converted[col] = df[col] * self.CONVERSION_FACTOR
        
        # Show AFTER stats
        self.logger.info("  After conversion (m/s²):")
        for col in accel_cols:
            self.logger.info(f"    {col}: mean={df_converted[col].mean():10.3f}, std={df_converted[col].std():10.3f}")
        
        # Validate conversion (Az should be ~-9.8 m/s² if stationary)
        if 'Az' in accel_cols or 'Az_w' in accel_cols:
            az_col = 'Az' if 'Az' in accel_cols else 'Az_w'
            az_mean = df_converted[az_col].mean()
            expected_gravity = -9.8
            diff = abs(az_mean - expected_gravity)
            
            if diff < 5:
                self.logger.info(f"  ✓ Validation: Az mean = {az_mean:.2f} m/s² (close to gravity -9.8)")
            else:
                self.logger.warning(f"  ⚠ Validation: Az mean = {az_mean:.2f} m/s² (differs from gravity -9.8)")
        
        return df_converted
    
    def process_units(self, df: pd.DataFrame, accel_cols: List[str]) -> Tuple[pd.DataFrame, bool]:
        """
        Main entry point: detect units and convert if needed
        
        Returns:
            (processed_df, conversion_applied)
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("UNIT DETECTION AND CONVERSION")
        self.logger.info("="*80)
        
        # Detect current units
        detected_units = self.detect_units(df, accel_cols)
        
        # Decision: convert or skip
        if detected_units == 'milliG':
            self.logger.info("  → Decision: CONVERSION REQUIRED")
            df_processed = self.convert_to_ms2(df, accel_cols)
            conversion_applied = True
        elif detected_units == 'm/s²':
            self.logger.info("  → Decision: CONVERSION NOT NEEDED (already in m/s²)")
            df_processed = df.copy()
            conversion_applied = False
        else:
            self.logger.error("  → Decision: CANNOT PROCEED (ambiguous units)")
            raise ValueError(
                f"Cannot determine accelerometer units. "
                f"Please verify data manually or specify units explicitly."
            )
        
        self.logger.info("="*80)
        return df_processed, conversion_applied


# ============================================================================
# DATA PREPROCESSING PIPELINE
# ============================================================================

class UnifiedPreprocessor:
    """Unified preprocessing for training and production data"""
    
    def __init__(self, logger: logging.Logger, window_size: int = WINDOW_SIZE, overlap: float = OVERLAP):
        self.logger = logger
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = StandardScaler()
        self.activity_to_label = {}
        self.label_to_activity = {}
        
        self.logger.info(f"Preprocessor initialized:")
        self.logger.info(f"  Window size: {window_size} samples")
        self.logger.info(f"  Overlap: {overlap*100:.0f}%")
        self.logger.info(f"  Step size: {self.step_size} samples")
    
    def detect_data_format(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Detect if data is labeled (training) or unlabeled (production)
        
        Returns:
            (format, sensor_columns)
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("DATA FORMAT DETECTION")
        self.logger.info("="*80)
        
        has_activity = 'activity' in df.columns
        has_user = 'User' in df.columns
        
        # Check sensor column formats
        labeled_sensors = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        unlabeled_sensors = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        
        has_labeled_format = all(col in df.columns for col in labeled_sensors)
        has_unlabeled_format = all(col in df.columns for col in unlabeled_sensors)
        
        if has_activity and has_user and has_labeled_format:
            self.logger.error("  ✗ Labeled training data is not supported in this script")
            self.logger.error("  Please use the archived training pipeline for retraining tasks")
            raise ValueError("Training (labeled) data is not supported here")
        elif has_unlabeled_format:
            data_format = 'unlabeled'
            sensor_cols = unlabeled_sensors
            self.logger.info(f"  Format: UNLABELED (production data)")
            self.logger.info(f"  Columns: {', '.join(sensor_cols)} + timestamp")
        else:
            self.logger.error("  ✗ Unknown data format!")
            self.logger.error(f"  Available columns: {df.columns.tolist()}")
            raise ValueError(f"Cannot determine data format from columns: {df.columns.tolist()}")
        
        self.logger.info(f"  Samples: {len(df):,}")
        self.logger.info("="*80)
        
        return data_format, sensor_cols
    
    def create_activity_encoding(self, activities: pd.Series) -> Dict[str, int]:
        """Create mapping between activity names and numeric labels"""
        unique_activities = sorted(activities.unique())
        self.activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}
        self.label_to_activity = {idx: activity for activity, idx in self.activity_to_label.items()}
        
        self.logger.info("")
        self.logger.info("Activity Encoding:")
        for activity, label in self.activity_to_label.items():
            count = (activities == activity).sum()
            self.logger.info(f"  {label:2d}: {activity:20s} ({count:6,} samples)")
        
        return self.activity_to_label
    
    def normalize_data(self, df: pd.DataFrame, sensor_cols: List[str], mode: str = 'transform') -> pd.DataFrame:
        """
        Normalize sensor values using pre-fitted StandardScaler
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor column names
            mode: only 'transform' is supported (production)
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("DATA NORMALIZATION")
        self.logger.info("="*80)
        
        df_normalized = df.copy()
        
        # Check and handle NaN values
        nan_count = df[sensor_cols].isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"  Found {nan_count} NaN values in sensor data")
            self.logger.warning(f"  Applying forward fill + backward fill")
            df_normalized[sensor_cols] = df_normalized[sensor_cols].ffill().bfill()
            
            # Check if any NaN remain
            remaining_nan = df_normalized[sensor_cols].isna().sum().sum()
            if remaining_nan > 0:
                self.logger.warning(f"  {remaining_nan} NaN values remain after filling")
                self.logger.warning(f"  Dropping rows with NaN")
                df_normalized = df_normalized.dropna(subset=sensor_cols).reset_index(drop=True)
            
            self.logger.info(f"  After NaN handling: {len(df_normalized):,} samples")
        
        if mode == 'transform':
            # Production: use pre-fitted scaler
            self.logger.info("  Mode: TRANSFORM (production data with saved scaler)")
            
            # Load scaler config
            config_path = DATA_PREPARED / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Scaler config not found: {config_path}\n"
                    f"Please run preprocessing on training data first!"
                )
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.scaler.mean_ = np.array(config['scaler_mean'])
            self.scaler.scale_ = np.array(config['scaler_scale'])
            
            self.logger.info("  Loaded scaler from training:")
            self.logger.info(f"    Mean:  {self.scaler.mean_}")
            self.logger.info(f"    Scale: {self.scaler.scale_}")
            
            df_normalized[sensor_cols] = self.scaler.transform(df[sensor_cols])
        else:
            raise ValueError(f"Invalid mode: {mode}. Only 'transform' (production) is supported.")
        
        # Log statistics
        self.logger.info("  After normalization:")
        self.logger.info(f"    Range: [{df_normalized[sensor_cols].values.min():.3f}, "
                        f"{df_normalized[sensor_cols].values.max():.3f}]")
        self.logger.info(f"    Mean:  {df_normalized[sensor_cols].values.mean():.3f}")
        self.logger.info(f"    Std:   {df_normalized[sensor_cols].values.std():.3f}")
        self.logger.info("="*80)
        
        return df_normalized
    
    def create_windows(self, df: pd.DataFrame, sensor_cols: List[str], 
                      data_format: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict]]:
        """
        Create sliding windows from time series data
        
        Returns:
            X: (n_windows, window_size, n_sensors) array
            y: (n_windows,) array or None if unlabeled
            metadata: List of window info dicts
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("SLIDING WINDOW CREATION")
        self.logger.info("="*80)
        
        n_samples = len(df)
        n_possible_windows = (n_samples - self.window_size) // self.step_size + 1
        
        self.logger.info(f"  Total samples: {n_samples:,}")
        self.logger.info(f"  Window size: {self.window_size}")
        self.logger.info(f"  Step size: {self.step_size}")
        self.logger.info(f"  Maximum possible windows: {n_possible_windows:,}")
        
        X = []
        y = []
        metadata = []
        
        for i in range(n_possible_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            if end_idx > n_samples:
                break
            
            # Extract window
            window_data = df.iloc[start_idx:end_idx][sensor_cols].values
            
            # Validate shape
            if window_data.shape[0] != self.window_size:
                continue
            
            # Skip windows with NaN values
            if np.isnan(window_data).any():
                continue
            
            X.append(window_data)
            
            # Add label if available (majority vote)
            if data_format == 'labeled':
                window_activities = df.iloc[start_idx:end_idx]['activity']
                majority_activity = window_activities.mode()[0]
                label = self.activity_to_label[majority_activity]
                y.append(label)
            
            # Metadata
            meta = {
                'window_id': len(X) - 1,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
            }
            
            # Add timestamps if available
            if 'timestamp_iso' in df.columns:
                meta['timestamp_start'] = str(df.iloc[start_idx]['timestamp_iso'])
                meta['timestamp_end'] = str(df.iloc[end_idx - 1]['timestamp_iso'])
            
            # Add user if available
            if 'User' in df.columns:
                meta['user'] = str(df.iloc[start_idx]['User'])
            
            metadata.append(meta)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32) if len(y) > 0 else None
        
        self.logger.info(f"  ✓ Created {len(X):,} windows")
        self.logger.info(f"  Shape: {X.shape}")
        self.logger.info(f"  Memory: {X.nbytes / 1024 / 1024:.2f} MB")
        
        if y is not None:
            self.logger.info(f"  Labels shape: {y.shape}")
            self.logger.info(f"  Label distribution:")
            for label_id in np.unique(y):
                count = (y == label_id).sum()
                activity = self.label_to_activity[label_id]
                self.logger.info(f"    {label_id:2d} ({activity:20s}): {count:5,} windows")
        
        self.logger.info("="*80)
        
        return X, y, metadata
    
    def split_by_user(self, X: np.ndarray, y: np.ndarray, metadata: List[Dict],
                     train_ratio: float = 0.66, val_ratio: float = 0.17) -> Dict:
        """
        Split data by user for train/val/test
        
        Ensures no user appears in multiple splits (prevents data leakage)
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("TRAIN/VAL/TEST SPLIT BY USER")
        self.logger.info("="*80)
        
        # Get unique users
        users = np.array([m['user'] for m in metadata])
        unique_users = np.unique(users)
        n_users = len(unique_users)
        
        self.logger.info(f"  Total users: {n_users}")
        self.logger.info(f"  Split ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={1-train_ratio-val_ratio:.2f}")
        
        # Shuffle users
        np.random.seed(42)
        shuffled_users = np.random.permutation(unique_users)
        
        # Split users
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)
        
        train_users = shuffled_users[:n_train]
        val_users = shuffled_users[n_train:n_train + n_val]
        test_users = shuffled_users[n_train + n_val:]
        
        self.logger.info(f"  Train users ({len(train_users)}): {list(train_users)}")
        self.logger.info(f"  Val users ({len(val_users)}): {list(val_users)}")
        self.logger.info(f"  Test users ({len(test_users)}): {list(test_users)}")
        
        # Split data
        train_mask = np.isin(users, train_users)
        val_mask = np.isin(users, val_users)
        test_mask = np.isin(users, test_users)
        
        splits = {
            'train_X': X[train_mask],
            'train_y': y[train_mask],
            'val_X': X[val_mask],
            'val_y': y[val_mask],
            'test_X': X[test_mask],
            'test_y': y[test_mask],
        }
        
        self.logger.info(f"  Train: {splits['train_X'].shape[0]:,} windows")
        self.logger.info(f"  Val:   {splits['val_X'].shape[0]:,} windows")
        self.logger.info(f"  Test:  {splits['test_X'].shape[0]:,} windows")
        self.logger.info("="*80)
        
        return splits
    
    def save_data(self, data: Dict, metadata: List[Dict], data_format: str, conversion_applied: bool):
        """Save preprocessed production data and metadata"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("SAVING PREPROCESSED DATA")
        self.logger.info("="*80)
        
        DATA_PREPARED.mkdir(parents=True, exist_ok=True)
        
        # Save production data
        output_path = DATA_PREPARED / "production_X.npy"
        np.save(output_path, data['X'])
        self.logger.info(f"  Saved: {output_path.name} {data['X'].shape}")
        
        # Save metadata
        meta = {
            'created_date': datetime.now().isoformat(),
            'data_format': data_format,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'total_windows': data['X'].shape[0],
            'conversion_applied': conversion_applied,
            'windows': metadata,
        }
        
        meta_path = DATA_PREPARED / "production_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        self.logger.info(f"  Saved: {meta_path.name}")
        
        self.logger.info("="*80)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Production preprocessing pipeline (unit auto-detection)")
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path (relative to project root)')
    args = parser.parse_args()
    
    # Setup logging
    logger_setup = PreprocessLogger("production_preprocessing")
    logger = logger_setup.get_logger()
    
    try:
        logger.info("Mode: PRODUCTION")
        logger.info(f"Input: {args.input}")
        
        # Load data
        input_path = PROJECT_ROOT / args.input
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Initialize components
        unit_detector = UnitDetector(logger)
        preprocessor = UnifiedPreprocessor(logger)
        
        # Detect data format
        data_format, sensor_cols = preprocessor.detect_data_format(df)
        
        # Detect and convert units (accelerometer only)
        accel_cols = [col for col in sensor_cols if col.startswith('A')]
        df, conversion_applied = unit_detector.process_units(df, accel_cols)
        
        # Normalize data (production only)
        df_normalized = preprocessor.normalize_data(df, sensor_cols, mode='transform')
        
        # Create windows
        X, _, metadata = preprocessor.create_windows(df_normalized, sensor_cols, data_format)
        
        # Save production data
        data = {'X': X}
        preprocessor.save_data(data, metadata, data_format, conversion_applied)
        
        logger.info("")
        logger.info("="*80)
        logger.info("✓ PREPROCESSING COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"✗ PREPROCESSING FAILED: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

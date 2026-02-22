"""
Data Preparation Pipeline for Activity Recognition Model

This script prepares sensor data for training/inference:
1. Loads raw sensor data (labeled or unlabeled)
2. Normalizes sensor values to consistent scale
3. Creates sliding windows (200 timesteps = 4 seconds at 50Hz)
4. Splits data by user (train/val/test)
5. Saves prepared datasets

Input formats:
- Training: all_users_data_labeled.csv (with activity + User columns)
- Production: sensor_fused_50Hz.csv (only sensors + timestamp)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import ACTIVITY_LABELS, DATA_PREPARED, DATA_PROCESSED, LOGS_DIR, SENSOR_COLUMNS


class DataPreparationPipeline:
    """Prepare sensor data for model training/inference"""

    def __init__(self, window_size=200, overlap=0.5, target_hz=50):
        """
        Args:
            window_size: Number of timesteps per window (200 = 4 seconds at 50Hz)
            overlap: Overlap ratio between windows (0.5 = 50% overlap)
            target_hz: Expected sample rate (50 Hz)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.target_hz = target_hz

        # Sensor column names (different in labeled vs unlabeled data)
        self.labeled_sensors = ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
        self.unlabeled_sensors = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

        self.scaler = StandardScaler()
        self.activity_to_label = {}
        self.label_to_activity = {}

    def detect_data_format(self, df):
        """Detect if data is labeled (training) or unlabeled (production)"""
        has_activity = "activity" in df.columns
        has_user = "User" in df.columns
        has_labeled_sensors = all(col in df.columns for col in self.labeled_sensors)
        has_unlabeled_sensors = all(col in df.columns for col in self.unlabeled_sensors)

        if has_activity and has_user and has_labeled_sensors:
            return "labeled", self.labeled_sensors
        elif has_unlabeled_sensors:
            return "unlabeled", self.unlabeled_sensors
        else:
            raise ValueError(f"Unknown data format. Columns: {df.columns.tolist()}")

    def normalize_sensor_values(self, df, sensor_cols, mode="fit_transform"):
        """
        Normalize sensor values to consistent scale

        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor column names
            mode: 'fit_transform' (training) or 'transform' (inference)
        """
        df_normalized = df.copy()

        if mode == "fit_transform":
            # Training: Fit scaler on training data
            df_normalized[sensor_cols] = self.scaler.fit_transform(df[sensor_cols])
            print(f"✓ Fitted scaler on {len(df)} samples")
            print(f"  Mean: {self.scaler.mean_}")
            print(f"  Std: {self.scaler.scale_}")
        elif mode == "transform":
            # Inference: Use pre-fitted scaler
            df_normalized[sensor_cols] = self.scaler.transform(df[sensor_cols])
            print(f"✓ Applied pre-fitted scaler to {len(df)} samples")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'fit_transform' or 'transform'")

        return df_normalized

    def create_activity_encoding(self, activities):
        """Create mapping between activity names and numeric labels"""
        unique_activities = sorted(activities.unique())
        self.activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}
        self.label_to_activity = {idx: activity for activity, idx in self.activity_to_label.items()}

        print(f"\n✓ Created activity encoding for {len(unique_activities)} classes:")
        for activity, label in self.activity_to_label.items():
            print(f"  {label}: {activity}")

        return self.activity_to_label

    def create_sliding_windows(self, df, sensor_cols, data_format):
        """
        Create sliding windows from time series data

        Returns:
            X: (n_windows, window_size, n_sensors) numpy array
            y: (n_windows,) numpy array (None if unlabeled)
            metadata: List of dicts with window info
        """
        n_samples = len(df)
        n_windows = (n_samples - self.window_size) // self.step_size + 1

        X = []
        y = []
        metadata = []

        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size

            if end_idx > n_samples:
                break

            # Extract window
            window_data = df.iloc[start_idx:end_idx][sensor_cols].values

            # Validate window shape
            if window_data.shape[0] != self.window_size:
                continue

            X.append(window_data)

            # Add label if available (use majority vote for window)
            if data_format == "labeled":
                window_activities = df.iloc[start_idx:end_idx]["activity"]
                majority_activity = window_activities.mode()[0]
                label = self.activity_to_label[majority_activity]
                y.append(label)
            else:
                y.append(None)  # No label for production data

            # Store metadata
            meta = {
                "window_idx": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "timestamp_start": df.iloc[start_idx][
                    "timestamp" if "timestamp" in df.columns else "timestamp_ms"
                ],
                "timestamp_end": df.iloc[end_idx - 1][
                    "timestamp" if "timestamp" in df.columns else "timestamp_ms"
                ],
            }

            if data_format == "labeled":
                meta["activity"] = majority_activity
                meta["user"] = df.iloc[start_idx]["User"]

            metadata.append(meta)

        X = np.array(X)
        y = np.array(y) if data_format == "labeled" else None

        print(f"✓ Created {len(X)} windows of shape {X.shape}")

        return X, y, metadata

    def split_by_user(self, X, y, metadata, train_users, val_users, test_users):
        """
        Split data by user to avoid data leakage

        Args:
            X, y, metadata: Output from create_sliding_windows
            train_users: List of user IDs for training (e.g., [1, 2, 3, 4])
            val_users: List of user IDs for validation (e.g., [5])
            test_users: List of user IDs for testing (e.g., [6])
        """
        # Get user for each window
        users = np.array([meta["user"] for meta in metadata])

        # Create masks
        train_mask = np.isin(users, train_users)
        val_mask = np.isin(users, val_users)
        test_mask = np.isin(users, test_users)

        # Split data
        splits = {
            "train": {
                "X": X[train_mask],
                "y": y[train_mask],
                "metadata": [m for m, mask in zip(metadata, train_mask) if mask],
            },
            "val": {
                "X": X[val_mask],
                "y": y[val_mask],
                "metadata": [m for m, mask in zip(metadata, val_mask) if mask],
            },
            "test": {
                "X": X[test_mask],
                "y": y[test_mask],
                "metadata": [m for m, mask in zip(metadata, test_mask) if mask],
            },
        }

        print(f"\n✓ Split by user:")
        print(f"  Train (users {train_users}): {len(splits['train']['X'])} windows")
        print(f"  Val   (users {val_users}):   {len(splits['val']['X'])} windows")
        print(f"  Test  (users {test_users}):  {len(splits['test']['X'])} windows")

        return splits

    def save_prepared_data(self, splits, output_dir, config):
        """Save prepared datasets to disk"""
        os.makedirs(output_dir, exist_ok=True)

        for split_name, split_data in splits.items():
            # Save arrays
            np.save(os.path.join(output_dir, f"{split_name}_X.npy"), split_data["X"])
            np.save(os.path.join(output_dir, f"{split_name}_y.npy"), split_data["y"])

            # Save metadata
            with open(os.path.join(output_dir, f"{split_name}_metadata.json"), "w") as f:
                json.dump(split_data["metadata"], f, indent=2, default=str)

            print(f"✓ Saved {split_name} split to {output_dir}")

        # Save configuration and scaler
        config["activity_to_label"] = self.activity_to_label
        config["label_to_activity"] = self.label_to_activity
        config["scaler_mean"] = self.scaler.mean_.tolist()
        config["scaler_scale"] = self.scaler.scale_.tolist()

        # Convert Path objects to strings for JSON serialization
        config["input_file"] = str(config["input_file"])
        config["output_dir"] = str(output_dir)

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved configuration to {output_dir}/config.json")

    def prepare_labeled_data(self, input_file, output_dir, train_users, val_users, test_users):
        """
        Full pipeline for labeled training data

        Args:
            input_file: Path to all_users_data_labeled.csv
            output_dir: Output directory for prepared data
            train_users, val_users, test_users: User IDs for splits
        """
        print("=" * 80)
        print("PREPARING LABELED TRAINING DATA")
        print("=" * 80)

        # Load data
        print(f"\n1. Loading data from {input_file}")
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} samples with {df.shape[1]} columns")

        # Detect format
        data_format, sensor_cols = self.detect_data_format(df)
        print(f"✓ Detected format: {data_format}")
        print(f"✓ Sensor columns: {sensor_cols}")

        # Create activity encoding
        print(f"\n2. Creating activity encoding")
        self.create_activity_encoding(df["activity"])

        # Normalize sensor values
        print(f"\n3. Normalizing sensor values")
        df_normalized = self.normalize_sensor_values(df, sensor_cols, mode="fit_transform")

        # Create sliding windows
        print(f"\n4. Creating sliding windows (size={self.window_size}, overlap={self.overlap})")
        X, y, metadata = self.create_sliding_windows(df_normalized, sensor_cols, data_format)

        # Split by user
        print(f"\n5. Splitting by user")
        splits = self.split_by_user(X, y, metadata, train_users, val_users, test_users)

        # Save data
        print(f"\n6. Saving prepared data")
        config = {
            "input_file": input_file,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "step_size": self.step_size,
            "target_hz": self.target_hz,
            "sensor_cols": sensor_cols,
            "n_classes": len(self.activity_to_label),
            "train_users": train_users,
            "val_users": val_users,
            "test_users": test_users,
            "created_at": datetime.now().isoformat(),
        }
        self.save_prepared_data(splits, output_dir, config)

        print(f"\n{'='*80}")
        print("✅ DATA PREPARATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Total windows: {len(X)}")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        print(f"\nReady for training! 🚀")


def main():
    """Main execution"""
    # Configuration
    INPUT_FILE = PROJECT_ROOT / "all_users_data_labeled.csv"
    OUTPUT_DIR = DATA_PREPARED

    # User split (6 users total: 1-4 train, 5 val, 6 test)
    TRAIN_USERS = [1, 2, 3, 4]  # 60% of data
    VAL_USERS = [5]  # 20% of data
    TEST_USERS = [6]  # 20% of data

    # Window parameters
    WINDOW_SIZE = 200  # 4 seconds at 50Hz
    OVERLAP = 0.5  # 50% overlap
    TARGET_HZ = 50

    # Create pipeline
    pipeline = DataPreparationPipeline(
        window_size=WINDOW_SIZE, overlap=OVERLAP, target_hz=TARGET_HZ
    )

    # Prepare labeled training data
    pipeline.prepare_labeled_data(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        train_users=TRAIN_USERS,
        val_users=VAL_USERS,
        test_users=TEST_USERS,
    )


if __name__ == "__main__":
    main()

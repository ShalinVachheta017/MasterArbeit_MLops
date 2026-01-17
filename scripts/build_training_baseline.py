#!/usr/bin/env python
"""
Build Training Baseline Statistics
===================================

Compute baseline feature statistics from labeled training data.
These baselines are used for drift detection in production.

Usage:
    python scripts/build_training_baseline.py
    python scripts/build_training_baseline.py --input data/raw/all_users_data_labeled.csv
    python scripts/build_training_baseline.py --output data/prepared/baseline_stats.json

Outputs:
    - data/prepared/baseline_stats.json: Feature statistics per channel
    - data/prepared/baseline_embeddings.npz: (Optional) Model embeddings

Author: Master Thesis MLOps Project
Date: January 15, 2026
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PROJECT_ROOT, DATA_RAW, DATA_PREPARED, LABELED_DATA_FILE,
    SENSOR_COLUMNS, ACTIVITY_LABELS, WINDOW_SIZE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class BaselineBuilder:
    """
    Build training baseline statistics for drift detection.
    
    Computes per-channel statistics:
    - mean, std, min, max
    - percentiles (5, 25, 50, 75, 95)
    - histogram bins
    
    Can also compute model embeddings for embedding drift detection.
    """
    
    SENSOR_NAMES = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    def __init__(self):
        self.baseline = {}
    
    def build_from_csv(self, 
                       input_path: Path,
                       sensor_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build baseline from labeled CSV data.
        
        Args:
            input_path: Path to labeled CSV file
            sensor_columns: List of sensor column names
        
        Returns:
            Dictionary with baseline statistics
        """
        logger.info(f"Loading training data from: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Determine sensor columns
        if sensor_columns is None:
            # Try common column patterns
            if 'Ax_w' in df.columns:
                sensor_columns = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
            elif 'Ax' in df.columns:
                sensor_columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
            else:
                raise ValueError(f"Cannot find sensor columns. Available: {df.columns.tolist()}")
        
        logger.info(f"Using sensor columns: {sensor_columns}")
        
        # Build baseline
        self.baseline = {
            "created_at": datetime.now().isoformat(),
            "source_file": str(input_path),
            "n_samples": len(df),
            "sensor_columns": sensor_columns,
            "per_channel": {},
            "global": {}
        }
        
        # Per-channel statistics
        for col_idx, col_name in enumerate(sensor_columns):
            if col_name not in df.columns:
                logger.warning(f"Column {col_name} not found - skipping")
                continue
            
            data = df[col_name].dropna().values
            
            # Map to standard sensor name
            std_name = self.SENSOR_NAMES[col_idx] if col_idx < len(self.SENSOR_NAMES) else col_name
            
            stats = {
                "original_column": col_name,
                "n_values": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "percentile_5": float(np.percentile(data, 5)),
                "percentile_25": float(np.percentile(data, 25)),
                "percentile_50": float(np.percentile(data, 50)),  # median
                "percentile_75": float(np.percentile(data, 75)),
                "percentile_95": float(np.percentile(data, 95)),
            }
            
            # Histogram for distribution comparison (PSI computation)
            hist, bin_edges = np.histogram(data, bins=50)
            stats["histogram"] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
            
            # Store REAL samples for KS test (scientifically defensible!)
            # Random subsample of 10k values avoids memory issues while
            # preserving the actual distribution for non-parametric tests
            max_samples = 10000
            if len(data) > max_samples:
                sample_indices = np.random.choice(len(data), max_samples, replace=False)
                samples = data[sample_indices]
            else:
                samples = data
            stats["samples"] = samples.tolist()
            stats["n_stored_samples"] = len(samples)
            
            self.baseline["per_channel"][std_name] = stats
            
            logger.info(f"  {std_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, stored {len(samples)} samples")
        
        # Global statistics (useful for sanity checks)
        all_sensor_data = df[sensor_columns].values.flatten()
        all_sensor_data = all_sensor_data[~np.isnan(all_sensor_data)]
        
        self.baseline["global"] = {
            "n_total_values": len(all_sensor_data),
            "overall_mean": float(np.mean(all_sensor_data)),
            "overall_std": float(np.std(all_sensor_data))
        }
        
        # Activity distribution (for reference)
        if 'activity' in df.columns:
            activity_counts = df['activity'].value_counts().to_dict()
            self.baseline["activity_distribution"] = {
                str(k): int(v) for k, v in activity_counts.items()
            }
            logger.info(f"\n  Activity distribution:")
            for act, count in sorted(activity_counts.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {act}: {count} ({100*count/len(df):.1f}%)")
        
        return self.baseline
    
    def build_from_numpy(self, 
                         X: np.ndarray,
                         y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Build baseline from windowed NumPy arrays.
        
        Args:
            X: Array of shape (n_windows, timesteps, channels)
            y: Optional labels array
        
        Returns:
            Dictionary with baseline statistics
        """
        logger.info(f"Building baseline from NumPy array: {X.shape}")
        
        n_windows, n_timesteps, n_channels = X.shape
        
        # Flatten to (n_samples, n_channels)
        X_flat = X.reshape(-1, n_channels)
        
        self.baseline = {
            "created_at": datetime.now().isoformat(),
            "source": "numpy_array",
            "n_windows": n_windows,
            "n_timesteps": n_timesteps,
            "n_channels": n_channels,
            "n_total_samples": len(X_flat),
            "per_channel": {},
            "per_window": {},
            "global": {}
        }
        
        # Per-channel statistics
        for ch_idx in range(n_channels):
            ch_name = self.SENSOR_NAMES[ch_idx] if ch_idx < len(self.SENSOR_NAMES) else f"channel_{ch_idx}"
            data = X_flat[:, ch_idx]
            
            stats = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "percentile_5": float(np.percentile(data, 5)),
                "percentile_25": float(np.percentile(data, 25)),
                "percentile_50": float(np.percentile(data, 50)),
                "percentile_75": float(np.percentile(data, 75)),
                "percentile_95": float(np.percentile(data, 95)),
            }
            
            # Histogram for PSI computation
            hist, bin_edges = np.histogram(data, bins=50)
            stats["histogram"] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
            
            # Store REAL samples for KS test (scientifically defensible!)
            # Random subsample of 10k values avoids memory issues while
            # preserving the actual distribution for non-parametric tests
            max_samples = 10000
            if len(data) > max_samples:
                sample_indices = np.random.choice(len(data), max_samples, replace=False)
                samples = data[sample_indices]
            else:
                samples = data
            stats["samples"] = samples.tolist()
            stats["n_stored_samples"] = len(samples)
            
            self.baseline["per_channel"][ch_name] = stats
            
            logger.info(f"  {ch_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, stored {len(samples)} samples")
        
        # Per-window statistics (for window-level drift detection)
        window_means = X.mean(axis=1)  # (n_windows, n_channels)
        window_stds = X.std(axis=1)
        
        for ch_idx in range(n_channels):
            ch_name = self.SENSOR_NAMES[ch_idx] if ch_idx < len(self.SENSOR_NAMES) else f"channel_{ch_idx}"
            self.baseline["per_window"][ch_name] = {
                "window_mean_mean": float(np.mean(window_means[:, ch_idx])),
                "window_mean_std": float(np.std(window_means[:, ch_idx])),
                "window_std_mean": float(np.mean(window_stds[:, ch_idx])),
                "window_std_std": float(np.std(window_stds[:, ch_idx]))
            }
        
        # Global statistics
        self.baseline["global"] = {
            "overall_mean": float(np.mean(X_flat)),
            "overall_std": float(np.std(X_flat))
        }
        
        # Label distribution if available
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            self.baseline["label_distribution"] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }
        
        return self.baseline
    
    def save(self, output_path: Path):
        """Save baseline to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.baseline, f, indent=2)
        
        logger.info(f"✅ Baseline saved to: {output_path}")


def build_embeddings_baseline(
    model_path: Path,
    X_train: np.ndarray,
    output_path: Path,
    layer_name: str = "bidirectional"  # BiLSTM layer name
):
    """
    Build embedding baseline using model's intermediate representations.
    
    This is optional and requires TensorFlow.
    
    Args:
        model_path: Path to trained Keras model
        X_train: Training data array
        output_path: Where to save embeddings
        layer_name: Name of layer to extract embeddings from
    """
    logger.info("Building embedding baseline...")
    
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not available - skipping embedding baseline")
        return
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Find embedding layer
    embedding_layer = None
    for layer in model.layers:
        if layer_name.lower() in layer.name.lower():
            embedding_layer = layer
            break
    
    if embedding_layer is None:
        logger.warning(f"Layer '{layer_name}' not found in model - skipping embeddings")
        return
    
    # Create embedding extractor
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output
    )
    
    # Extract embeddings (in batches to avoid memory issues)
    batch_size = 256
    embeddings = []
    
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        batch_embeddings = embedding_model.predict(batch, verbose=0)
        
        # Global average pooling if needed (for sequence output)
        if len(batch_embeddings.shape) == 3:
            batch_embeddings = np.mean(batch_embeddings, axis=1)
        
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    
    # Compute embedding statistics
    embedding_mean = embeddings.mean(axis=0)
    embedding_std = embeddings.std(axis=0)
    
    # Save embeddings summary (not all embeddings - too large)
    np.savez(
        output_path,
        mean=embedding_mean,
        std=embedding_std,
        n_samples=len(embeddings),
        sample_embeddings=embeddings[:1000]  # Keep sample for comparison
    )
    
    logger.info(f"✅ Embedding baseline saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Training Baseline")
    parser.add_argument('--input', type=str, default=None,
                       help='Path to labeled training data (CSV or NPY)')
    parser.add_argument('--output', type=str, default='data/prepared/baseline_stats.json',
                       help='Output path for baseline statistics')
    parser.add_argument('--embeddings', action='store_true',
                       help='Also compute embedding baseline (requires model)')
    parser.add_argument('--model', type=str, default='models/pretrained/fine_tuned_model_1dcnnbilstm.keras',
                       help='Path to model (for embeddings)')
    args = parser.parse_args()
    
    # Determine input file
    # PRIORITY ORDER:
    # 1. Explicit --input argument
    # 2. Windowed X_train.npy (PREFERRED - same preprocessing as production!)
    # 3. Raw labeled CSV (fallback - may have different scaling/normalization)
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            input_path = PROJECT_ROOT / args.input
    else:
        # PREFER windowed data to match production preprocessing!
        # This is critical: baseline MUST come from same preprocessing stage
        if (DATA_PREPARED / "X_train.npy").exists():
            input_path = DATA_PREPARED / "X_train.npy"
            logger.info("📋 Using windowed X_train.npy (matches production preprocessing)")
        elif LABELED_DATA_FILE.exists():
            input_path = LABELED_DATA_FILE
            logger.warning("⚠️ Using raw CSV - may not match production preprocessing!")
        else:
            # Try to find any labeled data
            csv_files = list(DATA_RAW.glob("*labeled*.csv"))
            if csv_files:
                input_path = csv_files[0]
                logger.warning("⚠️ Using raw CSV - may not match production preprocessing!")
            else:
                logger.error("❌ No training data found. Specify with --input")
                sys.exit(1)
    
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"{'='*60}")
    logger.info("📊 BUILDING TRAINING BASELINE")
    logger.info(f"{'='*60}")
    
    # Build baseline
    builder = BaselineBuilder()
    
    if input_path.suffix == '.npy':
        X = np.load(input_path)
        # Try to load labels
        y_path = input_path.parent / input_path.name.replace('X_', 'y_')
        y = np.load(y_path) if y_path.exists() else None
        builder.build_from_numpy(X, y)
    else:
        builder.build_from_csv(input_path)
    
    # Save baseline
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.output
    
    builder.save(output_path)
    
    # Build embeddings if requested
    if args.embeddings:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / args.model
        
        if model_path.exists():
            # Load training data for embeddings
            if input_path.suffix == '.npy':
                X_train = np.load(input_path)
            else:
                # Need windowed data for embeddings
                X_train_path = DATA_PREPARED / "X_train.npy"
                if X_train_path.exists():
                    X_train = np.load(X_train_path)
                else:
                    logger.warning("No windowed X_train.npy found - skipping embeddings")
                    return
            
            embeddings_path = output_path.parent / "baseline_embeddings.npz"
            build_embeddings_baseline(model_path, X_train, embeddings_path)
        else:
            logger.warning(f"Model not found: {model_path} - skipping embeddings")
    
    logger.info("\n✅ Baseline building complete!")


if __name__ == "__main__":
    main()

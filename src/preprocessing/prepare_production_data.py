"""
Production Data Preprocessing Pipeline
=======================================
Preprocesses unlabeled production data using the SAME scaler and windowing
strategy as the training data preparation.

Input:  data/processed/sensor_fused_50Hz.csv (unlabeled)
Output: data/prepared/production_X.npy (windows, no labels)
        data/prepared/production_metadata.json

Key Difference from Training Pipeline:
- NO labels (production data is unlabeled)
- Uses SAVED scaler from training data (no fitting!)
- Same windowing (200 samples, 50% overlap)
- Same sensor columns and order

Usage:
------
python src/preprocessing/prepare_production_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT,
    DATA_PROCESSED,
    DATA_PREPARED,
    SENSOR_COLUMNS,
    WINDOW_SIZE,
    OVERLAP
)


def load_scaler_config():
    """Load scaler parameters from training data preparation."""
    config_path = DATA_PREPARED / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Scaler config not found: {config_path}\n"
            f"Please run prepare_training_data.py first!"
        )
    
    print(f"Loading scaler config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Extract scaler parameters (scale = std for StandardScaler)
    scaler_mean = np.array(config['scaler_mean'])
    scaler_scale = np.array(config['scaler_scale'])  # scale = std in StandardScaler
    
    print(f"Scaler parameters loaded:")
    print(f"  Mean:  {scaler_mean}")
    print(f"  Scale: {scaler_scale}")
    
    return scaler_mean, scaler_scale, config


def load_production_data():
    """Load unlabeled production data."""
    csv_path = DATA_PROCESSED / "sensor_fused_50Hz.csv"
    
    print(f"\nLoading production data from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Production data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_production_data(df, scaler_mean, scaler_scale):
    """Preprocess production data using saved scaler parameters."""
    print(f"\n{'='*60}")
    print("PREPROCESSING PRODUCTION DATA")
    print(f"{'='*60}")
    
    # Map column names (production data has different names)
    # Production: ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    # Training:   ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
    
    production_sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    # Check if columns exist
    if not all(col in df.columns for col in production_sensor_cols):
        raise ValueError(
            f"Missing sensor columns in production data!\n"
            f"Expected: {production_sensor_cols}\n"
            f"Found: {list(df.columns)}"
        )
    
    # Extract sensor data
    print(f"\nStep 1: Extracting sensor data...")
    X = df[production_sensor_cols].values
    print(f"  Shape: {X.shape}")
    print(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Mean: {X.mean():.3f}")
    
    # Check for NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\n  WARNING: Found {nan_count} NaN values!")
        print(f"  Strategy: Forward fill + backward fill + drop remaining")
        
        # Fill NaN values
        df_clean = df[production_sensor_cols].fillna(method='ffill').fillna(method='bfill')
        X = df_clean.values
        
        # If still NaN, drop those rows
        valid_rows = ~np.isnan(X).any(axis=1)
        X = X[valid_rows]
        df = df[valid_rows].reset_index(drop=True)
        
        print(f"  After cleaning: {len(X)} samples (removed {nan_count - np.isnan(X).sum()})")
    
    # Apply StandardScaler using SAVED parameters (NO fitting!)
    print(f"\nStep 2: Applying StandardScaler (using training scaler)...")
    print(f"  Using saved mean:  {scaler_mean}")
    print(f"  Using saved scale: {scaler_scale}")
    
    X_scaled = (X - scaler_mean) / scaler_scale
    
    print(f"  After scaling:")
    print(f"    Range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"    Mean: {X_scaled.mean():.3f}")
    print(f"    Std: {X_scaled.std():.3f}")
    
    # Create sliding windows
    print(f"\nStep 3: Creating sliding windows...")
    print(f"  Window size: {WINDOW_SIZE} samples")
    print(f"  Overlap: {OVERLAP*100:.0f}%")
    
    stride = int(WINDOW_SIZE * (1 - OVERLAP))
    print(f"  Stride: {stride} samples")
    
    windows = []
    metadata = []
    
    for i in range(0, len(X_scaled) - WINDOW_SIZE + 1, stride):
        window = X_scaled[i:i + WINDOW_SIZE]
        
        if len(window) == WINDOW_SIZE:
            windows.append(window)
            
            # Store metadata (for tracking, no labels!)
            window_meta = {
                'window_id': len(windows) - 1,
                'start_index': int(i),
                'end_index': int(i + WINDOW_SIZE),
                'timestamp_start': str(df.iloc[i]['timestamp_iso']) if 'timestamp_iso' in df.columns else None,
                'timestamp_end': str(df.iloc[i + WINDOW_SIZE - 1]['timestamp_iso']) if 'timestamp_iso' in df.columns else None
            }
            metadata.append(window_meta)
    
    X_windows = np.array(windows, dtype=np.float32)
    
    print(f"  Created {len(X_windows)} windows")
    print(f"  Window shape: {X_windows.shape}")
    
    return X_windows, metadata


def save_production_data(X_windows, metadata):
    """Save preprocessed production data."""
    print(f"\n{'='*60}")
    print("SAVING PREPROCESSED PRODUCTION DATA")
    print(f"{'='*60}")
    
    # Ensure output directory exists
    DATA_PREPARED.mkdir(parents=True, exist_ok=True)
    
    # Save windows
    output_X = DATA_PREPARED / "production_X.npy"
    np.save(output_X, X_windows)
    print(f"Saved windows: {output_X}")
    print(f"  Shape: {X_windows.shape}")
    print(f"  Size: {X_windows.nbytes / 1024 / 1024:.2f} MB")
    
    # Save metadata
    output_meta = DATA_PREPARED / "production_metadata.json"
    
    meta_summary = {
        'created_date': datetime.now().isoformat(),
        'source_file': 'data/processed/sensor_fused_50Hz.csv',
        'preprocessing_pipeline': 'prepare_production_data.py',
        'total_windows': len(X_windows),
        'window_size': WINDOW_SIZE,
        'overlap': OVERLAP,
        'sensor_columns': SENSOR_COLUMNS,
        'normalization': 'StandardScaler (from training data)',
        'scaler_source': 'data/prepared/config.json',
        'has_labels': False,
        'purpose': 'Production inference',
        'windows': metadata
    }
    
    with open(output_meta, 'w', encoding='utf-8') as f:
        json.dump(meta_summary, f, indent=2)
    
    print(f"Saved metadata: {output_meta}")
    print(f"  Total windows: {len(metadata)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Production windows: {len(X_windows)}")
    print(f"Window shape: {X_windows.shape}")
    print(f"Data range: [{X_windows.min():.3f}, {X_windows.max():.3f}]")
    print(f"Data mean: {X_windows.mean():.3f}")
    print(f"Data std: {X_windows.std():.3f}")
    
    return output_X, output_meta


def compare_with_training_data():
    """Compare production data distribution with training data."""
    print(f"\n{'='*60}")
    print("COMPARING WITH TRAINING DATA")
    print(f"{'='*60}")
    
    # Load training data
    train_X_path = DATA_PREPARED / "train_X.npy"
    production_X_path = DATA_PREPARED / "production_X.npy"
    
    if not train_X_path.exists():
        print("Training data not found - skipping comparison")
        return
    
    train_X = np.load(train_X_path)
    production_X = np.load(production_X_path)
    
    print(f"\nTraining Data:")
    print(f"  Shape: {train_X.shape}")
    print(f"  Range: [{train_X.min():.3f}, {train_X.max():.3f}]")
    print(f"  Mean: {train_X.mean():.3f}")
    print(f"  Std: {train_X.std():.3f}")
    
    print(f"\nProduction Data:")
    print(f"  Shape: {production_X.shape}")
    print(f"  Range: [{production_X.min():.3f}, {production_X.max():.3f}]")
    print(f"  Mean: {production_X.mean():.3f}")
    print(f"  Std: {production_X.std():.3f}")
    
    # Check if distributions are similar
    print(f"\nDistribution Check:")
    
    mean_diff = abs(train_X.mean() - production_X.mean())
    std_diff = abs(train_X.std() - production_X.std())
    
    print(f"  Mean difference: {mean_diff:.4f} {'✓ GOOD' if mean_diff < 0.1 else '⚠ CHECK'}")
    print(f"  Std difference:  {std_diff:.4f} {'✓ GOOD' if std_diff < 0.1 else '⚠ CHECK'}")
    
    if mean_diff < 0.1 and std_diff < 0.1:
        print(f"\n✓ Production data distribution MATCHES training data!")
        print(f"  Ready for inference with trained model.")
    else:
        print(f"\n⚠ Production data distribution DIFFERS from training data!")
        print(f"  Possible data drift - review preprocessing steps.")


def create_inference_readme():
    """Create README for production data usage."""
    readme_path = DATA_PREPARED / "PRODUCTION_DATA_README.md"
    
    content = """# Production Data - Preprocessed for Inference

**Created:** {date}  
**Source:** `data/processed/sensor_fused_50Hz.csv`  
**Pipeline:** `src/preprocessing/prepare_production_data.py`

---

## Files

- `production_X.npy` - Preprocessed sensor windows (no labels)
- `production_metadata.json` - Window metadata (timestamps, indices)

---

## Preprocessing Applied

1. **Sensor Extraction:** Ax, Ay, Az, Gx, Gy, Gz
2. **Normalization:** StandardScaler (using training data parameters)
3. **Windowing:** 200 samples, 50% overlap
4. **Output:** (N, 200, 6) float32 array

---

## Usage for Inference

```python
import numpy as np
import tensorflow as tf

# Load preprocessed data
X = np.load('data/prepared/production_X.npy')
print(f"Shape: {{X.shape}}")  # (N, 200, 6)

# Load model
model = tf.keras.models.load_model('models/pretrained/fine_tuned_model_1dcnnbilstm.keras')

# Predict
predictions = model.predict(X)
pred_classes = np.argmax(predictions, axis=1)
pred_confidence = np.max(predictions, axis=1)

print(f"Predicted classes: {{pred_classes[:10]}}")
print(f"Confidence scores: {{pred_confidence[:10]}}")
```

---

## Comparison with Training Data

- Same normalization (StandardScaler from training)
- Same window size (200 samples)
- Same overlap (50%)
- Same sensor order (Ax, Ay, Az, Gx, Gy, Gz)

**Ready for production inference!** ✓

---

## Next Steps

1. Run inference: `python src/inference/predict.py`
2. Monitor predictions: `python src/monitoring/drift_detection.py`
3. Analyze results: `python src/evaluation/analyze_production.py`
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nCreated README: {readme_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("PRODUCTION DATA PREPROCESSING PIPELINE")
    print("="*60)
    print("Goal: Preprocess unlabeled production data using")
    print("      SAME scaler and windowing as training data")
    print("="*60)
    
    try:
        # Step 1: Load scaler config from training data
        scaler_mean, scaler_scale, config = load_scaler_config()
        
        # Step 2: Load production data
        df = load_production_data()
        
        # Step 3: Preprocess production data
        X_windows, metadata = preprocess_production_data(df, scaler_mean, scaler_scale)
        
        # Step 4: Save preprocessed data
        output_X, output_meta = save_production_data(X_windows, metadata)
        
        # Step 5: Compare distributions
        compare_with_training_data()
        
        # Step 6: Create README
        create_inference_readme()
        
        print(f"\n{'='*60}")
        print("✓ PRODUCTION DATA PREPROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"\nOutput files:")
        print(f"  1. {output_X}")
        print(f"  2. {output_meta}")
        print(f"  3. {DATA_PREPARED / 'PRODUCTION_DATA_README.md'}")
        
        print(f"\nNext steps:")
        print(f"  1. Run inference on production data")
        print(f"  2. Monitor prediction distribution")
        print(f"  3. Detect data drift vs training data")
        print(f"  4. Compare model performance (if labels available)")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        raise


if __name__ == "__main__":
    main()

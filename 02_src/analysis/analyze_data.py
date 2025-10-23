"""
Data Analysis Script for MLOps Thesis Project

This script analyzes the existing data files to understand:
1. Data structure and format
2. Presence of labels/classes
3. Data quality and statistics
4. Recommendations for training pipeline

Run this BEFORE creating training scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths - navigate to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "01_data" / "raw"
PROCESSED_DIR = BASE_DIR / "01_data" / "processed"
OUTPUT_DIR = BASE_DIR / "05_outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
F_DATA_PATH = BASE_DIR / "01_data" / "samples" / "f_data_50hz.csv"
SENSOR_FUSED_PATH = PROCESSED_DIR / "sensor_fused_50Hz.csv"
META_PATH = PROCESSED_DIR / "sensor_fused_meta.json"

print("="*80)
print("DATA ANALYSIS FOR MLOPS THESIS PROJECT")
print("="*80)

# ============================================================================
# 1. ANALYZE f_data_50hz.csv
# ============================================================================
print("\n" + "="*80)
print("1. ANALYZING: f_data_50hz.csv")
print("="*80)

if F_DATA_PATH.exists():
    try:
        # Load data (handle potential whitespace in column names)
        df_f = pd.read_csv(F_DATA_PATH)
        
        # Clean column names (strip whitespace)
        df_f.columns = df_f.columns.str.strip()
        
        print(f"\n📊 Basic Information:")
        print(f"   - Shape: {df_f.shape[0]:,} rows × {df_f.shape[1]} columns")
        print(f"   - Size: {F_DATA_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   - Memory usage: {df_f.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        print(f"\n📋 Columns:")
        for i, col in enumerate(df_f.columns, 1):
            print(f"   {i}. {col:<25} (dtype: {df_f[col].dtype})")
        
        # Check for label column
        potential_label_cols = ['label', 'class', 'target', 'activity', 'state', 'anxiety_level']
        has_labels = any(col in df_f.columns.str.lower() for col in potential_label_cols)
        
        if has_labels:
            label_col = [col for col in df_f.columns if col.lower() in potential_label_cols][0]
            print(f"\n✅ LABELS FOUND: Column '{label_col}'")
            print(f"   - Unique values: {df_f[label_col].unique()}")
            print(f"   - Value counts:")
            print(df_f[label_col].value_counts().to_string())
        else:
            print(f"\n❌ NO LABELS FOUND")
            print(f"   - This appears to be UNLABELED sensor data")
            print(f"   - You will need to:")
            print(f"     1. Obtain labeled data from your mentor")
            print(f"     2. OR manually label this data")
            print(f"     3. OR use the model to generate pseudo-labels")
        
        # Statistical summary
        print(f"\n📈 Statistical Summary (Sensor Columns):")
        sensor_cols = [col for col in df_f.columns if col not in ['timestamp', 'label', 'class']]
        print(df_f[sensor_cols].describe().to_string())
        
        # Check for missing values
        missing = df_f.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing Values Detected:")
            print(missing[missing > 0].to_string())
        else:
            print(f"\n✅ No missing values")
        
        # Time range
        if 'timestamp' in df_f.columns:
            df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], errors='coerce')
            print(f"\n⏰ Time Range:")
            print(f"   - Start: {df_f['timestamp'].min()}")
            print(f"   - End: {df_f['timestamp'].max()}")
            print(f"   - Duration: {df_f['timestamp'].max() - df_f['timestamp'].min()}")
            
            # Note: If timestamp shows year 2005, it's likely a placeholder
            if df_f['timestamp'].min().year < 2020:
                print(f"   ⚠️  WARNING: Timestamps show year {df_f['timestamp'].min().year}")
                print(f"   This might be a placeholder or demo data timestamp issue")
        
        # Save summary
        summary_f = {
            "file": "f_data_50hz.csv",
            "shape": df_f.shape,
            "columns": df_f.columns.tolist(),
            "has_labels": has_labels,
            "sensor_columns": sensor_cols,
            "missing_values": missing.to_dict(),
            "statistics": df_f[sensor_cols].describe().to_dict()
        }
        
        with open(OUTPUT_DIR / "f_data_analysis.json", "w") as f:
            json.dump(summary_f, f, indent=2, default=str)
        
        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        
        # Plot sensor distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Sensor Data Distributions - f_data_50hz.csv', fontsize=16)
        
        for idx, col in enumerate(sensor_cols[:6]):  # Plot first 6 sensor columns
            ax = axes[idx // 3, idx % 3]
            ax.hist(df_f[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "f_data_distributions.png", dpi=150)
        print(f"   ✅ Saved: {OUTPUT_DIR / 'f_data_distributions.png'}")
        
        # Plot time series sample (first 1000 points)
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        fig.suptitle('Sample Time Series - First 1000 Points', fontsize=16)
        
        sample_data = df_f.head(1000)
        
        # Accelerometer
        axes[0].plot(sample_data[sensor_cols[0]], label=sensor_cols[0], alpha=0.7)
        axes[0].plot(sample_data[sensor_cols[1]], label=sensor_cols[1], alpha=0.7)
        axes[0].plot(sample_data[sensor_cols[2]], label=sensor_cols[2], alpha=0.7)
        axes[0].set_title('Accelerometer (First 3 columns)')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gyroscope
        axes[1].plot(sample_data[sensor_cols[3]], label=sensor_cols[3], alpha=0.7)
        axes[1].plot(sample_data[sensor_cols[4]], label=sensor_cols[4], alpha=0.7)
        axes[1].plot(sample_data[sensor_cols[5]], label=sensor_cols[5], alpha=0.7)
        axes[1].set_title('Gyroscope (Last 3 columns)')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "f_data_timeseries_sample.png", dpi=150)
        print(f"   ✅ Saved: {OUTPUT_DIR / 'f_data_timeseries_sample.png'}")
        
        plt.close('all')
        
    except Exception as e:
        print(f"\n❌ Error loading f_data_50hz.csv: {e}")
else:
    print(f"\n❌ File not found: {F_DATA_PATH}")

# ============================================================================
# 2. ANALYZE sensor_fused_50Hz.csv
# ============================================================================
print("\n" + "="*80)
print("2. ANALYZING: sensor_fused_50Hz.csv")
print("="*80)

if SENSOR_FUSED_PATH.exists():
    try:
        df_sensor = pd.read_csv(SENSOR_FUSED_PATH)
        
        print(f"\n📊 Basic Information:")
        print(f"   - Shape: {df_sensor.shape[0]:,} rows × {df_sensor.shape[1]} columns")
        print(f"   - Size: {SENSOR_FUSED_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   - Memory usage: {df_sensor.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        print(f"\n📋 Columns:")
        for i, col in enumerate(df_sensor.columns, 1):
            print(f"   {i}. {col:<25} (dtype: {df_sensor[col].dtype})")
        
        # Check for labels
        has_labels_sensor = any(col.lower() in ['label', 'class', 'target'] for col in df_sensor.columns)
        
        if has_labels_sensor:
            print(f"\n✅ LABELS FOUND in sensor_fused_50Hz.csv")
        else:
            print(f"\n❌ NO LABELS in sensor_fused_50Hz.csv")
            print(f"   - This is output from your preprocessing pipeline")
            print(f"   - This data needs to be labeled for training")
        
        # Statistical summary
        print(f"\n📈 Statistical Summary (Sensor Columns):")
        sensor_cols_fused = [col for col in df_sensor.columns if col not in ['timestamp_ms', 'timestamp_iso', 'label']]
        print(df_sensor[sensor_cols_fused].describe().to_string())
        
        # Check for missing values
        missing_sensor = df_sensor.isnull().sum()
        if missing_sensor.any():
            print(f"\n⚠️  Missing Values Detected:")
            print(missing_sensor[missing_sensor > 0].to_string())
        else:
            print(f"\n✅ No missing values")
        
        # Time range
        if 'timestamp_iso' in df_sensor.columns:
            df_sensor['timestamp_iso'] = pd.to_datetime(df_sensor['timestamp_iso'], errors='coerce')
            print(f"\n⏰ Time Range:")
            print(f"   - Start: {df_sensor['timestamp_iso'].min()}")
            print(f"   - End: {df_sensor['timestamp_iso'].max()}")
            print(f"   - Duration: {df_sensor['timestamp_iso'].max() - df_sensor['timestamp_iso'].min()}")
        
        # Load metadata
        if META_PATH.exists():
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            print(f"\n📝 Preprocessing Metadata:")
            print(f"   - Target frequency: {meta['config']['target_hz']} Hz")
            print(f"   - Merge tolerance: {meta['config']['tolerance_ms']} ms")
            print(f"   - Native samples: {meta['rows']['native']:,}")
            print(f"   - Resampled samples: {meta['rows']['resampled']:,}")
        
    except Exception as e:
        print(f"\n❌ Error loading sensor_fused_50Hz.csv: {e}")
else:
    print(f"\n❌ File not found: {SENSOR_FUSED_PATH}")

# ============================================================================
# 3. COMPARISON & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("3. COMPARISON & RECOMMENDATIONS")
print("="*80)

if F_DATA_PATH.exists() and SENSOR_FUSED_PATH.exists():
    print(f"\n📊 Data Files Comparison:")
    print(f"   {'File':<30} {'Samples':<12} {'Columns':<10} {'Has Labels':<12}")
    print(f"   {'-'*30} {'-'*12} {'-'*10} {'-'*12}")
    print(f"   {'f_data_50hz.csv':<30} {df_f.shape[0]:>11,} {df_f.shape[1]:>9} {'TBD':<12}")
    print(f"   {'sensor_fused_50Hz.csv':<30} {df_sensor.shape[0]:>11,} {df_sensor.shape[1]:>9} {'No':<12}")

# ============================================================================
# 4. CRITICAL QUESTIONS TO ANSWER
# ============================================================================
print("\n" + "="*80)
print("4. CRITICAL QUESTIONS YOU MUST ANSWER")
print("="*80)

questions = [
    {
        "Q": "What are you classifying?",
        "Options": [
            "Binary: Calm (0) vs Anxious (1)",
            "Multi-class: Low/Medium/High anxiety (0/1/2)",
            "Activity recognition: Sitting/Walking/Running/etc.",
            "Other: ___________"
        ]
    },
    {
        "Q": "Which file has the training labels?",
        "Options": [
            "f_data_50hz.csv (but no label column visible)",
            "A separate label file (where?)",
            "Labels need to be created manually",
            "Ask mentor for labeled data"
        ]
    },
    {
        "Q": "What is the window size for sequences?",
        "Options": [
            "50 samples (1 second at 50Hz)",
            "100 samples (2 seconds)",
            "250 samples (5 seconds)",
            "Other: ___ samples"
        ]
    },
    {
        "Q": "What is the class distribution?",
        "Options": [
            "Balanced (equal samples per class)",
            "Imbalanced (need to handle)",
            "Unknown (need to check)"
        ]
    }
]

for i, q in enumerate(questions, 1):
    print(f"\n🔴 QUESTION {i}: {q['Q']}")
    for j, opt in enumerate(q['Options'], 1):
        print(f"   {j}. {opt}")

# ============================================================================
# 5. NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("5. RECOMMENDED NEXT STEPS")
print("="*80)

steps = [
    "✅ DONE: Data analysis complete",
    "🔴 CRITICAL: Install TensorFlow and inspect model architecture",
    "🔴 CRITICAL: Get labeled data or clarify label source",
    "🔴 CRITICAL: Contact mentor for model training details",
    "🟡 Next: Create data preparation script (windowing, normalization)",
    "🟡 Next: Define model architecture file",
    "🟢 Later: Implement training script with MLflow",
    "🟢 Later: Create evaluation and metrics script"
]

for i, step in enumerate(steps, 1):
    print(f"   {i}. {step}")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("6. SUMMARY")
print("="*80)

print(f"""
✅ COMPLETED:
   - Analyzed data files
   - Generated statistics and visualizations
   - Saved analysis results to: {OUTPUT_DIR}

❌ MISSING (Critical):
   - Label information (no labels found in data files)
   - Model architecture details
   - Training hyperparameters
   - Window size and data preparation strategy

📋 OUTPUTS CREATED:
   - {OUTPUT_DIR / 'f_data_analysis.json'}
   - {OUTPUT_DIR / 'f_data_distributions.png'}
   - {OUTPUT_DIR / 'f_data_timeseries_sample.png'}

📞 ACTION REQUIRED:
   1. Contact your mentor to get:
      - Labeled training data
      - Model training details
      - Classification task definition
   
   2. Install TensorFlow and inspect model:
      pip install tensorflow
      python -c "import tensorflow as tf; model = tf.keras.models.load_model('model/fine_tuned_model_1dcnnbilstm.keras'); model.summary()"
   
   3. After getting answers, I can help you build:
      - Data preparation pipeline
      - Training script with MLflow
      - Evaluation script
      - MLOps components (Docker, CI/CD, monitoring)
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE! Check the outputs in:", OUTPUT_DIR)
print("="*80)

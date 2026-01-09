 #!/usr/bin/env python3
"""
GARMIN DATA LABELING SCRIPT
============================

This script labels your Garmin sensor data with activity labels based on 
timestamp ranges. This is an independent script that does NOT modify your 
guide's notebooks.

Usage:
    python src/label_garmin_data.py

The script will:
1. Load your processed Garmin data (f_data_50hz.csv or sensor_fused_50Hz.csv)
2. Apply activity labels based on time ranges
3. Save as a new labeled CSV file for training/fine-tuning

Author: Thesis Project
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 70)
print("GARMIN DATA LABELING SCRIPT")
print("=" * 70)

# ============================================================================
# CONFIGURATION: Define your activity time ranges here
# ============================================================================
# You can add more users by copying the pattern below
# Format: (start_time, end_time, activity_name)

ACTIVITY_RANGES = {
    # User F - from your recordings
    "f": [
        ("00:04:59", "00:06:58", "ear_rubbing"),
        ("00:07:01", "00:08:56", "forehead_rubbing"),
        ("00:09:00", "00:11:10", "hair_pulling"),
        ("00:11:12", "00:13:03", "hand_scratching"),
        ("00:13:14", "00:14:55", "hand_tapping"),
        ("00:15:05", "00:16:58", "knuckles_cracking"),
        ("00:17:05", "00:19:05", "nail_biting"),
        ("00:19:14", "00:21:03", "nape_rubbing"),
        ("00:21:11", "00:22:40", "smoking"),
        ("00:23:05", "00:23:10", "smoking"),
        ("00:23:31", "00:23:45", "sitting"),
        ("00:23:51", "00:24:03", "sitting"),
        ("00:24:05", "00:25:35", "sitting"),
        ("00:25:52", "00:27:24", "standing"),
    ],
    
    # User M - from your recordings
    "m": [
        ("09:10:35", "09:12:36", "ear_rubbing"),
        ("09:12:41", "09:14:45", "forehead_rubbing"),
        ("09:14:50", "09:16:53", "hair_pulling"),
        ("09:17:02", "09:19:10", "hand_scratching"),
        ("09:19:18", "09:21:19", "hand_tapping"),
        ("09:21:31", "09:23:31", "knuckles_cracking"),
        ("09:23:36", "09:25:35", "nail_biting"),
        ("09:25:40", "09:27:43", "nape_rubbing"),
        ("09:27:55", "09:29:48", "sitting"),
        ("09:29:55", "09:32:38", "smoking"),
        ("09:32:54", "09:34:32", "standing"),
    ],
    
    # User G - from your recordings
    "g": [
        ("20:36:16", "20:38:16", "ear_rubbing"),
        ("20:38:18", "20:40:15", "forehead_rubbing"),
        ("20:40:19", "20:42:14", "hair_pulling"),
        ("20:42:22", "20:44:11", "hand_scratching"),
        ("20:44:18", "20:46:13", "hand_tapping"),
        ("20:46:18", "20:48:22", "knuckles_cracking"),
        ("20:48:27", "20:51:10", "nail_biting"),
        ("20:51:19", "20:53:13", "nape_rubbing"),
        ("20:53:17", "20:55:14", "smoking"),
        ("20:55:24", "20:57:39", "sitting"),
        ("20:57:50", "21:00:13", "standing"),
    ],
}

# ============================================================================
# STEP 1: Find and load Garmin data files
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: FINDING GARMIN DATA FILES")
print("=" * 70)

# Possible locations for Garmin data
possible_paths = [
    PROJECT_ROOT / "data" / "samples_2005 dataset" / "f_data_50hz.csv",
    PROJECT_ROOT / "data" / "preprocessed" / "sensor_fused_50Hz.csv",
    PROJECT_ROOT / "data" / "raw" / "garmin_data.csv",
]

# Find existing files
found_files = []
for path in possible_paths:
    if path.exists():
        found_files.append(path)
        print(f"✓ Found: {path}")

if not found_files:
    print("\n❌ No Garmin data files found!")
    print("Expected files in one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nPlease ensure your Garmin data is in one of these locations.")
    exit(1)

# ============================================================================
# STEP 2: Process each file
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: PROCESSING AND LABELING DATA")
print("=" * 70)

def parse_time_ranges(ranges_list):
    """Convert time string ranges to datetime.time objects."""
    parsed = []
    for start, end, label in ranges_list:
        start_time = datetime.strptime(start, "%H:%M:%S").time()
        end_time = datetime.strptime(end, "%H:%M:%S").time()
        parsed.append((start_time, end_time, label))
    return parsed

def label_data(df, activity_ranges):
    """Apply activity labels to dataframe based on timestamp."""
    # Parse time ranges
    parsed_ranges = parse_time_ranges(activity_ranges)
    
    def get_label(timestamp):
        """Get activity label for a given timestamp."""
        if pd.isna(timestamp):
            return "unknown"
        
        # Extract time from timestamp
        if isinstance(timestamp, str):
            try:
                t = pd.to_datetime(timestamp).time()
            except:
                return "unknown"
        else:
            t = timestamp.time() if hasattr(timestamp, 'time') else timestamp
        
        # Check each range
        for start, end, label in parsed_ranges:
            if start <= t <= end:
                return label
        return "unknown"
    
    # Apply labeling
    df['activity'] = df['timestamp'].apply(get_label)
    return df

def detect_user_from_timestamps(df):
    """Detect which user's time ranges match the data."""
    # Get sample timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sample_times = df['timestamp'].dt.time.head(100)
    
    for user, ranges in ACTIVITY_RANGES.items():
        parsed = parse_time_ranges(ranges)
        for start, end, _ in parsed:
            for t in sample_times:
                if start <= t <= end:
                    return user
    return None

# Process each found file
all_labeled_data = []

for file_path in found_files:
    print(f"\nProcessing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Standardize column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Check for timestamp column
    timestamp_cols = [c for c in df.columns if 'time' in c.lower()]
    if timestamp_cols:
        # Rename to standard 'timestamp'
        df = df.rename(columns={timestamp_cols[0]: 'timestamp'})
    
    if 'timestamp' not in df.columns:
        print(f"  ⚠️ No timestamp column found, skipping...")
        continue
    
    # Detect user
    user = detect_user_from_timestamps(df)
    if user:
        print(f"  ✓ Detected user: {user}")
        df = label_data(df, ACTIVITY_RANGES[user])
        df['User'] = user
    else:
        print(f"  ⚠️ Could not detect user from timestamps")
        # Try all users
        for user, ranges in ACTIVITY_RANGES.items():
            df_copy = df.copy()
            df_copy = label_data(df_copy, ranges)
            labeled_count = (df_copy['activity'] != 'unknown').sum()
            if labeled_count > 0:
                print(f"  → User {user}: {labeled_count} rows labeled")
                df_copy['User'] = user
                all_labeled_data.append(df_copy[df_copy['activity'] != 'unknown'])
        continue
    
    # Remove unknown labels
    labeled_df = df[df['activity'] != 'unknown'].copy()
    print(f"  ✓ Labeled {len(labeled_df)} rows (removed {len(df) - len(labeled_df)} unknown)")
    
    if len(labeled_df) > 0:
        all_labeled_data.append(labeled_df)
        
        # Show activity distribution
        print(f"\n  Activity distribution:")
        for activity, count in labeled_df['activity'].value_counts().items():
            print(f"    {activity}: {count}")

# ============================================================================
# STEP 3: Combine and save labeled data
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SAVING LABELED DATA")
print("=" * 70)

if all_labeled_data:
    # Combine all labeled data
    combined_df = pd.concat(all_labeled_data, ignore_index=True)
    
    # Standardize column names to match training data format
    column_mapping = {
        'Ax': 'Ax_w', 'Ay': 'Ay_w', 'Az': 'Az_w',
        'Gx': 'Gx_w', 'Gy': 'Gy_w', 'Gz': 'Gz_w',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in combined_df.columns and new_name not in combined_df.columns:
            combined_df = combined_df.rename(columns={old_name: new_name})
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w', 'activity', 'User']
    available_cols = [c for c in required_cols if c in combined_df.columns]
    
    # Save to file
    output_path = PROJECT_ROOT / "data" / "prepared" / "garmin_labeled.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_df[available_cols].to_csv(output_path, index=False)
    
    print(f"\n✓ Saved labeled data to: {output_path}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Users: {combined_df['User'].nunique()}")
    print(f"  Activities: {combined_df['activity'].nunique()}")
    
    print(f"\n  Activity summary:")
    for activity, count in combined_df['activity'].value_counts().items():
        print(f"    {activity}: {count}")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Your Garmin data is now labeled.")
    print("=" * 70)
    print(f"""
NEXT STEPS:
1. Use this labeled data for fine-tuning:
   python src/train_model.py --data data/prepared/garmin_labeled.csv

2. Or run cross-validation on it:
   python src/k_fold_evaluator.py --data data/prepared/garmin_labeled.csv

3. For preprocessing into the model format:
   python src/preprocess_data.py --input data/prepared/garmin_labeled.csv
""")

else:
    print("\n❌ No data could be labeled!")
    print("""
POSSIBLE ISSUES:
1. Your Garmin data timestamps don't match the defined time ranges
2. The data file format is different than expected

SOLUTIONS:
1. Check your Garmin data timestamps
2. Update the ACTIVITY_RANGES dictionary in this script with your actual recording times
3. Ensure your data has a 'timestamp' column

To add your own time ranges, edit the ACTIVITY_RANGES dictionary at the top of this script.
""")

print("\n" + "=" * 70)
print("SCRIPT COMPLETE")
print("=" * 70)

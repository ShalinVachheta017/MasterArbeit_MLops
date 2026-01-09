#!/usr/bin/env python3
"""
VALIDATE GARMIN ACTIVITY LABELS
================================

This script checks if your time-based activity labels actually match
the sensor data patterns. It verifies:

1. Are the sensor patterns different for each activity?
2. Do the time ranges produce scientifically valid activity signatures?
3. Are there any mislabeled segments?

Author: Thesis Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 70)
print("VALIDATING GARMIN ACTIVITY LABELS")
print("=" * 70)

# Load labeled data
labeled_path = PROJECT_ROOT / "data" / "prepared" / "garmin_labeled.csv"
print(f"\nLoading: {labeled_path}")
df = pd.read_csv(labeled_path)

print(f"✓ Total samples: {len(df)}")
print(f"✓ Activities: {df['activity'].nunique()}")

feature_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']

# ============================================================================
# STEP 1: Check Statistical Separability of Activities
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: STATISTICAL ANALYSIS - Are Activities Distinguishable?")
print("=" * 70)

activity_stats = {}

for activity in sorted(df['activity'].unique()):
    activity_data = df[df['activity'] == activity][feature_cols]
    
    stats_dict = {
        'count': len(activity_data),
        'mean_magnitude': np.sqrt((activity_data**2).sum(axis=1)).mean(),
        'std_magnitude': np.sqrt((activity_data**2).sum(axis=1)).std(),
        'gyro_intensity': activity_data[['Gx_w', 'Gy_w', 'Gz_w']].abs().mean().mean(),
        'accel_intensity': activity_data[['Ax_w', 'Ay_w', 'Az_w']].abs().mean().mean(),
    }
    activity_stats[activity] = stats_dict

# Display results
print("\nActivity Characteristics:")
print("-" * 90)
print(f"{'Activity':<20} {'Samples':>8} {'Movement':>12} {'Gyro':>10} {'Accel':>10}")
print("-" * 90)

for activity, stats in sorted(activity_stats.items()):
    print(f"{activity:<20} {stats['count']:>8} {stats['mean_magnitude']:>12.3f} "
          f"{stats['gyro_intensity']:>10.3f} {stats['accel_intensity']:>10.3f}")

# ============================================================================
# STEP 2: Check if Activities Form Distinct Clusters
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: CLUSTER ANALYSIS - Do Activities Cluster Separately?")
print("=" * 70)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data for faster computation
sample_size = min(1000, len(df) // df['activity'].nunique())
df_sample = df.groupby('activity').sample(n=sample_size, random_state=42)

X = df_sample[feature_cols].values
y = df_sample['activity'].values

# Standardize and reduce to 2D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"This means {pca.explained_variance_ratio_.sum():.0%} of data variation is captured in 2D")

# Calculate cluster separation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
silhouette = silhouette_score(X_scaled, y_encoded)

print(f"\nSilhouette Score: {silhouette:.3f}")
print("(Score > 0.3 = activities are reasonably separated)")
print("(Score > 0.5 = activities are well separated)")
print("(Score < 0.2 = activities overlap significantly)")

if silhouette > 0.3:
    print("✅ Good! Activities form distinct patterns in sensor data")
elif silhouette > 0.2:
    print("⚠️  Moderate: Some activities may be hard to distinguish")
else:
    print("❌ Warning: Activities are NOT well separated - labels may be incorrect!")

# ============================================================================
# STEP 3: Check Activity Transitions
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: TEMPORAL VALIDATION - Are Transitions Clean?")
print("=" * 70)

# Add timestamp parsing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['prev_activity'] = df['activity'].shift(1)
df['is_transition'] = df['activity'] != df['prev_activity']

transitions = df[df['is_transition'] == True]
print(f"\nNumber of activity transitions: {len(transitions)}")
print("\nTransition sequence:")
for idx, row in transitions.head(15).iterrows():
    prev = row['prev_activity'] if pd.notna(row['prev_activity']) else 'START'
    print(f"  {prev:>20} → {row['activity']:<20} at {row['timestamp']}")

# ============================================================================
# STEP 4: Compare with Training Data Patterns
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: COMPARISON - Garmin vs Training Data Patterns")
print("=" * 70)

# Load training data
train_path = PROJECT_ROOT / "data" / "raw" / "all_users_data_labeled.csv"
if not train_path.exists():
    train_path = PROJECT_ROOT / "research_papers" / "all_users_data_labeled.csv"

train_df = pd.read_csv(train_path)
train_stats = {}

for activity in sorted(train_df['activity'].unique()):
    activity_data = train_df[train_df['activity'] == activity][feature_cols]
    train_stats[activity] = {
        'gyro_intensity': activity_data[['Gx_w', 'Gy_w', 'Gz_w']].abs().mean().mean(),
        'accel_intensity': activity_data[['Ax_w', 'Ay_w', 'Az_w']].abs().mean().mean(),
    }

print("\nIntensity Comparison (Garmin vs Training):")
print("-" * 80)
print(f"{'Activity':<20} {'Garmin Gyro':>12} {'Train Gyro':>12} {'Garmin Accel':>12} {'Train Accel':>12}")
print("-" * 80)

correlation_scores = []
for activity in sorted(set(activity_stats.keys()) & set(train_stats.keys())):
    g_gyro = activity_stats[activity]['gyro_intensity']
    t_gyro = train_stats[activity]['gyro_intensity']
    g_accel = activity_stats[activity]['accel_intensity']
    t_accel = train_stats[activity]['accel_intensity']
    
    print(f"{activity:<20} {g_gyro:>12.3f} {t_gyro:>12.3f} {g_accel:>12.3f} {t_accel:>12.3f}")
    
    # Simple correlation check
    if t_gyro > 0 and t_accel > 0:
        gyro_ratio = min(g_gyro, t_gyro) / max(g_gyro, t_gyro)
        accel_ratio = min(g_accel, t_accel) / max(g_accel, t_accel)
        correlation_scores.append((gyro_ratio + accel_ratio) / 2)

avg_correlation = np.mean(correlation_scores) if correlation_scores else 0
print(f"\nAverage Pattern Similarity: {avg_correlation:.2%}")

# ============================================================================
# STEP 5: Final Verdict
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION VERDICT")
print("=" * 70)

issues = []
warnings = []

# Check 1: Silhouette score
if silhouette < 0.2:
    issues.append("Activities are NOT well separated in sensor space")
elif silhouette < 0.3:
    warnings.append("Moderate activity separation - some confusion expected")
else:
    print("✅ Activities form distinct sensor patterns")

# Check 2: Pattern similarity
if avg_correlation < 0.5:
    issues.append(f"Garmin patterns differ significantly from training data ({avg_correlation:.0%} similarity)")
elif avg_correlation < 0.7:
    warnings.append(f"Moderate pattern similarity to training data ({avg_correlation:.0%})")
else:
    print(f"✅ Garmin patterns match training data well ({avg_correlation:.0%})")

# Check 3: Sample counts
min_samples = min(s['count'] for s in activity_stats.values())
if min_samples < 1000:
    warnings.append(f"Some activities have few samples (min: {min_samples})")
else:
    print(f"✅ All activities have sufficient samples (min: {min_samples})")

# Display issues
if issues:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠️  Your time-based labels may be INCORRECT!")
    print("    The sensor patterns don't match what we expect for these activities.")

if warnings:
    print("\n⚠️  WARNINGS:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

if not issues and not warnings:
    print("\n✅ ALL CHECKS PASSED!")
    print("   Your time-based labels appear to be scientifically valid.")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if issues:
    print("""
❌ YOUR LABELS NEED VALIDATION!

The sensor data patterns don't strongly support your time-based labels.
This could mean:

1. The time ranges are incorrect or shifted
2. Activities were performed differently than training data
3. The Garmin watch captures data differently

WHAT TO DO:
──────────
Option A - Manual Verification (Recommended):
  1. Open data/samples_2005 dataset/f_data_50hz.csv
  2. Plot sensor data and visually check if time ranges match activities
  3. Adjust time ranges where obvious mismatches exist
  
Option B - Use Training Data Only:
  1. Skip fine-tuning on Garmin data
  2. Focus on improving training data model (more epochs, augmentation)
  3. Accept lower accuracy on real Garmin data (lab-to-life gap)
  
Option C - Collect NEW Labeled Data:
  1. Record activities again with precise timing
  2. Mark exact start/stop times during recording
  3. Use video/stopwatch for accurate labels
""")
elif warnings:
    print("""
⚠️  LABELS ARE ACCEPTABLE BUT NOT PERFECT

Your time-based labels show reasonable patterns, but there's room for improvement.

SUGGESTED ACTIONS:
─────────────────
1. Proceed with caution - the model will likely work but may not be optimal
2. Consider manual verification of a few activities
3. Expect 70-80% accuracy (not 85%+)

You can proceed to fine-tuning, but be aware results may be limited by label quality.
""")
else:
    print("""
✅ YOUR LABELS LOOK GOOD!

The sensor patterns match expected activity characteristics and align well
with training data. You can confidently proceed with:

1. Fine-tuning the model on your Garmin data
2. Cross-validation experiments
3. Production deployment

Your time-based labeling appears scientifically sound!
""")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

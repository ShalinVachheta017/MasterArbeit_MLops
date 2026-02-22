#!/usr/bin/env python3
"""
CROSS-VALIDATION ON YOUR GARMIN DATA
=====================================

This script runs 5-fold cross-validation on your labeled Garmin data
to see how well the model can learn YOUR data specifically.

This will show:
1. If the Garmin data quality is good enough for training
2. Expected accuracy when fine-tuning on your data
3. Comparison with training data performance (76.5%)
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 70)
print("5-FOLD CROSS-VALIDATION ON YOUR GARMIN DATA")
print("=" * 70)

# Load labeled Garmin data
garmin_path = PROJECT_ROOT / "data" / "prepared" / "garmin_labeled.csv"
print(f"\nLoading: {garmin_path}")

df = pd.read_csv(garmin_path)
print(f"✓ Shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"✓ Activities: {df['activity'].nunique()}")

print("\nActivity Distribution:")
print(df["activity"].value_counts())

# Prepare features
feature_cols = ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]

# Check column names
available_cols = [c for c in feature_cols if c in df.columns]
if len(available_cols) < 6:
    # Try alternate names
    alt_mapping = {
        "Ax": "Ax_w",
        "Ay": "Ay_w",
        "Az": "Az_w",
        "Gx": "Gx_w",
        "Gy": "Gy_w",
        "Gz": "Gz_w",
    }
    for old, new in alt_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

print(f"\nUsing feature columns: {feature_cols}")
X_raw = df[feature_cols].values
y_raw = df["activity"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_
print(f"Classes: {classes}")

# Create sliding windows
WINDOW_SIZE = 200
STEP_SIZE = 100


def create_windows(X, y, window_size=200, step_size=100):
    windows = []
    labels = []

    for i in range(0, len(X) - window_size + 1, step_size):
        window = X[i : i + window_size]
        window_labels = y[i : i + window_size]
        majority_label = Counter(window_labels).most_common(1)[0][0]
        windows.append(window)
        labels.append(majority_label)

    return np.array(windows), np.array(labels)


print(f"\nCreating windows: size={WINDOW_SIZE}, step={STEP_SIZE}")
X_windows, y_windows = create_windows(X_raw, y_encoded, WINDOW_SIZE, STEP_SIZE)
print(f"✓ Created {len(X_windows)} windows")

# Check data statistics
print(f"\nData statistics (before standardization):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: mean={X_raw[:, i].mean():.3f}, std={X_raw[:, i].std():.3f}")


# Model definition
def create_model(input_shape=(200, 6), num_classes=11):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(64, 3, activation="relu", input_shape=input_shape),
            tf.keras.layers.Conv1D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv1D(128, 3, activation="relu"),
            tf.keras.layers.Conv1D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# 5-Fold Cross-Validation
print("\n" + "=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_windows, y_windows), 1):
    print(f"\n{'─' * 50}")
    print(f"FOLD {fold}/5")
    print(f"{'─' * 50}")

    X_train, X_val = X_windows[train_idx], X_windows[val_idx]
    y_train, y_val = y_windows[train_idx], y_windows[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Standardize
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 6)
    X_val_flat = X_val.reshape(-1, 6)

    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

    # Train
    model = create_model()
    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=10,
        batch_size=64,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    fold_accuracies.append(val_acc)

    predictions = model.predict(X_val_scaled, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    pred_dist = Counter(pred_classes)

    print(f"\n✓ Fold {fold} Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  Predictions: {len(set(pred_classes))} different classes predicted")

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\nFold Accuracies:")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"  Fold {i}: {acc:.4f} ({acc*100:.1f}%)")

print(f"\n{'─' * 50}")
print(f"GARMIN DATA MEAN ACCURACY: {mean_acc:.4f} ({mean_acc*100:.1f}%)")
print(f"STD DEVIATION: {std_acc:.4f}")
print(f"{'─' * 50}")

print(
    f"""
COMPARISON:
  Training data (all_users_data_labeled.csv): 76.5%
  Your Garmin data (garmin_labeled.csv):      {mean_acc*100:.1f}%
  
{'✅ Your Garmin data works well!' if mean_acc > 0.70 else '⚠️ Garmin data may need preprocessing adjustments'}
"""
)

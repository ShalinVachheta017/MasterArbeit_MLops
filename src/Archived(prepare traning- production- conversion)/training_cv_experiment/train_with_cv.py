#!/usr/bin/env python3
"""
TRAINING SCRIPT WITH 5-FOLD CROSS-VALIDATION
=============================================

This script implements the training methodology from ICTH_16 paper:
1. 5-Fold Cross-Validation to measure model performance
2. Train final model on 100% data with 100 epochs
3. Save the best model for deployment

Scientific Basis:
- ICTH_16: "5-fold cross-validation demonstrates robust mean accuracy of 87.0% (± 1.2%)"
- EHB_2025_71: "HAR model achieved stable performance across 5-fold cross-validation"

Usage:
    python src/train_with_cv.py --epochs 100

Author: Thesis Project
Date: 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

# ML imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# CONFIGURATION
# ============================================================================
WINDOW_SIZE = 200  # 4 seconds at 50Hz (per ICTH_16)
STEP_SIZE = 100    # 50% overlap (per ICTH_16)
N_FOLDS = 5        # 5-fold CV (per ICTH_16)
BATCH_SIZE = 64
FEATURE_COLS = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']

# ============================================================================
# DATA LOADING
# ============================================================================
def load_training_data():
    """Load the labeled training data (all_users_data_labeled.csv)."""
    data_path = PROJECT_ROOT / "data" / "raw" / "all_users_data_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "research_papers" / "all_users_data_labeled.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    print(f"Loading training data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")
    print(f"  Users: {df['User'].nunique()}")
    print(f"  Activities: {df['activity'].nunique()}")
    
    return df

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def create_windows(X, y, window_size=200, step_size=100):
    """
    Create sliding windows from sensor data.
    
    Per ICTH_16: "window size of 200 time steps (4 seconds at 50Hz) 
    with 50% overlap between consecutive windows"
    """
    windows = []
    labels = []
    
    for i in range(0, len(X) - window_size + 1, step_size):
        window = X[i:i + window_size]
        window_labels = y[i:i + window_size]
        # Use majority label in window
        majority_label = Counter(window_labels).most_common(1)[0][0]
        windows.append(window)
        labels.append(majority_label)
    
    return np.array(windows), np.array(labels)

def prepare_data(df):
    """Prepare data for training."""
    # Extract features and labels
    X = df[FEATURE_COLS].values
    y = df['activity'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create windows
    print(f"\nCreating windows: size={WINDOW_SIZE}, step={STEP_SIZE}")
    X_windows, y_windows = create_windows(X, y_encoded, WINDOW_SIZE, STEP_SIZE)
    print(f"  Created {len(X_windows)} windows")
    
    return X_windows, y_windows, label_encoder

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
def create_model(input_shape=(200, 6), num_classes=11):
    """
    Create 1D-CNN-BiLSTM model per ICTH_16 paper.
    
    Architecture per ICTH_16:
    - 1D CNN layers for feature extraction
    - BiLSTM layers for temporal dependencies
    - Batch normalization and dropout for regularization
    """
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=input_shape),
        
        # CNN Block 1
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.25),
        
        # CNN Block 2
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.25),
        
        # BiLSTM layers (per ICTH_16: "BiLSTM layers model temporal dependencies")
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        # Dense layers
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================
def run_cross_validation(X, y, num_classes, epochs=100):
    """
    Run 5-fold cross-validation per ICTH_16 methodology.
    
    Per ICTH_16: "5-fold cross-validation protocol. The data was partitioned 
    into five folds. In each iteration, four folds were used for training, 
    and the remaining fold was used for testing."
    """
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION (ICTH_16 Methodology)")
    print("=" * 70)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'─' * 60}")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"{'─' * 60}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # Standardize (fit on train, transform both)
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, 6)
        X_val_flat = X_val.reshape(-1, 6)
        
        scaler.fit(X_train_flat)
        X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # Create and train model
        model = create_model(input_shape=(WINDOW_SIZE, 6), num_classes=num_classes)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
        
        # Get per-class metrics
        predictions = model.predict(X_val_scaled, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import classification_report, f1_score
        f1 = f1_score(y_val, pred_classes, average='macro')
        
        fold_results.append({
            'fold': fold,
            'accuracy': val_acc,
            'f1_score': f1,
            'best_epoch': len(history.history['accuracy'])
        })
        
        print(f"\n✓ Fold {fold} Results:")
        print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Best Epoch: {fold_results[-1]['best_epoch']}")
        
        # Clean up
        del model
        keras.backend.clear_session()
    
    return fold_results

# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================
def train_final_model(X, y, num_classes, epochs=100, scaler_params=None):
    """
    Train the final model on 100% of data.
    
    Per ICTH_16: After CV validation, train final model on all data.
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON 100% DATA")
    print("=" * 70)
    
    # Standardize all data
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 6)
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    
    # Save scaler parameters
    scaler_config = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    print(f"\nScaler parameters:")
    print(f"  Mean: {scaler.mean_}")
    print(f"  Scale: {scaler.scale_}")
    
    # Create model
    model = create_model(input_shape=(WINDOW_SIZE, 6), num_classes=num_classes)
    
    # Split for validation during training (10%)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpoint
    model_dir = PROJECT_ROOT / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / "best_model.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print(f"\nTraining for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Final evaluation
    final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Final Model Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    
    return model, scaler_config, history

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train HAR model with 5-fold CV')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HAR MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Methodology: ICTH_16 Paper")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)
    
    # Load data
    df = load_training_data()
    
    # Prepare data
    X, y, label_encoder = prepare_data(df)
    num_classes = len(label_encoder.classes_)
    
    print(f"\nClasses ({num_classes}):")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")
    
    # Run cross-validation
    if not args.skip_cv:
        cv_results = run_cross_validation(X, y, num_classes, epochs=args.epochs)
        
        # Summary
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        
        accuracies = [r['accuracy'] for r in cv_results]
        f1_scores = [r['f1_score'] for r in cv_results]
        
        print("\nFold Results:")
        for r in cv_results:
            print(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}, F1={r['f1_score']:.4f}")
        
        print(f"\n{'─' * 50}")
        print(f"MEAN ACCURACY: {np.mean(accuracies):.4f} (± {np.std(accuracies):.4f})")
        print(f"MEAN F1-SCORE: {np.mean(f1_scores):.4f} (± {np.std(f1_scores):.4f})")
        print(f"{'─' * 50}")
        
        # Compare with ICTH_16
        print(f"\nComparison with ICTH_16 Paper:")
        print(f"  ICTH_16 Baseline (ADAMSense): 89.11%")
        print(f"  ICTH_16 After Fine-tuning:    87.0% (± 1.2%)")
        print(f"  Our Result:                   {np.mean(accuracies)*100:.1f}% (± {np.std(accuracies)*100:.1f}%)")
    
    # Train final model
    model, scaler_config, history = train_final_model(X, y, num_classes, epochs=args.epochs)
    
    # Save artifacts
    output_dir = PROJECT_ROOT / "models" / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save(output_dir / "har_model_final.keras")
    print(f"\n✓ Model saved to: {output_dir / 'har_model_final.keras'}")
    
    # Save scaler config
    import json
    with open(output_dir / "scaler_config.json", "w") as f:
        json.dump(scaler_config, f, indent=2)
    print(f"✓ Scaler config saved to: {output_dir / 'scaler_config.json'}")
    
    # Save label encoder
    label_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✓ Label mapping saved to: {output_dir / 'label_mapping.json'}")
    
    # Save training report
    report = {
        "date": datetime.now().isoformat(),
        "epochs": args.epochs,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "n_folds": N_FOLDS,
        "total_samples": len(df),
        "total_windows": len(X),
        "num_classes": num_classes,
        "classes": list(label_encoder.classes_),
        "scaler": scaler_config,
        "methodology": "ICTH_16 Paper - 5-fold Cross-Validation"
    }
    
    if not args.skip_cv:
        report["cv_results"] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "fold_accuracies": [float(a) for a in accuracies]
        }
    
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Training report saved to: {output_dir / 'training_report.json'}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    if not args.skip_cv:
        print(f"""
RESULTS SUMMARY:
────────────────
Cross-Validation Accuracy: {np.mean(accuracies)*100:.1f}% (± {np.std(accuracies)*100:.1f}%)
Cross-Validation F1-Score: {np.mean(f1_scores)*100:.1f}% (± {np.std(f1_scores)*100:.1f}%)

COMPARISON WITH ICTH_16 PAPER:
──────────────────────────────
Paper reports 87.0% (± 1.2%) after fine-tuning on Garmin data.
Our training data accuracy: {np.mean(accuracies)*100:.1f}%

IMPORTANT NOTE (Lab-to-Life Gap):
─────────────────────────────────
Per ICTH_16 paper, when this model is applied to UNLABELED production 
data from a commercial device, accuracy drops to ~49%.

This is the documented "lab-to-life gap" and is EXPECTED behavior,
not a bug in the pipeline.

Citation: ICTH_16, Section 4.3: "Without any fine-tuning, the model 
performed poorly, achieving an accuracy of only 48.7%"
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

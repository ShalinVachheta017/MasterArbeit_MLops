# 11 — Stage Deep-Dive: Training, Evaluation, and Inference

> Part of [Opus Understanding Audit Pack](00_README.md) | Phase 2 — Technical Stage Deep-Dives
> **Commit:** `168c05bb` | **Audit Date:** 2026-02-22

---

## 1. Model Architecture — 1D-CNN-BiLSTM

### 1.1 Architecture Definition

**FACT:** Built by `HARModelBuilder.create_1dcnn_bilstm()` in `src/train.py`.
[CODE: src/train.py | class:HARModelBuilder | method:create_1dcnn_bilstm]

```
Input: (batch, 200, 6)  — 200 timesteps × 6 IMU channels

CNN Block 1:
    Conv1D(64, k=3, relu, padding=same) → BatchNorm → 
    Conv1D(64, k=3, relu, padding=same) → BatchNorm → MaxPool1D(2) → Dropout(0.25)
    Output: (batch, 100, 64)

CNN Block 2:
    Conv1D(128, k=3, relu, padding=same) → BatchNorm →
    Conv1D(128, k=3, relu, padding=same) → BatchNorm → MaxPool1D(2) → Dropout(0.25)
    Output: (batch, 50, 128)

BiLSTM Block:
    Bidirectional(LSTM(64, return_sequences=True)) → BatchNorm → Dropout(0.3)
    Bidirectional(LSTM(64, return_sequences=False)) → BatchNorm → Dropout(0.5)
    Output: (batch, 128)

Classifier:
    Dense(128, relu) → BatchNorm → Dropout(0.5)
    Dense(11, softmax)
    Output: (batch, 11)
```

**FACT:** ~1.5M parameters. 6 BatchNormalization layers (critical for AdaBN/TENT adaptation).

### 1.2 Training Configuration

| Parameter | Default | Source |
|-----------|---------|--------|
| `window_size` | 200 | 4 seconds at 50Hz (ICTH_16) |
| `step_size` | 100 | 50% overlap |
| `n_sensors` | 6 | Ax, Ay, Az, Gx, Gy, Gz |
| `n_classes` | 11 | Activity classes |
| `epochs` | 100 | |
| `batch_size` | 64 | |
| `learning_rate` | 0.001 | Adam optimizer |
| `early_stopping_patience` | 15 | Monitor `val_accuracy` |
| `reduce_lr_patience` | 5 | Factor 0.5, min_lr 1e-6 |
| `n_folds` | 5 | Stratified K-Fold |
| `cv_random_seed` | 42 | Reproducibility |
| `dropout_cnn` | 0.25 | |
| `dropout_lstm` | 0.3 | |
| `dropout_dense` | 0.5 | |

**FACT:** [CODE: src/train.py | class:TrainingConfig]

### 1.3 Loss and Metrics

- **Loss:** `sparse_categorical_crossentropy`
- **Metrics tracked:** `accuracy`, `f1_macro`, `f1_weighted`, `cohen_kappa`, per-class F1
- **MLflow:** All metrics logged per fold and for final model

---

## 2. Training Pipeline (`HARTrainer`)

### 2.1 5-Fold Stratified Cross-Validation

**FACT:** [CODE: src/train.py | class:HARTrainer | method:run_cross_validation]

```python
# Per ICTH_16: "5-fold cross-validation protocol"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Standardize: fit on train, transform both
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 6))
    X_train_scaled = scaler.transform(X_train.reshape(-1, 6)).reshape(X_train.shape)
    X_val_scaled   = scaler.transform(X_val.reshape(-1, 6)).reshape(X_val.shape)
    
    # Fresh model per fold
    model = create_1dcnn_bilstm()
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), ...)
    
    # Nested MLflow run per fold
    tracker.log_metrics({val_accuracy, f1_macro, f1_weighted, cohen_kappa})
```

**INFERENCE:** Each fold gets a fresh model and a fresh scaler fitted only on training data — this is correct methodology that avoids data leakage.

### 2.2 Final Model Training

After CV, trains on **100% of data** with a 10% hold-out for monitoring training progress:
```python
X_train, X_val = train_test_split(X_scaled, y, test_size=0.1, stratify=y)
model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
```

**Artifacts saved:**
- `har_model_{timestamp}.keras` + `har_model_latest.keras`
- `scaler_config.json` (mean, scale arrays)
- `label_mapping.json` (index → activity name)
- `training_report_{timestamp}.json` (full config, metrics, history)
- All logged to MLflow

**FACT:** [CODE: src/train.py | method:train_final_model, _save_training_artifacts]

---

## 3. Domain Adaptation Training (`DomainAdaptationTrainer`)

Extends `HARTrainer` with domain adaptation methods for retraining scenarios.

**FACT:** [CODE: src/train.py | class:DomainAdaptationTrainer]

| Method Arg | Implementation | Actual Behavior |
|------------|---------------|-----------------|
| `pseudo_label` | `_retrain_pseudo_labeling` | Full calibrated pseudo-labeling (see doc 13) |
| `mmd` | `_retrain_mmd` | **Redirects to pseudo_label** (placeholder) |
| `dann` | `_retrain_dann` | **Redirects to pseudo_label** (placeholder) |

**RISK:** MMD and DANN are listed as adaptation options but silently fall back to pseudo-labeling. The thesis must not claim these are implemented.
[CODE: src/train.py | methods:_retrain_mmd, _retrain_dann]

---

## 4. Stage 4 — Model Inference (Component 4)

### 4.1 Batch Inference

**FACT:** Implemented in `src/components/model_inference.py`.
[CODE: src/components/model_inference.py]

**Flow:**
1. Load model from `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`
2. Load windowed data from `production_X.npy` (output of Stage 3)
3. `model.predict(X, batch_size=64)` → probabilities `(N, 11)`
4. `argmax(probs)` → predicted classes
5. Compute: activity distribution, confidence statistics, inference time

### 4.2 Output Artifact

```
ModelInferenceArtifact:
    predictions_csv_path: Path      # window, activity, activity_id, confidence
    predictions_npy_path: Path      # class indices array
    probabilities_npy_path: Path    # full probability matrix (N, 11)
    n_predictions: int
    inference_time_seconds: float
    activity_distribution: Dict[str, int]
    confidence_stats: Dict          # mean, std, min, max, median, n_uncertain
    model_version: str
```

---

## 5. Stage 5 — Model Evaluation (Component 5)

### 5.1 Two Evaluation Modes

**FACT:** Implemented in `src/components/model_evaluation.py` (~66 lines).
[CODE: src/components/model_evaluation.py]

| Mode | Condition | Metrics Computed |
|------|-----------|-----------------|
| **Labeled** | Ground-truth labels available | Accuracy, F1 (macro/weighted), Cohen's Kappa, per-class F1, confusion matrix |
| **Unlabeled** | No labels (production mode) | Distribution summary, confidence summary only |

### 5.2 Unlabeled Evaluation (Production)

When no labels are available (typical production scenario):
- **Distribution summary:** Number of distinct activities detected, dominant activity percentage, class balance
- **Confidence summary:** Mean, median, std, min, max confidence

**INFERENCE:** The evaluation component is lightweight by design — deep analysis (ECE, calibration) is in `src/calibration.py` but not orchestrated here.

---

## 6. Activity Classes (11-Class HAR)

| ID | Activity | Domain |
|----|----------|--------|
| 0 | ear_rubbing | Anxiety-related |
| 1 | forehead_rubbing | Anxiety-related |
| 2 | hair_pulling | Anxiety-related |
| 3 | hand_scratching | Anxiety-related |
| 4 | hand_tapping | Anxiety-related |
| 5 | knuckles_cracking | Anxiety-related |
| 6 | nail_biting | Anxiety-related |
| 7 | nape_rubbing | Anxiety-related |
| 8 | sitting | Baseline |
| 9 | smoking | Ambiguous |
| 10 | standing | Baseline |

**FACT:** [CODE: src/api/app.py | ACTIVITY_CLASSES dict]
**INFERENCE:** This is an anxiety-detection HAR task — 8 of 11 classes are anxiety-related behaviors, plus 2 baseline activities and 1 ambiguous (smoking).

---

## 7. MLflow Integration

**FACT:** All training/evaluation metrics flow through `src/mlflow_tracking.py` — class `MLflowTracker`.

| Stage | What's Logged |
|-------|--------------|
| **CV Training** | Nested runs per fold: val_accuracy, f1_macro, f1_weighted, cohen_kappa |
| **Final Training** | Final metrics + model artifacts + scaler/label mapping |
| **Pipeline** | Per-stage metrics (see `_log_stage_to_mlflow` in production_pipeline.py) |
| **Model Registry** | Model registered as `har-1dcnn-bilstm` in MLflow Model Registry |

**FACT:** [CODE: src/pipeline/production_pipeline.py | method:_log_stage_to_mlflow]

---

## 8. Critical Findings

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| TR-1 | Correct CV methodology — no data leakage | **STRENGTH** | Fresh scaler per fold, fresh model per fold |
| TR-2 | DANN/MMD silently redirect to pseudo-labeling | **HIGH** | [CODE: src/train.py:_retrain_dann, _retrain_mmd] |
| TR-3 | Evaluation component is lightweight (no ECE) | **MEDIUM** | Calibration exists but not integrated |
| TR-4 | MLflow tracking comprehensive (per-fold, per-stage) | **STRENGTH** | [CODE: src/train.py, src/pipeline/production_pipeline.py] |
| TR-5 | 11-class anxiety HAR is a narrow but well-defined domain | **INFO** | Domain-specific label set |

---

## 9. Recommendations for Thesis

1. **Architecture figure**: Create a detailed 1D-CNN-BiLSTM diagram showing layer shapes and parameter counts
2. **CV results table**: Standard thesis Table showing per-fold accuracy, F1, Kappa with mean±std
3. **DANN/MMD**: Either implement or remove — do not list as capabilities
4. **Integrate ECE**: Add calibration evaluation to Stage 5 for thesis completeness
5. **Confusion matrix**: Generate and include as thesis figure (standard for HAR papers)

# Big Questions for Thesis MLOps Pipeline
**Date:** January 18, 2026

---

## How to Use This File

Each section contains **one focused question** with **ANSWERS PROVIDED** based on repo analysis and research papers.

**Context:**
- `decoded_csv_files/` = multiple unlabeled production datasets (different recording sessions)
- `data/raw/` = main production dataset already processed
- Thesis requires research paper citations for all decisions

---

# ANSWERED QUESTIONS

---

## Section 1: Dataset Management

### Q1.1: Do we need to label ALL decoded CSV datasets?

**Background:** We have many datasets in `decoded_csv_files/`. Data distribution may vary based on:
- Dominant hand vs non-dominant hand wearing the watch
- Activity performed with same or different hand

**Core Question:** Must we label every dataset, or can we use unlabeled data strategically?

---

### ✅ ANSWER Q1.1:

**Decision:** **NO, you do NOT need to label ALL datasets.** Use a strategic approach:

**Tier 1 - Label Small Audit Set (REQUIRED):**
- Label **3-5 complete sessions** (~200-500 windows total)
- Use stratified selection: cover all activity classes, both dominant/non-dominant
- Purpose: Validate model accuracy, enable per-class F1, detect drift

**Tier 2 - Keep Rest Unlabeled (STRATEGIC):**
- Use unlabeled data for:
  - **Drift detection** (compare distributions to training baseline)
  - **Uncertainty monitoring** (entropy, margin, confidence)
  - **Active learning** (select most uncertain samples for labeling)

**Dominant Hand Variation Handling:**
From [docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md](docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md):
- ~70% of people wear watches on left wrist (typically non-dominant)
- Most anxiety behaviors performed with dominant hand
- **Signal attenuation** expected when watch on non-dominant wrist
- Store metadata: `dominant_hand`, `watch_wrist`, `dominance_match`

**How to Detect Distribution Differences Without Labels:**
1. **Statistical tests**: KS-test, Wasserstein distance between training baseline and production
2. **Confidence drop**: Mean confidence dropping >0.15 indicates domain shift
3. **Embedding distance**: Mahalanobis distance from training embeddings

**CITATIONS:**
1. (docs/output_1801_2026-01-18.md, lines 69-143, Pair 01): "Label 3-5 complete sessions (~200-500 windows total) using stratified selection rules. This minimal audit set enables per-class F1 evaluation, drift detection, and retraining trigger decisions."
2. (docs/output_1801_2026-01-18.md, lines 150-352, Pair 02): "Non-dominant wrist has lower signal-to-noise ratio. Use adaptive thresholds: Normal confidence 0.50 → Relaxed (non-dominant): 0.35"
3. (papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring.pdf, p.9-11): "Two-sample tests enable distribution shift detection on unlabeled production data"
4. (docs/research/KEEP_Research_QA_From_Papers.md, lines 140-196): "Detecting drift without ground truth relies on KS test, MMD, or Jensen-Shannon divergence between training and production distributions"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 16 — Labeling strategy for multiple production datasets

Q1: Do we need to label ALL production datasets in decoded_csv_files/, or can we use unlabeled data? Consider: dominant hand variations, data distribution differences, active learning strategies. Cite 2-4 papers.

Q2: How to detect and handle distribution differences between dominant hand and non-dominant hand data without labels? Cite 2-4 papers.
```

---

### Q1.2: DVC Tracking - Raw, Processed, AND Prepared?

**Background:** Currently have `raw.dvc`, `processed.dvc`, `prepared.dvc` files.

**Core Question:** Do we need to track all three stages in DVC, or is tracking only certain stages sufficient?

---

### ✅ ANSWER Q1.2:

**Decision:** **YES, track ALL THREE stages** (raw, processed, prepared). This is best practice for full reproducibility.

**What Each Stage Contains:**

| Stage | File | Content | Size | Purpose |
|-------|------|---------|------|---------|
| `raw.dvc` | `data/raw/` | Original Excel files from Garmin | ~64MB | Immutable source, never modify |
| `processed.dvc` | `data/processed/` | `sensor_fused_50Hz.csv` | ~15MB | Resampled, time-aligned sensor data |
| `prepared.dvc` | `data/prepared/` | `.npy` arrays + `config.json` | ~10MB | ML-ready windowed data with scaler params |

**Why Track All Three:**

1. **Full reproducibility**: Can recreate any pipeline stage from any version
2. **Debugging**: If inference fails, trace back through stages
3. **Experimentation**: Test different preprocessing (gravity removal ON/OFF) without re-fusion
4. **Storage efficient**: DVC deduplicates unchanged files

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| **Track all 3** ✅ | Full reproducibility, easy debugging | More storage (~90MB total) |
| Track only prepared | Less storage | Can't reproduce intermediate steps |
| Track only raw | Minimal storage | Must rerun entire pipeline |

**CITATIONS:**
1. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 429-443): "DVC tracks raw.dvc (64MB), processed.dvc, prepared.dvc. Evidence: .dvc files at data/raw.dvc"
2. (docs/research/KEEP_Research_QA_From_Papers.md, lines 101-103): "DVC stores actual data in remote storage while keeping lightweight pointer files in Git. This versions data alongside code."
3. (README.md, lines 86-154): "Data Versioning: DVC ✅ Complete"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 17 — DVC tracking strategy

Q1: Should we track raw, processed, AND prepared data in DVC? What are trade-offs (storage, reproducibility, versioning)? Cite 2-3 papers on data versioning best practices.

Q2: What's the recommended DVC pipeline structure for HAR projects with sensor data? Cite 2-3 papers.
```

---

### Q1.3: Combine Sensor Fusion + Preprocessing into One Script?

**Background:** Currently have separate scripts for sensor fusion (merging accelerometer + gyroscope) and preprocessing.

**Core Question:** Can/should we combine them into one script?

---

### ✅ ANSWER Q1.3:

**Decision:** **KEEP THEM SEPARATE** (recommended) but can combine for simplicity.

**Current Structure:**
- `src/sensor_data_pipeline.py` → Sensor fusion (accel + gyro → fused CSV)
- `src/preprocess_data.py` → Preprocessing (fused CSV → .npy arrays)

**Why Keep Separate (Recommended):**

| Benefit | Explanation |
|---------|-------------|
| **Modularity** | Can modify preprocessing without re-fusing |
| **Experimentation** | Test different gravity removal settings quickly |
| **Debugging** | Isolate issues to specific stage |
| **Reusability** | Fusion works for any sensor pair |

**When to Combine:**
- Simple deployment with no experimentation needed
- One-shot pipeline (run once, never modify)

**If Combining, Structure Would Be:**
```python
# src/full_pipeline.py
class FullPipeline:
    def run(self, accel_path, gyro_path, output_path):
        # Step 1: Fusion
        fused_df = self.fuse_sensors(accel_path, gyro_path)
        
        # Step 2: Preprocess
        X, config = self.preprocess(fused_df)
        
        # Step 3: Save
        np.save(output_path, X)
```

**CITATIONS:**
1. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 37-43): "Pipeline: Fusion → Preprocessing → Inference. Each stage has dedicated script."
2. (docs/output_1801_2026-01-18.md, lines 1141): "Testing different enable_gravity_removal settings doesn't require re-fusion"
3. (src/README.md, lines 63): "Pipeline order: sensor_data_pipeline.py → preprocess_data.py → run_inference.py"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 18 — Script consolidation

Q1: Can we combine sensor fusion (merging accel + gyro on nearest time grid) and preprocessing into one script? What are trade-offs (modularity vs simplicity)? Cite 2-3 papers.

Q2: Recommend script structure for our pipeline. Should sensor fusion be separate or combined with preprocessing?
```

---

### Q1.4: File Naming Convention for Merged Datasets

**Background:** After merging accelerometer + gyroscope, we want filenames that show recording date without opening the file.

**Core Question:** Best naming convention for merged sensor data files?

---

### ✅ ANSWER Q1.4:

**Decision:** Use format: `{YYYY-MM-DD-HH-MM-SS}_sensor_fused.csv`

**Recommended Naming Convention:**

| Component | Example | Purpose |
|-----------|---------|---------|
| Date-Time | `2025-07-16-21-03-13` | Recording timestamp (from filename) |
| Type | `sensor_fused` | Indicates merged accel+gyro | 
| Extension | `.csv` | File format |

**Full Example:**
```
decoded_csv_files/
├── 2025-07-16-21-03-13_accelerometer.csv  (raw)
├── 2025-07-16-21-03-13_gyroscope.csv      (raw)
└── 2025-07-16-21-03-13_sensor_fused.csv   (merged) ← NEW
```

**Implementation (add to sensor_data_pipeline.py):**
```python
def get_output_filename(accel_path: Path) -> str:
    """Extract date from accelerometer filename and create fused filename."""
    # Input: 2025-07-16-21-03-13_accelerometer.csv
    # Output: 2025-07-16-21-03-13_sensor_fused.csv
    
    stem = accel_path.stem  # "2025-07-16-21-03-13_accelerometer"
    date_part = stem.replace("_accelerometer", "")  # "2025-07-16-21-03-13"
    return f"{date_part}_sensor_fused.csv"
```

**Alternative Formats:**
| Format | Example | Pro | Con |
|--------|---------|-----|-----|
| **ISO timestamp** ✅ | `2025-07-16-21-03-13_sensor_fused.csv` | Sortable, readable | Verbose |
| Compact | `20250716_210313_fused.csv` | Shorter | Less readable |
| With user | `user01_2025-07-16_fused.csv` | User context | Requires user info |

**CITATIONS:**
1. (decoded_csv_files/): Current files already use `YYYY-MM-DD-HH-MM-SS_sensor.csv` format
2. (docs/output_1801_2026-01-18.md, lines 200-310, Pair 02): "Store session metadata including recording timestamp"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 19 — File naming conventions

Q1: Recommend naming convention for merged sensor datasets that includes recording date. Example: 2025-07-16-21-03-13_merged.csv or activity_YYYYMMDD_HHMMSS.csv? What metadata should be in filename vs inside file?

Q2: How to implement automatic renaming during sensor fusion that extracts date from original files?
```

---

## Section 2: Pipeline Components Explanation

### Q2.1: What is Inference? What does the script do?

**Background:** Have `src/run_inference.py` but need clear explanation.

**Core Question:** What is inference in ML pipeline context? What does our script do?

---

### ✅ ANSWER Q2.1:

**Definition:** **Inference** is applying a trained model to new (unlabeled) data to generate predictions. It's the "production use" of the model.

**What `src/run_inference.py` Does (Step-by-Step):**

| Step | Action | Code Location |
|------|--------|---------------|
| 1 | **Load Model** | `model = tf.keras.models.load_model('model.keras')` |
| 2 | **Load Data** | `data = np.load('production_X.npy')` → Shape: (n_windows, 200, 6) |
| 3 | **Run Prediction** | `probabilities = model.predict(data)` → Shape: (n_windows, 11) |
| 4 | **Extract Results** | `predictions = np.argmax(probabilities, axis=1)` |
| 5 | **Compute Confidence** | `confidences = np.max(probabilities, axis=1)` |
| 6 | **Export CSV** | `predictions.csv` with columns: window_id, predicted_class, confidence |

**Inputs:**
- `data/prepared/production_X.npy` - Preprocessed sensor windows (n_windows, 200, 6)
- `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` - Trained model

**Outputs:**
```
data/prepared/predictions/
├── predictions_20260118_143100.csv      # Human-readable results
├── predictions_20260118_143100_probs.npy # Full probability matrix
└── predictions_20260118_143100_metadata.json # Run metadata
```

**CSV Output Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `window_id` | Unique identifier | 42 |
| `predicted_class` | Class index (0-10) | 3 |
| `predicted_activity` | Activity name | "sitting" |
| `confidence` | Max softmax prob | 0.84 |
| `is_uncertain` | confidence < 0.50 | False |
| `prob_class_0`...`prob_class_10` | All class probs | 0.02, 0.84, 0.01... |

**Confidence Levels:**
| Level | Range | Action |
|-------|-------|--------|
| HIGH | >90% | Trust prediction |
| MODERATE | 70-90% | Likely correct |
| LOW | 50-70% | Review manually |
| UNCERTAIN | <50% | Flag for review |

**CITATIONS:**
1. (src/run_inference.py, lines 80-100): InferenceConfig defines model_path, input_path, confidence_threshold=0.50
2. (src/run_inference.py, lines 429-450): `predict_batch()` computes `predictions = np.argmax(probabilities, axis=1)`, `confidences = np.max(probabilities, axis=1)`
3. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 233-250): "Confidence is max softmax probability, computed in run_inference.py line 436"
4. (src/README.md, lines 400-460): Full documentation of run_inference.py

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 20 — Inference explained

Q1: What is "inference" in ML pipeline? Explain what src/run_inference.py does step-by-step. What inputs, outputs, and metrics does it produce?

Q2: What should be logged during inference for thesis documentation? (confidence, entropy, predictions, timestamps, etc.)
```

---

### Q2.2: What is Evaluation? What does the script do?

**Background:** Have `src/evaluate_predictions.py` but need clear explanation.

**Core Question:** What is evaluation vs inference? What does evaluation script compute?

---

### ✅ ANSWER Q2.2:

**Definition:** **Evaluation** compares predictions against ground truth labels to measure model performance. It answers: "How good is the model?"

**Inference vs Evaluation:**

| Aspect | Inference | Evaluation |
|--------|-----------|------------|
| **Purpose** | Generate predictions | Measure accuracy |
| **Labels needed?** | ❌ NO | ✅ YES (for full eval) |
| **Output** | Predictions CSV | Metrics (accuracy, F1, confusion matrix) |
| **When used** | Production (always) | Validation (when labels available) |

**What `src/evaluate_predictions.py` Computes:**

**With Labels (Full Evaluation):**
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | correct / total | Overall correctness |
| **Precision** | TP / (TP+FP) | How many predicted positives are correct |
| **Recall** | TP / (TP+FN) | How many actual positives were found |
| **F1 Score** | 2 × (P×R)/(P+R) | Balance of precision/recall |
| **Confusion Matrix** | NxN grid | Per-class errors |
| **ECE** | Expected Calibration Error | Confidence reliability |

**Without Labels (Proxy Evaluation):**
| Metric | Purpose |
|--------|---------|
| **Mean Confidence** | Model certainty |
| **Entropy** | Overall uncertainty |
| **Margin** | Top1 - Top2 ambiguity |
| **Prediction Distribution** | Class balance |

**Two Modes in Script:**
```python
# Mode 1: WITH labels (full metrics)
python evaluate_predictions.py --predictions preds.csv --labels test_y.npy

# Mode 2: WITHOUT labels (proxy metrics only)
python evaluate_predictions.py --predictions preds.csv
```

**Key Classes:**
- `ClassificationEvaluator` - Computes accuracy, F1, confusion matrix (needs labels)
- `PredictionAnalyzer` - Computes confidence stats, entropy, distribution (no labels)

**CITATIONS:**
1. (src/evaluate_predictions.py, lines 424-550): `ClassificationEvaluator.compute_metrics()` uses sklearn's accuracy_score, precision_recall_fscore_support, confusion_matrix
2. (docs/output_1801_2026-01-18.md, lines 1776-1850): "Two evaluation modes: ground-truth (labeled) and proxy (unlabeled)"
3. (src/README.md, lines 470-520): "evaluate_predictions.py: ClassificationEvaluator for labeled, PredictionAnalyzer for unlabeled"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 21 — Evaluation explained

Q1: What is "evaluation" in ML pipeline? How is it different from inference? Explain what src/evaluate_predictions.py computes (accuracy, F1, confusion matrix, ECE, etc.).

Q2: When do we run evaluation vs inference? Can we evaluate without labels?
```

---

## Section 3: MLflow Integration

### Q3.1: Log Confidence and Metrics to MLflow

**Background:** Want to log confidence, entropy, and other uncertainty metrics to MLflow.

**Core Question:** How to log these metrics to MLflow during inference?

---

### ✅ ANSWER Q3.1:

**Decision:** **YES, log everything to MLflow.** Already partially implemented.

**What Gets Logged (Current):**
From `src/run_inference.py` lines 702-760:
```python
with mlflow.start_run(run_name=f"inference_{timestamp}"):
    # Parameters
    mlflow.log_params({
        "mode": "batch",
        "batch_size": 32,
        "confidence_threshold": 0.50,
        "model_path": "models/pretrained/model.keras"
    })
    
    # Metrics
    mlflow.log_metrics({
        "total_windows": len(results_df),
        "uncertain_count": results_df['is_uncertain'].sum(),
        "avg_confidence": results_df['confidence'].mean(),
        "std_confidence": results_df['confidence'].std(),
    })
    
    # Activity distribution
    for activity, count in activity_dist.items():
        mlflow.log_metric(f"count_{activity}", count)
```

**What to ADD (Recommended):**
```python
# Add entropy and margin
entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
sorted_probs = np.sort(probs, axis=1)
margin = sorted_probs[:, -1] - sorted_probs[:, -2]

mlflow.log_metrics({
    "mean_entropy": float(np.mean(entropy)),
    "max_entropy": float(np.max(entropy)),
    "mean_margin": float(np.mean(margin)),
    "min_margin": float(np.min(margin)),
    "high_entropy_ratio": float(np.mean(entropy > 2.0)),
    "low_margin_ratio": float(np.mean(margin < 0.10)),
})

# Log artifacts
mlflow.log_artifact("predictions.csv")
mlflow.log_figure(confidence_histogram, "confidence_distribution.png")
```

**MLflow Artifacts to Store:**

| Artifact | Format | Purpose |
|----------|--------|---------|
| `predictions.csv` | CSV | Per-window predictions |
| `confidence_distribution.png` | PNG | Visual inspection |
| `entropy_distribution.png` | PNG | Uncertainty analysis |
| `confusion_matrix.png` | PNG | If labels available |
| `drift_report.json` | JSON | Layer 3 drift results |

**View Results:**
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

**CITATIONS:**
1. (src/run_inference.py, lines 702-760): MLflow logging already implemented for basic metrics
2. (docs/output_1801_2026-01-18.md, lines 2674-2736, Pair 09): "Log confidence/entropy as metrics, histograms as artifacts"
3. (scripts/post_inference_monitoring.py, lines 1349-1500): Extended MLflow logging with drift metrics

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 22 — MLflow logging for uncertainty metrics

Q1: How to log confidence, entropy, margin, and other uncertainty metrics to MLflow during inference? Show code integration points.

Q2: What MLflow artifacts should we store for thesis (histograms, CSVs, model versions)? Best practices for experiment tracking. Cite 2-3 papers.
```

---

## Section 4: Uncertainty & OOD Methods

### Q4.1: Which Uncertainty Methods for Thesis?

**Background:** Options include entropy, margin, temperature scaling, energy score, embedding distance, deep ensembles.

**Core Question:** Which methods are appropriate for thesis even if expensive/advanced?

---

### ✅ ANSWER Q4.1:

**Decision:** Implement **3-Tier Framework** from simple to advanced:

### Tier 1: Entropy + Margin (Already Implemented ✅)

**Entropy:** H(p) = -Σ pᵢ log(pᵢ)
- Measures overall uncertainty
- H=0: Certain (one class dominates)
- H=log(k): Maximum uncertainty (uniform)
- **Threshold:** H > 2.0 = high uncertainty

**Margin:** p_top1 - p_top2
- Measures ambiguity between top classes
- Margin > 0.5: Clear decision
- Margin < 0.1: Very ambiguous
- **Threshold:** Margin < 0.10 = flag for review

**Location:** scripts/post_inference_monitoring.py lines 339-370

---

### Tier 2: Temperature Scaling (Recommended for Thesis ✅)

**Purpose:** Make confidence scores match actual accuracy (calibration)

**Formula:** p'ᵢ = exp(zᵢ/T) / Σ exp(zⱼ/T)

**How It Works:**
1. Fit single parameter T on validation set (one-time)
2. T > 1 reduces overconfidence
3. Apply at inference: divide logits by T before softmax

**Expected Improvement:**
- ECE: 0.12 → 0.05 (better calibration)

**Implementation:** ~1-2 days work, add to src/evaluate_predictions.py

---

### Tier 3: Mahalanobis Distance (Optional, Advanced)

**Purpose:** Detect out-of-distribution (OOD) samples in embedding space

**Formula:** D_M = √[(x-μ)ᵀ Σ⁻¹ (x-μ)]

**Interpretation:**
- D_M < 2: In-distribution
- D_M > 4: Out-of-distribution

**Location:** scripts/post_inference_monitoring.py lines 956-1050 (EmbeddingDriftDetector)

---

### NOT Recommended for Thesis:

| Method | Why Skip |
|--------|----------|
| **Energy Score** | Requires logits not saved by current pipeline |
| **Deep Ensembles** | 5x training cost, 5x inference time |
| **Conformal Prediction** | Too complex for thesis scope |
| **MC Dropout** | Requires architecture changes, slow inference |

---

### Summary Table:

| Method | Implemented | Effort | Benefit | Use For |
|--------|-------------|--------|---------|---------|
| **Entropy** ✅ | Yes | 0 | High | Multi-way confusion |
| **Margin** ✅ | Yes | 0 | High | Binary ambiguity |
| **Temperature Scaling** | No | 1-2 days | Medium | Calibration |
| **Mahalanobis** ✅ | Yes | 0 | Medium | OOD detection |

**CITATIONS:**
1. (docs/output_1801_2026-01-18.md, lines 7977-8200, Pair 15): Full 3-tier uncertainty framework with formulas and thresholds
2. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 963-1000): "Better alternatives: Entropy, Margin, Temperature Scaling"
3. (new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf): Energy score for OOD
4. (scripts/post_inference_monitoring.py, lines 339-370): Entropy and margin computation
5. (scripts/post_inference_monitoring.py, lines 956-1050): EmbeddingDriftDetector with Mahalanobis distance

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 23 — Uncertainty methods selection

Q1: Compare uncertainty methods for HAR thesis: entropy, margin, temperature scaling/ECE, energy score, embedding distance (Mahalanobis), deep ensembles. Which should we implement? Cite 4-6 papers.

Q2: Implementation priority order - which are quick wins vs which require significant effort? Cost-benefit analysis.
```

---

## Section 5: Standard Scaler Questions

### Q5.1: Why Fit Scaler on Training Data Only?

**Background:** StandardScaler must be fit on training data only, but production data may have different distributions.

**Core Question:** Why training-only fit? What about production data variations?

---

### ✅ ANSWER Q5.1:

**Decision:** **ALWAYS fit scaler on training data only.** This is fundamental ML practice.

### Why Training-Only Fit?

**What StandardScaler Does:**
```
X_normalized = (X - mean) / std
```
where mean and std are computed during `fit()`.

**Three Failure Modes if You Fit on Production:**

| Failure | Explanation | Consequence |
|---------|-------------|-------------|
| **Data Leakage** | Test/production statistics contaminate training | Overly optimistic validation metrics |
| **Input Mismatch** | Model trained on different normalized values | Broken learned representations |
| **Temporal Contamination** | Future data influences past decisions | Violates causality in time-series |

### Example:

**Correct (Training-Only):**
```python
# During training
scaler = StandardScaler()
scaler.fit(X_train)  # Compute mean/std from training ONLY
X_train_scaled = scaler.transform(X_train)

# Save scaler parameters
config = {
    "scaler_mean": scaler.mean_.tolist(),  # [3.22, 1.28, -3.53, 0.60, 0.23, 0.09]
    "scaler_scale": scaler.scale_.tolist()  # [5.1, 4.2, 8.3, 2.1, 1.8, 0.9]
}

# During production
X_prod_scaled = scaler.transform(X_prod)  # Use SAME mean/std
```

**Wrong (Causes Leakage):**
```python
# NEVER DO THIS
scaler.fit(X_prod)  # Production statistics leak!
```

### What if Production Distribution Differs?

**This is Expected!** Production data WILL differ (new users, dominant hand variation, etc.)

**How to Handle:**
1. **Monitor drift**: Compare production stats to training baseline
2. **Detect when scaler inappropriate**: If KS-test p < 0.05 on >3 channels → investigate
3. **Do NOT refit scaler**: Instead, consider retraining entire model

**CITATIONS:**
1. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 342-370): "If you fit scaler on production data, you're using information from the future/test set. This leads to data leakage."
2. (docs/output_1801_2026-01-18.md, lines 3109-3200, Pair 10): Full explanation of StandardScaler leakage with HAR examples
3. (src/preprocess_data.py, lines 507-521): Scaler loading from config.json

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 24 — StandardScaler fitting strategy

Q1: Why must StandardScaler be fit on training data ONLY? What happens if we fit on production data? Cite 2-3 papers on data leakage.

Q2: If production data has different distribution (dominant hand variation), should we use same scaler or adapt? How to detect when scaler is no longer appropriate?
```

---

### Q5.2: Saved Scaler Parameters - Mean and Scale

**Background:** `models/active/scaler.pkl` contains mean and scale parameters.

**Core Question:** What are these parameters? What if fine-tuning changes them?

---

### ✅ ANSWER Q5.2:

**Where Scaler Parameters Are Stored:**
- **File:** `data/prepared/config.json`
- **NOT** `models/active/scaler.pkl` (your repo uses JSON, not pickle)

**Parameters Stored:**
```json
{
  "scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
  "scaler_scale": [5.1, 4.2, 8.3, 2.1, 1.8, 0.9]
}
```

| Parameter | Meaning | Channels |
|-----------|---------|----------|
| `scaler_mean` | Per-channel mean from training | [Ax, Ay, Az, Gx, Gy, Gz] |
| `scaler_scale` | Per-channel std from training | [Ax, Ay, Az, Gx, Gy, Gz] |

**How to Inspect:**
```python
import json
with open('data/prepared/config.json') as f:
    config = json.load(f)

print("Mean:", config['scaler_mean'])
print("Scale:", config['scaler_scale'])
```

### What if Fine-Tuning Changes Scaler?

**Scenario:** After collecting new data and fine-tuning, mean/scale may shift.

**Options:**

| Approach | When to Use | Risk |
|----------|-------------|------|
| **Keep original scaler** ✅ | Small distribution shift | Model expects original normalization |
| **Refit on combined data** | Large shift, full retraining | Creates new model version |
| **Adaptive normalization** | Research topic | Complex, not recommended for thesis |

**Best Practice:**
1. **Version scaler with model**: Scaler fitted on dataset X → Model trained on dataset X
2. **Check if refit needed**: Compare new training stats to old scaler
3. **If significant shift** (>20% change in mean): Retrain model with new scaler

**Versioning Strategy:**
```
models/
├── v1/
│   ├── model.keras
│   └── scaler_config.json  # Mean/scale for v1
├── v2/
│   ├── model.keras
│   └── scaler_config.json  # Mean/scale for v2 (different!)
└── active -> v2/  # Symlink to current version
```

**CITATIONS:**
1. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 68, 366-367): "config.json contains scaler_mean, scaler_scale from training"
2. (docs/output_1801_2026-01-18.md, lines 2095, 2203): "Scaler changed (different mean/std) → predictions will differ"
3. (src/preprocess_data.py, lines 513-521): Scaler loading: `self.scaler.mean_ = np.array(config['scaler_mean'])`

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 25 — Scaler versioning and fine-tuning

Q1: Explain StandardScaler's mean and scale parameters. Where are they saved? How to inspect them?

Q2: If we fine-tune model with new data, do we need new scaler? How to version scaler with model? What if mean/scale shift significantly? Cite 2-3 papers.
```

---

## Section 6: Gravity Removal

### Q6.1: Gravity in Accelerometer Data

**Background:** Need to understand gravity's effect on sensor data.

**Core Question:** Does gravity affect all axes (X,Y,Z) or just Z? Is removal necessary?

---

### ✅ ANSWER Q6.1:

### Does Gravity Affect All Axes or Just Z?

**Answer: ALL AXES (Ax, Ay, Az)**, but the distribution depends on device orientation.

**Physics Explanation:**
- Accelerometer measures **total acceleration** = body movement + gravity
- Gravity is always ~9.81 m/s² **downward** (Earth's reference frame)
- But accelerometer measures in **device reference frame**
- Device orientation determines how gravity projects onto each axis

**Example Orientations:**

| Device Position | Ax | Ay | Az | Gravity on |
|-----------------|----|----|-----|------------|
| Flat on table, screen up | 0 | 0 | -9.81 | Az only |
| Standing upright | 0 | -9.81 | 0 | Ay only |
| Tilted 45° | 0 | -6.94 | -6.94 | Ay + Az |
| Wrist horizontal | varies | varies | ~-9.81 | Mostly Az |

**From Your Data:**
```csv
# decoded_csv_files/2025-07-16-21-03-13_accelerometer.csv
accel_z = -1002.9570922851562  # milliG ≈ -9.83 m/s² (gravity!)
```

This confirms **gravity is present in Az** in your production data.

---

### Is Gravity Removal Necessary?

**Answer: IT DEPENDS on training data.**

**Key Insight:** Training data (from research paper) **already had gravity removed**. Production data **still contains gravity**. This mismatch causes problems.

| Dataset | Az Mean | Gravity | Status |
|---------|---------|---------|--------|
| Training (Paper) | -3.42 m/s² | REMOVED | High-pass filtered |
| Production (Your data) | -9.83 m/s² | PRESENT | Raw sensor |

**Recommendation:** **REMOVE GRAVITY** from production data to match training.

**How to Remove (High-Pass Filter):**
```python
# src/sensor_data_pipeline.py, GravityRemovalPreprocessor class
from scipy.signal import butter, filtfilt

def remove_gravity(acceleration_data, cutoff_hz=0.3, sampling_rate=50):
    """
    High-pass filter removes low-frequency components (gravity).
    
    - Gravity = DC component (0 Hz)
    - Human movement = higher frequencies (>0.3 Hz)
    - Butterworth high-pass filter removes gravity
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(3, normalized_cutoff, btype='high')
    
    return filtfilt(b, a, acceleration_data, axis=0)
```

**When to Skip Gravity Removal:**
- If training data also contains gravity (same preprocessing)
- If model was trained to use gravity as orientation signal

**CITATIONS:**
1. (docs/output_1801_2026-01-18.md, lines 5039-5200, Pair 12): Full gravity analysis with physics, UCI HAR reference
2. (src/sensor_data_pipeline.py, lines 720-850): GravityRemovalPreprocessor implementation
3. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 384-422): "Gravity removal: High-pass filter (0.3Hz) on accelerometer. Removes DC offset from orientation."
4. (notebooks/exploration/gravity_removal_demo.ipynb): "Training Az mean = -3.42 m/s² (REMOVED), Production Az mean = -9.83 (PRESENT)"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 26 — Gravity removal in accelerometer data

Q1: Does gravity affect all accelerometer axes (Ax, Ay, Az) or just Az? Explain with physics. Cite 2-3 papers.

Q2: Is gravity removal necessary for HAR? When to remove vs keep? What methods exist (high-pass filter, calibration)? Cite 2-3 papers.
```

---

## Section 7: Retraining Without Labels

### Q7.1: How to Measure Accuracy Without Labels?

**Background:** If data drift triggers retraining but we have no labels, how to know if model improved?

**Core Question:** How to measure/validate model accuracy without labeled production data?

---

### ✅ ANSWER Q7.1:

### The Hard Truth

**You CANNOT directly measure accuracy without labels.** Accuracy requires ground truth.

### But You CAN Use Proxy Metrics:

| Proxy Metric | What It Measures | Interpretation |
|--------------|------------------|----------------|
| **Mean Confidence** | Model certainty | Higher = more certain (but may be overconfident) |
| **Entropy** | Overall uncertainty | Lower = more decisive |
| **Margin** | Top1 - Top2 gap | Higher = clearer decisions |
| **Prediction Stability** | Temporal consistency | Same input → same output |
| **Drift Metrics** | Distribution shift | Lower = closer to training |

### Strategies for Validation Without Labels:

**Strategy 1: Label Small Audit Set (RECOMMENDED)**
- Label 50-200 windows (3-5 hours work)
- Use for validation accuracy
- Enables A/B testing: old model vs new model

**Strategy 2: Active Learning**
- Select most uncertain samples (high entropy, low margin)
- Label only those (~50 windows)
- Maximum information per labeled sample

**Strategy 3: Pseudo-Labeling with Filtering**
- Use old model's high-confidence predictions as pseudo-labels
- Filter: Only confidence > 0.80, margin > 0.30
- Risk: Confirmation bias (old model's errors propagate)

**Strategy 4: Agreement-Based Validation**
- Train two models independently
- Where they agree = likely correct
- Where they disagree = investigate

### Practical Workflow:

```
1. Detect drift (PSI > 0.25 on >2 channels)
         ↓
2. Sample 100 windows (50 random + 50 uncertain)
         ↓
3. Human labels 100 windows (~2 hours)
         ↓
4. Retrain model on mixed data (30% old + 70% new)
         ↓
5. Evaluate on labeled audit set
         ↓
6. If accuracy > 70% AND improvement > 5%: Deploy
   Else: Rollback
```

**CITATIONS:**
1. (docs/output_1801_2026-01-18.md, lines 6681-7948, Pair 14): Full retraining protocol without labels
2. (docs/research/KEEP_Research_QA_From_Papers.md, lines 145-196): "Model uncertainty/confidence can serve as proxy for OOD detection"
3. (papers/mlops_production/Essential_MLOps_Data_Science_Horizons_2023.pdf, p.97-104): "Proxy metrics (confidence, prediction drift) as substitutes for accuracy when ground truth unavailable"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 27 — Model validation without labels

Q1: If data drift triggers retraining but production data has NO labels, how do we measure accuracy? What proxy metrics can we use? Cite 3-4 papers.

Q2: Strategies: active learning (label small subset), pseudo-labeling, agreement-based validation. Which is best for thesis? Implementation steps. Cite 3-4 papers.
```

---

## Section 8: Drift Types & Monitoring

### Q8.1: Five Drift Types - What to Monitor Without Labels?

**Background:** Drift types: covariate, prior, concept, prediction, confidence. Need to monitor without labels.

**Core Question:** Which drifts can we detect without labels? How?

---

### ✅ ANSWER Q8.1:

### The Five Drift Types Explained:

| Drift Type | What Changes | Example | Detectable Without Labels? |
|------------|--------------|---------|---------------------------|
| **Covariate Drift** | P(X) - input distribution | Sensor values shift (new device) | ✅ YES |
| **Prior Drift** | P(Y) - label distribution | More "sitting" in production | ⚠️ PARTIAL (via predictions) |
| **Concept Drift** | P(Y\|X) - relationship | Same sensor → different activity | ❌ NO (needs labels) |
| **Prediction Drift** | Model outputs change | Different predicted class distribution | ✅ YES |
| **Confidence Drift** | Confidence distribution | Mean confidence drops | ✅ YES |

### What You CAN Monitor Without Labels:

**Layer 1: Confidence Metrics**
| Metric | Formula | Alert Threshold |
|--------|---------|-----------------|
| Mean Confidence | avg(max(softmax)) | Drop > 0.15 |
| Mean Entropy | avg(-Σ p log p) | Increase > 0.50 |
| Mean Margin | avg(p₁ - p₂) | Drop > 0.15 |
| Uncertain Ratio | count(conf < 0.50) / total | > 20% |

**Layer 2: Temporal Consistency**
| Metric | What It Detects | Alert |
|--------|-----------------|-------|
| Flip Rate | Rapid prediction changes | > 30% windows flip |
| Dwell Time | Time in same activity | < 3 seconds average |
| Streak Length | Consecutive same prediction | < 5 windows average |

**Layer 3: Statistical Drift (Input Features)**
| Test | Purpose | Alert Threshold |
|------|---------|-----------------|
| **KS Test** | Distribution shape | p < 0.05 on >3/6 channels |
| **PSI** | Population Stability Index | PSI > 0.25 |
| **Wasserstein** | Distribution distance | > 0.30 |

**Layer 4: Embedding Drift**
| Metric | What It Detects | Alert |
|--------|-----------------|-------|
| Mean Mahalanobis | Distance from training | > 3.0 average |
| Cosine Similarity | Direction change | < 0.85 |

### What You CANNOT Detect Without Labels:

- **Concept drift**: When same input should map to different output
- **True accuracy**: Need labels to measure correctness
- **Per-class performance**: Need labels for confusion matrix

### Implementation in Repo:

```python
# scripts/post_inference_monitoring.py

# Layer 1: Confidence (lines 339-370)
confidence_analyzer = ConfidenceAnalyzer(config)
results = confidence_analyzer.analyze(predictions_df)

# Layer 3: Drift (lines 615-900)
drift_detector = DriftDetector(baseline_path)
drift_results = drift_detector.analyze(production_data)

# Layer 4: Embedding (lines 952-1050)
embedding_detector = EmbeddingDriftDetector(baseline_embeddings_path)
embedding_results = embedding_detector.analyze(model, production_data)
```

**CITATIONS:**
1. (docs/PIPELINE_DEEP_DIVE_opus.md, lines 606-672): Drift detection table with methods
2. (docs/output_1801_2026-01-18.md, lines 5843-6679, Pair 13): 4-layer monitoring framework
3. (scripts/post_inference_monitoring.py, lines 615-1050): Full drift detection implementation
4. (docs/research/KEEP_Research_QA_From_Papers.md, lines 140-196): "Detecting drift without labels relies on KS test, MMD, model confidence"

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 28 — Drift detection without labels

Q1: Explain 5 drift types (covariate, prior, concept, prediction, confidence). Which can be detected WITHOUT labels? Cite 3-4 papers.

Q2: Monitoring strategy for unlabeled data: confidence, entropy, prediction distribution, sensor quality, temporal consistency, OOD embedding distance. Implementation for each. Cite 3-4 papers.
```

---

## Section 9: Dominant Hand Pattern Detection

### Q9.1: Detecting Dominant Hand Patterns

**Background:** Data distribution varies based on watch worn on dominant vs non-dominant hand, and activity performed with same or different hand.

**Core Question:** Can we detect these patterns from unlabeled data?

---

### ✅ ANSWER Q9.1:

### Can We Detect Dominant vs Non-Dominant Patterns?

**Answer: YES, through signal characteristics.**

### Signal Differences:

| Characteristic | Dominant Wrist Match | Non-Dominant Wrist Mismatch |
|----------------|---------------------|----------------------------|
| **Signal Amplitude** | High (direct motion) | Low (secondary motion) |
| **Signal Variance** | High (active movement) | Low (passive movement) |
| **Peak Frequency** | Activity-specific | Damped/attenuated |
| **SNR** | High | Low |

### What Happens in Each Scenario:

| Watch Wrist | Activity Hand | Signal Type | Model Performance |
|-------------|---------------|-------------|-------------------|
| **Left (non-dom)** | **Right (dom)** | Secondary motion only | ⚠️ Degraded (-10-15% F1) |
| **Right (dom)** | **Right (dom)** | Direct motion | ✅ Optimal |
| **Left** | **Left** | Direct motion | ✅ Good |
| **Right** | **Left** | Secondary motion | ⚠️ Degraded |

### Detection Methods (Without Labels):

**Method 1: Variance Analysis**
```python
# Non-dominant typically has lower variance
variance_per_session = df.groupby('session_id')[['Ax', 'Ay', 'Az']].var()

# If variance is significantly lower than training baseline → likely non-dominant
if variance_per_session.mean() < 0.6 * training_variance:
    flag = "Possible non-dominant wrist placement"
```

**Method 2: Confidence Drop**
```python
# Non-dominant sessions typically have lower confidence
mean_conf_per_session = predictions.groupby('session_id')['confidence'].mean()

# If confidence drops significantly → investigate wrist placement
if mean_conf_per_session < 0.60:
    flag = "Low confidence - check wrist placement"
```

**Method 3: Amplitude Analysis**
```python
# Peak-to-peak amplitude lower for non-dominant
amplitude = df[['Ax', 'Ay', 'Az']].max() - df[['Ax', 'Ay', 'Az']].min()

# Non-dominant shows ~60-80% of dominant amplitude
```

### Adaptation Strategies:

1. **Metadata tagging**: Record `dominant_hand`, `watch_wrist`, `dominance_match` per session
2. **Relaxed thresholds**: Use confidence threshold 0.35 (instead of 0.50) for non-dominant
3. **Signal augmentation**: During training, attenuate signals by 0.6-0.8x to simulate non-dominant
4. **Separate reporting**: Report F1 separately for dominant vs non-dominant

**CITATIONS:**
1. (docs/output_1801_2026-01-18.md, lines 150-352, Pair 02): "Non-dominant wrist has lower SNR. Use adaptive thresholds: 0.50 → 0.35"
2. (docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md): "~70% wear watch on left (non-dominant for right-handers). 80-95% of fine motor tasks performed by dominant hand."
3. (scripts/post_inference_monitoring.py, lines 92-200): MonitoringConfig with dominance_match parameter and relaxed thresholds

---

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 29 — Dominant hand pattern detection

Q1: Can we detect dominant hand vs non-dominant hand patterns from accelerometer/gyroscope data without labels? What signal characteristics differ? Cite 2-3 papers.

Q2: If user wears watch on left hand but does activities with right hand (and vice versa), how does this affect HAR model? Can we detect/adapt? Cite 2-3 papers.
```

---

## Section 10: FastAPI UI Design

### Q10.1: UI Design for FastAPI with Limitations

**Background:** Need simple UI for FastAPI inference endpoint with research backing.

**Core Question:** How to design minimal but effective UI?

---

### ✅ ANSWER Q10.1:

### FastAPI UI Design Recommendations

**Current Implementation:**
The API exists at `docker/api/main.py` with these endpoints:
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single window prediction
- `POST /predict/batch` - Batch prediction

### Response Fields Already Available:

| Field | Type | Description |
|-------|------|-------------|
| `activity` | string | Predicted activity name |
| `activity_id` | int | Class ID (0-10) |
| `confidence` | float | Max softmax probability |
| `probabilities` | dict | Per-class probabilities |
| `timestamp` | string | Prediction timestamp |
| `model_version` | string | Version identifier |

### 📊 Recommended UI Display Structure:

```
┌─────────────────────────────────────────────────────────────────┐
│ 🏃 HAR Prediction Dashboard                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📁 Upload: [Choose CSV file]  [Upload]                        │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  🎯 PREDICTION RESULT                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Activity: HAND_SCRATCHING                                 │  │
│  │ Confidence: 87.3%  [████████░░] HIGH                     │  │
│  │ Entropy: 0.23 (LOW = confident)                          │  │
│  │ Margin: 0.52 (HIGH = confident)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  📊 CLASS PROBABILITIES                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ hand_scratching  ████████████████████░░░░░  87.3%        │  │
│  │ nail_biting      ████░░░░░░░░░░░░░░░░░░░░   8.2%        │  │
│  │ hair_pulling     ██░░░░░░░░░░░░░░░░░░░░░░░  2.1%        │  │
│  │ ... (8 more)                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ⚠️ UNCERTAINTY INDICATORS                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Status:  ✅ HIGH CONFIDENCE - Trust this prediction      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  🔧 MODEL INFO: 1D-CNN-BiLSTM v1.0.0 | 11 classes              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Additional Response Fields:

| Field | Formula | Purpose |
|-------|---------|---------|
| `entropy` | $-\sum p_i \log p_i$ | Uncertainty measure (lower = confident) |
| `margin` | $p_{max} - p_{2nd}$ | Gap between top two classes |
| `confidence_band` | Based on threshold | "HIGH", "MODERATE", "LOW" |
| `top_3_classes` | Sorted probabilities | Show alternatives |

### Confidence Bands for UI:

| Confidence | Color | Label | Interpretation |
|------------|-------|-------|----------------|
| ≥ 0.80 | 🟢 Green | HIGH | Trust prediction |
| 0.50-0.79 | 🟡 Yellow | MODERATE | Review recommended |
| < 0.50 | 🔴 Red | LOW | Needs human review |

### Extended Response JSON:

```json
{
  "activity": "hand_scratching",
  "activity_id": 3,
  "confidence": 0.873,
  "entropy": 0.23,
  "margin": 0.52,
  "confidence_band": "HIGH",
  "top_3_activities": [
    {"name": "hand_scratching", "prob": 0.873},
    {"name": "nail_biting", "prob": 0.082},
    {"name": "hair_pulling", "prob": 0.021}
  ],
  "uncertainty_warning": null,
  "timestamp": "2026-01-18T14:30:00Z",
  "model_version": "1.0.0"
}
```

**CITATIONS:**
1. (docker/api/main.py, lines 1-150): FastAPI implementation with PredictionResponse schema
2. (scripts/post_inference_monitoring.py, lines 339-370): Entropy and margin calculation formulas
3. (docs/output_1801_2026-01-18.md, Pair 07): "UI should show confidence bars, entropy/margin values, and color-coded uncertainty levels"

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 30 — FastAPI UI design

Q1: Design minimal FastAPI UI for HAR inference: what fields needed (file upload, confidence threshold, visualization)? Cite 2-3 papers on ML system UI.

Q2: What should the response display? (predictions, confidence bars, uncertainty indicators, alerts). Example JSON response structure.
```

---

## Section 11: Improvements from Papers

### Q11.1: Data Augmentation and Other Improvements

**Background:** Looking for research-backed improvements to prediction confidence and class metrics.

**Core Question:** What improvements can we implement based on papers?

---

### ✅ ANSWER Q11.1:

### Research-Backed Improvements for HAR Pipeline

### 1️⃣ DATA AUGMENTATION TECHNIQUES

| Technique | Description | Expected Gain | Priority |
|-----------|-------------|---------------|----------|
| **Jittering** | Add Gaussian noise (σ=0.05) | +2-3% F1 | ⭐⭐⭐ |
| **Scaling** | Random scale [0.9, 1.1] | +1-2% F1 | ⭐⭐⭐ |
| **Rotation** | Random rotation [-10°, 10°] | +2-4% F1 | ⭐⭐ |
| **Time Warping** | Stretch/compress time segments | +1-3% F1 | ⭐⭐ |
| **Axis Mirroring** | Flip X/Y axes | +2-3% F1 for handedness | ⭐⭐⭐ |
| **Signal Attenuation** | Multiply by 0.6-0.8 | Simulates non-dominant wrist | ⭐⭐ |

### Implementation Example:

```python
# config/pipeline_config.yaml
preprocessing:
  augmentation:
    enabled: true
    jitter_sigma: 0.05
    scaling_range: [0.9, 1.1]
    rotation_range: [-10, 10]
    time_warp: true
    axis_mirror_prob: 0.3  # For handedness robustness
```

```python
# src/data_augmentation.py
import numpy as np

def augment_window(window, config):
    """Apply augmentation to a single window (200, 6)."""
    augmented = window.copy()
    
    # Jittering
    if config.get('jitter_sigma', 0) > 0:
        noise = np.random.normal(0, config['jitter_sigma'], window.shape)
        augmented += noise
    
    # Scaling
    if config.get('scaling_range'):
        scale = np.random.uniform(*config['scaling_range'])
        augmented *= scale
    
    # Axis mirroring (for handedness)
    if np.random.random() < config.get('axis_mirror_prob', 0):
        augmented[:, 0] *= -1  # Flip Ax
        augmented[:, 3] *= -1  # Flip Gx
    
    return augmented
```

### 2️⃣ MODEL ARCHITECTURE IMPROVEMENTS

| Improvement | Source | Implementation | Expected Gain |
|-------------|--------|----------------|---------------|
| **Self-Attention** | Zhang et al. (2022) | Add MultiHeadAttention after BiLSTM | +3-5% F1 |
| **Multi-Head CNN** | Multi-scale paper | Parallel kernels (3,5,7) | +2-3% F1 |
| **Residual Connections** | ResNet principles | Skip connections in CNN | +1-2% F1 |

### Self-Attention Implementation:

```python
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=64):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm = LayerNormalization()
        
    def call(self, x):
        attention_output = self.attention(x, x)
        return self.norm(x + attention_output)  # Residual connection

# Insert in model: BiLSTM → AttentionBlock → GlobalAveragePooling → Dense
```

### 3️⃣ CLASS BALANCING STRATEGIES

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **Class Weights** | Imbalanced training data | `class_weight={0: 2.0, 1: 1.0, ...}` |
| **SMOTE** | Severe imbalance (<10% minority) | `from imblearn.over_sampling import SMOTE` |
| **Focal Loss** | Many hard examples | $FL = -α(1-p)^γ \log(p)$ |
| **Stratified Sampling** | Production distribution differs | Sample proportionally |

### 4️⃣ TRANSFER LEARNING / FINE-TUNING

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Freeze base layers** | Keep CNN frozen, retrain LSTM+Dense | Limited new data (<1000 samples) |
| **Fine-tune all** | Unfreeze all with lower LR | Sufficient new data (>5000 samples) |
| **Domain adaptation** | Lab-to-life calibration | New sensor/device |

### 5️⃣ PRIORITY RECOMMENDATIONS

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| 🥇 **HIGH** | Jittering + Scaling augmentation | Low | +3-5% F1 |
| 🥇 **HIGH** | Class weights for imbalance | Low | +2-4% F1 |
| 🥈 **MEDIUM** | Self-Attention layer | Medium | +3-5% F1 |
| 🥈 **MEDIUM** | Axis mirroring for handedness | Low | +2-3% F1 |
| 🥉 **LOW** | Multi-Head CNN | High | +2-3% F1 |
| 🥉 **LOW** | Focal Loss | Medium | +1-2% F1 |

### Current Status in Pipeline:

| Feature | Implemented? | Location |
|---------|--------------|----------|
| Augmentation | ❌ Not yet | Recommended: `src/data_augmentation.py` |
| Class weights | ❌ Not yet | Add to training config |
| Self-Attention | ❌ Not yet | Model architecture |
| Transfer learning | ✅ Yes | Fine-tuning implemented |

**CITATIONS:**
1. (docs/research/RESEARCH_PAPER_INSIGHTS.md, lines 55-80): Self-Attention implementation with MultiHeadAttention
2. (docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md, line 404): "data augmentation including axis mirroring and signal attenuation to simulate non-dominant wrist"
3. (papers needs to read/Deep Learning in Human Activity Recognition with Wearable Sensors.pdf): "Data augmentation improves generalization across sensor placements"
4. (docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md, lines 1739-1746): "Implement Data Augmentation" and "Add Self-Attention Layer" in TODO list
5. (docs/PIPELINE_DEEP_DIVE_opus.md, line 288): Deep Ensembles comparison table

#### Prompt for Copilot:
```
Update docs/output_1801_2026-01-18.md (append; don't overwrite).
Add: ## Pair 31 — Research-backed improvements

Q1: What data augmentation techniques improve HAR prediction confidence? (rotation, jittering, scaling, time warping). Cite 4-5 papers.

Q2: Beyond augmentation, what other improvements from papers? (class balancing, ensemble methods, attention mechanisms, transfer learning). Cite 4-5 papers.
```

---

## Quick Reference: Question Summary

| # | Topic | Key Question |
|---|-------|--------------|
| 1.1 | Labeling | Label ALL datasets or use unlabeled strategically? |
| 1.2 | DVC | Track raw + processed + prepared? |
| 1.3 | Scripts | Combine sensor fusion + preprocessing? |
| 1.4 | Naming | File naming convention with dates? |
| 2.1 | Inference | What is inference? What does script do? |
| 2.2 | Evaluation | What is evaluation? What does script compute? |
| 3.1 | MLflow | Log confidence/entropy to MLflow? |
| 4.1 | Uncertainty | Which uncertainty methods for thesis? |
| 5.1 | Scaler | Why fit on training only? |
| 5.2 | Scaler | What if fine-tuning changes mean/scale? |
| 6.1 | Gravity | Affects all axes or just Z? Necessary? |
| 7.1 | Retraining | Measure accuracy without labels? |
| 8.1 | Drift | Which drifts detectable without labels? |
| 9.1 | Hand | Detect dominant hand patterns? |
| 10.1 | UI | FastAPI UI design with limitations? |
| 11.1 | Improve | Data augmentation and improvements? |

---

## Recommended Order

**Phase 1 - Understanding (start here):**
1. Q2.1 (Inference explained)
2. Q2.2 (Evaluation explained)
3. Q5.1 (Scaler basics)

**Phase 2 - Core Decisions:**
4. Q1.1 (Labeling strategy)
5. Q6.1 (Gravity removal)
6. Q4.1 (Uncertainty methods)

**Phase 3 - Implementation:**
7. Q3.1 (MLflow logging)
8. Q8.1 (Drift monitoring)
9. Q7.1 (Retraining without labels)

**Phase 4 - Polish:**
10. Q1.2-1.4 (DVC, scripts, naming)
11. Q9.1-11.1 (Hand detection, UI, improvements)

---

## Notes

- Each prompt is designed to add ~500-800 lines to output file
- All prompts request 2-6 citations from papers in repo
- Run prompts one at a time to avoid credit issues
- Current output file: `docs/output_1801_2026-01-18.md` (9193 lines after Pair 15)

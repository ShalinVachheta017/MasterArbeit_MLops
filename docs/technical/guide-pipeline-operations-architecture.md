# Pipeline Operations and Architecture

**HAR MLOps Production Pipeline — Operational Reference**

> Master Thesis: MLOps-Enhanced Human Activity Recognition for Anxiety Detection Using Wearable Sensors  
> Author: Shalin Vachheta  
> Pipeline Version: 2.1.0  
> Last Updated: February 2026

---

## Table of Contents

1. [Automatic Detection of New Raw Files](#1-automatic-detection-of-new-raw-files)
2. [Handling Multiple File Types](#2-handling-multiple-file-types)
3. [Stage-Based Execution](#3-stage-based-execution)
4. [Calibration vs Normal Run](#4-calibration-vs-normal-run)
5. [Adaptation and Pseudo-Labeling](#5-adaptation-and-pseudo-labeling)
6. [Custom Model Path and Versioning](#6-custom-model-path-and-versioning)
7. [Project Cleanup Plan](#7-project-cleanup-plan)
8. [Environment Setup](#8-environment-setup)

---

## 1. Automatic Detection of New Raw Files

### 1.1 Problem Statement

Raw wearable sensor data arrives as CSV or Excel file pairs in `data/raw/`. Each recording session produces multiple files sharing a common timestamp prefix. The pipeline must determine which sessions are new and which have already been processed, ensuring that no session is ingested twice and no new session is missed.

### 1.2 File Discovery Logic

The ingestion stage scans `data/raw/` for sensor files using glob matching. Files are identified by sensor keyword in the filename:

```
data/raw/
├── 2025-03-23-15-23-10-accelerometer_data.xlsx
├── 2025-03-23-15-23-10-gyroscope_data.xlsx
├── 2025-08-19-13-05-35_accelerometer.csv        ← copied from Decoded/
├── 2025-08-19-13-05-35_gyroscope.csv             ← copied from Decoded/
└── Decoded/
    ├── 2025-07-16-21-03-13_accelerometer.csv
    ├── 2025-07-16-21-03-13_gyroscope.csv
    ├── 2025-07-16-21-03-13_record.csv
    └── ... (26 sessions, 78 files)
```

The function `find_latest_sensor_pair()` in `src/sensor_data_pipeline.py` performs the following steps:

1. **Glob for accelerometer files:** `raw_dir.glob("*accelerometer*.*")`
2. **Glob for gyroscope files:** `raw_dir.glob("*gyroscope*.*")`
3. **Sort by modification time** (newest first)
4. **Pair by prefix match:** Extract the portion before the keyword `accelerometer` and find a gyroscope file sharing the same prefix
5. **Fallback:** If no prefix match exists, pair the newest accelerometer with the newest gyroscope

### 1.3 Session Identification via Timestamp

Each recording session is uniquely identified by its timestamp prefix. Grouping logic:

```
Timestamp Prefix               Files in Session
───────────────────────────     ──────────────────────────────────────────────
2025-07-16-21-03-13             _accelerometer.csv, _gyroscope.csv, _record.csv
2025-07-22-09-15-44             _accelerometer.csv, _gyroscope.csv, _record.csv
2025-08-19-13-05-35             _accelerometer.csv, _gyroscope.csv
```

The timestamp string (e.g., `2025-07-16-21-03-13`) serves as the **session identifier**. This is extracted by splitting the filename on the sensor keyword:

```python
session_id = filename.split("_accelerometer")[0]
# or
session_id = filename.split("-accelerometer")[0]
```

### 1.4 Processed Session Registry

To prevent duplicate processing, the pipeline maintains an **artifact registry** that tracks which sessions have already been ingested. The registry is implemented as a JSON file:

```
logs/pipeline/processed_sessions.json
```

Structure:

```json
{
  "sessions": {
    "2025-03-23-15-23-10": {
      "status": "completed",
      "ingested_at": "2026-02-12T10:30:00",
      "pipeline_run_id": "20260212_103000",
      "source_type": "excel",
      "fused_csv": "data/processed/sensor_fused_50Hz.csv",
      "n_rows": 542810
    },
    "2025-08-19-13-05-35": {
      "status": "completed",
      "ingested_at": "2026-02-12T14:15:00",
      "pipeline_run_id": "20260212_141500",
      "source_type": "csv",
      "fused_csv": "data/processed/sensor_fused_50Hz.csv",
      "n_rows": 261002
    }
  }
}
```

### 1.5 Detection and Decision Flow

```
┌──────────────────────────────────┐
│  Scan data/raw/ for sensor files │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Group files by timestamp prefix │
│  (identify unique sessions)      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Load processed_sessions.json    │
└──────────────┬───────────────────┘
               │
               ▼
       ┌───────┴────────┐
       │                │
       ▼                ▼
  ┌─────────┐    ┌────────────┐
  │  Known  │    │  Unknown   │
  │ session │    │  session   │
  └────┬────┘    └─────┬──────┘
       │               │
       ▼               ▼
  ┌──────────┐   ┌────────────────┐
  │  SKIP    │   │ Validate files │
  │  + log   │   │ (acc + gyro    │
  │ "Already │   │  present?)     │
  │ processed│   └───────┬────────┘
  │  "       │           │
  └──────────┘     ┌─────┴─────┐
                   │           │
                   ▼           ▼
             ┌──────────┐ ┌────────────┐
             │ Complete │ │  Missing   │
             │  pair    │ │  file(s)   │
             └────┬─────┘ └─────┬──────┘
                  │             │
                  ▼             ▼
             ┌──────────┐ ┌────────────┐
             │ PROCESS  │ │   SKIP     │
             │ + update │ │  + NOTIFY  │
             │ registry │ │  "Missing  │
             └──────────┘ │  gyroscope"│
                          └────────────┘
```

### 1.6 Logging Policy

Every detection decision is logged explicitly:

| Scenario | Log Level | Message |
|----------|-----------|---------|
| Session already processed | `INFO` | `Session 2025-07-16-21-03-13 already processed (run 20260212_103000). Skipping.` |
| New session detected | `INFO` | `New session detected: 2025-08-19-13-05-35. Starting pipeline.` |
| Missing required file | `WARNING` | `Session 2025-08-01-10-22-15 missing gyroscope file. Skipping.` |
| Partially processed | `WARNING` | `Session 2025-07-22-09-15-44 has status 'partial'. Manual review recommended.` |
| Registry updated | `INFO` | `Registered session 2025-08-19-13-05-35 as completed.` |

### 1.7 Guarantees

This design provides the following production-ready guarantees:

- **Idempotent pipeline runs.** Running the pipeline multiple times with the same data produces the same result. Completed sessions are skipped, not reprocessed.
- **No duplicate artifacts.** Each session produces exactly one fused CSV, one set of predictions, and one MLflow run entry.
- **Auditability.** The processed session registry provides a complete history of which raw files were ingested, when, and by which pipeline run.
- **Fail-safe behaviour.** If a session fails mid-processing, it is recorded as `partial` rather than `completed`, ensuring it can be retried.

---

## 2. Handling Multiple File Types

### 2.1 File Types per Recording Session

Each Garmin wearable recording session produces up to three file types:

| File | Content | Sampling Rate | Required |
|------|---------|--------------|----------|
| `{ts}_accelerometer.csv` | Tri-axial acceleration (`accel_x`, `accel_y`, `accel_z`) | ~50 Hz (high-rate) | **Yes** |
| `{ts}_gyroscope.csv` | Tri-axial angular velocity (`gyro_x`, `gyro_y`, `gyro_z`) | ~50 Hz (high-rate) | **Yes** |
| `{ts}_record.csv` | Aggregated per-second summary with heart rate, cadence, distance, plus compressed sensor arrays | 1 Hz (summary) | **No** (optional) |

For the original Garmin export format, the same data arrives as Excel files:

| File | Content | Required |
|------|---------|----------|
| `{ts}-accelerometer_data.xlsx` | Sensor data with list-encoded columns | **Yes** |
| `{ts}-gyroscope_data.xlsx` | Sensor data with list-encoded columns | **Yes** |

### 2.2 Session Integrity Validation

Before processing a session, the ingestion stage validates file completeness:

```
Required pairing:
  accelerometer + gyroscope → VALID (proceed with sensor fusion)
  accelerometer only       → VALID (proceed in accel-only mode, reduced accuracy)
  gyroscope only           → INVALID (skip session, notify user)
  neither found            → INVALID (skip session, notify user)
```

Validation rules:

1. **Timestamp match:** The accelerometer and gyroscope filenames must share the same timestamp prefix, confirming they belong to the same recording session.
2. **Column presence:** After loading, required columns are validated:
   - Accelerometer: `timestamp`, `timestamp_ms`, `sample_time_offset`, `accel_x` (or `x`), `accel_y` (or `y`), `accel_z` (or `z`)
   - Gyroscope: `timestamp`, `timestamp_ms`, `sample_time_offset`, `gyro_x` (or `x`), `gyro_y` (or `y`), `gyro_z` (or `z`)
3. **Row count:** Both files should have comparable row counts (within an order of magnitude). Large discrepancies indicate corrupted data.

### 2.3 Record File Handling

The `record.csv` file is optional and provides supplementary metadata:

```csv
accelerometer_x, accelerometer_y, accelerometer_z,
cadence, distance, fractional_cadence,
gyroscope_x, gyroscope_y, gyroscope_z,
heart_rate, timestamp, unknown_134, ...
```

Processing logic:

- **If `record.csv` exists:** Extract heart rate, cadence, and distance columns. Merge as metadata alongside the fused sensor data. These auxiliary features can support multi-modal analysis but are not required for core HAR inference.
- **If `record.csv` is absent:** Continue with accelerometer and gyroscope only. The pipeline operates in **sensor-only mode**, which is sufficient for all 14 pipeline stages.

### 2.4 Format Handling — CSV vs Excel

The `DataIngestion` component in `src/components/data_ingestion.py` supports two ingestion paths:

| Path | Trigger | Processing |
|------|---------|------------|
| **Path A: Direct CSV** | `--input-csv` flag provided | Load CSV directly, copy to `data/processed/` |
| **Path B: Excel Pair** | No `--input-csv` flag | Auto-detect Excel pair via `find_latest_sensor_pair()`, then process through `SensorDataLoader` → `DataProcessor` → `SensorFusion` → `Resampler` |

For the Decoded CSV format (which already has per-sample columns rather than list-encoded values), Path A is used. The CSV is loaded directly and passed to the transformation stage.

### 2.5 Sensor Fusion Pipeline

When both accelerometer and gyroscope data are available, the sensor fusion pipeline produces a unified time-aligned output:

```
Accelerometer CSV        Gyroscope CSV
  (accel_x/y/z)           (gyro_x/y/z)
       │                       │
       ▼                       ▼
  ┌──────────┐           ┌──────────┐
  │ Load +   │           │ Load +   │
  │ Validate │           │ Validate │
  └────┬─────┘           └────┬─────┘
       │                       │
       ▼                       ▼
  ┌──────────┐           ┌──────────┐
  │ Column   │           │ Column   │
  │ Rename   │           │ Rename   │
  │ (→x,y,z) │           │ (→x,y,z) │
  └────┬─────┘           └────┬─────┘
       │                       │
       ▼                       ▼
  ┌──────────┐           ┌──────────┐
  │ Explode  │           │ Explode  │
  │ List     │           │ List     │
  │ Columns  │           │ Columns  │
  └────┬─────┘           └────┬─────┘
       │                       │
       └──────────┬────────────┘
                  ▼
          ┌──────────────┐
          │ Timestamp    │
          │ Alignment    │
          │ (merge_asof) │
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │ Resample to  │
          │ 50 Hz        │
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │ Output:      │
          │ sensor_fused │
          │ _50Hz.csv    │
          └──────────────┘
```

Output schema (`sensor_fused_50Hz.csv`):

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 datetime |
| `Ax_w` | Accelerometer X (m/s² or mG) |
| `Ay_w` | Accelerometer Y |
| `Az_w` | Accelerometer Z |
| `Gx_w` | Gyroscope X (°/s) |
| `Gy_w` | Gyroscope Y |
| `Gz_w` | Gyroscope Z |

---

## 3. Stage-Based Execution

### 3.1 The 14-Stage Pipeline Architecture

The pipeline is divided into 14 stages organized into three execution groups:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE CYCLE (Stages 1–7)                     │
│                    Default execution — every run                    │
├──────┬──────────────────────┬───────────────────────────────────────┤
│  #   │  Stage               │  Purpose                             │
├──────┼──────────────────────┼───────────────────────────────────────┤
│  1   │  ingestion           │  Raw files → sensor_fused_50Hz.csv   │
│  2   │  validation          │  Schema + value-range checks         │
│  3   │  transformation      │  CSV → normalised, windowed .npy     │
│  4   │  inference           │  .npy + model → predictions          │
│  5   │  evaluation          │  Confidence / distribution / ECE     │
│  6   │  monitoring          │  3-layer: confidence, temporal, drift │
│  7   │  trigger             │  Automated retraining decision       │
└──────┴──────────────────────┴───────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  RETRAINING CYCLE (Stages 8–10)                     │
│                  Enabled via --retrain flag                          │
├──────┬──────────────────────┬───────────────────────────────────────┤
│  8   │  retraining          │  Standard / AdaBN / pseudo-label      │
│  9   │  registration        │  Version, deploy, rollback            │
│  10  │  baseline_update     │  Rebuild drift baselines              │
└──────┴──────────────────────┴───────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                ADVANCED ANALYTICS (Stages 11–14)                    │
│                Enabled via --advanced flag                           │
├──────┬──────────────────────┬───────────────────────────────────────┤
│  11  │  calibration         │  Temperature scaling, MC Dropout, ECE │
│  12  │  wasserstein_drift   │  Wasserstein distance, change-points  │
│  13  │  curriculum_pseudo   │  Progressive self-training with EWC   │
│      │  _labeling           │                                       │
│  14  │  sensor_placement    │  Hand detection, axis mirroring       │
└──────┴──────────────────────┴───────────────────────────────────────┘
```

### 3.2 Why Not Always Run the Full Pipeline?

Running all 14 stages on every invocation is neither necessary nor efficient. Stage-based execution provides five critical benefits:

**1. Faster Debugging**

When a failure occurs in Stage 6 (monitoring), re-running stages 1–5 wastes time. Targeted execution isolates the problem:

```bash
python run_pipeline.py --stages monitoring
```

**2. Reproducibility**

Intermediate artifacts (CSVs, NPY arrays, JSON reports) are persisted after each stage. A specific stage can be re-run against fixed inputs, producing deterministic outputs. This is essential for thesis experiments where exact reproduction of results is required.

**3. Production Control**

In production, the inference cycle (stages 1–7) runs on every new data batch. Retraining (stages 8–10) runs only when drift is detected. Advanced analytics (stages 11–14) run on demand for deep diagnostics. These have fundamentally different scheduling requirements.

**4. Reduced Compute Cost**

| Stage Group | Approx. Runtime | Resources |
|-------------|----------------|-----------|
| Inference Cycle (1–7) | ~2 minutes | CPU sufficient |
| Retraining Cycle (8–10) | ~15–30 minutes | GPU recommended |
| Advanced Analytics (11–14) | ~5–10 minutes | CPU sufficient |
| Full Pipeline (1–14) | ~25–45 minutes | GPU recommended |

Running stages 1–7 repeatedly instead of 1–14 saves over 80% of compute time.

**5. Easier CI/CD Integration**

In the GitHub Actions workflow, stages 1–7 are run as smoke tests on every push. Retraining stages are run only on tagged releases. This separation maps cleanly to CI/CD job boundaries.

### 3.3 CLI Reference for Stage Selection

```bash
# Default: stages 1–7 (inference cycle)
python run_pipeline.py

# Specific stages only
python run_pipeline.py --stages inference evaluation

# Stages 1–7 + 8–10 (inference + retraining)
python run_pipeline.py --retrain

# Stages 1–7 + 11–14 (inference + advanced)
python run_pipeline.py --advanced

# Stages 1–14 (full pipeline)
python run_pipeline.py --retrain --advanced

# Retraining with AdaBN domain adaptation
python run_pipeline.py --retrain --adapt adabn

# Only advanced analytics on existing data
python run_pipeline.py --stages calibration wasserstein_drift sensor_placement

# Skip early stages (use existing processed data)
python run_pipeline.py --skip-ingestion --skip-validation

# Continue even if a stage fails
python run_pipeline.py --continue-on-failure
```

### 3.4 Stage Dependency Graph

Stages have ordered dependencies. The pipeline resolves these automatically through fallback artifacts:

```
ingestion ──▶ validation ──▶ transformation ──▶ inference ──▶ evaluation
                                                    │
                                                    ▼
                                              monitoring ──▶ trigger
                                                    │
                                         ┌──────────┼──────────┐
                                         ▼          ▼          ▼
                                    retraining  registration  baseline_update
                                         │
                              ┌──────────┼──────────┬──────────┐
                              ▼          ▼          ▼          ▼
                         calibration  wasserstein  curriculum  sensor
                                      _drift     _pseudo     _placement
                                                  _labeling
```

When a stage is skipped (e.g., `--skip-ingestion`), the pipeline creates a **fallback artifact** pointing to the most recent output file. This allows downstream stages to proceed without re-processing upstream data.

### 3.5 Argument Parser Design

The pipeline uses Python's `argparse` for modular stage selection. The `--stages` argument accepts one or more stage names from the defined set:

```python
parser.add_argument("--stages", nargs="+", choices=[
    "ingestion", "validation", "transformation",
    "inference", "evaluation", "monitoring", "trigger",
    "retraining", "registration", "baseline_update",
    "calibration", "wasserstein_drift",
    "curriculum_pseudo_labeling", "sensor_placement",
], default=None)
```

This design follows the **composable pipeline** pattern: each stage is an independently executable unit with defined inputs and outputs, composed at runtime based on CLI arguments.

---

## 4. Calibration vs Normal Run

### 4.1 Two Execution Modes

The pipeline distinguishes between two fundamentally different execution modes:

| Aspect | Normal Run | Calibration Run |
|--------|-----------|-----------------|
| **Purpose** | Process data → inference → monitoring | Establish baselines and tune thresholds |
| **Frequency** | Every new data batch | Once at deployment, or after model retraining |
| **Stages** | 1–7 (inference cycle) | 11–14 (advanced analytics) |
| **Output** | Predictions, monitoring reports | Calibration parameters, baseline distributions |
| **CLI** | `python run_pipeline.py` | `python run_pipeline.py --advanced` |

### 4.2 What Calibration Mode Does

#### Temperature Scaling (Stage 11)

Raw neural network softmax outputs are known to be **overconfident** (Guo et al., 2017). Temperature scaling learns a single parameter $T$ that divides the logits before softmax:

$$\hat{p}_i = \text{softmax}(z_i / T)$$

The calibration stage:

1. Loads validation logits from the model
2. Optimises $T$ via negative log-likelihood minimisation
3. Saves the learned $T$ to `artifacts/calibration/temperature.json`
4. Computes the Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

#### MC Dropout Uncertainty (Stage 11)

Monte Carlo Dropout (Gal & Ghahramani, 2016) estimates predictive uncertainty by performing $K$ stochastic forward passes with dropout enabled at inference time:

1. Run $K=30$ forward passes (configurable via `--mc-dropout-passes`)
2. Compute **predictive entropy:** $H[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$
3. Compute **mutual information** (epistemic uncertainty): $I[y, \omega | x] = H[\bar{p}] - \mathbb{E}[H[p_k]]$
4. Flag high-entropy samples as uncertain

#### Gravity Removal Tuning

When `--gravity-removal` is enabled, the calibration mode determines the optimal high-pass filter parameters for removing the gravitational component from accelerometer readings:

- Cutoff frequency: 0.3 Hz (configurable in `config/pipeline_config.yaml`)
- Filter order: 3rd-order Butterworth
- These parameters are tuned once and reused for all subsequent normal runs

#### Baseline Distribution Capture

Stages 12 and 14 establish reference distributions:

- **Wasserstein baselines** (Stage 12): Per-channel distribution snapshots from training or validation data, used as the reference point for drift detection in normal runs
- **Sensor placement baselines** (Stage 14): Dominant vs non-dominant hand feature distributions, used for automatic hand-side detection

### 4.3 Why Calibration Should Be Separate

1. **One-time setup cost.** Temperature scaling, gravity filter tuning, and baseline computation need to run only once (or after model retraining). Embedding them in every inference run wastes compute.

2. **Different thresholds.** Calibration determines the thresholds that normal runs use for classification (confidence, entropy, drift). If calibration ran simultaneously, thresholds would be computed and applied in the same run, creating a circular dependency.

3. **Cleaner architecture.** Separating calibration from inference follows the **train/calibrate/serve** paradigm. The model is trained (Stages 8–10), calibrated (Stages 11–14), and then served (Stages 1–7) in distinct phases.

4. **Reproducibility.** Calibration parameters are saved as artifacts. Multiple normal runs share the same calibration, ensuring consistent behaviour across inference batches.

### 4.4 When to Re-Run Calibration

| Trigger | Action |
|---------|--------|
| Model retrained (Stages 8–10 completed) | Re-run calibration (Stages 11–14) |
| New sensor hardware deployed | Re-run gravity removal tuning |
| Wasserstein drift detected above critical threshold | Re-run baseline capture |
| Switch between dominant/non-dominant hand | Re-run sensor placement calibration |

---

## 5. Adaptation and Pseudo-Labeling

### 5.1 The Problem: No Ground-Truth Labels in Production

In a research setting, training and validation data are fully labelled. In production, new wearable recordings arrive without activity labels. The model must still adapt to:

- **Domain shift:** Different users, different wrist placement, different movement patterns
- **Temporal drift:** Gradual changes in user behaviour over weeks or months
- **Sensor variance:** Manufacturing tolerances, firmware updates, battery degradation

Traditional supervised retraining requires labelled data, which is expensive and impractical to obtain continuously.

### 5.2 Solution: Pseudo-Labeling with Curriculum Learning

The pipeline implements **curriculum pseudo-labeling** (Stage 13), a self-training strategy that generates synthetic labels from high-confidence model predictions.

#### Algorithm Overview

```
Input:  Trained model M, labelled source data (X_s, y_s),
        unlabelled production data X_u
Output: Fine-tuned model M'

for iteration i = 1 to N:
    1. Predict on X_u using M:
         ŷ_u = M(X_u),  confidence c_u = max(softmax(M(X_u)))

    2. Compute threshold τ_i:
         τ_i = τ_start - (τ_start - τ_end) × (i / N)
         (Linear decay from 0.95 → 0.80)

    3. Select pseudo-labelled samples:
         S_i = { (x, ŷ) : c(x) ≥ τ_i }
         Apply class balancing: max K samples per class

    4. Combine with source data:
         D_i = (X_s, y_s) ∪ S_i

    5. Fine-tune M on D_i with EWC regularisation:
         L = L_task(D_i) + λ × L_EWC(θ, θ*)

    6. Update teacher model (EMA):
         θ_teacher = α × θ_teacher + (1 - α) × θ_student

return M'
```

#### Curriculum Schedule

The confidence threshold decays linearly across iterations, implementing a curriculum that starts with easy (high-confidence) samples and progressively includes harder ones:

```
Iteration    Threshold    Effect
─────────    ─────────    ────────────────────────────────────────
    1          0.95       Only very confident predictions (easy)
    2          0.91       Slightly lower bar
    3          0.87       More samples included
    4          0.83       Approaching boundary region
    5          0.80       Moderate confidence (harder samples)
```

Configuration via CLI:

```bash
python run_pipeline.py --stages curriculum_pseudo_labeling \
    --curriculum-iterations 10 \
    --ewc-lambda 500.0
```

### 5.3 Elastic Weight Consolidation (EWC)

When fine-tuning on pseudo-labelled production data, the model risks **catastrophic forgetting** — losing performance on the original training distribution. EWC (Kirkpatrick et al., 2017) adds a quadratic penalty that anchors important parameters near their pre-trained values:

$$L_{\text{EWC}} = \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^*)^2$$

Where:
- $\theta^*$ is the pre-trained parameter vector
- $F_i$ is the diagonal of the Fisher Information Matrix, estimating each parameter's importance
- $\lambda$ controls the regularisation strength (default: 1000.0)

Higher $\lambda$ values preserve the original model more aggressively, at the cost of slower adaptation. Lower values allow faster adaptation but risk greater forgetting.

### 5.4 Domain Adaptation via AdaBN

For cases where pseudo-labeling is insufficient, the pipeline supports **Adaptive Batch Normalisation (AdaBN)**. This method adapts the model to a new domain by recalculating batch normalisation statistics on the target data, without modifying any model weights:

```bash
python run_pipeline.py --retrain --adapt adabn
```

AdaBN procedure:
1. Freeze all model parameters
2. Pass $N$ batches of production data through the network
3. Update running mean and variance in all BatchNorm layers
4. The adapted model reflects the production data distribution

This is a lightweight adaptation method (no gradient descent, no labels needed) suitable for small domain shifts such as same user on a different day.

### 5.5 Addressing Wrist-Side Domain Shift

A specific source of domain shift in wrist-worn HAR is the difference between dominant and non-dominant hand placement. The **sensor placement module** (Stage 14) addresses this:

| Case | Training Data | Deployment | Strategy |
|------|--------------|------------|----------|
| A | Dominant hand | Dominant hand | No adaptation needed |
| B | Dominant hand | Non-dominant hand | Axis mirroring augmentation |
| C | Non-dominant hand | Dominant hand | Axis mirroring augmentation |
| D | Both hands available | Either hand | Sensor fusion |

The axis mirroring augmentation flips the Y and Z axes of both accelerometer and gyroscope channels (indices `[1, 2, 4, 5]`), simulating the effect of wearing the sensor on the opposite wrist.

---

## 6. Custom Model Path and Versioning

### 6.1 Current Model Loading

The pipeline loads the pretrained model from a default path:

```
models/pretrained/fine_tuned_model_1dcnnbilstm.keras
```

This path can be overridden at runtime:

```bash
python run_pipeline.py --model "models/pretrained/my_custom_model.keras"
```

### 6.2 Model Versioning Strategy

The model registration stage (Stage 9) implements a versioned model directory structure:

```
models/
├── pretrained/
│   └── fine_tuned_model_1dcnnbilstm.keras     ← production model
├── archived_experiments/
│   ├── v1.0_baseline_20260115/
│   │   ├── model.keras
│   │   └── metrics.json
│   ├── v1.1_adabn_20260120/
│   │   ├── model.keras
│   │   └── metrics.json
│   └── v2.0_curriculum_20260210/
│       ├── model.keras
│       └── metrics.json
```

Each version directory contains:
- The serialised Keras model (`.keras` format)
- A `metrics.json` file with validation performance
- Optional: scaler configuration, calibration temperature

### 6.3 MLflow Model Registry

For more sophisticated versioning, the pipeline integrates with the MLflow Model Registry:

```python
# During registration (Stage 9)
mlflow.keras.log_model(model, "har-model")
mlflow.register_model(
    model_uri=f"runs:/{run_id}/har-model",
    name="har-1dcnn-bilstm"
)
```

MLflow tracks:
- Model versions (auto-incrementing)
- Stage transitions (`None` → `Staging` → `Production` → `Archived`)
- Version metadata (parameters, metrics, tags)
- Lineage (which training run produced which model version)

### 6.4 Model Loading Priority

When the pipeline resolves a model path, it follows this priority order:

1. **CLI argument:** `--model path/to/model.keras` (highest priority)
2. **MLflow registry:** Latest model in `Production` stage
3. **Default path:** `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`

### 6.5 Rollback Procedure

If a newly deployed model underperforms:

```bash
# 1. Check archived versions
ls models/archived_experiments/

# 2. Roll back to a specific version
python run_pipeline.py --model "models/archived_experiments/v1.0_baseline_20260115/model.keras"

# 3. Or via MLflow: transition current Production model back to Staging
```

The registration stage performs **proxy validation** before deployment: it compares the new model's confidence distribution and ECE against the current production model. If the new model is worse on these proxy metrics, deployment is blocked and the operator is notified.

---

## 7. Project Cleanup Plan

### 7.1 Current Repository State

The repository currently serves dual purposes: research exploration and production pipeline. This results in a mixed structure containing experimental notebooks, draft papers, and production code in the same tree.

Current structure (simplified):

```
MasterArbeit_MLops/
├── src/                          ← Production pipeline code
├── tests/                        ← Test suites
├── config/                       ← Configuration files
├── data/                         ← Raw and processed data
├── models/                       ← Trained models
├── notebooks/                    ← Experimental notebooks (research)
├── papers/                       ← 76+ research papers (research)
├── archive/                      ← Old scripts and notes
├── docs/                         ← Documentation
├── ai helps/                     ← AI conversation logs
├── cheat sheet/                  ← Personal reference material
├── research_papers/              ← Duplicate paper storage
├── images/                       ← Miscellaneous images
├── outputs/                      ← Pipeline outputs
├── reports/                      ← Generated reports
├── logs/                         ← Pipeline logs
├── mlruns/                       ← MLflow tracking data
├── docker/                       ← Docker configuration
├── scripts/                      ← Utility scripts
└── .github/                      ← CI/CD workflows
```

### 7.2 Proposed Clean Structure

The repository should be separated into two distinct repositories:

#### Production Pipeline Repository

```
har-mlops-pipeline/
├── src/
│   ├── components/               ← 14 pipeline stage components
│   ├── entity/                   ← Config + artifact dataclasses
│   ├── pipeline/                 ← ProductionPipeline orchestrator
│   ├── calibration.py            ← Calibration & UQ
│   ├── wasserstein_drift.py      ← Wasserstein drift detection
│   ├── curriculum_pseudo_labeling.py  ← Self-training
│   ├── sensor_placement.py       ← Hand detection & mirroring
│   ├── robustness.py             ← Noise injection testing
│   ├── sensor_data_pipeline.py   ← Raw data processing
│   ├── mlflow_tracking.py        ← Experiment tracking
│   └── train.py                  ← Model training
├── tests/                        ← Unit + integration tests
├── config/
│   ├── pipeline_config.yaml
│   └── mlflow_config.yaml
├── data/
│   ├── raw/                      ← Raw sensor files
│   ├── processed/                ← Fused CSVs
│   └── prepared/                 ← Windowed NPY arrays
├── models/
│   ├── pretrained/               ← Production model
│   └── archived_experiments/     ← Version history
├── docker/
│   ├── Dockerfile.inference
│   └── Dockerfile.training
├── scripts/                      ← Utility scripts
├── logs/                         ← Pipeline logs
├── outputs/                      ← Predictions
├── .github/workflows/            ← CI/CD
├── pyproject.toml                ← Package definition
├── run_pipeline.py               ← Entry point
└── README.md
```

#### Research Repository

```
har-thesis-research/
├── notebooks/                    ← Experimental notebooks
├── papers/                       ← Research paper collection
├── docs/
│   ├── thesis/                   ← Thesis chapters
│   └── research/                 ← Literature notes
├── experiments/                  ← One-off experiment scripts
└── figures/                      ← Generated figures for thesis
```

### 7.3 Cleanup Actions

| Priority | Action | Files Affected |
|----------|--------|---------------|
| 1 | Move `papers/`, `research_papers/` to research repo | ~80 files |
| 2 | Move `notebooks/exploration/` to research repo | ~5 files |
| 3 | Move `archive/`, `ai helps/`, `cheat sheet/` to research repo | ~30 files |
| 4 | Remove duplicate files (`research_papers/` mirrors `papers/`) | ~20 files |
| 5 | Resolve merge conflicts in `config/requirements.txt` | 1 file |
| 6 | Remove `images/` (not used by pipeline) | ~5 files |
| 7 | Add `.gitignore` entries for `mlruns/`, `outputs/`, `logs/` | 1 file |
| 8 | Ensure `pyproject.toml` is single source of truth for dependencies | 2 files |

### 7.4 Data Management

Large data files should not reside in Git. The project already uses DVC (Data Version Control) for this:

```
data/raw.dvc            ← DVC pointer to raw data
data/processed.dvc      ← DVC pointer to processed data
data/prepared.dvc       ← DVC pointer to prepared data
models/pretrained.dvc   ← DVC pointer to model files
```

After cleanup, the `.gitignore` should exclude:

```gitignore
# Data (tracked by DVC)
data/raw/
data/processed/
data/prepared/
models/pretrained/

# Pipeline outputs
outputs/
logs/
mlruns/

# Python
__pycache__/
*.pyc
.eggs/
*.egg-info/
```

---

## 8. Environment Setup

### 8.1 Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.10 (recommended: 3.11) | Runtime |
| pip | ≥ 23.0 | Package management |
| Git | ≥ 2.40 | Version control |
| DVC | ≥ 3.50 | Data versioning |
| Docker | ≥ 24.0 | Containerisation (optional) |
| CUDA | ≥ 12.0 | GPU acceleration (optional) |

### 8.2 Environment Setup — Step by Step

#### Option A: Virtual Environment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/<user>/MasterArbeit_MLops.git
cd MasterArbeit_MLops

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 4. Install the package with all dependencies
pip install -e ".[dev]"

# 5. Verify installation
python -c "from src.pipeline.production_pipeline import ProductionPipeline; print('OK')"
```

#### Option B: Using requirements.txt

```bash
# 1. Create and activate virtual environment (same as above)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r config/requirements.txt

# 3. Install the project in editable mode
pip install -e .
```

#### Option C: Conda Environment

```bash
# 1. Create conda environment
conda create -n har-mlops python=3.11 -y
conda activate har-mlops

# 2. Install dependencies
pip install -e ".[dev]"
```

### 8.3 Data Setup

```bash
# Pull data tracked by DVC
dvc pull

# Verify data structure
ls data/raw/
ls data/processed/
ls models/pretrained/
```

### 8.4 Configuration Files

| File | Purpose |
|------|---------|
| `config/pipeline_config.yaml` | Preprocessing parameters (sampling rate, window size, filters) |
| `config/mlflow_config.yaml` | MLflow tracking URI, experiment name, model registry settings |
| `pyproject.toml` | Package metadata, dependencies, tool configurations |
| `pytest.ini` | Test runner configuration |
| `docker-compose.yml` | Multi-container orchestration (API + monitoring) |

### 8.5 Running the Pipeline

```bash
# Basic inference pipeline (stages 1–7)
python run_pipeline.py

# With specific input data
python run_pipeline.py --input-csv "data/raw/Decoded/2025-08-19-13-05-35_accelerometer.csv"

# Full pipeline with retraining + advanced analytics
python run_pipeline.py --retrain --advanced --continue-on-failure

# Start MLflow UI to view results
mlflow ui --backend-store-uri mlruns --port 5000
```

### 8.6 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific test category
pytest tests/ -m "calibration" -v
pytest tests/ -m "robustness" -v
```

### 8.7 Docker Deployment

```bash
# Build inference container
docker build -f docker/Dockerfile.inference -t har-inference:latest .

# Run inference API
docker run -d -p 8000:8000 --name har-api har-inference:latest

# Health check
curl http://localhost:8000/health

# Or use docker-compose for full stack
docker-compose up -d
```

### 8.8 Why Environment Isolation Matters

Environment isolation is critical for reproducibility in an MLOps context:

1. **Dependency pinning.** The `pyproject.toml` and `requirements.txt` specify minimum versions for all packages. A virtual environment ensures these exact versions are used, preventing conflicts with system-wide packages.

2. **Cross-platform consistency.** The CI/CD pipeline (GitHub Actions) uses `ubuntu-latest`. Development occurs on Windows. A virtual environment ensures the same package versions regardless of OS.

3. **Experiment reproducibility.** MLflow logs the Python environment for each run. If a virtual environment is used, the logged environment matches the actual execution environment exactly.

4. **Deployment parity.** The Docker container installs the same `requirements.txt`. Using a virtual environment during development mirrors the container's isolated dependency tree.

---

## Appendix: Complete CLI Reference

```
usage: run_pipeline.py [options]

Stage Selection:
  --stages STAGE [STAGE ...]    Run only these stages
  --skip-ingestion              Skip Stage 1
  --skip-validation             Skip Stage 2
  --retrain                     Enable Stages 8–10
  --advanced                    Enable Stages 11–14
  --continue-on-failure         Log errors, continue to next stage

Input Overrides:
  --input-csv PATH              Direct CSV file input
  --model PATH                  Custom model file path
  --gravity-removal             Enable gravity component removal
  --calibrate                   Enable sensor calibration

Retraining Parameters:
  --adapt {adabn,pseudo_label,none}  Adaptation method
  --labels PATH                 Path to label file (supervised retraining)
  --epochs N                    Training epochs (default: 100)
  --auto-deploy                 Auto-deploy after registration

Advanced Analytics:
  --mc-dropout-passes N         MC Dropout forward passes (default: 30)
  --curriculum-iterations N     Pseudo-labeling iterations (default: 5)
  --ewc-lambda FLOAT            EWC regularisation strength (default: 1000.0)
```

---

*This document is part of the Master Thesis: "MLOps-Enhanced Human Activity Recognition for Anxiety Detection Using Wearable Sensors." All code references correspond to the implementation in the `src/` directory of the project repository.*

---

## 9. Thesis Completion Progress Tracker

> **Snapshot Date:** February 12, 2026  
> **Previous Assessment (Jan 31):** 68%  
> **Current Assessment:** ~76%  
> **Deadline:** May 20, 2026 (14 weeks remaining)

### 9.1 Month-by-Month Progress vs Thesis Plan

The thesis plan defines 6 months of work (October 2025 – April 2026). The table below maps each planned milestone against actual deliverables, with completion percentages and evidence files.

---

#### Month 1: Data Ingestion and Preprocessing (Oct 2025)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| Literature review and detailed planning | DONE | 100% | [docs/APPENDIX_PAPER_INDEX.md](APPENDIX_PAPER_INDEX.md) — 35+ papers indexed; [docs/Bibliography_From_Local_PDFs.md](Bibliography_From_Local_PDFs.md) — formatted bibliography; `papers/` folder — 76+ PDFs across 7 subfolders |
| Data ingestion system for raw Garmin data | DONE | 95% | [src/sensor_data_pipeline.py](../src/sensor_data_pipeline.py) — 1,182 lines: Excel loader, column parser, sensor fusion, resampling; [src/components/data_ingestion.py](../src/components/data_ingestion.py) — 143 lines: two ingestion paths (CSV direct + Excel pair) |
| Preprocessing (missing values, resampling, features) | DONE | 95% | [src/preprocess_data.py](../src/preprocess_data.py) — 798 lines: unit detection, conversion, sliding windows, normalisation; [src/components/data_transformation.py](../src/components/data_transformation.py) — 131 lines: CSV to windowed NPY |
| Automated and reproducible pipeline | DONE | 90% | [run_pipeline.py](../run_pipeline.py) — 296 lines: single entry point with 20 CLI arguments; DVC tracking for data versioning (`data/raw.dvc`, `data/processed.dvc`) |

**Month 1 Overall: 95%**

---

#### Month 2: Model Training and Versioning (Nov 2025)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| Automated training loop with logging | DONE | 85% | [src/train.py](../src/train.py) — 925 lines: 5-fold stratified CV, MLflow tracking; [src/components/model_retraining.py](../src/components/model_retraining.py) — 234 lines: standard + AdaBN + pseudo-label |
| Hyperparameter optimisation | PARTIAL | 60% | Manual CLI flags for epochs, learning rate, batch size; no automated search (Optuna/Ray Tune) integrated yet |
| MLflow experiment tracking | DONE | 90% | [src/mlflow_tracking.py](../src/mlflow_tracking.py) — 654 lines: experiments, runs, metrics, model registry; [config/mlflow_config.yaml](../config/mlflow_config.yaml) — experiment configuration |
| Model versioning and registry | DONE | 85% | [src/model_rollback.py](../src/model_rollback.py) — 532 lines: version history, safe rollback; [src/components/model_registration.py](../src/components/model_registration.py) — 93 lines: MLflow registry + proxy validation |
| Docker training environment | DONE | 90% | [docker/Dockerfile.training](../docker/Dockerfile.training) — 54 lines: reproducible training container |

**Month 2 Overall: 82%**

---

#### Month 3: CI/CD and Deployment (Dec 2025)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| GitHub Actions CI/CD pipeline | DONE | 80% | [.github/workflows/ci-cd.yml](../.github/workflows/ci-cd.yml) — 282 lines: lint → test → Docker build → integration test → notify; [docs/GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) — step-by-step explanation |
| Automated testing in CI | DONE | 85% | 18 test files in `tests/`: unit tests for calibration, drift, retraining, robustness, OOD, etc.; pytest with coverage in CI; `pytest.ini` configured |
| Docker image build and push | DONE | 80% | CI workflow builds from `docker/Dockerfile.inference` and pushes to `ghcr.io`; GitHub Actions cache enabled |
| FastAPI inference API | DONE | 90% | [docker/api/main.py](../docker/api/main.py) — 447 lines: `/health`, `/model/info`, `/predict` endpoints |
| Docker containerised deployment | DONE | 85% | [docker-compose.yml](../docker-compose.yml) — 143 lines: MLflow server + inference API + training environment |
| GitHub Actions self-hosted runner | NOT DONE | 0% | No self-hosted runner configuration; using GitHub-hosted `ubuntu-latest` only |
| Integration / smoke tests in CI | DONE | 75% | CI workflow has smoke test job (health check + `scripts/inference_smoke.py`); model-validation job exists as placeholder |

**Month 3 Overall: 71%**

---

#### Month 4: Monitoring and Integration (Jan 2026)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| Data drift detection | DONE | 90% | [src/wasserstein_drift.py](../src/wasserstein_drift.py) — 453 lines: Wasserstein distance + change-point detection; [src/components/post_inference_monitoring.py](../src/components/post_inference_monitoring.py) — 96 lines: 3-layer monitoring (confidence, temporal, drift); PSI + KS-test in monitoring scripts |
| Prediction drift monitoring | DONE | 85% | [scripts/post_inference_monitoring.py](../scripts/post_inference_monitoring.py) — prediction distribution tracking; `reports/monitoring/` — 6 timestamped monitoring reports |
| Prometheus metrics exporter | DONE | 80% | [src/prometheus_metrics.py](../src/prometheus_metrics.py) — 623 lines: F1, accuracy, prediction metrics; [config/prometheus.yml](../config/prometheus.yml) — 70 lines: scrape configuration |
| Grafana dashboard | DONE | 75% | [config/grafana/har_dashboard.json](../config/grafana/har_dashboard.json) — dashboard definition |
| Alert rules | DONE | 75% | [config/alerts/har_alerts.yml](../config/alerts/har_alerts.yml) — 191 lines: model performance, drift, latency alerts |
| Prometheus/Grafana deployed and tested | NOT DONE | 20% | Config files exist but no evidence of live deployment or screenshots in thesis figures |
| Retraining trigger policy | DONE | 85% | [src/trigger_policy.py](../src/trigger_policy.py) — 812 lines: multi-metric voting for retrain decision; [src/components/trigger_evaluation.py](../src/components/trigger_evaluation.py) — 79 lines |
| OOD detection | DONE | 85% | [src/ood_detection.py](../src/ood_detection.py) — 462 lines: energy-based OOD (NeurIPS 2020) |

**Month 4 Overall: 74%**

---

#### Month 5: Refinement and Retraining Strategy (Feb 2026)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| Pipeline robustness and error handling | DONE | 85% | [src/robustness.py](../src/robustness.py) — 449 lines: noise injection, missing data, jitter; `--continue-on-failure` flag in CLI; fallback artifacts in production pipeline |
| Model calibration and uncertainty | DONE | 90% | [src/calibration.py](../src/calibration.py) — 544 lines: temperature scaling, MC Dropout, ECE, reliability diagrams; [src/components/calibration_uncertainty.py](../src/components/calibration_uncertainty.py) — 139 lines |
| Pseudo-labeling retraining prototype | DONE | 90% | [src/curriculum_pseudo_labeling.py](../src/curriculum_pseudo_labeling.py) — 466 lines: progressive self-training with EWC; [src/components/curriculum_pseudo_labeling.py](../src/components/curriculum_pseudo_labeling.py) — 134 lines |
| Domain adaptation (sensor placement) | DONE | 85% | [src/sensor_placement.py](../src/sensor_placement.py) — 345 lines: axis mirroring, hand detection; [src/domain_adaptation/adabn.py](../src/domain_adaptation/adabn.py) — AdaBN implementation |
| Active learning sample selection | DONE | 80% | [src/active_learning_export.py](../src/active_learning_export.py) — 715 lines: low confidence / high entropy sample export |
| Documentation for all new modules | DONE | 90% | [docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md](PIPELINE_OPERATIONS_AND_ARCHITECTURE.md) — this document: 8 sections covering all operational aspects |

**Month 5 Overall: 87%**

---

#### Month 6: Thesis Writing and Final Deliverables (Mar–Apr 2026)

| Planned Milestone | Status | % | Evidence |
| --- | --- | --- | --- |
| Thesis Chapter 1: Introduction | NOT STARTED | 0% | — |
| Thesis Chapter 2: Literature Review | PARTIAL | 25% | [docs/thesis/CONCEPTS_EXPLAINED.md](thesis/CONCEPTS_EXPLAINED.md) — key concepts; [docs/research/RESEARCH_PAPERS_ANALYSIS.md](research/RESEARCH_PAPERS_ANALYSIS.md) — paper analysis; bibliography generated — needs formal writing |
| Thesis Chapter 3: Methodology | PARTIAL | 20% | [docs/thesis/THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) — structure outline exists; pipeline architecture documented but not in thesis format |
| Thesis Chapter 4: Implementation | PARTIAL | 30% | All code exists and is documented; `docs/stages/` has per-stage deep dives — needs formal thesis chapter writing |
| Thesis Chapter 5: Results and Evaluation | NOT STARTED | 5% | `reports/monitoring/` has raw data; thesis figures script exists (`scripts/generate_thesis_figures.py`) — no formal results chapter |
| Thesis Chapter 6: Discussion and Future Work | NOT STARTED | 0% | — |
| Final presentation preparation | NOT STARTED | 0% | — |
| Code organisation and repository cleanup | PARTIAL | 40% | Section 7 of this doc defines cleanup plan; `archive/` folder created; still mixed research + production files |

**Month 6 Overall: 15%**

---

### 9.2 Overall Completion Summary

```
Month 1: Data Ingestion & Preprocessing     ████████████████████░  95%
Month 2: Training & Versioning              ████████████████░░░░░  82%
Month 3: CI/CD & Deployment                 ██████████████░░░░░░░  71%
Month 4: Monitoring & Integration           ███████████████░░░░░░  74%
Month 5: Refinement & Retraining            █████████████████░░░░  87%
Month 6: Thesis Writing                     ███░░░░░░░░░░░░░░░░░░  15%
─────────────────────────────────────────────────────────────────────
WEIGHTED OVERALL                             ██████████████░░░░░░░  76%
```

**Change since last assessment (Jan 31):** +8 percentage points (68% → 76%)

**Key improvements since Jan 31:**
- Calibration, Wasserstein drift, curriculum pseudo-labeling, sensor placement modules implemented (+4 new `src/` modules, +4 components)
- `pyproject.toml` created as single source of truth for packaging
- Pipeline expanded from 10 to 14 stages
- 6 new test files added
- 3 new documentation guides created

---

### 9.3 What Remains — Ranked by Priority

#### CRITICAL (Must Complete Before Submission)

| # | Task | Effort | Relevant Guide |
| --- | --- | --- | --- |
| 1 | **Write thesis chapters 1–6** | 6–8 weeks | [docs/thesis/THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) |
| 2 | **Generate formal results and figures** | 1 week | `scripts/generate_thesis_figures.py` |
| 3 | **Final presentation slides** | 1 week | — |

#### HIGH PRIORITY (Strengthens Thesis Significantly)

| # | Task | Effort | Relevant Guide |
| --- | --- | --- | --- |
| 4 | **Deploy Prometheus + Grafana locally, capture screenshots** | 2–3 hours | [config/prometheus.yml](../config/prometheus.yml), [config/grafana/har_dashboard.json](../config/grafana/har_dashboard.json), [docker-compose.yml](../docker-compose.yml) |
| 5 | **Run full pipeline on all 26 Decoded sessions, collect MLflow results** | 2–3 hours | [docs/DATA_INGESTION_AND_INFERENCE_GUIDE.md](DATA_INGESTION_AND_INFERENCE_GUIDE.md) — Section 3: batch processing |
| 6 | **GitHub Actions — verify CI passes on push** | 1 hour | [docs/GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) — Section 10: push and trigger |
| 7 | **Repository cleanup — separate research from production** | 2–3 hours | This document — Section 7: Project Cleanup Plan |
| 8 | **Resolve merge conflicts in `config/requirements.txt`** | 15 min | Delete conflicting sections; `pyproject.toml` is the source of truth |

#### MEDIUM PRIORITY (Nice to Have)

| # | Task | Effort | Relevant Guide |
| --- | --- | --- | --- |
| 9 | **Add automated hyperparameter search (Optuna)** | 1 day | Not yet documented — add to `src/train.py` |
| 10 | **GitHub Actions self-hosted runner setup** | 2–3 hours | [docs/GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) — Section 13: customising |
| 11 | **CI/CD: Add pipeline smoke test job** | 1 hour | [docs/GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) — Section 13: add pipeline run |
| 12 | **Model validation job in CI (replace placeholder)** | 1 hour | [.github/workflows/ci-cd.yml](../.github/workflows/ci-cd.yml) — line 230: uncomment DVC pull + validation |
| 13 | **File watcher for auto-ingestion** | 1 hour | [docs/DATA_INGESTION_AND_INFERENCE_GUIDE.md](DATA_INGESTION_AND_INFERENCE_GUIDE.md) — Section 4: watchdog script |
| 14 | **Implement processed-session registry (JSON)** | 2 hours | This document — Section 1.4: artifact registry |

---

### 9.4 Documentation File Map

Every pipeline topic is covered by at least one documentation file. This cross-reference shows where to find guidance for each area:

| Topic | Primary Document | Supporting Files |
| --- | --- | --- |
| **Data ingestion and new data** | [DATA_INGESTION_AND_INFERENCE_GUIDE.md](DATA_INGESTION_AND_INFERENCE_GUIDE.md) | This document §1–2 |
| **CI/CD and GitHub Actions** | [GITHUB_ACTIONS_CICD_GUIDE.md](GITHUB_ACTIONS_CICD_GUIDE.md) | [.github/workflows/ci-cd.yml](../.github/workflows/ci-cd.yml) |
| **Pipeline architecture (14 stages)** | This document (§3–5) | [PIPELINE_DEEP_DIVE_opus.md](PIPELINE_DEEP_DIVE_opus.md), [PIPELINE_STAGE_PROGRESS_DASHBOARD.md](PIPELINE_STAGE_PROGRESS_DASHBOARD.md) |
| **Calibration and uncertainty** | This document (§4) | [docs/stages/07_EVALUATION_METRICS.md](stages/07_EVALUATION_METRICS.md) |
| **Pseudo-labeling and adaptation** | This document (§5) | [docs/thesis/UNLABELED_EVALUATION.md](thesis/UNLABELED_EVALUATION.md), [docs/thesis/FINE_TUNING_STRATEGY.md](thesis/FINE_TUNING_STRATEGY.md) |
| **Model versioning and rollback** | This document (§6) | [docs/stages/09_DEPLOYMENT_AUDIT.md](stages/09_DEPLOYMENT_AUDIT.md) |
| **Monitoring and drift detection** | [docs/stages/06_MONITORING_DRIFT.md](stages/06_MONITORING_DRIFT.md) | [config/prometheus.yml](../config/prometheus.yml), [config/alerts/har_alerts.yml](../config/alerts/har_alerts.yml) |
| **Environment setup** | This document (§8) | [pyproject.toml](../pyproject.toml), [config/requirements.txt](../config/requirements.txt) |
| **Docker deployment** | [docker-compose.yml](../docker-compose.yml) | [docker/Dockerfile.inference](../docker/Dockerfile.inference), [docker/api/main.py](../docker/api/main.py) |
| **Research papers and bibliography** | [APPENDIX_PAPER_INDEX.md](APPENDIX_PAPER_INDEX.md) | [Bibliography_From_Local_PDFs.md](Bibliography_From_Local_PDFs.md), [docs/research/](research/) |
| **Mentor Q&A preparation** | [MENTOR_QA_SIMPLE_WITH_PAPERS.md](MENTOR_QA_SIMPLE_WITH_PAPERS.md) | [HAR_MLOps_QnA_With_Papers.md](HAR_MLOps_QnA_With_Papers.md) |
| **Thesis structure and writing** | [docs/thesis/THESIS_STRUCTURE_OUTLINE.md](thesis/THESIS_STRUCTURE_OUTLINE.md) | [docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md](thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md) |
| **Previous progress tracking** | [THESIS_MASTER_PROGRESS_2026-01-31.md](THESIS_MASTER_PROGRESS_2026-01-31.md) | [PIPELINE_STAGE_PROGRESS_DASHBOARD.md](PIPELINE_STAGE_PROGRESS_DASHBOARD.md) |

---

### 9.5 Recommended 14-Week Plan (Feb 12 → May 20)

| Weeks | Focus | Deliverables |
| --- | --- | --- |
| **Week 1–2** (Feb 12–25) | Pipeline execution and results | Run all 26 sessions; deploy Prometheus/Grafana; capture MLflow screenshots; resolve merge conflicts |
| **Week 3–4** (Feb 26 – Mar 11) | Thesis Ch 1–2 | Introduction (problem, objectives, scope); Literature Review (HAR, MLOps, domain adaptation, UQ) |
| **Week 5–6** (Mar 12–25) | Thesis Ch 3 | Methodology (pipeline design, 14 stages, data flow, architecture diagrams) |
| **Week 7–8** (Mar 26 – Apr 8) | Thesis Ch 4 | Implementation (code walkthrough, config, deployment, CI/CD, Docker) |
| **Week 9–10** (Apr 9–22) | Thesis Ch 5 | Results (MLflow metrics, monitoring reports, calibration results, drift analysis, robustness tests) |
| **Week 11–12** (Apr 23 – May 6) | Thesis Ch 6 + cleanup | Discussion, future work, limitations; repository cleanup; finalise all figures |
| **Week 13–14** (May 7–20) | Review and submission | Proofread, format, final presentation slides, submit |

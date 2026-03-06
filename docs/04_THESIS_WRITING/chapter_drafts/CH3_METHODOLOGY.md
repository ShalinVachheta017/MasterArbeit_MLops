# Chapter 3 — Methodology

---

## 3.1 System Overview

The proposed system is a stage-based production pipeline that transforms raw wearable sensor recordings into activity predictions and continuously monitors, evaluates, and adapts the underlying model. The pipeline comprises 14 stages, logically partitioned into three groups that can be executed independently or in combination.

**Stage Groups**

| Group | Stages | Purpose | Typical Trigger |
|-------|--------|---------|-----------------|
| Inference Cycle | 1 – 7 | Ingest data, predict activities, monitor quality | Every new data batch |
| Retraining Cycle | 8 – 10 | Retrain, register, and update baselines | Triggered by monitoring |
| Advanced Analytics | 11 – 14 | Calibrate, detect drift, pseudo-label, adapt | After model retraining or at deployment |

This decomposition follows the **composable pipeline** design principle: each stage is an independently executable unit with a typed configuration input and a typed artifact output. The orchestrator composes stages at runtime according to command-line arguments. Default execution runs stages 1–7 only; retraining and advanced analytics are enabled via explicit flags, ensuring that the costliest computations occur only when needed.

> **What is new.** Existing HAR systems in the literature typically implement training and inference as monolithic scripts. This pipeline introduces a stage-separated architecture with fallback artifacts, allowing any subset of stages to execute while downstream stages resolve their inputs from the most recent upstream artifacts. This design enables incremental deployment: a team can begin with inference-only operation and progressively enable monitoring, triggering, and adaptation without modifying existing stages.

### 3.1.1 Stage Enumeration

The complete stage list with input/output types is as follows:

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | Data Ingestion | Raw Excel or CSV files | Fused, resampled CSV (50 Hz) |
| 2 | Data Validation | Fused CSV | Validation report (pass/fail + diagnostics) |
| 3 | Data Transformation | Validated CSV | Windowed NumPy arrays (.npy) |
| 4 | Model Inference | .npy arrays + Keras model | Prediction CSV + probability arrays |
| 5 | Model Evaluation | Predictions + probabilities | Confidence distribution, ECE report |
| 6 | Post-Inference Monitoring | Predictions (unlabelled) | Three-layer monitoring report |
| 7 | Trigger Evaluation | Monitoring report | Retrain decision (boolean + reason) |
| 8 | Model Retraining | Training data + current model | Retrained model checkpoint |
| 9 | Model Registration | Retrained model + proxy metrics | Versioned model in registry |
| 10 | Baseline Update | New model + reference data | Updated drift baselines |
| 11 | Calibration & UQ | Validation logits | Temperature parameter, ECE, uncertainty estimates |
| 12 | Wasserstein Drift | Production features + baselines | Per-channel drift scores, change-point flags |
| 13 | Curriculum Pseudo-Labeling | Model + unlabelled data | Fine-tuned model via self-training |
| 14 | Sensor Placement | Accelerometer + gyroscope features | Hand-side classification, mirrored features |

### 3.1.2 Architecture Diagram

[FIGURE: pipeline_architecture — 14-stage pipeline with three groups, data flow arrows, and artifact storage]

```
                        ┌──────────────────────────────────────────────────────┐
                        │              I N F E R E N C E   C Y C L E          │
  Raw Sensor Files ───► │  1.Ingest → 2.Validate → 3.Transform → 4.Infer     │
                        │       → 5.Evaluate → 6.Monitor → 7.Trigger         │
                        └────────────────────────┬─────────────────────────────┘
                                                 │
                                    trigger = True?
                                                 │
                        ┌────────────────────────▼─────────────────────────────┐
                        │           R E T R A I N I N G   C Y C L E           │
                        │  8.Retrain → 9.Register → 10.Update Baselines       │
                        └────────────────────────┬─────────────────────────────┘
                                                 │
                                    calibrate?
                                                 │
                        ┌────────────────────────▼─────────────────────────────┐
                        │        A D V A N C E D   A N A L Y T I C S          │
                        │  11.Calibrate → 12.Wasserstein → 13.Pseudo-Label    │
                        │       → 14.Sensor Placement                         │
                        └─────────────────────────────────────────────────────┘

  Artifacts flow downward.  Each stage reads from the artifact registry and
  writes its output artifact.  Missing upstream artifacts are resolved via
  fallback to the most recent successful output.
```

## 3.2 Data Flow and Artifact Registry

### 3.2.1 Ingestion Paths

The pipeline supports two raw data formats, reflecting the evolution of the data collection methodology:

1. **Garmin Connect Excel archive.** A pair of Excel files — one for heart rate/stress metrics, one for accelerometer/gyroscope samples — exported from the Garmin Connect web interface. The ingestion stage parses these files, aligns timestamps, performs sensor fusion (accelerometer + gyroscope into a unified 6-channel array), and resamples to a fixed 50 Hz sampling rate.

2. **Decoded per-session CSV triplets.** Later recordings were exported in the Garmin Decoded format, producing three CSV files per session: `{timestamp}_accelerometer.csv`, `{timestamp}_gyroscope.csv`, and `{timestamp}_record.csv`. The ingestion stage detects this format by filename pattern, merges the accelerometer and gyroscope channels on their nearest timestamps, and produces the same fused CSV output as path (1).

Both paths converge on a single intermediate representation: a fused CSV file with columns `[timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]` sampled at 50 Hz. This idempotency — running ingestion twice on the same raw input produces identical output — is a prerequisite for pipeline reproducibility.

### 3.2.2 Artifact Registry Concept

Each stage writes its output to a timestamped artifact directory:

```
artifacts/
└── 20260212_143022/
    ├── ingestion/
    │   └── sensor_fused_50Hz.csv
    ├── validation/
    │   └── validation_report.json
    ├── transformation/
    │   ├── production_X.npy
    │   └── scaler_params.json
    ├── inference/
    │   ├── predictions.csv
    │   └── probabilities.npy
    └── ...
```

The orchestrator maintains a registry of the most recent artifact per stage. When a stage is skipped (e.g., `--skip-ingestion`), the pipeline creates a **fallback artifact** that points to the latest successful output. This mechanism decouples stage execution from stage ordering: a researcher can re-run only the evaluation stage without re-processing raw data.

[FIGURE: artifact_flow — Diagram showing artifact registry with fallback pointers]

*TODO: Create a visual diagram showing the artifact registry with fallback resolution arrows.*

## 3.3 Monitoring Signals and Trigger Policy

### 3.3.1 The Three-Layer Monitoring Framework

In a production deployment, ground-truth labels are unavailable. The monitoring framework therefore operates entirely on unlabelled model outputs, extracting three categories of proxy signal:

**Layer 1 — Confidence Analysis.** The softmax output of the classifier provides a per-sample confidence score. The monitoring stage computes:

- Mean confidence across the batch
- Fraction of low-confidence predictions (below a configurable threshold, default 0.6)
- Entropy of the confidence distribution

A decline in mean confidence or an increase in the low-confidence fraction relative to the calibration baseline indicates that the model is encountering data it was not trained to handle.

**Layer 2 — Temporal Consistency.** Activity predictions should exhibit temporal coherence: a person does not switch between walking and sitting every second. The monitoring stage computes:

- **Flip rate:** The fraction of consecutive prediction pairs that differ. An abnormally high flip rate suggests the model is oscillating between classes in ambiguous regions of the feature space.
- **Mean dwell time:** The average number of consecutive windows assigned to the same class. A dwell time significantly shorter than the baseline indicates reduced prediction stability.

**Layer 3 — Statistical Drift Detection.** The monitoring stage compares the current batch's feature distributions against calibration baselines using two statistical tests:

- **Population Stability Index (PSI):** Measures the divergence between two discrete distributions. PSI values above 0.2 indicate significant drift.
- **Kolmogorov–Smirnov (KS) test:** A non-parametric test for whether two samples originate from the same distribution. A per-channel KS p-value below 0.05 flags that channel as drifted.

### 3.3.2 Trigger Policy Engine

The trigger evaluation stage (Stage 7) aggregates the three monitoring layers into a retraining decision. A naive approach — triggering on any single anomalous metric — produces excessive false positives, as individual metrics fluctuate with natural variation. The implemented policy uses a **two-of-three voting scheme:** retraining is triggered only when at least two of the three monitoring layers independently flag the batch as degraded.

Additionally, the engine implements:

- **Tiered alerting:** Metrics are classified as INFO (within normal range), WARNING (approaching threshold), or CRITICAL (threshold exceeded). Only CRITICAL signals contribute to the vote.
- **Cooldown period:** After a retraining trigger fires, all subsequent triggers are suppressed for a configurable duration (default: 24 hours). This prevents cascading retrains when a single distribution shift causes multiple batches to fail.
- **Audit logging:** Every trigger decision — including the monitoring inputs, vote tally, and outcome — is logged to MLflow as a structured JSON artifact for retrospective analysis.

[FIGURE: trigger_policy_flowchart — Decision tree showing 2-of-3 voting with cooldown]

*TODO: Create flowchart showing the trigger decision path with tiered alerting and cooldown logic.*

## 3.4 Adaptation Methods

When the trigger policy determines that adaptation is necessary, the pipeline supports three strategies, selectable via a command-line argument. Each strategy addresses a different type and magnitude of distribution shift.

### 3.4.1 Adaptive Batch Normalisation (AdaBN)

For mild domain shifts — such as the same user on a different day, or minor sensor placement variation — Adaptive Batch Normalisation (AdaBN) provides a lightweight adaptation method that requires no gradient descent and no labels. The procedure is:

1. Freeze all model weights.
2. Pass $N$ batches of production data through the network.
3. Allow the running mean $\mu$ and variance $\sigma^2$ of each Batch Normalisation layer to update.
4. The adapted model reflects the production data's first- and second-order statistics.

Formally, for each Batch Normalisation layer $l$ with parameters $(\gamma_l, \beta_l)$, the normalisation changes from the training statistics $(\mu_l^{\text{train}}, \sigma_l^{\text{train}})$ to production statistics $(\mu_l^{\text{prod}}, \sigma_l^{\text{prod}})$:

$$\hat{x}_l = \gamma_l \cdot \frac{x_l - \mu_l^{\text{prod}}}{\sqrt{(\sigma_l^{\text{prod}})^2 + \epsilon}} + \beta_l$$

This is the fastest adaptation path, completing in seconds rather than minutes, and is recommended as the first intervention when drift is detected.

### 3.4.2 Curriculum Pseudo-Labeling with EWC

For more substantial distribution shifts — such as a new user or a change in sensor hardware — AdaBN may be insufficient. The curriculum pseudo-labeling strategy (Stage 13) implements a self-training loop that fine-tunes the model on unlabelled production data:

1. **Predict.** The current model generates predictions on the unlabelled production batch, along with per-sample confidence scores.
2. **Threshold.** Only samples whose confidence exceeds a threshold $\tau_i$ are selected as pseudo-labelled training examples. The threshold decays linearly across iterations from $\tau_{\text{start}} = 0.95$ to $\tau_{\text{end}} = 0.80$, implementing a curriculum that begins with easy (high-confidence) samples and progressively incorporates harder ones.
3. **Balance.** To prevent class collapse, a maximum of $K$ pseudo-labelled samples are drawn per class per iteration.
4. **Combine.** The pseudo-labelled samples are combined with the original labelled source data.
5. **Fine-tune.** The model is fine-tuned on the combined dataset with an EWC penalty term.
6. **Update teacher.** An exponential moving average (EMA) of the student model parameters serves as the teacher for the next iteration.

The EWC regularisation term prevents catastrophic forgetting of the original training distribution:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ is the diagonal of the Fisher Information Matrix, $\theta^*$ are the pre-trained parameters, and $\lambda$ (default: 1000) controls the regularisation strength.

### 3.4.3 Safeguards Against Adaptation Failure

Unsupervised adaptation can degenerate if the pseudo-labels are systematically wrong (confirmation bias). The pipeline implements three safeguards:

1. **Entropy monitoring.** If the mean entropy of pseudo-label predictions increases across iterations (instead of decreasing), the training loop is terminated early. Increasing entropy indicates that the model is becoming less certain, a sign of degenerate pseudo-labels.
2. **Class diversity gate.** If fewer than a configurable minimum number of classes (default: 3) are represented in the pseudo-labelled batch, the iteration is skipped. This prevents mode collapse into a small number of dominant classes.
3. **Proxy validation.** After adaptation, the registration stage (Stage 9) compares the adapted model's confidence distribution and Expected Calibration Error against the current production model. If the adapted model is worse on these proxy metrics, deployment is blocked and the operator is notified.

### 3.4.4 Sensor Placement Adaptation

A specific source of domain shift in wrist-worn HAR is the difference between dominant-hand and non-dominant-hand placement. The sensor placement module (Stage 14) addresses this through:

- **Automatic hand-side detection** based on statistical features of the gyroscope signal (dominant-hand movements typically exhibit higher angular velocity variance).
- **Axis mirroring augmentation** that flips the Y and Z axes of both accelerometer and gyroscope channels (indices [1, 2, 4, 5]), simulating the effect of wearing the sensor on the opposite wrist.

This adaptation is applied as a data-level transformation before inference, requiring no model modification.

[FIGURE: adaptation_strategies_comparison — Table/diagram comparing AdaBN, curriculum pseudo-labeling, and axis mirroring on scope, compute cost, and label requirements]

*TODO: Create comparison figure showing the three adaptation strategies side-by-side with pros/cons.*

---

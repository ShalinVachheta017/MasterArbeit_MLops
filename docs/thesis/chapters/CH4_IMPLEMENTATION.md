# Chapter 4 — Implementation

---

## 4.1 Repository Structure and Module Mapping

The implementation is organised as an installable Python package rooted in the `src/` directory. The repository follows a layered architecture that separates concerns into four tiers: entities, components, core modules, and the pipeline orchestrator.

```
MasterArbeit_MLops/
├── run_pipeline.py              ← Single entry point (CLI)
├── pyproject.toml               ← Package definition, dependencies, tool config
├── src/
│   ├── entity/
│   │   ├── config_entity.py     ← 14 typed configuration dataclasses
│   │   └── artifact_entity.py   ← 14 typed artifact dataclasses + PipelineResult
│   ├── components/
│   │   ├── data_ingestion.py    ← Stage 1: Excel/CSV → fused CSV
│   │   ├── data_validation.py   ← Stage 2: schema + range checks
│   │   ├── data_transformation.py ← Stage 3: CSV → windowed .npy
│   │   ├── model_inference.py   ← Stage 4: .npy → predictions
│   │   ├── model_evaluation.py  ← Stage 5: confidence/ECE analysis
│   │   ├── post_inference_monitoring.py ← Stage 6: 3-layer monitoring
│   │   ├── trigger_evaluation.py ← Stage 7: retraining decision
│   │   ├── model_retraining.py  ← Stage 8: standard/AdaBN/pseudo-label
│   │   ├── model_registration.py ← Stage 9: version, deploy, rollback
│   │   ├── baseline_update.py   ← Stage 10: rebuild baselines
│   │   ├── calibration_uncertainty.py ← Stage 11: temp scaling, MC Dropout
│   │   ├── wasserstein_drift.py ← Stage 12: distribution drift
│   │   ├── curriculum_pseudo_labeling.py ← Stage 13: self-training
│   │   └── sensor_placement.py  ← Stage 14: hand detection + mirroring
│   ├── pipeline/
│   │   └── production_pipeline.py ← Orchestrator (470 lines)
│   ├── calibration.py           ← Core calibration logic (544 lines)
│   ├── wasserstein_drift.py     ← Core drift detection (453 lines)
│   ├── curriculum_pseudo_labeling.py ← Core self-training (466 lines)
│   ├── sensor_placement.py      ← Core hand detection (345 lines)
│   ├── robustness.py            ← Noise injection testing (449 lines)
│   ├── sensor_data_pipeline.py  ← Raw data processing (1,182 lines)
│   ├── train.py                 ← Model training (925 lines)
│   ├── mlflow_tracking.py       ← Experiment tracking (654 lines)
│   ├── trigger_policy.py        ← Trigger engine (812 lines)
│   ├── prometheus_metrics.py    ← Metrics exporter (623 lines)
│   ├── model_rollback.py        ← Version history (532 lines)
│   └── ...
├── tests/                       ← 18 test files + conftest.py
├── config/
│   ├── pipeline_config.yaml     ← Preprocessing parameters
│   ├── mlflow_config.yaml       ← Tracking URI, experiment name
│   ├── prometheus.yml           ← Scrape configuration
│   ├── grafana/                 ← Dashboard JSON definitions
│   └── alerts/                  ← Alerting rules
├── docker/
│   ├── Dockerfile.inference     ← Inference container (71 lines)
│   ├── Dockerfile.training      ← Training container (54 lines)
│   └── api/
│       └── main.py              ← FastAPI server (447 lines)
├── docker-compose.yml           ← Multi-container orchestration (143 lines)
└── .github/workflows/
    └── ci-cd.yml                ← CI/CD workflow (282 lines, 6 jobs)
```

> **What is new.** The two-tier architecture — thin *components* that delegate to heavyweight *core modules* — is a deliberate design choice. Components contain only stage-level orchestration logic (typically 80–140 lines each). All algorithmic complexity resides in the core modules, which can be tested, profiled, and reused independently of the pipeline. This separation was inspired by production ML pipeline patterns, adapted for the specific requirements of wearable sensor data.

### 4.1.1 Entity Layer

The entity layer defines the contract between stages using Python dataclasses. Each stage receives a configuration dataclass and returns an artifact dataclass.

**Configuration dataclasses** (`config_entity.py`, 344 lines) encode all parameters a stage might need — file paths, thresholds, hyperparameters — with sensible defaults. A master `PipelineConfig` dataclass holds project-wide settings (root directory, timestamp, artifact directory) from which stage-specific configs derive their paths.

**Artifact dataclasses** (`artifact_entity.py`, 217 lines) are the typed outputs of each stage. A `PipelineResult` aggregator collects all artifacts from a pipeline run, enabling the orchestrator to pass results between stages and serialise the full execution record to JSON.

This typed contract eliminates the class of bug where a stage silently receives an incorrect file path or a missing configuration value. If a required field is not provided, the dataclass constructor raises an error before the stage begins execution.

### 4.1.2 Component Layer

Each of the 14 pipeline stages is implemented as a component class in `src/components/`. Components follow a uniform interface:

```python
class DataIngestionComponent:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> DataIngestionArtifact:
        # delegate to core module
        ...
        return DataIngestionArtifact(output_path=output_path)
```

Components are responsible for:
- Reading configuration
- Invoking the appropriate core module
- Wrapping the result in a typed artifact
- Logging entry/exit messages

Components are *not* responsible for algorithmic logic. A component that performs feature extraction, for example, delegates to `sensor_data_pipeline.py` for the actual computation. This separation ensures that core algorithms can be unit-tested without instantiating the pipeline.

## 4.2 Pipeline Runner and Stage Selection

### 4.2.1 CLI Design

The pipeline is invoked through a single entry point, `run_pipeline.py` (296 lines), which parses 20 command-line arguments and constructs the appropriate configuration objects. The argument parser is organised into four groups:

| Group | Arguments | Purpose |
|-------|-----------|---------|
| Stage Selection | `--stages`, `--skip-ingestion`, `--skip-validation`, `--retrain`, `--advanced` | Control which stages execute |
| Input Overrides | `--input-csv`, `--model`, `--gravity-removal`, `--calibrate` | Override default data/model paths |
| Retraining | `--adapt`, `--labels`, `--epochs`, `--auto-deploy` | Configure adaptation method |
| Advanced Analytics | `--mc-dropout-passes`, `--curriculum-iterations`, `--ewc-lambda` | Tune calibration/pseudo-labeling |

The default invocation (`python run_pipeline.py` with no arguments) executes stages 1–7, the inference cycle. This is the most common usage pattern: ingest new data, produce predictions, evaluate model health.

### 4.2.2 Stage Orchestration

The `ProductionPipeline` class (470 lines) implements the orchestration logic. On initialisation, it receives all 14 stage configurations. On execution, it iterates through the requested stages in order, handling three concerns:

1. **Fallback artifacts.** If a stage is skipped, the orchestrator locates the most recent artifact from a prior run and creates a fallback artifact object pointing to it. Downstream stages consume this fallback transparently.

2. **Error isolation.** When `--continue-on-failure` is set, a stage failure is logged and the pipeline advances to the next stage. The failure is recorded in the `PipelineResult` with its traceback.

3. **MLflow integration.** The orchestrator wraps the entire pipeline run in an MLflow run context, logging stage durations, success/failure status, and configuration parameters. Individual stages may create nested runs for finer-grained tracking.

### 4.2.3 Execution Modes

The CLI enables three execution modes through flag composition:

```bash
# Mode 1: Inference (default) — stages 1–7
python run_pipeline.py

# Mode 2: Inference + Retraining — stages 1–10
python run_pipeline.py --retrain

# Mode 3: Full pipeline — stages 1–14
python run_pipeline.py --retrain --advanced

# Mode 4: Targeted — any subset
python run_pipeline.py --stages calibration wasserstein_drift
```

The `--stages` argument accepts an arbitrary subset of stage names. The orchestrator validates that each name is in the allowed set and executes them in their canonical order, regardless of the order specified on the command line.

## 4.3 CI/CD Overview

### 4.3.1 GitHub Actions Workflow

The project uses a single GitHub Actions workflow file (`.github/workflows/ci-cd.yml`, 282 lines) that defines six jobs triggered on pushes to `main` or `develop` and on pull requests:

| Job | Purpose | Dependencies |
|-----|---------|-------------|
| `lint` | Runs `flake8` for code quality | None |
| `test` | Executes `pytest` with coverage | lint |
| `docker-build` | Builds inference Docker image | test |
| `integration-test` | Starts container, runs health check + smoke test | docker-build |
| `model-validation` | Validates model metrics against thresholds | test |
| `notify` | Posts status to Slack/email | All preceding jobs |

The workflow uses path filtering: only changes to `src/`, `tests/`, `docker/`, `config/`, or workflow files trigger the pipeline. This prevents documentation-only commits from consuming CI minutes.

### 4.3.2 Docker Containerisation

Two Dockerfiles serve distinct purposes:

- **`Dockerfile.inference`** (71 lines): Builds a lightweight image containing the trained model, the FastAPI serving application, and production dependencies only. The image exposes port 8000 and starts the Uvicorn ASGI server.

- **`Dockerfile.training`** (54 lines): Builds a heavier image with training dependencies (full TensorFlow, development tools). Used for reproducible retraining in controlled environments.

The `docker-compose.yml` (143 lines) orchestrates three services:

```yaml
services:
  mlflow:        # Experiment tracking server
  inference-api: # FastAPI prediction endpoint
  training:      # On-demand retraining container
```

### 4.3.3 FastAPI Inference API

The serving layer is implemented as a FastAPI application (`docker/api/main.py`, 447 lines) with three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe for orchestrators |
| `/model/info` | GET | Returns model version, architecture, and training metadata |
| `/predict` | POST | Accepts sensor data, returns activity predictions |

The `/predict` endpoint accepts a JSON payload containing a windowed sensor array, passes it through the loaded Keras model, and returns class probabilities and the predicted label. The endpoint also logs inference latency and prediction distribution to Prometheus metrics.

## 4.4 Logging, Experiment Tracking, and Reproducibility

### 4.4.1 MLflow Integration

MLflow serves as the experiment tracking backbone. The integration (`mlflow_tracking.py`, 654 lines) provides:

- **Experiment management.** All pipeline runs are logged under a single experiment (`har-production-pipeline`). Each run records the CLI arguments, stage configuration, and execution timestamp.
- **Metric logging.** Per-stage metrics (e.g., mean confidence, drift scores, calibration ECE) are logged as MLflow metrics, enabling comparison across runs.
- **Artifact logging.** Key outputs — predictions, monitoring reports, model checkpoints — are logged as MLflow artifacts, providing a browsable history through the MLflow UI.
- **Model registry.** Trained models are registered in the MLflow Model Registry with automatic version incrementing. The registry supports stage transitions (`None` → `Staging` → `Production` → `Archived`) for deployment governance.

The tracking URI defaults to a local `mlruns/` directory but can be reconfigured via `config/mlflow_config.yaml` to point to a remote tracking server.

### 4.4.2 DVC Data Versioning

Raw sensor data and trained model files are excluded from Git (via `.gitignore`) and tracked by DVC (Data Version Control). Four DVC pointer files maintain the version chain:

```
data/raw.dvc           ← Points to raw sensor recordings
data/processed.dvc     ← Points to fused, resampled CSVs
data/prepared.dvc      ← Points to windowed NumPy arrays
models/pretrained.dvc  ← Points to the current production model
```

This ensures that any historical pipeline run can be reproduced: the Git commit records the code state, and the DVC pointer records the data state. Together, they form a complete provenance chain.

### 4.4.3 Deterministic Artifact Naming

Every pipeline execution generates a timestamped artifact directory:

```
artifacts/YYYYMMDD_HHMMSS/
```

Within this directory, each stage writes to a dedicated subdirectory. This convention guarantees that artefacts from different runs never collide and that the chronological order of executions is immediately apparent from the directory listing.

The pipeline result — a JSON document summarising all stage outcomes — is additionally written to:

```
logs/pipeline/pipeline_result_YYYYMMDD_HHMMSS.json
```

This file records the full execution metadata: which stages ran, their durations, success/failure status, error tracebacks (if any), and all artifact paths. It serves as the primary audit record for each pipeline invocation.

### 4.4.4 Monitoring Infrastructure

The monitoring infrastructure comprises three components configured declaratively:

- **Prometheus** (`config/prometheus.yml`, 70 lines): Scrapes the inference API endpoint and the metrics exporter at configurable intervals. Metrics include prediction counts per class, inference latency histograms, and model confidence distributions.
- **Grafana** (`config/grafana/har_dashboard.json`): A pre-built dashboard visualising prediction distributions, drift scores, confidence trends, and alert status over time.
- **Alert rules** (`config/alerts/har_alerts.yml`, 191 lines): Define thresholds for model performance degradation, excessive drift, and inference latency spikes. Alerts are routed to the configured notification channels.

[FIGURE: monitoring_stack — Prometheus + Grafana architecture with scrape targets and dashboard panels]

*TODO: Capture Grafana dashboard screenshots after deploying the monitoring stack and include them as figures.*

### 4.4.5 Test Suite

The test suite comprises 18 test files and a shared `conftest.py`, totalling over 1,000 test assertions. Tests are organised by module:

| Test File | Target | Markers |
|-----------|--------|---------|
| `test_calibration.py` | Temperature scaling, ECE computation | `calibration` |
| `test_wasserstein_drift.py` | Wasserstein distance, change-point detection | `unit` |
| `test_curriculum_pseudo_labeling.py` | Self-training loop, EWC penalty | `unit` |
| `test_robustness.py` | Noise injection, missing data | `robustness` |
| `test_sensor_placement.py` | Hand detection, axis mirroring | `unit` |
| `test_pipeline_integration.py` | End-to-end pipeline execution | `integration` |
| ... | ... | ... |

Tests are executed via `pytest` with configuration in `pytest.ini` and `pyproject.toml`. The CI workflow runs the full suite with coverage reporting on every push. Markers allow selective execution (e.g., `pytest -m "calibration"` runs only calibration tests).

[FIGURE: test_coverage_report — HTML coverage report showing per-module line coverage]

*TODO: Generate and include a coverage report screenshot after running `pytest --cov=src --cov-report=html`.*

## 4.5 Session-Based File Detection and Ingestion

### 4.5.1 Raw Data Layout

Raw sensor recordings reside in `data/raw/`. Each recording session produces three CSV files that share a common timestamp prefix:

```
2025-07-16-21-03-13_accelerometer.csv
2025-07-16-21-03-13_gyroscope.csv
2025-07-16-21-03-13_record.csv
```

The timestamp prefix (`YYYY-MM-DD-HH-MM-SS`) serves as the **session identifier**. Files belonging to the same session are grouped by matching this prefix. The dataset currently contains 26 such sessions (78 CSV files), plus one legacy Excel pair and the composite `all_users_data_labeled.csv`.

### 4.5.2 File Discovery and Matching

The `find_latest_sensor_pair()` function in `sensor_data_pipeline.py` discovers sensor files by globbing for `*accelerometer*.*` and `*gyroscope*.*` patterns. It sorts results by file modification time and matches accelerometer and gyroscope files by comparing the filename prefix before the keyword `accelerometer` or `gyroscope`. This ensures that files from the same recording session are paired correctly even when multiple sessions exist in the directory.

The current implementation returns only the **newest** pair, optimised for the single-session inference use case. A planned extension (`discover_all_sessions()`) will return all session groups and compare them against a session registry to identify unprocessed sessions.

### 4.5.3 Already-Processed Detection (Planned)

To prevent redundant processing, the system will maintain a session registry at `data/processed/session_registry.json`. This registry maps each session ID to its processing timestamp and output path:

```json
{
  "2025-07-16-21-03-13": {
    "processed_at": "2026-02-15T14:30:00",
    "output": "data/processed/sensor_fused_50Hz_2025-07-16-21-03-13.csv"
  }
}
```

Before processing a session, the ingestion component checks the registry. If the session ID is already present and the output file still exists, the session is skipped. This design supports incremental ingestion: when new recordings are added to `data/raw/`, only the new sessions are processed.

## 4.6 Stage Group Rationale

### 4.6.1 Why Three Groups?

The 14 pipeline stages are partitioned into three execution groups based on **frequency of use** and **computational cost**:

| Group | Stages | Flag | Frequency | Typical Duration |
|-------|--------|------|-----------|-----------------|
| Inference cycle | 1–7 | *(default)* | Every new recording | ~2 minutes (CPU) |
| Retraining cycle | 8–10 | `--retrain` | Only when trigger fires | ~10–30 minutes |
| Advanced calibration | 11–14 | `--advanced` | After retraining or at deployment | ~5–15 minutes |

**Stages 1–7 (Inference)** constitute the standard operational loop: ingest data, validate, transform, run inference, evaluate model health, monitor for drift, and decide whether retraining is necessary. This group runs on every new data arrival and is designed to be lightweight.

**Stages 8–10 (Retraining)** execute only when Stage 7 determines that the model has degraded beyond an acceptable threshold. This group is computationally expensive (it involves model training) and is gated behind the `--retrain` flag to prevent accidental triggering. It also supports three adaptation methods via `--adapt {adabn, pseudo_label, none}`.

**Stages 11–14 (Advanced)** perform calibration, drift baseline establishment, self-training refinement, and sensor placement analysis. These stages are calibration and analysis operations that establish the thresholds and baselines used by stages 1–7 in subsequent inference cycles. They should be run once after retraining to recalibrate the monitoring thresholds.

### 4.6.2 Combining Groups

All three groups can be executed in a single invocation:

```bash
python run_pipeline.py --retrain --advanced --continue-on-failure
```

This runs all 14 stages in canonical order. However, in production operation, sequential execution is preferred: run stages 1–7, inspect the trigger decision, and only proceed to stages 8–10 if the trigger fires. This avoids unnecessary retraining on every data arrival.

## 4.7 Domain Adaptation Methods

### 4.7.1 Adaptive Batch Normalisation (AdaBN)

When model degradation is caused by a mild distribution shift — for example, the same user wearing the sensor on a slightly different wrist position, or data collected on a different day with different baseline movement patterns — AdaBN provides the lightest possible adaptation. The method freezes all model weights and updates only the running mean and variance statistics in the BatchNormalisation layers by performing a forward pass over the new production data.

Key characteristics:
- **No labels required.** AdaBN uses only unlabeled production data.
- **No gradient descent.** Only first-order statistics are updated.
- **Execution time.** Seconds, not minutes. Suitable for real-time adaptation.
- **Limitation.** Cannot compensate for large distribution shifts (e.g., entirely new user with different movement patterns). In such cases, pseudo-labeling should be used.

### 4.7.2 Curriculum Pseudo-Labeling with EWC

For larger distribution shifts where AdaBN is insufficient, the system employs curriculum pseudo-labeling. This method uses the model's own high-confidence predictions as surrogate labels for fine-tuning:

1. **Confidence thresholding.** Only predictions above a confidence threshold (starting at 0.95) are accepted as pseudo-labels.
2. **Curriculum schedule.** The threshold is gradually lowered across iterations (0.95 → 0.90 → 0.85 → 0.80), introducing harder examples progressively.
3. **EWC regularisation.** Elastic Weight Consolidation adds a penalty term that constrains weight updates to preserve knowledge from the original training distribution. This prevents catastrophic forgetting.

Three safeguards prevent degenerate self-training:
- **Entropy monitoring:** If the model becomes more uncertain across iterations, training is halted.
- **Class diversity check:** If pseudo-labels cover fewer than three activity classes, the iteration is skipped.
- **Proxy validation:** The adapted model must outperform the current model on proxy metrics before deployment is permitted.

### 4.7.3 Adaptation Selection

The choice of adaptation method is configured via the `--adapt` flag:

```bash
# Light adaptation (seconds, no labels)
python run_pipeline.py --retrain --adapt adabn

# Heavy adaptation (minutes, no labels, self-training)
python run_pipeline.py --retrain --adapt pseudo_label --curriculum-iterations 10 --ewc-lambda 500

# Standard retraining (requires labels)
python run_pipeline.py --retrain --adapt none --labels data/new_labels.csv
```

The decision of which method to use should follow a hierarchy: try AdaBN first; if performance does not recover, escalate to pseudo-labeling; if pseudo-labeling also fails (safeguards trigger), revert to supervised retraining with newly collected labels.

---

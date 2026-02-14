# Data Ingestion & Inference Guide

> **How to ingest new sensor data → run inference → view results in MLflow**

---

## Table of Contents

1. [Your Data Overview](#1-your-data-overview)
2. [Manual Ingestion — Single Recording](#2-manual-ingestion--single-recording)
3. [Manual Ingestion — Batch (All 26 Sessions)](#3-manual-ingestion--batch-all-26-sessions)
4. [Auto-Detection — Pipeline Finds New Data](#4-auto-detection--pipeline-finds-new-data)
5. [Viewing Results in MLflow](#5-viewing-results-in-mlflow)
6. [Advanced Pipeline Options](#6-advanced-pipeline-options)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Your Data Overview

### What's in `data/raw/Decoded/`

You have **26 recording sessions** (July–August 2025), each with 3 CSV files:

| File Pattern | Description | Example Columns |
|---|---|---|
| `{timestamp}_accelerometer.csv` | Raw accelerometer at ~50 Hz | `timestamp`, `timestamp_ms`, `sample_index`, `sample_time_offset`, `accel_x`, `accel_y`, `accel_z` |
| `{timestamp}_gyroscope.csv` | Raw gyroscope at ~50 Hz | `timestamp`, `timestamp_ms`, `sample_index`, `sample_time_offset`, `gyro_x`, `gyro_y`, `gyro_z` |
| `{timestamp}_record.csv` | Garmin FIT summary (1 Hz) | `accelerometer_x`, `gyroscope_x`, `heart_rate`, `cadence`, `distance`, etc. |

**Total: 78 files** (26 × 3)

### Existing Pipeline Data Flow

```
data/raw/            →  data/processed/           →  data/prepared/         →  outputs/
  Excel or CSV            sensor_fused_50Hz.csv        production_X.npy         predictions.csv
  (accel + gyro)          (fused & resampled)          (windowed, normalized)   (class labels)
```

**Pipeline stages that handle this:**
```
Stage 1: Ingestion      →  raw files → sensor_fused_50Hz.csv
Stage 2: Validation      →  schema + range checks
Stage 3: Transformation  →  CSV → production_X.npy (windowed, normalized)
Stage 4: Inference        →  predictions using pretrained model
Stage 5: Evaluation       →  confidence + distribution analysis
Stage 6: Monitoring       →  drift detection
Stage 7: Trigger          →  retraining decision
```

---

## 2. Manual Ingestion — Single Recording

### Option A: Use the Accelerometer CSV Directly

The accelerometer CSV files in `Decoded/` already have the right column format (`accel_x`, `accel_y`, `accel_z` with timestamps). You can feed a single file directly:

```bash
# From project root
python run_pipeline.py --input-csv "data/raw/Decoded/2025-07-16-21-03-13_accelerometer.csv"
```

> **Note:** This skips sensor fusion (gyroscope not included). The pipeline will process whatever columns it finds.

### Option B: Pre-Fuse Accel + Gyro, Then Ingest

For best results, fuse accelerometer + gyroscope first, then feed the combined CSV:

```bash
# Step 1: Fuse one session using the sensor_data_pipeline
python -c "
from pathlib import Path
from src.sensor_data_pipeline import (
    ProcessingConfig, LoggerSetup, SensorDataLoader,
    DataProcessor, SensorFusion, Resampler
)
import logging

cfg = ProcessingConfig(target_hz=50)
log = logging.getLogger('fusion')
logging.basicConfig(level=logging.INFO)

loader = SensorDataLoader(log)
proc   = DataProcessor(log)
fusion = SensorFusion(cfg, log)
resamp = Resampler(cfg, log)

# Load the Decoded CSVs (they use CSV format, not Excel)
import pandas as pd
accel = pd.read_csv('data/raw/Decoded/2025-07-16-21-03-13_accelerometer.csv')
gyro  = pd.read_csv('data/raw/Decoded/2025-07-16-21-03-13_gyroscope.csv')

# Rename columns to match expected format
accel = accel.rename(columns={'accel_x': 'x', 'accel_y': 'y', 'accel_z': 'z'})
gyro  = gyro.rename(columns={'gyro_x': 'x', 'gyro_y': 'y', 'gyro_z': 'z'})

# Process
accel = proc.process_sensor_data(accel, 'accelerometer')
gyro  = proc.process_sensor_data(gyro, 'gyroscope')
fused = fusion.merge_sensor_data(accel, gyro)
fused = resamp.resample_data(fused)
fused = resamp.add_timestamp_columns(fused)
fused.to_csv('data/processed/sensor_fused_50Hz.csv', index=False)
print(f'Fused: {len(fused)} rows, {list(fused.columns)}')
"

# Step 2: Run the full pipeline (skip ingestion since data is already fused)
python run_pipeline.py --skip-ingestion
```

### Option C: Point to Specific Fused CSV

If you already have a fused/processed CSV:

```bash
python run_pipeline.py --input-csv "data/processed/sensor_fused_50Hz.csv"
```

---

## 3. Manual Ingestion — Batch (All 26 Sessions)

### PowerShell Script — Process All Decoded Sessions

Create and run this script to batch-process all 26 sessions:

```powershell
# batch_process_decoded.ps1
# Processes all 26 recording sessions in data/raw/Decoded/

$decodedDir = "data\raw\Decoded"
$outputDir  = "outputs\batch_results"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Get unique session prefixes (e.g., "2025-07-16-21-03-13")
$sessions = Get-ChildItem "$decodedDir\*_accelerometer.csv" |
    ForEach-Object { $_.Name -replace '_accelerometer\.csv$', '' } |
    Sort-Object -Unique

Write-Host "Found $($sessions.Count) sessions to process"

foreach ($session in $sessions) {
    Write-Host "`n=== Processing: $session ==="
    $accelFile = "$decodedDir\${session}_accelerometer.csv"

    # Run pipeline with each session's accelerometer data
    python run_pipeline.py --input-csv $accelFile 2>&1 | Tee-Object -Variable output

    # Copy results
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if (Test-Path "outputs\predictions_fresh.csv") {
        Copy-Item "outputs\predictions_fresh.csv" "$outputDir\predictions_${session}.csv"
    }

    Write-Host "=== Completed: $session ==="
}

Write-Host "`nAll sessions processed. Results in: $outputDir"
```

Run it:

```powershell
.\batch_process_decoded.ps1
```

### Python Script — Batch Process

```python
# batch_ingest.py
"""Process all 26 Decoded recording sessions through the pipeline."""
import subprocess
from pathlib import Path

decoded_dir = Path("data/raw/Decoded")
output_dir = Path("outputs/batch_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all unique sessions
sessions = sorted(set(
    f.name.rsplit("_accelerometer", 1)[0]
    for f in decoded_dir.glob("*_accelerometer.csv")
))

print(f"Found {len(sessions)} sessions")

for i, session in enumerate(sessions, 1):
    accel = decoded_dir / f"{session}_accelerometer.csv"
    print(f"\n[{i}/{len(sessions)}] Processing: {session}")

    result = subprocess.run(
        ["python", "run_pipeline.py", "--input-csv", str(accel)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"  ✓ Success")
    else:
        print(f"  ✗ Failed: {result.stderr[-200:]}")

print("\nBatch processing complete!")
```

---

## 4. Auto-Detection — Pipeline Finds New Data

### How It Currently Works

The pipeline's `find_latest_sensor_pair()` function automatically finds the **newest** accelerometer + gyroscope file pair in `data/raw/`:

```
python run_pipeline.py
# → Automatically picks the newest accel/gyro pair in data/raw/
```

It matches files by:
1. Looking for files with "accelerometer" and "gyroscope" in the name
2. Pairing by shared prefix (e.g., `2025-03-23-15-23-10-`)
3. Falling back to newest accel + newest gyro

### Making Decoded CSVs Auto-Detectable

Currently `find_latest_sensor_pair()` searches `data/raw/` (not subfolders). To use Decoded data automatically:

**Quick Fix — Copy to `data/raw/`:**

```powershell
# Copy the latest Decoded session to data/raw/ where auto-detection works
$latest = Get-ChildItem "data\raw\Decoded\*_accelerometer.csv" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$prefix = $latest.Name -replace '_accelerometer\.csv$', ''
Copy-Item "data\raw\Decoded\${prefix}_accelerometer.csv" "data\raw\"
Copy-Item "data\raw\Decoded\${prefix}_gyroscope.csv" "data\raw\"

# Now auto-detection will find them:
python run_pipeline.py
```

**Better Fix — File Watcher for Continuous Auto-Detection:**

```python
# watch_new_data.py
"""Watch data/raw/Decoded/ for new recordings and auto-run pipeline."""
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewDataHandler(FileSystemEventHandler):
    def __init__(self):
        self.processed = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if "_accelerometer.csv" not in path.name:
            return

        session = path.name.replace("_accelerometer.csv", "")
        if session in self.processed:
            return

        # Wait for gyroscope file to appear too
        gyro = path.parent / f"{session}_gyroscope.csv"
        for _ in range(30):  # wait up to 30 seconds
            if gyro.exists():
                break
            time.sleep(1)

        print(f"\n🔔 New data detected: {session}")
        print(f"   Running pipeline...")

        result = subprocess.run(
            ["python", "run_pipeline.py", "--input-csv", str(path)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"   ✓ Pipeline completed successfully")
            self.processed.add(session)
        else:
            print(f"   ✗ Pipeline failed: {result.stderr[-200:]}")

if __name__ == "__main__":
    watch_dir = "data/raw/Decoded"
    print(f"👁 Watching {watch_dir} for new data...")
    print("   Drop new accelerometer CSV files to trigger pipeline")
    print("   Press Ctrl+C to stop\n")

    observer = Observer()
    observer.schedule(NewDataHandler(), watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

Install watchdog and run:

```bash
pip install watchdog
python watch_new_data.py
```

---

## 5. Viewing Results in MLflow

### Step 1: Start MLflow UI

```bash
# From project root
mlflow ui --backend-store-uri mlruns --port 5000
```

Then open: **http://localhost:5000**

### Step 2: Find Your Pipeline Run

1. In the MLflow UI, select experiment: **`anxiety-activity-recognition`** or **`har-production-pipeline`**
2. Your most recent run appears at the top, named `pipeline_<timestamp>`
3. Click on it to see details

### Step 3: What's Logged

Each pipeline run logs:

| Category | Metrics / Params |
|---|---|
| **Pipeline Info** | `stages_completed`, `stages_failed`, `overall_status` |
| **Inference** | `inference_count`, `inference_time_seconds` |
| **Monitoring** | `monitoring_status`, drift flags |
| **Calibration** (if `--advanced`) | `calibration_temperature`, `calibration_ece`, `mc_dropout_entropy` |
| **Wasserstein Drift** (if `--advanced`) | Wasserstein distances per feature |
| **Curriculum** (if `--advanced`) | `curriculum_accuracy`, `curriculum_n_pseudo_labels` |

### Step 4: Compare Runs

1. Select multiple runs (checkboxes)
2. Click **"Compare"**
3. View metrics side-by-side (useful for comparing different recording sessions)

### Step 5: View Pipeline Result JSON

Results are also saved locally:

```
logs/pipeline/pipeline_result_<timestamp>.json
```

```bash
# View the latest result
Get-ChildItem logs\pipeline\pipeline_result_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content | ConvertFrom-Json
```

### Prediction Output Files

After inference, find predictions in:

```
outputs/predictions_fresh.csv           # Latest predictions (class labels + confidence)
outputs/production_predictions_fresh.npy # Raw prediction probabilities
outputs/production_labels_fresh.npy     # Predicted class indices
```

---

## 6. Advanced Pipeline Options

### Run with Calibration & Drift Detection

```bash
python run_pipeline.py --advanced
# Runs stages 1-7 + stages 11-14 (calibration, Wasserstein drift, curriculum, sensor placement)
```

### Run Specific Stages Only

```bash
# Only inference + evaluation (assumes data is already prepared)
python run_pipeline.py --stages inference evaluation

# Only advanced analytics on existing data
python run_pipeline.py --stages calibration wasserstein_drift
```

### Run with Retraining

```bash
# Standard retraining with AdaBN adaptation
python run_pipeline.py --retrain --adapt adabn

# Pseudo-label self-training
python run_pipeline.py --retrain --adapt pseudo_label --epochs 50
```

### Custom Model Path

```bash
python run_pipeline.py --model "models/pretrained/fine_tuned_model_1dcnnbilstm.keras"
```

### Full Pipeline Command Reference

```bash
python run_pipeline.py \
    --input-csv "data/raw/Decoded/2025-08-19-13-05-35_accelerometer.csv" \
    --advanced \
    --mc-dropout-passes 50 \
    --continue-on-failure
```

| Flag | Purpose |
|---|---|
| `--input-csv PATH` | Use a specific CSV file instead of auto-detection |
| `--skip-ingestion` | Skip Stage 1 (use existing processed data) |
| `--skip-validation` | Skip Stage 2 |
| `--stages STAGE [...]` | Run only specific stages |
| `--retrain` | Enable retraining stages (8-10) |
| `--adapt {adabn,pseudo_label,none}` | Adaptation method |
| `--advanced` | Enable advanced stages (11-14) |
| `--mc-dropout-passes N` | MC Dropout forward passes (default: 30) |
| `--curriculum-iterations N` | Curriculum self-training iterations (default: 5) |
| `--ewc-lambda FLOAT` | Elastic Weight Consolidation strength (default: 1000) |
| `--continue-on-failure` | Don't stop if a stage fails |
| `--model PATH` | Path to a specific model file |
| `--gravity-removal` | Enable gravity component removal |
| `--epochs N` | Training epochs for retraining (default: 100) |
| `--auto-deploy` | Auto-deploy after model registration |

---

## 7. Troubleshooting

### "No accelerometer files found"

The auto-detection searches `data/raw/` (not subfolders). Either:
- Copy files from `Decoded/` to `data/raw/`
- Use `--input-csv` to point directly to a file

### "Missing columns" during ingestion

The Decoded CSVs use `accel_x, accel_y, accel_z` column names. If the pipeline expects different names (like `x, y, z`), it will auto-rename if going through the Excel path. For CSV direct input, the transformation stage handles column mapping.

### MLflow UI not showing runs

```bash
# Make sure you're pointing to the right tracking directory
mlflow ui --backend-store-uri mlruns --port 5000

# Check that mlruns/ has content
Get-ChildItem mlruns\ -Directory
```

### Pipeline fails at transformation

Ensure the fused CSV has the required 6 sensor columns (Ax, Ay, Az, Gx, Gy, Gz). If using only accelerometer data, the transformation stage may need to handle 3-column input.

### "Module not found" errors

```bash
# Install all dependencies
pip install -e ".[dev]"
# OR
pip install -r config/requirements.txt
```

---

## Quick Start — Complete Workflow

```bash
# 1. Run pipeline with newest Decoded session
python run_pipeline.py --input-csv "data/raw/Decoded/2025-08-19-13-05-35_accelerometer.csv"

# 2. Start MLflow to see results
mlflow ui --backend-store-uri mlruns --port 5000

# 3. Open http://localhost:5000 → experiment "har-production-pipeline"

# 4. Check predictions
type outputs\predictions_fresh.csv | Select-Object -First 10
```

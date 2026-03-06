# 10 — Stage Deep-Dive: Ingestion, Preprocessing, and Quality Control

> Part of [Opus Understanding Audit Pack](00_README.md) | Phase 2 — Technical Stage Deep-Dives
> **Commit:** `168c05bb` | **Audit Date:** 2026-02-22

---

## 1. Stage 1 — Data Ingestion (Component 1)

### 1.1 Overview

**FACT:** Implemented in `src/components/data_ingestion.py` (~607 lines).
[CODE: src/components/data_ingestion.py | class:DataIngestion]

The ingestion component handles heterogeneous IMU sensor data from multiple sources (accelerometer + gyroscope files per user/session) and produces a single fused, resampled CSV.

### 1.2 Three Ingestion Paths

| Path | Condition | Behavior |
|------|-----------|----------|
| **Direct CSV** | `ingestion_config.input_csv` is set | Read existing fused CSV, skip discovery |
| **Explicit Pair** | `accel_file` + `gyro_file` set in config | Merge the single pair, no discovery |
| **Auto-Discover** | Default — neither above set | Walk `data_raw_dir`, discover accel/gyro pairs by naming patterns |

**FACT:** [CODE: src/components/data_ingestion.py | method:initiate_data_ingestion — 3 branches]

### 1.3 Auto-Discovery Logic

```python
def discover_sensor_files(base_dir):
    # Walks base_dir looking for accel/gyro file pairs
    # Matches by dataset_name + user_id + session_id
    # Returns list of (accel_path, gyro_path, metadata_dict)
```

**Column renaming maps:**
```python
ACCEL_RENAME = {"Ax": "Ax", "Ay": "Ay", "Az": "Az", ...}  # normalize column names
GYRO_RENAME  = {"Gx": "Gx", "Gy": "Gy", "Gz": "Gz", ...}
```

**FACT:** [CODE: src/components/data_ingestion.py | function:discover_sensor_files, ACCEL_RENAME, GYRO_RENAME]

### 1.4 Sensor Fusion — merge_asof

The core fusion operates via `pd.merge_asof`:
```python
merged = pd.merge_asof(
    accel_df.sort_values("timestamp"),
    gyro_df.sort_values("timestamp"),
    on="timestamp",
    tolerance=pd.Timedelta("20ms"),   # max 20ms mismatch
    direction="nearest"
)
```

**Design rationale (INFERENCE):** `merge_asof` with 20ms tolerance handles IMU clock jitter — accelerometer and gyroscope often have slightly different sampling clocks. Nearest-neighbor time matching within 20ms (1 sample at 50Hz) ensures no artificial lag is introduced.

### 1.5 Resampling to 50Hz

After fusion, the data is resampled to a uniform 50Hz grid:
```python
# Create 50Hz timestamp grid
# Interpolate merged data to grid
# Result: exactly 50Hz sampling, (0, 200] samples per 4-second window
```

**FACT:** Output sampling rate is configurable but defaults to 50Hz.
[CODE: src/components/data_ingestion.py | resampling logic]

### 1.6 Manifest-Based Skip

**FACT:** If a manifest file exists listing previously ingested sessions, those sessions are skipped (incremental ingestion).
[CODE: src/components/data_ingestion.py | manifest logic]

### 1.7 Output Artifact

```
DataIngestionArtifact:
    fused_csv_path: Path          # e.g., data/processed/sensor_fused_50Hz.csv
    n_rows: int                    # total rows in fused CSV
    n_columns: int                 # number of columns
    sampling_hz: int               # 50
    ingestion_timestamp: str       # ISO format
    source_type: str               # "auto_discover" | "explicit_pair" | "direct_csv"
```

---

## 2. Stage 2 — Data Validation (Component 2)

### 2.1 Overview

**FACT:** Implemented in `src/data_validator.py` + wrapped by `src/components/data_validation.py`.
[CODE: src/data_validator.py]
[CODE: src/components/data_validation.py]

### 2.2 Validation Rules

| Rule | What It Checks | Pass Condition |
|------|---------------|----------------|
| **Schema check** | Required columns present | All 6 sensor columns + metadata columns exist |
| **Range check** | Sensor values within physical bounds | Accel within ±20 m/s², Gyro within ±2000 °/s |
| **Missing data** | NaN/null percentage | Missing < threshold (configurable) |
| **Sampling rate** | Temporal consistency | Median sample interval ≈ 20ms (50Hz) |
| **Minimum rows** | Sufficient data | At least `min_samples` rows present |

### 2.3 Output

```
DataValidationArtifact:
    is_valid: bool          # True if all rules pass
    errors: List[str]       # Critical validation failures
    warnings: List[str]     # Non-blocking concerns
```

**Pipeline behavior:** If `is_valid=False` and `continue_on_failure=False` (default), the pipeline **aborts** before transformation.
[CODE: src/pipeline/production_pipeline.py | stage:validation — break on failure]

---

## 3. Stage 3 — Data Transformation (Component 3)

### 3.1 Overview

**FACT:** Implemented in `src/components/data_transformation.py` (~130 lines) delegating to `src/preprocess_data.py` (~779 lines).
[CODE: src/components/data_transformation.py]
[CODE: src/preprocess_data.py | class:UnifiedPreprocessor]

### 3.2 Preprocessing Pipeline (4 Steps)

The `UnifiedPreprocessor` applies an ordered pipeline:

```
Raw CSV → Unit Detection → Gravity Removal → Domain Calibration → Windowing + Scaling → .npy
```

#### Step 1: Unit Detection (`UnitDetector`)

Automatically detects whether accelerometer data is in **m/s²**, **g**, or **milliG**:
```python
class UnitDetector:
    def detect(accel_data) -> str:
        median_magnitude = median(sqrt(ax² + ay² + az²))
        if median_magnitude > 100:      return "milliG"    # ~981 milliG standing
        elif median_magnitude > 5:      return "m/s²"      # ~9.81 m/s²
        else:                           return "g"          # ~1g standing
```

Converts all to **m/s²** using: milliG → ×0.00981, g → ×9.81.

**FACT:** [CODE: src/preprocess_data.py | class:UnitDetector]

#### Step 2: Gravity Removal (`GravityRemover`)

Applies a **4th-order Butterworth low-pass filter** at 0.3Hz to estimate the gravity component, then subtracts it:
```python
class GravityRemover:
    def remove(accel_data, fs=50) -> accel_linear:
        b, a = butter(4, 0.3, btype='low', fs=fs)
        gravity_estimate = filtfilt(b, a, accel_data, axis=0)
        return accel_data - gravity_estimate
```

**FACT:** [CODE: src/preprocess_data.py | class:GravityRemover]

**INFERENCE:** 0.3Hz cutoff isolates the gravity vector (essentially DC + very slow tilt changes). This is standard in IMU-based HAR literature.

#### Step 3: Domain Calibration (`DomainCalibrator`)

Applies per-channel z-score standardization using training baseline statistics:
```python
class DomainCalibrator:
    def calibrate(data, baseline_mean, baseline_std):
        return (data - baseline_mean) / (baseline_std + 1e-8)
```

**FACT:** [CODE: src/preprocess_data.py | class:DomainCalibrator]

**ASSUMPTION:** Calibration uses training-time baseline statistics. If the production domain has shifted, this normalization may not be appropriate (but drift detection will catch this downstream).

#### Step 4: Windowing and Scaling

```python
window_size = 200   # 4 seconds at 50Hz
step_size   = 100   # 50% overlap

# Create sliding windows
windows = []
for i in range(0, len(data) - window_size + 1, step_size):
    windows.append(data[i:i+window_size])

# Per-window StandardScaler (fit on training distribution)
X = np.array(windows)  # shape: (N, 200, 6)
```

**FACT:** Window parameters match the ICTH_16 paper specification: "window size of 200 time steps (4 seconds at 50Hz) with 50% overlap."
[CODE: src/preprocess_data.py | windowing logic]

### 3.3 Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `enable_unit_conversion` | True | Run UnitDetector + conversion |
| `enable_gravity_removal` | True | Run GravityRemover |
| `enable_calibration` | True | Run DomainCalibrator |

**FACT:** Logged at pipeline start: "Unit Conversion (milliG→m/s²): ENABLED/DISABLED", etc.
[CODE: src/pipeline/production_pipeline.py | run() logging lines]

### 3.4 Output Artifact

```
DataTransformationArtifact:
    production_X_path: Path           # data/prepared/production_X.npy (N × 200 × 6)
    metadata_path: Path               # production_metadata.json
    n_windows: int                     # number of windows created
    window_size: int                   # 200
    unit_conversion_applied: bool
    gravity_removal_applied: bool
    preprocessing_timestamp: str
```

---

## 4. Data Flow: Stages 1 → 2 → 3

```
Raw sensor files (accel + gyro CSVs per user/session)
    │
    ▼ Stage 1: Data Ingestion
    ├── discover_sensor_files()
    ├── merge_asof() with 20ms tolerance
    ├── resample to 50Hz
    └── Output: sensor_fused_50Hz.csv
    │
    ▼ Stage 2: Data Validation
    ├── Schema check (columns present?)
    ├── Range check (physical bounds?)
    ├── Missing data check
    ├── Sampling rate check
    └── Output: is_valid + errors/warnings
    │   (abort if is_valid=False)
    │
    ▼ Stage 3: Data Transformation
    ├── Unit detection → convert to m/s²
    ├── Gravity removal (Butterworth 0.3Hz)
    ├── Domain calibration (z-score with baseline)
    ├── Sliding window (200 × 6 @ 50% overlap)
    └── Output: production_X.npy (N × 200 × 6)
```

---

## 5. Critical Findings

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| I-1 | Three ingestion paths provide flexibility but add code complexity | **INFO** | [CODE: src/components/data_ingestion.py] |
| I-2 | merge_asof with 20ms tolerance is appropriate for IMU clock jitter | **STRENGTH** | Standard practice for multi-sensor fusion |
| I-3 | Automatic unit detection (milliG/g/m/s²) prevents common preprocessing errors | **STRENGTH** | [CODE: src/preprocess_data.py:UnitDetector] |
| I-4 | Gravity removal uses standard Butterworth filter | **STRENGTH** | 4th order, 0.3Hz cutoff |
| I-5 | Validation aborts pipeline on failure (safe default) | **STRENGTH** | [CODE: src/pipeline/production_pipeline.py] |
| I-6 | No data versioning (DVC not wired) | **MEDIUM** | No `dvc.yaml` or `.dvc` files in repo |

---

## 6. Recommendations for Thesis

1. **Document the 3-path ingestion pattern**: This is a reusable design for multi-dataset HAR pipelines
2. **Validate the unit detector**: Run it against all datasets and report detection accuracy
3. **Thesis figure**: Create a preprocessing pipeline diagram showing the 4-step transformation
4. **Data versioning**: Consider adding DVC to track raw data and intermediate artifacts

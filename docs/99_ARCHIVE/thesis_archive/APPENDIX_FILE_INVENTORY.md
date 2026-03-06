# Appendix: Full Repository File Inventory (Grouped by Macro-Stage)

Last updated: 2026-01-25

Notes:
- Full workspace scan excludes `.git` objects by request.
- Each file is assigned to exactly one macro-stage (A-I).

<details>
<summary>Stage A) Data & Ingestion (79 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `data/raw/2025-03-23-15-23-10-accelerometer_data.xlsx` | xlsx | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/2025-03-23-15-23-10-gyroscope_data.xlsx` | xlsx | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/all_users_data_labeled.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-03-13_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-03-13_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-03-13_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-46-56_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-46-56_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-21-46-56_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-22-29-04_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-22-29-04_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-22-29-04_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-23-13-09_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-23-13-09_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-16-23-13-09_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-01-48-33_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-01-48-33_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-01-48-33_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-13-49-10_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-13-49-10_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-13-49-10_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-14-16-20_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-14-16-20_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-14-16-20_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-05-26_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-05-26_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-05-26_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-59-12_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-59-12_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-15-59-12_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-17-39-05_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-17-39-05_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-17-39-05_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-18-53-05_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-18-53-05_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-18-53-05_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-19-27-18_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-19-27-18_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-19-27-18_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-21-42-30_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-21-42-30_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-17-21-42-30_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-19-11-24-04_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-19-11-24-04_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-19-11-24-04_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-13-58-56_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-13-58-56_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-13-58-56_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-16-30-24_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-16-30-24_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-21-16-30-24_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-12-15-48_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-12-15-48_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-12-15-48_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-22-11-22_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-22-11-22_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-07-25-22-11-22_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-01-12-52-49_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-01-12-52-49_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-01-12-52-49_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-12-33-33_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-12-33-33_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-12-33-33_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-13-19-03_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-13-19-03_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-11-13-19-03_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-12-09-08_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-12-09-08_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-12-09-08_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-17-01-38_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-17-01-38_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-12-17-01-38_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-13-04-18-52_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-13-04-18-52_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-13-04-18-52_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-19-08-58-40_accelerometer.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-19-08-58-40_gyroscope.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `data/raw/Decoded/2025-08-19-08-58-40_record.csv` | csv | Raw input data | Input to `src/sensor_data_pipeline.py` | N/A | data/processed/* |  |
| `src/sensor_data_pipeline.py` | py | Raw ingestion + sensor fusion pipeline | CLI: `python src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/processed/* |  |

</details>

<details>
<summary>Stage B) Preprocessing & Sensor Fusion (25 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `config/pipeline_config.yaml` | yaml | Pipeline preprocessing/inference config | Loaded by src scripts | N/A | N/A |  |
| `data/prepared/config.json` | json | Preprocessed features / metadata | Output of `src/preprocess_data.py` | data/processed/* | N/A |  |
| `data/prepared/PRODUCTION_DATA_README.md` | md | Preprocessed features / metadata | Output of `src/preprocess_data.py` | data/processed/* | N/A |  |
| `data/prepared/production_metadata.json` | json | Preprocessed features / metadata | Output of `src/preprocess_data.py` | data/processed/* | N/A |  |
| `data/prepared/production_X.npy` | npy | Preprocessed features / metadata | Output of `src/preprocess_data.py` | data/processed/* | N/A |  |
| `data/prepared/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `data/preprocessed/sensor_fused_50Hz.csv` | csv | Intermediate preprocessed data | Output of preprocessing | data/processed/* | data/prepared/* |  |
| `data/preprocessed/sensor_fused_meta.json` | json | Intermediate preprocessed data | Output of preprocessing | data/processed/* | data/prepared/* |  |
| `data/preprocessed/sensor_merged_native_rate.csv` | csv | Intermediate preprocessed data | Output of preprocessing | data/processed/* | data/prepared/* |  |
| `data/processed/sensor_fused_50Hz.csv` | csv | Processed sensor data | Output of `src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/prepared/* |  |
| `data/processed/sensor_fused_50Hz_converted.csv` | csv | Processed sensor data | Output of `src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/prepared/* |  |
| `data/processed/sensor_fused_meta.json` | json | Processed sensor data | Output of `src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/prepared/* |  |
| `data/processed/sensor_merged_native_rate.csv` | csv | Processed sensor data | Output of `src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/prepared/* |  |
| `data/processed/test_fused_decoded.csv` | csv | Processed sensor data | Output of `src/sensor_data_pipeline.py` | data/raw/*.xlsx | data/prepared/* |  |
| `logs/preprocessing/pipeline.log` | log | Preprocessing log output | Output of preprocessing scripts | data/raw/* | N/A |  |
| `logs/preprocessing/production_preprocessing_20251212_175059.log` | log | Preprocessing log output | Output of preprocessing scripts | data/raw/* | N/A |  |
| `logs/preprocessing/production_preprocessing_20260106_115108.log` | log | Preprocessing log output | Output of preprocessing scripts | data/raw/* | N/A |  |
| `logs/preprocessing/production_preprocessing_20260115_130155.log` | log | Preprocessing log output | Output of preprocessing scripts | data/raw/* | N/A |  |
| `notebooks/data_preprocessing_step1.ipynb` | ipynb | Notebook | Jupyter/manual reference | N/A | N/A |  |
| `notebooks/from_guide_processing.ipynb` | ipynb | Notebook | Jupyter/manual reference | N/A | N/A |  |
| `notebooks/production_preprocessing.ipynb` | ipynb | Notebook | Jupyter/manual reference | N/A | N/A |  |
| `notebooks/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `src/Archived(prepare traning- production- conversion)/convert_production_units.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/Archived(prepare traning- production- conversion)/prepare_production_data.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/preprocess_data.py` | py | CSV -> windowed NPY preprocessing | CLI: `python src/preprocess_data.py` | data/processed/*.csv | data/prepared/*.npy |  |

</details>

<details>
<summary>Stage C) Data QC / Validation (12 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `reports/preprocess_qc/qc_20260109_141008.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `reports/preprocess_qc/qc_20260109_141031.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `reports/preprocess_qc/qc_20260109_141126.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `reports/preprocess_qc/qc_20260109_141152.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `reports/preprocess_qc/qc_20260109_141208.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `reports/preprocess_qc/qc_20260109_141213.json` | json | QC report output | Output of `scripts/preprocess_qc.py` | data/processed/* | N/A |  |
| `scripts/preprocess_qc.py` | py | Preprocessing QC checks | CLI: `python scripts/preprocess_qc.py` | data/processed/*.csv or data/prepared/*.npy | reports/preprocess_qc/* |  |
| `src/Archived(prepare traning- production- conversion)/training_cv_experiment/validate_labels.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/Archived(prepare traning- production- conversion)/validate_garmin_data.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/Archived(prepare traning- production- conversion)/validate_model_and_diagnose.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/data_validator.py` | py | Data validation utilities | Imported by QC scripts | N/A | N/A |  |
| `src/diagnostic_pipeline_check.py` | py | Pipeline diagnostics | CLI/Imported | N/A | N/A |  |

</details>

<details>
<summary>Stage D) Baselines & Reference Statistics (4 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `data/prepared/baseline_stats.json` | json | Baseline statistics | Output of `scripts/build_training_baseline.py` | data/raw/all_users_data_labeled.csv | N/A |  |
| `models/normalized_baseline.json` | json | Normalized baseline statistics | Output of `scripts/create_normalized_baseline.py` | data/raw/all_users_data_labeled.csv | N/A | Uses hard-coded absolute paths. |
| `scripts/build_training_baseline.py` | py | Baseline statistics builder | CLI: `python scripts/build_training_baseline.py` | data/raw/all_users_data_labeled.csv | data/prepared/baseline_stats.json |  |
| `scripts/create_normalized_baseline.py` | py | Normalized baseline builder | CLI: `python scripts/create_normalized_baseline.py` | data/raw/all_users_data_labeled.csv | models/normalized_baseline.json | Uses hard-coded absolute paths. |

</details>

<details>
<summary>Stage E) Training & Experiment Tracking (165 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `.dvc/.gitignore` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/00/a3eaf2c3ccf3394e78f4d983e794bf` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/02/7b59b4d617365818d3061f0016bfd1` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/03/7112a5510c71ddb1d010cdaf9d8a59` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/09/abab81f7ca90f75862a435be890620` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/0a/f976dc1f94eba068dc612bd8dcef67` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/0b/7f86042d2b050d738c281de5df85fa` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/13/177b68efea3c0e7a1d5599eda65c04` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/3b/b89afbead08f80d8435437d67a6eb2.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/3f/79bfa694c7575282696d1a8959d5a8` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/4f/403fe417e2086e6088caf7c8699869` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/51/978b0c7b3b1d19a3e15f55e3b5d3a9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/52/a4423d2abc56614022864c89faf258` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/56/5e8c51d1cf617dfa71d0b7590c300d` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/5a/8f22bcba8db25ff88bc2991517f47b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/62/e13916bcac592e412f47020257831b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/69/c45431805a9772975fa73c8811fb97.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/75/df19bfa57c6d4a8ee84b84d6c3f334` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/76/80434d06d908581f48d50247f71e54` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/84/b076a1c1ebe9c607647bf757a42046` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/84/b284ec7a51f8c8e00a385b57835fb5` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/85/d003a961c72496dd3d4c0169c45da9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/90/251d79499a47a75f41e9910942a181` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/98/fe03a240c410fe80df8174f6bdd957` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/9a/2e6cd7fe75c6773e6790e24ec40094` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/9c/df8d425e35bc7a90c92db039c3dde7` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/a1/df11782807ac51484f9e9747bc68f2.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/a3/378df65380f9062735e1f541f32b01.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/ac/9bcb5de421cf6b4872fb67385f62c6` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/af/5803c623688a0b1e2ba18b1aa81637` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/ca/3a519f5250d7fe836802533f66955e` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/ce/480e07aef1d8933ab0852e638a148a` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/cache/files/md5/f9/3e297b2e1e4fcb86ca69966c721ac9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/config` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/btime` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/dag.md` | md | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/15/5e05372f8cd8d2c06c78fc53c90436979c4ff5` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/1a/e96728c46084994ab1c38ad2a0f4b0d5a0afbd` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/20/f89f0d63fa7b6aa46d0bd21ba0bb561e2c5a86` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/23/7ef7817444f8acf81c94883455bf4388e65d5b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/26/299aec244f513e4ca807c420c6807bfce17e97` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/31/9094330a1b338b15fec0155a1ddedf53674048` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/39/aa52e04a566e53580e5bfeaa8aef62cd443570` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/40/2189ffee150c1bccae79343179e7df1f8cf7a4` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/43/985d480007447ded99a446c06009bc10d05abe` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/46/515dda730104bedfdcb2eb838651a57e912269` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/49/f3ad381cbdf278e72ed0359685e7396413bef8` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/51/5a0b74825597b92c4864350da2a104e72b71b5` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/54/d4efe4e5fb427afdd49f3842e6b60b035badcb` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/5c/dc93819f351475077ce35cdf2a3ec75ee21fb6` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/6a/6130176f46ba584065fcd1e1d67132b1e0642a` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/72/6ab040f8d4d7c82e633680b76fb9ea9144c08f` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/90/391c17c6c83c575bc5685fffb961d2047e9e39` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/94/69bf14f47d3db99a3972998137e174f3029e3b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/95/69c676f58a9aad559b7d58d465c08967f8a7d7` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/ad/e66314ab5733c1bada1563117281141e06cfda` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/ae/f99974e6e977aafa1b7a53198080ca4e490ad8` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/c9/c95b30d20e90af8641fc620bd3fcdbec1fc6c7` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/d2/aef243448c6c98f8d505869d99b6e349907bd6` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/e2/db36c56c48e838e0c3cba32b12f698f8bb2120` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/f6/5a8da7c1b3b5ad6954f9789a838ce709c3f992` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/f8/f953d9c09c76fe2e8138f0385dedf3a4c0b36a` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/exps/cache/fc/f0b0b4dd3779f065712fcb511aa302fd3340e7` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/lock` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/rwlock` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc/tmp/rwlock.lock` | lock | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/00/a3eaf2c3ccf3394e78f4d983e794bf` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/02/7b59b4d617365818d3061f0016bfd1` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/03/7112a5510c71ddb1d010cdaf9d8a59` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/09/abab81f7ca90f75862a435be890620` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/0a/f976dc1f94eba068dc612bd8dcef67` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/0b/7f86042d2b050d738c281de5df85fa` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/13/177b68efea3c0e7a1d5599eda65c04` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/3b/b89afbead08f80d8435437d67a6eb2.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/3f/79bfa694c7575282696d1a8959d5a8` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/4f/403fe417e2086e6088caf7c8699869` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/51/978b0c7b3b1d19a3e15f55e3b5d3a9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/52/a4423d2abc56614022864c89faf258` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/56/5e8c51d1cf617dfa71d0b7590c300d` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/5a/8f22bcba8db25ff88bc2991517f47b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/62/e13916bcac592e412f47020257831b` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/69/c45431805a9772975fa73c8811fb97.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/75/df19bfa57c6d4a8ee84b84d6c3f334` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/76/80434d06d908581f48d50247f71e54` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/84/b076a1c1ebe9c607647bf757a42046` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/84/b284ec7a51f8c8e00a385b57835fb5` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/85/d003a961c72496dd3d4c0169c45da9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/90/251d79499a47a75f41e9910942a181` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/98/fe03a240c410fe80df8174f6bdd957` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/9a/2e6cd7fe75c6773e6790e24ec40094` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/9c/df8d425e35bc7a90c92db039c3dde7` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/a1/df11782807ac51484f9e9747bc68f2.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/a3/378df65380f9062735e1f541f32b01.dir` | dir | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/ac/9bcb5de421cf6b4872fb67385f62c6` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/af/5803c623688a0b1e2ba18b1aa81637` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/ca/3a519f5250d7fe836802533f66955e` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/ce/480e07aef1d8933ab0852e638a148a` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `.dvc_storage/files/md5/f9/3e297b2e1e4fcb86ca69966c721ac9` | no-ext | DVC cache/metadata (internal) | dvc (internal) | N/A | N/A | Internal cache |
| `config/mlflow_config.yaml` | yaml | MLflow tracking config | Loaded by src/mlflow_tracking.py | N/A | N/A |  |
| `data/prepared.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `data/processed.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `data/raw.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `mlruns/0/meta.yaml` | yaml | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/artifacts/predictions_20260106_115143.csv` | csv | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/artifacts/predictions_20260106_115143_metadata.json` | json | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/artifacts/predictions_20260106_115143_probs.npy` | npy | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/meta.yaml` | yaml | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/metrics/avg_confidence` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/metrics/count_hand_tapping` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/metrics/std_confidence` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/metrics/total_windows` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/metrics/uncertain_count` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/batch_size` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/channels` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/confidence_threshold` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/input_path` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/mode` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/model_params` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/model_path` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/n_windows` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/params/timesteps` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/tags/mlflow.runName` | runname | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/tags/mlflow.source.git.commit` | commit | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/tags/mlflow.source.name` | name | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/tags/mlflow.source.type` | type | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/62435f47bef54ac9840ef3e3b413b3e9/tags/mlflow.user` | user | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/artifacts/predictions_20251212_175115.csv` | csv | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/artifacts/predictions_20251212_175115_metadata.json` | json | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/artifacts/predictions_20251212_175115_probs.npy` | npy | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/meta.yaml` | yaml | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/avg_confidence` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_forehead_rubbing` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_hand_tapping` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_nape_rubbing` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_sitting` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_smoking` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/count_standing` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/std_confidence` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/total_windows` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/metrics/uncertain_count` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/batch_size` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/channels` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/confidence_threshold` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/input_path` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/mode` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/model_params` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/model_path` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/n_windows` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/params/timesteps` | no-ext | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/tags/mlflow.runName` | runname | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/tags/mlflow.source.git.commit` | commit | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/tags/mlflow.source.name` | name | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/tags/mlflow.source.type` | type | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/63f4a91bc5924b5cafb4bcb028f69d6b/tags/mlflow.user` | user | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/meta.yaml` | yaml | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `mlruns/950614147457743858/tags/mlflow.experimentKind` | experimentkind | MLflow tracking data (internal) | mlflow (internal) | N/A | N/A | Internal tracking data |
| `models/.gitignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `models/pretrained.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` | keras | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `models/pretrained/model_info.json` | json | JSON artifact/config | Used by scripts | N/A | N/A |  |
| `papers/research_papers/all_users_data_labeled.csv.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `papers/research_papers/anxiety_dataset.csv.dvc` | dvc | DVC tracking file | dvc (internal) | N/A | N/A | Tracks large artifact via DVC |
| `src/Archived(prepare traning- production- conversion)/prepare_training_data.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/Archived(prepare traning- production- conversion)/training_cv_experiment/train_with_cv.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/mlflow_tracking.py` | py | MLflow tracking helper | Imported by pipeline scripts | mlflow tracking URI | mlruns/* |  |

</details>

<details>
<summary>Stage F) Evaluation & Confidence / Calibration (7 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `logs/evaluation/cross_user_evaluation.json` | json | Evaluation log output | Output of `src/evaluate_predictions.py` | data/prepared/predictions/*.csv | N/A |  |
| `logs/evaluation/cross_user_summary.txt` | txt | Evaluation log output | Output of `src/evaluate_predictions.py` | data/prepared/predictions/*.csv | N/A |  |
| `logs/evaluation/evaluation_20251212_175119.log` | log | Evaluation log output | Output of `src/evaluate_predictions.py` | data/prepared/predictions/*.csv | N/A |  |
| `outputs/evaluation/evaluation_20251212_175119.json` | json | Evaluation report output | Output of `src/evaluate_predictions.py` | data/prepared/predictions/*.csv | N/A |  |
| `outputs/evaluation/evaluation_20251212_175119.txt` | txt | Evaluation report output | Output of `src/evaluate_predictions.py` | data/prepared/predictions/*.csv | N/A |  |
| `src/Archived(prepare traning- production- conversion)/training_cv_experiment/k_fold_evaluator.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/evaluate_predictions.py` | py | Prediction evaluation pipeline | CLI: `python src/evaluate_predictions.py` | data/prepared/predictions/*.csv | outputs/evaluation/* |  |

</details>

<details>
<summary>Stage G) Inference & Smoke Tests (22 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `data/prepared/predictions/predictions_20251212_165815.csv` | csv | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20251212_165815_metadata.json` | json | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20251212_165815_probs.npy` | npy | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20251212_175115.csv` | csv | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20251212_175115_metadata.json` | json | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20251212_175115_probs.npy` | npy | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20260106_115143.csv` | csv | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20260106_115143_metadata.json` | json | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `data/prepared/predictions/predictions_20260106_115143_probs.npy` | npy | Inference predictions output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `docker/api/__init__.py` | py | Package marker | Imported by docker/api | N/A | N/A |  |
| `docker/api/main.py` | py | FastAPI inference service | Run in Docker container | models + config | HTTP inference API |  |
| `logs/inference/inference_20251212_175104.log` | log | Inference log output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `logs/inference/inference_20260106_115126.log` | log | Inference log output | Output of `src/run_inference.py` | data/prepared/production_X.npy | N/A |  |
| `outputs/gravity_removal_comparison.png` | png | Output artifact | Output of pipeline scripts | N/A | N/A |  |
| `outputs/predictions_fresh.csv` | csv | Output artifact | Output of pipeline scripts | N/A | N/A |  |
| `outputs/production_labels_fresh.npy` | npy | Output artifact | Output of pipeline scripts | N/A | N/A |  |
| `outputs/production_predictions_fresh.npy` | npy | Output artifact | Output of pipeline scripts | N/A | N/A |  |
| `reports/inference_smoke/smoke_20260109_141333.json` | json | Inference smoke report | Output of `scripts/inference_smoke.py` | data/prepared/production_X.npy | N/A |  |
| `reports/inference_smoke/smoke_20260109_141421.json` | json | Inference smoke report | Output of `scripts/inference_smoke.py` | data/prepared/production_X.npy | N/A |  |
| `reports/inference_smoke/smoke_20260109_141448.json` | json | Inference smoke report | Output of `scripts/inference_smoke.py` | data/prepared/production_X.npy | N/A |  |
| `scripts/inference_smoke.py` | py | Inference smoke test | CLI: `python scripts/inference_smoke.py` | data/prepared/production_X.npy | reports/inference_smoke/* |  |
| `src/run_inference.py` | py | Batch inference pipeline | CLI: `python src/run_inference.py` | data/prepared/production_X.npy | data/prepared/predictions/* |  |

</details>

<details>
<summary>Stage H) Monitoring & Drift Detection (21 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `reports/monitoring/2026-01-15_12-58-37_20260106_115143/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-37_20260106_115143/drift_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-37_20260106_115143/temporal_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-53_20260106_115143/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-53_20260106_115143/drift_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-53_20260106_115143/summary.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_12-58-53_20260106_115143/temporal_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-05-02_fresh/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-05-56_fresh/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-05-56_fresh/drift_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-05-56_fresh/summary.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-05-56_fresh/temporal_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-11-41_fresh/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-11-41_fresh/drift_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-11-41_fresh/summary.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-11-41_fresh/temporal_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-13-21_fresh/confidence_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-13-21_fresh/drift_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-13-21_fresh/summary.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `reports/monitoring/2026-01-15_13-13-21_fresh/temporal_report.json` | json | Monitoring report output | Output of `scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | N/A |  |
| `scripts/post_inference_monitoring.py` | py | Post-inference monitoring + drift | CLI: `python scripts/post_inference_monitoring.py` | predictions CSV + baseline JSON | reports/monitoring/* |  |

</details>

<details>
<summary>Stage I) Retraining & CI/CD & Packaging (369 files)</summary>

| File Path | Type | Purpose (1 line) | Entry/Used By | Inputs | Outputs | Notes |
|---|---|---|---|---|---|---|
| `.dockerignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `.dvcignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `.gitignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `ai helps/image-1.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `ai helps/image-2.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `ai helps/image.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `ai helps/MLOps Lifecycle Framework and LLMOps Integration – Repository Analysis.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `cheat sheet/1697955590966.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `cheat sheet/DVC_cheatsheet.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `config/.pylintrc` | no-ext | Pylint configuration | Used by linting tools | N/A | N/A |  |
| `config/requirements.txt` | txt | Python dependencies list | Used by Docker and local installs | N/A | N/A |  |
| `data/.gitignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `data/all_users_data_labeled.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `data/anxiety_dataset.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `data/prepared/garmin_labeled.csv` | csv | Labeled data for retraining | Output of `label_garmin_data.py` | data/raw/* | N/A |  |
| `data/samples_2005 dataset/f_data_50hz.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `docker-compose.yml` | yml | Docker Compose orchestrator | Run via `docker-compose up -d` | Dockerfiles + images | Running services |  |
| `docker/Dockerfile.inference` | dockerfile | Docker image for inference API | Built by docker-compose | config/requirements.txt | Inference image |  |
| `docker/Dockerfile.training` | dockerfile | Docker image for training/preprocessing | Built by docker-compose | config/requirements.txt | Training image |  |
| `docs/APPENDIX_PAPER_INDEX.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/APPENDIX_FILE_INVENTORY.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/archive/extranotes.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/archive/LATER_Offline_MLOps_Guide.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/archive/Mondaymeet.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/archive/RESTRUCTURE_PIPELINE_PACKAGES.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/BIG_QUESTIONS_2026-01-18.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/BIG_QUESTIONS_RISK_PAPERS_2026-01-18.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/HOW_TO_READ_DOCS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/MENTOR_QA_SIMPLE_WITH_PAPERS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/ORGANIZATION_MAP.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/output_1801_2026-01-18.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/PIPELINE_DEEP_DIVE_opus.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/PIPELINE_STAGE_PROGRESS_DASHBOARD.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/REPOSITORY_CLEANUP_ANALYSIS_2026-01-18.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/research/KEEP_Research_QA_From_Papers.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/research/RESEARCH_PAPER_INSIGHTS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/research/RESEARCH_PAPERS_ANALYSIS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/evaluation_audit.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/FRESH_START_CLEANUP_GUIDE.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/pipeline_audit_map.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/PIPELINE_RERUN_GUIDE.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/PIPELINE_TEST_RESULTS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/PIPELINE_VISUALIZATION_CURRENT.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/QC_EXECUTION_SUMMARY.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/root_cause_low_accuracy.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/technical/tracking_audit.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/CONCEPTS_EXPLAINED.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/FINE_TUNING_STRATEGY.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/production refrencxe/KEEP_Production_Robustness_Guide.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/production refrencxe/KEEP_Reference_Project_Learnings.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/production refrencxe/KEEP_Technology_Stack_Analysis.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/QA_LAB_TO_LIFE_GAP.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/THESIS_READY_UNLABELED_EVALUATION_PLAN.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/thesis/UNLABELED_EVALUATION.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `docs/THESIS_PROGRESS_DASHBOARD_2026-01-20.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `EHB_2025_71.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `HOW_TO_READ_THIS_REPOSITORY.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `ICTH_16.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `images/1D CNN-3BiLSTm.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/1D CNN-BiLSTM.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/1DCNN-BiLSTM_for_Wearable_Anxiety_Detection.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `images/83f2361f.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/Data structure of garmin.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/garmin data -Motion_Data_Intelligence.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `images/Gemini_Generated_Image_kk7036kk7036kk70.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/LM01.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/LM02.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/mlflow.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/MLOps_Production_System_Blueprint.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `images/PIPELINE LM1.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/Prognosis model.png` | png | Image asset | Manual reference | N/A | N/A |  |
| `images/Prognosis_Models_Building_Predictive_Foresight.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `images/unnamed.jpg` | jpg | Image asset | Manual reference | N/A | N/A |  |
| `MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `New Microsoft Word Document (AutoRecovered).docx` | docx | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `papers/anxiety_detection/A Survey on Wearable Sensors for Mental Health Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/anxiety_detection/ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/anxiety_detection/Anxiety Detection Leveraging Mobile Passive Sensing.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/anxiety_detection/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/domain_adaptation/Domain Adaptation for Inertial Measurement Unit-based Human.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/domain_adaptation/Transfer Learning in Human Activity Recognition  A Survey.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Building-Scalable-MLOps-Optimizing-Machine-Learning-Deployment-and-Operations.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Developing a Scalable MLOps Pipeline for Continuou.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Essential_MLOps_Data_Science_Horizons_2023_Data_Science_Horizons_Final_2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/MLOps A Step Forward to Enterprise Machine Learning 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/MLOps and LLMOps with Python_ A Comprehensive Guide with Tools and Best Practices _ by André Castro _ Medium.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Practical-mlops-operationalizing-machine-learning-models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Research Roadmap_ Developing a Scalable MLOps Pipeline for Continuous Mental Health Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Resilience-aware MLOps for AI-based medical diagnostic system  2024.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Roadmap for a Scalable MLOps Pipeline in Mental Health Monitoring (Master’s Thesis).pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/The Role of MLOps in Healthcare Enhancing Predictive Analytics and Patient Outcomes.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/mlops_production/Thesis_MLOps_FullPlan_with_GanttChart.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1-s2.0-S1574119220300353-main.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1-s2.0-S1574119223000755-main (1).pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1-s2.0-S1574119223000755-main.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1706.04599v2 (1).pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1706.04599v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1711.10160v1.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/1802.03916v3.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/2010.03759v4.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/2404.15331v1.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/2406.01416v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/2505.04608v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/activear.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/preprints202601.0069.v1 (1).pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/preprints202601.0069.v1.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/s10489-025-06708-7.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/s41746-024-01062-3.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/new paper/sensors-21-01669-v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/01.AUCS10080.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/1806.05208v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/2202.10169v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/2503.15577v1.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/978-981-15-0474-7.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Close Look into Human Activity Recognition Models using Deep Learning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A DIFFUSION MODEL FOR MULTIVARIATE.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Self-supervised Framework for Improved 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Survey on Wearable Sensors for Mental Health Monitoring 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A systematic review of passive data for remote monitoring in psychosis and schizophrenia.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A Visual Data and Detection Pipeline for Wearable Industrial Assistants.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A_Comprehensive_Review_on_Harnessing_Wearable_Tech.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A_Multi-Resolution_Deep_Learning_Approach_for_IoT-Integrated_Biomedical_Signal_Processing_using_CNN-LSTM.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/A_Survey_on_Wearable_Sensors_for_Mental_Health_Mon.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Advancing digital sensing in mental health research.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/AI-Driven Tools for Detecting and Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/AI-Driven_Health_Monitoring_using_Wearable_Devices_for_Physical_and_Mental_Well-Being.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/An AI-native Runtime for Multi-Wearable Environments.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Anxolotl_an_Anxiety_Companion_App_-_Stress_Detecti.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Atechnical framework for deploying custom real-time machine 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Atechnical framework for deploying custom real-time machine.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/AutoMR- A Universal Time Series Motion Recognition Pipeline.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/AWearable Device Dataset for Mental Health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Building-Scalable-MLOps-Optimizing-Machine-Learning-Deployment-and-Operations.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/CNN and LSTM-Based Emotion Charting Using Physiological Signals 2020.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Comparative Study on the Effects of Noise in.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Complete Reading List with Detailed Rationale.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep CNN-LSTM With Self-Attention Model for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep learning for sensor-based activity recognition_ A survey.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep Learning in Human Activity Recognition with Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years 2020.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Deep_Learning-based_Fall_Detection_Algorithm_Using_Ensemble_Model_of_Coarse-fine_CNN_and_GRU_Networks.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Detecting Mental Disorders with Wearables 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Detecting Prolonged Stress in Real Life Using Wearable Biosensors and Ecological Momentary Assessments 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Detection and monitoring of stress using wearable a systematic review.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Developing a Scalable MLOps Pipeline for Continuou.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Development and Testing of Retrieval Augmented Generation in Large Language Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Development of a two-stage depression symptom detection model.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/DevOps-Driven Real-Time Health Analytics.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Editorial Reward Processing in Motivational and Affective Disorders.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/EEGEmotion Recognition via a Lightweight.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/EHB_2025_71.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enabling End-To-End Machine Learning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enhancing generalized anxiety disorder diagnosis precision MSTCNN model utilizing high-frequency EEG signals.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enhancing Multimodal Electronic Health Records.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Enhancing the Structural Health Monitoring (SHM) through data.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Essential_MLOps_Data_Science_Horizons_2023_Data_Science_Horizons_Final_2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Evaluating large language models on medical evidence summarization.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Exploring the Capabilities of LLMs for IMU-based Fine-grained.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/extract_paper_info.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `papers/papers needs to read/fninf-16-859309.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Fusing Wearable Biosensors with Artificial Intelligence for Mental Health Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/generate_paper_summary_excel.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `papers/papers needs to read/Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Human Activity Recognition Using Tools of Convolutional Neural Networks.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/ICTH_16.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/ICTH_2025_Oleh_Paper_MLOps_Summary.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Identifying Objective Physiological Markers and Modifiable Behaviors for Self-Reported Stress and Mental Health Status Using Wearable Sensors and Mobile Phones Observational Study 2018.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Implications on Human Activity Recognition Research.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/informatics-09-00056.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/information-16-00087-v2.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Integrating Environmental Data for Mental Health Monitoring_ A Data-Driven IoT-Based Approach.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Integrating-AI_ML-With-Wearable-Devices-for-Monitoring-Student-Mental-Health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/IoT-based_Mental_Health_Monitoring_with_Machine_Learning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/IV2023_Anxolotl.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/jpm-14-00203.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Learning the Language of wearable sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/LLM Chatbot-Creation Approaches.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/LSM-2-Learning from Incomplete Wearable Sensor Data.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Machine Learning Based Anxiety Detection in Older 2020.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Machine Learning based Anxiety Detection using Physiological Signals and Context Features.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Machine Learning for Healthcare Wearable Devices  The Big Picture.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Machine Learning Operations in Health Care A Scoping Review.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Medical Graph RAG Towards Safe Medical Large Language Model via.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/medium.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Mental Disorders Detection in the Era of Large Language Models 2024.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Mental Health Disorder Detection System Based on Wearable Sensors and Artificial Neural Networks 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Mental-Perceiver Audio-Textual Multi-Modal Learning for Estimating Mental.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/MLHOps Machine Learning for Healthcare Operations.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/MLOps A Step Forward to Enterprise Machine Learning 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/MLOps and LLMOps with Python_ A Comprehensive Guide with Tools and Best Practices _ by André Castro _ Medium.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Momentary Stressor Logging and Reflective Visualizations Implications for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Multimodal Frame-Scoring Transformer for Video Summarization.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Panic Attack Prediction Using Wearable Devices and Machine.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/papers_summary.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/papers needs to read/papers_summary.xlsx` | xlsx | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/papers needs to read/Passive Sensing for Mental Health Monitoring Using Machine Learning With Wearables and Smartphones.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Passive Sensing for Mental Health Monitoring Using Machine.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Personalized machine learning of depressed mood using wearables 2021.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Practical-mlops-operationalizing-machine-learning-models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Pre-trained 1DCNN-BiLSTM Hybrid Network for Temperature Prediction of Wind Turbine Gearboxes.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Quantifying Digital Biomarkers for Well-Being Stress, Anxiety, Positive and Negative Affect via Wearable Devices and Their Time-Based Predictions 2023.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Real-Time Stress Monitoring Detection and Management in College Students.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Reproducible workflow for online AI in digital health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Research Roadmap_ Developing a Scalable MLOps Pipeline for Continuous Mental Health Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Research_Papers_Summary.xlsx` | xlsx | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/papers needs to read/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Resilience-aware MLOps for AI-based medical diagnostic system  2024.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Roadmap for a Scalable MLOps Pipeline in Mental Health Monitoring (Master’s Thesis).pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Scientific Evidence for Clinical Text Summarization Using Large.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/sensors-23-00849.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Smart-Wearable-Technologies-for-Real-Time-Mental-Health-Monitoring-A-Holistic-Approach-to-Early-Detection-and-Intervention.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Spatiotemporal Feature Fusion for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Stress and Emotion Open Access Data A Review on Datasets, Modalities, Methods, Challenges, and Future Research Perspectives.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Stress detection through wearable sensors a cnn.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/techniques for detection, prediction and monitoring of stress and stress related mental disorders with use of ML,DL.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/The Agile Deployment of Machine Learning Models in Healthcar 2019.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/The Role of MLOps in Healthcare Enhancing Predictive Analytics and Patient Outcomes.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Thesis_MLOps_FullPlan_with_GanttChart.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Toward Reusable Science with Readable Code and.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Transforming Wearable Data into Personal Health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Tutorial on time series prediction using 1D-CNN and BiLSTM.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Wearable Artificial Intelligence for Detecting Anxiety.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Wearable Sensor‐Based Human Activity Recognition Using Hybrid Deep Learning Techniques 2020.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Wearable, Environmental, and Smartphone-Based Passive Sensing for Mental Health Monitoring 2021.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/Wearable_Tech_for_Mental_Health_Monitoring_Real-Time_Stress_Detection_Using_Biometric_Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/papers needs to read/When Does Optimizing a Proper Loss Yield Calibration.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/.gitignore` | no-ext | Ignore rules | Used by tooling | N/A | N/A |  |
| `papers/research_papers/76 papers/A Close Look into Human Activity Recognition Models using Deep Learning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A DIFFUSION MODEL FOR MULTIVARIATE.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Survey on Wearable Sensors for Mental Health Monitoring.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/A Visual Data and Detection Pipeline for Wearable Industrial Assistants.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/An AI-native Runtime for Multi-Wearable Environments.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Anxiety Detection Leveraging Mobile Passive Sensing.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Atechnical framework for deploying custom real-time machine.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/AutoMR- A Universal Time Series Motion Recognition Pipeline.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Comparative Study on the Effects of Noise in.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Deep CNN-LSTM With Self-Attention Model for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Deep Learning in Human Activity Recognition with Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Development and Testing of Retrieval Augmented Generation in Large Language Models.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Development of a two-stage depression symptom detection model.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/DevOps-Driven Real-Time Health Analytics.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Domain Adaptation for Inertial Measurement Unit-based Human.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Enabling End-To-End Machine Learning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Enhancing Multimodal Electronic Health Records.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Evaluating large language models on medical evidence summarization.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Exploring the Capabilities of LLMs for IMU-based Fine-grained.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Human Activity Recognition Using Tools of Convolutional Neural Networks.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Implications on Human Activity Recognition Research.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Learning the Language of wearable sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/LLM Chatbot-Creation Approaches.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/LSM-2-Learning from Incomplete Wearable Sensor Data.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Machine Learning based Anxiety Detection using Physiological Signals and Context Features.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Medical Graph RAG Towards Safe Medical Large Language Model via.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/MLHOps Machine Learning for Healthcare Operations.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Momentary Stressor Logging and Reflective Visualizations Implications for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Multimodal Frame-Scoring Transformer for Video Summarization.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Panic Attack Prediction Using Wearable Devices and Machine.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Real-Time Stress Monitoring Detection and Management in College Students.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Reproducible workflow for online AI in digital health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Scientific Evidence for Clinical Text Summarization Using Large.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Spatiotemporal Feature Fusion for.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Toward Reusable Science with Readable Code and.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Transfer Learning in Human Activity Recognition  A Survey.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Transforming Wearable Data into Personal Health.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/Wearable Artificial Intelligence for Detecting Anxiety.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76 papers/When Does Optimizing a Proper Loss Yield Calibration.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `papers/research_papers/76_papers_suggestions.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `papers/research_papers/76_papers_summarizzation.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `papers/research_papers/COMPREHENSIVE_RESEARCH_PAPERS_SUMMARY.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `papers/research_papers/EHB_2025_71_extracted.txt` | txt | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `papers/research_papers/Final_resorecs_paper_list.xlsx` | xlsx | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/research_papers/ICTH_16_extracted.txt` | txt | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `papers/research_papers/PApers.xlsx` | xlsx | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/research_papers/Research_Papers_Summary.xlsx` | xlsx | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `papers/research_papers/Summarize papers and create file.pdf` | pdf | Research paper or research asset | Manual reference | N/A | N/A |  |
| `PROJECT_GUIDE.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `scripts/__pycache__/build_training_baseline.cpython-313.pyc` | pyc | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `scripts/__pycache__/post_inference_monitoring.cpython-313.pyc` | pyc | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `src/Archived(prepare traning- production- conversion)/Archived  files.txt` | txt | NOT FOUND: add purpose | NOT FOUND: add producer/consumer | NOT FOUND | NOT FOUND |  |
| `src/Archived(prepare traning- production- conversion)/training_cv_experiment/label_garmin_data.py` | py | Python script/module | Imported or CLI | N/A | N/A |  |
| `src/config.py` | py | Path and constant configuration | Imported by src modules | N/A | N/A |  |
| `src/README.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `Summary_of_7_Research_Themes_in_HAR.csv` | csv | Data artifact | Used by pipeline scripts | N/A | N/A |  |
| `Thesis_Plan.md` | md | Documentation | Manual reference | N/A | N/A |  |
| `unnamed.jpg` | jpg | Image asset | Manual reference | N/A | N/A |  |

</details>

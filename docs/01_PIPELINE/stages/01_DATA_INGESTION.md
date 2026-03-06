# Stage 0: Data Ingestion

**Pipeline Stage:** Raw data acquisition and initial organization  
**Input:** Raw sensor files (FIT, Excel, CSV)  
**Output:** Organized raw files with consistent naming

---

## Key Questions

### Q1: What naming conventions should we use?

**Pattern:** `{YYYY-MM-DD}_{source}_{datatype}_{version}.{ext}`

**Examples:**
```
2025-03-23_garmin_accelerometer_raw.xlsx
2025-03-23_garmin_gyroscope_raw.xlsx
2025-03-23_garmin_fused_50Hz_v1.csv
```

**Rules:**
- ISO date prefix for chronological sorting
- Lowercase, underscores instead of spaces
- Version suffix for reprocessed files

---

### Q2: DVC vs Metadata JSON — Do we need both?

**YES, both are needed:**

| Aspect | DVC (.dvc files) | Metadata JSON |
|--------|------------------|---------------|
| **Primary purpose** | Version control large files | Record processing parameters |
| **What it stores** | File hash, remote storage path | Source files, params, QC results |
| **When to query** | Need to pull file or check version | Need to inspect processing details |
| **Searchable** | No (binary reference) | Yes (JSON is human-readable) |
| **Git tracked** | Yes (.dvc file) | Yes (JSON file) |

**Why both:**

1. **DVC alone is not enough because:**
   - You can't see processing parameters from a .dvc file
   - You need to `dvc pull` to inspect data
   - No record of which source files were used

2. **JSON alone is not enough because:**
   - No efficient storage for large files
   - No automatic deduplication
   - No remote storage management

**Recommendation:**
```
data/
├── raw/
│   ├── 2025-03-23_accelerometer.xlsx
│   ├── 2025-03-23_accelerometer.xlsx.dvc  ← DVC tracking
│   └── raw_manifest.json                   ← Metadata
├── prepared/
│   ├── production_X.npy
│   ├── production_X.npy.dvc               ← DVC tracking
│   ├── production_metadata.json           ← Processing params
│   └── baseline_stats.json                ← Training statistics
```

---

### Q3: What should the metadata JSON contain?

**For raw data (`raw_manifest.json`):**
```json
{
  "files": [
    {
      "filename": "2025-03-23_accelerometer.xlsx",
      "original_name": "accelerometer_data.xlsx",
      "ingested_at": "2026-01-15T10:30:00Z",
      "file_hash": "sha256:abc123...",
      "source_device": "garmin_venu3",
      "collection_date": "2025-03-23",
      "rows": 345418,
      "columns": ["timestamp", "x", "y", "z"]
    }
  ],
  "dvc_tracked": true,
  "ingestion_script": "scripts/ingest_raw_data.py",
  "ingestion_version": "1.0.0"
}
```

**For prepared data (`production_metadata.json`):**
```json
{
  "created_date": "2026-01-15T13:01:56Z",
  "source_files": ["sensor_fused_50Hz.csv"],
  "source_hash": "sha256:def456...",
  "preprocessing": {
    "target_hz": 50,
    "window_size": 200,
    "overlap": 0.5,
    "normalization": "standard_scaler",
    "scaler_path": "models/scaler.pkl",
    "gravity_removal": false
  },
  "output": {
    "shape": [2609, 200, 6],
    "dtype": "float32"
  },
  "qc_passed": true,
  "qc_report": "reports/preprocess_qc/qc_20260115.json"
}
```

---

## What to Do Checklist

- [ ] Create ingestion script that renames files to standard pattern
- [ ] Track all raw files with DVC (`dvc add data/raw/`)
- [ ] Generate raw_manifest.json on ingestion
- [ ] Store file hashes for provenance
- [ ] Log original → renamed file mapping
- [ ] Validate filename pattern before accepting

---

## Evidence from Papers

**[MLDEV: Data Science Experiment Automation | PDF: papers/research_papers/76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf]**
- DVC + metadata combination recommended for comprehensive tracking
- Metadata files enable quick inspection without pulling data

**[Toward Reusable Science with Readable Code | PDF: papers/research_papers/76 papers/Toward Reusable Science with Readable Code and.pdf]**
- Consistent naming conventions are essential for reproducibility
- Version suffixes prevent accidental overwrites

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add file hash validation on ingest | Low | Detect corrupted files |
| **HIGH** | Auto-generate manifest on `dvc add` | Medium | Ensure metadata always exists |
| **MEDIUM** | Add schema validation for raw files | Medium | Catch format issues early |
| **LOW** | Support additional file formats (Parquet) | Medium | Future-proof for larger datasets |

---

**Next Stage:** [02_PREPROCESSING_FUSION.md](02_PREPROCESSING_FUSION.md)

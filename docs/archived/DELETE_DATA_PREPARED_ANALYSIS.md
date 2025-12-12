# Data/Prepared Folder Analysis & Cleanup Guide

**Date:** December 12, 2025  
**Purpose:** Understand what files are in data/prepared and whether to keep or delete them

---

## 📁 Current Contents of data/prepared/

### Files Present:
```
data/prepared/
├── config.json                  (1.2 KB)  ← Scaler configuration
├── PRODUCTION_DATA_README.md    (3.5 KB)  ← Documentation
├── production_metadata.json     (2.8 KB)  ← Prepared data metadata
└── predictions/                           ← Subfolder (empty)
    └── (no files currently)
```

### What Each File Does:

#### 1. **config.json** ✅ KEEP
**Purpose:** Scaler parameters from model training  
**Contains:**
```json
{
  "scaler_mean": [3.22, 1.28, -3.53, 0.60, 0.23, 0.09],
  "scaler_scale": [6.57, 4.35, 3.24, ...]
}
```
**Why keep:** Needed for normalizing inference data to match training distribution  
**Size:** 1.2 KB (negligible)  
**Action:** ✅ **KEEP - DO NOT DELETE**

---

#### 2. **PRODUCTION_DATA_README.md** ✅ KEEP
**Purpose:** Documentation about prepared production data  
**Contains:** Metadata, description of columns, preprocessing steps applied  
**Why keep:** Useful reference for understanding data format  
**Size:** 3.5 KB (negligible)  
**Action:** ✅ **KEEP - DO NOT DELETE**

---

#### 3. **production_metadata.json** ⚠️ CAN DELETE (Will Regenerate)
**Purpose:** Metadata about prepared arrays  
**Contains:**
```json
{
  "total_windows": 1815,
  "window_size": 200,
  "overlap_percent": 50,
  "channels": 6,
  "scaler_applied": true,
  "timestamp": "2025-12-11T22:52:48"
}
```
**Why keep:** Useful for understanding how data was preprocessed  
**Size:** 2.8 KB (negligible)  
**Action:** ⚠️ **Optional - will be regenerated on next preprocessing run**

---

#### 4. **predictions/ (Folder)** ✅ EMPTY NOW - KEEP FOLDER
**Purpose:** Storage for prediction outputs  
**Contains:** Empty (will be filled after inference runs)  
**Why keep:** Empty folder needed for pipeline  
**Action:** ✅ **KEEP FOLDER - It's empty so no harm**

---

## ❓ About the "23 Different Files" You Mentioned

You mentioned seeing 23 different files in data/prepared. This likely refers to:

### Scenario 1: You saw all prediction files
If there were old prediction runs:
- `predictions_20251211_225312.csv` (predictions)
- `predictions_20251211_225312.json` (metadata)
- `predictions_20251211_225312_probs.npy` (probabilities)
- × ~7 different timestamp versions = 21 files

**Action:** ✅ **DELETE - These are from old runs**

```powershell
Remove-Item -Path "data/prepared/predictions/*.csv" -Force
Remove-Item -Path "data/prepared/predictions/*.json" -Force
Remove-Item -Path "data/prepared/predictions/*.npy" -Force
```

### Scenario 2: You saw old .npy array files
If there were old prepared arrays:
- `train_X.npy`, `train_y.npy` (training data)
- `val_X.npy`, `val_y.npy` (validation data)
- `test_X.npy`, `test_y.npy` (test data)
- `production_X.npy` (production data for inference)
- × various versions = 10-15 files

**Action:** ✅ **DELETE - These will be regenerated**

```powershell
Remove-Item -Path "data/prepared/*.npy" -Force
Remove-Item -Path "data/prepared/*.h5" -Force
```

### Scenario 3: Mix of everything
You might have seen:
- Old `.npy` files (prepared arrays)
- Old `.csv` files (predictions)
- `.json` metadata files
- Configuration files
- = 23 total files

---

## 🧹 Recommended Cleanup

### Minimal Clean (Keep configs, delete outputs)
```powershell
# Keep: config.json, PRODUCTION_DATA_README.md
# Delete: old .npy files, predictions, metadata

Remove-Item -Path "data/prepared/*.npy" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/*.h5" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/production_metadata.json" -Force -ErrorAction SilentlyContinue
```

### Aggressive Clean (Delete everything except configs)
```powershell
# Keep: config.json, PRODUCTION_DATA_README.md, predictions/ folder
# Delete: everything else

Get-ChildItem -Path "data/prepared" -File | 
    Where-Object { $_.Name -notmatch "(config\.json|PRODUCTION_DATA_README\.md)" } | 
    Remove-Item -Force

Get-ChildItem -Path "data/prepared/predictions" -Recurse | Remove-Item -Force
```

### Nuclear Option (Clean everything)
```powershell
# This will delete EVERYTHING in data/prepared
# Only use if you have backups or DVC remote

Remove-Item -Path "data/prepared" -Recurse -Force
mkdir data/prepared
mkdir data/prepared/predictions
```

---

## 📊 What Will Be Created on Fresh Run

After running the pipeline from scratch, you'll have:

```
data/prepared/
├── config.json                        (existing - keeps scaler config)
├── PRODUCTION_DATA_README.md          (existing - documentation)
├── production_metadata.json           (NEW - regenerated)
├── production_X.npy                   (NEW - 1815 windows × 200 samples × 6 channels)
└── predictions/
    ├── predictions_20251212_143022.csv        (NEW - activity predictions)
    ├── predictions_20251212_143022.json       (NEW - metadata)
    ├── predictions_20251212_143022_probs.npy (NEW - confidence scores)
    └── predictions_20251212_143022_metadata.json (NEW - summary)
```

**Total size after fresh run:** ~15-20 MB (depends on data size)

---

## ✅ Final Recommendation

**Safe Action (Recommended for Fresh Start):**

```powershell
# 1. Keep config files (needed for inference)
Write-Host "Keeping: config.json, PRODUCTION_DATA_README.md"

# 2. Delete all old outputs
Write-Host "Deleting old outputs..."
Remove-Item -Path "data/prepared/*.npy" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/*.h5" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/production_metadata.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/prepared/predictions/*.npy" -Force -ErrorAction SilentlyContinue

Write-Host "✅ Clean! Ready for fresh pipeline run"
```

---

## 🔄 After Fresh Run

Once you run the pipeline again:

```powershell
# The new files will be created automatically:
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate
python src/run_inference.py

# Result: data/prepared/ will be populated with fresh outputs
# Old files will be gone, new files with current timestamps will exist
```

---

## 📝 Summary Table

| File | Size | Keep? | Reason |
|------|------|-------|--------|
| config.json | 1.2 KB | ✅ YES | Scaler config needed for inference |
| PRODUCTION_DATA_README.md | 3.5 KB | ✅ YES | Documentation |
| production_metadata.json | 2.8 KB | ⚠️ OPTIONAL | Will regenerate |
| *.npy (old prepared data) | 10-50 MB | ❌ NO | Will regenerate on fresh run |
| predictions/*.csv | 1-5 MB | ❌ NO | Old prediction results |
| predictions/*.json | 500 KB | ❌ NO | Old metadata |
| predictions/*.npy | 5-10 MB | ❌ NO | Old probabilities |

---

**Bottom Line:** Delete all the `.npy` files and everything in `predictions/` folder. Keep `config.json` and the README. Everything else will be regenerated on your fresh run! 🚀

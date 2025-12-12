# 🗂️ Complete File System Map - What to Delete vs Keep

**For:** Understanding fresh start cleanup  
**Date:** December 12, 2025

---

## 📦 Full Directory Structure

```
MasterArbeit_MLops/
│
├── 📂 data/
│   ├── 📂 raw/                          ✅ KEEP (original sensor data)
│   │   ├── 2025-03-23-15-23-10-accelerometer_data.xlsx
│   │   ├── 2025-03-23-15-23-10-gyroscope_data.xlsx
│   │   └── raw.dvc                      ✅ KEEP (DVC pointer)
│   │
│   ├── 📂 preprocessed/                 ❌ DELETE (will regenerate)
│   │   ├── sensor_fused_50Hz.csv        ❌ DELETE
│   │   ├── sensor_merged_native_rate.csv ❌ DELETE
│   │   ├── sensor_fused_meta.json       ❌ DELETE
│   │   └── processed.dvc                ✅ KEEP (DVC pointer)
│   │
│   └── 📂 prepared/                     ⚠️ MIXED
│       ├── config.json                  ✅ KEEP (scaler config - CRITICAL!)
│       ├── PRODUCTION_DATA_README.md    ✅ KEEP (documentation)
│       ├── production_metadata.json     ❌ DELETE (will regenerate)
│       ├── production_X.npy             ❌ DELETE (will regenerate)
│       ├── prepared.dvc                 ✅ KEEP (DVC pointer)
│       └── 📂 predictions/
│           ├── predictions_*.csv        ❌ DELETE (old results)
│           ├── predictions_*.json       ❌ DELETE (old metadata)
│           ├── predictions_*_probs.npy  ❌ DELETE (old probabilities)
│           └── predictions_*_metadata.json ❌ DELETE (old metadata)
│
├── 📂 models/
│   ├── 📂 pretrained/                   ✅ KEEP (fine-tuned model)
│   │   ├── fine_tuned_model_1dcnnbilstm.keras
│   │   └── model_info.json
│   │
│   ├── 📂 trained/                      ❌ DELETE (if any, will recreate)
│   │   └── (custom trained models)
│   │
│   └── pretrained.dvc                   ✅ KEEP (DVC pointer)
│
├── 📂 src/                              ✅ KEEP (source code)
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess_data.py
│   ├── run_inference.py
│   ├── evaluate_predictions.py
│   └── (other Python files)
│
├── 📂 docker/                           ✅ KEEP (Docker files)
│   ├── Dockerfile.inference
│   ├── Dockerfile.training
│   └── api/
│       ├── __init__.py
│       └── main.py
│
├── 📂 config/                           ✅ KEEP (configuration)
│   ├── pipeline_config.yaml
│   ├── mlflow_config.yaml
│   ├── requirements.txt
│   └── .pylintrc
│
├── 📂 logs/                             ❌ DELETE (will regenerate)
│   ├── 📂 preprocessing/
│   │   └── *.log                        ❌ DELETE
│   ├── 📂 training/
│   │   └── *.log                        ❌ DELETE
│   ├── 📂 inference/
│   │   └── *.log                        ❌ DELETE
│   └── 📂 evaluation/
│       └── *.log                        ❌ DELETE
│
├── 📂 outputs/                          ❌ DELETE (will regenerate)
│   ├── 📂 evaluation/
│   │   ├── evaluation_20251208_*.json  ❌ DELETE
│   │   ├── evaluation_20251208_*.txt   ❌ DELETE
│   │   ├── evaluation_20251211_*.json  ❌ DELETE
│   │   └── evaluation_20251211_*.txt   ❌ DELETE
│   │
│   ├── 📂 predictions/
│   │   └── (old prediction files)      ❌ DELETE
│   │
│   └── 📂 analysis/
│       └── (old analysis files)        ❌ DELETE
│
├── 📂 research_papers/                  ✅ KEEP (reference datasets)
│   ├── anxiety_dataset.csv              ✅ KEEP (~50MB)
│   ├── anxiety_dataset.csv.dvc          ✅ KEEP (DVC pointer)
│   ├── all_users_data_labeled.csv       ✅ KEEP (~70MB)
│   ├── all_users_data_labeled.csv.dvc   ✅ KEEP (DVC pointer)
│   └── temp.ipynb                       ✅ KEEP (analysis notebook)
│
├── 📂 notebooks/                        ✅ KEEP (notebooks)
│   └── (jupyter notebooks)
│
├── 📂 docs/                             ✅ KEEP (documentation)
│   ├── FRESH_START_CLEANUP_GUIDE.md     ✅ KEEP (NEW!)
│   ├── FRESH_START_SUMMARY.md           ✅ KEEP (NEW!)
│   ├── DATA_PREPARED_ANALYSIS.md        ✅ KEEP (NEW!)
│   ├── QUICK_START_FRESH.md             ✅ KEEP (NEW!)
│   ├── PIPELINE_RERUN_GUIDE.md          ✅ KEEP (updated)
│   ├── PIPELINE_VISUALIZATION_PROMPTS.md ✅ KEEP
│   ├── MENTOR_EMAIL_FOLLOWUP.md         ✅ KEEP
│   └── (other docs)
│
├── 📂 scripts/                          ✅ KEEP
│   ├── complete_fresh_start.ps1         ✅ KEEP (NEW!)
│   ├── health_check.ps1                 ✅ KEEP (from PIPELINE_RERUN_GUIDE)
│   └── (other scripts)
│
├── 📂 .git/                             ✅ KEEP (Git repository)
│
├── 📂 .dvc/                             ✅ KEEP (DVC configuration)
│   ├── config                           ✅ KEEP (DVC settings)
│   └── cache/                           ⚠️ OPTIONAL (local cache)
│       └── (can clean with: dvc gc)
│
├── 📂 mlruns/                           ❌ DELETE (experiment history)
│   ├── 0/                               ❌ DELETE (MLflow runs)
│   └── (experiment data)
│
├── 📄 mlflow.db                         ❌ DELETE (MLflow database)
├── 📄 docker-compose.yml                ✅ KEEP (Docker config)
├── 📄 README.md                         ✅ KEEP (Main documentation)
├── 📄 .gitignore                        ✅ KEEP (Git ignore rules)
├── 📄 .dockerignore                     ✅ KEEP (Docker ignore rules)
├── 📄 .dvcignore                        ✅ KEEP (DVC ignore rules)
└── 📄 (other config files)              ✅ KEEP

```

---

## 📊 Storage Breakdown

### ❌ DELETE (Can Free ~100-150 MB)
```
outputs/evaluation/          ~2 MB    (8 evaluation files)
logs/                        ~10 MB   (old run logs)
data/preprocessed/           ~50 MB   (old CSVs)
data/prepared/*.npy          ~15 MB   (old arrays)
data/prepared/predictions/   ~10 MB   (old predictions)
mlruns/                      ~5 MB    (MLflow database)
mlflow.db                    ~1 MB    (MLflow DB)
─────────────────────────────────
Total deletable:             ~93 MB
```

### ✅ KEEP (Required for Fresh Run)
```
data/raw/                    ~60 MB   (original sensor data - DVC)
models/pretrained/           ~18 MB   (fine-tuned model - DVC)
research_papers/             ~120 MB  (reference datasets - DVC)
config/                      ~1 MB    (configuration files)
src/                         ~5 MB    (source code)
docker/                      ~2 MB    (Docker files)
docs/                        ~5 MB    (documentation)
.git/                        ~50 MB   (Git history)
.dvc/                        ~10 MB   (DVC config)
─────────────────────────────────
Total to keep:               ~271 MB
```

---

## 🎯 Cleanup Categories

### Category 1: Old Results (ALWAYS SAFE TO DELETE)
```
outputs/evaluation/*.json          ❌ DELETE
outputs/evaluation/*.txt           ❌ DELETE
data/prepared/predictions/*.csv    ❌ DELETE
data/prepared/predictions/*.json   ❌ DELETE
```
**Impact:** Removes old experiment results  
**Regenerates:** On next `python src/evaluate_predictions.py`  
**Loss:** None (results are timestamped, easy to track)

### Category 2: Old Logs (ALWAYS SAFE TO DELETE)
```
logs/preprocessing/*.log           ❌ DELETE
logs/training/*.log                ❌ DELETE
logs/inference/*.log               ❌ DELETE
logs/evaluation/*.log              ❌ DELETE
```
**Impact:** Removes debug/execution logs  
**Regenerates:** On next pipeline run  
**Loss:** None (new logs have same info)

### Category 3: Old Generated Data (SAFE WITH BACKUP)
```
data/preprocessed/*.csv            ❌ DELETE
data/preprocessed/*.json           ❌ DELETE
data/prepared/*.npy                ❌ DELETE
data/prepared/predictions/         ❌ DELETE
```
**Impact:** Removes preprocessed/prepared data  
**Regenerates:** On next `python src/preprocess_data.py` + `python src/run_inference.py`  
**Loss:** None (DVC backs up originals, can restore)

### Category 4: Experiment Tracking (PERMANENT LOSS)
```
mlruns/                            ❌ DELETE (PERMANENT!)
mlflow.db                          ❌ DELETE (PERMANENT!)
```
**Impact:** Removes ALL MLflow experiment history  
**Regenerates:** No (experiments are gone forever)  
**Loss:** Permanent (but can restart fresh)

---

## 🔐 Critical Files (NEVER DELETE!)

```
❌ DO NOT DELETE THESE UNDER ANY CIRCUMSTANCES:

data/raw/                              (original sensor data)
models/pretrained/                     (fine-tuned model)
data/prepared/config.json              (scaler configuration - CRITICAL!)
research_papers/                       (reference datasets)
src/                                   (source code)
.git/                                  (Git repository)
docker-compose.yml                     (Docker configuration)
README.md                              (Main documentation)
```

**If you accidentally delete critical files:**
```powershell
# Restore from Git
git restore <filename>

# Restore from DVC
dvc pull <file.dvc>
```

---

## 📋 Pre-Cleanup Verification

**Before deleting, verify these files exist:**
```powershell
# Check raw data (should be ~60MB)
ls -la data/raw/*.xlsx

# Check model (should be ~18MB)
ls -la models/pretrained/*.keras

# Check critical config
ls -la data/prepared/config.json

# Check Git is intact
git log --oneline | head -5

# Check DVC is intact
dvc status
```

---

## 🚀 Step-by-Step Fresh Start

### Step 1: Verify Before Deleting
```powershell
ls data/raw/*.xlsx                    # Should show 2 files
ls models/pretrained/*.keras          # Should show 1 file
ls data/prepared/config.json          # Should exist
git log | head -5                     # Should show commits
dvc status                            # Should be up to date
```

### Step 2: Delete Old Records
```powershell
.\scripts\complete_fresh_start.ps1    # ONE COMMAND!
```

### Step 3: Verify Cleanup
```powershell
ls outputs/evaluation/                # Should be empty
ls logs/preprocessing/                # Should be empty
ls data/prepared/predictions/         # Should be empty
ls data/prepared/*.npy                # Should be empty
```

### Step 4: Run Fresh Pipeline
```powershell
python src/sensor_data_pipeline.py
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate
python src/run_inference.py
python src/evaluate_predictions.py
```

### Step 5: Verify New Files
```powershell
ls data/preprocessed/sensor_fused_50Hz.csv    # Should exist (new)
ls data/prepared/predictions/predictions_*.csv # Should exist (new)
ls outputs/evaluation/evaluation_*.json        # Should exist (new)
```

---

## 📱 Reference During Cleanup

**Keep this window open while cleaning:**

**DELETE THESE:**
- ❌ `outputs/evaluation/` - Old reports
- ❌ `logs/` - Old logs
- ❌ `data/preprocessed/` - Old CSVs
- ❌ `data/prepared/*.npy` - Old arrays
- ❌ `data/prepared/predictions/` - Old predictions
- ❌ `mlruns/` - MLflow experiments
- ❌ `mlflow.db` - MLflow database

**KEEP THESE:**
- ✅ `data/raw/` - Original data
- ✅ `models/pretrained/` - Fine-tuned model
- ✅ `data/prepared/config.json` - Scaler config
- ✅ `data/prepared/PRODUCTION_DATA_README.md` - Docs
- ✅ `research_papers/` - Reference datasets
- ✅ `src/` - Source code
- ✅ `.git/` - Version control
- ✅ All `.dvc` files - DVC pointers

---

**Ready? Run:** `.\scripts\complete_fresh_start.ps1` 🚀

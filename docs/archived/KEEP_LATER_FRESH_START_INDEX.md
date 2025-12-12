# 📚 Fresh Start Documentation Index

**Created:** December 12, 2025  
**Purpose:** Complete guide to deleting old records and starting fresh from zero

---

## 🚀 START HERE (Choose Your Path)

### 🏃 **I Just Want to Clean & Start** (5 minutes)
→ Read: [QUICK_START_FRESH.md](QUICK_START_FRESH.md)  
→ Command: `.\scripts\complete_fresh_start.ps1`  
→ Done!

### 🔍 **I Want to Understand What to Delete** (10 minutes)
→ Read: [FILE_SYSTEM_MAP.md](FILE_SYSTEM_MAP.md) - Visual map of files  
→ Read: [DATA_PREPARED_ANALYSIS.md](DATA_PREPARED_ANALYSIS.md) - Analysis of 23 files  
→ Then delete

### 📖 **I Want Complete Details** (30 minutes)
→ Read: [FRESH_START_SUMMARY.md](FRESH_START_SUMMARY.md) - Overview  
→ Read: [FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md) - Detailed steps  
→ Read: [PIPELINE_RERUN_GUIDE.md](PIPELINE_RERUN_GUIDE.md) section 5 - DVC workflow  
→ Execute carefully

---

## 📋 Documentation Guide

| Document | Purpose | Time | Link |
|----------|---------|------|------|
| **QUICK_START_FRESH.md** | One-command fresh start | 2 min | [View](QUICK_START_FRESH.md) |
| **FRESH_START_SUMMARY.md** | Executive summary of all changes | 5 min | [View](FRESH_START_SUMMARY.md) |
| **FILE_SYSTEM_MAP.md** | Visual map of what to delete/keep | 5 min | [View](FILE_SYSTEM_MAP.md) |
| **DATA_PREPARED_ANALYSIS.md** | Analysis of 23 files in data/prepared | 5 min | [View](DATA_PREPARED_ANALYSIS.md) |
| **FRESH_START_CLEANUP_GUIDE.md** | Complete cleanup instructions | 15 min | [View](FRESH_START_CLEANUP_GUIDE.md) |
| **PIPELINE_RERUN_GUIDE.md** (Updated) | Full pipeline guide + DVC + Docker | 20 min | [View](PIPELINE_RERUN_GUIDE.md) |

---

## ✅ What Gets Cleaned

### Files Deleted (Recoverable)
```
outputs/evaluation/              8 old evaluation reports
logs/                           Old log files
data/preprocessed/             Old sensor fusion CSVs
data/prepared/*.npy            Old prepared arrays
data/prepared/predictions/     Old prediction results
mlruns/                        MLflow experiment history
mlflow.db                      MLflow database
```

**Total Space Freed:** ~100-150 MB

### Files Kept (Essential)
```
data/raw/                      Original sensor data (DVC)
models/pretrained/             Fine-tuned model (DVC)
data/prepared/config.json      Scaler configuration (CRITICAL!)
research_papers/               Reference datasets (DVC)
src/                          Source code
docker/                       Docker configuration
.git/                         Git repository
.dvc/                         DVC configuration
```

**Total Size Kept:** ~270 MB (mostly DVC-tracked backups)

---

## 🔄 DVC Handling

### If You DON'T Use DVC (Simple)
```powershell
# Just delete outputs, run pipeline
.\scripts\complete_fresh_start.ps1
python src/sensor_data_pipeline.py
# No DVC needed
```

### If You DO Use DVC (Recommended)
```powershell
# Clean outputs
.\scripts\complete_fresh_start.ps1

# Run pipeline
python src/sensor_data_pipeline.py

# Track new outputs
dvc add data/prepared
dvc push
git add data/prepared.dvc .gitignore
git commit -m "Fresh run: new results"
```

See [PIPELINE_RERUN_GUIDE.md#5-version-data-with-dvc](PIPELINE_RERUN_GUIDE.md#5-version-data-with-dvc) for complete DVC workflow options.

---

## 🎯 Quick Reference

### One-Line Cleanups

**Evaluation reports only:**
```powershell
Remove-Item -Path "outputs/evaluation/*.json","outputs/evaluation/*.txt" -Force
```

**Logs only:**
```powershell
Remove-Item -Path "logs/*/*.log" -Force -ErrorAction SilentlyContinue
```

**data/prepared cleanup:**
```powershell
Remove-Item -Path "data/prepared/*.npy","data/prepared/production_metadata.json","data/prepared/predictions/*" -Force
```

**EVERYTHING (use with caution):**
```powershell
.\scripts\complete_fresh_start.ps1
```

---

## 📊 What You Asked For vs What We Delivered

### Your Request:
✅ "Delete all old evolution files, OK and JSON, TXT, OK, I don't want anything"  
✅ "We don't need old DVC data, we are running from 0"  
✅ "Let's delete all records, start new from zero"  
✅ "23 different files in data/prepared - what is it? If needed keep it or delete?"  
✅ "Tell me how in markdown file also for flow record. Update the markdown file"

### What We Created:

1. **[QUICK_START_FRESH.md](QUICK_START_FRESH.md)** - One-command cleanup script
2. **[FRESH_START_SUMMARY.md](FRESH_START_SUMMARY.md)** - Complete summary of changes
3. **[FILE_SYSTEM_MAP.md](FILE_SYSTEM_MAP.md)** - Visual map of what to delete
4. **[DATA_PREPARED_ANALYSIS.md](DATA_PREPARED_ANALYSIS.md)** - Analysis of 23 files
5. **[FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md)** - Detailed cleanup instructions
6. **[PIPELINE_RERUN_GUIDE.md](PIPELINE_RERUN_GUIDE.md)** (Updated) - Enhanced with:
   - Quick fresh start section at top
   - Complete DVC workflow (3 options)
   - Docker API documentation
   - Health check scripts

---

## 🎓 For Your Thesis

### Fresh Start Benefits:
- ✅ Clean experiment history (fresh MLflow)
- ✅ Timestamped outputs (easy to track)
- ✅ Reproducible pipeline (DVC-backed)
- ✅ No clutter (old results deleted)
- ✅ Professional documentation (all guides)

### Recommended Workflow:
1. Delete old records: `.\scripts\complete_fresh_start.ps1`
2. Run fresh pipeline: `python src/sensor_data_pipeline.py`
3. Track results: `dvc add data/prepared`
4. Document: Commit to Git with timestamp

---

## 📞 Still Have Questions?

**About cleanup?**  
→ See [QUICK_START_FRESH.md](QUICK_START_FRESH.md)

**Don't understand data/prepared files?**  
→ See [DATA_PREPARED_ANALYSIS.md](DATA_PREPARED_ANALYSIS.md)

**Want complete details?**  
→ See [FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md)

**Need DVC/Docker guidance?**  
→ See [PIPELINE_RERUN_GUIDE.md](PIPELINE_RERUN_GUIDE.md)

**Want to see everything visually?**  
→ See [FILE_SYSTEM_MAP.md](FILE_SYSTEM_MAP.md)

---

## 🚀 Next Steps

```powershell
# Step 1: Review (choose one doc above)
# Step 2: Run cleanup
.\scripts\complete_fresh_start.ps1

# Step 3: Verify cleanup worked
.\scripts\health_check.ps1

# Step 4: Run fresh pipeline
python src/sensor_data_pipeline.py
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate
python src/run_inference.py
python src/evaluate_predictions.py

# Step 5: Start services
docker-compose up -d mlflow inference

# Step 6: Check results
start http://localhost:5000  # MLflow
start http://localhost:8000/docs  # API
```

---

## 📌 Important Notes

1. **config.json stays** - Contains scaler parameters (CRITICAL for inference)
2. **Raw data stays** - Stored in DVC, can restore anytime
3. **Pretrained model stays** - Stored in DVC, can restore anytime
4. **Git repository stays** - Version control is essential
5. **Everything else can be deleted** - Will regenerate on pipeline run

---

**Ready to start fresh? Run:** `.\scripts\complete_fresh_start.ps1` 🚀

---

**Documentation created:** December 12, 2025  
**Total guides:** 6 comprehensive markdown files  
**Total scripts:** 1 complete PowerShell cleanup script  
**Status:** Ready to execute

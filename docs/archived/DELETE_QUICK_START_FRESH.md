# 🚀 QUICK START: Fresh Pipeline Run from Zero

**Last Updated:** December 12, 2025

---

## ⚡ One-Command Fresh Start

```powershell
# Run this ONE command to clean everything and start fresh
.\scripts\complete_fresh_start.ps1
```

**What it does:**
- ❌ Deletes all old evaluation reports (8 files)
- ❌ Deletes all old logs  
- ❌ Deletes all old predictions
- ❌ Deletes all old preprocessed data
- ❌ Deletes MLflow experiment history
- ✅ Keeps raw data, pretrained model, reference datasets
- ✅ Keeps code, configs, Git repository

---

## 📋 Then Run Pipeline (3 Commands)

```powershell
# 1. Fuse sensors (raw Excel → CSV)
python src/sensor_data_pipeline.py

# 2. Prepare for inference (CSV → windowed arrays)
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate

# 3. Get predictions (arrays → activity predictions)
python src/run_inference.py

# 4. Evaluate results
python src/evaluate_predictions.py
```

**Expected output:**
- ✅ New predictions in: `data/prepared/predictions/predictions_<timestamp>.csv`
- ✅ Evaluation in: `outputs/evaluation/evaluation_<timestamp>.json`
- ✅ Logs in: `logs/preprocessing/`, `logs/inference/`, `logs/evaluation/`

---

## 🐳 Then Start Services

```powershell
# Start MLflow (experiment tracking)
docker-compose up -d mlflow
start http://localhost:5000

# Start Inference API
docker-compose up -d inference
start http://localhost:8000/docs
```

**Check status:**
```powershell
docker-compose ps
```

---

## 📊 What About data/prepared Files?

**Keep these:**
- ✅ `config.json` - Scaler parameters (needed for inference)
- ✅ `PRODUCTION_DATA_README.md` - Documentation

**Delete these (will regenerate):**
- ❌ `production_metadata.json` - Old metadata
- ❌ `*.npy` files - Old prepared arrays
- ❌ `predictions/` folder contents - Old predictions

**Run this to clean:**
```powershell
Remove-Item -Path "data/prepared/*.npy","data/prepared/production_metadata.json","data/prepared/predictions/*" -Force -ErrorAction SilentlyContinue
```

For complete analysis: See [DATA_PREPARED_ANALYSIS.md](DATA_PREPARED_ANALYSIS.md)

---

## 🔄 DVC Workflow (Optional)

**If you want to track new outputs with DVC:**

```powershell
# After fresh preprocessing, add to DVC
dvc add data/prepared
dvc push

# Commit to Git
git add data/prepared.dvc .gitignore
git commit -m "Fresh run: new prepared data"
git push
```

**If you DON'T want DVC tracking:**
Just delete outputs as described above and run fresh.

For complete DVC guide: See [FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md)

---

## ✅ Verification Checklist

After everything runs:

```powershell
# Check pipeline outputs exist
ls data/preprocessed/sensor_fused_50Hz.csv
ls data/prepared/predictions/predictions_*.csv
ls outputs/evaluation/evaluation_*.json

# Check services running
docker-compose ps

# Check MLflow has new experiment
start http://localhost:5000

# Check API is working
start http://localhost:8000/docs
```

---

## 📚 Full Documentation

- **Complete cleanup guide:** [FRESH_START_CLEANUP_GUIDE.md](FRESH_START_CLEANUP_GUIDE.md)
- **data/prepared analysis:** [DATA_PREPARED_ANALYSIS.md](DATA_PREPARED_ANALYSIS.md)
- **Full pipeline guide:** [PIPELINE_RERUN_GUIDE.md](PIPELINE_RERUN_GUIDE.md)
- **Docker & inspection:** [PIPELINE_RERUN_GUIDE.md#13](PIPELINE_RERUN_GUIDE.md#13-how-to-inspect-versioning-tracking-and-docker)

---

## 🆘 Troubleshooting

**Problem:** Docker services won't start
```powershell
docker-compose logs -f
# Check for missing models or data files
```

**Problem:** Pipeline runs but predictions are blank
```powershell
# Check if raw data exists
ls data/raw/*.xlsx

# Check if model exists
ls models/pretrained/*.keras

# Run with verbose logging
python src/run_inference.py --verbose
```

**Problem:** DVC keeps asking for credentials
```powershell
# Skip DVC tracking for now
# Just delete outputs manually instead
```

**Problem:** Need to recover deleted files
```powershell
# Restore from DVC
dvc pull

# Restore from Git
git restore <filepath>
```

---

**Ready? Run:** `.\scripts\complete_fresh_start.ps1` 🚀

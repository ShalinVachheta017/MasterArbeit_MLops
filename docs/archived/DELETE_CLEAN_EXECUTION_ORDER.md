# 🚀 CLEAN EXECUTION PLAN (Correct Order)

## Step 1: Delete Old Files FIRST ✅

```powershell
# Delete old evaluation files
Remove-Item -Path "outputs/evaluation/*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "outputs/evaluation/*.txt" -Force -ErrorAction SilentlyContinue

# Delete old logs
Remove-Item -Path "logs/*/*.log" -Force -Recurse -ErrorAction SilentlyContinue

# Delete old preprocessed data
Remove-Item -Path "data/preprocessed/*.csv" -Force -ErrorAction SilentlyContinue

# Delete old prepared arrays
Remove-Item -Path "data/prepared/*.npy" -Force -ErrorAction SilentlyContinue

# Delete old MLflow history
Remove-Item -Path "mlruns" -Force -Recurse -ErrorAction SilentlyContinue

Write-Host "✅ Cleanup complete!" -ForegroundColor Green
```

---

## Step 2: Verify MLflow is Ready ✅

```powershell
# Check MLflow can be reached (should be local by default)
Write-Host "Checking MLflow..." -ForegroundColor Cyan

# Verify we can access MLflow
mlflow --version

# Show MLflow backend storage
$env:MLFLOW_TRACKING_URI = "mlruns"
Write-Host "MLflow tracking URI: $env:MLFLOW_TRACKING_URI" -ForegroundColor Green
```

---

## Step 3: Run Fresh Pipeline (with MLflow tracking) ✅

```powershell
Write-Host "Starting fresh pipeline run with MLflow tracking..." -ForegroundColor Cyan
Write-Host ""

# 1. Sensor fusion
Write-Host "Step 1: Sensor data pipeline..." -ForegroundColor Yellow
python src/sensor_data_pipeline.py

# 2. Preprocessing with calibration
Write-Host "Step 2: Preprocessing with calibration..." -ForegroundColor Yellow
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate

# 3. Inference (NOW WITH MLFLOW!)
Write-Host "Step 3: Inference with MLflow tracking..." -ForegroundColor Yellow
python src/run_inference.py

# 4. Evaluation
Write-Host "Step 4: Evaluation..." -ForegroundColor Yellow
python src/evaluate_predictions.py

Write-Host ""
Write-Host "✅ Pipeline complete!" -ForegroundColor Green
```

---

## Step 4: Verify MLflow Shows Experiments ✅

```powershell
Write-Host "Verifying MLflow experiments..." -ForegroundColor Cyan

# Start MLflow UI in new terminal
Write-Host ""
Write-Host "Opening MLflow UI (in new terminal)..." -ForegroundColor Yellow
Write-Host "Command: mlflow ui" -ForegroundColor Green

# Show what to look for
Write-Host ""
Write-Host "Expected in MLflow:" -ForegroundColor Cyan
Write-Host "  ✅ Experiment: 'inference-production'" 
Write-Host "  ✅ Run name: 'inference_YYYYMMDD_HHMMSS'"
Write-Host "  ✅ Metrics: confidence stats, activity distribution"
Write-Host "  ✅ Artifacts: CSV prediction files"
```

---

## Quick Command Summary

```powershell
# ALL IN ONE (Copy & Paste)

# 1. Delete old files
Remove-Item -Path "outputs/evaluation/*.json", "outputs/evaluation/*.txt", `
                    "logs/*/*.log", "data/preprocessed/*.csv", `
                    "data/prepared/*.npy", "mlruns" -Force -Recurse -ErrorAction SilentlyContinue
Write-Host "✅ Cleanup done"

# 2. Run pipeline
python src/sensor_data_pipeline.py
python src/preprocess_data.py --input data/preprocessed/sensor_fused_50Hz.csv --calibrate
python src/run_inference.py
python src/evaluate_predictions.py
Write-Host "✅ Pipeline done"

# 3. Check MLflow
mlflow ui
# Then open: http://localhost:5000
```

---

## Checklist Before Running

- [ ] Old evaluation files deleted
- [ ] Old logs deleted
- [ ] Old preprocessed data deleted
- [ ] Old prepared .npy files deleted
- [ ] MLflow mlruns/ folder deleted (to start fresh)
- [ ] Python environment activated
- [ ] Dependencies installed (`pip install -r config/requirements.txt`)

---

## Checklist After Running

- [ ] sensor_fused_50Hz.csv created ✅
- [ ] Prepared data (.npy) created ✅
- [ ] Predictions CSV created ✅
- [ ] Evaluation reports created ✅
- [ ] MLflow UI opens without errors ✅
- [ ] "inference-production" experiment visible ✅
- [ ] Run metrics showing (confidence, activity distribution) ✅

---

## If MLflow Shows Nothing

**Possible issues:**
1. ❌ MLflow server not running → Run `mlflow ui` in new terminal
2. ❌ Port 5000 in use → Use different port: `mlflow ui --port 5001`
3. ❌ MLflow tracking code not working → Check [src/run_inference.py](../src/run_inference.py) has MLflow imports
4. ❌ Old mlruns corrupted → Delete entire `mlruns/` folder and re-run

**To verify MLflow code is there:**
```powershell
grep -n "mlflow.set_experiment" src/run_inference.py
# Should show match (confirms code is in file)
```

---

**Order: DELETE → VERIFY MLFLOW → RUN PIPELINE → CHECK MLFLOW UI** ✅

# 🎯 VIEW YOUR MLFLOW EXPERIMENTS NOW!

## ⚡ Quick Start (2 steps)

### Step 1: Open NEW PowerShell Terminal
```powershell
# In a NEW terminal (don't use the one with the pipeline run)
cd "D:\study apply\ML Ops\MasterArbeit_MLops"
```

### Step 2: Start MLflow UI
```powershell
mlflow ui
```

**Output should say:**
```
INFO: Uvicorn running on http://127.0.0.1:5000
```

### Step 3: Open in Browser
```
http://localhost:5000
```

---

## ✅ WHAT YOU'LL SEE

### Experiment List:
```
inference-production          ← Your experiment
├─ Run: 2025-12-12 17:51:05  ← Your pipeline run
│  ├─ Duration: 4 seconds
│  ├─ Metrics:
│  │  ├─ model_params: 499131
│  │  ├─ n_windows: 1815
│  │  ├─ avg_confidence: 0.52
│  │  ├─ count_forehead_rubbing: 1799
│  │  ├─ count_nape_rubbing: 9
│  │  └─ ... (more metrics)
│  └─ Artifacts:
│     └─ predictions_20251212_175115.csv ✅
```

---

## 📊 METRICS YOU'LL SEE

| Metric | Value | Meaning |
|--------|-------|---------|
| `model_params` | 499,131 | Model size |
| `n_windows` | 1,815 | Windows processed |
| `timesteps` | 200 | Time points per window |
| `channels` | 6 | Sensor channels |
| `avg_confidence` | 0.52 | Average prediction confidence |
| `std_confidence` | 0.27 | Confidence variation |
| `count_forehead_rubbing` | 1,799 | Most common activity |
| `count_nape_rubbing` | 9 | Second activity |
| `count_standing` | 2 | Other activities |

---

## 🎁 ARTIFACTS YOU'LL SEE

**Prediction Files:**
```
predictions_20251212_175115.csv
├─ window_id: 0, 1, 2, ...
├─ predicted_activity: forehead_rubbing
├─ confidence: 0.52, 0.48, ...
└─ confidence_level: UNCERTAIN, LOW, ...
```

**Metadata:**
```
predictions_20251212_175115_metadata.json
├─ total_windows: 1815
├─ uncertain_count: 952
├─ avg_confidence: 0.52
└─ activity_distribution: {...}
```

**Probabilities:**
```
predictions_20251212_175115_probs.npy
└─ (1815, 11) matrix of class probabilities
```

---

## 🖼️ UI Layout

```
Top Navigation:
  [Experiments] [Runs] [Compare Runs]

Left Sidebar:
  ✅ Experiments
    └─ inference-production (ACTIVE)
       └─ Runs (1)
          └─ 2025-12-12 17:51:05 (Success)

Main Panel:
  Run Overview:
  ├─ Status: Completed ✅
  ├─ Start time: 2025-12-12 17:51:05
  ├─ Duration: 4 seconds
  ├─ Parameters (8): model_path, batch_size, ...
  ├─ Metrics (10): avg_confidence, std_confidence, ...
  └─ Artifacts (3): CSV, JSON, NPY files

Tabs:
  ├─ Overview (current)
  ├─ Metrics (line charts)
  ├─ Artifacts (download files)
  └─ Metadata
```

---

## 🔗 URLS

| Page | URL |
|------|-----|
| Home | http://localhost:5000 |
| Experiments | http://localhost:5000/experiments |
| Your Experiment | http://localhost:5000/experiments/950614147457743858 |
| Your Run | http://localhost:5000/experiments/950614147457743858/runs/63f4a91bc5924b5cafb4bcb028f69d6b |

---

## 🎯 THINGS TO EXPLORE

### 1. View Metrics Over Time
- Click "Metrics" tab
- See confidence distribution
- View activity counts as bar chart

### 2. Download Artifacts
- Click "Artifacts" tab
- Download CSV file
- Import to Excel or Python

### 3. Compare Multiple Runs
- Run pipeline again (tomorrow)
- Both runs appear in list
- Compare metrics side-by-side

### 4. Export Data
- Right-click on run
- Export to CSV
- Share with mentor

---

## 🆘 TROUBLESHOOTING

### MLflow UI Won't Open

**Problem:** Browser shows "can't reach localhost:5000"

**Solution:**
```powershell
# Check if port 5000 is busy
netstat -ano | grep :5000

# If busy, use different port
mlflow ui --port 5001
# Then open: http://localhost:5001
```

### No Experiments Shown

**Problem:** "Experiments" page is empty

**Solution:**
```powershell
# Verify mlruns folder exists
ls mlruns/

# If empty, re-run pipeline
python src/run_inference.py
```

### Artifacts Not Showing

**Problem:** Artifacts tab is empty

**Solution:**
```powershell
# Check artifacts were saved
ls data/prepared/predictions/

# If empty, pipeline didn't complete
# Re-run and check for errors
```

---

## 📱 MOBILE ACCESS

To view MLflow from phone/tablet:

```powershell
# Find your computer's IP
ipconfig

# Get IPv4 address (e.g., 192.168.1.100)

# Start MLflow with host binding
mlflow ui --host 0.0.0.0 --port 5000

# From phone, open:
http://192.168.1.100:5000
```

---

## 💾 BACKUP YOUR EXPERIMENTS

```powershell
# MLflow data is in mlruns/ folder
# To backup:
Copy-Item -Path "mlruns" -Destination "mlruns_backup_20251212" -Recurse

# To restore:
Copy-Item -Path "mlruns_backup_20251212" -Destination "mlruns" -Recurse
```

---

## 🔄 RUN PIPELINE AGAIN

To track new experiments:

```powershell
# This will create NEW run automatically
python src/sensor_data_pipeline.py
python src/preprocess_data.py --calibrate
python src/run_inference.py
python src/evaluate_predictions.py

# New run appears in MLflow UI automatically ✅
```

---

## 📈 NEXT STEPS

1. ✅ MLflow UI open
2. → View your experiment
3. → Check metrics and artifacts
4. → Download predictions CSV
5. → Share results with mentor
6. → Run again tomorrow for trend tracking

---

**🎉 Your experiments are now being tracked in MLflow!**

**Next Command:**
```powershell
mlflow ui
```

**Then Open:**
```
http://localhost:5000
```

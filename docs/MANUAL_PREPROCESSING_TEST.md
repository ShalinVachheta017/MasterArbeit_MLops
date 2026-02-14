# Manual Preprocessing Testing Guide

## Quick Overview

Test your preprocessing pipeline manually by running 10 different recording sessions both **with** and **without** the `--gravity-removal` flag, then compare the results.

---

## 📋 Step-by-Step Instructions

### **Step 1: List Available Datasets**

See which recording sessions you have:

```powershell
cd "D:\study apply\ML Ops\MasterArbeit_MLops"
Get-ChildItem data\raw\*_accelerometer.csv | Select-Object -First 20 Name
```

You should see files like:
- `2025-07-16-21-03-13_accelerometer.csv`
- `2025-07-17-13-49-10_accelerometer.csv`
- `2025-08-11-12-33-33_accelerometer.csv`
- etc.

### **Step 2: Pick 10 Sessions to Test**

Choose 10 different session IDs (the timestamp part). For example:
```
1. 2025-07-16-21-03-13
2. 2025-07-17-13-49-10
3. 2025-07-17-21-42-30
4. 2025-07-21-13-58-56
5. 2025-07-25-12-15-48
6. 2025-08-01-12-52-49
7. 2025-08-11-12-33-33
8. 2025-08-12-12-09-08
9. 2025-08-13-04-18-52
10. 2025-08-19-08-58-40
```

### **Step 3: Isolate Each Session**

For **each session**, you need to temporarily move other data files out of the way:

```powershell
# Create backup and temp folders
mkdir "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw_backup" -ErrorAction SilentlyContinue
mkdir "D:\study apply\ML Ops\MasterArbeit_MLops\data\temp_session" -ErrorAction SilentlyContinue

# Move all files to backup
Move-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw\*" "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw_backup\"

# Copy ONLY the session you want to test back to raw/
$SESSION = "2025-07-16-21-03-13"  # CHANGE THIS for each test
Copy-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw_backup\${SESSION}_*" "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw\"
```

### **Step 4: Test WITHOUT Preprocessing**

```powershell
# Clean previous run artifacts
Remove-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\processed" -Recurse -ErrorAction SilentlyContinue
Remove-Item "D:\study apply\ML Ops\MasterArbeit_MLops\artifacts" -Recurse -ErrorAction SilentlyContinue

# Run pipeline WITHOUT preprocessing
cd "D:\study apply\ML Ops\MasterArbeit_MLops"
python run_pipeline.py

# Save the log
$SESSION = "2025-07-16-21-03-13"  # Same as above
$LATEST_LOG = Get-ChildItem "logs\pipeline\pipeline_result_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $LATEST_LOG "logs\pipeline\${SESSION}_BEFORE.log"
```

### **Step 5: Test WITH Preprocessing**

```powershell
# Clean previous run artifacts
Remove-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\processed" -Recurse -ErrorAction SilentlyContinue
Remove-Item "D:\study apply\ML Ops\MasterArbeit_MLops\artifacts" -Recurse -ErrorAction SilentlyContinue

# Run pipeline WITH preprocessing
python run_pipeline.py --gravity-removal

# Save the log
$SESSION = "2025-07-16-21-03-13"  # Same as above
$LATEST_LOG = Get-ChildItem "logs\pipeline\pipeline_result_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $LATEST_LOG "logs\pipeline\${SESSION}_AFTER.log"
```

### **Step 6: Restore All Data**

After testing each session, restore all the data files:

```powershell
# Remove test session files
Remove-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw\*"

# Move all files back
Move-Item "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw_backup\*" "D:\study apply\ML Ops\MasterArbeit_MLops\data\raw\"
```

### **Step 7: Repeat for All 10 Sessions**

Repeat Steps 3-6 for each of your 10 selected sessions.

---

## 📊 Analyzing Results

After testing all 10 sessions, you'll have 20 log files:
```
logs/pipeline/
    2025-07-16-21-03-13_BEFORE.log
    2025-07-16-21-03-13_AFTER.log
    2025-07-17-13-49-10_BEFORE.log
    2025-07-17-13-49-10_AFTER.log
    ...
```

### **What to Look For in Each Log**

Open each pair of BEFORE/AFTER logs and extract these metrics:

| Metric | Where to Find in Log | Before | After |
|--------|---------------------|--------|-------|
| **Windows** | Stage 3: "Total Windows : N" | ? | ? |
| **Mean Confidence** | Stage 4: "Mean Confidence : X.XXXX" | ? | ? |
| **Uncertain Ratio** | Stage 6: "Uncertain Ratio : X.XX%" | ? | ? |
| **Flip Rate** | Stage 6: "Flip Rate : X.XX%" | ? | ? |
| **Top Activity** | Stage 4: Activity distribution (highest bar) | ? | ? |

### **Example Comparison**

**Session: 2025-07-16-21-03-13**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Windows | 1,603 | 18,636 | +17,033 |
| Mean Confidence | 0.7484 | 0.9788 | +0.2304 (+30.8%) |
| Uncertain Ratio | 10.12% | 0.10% | -10.02% |
| Flip Rate | 36.82% | 0.60% | -36.22% |
| Top Activity | smoking (40.7%) | hand_tapping (99.7%) | — |

---

## 🎯 Creating Your Comparison Table

Create a spreadsheet or markdown table with all 10 sessions:

```markdown
# Preprocessing Validation Results

Tested 10 recording sessions (July-August 2025) to validate preprocessing improvements.

## Summary Statistics

| Session | Before Conf | After Conf | Improvement |
|---------|-------------|------------|-------------|
| 2025-07-16-21-03-13 | 0.7484 | 0.9788 | +30.8% |
| 2025-07-17-13-49-10 | 0.XXXX | 0.XXXX | +XX.X% |
| ... | ... | ... | ... |
| **Average** | **0.XXXX** | **0.XXXX** | **+XX.X%** |

## Conclusion

- ✅ Improved: X/10 sessions
- ⚠️ Degraded: Y/10 sessions
- ➖ Unchanged: Z/10 sessions

Preprocessing shows consistent improvement across XX% of tested sessions,
validating that it should be **mandatory for production deployment**.
```

---

## 🔧 Faster Testing Script

If you want to automate the repetitive parts, create a file `test_one_session.ps1`:

```powershell
param(
    [string]$SessionID
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing Session: $SessionID" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Backup and isolate
Write-Host "[1/5] Isolating session..." -ForegroundColor Yellow
if (-not (Test-Path "data\raw_backup")) {
    Move-Item "data\raw\*" "data\raw_backup\" -Force
}
Copy-Item "data\raw_backup\${SessionID}_*" "data\raw\" -Force

# Test WITHOUT preprocessing
Write-Host "[2/5] Testing WITHOUT preprocessing..." -ForegroundColor Yellow
Remove-Item "data\processed" -Recurse -ErrorAction SilentlyContinue
Remove-Item "artifacts" -Recurse -ErrorAction SilentlyContinue
python run_pipeline.py | Out-Null
$latestLog = Get-ChildItem "logs\pipeline\pipeline_result_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $latestLog "logs\pipeline\${SessionID}_BEFORE.log" -Force
Write-Host "   ✓ Saved: ${SessionID}_BEFORE.log" -ForegroundColor Green

# Test WITH preprocessing
Write-Host "[3/5] Testing WITH preprocessing..." -ForegroundColor Yellow
Remove-Item "data\processed" -Recurse -ErrorAction SilentlyContinue
Remove-Item "artifacts" -Recurse -ErrorAction SilentlyContinue
python run_pipeline.py --gravity-removal | Out-Null
$latestLog = Get-ChildItem "logs\pipeline\pipeline_result_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $latestLog "logs\pipeline\${SessionID}_AFTER.log" -Force
Write-Host "   ✓ Saved: ${SessionID}_AFTER.log" -ForegroundColor Green

# Restore data
Write-Host "[4/5] Restoring all data..." -ForegroundColor Yellow
Remove-Item "data\raw\*" -Force
Move-Item "data\raw_backup\*" "data\raw\" -Force

Write-Host "[5/5] Session complete!`n" -ForegroundColor Green
```

Then test each session by running:

```powershell
.\test_one_session.ps1 -SessionID "2025-07-16-21-03-13"
.\test_one_session.ps1 -SessionID "2025-07-17-13-49-10"
# ... etc for all 10 sessions
```

---

## 📁 File Organization

After testing, organize your results:

```
logs/pipeline/
├── preprocessing_tests_2026-02/
│   ├── 01_2025-07-16-21-03-13_BEFORE.log
│   ├── 01_2025-07-16-21-03-13_AFTER.log
│   ├── 02_2025-07-17-13-49-10_BEFORE.log
│   ├── 02_2025-07-17-13-49-10_AFTER.log
│   └── ...
└── PREPROCESSING_RESULTS.md  (your comparison table)
```

---

## 💡 Tips

1. **Test chronologically** - Start with earliest sessions (July), then work towards latest (August)
2. **Check for failures** - If a pipeline run fails, note it and check why in the log
3. **Look for patterns** - Do newer sessions perform better? Worse? Why?
4. **Document anomalies** - If one session shows unusual results, investigate it separately
5. **Save timestamps** - Each test takes ~30-60 seconds, so 10 sessions = ~10-20 minutes total

---

## 🎓 For Your Thesis

Use the comparison table in your **Results** chapter to show:
- Preprocessing improves confidence by X% on average
- Y out of 10 sessions showed improvement
- Uncertain predictions dropped by Z% on average

This provides **statistical evidence** that preprocessing is essential for production deployment.

---

## ❓ Questions?

- **Why isolate sessions?** - The ingestion component automatically picks the latest file pair. To test a specific session, you need to temporarily hide the others.
- **Why clean artifacts?** - Each pipeline run creates processed data and artifacts. Cleaning ensures each test starts fresh.
- **Can I test more than 10?** - Yes! But 10 is a good balance between statistical validity and time investment.

---

**Next Steps:**
1. Pick your 10 sessions
2. Run the tests (manual or with the PowerShell script)
3. Extract metrics from logs
4. Create comparison table
5. Use results in thesis

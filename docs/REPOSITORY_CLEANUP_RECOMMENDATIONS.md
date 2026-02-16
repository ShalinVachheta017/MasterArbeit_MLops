# 🧹 Repository Cleanup Recommendations

**Date:** February 16, 2026  
**Current Repo Size:** ~12GB total, ~3GB tracked in git  
**Goal:** Reduce git-tracked size, improve repo cleanliness

---

## 📊 Current Size Analysis

| Directory | Size | Status | Recommendation |
|-----------|------|--------|----------------|
| `data/` | 7.5GB | ✅ Mostly in .gitignore | Keep (DVC tracked) |
| `artifacts/` | 2.5GB | ✅ In .gitignore | ⚠️ Clean old runs |
| `archive/` | 1.2GB | ✅ In .gitignore | ✅ Keep as is |
| `.dvc_storage/` | 346MB | ✅ In .gitignore | ✅ Keep (DVC cache) |
| `.mypy_cache/` | 178MB | ✅ In .gitignore | ❌ Delete (auto-generated) |
| `mlruns/` | 71MB | ✅ In .gitignore | ⚠️ Clean old runs |
| `logs/` | 13MB | ❌ Partially tracked | ⚠️ Move old logs to archive |
| `outputs/` | 24MB | ✅ In .gitignore | ⚠️ Clean old predictions |
| `reports/` | 0.7MB | ✅ In .gitignore | ⚠️ Clean old reports |

**Total Reclaimable Space:** ~2.8GB (from artifacts, logs, outputs)  
**Git-tracked Size:** ~150MB (mostly code, docs, configs)

---

## 🎯 Action Items

### 💯 Priority 1: DELETE NOW (Auto-Generated Caches)

These are temporary caches that can be regenerated:

```bash
# Safe to delete - will be regenerated
rm -rf .mypy_cache/
rm -rf .pytest_cache/
rm -rf src/__pycache__/
rm -rf tests/__pycache__/
rm -rf scripts/__pycache__/
```

**Savings:** ~180MB

---

### 🔥 Priority 2: CLEAN OLD ARTIFACTS (Keep Recent Only)

#### A. Artifacts Directory (2.5GB!)

**Problem:** 45 timestamped directories from Feb 14-15 test runs

```bash
# Keep only the 5 most recent, delete the rest
cd artifacts/
ls -lt | head -6  # View newest 5

# Delete old artifacts (older than 2 days)
Get-ChildItem -Directory | Where-Object { $_.CreationTime -lt (Get-Date).AddDays(-2) } | Remove-Item -Recurse -Force
```

**Savings:** ~2GB (keep ~500MB of recent artifacts)

#### B. MLflow Runs (71MB)

**Current:** 5+ experiment runs tracked in `mlruns/`

```bash
# Keep only recent experiments, archive old ones
# Option 1: Use MLflow UI to delete old experiments
# Option 2: Move old runs to archive

mkdir -p archive/mlruns_archive/
mv mlruns/0/[old-run-ids] archive/mlruns_archive/
```

**Savings:** ~40MB

#### C. Logs Directory (13MB)

**Problem:** 33 log files from Feb 14 testing

```bash
# Move old logs to archive, keep only last 5
cd logs/
mkdir -p ../archive/logs_feb14_2026/

# Move logs older than 2 days
Get-ChildItem *.log | Where-Object { $_.CreationTime -lt (Get-Date).AddDays(-2) } | Move-Item -Destination ../archive/logs_feb14_2026/

# Or compress them
Compress-Archive -Path *.log -DestinationPath ../archive/logs_feb14_2026.zip
rm *.log
```

**Savings:** ~12MB

---

### 📦 Priority 3: ARCHIVE OLD OUTPUTS

#### A. Outputs Directory (24MB)

```bash
# Move old predictions to archive
cd outputs/
mkdir -p ../archive/predictions_feb_2026/

# Move old prediction files
mv predictions_20260212*.csv ../archive/predictions_feb_2026/
mv predictions_20260212*.json ../archive/predictions_feb_2026/
mv predictions_20260212*.npy ../archive/predictions_feb_2026/

# Keep only production_* files and recent analysis
```

**Savings:** ~15MB

#### B. Reports Directory (0.7MB)

```bash
# Archive old monitoring reports
cd reports/
mkdir -p ../archive/reports_feb_2026/

# Move old reports
mv inference_smoke/ ../archive/reports_feb_2026/ 2>/dev/null
mv monitoring/ ../archive/reports_feb_2026/ 2>/dev/null
```

**Savings:** ~0.7MB

---

### 🗂️ Priority 4: ROOT DIRECTORY CLEANUP

#### Files to Move to Archive

These are one-off scripts/CSVs that are no longer actively used:

```bash
# Move to archive
mv batch_process_all_datasets.py archive/scripts_feb2026/
mv generate_summary_report.py archive/scripts_feb2026/
mv Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv archive/docs_feb2026/
mv Summary_of_7_Research_Themes_in_HAR.csv archive/docs_feb2026/
```

**Why:** These were useful during development but aren't part of the core pipeline anymore.

**Keep in root:**
- `run_pipeline.py` ✅ (main entry point)
- `setup.py` ✅ (installation)
- `pyproject.toml` ✅ (config)
- `pytest.ini` ✅ (testing)
- `docker-compose.yml` ✅ (deployment)
- `README.md` ✅ (documentation)
- `Thesis_Plan.md` ✅ (thesis)

---

### 📚 Priority 5: NOTEBOOKS CLEANUP

#### Current State

```
notebooks/
├── data_preprocessing_step1.ipynb
├── from_guide_processing.ipynb
├── production_preprocessing.ipynb
├── exploration/
```

**Recommendation:**

```bash
# Move old dev notebooks to archive
cd notebooks/
mkdir -p ../archive/notebooks_dev/

# Move superseded notebooks
mv data_preprocessing_step1.ipynb ../archive/notebooks_dev/
mv from_guide_processing.ipynb ../archive/notebooks_dev/

# Keep only production_preprocessing.ipynb and exploration/
```

---

## ✅ What to KEEP in Repo

### Essential Files (Git-Tracked)

- ✅ `src/` - All source code
- ✅ `tests/` - All test files
- ✅ `docs/` - All documentation
- ✅ `config/` - Configuration files
- ✅ `docker/` - Dockerfiles
- ✅ `.github/workflows/` - CI/CD
- ✅ `scripts/` - Utility scripts (production-ready only)
- ✅ `*.dvc` files - DVC pointers
- ✅ `.dvc/config` - DVC config

### Essential Directories (Not Git-Tracked, Keep Locally)

- ✅ `data/` - DVC tracked, in .gitignore ✅
- ✅ `.dvc_storage/` - DVC cache ✅
- ✅ `models/` - DVC tracked (keep pretrained/) ✅
- ✅ `archive/` - Historical reference ✅
- ✅ Last 5 recent artifacts in `artifacts/` ✅
- ✅ Last 5 recent logs in `logs/` ✅

---

## 🚀 Cleanup Script (PowerShell)

Here's a script to automate the cleanup:

```powershell
# Repository Cleanup Script
# Run from repo root: .\cleanup_repo.ps1

Write-Host "🧹 Starting Repository Cleanup..." -ForegroundColor Green

# 1. Delete auto-generated caches
Write-Host "`n1. Removing caches..." -ForegroundColor Yellow
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .mypy_cache, .pytest_cache

# 2. Clean old artifacts (keep 5 newest)
Write-Host "`n2. Cleaning old artifacts..." -ForegroundColor Yellow
cd artifacts
$keep = Get-ChildItem -Directory | Sort-Object CreationTime -Descending | Select-Object -First 5
$remove = Get-ChildItem -Directory | Where-Object { $_.Name -notin $keep.Name }
$remove | ForEach-Object { 
    Write-Host "  Removing: $($_.Name)" -ForegroundColor Gray
    Remove-Item -Recurse -Force $_.FullName 
}
cd ..

# 3. Archive old logs
Write-Host "`n3. Archiving old logs..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "archive/logs_feb2026" | Out-Null
Get-ChildItem logs/*.log | Where-Object { $_.CreationTime -lt (Get-Date).AddDays(-2) } | 
    Move-Item -Destination "archive/logs_feb2026/"

# 4. Move root scripts to archive
Write-Host "`n4. Moving root scripts to archive..." -ForegroundColor Yellow
Move-Item -Force -ErrorAction SilentlyContinue batch_process_all_datasets.py archive/scripts_feb2026/
Move-Item -Force -ErrorAction SilentlyContinue generate_summary_report.py archive/scripts_feb2026/

# 5. Move CSVs to archive
Write-Host "`n5. Moving research CSVs to archive..." -ForegroundColor Yellow
Move-Item -Force -ErrorAction SilentlyContinue Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv archive/docs_feb2026/
Move-Item -Force -ErrorAction SilentlyContinue Summary_of_7_Research_Themes_in_HAR.csv archive/docs_feb2026/

# 6. Clean outputs
Write-Host "`n6. Archiving old outputs..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "archive/predictions_feb2026" | Out-Null
Get-ChildItem outputs/predictions_20260212* | Move-Item -Destination "archive/predictions_feb2026/"

# 7. Summary
Write-Host "`n✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "`nSpace freed:" -ForegroundColor Cyan
$beforeSize = 12000  # MB (estimated)
$afterSize = 9000    # MB (estimated)
Write-Host "  Before: ~$beforeSize MB" -ForegroundColor Gray
Write-Host "  After:  ~$afterSize MB" -ForegroundColor Gray
Write-Host "  Saved:  ~$($beforeSize - $afterSize) MB" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Verify changes: git status"
Write-Host "  2. Test pipeline: pytest tests/"
Write-Host "  3. Commit if all looks good: git add -A && git commit -m 'chore: cleanup old artifacts and logs'"
```

---

## 🔍 Verification Checklist

After cleanup, verify:

```bash
# 1. Check git status (should be clean)
git status

# 2. Run tests (everything should still work)
pytest tests/

# 3. Check pipeline works
python run_pipeline.py --help

# 4. Verify DVC still works
dvc status

# 5. Check Docker builds
docker-compose build

# 6. Verify CI/CD (push and watch GitHub Actions)
git push origin main
```

---

## 📋 Summary

### Space Savings Breakdown

| Action | Space Saved |
|--------|-------------|
| Delete caches (.mypy, pytest) | ~180MB |
| Clean old artifacts (keep 5) | ~2GB |
| Archive old logs | ~12MB |
| Archive old outputs | ~15MB |
| Clean MLflow runs | ~40MB |
| **Total Savings** | **~2.25GB** |

### Final Repository Size

- **Before:** ~12GB total
- **After:** ~9.75GB total
- **Git-tracked:** ~150MB (mostly code/docs)

---

## ⚠️ Important Notes

1. **Don't delete `data/`** - This is DVC tracked and essential
2. **Don't delete `.dvc_storage/`** - This is your DVC cache
3. **Don't delete `archive/`** - Historical reference for thesis
4. **Keep recent artifacts** - Last 5 for debugging
5. **Verify before committing** - Run tests after cleanup

---

## 🎓 Thesis Context

**Why this matters for your thesis:**

1. **Clean repo = Professional presentation**
   - Reviewers will browse your code
   - Clean structure shows good software engineering

2. **Faster CI/CD**
   - Less data = faster GitHub Actions
   - Smaller clones for reviewers

3. **Clear separation**
   - Working code in git
   - Data in DVC
   - History in archive

4. **Documentation**
   - Shows you understand DevOps best practices
   - Demonstrates repo hygiene

---

## 📝 Commit Message After Cleanup

```bash
git add -A
git commit -m "chore: cleanup repository - archive old artifacts and logs

- Remove auto-generated caches (.mypy, pytest)
- Archive artifacts older than 2 days (kept 5 recent)
- Move old logs to archive (kept 5 recent)
- Move batch scripts to archive/scripts_feb2026
- Move research CSVs to archive/docs_feb2026
- Archive old predictions and reports

Space saved: ~2.25GB
All tests passing, pipeline functional
"
git push origin main
```

---

## 🚀 Ready to Clean?

**Conservative approach (recommended):**
1. Create a backup branch: `git checkout -b backup-before-cleanup`
2. Run the cleanup script
3. Verify everything works
4. Merge to main if all good

**Aggressive approach:**
1. Run cleanup script directly on main
2. Commit and push
3. CI/CD will verify

**Your choice!** 😊

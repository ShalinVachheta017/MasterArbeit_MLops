# 📋 Markdown Files Cleanup Guide

**Last Updated:** December 12, 2025  
**Total Files:** 16 markdown files in `/docs/`  
**Action:** Consolidate from 16 → 6 essential files

---

## ✅ FILES TO KEEP (6 essential)

### 1. **PIPELINE_RERUN_GUIDE.md** ⭐ PRIMARY
- **Purpose:** Complete end-to-end workflow reference
- **Contains:** Steps 1-10, Docker API docs, cleanup scripts, DVC integration
- **Size:** ~900 lines
- **Essential:** YES - This is your main reference for running pipeline
- **Keep:** ✅ ABSOLUTELY

### 2. **FRESH_START_CLEANUP_GUIDE.md** 🧹 CLEANUP
- **Purpose:** Comprehensive cleanup instructions
- **Contains:** PowerShell scripts, manual deletion, recovery procedures
- **Size:** ~300 lines
- **Essential:** YES - For fresh starts and deleting old records
- **Keep:** ✅ ABSOLUTELY

### 3. **CONCEPTS_EXPLAINED.md** 📚 THEORY
- **Purpose:** Technical concepts (unit conversion, domain calibration, confidence)
- **Contains:** milliG vs m/s², how domain shift works, confidence interpretation
- **Size:** ~600 lines
- **Essential:** YES - Needed to understand WHY things work
- **Keep:** ✅ ABSOLUTELY

### 4. **SRC_FOLDER_ANALYSIS.md** 📁 CODE STRUCTURE
- **Purpose:** Explains what each Python file does
- **Contains:** File-by-file breakdown of src/ modules
- **Size:** ~250 lines
- **Essential:** YES - Helps navigate your codebase
- **Keep:** ✅ KEEP

### 5. **FRESH_START_INDEX.md** 🗺️ NAVIGATION
- **Purpose:** Choose-your-path interface for fresh start docs
- **Contains:** Quick reference matrix showing which guide to read for each scenario
- **Size:** ~100 lines
- **Essential:** MEDIUM - Nice to have, but not critical
- **Keep:** ✅ OPTIONAL (consolidate into PIPELINE_RERUN_GUIDE.md instead)

### 6. **README.md** (at root) 🏠 PROJECT OVERVIEW
- **Purpose:** Main project documentation
- **Contains:** Quick start, architecture, features, installation
- **Size:** ~200 lines
- **Essential:** YES - First thing readers see
- **Keep:** ✅ ABSOLUTELY

---

## ❌ FILES TO DELETE (9 files)

### ❌ 1. **FRESH_START_SUMMARY.md** 
- **Reason:** REDUNDANT with FRESH_START_INDEX.md
- **What's covered elsewhere:** Same info in FRESH_START_CLEANUP_GUIDE.md
- **Size:** 317 lines (not needed)

### ❌ 2. **QUICK_START_FRESH.md**
- **Reason:** DUPLICATE of PIPELINE_RERUN_GUIDE.md (section "Quick Fresh Start")
- **What's covered elsewhere:** Same one-command cleanup
- **Size:** ~100 lines (not needed)

### ❌ 3. **QUICK_RUN.md**
- **Reason:** OBSOLETE - Covered in PIPELINE_RERUN_GUIDE.md
- **What's covered elsewhere:** Same step-by-step pipeline
- **Size:** ~150 lines (not needed)

### ❌ 4. **FILE_SYSTEM_MAP.md**
- **Reason:** OUTDATED - Directory structure changed, info stale
- **What's covered elsewhere:** Current structure in README.md
- **Size:** ~200 lines (not needed)

### ❌ 5. **DATA_PREPARED_ANALYSIS.md**
- **Reason:** ONE-TIME ANALYSIS - No longer needed after cleanup
- **What's covered elsewhere:** Already analyzed; now we delete those files anyway
- **Size:** ~150 lines (not needed)

### ❌ 6. **PATH_COMPARISON_ANALYSIS.md**
- **Reason:** HISTORICAL ANALYSIS - No ongoing relevance
- **What's covered elsewhere:** Path issues resolved months ago
- **Size:** ~100 lines (not needed)

### ❌ 7. **UNIT_CONVERSION_SOLUTION.md**
- **Reason:** COVERED IN CONCEPTS_EXPLAINED.md
- **What's covered elsewhere:** Section "1️⃣ Units: milliG vs m/s²"
- **Size:** ~120 lines (not needed)

### ❌ 8. **RESEARCH_PAPERS_ANALYSIS.md**
- **Reason:** NOT NEEDED FOR PIPELINE - Research archive only
- **What's covered elsewhere:** Papers already in research_papers/ folder
- **Size:** ~180 lines (not needed)

### ❌ 9. **PIPELINE_RUNBOOK.md**
- **Reason:** DUPLICATE of PIPELINE_RERUN_GUIDE.md (different name, same content)
- **What's covered elsewhere:** Exact same pipeline steps
- **Size:** ~850 lines (not needed)

### ❌ 10. **MENTOR_EMAIL_FOLLOWUP.md** (OPTIONAL)
- **Reason:** ONE-TIME COMMUNICATION - No ongoing reference
- **What's covered:** Email draft already sent/archived
- **Size:** ~120 lines
- **Status:** Can archive if mentor replies, delete if no longer needed

---

## 📊 Cleanup Summary

| Action | Count | Files | Size |
|--------|-------|-------|------|
| **KEEP** | 6 | PIPELINE_RERUN_GUIDE, FRESH_START_CLEANUP_GUIDE, CONCEPTS_EXPLAINED, SRC_FOLDER_ANALYSIS, FRESH_START_INDEX, README.md | ~2.5 MB |
| **DELETE** | 10 | All listed above | ~1.8 MB |
| **Total** | 16 | Current state | ~4.3 MB |

---

## 🚀 How to Clean Up

### Option A: Delete All At Once
```powershell
# Delete all redundant markdown files
Remove-Item -Path "docs/FRESH_START_SUMMARY.md", `
                    "docs/QUICK_START_FRESH.md", `
                    "docs/QUICK_RUN.md", `
                    "docs/FILE_SYSTEM_MAP.md", `
                    "docs/DATA_PREPARED_ANALYSIS.md", `
                    "docs/PATH_COMPARISON_ANALYSIS.md", `
                    "docs/UNIT_CONVERSION_SOLUTION.md", `
                    "docs/RESEARCH_PAPERS_ANALYSIS.md", `
                    "docs/PIPELINE_RUNBOOK.md", `
                    "docs/MENTOR_EMAIL_FOLLOWUP.md" -Force
```

### Option B: Delete One by One (safer)
```powershell
# Verify file exists first, then delete
$files = "FRESH_START_SUMMARY.md", "QUICK_START_FRESH.md", "QUICK_RUN.md"
foreach ($file in $files) {
    if (Test-Path "docs/$file") {
        Remove-Item "docs/$file" -Force
        Write-Host "✓ Deleted: $file"
    }
}
```

### After Cleanup:
1. Verify only 6 files remain in docs/
2. Update archived/ folder with deleted files if needed
3. Git commit: `git add docs/; git commit -m "Cleanup: consolidate 16 markdown files → 6 essential guides"`

---

## 📖 Which File to Read When?

| **I want to...** | **Read this file** |
|---|---|
| Get quick overview | `README.md` |
| Run full pipeline fresh | `PIPELINE_RERUN_GUIDE.md` |
| Understand unit conversion | `CONCEPTS_EXPLAINED.md` |
| Delete old data & logs | `FRESH_START_CLEANUP_GUIDE.md` |
| Find what a code file does | `SRC_FOLDER_ANALYSIS.md` |
| Pick a guide for my scenario | `FRESH_START_INDEX.md` _(or just read PIPELINE_RERUN_GUIDE.md)_ |

---

## ✨ New MLflow Fix

**Just added:** MLflow tracking to `run_inference.py`

Now your inference pipeline will:
- ✅ Appear in MLflow UI automatically
- ✅ Log model parameters (param count)
- ✅ Log data shape and size  
- ✅ Log inference metrics (confidence stats)
- ✅ Log activity distribution
- ✅ Save output artifacts

**To verify it works:**
```powershell
# Run inference
python src/run_inference.py

# Check MLflow (in new terminal)
mlflow ui

# Open browser to http://localhost:5000
# Should see new "inference-production" experiment
```

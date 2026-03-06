# 🗂️ REPOSITORY CLEANUP & ORGANIZATION ANALYSIS
## Master Thesis MLOps Project - Comprehensive File Review

**Generated:** January 18, 2026  
**Total Files Analyzed:** ~500+ files across 30+ folders  
**Purpose:** Categorize all files by importance and provide cleanup recommendations

---

## 📊 CATEGORY LEGEND

| Category | Symbol | Meaning | Action |
|----------|--------|---------|--------|
| **ESSENTIAL** | 🟢 | Core files needed for thesis/pipeline | NEVER DELETE |
| **KEEP** | 🔵 | Useful files, actively used | Keep indefinitely |
| **ARCHIVE** | 🟡 | May be useful later, not active | Move to archive folder |
| **CAN DELETE** | 🟠 | Redundant/outdated, safe to remove | Delete when ready |
| **MUST DELETE** | 🔴 | Clutter, duplicates, temporary files | Delete immediately |

---

# 📁 ROOT LEVEL FILES

## 🟢 ESSENTIAL (Keep Forever)

| File | Why Essential | Usage |
|------|---------------|-------|
| `README.md` | Main project documentation | **Present & Future:** Primary reference for understanding project |
| `PROJECT_GUIDE.md` | Complete folder/file reference with diagrams | **Present:** Navigate project structure |
| `Thesis_Plan.md` | 6-month thesis timeline (Oct 2025 - Apr 2026) | **Present:** Track thesis milestones |
| `docker-compose.yml` | Docker orchestration config | **Future:** Production deployment |
| `.gitignore` | Git ignore rules | **Present:** Version control |
| `.dvcignore` | DVC ignore rules | **Present:** Data versioning |
| `.dockerignore` | Docker ignore rules | **Future:** Containerization |

## 🔵 KEEP (Useful Reference)

| File | Why Keep | Usage |
|------|----------|-------|
| `MASTER_FILE_ANALYSIS_AND_NEXT_STEPS.md` | File categorization + next steps | **Present:** Action planning |
| `FINAL_3_PATHWAYS_TO_COMPLETE_THESIS.md` | 3 implementation pathways for thesis | **Present:** Decision making |

## 🔴 MUST DELETE (Clutter)

| File | Why Delete | Notes |
|------|------------|-------|
| `ans.md` | Empty file | No content, just placeholder |
| `unnamed.jpg` | Random image in root | Move to `images/` if needed |
| `EHB_2025_71.pdf` | PDF in wrong location | Already exists in `papers needs to read/` |
| `ICTH_16.pdf` | PDF in wrong location | Already exists in `papers needs to read/` |
| `New Microsoft Word Document.docx` | Empty/temp Word doc | Likely accidental creation |

---

# 📁 src/ FOLDER - SOURCE CODE

## 🟢 ESSENTIAL (Core Pipeline)

| File | Purpose | Pipeline Stage |
|------|---------|----------------|
| `config.py` | Central path configuration | All stages |
| `sensor_data_pipeline.py` | Raw Excel → CSV processing | Preprocessing |
| `preprocess_data.py` | CSV → .npy windowing | Preprocessing |
| `run_inference.py` | Model inference | Inference |
| `evaluate_predictions.py` | Prediction analysis | Evaluation |
| `mlflow_tracking.py` | Experiment tracking | Tracking |
| `data_validator.py` | Data quality checks | Validation |
| `README.md` | Source code documentation | Reference |

## 🟡 ARCHIVE (Old Code - Keep for Reference)

| Folder | Contents | Why Archive |
|--------|----------|-------------|
| `Archived(prepare traning- production- conversion)/` | Old preprocessing scripts | Historical reference, not actively used |
| - `prepare_production_data.py` | Old production prep | Superseded by current pipeline |
| - `prepare_training_data.py` | Old training prep | Superseded by current pipeline |
| - `convert_production_units.py` | Unit conversion | May be useful for debugging |
| - `validate_garmin_data.py` | Garmin validation | Reference implementation |
| - `training_cv_experiment/` | Old training experiments | Historical reference |

## 🟠 CAN DELETE

| File | Why Delete |
|------|------------|
| `diagnostic_pipeline_check.py` | One-time diagnostic, not part of main pipeline |
| `__pycache__/` | Python cache, auto-regenerated |

---

# 📁 scripts/ FOLDER

## 🔵 KEEP

| File | Purpose | Usage |
|------|---------|-------|
| `build_training_baseline.py` | Create training baseline | **Future:** Retraining pipeline |
| `create_normalized_baseline.py` | Normalization baseline | **Future:** Data normalization |
| `inference_smoke.py` | Smoke test for inference | **Present:** Testing |
| `post_inference_monitoring.py` | Post-inference monitoring | **Future:** Production monitoring |
| `preprocess_qc.py` | Quality control for preprocessing | **Present:** Data quality |

## 🟡 ARCHIVE (Empty Folders)

| Folder | Status | Recommendation |
|--------|--------|----------------|
| `evaluation/` | Empty | Delete or populate with evaluation scripts |
| `labeling/` | Empty | Delete or populate with labeling tools |

---

# 📁 notebooks/ FOLDER

## 🟢 ESSENTIAL

| File | Purpose | Usage |
|------|---------|-------|
| `data_preprocessing_step1.ipynb` | Step 1 preprocessing | **Present:** Interactive preprocessing |
| `production_preprocessing.ipynb` | Production preprocessing | **Present:** Production data pipeline |
| `README.md` | Notebook documentation | Reference |

## 🟡 ARCHIVE

| File | Purpose | Recommendation |
|------|---------|----------------|
| `from_guide_processing.ipynb` | Experimental processing | Archive for reference |
| `exploration/gravity_removal_demo.ipynb` | Gravity removal demo | Keep for educational purposes |

---

# 📁 config/ FOLDER

## 🟢 ESSENTIAL (All Files - Keep Everything)

| File | Purpose | Usage |
|------|---------|-------|
| `pipeline_config.yaml` | Main pipeline configuration | **Present:** All pipeline stages |
| `mlflow_config.yaml` | MLflow tracking config | **Present:** Experiment tracking |
| `requirements.txt` | Python dependencies | **Present:** Environment setup |
| `.pylintrc` | Python linting rules | **Present:** Code quality |

---

# 📁 data/ FOLDER

## 🟢 ESSENTIAL (Core Data)

| Subfolder | Contents | Usage |
|-----------|----------|-------|
| `raw/` | Original source data (Excel files, labeled CSV) | **Never delete:** Source of truth |
| `prepared/` | Model-ready data (production_X.npy, config.json) | **Present:** Inference input |
| `.dvc` files | `raw.dvc`, `prepared.dvc`, `processed.dvc` | **Present:** Data versioning |

## 🔵 KEEP

| Subfolder | Contents | Usage |
|-----------|----------|-------|
| `preprocessed/` | Intermediate sensor fusion files | **Present:** Pipeline intermediate stage |
| `processed/` | Additional processed data | **Present:** Pipeline output |
| `samples_2005 dataset/` | Sample dataset (f_data_50hz.csv) | **Reference:** Sample data |

## Files Inside data/prepared/ - KEEP ALL:
- `production_X.npy` - Model input data
- `config.json` - Preprocessing config
- `baseline_stats.json` - Statistics for normalization
- `production_metadata.json` - Metadata
- `garmin_labeled.csv` - Labeled Garmin data
- `predictions/` - Prediction outputs
- `README.md` - Documentation

---

# 📁 decoded_csv_files/ FOLDER

## 🟡 ARCHIVE or 🟠 CAN DELETE

| Status | Contents | Recommendation |
|--------|----------|----------------|
| 75+ CSV files | Decoded accelerometer/gyroscope/record files | **Large folder (~50MB+)** |
| Timestamps | July 2025 data | Move to `data/raw/decoded/` OR delete if already processed |

**Reason:** These appear to be intermediate decoded files from Garmin data. If they've been processed into the main pipeline, they can be archived or deleted.

---

# 📁 models/ FOLDER

## 🟢 ESSENTIAL

| Item | Contents | Usage |
|------|----------|-------|
| `pretrained/` | `fine_tuned_model_1dcnnbilstm.keras`, `model_info.json` | **Critical:** Core ML model |
| `normalized_baseline.json` | Normalization baseline | **Present:** Data normalization |
| `.gitignore` | Model folder ignore rules | Version control |

## 🔵 KEEP

| Item | Contents | Usage |
|------|----------|-------|
| `archived_experiments/cv_training_20260106/` | Training experiment artifacts | **Reference:** Historical experiments |
| - `best_model.keras` | Best model from training | Backup model |
| - `training_report.json` | Training metrics | Reference |

## 🟠 CAN DELETE

| Item | Reason |
|------|--------|
| `trained/` | Empty folder, no contents |

---

# 📁 docker/ FOLDER

## 🟢 ESSENTIAL (All Files)

| File | Purpose | Usage |
|------|---------|-------|
| `Dockerfile.inference` | Inference container | **Future:** Deployment |
| `Dockerfile.training` | Training container | **Future:** Automated training |
| `api/main.py` | FastAPI inference endpoint | **Future:** API deployment |
| `api/__init__.py` | Python package init | Code structure |

---

# 📁 docs/ FOLDER - DOCUMENTATION

## 🟢 ESSENTIAL

| File | Purpose | Usage |
|------|---------|-------|
| `README.md` | Docs folder overview | Navigation |
| `ORGANIZATION_MAP.md` | Repository structure map | **Present:** Understanding repo |

## 🔵 KEEP (docs/thesis/)

| File | Thesis Chapter | Keep For |
|------|----------------|----------|
| `CONCEPTS_EXPLAINED.md` | Chapter 2: Background | Thesis writing |
| `FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md` | Timeline | Progress tracking |
| `FINE_TUNING_STRATEGY.md` | Methodology | Implementation |
| `HANDEDNESS_WRIST_PLACEMENT_ANALYSIS.md` | Discussion | Domain shift analysis |
| `KEEP_Production_Robustness_Guide.md` | Future work | Production improvements |
| `KEEP_Reference_Project_Learnings.md` | Related work | Best practices |
| `KEEP_Technology_Stack_Analysis.md` | Implementation | Tech justification |
| `QA_LAB_TO_LIFE_GAP.md` | Discussion | Lab-to-production gap |
| `THESIS_READY_UNLABELED_EVALUATION_PLAN.md` | Evaluation | Thesis-ready text |
| `UNLABELED_EVALUATION.md` | Methodology | 4-layer monitoring framework |

## 🔵 KEEP (docs/research/)

| File | Purpose |
|------|---------|
| `KEEP_Research_QA_From_Papers.md` | Q&A from literature |
| `RESEARCH_PAPERS_ANALYSIS.md` | ICTH_16 & EHB analysis |
| `RESEARCH_PAPER_INSIGHTS.md` | Cross-paper synthesis |

## 🔵 KEEP (docs/technical/)

| File | Purpose |
|------|---------|
| `PIPELINE_RERUN_GUIDE.md` | **Critical:** How to run pipeline |
| `PIPELINE_VISUALIZATION_CURRENT.md` | Architecture diagrams |
| `PIPELINE_TEST_RESULTS.md` | Validation results |
| `pipeline_audit_map.md` | Code inventory |
| `evaluation_audit.md` | Evaluation framework |
| `tracking_audit.md` | MLflow/DVC setup |
| `QC_EXECUTION_SUMMARY.md` | QC results |
| `root_cause_low_accuracy.md` | Debugging analysis |

## 🟡 ARCHIVE (docs/archive/)

| File | Recommendation |
|------|----------------|
| `extranotes.md` | Archive or delete |
| `FRESH_START_CLEANUP_GUIDE.md` | Keep in archive for reference |
| `LATER_Offline_MLOps_Guide.md` | Keep for future edge deployment |
| `Mondaymeet.md` | Archive - meeting notes |
| `RESTRUCTURE_PIPELINE_PACKAGES.md` | Archive - restructuring notes |

## 🟠 CAN DELETE (docs/)

| File | Reason |
|------|--------|
| `BIG_QUESTIONS_2026-01-18.md` | Temporary questions file |
| `output_1801_2026-01-18.md` | Temporary output file |
| `PIPELINE_DIVE.md` | Superseded by PIPELINE_DEEP_DIVE_opus.md |

---

# 📁 PAPER FOLDERS ANALYSIS

## ⚠️ MAJOR DUPLICATION ISSUE

You have **5 different folders** containing papers with significant overlap:

| Folder | Paper Count | Status |
|--------|-------------|--------|
| `papers needs to read/` | 140+ PDFs | **Largest collection** |
| `research_papers/76 papers/` | 76 PDFs | **Organized subset** |
| `papers/` | ~20 PDFs (4 subfolders) | **Categorized by topic** |
| `new paper/` | 19 PDFs | **Recent additions** |
| `np/papers/` | Empty | **Delete** |

### 🔴 RECOMMENDED ACTION: CONSOLIDATE PAPERS

**Create single `papers/` structure:**
```
papers/
├── anxiety_detection/       (Keep - 4 papers)
├── domain_adaptation/       (Keep - 2 papers)
├── mlops_production/        (Keep - 14 papers)
├── uncertainty_confidence/  (Delete - Empty)
├── wearable_sensors/        (NEW - consolidate HAR papers)
├── deep_learning/           (NEW - CNN/LSTM papers)
├── healthcare_ml/           (NEW - healthcare ML papers)
└── general_reference/       (NEW - other papers)
```

## 🔴 MUST DELETE/CONSOLIDATE

| Folder | Recommendation |
|--------|----------------|
| `papers needs to read/` | **MERGE** into organized `papers/` structure |
| `research_papers/76 papers/` | **MERGE** - duplicates of above |
| `new paper/` | **MERGE** - recent papers to categorize |
| `np/papers/` | **DELETE** - empty folder |
| `np/` | **DELETE** - only contains empty subfolder |

## Files in research_papers/ (NOT in 76 papers/)

| File | Recommendation |
|------|----------------|
| `76_papers_suggestions.md` | 🔵 KEEP - Paper recommendations |
| `76_papers_summarizzation.md` | 🔵 KEEP - Paper summaries |
| `COMPREHENSIVE_RESEARCH_PAPERS_SUMMARY.md` | 🔵 KEEP - Best analysis |
| `all_users_data_labeled.csv` | 🟡 Move to `data/raw/` |
| `anxiety_dataset.csv` | 🟡 Move to `data/raw/` |
| `*.xlsx` files | 🟡 Move to `data/raw/` |
| `temp.ipynb` | 🔴 DELETE - Temporary |
| `papers_temp.csv` | 🔴 DELETE - Temporary |
| `extract_paper_info.py` | 🟡 Move to `scripts/` |
| `generate_paper_summary_excel.py` | 🟡 Move to `scripts/` |

---

# 📁 logs/ FOLDER

## 🔵 KEEP (Organized Logging)

| Subfolder | Contents | Usage |
|-----------|----------|-------|
| `evaluation/` | Evaluation logs | Debugging |
| `inference/` | Inference logs | Debugging |
| `preprocessing/` | Pipeline logs | Debugging |
| `training/` | Training logs (empty) | Future use |

**Note:** Log files are auto-generated. Keep folder structure, individual logs can be cleaned periodically.

---

# 📁 outputs/ FOLDER

## 🔵 KEEP

| Subfolder | Contents | Usage |
|-----------|----------|-------|
| `evaluation/` | Evaluation JSONs | Results |
| `predictions_fresh.csv` | Fresh predictions | Current output |
| `production_labels_fresh.npy` | Labels | Current output |
| `production_predictions_fresh.npy` | Predictions | Current output |
| `gravity_removal_comparison.png` | Visualization | Documentation |

---

# 📁 reports/ FOLDER

## 🔵 KEEP (All Subfolders)

| Subfolder | Contents | Usage |
|-----------|----------|-------|
| `inference_smoke/` | Smoke test reports | Testing |
| `monitoring/` | Monitoring reports | Production monitoring |
| `preprocess_qc/` | QC reports | Quality control |

---

# 📁 mlruns/ FOLDER

## 🔵 KEEP (MLflow Data)

| Contents | Usage |
|----------|-------|
| Experiment tracking data | **Present:** Experiment comparison |
| Model artifacts | **Present:** Model versioning |

**Note:** Can be cleaned with `mlflow gc` if storage becomes an issue.

---

# 📁 OTHER FOLDERS

## 🔵 KEEP

| Folder | Purpose |
|--------|---------|
| `images/` | Project diagrams and visualizations |
| `cheat sheet/` | DVC cheatsheet, reference PDFs |
| `.dvc/` | DVC internal configuration |
| `.dvc_storage/` | DVC local cache |
| `.git/` | Git version control |
| `tests/` | Unit tests (currently empty - populate!) |
| `.cursor/` | Cursor editor config |
| `.qodo/` | Qodo config |

## 🟡 ARCHIVE

| Folder | Contents | Recommendation |
|--------|----------|----------------|
| `ai helps/` | AI-generated help documents | Merge into `docs/` |

---

# 📊 SUMMARY STATISTICS

## Files by Category

| Category | Count | Action |
|----------|-------|--------|
| 🟢 ESSENTIAL | ~50 files | NEVER DELETE |
| 🔵 KEEP | ~100 files | Keep indefinitely |
| 🟡 ARCHIVE | ~50 files | Move to archive folders |
| 🟠 CAN DELETE | ~30 files | Delete when ready |
| 🔴 MUST DELETE | ~10 files | Delete immediately |
| 📄 Papers (Duplicates) | 200+ PDFs | Consolidate to ~100 unique |

## Storage Savings Estimate

| Action | Potential Savings |
|--------|-------------------|
| Delete duplicate papers | ~500 MB |
| Delete decoded_csv_files | ~50 MB |
| Clean mlruns (optional) | ~100 MB |
| Clean logs (optional) | ~10 MB |
| **Total Potential** | **~660 MB** |

---

# 🎯 RECOMMENDED CLEANUP ACTIONS

## Priority 1: IMMEDIATE (5 minutes)

```powershell
# Delete empty/useless files
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\ans.md" -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\New Microsoft Word Document.docx" -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\np" -Recurse -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\papers\uncertainty_confidence" -Recurse -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\scripts\evaluation" -Recurse -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\scripts\labeling" -Recurse -Force
Remove-Item "d:\study apply\ML Ops\MasterArbeit_MLops\models\trained" -Recurse -Force
```

## Priority 2: MOVE MISPLACED FILES (10 minutes)

```powershell
# Move root PDFs to papers folder
Move-Item "d:\study apply\ML Ops\MasterArbeit_MLops\EHB_2025_71.pdf" "d:\study apply\ML Ops\MasterArbeit_MLops\papers needs to read\"
Move-Item "d:\study apply\ML Ops\MasterArbeit_MLops\ICTH_16.pdf" "d:\study apply\ML Ops\MasterArbeit_MLops\papers needs to read\"
Move-Item "d:\study apply\ML Ops\MasterArbeit_MLops\unnamed.jpg" "d:\study apply\ML Ops\MasterArbeit_MLops\images\"

# Move datasets from research_papers to data/raw
Move-Item "d:\study apply\ML Ops\MasterArbeit_MLops\research_papers\all_users_data_labeled.csv" "d:\study apply\ML Ops\MasterArbeit_MLops\data\raw\"
Move-Item "d:\study apply\ML Ops\MasterArbeit_MLops\research_papers\anxiety_dataset.csv" "d:\study apply\ML Ops\MasterArbeit_MLops\data\raw\"
```

## Priority 3: PAPER CONSOLIDATION (30 minutes)

1. **Audit duplicates** between:
   - `papers needs to read/`
   - `research_papers/76 papers/`
   - `new paper/`

2. **Create organized structure** in `papers/`:
   - Move unique papers to appropriate subfolders
   - Delete duplicates
   - Keep summary markdown files in `research_papers/`

## Priority 4: ARCHIVE OLD FILES (15 minutes)

Move to `docs/archive/`:
- Old status files
- Meeting notes
- One-time analysis files

---

# 🔮 FUTURE RECOMMENDATIONS

1. **Add `.gitkeep` files** to empty folders you want to preserve
2. **Create `scripts/utils/`** for helper scripts (paper extraction, etc.)
3. **Populate `tests/`** with unit tests for src/ modules
4. **Use DVC** to track large decoded_csv_files if needed
5. **Regular cleanup** - Run `mlflow gc` and delete old logs monthly

---

**Document End**  
*Generated by GitHub Copilot - Repository Analysis Tool*

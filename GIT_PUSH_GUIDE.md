"""
GIT PUSH GUIDE FOR HAR MLOps REPOSITORY
========================================

WHAT GETS PUSHED TO GITHUB:
==========================

✅ SOURCE CODE:
  - src/ — All production Python code (components, pipeline, utilities)
  - tests/ — All test files (124 tests)
  - scripts/ — Utility scripts for preprocessing, monitoring, etc.
  
✅ CONFIGURATION:
  - config/ — YAML configs (pipeline, prometheus, etc.)
  - docker/ — Docker build files
  - pytest.ini — Test configuration
  - .github/workflows/ — CI/CD pipelines
  
✅ DOCUMENTATION:
  - README.md — Project overview
  - PROJECT_GUIDE.md — How to use the pipeline
  - Thesis_Plan.md — Thesis structure
  - FEBRUARY_2026_ACTION_PLAN.md — Current progress tracking
  - docs/ — All thesis writing and documentation (except archive)
  
✅ METADATA:
  - .dvc files — Data version control pointers (*.dvc, data/*.dvc)
  - requirements.txt — Python dependencies
  - docker-compose.yml — Docker orchestration

---

WHAT DOES NOT GET PUSHED (.gitignore):
======================================

❌ DATA (tracked by DVC instead):
  - data/ — All raw, processed, and prepared data
  - *.csv — CSV data files
  - decoded_csv_files/ — Large decoded datasets
  
❌ LARGE MODEL FILES:
  - models/ — Pretrained and trained model files (.keras)
  
❌ PAPERS & RESEARCH:
  - papers/ — Research papers (PDFs, 200+ MB)
  - research_papers/ — Additional research materials
  - *.pdf — All PDF files
  
❌ GENERATED/RUNTIME FILES:
  - outputs/ — Predictions, reports, generated files
  - logs/ — Runtime logs
  - reports/ — Generated reports
  - mlruns/ — MLflow experiment tracking
  
❌ ARCHIVES:
  - archive/ — Old/archived files
  
❌ ENVIRONMENT & CACHE:
  - venv/, .venv/ — Virtual environments
  - __pycache__/, .pytest_cache/ — Python cache
  - .dvc_storage/ — DVC local cache
  
---

CURRENT GITHUB-READY STATUS:
============================

Repository size without data/papers:
  - Estimated: ~50-100 MB (code, tests, docs, config)
  - Without this ignore: ~5+ GB

✅ Ready to push to GitHub!

COMMANDS TO GIT PUSH:
====================

1. Check what will be pushed:
   $ git status
   $ git diff --cached

2. Add all tracked files:
   $ git add .

3. Commit with meaningful message:
   $ git commit -m "Add 10-stage production pipeline with AdaBN domain adaptation"

4. Push to GitHub:
   $ git push origin main

---

DATA REPRODUCIBILITY WITH DVC:
==============================

Even though data/ is ignored by git, data is reproducible via DVC:

1. .dvc files ARE tracked by git (small metadata files)
2. Actual data is stored in .dvc_storage/ for local development
3. For production/CI: Use `dvc pull` to fetch data from .dvc_storage/

Commands:
  $ dvc status          # Check data status
  $ dvc add data/raw/   # Track new data
  $ dvc push            # Push data to storage
  $ dvc pull            # Pull data locally
  $ git add *.dvc       # Commit .dvc pointers
  $ git push            # Push to GitHub

---

GITHUB REPOSITORY BEST PRACTICES:
=================================

✅ Lean repository (code + config only) 
✅ Data versioned separately with DVC
✅ Easy for collaborators to clone and get started
✅ All docs in repo for transparency
✅ Test coverage in repo for CI/CD

Result: Clean, professional, production-ready repository! 🚀
"""

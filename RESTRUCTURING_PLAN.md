# MLOps Project Restructuring Plan

## Current Issues
- ❌ Complex numbered folder structure (01_data, 02_src, etc.)
- ❌ Too many analysis scripts scattered in root directory
- ❌ Large archive folder with duplicates
- ❌ Not following standard MLOps project conventions

## New Simplified Structure

```
MasterArbeit_MLops/
├── data/
│   ├── raw/                    # Original labeled dataset
│   ├── processed/              # Cleaned and prepared data
│   └── production/             # Unlabeled production data
│
├── models/
│   └── pretrained/             # Mentor's pretrained model (DO NOT RETRAIN)
│
├── src/
│   ├── inference/              # Prediction pipeline (NEW)
│   ├── preprocessing/          # Data preparation scripts
│   ├── monitoring/             # Model monitoring (NEW)
│   └── utils/                  # Helper functions (NEW)
│
├── api/                        # FastAPI serving endpoint (NEW)
│   ├── app.py
│   ├── schemas.py
│   └── routes/
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── experiments/
│
├── tests/                      # Unit tests (NEW)
│   ├── test_inference.py
│   └── test_preprocessing.py
│
├── config/
│   ├── model_config.yaml       # Model settings
│   └── deployment_config.yaml  # Deployment settings
│
├── docker/                     # Docker setup (NEW)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/              # CI/CD pipelines (NEW)
│       └── mlops-pipeline.yml
│
├── logs/                       # Application logs
├── docs/                       # Documentation
├── requirements.txt
└── README.md
```

## Actions to Take

### 1. DELETE (Cleanup)
- [ ] `09_archive/` - entire folder (backups, old files)
- [ ] Root analysis scripts: `analyze_labeled_data.py`, `check_activities.py`, `check_data_leakage.py`, `compare_datasets.py`
- [ ] `07_docs/` - move useful content to `docs/`, delete rest
- [ ] `05_outputs/` - can recreate as needed
- [ ] `06_logs/` - empty folders

### 2. RENAME (Simplify)
- [ ] `01_data/` → `data/`
- [ ] `02_src/` → `src/`
- [ ] `03_models/` → `models/`
- [ ] `04_notebooks/` → `notebooks/`
- [ ] `08_config/` → `config/`

### 3. CREATE (New MLOps Components)
- [ ] `src/inference/` - prediction pipeline
- [ ] `src/monitoring/` - model monitoring
- [ ] `src/utils/` - helper functions
- [ ] `api/` - FastAPI serving
- [ ] `tests/` - unit tests
- [ ] `docker/` - containerization
- [ ] `.github/workflows/` - CI/CD

### 4. CONSOLIDATE
- [ ] Move `02_src/path_config.py` → `src/config.py`
- [ ] Keep only `prepare_training_data.py` in preprocessing
- [ ] Archive `sensor_data_pipeline.py` (not needed)

## Focus: MLOps, NOT Model Training!

**Remember:** Your thesis is about building a production-ready MLOps pipeline, not retraining models.

### Core Components:
1. **Inference Pipeline** - Use pretrained model AS-IS
2. **API Serving** - FastAPI endpoint
3. **Monitoring** - Track performance
4. **CI/CD** - Automated deployment
5. **Containerization** - Docker
6. **Model Registry** - MLflow

---
**Status:** Ready to execute
**Date:** November 4, 2025

# 📋 MASTER FILE ANALYSIS & NEXT STEPS
## Thesis MLOps Project - January 2026

**Generated:** January 6, 2026  
**Purpose:** Categorize all files (KEEP/DELETE/LATER) and define clear next steps

---

## 📁 FILE CATEGORIZATION

### Legend
- 🟢 **KEEP** - Essential, actively used
- 🔴 **DELETE** - Outdated, redundant, or already archived
- 🟡 **LATER** - May be useful in future phases

---

## ROOT LEVEL FILES

| File | Decision | Reason |
|------|----------|--------|
| `README.md` | 🟢 KEEP | Main project documentation |
| `PROJECT_GUIDE.md` | 🟢 KEEP | Complete folder/file reference |
| `Thesis_Plan.md` | 🟢 KEEP | 6-month timeline (essential) |
| `COMPREHENSIVE_THESIS_STATUS.md` | 🟡 LATER | Outdated (Dec 2025), merge into new status |
| `WHAT_TO_DO_NEXT.md` | 🔴 DELETE | Outdated (Dec 2025), will be replaced |
| `TODO_THIS_WEEK.md` | 🔴 DELETE | Outdated (Dec 2025), no longer relevant |
| `LEARNINGS_FROM_REFERENCE_PROJECT.md` | 🟢 KEEP | Valuable architecture patterns |
| `Technology Stack Analysis.md` | 🟢 KEEP | Tech decisions reference |
| `imp.md` | 🟢 KEEP | Production robustness guide |
| `docker-compose.yml` | 🟢 KEEP | Docker orchestration |
| `dvc_experiments.html` | 🔴 DELETE | Auto-generated, can recreate |

---

## docs/ FOLDER

| File | Decision | Reason |
|------|----------|--------|
| `docs/CURRENT_STATUS.md` | 🔴 DELETE | Outdated (Dec 6, 2025) |
| `docs/PIPELINE_RERUN_GUIDE.md` | 🟢 KEEP | Essential for running pipeline |
| `docs/FRESH_START_CLEANUP_GUIDE.md` | 🟢 KEEP | Useful for cleanup |
| `docs/FILE_ORGANIZATION_SUMMARY.md` | 🔴 DELETE | Outdated cleanup reference |
| `docs/MARKDOWN_CLEANUP_GUIDE.md` | 🔴 DELETE | Already done cleanup |
| `docs/CONCEPTS_EXPLAINED.md` | 🟢 KEEP | Educational reference |
| `docs/RESEARCH_PAPER_INSIGHTS.md` | 🟢 KEEP | Valuable paper analysis |
| `docs/RESEARCH_PAPERS_ANALYSIS.md` | 🟢 KEEP | ICTH_16 & EHB_2025_71 analysis |
| `docs/SRC_FOLDER_ANALYSIS.md` | 🔴 DELETE | One-time analysis |
| `docs/QA_LAB_TO_LIFE_GAP.md` | 🟢 KEEP | Important Q&A for thesis |
| `docs/archived/` | 🔴 DELETE | Already marked for deletion |
| `docs/archived_status/` | 🔴 DELETE | Outdated status files |

---

## ai helps/ FOLDER

| File | Decision | Reason |
|------|----------|--------|
| `ai helps/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md` | 🟢 KEEP | Comprehensive plan |
| `ai helps/offline-mlops-guide.md` | 🟡 LATER | Edge deployment reference |
| `ai helps/extranotes.md` | 🔴 DELETE | Temporary notes |

---

## research_papers/ FOLDER

| File | Decision | Reason |
|------|----------|--------|
| `76_papers_suggestions.md` | 🟢 KEEP | Paper recommendations |
| `76_papers_summarizzation.md` | 🟢 KEEP | Paper summaries |
| `COMPREHENSIVE_RESEARCH_PAPERS_SUMMARY.md` | 🟢 KEEP | Best paper analysis |
| `all_users_data_labeled.csv` | 🟢 KEEP | Training data |
| `anxiety_dataset.csv` | 🟢 KEEP | Additional dataset |
| `76 papers/` | 🟢 KEEP | PDF collection |

---

## data/ FOLDER

| Subfolder | Decision | Reason |
|-----------|----------|--------|
| `data/raw/` | 🟢 KEEP | Source data |
| `data/preprocessed/` | 🟢 KEEP | Pipeline output |
| `data/prepared/` | 🟢 KEEP | Model-ready data |
| `data/prepared/DATA_COMPARISON_REPORT.md` | 🔴 DELETE | One-time analysis |
| `data/prepared/PRODUCTION_DATA_README.md` | 🟢 KEEP | Data documentation |

---

## notebooks/ FOLDER

| File | Decision | Reason |
|------|----------|--------|
| `data_preprocessing_step1.ipynb` | 🟢 KEEP | Preprocessing reference |
| `production_preprocessing.ipynb` | 🟢 KEEP | Production preprocessing |
| `from_guide_processing.ipynb` | 🟡 LATER | Experimental |
| `data_comparison.ipynb` | 🔴 DELETE | One-time comparison |
| `scalable.ipynb` | 🔴 DELETE | Experimental |
| `exploration/` | 🟡 LATER | Exploration notebooks |

---

## src/ FOLDER

| File | Decision | Reason |
|------|----------|--------|
| `config.py` | 🟢 KEEP | Core configuration |
| `preprocess_data.py` | 🟢 KEEP | Core preprocessing |
| `run_inference.py` | 🟢 KEEP | Core inference |
| `sensor_data_pipeline.py` | 🟢 KEEP | Core data pipeline |
| `mlflow_tracking.py` | 🟢 KEEP | Experiment tracking |
| `data_validator.py` | 🟢 KEEP | Data validation |
| `evaluate_predictions.py` | 🟢 KEEP | Evaluation logic |
| `compare_data.py` | 🔴 DELETE | One-time comparison |
| `Archived(...)/` | 🟢 KEEP | Archive for old code |

---

# 🎯 WHAT TO DO NEXT

## The Core Problem

**Your pipeline is INFERENCE-ONLY, not TRAINING.**

Current flow:
```
Raw Garmin Data → Preprocess → Pretrained Model → Predictions
```

What's missing for production:
```
New Labeled Data → Retrain with CV → Updated Model → Deploy → Monitor → Repeat
```

---

## Priority 1: RETRAINING PIPELINE (Week 1-2)

### Why Retraining?
Per ICTH_16 paper: *"Weekly retraining with 10-20% new labeled data maintains 85%+ accuracy"*

### When to Trigger Retraining?
Options from research papers:

| Trigger | Description | Paper Source |
|---------|-------------|--------------|
| **Scheduled** | Every week (cron job) | ICTH_16 |
| **Data Volume** | After N new labeled samples | MLOps Survey |
| **Drift Detected** | When data distribution shifts | Domain Adaptation papers |
| **Performance Drop** | When accuracy < threshold | MLOps Best Practices |

### Recommended: Scheduled + Drift-based

```
┌─────────────────────────────────────────────────────────────┐
│  RETRAINING TRIGGER LOGIC                                   │
├─────────────────────────────────────────────────────────────┤
│  IF (weekly_schedule_reached) OR (drift_score > 0.1):       │
│      IF (new_labeled_samples > 100):                        │
│          run_retraining_with_cv()                           │
│          IF (new_accuracy > current_accuracy - 0.02):       │
│              deploy_new_model()                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Priority 2: CI/CD PIPELINE (Week 2-3)

### Minimal CI/CD for Thesis

```yaml
# .github/workflows/mlops.yml
name: MLOps Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly (Sunday midnight)

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/ -v

  retrain:
    needs: test
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - name: Check for new data
        run: python scripts/check_new_data.py
      - name: Retrain with CV
        run: python src/retrain_with_cv.py
      - name: Deploy if better
        run: python scripts/deploy_if_better.py
```

---

## Priority 3: DRIFT DETECTION (Week 3-4)

### Simple Drift Detection

```python
# src/drift_detector.py
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference_data, new_data, threshold=0.1):
    """
    Detect distribution shift using Kolmogorov-Smirnov test.
    Per Domain Adaptation papers: KS-test is simple but effective.
    """
    drift_scores = {}
    for col in ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']:
        statistic, p_value = ks_2samp(reference_data[col], new_data[col])
        drift_scores[col] = statistic
    
    avg_drift = np.mean(list(drift_scores.values()))
    return {
        'drift_detected': avg_drift > threshold,
        'drift_score': avg_drift,
        'per_feature': drift_scores
    }
```

---

## Priority 4: MONITORING DASHBOARD (Week 4-5)

### Simple Prometheus + Grafana

Already in docker-compose pattern, just needs metrics endpoints:

```python
# Add to FastAPI
from prometheus_client import Counter, Histogram, generate_latest

PREDICTIONS = Counter('predictions_total', 'Total predictions', ['activity'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

# 🔬 RESEARCH QUESTIONS TO EXPLORE

These questions can be researched online or asked in NotebookLM:

## Domain Adaptation Questions

1. **"How to implement unsupervised domain adaptation for wearable HAR when target domain has no labels?"**
   - Key papers: CORAL, MMD, DANN
   - Search: "unsupervised domain adaptation IMU sensors"

2. **"What is the minimum number of labeled target samples needed for effective fine-tuning?"**
   - ICTH_16 suggests: 10-20% of source domain size
   - Search: "few-shot domain adaptation HAR"

3. **"Can we use contrastive learning to align source and target sensor distributions?"**
   - Key papers: SimCLR, MoCo for time-series
   - Search: "contrastive learning sensor data domain adaptation"

## Retraining Questions

4. **"How often should HAR models be retrained in production?"**
   - ICTH_16: Weekly with new data
   - Search: "model retraining frequency machine learning production"

5. **"What triggers model retraining: scheduled, drift-based, or performance-based?"**
   - MLOps papers suggest: combination
   - Search: "model retraining triggers MLOps"

6. **"How to implement online learning for HAR without forgetting old activities?"**
   - Key concept: Catastrophic forgetting
   - Search: "continual learning HAR wearables"

## MLOps Questions

7. **"What's the minimal CI/CD pipeline for a thesis-level MLOps project?"**
   - GitHub Actions + pytest + Docker
   - Search: "minimal MLOps pipeline academic project"

8. **"How to version models and data together in a reproducible way?"**
   - DVC + MLflow combination
   - Search: "DVC MLflow integration reproducibility"

---

# 📊 SUMMARY TABLE

| Priority | Task | Effort | Research Support |
|----------|------|--------|------------------|
| 1 | Retraining Pipeline with CV | 1-2 weeks | ICTH_16: weekly retraining |
| 2 | CI/CD (GitHub Actions) | 1 week | MLOps Survey |
| 3 | Drift Detection | 3-4 days | Domain Adaptation papers |
| 4 | Monitoring (Prometheus) | 3-4 days | MLOps Best Practices |
| 5 | Thesis Writing | 4-6 weeks | - |

---

# 🗑️ FILES TO DELETE NOW

Run this command to clean up:

```powershell
# Root level
Remove-Item -Path "COMPREHENSIVE_THESIS_STATUS.md" -Force
Remove-Item -Path "WHAT_TO_DO_NEXT.md" -Force
Remove-Item -Path "TODO_THIS_WEEK.md" -Force
Remove-Item -Path "dvc_experiments.html" -Force

# docs/
Remove-Item -Path "docs/CURRENT_STATUS.md" -Force
Remove-Item -Path "docs/FILE_ORGANIZATION_SUMMARY.md" -Force
Remove-Item -Path "docs/MARKDOWN_CLEANUP_GUIDE.md" -Force
Remove-Item -Path "docs/SRC_FOLDER_ANALYSIS.md" -Force
Remove-Item -Path "docs/archived" -Recurse -Force
Remove-Item -Path "docs/archived_status" -Recurse -Force

# ai helps/
Remove-Item -Path "ai helps/extranotes.md" -Force

# data/prepared/
Remove-Item -Path "data/prepared/DATA_COMPARISON_REPORT.md" -Force

# notebooks/
Remove-Item -Path "notebooks/data_comparison.ipynb" -Force
Remove-Item -Path "notebooks/scalable.ipynb" -Force

# src/
Remove-Item -Path "src/compare_data.py" -Force
```

---

# ✅ NEXT IMMEDIATE ACTION

1. **Run the cleanup command above**
2. **Read `ai helps/FINAL_Thesis_Status_and_Plan_Jan_to_Jun_2026.md`** - This is your roadmap
3. **Create `src/retrain_with_cv.py`** - Retraining script with 5-fold CV
4. **Ask NotebookLM**: "How to trigger model retraining based on drift detection in HAR systems?"

---

*This file replaces: WHAT_TO_DO_NEXT.md, TODO_THIS_WEEK.md, COMPREHENSIVE_THESIS_STATUS.md*

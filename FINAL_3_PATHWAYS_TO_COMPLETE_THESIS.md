# FINAL THESIS IMPLEMENTATION ROADMAP - 3 PATHWAYS
## HAR Wearable Sensor MLOps Pipeline - Decision Document

**Created:** January 6, 2026  
**Thesis Month:** 3-4 (CI/CD, Monitoring, Retraining Phase)  
**Current Status:** Inference pipeline working, Retraining pipeline needed  

---

## 📊 CURRENT SITUATION SUMMARY

### What We Have ✅
- **Pretrained Model:** 1D-CNN-BiLSTM (499K params) trained on ADAMSense
- **Inference Pipeline:** Raw Garmin → Preprocess → Model → Predictions
- **Model Versioning:** MLflow/DVC tracking in place
- **Research Insights:** 150+ papers analyzed via NotebookLM

### What's Missing ❌
- **Retraining Pipeline:** No mechanism to update model with new data
- **Drift Detection:** No monitoring for data/concept drift
- **CI/CD:** No automated deployment pipeline
- **Lab-to-Life Bridge:** Model trained on lab data, deployed on Garmin (49% → 87% gap)

---

## 🛤️ THREE PATHWAYS TO COMPLETE THESIS

---

# PATH A: Research Paper-Based Solutions (Academic Focus)
## *Implement Domain Adaptation from Literature*

### Core Methodology: AdaBN + UDA
Based on ICTH_16 and 150+ paper insights, implement:

#### 1. **AdaBN (Adaptive Batch Normalization)** - Simplest UDA Method
```
How it works:
- Freeze all model weights during inference
- Compute target-specific Batch Normalization statistics (mean/variance)
- Update only BN layers with Garmin data statistics
- No labels required from target domain

Implementation:
1. Load pretrained model
2. Forward pass through Garmin data (inference mode but collecting BN stats)
3. Update running_mean and running_var in BN layers
4. Use updated model for final predictions
```

**Code Location:** `src/domain_adaptation/adabn.py`

#### 2. **Contrastive Learning** (Optional Advanced)
```
From papers:
- Create positive pairs (same activity, different time)
- Create negative pairs (different activities)
- Learn domain-invariant representations
```

#### 3. **MMD/DANN** (If AdaBN insufficient)
- Maximum Mean Discrepancy for distribution matching
- Domain Adversarial Neural Networks for feature alignment

### Pros & Cons
| Pros | Cons |
|------|------|
| ✅ Academically rigorous | ❌ Complex to implement |
| ✅ Novel thesis contribution | ❌ May need hyperparameter tuning |
| ✅ Addresses lab-to-life gap | ❌ No production-ready patterns |
| ✅ Paper citations ready | ❌ Limited MLOps focus |

### Implementation Effort: **4-6 weeks**

---

# PATH B: Practical MLOps Pipeline (Industry Focus)
## *Based on Research Answers + Best Practices*

### Core Components (From NotebookLM Research)

#### 1. **Drift Detection System**
```python
# KS-Test for Feature Drift (No Labels Required)
from scipy.stats import ks_2samp

def detect_drift(baseline_data, new_data, threshold=0.05):
    """
    Kolmogorov-Smirnov test for each feature
    Returns: dict of {feature: (statistic, p_value, drifted)}
    """
    results = {}
    for feature in baseline_data.columns:
        stat, p_val = ks_2samp(baseline_data[feature], new_data[feature])
        results[feature] = {
            'statistic': stat,
            'p_value': p_val,
            'drifted': p_val < threshold
        }
    return results
```

#### 2. **Combined Retraining Triggers**
```
From ICTH_16 Paper Research:
├── Drift-Based: KS-test detects significant distribution shift
├── Performance-Based: Prediction entropy > threshold (low confidence)  
├── Scheduled: Weekly/monthly automatic retraining
└── Human-Initiated: Manual trigger for new labeled data

Recommendation: Use ALL triggers with priority:
1. Performance drop → immediate retraining
2. Drift detected → queue retraining
3. Scheduled → background retraining
```

#### 3. **EWC (Elastic Weight Consolidation)**
```
Prevents catastrophic forgetting during incremental retraining:
- Compute Fisher Information Matrix on old data
- Add regularization term to loss function
- Protects important weights from changing too much

Loss = CrossEntropy + λ * Σ F_i * (θ - θ_old)²
```

#### 4. **Cross-Validation Strategy**
```
From ICTH_16: 6 volunteers fine-tuning improved 49% → 87%

For Retraining Pipeline:
├── Collect labeled data from users (or pseudo-labels)
├── 5-Fold CV on new data to validate model updates
├── Compare against baseline model
└── Deploy only if improvement > threshold
```

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION MLOPS PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │ Raw Data │────▶│ Preprocess│────▶│ Inference │           │
│  │ (Garmin) │     │   Stage   │     │  Model    │           │
│  └──────────┘     └──────────┘     └─────┬─────┘           │
│                                          │                  │
│                                          ▼                  │
│                                   ┌──────────┐              │
│                                   │Predictions│              │
│                                   └─────┬─────┘             │
│                                         │                   │
│       ┌─────────────────────────────────┼───────────────┐   │
│       │            MONITORING LAYER     │               │   │
│       │  ┌─────────┐  ┌─────────┐  ┌───▼────┐          │   │
│       │  │ Drift   │  │Scheduled│  │Entropy │          │   │
│       │  │Detector │  │ Trigger │  │Monitor │          │   │
│       │  └────┬────┘  └────┬────┘  └───┬────┘          │   │
│       └───────┼────────────┼───────────┼────────────────┘   │
│               │            │           │                    │
│               └────────────┴───────────┘                    │
│                            │                                │
│                            ▼                                │
│                    ┌──────────────┐                         │
│                    │  RETRAINING  │                         │
│                    │   PIPELINE   │                         │
│                    ├──────────────┤                         │
│                    │ • Load Data  │                         │
│                    │ • K-Fold CV  │                         │
│                    │ • EWC Loss   │                         │
│                    │ • Evaluate   │                         │
│                    │ • Version    │                         │
│                    └──────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pros & Cons
| Pros | Cons |
|------|------|
| ✅ Production-ready patterns | ❌ Less academically novel |
| ✅ Industry-standard MLOps | ❌ Doesn't solve domain shift directly |
| ✅ Practical for thesis demo | ❌ May need labeled data |
| ✅ Clear implementation path | ❌ Performance may plateau |

### Implementation Effort: **3-4 weeks**

---

# PATH C: Reference Implementation (From GitHub Repos)
## *Adapt Vehicle Insurance / YT-Capstone Patterns*

### Learning from Your Previous Projects

#### From `Vehicle-Insurance-DataPipeline-MLops-`
```
📦 Key Components to Adapt:
├── src/components/
│   ├── data_ingestion.py      → Load new Garmin data
│   ├── data_validation.py     → Schema checks + drift detection
│   ├── data_transformation.py → Preprocessing pipeline
│   ├── model_trainer.py       → Retraining logic
│   ├── model_evaluation.py    → Compare with production model
│   └── model_pusher.py        → Deploy to S3/local
│
├── src/entity/
│   ├── config_entity.py       → Configuration dataclasses
│   └── artifact_entity.py     → Pipeline artifacts tracking
│
├── src/pipeline/
│   ├── training_pipeline.py   → Orchestrate full retraining
│   └── prediction_pipeline.py → Inference pipeline (already have)
│
└── CI/CD:
    └── .github/workflows/     → GitHub Actions for automation
```

#### Key Pattern: Model Evaluation Gate
```python
# From Vehicle Insurance project
class ModelEvaluation:
    def start_model_evaluation(self, data_ingestion_artifact, model_trainer_artifact):
        """
        Compares newly trained model with production model
        Only promotes if performance improves
        """
        if not model_evaluation_artifact.is_model_accepted:
            logging.info("Model not accepted - keeping production version")
            return None
        
        # Push new model to production
        model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
```

#### From `YT-Capstone-Project`
```
📦 Additional Components:
├── MLflow Integration
│   ├── DagsHub for remote tracking
│   ├── Model Registry (Staging → Production)
│   └── Experiment comparison
│
├── CI/CD Pipeline
│   ├── tests/test_model.py    → Model validation tests
│   ├── scripts/promote_model.py → Stage → Production promotion
│   └── Prometheus/Grafana     → Monitoring (optional)
│
└── Deployment
    ├── Docker + ECR
    ├── EKS for scaling
    └── GitHub Actions automation
```

#### From `house-price-predictor_MLops_U`
```
📦 Streamlined Patterns:
├── src/data/run_processing.py  → Clean data processing script
├── src/features/engineer.py    → Feature engineering pipeline
├── src/models/train_model.py   → Model training with MLflow
├── src/api/
│   ├── main.py                 → FastAPI endpoints
│   ├── inference.py            → Prediction logic
│   └── schemas.py              → Pydantic models
│
└── docker-compose.yaml         → Multi-container deployment
```

### Proposed HAR Pipeline Structure
```
MasterArbeit_MLops/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Load Garmin CSV/real-time
│   │   ├── data_validation.py       # Schema + Drift Detection
│   │   ├── data_transformation.py   # Sensor preprocessing
│   │   ├── model_trainer.py         # Fine-tuning with EWC
│   │   ├── model_evaluation.py      # Compare vs production
│   │   └── model_pusher.py          # Version & deploy
│   │
│   ├── entity/
│   │   ├── config_entity.py         # DataIngestionConfig, etc.
│   │   └── artifact_entity.py       # DataIngestionArtifact, etc.
│   │
│   ├── pipeline/
│   │   ├── inference_pipeline.py    # Current: Garmin → Prediction
│   │   └── training_pipeline.py     # NEW: Retraining orchestration
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py        # KS-test implementation
│   │   └── retraining_triggers.py   # Trigger logic
│   │
│   └── utils/
│       └── main_utils.py            # Helper functions
│
├── config/
│   ├── schema.yaml                  # Data schema definition
│   └── model_config.yaml            # Model hyperparameters
│
├── docker/
│   ├── Dockerfile.training
│   └── Dockerfile.inference
│
├── .github/workflows/
│   └── ci_cd.yaml                   # GitHub Actions pipeline
│
└── tests/
    ├── test_model_loading.py
    └── test_drift_detection.py
```

### Pros & Cons
| Pros | Cons |
|------|------|
| ✅ Proven patterns you've used | ❌ Adapting for sensor data |
| ✅ Quick to implement | ❌ May not address domain shift |
| ✅ Complete MLOps structure | ❌ Extra work to integrate research |
| ✅ GitHub Actions ready | ❌ Need to adapt from tabular to time-series |

### Implementation Effort: **2-3 weeks** (structure), **+2 weeks** (integration)

---

## 🎯 RECOMMENDATION: HYBRID APPROACH

### Best Strategy for Your Thesis

**Combine Path B + Path C with Path A as optional enhancement**

```
Week 1-2: Set up MLOps Structure (Path C)
├── Create entity/component structure
├── Adapt from Vehicle Insurance patterns
├── Set up basic training pipeline skeleton

Week 3-4: Implement Monitoring (Path B)
├── KS-test drift detection
├── Entropy-based confidence monitoring
├── Retraining triggers

Week 5-6: Add Retraining Pipeline (Path B + C)
├── Fine-tuning with EWC (prevents forgetting)
├── K-Fold CV for validation
├── Model evaluation gate

Week 7-8: CI/CD & Deployment (Path C)
├── GitHub Actions workflow
├── Docker containerization
├── MLflow model registry

Week 9-10 (Optional): Domain Adaptation (Path A)
├── Implement AdaBN if time permits
├── Test on Garmin data
├── Compare with baseline

Final: Documentation & Thesis Writing
├── Document all components
├── Create architecture diagrams
├── Write thesis chapters
```

---

## 📁 FILES TO CREATE NEXT

### Priority 1: Pipeline Structure
```
src/components/data_ingestion.py
src/components/data_validation.py  
src/entity/config_entity.py
src/entity/artifact_entity.py
src/pipeline/training_pipeline.py
```

### Priority 2: Monitoring
```
src/monitoring/drift_detector.py
src/monitoring/retraining_triggers.py
```

### Priority 3: CI/CD
```
.github/workflows/ci_cd.yaml
docker-compose.yaml (update)
```

---

## 📚 KEY RESEARCH REFERENCES

| Concept | Source | Use Case |
|---------|--------|----------|
| AdaBN | Multiple UDA papers | Simple domain adaptation |
| KS-Test | Statistical literature | Drift detection without labels |
| EWC | Kirkpatrick et al. 2017 | Prevent catastrophic forgetting |
| 6-volunteer fine-tuning | ICTH_16 | Validation benchmark (49%→87%) |
| Combined triggers | ICTH_16 + others | Retraining decision logic |

---

## ✅ NEXT ACTION

**Tell me which path or combination you want to start with, and I'll create the implementation files.**

Options:
1. **"Start with Path B+C"** → I'll create the MLOps structure + monitoring
2. **"Start with Path A"** → I'll implement AdaBN first
3. **"Full hybrid"** → I'll create complete structure with all components
4. **"Just CI/CD"** → Focus on deployment pipeline first

---

*Document maintained as thesis roadmap. Update as implementation progresses.*

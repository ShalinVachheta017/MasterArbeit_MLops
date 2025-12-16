# 📚 Research Paper Analysis: HAR MLOps Pipeline Improvements

> **Analysis Date:** December 13, 2025  
> **Papers Reviewed:** 76+ papers from `research_papers/76 papers/` folder  
> **Purpose:** Extract actionable improvements for the current HAR MLOps pipeline

---

## 📊 Summary of Key Papers Analyzed

| Paper | Core Contribution |
|-------|------------------|
| **EHB_2025_71** | RAG-enhanced multi-stage pipeline: HAR → bout analysis → LLM report generation |
| **ICTH_16** | Domain adaptation (lab-to-life gap): 49% → 87% accuracy via fine-tuning on Garmin data |
| **ADAM-sense** (Khan et al., 2021) | Foundational dataset for 11 anxiety-related activities |
| Deep CNN-LSTM With Self-Attention | Self-attention for improved temporal modeling |
| Multi-Head CNN followed by LSTM | Multi-scale feature extraction |
| MLOps: A Survey | Best practices for ML operations |
| Transfer Learning in HAR: A Survey | Domain adaptation strategies |
| Foundation Model for Wearable Sensing | Pre-trained models for sensor data |

---

## 🎯 Current Pipeline Status

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CURRENT IMPLEMENTATION                             │
├──────────────────────────────────────────────────────────────────────┤
│  Model:        1D-CNN-BiLSTM (499,131 parameters)                    │
│  Input:        200 timesteps × 6 sensors at 50Hz                     │
│  Output:       11 anxiety-related activity classes                   │
│  Calibration:  -6.295 m/s² Az offset for Garmin                     │
│  Tracking:     MLflow for experiments                                │
│  Versioning:   DVC for data                                          │
│  Deploy:       Docker containerization                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ Model Architecture Improvements

### Current Architecture
- **1D-CNN-BiLSTM** (499,131 parameters)
- Input: 200 timesteps × 6 sensors at 50Hz
- Output: 11 activity classes

### 📈 Recommended Improvements

| Improvement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **Add Self-Attention** | "Deep CNN-LSTM With Self-Attention Model" | Add attention layer between BiLSTM and Dense for long-range dependencies | ⭐⭐⭐ |
| **Multi-Head CNN** | "Multi-Head CNN followed by LSTM" | Parallel CNN heads with kernel sizes (3, 5, 7) for multi-scale extraction | ⭐⭐ |
| **Ablation Study** | ICTH_16 | Document: 1DCNNBiLSTM (F1: 0.871) > BiLSTM (0.813) > LSTM (0.828) > CNN (0.697) | ⭐⭐⭐ |
| **Lightweight Models** | "Lightweight HAR Framework" | Model pruning/quantization for edge deployment (TensorFlow Lite) | ⭐ |
| **Foundation Models** | "Foundation Model for HAR" | Pre-trained HAR models as feature extractors | ⭐ |

### 💻 Implementation Example: Self-Attention Layer

```python
# Add to model after BiLSTM layer
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=64):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm = LayerNormalization()
        
    def call(self, x):
        attention_output = self.attention(x, x)
        return self.norm(x + attention_output)  # Residual connection

# Usage in model:
# x = BiLSTM(64, return_sequences=True)(x)
# x = AttentionBlock(num_heads=4, key_dim=64)(x)
# x = GlobalAveragePooling1D()(x)
```

---

## 2️⃣ Preprocessing & Feature Engineering

### Current Preprocessing
- Sensor fusion: accelerometer + gyroscope
- 50Hz resampling
- 200-timestep windows (4 seconds) with 50% overlap
- Domain calibration: -6.295 m/s² Az offset

### 📈 Recommended Improvements

| Improvement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **Temporal Bout Analysis** | EHB_2025_71 | Add bout detection (consecutive same-activity windows) with gap thresholds | ⭐⭐⭐ |
| **Multi-Scale Windowing** | "ML-EX-SW: MLOps for Sliding Windows" | Test window sizes: 2s, 4s, 8s - select optimal per-activity | ⭐⭐ |
| **Data Augmentation** | "Self-Supervised HAR" | Add jittering, scaling, rotation, time-warping | ⭐⭐⭐ |
| **Time-of-Day Features** | EHB_2025_71 | Extract metadata: morning/afternoon/evening patterns | ⭐ |
| **Gravity Removal Comparison** | Existing pipeline | Document effect on classification accuracy | ⭐⭐ |

### 💻 Configuration Example

```yaml
# Add to config/pipeline_config.yaml
preprocessing:
  windows:
    sizes: [100, 200, 400]  # 2s, 4s, 8s at 50Hz
    overlap: 0.5
    
  augmentation:
    enabled: true
    jitter_sigma: 0.05
    scaling_range: [0.9, 1.1]
    rotation_range: [-10, 10]
    time_warp: true
    
  bout_analysis:
    enabled: true
    hr_gap_threshold_seconds: 120
    behavior_gap_threshold_seconds: 300
    min_bout_duration_seconds: 5
```

### 💻 Data Augmentation Code

```python
# Add to src/data_augmentation.py
import numpy as np

def augment_sensor_data(X, jitter=0.05, scaling=(0.9, 1.1)):
    """Apply data augmentation to sensor windows"""
    augmented = X.copy()
    
    # Jittering: Add Gaussian noise
    noise = np.random.normal(0, jitter, X.shape)
    augmented += noise
    
    # Scaling: Random magnitude scaling
    scale = np.random.uniform(scaling[0], scaling[1])
    augmented *= scale
    
    return augmented

def time_warp(X, sigma=0.2):
    """Apply time warping augmentation"""
    # Smooth random time distortion
    pass  # Implementation here
```

---

## 3️⃣ MLOps/Pipeline Improvements

### Current MLOps Stack
- **MLflow** for experiment tracking
- **DVC** for data versioning  
- **Docker** for containerization
- Python scripts for pipeline execution

### 📈 Recommended Improvements

| Improvement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **Drift Detection** | "MLOps for Wearable HAR" | Add KS-test/PSI between train and production data | ⭐⭐⭐ |
| **Auto Retraining Triggers** | "MLOps: A Survey" | Define performance thresholds for automatic fine-tuning | ⭐⭐ |
| **Enhanced MLflow Logging** | "MLDev" | Log confusion matrices, per-class metrics as artifacts | ⭐⭐⭐ |
| **CI/CD Pipeline** | "End-to-End ML Replicability" | GitHub Actions for automated testing/deployment | ⭐⭐ |
| **Model Registry** | MLflow native | Use stages: Staging → Production | ⭐⭐ |
| **A/B Testing** | "MLOps Framework for HAR" | Shadow mode for new model versions | ⭐ |

### 💻 Drift Detection Implementation

```python
# Add to src/data_validator.py
from scipy.stats import ks_2samp
import numpy as np

def detect_distribution_drift(train_data, prod_data, threshold=0.05):
    """
    Detect drift using Kolmogorov-Smirnov test
    
    Args:
        train_data: Training data array (n_samples, timesteps, features)
        prod_data: Production data array
        threshold: P-value threshold for drift detection
        
    Returns:
        dict: Drift detection results per sensor
    """
    sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    drift_results = {}
    
    for idx, sensor in enumerate(sensor_names):
        train_flat = train_data[:, :, idx].flatten()
        prod_flat = prod_data[:, :, idx].flatten()
        
        stat, p_value = ks_2samp(train_flat, prod_flat)
        
        drift_results[sensor] = {
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'drift_detected': p_value < threshold
        }
    
    return drift_results

def log_drift_to_mlflow(drift_results):
    """Log drift metrics to MLflow"""
    import mlflow
    
    for sensor, results in drift_results.items():
        mlflow.log_metric(f"drift_ks_{sensor}", results['ks_statistic'])
        mlflow.log_metric(f"drift_pvalue_{sensor}", results['p_value'])
    
    any_drift = any(r['drift_detected'] for r in drift_results.values())
    mlflow.log_metric("drift_detected", int(any_drift))
```

---

## 4️⃣ Domain Adaptation Techniques

### Current Approach
- Pre-trained on ADAM-sense dataset
- Fine-tuned on Garmin Venu 3 data
- Performance improvement: **49% → 87%** after fine-tuning

### 📈 Recommended Improvements

| Improvement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **Quantify Domain Shift** | ICTH_16 | Always log baseline accuracy before fine-tuning | ⭐⭐⭐ |
| **Layer Freezing Strategy** | "Transfer Learning in HAR" | Compare: freeze early layers vs. full fine-tuning | ⭐⭐ |
| **Device Calibration Pipeline** | ICTH_16 | Automate unit conversion + offset calculation | ⭐⭐⭐ |
| **Cross-Device Validation** | "Domain Adaptation for IMU" | Test on multiple device types | ⭐ |
| **Self-Supervised Pre-Training** | "Self-Supervised HAR" | Contrastive learning for device-agnostic features | ⭐ |

### 💻 Domain Adaptation Utility

```python
# Add to src/domain_adaptation.py
import numpy as np

class DomainAdapter:
    """Utility for adapting models to new device data"""
    
    def __init__(self, base_model, freeze_layers=5):
        self.model = base_model
        self.freeze_layers = freeze_layers
        
    def prepare_for_finetuning(self):
        """Freeze early CNN layers for fine-tuning"""
        for i, layer in enumerate(self.model.layers):
            if i < self.freeze_layers:
                layer.trainable = False
            else:
                layer.trainable = True
        return self.model
    
    @staticmethod
    def calculate_calibration_offset(source_data, target_data):
        """
        Calculate calibration offset between source and target device
        
        Args:
            source_data: Data from source device (ADAM-sense)
            target_data: Data from target device (Garmin)
            
        Returns:
            dict: Calibration offsets per sensor axis
        """
        offsets = {}
        sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        
        for idx, sensor in enumerate(sensor_names):
            source_mean = np.mean(source_data[:, :, idx])
            target_mean = np.mean(target_data[:, :, idx])
            offsets[sensor] = source_mean - target_mean
            
        return offsets
    
    @staticmethod
    def apply_calibration(data, offsets):
        """Apply calibration offsets to target data"""
        calibrated = data.copy()
        for idx, sensor in enumerate(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']):
            if sensor in offsets:
                calibrated[:, :, idx] += offsets[sensor]
        return calibrated
```

---

## 5️⃣ Evaluation Methods

### Current Evaluation
- Confusion matrix
- Per-class accuracy
- Activity distribution

### 📈 Recommended Improvements

| Improvement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **5-Fold Cross-Validation** | ICTH_16 | Report mean ± std (achieved 87.0% ± 1.2%) | ⭐⭐⭐ |
| **F1-Macro Score** | Standard practice | Primary metric for class imbalance | ⭐⭐⭐ |
| **Ablation Study** | ICTH_16 | Compare CNN-only, LSTM-only, BiLSTM-only, hybrid | ⭐⭐ |
| **Confidence Calibration** | "Wearable AI for Anxiety" | Reject low-confidence predictions | ⭐⭐ |
| **Per-Activity Analysis** | EHB_2025_71 | Identify weak activities (e.g., hand_scratching: F1=0.78) | ⭐⭐ |

### 💻 Enhanced Evaluation Code

```python
# Add to src/evaluate_predictions.py
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np

def calculate_advanced_metrics(y_true, y_pred, y_prob, class_names):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (n_samples, n_classes)
        class_names: List of activity names
        
    Returns:
        dict: Comprehensive metrics
    """
    metrics = {
        # Overall metrics
        'accuracy': np.mean(y_true == y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        
        # Per-class F1
        'per_class_f1': dict(zip(
            class_names, 
            f1_score(y_true, y_pred, average=None)
        )),
        
        # Confidence metrics
        'confidence_mean': float(y_prob.max(axis=1).mean()),
        'confidence_std': float(y_prob.max(axis=1).std()),
        
        # Low confidence samples
        'low_confidence_count': int((y_prob.max(axis=1) < 0.5).sum()),
        'low_confidence_pct': float((y_prob.max(axis=1) < 0.5).mean() * 100),
    }
    
    return metrics

def identify_weak_classes(per_class_f1, threshold=0.80):
    """Identify activities with F1 score below threshold"""
    weak = {k: v for k, v in per_class_f1.items() if v < threshold}
    return weak
```

---

## 6️⃣ Novel Approaches: RAG, LLMs, and Foundation Models

### Key Insight from EHB_2025_71

The research describes a **multi-stage pipeline** that extends beyond prediction:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Sensor  │ →  │  HAR Model  │ →  │ Temporal    │ →  │ RAG-Enhanced│
│    Data     │    │ Predictions │    │Bout Analysis│    │ LLM Reports │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                    ┌───────────────────────────┼───────────┐
                                    ▼                           ▼           ▼
                            ┌─────────────┐           ┌─────────────┐  ┌─────────────┐
                            │  Clinical   │           │   Patient   │  │  Detailed   │
                            │   Summary   │           │   Summary   │  │   Report    │
                            └─────────────┘           └─────────────┘  └─────────────┘
```

### 📈 Future Enhancements

| Enhancement | Source Paper(s) | Specific Recommendation | Priority |
|-------------|-----------------|------------------------|----------|
| **RAG for Explainability** | EHB_2025_71 | Knowledge base of activity definitions + clinical interpretations | ⭐⭐ |
| **Multi-Audience Reports** | EHB_2025_71 | Different formats: clinical, patient-friendly, research | ⭐ |
| **Knowledge Graph** | EHB_2025_71 | Neo4j for activity-behavior-clinical relationships | ⭐ |
| **LLM Integration** | "In-context learning" | Natural language summaries of prediction patterns | ⭐ |
| **Foundation Models** | "SensorLM" | Pre-trained sensor language models for features | ⭐ |

### 💻 Conceptual RAG Pipeline (Future Work)

```python
# Conceptual implementation for future thesis extension
class HARReportGenerator:
    """Generate clinical reports from HAR predictions using RAG"""
    
    def __init__(self, llm_model, knowledge_base):
        self.llm = llm_model
        self.kb = knowledge_base  # Vector store or Neo4j
        
    def analyze_bouts(self, predictions, timestamps):
        """
        Detect activity bouts (consecutive same-activity periods)
        
        Returns:
            DataFrame with bout statistics (activity, duration, count)
        """
        bouts = []
        current_activity = None
        bout_start = None
        
        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred != current_activity:
                if current_activity is not None:
                    bouts.append({
                        'activity': current_activity,
                        'start': bout_start,
                        'end': timestamps[i-1],
                        'duration_seconds': (timestamps[i-1] - bout_start).total_seconds()
                    })
                current_activity = pred
                bout_start = ts
                
        return pd.DataFrame(bouts)
    
    def generate_clinical_summary(self, bout_analysis):
        """Generate RAG-enhanced clinical summary"""
        # Retrieve relevant clinical knowledge
        context = self.kb.retrieve_similar(
            f"anxiety activities: {bout_analysis.activity.unique().tolist()}"
        )
        
        prompt = f"""
        Based on the following activity bout analysis:
        {bout_analysis.to_json(orient='records')}
        
        And clinical context from knowledge base:
        {context}
        
        Generate a clinical summary for the treating psychologist.
        Include:
        1. Key behavioral patterns observed
        2. Comparison to typical anxiety presentation
        3. Suggested monitoring focus areas
        """
        
        return self.llm.generate(prompt)
```

---

## 🎯 Prioritized Action Plan

### Phase 1: Quick Wins (1-2 weeks) ⭐⭐⭐

| Task | File to Modify | Effort |
|------|---------------|--------|
| Add 5-fold cross-validation | `src/evaluate_predictions.py` | Low |
| Implement drift detection | `src/data_validator.py` | Low |
| Log more MLflow metrics | `src/mlflow_tracking.py` | Low |
| Document ablation study | `docs/` | Low |
| Add F1-macro as primary metric | `src/evaluate_predictions.py` | Low |

### Phase 2: Model Enhancements (2-4 weeks) ⭐⭐

| Task | File to Create/Modify | Effort |
|------|----------------------|--------|
| Add self-attention layer | `src/model.py` | Medium |
| Implement data augmentation | `src/data_augmentation.py` | Medium |
| Add temporal bout analysis | `src/bout_analysis.py` | Medium |
| Create domain adaptation utilities | `src/domain_adaptation.py` | Medium |

### Phase 3: MLOps Maturity (4-6 weeks) ⭐

| Task | Location | Effort |
|------|----------|--------|
| Automated retraining triggers | `src/retraining.py` | High |
| CI/CD with GitHub Actions | `.github/workflows/` | Medium |
| MLflow Model Registry | `src/mlflow_tracking.py` | Medium |
| A/B testing capability | `docker/` | High |

### Phase 4: Advanced Features (Future Work) 🔮

| Task | Purpose | Effort |
|------|---------|--------|
| RAG-enhanced report generation | Clinical interpretability | High |
| Knowledge graph (Neo4j) | Activity-symptom relationships | High |
| Foundation model integration | Better feature extraction | High |

---

## 📚 Key Citations for Thesis

```bibtex
@article{oleh2025multistage,
  title={A Multi-Stage, RAG-Enhanced Pipeline for Generating Longitudinal, 
         Clinically Actionable Mental Health Reports from Wearable Sensor Data},
  author={Oleh, Ugonna and Obermaisser, Roman and Malchulska, Alla and Klucken, Tim},
  journal={EHB 2025},
  year={2025}
}

@inproceedings{oleh2025anxiety,
  title={Recognition of Anxiety-Related Activities using 1DCNNBiLSTM 
         on Sensor Data from a Commercial Wearable Device},
  author={Oleh, Ugonna and Obermaisser, Roman},
  booktitle={ICTH 2025},
  year={2025}
}

@article{khan2021adamsense,
  title={ADAM-sense: Anxiety-displaying activities recognition by motion sensors},
  author={Khan, NS and Ghani, MS and Anjum, G},
  journal={Pervasive and Mobile Computing},
  volume={72},
  pages={101333},
  year={2021}
}

@article{hewage2022mlops,
  title={Machine Learning Operations: A Survey on MLOps},
  author={Hewage, N and Meedeniya, D},
  journal={arXiv preprint arXiv:2202.10169},
  year={2022}
}

@article{khatun2022cnnlstm,
  title={Deep CNN-LSTM With Self-Attention Model for Human Activity 
         Recognition Using Wearable Sensor},
  author={Khatun, Mst Alema and Yousuf, Mohammad Abu and others},
  journal={IEEE Journal of Translational Engineering in Health and Medicine},
  year={2022}
}

@article{dhekane2024transfer,
  title={Transfer Learning in Human Activity Recognition: A Survey},
  author={Dhekane, Sourish G and others},
  journal={ACM Computing Surveys},
  year={2024}
}
```

---

## 📊 Summary: Key Takeaways

### ✅ What's Already Good
1. **Architecture validated**: 1D-CNN-BiLSTM matches ICTH_16 research (F1: 0.87)
2. **Domain adaptation works**: 49% → 87% via fine-tuning
3. **MLOps foundation solid**: MLflow + DVC + Docker

### 🔄 Main Improvement Areas
1. **Add self-attention** for better temporal modeling
2. **Implement drift detection** for production monitoring
3. **Enhance evaluation** with cross-validation and F1-macro
4. **Add data augmentation** for robustness

### 🔮 Future Research Directions
1. **RAG-enhanced clinical reports** (EHB_2025_71 pipeline)
2. **Foundation models** for sensor data
3. **Multi-device generalization** without fine-tuning

---

*Generated from analysis of 76+ research papers in `research_papers/76 papers/`*  
*Last Updated: December 13, 2025*

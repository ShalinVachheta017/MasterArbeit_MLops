# MLOps Thesis Project - Current State Assessment
**Date:** October 12, 2025  
**Status:** Pre-Training Phase - Setting Foundation

---

## 🎯 **PROJECT OVERVIEW**

**Thesis Goal:** Developing an MLOps Pipeline for Continuous Mental Health Monitoring using Wearable Sensor Data

**Model Type:** 1D-CNN-BiLSTM for Anxiety Activity Recognition

**Timeline:** 6-month thesis (You appear to be in Month 1-2)

---

## ✅ **WHAT YOU HAVE (STRENGTHS)**

### 1. **Data Preprocessing Pipeline** ✅ EXCELLENT
- **Location:** `src/data_preprocessing.py` and `src/MDP.py`
- **Status:** Production-ready, modular architecture
- **Features:**
  - Automated sensor fusion (accelerometer + gyroscope)
  - Resampling to 50Hz with precise timestamp alignment
  - Robust logging and error handling
  - Quality: 95.1% sensor alignment success rate

### 2. **Pre-trained Model** ✅ AVAILABLE
- **Location:** `model/fine_tuned_model_1dcnnbilstm.keras`
- **Size:** 6MB
- **Format:** Keras/TensorFlow format
- **Status:** Model exists but architecture/training code is missing

### 3. **Data Files**
- **Raw Data:** 
  - `data/2025-03-23-15-23-10-accelerometer_data.xlsx` (14,536 rows)
  - `data/2025-03-23-15-23-10-gyroscope_data.xlsx` (14,536 rows)
  
- **Processed Data:**
  - `pre_processed_data/sensor_fused_50Hz.csv` (181,699 samples)
  - `pre_processed_data/sensor_merged_native_rate.csv` (345,418 samples)
  
- **Training Data (?):**
  - `f_data_50hz.csv` (69,365 samples, 7 columns)
  - **Question:** Does this have labels? What are the classes?

### 4. **Infrastructure**
- Proper directory structure (`logs/`, `model/`, `src/`, `data/`)
- Logging system in place
- Requirements file started

---

## ⚠️ **CRITICAL GAPS - MUST ADDRESS BEFORE TRAINING**

### 1. **MISSING: Training Code** 🔴 CRITICAL
**Problem:** You have a trained model but no code to train it

**What You Need:**
- Model architecture definition (1D-CNN-BiLSTM structure)
- Training loop with hyperparameters
- Data loading and windowing strategy
- Loss function and optimizer configuration

**Questions to Answer:**
- How was the existing model trained?
- What are the input/output dimensions?
- What is the window size for sequences?
- What are the hyperparameters?

### 2. **MISSING: Labels/Classes** 🔴 CRITICAL
**Problem:** No clear label information visible

**Questions to Answer:**
- What are you classifying? (Anxiety states? Activities? Stress levels?)
- How many classes? (Binary: anxious/calm? Multi-class: 3+ states?)
- Where are the labels? (Separate file? Embedded in data?)
- What is the label format?

**Example Expected Structure:**
```csv
timestamp, Ax, Ay, Az, Gx, Gy, Gz, label
..., ..., ..., ..., ..., ..., ..., 0  # Class 0: Calm
..., ..., ..., ..., ..., ..., ..., 1  # Class 1: Anxious
```

### 3. **MISSING: Data Preparation for Training** 🔴 CRITICAL
**Problem:** Raw time series needs to be converted to training samples

**What You Need:**
- **Sliding window approach:** Convert continuous time series into fixed-length sequences
- **Window size:** e.g., 100 samples (2 seconds at 50Hz) or 250 samples (5 seconds)
- **Overlap:** e.g., 50% overlap for more training samples
- **Normalization:** Standardize sensor values (mean=0, std=1)
- **Train/Val/Test split:** e.g., 70/15/15 or 60/20/20

**Example Expected Shape:**
```python
X_train.shape = (num_samples, window_size, num_features)
# e.g., (10000, 100, 6) = 10k samples, 100 timesteps, 6 sensors

y_train.shape = (num_samples, num_classes)
# e.g., (10000, 2) for binary classification (one-hot encoded)
```

### 4. **MISSING: TensorFlow Installation** 🟡 MEDIUM
**Problem:** Model is in Keras format, but TensorFlow not installed

**Solution:** Add to `requirements.txt`:
```python
tensorflow>=2.14  # For CPU
# OR
tensorflow-gpu>=2.14  # For GPU (if CUDA available)
```

---

## 📊 **DATA ANALYSIS NEEDED**

### File: `f_data_50hz.csv` (69,365 samples)
**Questions:**
1. Is this your **training dataset** with labels?
2. What does `Ax_w`, `Ay_w`, etc. mean? (Why "_w" suffix?)
3. Does this data come from the same preprocessing pipeline?
4. Time range: 2005? (Seems like timestamp issue or demo data?)

### File: `sensor_fused_50Hz.csv` (181,699 samples)
**Questions:**
1. Is this **unlabeled** sensor data from your preprocessing?
2. Should this be labeled manually or automatically?
3. Is this the data for inference/testing?

---

## 🎯 **RECOMMENDED ACTION PLAN**

### **PHASE 1: UNDERSTAND YOUR DATA (CURRENT - Week 1)**

#### Step 1.1: Analyze Data Files ✅ (Today)
```python
# Create: src/analyze_data.py
# Objectives:
# - Load f_data_50hz.csv and sensor_fused_50Hz.csv
# - Compare structures
# - Check for labels
# - Visualize sensor patterns
# - Compute statistics (mean, std, min, max per sensor)
```

#### Step 1.2: Understand Your Model (Today)
```python
# Install TensorFlow
# Load model and inspect:
# - Input shape
# - Output shape (number of classes)
# - Layer architecture
# - Total parameters
```

#### Step 1.3: Define Your Classification Task (Today/Tomorrow)
**YOU MUST ANSWER:**
- What am I classifying? (e.g., "Anxiety level: Low/Medium/High")
- How many classes? (e.g., 2 for binary, 3+ for multi-class)
- Where are my labels? (Do I need to create them?)

---

### **PHASE 2: PREPARE FOR TRAINING (Week 1-2)**

#### Step 2.1: Create Data Preparation Script
```python
# File: src/prepare_training_data.py

# Functions needed:
1. load_sensor_data(filepath) → DataFrame
2. create_sliding_windows(data, window_size, overlap) → numpy arrays
3. normalize_data(windows) → normalized arrays
4. split_data(X, y, test_size) → train/val/test sets
5. save_prepared_data(X, y, output_dir) → saved .npy files
```

**Example Expected Output:**
```
prepared_data/
  ├── X_train.npy  (shape: (8000, 100, 6))
  ├── y_train.npy  (shape: (8000, 2))
  ├── X_val.npy    (shape: (1000, 100, 6))
  ├── y_val.npy    (shape: (1000, 2))
  ├── X_test.npy   (shape: (1000, 100, 6))
  ├── y_test.npy   (shape: (1000, 2))
  └── metadata.json (window_size, overlap, normalization params)
```

#### Step 2.2: Create Model Architecture File
```python
# File: src/model_architecture.py

def build_1dcnn_bilstm(input_shape, num_classes):
    """
    Build 1D-CNN-BiLSTM model
    
    Args:
        input_shape: tuple (window_size, num_features) e.g., (100, 6)
        num_classes: int, number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Define layers:
    # 1. Conv1D layers for feature extraction
    # 2. MaxPooling1D for dimensionality reduction
    # 3. Bidirectional LSTM for temporal patterns
    # 4. Dropout for regularization
    # 5. Dense layers for classification
```

#### Step 2.3: Create Training Script with MLflow
```python
# File: src/train_model.py

# Components:
1. Load prepared data
2. Build model architecture
3. Set up MLflow experiment tracking
4. Define callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
5. Train model with validation
6. Log metrics, parameters, and model to MLflow
7. Save best model to model/ directory
```

#### Step 2.4: Create Evaluation Script
```python
# File: src/evaluate_model.py

# Functions:
1. load_model_and_data()
2. make_predictions()
3. compute_metrics(y_true, y_pred):
   - Accuracy
   - Precision, Recall, F1-Score (per class and macro/micro avg)
   - Confusion Matrix
   - Classification Report
4. visualize_results()
5. save_evaluation_report()
```

---

### **PHASE 3: MLOps INTEGRATION (Week 3-4)**

#### Step 3.1: Dockerize Training
```dockerfile
# Dockerfile
FROM tensorflow/tensorflow:2.14-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY data/ ./data/
CMD ["python", "src/train_model.py"]
```

#### Step 3.2: Set Up MLflow
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000
```

#### Step 3.3: Create CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/train_model.yml
name: Train Model
on:
  push:
    paths:
      - 'src/**'
      - 'data/**'
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and train
        run: |
          docker build -t anxiety-model .
          docker run anxiety-model
```

---

## 🚨 **IMMEDIATE NEXT STEPS (TODAY)**

### Step 1: Install TensorFlow and Inspect Model
```bash
# Update requirements.txt
echo "tensorflow>=2.14" >> requirements.txt
pip install tensorflow

# Inspect model
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('model/fine_tuned_model_1dcnnbilstm.keras')
print('='*60)
print('MODEL ARCHITECTURE')
print('='*60)
model.summary()
print('\n' + '='*60)
print('INPUT/OUTPUT SHAPES')
print('='*60)
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
print(f'Number of classes: {model.output_shape[-1]}')
"
```

### Step 2: Analyze Your Data
I'll create a script for you to run that will:
- Load both CSV files
- Compare structures
- Check for labels
- Provide recommendations

### Step 3: Answer Critical Questions
**YOU MUST PROVIDE:**
1. **What are you classifying?** (e.g., "Anxiety levels: 0=Calm, 1=Mild, 2=Severe")
2. **Do you have labels?** (Yes/No - where are they?)
3. **What is your window size?** (How many samples per training example?)
4. **What was the model trained on?** (Ask your mentor for details)

---

## 📝 **EVALUATION METRICS (For Classification)**

### For Binary Classification (2 classes):
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred),
    'confusion_matrix': confusion_matrix(y_true, y_pred)
}
```

### For Multi-class Classification (3+ classes):
```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision_macro': precision_score(y_true, y_pred, average='macro'),
    'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
    'recall_macro': recall_score(y_true, y_pred, average='macro'),
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'confusion_matrix': confusion_matrix(y_true, y_pred),
    'classification_report': classification_report(y_true, y_pred)
}
```

### Additional Metrics:
- **ROC-AUC Score** (for binary classification)
- **Cohen's Kappa** (inter-rater agreement)
- **Matthews Correlation Coefficient** (balanced accuracy)
- **Per-class Precision/Recall/F1**

---

## 🎯 **SUCCESS CRITERIA**

### For Your Thesis (Proof-of-Concept MLOps):
1. ✅ **Data Pipeline:** Automated preprocessing (YOU HAVE THIS)
2. ⏳ **Training Pipeline:** Reproducible training with experiment tracking
3. ⏳ **Model Versioning:** Track all models with metadata
4. ⏳ **Deployment:** Simple API for inference
5. ⏳ **Monitoring:** Basic data drift detection
6. ⏳ **CI/CD:** Automated testing and deployment
7. ⏳ **Documentation:** Complete thesis documentation

---

## 📚 **RESOURCES FOR YOUR THESIS**

### MLflow Experiment Tracking:
```python
import mlflow

# Log parameters
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("epochs", 50)
mlflow.log_param("batch_size", 32)

# Log metrics
mlflow.log_metric("train_loss", 0.15)
mlflow.log_metric("val_accuracy", 0.92)

# Log model
mlflow.keras.log_model(model, "model")
```

### Model Versioning:
```python
# Register model in MLflow Model Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="anxiety_activity_recognition"
)

# Transition to production
client.transition_model_version_stage(
    name="anxiety_activity_recognition",
    version=1,
    stage="Production"
)
```

---

## ⚠️ **CRITICAL QUESTIONS FOR YOUR MENTOR**

Before proceeding, **ASK YOUR MENTOR:**

1. **Model Details:**
   - What are the exact input/output dimensions?
   - What is the window size (sequence length)?
   - How many classes and what do they represent?
   - What were the training hyperparameters?

2. **Data Details:**
   - What is `f_data_50hz.csv`? Does it have labels?
   - How were the classes/labels defined?
   - What is the class distribution? (Balanced/Imbalanced?)

3. **Expected Performance:**
   - What was the original model's accuracy?
   - What metrics should I aim for?
   - What is considered "good" performance for this task?

4. **Thesis Scope:**
   - Should I focus on reproducing the model first?
   - Or should I prioritize building the MLOps pipeline?
   - What is the minimum viable deliverable?

---

## 🚀 **NEXT STEPS SUMMARY**

| Priority | Task | Time | Status |
|----------|------|------|--------|
| 🔴 HIGH | Install TensorFlow & inspect model | 30 min | NOT STARTED |
| 🔴 HIGH | Analyze data files (labels?) | 1 hour | IN PROGRESS |
| 🔴 HIGH | Get model training details from mentor | ASAP | PENDING |
| 🟡 MEDIUM | Create data preparation script | 2-3 hours | NOT STARTED |
| 🟡 MEDIUM | Create model architecture file | 1-2 hours | NOT STARTED |
| 🟢 LOW | Write training script with MLflow | 3-4 hours | NOT STARTED |
| 🟢 LOW | Write evaluation script | 2 hours | NOT STARTED |

---

## 💡 **MY RECOMMENDATIONS**

### **Immediate (Today):**
1. ✅ Install TensorFlow
2. ✅ Inspect the model architecture
3. ✅ Analyze both CSV files to find labels
4. ✅ Contact mentor for model training details

### **This Week:**
1. Create data analysis script
2. Build data preparation pipeline
3. Write model architecture definition
4. Test model loading and inference

### **Next Week:**
1. Implement training script
2. Set up MLflow tracking
3. Train a baseline model
4. Evaluate and compare with original

### **For Your Thesis:**
- **Focus on proof-of-concept**, not production-grade system
- **Document everything** you build
- **Version control** with Git
- **Reproducibility** is key (Docker, requirements.txt, configs)

---

**Ready to start? Let me know what you find when you inspect the model and data, and I'll help you build the complete training and evaluation pipeline! 🚀**

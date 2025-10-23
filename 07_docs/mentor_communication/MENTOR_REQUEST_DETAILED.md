# Request for Training Labels and Guidance
**Date:** October 23, 2025  
**Student:** [Your Name]  
**Thesis Title:** Developing an MLOps Pipeline for Continuous Mental Health Monitoring using Wearable Sensor Data  
**Timeline:** 6 months (Started: October 2025)

---

## 📊 Current Progress Summary (Phase 1 Complete - 25%)

### ✅ What Has Been Completed

#### 1. Data Preprocessing Pipeline
- **Built:** Modular preprocessing system with 8 specialized classes
- **Input:** Raw Excel files from March 2025 session
  - `2025-03-23-15-23-10-accelerometer_data.xlsx`
  - `2025-03-23-15-23-10-gyroscope_data.xlsx`
- **Processing Steps:**
  - Normalized column names and parsed list-format cells
  - Exploded array data to individual samples
  - Aligned accelerometer ↔ gyroscope by timestamp (1ms precision)
  - Achieved 95.1% successful sensor alignment
  - Resampled all data to exactly 50Hz sampling rate
  - Added epoch millisecond timestamps for ML pipeline
- **Output Files Generated:**
  - `sensor_fused_50Hz.csv` (181,699 samples)
  - `sensor_merged_native_rate.csv` (native rate)
  - Metadata JSON with processing configuration

#### 2. Model Architecture Analysis
- **Inspected:** Pre-trained 1D-CNN-BiLSTM model
- **Model Specifications:**
  - Input Shape: (None, 200, 6) → 200 timesteps × 6 sensors
  - Output Shape: (None, 11) → 11-class classification
  - Total Parameters: 1,496,307 (1.5M)
  - Trainable Parameters: 498,587
  - Architecture Layers:
    - 2× Conv1D (16, 32 filters)
    - 2× Bidirectional LSTM (64, 32 units)
    - 5× BatchNormalization
    - 5× Dropout
    - 2× Dense layers (32, 11 units)
  - Optimizer: Adam (learning_rate = 0.0001)
  - Loss Function: categorical_crossentropy
- **Output:** `model/model_info.json` with complete architecture details

#### 3. Data Quality Analysis
- **Analyzed Files:**
  1. `f_data_50hz.csv` (69,365 samples) - Original file you provided
  2. `sensor_fused_50Hz.csv` (181,699 samples) - My preprocessed output
  3. Raw Excel files - Original sensor data
- **Quality Metrics:**
  - Missing values: Only 0.014% (26 per sensor out of 181,699)
  - Sensor alignment: 95.1% success rate
  - All sensor readings within expected physical ranges
  - Timestamps properly synchronized
- **Outputs Generated:**
  - Statistical analysis JSON files
  - Distribution histograms for all 6 sensors
  - Time series visualization samples

---

## 🔴 CRITICAL BLOCKER - Missing Training Labels

### The Problem

**None of the data files contain training labels (ground truth classifications).**

### Files Analyzed - No Labels Found

#### 1. Preprocessed Output (from my pipeline)
**File:** `sensor_fused_50Hz.csv` (181,699 samples)
```
Columns present:
- timestamp_ms (epoch milliseconds)
- timestamp_iso (ISO format timestamp)
- Ax, Ay, Az (accelerometer x, y, z)
- Gx, Gy, Gz (gyroscope x, y, z)

Missing: label, class, anxiety_level, or any classification column
```

**File:** `sensor_merged_native_rate.csv` (native sampling rate)
```
Same column structure as above
Missing: No label column
```

#### 2. Original Data Files
**File:** `f_data_50hz.csv` (69,365 samples)
```
Columns present:
- timestamp
- Ax_w, Ay_w, Az_w (accelerometer)
- Gx_w, Gy_w, Gz_w (gyroscope)

Missing: No label column
```

#### 3. Raw Excel Files (March 2025 session)
**Files:** 
- `2025-03-23-15-23-10-accelerometer_data.xlsx`
- `2025-03-23-15-23-10-gyroscope_data.xlsx`

```
Columns present:
- timestamp, timestamp_ms, sample_time_offset
- x, y, z (sensor values in list format)

Missing: No label columns in either file
```

### What This Means

All files contain **only sensor readings** (accelerometer and gyroscope values) but **no information about:**
- Which anxiety level each sample represents
- Which mental state (calm/anxious/stressed) each reading corresponds to
- Any ground truth classification that the model should learn

---

## ❓ Questions - Why I Need This Information

### Question 1: Where Are the Training Labels?

**I need to know:**
- Do you have a separate file containing labels for this sensor data?
- Are labels stored in a different format (database, Excel sheet, text file)?
- Were labels collected during the same March 2025 session?
- How should I match labels to the sensor timestamps?

**Why I need this:**
Without labels, I cannot create the training dataset. The model needs to learn "when sensor pattern X occurs, it means anxiety level Y" but I don't have the Y values.

### Question 2: What Do the 11 Classes Represent?

**I need to know:**
- What does each of the 11 output classes mean?
- Are they anxiety levels 0-10?
- Are they discrete mental states (calm, mild anxiety, moderate anxiety, etc.)?
- Is there a class description document?

**Why I need this:**
To properly interpret model predictions and create meaningful evaluation metrics. Also needed for thesis documentation and clinical interpretation.

### Question 3: What Is the Label Format?

**I need to know:**
- Integer class IDs (0, 1, 2, ... 10)?
- Text labels ("calm", "anxious", etc.)?
- One-hot encoded vectors?
- Continuous values that need binning?

**Why I need this:**
Different formats require different preprocessing. I need to convert labels to the format the model expects (one-hot encoded for categorical_crossentropy loss).

### Question 4: What Training Hyperparameters Were Used?

**I need to know:**
- Batch size during training
- Number of epochs trained
- Learning rate schedule (if different from 0.0001)
- Early stopping criteria
- Any data augmentation techniques used

**Why I need this:**
For reproducibility and thesis documentation. Also, if I need to fine-tune the model, I should use similar hyperparameters.

### Question 5: How Was Training Data Prepared?

**I need to know:**
- How did you create the 200-timestep sliding windows?
  - Window size: 200 samples = 4 seconds at 50Hz
  - Step size: overlapping or non-overlapping?
  - Padding for short sequences?
- What normalization was applied?
  - StandardScaler per sensor?
  - Min-max normalization?
  - Per-window or global statistics?
- Train/validation/test split percentages
- Was stratified sampling used for class balance?

**Why I need this:**
Must replicate your exact preprocessing to ensure consistency. Using different window sizes or normalization would make the model predictions unreliable.

### Question 6: What Was Model Performance?

**I need to know:**
- Overall accuracy achieved
- Per-class precision, recall, F1-score
- Confusion matrix results
- Were there class imbalances or problematic classes?

**Why I need this:**
Establishes baseline performance. Helps me understand if the model is production-ready or needs improvement. Critical for thesis evaluation chapter.

### Question 7: Alternative Path if Labels Unavailable

**I need to know:**
If labeled data cannot be provided, would it be acceptable to:
- Focus thesis purely on MLOps infrastructure (deployment, monitoring, versioning)
- Use the pre-trained model for inference only (no training pipeline)
- Demonstrate continuous deployment and monitoring without the training component

**Why I need this:**
Need to know if I should pivot my thesis scope or wait for labeled data. Both are valid thesis approaches but require different timelines.

---

## ⏸️ What Is Currently Blocked

All subsequent thesis phases cannot proceed without labels:

### Phase 2: Data Preparation (BLOCKED)
**Cannot build:** `prepare_training_data.py`
- Cannot create labeled 200-timestep sliding windows
- Cannot apply normalization without knowing your method
- Cannot split train/val/test without labels
- Cannot save labeled .npy files for training

### Phase 3: Model Architecture (BLOCKED)
**Cannot build:** `model_architecture.py`
- Need to verify architecture matches your training approach
- Cannot test model with sample data without labels
- Cannot validate input/output shapes without creating actual training batches

### Phase 4: Training Pipeline (BLOCKED)
**Cannot build:** `train_model.py`
- Cannot implement training loop without labeled data
- Cannot set up MLflow experiment tracking
- Cannot implement callbacks (need validation data with labels)
- Cannot save model checkpoints (no training to checkpoint)

### Phase 5: Evaluation System (BLOCKED)
**Cannot build:** `evaluate_model.py`
- Cannot calculate accuracy (no ground truth to compare against)
- Cannot create confusion matrix (no true labels vs predicted labels)
- Cannot compute precision/recall/F1-score per class
- Cannot generate evaluation reports

### Phase 6: MLOps Infrastructure (BLOCKED)
**Cannot deploy:**
- Cannot validate model performance before deployment
- Cannot set up drift detection (need baseline performance metrics)
- Cannot create monitoring dashboards (no metrics to monitor)
- Cannot document model limitations without evaluation results

---

## 🎯 What I Will Do Immediately After Receiving Labels

### Step 1: Data Preparation (Week 1)
1. **Create sliding windows script:**
   - Generate 200-timestep windows from continuous sensor data
   - Match each window to its corresponding label
   - Handle edge cases (short sequences, missing data)

2. **Apply normalization:**
   - Replicate your normalization method
   - StandardScaler per sensor channel
   - Save scaler parameters for inference

3. **Split dataset:**
   - Train: 70%, Validation: 15%, Test: 15%
   - Stratified sampling to maintain class distribution
   - Save splits with metadata

4. **Save prepared data:**
   - `X_train.npy`, `y_train.npy`
   - `X_val.npy`, `y_val.npy`
   - `X_test.npy`, `y_test.npy`
   - `data_preparation_config.json`

### Step 2: Training Pipeline (Week 2-3)
1. **Implement training script:**
   - MLflow experiment tracking
   - Model checkpointing
   - Early stopping callback
   - Learning rate reduction on plateau

2. **Log all metrics:**
   - Training/validation loss and accuracy per epoch
   - Per-class metrics
   - Training time and resource usage

3. **Save artifacts:**
   - Best model checkpoint
   - Training history plots
   - Hyperparameters configuration

### Step 3: Evaluation System (Week 4)
1. **Comprehensive evaluation:**
   - Test set accuracy, precision, recall, F1-score
   - Confusion matrix visualization
   - Per-class performance analysis
   - Error analysis (misclassified samples)

2. **Generate reports:**
   - PDF evaluation report
   - Visualization plots
   - Model card documentation

### Step 4: MLOps Deployment (Week 5-6)
1. **Build inference API:**
   - FastAPI RESTful API
   - Input validation
   - Preprocessing pipeline integration
   - Model serving with TensorFlow Serving

2. **Containerization:**
   - Docker containers
   - Docker Compose for orchestration
   - Environment configuration

3. **Monitoring setup:**
   - Drift detection (data distribution changes)
   - Performance monitoring (prediction latency, accuracy)
   - Logging and alerting

4. **CI/CD pipeline:**
   - GitHub Actions workflows
   - Automated testing
   - Model versioning with MLflow Registry

---

## 📅 Additional Request - Thesis Registration Guidance

### Thesis Registration Details

**Registration Date:** November 1st, 2025  
**University:** [Your University Name]  
**Department:** [Your Department]

### Documents Required

I will be submitting the thesis registration form (attached separately) and need guidance on:

1. **Thesis Scope Confirmation**
   - Should the thesis focus include the full training pipeline, or can it be MLOps infrastructure only?
   - Are there specific deliverables the department expects?

2. **Form Completion Guidance**
   - Research objectives section - should I list both training and deployment, or just deployment if labels unavailable?
   - Expected outcomes - what should I commit to delivering by the deadline?
   - Timeline - should I build contingencies for the label availability issue?

3. **Supervisor Approval**
   - What sections need your signature/approval?
   - Any comments you'd like me to include in the research plan?

4. **Technical Requirements**
   - Are there specific technologies the department requires/prefers?
   - Any restrictions on cloud services vs local deployment?

### What I've Prepared for Registration

- ✅ Clear problem statement (mental health monitoring MLOps pipeline)
- ✅ Literature review notes on MLOps for healthcare
- ✅ Technical stack identified (Python, TensorFlow, MLflow, FastAPI, Docker)
- ✅ Initial project structure and code organization
- ⏸️ Timeline - needs adjustment based on label availability

---

## 🕐 Timeline Impact

### Current Status
- **Started:** October 2025
- **Current Progress:** 25% (Phase 1 complete)
- **Time Elapsed:** ~3 weeks
- **Time Remaining:** ~5 months

### If Labels Received This Week
- **November:** Data preparation + training pipeline (2-3 weeks)
- **December:** Evaluation system + initial MLOps infrastructure (4 weeks)
- **January:** Complete MLOps deployment (4 weeks)
- **February:** Testing, monitoring, documentation (4 weeks)
- **March:** Thesis writing, final testing (4 weeks)
- **April:** Thesis submission
- **Status:** ✅ On track

### If Labels Delayed by 2-3 Weeks
- **Late November:** Data preparation + training pipeline (compressed)
- **December-January:** Fast-track evaluation and deployment
- **February-March:** Compressed testing and documentation
- **Status:** ⚠️ Tight but manageable

### If Labels Unavailable
- **November:** Pivot to MLOps-only focus (update registration)
- **November-December:** Inference pipeline + API deployment (6 weeks)
- **January:** Monitoring, drift detection, model versioning (4 weeks)
- **February:** CI/CD, testing, performance optimization (4 weeks)
- **March:** Documentation and thesis writing (4 weeks)
- **April:** Thesis submission
- **Status:** ✅ Alternative valid approach

---

## 📎 Attachments Included with This Document

1. **Thesis Registration Form** - To be filled out with your guidance
2. **This Document** - Complete context of work completed and blockers

---

## 🙏 Request Summary

**Immediate Needs:**
1. ✅ Labeled dataset or guidance on where to find it
2. ✅ Answers to 7 critical questions above
3. ✅ Guidance on thesis registration form completion
4. ✅ Confirmation of thesis scope (training pipeline vs MLOps-only)

**Optional:**
- Brief meeting to discuss if easier than written response
- Access to any documentation you used during model training

**Timeline:**
- Response this week would keep thesis on track
- Response within 2-3 weeks still manageable
- If labels unavailable, need confirmation to pivot approach

---

## 📧 Contact Information

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Email:** [Your Email]  
**Phone:** [Your Phone]  
**Preferred Contact Method:** [Email/Phone/Meeting]

---

**Thank you for your guidance and support!**

Looking forward to your response so I can proceed with the next phases of the thesis.

Best regards,  
[Your Name]

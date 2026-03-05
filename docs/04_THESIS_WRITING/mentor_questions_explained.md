# Mentor Questions — Explained (Technical Terms Guide)

**Date:** February 24, 2026  
**Purpose:** This document explains technical terms and concepts from the mentor questions to help understand what is being asked and why it matters for the thesis.

**Important Context:**
- **Final Deadline:** June 30, 2026 (aiming to finish earlier)
- **Main Thesis Focus:** MLOps + Prognosis model
- **Evaluation Approach:** Window-level K-fold (subject-wise NOT required per mentor)
- **Drift Detection:** Using Z-score for unlabeled data
- **Test-Time Adaptation:** AdaBN and TENT as separate alternatives (NOT combined due to bug)

---

## Table of Contents

1. [Monitoring Thresholds](#1-monitoring-thresholds)
2. [Scalability and Performance](#2-scalability-and-performance)
3. [Prognosis Model](#3-prognosis-model)
4. [Online Monitoring & Automation](#4-online-monitoring--automation)
5. [Evaluation Integrity](#5-evaluation-integrity)
6. [Data Quality & Drift Detection](#6-data-quality--drift-detection)
7. [Adaptation Methods](#7-adaptation-methods)
8. [CI/CD & Code Completeness](#8-cicd--code-completeness)
9. [Thesis Timeline & Exams](#9-thesis-timeline--exams)
10. [Code Freeze & Future Work](#10-code-freeze--future-work)

---

## 1. Monitoring Thresholds

### What is "Synthetic Drift Injection"?

**Definition:** Artificially creating data drift by modifying the test data to simulate real-world changes.

**Example:**
- Take normal accelerometer data
- Add noise: `data_modified = data_original + noise`
- Shift mean: `data_modified = data_original * 1.2`
- Change distribution to see if your monitoring system detects it

**Why do this?**
- To test if your drift detection thresholds actually work
- To verify PSI, Wasserstein distance, etc. catch problems before they hurt performance

---

### What is "Full Sensitivity Analysis"?

**Definition:** A comprehensive study of how different threshold values affect system behavior.

**What it involves:**
- Test many threshold combinations: PSI = [0.1, 0.2, 0.3, ..., 1.0]
- Measure: false positives (too sensitive), false negatives (too relaxed)
- Create plots showing trade-offs
- Formally justify the final choice

**Difference from our simple approach:**
- **Simple:** Pick thresholds, run 26 sessions, check if they seem reasonable
- **Full sensitivity:** Systematic grid search, statistical validation, publication-quality analysis

**For thesis:** Simple approach is usually enough for master's thesis; full analysis is more for research papers.

---

### What is a "Canonical Config"?

**Definition:** One single, official configuration file that is the "source of truth" for all threshold values.

**Current problem:**
- `config/monitoring_thresholds.yaml` has one set of values
- `src/api/monitoring.py` might have different hardcoded values
- Pipeline scripts might use yet another set

**Solution (canonical config):**
```python
# One file: config/canonical_thresholds.yaml
confidence_threshold: 0.75
psi_threshold: 0.75  # Multi-channel calibrated
transition_rate_threshold: 0.40
cooldown_periods:
  confidence: 3
  drift: 2
```

All code imports from this one file. No duplicates, no confusion.

**Why it matters for thesis:**
- Reproducibility: anyone can see exactly what thresholds you used
- Consistency: same values everywhere
- Documentation: one place to explain choices

---

### Why are thresholds different between API and pipeline?

**What happened:**
- Pipeline (batch processing): uses values from YAML config files
- API (online inference): might have different hardcoded values in Python

**Why this is bad:**
- If pipeline says "drift detected" but API says "no drift", which is correct?
- Inconsistent results
- Hard to debug
- Looks unprofessional in thesis

**Effect on thesis:**
- Examiners will ask: "Why are there different thresholds?"
- You need one answer: "All use the same canonical config"

---

## 2. Scalability and Performance

### What is "Hyperparameter Optimization (HPO)"?

**Definition:** Automatically searching for the best model hyperparameters (learning rate, batch size, dropout, etc.).

**Tools:**
- **Optuna:** Bayesian optimization
- **Grid Search:** Try all combinations
- **Random Search:** Try random combinations

**Example:**
```python
# Without HPO (fixed config)
learning_rate = 0.001
batch_size = 32
dropout = 0.3

# With HPO (automatic search)
best_params = optuna_study.optimize()
# Optuna tries: lr=0.01, bs=64, dropout=0.2
# Then: lr=0.001, bs=32, dropout=0.5
# Finds best combination
```

**What the thesis plan said:**
- Month 2: "integrate automated hyperparameter optimization"

**What you actually have:**
- Fixed configuration justified by literature and initial experiments
- Tracked in MLflow

**For thesis:**
- Ask mentor: is fixed config OK, or must you implement HPO?
- HPO is time-consuming and may not improve results much

---

## 3. Prognosis Model

### What is the "Prognosis Model"?

**Definition:** A second model that predicts future mental health outcomes based on the activity patterns recognized by the HAR model.

**Flow:**
```
Sensor Data (accel/gyro) 
  → HAR Model (1D-CNN-BiLSTM) 
    → Activities (walking, sitting, hand-tapping, etc.)
      → Prognosis Model 
        → Prediction (anxiety level, stress probability, etc.)
```

**The Challenge:**
- **I don't have a clear understanding of what the prognosis model should do**
- Need mentor to explain:
  - What should it predict? (anxiety levels, mental health risk scores, clinical outcomes?)
  - What inputs from HAR? (activity percentages, sequences, transitions?)
  - What output format?

**Why this is the main focus:**
- Thesis is **MLOps + Prognosis** (NOT just monitoring/adaptation)
- MLOps infrastructure (14-stage pipeline, 3-layer monitoring, CI/CD) is the technical foundation
- Prognosis model is co-equal focus alongside MLOps

---

### What needs to be done?

**Minimum deliverable (need mentor confirmation):**
1. **Design and document** the data flow from HAR → prognosis
2. **Architectural diagrams** showing integration points
3. **Interface specification** (clear input/output)
4. **Placeholder implementation** or mark as future work

**Three options:**
1. **Design only:** Draw diagrams, explain how it would work, no code
2. **Placeholder:** Create empty class with clear input/output interface
   ```python
   class PrognosisModel:
       def predict(self, activity_sequence: List[str]) -> float:
           """Predicts anxiety level from activity sequence.
           
           Args:
               activity_sequence: ["walking", "sitting", "hand-tapping", ...]
           
           Returns:
               anxiety_score: 0.0 to 1.0
           """
           raise NotImplementedError("Future work")
   ```
3. **Full implementation:** Train model, evaluate it, report results

**For thesis:**
- Need mentor guidance urgently on which option is expected
- Given June 30 deadline, options 1 or 2 are more realistic

---

### What is "CORAL"?

**Possible meanings:**

1. **CORAL (Domain Adaptation Method):**
   - CORrelation ALignment
   - A technique to align feature distributions between source and target domains
   - Paper: "Return of Frustratingly Easy Domain Adaptation" (Sun & Saenko, 2016)
   - Used to make models generalize better to new environments

2. **Clinical Assessment of Repetitive and Stereotyped Movement (CORAL):**
   - A clinical scale for measuring behaviors in mental health
   - Maybe mentor means using this as target for prognosis?

3. **Something else specific to your project**

**What you need:**
- Ask mentor: "When you mentioned 'coral', did you mean the CORAL domain adaptation method, a clinical assessment tool, or something else?"
- Then document it correctly in pipeline design

**Where it might go:**
- If CORAL adaptation: between preprocessing and model inference
- If CORAL clinical score: as output target for prognosis model

---

## 4. Online Monitoring & Automation

### What is "Online" vs "Batch" Processing?

**Batch Processing (what you have now):**
```
Upload file 1 → Process entire file → Results
Upload file 2 → Process entire file → Results
```

**True Online Processing:**
```
Sensor → BLE stream → Process each window immediately → Results in real-time
```

**Your simulation approach:**
```
Replay file 1 through API → Monitor
Replay file 2 through API → Monitor
...
(Simulate "online" by processing sequentially)
```

**For thesis:**
- True online requires live sensor connection (you don't have this)
- Simulated online is acceptable for master's thesis

---

### WebUI for Data Upload and Inference

**What you could add:**

A simple web interface where users can:
1. **Upload sensor files** (CSV with accelerometer + gyroscope)
2. **See inference results** (predicted activities)
3. **View monitoring dashboard** (drift warnings, confidence scores)
4. **Submit labels** (if ground truth becomes available)
5. **Trigger retraining** (manual button to start retrain)

**Example Flask/FastAPI UI:**
```python
@app.route("/upload", methods=["POST"])
def upload_and_infer():
    file = request.files['sensor_data']
    results = inference_pipeline.predict(file)
    drift_status = monitor.check_drift(results)
    return render_template("results.html", 
                          predictions=results,
                          drift_warning=drift_status)
```

**Benefits:**
- Makes "online" processing more realistic
- Easy for mentor to test the system
- Good demo for thesis defense

**Effort:**
- ~1-2 days to build basic version
- Ask mentor if this is expected or nice-to-have

---

### New Device / New Dataset Processing

**Scenario:**
You get a new wearable device or new data sessions with:
- Different sensor placement (wrist vs chest)
- Different sampling rate
- Different patient population

**What needs to happen:**
1. **Data preprocessing:** Resample, normalize, align channels
2. **Unit detection:** Check if milliG or m/s²
3. **Inference:** Run through model
4. **Monitoring:** Check for drift (new device = new distribution)
5. **Adaptation:** Apply AdaBN/TENT to adapt to new device
6. **Optional retraining:** If drift is severe, retrain with new data

**For thesis:**
- Document this pipeline clearly
- Show that your monitoring system catches new-device drift
- Demonstrate adaptation methods help with new devices

---

### Manual vs Automatic Retraining

**Manual (what you have now):**
```bash
# 1. System detects drift, saves warning
$ python detect_drift.py  # Output: "Drift detected: PSI = 0.89"

# 2. You manually decide to retrain
$ python retrain.py --trigger-reason drift

# 3. You manually promote model
$ python promote_model.py --model-id mlflow-run-123
```

**Automatic (what might be expected):**
```python
# CI/CD system runs continuously
while True:
    drift_status = monitor.check()
    if drift_status.should_retrain():
        new_model = retrain_pipeline.run()
        if new_model.performance > current_model.performance:
            deploy_automatically(new_model)
    sleep(3600)  # Check every hour
```

**For thesis:**
- Manual is OK for proof-of-concept
- Document clearly how automatic would work
- Show the decision logic even if not implemented

---

### Prometheus/Grafana vs Just MLflow

**What each does:**

**MLflow:**
- Tracks experiments during training
- Stores models and parameters
- Shows historical results
- Good for: "Which model version performed best?"

**Prometheus + Grafana:**
- Monitor deployed model in production
- Real-time metrics (latency, throughput, drift scores)
- Alerts when thresholds exceeded
- Good for: "Is the deployed model healthy right now?"

**What you have:**
- MLflow: ✅ Fully implemented
- Prometheus + Grafana: ⚠️ Configured but not running live

**Key Question:**
- **Can everything be done with MLflow alone?**
- If yes: Remove Prometheus/Grafana complexity, focus on MLflow
- If no: Need to justify why both are needed

**For thesis:**
- Proposal: Use MLflow for all tracking needs
- Treat Prometheus/Grafana as optional for production deployments
- Simple is better unless mentor requires both

---

## 5. Evaluation Integrity

### What is "Subject-wise Evaluation"?

**Important Update from Mentor:**
- **Subject-wise evaluation is NOT required for this thesis**
- We will use window-level K-fold cross-validation
- This should be clearly documented as a limitation

**Problem with window-level split:**
```
Person A: [window1, window2, window3, window4, window5]
           ↓         ↓         ↓         ↓         ↓
Train:    [window1, window3]
Val:                [window2, window4]
Test:                                   [window5]
```

Windows from **same person** appear in train, validation, AND test sets.

**Why this is a limitation:**
- Model learns person-specific patterns ("Person A walks like X")
- Test set is not truly independent
- **Inflated performance:** Model seems better than it really is
- Doesn't test generalization to new people

**For thesis:**
- Use window-level StratifiedKFold for experiments
- **Clearly state this limitation** in Methods section
- Explain that subject-wise generalization is important for real deployment
- Mark subject-wise evaluation as **Future Work**

---

### When to use K-fold Cross-Validation?

**K-fold cross-validation:**
- Used when we have **labeled data**
- Split data into K parts
- Train on K-1 parts, validate on 1 part
- Repeat K times, average results

**In our thesis:**
- Use K-fold when doing **retraining experiments** with labeled data
- Use **StratifiedKFold** to keep class distributions balanced
- **Window-level split** (not subject-wise, per mentor guidance)

**Code:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and evaluate
```

**Note:** We removed LOSO (Leave-One-Subject-Out) from consideration since subject-wise evaluation is not required.

---

### Current Performance and Expectations

**Current performance (window-level):**
- Accuracy: ~93.8%
- Macro F1: ~0.939

**For thesis:**
- Continue using window-level evaluation
- Document limitations clearly
- Report only MLflow-traceable results
- Include per-class recall for all 11 classes

---

### What is "Minimum Acceptable Performance"?

**The question:**
- If subject-wise F1 drops to 0.70, is that acceptable?
- Or must it be > 0.75? > 0.80?

**Why this matters:**
- Sets expectations for passing thesis
- Helps decide if more work is needed
- Some methods might not generalize well (need to know if that's OK)

**For thesis:**
- Ask mentor explicitly: "What macro F1 on subject-wise evaluation would you consider adequate?"
- This avoids surprises during defense

**Typical expectations:**
- Excellent: > 0.85
- Good: 0.80-0.85
- Acceptable: 0.75-0.80
- Borderline: 0.70-0.75
- Problematic: < 0.70

(But depends on difficulty of task and literature baselines)

---

## 6. Data Quality & Drift Detection

### Z-score for Drift Detection (Our Approach)

**What is Z-score?**

- Compares **individual values** or **batch statistics** to reference distribution
- Formula: `z = (x - mean) / std_dev`
- Good for: detecting shifts in streaming unlabeled data

**Our use case:**
```python
# Reference statistics from training data
reference_mean = 9.8
reference_std = 0.5

# Current batch statistics
current_batch_mean = 12.3
z = (12.3 - 9.8) / 0.5 = 5.0  # Significant drift!
```

**Why Z-score for unlabeled data:**
- Works on summary statistics (don't need individual labels)
- Can detect distribution shifts batch-by-batch
- Simple to implement and interpret

**For multi-channel sensor data:**
```python
# Calculate Z-score for each channel
z_accel_x = (current_mean_x - ref_mean_x) / ref_std_x
z_accel_y = (current_mean_y - ref_mean_y) / ref_std_y
z_accel_z = (current_mean_z - ref_mean_z) / ref_std_z
z_gyro_x = ...

# Aggregate (e.g., max absolute Z-score)
z_max = max(abs(z_accel_x), abs(z_accel_y), ...) 

if z_max > 3.0:  # Threshold (3 sigma rule, or empirically calibrated)
    raise DriftWarning("Significant distribution shift detected")
```

**For thesis:**
- Document Z-score threshold calibration for multi-channel data
- Show experiments on 26 sessions to choose appropriate threshold
- This is a **methodological contribution** for multi-channel time series

---

### PSI vs Z-score (Comparison)

**PSI (Population Stability Index):**
- Compares **distributions** (entire shape)
- Bins data into histogram, compares bin counts
- Good for: detecting drift when you have enough data to build histograms
- Typically used in batch settings

**Z-score:**
- Compares **summary statistics** (mean, std)
- Can work on smaller batches
- Good for: streaming data, unlabeled data

**Why we chose Z-score:**
- Works better for **unlabeled streaming data**
- Simpler to implement and tune
- Appropriate for our use case

---

## 7. Adaptation Methods

### What is AdaBN (Adaptive Batch Normalization)?

**Problem:**
- Model trained on Persons A, B, C (source domain)
- At test time, see Person D (target domain)
- Person D has different movement patterns → features have different statistics

**Batch Normalization (BN) layers store:**
```python
# Learned during training
mean_train = 2.5
std_train = 1.2

# At test time
output = (input - mean_train) / std_train  # Uses training statistics
```

**AdaBN:** Replace training statistics with **test batch statistics**

```python
# At test time
mean_test = 3.1  # Computed from Person D's data
std_test = 1.5
output = (input - mean_test) / std_test  # Uses test statistics
```

**Why this helps:**
- Adapts feature distribution to match new person/environment
- No gradient updates needed (just replace statistics)
- Very fast and simple

**Paper:** "Revisiting Batch Normalization For Practical Domain Adaptation" (Li et al., 2016)

---

### What is TENT (Test Entropy Minimization)?

**Problem:**
- Model uncertain on new test data
- Predictions have high entropy (e.g., [0.3, 0.4, 0.3] instead of [0.9, 0.05, 0.05])

**TENT:** Update model at test time to make predictions more confident

**How it works:**
1. Get test sample
2. Forward pass → prediction (e.g., [0.3, 0.4, 0.3])
3. Calculate entropy: H = -Σ p_i log(p_i)  # High entropy = uncertain
4. Backpropagate to minimize entropy
5. Update model weights (only BN layers, not full model)
6. Now prediction is sharper: [0.7, 0.2, 0.1]

**Key idea:** Force model to be confident, assuming confident predictions are correct.

**Paper:** "Tent: Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2021)

**Why only update BN layers:**
- Safe: BN layers are less likely to cause catastrophic forgetting
- Fast: Fewer parameters to update
- Effective: BN captures domain-specific statistics

---

### Critical Bug: AdaBN + TENT Should NOT Be Combined

**What happened:**

**Step 1 (AdaBN):**
```python
# Replace BN statistics with test batch statistics
bn_layer.running_mean = test_batch_mean  # [2.5, 3.1, 1.8, ...]
bn_layer.running_std = test_batch_std    # [1.2, 1.5, 0.9, ...]
```

**Step 2 (TENT):**
```python
# Minimize entropy, update BN parameters
loss = entropy(predictions)
loss.backward()
optimizer.step()  # Updates BN.weight and BN.bias

# BUT ALSO accidentally resets:
bn_layer.running_mean = train_mean  # ❌ Overwrites AdaBN statistics!
bn_layer.running_std = train_std    # ❌
```

**Problem:** TENT's backward pass resets BN statistics back to training values, **completely undoing AdaBN's work**!

**Symptom:** When using AdaBN + TENT together, performance is **WORSE** than using AdaBN alone.

---

**The Decision:**

Instead of trying to fix this bug (snapshot/restore approach), we decided:

**Present AdaBN and TENT as ALTERNATIVE approaches:**
- Use AdaBN **OR** TENT, **NOT both**
- Evaluate them separately
- Document the interaction issue as a cautionary finding

**Why this approach:**
- Attempting to combine them makes performance worse
- Each method works well independently
- Cleaner thesis narrative: "Two alternative test-time adaptation methods"- Warns others about this interaction problem

**Ablation experiments:**
1. No adaptation (baseline)
2. AdaBN only
3. TENT only
4. Pseudo-labeling with self-consistency filter
5. Pseudo-labeling without filter

**For thesis:**
- Present as **separate alternative methods**
- Document interaction bug as cautionary finding
- Show ablation results for each method independently

---

### What is the Self-Consistency Filter?

**Context:** Pseudo-labeling for test-time adaptation

**Pseudo-labeling:**
```python
# Model makes prediction on unlabeled test data
pred = model.predict(test_sample)  # [0.8, 0.1, 0.1] → class 0

# Use prediction as "pseudo-label" to update model
pseudo_label = argmax(pred)  # 0
loss = cross_entropy(pred, pseudo_label)
loss.backward()
```

**Problem:** If prediction is wrong, you train on wrong label → model gets worse!

---

**Self-Consistency Filter:**

Only use pseudo-labels when multiple forward passes agree:

```python
# Forward pass 1 (with dropout/noise)
pred1 = model.predict(test_sample)  # class 2

# Forward pass 2
pred2 = model.predict(test_sample)  # class 2

# Forward pass 3
pred3 = model.predict(test_sample)  # class 0 (different!)

# Only use if all agree
if pred1 == pred2 == pred3:
    pseudo_label = pred1  # Use it
    update_model(pseudo_label)
else:
    skip  # Uncertain, don't use
```

**Result in your thesis:**
- Filter keeps only **14.5% of samples** (strict!)
- Why so low: 11 activity classes, many borderline cases
- Trade-off: High precision (confident labels) vs low recall (few labels)

---

### Self-Consistency and Data Characteristics

**Problem 1: Rare Classes**
Some activity classes are rare:
- Common: sitting, standing, walking (thousands of samples)
- Rare: hand-tapping (hundreds of samples)

**Effect of strict filter:**
```
Before filter:
  - Sitting: 5000 samples
  - Hand-tapping: 200 samples

After filter (keep only 14.5%):
  - Sitting: 725 samples (still many)
  - Hand-tapping: 29 samples (very few!)
```

**Consequence:** Model may "forget" rare classes during adaptation.

---

**Problem 2: Not All 11 Classes in Every Dataset**

**Critical data characteristic:**
- **We don't have all 11 activity classes in every single dataset**
- Getting all 11 classes in one recording session is difficult
- Some sessions may only have 5-7 different activities

**Why this matters:**
- Makes class imbalance worse
- Some adaptation methods may never see certain classes
- Need to document this limitation clearly

---

**Problem 3: High Hand-Tapping Prevalence**

**Observation:** Inference results show **very high frequency of hand-tapping activity** across most of the 26 datasets.

**Possible explanation:**
- When someone is working at a desk, **hands are constantly on the table**
- This position/motion gets detected as "hand-tapping" by the model
- May be a **labeling artifact** rather than actual hand-tapping behavior

**What to check:**
1. Analyze class distribution across all 26 sessions
2. Count how often hand-tapping appears
3. Check if it correlates with "working at desk" scenarios

**For thesis:**
- Run ablation: pseudo-labeling with vs without self-consistency filter
- Analyze and document class distribution across sessions
- Note hand-tapping prevalence and discuss possible causes
- Document that not all 11 classes appear in every session

---

## 8. CI/CD & Code Completeness

### What are DANN and MMD?

**DANN (Domain-Adversarial Neural Network):**

**Goal:** Learn features that work on both source and target domains.

**How it works:**
```
Input → Feature Extractor → Task Classifier (predict activity)
           ↓
     Domain Classifier (predict: training data or test data?)
```

**Training:**
1. Task Classifier: Maximize activity prediction accuracy
2. Domain Classifier: Predict which domain sample came from
3. Feature Extractor: **Fool domain classifier** (make features domain-invariant)

**Result:** Features that work well on both training people and new people.

**Paper:** "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)

---

**MMD (Maximum Mean Discrepancy):**

**Goal:** Minimize difference between source and target feature distributions.

**How it works:**

Measure distance between distributions using kernel methods:
```
MMD = ||mean(features_train) - mean(features_test)||²
```

More formally, uses kernel trick:
```
MMD² = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2E[k(x_s, x_t)]
```

**Training:** Add MMD loss to regular classification loss:
```python
loss_classification = cross_entropy(pred, label)
loss_mmd = calculate_mmd(features_train, features_test)
total_loss = loss_classification + lambda * loss_mmd
```

**Result:** Model learns features that are similar between training and test data.

**Paper:** "Learning Transferable Features with Deep Adaptation Networks" (Long et al., 2015)

---

**Why they're in the code (and should be removed):**

**Current status:**
```python
class DANNAdapter:
    def adapt(self, model, test_data):
        raise NotImplementedError("DANN not implemented - future work")

class MMDAdapter:
    def adapt(self, model, test_data):
        raise NotImplementedError("MMD not implemented - future work")
```

**Decision: REMOVE from codebase**

**Reasons:**
1. They're not implemented and won't be implemented for this thesis
2. Having `NotImplementedError` stubs makes code look incomplete
3. Cleaner to remove them entirely
4. Can still mention in thesis:
   - **Related Work section:** "DANN and MMD are alternative domain adaptation approaches"
   - **Future Work section:** "DANN and MMD could be explored for more complex domain shifts"

**For thesis:**
- **Remove DANN.py and MMD.py** from src/adaptation/
- Keep only: AdaBN.py, TENT.py, pseudo_labeling.py
- Mention DANN/MMD only in Related Work and Future Work chapters
- Cleaner, more focused codebase

---

### About Question 31

**Question 31 asks:** "Which should be the main scientific contribution?"
- 14-stage pipeline?
- 3-layer monitoring?
- Test-time adaptation findings?

**Updated Understanding:**

**Main thesis focus is MLOps + Prognosis**, so contributions should reflect this:

1. **Complete 14-stage MLOps pipeline** for HAR
   - Model registry
   - CI/CD design
   - Deployment architecture
   - Docker containerization

2. **3-layer monitoring framework** (needs citations)
   - Confidence monitoring
   - Drift detection (Z-score based)
   - Transition rate monitoring
   - Multi-channel threshold calibration

3. **Prognosis model architectural design**
   - Integration approach
   - Data flow design
   - Interface specification

4. **Test-time adaptation** (technical detail, not main focus)
   - AdaBN and TENT as alternatives
   - Interaction bug finding

**Should you keep Q31?**

**Yes, keep it** — it helps mentor guide you on thesis structure and defense strategy.

Rephrase slightly: "Which contribution should receive most emphasis in thesis and defense?"

---

## 9. Thesis Timeline & Exams

### Important Timeline Information

**Final Submission Deadline:** June 30, 2026

**Note:** While aiming to finish earlier, the hard deadline is end of June, giving more flexibility than originally thought (was thinking April).

**Work Planning:**

**Priority 1 (Must complete before June):**
- ✅ Prognosis model scope clarification (from mentor)
- ✅ Complete evaluation experiments (window-level K-fold)
- ✅ Ablation studies:
  - No adaptation baseline
  - AdaBN only
  - TENT only
  - Pseudo-labeling with filter
  - Pseudo-labeling without filter
- ✅ Class distribution analysis (hand-tapping prevalence, class availability)
- ✅ Remove DANN/MMD from codebase
- ✅ Timing benchmarks and scalability analysis
- ✅ Code cleanup and documentation

**Priority 2 (After core experiments):**
- Thesis writing:
  - Methods chapter (detailed)
  - Results chapter (all experiments)
  - Discussion and Future Work
  - Introduction and Related Work (keep concise)
- Code freeze and commit-pinned links
- Plots and figures
- Final proofreading

**Note:** Having until end of June provides good buffer time for thesis writing after experiments are complete.

---

## 10. Code Freeze & Future Work

### What is "Code Freeze"?

**Definition:** Stop making changes to the codebase and lock it at a specific version.

**Why needed:**
```
# Problem without freeze
Your thesis says: "See line 245 in train.py"
But you keep editing code after writing
Now line 245 has different code!
Examiner looks at GitHub → confused
```

**Solution: Code freeze**
```
# 1. Stop editing code on specific date
git commit -m "FREEZE: Code for thesis submission"
git tag thesis-final-v1.0

# 2. Use commit-pinned links in thesis
"See training loop: https://github.com/You/Repo/blob/abc123def/src/train.py#L245"
                                                              ↑ specific commit hash

# 3. Never change that tagged commit
```

**When to freeze:**
- **2-3 weeks before thesis submission**
- After all experiments are done
- Before you start writing detailed code discussion in thesis

**Benefits:**
- Thesis and code always match
- Reproducible: anyone can checkout exact version you used
- Professional: shows good software engineering practice

---

### What Can Still Change After Freeze?

**Frozen (don't touch):**
- ❌ Core pipeline code
- ❌ Model training scripts
- ❌ Data processing logic
- ❌ Anything you reference in thesis

**OK to change:**
- ✅ README documentation
- ✅ Separate analysis scripts
- ✅ Future work branch (clearly labeled)
- ✅ Docker configs for deployment (if not discussed in thesis)

**Best practice:**
```bash
# Frozen branch
git checkout main
git tag thesis-v1.0

# New work goes in separate branch
git checkout -b post-thesis-improvements
# Make changes here
```

---

### What Must Be Finished vs Future Work (Q34)

**This question asks:** For advanced features you've partially implemented, which must have results in thesis vs which can be "Future Work"?

**Features in question:**
1. **Wasserstein distance** (drift detection metric)
2. **Curriculum pseudo-labeling** (adaptive learning)
3. **OOD detection** (out-of-distribution samples)
4. **Sensor placement analysis** (robustness to sensor position)

**Three categories:**

**Category A: Must be evaluated (core contributions)**
- Subject-wise evaluation
- AdaBN + TENT adaptation
- PSI threshold calibration
- Basic monitoring (confidence, drift, transitions)

**Category B: Implemented but not evaluated (document as available)**
- Wasserstein drift detection (implemented, results optional)
- Blue-green deployment (designed, not tested)
- DANN/MMD stubs (future work)

**Category C: Clear future work (mention only)**
- Real-time BLE streaming
- Full prognosis model
- Sensor placement robustness study
- Multi-device heterogeneous adaptation

**For thesis:**
- Ask mentor to categorize each feature
- Category A: Must complete before submission
- Category B: OK to document as "available but not evaluated in this work"
- Category C: 1-2 paragraphs in Future Work section

**Example discussion:**
> "While OOD detection capability is implemented (see src/monitoring/ood_detector.py), 
> a comprehensive evaluation on synthetic OOD samples is beyond the scope of this thesis 
> and left for future work. The implementation uses Mahalanobis distance on feature 
> representations and has been unit-tested but not validated on real anomalous data."

---

## Summary: What to Ask Your Mentor

**Critical decisions (must answer):**
1. **What should the prognosis model do?** (Q5 - need detailed guidance)
2. **Is prognosis implementation required, or design+interface sufficient?** (Q7 - CRITICAL)
3. **Can MLflow handle all monitoring, or is Prometheus/Grafana needed?** (Q11)
4. **Confirm: window-level evaluation is OK?** (Q12)
5. **What is minimum acceptable performance?** (Q13)
6. **Should I remove DANN/MMD from code?** (Q25)
7. **Which contributions need literature citations?** (3-layer monitoring needs support)

**Important clarifications:**
- Is Z-score approach for drift detection sufficient?
- Should AdaBN and TENT be presented as alternatives (NOT combined)?
- Is fixed config OK or must implement HPO?
- When should I freeze the code?

**Timeline:**
- Final deadline: June 30, 2026
- What should be prioritized in Methods/Results chapters?

---

**Next steps:**
1. **Send email to mentor** with questions from mentor_questions.md Section 11
2. **Wait for answers** on prognosis model scope (most critical)
3. **Remove DANN/MMD** from codebase if mentor agrees
4. **Run ablation studies**: AdaBN, TENT, pseudo-labeling (with/without filter)
5. **Analyze class distribution** in artifacts/ folder (hand-tapping prevalence)
6. **Find citations** for 3-layer monitoring framework
7. **Implement prognosis design** based on mentor guidance

**TODO immediately:**
- Script to analyze class distribution in artifacts/ folders
- Document Z-score threshold calibration approach
- Clean up code (remove DANN/MMD after mentor confirmation)
- Start Methods chapter draft while waiting for mentor feedback

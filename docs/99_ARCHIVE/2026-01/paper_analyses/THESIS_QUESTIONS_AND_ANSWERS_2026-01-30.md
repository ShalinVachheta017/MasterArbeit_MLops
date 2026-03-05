# 📋 Thesis Critical Questions & Answers
## Date: January 30, 2026

---

## 📊 Current Status Summary

### Where Are We Stuck?

| Blocker | Priority | Status | Effort Needed |
|---------|----------|--------|---------------|
| **No Retraining Pipeline** | 🔴 CRITICAL | Missing | 1-2 weeks |
| **No Unit Tests** | 🔴 CRITICAL | `tests/` folder is EMPTY | 3-4 days |
| **No CI/CD Pipeline** | 🔴 CRITICAL | No `.github/workflows/` | 2-3 days |
| **Thesis Writing** | 🔴 CRITICAL | 0% complete | 4-6 weeks |
| **No Temperature Scaling** | 🟠 HIGH | Calibration missing | 1-2 days |

**Current Pipeline:**
```
Raw Garmin Data → Preprocess → Pretrained Model → Predictions (DONE)
```

**Missing (Required for thesis):**
```
New Data → Drift Detection → Retrain → Validate → Deploy → Monitor (NOT DONE)
```

---

# Question 1: How to Show Model Degradation?

## 1.1 What is Model Degradation?

Model degradation is when a trained model's performance drops over time due to:
- **Data drift**: Input data distribution changes
- **Concept drift**: Relationship between features and labels changes
- **Domain shift**: Lab-trained model deployed on real-world data

### Evidence from Your Papers

From **ICTH_16.pdf** (your foundation paper):
> Lab-trained model achieved **89% accuracy** on lab data but only **49% accuracy** on production (Garmin) data — **40% degradation!**

After domain adaptation (fine-tuning): **49% → 87%** recovery

## 1.2 Metrics to Detect Degradation

### A. Distribution Drift Metrics (No Labels Needed)

| Metric | Threshold | Python Implementation | Interpretation |
|--------|-----------|----------------------|----------------|
| **KS-Test** | p < 0.05 | `scipy.stats.ks_2samp(train, prod)` | Distribution shift detected |
| **PSI** | > 0.25 | Binned histogram comparison | Major population shift |
| **MMD** | Project-specific | Kernel-based distance | Feature distribution change |
| **Wasserstein** | > 0.5 | `scipy.stats.wasserstein_distance` | Earth mover's distance |

**Implementation (from your papers):**
```python
from scipy.stats import ks_2samp
import numpy as np

def detect_distribution_drift(train_data, prod_data, threshold=0.05):
    """
    Per Domain Adaptation papers: KS-test is simple but effective.
    """
    sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    drift_detected = []
    
    for idx, sensor in enumerate(sensor_names):
        train_flat = train_data[:, :, idx].flatten()
        prod_flat = prod_data[:, :, idx].flatten()
        stat, p_value = ks_2samp(train_flat, prod_flat)
        
        if p_value < threshold:
            drift_detected.append({
                'sensor': sensor,
                'ks_statistic': stat,
                'p_value': p_value,
                'drift': True
            })
    
    return drift_detected
```

### B. Prediction Behavior Metrics (No Labels Needed)

| Metric | Normal Range | Degradation Signal | Formula |
|--------|--------------|-------------------|---------|
| **Mean Confidence** | 0.85-0.95 | < 0.70 | `np.mean(max_probs)` |
| **Entropy** | 0.2-0.8 | > 1.5 | `-sum(p * log(p))` |
| **Margin** | > 0.40 | < 0.15 | `top1_prob - top2_prob` |
| **Flip Rate** | < 15% | > 30% | `% adjacent different predictions` |
| **Idle Percentage** | < 20% | > 50% | `% sitting/standing predictions` |

**Implementation:**
```python
def compute_degradation_indicators(probabilities):
    """
    Compute proxy metrics for model degradation without labels.
    """
    max_probs = np.max(probabilities, axis=1)
    
    # Confidence drop indicator
    mean_confidence = np.mean(max_probs)
    low_confidence_rate = np.mean(max_probs < 0.50)
    
    # Entropy (uncertainty)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    
    # Margin (decision boundary ambiguity)
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    mean_margin = np.mean(margins)
    
    # Thresholds from research papers
    degradation_signals = {
        'confidence_drop': mean_confidence < 0.70,
        'high_uncertainty': mean_entropy > 1.5,
        'low_margin': mean_margin < 0.15,
        'degradation_score': (
            (0.85 - mean_confidence) / 0.85 +  # Confidence contribution
            (mean_entropy - 0.5) / 2.0 +        # Entropy contribution
            (0.40 - mean_margin) / 0.40          # Margin contribution
        ) / 3
    }
    
    return degradation_signals
```

### C. Performance Metrics (Requires Labels)

| Metric | Baseline | Degradation Threshold | Action |
|--------|----------|----------------------|--------|
| **Accuracy** | 87% | < 80% | Trigger retraining |
| **F1 (macro)** | 0.85 | < 0.75 | Investigate per-class |
| **ECE** | < 0.05 | > 0.15 | Needs calibration |

## 1.3 How to Demonstrate in Thesis

### Experiment Design

1. **Baseline Measurement**
   - Run inference on test split → Record accuracy, F1, confidence
   - This is your "Day 0" reference

2. **Simulate Degradation** (3 methods)
   
   **Method A: Add Noise**
   ```python
   # Add Gaussian noise to simulate sensor drift
   noisy_data = clean_data + np.random.normal(0, 0.1, clean_data.shape)
   ```
   
   **Method B: Domain Shift**
   ```python
   # Test on different user's data (cross-person)
   # Or different time period (temporal shift)
   ```
   
   **Method C: Sensor Attenuation**
   ```python
   # Simulate non-dominant hand scenario
   attenuated_data = clean_data * 0.3  # Reduce signal strength
   ```

3. **Track Metrics Over Time**
   - Run inference weekly → Log all metrics to MLflow
   - Plot degradation curves (confidence over time)

### Visualization for Thesis

```python
import matplotlib.pyplot as plt

# Figure: Model Degradation Over Time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Accuracy drop
axes[0,0].plot(weeks, accuracy_scores)
axes[0,0].axhline(y=0.80, color='r', linestyle='--', label='Retraining Threshold')
axes[0,0].set_title('Accuracy Over Time')

# Plot 2: Confidence distribution shift
axes[0,1].hist(week1_confidence, alpha=0.5, label='Week 1')
axes[0,1].hist(week4_confidence, alpha=0.5, label='Week 4')
axes[0,1].set_title('Confidence Distribution Shift')

# Plot 3: Drift score (KS-statistic)
axes[1,0].bar(sensors, ks_statistics)
axes[1,0].axhline(y=0.15, color='r', linestyle='--', label='Warning')
axes[1,0].set_title('Per-Sensor Distribution Drift')

# Plot 4: Entropy over time
axes[1,1].plot(weeks, mean_entropy)
axes[1,1].set_title('Mean Entropy Over Time')

plt.savefig('docs/figures/model_degradation.png', dpi=300)
```

---

# Question 2: How to Retrain the Model?

## 2.1 Retraining Techniques from Papers

### A. Supervised Fine-Tuning (With Labels)

From **ICTH_16.pdf**:
> "Weekly retraining with 10-20% new labeled data maintains 85%+ accuracy"

**Implementation:**
```python
# src/retrain_with_cv.py
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

def retrain_with_cv(
    base_model_path: str,
    new_X: np.ndarray,
    new_y: np.ndarray,
    n_folds: int = 5
):
    """
    Retrain model using 5-fold cross-validation.
    Per ICTH_16: Weekly retraining with CV maintains accuracy.
    """
    base_model = tf.keras.models.load_model(base_model_path)
    
    # Freeze early layers (feature extraction)
    for layer in base_model.layers[:5]:
        layer.trainable = False
    
    # Unfreeze later layers (task-specific)
    for layer in base_model.layers[5:]:
        layer.trainable = True
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(new_X, new_y)):
        X_train, X_val = new_X[train_idx], new_X[val_idx]
        y_train, y_val = new_y[train_idx], new_y[val_idx]
        
        # Clone and train
        model = tf.keras.models.clone_model(base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
        )
        
        _, accuracy = model.evaluate(X_val, y_val)
        cv_scores.append(accuracy)
    
    return np.mean(cv_scores), np.std(cv_scores)
```

### B. Self-Training / Pseudo-Labeling (No Labels)

From **Transfer_Learning_of_Human_Activities_Based_on_IMU_Sensors_A_Review.pdf**:
> "Self-training with confidence threshold >0.90 prevents error propagation"

**Implementation:**
```python
def self_training_retrain(model, unlabeled_X, confidence_threshold=0.90):
    """
    Self-training: Use high-confidence predictions as pseudo-labels.
    Per Transfer Learning Survey: threshold >0.90 is recommended.
    """
    # Get predictions on unlabeled data
    probabilities = model.predict(unlabeled_X)
    max_probs = np.max(probabilities, axis=1)
    pseudo_labels = np.argmax(probabilities, axis=1)
    
    # Select high-confidence samples
    confident_mask = max_probs >= confidence_threshold
    confident_X = unlabeled_X[confident_mask]
    confident_y = pseudo_labels[confident_mask]
    
    print(f"Selected {np.sum(confident_mask)}/{len(unlabeled_X)} samples "
          f"({100*np.mean(confident_mask):.1f}%) above threshold {confidence_threshold}")
    
    # Retrain with pseudo-labeled data
    model.fit(confident_X, confident_y, epochs=10, batch_size=32)
    
    return model, confident_mask
```

### C. Elastic Weight Consolidation (Prevent Forgetting)

From **Lifelong_Learning_in_Sensor-Based_Human_Activity_Recognition.pdf**:
> "EWC penalty prevents catastrophic forgetting when learning new tasks"

**Implementation:**
```python
def compute_fisher_information(model, X, y, num_samples=200):
    """
    Compute Fisher Information Matrix for EWC.
    Per Lifelong Learning paper: Protects important weights.
    """
    fisher = {n: np.zeros(w.shape) for n, w in enumerate(model.trainable_weights)}
    
    for i in range(min(num_samples, len(X))):
        with tf.GradientTape() as tape:
            output = model(X[i:i+1], training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y[i:i+1], output)
        
        gradients = tape.gradient(loss, model.trainable_weights)
        
        for n, grad in enumerate(gradients):
            if grad is not None:
                fisher[n] += grad.numpy() ** 2
    
    for n in fisher:
        fisher[n] /= num_samples
    
    return fisher

def ewc_loss(model, fisher, old_weights, lambda_ewc=1000):
    """
    EWC penalty term to add to training loss.
    """
    ewc_penalty = 0
    for n, (new_w, old_w) in enumerate(zip(model.trainable_weights, old_weights)):
        ewc_penalty += tf.reduce_sum(fisher[n] * (new_w - old_w) ** 2)
    return lambda_ewc * ewc_penalty
```

### D. Adaptive Batch Normalization (AdaBN)

From **XHAR_Deep_Domain_Adaptation_for_Human_Activity_Recognition_with_Smart_Devices.pdf**:
> "AdaBN: Replace source BN statistics with target statistics at inference"

**Implementation (No Retraining Needed!):**
```python
def adapt_batch_norm(model, target_X, num_batches=50):
    """
    Adaptive Batch Normalization - update BN statistics using target data.
    Per XHAR: Simple but effective, no gradient updates needed.
    """
    # Set model to training mode for BN statistic updates
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    # Run forward passes to update BN running statistics
    batch_size = 32
    for i in range(num_batches):
        batch_idx = np.random.choice(len(target_X), batch_size)
        batch = target_X[batch_idx]
        _ = model(batch, training=True)  # Update BN statistics
    
    # Freeze BN layers again
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    return model
```

## 2.2 Retraining Triggers

| Trigger Type | Condition | Paper Source |
|--------------|-----------|--------------|
| **Scheduled** | Every week (cron job) | ICTH_16 |
| **Drift-Based** | KS-test > 0.25 or PSI > 0.25 | Domain Adaptation Survey |
| **Performance-Based** | Accuracy drops > 5% | MLOps Best Practices |
| **Data Volume** | > 100 new labeled samples | Transfer Learning Survey |

**Recommended Trigger Logic:**
```python
def should_retrain(metrics):
    """
    Combined trigger: Scheduled + Drift + Performance
    """
    return (
        metrics['days_since_last_retrain'] >= 7 or  # Weekly
        metrics['drift_score'] > 0.25 or            # Distribution shift
        metrics['accuracy_drop'] > 0.05 or          # Performance degradation
        metrics['new_labeled_samples'] >= 100       # Data volume
    )
```

---

# Question 3: Drift Metrics and Unsupervised Domain Adaptation

## 3.1 Drift Detection Metrics

| Metric | Type | Use Case | Threshold |
|--------|------|----------|-----------|
| **KS-Test** | Statistical | Per-feature distribution | p < 0.05 |
| **PSI** | Binned | Overall population | > 0.25 |
| **MMD** | Kernel | High-dimensional features | Project-specific |
| **KL Divergence** | Information | Probability distributions | > 0.1 |
| **Wasserstein** | Optimal Transport | Distribution distance | > 0.5 |

## 3.2 Use Cases for Unsupervised Domain Adaptation

From **Unsupervised_Domain_Adaptation_in_Activity_Recognition_A_GAN-Based_Approach.pdf**:

### When to Use UDA

| Scenario | UDA Technique | Rationale |
|----------|---------------|-----------|
| **Lab → Real World** | AdaBN, MMD | Different environments, sensors |
| **User A → User B** | Fine-tuning, DANN | Different body types, habits |
| **Dominant → Non-dominant** | Axis mirroring + AdaBN | Sensor placement shift |
| **Old sensor → New sensor** | Feature alignment | Hardware differences |

### UDA Technique Hierarchy

```
Complexity Low → High:
│
├── Level 1: AdaBN (NO retraining)
│   └── Just update Batch Norm statistics from target data
│   └── Effort: 30 minutes
│
├── Level 2: Self-Training (pseudo-labeling)
│   └── Use high-confidence predictions as labels
│   └── Effort: 1-2 days
│
├── Level 3: MMD/CORAL Loss
│   └── Add distribution alignment to training loss
│   └── Effort: 2-3 days
│
└── Level 4: Adversarial (DANN/GAN)
    └── Domain discriminator + feature extractor
    └── Effort: 1-2 weeks
```

### Recommendation for Your Thesis

**Implement Levels 1-2:**
1. **AdaBN** (simple, effective, can show in thesis)
2. **Self-Training** (demonstrates MLOps retraining loop)

---

# Question 4: Hand Placement Statistics (ABCD Cases)

## 4.1 Population Statistics

| Factor | Statistic | Source |
|--------|-----------|--------|
| Left wrist wearing | ~70% | Smartwatch usage studies |
| Right-handedness | ~90% | General population |
| Dominant hand for fine tasks | 80-95% | Biomechanics research |
| Watch-Activity mismatch | ~63% | Computed: 70% × 90% |

## 4.2 The Four Domain Cases

| Case | Watch Wrist | Activity Hand | Signal Quality | Expected % |
|------|-------------|---------------|----------------|------------|
| **A** | Dominant (Right) | Dominant (Right) | **BEST** — Full motion (±2-10 m/s²) | ~7% |
| **B** | Non-dominant (Left) | Dominant (Right) | **WEAKEST** — Indirect only (±0.1-0.5 m/s²) | **~63%** |
| **C** | Dominant (Right) | Non-dominant (Left) | GOOD — Decent signal | ~3% |
| **D** | Non-dominant (Left) | Non-dominant (Left) | MODERATE | ~27% |

### Calculation Breakdown

```
Population Statistics:
- P(Right-handed) = 90%
- P(Left-handed) = 10%
- P(Watch on Left) = 70%
- P(Watch on Right) = 30%

For Right-Handed Users (90% of population):
- Case A: Watch=Right, Activity=Right → 30% × 90% = 27% → Signal BEST
- Case B: Watch=Left, Activity=Right → 70% × 90% = 63% → Signal WEAKEST
- Case C: Watch=Right, Activity=Left → 30% × 10% = 3% → Signal GOOD
- Case D: Watch=Left, Activity=Left → 70% × 10% = 7% → Signal MODERATE

For Left-Handed Users (10% of population):
- (Mirror of above)

DOMINANT CASE IN PRODUCTION: Case B (63%) — WORST signal quality!
```

## 4.3 Expected Model Behavior by Case

| Case | Confidence | Entropy | Flip Rate | Idle % |
|------|------------|---------|-----------|--------|
| **A (BEST)** | 85-95% | 0.2-0.8 | 10-20% | 10-20% |
| **B (WORST)** | 50-75% | 1.0-2.0 | 25-50% | 40-70% |
| **C** | 75-90% | 0.5-1.2 | 15-25% | 20-35% |
| **D** | 60-80% | 0.8-1.5 | 20-35% | 30-50% |

## 4.4 Visualization for Thesis

```python
import matplotlib.pyplot as plt

# Pie chart: Domain distribution
cases = ['Case A\n(Match: Best)', 'Case B\n(Mismatch: Worst)', 
         'Case C\n(Partial)', 'Case D\n(Partial)']
percentages = [7, 63, 3, 27]
colors = ['green', 'red', 'lightgreen', 'orange']

plt.figure(figsize=(10, 8))
plt.pie(percentages, labels=cases, colors=colors, autopct='%1.0f%%')
plt.title('Domain Distribution in Real-World Deployment\n(Most users are in Case B - worst signal)')
plt.savefig('docs/figures/domain_distribution.png', dpi=300)
```

---

# Question 5: Unit Tests and CI/CD Setup

## 5.1 Unit Test Structure

Create these files in `tests/` folder:

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_data_validator.py   # Data validation tests
├── test_preprocess.py       # Preprocessing tests
├── test_inference.py        # Inference tests
├── test_drift_detector.py   # Drift detection tests
└── test_integration.py      # End-to-end tests
```

### Example: `tests/test_data_validator.py`

```python
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')
from data_validator import DataValidator

class TestDataValidator:
    """Tests for src/data_validator.py"""
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sensor data: (n_windows, 200, 6)"""
        return np.random.randn(100, 200, 6).astype(np.float32)
    
    @pytest.fixture
    def invalid_data_shape(self):
        """Wrong shape: should be (n, 200, 6) not (n, 150, 6)"""
        return np.random.randn(100, 150, 6).astype(np.float32)
    
    def test_valid_data_passes(self, validator, valid_data):
        """Valid data should pass validation"""
        result = validator.validate(valid_data)
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_invalid_shape_fails(self, validator, invalid_data_shape):
        """Invalid shape should fail with clear error"""
        result = validator.validate(invalid_data_shape)
        assert result.is_valid == False
        assert 'shape' in result.errors[0].lower()
    
    def test_nan_values_detected(self, validator, valid_data):
        """NaN values should be detected"""
        valid_data[0, 0, 0] = np.nan
        result = validator.validate(valid_data)
        assert result.is_valid == False
        assert 'nan' in result.errors[0].lower()
    
    def test_out_of_range_values(self, validator, valid_data):
        """Accelerometer values > 20 m/s² should be flagged"""
        valid_data[0, 0, 0] = 100.0  # Unrealistic value
        result = validator.validate(valid_data)
        assert result.has_warnings == True
```

### Example: `tests/test_inference.py`

```python
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

class TestInference:
    """Tests for src/run_inference.py"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample inference data"""
        return np.random.randn(10, 200, 6).astype(np.float32)
    
    @pytest.fixture
    def mock_model(self, tmp_path):
        """Create a simple mock model for testing"""
        inputs = tf.keras.Input(shape=(200, 6))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(11, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        
        model_path = tmp_path / "mock_model.keras"
        model.save(model_path)
        return str(model_path)
    
    def test_prediction_shape(self, sample_data, mock_model):
        """Predictions should have shape (n_samples, 11)"""
        model = tf.keras.models.load_model(mock_model)
        predictions = model.predict(sample_data)
        
        assert predictions.shape == (10, 11)
    
    def test_probabilities_sum_to_one(self, sample_data, mock_model):
        """Softmax probabilities should sum to 1.0"""
        model = tf.keras.models.load_model(mock_model)
        predictions = model.predict(sample_data)
        
        row_sums = np.sum(predictions, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=5)
    
    def test_confidence_in_range(self, sample_data, mock_model):
        """Confidence should be between 0 and 1"""
        model = tf.keras.models.load_model(mock_model)
        predictions = model.predict(sample_data)
        confidence = np.max(predictions, axis=1)
        
        assert np.all(confidence >= 0.0)
        assert np.all(confidence <= 1.0)
```

## 5.2 CI/CD Pipeline with GitHub Actions

Create `.github/workflows/mlops.yml`:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  # Job 1: Lint and Format Check
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install linting tools
        run: |
          pip install flake8 black isort
      
      - name: Run flake8
        run: flake8 src/ scripts/ --max-line-length=120 --ignore=E501
      
      - name: Check formatting with black
        run: black --check src/ scripts/
      
      - name: Check import sorting
        run: isort --check-only src/ scripts/

  # Job 2: Run Tests
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r config/requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests with coverage
        run: |
          pytest tests/ -v --cov=src --cov-report=xml --cov-report=html
      
      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  # Job 3: Data Validation
  validate-data:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r config/requirements.txt
      
      - name: Run data validation
        run: python scripts/preprocess_qc.py
      
      - name: Check for data drift
        run: python scripts/post_inference_monitoring.py --check-drift-only

  # Job 4: Weekly Retraining (Scheduled)
  retrain:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r config/requirements.txt
      
      - name: Check for new labeled data
        id: check-data
        run: |
          python scripts/check_new_data.py
          echo "has_new_data=$?" >> $GITHUB_OUTPUT
      
      - name: Retrain if new data available
        if: steps.check-data.outputs.has_new_data == '0'
        run: python src/retrain_with_cv.py
      
      - name: Validate new model
        if: steps.check-data.outputs.has_new_data == '0'
        run: python scripts/validate_model.py
      
      - name: Deploy if better
        if: steps.check-data.outputs.has_new_data == '0'
        run: python scripts/deploy_if_better.py

  # Job 5: Build Docker Image
  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -f docker/Dockerfile.inference -t har-inference:latest .
      
      - name: Test Docker image
        run: |
          docker run --rm har-inference:latest python -c "import tensorflow; print('OK')"
```

## 5.3 Running Tests Locally

```powershell
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_validator.py -v

# Run tests matching pattern
pytest tests/ -k "inference" -v
```

---

# Question 6: Other Important Thesis Questions

## 6.1 Questions from Your Docs (Still Open)

| Question | Priority | File Reference |
|----------|----------|----------------|
| Should we train separate models for dominant vs non-dominant? | 🟠 HIGH | THESIS_PROGRESS_DASHBOARD |
| How to implement temperature scaling for calibration? | 🟠 HIGH | BIG_QUESTIONS |
| What's the minimum labeled samples for effective fine-tuning? | 🟡 MEDIUM | BIG_QUESTIONS |
| How to implement active learning for label acquisition? | 🟡 MEDIUM | RESEARCH_QA |

## 6.2 Critical Next Steps (Priority Order)

| Week | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Create `src/retrain_with_cv.py` | Working retraining script |
| **Week 1** | Add 5-10 unit tests | `tests/` with pytest |
| **Week 2** | Create CI/CD workflow | `.github/workflows/mlops.yml` |
| **Week 2** | Implement drift detector | `src/drift_detector.py` |
| **Week 3** | Add temperature scaling | Calibration in `evaluate_predictions.py` |
| **Week 3** | Label 100-200 windows | Audit set for validation |
| **Week 4+** | Start thesis writing | Chapter 1 draft |

## 6.3 Key Papers for Each Topic

### Model Degradation
- `Out-of-distribution_in_Human_Activity_Recognition.pdf`
- `WATCH_Wasserstein_Change_Point_Detection_for_High-Dimensional_Time_Series_Data.pdf`
- `Learning_Sinkhorn_Divergences_for_Change_Point_Detection.pdf`

### Retraining & Adaptation
- `Transfer_Learning_of_Human_Activities_Based_on_IMU_Sensors_A_Review.pdf`
- `Lifelong_Learning_in_Sensor-Based_Human_Activity_Recognition.pdf`
- `AdaptNet_Human_Activity_Recognition_via_Bilateral_Domain_Adaptation_Using_Semi-Supervised_Deep_Translation_Networks.pdf`

### Drift Detection
- `XHAR_Deep_Domain_Adaptation_for_Human_Activity_Recognition_with_Smart_Devices.pdf`
- `15027_Which_Time_Series_Domain_Shifts_can_Neural_Networks_Adapt_to_revised.pdf`

### Hand Placement
- `Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf`
- `Personalizing_Activity_Recognition_Models_Through_Quantifying_Different_Types_of_Uncertainty_Using_Wearable_Sensors.pdf`

### Uncertainty & Confidence
- `XAI-BayesHAR_A_novel_Framework_for_Human_Activity_Recognition_with_Integrated_Uncertainty_and_Shapely_Values.pdf`
- `A_Deep_Learning_Assisted_Method_for_Measuring_Uncertainty_in_Activity_Recognition_with_Wearable_Sensors.pdf`

---

# 📋 Summary Action Items

## Immediate (This Week)

1. ✅ **Create `tests/` folder with basic tests**
2. ✅ **Create `src/retrain_with_cv.py`**
3. ✅ **Create `.github/workflows/mlops.yml`**

## Short-term (2-3 Weeks)

4. 📝 **Implement drift detection metrics**
5. 📝 **Add temperature scaling for calibration**
6. 📝 **Create degradation visualization scripts**

## Medium-term (Month 2)

7. 📝 **Run domain adaptation experiments (AdaBN)**
8. 📝 **Collect hand placement statistics from your data**
9. 📝 **Start thesis Chapter 1 (Introduction)**

---

**Generated by:** Analysis of `paper for questions/` papers and existing documentation  
**Next Review:** February 7, 2026

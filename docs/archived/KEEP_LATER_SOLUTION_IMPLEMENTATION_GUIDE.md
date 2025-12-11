> 📦 **ARCHIVED - KEEP FOR LATER**
> 
> **Reason:** Domain shift solutions - needed when we resume that work
> 
> **Why keeping:** Contains step-by-step implementation guide for gravity removal, personalization, and domain adaptation. Paper-backed solutions ready to implement.
> 
> **When to use:** Week 5+ when domain shift work resumes (or if mentor approves hybrid approach)

---

# HAR Domain Shift: Implementation Guide (Paper-Backed Solutions)

**Status:** Ready to implement  
**Timeline:** Week 1 priorities listed below  
**Thesis Value:** HIGH - Demonstrates real-world MLOps challenges and solutions

---

## Solution 1: Gravity Removal via High-Pass Filter ⭐ START HERE

**Paper:** Yurtman et al. (2017) "Activity Recognition Invariant to Wearable Sensor Unit Orientation"  
**Effort:** 1 day  
**Impact:** Very High (eliminates the gravity shortcut)

### Why It Works
- Removes the **static gravity component** (constant -9.8 m/s²)
- Leaves only **dynamic motion** (what activities actually are)
- Tested on 7 different sensor placements - proven robust

### Implementation (Add to `src/preprocess_data.py`)

```python
from scipy.signal import butter, filtfilt

def remove_gravity_component(accelerometer_data, sampling_rate=50, cutoff_hz=0.3):
    """
    Remove static gravity component using high-pass Butterworth filter.
    
    Paper: Yurtman et al. (2017)
    - Cutoff frequency: 0.3 Hz (removes frequencies below 0.3 Hz)
    - Order: 3 (standard for this application)
    - This leaves only dynamic motion, removes static gravity bias
    
    Args:
        accelerometer_data: shape (n_samples, 3) with Ax, Ay, Az columns
        sampling_rate: Hz (default 50 for your data)
        cutoff_hz: cutoff frequency (default 0.3 Hz from paper)
    
    Returns:
        filtered_data: shape (n_samples, 3) with gravity removed
    """
    # Design high-pass Butterworth filter
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_hz / nyquist_freq
    
    b, a = butter(N=3, Wn=normalized_cutoff, btype='high')
    
    # Apply forward-backward filter (zero phase distortion)
    filtered_data = filtfilt(b, a, accelerometer_data, axis=0)
    
    return filtered_data


# USAGE IN PREPROCESSING PIPELINE:
# ================================
# After loading and unit conversion, before StandardScaler:

# 1. Load data
data = pd.read_csv('production_data.csv')
acc_raw = data[['Ax_w', 'Ay_w', 'Az_w']].values

# 2. Remove gravity (THIS IS NEW)
acc_no_gravity = remove_gravity_component(acc_raw, sampling_rate=50)
data[['Ax_w', 'Ay_w', 'Az_w']] = acc_no_gravity

# 3. Continue with existing preprocessing
data = scaler.transform(data)  # StandardScaler
windows = create_windows(data)  # Your windowing logic
```

### Expected Results
- **Before:** Production Az mean = -9.83 m/s² (constant)
- **After:** Production Az mean ≈ 0.0 m/s² (gravity removed)
- **Impact:** Model will see dynamic motion patterns, not gravity alignment

### Test This First
```python
# Quick validation
import matplotlib.pyplot as plt

acc_before = data[['Ax_w', 'Ay_w', 'Az_w']].values[:500]
acc_after = remove_gravity_component(acc_before)

plt.subplot(2,1,1)
plt.plot(acc_before[:, 2])  # Az before
plt.title('Before: Gravity Removal (Az constant at -9.83)')
plt.ylabel('m/s²')

plt.subplot(2,1,2)
plt.plot(acc_after[:, 2])   # Az after
plt.title('After: Gravity Removal (Az oscillates around 0)')
plt.ylabel('m/s²')
plt.show()
```

---

## Solution 2: g-Unit Normalization

**Paper:** Dhekane & Ploetz (2024) "Transfer Learning in Sensor-Based HAR: A Survey"  
**Effort:** 2 hours  
**Impact:** Medium (aligns with pre-training data range)

### Why It Works
- Your pre-training data was in **normalized g-units: [-2, +2]**
- Your fine-tuning data was in **raw m/s²: [-45, +45]**
- This 10x scale difference confuses the model
- Normalizing to g-units makes all data comparable

### Implementation

```python
def normalize_to_g_units(accelerometer_data_ms2):
    """
    Convert m/s² to g-units and clip to [-2, +2] range (ADAMSense standard).
    
    Paper: Dhekane & Ploetz (2024) - Section 4.2 "Sensor Heterogeneity"
    Recommendation: "Normalize all sensor units to common scale before training"
    """
    # Convert: 1g = 9.81 m/s²
    acc_g = accelerometer_data_ms2 / 9.81
    
    # Clip to match ADAMSense range
    acc_g_clipped = np.clip(acc_g, -2.0, 2.0)
    
    return acc_g_clipped


# USAGE:
acc_g = normalize_to_g_units(acc_raw)  # Convert to g-units
# Then apply StandardScaler with training's mean/std
```

### Optional Combination
Can be applied **after** gravity removal:
```python
acc_raw → gravity_removed → normalized_to_g_units → StandardScaler
```

---

## Solution 3: Lightweight Personalization ⭐ RECOMMENDED FOR PRODUCTION

**Paper:** Dey et al. (2015) "Toward Personalized Activity Recognition with Semipopulation Approach"  
**Effort:** 3 days  
**Impact:** Very High (20-30% accuracy improvement)  
**Data Needed:** 1-2 minutes labeled activities from production user

### Why It Works
- Shows that fine-tuning **only the final layer** on just **1-2 minutes of data** dramatically improves accuracy
- No need for weeks of data - just a quick calibration session
- Preserves learned features from pre-training/fine-tuning

### Implementation

```python
from tensorflow.keras.models import Model

def create_personalization_model(pretrained_model, freeze_until_layer='flatten'):
    """
    Freeze all layers except the last dense layer for quick personalization.
    
    Paper: Dey et al. (2015)
    Insight: "Fine-tuning only final layer on 1-2 minutes of user data improves accuracy"
    """
    # Freeze all layers up to specified point
    for layer in pretrained_model.layers:
        if layer.name == freeze_until_layer:
            break
        layer.trainable = False
    
    # Only final dense layers remain trainable
    return pretrained_model


# WORKFLOW:
# =========
# 1. Ask production user to perform 5-6 activities for 10 sec each:
#    - Sitting, standing, hand_tapping, scratching, knuckle_cracking, nape_rubbing
#    - Collect ~300 labeled samples (5 activities × 10 sec × 50 Hz)

# 2. Load pre-trained model
model = load_model('fine_tuned_model_1dcnnbilstm.keras')

# 3. Create personalization version
personalization_model = create_personalization_model(model)

# 4. Fine-tune on user's data
personalization_model.compile(optimizer='adam', loss='categorical_crossentropy', 
                              metrics=['accuracy'])
personalization_model.fit(user_X_train, user_y_train, 
                          epochs=10, batch_size=32)

# 5. Save personalized model
personalization_model.save('personalized_model_user1.keras')
```

### Data Collection Protocol
**To ask production user:**
> "Could you perform each of these 6 activities for 10 seconds?
> 1. Sitting normally
> 2. Standing
> 3. Tapping fingers on table
> 4. Scratching arm
> 5. Cracking knuckles
> 6. Rubbing nape of neck
>
> This takes ~2 minutes total and will calibrate the model to your device."

---

## Solution 4: Unsupervised Domain Adaptation (Advanced)

**Paper:** Fu et al. (2021) "Personalized Activity Recognition via Unsupervised Domain Adaptation"  
**Effort:** 1 week  
**Impact:** High (learns device-invariant features)  
**Data:** Your 2-4 weeks of unlabeled production data

### Why It Works
- **No new labels needed** - uses only unlabeled production data
- Adds a **domain classifier** that tries to predict "is this source or target?"
- **Gradient Reversal Layer (GRL)** flips gradients, forcing main network to ignore domain differences
- Result: Features work on ANY sensor/orientation

### Simplified Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda
from tensorflow.keras.models import Model

class GradientReversalLayer(Layer):
    """Gradient Reversal Layer - flips gradients during backprop."""
    
    def __init__(self, hp_lambda=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
    
    @tf.custom_gradient
    def reverse_gradient(self, x):
        def grad(dy):
            return -self.hp_lambda * dy
        return x, grad
    
    def call(self, x):
        return self.reverse_gradient(x)


def build_domain_adaptation_model(pretrained_model):
    """
    Add domain classifier head with Gradient Reversal Layer.
    
    Paper: Fu et al. (2021)
    Architecture:
    - Freeze pre-trained feature extractor
    - Add domain classifier head with GRL
    - Train on mixture of labeled (source) + unlabeled (target) data
    """
    
    # Use features from pre-trained model
    feature_extractor = Model(
        inputs=pretrained_model.input,
        outputs=pretrained_model.get_layer('flatten').output
    )
    
    # Add domain classifier with GRL
    features = feature_extractor.output
    reversed_features = GradientReversalLayer(hp_lambda=0.1)(features)
    domain_classifier = Dense(1, activation='sigmoid', name='domain')(reversed_features)
    
    # Multi-task model
    da_model = Model(
        inputs=pretrained_model.input,
        outputs=[pretrained_model.output, domain_classifier]
    )
    
    return da_model


# TRAINING:
# =========
# Requires mixing:
# - Labeled source data (training set): Y_activity, Y_domain=0
# - Unlabeled target data (production): Y_domain=1 (fake label for domain classifier)

da_model.compile(
    optimizer='adam',
    loss=['categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[1.0, 0.25]  # Adjust domain loss weight
)

da_model.fit(
    combined_data,  # Mix of training + production data
    [activity_labels, domain_labels],
    epochs=30,
    batch_size=64
)
```

### When to Use This
- **If you have 2-4 weeks of unlabeled production data**
- **Want thesis contribution on domain adaptation**
- **Need most robust long-term solution**

---

## Implementation Roadmap (Week-by-Week)

### Week 1 (Dec 10-16): Start with Gravity Removal
```
Day 1-2:  Understand Butterworth filter, test on sample data
Day 3-4:  Integrate into preprocess_data.py
Day 5:    Test on production data, verify Az mean → 0
Day 6-7:  Run inference pipeline, check if predictions improve
```

**Expected outcome:** Predictions spread across activities (not 100% hand_tapping)

### Week 2 (Dec 17-23): Add g-Unit Normalization (Optional)
```
Day 1-2:  Implement g-unit conversion
Day 3:    Compare with/without gravity removal
Day 4-5:  Run cross-validation on test set
```

### Week 3 (Dec 24-30): Lightweight Personalization
```
Day 1-2:  Prepare data collection protocol
Day 3-4:  Ask production user for 2 minutes of labeled data
Day 5-6:  Fine-tune last layer on user data
Day 7:    Evaluate personalized model accuracy
```

**Expected outcome:** 70-85% accuracy on production data

### Week 4+ (Jan 1+): Optional - Domain Adaptation
```
Day 1-7:  Collect unlabeled production data
Week 2:   Implement GRL-based domain adaptation
Week 3:   Train and validate
```

---

## Thesis Chapter Mapping

| Solution | Thesis Chapter | Contribution |
|----------|---|---|
| **Gravity Removal** | Methods → Preprocessing | Standard practice from literature |
| **g-Unit Normalization** | Methods → Preprocessing | Addresses sensor heterogeneity |
| **Personalization** | Experiments → Results | Practical improvement demonstration |
| **Domain Adaptation** | Results + Future Work | Novel application of GRL to HAR |

---

## Quick Reference: Which Solution to Use When?

```
IF you want quick results NOW:
→ Use Gravity Removal (1 day) + g-Unit Normalization (2 hours)
→ Get predictions spread across activities immediately

IF you want production-ready:
→ Add Lightweight Personalization (3 days)
→ Ask user for 2 min calibration data
→ Gets to 70-85% accuracy

IF you want thesis contribution + have 2-4 weeks:
→ Add Unsupervised Domain Adaptation (1 week)
→ Use your unlabeled production data
→ Novel paper-backed approach
```

---

## References (For Your Thesis)

1. **Yurtman et al. (2017).** "Activity Recognition Invariant to Wearable Sensor Unit Orientation"  
   *MDPI Sensors.* Vol. 17, No. 8.  
   https://www.mdpi.com/1424-8220/17/8/1838

2. **Dhekane & Ploetz (2024).** "Transfer Learning in Sensor-Based HAR: A Survey"  
   *arXiv.* https://arxiv.org/abs/2401.10185

3. **Dey et al. (2015).** "Toward Personalized Activity Recognition with Semipopulation Approach"  
   *ACM Computing Surveys.* https://dl.acm.org/doi/10.1145/2757290

4. **Fu et al. (2021).** "Personalized Activity Recognition via Unsupervised Domain Adaptation"  
   *MDPI Sensors.* Vol. 21, No. 3.  
   https://www.mdpi.com/1424-8220/21/3/885

---

## Files to Update

- `src/preprocess_data.py` - Add gravity removal + g-unit functions
- `src/run_inference.py` - Use gravity-removed preprocessing
- `notebook/experiments/solution_testing.ipynb` - Test and validate each approach
- `docs/SOLUTION_IMPLEMENTATION.md` - This file (reference guide)

---

**Status:** Ready to implement  
**Priority:** Start with Solution 1 (Gravity Removal)  
**Timeline:** 1 day to fix, 1 week to validate all 4 approaches  
**Thesis Value:** HIGH - Real-world domain shift with paper-backed solutions

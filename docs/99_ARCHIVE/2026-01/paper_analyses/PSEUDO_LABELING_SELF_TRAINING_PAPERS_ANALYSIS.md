# 📚 Pseudo-labeling, Self-training & Lifelong Learning Papers Analysis

> **Date:** January 30, 2026  
> **Thesis Constraints:**
> - Production data is **UNLABELED**
> - No online evaluation with labels
> - Deep model: **1D-CNN + BiLSTM**
> - Sensors: **AX AY AZ GX GY GZ** (6-axis IMU)

---

## Analysis Summary Table

| # | Paper | Year | Labels Assumed | Pseudo-label Method | Confidence Threshold | Critical for Thesis |
|---|-------|------|----------------|---------------------|---------------------|---------------------|
| 1 | Curriculum Labeling | 2021 | Partial (small labeled set) | Self-paced curriculum | Adaptive (top-k easiest) | ⭐⭐⭐ Key technique |
| 2 | SelfHAR | 2021 | Partial (small labeled set) | Teacher-student + contrastive | Confidence-based | ⭐⭐⭐ HAR-specific |
| 3 | Lifelong Learning HAR (Prototypical) | 2022 | Task-incremental | Prototype-based replay | Distance-based | ⭐⭐⭐ Continual learning |
| 4 | A*HAR (Semi-supervised) | 2021 | Partial (small labeled set) | Mean Teacher | Implicit (EMA) | ⭐⭐ Class imbalance |
| 5 | Sinkhorn Change Point | 2022 | Supervised CPD | Not pseudo-labeling | N/A | ⭐ Change detection |
| 6 | FusionActNet (Decoding HAR) | 2024 | Full supervision | None | N/A | ⭐ Architecture only |

---

## Paper 1: Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning |
| **Year** | 2021 |
| **Venue** | AAAI 2021 |
| **Authors** | Cascante-Bonilla, Tan, Qi, Ordonez |
| **Code** | https://github.com/uvavision/Curriculum-Labeling |

### 🎯 Problem Addressed
- **Core Problem:** Semi-supervised learning with small labeled set + large unlabeled set
- **Key Insight:** Standard pseudo-labeling suffers from confirmation bias and concept drift
- **Innovation:** Apply curriculum learning principles to pseudo-labeling + restart model parameters between cycles

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Small initial labeled set (e.g., 4,000 on CIFAR-10) |
| Unlabeled Data | **Yes** | Large unlabeled pool |
| Target Domain Labels | **No** | Uses pseudo-labels |
| Type | **Semi-Supervised Learning** | Self-training with curriculum |

### 🔄 Pseudo-labeling / Self-training Method
```
Algorithm: Curriculum Labeling
├── 1. Train initial model on labeled data
├── 2. FOR each self-training cycle:
│   ├── a. RESTART model parameters (avoid concept drift!)
│   ├── b. Generate pseudo-labels for ALL unlabeled data
│   ├── c. RANK samples by model confidence (curriculum)
│   ├── d. SELECT top-K easiest samples (self-paced)
│   ├── e. Add selected pseudo-labeled samples to training set
│   └── f. Retrain from scratch
└── 3. Repeat until convergence or all samples labeled
```

### 🎚️ Confidence Thresholds for Pseudo-labels

| Technique | Value | Description |
|-----------|-------|-------------|
| **Self-paced selection** | Top-K% per class | Selects easiest samples based on softmax confidence |
| **Growth rate** | ~10-20% per cycle | Gradually increases pseudo-labeled pool |
| **Per-class balancing** | Yes | Ensures class distribution is maintained |
| **Hard threshold** | Not used | Uses relative ranking instead of absolute threshold |

**Key Innovation:** Instead of using a fixed confidence threshold (e.g., >0.9), they use relative ranking:
- Cycle 1: Top 10% most confident per class
- Cycle 2: Top 20% most confident per class
- Continue until all samples included

### ⚠️ How to Avoid Confirmation Bias

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Model restart** | Reset parameters before each cycle | ⭐⭐⭐ Critical |
| **Curriculum ordering** | Easy samples first, hard samples later | ⭐⭐⭐ Critical |
| **Per-class balancing** | Maintain class proportions | ⭐⭐ Important |
| **No EMA/momentum** | Avoids error accumulation | ⭐⭐ Important |

**WHY RESTART WORKS:**
- Standard self-training: Errors accumulate as model "confirms" its own mistakes
- With restart: Fresh model sees pseudo-labels as new training data
- Prevents error amplification across cycles

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### ✅ Questions ANSWERED by This Paper

1. **"How to prevent pseudo-label errors from accumulating?"**
   - **Answer:** RESTART model parameters before each self-training cycle
   - **Implementation:** `model.load_state_dict(initial_checkpoint)` before each cycle

2. **"How to select which samples to pseudo-label?"**
   - **Answer:** Use curriculum learning - rank by confidence, select easiest first
   - **NOT a fixed threshold:** Use relative top-K% per class

3. **"How much labeled data is needed to start?"**
   - **Answer:** As low as ~4,000 samples on CIFAR-10 (10% of training data)
   - **For HAR:** Possibly 500-2000 labeled windows initially

4. **"How to handle class imbalance in unlabeled data?"**
   - **Answer:** Select top-K% PER CLASS to maintain balance
   - **Critical for HAR:** Walking/sitting may dominate, stairs/running rare

#### ❓ Questions RAISED but NOT ANSWERED

1. **"What if we have ZERO initial labels?"**
   - Paper assumes small labeled set exists
   - For pure production: Need bootstrap strategy (expert labeling, or clustering)

2. **"How to apply curriculum learning with streaming data?"**
   - Paper assumes batch setting with full unlabeled set available
   - Production: Need online curriculum adaptation

3. **"How many cycles until convergence?"**
   - Paper: 10-20 cycles typical
   - HAR-specific: Unknown, need to experiment

4. **"Does this work with deep BiLSTM architectures?"**
   - Paper: CNNs only (ResNet)
   - Need to verify for sequence models

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Designed for batch semi-supervised |
| Production Deployment | ⚠️ Adaptation needed | Requires periodic retraining cycles |
| Online Learning | ❌ No | Batch-based self-training |

### 🔧 Pipeline Stages Affected

| Stage | Impact | Implementation |
|-------|--------|----------------|
| **Retraining** | ⭐⭐⭐ High | Add curriculum pseudo-labeling cycle |
| **Active Learning** | ⭐⭐ Medium | Can combine with expert query |
| **Monitoring** | ⭐ Low | Track pseudo-label quality |

### 💻 Implementation Sketch for HAR
```python
def curriculum_labeling_cycle(model, labeled_data, unlabeled_data, 
                               initial_checkpoint, cycle_num, K_percent):
    """
    One cycle of curriculum labeling for HAR.
    """
    # 1. RESTART model to initial weights (CRITICAL!)
    model.load_state_dict(initial_checkpoint)
    
    # 2. Generate pseudo-labels with confidence
    pseudo_labels, confidences = [], []
    for batch in unlabeled_data:
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)
        pseudo_labels.append(probs.argmax(-1))
        confidences.append(probs.max(-1).values)
    
    # 3. Curriculum selection: top-K% per class
    K = K_percent * (cycle_num + 1)  # Increase each cycle
    selected_indices = select_top_k_per_class(
        pseudo_labels, confidences, K_percent=K
    )
    
    # 4. Create combined dataset
    combined = labeled_data + unlabeled_data[selected_indices]
    
    # 5. Retrain from scratch
    model.load_state_dict(initial_checkpoint)  # Restart again
    train(model, combined)
    
    return model
```

### 📊 Specific Thresholds and Techniques

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Initial K% | 10% | Start conservative |
| K growth rate | +10% per cycle | Linear growth |
| Number of cycles | 10-20 | Until all data included |
| Class balancing | Per-class top-K | Critical for imbalance |
| Temperature | T=1 (standard softmax) | No sharpening used |

---

## Paper 2: SelfHAR - Improving Human Activity Recognition through Self-training with Unlabeled Data

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data |
| **Year** | 2021 |
| **Venue** | IMWUT / UbiComp |
| **Authors** | Tang, Perez-Pozuelo, Spathis, Brage, Wareham, Mascolo |
| **DOI** | 10.1145/3448112 |
| **arXiv** | 2102.06073 |

### 🎯 Problem Addressed
- **Core Problem:** HAR with limited labeled data + massive unlabeled wearable sensor data
- **Key Insight:** Combine teacher-student self-training with multi-task self-supervision
- **HAR-Specific:** Designed specifically for accelerometer/IMU data

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Yes** | Small labeled HAR dataset |
| Unlabeled Data | **Yes** | Large unlabeled sensor data |
| Target Domain Labels | **No** | Uses pseudo-labels from teacher |
| Type | **Semi-Supervised HAR** | Teacher-student + self-supervision |

### 🔄 Pseudo-labeling / Self-training Method
```
SelfHAR Architecture:
├── Pre-training Phase (Self-Supervised)
│   ├── Multi-task learning on unlabeled data
│   ├── Tasks: Signal reconstruction, transformation prediction
│   └── Learns robust signal-level representations
│
├── Teacher Training Phase
│   ├── Fine-tune pre-trained model on labeled data
│   └── Creates "Teacher" model
│
├── Pseudo-labeling Phase
│   ├── Teacher generates pseudo-labels for unlabeled data
│   ├── Confidence-based filtering
│   └── Data augmentation applied
│
└── Student Training Phase
    ├── Student learns from:
    │   ├── Original labeled data (hard labels)
    │   └── Pseudo-labeled data (soft labels)
    └── Knowledge distillation from teacher
```

### 🎚️ Confidence Thresholds for Pseudo-labels

| Technique | Value | Description |
|-----------|-------|-------------|
| **Soft pseudo-labels** | Full probability distribution | Not hard thresholding |
| **Temperature scaling** | T > 1 (softened) | Preserves uncertainty |
| **Sample weighting** | Confidence-weighted loss | Higher confidence = higher weight |
| **Implicit threshold** | ~0.7-0.9 effective | Via sample weighting |

**Key Innovation:** Instead of hard threshold:
```python
# DON'T: Hard threshold
if confidence > 0.9:
    use_pseudo_label(sample)

# DO: Soft weighting (SelfHAR approach)
loss_weight = confidence ** alpha  # alpha typically 1-2
weighted_loss = loss_weight * cross_entropy(pseudo_label, prediction)
```

### ⚠️ How to Avoid Confirmation Bias

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Self-supervised pre-training** | Learn representations without labels first | ⭐⭐⭐ Critical |
| **Teacher-student separation** | Fixed teacher, trainable student | ⭐⭐⭐ Critical |
| **Data augmentation** | Apply different augmentations to student inputs | ⭐⭐ Important |
| **Soft pseudo-labels** | Use distribution, not hard labels | ⭐⭐ Important |
| **Multi-task pre-training** | Diverse pretext tasks prevent overfitting | ⭐⭐ Important |

**Self-Supervised Pretext Tasks for HAR:**
1. **Signal reconstruction:** Predict masked sensor values
2. **Transformation prediction:** Identify which augmentation was applied
3. **Temporal ordering:** Predict correct sequence order
4. **Contrastive learning:** Distinguish augmented versions of same sample

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### ✅ Questions ANSWERED by This Paper

1. **"How to leverage massive unlabeled HAR data?"**
   - **Answer:** Self-supervised pre-training on unlabeled data FIRST
   - **Then:** Teacher-student self-training for pseudo-labels
   - **Result:** Up to 12% F1 improvement, 10x less labeled data needed

2. **"What augmentations work for IMU data?"**
   - **Answer:** Time warping, scaling, rotation, noise addition, permutation
   - **HAR-specific:** Sensor-axis permutation, jittering

3. **"How to handle noisy pseudo-labels?"**
   - **Answer:** Use soft labels (probability distributions) + confidence weighting
   - **NOT:** Hard threshold filtering

4. **"How to transfer knowledge from teacher to student?"**
   - **Answer:** Knowledge distillation loss + weighted pseudo-label loss
   - **Temperature:** T > 1 for softer distributions

5. **"What architecture works for HAR self-training?"**
   - **Answer:** CNN-based encoder + classification head
   - **Compatible:** Can adapt for 1D-CNN + BiLSTM

#### ❓ Questions RAISED but NOT ANSWERED

1. **"What if teacher model is poor quality?"**
   - Paper assumes reasonable teacher from labeled data
   - Risk: Poor teacher → poor pseudo-labels → poor student

2. **"How to update teacher in production?"**
   - Paper uses fixed teacher
   - Need: EMA teacher or periodic teacher refresh

3. **"How to detect when self-training is harmful?"**
   - Paper doesn't provide validation strategy without labels
   - Need: Confidence monitoring, distribution tracking

4. **"How to handle concept drift in production?"**
   - Paper assumes static distribution
   - Production: Activities/users may change over time

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Designed for batch semi-supervised HAR |
| Production Deployment | ⚠️ Partial | Can do periodic retraining cycles |
| Online Learning | ⚠️ Partial | Would need streaming adaptation |

### 🔧 Pipeline Stages Affected

| Stage | Impact | Implementation |
|-------|--------|----------------|
| **Pre-training** | ⭐⭐⭐ High | Self-supervised on production data |
| **Retraining** | ⭐⭐⭐ High | Teacher-student pseudo-labeling |
| **Data Pipeline** | ⭐⭐ Medium | Add augmentation transforms |
| **Active Learning** | ⭐⭐ Medium | Combine with pseudo-labeling |

### 💻 Implementation Sketch for HAR
```python
class SelfHARPipeline:
    def __init__(self, encoder, classifier):
        self.encoder = encoder  # 1D-CNN
        self.classifier = classifier  # BiLSTM + head
        
    def self_supervised_pretrain(self, unlabeled_data):
        """Pre-train encoder on unlabeled data with pretext tasks."""
        for batch in unlabeled_data:
            # Task 1: Masked reconstruction
            masked_batch, mask = random_mask(batch)
            reconstructed = self.encoder.decode(self.encoder(masked_batch))
            loss_recon = F.mse_loss(reconstructed[mask], batch[mask])
            
            # Task 2: Transformation prediction
            aug_batch, aug_labels = apply_random_transform(batch)
            transform_pred = self.encoder.transform_head(self.encoder(aug_batch))
            loss_transform = F.cross_entropy(transform_pred, aug_labels)
            
            # Combine losses
            loss = loss_recon + loss_transform
            loss.backward()
            
    def train_teacher(self, labeled_data):
        """Fine-tune pre-trained model on labeled data."""
        self.teacher = deepcopy(self.encoder)
        for batch, labels in labeled_data:
            logits = self.classifier(self.teacher(batch))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
    def generate_pseudo_labels(self, unlabeled_data, temperature=2.0):
        """Generate soft pseudo-labels from teacher."""
        pseudo_data = []
        for batch in unlabeled_data:
            with torch.no_grad():
                logits = self.classifier(self.teacher(batch))
                soft_labels = F.softmax(logits / temperature, dim=-1)
                confidences = soft_labels.max(dim=-1).values
            pseudo_data.append((batch, soft_labels, confidences))
        return pseudo_data
        
    def train_student(self, labeled_data, pseudo_data, alpha=1.0):
        """Train student on labeled + pseudo-labeled data."""
        for batch, soft_labels, confidences in pseudo_data:
            # Apply augmentation to student input
            aug_batch = data_augment(batch)
            logits = self.classifier(self.encoder(aug_batch))
            
            # Confidence-weighted loss
            weights = confidences ** alpha
            loss = (weights * F.kl_div(
                F.log_softmax(logits, dim=-1), soft_labels, reduction='none'
            ).sum(-1)).mean()
            loss.backward()
```

### 📊 Specific Thresholds and Techniques

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Temperature (T) | 2.0 - 4.0 | Higher = softer distributions |
| Confidence weight α | 1.0 - 2.0 | Higher = stronger filtering |
| Pre-training epochs | 50-100 | Until representation converges |
| Labeled data % | 10-50% | SelfHAR effective with as low as 10% |
| Augmentation probability | 0.5 | Per-sample augmentation |

### 🆚 SelfHAR vs Curriculum Labeling Comparison

| Aspect | SelfHAR | Curriculum Labeling |
|--------|---------|---------------------|
| Pre-training | Self-supervised (required) | None |
| Teacher-student | Yes (fixed teacher) | No (single model) |
| Model restart | No | Yes (critical) |
| Pseudo-label type | Soft (distributions) | Hard (class labels) |
| Confidence handling | Weighted loss | Top-K selection |
| HAR-specific | Yes (designed for HAR) | No (general) |
| Data efficiency | 10x less labels | ~10% labels minimum |

---

## Paper 3: Lifelong Learning in Sensor-Based Human Activity Recognition (Prototypical Networks)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Lifelong Adaptive Machine Learning for Sensor-based Human Activity Recognition Using Prototypical Networks |
| **Year** | 2022 |
| **Venue** | Sensors (MDPI) |
| **Authors** | Rebecca Adaimi, Edison Thomaz |
| **DOI** | 10.3390/s22186881 |
| **arXiv** | 2203.05692 |

### 🎯 Problem Addressed
- **Core Problem:** Continual/lifelong learning for HAR without catastrophic forgetting
- **Key Challenge:** Model must adapt to new activities/users while retaining old knowledge
- **Innovation:** Prototypical networks + experience replay for task-free continual HAR

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Initial Training | **Yes** | Labeled data for initial classes |
| New Classes | **Partial** | Few-shot examples for new activities |
| Streaming Data | **Mixed** | Assumes task-free data-incremental scenario |
| Type | **Continual Learning** | Prototypical networks + replay |

### 🔄 Pseudo-labeling / Self-training Method
```
LAPNet-HAR Framework:
├── Prototypical Network Base
│   ├── Encoder extracts embeddings
│   ├── Class prototypes = mean of class embeddings
│   └── Classification via nearest prototype distance
│
├── Continual Prototype Adaptation
│   ├── Update prototypes with new samples
│   ├── Weighted update based on confidence
│   └── Prototype drift tracking
│
├── Experience Replay Buffer
│   ├── Store representative samples per class
│   ├── Replay during new class learning
│   └── Prevents catastrophic forgetting
│
└── Contrastive Loss
    ├── Pull same-class samples closer
    ├── Push different-class samples apart
    └── Facilitates inter-class separation
```

### 🎚️ Confidence Thresholds for Classification

| Technique | Value | Description |
|-----------|-------|-------------|
| **Distance-based confidence** | Euclidean/cosine distance | Closer to prototype = higher confidence |
| **Margin threshold** | Δ > τ for top-2 prototypes | Reject ambiguous samples |
| **Prototype update weight** | α = 0.1 - 0.5 | EMA for prototype updates |
| **Replay buffer size** | K per class | Maintains class balance |

**Distance-based Pseudo-labeling:**
```python
def classify_with_confidence(embedding, prototypes):
    distances = euclidean_distance(embedding, prototypes)  # [num_classes]
    sorted_dists = torch.sort(distances)
    
    pred_class = sorted_dists.indices[0]
    confidence = 1 / (1 + sorted_dists.values[0])  # Inverse distance
    
    # Margin-based confidence
    margin = sorted_dists.values[1] - sorted_dists.values[0]
    is_confident = margin > threshold
    
    return pred_class, confidence, is_confident
```

### ⚠️ How to Avoid Confirmation Bias (Catastrophic Forgetting)

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Experience replay** | Store and replay old samples | ⭐⭐⭐ Critical |
| **Prototype memory** | Maintain class prototypes separately | ⭐⭐⭐ Critical |
| **Contrastive loss** | Enforce class separation | ⭐⭐ Important |
| **Weighted prototype update** | Slow adaptation (low α) | ⭐⭐ Important |
| **Balanced replay** | Equal samples per class | ⭐⭐ Important |

**Key for Production:** Experience replay requires storing some labeled samples from each class. In purely unlabeled scenario, would need to store high-confidence pseudo-labeled samples.

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### ✅ Questions ANSWERED by This Paper

1. **"How to prevent forgetting old activities when adapting to new users?"**
   - **Answer:** Experience replay buffer + prototype memory
   - **Implementation:** Store K samples per class, replay during updates

2. **"How to handle task-free continual learning?"**
   - **Answer:** LAPNet-HAR designed for data-incremental (no task boundaries)
   - **Applicable:** Production data comes as continuous stream

3. **"How to quickly adapt to new activities with few examples?"**
   - **Answer:** Prototypical networks require only few-shot examples
   - **Advantage:** Fast prototype creation for new classes

4. **"How to measure adaptation quality without labels?"**
   - **Answer:** Track prototype drift and inter-class distances
   - **Metric:** Silhouette score, prototype stability

#### ❓ Questions RAISED but NOT ANSWERED

1. **"What if new samples have no labels at all?"**
   - Paper assumes few-shot labeled examples for new classes
   - Pure production: Need pseudo-labeling or active learning first

2. **"How to detect truly new activities vs domain shift?"**
   - Paper doesn't distinguish OOD detection from new class detection
   - Risk: Might create spurious new prototypes

3. **"How large should replay buffer be?"**
   - Paper: K = 50-200 per class
   - Trade-off: Memory vs forgetting prevention

4. **"How to combine with pseudo-labeling?"**
   - Not addressed: Integrating SelfHAR-style pseudo-labeling with prototypical continual learning

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Initial prototype creation |
| Production Deployment | ✅ Yes | Designed for continual adaptation |
| Online Learning | ✅ Yes | Task-free data-incremental |

### 🔧 Pipeline Stages Affected

| Stage | Impact | Implementation |
|-------|--------|----------------|
| **Model Architecture** | ⭐⭐⭐ High | Replace classifier with prototypes |
| **Retraining** | ⭐⭐⭐ High | Continual prototype updates |
| **Memory Management** | ⭐⭐⭐ High | Replay buffer implementation |
| **Active Learning** | ⭐⭐ Medium | Query samples near prototype boundaries |

### 💻 Implementation Sketch for HAR
```python
class LAPNetHAR:
    def __init__(self, encoder, num_classes, embed_dim, replay_size=100):
        self.encoder = encoder  # 1D-CNN + BiLSTM
        self.prototypes = torch.zeros(num_classes, embed_dim)
        self.prototype_counts = torch.zeros(num_classes)
        self.replay_buffer = ReplayBuffer(replay_size, num_classes)
        
    def classify(self, x):
        """Classify via nearest prototype."""
        embedding = self.encoder(x)
        distances = torch.cdist(embedding, self.prototypes)
        predictions = distances.argmin(dim=-1)
        confidences = 1 / (1 + distances.min(dim=-1).values)
        return predictions, confidences
        
    def update_prototype(self, x, label, alpha=0.1):
        """EMA update of class prototype."""
        embedding = self.encoder(x)
        self.prototypes[label] = (
            (1 - alpha) * self.prototypes[label] + 
            alpha * embedding.mean(dim=0)
        )
        self.prototype_counts[label] += x.shape[0]
        
    def continual_update(self, new_data, new_labels=None):
        """Update with new data, replay old samples."""
        # Classify new data if no labels
        if new_labels is None:
            new_labels, confidences = self.classify(new_data)
            # Only use high-confidence pseudo-labels
            mask = confidences > 0.8
            new_data = new_data[mask]
            new_labels = new_labels[mask]
        
        # Update prototypes with new data
        for c in new_labels.unique():
            class_data = new_data[new_labels == c]
            self.update_prototype(class_data, c)
            
        # Store in replay buffer
        self.replay_buffer.add(new_data, new_labels)
        
        # Replay old samples (prevent forgetting)
        replay_data, replay_labels = self.replay_buffer.sample()
        self.train_step(replay_data, replay_labels)
        
    def contrastive_loss(self, embeddings, labels):
        """Pull same-class closer, push different-class apart."""
        loss = 0
        for i in range(len(embeddings)):
            same_class = embeddings[labels == labels[i]]
            diff_class = embeddings[labels != labels[i]]
            
            # Positive: pull closer
            loss += (embeddings[i] - same_class.mean()).pow(2).mean()
            
            # Negative: push apart (with margin)
            margin = 1.0
            dist_diff = (embeddings[i] - diff_class).pow(2).sum(-1).sqrt()
            loss += F.relu(margin - dist_diff).mean()
            
        return loss
```

### 📊 Specific Thresholds and Techniques

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Prototype update α | 0.1 - 0.3 | Lower = more stable, higher = more adaptive |
| Replay buffer K | 50-200 per class | Trade-off memory vs forgetting |
| Confidence threshold | 0.7 - 0.9 | For pseudo-label acceptance |
| Contrastive margin | 1.0 | Minimum inter-class distance |
| Embedding dimension | 128 - 256 | For prototype comparison |

---

## Paper 4: A*HAR - Semi-supervised Learning for Class-imbalanced HAR

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | A*HAR: A New Benchmark towards Semi-supervised Learning for Class-imbalanced Human Activity Recognition |
| **Year** | 2021 |
| **Venue** | arXiv (IEEE format) |
| **Authors** | Narasimman, Lu, Raja, Foo, Aly, Lin, Chandrasekhar |
| **arXiv** | 2101.04859 |
| **Code** | https://github.com/I2RDL2/ASTAR-HAR |

### 🎯 Problem Addressed
- **Core Problem:** Semi-supervised HAR with severe class imbalance
- **Key Challenge:** Unlabeled data has unknown class distribution, often highly imbalanced
- **Innovation:** Benchmark + Mean Teacher evaluation for imbalanced HAR

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Source Domain Labels | **Partial** | Small labeled set (10-50%) |
| Unlabeled Data | **Yes** | Large unlabeled, class-imbalanced |
| Target Domain Labels | **No** | Uses Mean Teacher pseudo-labels |
| Type | **Semi-Supervised HAR** | Mean Teacher with CNN |

### 🔄 Pseudo-labeling / Self-training Method
```
Mean Teacher Framework:
├── Student Network
│   ├── Trainable parameters
│   ├── Standard supervised loss on labeled data
│   └── Consistency loss with teacher on unlabeled data
│
├── Teacher Network
│   ├── EMA of student weights: θ_t = α*θ_t + (1-α)*θ_s
│   ├── Generates pseudo-labels for unlabeled data
│   └── More stable predictions than student
│
└── Training Loop
    ├── For labeled: cross_entropy(student(x), y)
    ├── For unlabeled: consistency(student(x), teacher(x))
    └── Update teacher via EMA
```

### 🎚️ Confidence Thresholds for Pseudo-labels

| Technique | Value | Description |
|-----------|-------|-------------|
| **EMA decay α** | 0.999 | Teacher update momentum |
| **Consistency loss** | MSE or KL | Match student to teacher |
| **Implicit threshold** | Via EMA stability | No explicit threshold |
| **Ramp-up** | Linear over epochs | Increase consistency weight |

**Key Finding:** Mean Teacher improves overall accuracy but **fails on rare classes** in imbalanced setting.

### ⚠️ How to Avoid Confirmation Bias

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **EMA teacher** | Smooth weight updates | ⭐⭐ Moderate |
| **Consistency regularization** | Match predictions, not hard labels | ⭐⭐ Moderate |
| **Data augmentation** | Different augmentations for student/teacher | ⭐⭐ Moderate |

**CRITICAL FINDING:** Mean Teacher still suffers from class imbalance bias!
- Frequent classes get better pseudo-labels
- Rare classes get worse pseudo-labels
- Confirmation bias amplified for minority classes

### ❓ KEY QUESTIONS for Our Unlabeled Production Scenario

#### ✅ Questions ANSWERED by This Paper

1. **"Does semi-supervised learning help with imbalanced HAR?"**
   - **Answer:** Yes for majority classes, NO for minority classes
   - **Warning:** Activities like "stairs" or "running" may be harmed

2. **"What's a good baseline semi-supervised method?"**
   - **Answer:** Mean Teacher with CNN achieves reasonable performance
   - **But:** Not sufficient for class-imbalanced scenarios

3. **"Is there a benchmark for imbalanced semi-supervised HAR?"**
   - **Answer:** Yes, A*HAR benchmark released
   - **Useful:** For evaluating our methods

#### ❓ Questions RAISED but NOT ANSWERED

1. **"How to fix class imbalance in pseudo-labeling?"**
   - Paper identifies problem but doesn't solve it
   - Open problem: Class-imbalance-aware semi-supervised HAR

2. **"How to know class distribution in unlabeled data?"**
   - Paper assumes unknown distribution
   - Challenge: Can't balance what you don't know

3. **"Should we combine Mean Teacher with Curriculum Labeling?"**
   - Not explored: Curriculum + EMA teacher hybrid

4. **"How to weight minority class pseudo-labels higher?"**
   - Inverse frequency weighting?
   - Focal loss adaptation?

### 🏭 Production vs Offline
| Setting | Supported | Notes |
|---------|-----------|-------|
| Offline Training | ✅ Yes | Standard semi-supervised training |
| Production Deployment | ⚠️ Limited | Class imbalance problematic |
| Online Learning | ⚠️ Possible | EMA naturally supports streaming |

### 🔧 Pipeline Stages Affected

| Stage | Impact | Implementation |
|-------|--------|----------------|
| **Retraining** | ⭐⭐⭐ High | Add Mean Teacher training |
| **Class Balancing** | ⭐⭐⭐ High | CRITICAL for production |
| **Monitoring** | ⭐⭐ Medium | Track per-class pseudo-label quality |

### 📊 Specific Thresholds and Techniques

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| EMA decay α | 0.999 | Higher = more stable teacher |
| Consistency weight | 1.0 - 10.0 | Ramp up during training |
| Labeled ratio | 10-50% | A*HAR benchmark settings |
| Augmentation | Standard HAR augmentations | Jitter, scale, permute |

---

## Paper 5: Learning Sinkhorn Divergences for Supervised Change Point Detection

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Learning Sinkhorn Divergences for Supervised Change Point Detection |
| **Year** | 2022 |
| **Venue** | arXiv |
| **Authors** | Ahad, Dyer, Hengen, Xie, Davenport |
| **arXiv** | 2202.04000 |

### 🎯 Problem Addressed
- **Core Problem:** Detecting change points in complex sequential data
- **Key Challenge:** Unsupervised methods don't know which changes are important
- **Innovation:** Use labeled change point instances to learn a Sinkhorn divergence metric

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Change Point Labels | **Yes** | Supervised with labeled CPs |
| Time Series | **Unlabeled** | Streaming sequential data |
| Type | **Supervised Change Detection** | Learn metric from examples |

### ⚠️ Relevance to Our Scenario

**LIMITED RELEVANCE:** This paper is about change point detection, NOT pseudo-labeling or self-training.

| Aspect | Relevance | Notes |
|--------|-----------|-------|
| Pseudo-labeling | ❌ No | Not covered |
| Self-training | ❌ No | Not covered |
| HAR Domain | ⚠️ Partial | Could detect activity transitions |
| Unlabeled production | ❌ No | Requires labeled change points |

### 🔧 Potential Application for HAR MLOps

Could potentially use for:
1. Detecting distribution shift in production data
2. Identifying activity segment boundaries
3. Triggering retraining when change detected

But requires labeled change points for training, which contradicts our unlabeled scenario.

---

## Paper 6: FusionActNet - Decoding Human Activities (Static/Dynamic Classification)

### 📋 Basic Information
| Field | Value |
|-------|-------|
| **Full Title** | Decoding Human Activities: Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition |
| **Year** | 2024 (v3) |
| **Venue** | arXiv |
| **Authors** | Saha, Saha, Kabir, Fattah, Saquib |
| **arXiv** | 2310.02011 |

### 🎯 Problem Addressed
- **Core Problem:** HAR classification using accelerometer and gyroscope
- **Key Innovation:** Dual-network for static (immobile) vs dynamic (movement) activities
- **Architecture:** FusionActNet with guidance module

### 📊 Labels Assumed
| Category | Answer | Details |
|----------|--------|---------|
| Training Data | **Full labels** | Fully supervised |
| Type | **Supervised HAR** | Standard classification |

### ⚠️ Relevance to Our Scenario

**LIMITED RELEVANCE:** This paper is about supervised HAR architecture, NOT pseudo-labeling or self-training.

| Aspect | Relevance | Notes |
|--------|-----------|-------|
| Pseudo-labeling | ❌ No | Not covered |
| Self-training | ❌ No | Not covered |
| Architecture | ⭐⭐ Useful | Static/dynamic separation idea |
| Our sensors | ✅ Yes | Uses AX AY AZ GX GY GZ |

### 🔧 Potential Application for HAR MLOps

Architecture insights could be used for:
1. Pre-classifying static vs dynamic before detailed HAR
2. Using static/dynamic split for more targeted pseudo-labeling
3. Improving model architecture for 1D-CNN-BiLSTM

---

## Summary: Recommendations for Our Unlabeled Production Scenario

### 🏆 Most Relevant Papers (Ranked)

| Rank | Paper | Why Critical |
|------|-------|--------------|
| 1 | **SelfHAR** | HAR-specific, teacher-student, self-supervised pre-training |
| 2 | **Curriculum Labeling** | Model restart prevents bias, curriculum selection |
| 3 | **LAPNet-HAR** | Continual learning for production deployment |
| 4 | **A*HAR** | Class imbalance warning, Mean Teacher baseline |
| 5 | Sinkhorn CPD | Change detection (limited relevance) |
| 6 | FusionActNet | Architecture only (limited relevance) |

### 🎯 Recommended Hybrid Approach for Our Scenario

```
PROPOSED: SelfHAR + Curriculum + Prototypical Continual Learning

Phase 1: Bootstrap (Offline)
├── Self-supervised pre-training on production data (SelfHAR)
├── Fine-tune teacher on small labeled set
└── Create initial prototypes (LAPNet-HAR)

Phase 2: Initial Pseudo-labeling (Offline)
├── Teacher generates pseudo-labels
├── Curriculum selection: top-K% per class (Curriculum Labeling)
├── Class-balanced selection (avoid A*HAR imbalance issue)
└── Model restart before each cycle (Curriculum Labeling)

Phase 3: Continual Production Adaptation (Online)
├── Prototypical network updates (LAPNet-HAR)
├── Experience replay prevents forgetting
├── Periodic teacher refresh cycles
└── Drift detection triggers re-pseudo-labeling
```

### 🎚️ Recommended Thresholds Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| Self-training cycles | 10-20 | Curriculum Labeling |
| Initial K% selection | 10% per class | Curriculum Labeling |
| K growth rate | +10% per cycle | Curriculum Labeling |
| Teacher temperature | 2.0-4.0 | SelfHAR |
| Confidence weight α | 1.0-2.0 | SelfHAR |
| EMA decay (if used) | 0.999 | A*HAR |
| Prototype update α | 0.1-0.3 | LAPNet-HAR |
| Replay buffer K | 100 per class | LAPNet-HAR |
| Model restart | Yes, each cycle | Curriculum Labeling |

### ⚠️ Key Risks and Mitigations

| Risk | Mitigation | Source |
|------|------------|--------|
| Confirmation bias | Model restart between cycles | Curriculum Labeling |
| Class imbalance amplification | Per-class balanced selection | Curriculum + A*HAR warning |
| Catastrophic forgetting | Experience replay | LAPNet-HAR |
| Poor initial pseudo-labels | Self-supervised pre-training | SelfHAR |
| No validation without labels | Confidence monitoring, prototype drift | All papers |

### 📊 Pipeline Stages Affected Summary

| Stage | Papers Relevant | Key Technique |
|-------|-----------------|---------------|
| **Pre-training** | SelfHAR | Self-supervised on unlabeled |
| **Retraining** | Curriculum, SelfHAR, A*HAR | Teacher-student, curriculum |
| **Active Learning** | All | Query low-confidence samples |
| **Monitoring** | LAPNet-HAR | Prototype drift, confidence |
| **Drift Detection** | Sinkhorn (limited) | Distribution comparison |

---

## Questions Still Unanswered Across All Papers

### Critical Open Questions

1. **How to start with ZERO labels?**
   - All papers assume some initial labeled data
   - Need: Clustering-based initialization or expert bootstrapping

2. **How to validate pseudo-label quality without ground truth?**
   - No paper provides robust validation strategy
   - Need: Confidence calibration, agreement metrics

3. **How to combine curriculum + teacher-student + continual?**
   - Papers address separately
   - Need: Unified framework for production HAR

4. **What triggers retraining vs adaptation?**
   - Continual: Always adapt
   - Curriculum: Periodic cycles
   - Need: Drift-triggered decision

5. **How to handle sensors (AX AY AZ GX GY GZ) specifically?**
   - SelfHAR addresses accelerometer
   - FusionActNet addresses both
   - Need: IMU-specific augmentations and pre-training

---

*Analysis completed: January 30, 2026*

# Research Papers Analysis: ICTH_16 & EHB_2025_71

> **📝 Summary:** Deep-dive analysis of the two foundational papers for this thesis. Contains: training data source (ADAMSense dataset), 11 activity classes, model architecture details (1D-CNN-BiLSTM), domain adaptation methodology (49% → 87% accuracy), and RAG-enhanced report generation pipeline. Use this to understand the research foundation.

**Date:** December 9, 2025  
**Papers Analyzed:**
1. ICTH_16.pdf - "Recognition of Anxiety-Related Activities using 1DCNNBiLSTM on Sensor Data from a Commercial Wearable Device"
2. EHB_2025_71.pdf - "A Multi-Stage, RAG-Enhanced Pipeline for Generating Longitudinal, Clinically Actionable Mental Health Reports from Wearable Sensor Data"

**Authors:** Ugonna Oleh and Roman Obermaisser (University of Siegen)

---

## 1. TRAINING DATA SOURCE

### Paper ICTH_16:
- **Primary Training Dataset:** ADAMSense Dataset (public, research-grade)
  - **Source:** Khan et al. (2021) - "ADAM-sense: Anxiety-displaying activities recognition by motion sensors"
  - **Sensors:** Wrist-worn and chest-worn IMU sensors (accelerometer, gyroscope, magnetometer)
  - **Activities:** 11 anxiety-related activities:
    1. Ear rubbing or scratching
    2. Forehead rubbing or scratching
    3. Hair pulling
    4. Hand rubbing or scratching
    5. Hand tapping
    6. Knuckle cracking
    7. Nail biting
    8. Nape rubbing or scratching
    9. Smoking
    10. Idle sitting
    11. Idle standing

- **Fine-tuning Dataset:** Custom Garmin Venu 3 Dataset
  - **Device:** Garmin Venu 3 smartwatch
  - **Data Collection:** 6 volunteers
  - **Collection Method:** Each participant wore the watch on dominant wrist with sensor logger app enabled
  - **Controlled vs Free-living:** Participants were prompted to perform specific activities (controlled)
  - **Sensors Collected:** 3-axis accelerometer + 3-axis gyroscope only (magnetometer not available due to API limitations)

### Paper EHB_2025_71:
- **Model Source:** Same 1D-CNN-BiLSTM pre-trained on ADAMSense
- **Case Study Device:** Garmin Venu 3 smartwatch
- **Data Subject:** 39-year-old female with clinical diagnosis of panic disorder with agoraphobic avoidance
- **Monitoring Duration:** Week 1 (4 days baseline) + Week 2 (5 days comparison)

---

## 2. DATA COLLECTION METHOD

### ADAMSense Dataset:
- **Type:** Research-grade sensors (wrist-worn + chest-worn)
- **Device Orientation:** Wrist-worn (dominant wrist in ICTH_16 fine-tuning)
- **Mounting Position:** Wrist and chest
- **Number of Subjects:** Not explicitly stated for ADAMSense itself, but ICTH_16 used 6 volunteers for Garmin data

### Garmin Venu 3 Custom Dataset (ICTH_16):
- **Collection Environment:** Controlled setting - participants prompted to perform activities
- **Device Positioning:** Dominant wrist
- **Number of Subjects:** 6 volunteers
- **Consent:** Informed consent provided before data collection
- **Data Storage:** Proprietary FIT (Flexible and Interoperable Training) format initially
- **Logging Method:** Custom Monkey C app developed for the Garmin device
- **Data Access:** Manual extraction via USB connection to personal computer

### EHB_2025_71 Case Study:
- **Setting:** Continuous real-world monitoring (not prompted activities)
- **Device:** Garmin Venu 3 smartwatch
- **Data Streams:**
  - 3-axis accelerometer (motion)
  - 3-axis gyroscope (motion)
  - PPG (photoplethysmography for heart rate monitoring)
- **Supplementary Data:** Daily digital journal with mood, nervousness, sleep quality, and panic attack details
- **Ethics:** Study approved by University of Siegen Ethics Committee

---

## 3. PREPROCESSING STEPS

### ICTH_16 - ADAMSense Data Processing:
1. **Feature Selection:** Extracted wrist-worn accelerometer (3-axis) + gyroscope (3-axis) only
2. **Data Reduction:** From 25 features to 6 features to match Garmin Venu 3 capabilities
3. **Discarded Data:** Pocket and magnetometer sensor data

### ICTH_16 - Custom Garmin Venu 3 Data Processing:
1. **Format Conversion:** FIT files → CSV format
2. **Resampling/Downsampling:**
   - **Original Rate:** 100 Hz
   - **Target Rate:** 50 Hz
   - **Method:** "Resampling the data to 20-millisecond intervals and taking the mean of the values within each interval"
3. **Synchronization:** Data streams synchronized using their timestamps
4. **Unit Conversion:** "Accelerometer data was converted from milli-g to m/s²"
5. **Manual Labeling:** Data manually labeled against video recordings of the sessions

### Shared Processing (Both Datasets):
1. **Windowing Strategy:**
   - **Window Size:** 200 time steps (equivalent to 4 seconds at 50 Hz)
   - **Overlap:** 50% overlap between consecutive windows
   - **Rationale:** "4-second window is long enough to capture a complete cycle of the target activities, while 50% overlap prevents loss of information at window boundaries and serves as a form of data augmentation"

2. **No Explicit Preprocessing Mentioned:**
   - NO gravity removal mentioned
   - NO filtering (high-pass, low-pass, etc.) mentioned
   - NO normalization mentioned (but may be handled by batch normalization in the model)

### EHB_2025_71 Additional Processing:
1. **Temporal Bout Analysis:**
   - **Heart Rate Bouts:** Grouped by gaps < 120 seconds (2 minutes)
     - *Rationale:* "Estimated time constant for autonomic nervous system recovery, ensuring that only sustained periods of arousal, rather than momentary transient noise, are grouped as a single event"
   - **Behavioral Bouts:** Grouped by gaps < 300 seconds (5 minutes)
     - *Rationale:* "Ensure behavioral continuity, distinguishing a brief interruption (e.g., a hand adjustment) from a definitive shift in the psychomotor state"

2. **Metric Extraction per Period:**
   - Total duration of each activity
   - Number of distinct bouts
   - Average and maximum bout duration
   - Primary time of day activity occurs
   - Heart rate metrics calculated relative to daily mean HR

---

## 4. GRAVITY HANDLING

**Result: NO GRAVITY REMOVAL MENTIONED IN EITHER PAPER**

- Neither paper explicitly mentions gravity removal, offset, or filtering
- The preprocessing focuses on:
  - Downsampling
  - Unit conversion (milli-g → m/s²)
  - Timestamp synchronization
  - Windowing
- The raw accelerometer data appears to include the full 1g component (whether in milli-g or m/s²)
- This is different from many HAR applications where gravity is explicitly removed

---

## 5. DEVICE ORIENTATION

### Garmin Venu 3:
- **Mounting Position:** Dominant wrist
- **Sensor Axis Orientation:** Not explicitly specified in papers
- **Design:** Commercial smartwatch with built-in IMU sensors (assumes standard wrist orientation)

### ADAMSense:
- **Positions:** Wrist-worn and chest-worn IMU sensors
- **Axis Orientation:** Not explicitly specified

**Note:** The papers focus on the fact that orientation is standardized but don't detail how to handle rotational differences or accelerometer axes alignment.

---

## 6. SAMPLING RATE

### ADAMSense Dataset:
- **Original Sampling Rate:** Not explicitly stated in papers

### Garmin Venu 3 Custom Data:
- **Raw Collection Rate:** 100 Hz
- **Processing Rate:** Downsampled to 50 Hz
  - **Method:** Resampling to 20-millisecond intervals + mean aggregation
  - **Reason:** "To maintain consistency with the ADAMSense dataset and reduce computational load"

### Windowing:
- **Window Size:** 200 time steps at 50 Hz = 4 seconds
- **Sampling Points per Window:** 200 samples
- **Overlap:** 50% = 100 samples overlap between consecutive windows

---

## 7. UNITS

### Accelerometer Units:
- **Garmin Data (Raw):** Milligravity (milli-g)
- **After Conversion:** m/s² (gravitational acceleration)
- **Conversion Formula:** Implied: 1g ≈ 9.81 m/s², so 1000 milli-g = 9.81 m/s²

### Gyroscope Units:
- **Units:** Not explicitly stated in papers
- **Assumption:** Likely degrees per second (°/s) or radians per second (rad/s), standard for IMU gyroscopes

### Heart Rate (PPG):
- **Units:** beats per minute (bpm) - mentioned in EHB_2025_71
- **Example:** Baseline Week 1: 81.67 bpm, Week 2: 76.45 bpm

---

## 8. MODEL ARCHITECTURE

### 1D-CNN-BiLSTM Model:

#### Overall Design:
- **Type:** Hybrid deep learning architecture
- **Purpose:** Classify subtle, anxiety-related activities from time-series IMU data
- **Key Advantage:** Combines CNN's feature extraction with BiLSTM's temporal context awareness

#### Layer Composition:

**1. First Layer - 1D Convolutional Neural Network (1DCNN):**
- **Purpose:** Automatic hierarchical feature extraction from raw sensor signals
- **Advantages:**
  - Accepts time-series data directly without reshaping
  - Automatically learns features from raw signals (no manual feature engineering needed)
  - Identifies salient local patterns

**2. Middle Layers - Three BiLSTM Layers:**
- **Purpose:** Model temporal dependencies between features
- **Key Feature:** Bidirectional processing (forward + backward through sequence)
- **Advantages:**
  - Provides complete contextual understanding of activities
  - Crucial for distinguishing similar activities (e.g., "hand rubbing" vs. "knuckle cracking")
  - Bidirectional advantage over standard LSTM

**3. Regularization Techniques:**
- **Batch Normalization:** Applied after each layer
  - Purpose: Stabilize learning, improve convergence
- **Dropout:** Used consistently throughout the network
  - Purpose: Prevent overfitting

#### Ablation Study Results (on ADAMSense subset):
| Model Architecture | F1-Score |
|---|---|
| 1DCNNBiLSTM (Proposed) | **0.871** |
| LSTM-Only | 0.828 |
| 1DCNN-LSTM | 0.817 |
| BiLSTM-Only | 0.813 |
| 1DCNN-Only | 0.697 |

**Interpretation:** The hybrid model significantly outperforms single-component models, validating that CNN feature extraction + BiLSTM temporal modeling are synergistic.

#### Input/Output Format:
- **Input:** 4-second windows (200 time steps) of 6-feature data (3-axis accel + 3-axis gyro)
- **Output:** Activity classification label + confidence score per window

---

## 9. VALIDATION RESULTS

### ICTH_16 Performance:

#### Stage 1: Baseline on Full ADAMSense Dataset
- **Accuracy:** 98.8%
- **Loss:** 0.037
- **Average F1-Score:** 0.99
- **Training:** 28 epochs with early stopping (stops if validation accuracy doesn't improve after 5 epochs)
- **Note:** Improvement from 92% accuracy of the previous best model on ADAMSense

#### Stage 2: Domain Shift Quantification (Base Model on Garmin Data)
- **Training Data:** ADAMSense 6-feature subset (wrist accel + gyro only)
- **Test Data:** Custom Garmin Venu 3 dataset
- **Without Fine-tuning Accuracy:** **48.7%** (lab-to-life gap)
- **F1-Score (base model):** 0.90 (on ADAMSense subset)
- **Training:** 35 epochs with early stopping

**Key Finding:** This massive accuracy drop (89.11% → 48.7%) demonstrates the critical importance of domain adaptation.

#### Stage 3: Fine-Tuning with 5-Fold Cross-Validation
- **Validation Protocol:** 5-fold cross-validation on custom Garmin dataset
  - 4 folds for fine-tuning
  - 1 fold for testing
  - Repeated 5 times, each fold used as test set once

**Performance Metrics:**

| Metric | Before Fine-tuning | After Fine-tuning |
|---|---|---|
| Accuracy | 48.7% | 87.0% ± 1.2% |
| Weighted Avg Precision | 0.50 | 0.86 ± 0.02 |
| Weighted Avg Recall | 0.49 | 0.87 ± 0.01 |
| Weighted Avg F1-Score | 0.49 | 0.86 ± 0.01 |

**Per-Class Performance (After Fine-tuning, 5-fold average):**

| Activity | Precision | Recall | F1-Score |
|---|---|---|---|
| Ear rubbing | 0.77 ± 0.06 | 0.95 ± 0.03 | 0.85 ± 0.04 |
| Forehead rubbing | 0.90 ± 0.05 | 0.89 ± 0.03 | 0.89 ± 0.04 |
| Hair pulling | 0.92 ± 0.03 | 0.74 ± 0.09 | 0.82 ± 0.06 |
| Hand scratching | 0.82 ± 0.03 | 0.75 ± 0.04 | 0.78 ± 0.03 |
| Hand tapping | 0.77 ± 0.03 | 0.99 ± 0.01 | 0.86 ± 0.02 |
| Knuckles cracking | 0.82 ± 0.05 | 0.82 ± 0.02 | 0.82 ± 0.03 |
| Nail biting | 0.91 ± 0.03 | 0.87 ± 0.05 | 0.89 ± 0.02 |
| Nape rubbing | 0.87 ± 0.04 | 0.93 ± 0.03 | 0.90 ± 0.02 |
| Sitting | 0.93 ± 0.05 | 0.93 ± 0.04 | 0.93 ± 0.02 |
| Smoking | 0.86 ± 0.06 | 0.82 ± 0.06 | 0.84 ± 0.03 |
| Standing | 0.94 ± 0.04 | 0.95 ± 0.04 | 0.94 ± 0.02 |

**Variability Across Classes:**
- Best: Standing (0.94 F1) and Sitting (0.93 F1) - distinct motion signatures
- Weakest: Hand scratching (0.78 F1) - less consistent motion signature
- Note: Activities with clearer, more consistent motion patterns are easier to recognize

### EHB_2025_71 Performance:

#### HAR Model on Garmin Data (5-fold cross-validation):
- **Accuracy:** 0.87 ± 0.02
- **Precision:** 0.86 ± 0.02
- **Recall:** 0.87 ± 0.01
- **Macro F1-Score:** 0.86 ± 0.01
- **Note:** Same as ICTH_16 fine-tuning results (validates consistency)

#### Case Study Results:
**Week 1 vs Week 2 Analysis:**
- **Average Daily HR (Baseline Week 1):** 81.67 bpm
- **Average Daily HR (Week 2):** 76.45 bpm
  - *Apparent improvement but misleading without temporal analysis*

- **HR Bout Count (Baseline):** 82 bouts
- **HR Bout Count (Week 2):** 54 bouts (↓ 34%)
  - *Fewer bouts suggests improvement*

- **Total HR Arousal Duration (Baseline):** Unknown (implied shorter bouts)
- **Total HR Arousal Duration (Week 2):** 8 hours 6 minutes
- **Maximum Single Bout Duration (Week 2):** 37.5 minutes
  - *Critical insight: Fewer but much longer/intense episodes*

**Clinical Interpretation:**
- Shift from: Frequent, short arousal episodes
- To: Fewer, but substantially longer and more intense episodes
- **Likelihood of Future Panic Attacks:** Increasing (per RAG-enhanced LLM report)
- **Recommended Treatment:** Autonomic Regulation Techniques (biofeedback, paced breathing)

---

## 10. KEY DIFFERENCES: TRAINING VS PRODUCTION PREPROCESSING

### Training Data Pipeline (ICTH_16):
```
Raw Garmin Data (100 Hz)
  ↓ (Convert FIT → CSV)
  ↓ (Downsample to 50 Hz via mean aggregation)
  ↓ (Synchronize timestamps)
  ↓ (Convert milli-g → m/s²)
  ↓ (Manual labeling from video)
  ↓ (Windowing: 200 timesteps, 50% overlap)
  ↓
6-feature Time-series (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
```

### What's Notably ABSENT:
- ❌ No gravity removal
- ❌ No explicit filtering (butterworth, moving average, etc.)
- ❌ No explicit normalization (relies on batch norm in model)
- ❌ No offset removal
- ❌ No axis rotation/alignment mentioned

### Critical Implications for Production:
1. **Gravity Component:** The accelerometer data includes the full gravitational component
2. **Sampling Rate Match:** Production data must be 50 Hz (not 100 Hz)
3. **Unit Match:** Production acceleration must be in m/s² (not milli-g)
4. **Windowing Match:** 4-second windows with 50% overlap required
5. **No Ground Truth:** Production system won't have manual labeling; relies entirely on model confidence

---

## 11. RESEARCH CONTRIBUTIONS & METHODOLOGY SUMMARY

### ICTH_16 Main Contributions:
1. **Quantified the "Lab-to-Life" Gap:**
   - Model trained on research-grade (ADAMSense) achieves 89.11% on source data
   - Same model on commercial Garmin data: 48.7% (41.4% drop)
   - Clear evidence that domain adaptation is essential

2. **Validated Fine-Tuning as Solution:**
   - Simple fine-tuning on small custom dataset (6 subjects)
   - Recovers performance to 87.0% on target hardware
   - Practical pathway for practitioners without large custom datasets

3. **Hybrid 1DCNNBiLSTM Architecture:**
   - Specifically designed for subtle anxiety-related activities
   - Outperforms single-component models
   - Combines feature extraction (CNN) with temporal modeling (BiLSTM)

4. **Domain Adaptation Strategy:**
   - Pre-train on public research dataset (ADAMSense)
   - Fine-tune on limited commercial device data (Garmin)
   - Reduces need for massive new datasets per device

### EHB_2025_71 Main Contributions:
1. **Multi-Stage Automated Pipeline:**
   - HAR (recognize activities)
   - Temporal Bout Analysis (contextualize events)
   - RAG-Enhanced LLM (generate clinical reports)

2. **Graph-Based RAG System:**
   - Knowledge graph built from clinical literature
   - Semantic triples ensure factual grounding
   - Prevents LLM hallucinations in clinical context

3. **Multi-Audience Report Generation:**
   - Same data → three different reports
   - Psychologist (clinical, data-focused)
   - Patient (empathetic, supportive)
   - Research (comprehensive, analytical)

4. **Longitudinal, Baseline-Referenced Analysis:**
   - Establishes personal baseline from initial monitoring
   - Week-over-week comparative analysis
   - Identifies trends vs one-time measurements

---

## 12. CRITICAL FINDINGS FOR YOUR PROJECT

### ✅ What the Papers Use:
1. **Sampling Rate:** 50 Hz (downsampled from 100 Hz raw)
2. **Window Size:** 4-second windows (200 samples at 50 Hz)
3. **Window Overlap:** 50%
4. **Features:** 6-axis IMU (3-axis accel + 3-axis gyro)
5. **Unit Conversion:** milli-g → m/s²
6. **NO Explicit Preprocessing:** No gravity removal, no filtering, no normalization

### ⚠️ Common Production Pitfalls:
- Using 100 Hz data without downsampling
- Using different units (keeping in milli-g)
- Applying gravity removal (not in original preprocessing)
- Using different window sizes or overlap ratios
- Applying additional filtering not in original pipeline

### 🎯 Exact Preprocessing Reproducibility:
The papers provide sufficient detail to reproduce their exact preprocessing:
1. Collect 100 Hz raw Garmin IMU data
2. Downsample to 50 Hz (20ms intervals, mean aggregation)
3. Convert accelerometer from milli-g to m/s²
4. Synchronize timestamp streams
5. Create 200-timestep windows (4 seconds) with 50% overlap
6. Feed to 1DCNNBiLSTM model (exactly as ablation study validates)

---

## 13. LIMITATIONS ACKNOWLEDGED BY AUTHORS

### ICTH_16 Limitations:
1. **Small Dataset:** Only 6 volunteers for custom Garmin data
   - Intended as proof-of-concept, not generalizable model
   - Future work needs larger, more diverse cohort

2. **Activity Variability:** Some classes harder to recognize
   - Hand scratching: 0.78 F1 (inconsistent motion)
   - Standing: 0.94 F1 (distinct motion)
   - Suggests some activities need more training examples

3. **Device-Specific Adaptation:** Model fine-tuned for Garmin Venu 3
   - May not transfer to other smartwatch brands
   - Each device may require custom fine-tuning

### EHB_2025_71 Limitations:
1. **Case Study:** Single volunteer (39-year-old female)
   - Demonstrates proof-of-concept pipeline
   - Not validated across multiple patients or conditions

2. **Data Gaps:** Week 2 missing self-reported journal data
   - Pipeline had to handle missing modalities
   - Real-world messiness demonstrated

3. **RAG Knowledge Base:** Manually constructed from literature
   - May not cover all clinical scenarios
   - Requires clinical domain expert input

---

## REFERENCES

**ICTH_16 (ICTH 2025 Conference Paper):**
- Oleh, U., & Obermaisser, R. (2025). Recognition of Anxiety-Related Activities using 1DCNNBiLSTM on Sensor Data from a Commercial Wearable Device. *Procedia Computer Science*, Conference Paper for The 15th International Conference on Current and Future Trends of Information and Communication Technologies in Healthcare (ICTH 2025).

**EHB_2025_71:**
- Oleh, U., Obermaisser, R., Malchulska, A., & Klucken, T. (2025). A Multi-Stage, RAG-Enhanced Pipeline for Generating Longitudinal, Clinically Actionable Mental Health Reports from Wearable Sensor Data.

**ADAMSense Dataset Reference:**
- Khan, N.S., Ghani, M.S., & Anjum, G. (2021). ADAM-sense: Anxiety-displaying activities recognition by motion sensors. *Pervasive and Mobile Computing*, 78, 101485.

---

## DIRECT QUOTES FROM PAPERS

### On Preprocessing (ICTH_16, Page 4):
> "The raw sensor data collected from the Garmin Venu 3 was saved in the proprietary FIT file format. The first processing step was to convert these FIT files into a standard CSV format for analysis. The raw data was recorded at a frequency of 100Hz. However, to maintain consistency with the ADAMSense dataset and reduce computational load, the data was downsampled to 50Hz. This was achieved by resampling the data to 20-millisecond intervals and taking the mean of the values within each interval. Data streams were synchronised using their timestamps, and accelerometer data was converted from milli-g to m/s². Finally, the processed time-series data was manually labelled against video recordings of the sessions."

### On Windowing (ICTH_16, Page 4):
> "The continuous data streams were segmented using a window size of 200 time steps (equivalent to 4 seconds at 50Hz) with a 50% overlap between consecutive windows. This approach follows standard practice in HAR; the 4-second window is long enough to capture a complete cycle of the target activities, while the 50% overlap prevents loss of information at window boundaries and serves as a form of data augmentation."

### On Domain Shift (ICTH_16, Page 6):
> "Without any fine-tuning, the model performed poorly, achieving an accuracy of only 48.7%."

### On Fine-tuning Results (ICTH_16, Page 6):
> "The model achieved a mean accuracy of 87.0% (± 1.2%) across the five folds. This significant increase from 48.7% to 87.0% clearly demonstrates the critical impact and success of the fine-tuning strategy in adapting the model to the commercial wearable sensor data."

### On Temporal Bout Analysis (EHB_2025_71, Page 5):
> "Physiological Arousal (HR) bouts (periods of heart rate above the daily average) are broken by a time gap exceeding 120 seconds (2 minutes). This duration is chosen to align with the estimated time constant for autonomic nervous system recovery, ensuring that only sustained periods of arousal, rather than momentary transient noise, are grouped as a single event."

---

**Document Created:** December 9, 2025  
**Extracted by:** PDF Text Extraction + Manual Analysis  
**Confidence Level:** High (direct quotes from peer-reviewed publications)

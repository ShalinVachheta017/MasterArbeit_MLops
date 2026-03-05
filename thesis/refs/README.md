# Thesis Reference Papers

This folder contains the 7 academic papers **inside the `Thesis_report/` audit scope** that provide
evidence for the pipeline's design decisions.  They were copied here on 2026-02-27 from
`archive/research_papers/76 papers/` so that `scripts/extract_papers_to_text.py` (which only
searches inside `Thesis_report/`) can find them.

---

## Papers in This Folder

### 1. Deep CNN-LSTM With Self-Attention Model … Wearable Sensor (IEEE JTEHM 2022)
**File:** `Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf`  
**Supports:** `stage_4_model_architecture` — SUPPORTED  
**Key evidence:** 99.93% accuracy on H-Activity dataset, 98.76% on MHEALTH, 93.11% on UCI-HAR
using a CNN-LSTM architecture — establishes the 1D-CNN-BiLSTM as a state-of-the-art HAR backbone.

---

### 2. Evaluating BiLSTM and CNN+GRU Approaches for HAR Using WiFi CSI Data
**File:** `Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf`  
**Supports:** `stage_4_model_architecture` — SUPPORTED  
**Key evidence:** Direct comparison of BiLSTM vs CNN+GRU on HAR tasks, confirming BiLSTM as the
stronger baseline architecture for temporal activity classification.

---

### 3. Domain Adaptation for Inertial Measurement Unit-based Human (Activity Recognition)
**File:** `Domain Adaptation for Inertial Measurement Unit-based Human.pdf`  
**Supports:** `stage_8_adabn`, `stage_8_adabn_tent_twostage` — PARTIAL  
**Key evidence:** "The batch normalization technique standardizes the batch input in the deep learning
network training process, which subsequently helps the network converge faster and reduces the
internal covariate shift between source and target domains."  
**Missing for full support:** Li et al. (2016) AdaBN primary — arXiv:1603.04779

---

### 4. Transfer Learning in Human Activity Recognition: A Survey (ACM 2018)
**File:** `Transfer Learning in Human Activity Recognition  A Survey.pdf`  
**Supports:** `stage_8_adabn`, `stage_8_tent`, `stage_8_pseudo_label`, `stage_13_ewc` — PARTIAL  
**Key evidence:**
- Pseudo-labeling: "using similarity calculations, instance transfer and pseudo labeling can be performed"
- Continual update need: "the underlying HAR framework needs to be continually updated to keep up
  with changing environments" (motivates both TENT test-time adaptation and EWC regularisation)  
**Missing for full support:** TENT primary (OpenReview:uXl3bZLkr3c) and EWC primary (arXiv:1612.00796)

---

### 5. When Does Optimizing a Proper Loss Yield Calibration?
**File:** `When Does Optimizing a Proper Loss Yield Calibration.pdf`  
**Supports:** `stage_11_temperature_scaling` — PARTIAL  
**Key evidence:** "temperature scaling as in Guo et al. (2017). Google's data science practitioner's
guide recommends that, for recalibration methods, the calibration set should be representative
of deployment conditions."  
**Missing for full support:** Guo et al. (2017) primary — arXiv:1706.04599

---

### 6. MACHINE LEARNING OPERATIONS: A SURVEY ON MLOPS
**File:** `MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf`  
**Supports:** `mlops_pipeline_orchestration` — SUPPORTED  
**Key evidence:**
- "MLOps incorporates ML models for solution development and maintenance with continuous integration
  to provide efficient and reliable service."
- "Continuous Integration/Continuous Delivery (CI/CD) pipeline aims to produce software effectively
  and efficiently and supports software evolution."

---

### 7. Deep learning for sensor-based activity recognition: A survey
**File:** `Deep learning for sensor-based activity recognition_ A survey.pdf`  
**Supports:** `stage_6_layer3_zscore_drift` — PARTIAL  
**Key evidence:** Activities have varied time spans and signal distributions; CNN+generative model
combinations perform distribution alignment — establishing the need for per-channel statistical
distribution monitoring in HAR pipelines.

---

### 8. Li et al. (2016) — Revisiting Batch Normalization For Practical Domain Adaptation
**File:** `adabn_li2016_1603.04779.pdf`  
**arXiv:** 1603.04779  
**Supports:** `stage_8_adabn`, `stage_8_adabn_tent_twostage` — **SUPPORTED**  
**Key evidence:** Primary source for AdaBN. Shows that replacing source-domain BN statistics with
target-domain statistics at inference time reduces domain shift without any labelled target data.

---

### 9. Wang et al. (2021) — Tent: Fully Test-Time Adaptation by Entropy Minimization
**File:** `tent_wang2021_openreview_uXl3bZLkr3c.pdf`  
**arXiv:** 2006.10726 (ICLR 2021)  
**Supports:** `stage_8_tent`, `stage_8_adabn_tent_twostage` — **SUPPORTED**  
**Key evidence:** Primary source for TENT. Fine-tunes only BN affine parameters (γ, β) to minimise
prediction entropy on the target domain without any labelled data.

---

### 10. Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks
**File:** `ewc_kirkpatrick2017_1612.00796.pdf`  
**arXiv:** 1612.00796 (PNAS 2017)  
**Supports:** `stage_13_ewc`, `stage_13_curriculum_pseudo_labeling` — **SUPPORTED**  
**Key evidence:** Primary source for Elastic Weight Consolidation. Introduces the Fisher-information
penalty on weights important for previous tasks (lambda=1000 in our pipeline).

---

### 11. Guo et al. (2017) — On Calibration of Modern Neural Networks
**File:** `calibration_guo2017_1706.04599.pdf`  
**arXiv:** 1706.04599 (ICML 2017)  
**Supports:** `stage_11_temperature_scaling` — **SUPPORTED**  
**Key evidence:** Primary source for temperature scaling as a post-hoc calibration method.
Shows modern deep nets are miscalibrated and that scaling logits by scalar T minimises NLL.

---

### 12. Gal & Ghahramani (2016) — Dropout as a Bayesian Approximation
**File:** `mc_dropout_gal2016_1506.02142.pdf`  
**arXiv:** 1506.02142 (ICML 2016)  
**Supports:** `stage_11_mc_dropout` — **SUPPORTED**  
**Key evidence:** Primary source for Monte Carlo Dropout. Shows that keeping Dropout active at
inference time and averaging N forward passes approximates a Bayesian deep Gaussian process.

---

## Missing Foundational Papers (download and add here)

**All 5 papers have been downloaded.** Run `python scripts/fetch_foundation_papers.py` to re-download if missing.

| Paper | Filename | Pipeline Claim |
|---|---|---|
| Li et al. (2016) — Revisiting Batch Normalization For Practical Domain Adaptation | `adabn_li2016_1603.04779.pdf` | `stage_8_adabn` → **SUPPORTED** |
| Wang et al. (2021) — Tent: Fully Test-Time Adaptation by Entropy Minimization | `tent_wang2021_openreview_uXl3bZLkr3c.pdf` | `stage_8_tent` → **SUPPORTED** |
| Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks | `ewc_kirkpatrick2017_1612.00796.pdf` | `stage_13_ewc` → **SUPPORTED** |
| Guo et al. (2017) — On Calibration of Modern Neural Networks | `calibration_guo2017_1706.04599.pdf` | `stage_11_temperature_scaling` → **SUPPORTED** |
| Gal & Ghahramani (2016) — Dropout as a Bayesian Approximation | `mc_dropout_gal2016_1506.02142.pdf` | `stage_11_mc_dropout` → **SUPPORTED** |

---

## Support Map Summary (as of 2026-02-27)

| Tag | Count | Notes |
|---|---|---|
| SUPPORTED | 9 | PDF evidence found with matching quote in extracted text |
| EMPIRICAL_CALIBRATION | 9 | Backed by scripts + CSV/PNG in reports/ |
| SENSOR_SPEC | 1 | IMU hardware constraint |
| PROJECT_DECISION | 4 | Architecture choice; ablation accompanies |
| **UNSUPPORTED** | **0** | **None** |

Full details: see `reports/EVIDENCE_PACK_INDEX.md` and `reports/PAPER_SUPPORT_MAP.json`.

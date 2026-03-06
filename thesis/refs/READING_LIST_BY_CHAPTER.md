# Reading List by Chapter

> Generated: 2026-03-05 | Purpose: thesis writing guide — which paper to read for which chapter.
> All paths are relative to repo root.
> Papers in `thesis/refs/` are the PRIMARY citation files (already added to `thesis.bib`).
> Papers in `archive/` are secondary/supporting references.

---

## Chapter 1 — Introduction

**What you need to cite here:** Motivation for HAR + wearables, clinical/health relevance, why MLOps matters for real-world deployment, research gap.

| Paper | Cite For | Path |
|-------|----------|------|
| `MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | Define MLOps, motivate the pipeline approach | `archive/papers/by_topic/mlops_reproducibility/` |
| `Research Roadmap_ Developing a Scalable MLOps Pipeline for Continuous Mental Health Monitoring.pdf` | Prior similar work — position your contribution | `archive/papers/by_topic/mlops_reproducibility/` |
| `Roadmap for a Scalable MLOps Pipeline in Mental Health Monitoring (Master's Thesis).pdf` | Related master's thesis — gap statement | `archive/papers/by_topic/mlops_reproducibility/` |
| `The Role of MLOps in Healthcare Enhancing Predictive Analytics and Patient Outcomes.pdf` | Why MLOps matters in healthcare — Ch1 motivation | `archive/papers/by_topic/mlops_reproducibility/` |
| `A Survey on Wearable Sensors for Mental Health Monitoring.pdf` | Wearable sensing for mental health context | `archive/papers/by_topic/misc/` |
| `A_Survey_on_Human_Activity_Recognition_using_Wearable_Sensors.pdf` | HAR with wearables — broad motivation | `archive/papers/by_topic/har_models/` |
| `MLHOps Machine Learning for Healthcare Operations.pdf` | Healthcare operations MLOps gap | `archive/papers/by_topic/mlops_reproducibility/` |

---

## Chapter 2 — Related Work (Background & Literature Review)

**What you need to cite here:** HAR model architectures, IMU sensors, wearable data challenges, MLOps frameworks, domain adaptation surveys, drift detection background.

### 2.1 HAR Architectures & Wearable Sensing

| Paper | Cite For | Path |
|-------|----------|------|
| `Deep learning for sensor-based activity recognition_ A survey.pdf` | **Must-cite** seminal survey on DL for sensor-based HAR | `archive/papers/papers needs to read/` |
| `CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf` | Architecture comparison — contextualizes BiLSTM | `archive/papers/papers needs to read/` |
| `Deep Learning in Human Activity Recognition with Wearable Sensors.pdf` | DL-for-HAR overview | `archive/papers/papers needs to read/` |
| `A Close Look into Human Activity Recognition Models using Deep Learning.pdf` | Model comparison — supports architecture choice | `archive/papers/papers needs to read/` |
| `Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf` | Attention-enhanced CNN+LSTM for HAR | `archive/papers/papers needs to read/` |
| `Wearable Sensor-Based Human Activity Recognition Using Hybrid Deep Learning Techniques 2020.pdf` | Hybrid DL for HAR, 2020 baseline | `archive/papers/papers needs to read/` |
| `Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf` | IMU sensor data analysis | `archive/papers/papers needs to read/` |
| `Improved_Deep_Representation_Learning_for_Human_Activity_Recognition_using_IMU_Sensors.pdf` | Deep representations from IMU | `archive/papers/by_topic/har_models/` |
| `Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf` | CNN+LSTM hybrid baseline comparison | `archive/papers/papers needs to read/` |
| `Implications on Human Activity Recognition Research.pdf` | Open problems / limitations in HAR | `archive/papers/papers needs to read/` |
| `Evaluating BiLSTM and CNN+GRU Approaches for HAR Using WiFi CSI Data.pdf` | BiLSTM evaluation reference | `archive/papers/papers needs to read/` |
| `Transfer Learning in Human Activity Recognition  A Survey.pdf` | TL/DA survey for HAR | `archive/papers/by_topic/domain_adaptation_tta/` |

### 2.2 MLOps & Pipeline Background

| Paper | Cite For | Path |
|-------|----------|------|
| `MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` | MLOps definitions, lifecycle, CI/CD | `archive/papers/by_topic/mlops_reproducibility/` |
| `Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf` | Tool selection framework | `archive/papers/by_topic/mlops_reproducibility/` |
| `Practical-mlops-operationalizing-machine-learning-models.pdf` | MLOps book — comprehensive background | `archive/papers/by_topic/mlops_reproducibility/` |
| `MLOps A Step Forward to Enterprise Machine Learning 2023.pdf` | MLOps maturity | `archive/papers/by_topic/mlops_reproducibility/` |
| `Machine Learning Operations in Health Care A Scoping Review.pdf` | Healthcare MLOps scoping review | `archive/papers/papers needs to read/` |
| `Enabling End-To-End Machine Learning.pdf` | End-to-end ML pipelines | `archive/papers/papers needs to read/` |

### 2.3 Domain Adaptation Background

| Paper | Cite For | Path |
|-------|----------|------|
| `Transfer Learning in Human Activity Recognition  A Survey.pdf` | TL survey for HAR | `archive/papers/by_topic/domain_adaptation_tta/` |
| `Domain Adaptation for Inertial Measurement Unit-based Human.pdf` | DA for IMU-based HAR | `archive/papers/by_topic/domain_adaptation_tta/` |
| `thesis/refs/adabn_li2016_1603.04779.pdf` | AdaBN — batch norm-based domain adaptation | `thesis/refs/` |

### 2.4 Wearable Sensor Data & Datasets

| Paper | Cite For | Path |
|-------|----------|------|
| `Wearable, Environmental, and Smartphone-Based Passive Sensing for Mental Health Monitoring 2021.pdf` | Multi-modal wearable monitoring | `archive/papers/papers needs to read/` |
| `Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf` | Sensor fusion context | `archive/papers/papers needs to read/` |
| `Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf` | Wrist sensor displacement effects | `archive/papers/by_topic/har_models/` |

---

## Chapter 3 — Methodology

**What you need to cite here:** Pipeline design decisions, preprocessing choices, model architecture justification, domain adaptation methods (AdaBN, TENT, EWC), drift detection (WATCH/Wasserstein), calibration (temperature scaling).

### 3.1 Architecture Justification

| Paper | Cite For | Path |
|-------|----------|------|
| `thesis/refs/adabn_li2016_1603.04779.pdf` | AdaBN — normalizing BN for domain shift | `thesis/refs/` |
| `thesis/refs/tent_wang2021_openreview_uXl3bZLkr3c.pdf` | TENT — test-time entropy minimization | `thesis/refs/` |
| `thesis/refs/ewc_kirkpatrick2017_1612.00796.pdf` | EWC — preventing catastrophic forgetting | `thesis/refs/` |
| `2010.03759v4.pdf` (TENT arxiv) | Entropy min. TTA, primary method | `archive/papers/by_topic/domain_adaptation_tta/` |
| `Scaling Human Activity Recognition via Deep Learning-based Domain Adaptation.pdf` | DA scale + justification | `archive/papers/by_topic/domain_adaptation_tta/` |
| `Domain Adaptation for Inertial Measurement Unit-based Human.pdf` | IMU-specific DA techniques | `archive/papers/by_topic/domain_adaptation_tta/` |
| `Tutorial on time series prediction using 1D-CNN and BiLSTM.pdf` | BiLSTM for time series — architecture background | `archive/papers/papers needs to read/` |
| `Pre-trained 1DCNN-BiLSTM Hybrid Network for Temperature Prediction...pdf` | Pre-trained 1DCNN-BiLSTM pattern | `archive/papers/papers needs to read/` |

### 3.2 Calibration & Uncertainty

| Paper | Cite For | Path |
|-------|----------|------|
| `thesis/refs/calibration_guo2017_1706.04599.pdf` | Temperature scaling calibration | `thesis/refs/` |
| `thesis/refs/mc_dropout_gal2016_1506.02142.pdf` | MC-Dropout for uncertainty | `thesis/refs/` |
| `1706.04599v2.pdf` | Temperature scaling (Guo 2017) | `archive/papers/by_topic/evaluation_metrics/` |
| `Personalizing_Activity_Recognition_Models_Through_Quantifying_Different_Types_of_Uncertainty_Using_Wearable_Sensors.pdf` | Uncertainty types in HAR | `archive/papers/by_topic/har_models/` |

### 3.3 Drift Detection Methods

| Paper | Cite For | Path |
|-------|----------|------|
| `WATCH_Wasserstein_Change_Point_Detection_for_High-Dimensional_Time_Series_Data.pdf` | **Primary drift method** | `archive/papers/by_topic/drift_detection/` |
| `LIFEWATCH_Lifelong_Wasserstein_Change_Point_Detection.pdf` | Lifelong variant of WATCH | `archive/papers/by_topic/drift_detection/` |
| `Optimal_Transport_Based_Change_Point_Detection_and_Time_Series_Segment_Clustering.pdf` | Optimal transport foundation | `archive/papers/by_topic/drift_detection/` |
| `Normalizing_Self-Supervised_Learning_for_Provably_Reliable_Change_Point_Detection.pdf` | SSL change point detection | `archive/papers/by_topic/drift_detection/` |

### 3.4 Pipeline & Preprocessing Design

| Paper | Cite For | Path |
|-------|----------|------|
| `AutoMR- A Universal Time Series Motion Recognition Pipeline.pdf` | End-to-end motion recognition pipeline | `archive/papers/papers needs to read/` |
| `From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf` | MLOps monitoring design | `archive/papers/by_topic/mlops_reproducibility/` |
| `Reproducible workflow for online AI in digital health.pdf` | Reproducible workflow justification | `archive/papers/papers needs to read/` |
| `Lifelong_Learning_in_Sensor-Based_Human_Activity_Recognition.pdf` | Lifelong learning for HAR | `archive/papers/by_topic/domain_adaptation_tta/` |

---

## Chapter 4 — Pipeline Implementation

**What you need to cite here:** Specific tool choices (DVC, MLflow, Docker, FastAPI, Prometheus/Grafana, GitHub Actions), containerization, CI/CD for ML.

| Paper | Cite For | Path |
|-------|----------|------|
| `Building-Scalable-MLOps-Optimizing-Machine-Learning-Deployment-and-Operations.pdf` | Scalable MLOps deployment patterns | `archive/papers/by_topic/mlops_reproducibility/` |
| `Essential_MLOps_Data_Science_Horizons_2023_Data_Science_Horizons_Final_2023.pdf` | MLOps toolchain overview | `archive/papers/by_topic/mlops_reproducibility/` |
| `MLOps and LLMOps with Python_ A Comprehensive Guide...pdf` | MLflow, DVC, Docker implementation | `archive/papers/by_topic/mlops_reproducibility/` |
| `Atechnical framework for deploying custom real-time machine 2023.pdf` | Real-time ML deployment framework | `archive/papers/papers needs to read/` |
| `DevOps-Driven Real-Time Health Analytics.pdf` | Prometheus/Grafana for health analytics DevOps | `archive/papers/papers needs to read/` |
| `Developing a Scalable MLOps Pipeline for Continuou.pdf` | Continuous deployment pipeline architecture | `archive/papers/by_topic/mlops_reproducibility/` |
| `Resilience-aware MLOps for AI-based medical diagnostic system 2024.pdf` | Resilience patterns (retry, fallback) | `archive/papers/by_topic/mlops_reproducibility/` |

---

## Chapter 5 — Experiments & Results

**What you need to cite here:** Evaluation methodology, metrics (F1, ECE, Brier, Wasserstein), calibration results, drift detection evaluation, threshold selection.

| Paper | Cite For | Path |
|-------|----------|------|
| `thesis/refs/calibration_guo2017_1706.04599.pdf` | ECE metric + temperature scaling | `thesis/refs/` |
| `thesis/refs/mc_dropout_gal2016_1506.02142.pdf` | MC-Dropout baseline uncertainty | `thesis/refs/` |
| `When Does Optimizing a Proper Loss Yield Calibration.pdf` | When cross-entropy training calibrates | `archive/papers/papers needs to read/` |
| `NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf` | Energy-based OOD scoring | `archive/papers/by_topic/evaluation_metrics/` |
| `NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf` | Adaptive conformal prediction under shift | `archive/papers/by_topic/drift_detection/` |
| `WATCH_Wasserstein_Change_Point_Detection_for_High-Dimensional_Time_Series_Data.pdf` | Wasserstein drift detection evaluation | `archive/papers/by_topic/drift_detection/` |
| `Evaluating BiLSTM and CNN+GRU Approaches for HAR Using WiFi CSI Data.pdf` | BiLSTM baseline comparison | `archive/papers/papers needs to read/` |
| `XAI-BayesHAR_A_novel_Framework_for_Human_Activity_Recognition_with_Integrated_Uncertainty_and_Shapely_Values.pdf` | Uncertainty-aware HAR evaluation | `archive/papers/by_topic/har_models/` |

---

## Chapter 6 — Discussion, Limitations & Threats to Validity

**What you need to cite here:** Generalizability limitations, sensor variability, class imbalance, comparison with related work.

| Paper | Cite For | Path |
|-------|----------|------|
| `Implications on Human Activity Recognition Research.pdf` | HAR limitations and open problems | `archive/papers/papers needs to read/` |
| `Enhancing_Human_Activity_Recognition_in_Wrist-Worn_Sensor_Data_Through_Compensation_Strategies_for_Sensor_Displacement.pdf` | Wrist sensor placement threat to generalizability | `archive/papers/by_topic/har_models/` |
| `Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf` | Cross-population generalization challenges | `archive/papers/by_topic/misc/` |
| `Toward Reusable Science with Readable Code and.pdf` | Reproducibility / open science framing | `archive/papers/papers needs to read/` |
| `Reproducible workflow for online AI in digital health.pdf` | Reproducibility validation | `archive/papers/papers needs to read/` |

---

## Chapter 7 — Conclusion & Future Work

**What you need to cite here:** Future directions (federated learning, active learning, foundation models for wearables, LLM integration).

| Paper | Cite For | Path |
|-------|----------|------|
| `Lifelong_Learning_in_Sensor-Based_Human_Activity_Recognition.pdf` | Lifelong/continual learning for HAR | `archive/papers/by_topic/domain_adaptation_tta/` |
| `Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf` | Foundation model for wearables — future direction | `archive/papers/papers needs to read/` |
| `Learning the Language of wearable sensors.pdf` | LLM-like foundation models for sensors | `archive/papers/papers needs to read/` |
| `activear.pdf` | Active HAR — future annotation strategy | `archive/papers/by_topic/har_models/` |
| `Exploring the Capabilities of LLMs for IMU-based Fine-grained.pdf` | LLMs for IMU recognition — future | `archive/papers/papers needs to read/` |
| `A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning.pdf` | Federated learning for privacy-preserving HAR | `archive/papers/papers needs to read/` |

---

## Quick Reference: Primary BibTeX Citations

These are already in `thesis/refs/thesis.bib` and `thesis/refs/parameter_citations.bib`:

| Key | File | Use |
|-----|------|-----|
| `guo2017calibration` | `calibration_guo2017_1706.04599.pdf` | Temperature scaling |
| `gal2016dropout` | `mc_dropout_gal2016_1506.02142.pdf` | MC-Dropout uncertainty |
| `kirkpatrick2017overcoming` | `ewc_kirkpatrick2017_1612.00796.pdf` | EWC catastrophic forgetting |
| `li2016revisiting` | `adabn_li2016_1603.04779.pdf` | AdaBN domain adaptation |
| `wang2021tent` | `tent_wang2021_openreview_uXl3bZLkr3c.pdf` | TENT test-time adaptation |
| (add WATCH, LIFEWATCH) | `draft/WATCH_...pdf` | Wasserstein drift |

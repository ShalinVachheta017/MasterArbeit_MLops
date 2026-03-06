# Reading List by Thesis Chapter

> Maps papers from `thesis/refs/papers_all/` to thesis chapters.
> Use this to find the right citations while writing each chapter.

---

## Chapter 1 — Introduction & Motivation

| Paper | Folder | Why |
|-------|--------|-----|
| A Survey on Wearable Sensors for Mental Health Monitoring | 06_mental_health_anxiety | Motivates wearable HAR for mental health |
| ADAM-sense: Anxiety-displaying activities recognition | 06_mental_health_anxiety | Anxiety detection via activity recognition |
| MLHOps Machine Learning for Healthcare Operations | 01_mlops | Healthcare MLOps motivation |
| The Role of MLOps in Healthcare (in by_topic) | 01_mlops | Why MLOps matters in clinical settings |

---

## Chapter 2 — Related Work / Literature Review

### HAR Models & Architectures
| Paper | Folder |
|-------|--------|
| Deep learning for sensor-based activity recognition: A survey | 03_har_wearables |
| Deep CNN-LSTM With Self-Attention Model for HAR | 03_har_wearables |
| CNNs, RNNs and Transformers in Human Action Recognition | 03_har_wearables |
| A Close Look into HAR Models using Deep Learning | 03_har_wearables |
| Deep Learning in Human Activity Recognition with Wearable Sensors | 03_har_wearables |
| Human Activity Recognition using Multi-Head CNN followed by LSTM | 03_har_wearables |
| Evaluating BiLSTM and CNN+GRU Approaches for HAR | 03_har_wearables |
| Implications on Human Activity Recognition Research | 03_har_wearables |

### MLOps Fundamentals
| Paper | Folder |
|-------|--------|
| MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS | 01_mlops |
| Enabling End-To-End Machine Learning | 01_mlops |
| MLHOps Machine Learning for Healthcare Operations | 01_mlops |

### Transfer Learning & Domain Adaptation
| Paper | Folder |
|-------|--------|
| Transfer Learning in Human Activity Recognition: A Survey | 04_domain_adaptation |
| Domain Adaptation for IMU-based Human Activity | 04_domain_adaptation |

---

## Chapter 3 — Methodology / Pipeline Design

### Pipeline Architecture
| Paper | Folder | Why |
|-------|--------|-----|
| AutoMR: A Universal Time Series Motion Recognition Pipeline | 03_har_wearables | End-to-end pipeline comparison |
| An End-to-End Deep Learning Pipeline for Football Activity Recognition | 03_har_wearables | Pipeline design pattern |
| Reproducible workflow for online AI in digital health | 01_mlops | Reproducibility framework |

### Sensor Data Processing
| Paper | Folder |
|-------|--------|
| Combining Accelerometer and Gyroscope Data in Smartphone-Based | 03_har_wearables |
| Analyzing Wearable Accelerometer and Gyroscope Data | 03_har_wearables |
| Spatiotemporal Feature Fusion | 03_har_wearables |

### Model Architecture Justification
| Paper | Folder |
|-------|--------|
| Deep CNN-LSTM With Self-Attention Model for HAR | 03_har_wearables |
| A Multi-Task Deep Learning Approach for Sensor-based HAR | 03_har_wearables |

### Domain Adaptation (AdaBN, TENT, EWC)
| Paper | Folder |
|-------|--------|
| Domain Adaptation for IMU-based Human Activity | 04_domain_adaptation |
| adabn_li2016 (in thesis/refs/) | — |
| tent_wang2021 (in thesis/refs/) | — |
| ewc_kirkpatrick2017 (in thesis/refs/) | — |

---

## Chapter 4 — Implementation

| Paper | Folder | Why |
|-------|--------|-----|
| MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND | 01_mlops | Experiment automation |
| Atechnical framework for deploying custom real-time ML | 05_deployment_cicd | Deployment framework |
| DevOps-Driven Real-Time Health Analytics | 01_mlops | DevOps/monitoring stack |
| Building Flexible, Scalable ML-ready Multimodal Datasets | 05_deployment_cicd | Data pipeline design |
| Machine Learning Applied to Edge Computing and Wearable Devices | 05_deployment_cicd | Edge deployment |

---

## Chapter 5 — Evaluation & Results

### Monitoring & Drift
| Paper | Folder |
|-------|--------|
| Comparative Study on the Effects of Noise | 02_monitoring_drift |
| Resilience of ML Models in Anxiety Detection (Gaussian Noise) | 02_monitoring_drift |
| When Does Optimizing a Proper Loss Yield Calibration | 02_monitoring_drift |

### Calibration & Uncertainty
| Paper | Folder |
|-------|--------|
| calibration_guo2017 (in thesis/refs/) | — |
| mc_dropout_gal2016 (in thesis/refs/) | — |

---

## Chapter 6 — Discussion

| Paper | Folder | Why |
|-------|--------|-----|
| Are Anxiety Detection Models Generalizable | 06_mental_health_anxiety | Generalization limitations |
| Implications on Human Activity Recognition Research | 03_har_wearables | Open problems |
| Toward Reusable Science with Readable Code | 05_deployment_cicd | Reproducibility |

---

## Chapter 7 — Future Work

| Paper | Folder | Why |
|-------|--------|-----|
| Toward Foundation Model for Multivariate Wearable Sensing | 03_har_wearables | Foundation models for wearables |
| Exploring the Capabilities of LLMs for IMU-based Fine-grained | 04_domain_adaptation | LLM + IMU future direction |
| Learning the Language of wearable sensors | 03_har_wearables | Wearable foundation models |
| LSM-2: Learning from Incomplete Wearable Sensor Data | 03_har_wearables | Incomplete data handling |
| Beyond Sensor Data: Foundation Models of Behavioral Data | 03_har_wearables | Behavioral foundation models |

---

## Primary References (Already in thesis/refs/)

These are the core algorithmic papers already placed in `thesis/refs/`:

| File | Topic | Chapters |
|------|-------|----------|
| adabn_li2016_1603.04779.pdf | AdaBN domain adaptation | Ch3 |
| tent_wang2021_openreview.pdf | TENT test-time adaptation | Ch3, Ch5 |
| ewc_kirkpatrick2017_1612.00796.pdf | EWC catastrophic forgetting | Ch3 |
| calibration_guo2017_1706.04599.pdf | Temperature scaling | Ch3, Ch5 |
| mc_dropout_gal2016_1506.02142.pdf | MC-Dropout uncertainty | Ch3, Ch5 |

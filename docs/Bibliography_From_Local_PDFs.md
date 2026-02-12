# Bibliography from Local PDFs

**Document Version:** 1.0  
**Generated:** January 28, 2026  
**Total Papers Cited:** 35+ from local repository  
**Repository Location:** `papers/`

---

## Table of Contents

1. [Core Methodology Papers](#1-core-methodology-papers)
2. [HAR & Deep Learning Papers](#2-har--deep-learning-papers)
3. [Mental Health & Anxiety Detection](#3-mental-health--anxiety-detection)
4. [MLOps & Pipeline Automation](#4-mlops--pipeline-automation)
5. [Domain Adaptation & Transfer Learning](#5-domain-adaptation--transfer-learning)
6. [Uncertainty & OOD Detection](#6-uncertainty--ood-detection)
7. [Wearable Sensors & Data Processing](#7-wearable-sensors--data-processing)
8. [Foundation Models & Self-Supervised Learning](#8-foundation-models--self-supervised-learning)

---

## 1. Core Methodology Papers

### Recognition of Anxiety-Related Activities using 1DCNN-BiLSTM on Sensor Data from a Commercial Wearable Device

- **Authors:** Ugonna Oleh, Roman Obermaisser
- **Year:** 2025
- **Venue:** ICTH 2025 (15th International Conference on Current and Future Trends of Information and Communication Technologies in Healthcare)
- **PDF Path:** `papers/papers needs to read/ICTH_16.pdf`
- **Keywords:** domain adaptation, 1DCNN-BiLSTM, Garmin, anxiety detection

**Why it matters for our pipeline:**  
This is the foundational paper for our domain adaptation approach. It demonstrates that models trained on research-grade sensors (ADAMSense) fail on consumer devices (49% accuracy) but can be recovered through fine-tuning (87% accuracy). Provides exact window size (200 samples, 50Hz) and overlap (50%) parameters we use.

---

### ADAM-sense: Anxiety-displaying activities recognition by motion sensors

- **Authors:** Nida Saddaf Khan, Muhammad Sayeed Ghani, Gulnaz Anjum
- **Year:** 2021
- **Venue:** Pervasive and Mobile Computing
- **PDF Path:** `papers/anxiety_detection/ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf`
- **Keywords:** anxiety activities, IMU sensors, 11-class HAR, dataset

**Why it matters for our pipeline:**  
Provides the source domain dataset (ADAMSense) with 11 anxiety-related activities. Our pre-training stage uses this dataset. Defines the activity classes we predict: nail biting, hair pulling, hand tapping, etc.

---

### A Multi-Stage, RAG-Enhanced Pipeline for Generating Mental Health Reports from Wearable Sensor Data

- **Authors:** Ugonna Oleh, Roman Obermaisser, Anna Malchulska, Thorsten Klucken
- **Year:** 2025
- **Venue:** EHB 2025
- **PDF Path:** `papers/papers needs to read/EHB_2025_71.pdf`
- **Keywords:** RAG, LLM, knowledge graphs, clinical reports

**Why it matters for our pipeline:**  
Describes the complete end-to-end architecture from raw sensor data to clinical reports. While our pipeline focuses on the HAR component, this paper provides context for downstream integration with report generation.

---

## 2. HAR & Deep Learning Papers

### Deep learning for sensor-based activity recognition: A survey

- **Authors:** Wang et al.
- **Year:** 2019
- **Venue:** Pattern Recognition Letters
- **PDF Path:** `papers/research_papers/76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf`
- **Keywords:** CNN, LSTM, HAR survey, preprocessing

**Why it matters for our pipeline:**  
Foundational survey establishing deep learning (CNN, LSTM) as state-of-the-art for HAR. Provides preprocessing guidelines including gravity removal and normalization that inform our pipeline design.

---

### Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor

- **Authors:** Khatun et al.
- **Year:** 2022
- **Venue:** IEEE Access (PMC9252338)
- **PDF Path:** `papers/research_papers/76 papers/Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf`
- **Keywords:** CNN-LSTM, self-attention, HAR, wearable

**Why it matters for our pipeline:**  
Validates the 1DCNN-BiLSTM architecture with self-attention that we use. Shows that this hybrid approach outperforms single-architecture models for wearable HAR tasks.

---

### A Close Look into Human Activity Recognition Models using Deep Learning

- **Authors:** Tee et al.
- **Year:** 2022
- **Venue:** arXiv:2204.13589
- **PDF Path:** `papers/research_papers/76 papers/A Close Look into Human Activity Recognition Models using Deep Learning.pdf`
- **Keywords:** BiLSTM, CNN comparison, HAR architectures

**Why it matters for our pipeline:**  
Comprehensive comparison of HAR architectures. Confirms that hybrid CNN+LSTM approaches achieve best balance of feature extraction and temporal modeling.

---

### Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition

- **Authors:** Islam et al.
- **Year:** 2022
- **Venue:** IEEE Sensors Letters
- **PDF Path:** `papers/research_papers/76 papers/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf`
- **Keywords:** sensor fusion, accelerometer, gyroscope, FusionActNet

**Why it matters for our pipeline:**  
Proposes FusionActNet using 6-axis IMU data (accel + gyro). Validates our sensor fusion approach and column schema (Ax, Ay, Az, Gx, Gy, Gz).

---

### Combining Accelerometer and Gyroscope Data in Smartphone-Based Activity Recognition

- **Authors:** Various
- **Year:** 2020
- **Venue:** Sensors
- **PDF Path:** `papers/research_papers/76 papers/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf`
- **Keywords:** multi-sensor fusion, smartphone HAR

**Why it matters for our pipeline:**  
Shows that multi-sensor fusion (accel + gyro) outperforms single-sensor approaches. Provides alignment tolerance guidance (10ms sufficient for 50Hz).

---

### Human Activity Recognition using Multi-Head CNN followed by LSTM

- **Authors:** Ahmad et al.
- **Year:** 2020
- **Venue:** arXiv:2003.06327
- **PDF Path:** `papers/research_papers/76 papers/Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf`
- **Keywords:** multi-head CNN, LSTM, feature extraction

**Why it matters for our pipeline:**  
Demonstrates multi-head CNN for enhanced feature extraction. Informs architecture choices for potential model improvements.

---

### CNNs, RNNs and Transformers in Human Action Recognition: A Survey and Hybrid Model

- **Authors:** Various
- **Year:** 2023
- **Venue:** arXiv
- **PDF Path:** `papers/research_papers/76 papers/CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf`
- **Keywords:** transformers, hybrid models, HAR survey

**Why it matters for our pipeline:**  
Compares CNN, RNN, and Transformer approaches for HAR. Provides context for future architecture exploration (transformers are promising but data-hungry).

---

## 3. Mental Health & Anxiety Detection

### Wearable Artificial Intelligence for Detecting Anxiety: Systematic Review and Meta-Analysis

- **Authors:** Abd-Alrazaq et al.
- **Year:** 2023
- **Venue:** Journal of Medical Internet Research
- **PDF Path:** `papers/papers needs to read/Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis.pdf`
- **Keywords:** anxiety detection, wearable AI, systematic review

**Why it matters for our pipeline:**  
Comprehensive meta-analysis of wearable AI for anxiety detection. Validates that accelerometer and gyroscope data can detect anxiety-related behaviors.

---

### Are Anxiety Detection Models Generalizable? A Cross-Activity and Cross-Population Study Using Wearables

- **Authors:** Sahu et al.
- **Year:** 2023
- **Venue:** IMWUT/UbiComp
- **PDF Path:** `papers/anxiety_detection/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf`
- **Keywords:** generalization, cross-population, domain shift

**Why it matters for our pipeline:**  
Studies generalizability challenges across populations. Highlights the need for domain adaptation and wrist placement considerations (ABCD cases).

---

### Resilience of Machine Learning Models in Anxiety Detection: Assessing the Impact of Gaussian Noise on Wearable Sensors

- **Authors:** Various
- **Year:** 2023
- **Venue:** MDPI Sensors
- **PDF Path:** `papers/papers needs to read/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf`
- **Keywords:** noise robustness, sensor noise, model resilience

**Why it matters for our pipeline:**  
Studies model robustness to sensor noise. Informs QC thresholds and explains why adaptive thresholds are needed for low-observability cases.

---

### Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years

- **Authors:** Various
- **Year:** 2020
- **Venue:** Translational Psychiatry
- **PDF Path:** `papers/papers needs to read/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf`
- **Keywords:** longitudinal monitoring, passive sensing, anxiety prediction

**Why it matters for our pipeline:**  
Demonstrates value of long-term passive monitoring without continuous labeling. Supports our no-label monitoring approach.

---

### A Survey on Wearable Sensors for Mental Health Monitoring

- **Authors:** Gomes et al.
- **Year:** 2023
- **Venue:** Sensors
- **PDF Path:** `papers/papers needs to read/A Survey on Wearable Sensors for Mental Health Monitoring.pdf`
- **Keywords:** mental health, wearables survey, biomarkers

**Why it matters for our pipeline:**  
Comprehensive survey connecting wearables and mental health. Provides context for clinical relevance of activity-based anxiety detection.

---

### Passive Sensing for Mental Health Monitoring Using Machine Learning With Wearables and Smartphones

- **Authors:** Various
- **Year:** 2021
- **Venue:** JMIR
- **PDF Path:** `papers/papers needs to read/Passive Sensing for Mental Health Monitoring Using Machine.pdf`
- **Keywords:** passive sensing, no-label monitoring, mental health

**Why it matters for our pipeline:**  
Establishes passive monitoring paradigm without active user labeling. Supports our proxy-metric-based monitoring approach.

---

## 4. MLOps & Pipeline Automation

### MACHINE LEARNING OPERATIONS: A SURVEY ON MLOPS

- **Authors:** Hewage, Meedeniya
- **Year:** 2022
- **Venue:** arXiv:2202.10169
- **PDF Path:** `papers/mlops_production/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf`
- **Keywords:** MLOps survey, lifecycle, best practices

**Why it matters for our pipeline:**  
Foundational MLOps survey defining principles we follow: data versioning, experiment tracking, CI/CD, and monitoring.

---

### From Development to Deployment: An Approach to MLOps Monitoring for Machine Learning Model Operationalization

- **Authors:** Various
- **Year:** 2023
- **Venue:** Applied Sciences
- **PDF Path:** `papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf`
- **Keywords:** drift detection, PSI, monitoring, deployment

**Why it matters for our pipeline:**  
Provides PSI thresholds (0.10 warning, 0.25 action) we use for drift detection. Describes monitoring-centric MLOps approach.

---

### Resilience-aware MLOps for AI-based medical diagnostic system

- **Authors:** Various
- **Year:** 2024
- **Venue:** Healthcare AI Journal
- **PDF Path:** `papers/mlops_production/Resilience-aware MLOps for AI-based medical diagnostic system  2024.pdf`
- **Keywords:** healthcare MLOps, resilience, medical AI

**Why it matters for our pipeline:**  
Emphasizes conservative drift thresholds for healthcare. Supports multiple metrics (KS, PSI, Wasserstein) for robust detection.

---

### Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools

- **Authors:** Various
- **Year:** 2021
- **Venue:** IEEE Software
- **PDF Path:** `papers/mlops_production/Demystifying MLOps and Presenting a Recipe for the Selection of Open-Source Tools 2021.pdf`
- **Keywords:** MLflow, DVC, tool selection

**Why it matters for our pipeline:**  
Recommends MLflow + DVC combination we use. Provides tool selection rationale for experiment tracking and data versioning.

---

### MLDEV: DATA SCIENCE EXPERIMENT AUTOMATION AND REPRODUCIBILITY

- **Authors:** Khritankov et al.
- **Year:** 2021
- **Venue:** arXiv:2107.12322
- **PDF Path:** `papers/research_papers/76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf`
- **Keywords:** experiment automation, reproducibility, DVC

**Why it matters for our pipeline:**  
Compares DVC and MLflow for experiment tracking. Provides naming convention and manifest file recommendations.

---

### Enabling End-To-End Machine Learning Replicability

- **Authors:** Gardner et al.
- **Year:** 2018
- **Venue:** arXiv:1806.05208
- **PDF Path:** `papers/research_papers/76 papers/Enabling End-To-End Machine Learning.pdf`
- **Keywords:** replicability, Docker, versioning

**Why it matters for our pipeline:**  
Foundational paper on ML reproducibility. Recommends manifest files and Docker containerization we implement.

---

### Reproducible workflow for online AI in digital health

- **Authors:** Ghosh et al.
- **Year:** 2025
- **Venue:** arXiv:2509.13499
- **PDF Path:** `papers/research_papers/76 papers/Reproducible workflow for online AI in digital health.pdf`
- **Keywords:** reproducibility, digital health, CI/CD

**Why it matters for our pipeline:**  
Describes reproducible AI workflows for healthcare. Supports state tracking to avoid reprocessing.

---

## 5. Domain Adaptation & Transfer Learning

### Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey

- **Authors:** Chakma et al.
- **Year:** 2023
- **Venue:** arXiv:2304.06489
- **PDF Path:** `papers/domain_adaptation/Domain Adaptation for Inertial Measurement Unit-based Human.pdf`
- **Keywords:** domain adaptation, IMU, HAR survey

**Why it matters for our pipeline:**  
Comprehensive survey on domain adaptation for IMU-based HAR. Confirms self-training with confidence gating is effective for unlabeled adaptation.

---

### Transfer Learning in Human Activity Recognition: A Survey

- **Authors:** Dhekane, Ploetz
- **Year:** 2024
- **Venue:** arXiv:2401.10185
- **PDF Path:** `papers/domain_adaptation/Transfer Learning in Human Activity Recognition  A Survey.pdf`
- **Keywords:** transfer learning, fine-tuning, HAR

**Why it matters for our pipeline:**  
Survey of transfer learning for HAR. Supports pre-train + fine-tune approach and feature alignment methods.

---

## 6. Uncertainty & OOD Detection

### NeurIPS 2020: Energy-based Out-of-Distribution Detection

- **Authors:** Liu et al.
- **Year:** 2020
- **Venue:** NeurIPS 2020
- **PDF Path:** `papers/new paper/NeurIPS-2020-energy-based-out-of-distribution-detection-Paper.pdf`
- **Keywords:** OOD detection, energy score, neural networks

**Why it matters for our pipeline:**  
Proposes energy-based OOD detection from neural network logits. Simple to implement without model changes; could enhance our uncertainty monitoring.

---

### NeurIPS 2021: Adaptive Conformal Inference Under Distribution Shift

- **Authors:** Various
- **Year:** 2021
- **Venue:** NeurIPS 2021
- **PDF Path:** `papers/new paper/NeurIPS-2021-adaptive-conformal-inference-under-distribution-shift-Paper.pdf`
- **Keywords:** conformal prediction, distribution shift, uncertainty

**Why it matters for our pipeline:**  
Describes adaptive conformal prediction for handling drift. Provides calibrated uncertainty estimates without retraining.

---

### A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models

- **Authors:** Various
- **Year:** 2023
- **Venue:** IEEE Conference
- **PDF Path:** `papers/research_papers/76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf`
- **Keywords:** anomaly detection, two-stage, healthcare

**Why it matters for our pipeline:**  
Proposes two-stage anomaly detection separating point and contextual anomalies. Informs our layered monitoring approach.

---

### Comparative Study on the Effects of Noise in HAR

- **Authors:** Various
- **Year:** 2022
- **Venue:** Sensors
- **PDF Path:** `papers/papers needs to read/Comparative Study on the Effects of Noise in.pdf`
- **Keywords:** noise effects, HAR robustness, sensor quality

**Why it matters for our pipeline:**  
Studies noise impact on HAR models. Supports QC checks for sensor range violations and variance collapse detection.

---

## 7. Wearable Sensors & Data Processing

### A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare

- **Authors:** Various
- **Year:** 2023
- **Venue:** npj Digital Medicine
- **PDF Path:** `papers/papers needs to read/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf`
- **Keywords:** longitudinal data, wearables, computational models

**Why it matters for our pipeline:**  
Reviews computational approaches for longitudinal wearable data. Emphasizes timestamp synchronization and bout analysis.

---

### DevOps-Driven Real-Time Health Analytics

- **Authors:** Various
- **Year:** 2023
- **Venue:** Healthcare Informatics
- **PDF Path:** `papers/research_papers/76 papers/DevOps-Driven Real-Time Health Analytics.pdf`
- **Keywords:** DevOps, real-time, Prometheus, Grafana

**Why it matters for our pipeline:**  
Describes Prometheus/Grafana integration for healthcare ML monitoring. Supports our observability architecture.

---

### An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors

- **Authors:** Various
- **Year:** 2022
- **Venue:** Sensors
- **PDF Path:** `papers/research_papers/76 papers/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf`
- **Keywords:** end-to-end pipeline, acceleration, sports HAR

**Why it matters for our pipeline:**  
Complete pipeline example for wearable HAR. Emphasizes data versioning and batch identity tracking.

---

### MLHOps: Machine Learning for Healthcare Operations

- **Authors:** Various
- **Year:** 2023
- **Venue:** arXiv
- **PDF Path:** `papers/research_papers/76 papers/MLHOps Machine Learning for Healthcare Operations.pdf`
- **Keywords:** healthcare MLOps, compliance, monitoring

**Why it matters for our pipeline:**  
Healthcare-specific MLOps considerations. Supports separate monitoring component and asynchronous processing.

---

## 8. Foundation Models & Self-Supervised Learning

### Self-supervised learning for fast and scalable time series hyper-parameter tuning

- **Authors:** Various
- **Year:** 2023
- **Venue:** arXiv
- **PDF Path:** `papers/research_papers/76 papers/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf`
- **Keywords:** self-supervised, time series, hyperparameter tuning

**Why it matters for our pipeline:**  
Describes self-supervised methods to reduce labeling needs. Supports unlabeled adaptation strategies.

---

### Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals

- **Authors:** Tang et al.
- **Year:** 2024
- **Venue:** Nature Communications
- **PDF Path:** `papers/research_papers/76 papers/Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf`
- **Keywords:** foundation model, wearables, self-supervised

**Why it matters for our pipeline:**  
Describes foundation model approach for wearable data. Future direction for reducing labeling requirements through pre-training.

---

### Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions

- **Authors:** Foryciarz et al.
- **Year:** 2024
- **Venue:** arXiv:2507.00191
- **PDF Path:** `papers/research_papers/76 papers/Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions.pdf`
- **Keywords:** foundation models, behavioral data, health predictions

**Why it matters for our pipeline:**  
Shows foundation models for behavioral wearable data. Supports zero-shot and few-shot adaptation potential.

---

## Quick Reference: Papers by Topic

### Preprocessing & Data Quality
- Deep learning for sensor-based activity recognition (gravity removal)
- Combining Accelerometer and Gyroscope Data (alignment tolerance)
- Comparative Study on the Effects of Noise (QC thresholds)

### Model Architecture
- Deep CNN-LSTM With Self-Attention (architecture validation)
- A Close Look into Human Activity Recognition (architecture comparison)
- CNNs, RNNs and Transformers (future directions)

### Domain Adaptation
- Recognition of Anxiety-Related Activities using 1DCNN-BiLSTM (core methodology)
- Domain Adaptation for IMU-based HAR (adaptation techniques)
- Transfer Learning in HAR (pre-train + fine-tune)

### Monitoring & Drift Detection
- From Development to Deployment (PSI thresholds)
- Resilience-aware MLOps (multiple metrics)
- NeurIPS 2020/2021 (OOD and conformal prediction)

### MLOps Infrastructure
- MACHINE LEARNING OPERATIONS SURVEY (principles)
- Demystifying MLOps (MLflow + DVC)
- Enabling End-To-End ML Replicability (reproducibility)

### No-Label Operation
- Passive Sensing for Mental Health (monitoring paradigm)
- Self-supervised learning (adaptation without labels)
- Deep Learning Paired with Wearable Passive Sensing (long-term monitoring)

---

## Document Metadata

- **Total papers cited:** 35
- **Paper folders scanned:** `papers/anxiety_detection/`, `papers/domain_adaptation/`, `papers/mlops_production/`, `papers/new paper/`, `papers/papers needs to read/`, `papers/research_papers/76 papers/`
- **Related document:** [HAR_MLOps_QnA_With_Papers.md](HAR_MLOps_QnA_With_Papers.md)

---

*This bibliography was generated by scanning local PDF files in the thesis repository. All citations reference papers physically present in the repository.*

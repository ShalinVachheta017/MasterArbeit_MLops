# 76 Papers ? Summarizzation
**Generated:** 2025-12-13

This file summarizes the PDFs in `research_papers/76 papers/` and highlights how each can inform our HAR + anxiety + MLOps pipeline.

**Method note:** Abstract/conclusion excerpts are automatically extracted from PDFs (text quality varies by paper). For implementation decisions, verify against the full PDF.

## Index Table
| # | Paper | Tags | Primary Hook |
|---:|---|---|---|
| 1 | [A Close Look into Human Activity Recognition Models using Deep Learning](<76 papers/A Close Look into Human Activity Recognition Models using Deep Learning.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 2 | [A DIFFUSION MODEL FOR MULTIVARIATE](<76 papers/A DIFFUSION MODEL FOR MULTIVARIATE.pdf>) | Synthetic/Generative | Explore synthetic augmentation (e.g., diffusion) to address label scarcity and privacy; validate via downstream task gains. |
| 3 | [A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation](<76 papers/A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 4 | [A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation](<76 papers/A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation.pdf>) | Privacy/Federated | Consider privacy-preserving training (federated/secure aggregation) and data governance as first-class pipeline concerns. |
| 5 | [A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare](<76 papers/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf>) | Wearables/Longitudinal | Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts. |
| 6 | [A Survey on Wearable Sensors for Mental Health Monitoring](<76 papers/A Survey on Wearable Sensors for Mental Health Monitoring.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 7 | [A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models](<76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf>) | Anomaly Detection | Consider anomaly/OOD detection for monitoring, fallbacks, and ?unknown activity? handling in production. |
| 8 | [A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models](<76 papers/A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models.pdf>) | MLOps/Reproducibility, HAR/DL, HPO/AutoML | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 9 | [A Visual Data and Detection Pipeline for Wearable Industrial Assistants](<76 papers/A Visual Data and Detection Pipeline for Wearable Industrial Assistants.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 10 | [ADAM-sense_Anxietydisplayingactivitiesrecognitionby](<76 papers/ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 11 | [An AI-native Runtime for Multi-Wearable Environments](<76 papers/An AI-native Runtime for Multi-Wearable Environments.pdf>) | Edge/Runtime/IoT | Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups. |
| 12 | [An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors](<76 papers/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf>) | MLOps/Reproducibility, HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 13 | [Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition](<76 papers/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 14 | [Anxiety Detection Leveraging Mobile Passive Sensing](<76 papers/Anxiety Detection Leveraging Mobile Passive Sensing.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 15 | [Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables](<76 papers/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf>) | Domain Adaptation/Transfer, Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 16 | [Atechnical framework for deploying custom real-time machine](<76 papers/Atechnical framework for deploying custom real-time machine.pdf>) | Edge/Runtime/IoT | Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups. |
| 17 | [AutoMR- A Universal Time Series Motion Recognition Pipeline](<76 papers/AutoMR- A Universal Time Series Motion Recognition Pipeline.pdf>) | MLOps/Reproducibility, HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 18 | [Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions](<76 papers/Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions.pdf>) | Foundation/SSL | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 19 | [Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets](<76 papers/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf>) | Data Engineering | Use dataset engineering best practices (schema, provenance, splits, documentation) for scalable multi-modal expansion. |
| 20 | [CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model](<76 papers/CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 21 | [Combining Accelerometer and Gyroscope Data in Smartphone-Based](<76 papers/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 22 | [Comparative Study on the Effects of Noise in](<76 papers/Comparative Study on the Effects of Noise in.pdf>) | Robustness/Noise | Add robustness evaluation (noise, missingness) and sensor-quality monitoring to reduce brittle real-world performance. |
| 23 | [Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor](<76 papers/Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 24 | [Deep CNN-LSTM With Self-Attention Model for](<76 papers/Deep CNN-LSTM With Self-Attention Model for.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 25 | [Deep learning for sensor-based activity recognition_ A survey](<76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 26 | [Deep Learning in Human Activity Recognition with Wearable Sensors](<76 papers/Deep Learning in Human Activity Recognition with Wearable Sensors.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 27 | [Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years](<76 papers/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 28 | [Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study](<76 papers/Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study.pdf>) | Wearables/Longitudinal, Clinical UX/Dashboard | Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts. |
| 29 | [Development and Testing of Retrieval Augmented Generation in Large Language Models](<76 papers/Development and Testing of Retrieval Augmented Generation in Large Language Models.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 30 | [Development of a two-stage depression symptom detection model](<76 papers/Development of a two-stage depression symptom detection model.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 31 | [DevOps-Driven Real-Time Health Analytics](<76 papers/DevOps-Driven Real-Time Health Analytics.pdf>) | MLOps/Reproducibility, Edge/Runtime/IoT | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 32 | [Domain Adaptation for Inertial Measurement Unit-based Human](<76 papers/Domain Adaptation for Inertial Measurement Unit-based Human.pdf>) | Domain Adaptation/Transfer | Add explicit cross-device/cross-user evaluation and a fine-tuning workflow to close the lab-to-life gap. |
| 33 | [Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors](<76 papers/Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors.pdf>) | HAR/DL, Edge/Runtime/IoT | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 34 | [Enabling End-To-End Machine Learning](<76 papers/Enabling End-To-End Machine Learning.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 35 | [Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy](<76 papers/Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 36 | [Enhancing Multimodal Electronic Health Records](<76 papers/Enhancing Multimodal Electronic Health Records.pdf>) | EHR/Clinical Data | Align wearable features with EHR-style outcomes/labels and leverage established clinical dataset practices. |
| 37 | [Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot](<76 papers/Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot.pdf>) | RAG/LLM, Edge/Runtime/IoT | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 38 | [Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data](<76 papers/Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 39 | [Evaluating large language models on medical evidence summarization](<76 papers/Evaluating large language models on medical evidence summarization.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 40 | [Exploring the Capabilities of LLMs for IMU-based Fine-grained](<76 papers/Exploring the Capabilities of LLMs for IMU-based Fine-grained.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 41 | [Human Activity Recognition using Multi-Head CNN followed by LSTM](<76 papers/Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 42 | [Human Activity Recognition Using Tools of Convolutional Neural Networks](<76 papers/Human Activity Recognition Using Tools of Convolutional Neural Networks.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 43 | [I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses](<76 papers/I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.pdf>) | MLOps/Reproducibility, Privacy/Federated, Interoperability/FHIR | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 44 | [Implications on Human Activity Recognition Research](<76 papers/Implications on Human Activity Recognition Research.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 45 | [Learning the Language of wearable sensors](<76 papers/Learning the Language of wearable sensors.pdf>) | Foundation/SSL | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 46 | [Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises](<76 papers/Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises.pdf>) | EHR/Clinical Data, Data Engineering | Align wearable features with EHR-style outcomes/labels and leverage established clinical dataset practices. |
| 47 | [LLM Chatbot-Creation Approaches](<76 papers/LLM Chatbot-Creation Approaches.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 48 | [LSM-2-Learning from Incomplete Wearable Sensor Data](<76 papers/LSM-2-Learning from Incomplete Wearable Sensor Data.pdf>) | Foundation/SSL, Wearables/Longitudinal | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 49 | [Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature](<76 papers/Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature.pdf>) | Edge/Runtime/IoT | Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups. |
| 50 | [Machine Learning based Anxiety Detection using Physiological Signals and Context Features](<76 papers/Machine Learning based Anxiety Detection using Physiological Signals and Context Features.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 51 | [MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS](<76 papers/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 52 | [Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](<76 papers/Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition.pdf>) | Foundation/SSL, HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 53 | [Medical Graph RAG Towards Safe Medical Large Language Model via](<76 papers/Medical Graph RAG Towards Safe Medical Large Language Model via.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 54 | [MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND](<76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 55 | [MLHOps Machine Learning for Healthcare Operations](<76 papers/MLHOps Machine Learning for Healthcare Operations.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 56 | [Momentary Stressor Logging and Reflective Visualizations Implications for](<76 papers/Momentary Stressor Logging and Reflective Visualizations Implications for.pdf>) | Mental Health, Clinical UX/Dashboard | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 57 | [Multimodal Frame-Scoring Transformer for Video Summarization](<76 papers/Multimodal Frame-Scoring Transformer for Video Summarization.pdf>) | HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 58 | [Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework](<76 papers/Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework.pdf>) | RAG/LLM, HPO/AutoML | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 59 | [Panic Attack Prediction Using Wearable Devices and Machine](<76 papers/Panic Attack Prediction Using Wearable Devices and Machine.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 60 | [Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care](<76 papers/Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care.pdf>) | Mental Health, Clinical UX/Dashboard | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 61 | [Real-Time Stress Monitoring Detection and Management in College Students](<76 papers/Real-Time Stress Monitoring Detection and Management in College Students.pdf>) | Mental Health, Edge/Runtime/IoT | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 62 | [Reproducible workflow for online AI in digital health](<76 papers/Reproducible workflow for online AI in digital health.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 63 | [Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors](<76 papers/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf>) | Mental Health, Robustness/Noise | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 64 | [Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review](<76 papers/Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 65 | [Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster](<76 papers/Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster.pdf>) | RAG/LLM, Foundation/SSL | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 66 | [Scientific Evidence for Clinical Text Summarization Using Large](<76 papers/Scientific Evidence for Clinical Text Summarization Using Large.pdf>) | RAG/LLM | Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs. |
| 67 | [Self-supervised learning for fast and scalable time series hyper-parameter tuning](<76 papers/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf>) | Foundation/SSL, HPO/AutoML | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 68 | [Spatiotemporal Feature Fusion for](<76 papers/Spatiotemporal Feature Fusion for.pdf>) | HAR/DL, Edge/Runtime/IoT | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 69 | [Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals](<76 papers/Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf>) | Foundation/SSL | Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation. |
| 70 | [Toward Reusable Science with Readable Code and](<76 papers/Toward Reusable Science with Readable Code and.pdf>) | MLOps/Reproducibility | Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable. |
| 71 | [Transfer Learning in Human Activity Recognition  A Survey](<76 papers/Transfer Learning in Human Activity Recognition  A Survey.pdf>) | Domain Adaptation/Transfer, HAR/DL | Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable. |
| 72 | [Transforming Wearable Data into Personal Health](<76 papers/Transforming Wearable Data into Personal Health.pdf>) | Wearables/Longitudinal | Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts. |
| 73 | [Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study](<76 papers/Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 74 | [Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis](<76 papers/Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 75 | [Wearable Artificial Intelligence for Detecting Anxiety](<76 papers/Wearable Artificial Intelligence for Detecting Anxiety.pdf>) | Mental Health | Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation. |
| 76 | [When Does Optimizing a Proper Loss Yield Calibration](<76 papers/When Does Optimizing a Proper Loss Yield Calibration.pdf>) | Calibration/Uncertainty | Track calibration (ECE/Brier) and add post-hoc calibration (temperature scaling) for safer decision support. |

## Per-paper Notes
### 1. A Close Look into Human Activity Recognition Models using Deep Learning
- **PDF:** `<76 papers/A Close Look into Human Activity Recognition Models using Deep Learning.pdf>` (6 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** — Human activity recognition using deep learning techniques has become increasing popular because of its high effectivity with recognizing complex tasks, as well as being relatively low in costs compared to more traditional machine learning techniques. This paper surveys some state-of-the-art human activity recognition models that are based on deep learning architecture and has layers containing Convolution Neural Networks (CNN), Long Short-Term Memory (LSTM), or a mix of more than one type for a hybrid system.
- **Conclusion (excerpt):** can be drawn from the impressive results that models have obtained in recent years as deep learning methodologies continue to develop and show improvements
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 2. A DIFFUSION MODEL FOR MULTIVARIATE
- **PDF:** `<76 papers/A DIFFUSION MODEL FOR MULTIVARIATE.pdf>` (15 pages)
- **Tags:** Synthetic/Generative
- **Abstract (excerpt):** Kinematic sensors are often used to analyze movement behaviors in sports and daily activities due to their ease of use and lack of spatial restrictions, unlike video-based motion capturing systems. Still, the generation, and especially the labeling of motion data for specific activities can be time-consuming and costly.
- **Conclusion (excerpt):** In this paper we introduced IMUDiffusion, a diffusion model for inertial motion capturing systems. Based on the diffusion model architecture from the Computer Vision domain, we adapted their model to meet the requirements of generating high-quality sequences of human motion.
- **Pipeline hook:** Explore synthetic augmentation (e.g., diffusion) to address label scarcity and privacy; validate via downstream task gains.

### 3. A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation
- **PDF:** `<76 papers/A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation.pdf>` (14 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** —Sensor-based human activity segmentation and recognition are two important and challenging problems in many real-world applications and they have drawn increasing attention from the deep learning community in recent years. Most of the existing deep learning works were designed based on pre-segmented sensor streams and they have treated activity segmentation and recognition as two separate tasks.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 4. A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation
- **PDF:** `<76 papers/A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation.pdf>` (39 pages)
- **Tags:** Privacy/Federated
- **Abstract (excerpt):** The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time.
- **Pipeline hook:** Consider privacy-preserving training (federated/secure aggregation) and data governance as first-class pipeline concerns.

### 5. A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare
- **PDF:** `<76 papers/A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare.pdf>` (24 pages)
- **Tags:** Wearables/Longitudinal
- **Abstract (excerpt):** Wearable devices are increasingly used as tools for biomedi cal research, as the continuous stream of behavioral and physiological da ta they collect can provide insights about our health in everyday contexts. Long-term tracking, deﬁned in the timescale of months of year, can prov ide insights of patterns and changes as indicators of health changes.
- **Pipeline hook:** Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts.

### 6. A Survey on Wearable Sensors for Mental Health Monitoring
- **PDF:** `<76 papers/A Survey on Wearable Sensors for Mental Health Monitoring.pdf>` (17 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Mental illness, whether it is medically diagnosed or undiagnosed, affects a large proportion of the population. It is one of the causes of extensive disability , and f not properly treated, it can lead to severe emotional, behavioral, and physical health problems.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 7. A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models
- **PDF:** `<76 papers/A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.pdf>` (5 pages)
- **Tags:** Anomaly Detection
- **Abstract (excerpt):** —This paper presents an effective two-stage anomaly detection technique for wireless body area networks (WBANs)- based personalized healthcare monitoring. Our method uses SMO regression (Sequential Minimal Optimization) for more in-depth contextual analysis and Support Vector Machines (SVM) for initial anomaly categorization.
- **Conclusion (excerpt):** The two-stage anomaly detection methodology presented in this paper uses Support Vector Machines (SVM) and Support Vector Regression (SVR) to detect contextual and point anoma- lies in physiological data. The suggested method successfully separates physiologically significant patterns from sensor noise by combining machine learning approaches with domain- Fig.
- **Pipeline hook:** Consider anomaly/OOD detection for monitoring, fallbacks, and ?unknown activity? handling in production.

### 8. A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models
- **PDF:** `<76 papers/A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models.pdf>` (7 pages)
- **Tags:** MLOps/Reproducibility, HAR/DL, HPO/AutoML
- **Abstract (excerpt):** —Transformer-based models for time series forecasting (TSF) have attracted significant attention in recent years due to their effectiveness and versatility. However, these models often require extensive hyperparameter optimization (HPO) to achieve the best possible performance, and a unified pipeline for HPO in transformer-based TSF remains lacking.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 9. A Visual Data and Detection Pipeline for Wearable Industrial Assistants
- **PDF:** `<76 papers/A Visual Data and Detection Pipeline for Wearable Industrial Assistants.pdf>` (8 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** — Industrial assembly tasks increasingly demand rapid adaptation to complex procedures and varied compo- nents, yet are often conducted in environments with limited computing, connectivity, and strict privacy requirements. These constraints make conventional cloud-based or fully autonomous solutions impractical for factory deployment.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 10. ADAM-sense_Anxietydisplayingactivitiesrecognitionby
- **PDF:** `<76 papers/ADAM-sense_Anxietydisplayingactivitiesrecognitionby.pdf>` (13 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** PervasiveandMobileComputing78(2021)101485 Contents lists available at ScienceDirect PervasiveandMobileComputing journal homepage: www.elsevier.com/locate/pmc ADAM-sense:Anxiety-displayingactivitiesrecognitionby motionsensors NidaSaddafKhan a,∗,MuhammadSayeedGhani a,GulnazAnjum b,c aTelecommunication Research Lab (TRL), Institute of Business Administration Karachi, Plot # 68 & 88 Garden/ Kayani Shaheed Road, Karachi 74400, Pakistan bIntergroup Relations and Social Justice Lab, Department of Psychology, Simon Fraser University, Burnaby, Canada cDepartment of Social Sciences and Liberal Arts, Institute of Business Administration Karachi, Univers?
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 11. An AI-native Runtime for Multi-Wearable Environments
- **PDF:** `<76 papers/An AI-native Runtime for Multi-Wearable Environments.pdf>` (7 pages)
- **Tags:** Edge/Runtime/IoT
- **Abstract (excerpt):** The miniaturization of AI accelerators is paving the way for next-generation wearable applications within wearable technologies. We introduce Mojito, an AI-native runtime with advanced MLOps designed to facilitate the development and deployment of these applications on wearable devices.
- **Conclusion (excerpt):** We introduced exciting research challenges that MLOps need to address for next-generation wearable applications. Mojito highlighted the importance of dynamic and holistic orchestration of wearable devices.
- **Pipeline hook:** Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups.

### 12. An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors
- **PDF:** `<76 papers/An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors.pdf>` (28 pages)
- **Tags:** MLOps/Reproducibility, HAR/DL
- **Abstract (excerpt):** Action statistics in sports, such as the number of sprints and jumps, along with the details of the corresponding locomotor actions, are of high interest to coaches and players, as well as medical staff. Current video-based systems have the disadvantage that they are costly and not easily transportable to new locations.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 13. Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition
- **PDF:** `<76 papers/Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.pdf>` (6 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** A person’s movement or relative positioning can be effec- tively captured by different types of sensors and correspond- ing sensor output can be utilized in various manipulative techniques for the classification of different human activities. This letter proposes an effective scheme for human activity recognition, which introduces two unique approaches within a multi-structural architecture, named FusionActNet.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 14. Anxiety Detection Leveraging Mobile Passive Sensing
- **PDF:** `<76 papers/Anxiety Detection Leveraging Mobile Passive Sensing.pdf>` (5 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** —Anxiety disorders are the most common class of psychiatric problems affecting both children and adults. However, tools to effectively monitor and manage anxiety are lacking, and comparatively limited research has been applied to addressing the unique challenges around anxiety.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 15. Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables
- **PDF:** `<76 papers/Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables.pdf>` (32 pages)
- **Tags:** Domain Adaptation/Transfer, Mental Health
- **Abstract (excerpt):** Are Anxiety Detection Models Generalizable? A Cross-Activity and Cross-Population Study Using Wearables NILESH KUMAR SAHU, Indian Institute of Science Education and Research Bhopal (IISERB), India SNEHIL GUPTA, All India Institute of Medical Sciences Bhopal, India HAROON R LONE, Indian Institute of Science Education and Research Bhopal (IISERB), India Anxiety-provoking activities, such as public speaking, can trigger heightened anxiety responses in individuals with anxiety disorders.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 16. Atechnical framework for deploying custom real-time machine
- **PDF:** `<76 papers/Atechnical framework for deploying custom real-time machine.pdf>` (22 pages)
- **Tags:** Edge/Runtime/IoT
- **Abstract (excerpt):** DEPLOYR: A technical framework for deploying custom real-time machine learning models into the electronic medical record Conor K. Corbin∗,1, Rob Maclay∗,2, Aakash Acharya3, Sreedevi Mony3, Soumya Punnathanam3, Rahul Thapa3, Nikesh Kotecha, PhD3, Nigam H.
- **Pipeline hook:** Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups.

### 17. AutoMR- A Universal Time Series Motion Recognition Pipeline
- **PDF:** `<76 papers/AutoMR- A Universal Time Series Motion Recognition Pipeline.pdf>` (5 pages)
- **Tags:** MLOps/Reproducibility, HAR/DL
- **Abstract (excerpt):** — In this paper, we present an end-to-end automated motion recognition (AutoMR) pipeline designed for multimodal datasets. The proposed framework seamlessly integrates data preprocessing, model training, hyperparameter tuning, and evaluation, enabling robust performance across diverse scenar- ios.
- **Conclusion (excerpt):** The evaluation of AutoMR across ten datasets demon- strates its effectiveness in automating motion recognition by streamlining data preprocessing, model training, and hyperparameter tuning. AutoMR achieves state-of-the-art performance on eight datasets, highlighting its adaptability across different sensor modalities.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 18. Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions
- **PDF:** `<76 papers/Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions.pdf>` (26 pages)
- **Tags:** Foundation/SSL
- **Abstract (excerpt):** Wearable devices record physiological and behav- ioral signals that can improve health predictions. While foundation models are increasingly used for such predictions, they have been primarily applied to low-level sensor data, despite behavioral data often being more informative due to their align- ment with physiologically relevant timescales and quantities.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 19. Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets
- **PDF:** `<76 papers/Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.pdf>` (23 pages)
- **Tags:** Data Engineering
- **Abstract (excerpt):** The advancements in data acquisition, storage, and processing techniques have resulted in the rapid growth of heterogeneous medical data. Integrating radiological scans, histopathology images, and molecular information with clinical data is essential for developing a holistic understanding of the disease and optimizing treatment.
- **Pipeline hook:** Use dataset engineering best practices (schema, provenance, splits, documentation) for scalable multi-modal expansion.

### 20. CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model
- **PDF:** `<76 papers/CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model.pdf>` (29 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Human action recognition (HAR) encompasses the task of monitoring human activities across various domains, including but not limited to medical, educational, entertainment, visual surveillance, video retrieval, and the identification of anomalous activities. Over the past decade, the field of HAR has wit- nessed substantial progress by leveraging convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to effectively extract and comprehend intricate information, thereby enhancing the overall performance of HAR systems.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 21. Combining Accelerometer and Gyroscope Data in Smartphone-Based
- **PDF:** `<76 papers/Combining Accelerometer and Gyroscope Data in Smartphone-Based.pdf>` (14 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Physical activity patterns can be informative about a patient’s health status. Traditionally, activity data have been gathered using patient self-report.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 22. Comparative Study on the Effects of Noise in
- **PDF:** `<76 papers/Comparative Study on the Effects of Noise in.pdf>` (6 pages)
- **Tags:** Robustness/Noise
- **Abstract (excerpt):** 1 2
- **Pipeline hook:** Add robustness evaluation (noise, missingness) and sensor-quality monitoring to reduce brittle real-world performance.

### 23. Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor
- **PDF:** `<76 papers/Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf>` (16 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Human Activity Recognition (HAR) systems are devised for continuously observing human behavior - primarily in the ﬁelds of environmental compatibility, sports injury detection, senior care, rehabil- itation, entertainment, and the surveillance in intelligent home settings. Inertial sensors, e.g., accelerometers, linear acceleration, and gyroscopes are frequently employed for this purpose, which are now compacted into smart devices, e.g., smartphones.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 24. Deep CNN-LSTM With Self-Attention Model for
- **PDF:** `<76 papers/Deep CNN-LSTM With Self-Attention Model for.pdf>` (16 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Human Activity Recognition (HAR) systems are devised for continuously observing human behavior - primarily in the ﬁelds of environmental compatibility, sports injury detection, senior care, rehabil- itation, entertainment, and the surveillance in intelligent home settings. Inertial sensors, e.g., accelerometers, linear acceleration, and gyroscopes are frequently employed for this purpose, which are now compacted into smart devices, e.g., smartphones.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 25. Deep learning for sensor-based activity recognition_ A survey
- **PDF:** `<76 papers/Deep learning for sensor-based activity recognition_ A survey.pdf>` (9 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Pattern Recognition Letters 119 (2019) 3–11 Contents lists available at ScienceDirect Pattern Recognition Letters journal homepage: www.elsevier.com/locate/patrec Deep learning for sensor-based activity recognition: A survey Jindong Wang a , b , Yiqiang Chen a , b , ∗, Shuji Hao c , Xiaohui Peng a , b , Lisha Hu a , b a Beijing Key Laboratory of Mobile Computing and Pervasive Device, Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China b University of Chinese Academy of Sciences, Beijing, China c Institute of High Performance Computing, A ∗STAR, Singapore a r t i c l e i n f o Article history: Available online 21 Feb?
- **Conclusion (excerpt):** Human activity recognition is an important research topic in pattern recognition and pervasive computing. In this paper, we sur- vey the recent advance in deep learning approaches for sensor- based activity recognition.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 26. Deep Learning in Human Activity Recognition with Wearable Sensors
- **PDF:** `<76 papers/Deep Learning in Human Activity Recognition with Wearable Sensors.pdf>` (44 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Mobile and wearable devices have enabled numerous applications, including activity tracking, wellness monitoring, and human–computer interaction, that measure and improve our daily lives. Many of these applications are made possible by leveraging the rich collection of low- power sensors found in many mobile and wearable devices to perform human activity recognition (HAR).
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 27. Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years
- **PDF:** `<76 papers/Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years.pdf>` (18 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Background.—Recent studies have demonstrated that passive smartphone and wearable sensor data collected throughout daily life can predict anxiety symptoms cross-sectionally. However, to date, no research has demonstrated the capacity for these digital biomarkers to predict long-term prognosis.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 28. Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study
- **PDF:** `<76 papers/Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study.pdf>` (20 pages)
- **Tags:** Wearables/Longitudinal, Clinical UX/Dashboard
- **Abstract (excerpt):** Background: The recent growth of eHealth is unprecedented, especially after the COVID-19 pandemic. Within eHealth, wearable technology is increasingly being adopted because it can offer the remote monitoring of chronic and acute conditions in daily life environments.
- **Pipeline hook:** Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts.

### 29. Development and Testing of Retrieval Augmented Generation in Large Language Models
- **PDF:** `<76 papers/Development and Testing of Retrieval Augmented Generation in Large Language Models.pdf>` (22 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** Development and Testing of Retrieval Augmented Generation in Large Language Models - A Case Study Report Yu He Ke*1,2 Liyuan Jin*3,4,5 Kabilan Elangovan4,5 Hairil Rizal Abdullah1,2 Nan Liu3 Alex Tiong Heng Sia3,7 Chai Rick Soh1,3 Joshua Yi Min Tung2,6 Jasmine Chiat Ling Ong3,8 Daniel Shu Wei Ting+3,4,5 Affiliations: 1 Department of Anesthesiology, Singapore General Hospital, Singapore, Singapore 2 Data Science and Artificial Intelligence Lab, Singapore General Hospital, Singapore 3 Duke-NUS Medical School, Singapore, Singapore 4 Singapore National Eye Centre, Singapore Eye Research Institute, Singapore, Singapore 5 Singapore Health Services,?
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 30. Development of a two-stage depression symptom detection model
- **PDF:** `<76 papers/Development of a two-stage depression symptom detection model.pdf>` (11 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Frontiers in Computer Science 01 frontiersin.org Development of a two-stage depression symptom detection model: application of neural networks to twitter data Faye Beatriz Tumaliuan *, Lorelie Grepo and Eugene Rex Jalao Department of Industrial Engineering and Operations Research, University of the Philippines Diliman, Quezon City, Philippines This study aims to help in the area of depression screening in the Philippine setting, focusing on the detection of depression symptoms through language use and behavior in social media to help improve the accuracy of symptom tracking. A two-stage detection model is proposed, wherein the first stage dea?
- **Conclusion (excerpt):** Solutions that can identify depression patterns from daily living activity that does not hinder with depression symptoms and can help with initial screening methods are needed. This study aimed to help in the area of depression screening with the focus of detecting depression symptoms through language use in social media in the Philippine setting.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 31. DevOps-Driven Real-Time Health Analytics
- **PDF:** `<76 papers/DevOps-Driven Real-Time Health Analytics.pdf>` (10 pages)
- **Tags:** MLOps/Reproducibility, Edge/Runtime/IoT
- **Abstract (excerpt):** The rapid adoption of wearable health devices and IoT sensors has given us real -time health monitoring like never before, with opportunities for early disease detection, personalized treatment, and proactive healthcare interventions. However, scalability, latency, security, and regulatory compliance pose significant challenges.
- **Conclusion (excerpt):** This paper demonstrates the transformative potential of DevOps -driven real -time health analytics. By integrating CI/CD pipelines, MLOps automation, and DevSecOps security measures, healthcare organizations can achieve low-latency, scalable, and secure real-time health data processing.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 32. Domain Adaptation for Inertial Measurement Unit-based Human
- **PDF:** `<76 papers/Domain Adaptation for Inertial Measurement Unit-based Human.pdf>` (28 pages)
- **Tags:** Domain Adaptation/Transfer
- **Abstract (excerpt):** Machine learning-based wearable human activity recognition (WHAR) models enable the de- velopment of various smart and connected community applications such as sleep pattern moni- toring, medication reminders, cognitive health assessment, sports analytics, etc. However, the widespread adoption of these WHAR models is impeded by their degraded performance in the presence of data distribution heterogeneities caused by the sensor placement at diﬀerent body positions, inherent biases and heterogeneities across devices, and personal and environmental diversities.
- **Pipeline hook:** Add explicit cross-device/cross-user evaluation and a fine-tuning workflow to close the lab-to-life gap.

### 33. Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors
- **PDF:** `<76 papers/Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors.pdf>` (17 pages)
- **Tags:** HAR/DL, Edge/Runtime/IoT
- **Abstract (excerpt):** A wide range of applications, including sports and healthcare, use human activity recog- nition (HAR). The Internet of Things (IoT), using cloud systems, offers enormous resources but produces high delays and huge amounts of trafﬁc.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 34. Enabling End-To-End Machine Learning
- **PDF:** `<76 papers/Enabling End-To-End Machine Learning.pdf>` (10 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** The use of machine learning techniques has expanded in education research, driven by the rich data from digital learning environments and institutional data ware- houses. However, replication of machine learned models in the domain of the learning sciences is particularly challenging due to a conﬂuence of experimental, methodological, and data barriers.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 35. Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy
- **PDF:** `<76 papers/Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy.pdf>` (26 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** The exponential surge in online health information, coupled with its increasing use by non-experts, highlights the pressing need for advanced Health Informa- tion Retrieval models that consider not only topical relevance but also the factual accuracy of the retrieved information, given the potential risks associated with health misinformation. To this aim, this paper introduces a solution driven by Retrieval-Augmented Generation (RAG), which leverages the capabilities of gen- erative Large Language Models (LLMs) to enhance the retrieval of health-related documents grounded in scientific evidence.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 36. Enhancing Multimodal Electronic Health Records
- **PDF:** `<76 papers/Enhancing Multimodal Electronic Health Records.pdf>` (11 pages)
- **Tags:** EHR/Clinical Data
- **Abstract (excerpt):** The integration of multimodal Electronic Health Records (EHR) data has significantly advanced clinical predictive capabilities. Exist- ing models, which utilize clinical notes and multivariate time-series EHR data, often fall short of incorporating the necessary medical context for accurate clinical tasks, while previous approaches with knowledge graphs (KGs) primarily focus on structured knowledge extraction.
- **Pipeline hook:** Align wearable features with EHR-style outcomes/labels and leverage established clinical dataset practices.

### 37. Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot
- **PDF:** `<76 papers/Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot.pdf>` (16 pages)
- **Tags:** RAG/LLM, Edge/Runtime/IoT
- **Abstract (excerpt):** Retrieval-augmented generation (RAG) is a well-suited technique for retrieving privacy-sensitive Electronic Health Records (EHR). It can serve as a key module of the healthcare copilot, helping reduce misdiagnosis for healthcare practitioners and patients.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 38. Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data
- **PDF:** `<76 papers/Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf>` (8 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** —This paper compares the performance of BiLSTM and CNN+GRU deep learning models for Human Activity Recog- nition (HAR) on two WiFi-based Channel State Information (CSI) datasets: UT-HAR and NTU-Fi HAR. The findings indicate that the CNN+GRU model has a higher accuracy on the UT- HAR dataset (95.20%) thanks to its ability to extract spatial features.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 39. Evaluating large language models on medical evidence summarization
- **PDF:** `<76 papers/Evaluating large language models on medical evidence summarization.pdf>` (8 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** genera- tion7, there is no study yet on medical evidence summarization and appraisal. In this study, we conduct a systematic study of the potential and possible limitations of zero-shot prompt-based LLMs on medical evidence summarization using GPT-3.5 and ChatGPT models.
- **Conclusion (excerpt):** section; (2) ChatGPT- MainResult; (3) ChatGPT-Abstract; and (4) GPT3.5-MainResult. The order in which the summaries are presented is randomized to minimize potential order effects during the evaluation process.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 40. Exploring the Capabilities of LLMs for IMU-based Fine-grained
- **PDF:** `<76 papers/Exploring the Capabilities of LLMs for IMU-based Fine-grained.pdf>` (6 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Human activity recognition (HAR) using inertial measurement units (IMUs) increasingly leverages large language models (LLMs), yet existing approaches focus on coarse activities like walking or running. Our preliminary study indicates that pretrained LLMs fail catastrophically on fine-grained HAR tasks such as air-written letter recognition, achieving only near-random guessing accuracy.
- **Conclusion (excerpt):** This work explores the use of Large Language Models (LLMs) for fine-grained Human Activity Recognition, specifically focusing on mid-air letter recognition. Our investigation reveals important findings about how well LLMs understand IMU data.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 41. Human Activity Recognition using Multi-Head CNN followed by LSTM
- **PDF:** `<76 papers/Human Activity Recognition using Multi-Head CNN followed by LSTM.pdf>` (6 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** — This study presents a novel method to recognize human physical activities using CNN followed by LSTM. Achieving high accuracy by traditional machine learning algorithms, (such as SVM, KNN and random forest method) is a challenging task because the data acquired from the wearable sensors like accelerometer and gyroscope is a time-series data.
- **Conclusion (excerpt):** AND FUTURE WORK In this work, we proposed a novel multi-head CNN followed by LSTM architecture to recognize human physical activity recognition. We used the UCI database in which the data is divided into training and test subsets with a 7:3 ratio respectively.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 42. Human Activity Recognition Using Tools of Convolutional Neural Networks
- **PDF:** `<76 papers/Human Activity Recognition Using Tools of Convolutional Neural Networks.pdf>` (32 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** Human Activity Recognition (HAR) plays a significant role in the everyday life of people because of its ability to learn extensive high-level information about human activity from wearable or stationary devices. A substantial amount of research has been co nducted on HAR and numerous approaches based on deep learning and machine learning have been exploited by the research community to classify human activities.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 43. I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses
- **PDF:** `<76 papers/I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.pdf>` (26 pages)
- **Tags:** MLOps/Reproducibility, Privacy/Federated, Interoperability/FHIR
- **Abstract (excerpt):** Background:Clinicians are interested in better understanding complex diseases, such as cancer or rare diseases, so they need to produce and exchange data to mutualize sources and join forces. To do so and ensure privacy, a natural way consists in using a decentralized architecture and Federated Learning algorithms.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 44. Implications on Human Activity Recognition Research
- **PDF:** `<76 papers/Implications on Human Activity Recognition Research.pdf>` (8 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** The astonishing success of Large Language Models (LLMs) in Natu- ral Language Processing (NLP) has spurred their use in many ap- plication domains beyond text analysis, including wearable sensor- based Human Activity Recognition (HAR). In such scenarios, often sensor data are directly fed into an LLM along with text instructions for the model to perform activity classification.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 45. Learning the Language of wearable sensors
- **PDF:** `<76 papers/Learning the Language of wearable sensors.pdf>` (34 pages)
- **Tags:** Foundation/SSL
- **Abstract (excerpt):** arXiv:2506.09108v1 [cs.LG] 10 Jun 2025 2025-6-13 SensorLM: Learning the Language of Wearable Sensors Yuwei Zhang1∗, Kumar Ayush1∗, Siyuan Qiao2, A. Ali Heydari1, Girish Narayanswamy1, Maxwell A.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 46. Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises
- **PDF:** `<76 papers/Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises.pdf>` (42 pages)
- **Tags:** EHR/Clinical Data, Data Engineering
- **Abstract (excerpt):** The Medical Information Mart for Intensive Care (MIMIC) datasets have become the Kernel of Digital Health Research by providing freely accessible, deidentified records from tens of thousands of critical care admissions, enabling a broad spectrum of applications in clinical decision support, outcome prediction, and healthcare analytics. Although numerous studies and surveys have explored thepredictivepowerandclinicalutilityofMIMICbasedmodels,criticalchallengesindataintegration, representation, and interoperability remain underexplored.
- **Pipeline hook:** Align wearable features with EHR-style outcomes/labels and leverage established clinical dataset practices.

### 47. LLM Chatbot-Creation Approaches
- **PDF:** `<76 papers/LLM Chatbot-Creation Approaches.pdf>` (9 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** —This full research -to-practice paper explores approaches for developing course chatbots by comparing low-code platforms and custom-coded solutions in educational contexts. With the rise of Large Language Models (LLMs) like GPT-4 and LLaMA, LLM-based chatbots are being integrated into teaching workflows to automate tasks, provide assistance, and offer scalable support.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 48. LSM-2-Learning from Incomplete Wearable Sensor Data
- **PDF:** `<76 papers/LSM-2-Learning from Incomplete Wearable Sensor Data.pdf>` (32 pages)
- **Tags:** Foundation/SSL, Wearables/Longitudinal
- **Abstract (excerpt):** 2025-6-6 LSM-2: Learning from Incomplete Wearable Sensor Data Maxwell A. Xu1,3*,†, Girish Narayanswamy1,4*,†, Kumar Ayush1, Dimitris Spathis1, Shun Liao1, Shyam A.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 49. Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature
- **PDF:** `<76 papers/Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature.pdf>` (18 pages)
- **Tags:** Edge/Runtime/IoT
- **Abstract (excerpt):** The integration of machine learning (ML) with edge computing and wearable devices is rapidly advancing healthcare applications. This study systematically maps the literature in this emerg- ing ﬁeld, analyzing 171 studies and focusing on 28 key articles after rigorous selection.
- **Pipeline hook:** Plan for on-device/edge constraints (latency, battery) and consider runtime orchestration for multi-wearable setups.

### 50. Machine Learning based Anxiety Detection using Physiological Signals and Context Features
- **PDF:** `<76 papers/Machine Learning based Anxiety Detection using Physiological Signals and Context Features.pdf>` (6 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** —Anxiety is a common mental disorder that affects millions of people worldwide. Anxiety can be detected using physiological signals such as heart rate, skin conductance, blood pressure, and respiration.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 51. MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS
- **PDF:** `<76 papers/MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf>` (12 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** Machine Learning (ML) has become a fast-growing, trending approach in solution development in practice. Deep Learning (DL) which is a subset of ML, learns using deep neural networks to simulate the human brain.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 52. Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition
- **PDF:** `<76 papers/Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition.pdf>` (19 pages)
- **Tags:** Foundation/SSL, HAR/DL
- **Abstract (excerpt):** Compared with visual signals, Inertial Measurement Units (IMUs) placed onhuman limbs can capture accuratemotion signals while being robust to lighting variation and occlusion. While these character- istics are intuitively valuable to help egocentric action recognition, the potential of IMUs remains under-explored.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 53. Medical Graph RAG Towards Safe Medical Large Language Model via
- **PDF:** `<76 papers/Medical Graph RAG Towards Safe Medical Large Language Model via.pdf>` (10 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** We introduce a novel graph-based Retrieval- Augmented Generation (RAG) framework specifically designed for the medical domain, called MedGraphRAG, aimed at enhancing Large Language Model (LLM) capabilities for generating evidence-based medical responses, thereby improving safety and reliability when handling private medical data. Graph-based RAG (GraphRAG) leverages LLMs to orga- nize RAG data into graphs, showing strong po- tential for gaining holistic insights from long- form documents.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 54. MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND
- **PDF:** `<76 papers/MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND.pdf>` (11 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** In this paper we explore the challenges of automating experiments in data science. We propose an extensible experiment model as a foundation for integration of different open source tools for running research experiments.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 55. MLHOps Machine Learning for Healthcare Operations
- **PDF:** `<76 papers/MLHOps Machine Learning for Healthcare Operations.pdf>` (86 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** Machine Learning Health Operations (MLHOps) is the combination of pro- cesses for reliable, eﬃcient, usable, and ethical deployment and maintenance of machine learning models in healthcare settings. This paper provides both a survey of work in this area and guidelines for developers and clinicians to deploy and maintain their own models in clinical practice.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 56. Momentary Stressor Logging and Reflective Visualizations Implications for
- **PDF:** `<76 papers/Momentary Stressor Logging and Reflective Visualizations Implications for.pdf>` (36 pages)
- **Tags:** Mental Health, Clinical UX/Dashboard
- **Abstract (excerpt):** Momentary Stressor Logging and Reflective Visualizations: Implications for Stress Management with Wearables∗ SAMEER NEUPANE†, University of Memphis, USA MITHUN SAHA, University of Memphis, USA NASIR ALI‡, University of Memphis, USA TIMOTHY HNAT‡§, CuesHub, PBC, USA SHAHIN ALAN SAMIEI, University of Memphis, USA ANANDATIRTHA NANDUGUDI‡, University of Memphis, USA DAVID M. ALMEIDA,The Pennsylvania State University, USA SANTOSH KUMAR§, University of Memphis, USA Commercial wearables from Fitbit, Garmin, and Whoop have recently introduced real-time notifications based on detecting changes in physiological responses indicating potential stress.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 57. Multimodal Frame-Scoring Transformer for Video Summarization
- **PDF:** `<76 papers/Multimodal Frame-Scoring Transformer for Video Summarization.pdf>` (9 pages)
- **Tags:** HAR/DL
- **Abstract (excerpt):** As the number of video content has mushroomed in recent years, automatic video summarization has come useful when we want to just peek at the con- tent of the video. However, there are two under- lying limitations in generic video summarization task.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 58. Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework
- **PDF:** `<76 papers/Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework.pdf>` (9 pages)
- **Tags:** RAG/LLM, HPO/AutoML
- **Abstract (excerpt):** npj |digital medicine Article Published in partnership with Seoul National University Bundang Hospital https://doi.org/10.1038/s41746-024-01091-y Optimization of hepatological clinical guidelines interpretation by large language models: a retrieval augmented generation-based framework Check for updates Simone Kresevic1,2,4 , Mauro Giuffrè 2,4 , Milos Ajcevic1, Agostino Accardo1,L o r yS .C r o c è3 & Dennis L. Shung2 Large language models (LLMs) can potentially transform healthcare, particularly in providing the right information to the right provider at the right time in the hospital workﬂow.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 59. Panic Attack Prediction Using Wearable Devices and Machine
- **PDF:** `<76 papers/Panic Attack Prediction Using Wearable Devices and Machine.pdf>` (13 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Background: A panic attack (PA) is an intense form of anxiety accompanied by multiple somatic presentations, leading to frequent emergency department visits and impairing the quality of life. A prediction model for PAs could help clinicians and patients monitor, control, and carry out early intervention for recurrent PAs, enabling more personalized treatment for panic disorder (PD).
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 60. Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care
- **PDF:** `<76 papers/Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care.pdf>` (25 pages)
- **Tags:** Mental Health, Clinical UX/Dashboard
- **Abstract (excerpt):** PACM on Human-Computer Interaction, Vol. 3, No.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 61. Real-Time Stress Monitoring Detection and Management in College Students
- **PDF:** `<76 papers/Real-Time Stress Monitoring Detection and Management in College Students.pdf>` (30 pages)
- **Tags:** Mental Health, Edge/Runtime/IoT
- **Abstract (excerpt):** , references, tables, and figures) *Corresponding Author: Farzan Sasangohar Institution: Texas A&M University PO Box: College Station, TX, Email address: sasangohar@tamu.edu ABSTRACT College students are increasingly affected by stress, anxiety, and depression, yet face barriers to traditional mental health care. This study evaluated the efficacy of a mobile health (mHealth) intervention, Mental Health Evaluation and Lookout Program ( mHELP), which integrates a smartwatch sensor and machine learning (ML) algorithms for real -time stress detection and self -management.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 62. Reproducible workflow for online AI in digital health
- **PDF:** `<76 papers/Reproducible workflow for online AI in digital health.pdf>` (17 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** royalsocietypublishing.org/journal/rspa Research Article submitted to journal Subject Areas: Digital health, reproducibility, AI Keywords: Digital health, reproducibility, AI Author for correspondence: Susobhan Ghosh e-mail: susobhan_ghosh@g.harvard.edu Reproducible workflow for online AI in digital health Susobhan Ghosh1*, Bhanu T. Gullapalli1*, Daiqi Gao2, Asim Gazi1, Anna Trella1, Ziping Xu2, Kelly Zhang3, Susan A.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 63. Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors
- **PDF:** `<76 papers/Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.pdf>` (18 pages)
- **Tags:** Mental Health, Robustness/Noise
- **Abstract (excerpt):** The resilience of machine learning models for anxiety detection through wearable technology was explored. The effectiveness of feature-based and end-to-end machine learning models for anxiety detection was evaluated under varying conditions of Gaussian noise.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 64. Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review
- **PDF:** `<76 papers/Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review.pdf>` (30 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by inte- grating external knowledge retrieval to improve factual consistency and reduce halluci- nations. Despite growing interest, its use in healthcare remains fragmented.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 65. Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster
- **PDF:** `<76 papers/Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster.pdf>` (23 pages)
- **Tags:** RAG/LLM, Foundation/SSL
- **Abstract (excerpt):** Large Language Models (LLMs) and Foundation Models (FMs) have recently be- come prevalent for time series forecasting tasks. While fine-tuning LLMs enables domain adaptation, they often struggle to generalize across diverse and unseen datasets.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 66. Scientific Evidence for Clinical Text Summarization Using Large
- **PDF:** `<76 papers/Scientific Evidence for Clinical Text Summarization Using Large.pdf>` (18 pages)
- **Tags:** RAG/LLM
- **Abstract (excerpt):** Background: Information overload in electronic health records requires effective solutions to alleviate clinicians’ administrative tasks. Automatically summarizing clinical text has gained significant attention with the rise of large language models.
- **Pipeline hook:** Use retrieval-grounded report generation (prefer KG-RAG), and add factuality + provenance evaluation for clinical outputs.

### 67. Self-supervised learning for fast and scalable time series hyper-parameter tuning
- **PDF:** `<76 papers/Self-supervised learning for fast and scalable time series hyper-parameter tuning.pdf>` (10 pages)
- **Tags:** Foundation/SSL, HPO/AutoML
- **Abstract (excerpt):** Hyper-parameters of time series models play an important role in time series analysis. Slight differences in hyper-parameters might lead to very different forecast results for a given model, and there- fore, selecting good hyper-parameter values is indispensable.
- **Conclusion (excerpt):** and implications. International journal of forecasting16, 4 (2000), 451–476.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 68. Spatiotemporal Feature Fusion for
- **PDF:** `<76 papers/Spatiotemporal Feature Fusion for.pdf>` (15 pages)
- **Tags:** HAR/DL, Edge/Runtime/IoT
- **Abstract (excerpt):** —Early detection of anxiety is crucial for reducing the suff ering of individuals with mental disorders and improving tr eatment outcomes. Utilizing an mHealth platform for anxiety screen ing can be particularly practical in improving screening ef ﬁciency and reducing costs.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 69. Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals
- **PDF:** `<76 papers/Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.pdf>` (35 pages)
- **Tags:** Foundation/SSL
- **Abstract (excerpt):** Time-series foundation models excel at tasks like forecasting across diverse data types by leveraging informative waveform representations. Wearable sensing data, however, pose unique challenges due to their variability in patterns and frequency bands, especially for healthcare-related outcomes.
- **Pipeline hook:** Experiment with self-supervised / foundation-model pretraining to reduce labeling needs and improve few-shot adaptation.

### 70. Toward Reusable Science with Readable Code and
- **PDF:** `<76 papers/Toward Reusable Science with Readable Code and.pdf>` (10 pages)
- **Tags:** MLOps/Reproducibility
- **Abstract (excerpt):** —An essential part of research and scientiﬁc commu- nication is researchers’ ability to reproduce the results of others. While there have been increasing standards for authors to make data and code available, many of these ﬁles are hard to re- execute in practice, leading to a lack of research reproducibility.
- **Pipeline hook:** Strengthen reproducibility (data/model versioning, CI checks, experiment automation) and make deployments auditable.

### 71. Transfer Learning in Human Activity Recognition  A Survey
- **PDF:** `<76 papers/Transfer Learning in Human Activity Recognition  A Survey.pdf>` (40 pages)
- **Tags:** Domain Adaptation/Transfer, HAR/DL
- **Abstract (excerpt):** Transfer Learning in Human Activity Recognition: A Survey SOURISH GUNESH DHEKANE, THOMAS PLÖTZ ∗, School of Interactive Computing, College of Computing, Georgia Institute of Technology, USA Sensor-based human activity recognition (HAR) has been an active research area, owing to its applications in smart environ- ments, assisted living, fitness, healthcare, etc. Recently, deep learning based end-to-end training has resulted in state-of-the-art performance in domains such as computer vision and natural language, where large amounts of annotated data are available.
- **Pipeline hook:** Compare/extend HAR architectures (attention, multi-task, transformers) and keep windowing + sensor-fusion choices configurable.

### 72. Transforming Wearable Data into Personal Health
- **PDF:** `<76 papers/Transforming Wearable Data into Personal Health.pdf>` (53 pages)
- **Tags:** Wearables/Longitudinal
- **Abstract (excerpt):** 2025-9-1 Transforming Wearable Data into Personal Health Insights using Large Language Model Agents Mike A. Merrill∗, ‡, Akshay Paruchuri∗, ‡, Naghmeh Rezaei1, Geza Kovacs1, Javier Perez1, Yun Liu1, Erik Schenck1, Nova Hammerquist1, Jake Sunshine1, Shyam Tailor1, Kumar Ayush1, Hao-Wei Su1, Qian He1, Cory Y.
- **Pipeline hook:** Add longitudinal/bout analytics (trends, change detection) and store per-day/per-week aggregates as first-class artifacts.

### 73. Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study
- **PDF:** `<76 papers/Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study.pdf>` (9 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Background: Early identification of mental disorder symptoms is crucial for timely treatment and reduction of recurring symptoms and disabilities. A tool to help individuals recognize warning signs is important.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 74. Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis
- **PDF:** `<76 papers/Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis.pdf>` (23 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Background: Anxiety disorders rank among the most prevalent mental disorders worldwide. Anxiety symptoms are typically evaluated using self-assessment surveys or interview-based assessment methods conducted by clinicians, which can be subjective, time-consuming, and challenging to repeat.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 75. Wearable Artificial Intelligence for Detecting Anxiety
- **PDF:** `<76 papers/Wearable Artificial Intelligence for Detecting Anxiety.pdf>` (23 pages)
- **Tags:** Mental Health
- **Abstract (excerpt):** Background: Anxiety disorders rank among the most prevalent mental disorders worldwide. Anxiety symptoms are typically evaluated using self-assessment surveys or interview-based assessment methods conducted by clinicians, which can be subjective, time-consuming, and challenging to repeat.
- **Pipeline hook:** Incorporate longitudinal and context-aware signals; prioritize generalization and clinically meaningful validation.

### 76. When Does Optimizing a Proper Loss Yield Calibration
- **PDF:** `<76 papers/When Does Optimizing a Proper Loss Yield Calibration.pdf>` (25 pages)
- **Tags:** Calibration/Uncertainty
- **Abstract (excerpt):** Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth.
- **Pipeline hook:** Track calibration (ECE/Brier) and add post-hoc calibration (temperature scaling) for safer decision support.

# 76 Papers ? Suggestions & Pipeline Improvements
**Generated:** 2025-12-13

This file distills cross-paper, actionable improvements for the current pipeline (HAR ? temporal/bout analysis ? anxiety/stress inference ? RAG/LLM reporting ? MLOps).

## High-Impact Improvements (Actionable)
### 1) Add calibration + uncertainty as a first-class evaluation
- Track ECE/Brier score, reliability plots, and apply post-hoc calibration (temperature scaling) for safer decision support.
- Supporting papers: When Does Optimizing a Proper Loss Yield Calibration.

### 2) Add robustness to noise + sensor quality drift
- Evaluate under sensor noise, timing jitter, missing windows, and real-world motion artifacts; log data-quality metrics and alert on degradation.
- Supporting papers: Comparative Study on the Effects of Noise in; Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors.

### 3) Treat missing data as a core modeling concern
- Add systematic missingness simulation; consider SSL methods designed for incomplete streams; prefer models that degrade gracefully.
- Supporting papers: Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions; Learning the Language of wearable sensors; LSM-2-Learning from Incomplete Wearable Sensor Data; Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition; Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster; Self-supervised learning for fast and scalable time series hyper-parameter tuning; Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals.

### 4) Upgrade domain adaptation into an automated workflow
- Standardize cross-device/cross-user benchmarks, track the lab-to-life gap, and implement a re-tuning loop triggered by drift/shift detection.
- Supporting papers: Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables; Domain Adaptation for Inertial Measurement Unit-based Human; Transfer Learning in Human Activity Recognition  A Survey.

### 5) Add opt-in privacy-preserving training mode (federated learning)
- For sensitive mental-health signals, prototype a federated learning variant (or at least privacy threat modeling + governance), especially if scaling to real users.
- Supporting papers: A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation; I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.

### 6) Add longitudinal/bout analytics as a first-class layer
- Move from window-level predictions to bout-level events and long-term trends (daily/weekly aggregates) that clinicians can interpret.
- Supporting papers: A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare; Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study; LSM-2-Learning from Incomplete Wearable Sensor Data; Transforming Wearable Data into Personal Health.

### 7) Make clinician integration a deliverable, not an afterthought
- Build a clinician-facing dashboard view aligned to workflows; emphasize interpretability, longitudinal trends, and actionable summaries.
- Supporting papers: Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study; Momentary Stressor Logging and Reflective Visualizations Implications for; Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care.

### 8) Strengthen RAG safety: provenance, KG-RAG, and evaluation harness
- Prefer KG-based retrieval when possible; log retrieved chunks/triples; add factuality checks and ?no-answer? behavior for low confidence retrieval.
- Supporting papers: Development and Testing of Retrieval Augmented Generation in Large Language Models; Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy; Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot; Evaluating large language models on medical evidence summarization; LLM Chatbot-Creation Approaches; Medical Graph RAG Towards Safe Medical Large Language Model via; Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework; Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review; Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster; Scientific Evidence for Clinical Text Summarization Using Large.

### 9) Add automated HPO as a reproducible stage
- Integrate Optuna-style sweeps with MLflow logging; treat best-config selection as reproducible artifact (config + seed + data version).
- Supporting papers: A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models; Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework; Self-supervised learning for fast and scalable time series hyper-parameter tuning.

### 10) Plan early for edge constraints (latency, battery, multi-wearable)
- Define target inference latency/energy budgets; evaluate quantization/pruning; consider runtime orchestration when multiple devices are present.
- Supporting papers: An AI-native Runtime for Multi-Wearable Environments; Atechnical framework for deploying custom real-time machine; DevOps-Driven Real-Time Health Analytics; Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors; Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot; Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature; Real-Time Stress Monitoring Detection and Management in College Students; Spatiotemporal Feature Fusion for.

### 11) Adopt healthcare interoperability patterns (FHIR-aligned ETL/metadata)
- Define an interoperability-aware schema and metadata so wearable-derived events can map into clinical systems later.
- Supporting papers: I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses.

### 12) Consider anomaly/OOD detection as a safety layer
- Detect ?unknown? activities and distribution shift; gate report generation on sufficient confidence and data quality.
- Supporting papers: A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models.

### 13) Explore synthetic augmentation (carefully)
- Use generative models (e.g., diffusion) to augment rare anxiety-related activities; validate with strict splits to avoid leakage.
- Supporting papers: A DIFFUSION MODEL FOR MULTIVARIATE.

### 14) Strengthen MLOps for healthcare auditability
- Add model/data cards, immutable run records (config + seed + data hash), and deployment audit logs for compliance-ready traceability.
- Supporting papers: A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models; A Visual Data and Detection Pipeline for Wearable Industrial Assistants; An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors; AutoMR- A Universal Time Series Motion Recognition Pipeline; DevOps-Driven Real-Time Health Analytics; Enabling End-To-End Machine Learning; I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses; MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS; MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND; MLHOps Machine Learning for Healthcare Operations; Reproducible workflow for online AI in digital health; Toward Reusable Science with Readable Code and.

## Recommended Experiments Backlog
- **Robustness grid:** noise levels + missingness patterns + resampling rates + device placement changes.
- **Generalization:** leave-one-user-out and leave-one-device-out evaluation; report expected performance drop.
- **Ablations:** attention vs no-attention; multi-task segmentation vs single-task; transformer baseline if feasible.
- **Calibration:** pre/post temperature scaling; report ECE and Brier.
- **RAG safety:** hallucination/factuality tests, retrieval failure cases, and audit logs.

## Open Questions to Resolve (Thesis-Relevant)
- What is the **clinical target** (screening vs monitoring vs prediction) and what error type is most costly (FP vs FN)?
- Which **ground truth** is available/acceptable (EMA, clinician labels, validated questionnaires), and how will label noise be handled?
- What is the intended **deployment setting** (on-device vs phone vs cloud) and what privacy constraints apply?
- How will we ensure **interpretability** of anxiety-related activity patterns (bout-level explanations, trend narratives)?

## Paper Clusters (Quick Map)
- **HAR/DL** (22): A Close Look into Human Activity Recognition Models using Deep Learning; A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation; A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models; An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors; Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition; AutoMR- A Universal Time Series Motion Recognition Pipeline; CNNs, RNNs and Transformers in Human Action Recognition A Survey and a Hybrid Model; Combining Accelerometer and Gyroscope Data in Smartphone-Based; Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor; Deep CNN-LSTM With Self-Attention Model for; Deep learning for sensor-based activity recognition_ A survey; Deep Learning in Human Activity Recognition with Wearable Sensors; Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors; Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data; Exploring the Capabilities of LLMs for IMU-based Fine-grained; Human Activity Recognition using Multi-Head CNN followed by LSTM; Human Activity Recognition Using Tools of Convolutional Neural Networks; Implications on Human Activity Recognition Research; (+4 more)
- **Mental Health** (15): A Survey on Wearable Sensors for Mental Health Monitoring; ADAM-sense_Anxietydisplayingactivitiesrecognitionby; Anxiety Detection Leveraging Mobile Passive Sensing; Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables; Deep Learning Paired with Wearable Passive Sensing Data Predicts Deterioration in Anxiety Disorder Symptoms across 17–18 Years; Development of a two-stage depression symptom detection model; Machine Learning based Anxiety Detection using Physiological Signals and Context Features; Momentary Stressor Logging and Reflective Visualizations Implications for; Panic Attack Prediction Using Wearable Devices and Machine; Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care; Real-Time Stress Monitoring Detection and Management in College Students; Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors; Using Wearable Devices and Speech Data for Personalized Machine Learning in Early Detection of Mental Disorders Protocol for a Participatory Research Study; Wearable Artificial Intelligence for Detecting Anxiety Systematic Review and Meta-Analysis; Wearable Artificial Intelligence for Detecting Anxiety
- **Domain Adaptation/Transfer** (3): Are Anxiety Detection Models Generalizable-A Cross-Activity and Cross-Population Study Using Wearables; Domain Adaptation for Inertial Measurement Unit-based Human; Transfer Learning in Human Activity Recognition  A Survey
- **Foundation/SSL** (7): Beyond Sensor Data- Foundation Models of Behavioral Data from Wearables Improve Health Predictions; Learning the Language of wearable sensors; LSM-2-Learning from Incomplete Wearable Sensor Data; Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition; Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster; Self-supervised learning for fast and scalable time series hyper-parameter tuning; Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals
- **Wearables/Longitudinal** (4): A State-of-the-Art Review of Computational Models for Analyzing Longitudinal Wearable Sensor Data in Healthcare; Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study; LSM-2-Learning from Incomplete Wearable Sensor Data; Transforming Wearable Data into Personal Health
- **MLOps/Reproducibility** (12): A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models; A Visual Data and Detection Pipeline for Wearable Industrial Assistants; An End-to-End Deep Learning Pipeline for Football Activity Recognition Based on Wearable Acceleration Sensors; AutoMR- A Universal Time Series Motion Recognition Pipeline; DevOps-Driven Real-Time Health Analytics; Enabling End-To-End Machine Learning; I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses; MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS; MLDEV DATA SCIENCE EXPERIMENT AUTOMATION AND; MLHOps Machine Learning for Healthcare Operations; Reproducible workflow for online AI in digital health; Toward Reusable Science with Readable Code and
- **RAG/LLM** (10): Development and Testing of Retrieval Augmented Generation in Large Language Models; Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy; Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot; Evaluating large language models on medical evidence summarization; LLM Chatbot-Creation Approaches; Medical Graph RAG Towards Safe Medical Large Language Model via; Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework; Retrieval-Augmented Generation (RAG) in Healthcare A Comprehensive Review; Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster; Scientific Evidence for Clinical Text Summarization Using Large
- **Robustness/Noise** (2): Comparative Study on the Effects of Noise in; Resilience of Machine Learning Models in Anxiety Detection Assessing the Impact of Gaussian Noise on Wearable Sensors
- **Privacy/Federated** (2): A Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation; I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses
- **Edge/Runtime/IoT** (8): An AI-native Runtime for Multi-Wearable Environments; Atechnical framework for deploying custom real-time machine; DevOps-Driven Real-Time Health Analytics; Dynamic and Distributed Intelligence over Smart Devices, Internet of Things Edges, and Cloud Computing for Human Activity Recognition Using Wearable Sensors; Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot; Machine Learning Applied to Edge Computing and Wearable Devices for Healthcare- Systematic Mapping of the Literature; Real-Time Stress Monitoring Detection and Management in College Students; Spatiotemporal Feature Fusion for
- **Clinical UX/Dashboard** (3): Designing a Clinician-Centered Wearable Data Dashboard (CarePortal) Participatory Design Study; Momentary Stressor Logging and Reflective Visualizations Implications for; Provider Perspectives on Integrating Sensor-Captured Patient-Generated Data in Mental Health Care
- **Interoperability/FHIR** (1): I-ETL an interoperability-aware health (meta)data pipeline to enable federated analyses
- **Calibration/Uncertainty** (1): When Does Optimizing a Proper Loss Yield Calibration
- **HPO/AutoML** (3): A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models; Optimization of hepatological clinical guidelines interpretation by large language models- a retrieval augmented generation-based framework; Self-supervised learning for fast and scalable time series hyper-parameter tuning
- **Anomaly Detection** (1): A Two-Stage Anomaly Detection Framework for Improved Healthcare Using Support Vector Machines and Regression Models
- **Synthetic/Generative** (1): A DIFFUSION MODEL FOR MULTIVARIATE
- **EHR/Clinical Data** (2): Enhancing Multimodal Electronic Health Records; Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises
- **Data Engineering** (2): Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets; Leveraging MIMIC Datasets for Better Digital Health- A Review on Open Problems, Progress Highlights, and Future Promises
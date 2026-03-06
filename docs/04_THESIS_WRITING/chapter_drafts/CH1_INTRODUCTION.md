# Chapter 1 — Introduction

---

## 1.1 Problem Statement

Human Activity Recognition (HAR) using wearable inertial measurement units (IMUs) has progressed from a laboratory research problem to a practical tool for continuous health monitoring. In the context of anxiety detection, wrist-worn accelerometer and gyroscope data can be used to classify physical activities — such as walking, sitting, or fidgeting — whose patterns correlate with mental health states. Deep learning architectures, including one-dimensional convolutional neural networks combined with bidirectional long short-term memory layers (1D-CNN-BiLSTM), have demonstrated strong classification performance when evaluated on curated benchmark datasets.

However, a significant gap persists between research-grade model evaluation and reliable operation in production. The central problem addressed by this thesis is as follows:

> **Once a HAR model trained on labelled laboratory data is deployed to process continuous, unlabelled wearable recordings, how can the system detect that the model's reliability has degraded and adapt without requiring manual re-labelling?**

This question is rarely addressed in the HAR literature. The majority of published work evaluates models under closed-world assumptions: the training distribution matches the test distribution, all test data are labelled, and the model is evaluated once rather than continuously. In a real deployment, none of these assumptions hold. Users differ in anthropometry, handedness, and movement idiosyncrasies. Sensor placement may vary between the dominant and non-dominant wrist. Environmental conditions, firmware updates, and battery degradation introduce gradual sensor drift. Over days and weeks, the data distribution shifts away from the training distribution, and model outputs become progressively less trustworthy — yet no ground-truth labels exist to detect or quantify this degradation.

The consequences of undetected degradation are particularly severe in a mental health monitoring context. Misclassified activity sequences can lead to incorrect anxiety state estimates, eroding both clinical utility and user trust. A production HAR system therefore requires not only a performant model, but also a surrounding infrastructure that (a) continuously monitors prediction quality using proxy signals, (b) determines when retraining is necessary, (c) adapts the model without labelled data, and (d) ensures that the entire process is reproducible, versioned, and auditable.

This is fundamentally an MLOps problem. Machine Learning Operations (MLOps) extends the principles of DevOps — automation, continuous integration, monitoring, and infrastructure-as-code — to the machine learning lifecycle. While MLOps practices are well-established in domains such as natural language processing and recommender systems, their application to wearable sensor HAR remains nascent. The few published HAR pipelines that incorporate MLOps elements typically address only a subset of the lifecycle (e.g., experiment tracking or containerised inference) without integrating monitoring, drift detection, automated triggering, and self-supervised adaptation into a coherent, stage-based pipeline.

## 1.2 Research Objectives

This thesis designs, implements, and evaluates an end-to-end MLOps pipeline for wrist-worn HAR in an anxiety detection context. The pipeline spans the complete lifecycle from raw sensor data ingestion through inference, monitoring, adaptation, and deployment. The specific research objectives are:

1. **Continuous monitoring without labels.** Develop a multi-signal monitoring framework that detects model degradation using only unlabelled production predictions. The framework combines confidence distribution analysis, temporal consistency checks (activity flip rate, dwell time), and statistical drift detection (Population Stability Index, Kolmogorov–Smirnov test) into a composite health signal.

2. **Automated retraining trigger.** Design a trigger policy engine that decides when retraining is necessary based on monitoring outputs. The engine uses a multi-metric voting scheme (two-of-three confirmation) with tiered alerting (INFO → WARNING → CRITICAL) and cooldown periods to prevent false alarms.

3. **Self-supervised adaptation.** Implement a curriculum pseudo-labeling strategy that enables the model to adapt to new data distributions without manual annotation. The strategy progressively lowers the confidence threshold across iterations, incorporates Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting, and uses class-balanced sampling to maintain prediction diversity.

4. **Calibration and uncertainty quantification.** Integrate temperature scaling and Monte Carlo Dropout to produce calibrated confidence estimates and epistemic uncertainty measurements. These quantities serve both as monitoring inputs and as selection criteria for active learning sample export.

5. **Idempotent data ingestion.** Design the ingestion stage to handle heterogeneous sensor file formats (Garmin Connect Excel archives and per-session Decoded CSV triplets) through a unified, replayable interface. Re-running the pipeline on the same input must produce identical outputs.

6. **Safe model deployment with rollback.** Implement a model registration and rollback mechanism that versions every trained model, performs proxy validation before promotion to production, and enables instant rollback to a prior version if post-deployment monitoring detects regression.

7. **Reproducibility and auditability.** Ensure that every pipeline execution is fully logged through MLflow experiment tracking, DVC data versioning, and deterministic artifact naming, enabling any prior result to be reproduced from its recorded configuration.

## 1.3 Contributions

> **What is new.** This thesis contributes a complete, openly documented MLOps pipeline for wrist-worn HAR that closes the loop between inference, monitoring, and adaptation — a loop that existing HAR research leaves open. The specific contributions are:

- A **14-stage production pipeline** that decomposes the HAR MLOps lifecycle into independently executable, composable stages with typed configuration and artifact dataclasses.
- A **three-layer monitoring framework** (confidence, temporal, drift) that operates entirely on unlabelled predictions and feeds into a voting-based trigger policy.
- A **curriculum pseudo-labeling procedure** with EWC regularisation that adapts a pre-trained model to new user data without ground-truth labels.
- A **sensor placement adaptation module** that detects wrist-side placement (dominant vs. non-dominant) and applies axis mirroring augmentation to compensate for domain shift.
- A **calibrated uncertainty pipeline** combining temperature scaling and MC Dropout for both monitoring and active learning sample selection.
- Integration of these components into a **containerised, CI/CD-enabled deployment** with Docker, GitHub Actions, Prometheus metrics, and Grafana dashboards.

## 1.4 Scope and Delimitations

The scope of this thesis is bounded as follows. The sensor modality is restricted to wrist-worn triaxial accelerometers and gyroscopes (6-axis IMU), as collected by Garmin smartwatches. The classification target is an 11-class activity taxonomy derived from an anxiety-related behavioural protocol. The deep learning architecture is a fixed 1D-CNN-BiLSTM; architecture search is not within scope. The pipeline is designed for single-user adaptation scenarios; federated or multi-user aggregation is identified as future work.

The thesis does not collect new labelled data. The primary dataset comprises 26 recording sessions from the Garmin Decoded export format, each producing accelerometer, gyroscope, and metadata CSV files. Labelled training data originate from a prior data collection phase. This constraint is intentional: the pipeline is designed to operate precisely in the scenario where new labelled data are unavailable.

## 1.5 Thesis Structure

The remainder of this thesis is organised as follows. Chapter 2 reviews the literature across four domains: human activity recognition with deep learning, MLOps practices for continuous ML systems, domain adaptation and transfer learning for sensor data, and uncertainty quantification in neural classifiers. Chapter 3 presents the methodology, describing the 14-stage pipeline architecture, the monitoring and trigger framework, and the adaptation strategies. Chapter 4 details the implementation, covering the repository structure, CLI design, CI/CD workflow, experiment tracking, and containerised deployment. Chapter 5 reports the experimental results, including classification performance, calibration quality, drift detection accuracy, pseudo-labeling convergence, and robustness under simulated sensor degradation. Chapter 6 discusses the findings, identifies limitations, and outlines directions for future work.

[FIGURE: thesis_structure_overview — Visual map of how chapters relate to pipeline stages]

---

*TODO: Add figure showing the mapping between thesis chapters and pipeline lifecycle stages.*

---

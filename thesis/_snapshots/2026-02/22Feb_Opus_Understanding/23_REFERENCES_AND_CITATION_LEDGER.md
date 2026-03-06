# 23 — References and Citation Ledger

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22

---

## 1 Internal Code Citations Used in Phase 2

### File 10 — Ingestion, Preprocessing, QC

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| 3 ingestion paths | `src/components/data_ingestion.py` | L50-120 | ✅ |
| merge_asof 20ms tolerance | `src/components/data_ingestion.py` | L200+ | ✅ |
| 50Hz resampling | `src/components/data_ingestion.py` | L250+ | ✅ |
| Manifest-based skip | `src/components/data_ingestion.py` | L350+ | ✅ |
| UnifiedPreprocessor | `src/preprocess_data.py` | L1-779 | ✅ |
| UnitDetector milliG | `src/preprocess_data.py` | L150+ | ✅ |
| GravityRemover Butterworth | `src/preprocess_data.py` | L250+ | ✅ |
| DomainCalibrator | `src/preprocess_data.py` | L400+ | ✅ |
| data_transformation delegates | `src/components/data_transformation.py` | L1-130 | ✅ |

### File 11 — Training, Evaluation, Inference

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| 1D-CNN-BiLSTM architecture | `src/train.py` | L300-450 | ✅ |
| TrainingConfig 17 params | `src/train.py` | L50-100 | ✅ |
| 5-fold stratified CV | `src/train.py` | L500-600 | ✅ |
| DomainAdaptationTrainer | `src/train.py` | L700-900 | ✅ |
| DANN/MMD → pseudo_label redirect | `src/train.py` | L800+ | ✅ |
| Batch inference Component 4 | `src/components/` | — | ✅ |
| MLflowTracker | `src/train.py` | L1000+ | ✅ |

### File 12 — Monitoring 3-Layer Deep-Dive

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| PostInferenceMonitor class | `scripts/post_inference_monitoring.py` | L1-400+ | ✅ |
| Layer 1 confidence thresholds | `scripts/post_inference_monitoring.py` | L70-90 | ✅ |
| Layer 2 temporal analysis | `scripts/post_inference_monitoring.py` | L100-180 | ✅ |
| Layer 3 Wasserstein drift | `scripts/post_inference_monitoring.py` | L180-280 | ✅ |
| API monitoring divergent thresholds | `src/api/app.py` | L400+ | ✅ |
| Overall status determination | `scripts/post_inference_monitoring.py` | L300-350 | ✅ |

### File 13 — Drift, Calibration, Adaptation

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| WassersteinDriftDetector | `src/wasserstein_drift.py` | L1-200+ | ✅ |
| 2-of-6 / 4-of-6 voting | `src/wasserstein_drift.py` | L150+ | ✅ |
| TemperatureScaler | `src/calibration.py` | L1-100 | ✅ |
| MCDropoutEstimator | `src/calibration.py` | L100-200 | ✅ |
| AdaBN adapt_bn_statistics | `src/domain_adaptation/adabn.py` | full | ✅ |
| TENT 3 safety gates | `src/domain_adaptation/tent.py` | full | ✅ |
| Calibrated pseudo-labeling | `src/components/model_retraining.py` | L200+ | ✅ |
| is_better=True placeholder | `src/components/model_registration.py` | L50+ | ✅ |

### File 14 — Trigger, Governance, Rollback

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| TriggerPolicyEngine | `src/trigger_policy.py` | L1-822 | ✅ |
| 3-signal categories, 17 params | `src/trigger_policy.py` | L50-150 | ✅ |
| 2-of-3 voting | `src/trigger_policy.py` | L300-400 | ✅ |
| 5 action enums | `src/trigger_policy.py` | L30-45 | ✅ |
| 4 placeholder zeros | `src/components/trigger_evaluation.py` | L40-60 | ✅ |
| ModelRegistry JSON | `src/model_rollback.py` | L50-150 | ✅ |
| RollbackValidator | `src/model_rollback.py` | L300-400 | ✅ |
| baseline_update promote_to_shared | `src/components/baseline_update.py` | full | ✅ |

### File 15 — API, Docker, CI/CD, Tests

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| FastAPI 3 endpoints | `src/api/app.py` | L1-775 | ✅ |
| Embedded HTML dashboard | `src/api/app.py` | L500-775 | ✅ |
| Dockerfile.inference | `docker/Dockerfile.inference` | full | ✅ |
| Dockerfile.training | `docker/Dockerfile.training` | full | ✅ |
| docker-compose 4 services | `docker-compose.yml` | L1-143 | ✅ |
| CI/CD 7 jobs | `.github/workflows/ci-cd.yml` | L1-236 | ✅ |
| 215 tests, 19 files | `tests/` directory | all | ✅ |

### File 16 — Cross-Dataset Comparison

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| 26 datasets claimed | `batch_process_all_datasets.py` | L5 | ✅ |
| Dataset discovery glob | `batch_process_all_datasets.py` | L30-44 | ✅ |
| Z-score drift formula | `scripts/analyze_drift_across_datasets.py` | L66 | ✅ |
| milliG unit detection | `scripts/analyze_drift_across_datasets.py` | L71-72 | ✅ |
| Quantile threshold recommendations | `scripts/analyze_drift_across_datasets.py` | L96-99 | ✅ |
| Per-channel drift stats | `scripts/analyze_drift_across_datasets.py` | L103-108 | ✅ |
| Batch comparison report | `batch_process_all_datasets.py` | L170-300 | ✅ |
| Hardcoded CSV path | `generate_summary_report.py` | L9 | ✅ |

### File 17 — Prometheus/Grafana Decision

| Citation | File | Lines | Verified |
|----------|------|-------|----------|
| 18 metric definitions | `src/prometheus_metrics.py` | L49-145 | ✅ |
| MetricsExporter singleton | `src/prometheus_metrics.py` | L233-244 | ✅ |
| HTTP metrics server | `src/prometheus_metrics.py` | L532-555 | ✅ |
| record_from_monitoring_report | `src/prometheus_metrics.py` | L377-420 | ✅ |
| 5 Prometheus scrape targets | `config/prometheus.yml` | L31-67 | ✅ |
| 14 alert rules, 5 groups | `config/alerts/har_alerts.yml` | L1-191 | ✅ |
| No Prometheus in docker-compose | `docker-compose.yml` | L1-143 | ✅ |
| Grafana dashboard reference | `docs/thesis/chapters/CH4_IMPLEMENTATION.md` | L238 | ✅ |
| Grafana file does not exist | `file_search("**/grafana*")` | — | ✅ |

---

## 2 Research Paper References

The repository includes two curated research paper collections:

| File | Papers | Themes |
|------|--------|--------|
| `Comprehensive_Research_Paper_Table_Across_All_HAR_Topics.csv` | 37 papers | 7 HAR themes |
| `Summary_of_7_Research_Themes_in_HAR.csv` | 7 theme summaries | Sensor drift, adaptation, BiLSTM, UDA, pseudo-labeling, confidence filtering, wearable HAR drift |

---

## 3 External Tool Documentation References

| Tool | Version | Documentation URL |
|------|---------|-------------------|
| TensorFlow/Keras | 2.14+ | https://www.tensorflow.org/api_docs |
| FastAPI | 2.0.0 | https://fastapi.tiangolo.com/ |
| MLflow | (latest) | https://mlflow.org/docs/latest/index.html |
| Prometheus | — | https://prometheus.io/docs/ |
| Docker Compose | v2 | https://docs.docker.com/compose/ |
| pytest | 7.x+ | https://docs.pytest.org/ |
| GitHub Actions | — | https://docs.github.com/en/actions |

---

## 4 Citation Format Guide

All Phase 2 files use inline evidence citations in this format:

```
Code:    (source_file.py:L10-L80)  or  (source_file.py | class:ClassName)
Config:  (config/file.yml:L1-L50)
Test:    (tests/test_x.py)
Artifact: (artifacts/.../file.json)
```

For thesis BibTeX conversion, code citations become footnotes or appendix references, while paper citations map to standard `\cite{}` commands.

---

## 5 Citation TODO Queries (External Sources Needed)

These references are cited or implied in the audit but require proper bibliographic lookup:

| # | Citation TODO | Context | Suggested Search Query |
|--:|-------------|---------|----------------------|
| CT-1 | Bulling, A., Blanke, U., & Schiele, B. (2014). A tutorial on HAR using body-worn inertial sensors. *ACM Computing Surveys*. | Ch 2.1 — foundational HAR survey | `"Bulling" "tutorial" "human activity recognition" "body-worn"` |
| CT-2 | Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition. *Sensors*. | Ch 2.1 — CNN-LSTM architecture | `"Ordóñez" "Roggen" "deep convolutional LSTM" "activity recognition"` |
| CT-3 | Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*. | Ch 2.5 / Ch 3.8 — temperature scaling | `"Guo" "calibration" "modern neural networks" ICML 2017` |
| CT-4 | Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR 2021*. | Ch 2.3 / Ch 3.6 — TENT method | `"Wang" "Tent" "test-time adaptation" "entropy minimization" ICLR 2021` |
| CT-5 | Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018). Adaptive batch normalization for practical domain adaptation. *Pattern Recognition*. | Ch 2.3 / Ch 3.6 — AdaBN method | `"Li" "adaptive batch normalization" "domain adaptation"` |
| CT-6 | Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*. | Ch 2.7 — ML technical debt | `"Sculley" "hidden technical debt" "machine learning systems" NeurIPS 2015` |
| CT-7 | Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML production readiness. *NIPS 2017 Workshop*. | Ch 2.7 — ML test score | `"Breck" "ML test score" "production readiness" 2017` |
| CT-8 | Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS 2020*. | File 20: IMP-14 — energy OOD | `"Liu" "energy-based" "out-of-distribution detection" NeurIPS 2020` |
| CT-9 | Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method. *ICML Workshop*. | Ch 2.4 — pseudo-label origins | `"Lee" "pseudo-label" "semi-supervised" ICML 2013` |
| CT-10 | Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS 2019*. | Ch 2.6 — drift detection methods | `"Rabanser" "failing loudly" "dataset shift" NeurIPS 2019` |
| CT-11 | Google. (n.d.). TensorFlow Data Validation (TFDV) documentation. | File 20: IMP-06 — data quality gate | `TensorFlow data validation TFDV documentation` |
| CT-12 | Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *NeurIPS 2019*. | File 20: IMP-16 — conformal monitoring | `"Romano" "conformalized quantile regression" NeurIPS 2019` |

---

## 6 Phase 3 Cross-References

### File 20 — Improvement Roadmap Code Citations

| Improvement | Code Reference | Lines |
|------------|---------------|-------|
| IMP-01: Wire stages 11-14 | `src/pipeline/production_pipeline.py` | `ALL_STAGES` list |
| IMP-02: Fix placeholder zeros | `src/components/trigger_evaluation.py` | L40-60 |
| IMP-03: Replace is_better=True | `src/components/model_registration.py` | L50+ |
| IMP-07: Temperature scaling | `src/calibration.py:TemperatureScaler` | L1-100 |
| IMP-11: Unify monitoring thresholds | `scripts/post_inference_monitoring.py` vs `src/api/app.py` | L70-90 vs L400+ |

### File 21 — Thesis Blueprint Chapter References

| Chapter | Primary Code Sources |
|---------|---------------------|
| Ch 3 Methodology | All `src/` modules |
| Ch 4 Implementation | `docker-compose.yml`, `.github/workflows/ci-cd.yml`, `tests/` |
| Ch 5 Evaluation | `batch_process_all_datasets.py`, `scripts/analyze_drift_across_datasets.py` |

### File 22 — Mermaid Diagram Sources

| Diagram | Original Phase 2 Source |
|---------|------------------------|
| D-2 through D-3 | File 12 §5, §6 |
| D-4 through D-5 | File 13 §4, §5 |
| D-6 through D-7 | File 14 §3, §8 |
| D-1, D-8, D-9, D-10 | New for Phase 3 |

---

## 7 BibTeX Template for Thesis

```bibtex
% === Core Architecture References ===
@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={ICML},
  year={2017}
}

@inproceedings{wang2021tent,
  title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Saining and Olshausen, Bruno and Darrell, Trevor},
  booktitle={ICLR},
  year={2021}
}

@article{li2018adabn,
  title={Adaptive Batch Normalization for Practical Domain Adaptation},
  author={Li, Yanghao and Wang, Naiyan and Shi, Jianping and Liu, Jiaying and Hou, Xiaodi},
  journal={Pattern Recognition},
  year={2018}
}

@inproceedings{sculley2015debt,
  title={Hidden Technical Debt in Machine Learning Systems},
  author={Sculley, David and Holt, Gary and Golovin, Daniel and others},
  booktitle={NeurIPS},
  year={2015}
}

@inproceedings{breck2017mltest,
  title={The {ML} Test Score: A Rubric for {ML} Production Readiness and Technical Debt Reduction},
  author={Breck, Eric and Cai, Shanqing and Nielsen, Eric and Salib, Michael and Sculley, D.},
  booktitle={IEEE BigData},
  year={2017}
}

% === Add remaining entries after CT-1 through CT-12 lookup ===
```

---

## TODO: Phase 3 Additions

- Inventory all citations from Phase 3 files (20-28) when written
- Cross-reference with `docs/research/appendix-paper-index.md` if it exists
- Add page-level verification for key paper claims

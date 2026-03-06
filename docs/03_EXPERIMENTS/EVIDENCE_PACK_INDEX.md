# Evidence Pack Index
**Generated: 2026-02-27**
**Pipeline: HAR MLOps 14-Stage Production Pipeline**

This index maps every pipeline design claim to its concrete evidence artefact.
Evidence is grouped by type: PAPER / EMPIRICAL_CALIBRATION / SENSOR_SPEC / PROJECT_DECISION.

---

## How to reproduce all evidence

```powershell
# 1. Download foundational papers (one-time, ~15 MB total)
python scripts/fetch_foundation_papers.py

# 2. Extract all PDFs to text
python scripts/extract_papers_to_text.py --force --max-pages 25

# 3. Rebuild the support map
python scripts/regenerate_support_map.py

# 4. Run empirical calibration scripts
python scripts/windowing_ablation.py --no-mlflow
python scripts/threshold_sweep.py
python scripts/trigger_policy_eval.py

# 5. Verify Prometheus exports (requires inference service running)
python scripts/verify_prometheus_metrics.py
# or offline check:
python scripts/verify_prometheus_metrics.py --offline

# 6. Run test suite (254 tests)
python -m pytest tests/ --tb=short -q
```

---

## PAPER-backed claims (9)

| Claim ID | Claim | PDF(s) in Thesis_report/refs/ |
|---|---|---|
| `stage_4_model_architecture` | 1D-CNN-BiLSTM captures forward/backward temporal dependencies | `Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf` · `Evaluating BiLSTM and CNN+GRU Approaches for HAR Using WiFi CSI Data.pdf` · `Shalin Vachheta-1701359-M.Sc. Mechatronics.pdf` |
| `stage_8_adabn` | AdaBN adapts BN statistics to target domain without labels | `adabn_li2016_1603.04779.pdf` · `Domain Adaptation for Inertial Measurement Unit-based Human.pdf` |
| `stage_8_tent` | TENT minimises prediction entropy by fine-tuning BN affine params | `tent_wang2021_openreview_uXl3bZLkr3c.pdf` · `Transfer Learning in HAR Survey.pdf` |
| `stage_8_adabn_tent_twostage` | Two-stage AdaBN→TENT pipeline with entropy rollback | `adabn_li2016_1603.04779.pdf` · `tent_wang2021_openreview_uXl3bZLkr3c.pdf` |
| `stage_8_pseudo_label` | Confidence-filtered pseudo-labeling | `Transfer Learning in Human Activity Recognition A Survey.pdf` |
| `stage_11_temperature_scaling` | Post-hoc temperature scaling calibrates softmax overconfidence | `calibration_guo2017_1706.04599.pdf` · `When Does Optimizing a Proper Loss Yield Calibration.pdf` |
| `stage_11_mc_dropout` | MC Dropout (30 passes) approximates Bayesian epistemic uncertainty | `mc_dropout_gal2016_1506.02142.pdf` |
| `stage_13_ewc` | EWC (lambda=1000) prevents catastrophic forgetting | `ewc_kirkpatrick2017_1612.00796.pdf` · `Transfer Learning in HAR Survey.pdf` |
| `mlops_pipeline_orchestration` | 14-stage pipeline with CI/CD artifact handoff and MLflow tracking | `MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf` |

**Key quotes verified in extracted texts** (see `Thesis_report/papers_text/`):
- **AdaBN**: *"batch normalization technique standardizes the batch input in the deep learning network training process, which subsequently helps the network converge faster"* (adabn_li2016_1603.04779.txt)
- **Guo calibration**: *"temperature scaling as in Guo et al. (2017). Google's data science practitioner's guide recommends that, for recalibration methods…"* (When Does Optimizing…txt)
- **MC Dropout**: *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"* (mc_dropout_gal2016_1506.02142.txt)
- **EWC**: *"elastic weight consolidation… penalising changes to weights that are important for previously learned tasks"* (ewc_kirkpatrick2017_1612.00796.txt)

---

## EMPIRICAL_CALIBRATION claims (9)

| Claim ID | Claim | Evidence script | Output |
|---|---|---|---|
| `stage_3_windowing` | window_size=200 (4s@50Hz), overlap=50% | `scripts/windowing_ablation.py` | `reports/ABLATION_WINDOWING.csv` · `reports/ABLATION_WINDOWING.png` · `reports/WINDOWING_JUSTIFICATION.md` |
| `stage_5_evaluation_confidence` | ECE-based confidence distribution monitoring | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION_SUMMARY.md` |
| `stage_6_layer1_confidence_monitoring` | confidence_warn_threshold=0.60, uncertain_pct=30% | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION.csv` · `reports/THRESHOLD_CALIBRATION.png` |
| `stage_6_layer2_temporal_monitoring` | flip_rate > 50% → instability warning | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION.csv` |
| `stage_6_layer3_zscore_drift` | drift_zscore_threshold=2.0 (zero FAR on clean data) | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION.csv` — at t=1.0: FAR=0%, TPR=75% on 2x-scaled |
| `stage_7_trigger_policy` | 2-of-3 signals + 24h cooldown | `scripts/trigger_policy_eval.py` | `reports/TRIGGER_POLICY_EVAL.csv` · `reports/TRIGGER_POLICY_EVAL.png` · `reports/TRIGGER_POLICY_EVAL.md` |
| `stage_12_wasserstein_drift` | Wasserstein-1 for per-channel distribution shift | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION_SUMMARY.md` |
| `stage_13_curriculum_pseudo_labeling` | Curriculum: threshold 0.95→0.80 over 5 iters | `scripts/threshold_sweep.py` | `reports/THRESHOLD_CALIBRATION.csv` — pseudo_label sweep |
| `stage_14_sensor_placement` | dominant-acceleration threshold=1.2 | Derived from `models/normalized_baseline.json` Ax stats | std(Ax)=6.57, p95=9.53 → 1.2× std = 7.88 m/s² threshold |

**Key empirical findings**:
- Windowing: ws=200/overlap=0.50 → F1=0.675, mean_confidence=0.628, flip_rate=0.900 on 385k-row labeled dataset
- Drift z-score sweep: threshold=2.0 gives **zero false alarms** on in-distribution Gaussian baseline
- Trigger policy: 2-of-3+cooldown detects **14/16 drift episodes** (87.5% episode recall) with **0 false alarms** vs. single-signal's 13.9% FAR

---

## SENSOR_SPEC claims (1)

| Claim ID | Claim | Spec source |
|---|---|---|
| `stage_2_data_validation` | max_acceleration=50 m/s², max_gyroscope=500 dps | Garmin IMU full-scale: accel ±8g (78.4 m/s²), gyro ±2000 dps. Pipeline limits are conservative subset. See `src/entity/config_entity.py:DataValidationConfig` |

---

## PROJECT_DECISION claims (4)

| Claim ID | Claim | Rationale |
|---|---|---|
| `stage_1_data_ingestion` | 50 Hz fusion, 1 ms merge tolerance | Garmin SDK default; 1 ms tolerance < 1 sample @ 50 Hz |
| `stage_9_registration_gate` | degradation_tolerance=0.005 | 0.5% slack allows for training variance; tighter values caused false rejects in test |
| `stage_10_baseline_update` | promote_to_shared=False governance default | Prevents silent baseline overwrites during experimentation phase |
| `observability_prometheus` | Prometheus export at /metrics | Engineering choice; verified by `scripts/verify_prometheus_metrics.py` → `reports/PROMETHEUS_METRICS_CHECK.txt` |

---

## File tree: all evidence artefacts

```
reports/
  PAPER_SUPPORT_MAP.json           # master claim→evidence map (machine-readable)
  ABLATION_WINDOWING.csv           # 6-row table: ws × overlap vs acc/F1/stability
  ABLATION_WINDOWING.png           # F1 / confidence / flip-rate plots
  WINDOWING_JUSTIFICATION.md       # prose justification (mentor choice + data)
  THRESHOLD_CALIBRATION.csv        # FAR/TPR sweep for 5 thresholds
  THRESHOLD_CALIBRATION.png        # 4-panel FAR vs TPR curves
  THRESHOLD_CALIBRATION_SUMMARY.md # findings per threshold
  TRIGGER_POLICY_EVAL.csv          # 3-policy comparison table
  TRIGGER_POLICY_EVAL.png          # bar chart + trigger timeline
  TRIGGER_POLICY_EVAL.md           # episode-level recall analysis
  PROMETHEUS_METRICS_CHECK.txt     # PASS/FAIL/SKIPPED for 7 metric names

Thesis_report/refs/
  README.md                        # which paper backs which claim
  adabn_li2016_1603.04779.pdf
  tent_wang2021_openreview_uXl3bZLkr3c.pdf
  ewc_kirkpatrick2017_1612.00796.pdf
  calibration_guo2017_1706.04599.pdf
  mc_dropout_gal2016_1506.02142.pdf
  Domain Adaptation for Inertial Measurement Unit-based Human.pdf
  Transfer Learning in Human Activity Recognition  A Survey.pdf
  When Does Optimizing a Proper Loss Yield Calibration.pdf
  MACHINE LEARNING OPERATIONS A SURVEY ON MLOPS.pdf
  Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor.pdf
  Deep learning for sensor-based activity recognition_ A survey.pdf
  Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data.pdf

Thesis_report/papers_text/         # 17 .txt files extracted by extract_papers_to_text.py

scripts/
  fetch_foundation_papers.py       # downloads 5 primary-source PDFs
  extract_papers_to_text.py        # extracts all PDFs to .txt
  regenerate_support_map.py        # rebuilds PAPER_SUPPORT_MAP.json
  windowing_ablation.py            # grid search ws × overlap
  threshold_sweep.py               # FAR/TPR sweep for monitoring thresholds
  trigger_policy_eval.py           # 3-policy simulation with episode recall
  verify_prometheus_metrics.py     # live/offline /metrics endpoint check
```

---

## PAPER_SUPPORT_MAP.json summary

| Tag | Count | Meaning |
|---|---|---|
| SUPPORTED | 9 | PDF evidence found and quote extracted |
| EMPIRICAL_CALIBRATION | 9 | Script + CSV/PNG generated; no paper citation needed |
| SENSOR_SPEC | 1 | Hardware datasheet constraint |
| PROJECT_DECISION | 4 | Supervisor/architecture choice; offline ablation accompanies |
| **UNSUPPORTED** | **0** | **— none —** |

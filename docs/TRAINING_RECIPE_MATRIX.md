# Training Recipe Matrix

**Purpose:** Document the two-dataset training story and define experiments for thesis evaluation.  
**Last Updated:** Feb 19, 2026

---

## 1. Datasets

| Dataset | File | Rows | Channels | Role |
|---------|------|------|----------|------|
| **ADAMSense/2005** (anxiety) | `data/anxiety_dataset.csv` | 709,583 | 25 (pocket `_p` + wrist `_w`, accel+gyro+mag+GPS) | **Pretraining** — multi-user, multi-sensor, 11 anxiety activity classes |
| **Fine-tune** (wrist-only) | `data/all_users_data_labeled.csv` | 385,327 | 6 wrist channels (`Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w`) + `activity` + `User` | **Fine-tuning** — target domain, wrist-only, same 11 classes |
| **Production** (Garmin raw) | `data/raw/2025-*_accelerometer.csv` + `*_gyroscope.csv` | varies per recording | raw Garmin export (milliG / deg/s) | **Inference** — unlabeled, real-world, converted to m/s² at 50 Hz |

**Key relationships:**
- Pretraining learns general anxiety-activity representations from a larger, richer dataset.
- Fine-tuning adapts to the target sensor configuration (wrist-only, 6 channels, 50 Hz).
- Production data is the domain the pipeline monitors and adapts to — no labels available.

---

## 2. Training Recipes (Experiments)

| ID | Recipe | Dataset(s) | Thesis Purpose | Status |
|----|--------|-----------|----------------|--------|
| **T1** | Fine-tune only | `all_users_data_labeled.csv` | **Control baseline** — single-stage supervised training. Answers: "Does pretraining help?" | Needs separate run (or reference MLflow run if available) |
| **T2** | Pretrain → Fine-tune | `anxiety_dataset.csv` → `all_users_data_labeled.csv` | **Current deployed model** — transfer representation from multi-user to wrist-only | ✅ Done — `models/pretrained/fine_tuned_model_1dcnnbilstm.keras` (499K params) |
| **T3** | T2 + AdaBN+TENT on production | T2 model + unlabeled production data | **Unsupervised TTA** — no labels needed, fastest adaptation | = Audit run **A4** |
| **T4** | T2 + Calibrated pseudo-label | T2 model + unlabeled production data | **Semi-supervised adaptation** — calibrated, gated, class-balanced | = Audit run **A5** |

---

## 3. Metrics to Compare

| Metric | T2 baseline (A1) | A3 supervised retrain | T3 AdaBN+TENT (A4) | T4 Pseudo-label (A5) |
|--------|:----------------:|:---------------------:|:-------------------:|:--------------------:|
| **Adaptation method** | — (inference only) | Standard supervised | AdaBN + TENT | Calibrated pseudo-label |
| **Source samples** | — | 3,852 | 0 | 3,852 (558 after self-consistency filter) |
| **Target samples** | — | 0 | 86 | 86 (all used as pseudo-labels) |
| **f1_macro** | — | **0.814** | — | — |
| **cohen_kappa** | — | **0.795** | — | — |
| **val_accuracy** | — | — | — | **0.969** |
| **before_mean_confidence** | — | — | 0.899 | — |
| **after_mean_confidence** | — | — | 0.758 | — |
| **after_norm_entropy** | — | — | 0.281 | — |
| **Monitoring status** | PASS | PASS | ALERT (drift=1.33) | ALERT (drift=1.33) |
| **n_predictions** | 1,027 | 86 | 86 | 86 |
| **Pipeline stages** | 7 | 9 | 9 | 9 |
| **Calibration temperature** | — | — | — | 3.0 |
| **Pipeline status** | SUCCESS | SUCCESS | SUCCESS | SUCCESS |
| **Audit result** | 8/8 PASS | 12/12 PASS | 12/12 PASS | 12/12 PASS |
| **Run ID** | `20260219_115223` | `20260219_124825` | `20260219_125400` | `20260219_134233` |

**Key observations:**
1. **A3 supervised retrain** achieves 81.4% macro-F1 with 5-fold CV (skip-cv, 10 epochs), confirming the fine-tune dataset is usable.
2. **A4 AdaBN+TENT** reduces mean confidence from 89.9% → 75.8% (expected: BN stat update spreads predictions away from over-confident mode toward calibrated uncertainty).
3. **A5 pseudo-label** reaches 96.9% val_accuracy on the held-out 10% of the combined (filtered-source + pseudo-target) set. Self-consistency filtering retains 14.5% of source windows as clean training data.
4. **A1 baseline** produces 1,027 predictions on the full Garmin recording with no adaptation — monitoring shows PASS (no drift detected on larger dataset).

---

## 4. Model Architecture (Shared Across All Recipes)

From `models/pretrained/model_info.json`:

| Property | Value |
|----------|-------|
| Architecture | 1D-CNN + BiLSTM |
| Input shape | `(batch, 200, 6)` — 200 timesteps × 6 channels |
| Output shape | `(batch, 11)` — 11 activity classes |
| Parameters | 499,131 |
| Layers | Conv1D ×2, BatchNorm ×5, Dropout ×5, Bidirectional(LSTM) ×2, Flatten, Dense ×2 |
| Window | 200 samples (4 seconds at 50 Hz), 50% overlap |

---

## 5. The 11 Activity Classes

(Shared between pretrain and fine-tune datasets)

| # | Activity |
|---|----------|
| 1 | ear_rubbing |
| 2 | hand_tapping |
| 3 | hair_pulling |
| 4 | nail_biting |
| 5 | knuckles_cracking |
| 6 | head_scratching |
| 7 | lip_biting |
| 8 | face_touching |
| 9 | teeth_grinding |
| 10 | leg_shaking |
| 11 | finger_tapping |

---

## 6. Thesis Story

> "We pretrained a 1D-CNN-BiLSTM model on the ADAMSense multi-sensor anxiety dataset (709K samples) and fine-tuned on wrist-only data (385K samples) from the same 11 activity classes. In production, the model processes unlabeled Garmin smartwatch recordings. When monitoring detects distribution drift (PSI > threshold), the pipeline offers three adaptation strategies: (1) AdaBN — update BN statistics with zero labels, (2) AdaBN+TENT — additionally fine-tune BN affine parameters via entropy minimisation, (3) calibrated pseudo-labeling — temperature-scaled, entropy-gated, class-balanced self-training. We compare these against the unadapted baseline and a standard supervised retrain."

---

## 7. Notes

- **T1 vs T2 comparison** is a "nice-to-have" ablation (requires a separate training run without pretraining). If MLflow records from earlier experiments exist in `mlruns/`, reference those instead.
- **T3 and T4** are produced by audit runs A4 and A5 — no separate training needed.
- The anxiety dataset has 25 columns; only wrist columns (`_w` suffix) overlap with the fine-tune/production domain. The extra pocket/magnetometer/GPS channels are used during pretraining but dropped when the model is fine-tuned on 6-channel wrist data.

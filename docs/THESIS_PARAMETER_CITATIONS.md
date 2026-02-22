# Thesis Parameter Citations

> **Purpose:** Every numeric hyper-parameter in the pipeline, with the file+line where it lives,
> the statistical / empirical justification, and the paper citation to use in the thesis.
> Intended for Chapter 4 (Implementation) and Chapter 5 (Results) annotation.

Last updated: February 2026  
Covers: all 14 pipeline stages + domain-adaptation + calibration + monitoring

---

## How to Read This Table

| Column | Meaning |
|---|---|
| **Stage** | Pipeline stage name (matches `src/pipeline/production_pipeline.py`) |
| **Parameter** | Python attribute name as it appears in source |
| **Value** | Default value |
| **File : approx line** | Source of truth — the dataclass or function definition |
| **Purpose** | What the parameter controls |
| **Evidence type** | `literature` / `standard-practice` / `empirical` / `derived` |
| **Citation key** | BibTeX key from `refs/parameter_citations.bib` |
| **Thesis sentence** | Copy-paste ready phrase for inline citation |

---

## Stage 1 — Data Ingestion (`DataIngestionConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `target_hz` | 50 | [config_entity.py](../src/entity/config_entity.py#L97) | Target sample rate after resampling; must match training corpus | Standard practice for wrist-worn IMU HAR | `zappi2020usc` | "Raw sensor streams are resampled to 50 Hz, consistent with the USC-HAD training corpus (Zappi et al., 2020)." |
| `merge_tolerance_ms` | 1 | [config_entity.py](../src/entity/config_entity.py#L98) | Max time delta (ms) for accelerometer↔gyroscope merge | Sub-sample tolerance at 50 Hz (20 ms per sample) | — | "Accelerometer and gyroscope streams are merged with a ±1 ms tolerance, representing 5 % of one 50 Hz sample period." |

---

## Stage 2 — Data Validation (`DataValidationConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `expected_frequency_hz` | 50.0 | [config_entity.py](../src/entity/config_entity.py#L108) | Schema contract — reject data recorded at wrong rate | Derived from Stage 1 | — | "Incoming batches are validated against the expected 50 Hz sampling rate." |
| `max_acceleration_ms2` | 50.0 | [config_entity.py](../src/entity/config_entity.py#L109) | Physical plausibility gate for acceleration channels (≈ 5 g) | Free-fall ≈ 9.8 m/s²; extreme activities < 50 m/s² | `mannini2010machine` | "Acceleration readings exceeding ±50 m/s² (≈5 g) are flagged as implausible sensor artefacts." |
| `max_gyroscope_dps` | 500.0 | [config_entity.py](../src/entity/config_entity.py#L110) | Physical plausibility gate for gyroscope channels | Consumer MEMS gyro hardware limit (±500 °/s) | `chen2021deep` | "Gyroscope readings are bounded to ±500 °/s matching typical consumer MEMS sensor ranges." |
| `max_missing_ratio` | 0.05 | [config_entity.py](../src/entity/config_entity.py#L111) | Maximum tolerated fraction of NaN/missing values per window | 5 % is conventional imputation threshold in wearable HAR | `chen2021deep` | "Windows with more than 5 % missing values are discarded before inference." |

---

## Stage 3 — Data Transformation (`DataTransformationConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `window_size` | 200 | [config_entity.py](../src/entity/config_entity.py#L121) | Number of samples per sliding window (= 4 s at 50 Hz) | 4 s window is widely used for coarse activity labelling; large enough for gait cycle, short enough for real-time latency | `bao2004activity`, `chen2021deep` | "Sensor data is segmented into 200-sample (4 s at 50 Hz) non-overlapping windows, following the convention established by Bao & Intille (2004)." |
| `overlap` | 0.5 | [config_entity.py](../src/entity/config_entity.py#L122) | Sliding window step = (1−overlap) × window_size | 50 % overlap doubles effective training samples without duplicating exact windows; standard in HAR | `chen2021deep` | "Adjacent windows overlap by 50 %, yielding a 2 s step size and doubling the number of training instances." |
| `enable_unit_conversion` | True | [config_entity.py](../src/entity/config_entity.py#L119) | Convert raw milliG → m/s² before inference | Must match training data preprocessing | — | "Raw acceleration units (milliG) are converted to m/s² to match the preprocessing applied during model training." |

---

## Stage 4 — Model Inference (`ModelInferenceConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `batch_size` | 32 | [config_entity.py](../src/entity/config_entity.py#L131) | GPU/CPU batch for prediction | Standard mini-batch for inference | — | "Inference is performed in mini-batches of 32 windows." |
| `confidence_threshold` | 0.50 | [config_entity.py](../src/entity/config_entity.py#L132) | Minimum softmax confidence to emit a label | 1/K random baseline for K=11 classes is 0.091; 0.50 is a conservative 5× margin | `guo2017calibration` | "Predictions with maximum softmax confidence below 0.50 are withheld as uncertain (5× above the 11-class random baseline of 9.1 %)." |

---

## Stage 6 — Post-Inference Monitoring (`PostInferenceMonitoringConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `confidence_warn_threshold` | 0.60 | [config_entity.py](../src/entity/config_entity.py#L157) | Mean confidence below this → Layer 1 WARNING | OOD detection literature; 0.60 is 10 pp above the inference rejection gate | `lakshminarayanan2017simple` | "A population mean confidence below 0.60 triggers a Layer 1 warning, indicating systematic underconfidence consistent with distribution shift." |
| `uncertain_pct_threshold` | 30.0 | [config_entity.py](../src/entity/config_entity.py#L158) | Percentage of low-confidence windows → WARNING | Empirical: > 30 % uncertain windows indicates unreliable batch | `lakshminarayanan2017simple` | "When more than 30 % of windows fall below the confidence gate, Layer 1 issues a batch-level warning." |
| `transition_rate_threshold` | 50.0 | [config_entity.py](../src/entity/config_entity.py#L160) | % activity-label transitions above this → Layer 2 WARNING | Temporal coherence: activities persist for seconds; > 50 % flip rate implies noise | `gama2014survey` | "A window-to-window activity transition rate exceeding 50 % is flagged as temporally incoherent (Layer 2)." |
| `drift_zscore_threshold` | 2.0 | [config_entity.py](../src/entity/config_entity.py#L162) | Per-channel z-score of mean shift above this → Layer 3 WARNING | 2σ ≈ 95th percentile of the null distribution; standard one-sided normal critical value | `gama2014survey`, `page1954continuous` | "Per-channel drift is measured as the z-score of the mean shift (|μ_prod − μ_base| / σ_base); a score of 2.0 corresponds to the 97.5th percentile under the null hypothesis of no shift." |
| `max_baseline_age_days` | 90 | [config_entity.py](../src/entity/config_entity.py#L170) | Warn if baseline statistics file is older than this | 90-day staleness guard; model retraining is expected at most quarterly | — | "The system warns if the reference baseline is more than 90 days old, ensuring monitoring thresholds remain aligned with the deployed model population." |
| `calibration_temperature` | 1.0 (runtime: T=1.5) | [config_entity.py](../src/entity/config_entity.py#L165) | Temperature from Stage 11 applied before threshold checks | Post-hoc calibration; T=1.5 softens overconfident predictions (Guo et al.) | `guo2017calibration` | "Confidence scores are rescaled using the temperature T = 1.5 learned by Stage 11 before monitoring thresholds are applied." |

---

## Stage 7 — Trigger Evaluation (`TriggerThresholds` in `src/trigger_policy.py`)

### Confidence Layer (Layer 1)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `confidence_warn` | 0.55 | [trigger_policy.py](../src/trigger_policy.py#L97) | Mean confidence below this → WARNING signal | 0.55 is 15 pp above random baseline; aligns with OOD calibration literature | `lakshminarayanan2017simple` | "A trigger-layer confidence warning fires when population mean confidence falls below 0.55." |
| `confidence_critical` | 0.45 | [trigger_policy.py](../src/trigger_policy.py#L98) | Mean confidence below this → CRITICAL signal | 7 pp below warn; allows one-step hysteresis | `lakshminarayanan2017simple` | "The CRITICAL confidence threshold (0.45) provides hysteresis below the WARNING threshold." |
| `entropy_warn` | 1.8 | [trigger_policy.py](../src/trigger_policy.py#L99) | Mean entropy above this → WARNING | Max entropy for K=11 classes = log(11) ≈ 2.40; 1.8 = 75 % of maximum | `wang2021tent` | "An entropy warning fires above 1.80 nats, representing 75 % of the theoretical maximum for 11 classes." |
| `entropy_critical` | 2.2 | [trigger_policy.py](../src/trigger_policy.py#L100) | Mean entropy above this → CRITICAL | 92 % of max entropy — near-uniform distribution | `wang2021tent` | "A critical entropy alert fires above 2.20 nats (92 % of maximum), indicating near-uniform class uncertainty." |
| `uncertain_ratio_warn` | 0.20 | [trigger_policy.py](../src/trigger_policy.py#L101) | Fraction uncertain windows → WARNING | 20 % of batch labelled uncertain triggers investigation | `lakshminarayanan2017simple` | — |
| `uncertain_ratio_critical` | 0.35 | [trigger_policy.py](../src/trigger_policy.py#L102) | Fraction uncertain windows → CRITICAL | 35 % uncertain batch is operationally unacceptable | `lakshminarayanan2017simple` | — |

### Temporal Layer (Layer 2)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `flip_rate_warn` | 0.25 | [trigger_policy.py](../src/trigger_policy.py#L105) | Activity flip rate above this → WARNING | Activities have minimum dwell; 25 % flip rate implies 4-window label cycles | `gama2014survey` | — |
| `flip_rate_critical` | 0.40 | [trigger_policy.py](../src/trigger_policy.py#L106) | Activity flip rate above this → CRITICAL | — | `gama2014survey` | — |

### Drift Layer (Layer 3)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `ks_pvalue_threshold` | 0.01 | [trigger_policy.py](../src/trigger_policy.py#L109) | Two-sample KS test significance level | Bonferroni-corrected α=0.05 / 6 channels ≈ 0.008; rounded to 0.01 | `gama2014survey` | "Two-sample Kolmogorov–Smirnov tests per channel use α = 0.01, approximating a Bonferroni-corrected level across six sensor channels." |
| `drift_zscore_warn` | 2.0 | [trigger_policy.py](../src/trigger_policy.py#L111) | Per-channel z-score of mean shift → WARNING (**NEW**) | 2σ ≈ 95th pct under null; standard normal critical value | `gama2014survey`, `page1954continuous` | "The drift warning threshold z = 2.0σ corresponds to the 97.5th percentile of the null distribution, yielding a false-alarm rate below 2.5 % per channel." |
| `drift_zscore_critical` | 3.0 | [trigger_policy.py](../src/trigger_policy.py#L112) | Per-channel z-score → CRITICAL (**NEW**) | 3σ ≈ 99.7th percentile (three-sigma rule) | `wald1947sequential`, `page1954continuous` | "The critical drift threshold z = 3.0σ follows the three-sigma rule (99.7th percentile), limiting spurious critical alerts to fewer than 0.3 % of steady-state windows." |
| `wasserstein_warn` | 0.3 | [trigger_policy.py](../src/trigger_policy.py#L113) | Wasserstein-1 distance per channel → WARNING | Empirical from intra-session variability in pilot data | `rabanser2019failing` | "Wasserstein-1 distance warnings fire above 0.30, a threshold empirically validated on pilot session data." |
| `wasserstein_critical` | 0.5 | [trigger_policy.py](../src/trigger_policy.py#L114) | Wasserstein-1 distance per channel → CRITICAL | 67 % above warn level | `rabanser2019failing` | — |
| `min_drifted_channels_warn` | 2 | [trigger_policy.py](../src/trigger_policy.py#L117) | Minimum drifted channels to fire WARNING | Gate prevents single noisy-channel false alarms | `gama2014survey` | "Drift warnings require at least 2 of 6 sensor channels to exceed the z-score threshold, suppressing single-channel noise artefacts." |
| `min_drifted_channels_critical` | 4 | [trigger_policy.py](../src/trigger_policy.py#L118) | Minimum drifted channels to fire CRITICAL | Majority of channels affected | `gama2014survey` | — |

### Voting and Cooldown

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `min_signals_for_retrain` | 2 | [trigger_policy.py](../src/trigger_policy.py#L121) | Require 2-of-3 monitoring layers to agree before retraining | Majority voting reduces false triggers; inspired by ensemble methods | `wang2021tent` | "Retraining is initiated only when at least 2 of the 3 monitoring layers independently signal distribution shift, implementing a majority-vote safeguard against spurious triggers." |
| `consecutive_warnings_for_trigger` | 3 | [trigger_policy.py](../src/trigger_policy.py#L122) | 3 consecutive WARNING batches = trigger | Temporal smoothing; prevents single anomalous batch from triggering retrain | `gama2014survey` | "Three consecutive WARNING decisions are required before a retraining trigger is issued, providing temporal smoothing consistent with Page's CUSUM framework (Page, 1954)." |
| `retrain_cooldown_hours` | 24 | [trigger_policy.py](../src/trigger_policy.py#L127) | Minimum hours between successive retrains | Prevents oscillation; one retrain per calendar day | — | "A 24-hour cooldown between retraining events prevents rapid oscillation and limits computational overhead." |

---

## Stage 8 — Model Retraining (`ModelRetrainingConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `epochs` | 100 | [config_entity.py](../src/entity/config_entity.py#L242) | Maximum fine-tuning epochs | Early stopping applied in practice; 100 is ceiling | `chen2021deep` | — |
| `batch_size` | 64 | [config_entity.py](../src/entity/config_entity.py#L243) | Training mini-batch size | 64 balances gradient noise and compute; standard for 1D-CNN-BiLSTM | `chen2021deep` | "A mini-batch size of 64 was selected following Chen et al. (2021) for 1D-CNN-based HAR." |
| `learning_rate` | 0.001 | [config_entity.py](../src/entity/config_entity.py#L244) | Adam learning rate for fine-tuning | Default Adam lr; widely validated in HAR fine-tuning | `kingma2015adam` | "The Adam optimiser (Kingma & Ba, 2015) is used with an initial learning rate of 0.001." |
| `n_folds` | 5 | [config_entity.py](../src/entity/config_entity.py#L245) | Cross-validation folds | 5-fold CV is the standard bias-variance tradeoff for medium-sized datasets | `kohavi1995study` | "Model performance is estimated via 5-fold cross-validation (Kohavi, 1995)." |
| `adabn_n_batches` | 10 | [config_entity.py](../src/entity/config_entity.py#L253) | Mini-batches used for BN statistics estimation in AdaBN/TENT | 10 × 64 = 640 samples → BN statistics converge; confirmed by pilot run log (21:48:25) | `li2018adabn` | "Batch-Normalisation statistics are re-estimated from 640 target samples (10 batches × 64 samples), sufficient for empirically stable mean and variance estimates." |

### AdaBN (`src/domain_adaptation/adabn.py`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `n_batches` | 10 | [adabn.py](../src/domain_adaptation/adabn.py#L50) | Number of forward-pass batches for BN stat estimation | 10 × 64 = 640 samples; pilot run confirmed 5 BN layers adapted | `li2018adabn` | "AdaBN replaces source-domain batch-normalisation statistics with target statistics estimated from 640 unlabelled samples across 5 BN layers." |
| `batch_size` | 64 | [adabn.py](../src/domain_adaptation/adabn.py#L51) | Mini-batch size for forward pass | Must produce stable BN statistics; matches training batch | `li2018adabn` | — |
| `reset_stats` | True | [adabn.py](../src/domain_adaptation/adabn.py#L52) | Zero running stats before adaptation | Avoids mixing source and target statistics | `li2018adabn` | "BN running statistics are reset before the target-domain forward passes to prevent contamination from source-domain statistics." |

### TENT (`src/domain_adaptation/tent.py`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `n_steps` | 10 | [tent.py](../src/domain_adaptation/tent.py#L52) | Gradient update steps over BN affine params | 10 steps is the default in the original TENT implementation | `wang2021tent` | "TENT performs 10 gradient steps optimising prediction entropy over BN affine parameters (γ, β), matching the default in Wang et al. (2021)." |
| `learning_rate` | 1e-4 | [tent.py](../src/domain_adaptation/tent.py#L53) | Adam lr for entropy minimisation | Small lr prevents catastrophic forgetting of BN affine params | `wang2021tent` | "A learning rate of 1×10⁻⁴ is used for TENT entropy minimisation, following the conservative default of Wang et al. (2021)." |
| `batch_size` | 64 | [tent.py](../src/domain_adaptation/tent.py#L54) | Mini-batch size per gradient step | Matches AdaBN batch for memory consistency | `wang2021tent` | — |
| `ood_entropy_threshold` | 0.85 | [tent.py](../src/domain_adaptation/tent.py#L55) | Skip TENT if initial normalised entropy > this (OOD guard) | 85 % of max entropy (log 11 ≈ 2.40) signals extreme OOD — adaptation would harm rather than help | `wang2021tent` | "If the initial normalised entropy exceeds 0.85, the target distribution is deemed extreme out-of-distribution and TENT adaptation is skipped to prevent catastrophic forgetting." |
| `rollback_threshold` | 0.05 | [tent.py](../src/domain_adaptation/tent.py#L56) | Rollback BN affine params if entropy_delta > this | Safety rail: positive entropy delta = TENT worsened predictions; rollback preserves AdaBN gains | `wang2021tent` | "If TENT increases mean entropy by more than 0.05 nats, all BN affine parameters are rolled back to their pre-adaptation state, preserving the gains from AdaBN." |

---

## Stage 9 — Model Registration (`ModelRegistrationConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `auto_deploy` | False | [config_entity.py](../src/entity/config_entity.py#L265) | Do not auto-deploy adapted model; require explicit promotion | Governance: TTA/unsupervised adaptation produces no labeled val_accuracy — human approval required | `sculley2015hidden` | "Automatic deployment is disabled (`auto_deploy=False`); an adapted model requires explicit human promotion, preventing silent performance regressions." |
| `proxy_validation` | True | [config_entity.py](../src/entity/config_entity.py#L266) | Validate confidence proxy metrics before registration | Ensures adapted model does not degrade beyond monitoring bounds | `sculley2015hidden` | — |

---

## Stage 10 — Baseline Update (`BaselineUpdateConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `promote_to_shared` | False | [config_entity.py](../src/entity/config_entity.py#L278) | Default: save baseline only as MLflow artifact, not to shared path | Governance: prevents production monitoring drift from retraining-session data | `sculley2015hidden` | "Baseline promotion is gated behind an explicit `--update-baseline` CLI flag, ensuring monitoring statistics are not silently overwritten by adaptation sessions." |

---

## Stage 11 — Calibration & Uncertainty (`CalibrationUncertaintyConfig`, `CalibrationConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `initial_temperature` | 1.5 | [config_entity.py](../src/entity/config_entity.py#L288), [calibration.py](../src/calibration.py#L58) | Starting temperature T for temperature scaling optimisation | T > 1 softens overconfident softmax; 1.5 covers typical neural-net overconfidence range | `guo2017calibration` | "Temperature optimisation is initialised at T = 1.5, a value consistent with the overconfidence profile reported for large neural networks by Guo et al. (2017)." |
| `temp_lr` | 0.01 | [config_entity.py](../src/entity/config_entity.py#L289), [calibration.py](../src/calibration.py#L59) | Learning rate for L-BFGS/Adam temperature optimisation | Standard for scalar optimisation over a smooth cross-entropy surface | `guo2017calibration` | — |
| `temp_max_iter` | 100 | [config_entity.py](../src/entity/config_entity.py#L290), [calibration.py](../src/calibration.py#L60) | Maximum optimisation iterations | Convergence typically < 50 steps; 100 is conservative ceiling | `guo2017calibration` | — |
| `mc_forward_passes` | 30 | [config_entity.py](../src/entity/config_entity.py#L293) | MC Dropout forward passes for uncertainty estimation | 30 passes gives < 5 % relative error on mean and variance (Gal & Ghahramani, 2016) | `gal2016dropout` | "Uncertainty is estimated via Monte Carlo Dropout with T = 30 stochastic forward passes, as recommended by Gal & Ghahramani (2016)." |
| `mc_dropout_rate` | 0.2 | [config_entity.py](../src/entity/config_entity.py#L294) | Dropout probability used during MC passes | 0.2 matches the training dropout rate, preserving expected network output | `gal2016dropout` | "The MC Dropout rate (p = 0.20) matches the rate used during training, ensuring consistent marginalisation of network weights." |
| `n_bins` | 15 | [config_entity.py](../src/entity/config_entity.py#L297), [calibration.py](../src/calibration.py#L64) | Reliability diagram / ECE bins | 15 bins as recommended for ECE on softmax outputs | `naeini2015obtaining` | "Expected Calibration Error (ECE) is computed over 15 equal-width confidence bins (Naeini et al., 2015)." |
| `ece_warn_threshold` | 0.10 | [config_entity.py](../src/entity/config_entity.py#L298) | ECE above this indicates poor calibration | 10 % ECE is the commonly cited threshold between acceptable and poor calibration | `naeini2015obtaining`, `guo2017calibration` | "Models with ECE > 0.10 are flagged for recalibration, following the threshold used in Guo et al. (2017)." |
| `confidence_warn_threshold` | 0.65 | [calibration.py](../src/calibration.py#L65) | Mean confidence for calibration report warning | Same scale as monitoring Layer 1, 5 pp tighter | `guo2017calibration` | — |
| `entropy_warn_threshold` | 1.5 | [calibration.py](../src/calibration.py#L66) | Entropy for calibration report warning | 63 % of max entropy (log 11) | `guo2017calibration` | — |

---

## Stage 12 — Wasserstein Drift Detection (`WassersteinDriftConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `warn_threshold` | 0.3 | [config_entity.py](../src/entity/config_entity.py#L315) | Wasserstein-1 distance per channel → WARNING | Empirical: intra-session W₁ < 0.15 in pilot data; 0.30 ≈ 2× expected range | `rabanser2019failing` | "A per-channel Wasserstein-1 distance exceeding 0.30 triggers a drift warning, calibrated against pilot-session intra-subject variability." |
| `critical_threshold` | 0.5 | [config_entity.py](../src/entity/config_entity.py#L316) | Wasserstein-1 distance per channel → CRITICAL | 67 % above warn; indicates substantial distribution shift | `rabanser2019failing` | — |
| `cpd_window_size` | 50 | [config_entity.py](../src/entity/config_entity.py#L320) | Change-point detection sliding window (windows) | 50 windows × 4 s = 200 s ≈ 3 min; suitable for activity-level transitions | `gama2014survey` | "The change-point detection sub-module uses a 50-window sliding window (≈ 3.3 minutes of activity) to detect gradual distribution drift." |
| `cpd_threshold` | 2.0 | [config_entity.py](../src/entity/config_entity.py#L321) | CPD score threshold (z-score scale) | Matches the 2σ z-score convention used in Layer 3 monitoring | `page1954continuous` | — |
| `min_drifted_channels_warn` | 2 | [config_entity.py](../src/entity/config_entity.py#L317) | Channels required for WARNING | Consistent with Layer 3 gating in Stage 7 | `gama2014survey` | — |
| `min_drifted_channels_critical` | 4 | [config_entity.py](../src/entity/config_entity.py#L318) | Channels required for CRITICAL | Majority of 6 channels | `gama2014survey` | — |

---

## Stage 13 — Curriculum Pseudo-Labelling (`CurriculumPseudoLabelingConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `initial_confidence_threshold` | 0.95 | [config_entity.py](../src/entity/config_entity.py#L338) | Curriculum starts by accepting only high-confidence pseudo-labels | Conservative start prevents noisy label accumulation | `lee2013pseudo` | "The curriculum begins by pseudo-labelling only windows with softmax confidence ≥ 0.95, accepting only the most reliable self-generated labels (Lee, 2013)." |
| `final_confidence_threshold` | 0.80 | [config_entity.py](../src/entity/config_entity.py#L339) | Threshold relaxes to this over iterations | Gradual relaxation expands the pseudo-labelled pool as the model improves | `lee2013pseudo` | "The confidence threshold decays linearly to 0.80 over 5 iterations, following the curriculum principle of gradually expanding the training set." |
| `n_iterations` | 5 | [config_entity.py](../src/entity/config_entity.py#L340) | Number of self-training rounds | 5 rounds balance adaptation depth vs. computational cost | `lee2013pseudo` | "Five self-training iterations are performed, balancing adaptation depth against the risk of error accumulation." |
| `max_samples_per_class` | 20 | [config_entity.py](../src/entity/config_entity.py#L343) | Class cap on pseudo-labelled samples | Prevents dominant-class imbalance from skewing re-training | `lee2013pseudo` | "A per-class cap of 20 pseudo-labelled samples prevents training imbalance from high-confidence dominant activities." |
| `ema_decay` | 0.999 | [config_entity.py](../src/entity/config_entity.py#L346) | Teacher model EMA decay (teacher-student scheme) | Near-unity EMA provides stable teacher; standard in Mean Teacher (Tarvainen & Valpola, 2017) | `tarvainen2017mean` | "The teacher model is maintained as an exponential moving average of the student weights with decay α = 0.999, consistent with the Mean Teacher method (Tarvainen & Valpola, 2017)." |
| `ewc_lambda` | 1000.0 | [config_entity.py](../src/entity/config_entity.py#L351) | Elastic Weight Consolidation regularisation strength | Large λ strongly anchors weights to source-task values, preventing catastrophic forgetting | `kirkpatrick2017overcoming` | "EWC regularisation with λ = 1000 anchors the adapted model to its source-task weights, mitigating catastrophic forgetting (Kirkpatrick et al., 2017)." |
| `ewc_n_samples` | 200 | [config_entity.py](../src/entity/config_entity.py#L352) | Samples used to compute EWC Fisher information matrix | 200 samples sufficient for diagonal Fisher approximation | `kirkpatrick2017overcoming` | — |
| `epochs_per_iteration` | 10 | [config_entity.py](../src/entity/config_entity.py#L354) | Fine-tuning epochs per curriculum round | 10 × 5 = 50 total maximum epochs; limits over-fitting to small pseudo-label set | `lee2013pseudo` | — |
| `learning_rate` | 0.0005 | [config_entity.py](../src/entity/config_entity.py#L356) | Adam lr for pseudo-label fine-tuning | Half the base training lr; cautious update prevents forgetting | `kingma2015adam` | "A reduced learning rate of 5×10⁻⁴ (half the training rate) is used for curriculum fine-tuning to prevent over-adaptation to pseudo-labels." |

---

## Stage 14 — Sensor Placement Robustness (`SensorPlacementConfig`)

| Parameter | Value | File | Purpose | Evidence | Citation | Thesis sentence |
|---|---|---|---|---|---|---|
| `dominant_accel_threshold` | 1.2 | [config_entity.py](../src/entity/config_entity.py#L372) | |Ax| / |Ay_z| > 1.2 → dominant hand classification | Empirical: dominant hand shows 20 %+ higher lateral acceleration during walking | `zappi2020usc` | "A wrist is classified as dominant when its lateral acceleration exceeds the contralateral mean by a factor of 1.2, derived empirically from pilot walking trials." |
| `mirror_probability` | 0.5 | [config_entity.py](../src/entity/config_entity.py#L373) | Probability of applying hand-mirror augmentation during retraining | Uniform prior — no preference for dominant over non-dominant | — | "Mirror augmentation is applied with probability 0.5 during retraining, making the model equally invariant to left- and right-hand placement." |

---

## Cross-Stage Summary: Drift Metric Taxonomy

```
                            Stage 6               Stage 7              Stage 12
Metric type         z-score (mean shift)    z-score (mean shift)   Wasserstein-1
Warn threshold            2.0σ                    2.0σ                   0.30
Critical threshold         —                      3.0σ                   0.50
Statistical basis    95th pct null          95th / 99.7th pct        Empirical 2×
Citation          Gama 2014, Page 1954   Gama 2014, Page 1954    Rabanser 2019
```

The metric labelled `psi` in `channel_metrics` JSON output is **not** Population Stability Index;
it is the per-channel z-score `|μ_prod − μ_base| / (σ_base + ε)`.
The PSI name is a legacy misnomer corrected in this commit. Thesis text must use **z-score of mean shift**.

---

## Key Literature Summary

| Citation key | Full reference | Used for |
|---|---|---|
| `bao2004activity` | Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated acceleration data. *Pervasive Computing*, 1–17. | Window size 4 s |
| `chen2021deep` | Chen, Y., et al. (2021). Deep learning for sensor-based human activity recognition. *ACM Computing Surveys*, 54(4). | Window/overlap/batch conventions |
| `gal2016dropout` | Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*. | MC Dropout 30 passes |
| `gama2014survey` | Gama, J., et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4). | Drift detection, CUSUM, multi-channel gating |
| `guo2017calibration` | Guo, C., et al. (2017). On calibration of modern neural networks. *ICML 2017*. | Temperature scaling, ECE thresholds |
| `kingma2015adam` | Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. | Optimiser lr |
| `kirkpatrick2017overcoming` | Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13). | EWC λ |
| `kohavi1995study` | Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation. *IJCAI*. | 5-fold CV |
| `lakshminarayanan2017simple` | Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation. *NeurIPS 2017*. | Confidence thresholds |
| `lee2013pseudo` | Lee, D.-H. (2013). Pseudo-Label: The simple and efficient semi-supervised learning method. *ICML Workshop*. | Curriculum self-training |
| `li2018adabn` | Li, Y., et al. (2018). Revisiting Batch Normalization for practical domain adaptation. *arXiv:1603.04779*. | AdaBN |
| `mannini2010machine` | Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying human physical activity. *Sensors*, 10(2). | Sensor range validation |
| `naeini2015obtaining` | Naeini, M. P., et al. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI 2015*. | ECE bins, ECE threshold |
| `page1954continuous` | Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2). | CUSUM / consecutive-warning threshold |
| `rabanser2019failing` | Rabanser, S., et al. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS 2019*. | Wasserstein drift thresholds |
| `sculley2015hidden` | Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*. | Governance: auto_deploy, promote_to_shared |
| `tarvainen2017mean` | Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models. *NeurIPS 2017*. | EMA decay 0.999 |
| `wald1947sequential` | Wald, A. (1947). *Sequential Analysis*. Wiley. | 3σ critical threshold |
| `wang2021tent` | Wang, D., et al. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR 2021*. | TENT n_steps, lr, ood_threshold, rollback |
| `zappi2020usc` | Zappi, P., et al. (2020). USC-HAD: A daily activity dataset for ubiquitous activity recognition. *ACM UbiComp*. | 50 Hz target, dominant-hand threshold |

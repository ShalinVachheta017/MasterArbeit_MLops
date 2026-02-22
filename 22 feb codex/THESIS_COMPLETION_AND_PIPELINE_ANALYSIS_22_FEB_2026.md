# Thesis Completion and Pipeline Analysis (Code-First, Independent)

Date: 2026-02-22  
Project: HAR Wearable IMU MLOps Thesis Pipeline  
Method: Independent code/artifact/test/workflow analysis (not based on progress markdown percentages)

---

## 1. What I analyzed (repo-wide, code-first)

I analyzed the repository structure and implementation across:

- `src/` (pipeline, components, adaptation, calibration, drift, API, rollback, monitoring)
- `tests/` (unit/integration coverage across pipeline and advanced modules)
- `scripts/` (audit, drift analysis, baseline builders, verification)
- `.github/workflows/ci-cd.yml` (CI/CD implementation status)
- `docker/`, `docker-compose.yml` (packaging/deployment)
- `logs/pipeline/` and `artifacts/` (actual pipeline execution evidence)
- `docs/research/` and `docs/thesis/` (paper-index and paper-driven design reasoning)

Important: I did **not** use any markdown that says "we are 60%/80%/95% complete" as the source of truth. I used code presence, integration status, tests, and run artifacts.

---

## 2. Executive completion estimate (independent)

This project has strong engineering progress, but not all implemented modules are integrated into the main orchestrated pipeline yet.

### Independent completion estimate (my assessment)

- Core engineering pipeline (Stages 1-10 code + recent successful run): **~80-85% complete**
- Advanced thesis methods implemented in code but not fully integrated (Stages 11-14): **~40-55% complete**
- Operational hardening (CI/CD, validation gates, smoke tests, proxy deploy checks): **~55-65% complete**
- Testing breadth (many tests exist) with environment/marker cleanup still needed: **~70% complete**
- Thesis manuscript writing (structure/planning exists, final chapters not written): **~25-35% complete**
- Overall thesis project readiness (engineering + evidence + writing): **~68-74% complete**

### Practical interpretation

- You are past the prototype stage.
- You are not yet at thesis-finished pipeline stage, mainly because:
  - advanced stages are not wired into the main orchestrator,
  - some CI jobs are placeholders/misaligned,
  - some safety/validation logic is partial,
  - thesis writing is still mostly planning docs.

---

## 3. Completion breakdown by pipeline stage (actual code reality)

### Summary table (Stages 1-14)

| Stage | Status in code | Status in main orchestrator | Completion (my estimate) | Notes |
|---|---|---|---:|---|
| 1. Data ingestion | Implemented | Integrated | 90% | Auto-discovery, merge, resample, manifest skip logic |
| 2. Data validation | Implemented | Integrated | 90% | Schema/range/missing/sampling checks |
| 3. Data transformation/preprocessing | Implemented | Integrated | 90% | Units, gravity removal, calibration, scaler, windows |
| 4. Model inference | Implemented | Integrated | 90% | Batch inference, outputs, confidence stats |
| 5. Evaluation | Implemented | Integrated | 85% | Labeled and unlabeled paths, calibration metrics supported |
| 6. Post-inference monitoring | Implemented | Integrated | 80% | 3-layer monitoring works; baseline-dependent drift layer |
| 7. Trigger evaluation | Implemented | Integrated | 70% | Good engine, but wrapper feeds some placeholder values |
| 8. Retraining/adaptation | Implemented | Integrated | 80% | AdaBN/TENT/AdaBN+TENT/pseudo-label options wired |
| 9. Model registration | Implemented | Integrated | 70% | Registry/rollback exists; proxy better check partially placeholder |
| 10. Baseline update | Implemented | Integrated | 85% | Governance-aware, versioned baseline promotion support |
| 11. Calibration/uncertainty | Implemented | Not executed by main pipeline | 45% | Wrapper exists; not fully integrated into orchestrator |
| 12. Wasserstein drift | Implemented | Not executed by main pipeline | 40% | Wrapper exists; likely artifact-field bug + no orchestrator execution |
| 13. Curriculum pseudo-labeling | Implemented | Not executed by main pipeline | 50% | Advanced method implemented but not in main run path |
| 14. Sensor placement/handedness | Implemented (partial wrapper use) | Not executed by main pipeline | 40% | Detection/reporting exists; augmentation not fully used in wrapper |

### Key integration gap (most important finding)

`run_pipeline.py` advertises advanced stages and configs, but `src/pipeline/production_pipeline.py` currently runs only the first 10 stages. The advanced configs are accepted but ignored in the main orchestrator path.

This is the biggest reason the implemented features and end-to-end completed pipeline are not the same thing.

---

## 4. Evidence from actual runs (not docs)

### Latest pipeline result evidence

From `logs/pipeline/pipeline_result_20260219_173823.json`:

- `overall_status` is `SUCCESS`
- Stages successfully completed in that run include validation, transformation, inference, evaluation, monitoring, trigger, retraining, registration, baseline_update
- `ingestion` was skipped (likely no new files / already processed)
- Advanced outputs are `null`: calibration, wasserstein_drift, curriculum_pseudo_labeling, sensor_placement

Interpretation: the core loop is running, but advanced stage execution is not yet part of the orchestrated path.

### Audit evidence

`scripts/audit_artifacts.py --retrain` passed (`12/12 PASS`) after setting `PYTHONIOENCODING=utf-8` (Windows console encoding issue otherwise).

This is useful thesis evidence because it proves artifact completeness/consistency checks exist beyond just unit tests.

---

## 5. Completion by engineering area (more realistic than one number)

### A. Core data -> inference -> monitoring loop

- Status: Strong
- Completion: ~82-88%
- Why: Implemented components, successful pipeline run, tests exist, artifacts generated
- Remaining: edge-case hardening, richer trigger inputs, integration cleanup

### B. Adaptation/retraining sophistication (AdaBN/TENT/pseudo-labeling/etc.)

- Status: Strong code depth, partial orchestration
- Completion: ~65-75% at module level, ~45-55% at end-to-end pipeline level
- Why: Methods are implemented, but advanced wrappers are not all executed in main pipeline
- Remaining: integrate stages 11-14, evaluate and compare methods systematically

### C. Monitoring/drift/observability

- Status: Good foundation
- Completion: ~70-80%
- Why: Three-layer monitoring and drift scripts/modules exist; Prometheus/Grafana assets exist
- Remaining: unify monitoring outputs -> trigger inputs -> CI validation -> dashboards for your actual deployment mode

### D. CI/CD + deployment safety

- Status: Partial
- Completion: ~55-65%
- Why: Workflow exists, multiple jobs defined, Dockerfiles exist
- Remaining: fix missing smoke script reference, add schedule trigger, replace placeholder model-validation steps, tighten rollback gates

### E. Testing and quality gates

- Status: Broad test suite, execution environment issues
- Completion: ~65-75%
- Why: Many tests cover important modules; pytest markers exist
- Remaining: marker hygiene, Windows temp/cache permission stability in local runs, CI-to-local parity

### F. Thesis report writing

- Status: Planning-heavy, draft-writing-light
- Completion: ~25-35%
- Why: Excellent outline and supporting docs exist, but final thesis chapters are not yet written
- Remaining: convert docs into thesis chapters with results figures and reproducible evidence tables

---

## 6. Pipeline stages explained (why, what, and pseudocode)

This section explains the actual pipeline logic and why each stage exists. I include pseudocode for thesis-friendly explanation.

### Stage 1: Data Ingestion

#### Why we use it

- Raw smartwatch exports are not directly model-ready.
- Accelerometer and gyroscope streams may be separated, misaligned, or differently formatted.
- Production operation needs repeatable ingestion and duplicate-skip behavior.

#### What this stage does (in your repo)

- Reads CSV inputs (direct files or discovered pairs in `data/raw/`)
- Parses Garmin-like formats (including list-like cell parsing)
- Creates timestamps
- Aligns accel + gyro via merge-asof
- Resamples to target frequency
- Saves fused output
- Tracks processed pairs via `ingestion_manifest.json`

#### Pseudocode

```python
def stage1_data_ingestion(input_paths=None):
    pairs = discover_sensor_file_pairs(input_paths or "data/raw")
    pairs = skip_already_processed(pairs, manifest="ingestion_manifest.json")
    fused_outputs = []
    for accel_csv, gyro_csv in pairs:
        accel = load_and_parse_csv(accel_csv)
        gyro = load_and_parse_csv(gyro_csv)
        accel = normalize_timestamps(accel)
        gyro = normalize_timestamps(gyro)
        fused = merge_asof_on_time(accel, gyro)
        fused = resample_to_target_hz(fused, target_hz=50)
        out_file = save_fused_csv(fused)
        fused_outputs.append(out_file)
        update_manifest(accel_csv, gyro_csv, out_file)
    return fused_outputs
```

### Stage 2: Data Validation (Quality Control)

#### Why we use it

- Garbage input causes silent model failure.
- Wrong units, missing columns, or impossible sensor ranges can look valid but break predictions.

#### What this stage does

- Required column checks
- Numeric type checks
- Missing-value ratios
- Accelerometer/gyroscope range checks
- Sampling-rate consistency checks
- Summary stats report

#### Pseudocode

```python
def stage2_validate_data(fused_csv):
    df = read_csv(fused_csv)
    assert_required_columns(df, ["ax", "ay", "az", "gx", "gy", "gz"])
    check_numeric_types(df)
    check_missingness(df, max_missing_ratio=0.2)
    check_sensor_ranges(df, accel_bounds, gyro_bounds)
    check_sampling_consistency(df, expected_hz=50, tolerance=0.1)
    report = build_validation_report(df)
    save_json(report, "validation_report.json")
    return report
```

### Stage 3: Preprocessing / Transformation (including windowing + scaling)

#### Why we use it

- The model expects exactly the same preprocessing logic used during training.
- Domain mismatch (units, gravity offset, scaling) is a major cause of performance drop.

#### What this stage does

- Unit detection and conversion (e.g., `milliG` to `m/s^2`)
- Optional gravity removal (high-pass filter)
- Optional domain calibration (mean-shift toward training stats)
- Load and apply training scaler from `data/prepared/config.json`
- Generate sliding windows using vectorized logic

#### Pseudocode

```python
def stage3_preprocess_transform(fused_csv, config):
    df = read_csv(fused_csv)
    df = detect_and_convert_units(df)
    if config.remove_gravity:
        df = highpass_filter_accel(df, cutoff_hz=0.3)
    if config.enable_domain_calibration:
        df = apply_domain_mean_shift(df, training_means=config.training_means)
    windows = sliding_window(df[["ax", "ay", "az", "gx", "gy", "gz"]], size=config.window_size, step=config.step_size)
    scaler = load_training_scaler("data/prepared/config.json")
    windows_scaled = scaler.transform(windows)
    out = save_windows(windows_scaled)
    return out
```

### Stage 4: Model Inference

#### Why we use it

- Converts prepared windows into activity predictions and confidence scores.
- Produces the primary outputs that later stages monitor and evaluate.

#### What this stage does

- Loads model
- Runs batch inference
- Saves predictions and probabilities
- Computes confidence summary and activity distribution
- Logs run metadata/metrics (MLflow through inference pipeline path)

#### Pseudocode

```python
def stage4_inference(window_file, model_path):
    X = load_windows(window_file)
    model = load_model(model_path)
    probs = model.predict(X, batch_size=64)
    preds = argmax(probs, axis=1)
    conf = max_prob(probs)
    save_csv({"pred": preds, "confidence": conf, "probs": probs})
    summary = {
        "mean_confidence": conf.mean(),
        "uncertain_ratio": (conf < 0.6).mean(),
        "class_distribution": histogram(preds),
    }
    save_json(summary, "inference_summary.json")
    return summary
```

### Stage 5: Evaluation

#### Why we use it

- You need different evaluation behavior depending on whether labels exist.
- In production-like settings, labels may be unavailable, so unlabeled proxy analysis is essential.

#### What this stage does

- If labels are present: classification metrics (accuracy, F1, etc.), calibration metrics (ECE)
- If labels absent: distribution/confidence/uncertainty/temporal summaries

#### Pseudocode

```python
def stage5_evaluation(predictions, maybe_labels=None):
    if maybe_labels is not None:
        metrics = classification_metrics(y_true=maybe_labels, y_pred=predictions.preds)
        metrics["ece"] = expected_calibration_error(y_true=maybe_labels, probs=predictions.probs)
    else:
        metrics = unlabeled_proxy_evaluation(probs=predictions.probs, preds=predictions.preds)
    save_json(metrics, "evaluation_report.json")
    return metrics
```

### Stage 6: Post-Inference Monitoring (3-layer monitoring)

#### Why we use it

- Accuracy cannot be monitored directly without production labels.
- A single proxy metric is too fragile.
- Three layers give complementary views: prediction quality proxy, temporal behavior, and input-data shift.

#### What this stage does

- Layer 1: confidence/uncertainty monitoring
- Layer 2: temporal consistency monitoring (flip/transition behavior)
- Layer 3: baseline drift monitoring (feature distribution shift against baseline stats)

#### Pseudocode

```python
def stage6_monitoring(predictions, prepared_windows, baseline_stats=None):
    layer1 = confidence_monitor(confidences=predictions.confidences, low_conf_threshold=0.6, uncertain_ratio_threshold=0.3)
    layer2 = temporal_monitor(preds=predictions.preds, transition_rate_warn=0.5)
    if baseline_stats is not None:
        layer3 = baseline_drift_monitor(windows=prepared_windows, baseline_mean=baseline_stats.mean, baseline_std=baseline_stats.std)
    else:
        layer3 = {"status": "SKIPPED_NO_BASELINE"}
    report = {"layer1": layer1, "layer2": layer2, "layer3": layer3}
    save_json(report, "monitoring_report.json")
    return report
```

### Stage 7: Trigger Evaluation (retraining/adaptation decision)

#### Why we use it

- Monitoring outputs are signals, not decisions.
- Retraining every time any metric moves is expensive and unsafe.
- A trigger policy converts signals into actionable and controlled decisions.

#### Pseudocode

```python
def stage7_trigger_policy(monitoring_report, policy_state):
    signals = extract_signals(monitoring_report)
    confidence_status = eval_confidence_signals(signals)
    temporal_status = eval_temporal_signals(signals)
    drift_status = eval_drift_signals(signals)
    decision = aggregate_votes(confidence_status, temporal_status, drift_status)
    decision = apply_cooldown_and_escalation(decision, policy_state)
    save_policy_state(policy_state)
    save_json(decision, "trigger_decision.json")
    return decision
```

### Stage 8: Model Retraining / Adaptation

#### Why we use it

- Production data is mostly unlabeled.
- You need safe ways to adapt without full supervised retraining.
- Different shift types need different responses.

#### What this stage does (available modes)

- `none`
- `adabn`
- `tent`
- `adabn_tent`
- `pseudo_label`

#### Pseudocode

```python
def stage8_retraining_or_adaptation(mode, model, target_data, source_data=None):
    if mode == "none":
        return model, {"action": "no_change"}
    if mode == "adabn":
        model = adapt_bn_statistics(model, target_data)
        return model, {"action": "adabn"}
    if mode == "tent":
        model, report = tent_adapt(model, target_data)
        return model, report
    if mode == "adabn_tent":
        model = adapt_bn_statistics(model, target_data)
        model, tent_report = tent_adapt(model, target_data)
        return model, {"action": "adabn_tent", **tent_report}
    if mode == "pseudo_label":
        pseudo_set = select_high_confidence_pseudo_labels(model, target_data)
        model, train_report = retrain_with_pseudo_labels(model, source_data, pseudo_set)
        return model, train_report
```

### Stage 9: Model Registration

#### Why we use it

- Adapted/retrained models must be versioned for traceability and rollback.
- Thesis claims require proving which model version produced which results.

#### Pseudocode

```python
def stage9_model_registration(model_path, metrics, deploy=False):
    version = registry.register_model(model_path, metadata=metrics)
    if deploy:
        registry.deploy_model(version)
    return {"registered_version": version, "deployed": deploy}
```

### Stage 10: Baseline Update

#### Why we use it

- Monitoring Layer 3 depends on a baseline reference distribution.
- You must avoid silently changing the baseline after every run.

#### Pseudocode

```python
def stage10_baseline_update(data_for_baseline, promote_to_shared=False):
    baseline_stats = compute_reference_stats(data_for_baseline)
    artifact_path = save_baseline_artifact(baseline_stats)
    if promote_to_shared:
        archive_current_shared_baseline()
        copy_to_shared_models_baseline(artifact_path)
    return {"artifact_baseline": artifact_path, "promoted": promote_to_shared}
```

### Stage 11: Calibration and Uncertainty (advanced wrapper exists)

#### Why we use it

- Confidence scores are not always calibrated.
- A model can be highly confident and still wrong.
- Calibration and uncertainty improve the reliability of trigger decisions and pseudo-labeling gates.

#### What is implemented

- Temperature scaling utilities
- Calibration evaluation
- MC Dropout utilities
- Unlabeled confidence analysis

#### Current status

- Module is implemented, wrapper exists, but not fully integrated into the main orchestrated run path.

#### Pseudocode

```python
def stage11_calibration_uncertainty(model, preds, labeled_val=None):
    if labeled_val is not None:
        T = fit_temperature_scaling(model, labeled_val)
        calibrated_probs = apply_temperature(preds.probs, T)
        ece = expected_calibration_error(labeled_val.y, calibrated_probs)
        return {"temperature": T, "ece": ece}
    return analyze_unlabeled_confidence(preds.probs)
```

### Stage 12: Wasserstein Drift Detection (advanced wrapper exists)

#### Why we use it

- KS and PSI are useful but incomplete.
- Wasserstein distance captures magnitude/geometry of distribution change for multichannel time-series drift comparison.

#### What is implemented

- Per-channel Wasserstein drift
- Change-point style analysis
- Combined drift analysis (KS/PSI/Wasserstein)

#### Current status

- Advanced module exists but is not executed by the main pipeline orchestrator.
- Wrapper appears to have a likely artifact-field mismatch bug in one return path.

#### Pseudocode

```python
def stage12_wasserstein_drift(train_ref, prod_batch):
    scores = {ch: wasserstein_distance(train_ref[ch], prod_batch[ch]) for ch in channels6}
    cp = detect_change_point_over_windows(scores)
    verdict = aggregate_drift(scores, cp, thresholds)
    save_json({"scores": scores, "cp": cp, "verdict": verdict}, "wasserstein_drift.json")
    return verdict
```

### Stage 13: Curriculum Pseudo-Labeling (advanced wrapper exists)

#### Why we use it

- Naive pseudo-labeling amplifies mistakes.
- Curriculum selection (high confidence first, then lower) reduces confirmation bias and improves stability.

#### What is implemented

- Progressive threshold schedule
- Class-balanced selection
- Teacher-student EMA logic
- EWC regularization support
- Early stopping criteria

#### Current status

- Implemented as advanced module/wrapper, but not yet part of main orchestrated stage execution.

#### Pseudocode

```python
def stage13_curriculum_pseudo_labeling(model, target_data, source_replay):
    teacher = copy_model(model)
    student = reset_or_copy_model(model)
    threshold = 0.95
    for iter_id in range(max_iters):
        pseudo = select_pseudo_labels(teacher, target_data, conf_threshold=threshold, class_balanced=True)
        student = train_with_source_plus_pseudo(student, source_replay, pseudo, ewc_penalty=True)
        teacher = ema_update(teacher, student)
        threshold = decay_threshold(threshold, min_value=0.80)
        if early_stopping():
            break
    return student
```

### Stage 14: Sensor Placement / Handedness Compensation (advanced wrapper exists)

#### Why we use it

- Wrist side / handedness / placement mismatch can create large performance drops.
- Many users will not match the training setup exactly.

#### What is implemented

- Hand detection / placement analysis support
- Sensor mirroring augmentation utilities

#### Current status

- Wrapper reports hand/placement info, but augmentation is not fully applied/output in the wrapper path yet.

#### Pseudocode

```python
def stage14_sensor_placement_handling(batch_data, metadata=None):
    placement = detect_hand_or_placement(batch_data, metadata)
    if placement_mismatch_suspected(placement):
        batch_data = apply_axis_mirroring_or_compensation(batch_data)
    save_json({"placement": placement}, "sensor_placement_report.json")
    return batch_data
```

---

## 7. Three-layer monitoring (detailed): why exactly three?

You asked why just three. This is a very good thesis question.

### Short answer

Three layers are a good engineering compromise because they are:

- Orthogonal enough (different failure modes)
- Label-free (works in your unlabeled production setting)
- Computationally practical
- Explainable in a thesis and to supervisors

### The three layers and what each catches

#### Layer 1: Confidence / uncertainty proxy

- Inputs: model probabilities/confidences
- Detects: model hesitation, uncertainty spikes, low-confidence batches
- Good for: early warning of degradation
- Weakness: overconfident wrong predictions can hide here

#### Layer 2: Temporal consistency / transition behavior

- Inputs: predicted label sequence over time
- Detects: unrealistic rapid flipping, unstable predictions, broken temporal behavior
- Good for: HAR-specific sanity (activities have duration patterns)
- Weakness: true behavior change can look like instability

#### Layer 3: Input drift vs baseline statistics

- Inputs: prepared feature/window distributions vs baseline mean/std
- Detects: sensor drift, device changes, user domain shift, preprocessing mismatch
- Good for: catching problems before labels exist
- Weakness: not all drift is harmful (some drift is benign)

### Why not only one layer?

- Confidence alone misses overconfident failure.
- Temporal alone misses stable-but-wrong predictions.
- Input drift alone misses concept drift or confidence collapse.

### Why not 10 layers?

- More layers increase complexity, tuning burden, false positives, and thesis explanation overhead.
- For a thesis, three layers are a strong and defensible baseline architecture: output proxy, temporal behavior, input distribution.

### How the three layers help together

Typical interpretation:

- Layer 3 high drift + Layer 1 stable confidence -> likely covariate shift, try AdaBN first
- Layer 1 confidence drop + Layer 2 instability -> model behavior degradation, evaluate retraining trigger
- Layer 3 normal + Layer 1/2 degrade -> possible concept drift or label shift

This is exactly why a multi-signal trigger policy is needed.

---

## 8. Retraining trigger logic (detailed): what, why, and how

### Why a trigger layer exists

Monitoring generates alerts, but production systems need controlled decisions:

- do nothing
- increase monitoring
- adapt (AdaBN/TENT)
- retrain with pseudo-labels
- rollback/deploy guard

Without a trigger policy, the pipeline is either too passive or too reactive.

### Current design strengths in your repo (`src/trigger_policy.py`)

- Multiple thresholds supported (confidence, entropy, temporal, drift metrics)
- Voting/aggregation logic instead of single-signal trigger
- Cooldown mechanism
- State persistence for history and escalation
- Consecutive-warning escalation support
- Proxy validator concept for model comparison

### Current practical gaps

- Trigger wrapper (`src/components/trigger_evaluation.py`) maps some fields as placeholder zeros for certain metrics.
- Some advanced drift outputs are not yet feeding the trigger because stages 11-14 are not integrated in the main pipeline path.

### Recommended decision logic for thesis write-up (matches your implementation direction)

```python
def trigger_decision(monitoring, drift, state):
    conf_status = evaluate_confidence(monitoring.layer1)
    temp_status = evaluate_temporal(monitoring.layer2)
    drift_status = evaluate_drift(monitoring.layer3, drift)
    statuses = [conf_status, temp_status, drift_status]
    if in_cooldown(state):
        return "LOG_ONLY"
    if any(s == "CRITICAL" for s in statuses):
        action = "RETRAIN_EVAL"
    elif count_warning_or_above(statuses) >= 2:
        action = "ADAPT_OR_RETRAIN_EVAL"
    elif count_warning_or_above(statuses) == 1:
        action = "MONITOR_MORE_FREQUENTLY"
    else:
        action = "NO_ACTION"
    action = apply_consecutive_warning_escalation(action, state)
    persist_state(state)
    return action
```

### Why this helps your thesis

- It makes the retraining loop traceable and auditable
- It reduces false-positive retraining
- It gives a clear decision policy contribution, not just a collection of scripts

---

## 9. Adaptation methods in your pipeline (AdaBN, TENT, AdaBN+TENT)

You specifically asked what they are, why they are used, and how they help during execution.

### 9.1 AdaBN (Adaptive Batch Normalization)

#### What it is

- Updates BatchNorm running mean/variance using target-domain unlabeled data.
- Does not retrain all weights.

#### Why you use it

- Fast unlabeled adaptation for covariate shift (sensor/device/user distribution changes)
- Low risk compared to full pseudo-label retraining
- Good first-line response when input drift is detected

#### How it helps during execution

- If production sensor distribution shifts, BN statistics become mismatched.
- AdaBN recalibrates internal normalization to target data without labels.

#### In your code (important)

- `src/domain_adaptation/adabn.py`
- Freezes non-BN layers and updates only BN running stats via target batches

#### Pseudocode

```python
def adabn_adapt(model, target_loader):
    freeze_non_bn_layers(model)
    set_bn_layers_train_mode(model)
    for x_batch in target_loader:
        _ = model(x_batch, training=True)  # updates BN running mean/var
    return model
```

### 9.2 TENT (Test-time Entropy Minimization)

#### What it is

- Test-time adaptation method that minimizes prediction entropy on unlabeled target data.
- Updates only BatchNorm affine parameters (`gamma`, `beta`).

#### Why you use it

- Adapts confidence/decision boundaries under shift without labels
- More adaptive than AdaBN, but riskier if shift is harmful/non-stationary

#### How it helps during execution

- Reduces uncertain predictions in shifted environments
- Helps when confidence degrades but full retraining is too expensive

#### In your code (strong engineering detail)

- `src/domain_adaptation/tent.py`
- Includes OOD guard and rollback logic if entropy worsens / confidence drops
- Preserves AdaBN-updated BN stats during TENT steps (important stability fix)

#### Pseudocode

```python
def tent_adapt(model, target_loader):
    enable_only_bn_affine_params(model)
    opt = optimizer_for_bn_affine(model)
    baseline = snapshot_model_state(model)
    for x_batch in target_loader:
        probs = softmax(model(x_batch, training=True))
        entropy = mean_prediction_entropy(probs)
        if is_ood_risk(entropy):
            restore_state(model, baseline)
            return model, {"status": "ROLLBACK_OOD_GUARD"}
        opt.zero_grad()
        entropy.backward()
        opt.step()
        if harmful_regression_detected():
            restore_state(model, baseline)
            return model, {"status": "ROLLBACK_REGRESSION"}
    return model, {"status": "ADAPTED"}
```

### 9.3 AdaBN + TENT (combined)

#### What it is

- Two-step adaptation:
  1. AdaBN updates BN running statistics (distribution alignment)
  2. TENT updates BN affine parameters via entropy minimization (prediction refinement)

#### Why combine them

- AdaBN handles coarse domain-statistics mismatch
- TENT handles decision confidence refinement
- Combined approach can be stronger than either alone under some shifts

#### In your pipeline

- Supported mode in `src/components/model_retraining.py`
- Execution order is AdaBN first, then TENT

#### Pseudocode

```python
def adabn_plus_tent(model, target_loader):
    model = adabn_adapt(model, target_loader)
    model, report = tent_adapt(model, target_loader)
    return model, report
```

---

## 10. Pseudo-labeling, curriculum pseudo-labeling, and retraining triggers

### Why pseudo-labeling exists in your thesis

- Production data is mostly unlabeled
- Full supervised retraining is impossible in normal operation
- Pseudo-labeling is the practical bridge between no-label production and periodic model improvement

### Risks (and why your code has safeguards)

Main risk: confirmation bias

- If the model makes wrong predictions with high confidence, naive pseudo-labeling reinforces mistakes.

### Safeguards already implemented (good thesis strengths)

From `src/train.py` and `src/curriculum_pseudo_labeling.py`:

- Confidence gating
- Entropy gating
- Class-balanced top-k selection
- Label smoothing
- Temperature scaling estimate on source side
- Teacher-student / EMA concepts (curriculum module)
- EWC regularization support
- Early stopping
- Rollback / holdout safeguards in retraining path

### Why curriculum pseudo-labeling is better than naive pseudo-labeling

- Start with easiest samples (high confidence)
- Gradually include harder samples
- Reduces early contamination of training set
- More stable for thesis experiments and ablations

### Retraining trigger recommendation (how to connect methods)

Tiered response:

1. No action when monitoring is stable
2. AdaBN when Layer 3 drift rises but confidence/temporal remain mostly stable
3. AdaBN+TENT when confidence drops but OOD guard not triggered
4. Pseudo-label retraining evaluation when multiple signals persist or become critical
5. Rollback / canary reject if proxy metrics worsen

---

## 11. Calibration, uncertainty, and Wasserstein drift detection (why these matter)

### 11.1 Calibration (why)

#### Problem

- Confidence is not equal to correctness.
- A model can output 0.95 confidence and still be wrong.

#### Why calibration helps your pipeline

- Makes confidence thresholds more meaningful for monitoring alerts, pseudo-label acceptance, trigger decisions, and canary/proxy comparison.

#### What you have

- `src/calibration.py` supports temperature scaling and evaluation tools
- `src/components/calibration_uncertainty.py` wrapper exists
- Not fully integrated into the main orchestrator yet

#### Thesis angle

- Calibration is especially important because your pipeline relies on unlabeled proxy metrics.

### 11.2 Uncertainty (MC Dropout, confidence, entropy)

#### Why use uncertainty

- Accuracy is unavailable without labels
- Uncertainty proxies can detect degradation earlier than drift-only metrics

#### What you have

- Confidence and entropy monitoring integrated through monitoring/evaluation paths
- MC Dropout utilities available in calibration/uncertainty module

#### Practical note

- MC Dropout is computationally expensive; useful for audit batches or scheduled checks, not necessarily every request

### 11.3 Wasserstein drift detection (why)

#### Why not only KS / PSI?

- KS is shape-sensitive but not magnitude-aware enough in some cases
- PSI is useful but depends on binning
- Wasserstein adds a meaningful distance notion between distributions

#### Why it helps your pipeline

- Better drift characterization for multichannel IMU features
- Supports stronger trigger logic (combined with confidence + temporal signals)
- Good thesis justification because it is mathematically principled and label-free

#### Current repo status

- `src/wasserstein_drift.py` exists and is advanced
- wrapper exists in `src/components/wasserstein_drift.py`
- main pipeline does not yet execute it
- likely wrapper return-field bug needs fix

---

## 12. Docker, CI/CD, tests, and why there are so many tests

You asked:

- why Docker?
- why CI/CD?
- what tests are used in CI/CD?
- why so many tests in `tests/` beyond CI?
- why unit and integration tests?

### 12.1 Docker (why)

Docker solves reproducibility and deployment consistency:

- same Python/runtime/dependencies across machines
- easier thesis demo and supervisor reproduction
- versionable inference/training environments
- easier CI/CD builds and image publishing

Repo evidence:

- `docker/Dockerfile.inference`
- `docker/Dockerfile.training`
- `docker-compose.yml`

### 12.2 CI/CD (why)

CI/CD is thesis evidence for MLOps maturity:

- proves changes are tested automatically
- prevents regressions
- supports reproducibility claims
- enables repeatable packaging/integration checks

Repo evidence:

- `.github/workflows/ci-cd.yml` exists with multiple jobs (`lint`, `test`, `test-slow`, `build`, `integration-test`, `model-validation`, `notify`)

### 12.3 What tests CI/CD is trying to run (and why)

Workflow intent:

- `lint`: code quality / syntax standards
- `test`: fast tests (quick regression detection)
- `test-slow`: heavier tests (adaptation/retraining etc.)
- `integration-test`: end-to-end component interactions
- `build`: Docker/image/package build validation
- `model-validation`: scheduled model health checks (currently placeholder)

### 12.4 Important CI/CD gaps I found (must fix)

1. `scripts/inference_smoke.py` is referenced by CI but does not exist.
2. `model-validation` job checks for scheduled runs but workflow lacks `on.schedule`.
3. `model-validation` steps contain placeholder `echo` commands instead of real checks.

### 12.5 Why unit tests and integration tests both matter

#### Unit tests (fast, focused)

- Verify logic of small components in isolation
- Examples: trigger thresholds, AdaBN/TENT behavior, rollback registry behavior, drift score calculations

#### Integration tests (workflow confidence)

- Verify components work together with file I/O, configs, and artifacts
- Examples: pipeline stage contracts, artifact handoff compatibility, orchestration regressions

### 12.6 Why so many tests in `tests/` (even beyond CI)

This is a strength.

Reasons:

- CI cannot run every expensive case on every push
- Some tests are for research validation, robustness checks, offline audits, special modules not yet in default pipeline path, and future thesis ablations

So the `tests/` folder serves two roles:

1. engineering regression safety
2. thesis experiment confidence and reproducibility support

### 12.7 Current local test status (important nuance)

A local pytest run showed many tests passing but also failures/errors, with several issues caused by local Windows temp/cache permissions rather than only code logic.

Interpretation:

- test suite breadth is real and valuable
- local execution environment still needs stabilization (temp/cache paths, permissions)
- marker hygiene (`unit`, `integration`, `slow`, `gpu`) should be tightened for predictable CI/local behavior

---

## 13. Audit runs: why they exist and why they matter in this thesis

### Why audit runs are important

Audit runs are different from unit tests:

- Unit tests prove code behavior on controlled examples
- Audit runs prove a specific pipeline run produced all required artifacts and metadata

For a thesis, audit runs help answer:

- Can we reproduce this run?
- Do we have the artifacts for the claimed result?
- Were monitoring, trigger, and retraining outputs all produced?

### What your audit script contributes

`scripts/audit_artifacts.py` checks artifact presence/consistency and provides a pass/fail summary. This is excellent for thesis reproducibility appendices.

### Practical improvement

- Make audit script Windows-console-safe by default (avoid Unicode-only output or force UTF-8)

---

## 14. Cross-dataset comparison and drift analysis across datasets (why useful)

### Why this matters

Your thesis problem is fundamentally about domain shift:

- training data source (e.g., ADAMSense / lab-like)
- production data source (e.g., Garmin / real-world)

Cross-dataset comparison helps quantify:

- which channels shift most
- expected baseline domain mismatch before deployment
- threshold calibration for monitoring
- whether adaptation should focus on preprocessing vs model adaptation

### In your repo

- `scripts/analyze_drift_across_datasets.py` computes drift statistics and threshold recommendations

### How it helps the thesis

- Turns we-expect-drift into measured evidence
- Supports threshold choices in monitoring/trigger sections
- Supports RQ-style evaluation (does drift correlate with performance degradation?)

### Good thesis experiment design

1. Compare train-vs-production dataset distributions before adaptation
2. Run baseline model on target data
3. Run AdaBN/TENT/AdaBN+TENT
4. Recompute drift/proxy metrics
5. Report which metrics moved and whether proxy metrics improved

---

## 15. If many new models are pushed via Docker/CI/CD, how do we go back to baseline?

### Short answer

Yes, you can go back, but only if model versions and baselines are treated as first-class artifacts.

### What you already have

- Local model registry and rollback support in `src/model_rollback.py`
- Registry metadata at `models/registry/model_registry.json`
- Deploy/rollback history
- `current_model.keras` deployment target
- Baseline update stage with artifact storage and optional shared promotion

### What baseline can mean (important distinction)

There are two different baselines:

1. Model baseline (the known-good deployed model)
2. Monitoring baseline (reference feature statistics used for drift)

You need rollback strategy for both.

### Recommended rollback strategy (thesis + practical)

#### Model rollback

- Keep immutable version IDs/tags for every registered model
- Deploy by version, not by overwriting unnamed files
- If proxy metrics regress after deployment/canary, call rollback to previous version

#### Monitoring baseline rollback

- Never auto-promote new baseline after every run
- Save versioned baseline snapshots
- Promote baseline only after explicit retraining acceptance or supervisor-approved rule
- If a bad baseline was promoted, restore previous archived baseline version

### Docker/CI/CD best practice for rollback

- Tag images with commit SHA, model version ID, and optional `stable` alias
- Promote `stable` only after validation
- Rollback by redeploying last known-good image tag and registry version

### One current limitation in your code path

- Proxy validation in model registration is partially placeholder (`is_better=True` path), so safe automatic promotion still needs hardening.

---

## 16. Do you still need Prometheus and Grafana if data is mostly offline / local / FastAPI upload?

Short answer: not strictly required, but still useful depending on deployment mode.

### If usage is mostly offline batch runs (thesis experiments)

You can operate well with:

- MLflow (experiment tracking)
- JSON reports/artifacts
- audit scripts
- periodic plots/reports

In this mode, Prometheus/Grafana is optional.

### If usage includes a persistent FastAPI service (even local/lab)

Prometheus/Grafana becomes useful for:

- live service health (`latency`, `throughput`, errors)
- monitoring trends across uploads/sessions
- alerting on drift/confidence issues
- demos of operational monitoring maturity

### Best recommendation for your thesis

Treat Prometheus/Grafana as:

- Operational observability layer (optional for offline-only operation, recommended for service mode)

### Decision guide

- Offline thesis experiments only: optional (nice-to-have)
- Continuous API service / multiple users / long-running deployment: recommended
- Real clinical/production pilot: strongly recommended

### Repo reality (good news)

You already have the building blocks:

- `src/prometheus_metrics.py`
- `config/prometheus.yml`
- `config/grafana/har_dashboard.json`
- `config/alerts/har_alerts.yml`
- tests for Prometheus metrics

So this is more of an integration/use-mode decision than a fresh implementation effort.

---

## 17. Paper-informed improvement plan (using your local paper maps and summaries)

I used:

- `docs/research/appendix-paper-index.md` (global + stage-wise paper map)
- `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md` (88-paper question-driven synthesis)
- `docs/research/RESEARCH_PAPER_INSIGHTS.md` (improvement ideas + action phases)

Important note:

- Some paper-summary docs state page-level citations still need manual verification.
- So this is paper-guided engineering planning, not a final page-precise literature review chapter.

### 17.1 Highest-value improvement (before new fancy methods)

Integrate your existing advanced stages (11-14) into the main orchestrator.

Why first:

- You already implemented calibration, Wasserstein drift, curriculum pseudo-labeling, and sensor placement modules.
- Thesis value increases more by integrating and evaluating them than by adding another new method.

### 17.2 Monitoring and drift improvements (paper-aligned)

Paper-guided direction from WATCH/LIFEWATCH/OOD-HAR/uncertainty papers in your question map:

1. Add Wasserstein drift stage to main run path
2. Add energy-based OOD score as another label-free monitoring signal
3. Add pattern memory (LIFEWATCH-like) to reduce repeated false alarms on recurring benign patterns
4. Calibrate thresholds using cross-dataset drift analysis + small labeled audit batches

### 17.3 Trigger policy improvements (paper-informed + code reality)

1. Feed real entropy / uncertainty / drift outputs into trigger wrapper (remove placeholder zeros)
2. Add explicit tiered response mapping (`monitor`, `AdaBN`, `AdaBN+TENT`, `pseudo-label retraining eval`)
3. Persist and analyze consecutive warning patterns
4. Add threshold calibration study (false alarm vs missed degradation tradeoff)

### 17.4 Adaptation/retraining improvements (XHAR, Tent, SelfHAR, Curriculum, etc.)

1. Run controlled comparison: no adaptation vs AdaBN vs TENT vs AdaBN+TENT vs pseudo-labeling
2. Add acceptance criteria before promotion (proxy metrics improve, uncertainty not worse, temporal instability not increased)
3. Use a small labeled audit subset to validate proxy metrics correlation (very important thesis evidence)

### 17.5 Sensor placement / handedness improvement (paper gap + your module)

Practical improvements:

1. Require/store upload metadata (dominant hand, watch wrist)
2. Run sensor-placement stage and log mismatch risk
3. Add mirrored inference as fallback experiment (compare proxy stability)
4. Report placement-stratified performance (if labels available in audit subset)

### 17.6 CI/CD and reproducibility improvements (paper + MLOps practice)

Highest-priority fixes:

1. Fix missing `scripts/inference_smoke.py` reference or replace with existing smoke test
2. Add `on.schedule` to workflow if `model-validation` is intended
3. Replace placeholder `echo` validation steps with real checks (audit script, inference smoke, proxy thresholds)
4. Add artifact retention and CI run summaries

### 17.7 Calibration and uncertainty improvements (Guo et al. + your module)

1. Integrate Stage 11 into main pipeline (optional mode)
2. Use temperature scaling on a held-out labeled validation split
3. Store calibration parameters per model version
4. Reuse calibrated probabilities for pseudo-label gating and confidence monitoring

### 17.8 Thesis-quality experiments to add (most impactful)

If time is limited, prioritize these experiments:

1. End-to-end ablation of monitoring layers (1 vs 2 vs 3 layers)
2. Adaptation comparison (AdaBN vs TENT vs AdaBN+TENT vs pseudo-labeling)
3. Proxy metric correlation with labeled audit subset
4. Cross-dataset drift vs degradation analysis
5. Rollback/canary simulation with proxy-based acceptance

---

## 18. Thesis report preparation help (chapter structure, what to include, and why)

You already have useful planning docs (especially `docs/thesis/THESIS_STRUCTURE_OUTLINE.md`), so below is a practical version adapted to the current project reality.

### 18.1 Recommended thesis structure (practical and defensible)

#### Chapter 1: Introduction

Include: problem context, why MLOps matters (drift/unlabeled production/reproducibility), problem statement, research questions, contributions, thesis roadmap.

#### Chapter 2: Background and Related Work

Include: HAR with wearable IMU, domain shift in HAR, AdaBN/TENT, pseudo-labeling/curriculum, uncertainty/calibration, drift detection (KS/PSI/Wasserstein/OOD), MLOps in healthcare.

#### Chapter 3: Methodology (core chapter)

Include: full pipeline architecture, stage-by-stage logic, three-layer monitoring design, trigger policy, adaptation/retraining strategy, rollback and baseline governance, offline vs service observability strategy.

#### Chapter 4: Implementation

Include: repo structure, key modules/scripts, Docker/FastAPI setup, CI/CD workflow, test suite design, audit runs, known implementation gaps.

#### Chapter 5: Experimental Evaluation

Include: datasets, preprocessing alignment choices, baseline performance, cross-dataset degradation, monitoring behavior, adaptation comparison, proxy-vs-audit correlation, ablations, runtime/cost overhead.

#### Chapter 6: Discussion, Limitations, and Future Work

Include: what worked, what failed, limitations of unlabeled validation, optional role of Prometheus/Grafana for offline mode, future work (LIFEWATCH memory, OOD energy, conformal monitoring, active learning).

#### Chapter 7: Conclusion

Include: answers to each research question, final contributions summary, practical implications.

#### Appendices (recommended)

Include: configs, CI/CD workflow, test matrix, audit checklists, extra figures, artifact schemas.

### 18.2 Thesis chapter index (compact version)

```text
1. Introduction
   1.1 Motivation
   1.2 Problem Statement
   1.3 Research Questions
   1.4 Contributions
   1.5 Thesis Organization

2. Background and Related Work
   2.1 Wearable IMU-based HAR
   2.2 Domain Shift and Generalization in HAR
   2.3 Test-Time Adaptation (AdaBN, TENT)
   2.4 Pseudo-Labeling and Curriculum Self-Training
   2.5 Uncertainty, Calibration, and OOD Detection
   2.6 Drift Detection and Monitoring in MLOps
   2.7 MLOps for Health/Monitoring Systems
   2.8 Research Gap Summary

3. Methodology
   3.1 System Overview
   3.2 Data Ingestion and Validation
   3.3 Preprocessing, Windowing, and Scaling
   3.4 Inference and Evaluation
   3.5 Three-Layer Monitoring Framework
   3.6 Trigger Policy for Adaptation/Retraining
   3.7 Adaptation Methods (AdaBN, TENT, AdaBN+TENT)
   3.8 Pseudo-Label Retraining Strategy
   3.9 Model Registration, Baseline Governance, and Rollback
   3.10 Observability Strategy (MLflow vs Prometheus/Grafana)

4. Implementation
   4.1 Repository and Component Architecture
   4.2 Pipeline Orchestrator and Stage Interfaces
   4.3 API and Containerization
   4.4 CI/CD Workflow
   4.5 Test Strategy (Unit/Integration/Slow)
   4.6 Audit and Artifact Verification
   4.7 Implementation Gaps and Engineering Tradeoffs

5. Experimental Evaluation
   5.1 Experimental Setup and Datasets
   5.2 Source-to-Target Performance Degradation
   5.3 Monitoring and Drift Detection Evaluation
   5.4 Trigger Policy Behavior Analysis
   5.5 Adaptation/Retraining Comparison
   5.6 Proxy Metrics vs Labeled Audit Set Correlation
   5.7 Ablation Studies
   5.8 Runtime and Operational Cost Analysis

6. Discussion and Future Work
   6.1 Key Findings
   6.2 Practical Deployment Considerations (Offline vs Service)
   6.3 Limitations
   6.4 Future Work

7. Conclusion
References
Appendices
```

---

## 19. Difficulties you are likely to face (and how to overcome them)

### Difficulty 1: Implemented vs integrated confusion

- Problem: many advanced modules exist, but not all are in the orchestrated run path.
- Fix: use an integration checklist and distinguish in thesis text between implemented module, integrated stage, and evaluated stage.

### Difficulty 2: Unlabeled production evaluation gap

- Problem: hard to prove adaptation helped without labels.
- Fix: create a small labeled audit subset and correlate proxy metrics with accuracy/F1.

### Difficulty 3: CI/CD claims stronger than actual workflow

- Problem: workflow exists but some jobs are placeholders/mismatched.
- Fix: repair the workflow before thesis screenshots/claims and include one successful CI evidence table.

### Difficulty 4: Threshold tuning and false alarms

- Problem: monitoring/trigger thresholds can overfire or underfire.
- Fix: use cross-dataset drift analysis + labeled audit subset + sensitivity analysis.

### Difficulty 5: Test suite complexity and local environment issues

- Problem: Windows temp/cache permissions and marker inconsistencies distort test status.
- Fix: standardize pytest temp/cache config, enforce markers, separate environment failures from logic failures.

### Difficulty 6: Overclaiming Prometheus/Grafana necessity

- Problem: your deployment mode is often offline/local.
- Fix: present observability as tiered (offline vs service mode).

### Difficulty 7: Thesis writing delay despite strong code progress

- Problem: many support docs exist, but final chapter drafting lags.
- Fix: write Methodology and Implementation first (code-backed), then Evaluation after final experiments.

---

## 20. Concrete remaining work (prioritized)

### P0 (must do for strong thesis submission)

1. Integrate advanced stages 11-14 into `src/pipeline/production_pipeline.py`
2. Fix CI/CD workflow gaps (missing smoke script reference, add schedule, replace placeholder model-validation steps)
3. Harden trigger inputs (remove placeholder zero mappings where possible)
4. Fix `src/components/wasserstein_drift.py` artifact return mismatch
5. Replace placeholder proxy better-model logic with real proxy validation criteria
6. Run and document a complete end-to-end experiment set (including adaptation comparisons)

### P1 (high value, thesis strength)

1. Add/validate small labeled audit subset for proxy metric correlation
2. Integrate calibration stage for confidence-dependent decisions
3. Add energy-based OOD score to monitoring/trigger
4. Improve sensor placement wrapper to apply/report compensation path explicitly
5. Stabilize local test execution environment and marker usage

### P2 (nice if time allows / future work if not)

1. LIFEWATCH-like pattern memory for recurring drift suppression
2. More advanced online TTA (COA-HAR/OFTTA-like) benchmarking
3. Active learning sample query pipeline
4. Conformal-style monitoring for risk-aware alerts

---

## 21. Final assessment (plain language)

Your thesis project is not just documentation and it is also not fully finished.

It is best described as:

- a strong MLOps/HAR pipeline with real code depth and a functioning core loop
- plus advanced thesis methods implemented as modules
- but still needing final integration, validation hardening, and thesis write-up execution

If you focus next on integration + evaluation + thesis chapter drafting (instead of adding more new features), this can become a very strong and defensible thesis.

---

## 22. Files used as primary evidence (selected)

- `run_pipeline.py`
- `src/pipeline/production_pipeline.py`
- `src/components/data_ingestion.py`
- `src/components/data_validation.py`
- `src/components/data_transformation.py`
- `src/components/model_inference.py`
- `src/components/model_evaluation.py`
- `src/components/post_inference_monitoring.py`
- `src/components/trigger_evaluation.py`
- `src/trigger_policy.py`
- `src/components/model_retraining.py`
- `src/domain_adaptation/adabn.py`
- `src/domain_adaptation/tent.py`
- `src/train.py`
- `src/components/model_registration.py`
- `src/model_rollback.py`
- `src/components/baseline_update.py`
- `src/components/calibration_uncertainty.py`
- `src/calibration.py`
- `src/components/wasserstein_drift.py`
- `src/wasserstein_drift.py`
- `src/components/curriculum_pseudo_labeling.py`
- `src/curriculum_pseudo_labeling.py`
- `src/components/sensor_placement.py`
- `src/sensor_placement.py`
- `src/prometheus_metrics.py`
- `src/api/app.py`
- `.github/workflows/ci-cd.yml`
- `docker/Dockerfile.inference`
- `docker/Dockerfile.training`
- `docker-compose.yml`
- `scripts/post_inference_monitoring.py`
- `scripts/analyze_drift_across_datasets.py`
- `scripts/build_training_baseline.py`
- `scripts/audit_artifacts.py`
- `scripts/verify_repository.py`
- `logs/pipeline/pipeline_result_20260219_173823.json`
- `docs/research/appendix-paper-index.md`
- `docs/thesis/PAPER_DRIVEN_QUESTIONS_MAP.md`
- `docs/research/RESEARCH_PAPER_INSIGHTS.md`
- `docs/thesis/THESIS_STRUCTURE_OUTLINE.md`

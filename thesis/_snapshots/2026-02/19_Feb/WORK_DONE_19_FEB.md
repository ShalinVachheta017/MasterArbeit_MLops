# Work Done — 19 Feb 2026 (Full Day)

**Project:** HAR MLOps Pipeline — Master Thesis  
**Date:** Wednesday, 19 February 2026  
**Git Commits Today:** 7 (`34f80df` → `700381a` → `bd8dc1e` → `1ae27cc` → `3fd3c00` → `f47a48b` → `e2bc784`)  
**HEAD at end of day:** `e2bc784` on `origin/main`

---

## Sessions Overview

| Session | Time | Focus | Commits |
|---------|------|-------|---------|
| Morning | ~11:00 – 13:45 | Pipeline fixes + all 4 audit runs + docs | 5 |
| Afternoon | ~14:00 – 17:30 | Code quality hardening from 2nd review | 2 |

---

## SESSION 1 — Morning (Commits 1–5)

### Goal
Execute thesis closure plan: run all four audit experiments (A1/A3/A4/A5), fix any pipeline bugs encountered, verify artifacts, fill ablation table, commit everything.

---

### 1a. Pipeline Bug Fixes (Commits 1–4)

| Commit | Title | Key Fix |
|--------|-------|---------|
| `34f80df` | Bug fixes: baseline + metric loss | Wired `BaselineBuilder` correctly; fixed `.metrics` dict key names in model_retraining.py |
| `700381a` | Add `build_training_baseline.py`, CLI flags | Created missing `scripts/build_training_baseline.py`; exposed `adabn_tent`/`tent` as `--adapt` choices |
| `bd8dc1e` | Hardening: PSI test, Stage 6 schema guard | Fixed PSI test column mismatch; added schema guard so partial CSVs don't crash Stage 6 |
| `1ae27cc` | CI: split unit/slow tests | Added `unit`/`slow`/`integration` pytest markers; split GitHub Actions into fast + slow jobs |

---

### 1b. Audit Runs (Commit 5 — `3fd3c00`)

All four thesis audit experiments completed and artifacts verified:

| Audit | Command | Status | Key Metrics |
|-------|---------|:------:|-------------|
| **A1** — Inference baseline | `--skip-ingestion` | ✅ | 1,027 windows, conf 84.6%, drift PSI 0.203 |
| **A3** — Supervised retrain | `--retrain --adapt none --skip-cv --epochs 10` | ✅ | val_acc 0.969, F1 0.814 |
| **A4** — AdaBN+TENT | `--retrain --adapt adabn_tent` | ✅ | entropy 0.204→0.207 (Δ+0.003, threshold=0.05, accepted) |
| **A5** — Pseudo-label | `--retrain --adapt pseudo_label --epochs 10` | ✅ | val_acc 0.969, 43 pseudo-labeled samples selected |

**Bugs fixed during runs:**
- A3 "Training data not found" → fixed fallback path to prepared CSV in `model_retraining.py`
- A5 `_get_logits()` returned `None` for softmax head → added temperature re-scaling fallback

**Docs created:**
- `docs/THESIS_OBJECTIVES_TRACEABILITY.md` — every thesis RQ mapped to code + commit + test
- `docs/TRAINING_RECIPE_MATRIX.md` — ablation comparison table
- `scripts/audit_artifacts.py` — automated artifact presence/size checker

---

## SESSION 2 — Afternoon (Commits 6–7)

### Goal
Apply a second technical review's recommendations — 5 code-quality issues identified.

---

### 2a. Commit 6 — `f47a48b`

#### TENT Running-Stats Bug Fix (`src/domain_adaptation/tent.py`)

**Root cause:** `model(batch, training=True)` inside the gradient loop updates `moving_mean`/`moving_variance` as a side-effect every step. After AdaBN set those stats, TENT was corrupting them — entropy *increased* instead of decreasing.

**Fix applied:**
- Snapshot `initial_running` (all BN `moving_mean`/`moving_variance`) before the loop
- After each `apply_gradients`: restore with `.assign()` — only `gamma`/`beta` ever change
- Post-loop: if `entropy_delta > rollback_threshold (0.05)`, restore `gamma`/`beta` too
- Return changed to `(model, meta_dict)` with keys: `tent_rollback`, `tent_entropy_before/after/delta`, `tent_ood_skipped`

#### `--update-baseline` Governance (`baseline_update.py`, `config_entity.py`, `run_pipeline.py`)

`models/training_baseline.json` was silently overwritten on every run. Fixed:
- Default `BaselineUpdateConfig.promote_to_shared = False`
- Baseline built and saved to `artifacts/` + MLflow only by default
- `--update-baseline` CLI flag required to touch `models/`

#### Other additions in this commit
- `_detect_gpu()` startup function in `run_pipeline.py`
- `forced_retrain` / `retrain_forced_by_cli` params logged to MLflow per run
- **Pseudo-label rollback safety** in `src/train.py`: evaluate base_model on 20% source holdout before and after fine-tuning; revert if acc drops > 10 pp
- `scripts/export_mlflow_runs.py` — new CLI tool: `python scripts/export_mlflow_runs.py --experiment har-retraining` exports to CSV
- Three docs created: `DOCUMENTATION_INDEX.md`, `PIPELINE_RUNBOOK.md`, `REMAINING_WORK_FEB_TO_MAY_2026.md`

---

### 2b. Commit 7 — `e2bc784`

Four follow-up fixes after running the code and checking the logs:

#### Fix 1 — Baseline governance actually enforced (`baseline_update.py`)

**Problem:** Despite the "NOT promoted" log message, `builder.save()` was still called with the `models/` path before the governance check. The flag only guarded the archive copy.  
**Real fix:** When `promote=False`, output paths are redirected to `artifacts/.../models/` so `models/training_baseline.json` is *never created or modified*.

#### Fix 2 — TENT confidence-drop rollback (`tent.py`)

**Problem:** Latest run: entropy Δ=+0.003 (accepted, within 0.05 threshold), but **mean confidence dropped 0.8995→0.8207 (Δ=−0.079)**. Adaptation was harmful but passing.  
**Fix:** Added second rollback gate — rollback if `mean_conf_after < mean_conf_before − 0.01`.  
With the previous run's numbers: **would have rolled back** (Δconf=−0.079 << −0.01).  
New meta keys: `tent_confidence_before`, `tent_confidence_after`, `tent_confidence_delta`.

#### Fix 3 — `tf.function` retracing eliminated (`tent.py`)

Replaced `model.predict(...)` calls with `model(tf.constant(X), training=False)`. The latter reuses the already-compiled graph; `predict()` reconstructs its tf.function with each varying-shape call → 5–6 retracing warnings per run. Now zero.

#### Fix 4 — GPU message informative (`run_pipeline.py`)

When TF ≥ 2.11 runs on Windows-native without GPU detected:
> `CPU — TF 2.20 on Windows-native has no CUDA GPU support (last native-Windows GPU build was TF 2.10). Use WSL2: tensorflow.org/install`

This is correct behaviour (TF dropped Windows CUDA after 2.10); message is now thesis-reviewer-friendly instead of silent.

#### Fix 5 — MLflow `artifact_path` deprecation (`mlflow_tracking.py`, `production_pipeline.py`)

- `mlflow.keras.log_model` now uses `name=` (MLflow ≥2.9) with `try/except TypeError` fallback
- Auto-generates `input_example` from `model.input_shape` so MLflow stores the model signature

---

## End-of-Day State

```
HEAD:         e2bc784 (origin/main)
Pipeline:     9 stages end-to-end ✅ (ingestion skipped)
Audit runs:   A1 ✅  A3 ✅  A4 ✅  A5 ✅  (all PASS, artifacts committed)
TENT:         BN-stats freeze ✅ | rollback-entropy ✅ | rollback-confidence ✅
Baseline:     Governance gate real ✅ (models/ never written unless --update-baseline)
Pseudo-label: Rollback safety ✅ (pre/post holdout + 10pp threshold)
MLflow:       Signature ✅ | name= API ✅ | forced_retrain logged ✅
```

**One remaining validation item:**  
Re-run A4 to confirm the confidence-drop rollback triggers correctly:
```powershell
python run_pipeline.py --retrain --adapt adabn_tent --skip-ingestion 2>&1 | Tee-Object -FilePath "logs\a4_audit_v2.txt"
# Expect: tent_rollback=True, reason "confidence dropped 0.8995→0.8207 (Δ=−0.079 < −0.01)"
```

---

## Key Lessons

1. **"Log message ≠ actual behaviour"** — The baseline governance log said "NOT promoted" while `builder.save(models/...)` still executed. Always verify with logs which *path* the builder received.
2. **Entropy alone is not enough for TENT acceptance** — Confidence drop is faster and more interpretable. Δentropy=+0.003 passed; Δconf=−0.079 would not.
3. **`model.predict()` inside loops always causes retracing** — Use `model(tf.constant(X), training=False)` wherever input shape is stable.
4. **TF ≥ 2.11 + Windows native = no GPU** — Not a code bug. An informative message saves future confusion.
5. **Snapshot-restore is the correct test-time adaptation pattern** — TENT may only change affine parameters; running statistics must remain as set by AdaBN.

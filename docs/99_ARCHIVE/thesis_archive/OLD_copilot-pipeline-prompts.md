# Copilot Prompt — Pipeline Operations Guide

> **How to use:** Copy any prompt below into Copilot Chat.  
> These prompts help you understand and operate the pipeline day-to-day.

---

## SESSION DETECTION & PROCESSING

### P1 — Auto-Detect and Process All Unprocessed Sessions

```
Read src/sensor_data_pipeline.py (the find_latest_sensor_pair function) and
src/components/data_ingestion.py.

Our raw data is in data/raw/ with 26 Decoded sessions. Each session is a group
of 2-3 CSVs sharing the SAME timestamp prefix, for example:
  2025-07-16-21-03-13_accelerometer.csv
  2025-07-16-21-03-13_gyroscope.csv
  2025-07-16-21-03-13_record.csv

The timestamp prefix (e.g. "2025-07-16-21-03-13") is the SESSION ID.

Currently find_latest_sensor_pair() only finds the NEWEST pair by file
modification time. We need a batch processing flow:

1. Scan data/raw/ and group files by session ID (the prefix before _accelerometer / _gyroscope / _record).
2. For each session ID, check if an output already exists in data/processed/ by looking for a file named
   "sensor_fused_50Hz_{session_id}.csv" or by checking a JSON registry file at data/processed/session_registry.json.
3. If already processed → skip and log "Session {session_id} already processed, skipping."
4. If NOT processed → run ingestion (fuse accel + gyro → 50Hz CSV), save output, update the registry.
5. After all sessions: print summary of processed / skipped / failed.

Write a function called discover_all_sessions(raw_dir: Path) -> List[dict] that returns:
  [{"session_id": "2025-07-16-21-03-13", "accel": Path(...), "gyro": Path(...), "record": Path(...) or None}]

Write a function called load_session_registry(processed_dir: Path) -> Set[str] that reads
data/processed/session_registry.json and returns the set of already-processed session IDs.

Write a function called update_session_registry(processed_dir: Path, session_id: str, output_path: Path).

Then integrate these into the DataIngestion component so that when --input-csv is NOT provided
and no Excel files are found, it falls back to batch-processing all unprocessed Decoded sessions.

Also add a --batch flag to run_pipeline.py to explicitly trigger batch mode.
```

### P2 — How Session ID Detection Works (Explain It to Me)

```
Look at the files in data/raw/. The filenames follow this pattern:
  {YYYY-MM-DD-HH-MM-SS}_accelerometer.csv
  {YYYY-MM-DD-HH-MM-SS}_gyroscope.csv
  {YYYY-MM-DD-HH-MM-SS}_record.csv

The part before the underscore is the session timestamp, which acts as a
unique session ID. Files with the SAME prefix belong to the SAME recording session.

Explain:
1. How can we extract session IDs from filenames using Python?
2. How do we group accelerometer + gyroscope + record files into sessions?
3. How do we compare against already-processed sessions to find only NEW ones?
4. Give a concrete example with 3 sessions: 1 already processed, 2 new.

Use our actual file list from data/raw/ in your explanation.
```

### P3 — Skip Already Processed, Notify User

```
Read data/processed/ and data/raw/.

I want a quick status report. For each session in data/raw/:
1. Extract the session ID from the filename prefix.
2. Check if data/processed/ contains output for this session.
3. Report: "✓ {session_id} — already processed" or "✗ {session_id} — NEW, needs processing"
4. At the end: "Summary: X/Y sessions processed, Z remaining."

Write a Python script that does this check. No processing — just the status report.
```

---

## STAGE GROUPS EXPLAINED

### P4 — Why Are Stages Split Into Three Groups?

```
Read src/pipeline/production_pipeline.py (the ALL_STAGES, RETRAIN_STAGES, ADVANCED_STAGES
constants) and run_pipeline.py.

Explain to me like I'm defending my thesis:

1. WHY are stages split into [1-7], [8-10], [11-14]?
   - What is the cost (time, compute) of each group?
   - What is the frequency each group should run?

2. CAN I run all 14 stages at once? How?
   Answer: yes, `python run_pipeline.py --retrain --advanced`

3. When would I run ONLY stages 1-7?
   → Every time new data arrives. Quick inference cycle. ~2 minutes on CPU.

4. When would I run stages 8-10 (retraining)?
   → Only when Stage 7 trigger fires (model degradation detected).
   → Or manually when I want to retrain with new adaptation method.

5. When would I run stages 11-14 (advanced)?
   → After retraining to re-calibrate the model.
   → Or once at deployment to set baselines.
   → These are CALIBRATION stages — they establish the thresholds
     that stages 1-7 use for monitoring.

6. Can I run a single stage? How?
   → `python run_pipeline.py --stages calibration`
   → The pipeline creates fallback artifacts for missing upstream outputs.

7. What is the recommended workflow for a new recording session?
   Step 1: python run_pipeline.py --input-csv data/raw/2025-07-16-21-03-13_accelerometer.csv
   Step 2: Check MLflow UI for results
   Step 3: If trigger fires → python run_pipeline.py --retrain --adapt adabn
   Step 4: After retraining → python run_pipeline.py --advanced
```

### P5 — What Exactly Does Each Stage Do?

```
Read src/pipeline/production_pipeline.py and all 14 files in src/components/.

For each of the 14 stages, give me a one-paragraph explanation in plain language:
- What goes IN (input artifact)
- What comes OUT (output artifact)
- WHY it exists (what problem it solves)
- How LONG it takes approximately
- Can I SKIP it? What happens if I do?

Format as a table:
| Stage | Input | Output | Purpose | Time | Skippable? |
```

---

## AdaBN AND PSEUDO-LABELING EXPLAINED

### P6 — How Does AdaBN Help in Retraining?

```
Read src/domain_adaptation/adabn.py and src/components/model_retraining.py.

Explain AdaBN (Adaptive Batch Normalisation) to me as if I'm presenting it
at my thesis defence:

1. PROBLEM: Our model was trained on User A's data. User B moves differently.
   The feature distributions (mean, variance) are shifted. The BatchNorm layers
   in the neural network store User A's statistics → poor performance on User B.

2. SOLUTION: AdaBN freezes all model weights but updates ONLY the BatchNorm
   running_mean and running_variance by passing User B's data through the network.
   No gradient descent. No labels needed. Takes seconds, not minutes.

3. WHEN TO USE: Mild domain shift (same user different day, slight sensor
   position change). It's the LIGHTEST adaptation — try this first.

4. LIMITATION: Cannot fix large distribution shifts (completely new user,
   different sensor hardware). For that, use pseudo-labeling.

5. HOW TO RUN:
   python run_pipeline.py --retrain --adapt adabn

6. WHAT HAPPENS INTERNALLY:
   - Freeze all layers except BatchNorm
   - Forward pass N batches of production data
   - BatchNorm layers update their statistics to match production distribution
   - Save adapted model → run Stage 9 (registration) with proxy validation
```

### P7 — How Does Curriculum Pseudo-Labeling Help?

```
Read src/curriculum_pseudo_labeling.py completely.

Explain curriculum pseudo-labeling to me as if I'm presenting at my thesis defence:

1. PROBLEM: We have a trained model and NEW unlabeled production data.
   We want to adapt the model but we have NO LABELS. Traditional retraining
   requires labels. How do we retrain without labels?

2. SOLUTION: Use the model's OWN predictions as "pseudo-labels" — but only
   the HIGH-CONFIDENCE ones. Start with very strict threshold (0.95), gradually
   lower it (0.80) across iterations. This is the "curriculum" — easy first,
   hard later.

3. KEY INNOVATION vs SelfHAR:
   - SelfHAR uses contrastive pre-training (training-time technique)
   - Our approach uses curriculum thresholding (deployment-time technique)
   - We add EWC to prevent forgetting the original training distribution

4. EWC SAFEGUARD: Without EWC, the model "forgets" what it learned from
   labeled data. EWC adds a penalty that keeps important weights close to
   their original values. Think of it as "learn new things but don't forget
   the old things."

5. WHEN DOES IT FAIL? Three safeguards:
   - Entropy monitoring: if model gets MORE uncertain → stop (degenerate labels)
   - Class diversity: if fewer than 3 classes in pseudo-labels → skip iteration
   - Proxy validation: adapted model must be BETTER than current model on proxy
     metrics, otherwise deployment is blocked

6. HOW TO RUN:
   python run_pipeline.py --retrain --adapt pseudo_label
   python run_pipeline.py --stages curriculum_pseudo_labeling --curriculum-iterations 10 --ewc-lambda 500
```

---

## RUNNING THE PROJECT

### P8 — How to Set Up and Run Everything

```
Read pyproject.toml, run_pipeline.py, and config/pipeline_config.yaml.

Give me a step-by-step guide for a FRESH machine:

1. SETUP:
   git clone <repo>
   cd MasterArbeit_MLops
   python -m venv venv
   venv\Scripts\activate        (Windows)
   pip install -e ".[dev]"      (installs from pyproject.toml)

2. VERIFY:
   python -c "from src.pipeline.production_pipeline import ProductionPipeline; print('OK')"
   pytest tests/ -v --tb=short

3. RUN INFERENCE ON ONE SESSION:
   python run_pipeline.py --input-csv "data/raw/2025-07-16-21-03-13_accelerometer.csv"

4. VIEW RESULTS:
   mlflow ui --backend-store-uri mlruns --port 5000
   Open http://localhost:5000

5. RUN FULL PIPELINE:
   python run_pipeline.py --retrain --advanced --continue-on-failure

6. RUN TESTS:
   pytest tests/ -v
   pytest tests/ --cov=src --cov-report=html

7. DOCKER:
   docker-compose up -d
   curl http://localhost:8000/health

Explain what "pip install -e '.[dev]'" does:
- The -e flag installs in "editable" mode (changes to src/ take effect immediately)
- The [dev] part installs optional dev dependencies (pytest, flake8, black, etc.)
- This reads from pyproject.toml [project.dependencies] and [project.optional-dependencies.dev]
```

### P9 — What Is Our Current Model and Can I Use a Custom One?

```
Read run_pipeline.py (the --model argument) and src/entity/config_entity.py (ModelInferenceConfig).

1. DEFAULT MODEL: models/pretrained/fine_tuned_model_1dcnnbilstm.keras
   This is a 1D-CNN-BiLSTM trained on 11-class anxiety-related activities.
   It's the ONLY model we currently have. Model architecture is fixed.

2. CUSTOM MODEL: You can override it:
   python run_pipeline.py --model "path/to/my_model.keras"

3. MODEL PRIORITY (highest to lowest):
   a) --model CLI argument
   b) MLflow Model Registry (latest Production stage model)
   c) Default path: models/pretrained/fine_tuned_model_1dcnnbilstm.keras

4. For this thesis, we use the DEFAULT model. Custom model path exists
   for future extensibility (different architectures, student-teacher models).

5. After retraining (Stage 8), the new model goes through registration (Stage 9)
   with proxy validation. Only if it passes does it become the new production model.
```

---

## CLEANUP & FOCUS

### P10 — What Can I Ignore Right Now?

```
Look at the workspace root directory.

These are PRODUCTION (keep and use):
  src/                  ← All pipeline code
  tests/                ← All tests
  config/               ← Pipeline + MLflow + monitoring config
  docker/               ← Dockerfiles + API
  scripts/              ← Utility scripts
  data/                 ← Raw + processed data
  models/               ← Trained models
  docs/thesis/          ← Thesis chapters + evaluation framework
  .github/              ← CI/CD
  run_pipeline.py       ← Entry point
  pyproject.toml        ← Package definition
  docker-compose.yml    ← Container orchestration

These are RESEARCH (can be moved to separate repo later):
  papers/               ← 76+ PDFs (not used by pipeline)
  research_papers/      ← Duplicate of papers/
  notebooks/            ← Experimental notebooks
  archive/              ← Old scripts and notes
  ai helps/             ← AI conversation logs
  cheat sheet/          ← Personal reference
  images/               ← Miscellaneous

Don't delete anything now. Focus on running the pipeline and writing the thesis.
Cleanup is a one-time 2-hour task for later (see docs/PIPELINE_OPERATIONS_AND_ARCHITECTURE.md §7).
```

---

## QUICK OPERATIONS (Copy-Paste Commands)

```bash
# === INFERENCE (most common — run on new data) ===
python run_pipeline.py --input-csv "data/raw/2025-07-16-21-03-13_accelerometer.csv"

# === BATCH ALL SESSIONS (when implemented) ===
python run_pipeline.py --batch

# === FULL PIPELINE (inference + retrain + calibrate) ===
python run_pipeline.py --retrain --advanced --continue-on-failure

# === RETRAIN WITH AdaBN (light adaptation) ===
python run_pipeline.py --retrain --adapt adabn

# === RETRAIN WITH PSEUDO-LABELING (heavy adaptation) ===
python run_pipeline.py --retrain --adapt pseudo_label --curriculum-iterations 10

# === CALIBRATION ONLY (after retraining) ===
python run_pipeline.py --stages calibration wasserstein_drift sensor_placement

# === RUN TESTS ===
pytest tests/ -v --tb=short

# === UPDATE PROGRESS DASHBOARD ===
python scripts/update_progress_dashboard.py

# === VIEW RESULTS ===
mlflow ui --backend-store-uri mlruns --port 5000

# === DOCKER FULL STACK ===
docker-compose up -d
curl http://localhost:8000/health
```

---

*Copy any prompt above into Copilot Chat for step-by-step guidance about your pipeline.*

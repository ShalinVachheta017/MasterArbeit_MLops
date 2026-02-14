# Copilot Prompt Form — System Evaluation Queries

> **How to use:** Copy-paste any prompt below into GitHub Copilot Chat.
> Each prompt is self-contained and references your actual codebase.
> Prompts are grouped by evaluation lens (Researcher / Engineer / Examiner).

---

## RESEARCHER PROMPTS

### R1 — Validate Monitoring Equations Against Code

```
Read src/trigger_policy.py and src/evaluate_predictions.py.

For each of the three monitoring layers (confidence, temporal, drift):
1. Extract the exact threshold values used in code.
2. Write the formal equation for each signal (e.g., FlipRate, PSI, mean confidence).
3. Check: do the code thresholds match the values documented in docs/thesis/SYSTEM_EVALUATION_FRAMEWORK.md?
4. If any mismatch, list the discrepancy with file, line number, and both values.
```

### R2 — Verify EWC Implementation

```
Read src/curriculum_pseudo_labeling.py completely.

Answer:
1. How is the Fisher Information Matrix diagonal computed? Which samples are used?
2. What is the exact EWC loss formula implemented? Write it in LaTeX.
3. Is the EWC penalty applied to ALL parameters or only specific layers?
4. What happens if ewc_lambda = 0? Does the code degrade gracefully to standard fine-tuning?
5. Compare this implementation against Kirkpatrick et al. 2017 — are there any deviations?
```

### R3 — Audit Pseudo-Label Safeguards

```
Read src/curriculum_pseudo_labeling.py.

List all safeguards against confirmation bias / degenerate pseudo-labels:
1. Entropy monitoring — what triggers early stopping?
2. Class diversity gate — what is the minimum class count?
3. Proxy validation — how is the adapted model compared to the current model?
4. For each safeguard, give the exact code location (function name + approximate line).
5. Are there any failure modes NOT covered? Suggest improvements.
```

### R4 — Compare AdaBN vs Curriculum Pseudo-Labeling

```
Read src/domain_adaptation/adabn.py and src/curriculum_pseudo_labeling.py.

Create a comparison table with columns:
| Aspect | AdaBN | Curriculum Pseudo-Labeling |
Rows: requires labels?, modifies weights?, compute time, risk of forgetting,
suitable domain shift magnitude, number of hyperparameters, references.

Then: under what conditions should a user choose one over the other?
```

### R5 — Check Calibration Mathematical Correctness

```
Read src/calibration.py completely.

1. Write the ECE formula as implemented (not the textbook version — the actual code).
2. How many bins are used? Is adaptive binning or fixed-width binning used?
3. Is Brier score implemented? Write its formula.
4. For MC Dropout: how are K forward passes implemented? Is dropout forced on at inference?
5. Is predictive entropy computed correctly? Write the formula and verify against code.
6. What is the difference between aleatoric and epistemic uncertainty in this implementation?
```

### R6 — Audit Wasserstein Drift Detection

```
Read src/wasserstein_drift.py completely.

1. Which scipy function computes the Wasserstein distance? Is it W1 or W2?
2. How is change-point detection implemented? Is it a rolling-window z-score approach?
3. What are the default thresholds for warn and critical drift?
4. How does multi-resolution detection work (window, hourly, daily)?
5. Compare: what can Wasserstein detect that KS-test misses? Give a concrete example.
```

### R7 — Review OOD Detection

```
Read src/ood_detection.py completely.

1. Write the energy score formula as implemented.
2. How is the temperature parameter used in energy computation?
3. What is the ensemble approach? Which scores are combined and with what weights?
4. What are the default energy thresholds and how were they chosen?
5. How does this relate to the NeurIPS 2020 paper by Liu et al.?
```

---

## ENGINEER PROMPTS

### E1 — Full Test Coverage Audit

```
List all files in tests/ and all files in src/.
For each src/ module, check if a corresponding test file exists.
Create a coverage matrix:
| src module | test file | exists? | test count (grep for 'def test_') |
Flag any src module with 0 tests as UNTESTED.
```

### E2 — CI/CD Pipeline Review

```
Read .github/workflows/ci-cd.yml completely.

1. Draw the job dependency graph (which job depends on which).
2. For each job, list: trigger condition, runner OS, key steps, failure handling.
3. Is there a model validation job that actually runs? Or is it a placeholder?
4. What Docker registry is used? Is caching configured?
5. What is missing for a production-grade CI/CD? List gaps.
```

### E3 — Docker Deployment Audit

```
Read docker/Dockerfile.inference, docker/Dockerfile.training, docker-compose.yml, and docker/api/main.py.

1. Are the Dockerfiles using multi-stage builds? Should they?
2. What is the base image and Python version?
3. Is the model baked into the image or mounted at runtime?
4. Does docker-compose.yml define health checks for each service?
5. Is there a volume mount for MLflow data persistence?
6. What happens if the inference container runs out of memory?
```

### E4 — Pipeline Fault Tolerance

```
Read src/pipeline/production_pipeline.py and run_pipeline.py.

1. What happens when a stage fails and --continue-on-failure is set?
2. How are fallback artifacts created? Show the code path.
3. What is logged on failure (traceback, stage name, duration)?
4. Can the pipeline resume from a failed stage without re-running prior stages?
5. Is there a retry mechanism for transient failures?
```

### E5 — Reproducibility Verification

```
Read run_pipeline.py, src/mlflow_tracking.py, and pyproject.toml.

1. Are all CLI arguments logged to MLflow?
2. Is the Git commit SHA recorded per run?
3. Are random seeds set for NumPy, TensorFlow, and Python's random module?
4. Can I reproduce a specific past run using only its MLflow record? What would I need?
5. Are DVC file hashes logged alongside the MLflow run?
```

### E6 — Monitoring Stack Readiness

```
Read config/prometheus.yml, config/grafana/har_dashboard.json, config/alerts/har_alerts.yml, and src/prometheus_metrics.py.

1. What metrics are exported to Prometheus? List them with their types (counter, gauge, histogram).
2. What alert rules are defined? For each: metric, threshold, severity, action.
3. Is the Grafana dashboard JSON importable as-is? What datasource does it expect?
4. Has this stack been deployed and tested? Check for evidence in reports/ or logs/.
```

---

## EXAMINER PROMPTS

### X1 — Examiner Stress Test

```
Read docs/thesis/SYSTEM_EVALUATION_FRAMEWORK.md and docs/thesis/chapters/CH1_INTRODUCTION.md.

Act as a thesis examiner. For each of the 6 novelty claims:
1. Is the claim precisely defined with a formal equation?
2. Is there a corresponding experiment designed to validate it?
3. Is there an ablation that isolates the contribution?
4. What is the weakest claim? Why?
5. What question would you ask the candidate during the oral defence?
```

### X2 — Results Gap Analysis

```
Read docs/thesis/SYSTEM_EVALUATION_FRAMEWORK.md.

List every table with "TODO" entries. For each:
1. What data is needed to fill it?
2. Which script or command generates that data?
3. Estimate the time to run the experiment.
4. Rank by importance: which tables are essential vs. nice-to-have?
```

### X3 — Literature Positioning

```
Read docs/APPENDIX_PAPER_INDEX.md and docs/thesis/chapters/CH1_INTRODUCTION.md.

For each novelty claim, find at least 2 papers from our collection that address
the same problem differently. Create a comparison:
| Our Approach | Alternative (Paper X) | Key Difference |

This is needed for the Related Work section of Chapter 2.
```

### X4 — Reproducibility Challenge

```
Read docs/thesis/chapters/CH4_IMPLEMENTATION.md and pyproject.toml.

Imagine you are a reviewer trying to reproduce our results from scratch.
1. Can you set up the environment from pyproject.toml alone?
2. Is the data publicly available or do you need DVC remote credentials?
3. Are pre-trained model weights available?
4. Can you run the full pipeline with one command?
5. What would block you? List every friction point.
```

### X5 — Statistical Validity Check

```
Read src/evaluate_predictions.py and src/calibration.py.

1. Does the system report confidence intervals for F1 scores?
2. Is there a statistical test for comparing two models (e.g., McNemar)?
3. How many cross-validation folds are used?
4. Is stratification applied in the CV splits?
5. Are results averaged over multiple random seeds?
6. What is missing for publication-grade statistical reporting?
```

### X6 — System vs Literature Gap

```
Read all files in docs/thesis/chapters/ and src/curriculum_pseudo_labeling.py.

Compare our curriculum pseudo-labeling against:
1. SelfHAR (Tang et al., 2021) — what do they do that we don't?
2. ADAPT (2023) — how does their threshold strategy differ?
3. FixMatch (Sohn et al., 2020) — could we use their consistency regularisation?

For each: state what we could add and whether it's worth the complexity.
```

---

## QUICK CHECKS (Single-Line Prompts)

```
# Check if all 14 stages have matching config + artifact dataclasses:
Read src/entity/config_entity.py and src/entity/artifact_entity.py. Count dataclasses in each. Do they match the 14 stages?

# Check trigger thresholds are consistent:
Grep for 'confidence_critical', 'flip_rate', 'psi' across all src/ files. Are the values consistent?

# Check if EWC lambda default is the same everywhere:
Grep for 'ewc_lambda' across all files. List every occurrence with its value.

# Find all TODO/FIXME in the codebase:
Search for TODO|FIXME|HACK|XXX across all .py and .md files. List them grouped by file.

# Verify all test markers are registered:
Read pytest.ini and pyproject.toml. List all registered markers. Then grep tests/ for @pytest.mark — are all used markers registered?
```

---

*Copy any prompt above into Copilot Chat to get research-backed, evidence-grounded answers about your system.*

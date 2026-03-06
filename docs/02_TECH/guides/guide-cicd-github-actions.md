# GitHub Actions CI/CD Guide

> **How the automated pipeline works, step by step**

---

## Table of Contents

1. [What is GitHub Actions?](#1-what-is-github-actions)
2. [Your CI/CD Workflow Overview](#2-your-cicd-workflow-overview)
3. [When Does It Trigger?](#3-when-does-it-trigger)
4. [Job 1: Code Quality (Linting)](#4-job-1-code-quality-linting)
5. [Job 2: Unit Tests](#5-job-2-unit-tests)
6. [Job 3: Build Docker Image](#6-job-3-build-docker-image)
7. [Job 4: Integration Tests (Smoke)](#7-job-4-integration-tests-smoke)
8. [Job 5: Model Validation](#8-job-5-model-validation)
9. [Job 6: Failure Notification](#9-job-6-failure-notification)
10. [How to Push Code and Trigger the Pipeline](#10-how-to-push-code-and-trigger-the-pipeline)
11. [How to Add Secrets & Environment Variables](#11-how-to-add-secrets--environment-variables)
12. [Viewing Build Results on GitHub](#12-viewing-build-results-on-github)
13. [Customizing the Workflow](#13-customizing-the-workflow)
14. [Common Issues & Fixes](#14-common-issues--fixes)

---

## 1. What is GitHub Actions?

GitHub Actions is a **CI/CD platform** built into GitHub. It automatically runs tasks (linting, testing, building, deploying) whenever you push code or create a pull request.

**Key concepts:**

| Term | Meaning |
|---|---|
| **Workflow** | A `.yml` file in `.github/workflows/` that defines the automation |
| **Job** | A group of steps that run on a virtual machine (e.g., `ubuntu-latest`) |
| **Step** | A single task within a job (e.g., "install Python", "run tests") |
| **Trigger** | What causes the workflow to run (push, pull request, schedule, manual) |
| **Artifact** | Files produced by a job (test reports, Docker images) |
| **Secret** | Encrypted variable stored in GitHub (API keys, tokens) |

Your workflow file: **`.github/workflows/ci-cd.yml`**

---

## 2. Your CI/CD Workflow Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Code Push   │────▶│    LINT      │────▶│    TEST     │
│  or PR       │     │  (flake8,   │     │  (pytest +  │
│              │     │   black,    │     │   coverage) │
│              │     │   isort)    │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ INTEGRATION │◀────│   BUILD     │
                    │   TEST      │     │  (Docker    │
                    │ (smoke test)│     │   image)    │
                    └─────────────┘     └─────────────┘
                                               │
                    ┌─────────────┐             │
                    │   NOTIFY    │◀────────────┘
                    │ (on failure)│      (if any job fails)
                    └─────────────┘

  Separate trigger:
  ┌─────────────┐
  │   MODEL     │  ← manual trigger or scheduled
  │ VALIDATION  │
  └─────────────┘
```

**Execution order:** Lint → Test → Build → Integration Test → (Notify on failure)

Jobs depend on each other — if lint fails, tests won't run. If tests fail, Docker won't build.

---

## 3. When Does It Trigger?

The workflow runs automatically when:

```yaml
on:
  push:
    branches: [main, develop]        # Push to main or develop
    paths:                           # Only when these files change:
      - 'src/**'                     #   Source code
      - 'tests/**'                   #   Test files
      - 'docker/**'                  #   Docker configs
      - 'config/**'                  #   Configuration
      - 'requirements.txt'           #   Dependencies
      - '.github/workflows/**'       #   Workflow itself
  pull_request:
    branches: [main]                 # Any PR targeting main
  workflow_dispatch:                 # Manual button in GitHub UI
```

**What this means:**
- ✅ Pushing Python code changes → triggers pipeline
- ✅ Creating a PR to merge into `main` → triggers pipeline
- ✅ Clicking "Run workflow" button on GitHub → triggers pipeline
- ❌ Pushing changes to docs, papers, notebooks → does NOT trigger
- ❌ Pushing to branches other than `main`/`develop` → does NOT trigger

---

## 4. Job 1: Code Quality (Linting)

**Purpose:** Check code style and catch basic errors before running tests.

**What it does:**

| Step | Tool | What It Checks |
|---|---|---|
| flake8 (strict) | `flake8` | Syntax errors, undefined names, import issues (E9, F63, F7, F82) |
| flake8 (advisory) | `flake8` | Code complexity, line length (non-blocking, exit-zero) |
| black | `black` | Code formatting (PEP 8 style, non-blocking) |
| isort | `isort` | Import statement ordering (non-blocking) |

**When it fails:** Only if there are actual syntax errors or undefined variables. Formatting issues are reported but don't block the pipeline.

**Runs on:** `ubuntu-latest` with Python 3.11

---

## 5. Job 2: Unit Tests

**Purpose:** Run all tests and measure code coverage.

**Depends on:** Job 1 (Lint) must pass first.

**What it does:**

```bash
pytest tests/ -v \
    --cov=src \                    # Measure coverage of src/ folder
    --cov-report=xml \             # XML report for Codecov
    --cov-report=html \            # HTML report (browseable)
    --cov-report=term-missing \    # Terminal output showing missed lines
    --junitxml=test-results.xml    # JUnit XML for GitHub display
```

**Artifacts produced:**
- `test-results.xml` — test results (visible in GitHub Actions tab)
- `htmlcov/` — HTML coverage report (downloadable)
- `coverage.xml` — uploaded to Codecov for coverage tracking

**When it fails:** If any test in `tests/` fails. Fix the failing test, push again.

---

## 6. Job 3: Build Docker Image

**Purpose:** Build the inference Docker image and push it to GitHub Container Registry (ghcr.io).

**Depends on:** Job 2 (Test) must pass first.

**Only runs on:** Direct pushes (not PRs) — `if: github.event_name != 'pull_request'`

**What it does:**

1. **Login** to `ghcr.io` using `GITHUB_TOKEN` (automatic, no setup needed)
2. **Build** Docker image using `docker/Dockerfile.inference`
3. **Tag** with:
   - Branch name (e.g., `main`, `develop`)
   - Commit SHA (e.g., `abc1234`)
   - `latest` (only on `main` branch)
4. **Push** to `ghcr.io/<your-username>/MasterArbeit_MLops/har-inference`
5. **Cache** layers using GitHub Actions cache (faster rebuilds)

**Image naming:**
```
ghcr.io/<owner>/masterarbeit_mlops/har-inference:main
ghcr.io/<owner>/masterarbeit_mlops/har-inference:abc1234
ghcr.io/<owner>/masterarbeit_mlops/har-inference:latest
```

---

## 7. Job 4: Integration Tests (Smoke)

**Purpose:** Start the Docker container and verify it responds correctly.

**Depends on:** Job 3 (Build) must pass first.

**Only runs on:** `main` branch — `if: github.ref == 'refs/heads/main'`

**What it does:**

1. Pull the just-built Docker image
2. Start container on port 8000
3. Wait 10 seconds for startup
4. Run health check: `curl http://localhost:8000/health`
5. Run smoke test: `python scripts/inference_smoke.py --endpoint http://localhost:8000`
6. Stop and remove container

**When it fails:** The Docker container crashed, the API doesn't respond, or the health endpoint returns an error.

---

## 8. Job 5: Model Validation

**Purpose:** Validate the deployed model hasn't degraded (placeholder — needs implementation).

**Triggers:** Only on `workflow_dispatch` (manual) or `schedule` (not configured yet).

**Currently:** Echoes placeholder messages. To activate, uncomment the DVC pull and validation script lines.

**To configure scheduled runs**, add to the `on:` section:

```yaml
on:
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6:00 AM UTC
```

---

## 9. Job 6: Failure Notification

**Purpose:** Alert when the pipeline fails.

**Triggers:** Only when lint, test, or build jobs fail — `if: failure()`

**Currently:** Prints a message. To add Slack notifications:

```yaml
- name: Send Slack notification
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ml-pipeline'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 10. How to Push Code and Trigger the Pipeline

### First Time Setup

```bash
# 1. Initialize git (if not done)
git init

# 2. Add remote
git remote add origin https://github.com/<your-username>/MasterArbeit_MLops.git

# 3. Create main branch
git checkout -b main
```

### Daily Workflow

```bash
# 1. Check what changed
git status

# 2. Stage your changes
git add src/ tests/ config/

# 3. Commit with a clear message
git commit -m "feat: add Wasserstein drift detection module"

# 4. Push to GitHub
git push origin main
```

**After pushing:**
1. Go to your GitHub repository
2. Click the **"Actions"** tab
3. You'll see the workflow running
4. Click on it to see each job's progress and logs

### Manual Trigger

1. Go to **Actions** tab on GitHub
2. Select **"HAR MLOps CI/CD"** workflow
3. Click **"Run workflow"** dropdown
4. Select branch and click **"Run workflow"**

---

## 11. How to Add Secrets & Environment Variables

### Required Secrets

The workflow uses `GITHUB_TOKEN` automatically (no setup needed). If you add external services:

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**

| Secret Name | Purpose | Required? |
|---|---|---|
| `GITHUB_TOKEN` | Docker registry login | Auto-provided |
| `SLACK_WEBHOOK` | Slack notifications | Optional |
| `CODECOV_TOKEN` | Coverage uploads | Optional |

### Environment Variables

Already configured in the workflow:

```yaml
env:
  PYTHON_VERSION: '3.11'
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/har-inference
```

---

## 12. Viewing Build Results on GitHub

### Actions Tab

1. Navigate to your repo → **Actions** tab
2. Each workflow run shows:
   - ✅ Green checkmark = all jobs passed
   - ❌ Red X = one or more jobs failed
   - 🟡 Yellow dot = currently running
3. Click a run to see individual job logs

### Downloading Artifacts

1. Click on a completed workflow run
2. Scroll to **Artifacts** section at the bottom
3. Download `test-results` to see HTML coverage report

### Pull Request Checks

When you create a PR:
- GitHub shows check status next to the PR
- Lint and test results appear as status checks
- PR can require passing checks before merge (configure in repo Settings → Branches)

---

## 13. Customizing the Workflow

### Add a New Job

```yaml
  # Add after the test job
  my-new-job:
    name: My Custom Step
    runs-on: ubuntu-latest
    needs: test            # Run after tests pass
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: |
          pip install -r config/requirements.txt
          python my_script.py
```

### Add Pipeline Run to CI

To run the full ML pipeline in CI:

```yaml
  pipeline-test:
    name: Pipeline Smoke Test
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r config/requirements.txt
      - run: |
          python run_pipeline.py \
            --input-csv data/processed/sensor_fused_50Hz.csv \
            --skip-ingestion \
            --continue-on-failure
```

### Restrict to Specific Branches

```yaml
on:
  push:
    branches: [main]        # Only main (remove develop)
```

---

## 14. Common Issues & Fixes

### "pip install failed"

The workflow uses `config/requirements.txt`. If this file has merge conflicts:

```bash
# Fix locally
git checkout --theirs config/requirements.txt
# OR use pyproject.toml instead — update the workflow:
#   pip install -r config/requirements.txt  →  pip install -e .
```

### "Tests failed but work locally"

- CI uses `ubuntu-latest` (Linux), you develop on Windows
- Path separators differ: `\` vs `/`
- Check if tests use hardcoded Windows paths
- Review the test logs in the Actions tab

### "Docker build failed"

- Check `docker/Dockerfile.inference` exists and is valid
- Ensure all required files are not in `.gitignore`
- Check the build logs for missing dependencies

### "Permission denied" on ghcr.io

1. Go to repo **Settings** → **Actions** → **General**
2. Under "Workflow permissions", select **"Read and write permissions"**

### Workflow not triggering

- Check the `paths:` filter — only `src/`, `tests/`, `docker/`, `config/` changes trigger
- Pushing to branches other than `main`/`develop` won't trigger
- Check the `.github/workflows/ci-cd.yml` file is committed and pushed

---

## Workflow File Location

```
.github/
  workflows/
    ci-cd.yml          ← This is your CI/CD workflow (282 lines)
```

The full workflow is in `.github/workflows/ci-cd.yml`. Edit this file to customize the pipeline behavior.

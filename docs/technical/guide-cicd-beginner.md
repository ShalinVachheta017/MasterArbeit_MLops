# 🚀 GitHub Actions CI/CD: Complete Beginner's Guide

> **Learn how to build CI/CD from scratch for your MLOps project**  
> This guide explains GitHub Actions step-by-step using your HAR MLOps project as an example.

---

## 📚 Table of Contents

1. [What is CI/CD?](#what-is-cicd)
2. [GitHub Actions Basics](#github-actions-basics)
3. [Understanding the Workflow Structure](#understanding-the-workflow-structure)
4. [Building Your First CI/CD Pipeline](#building-your-first-cicd-pipeline)
5. [Complete Workflow Example](#complete-workflow-example)
6. [Step-by-Step Implementation](#step-by-step-implementation)
7. [Testing Without Cloud Services](#testing-without-cloud-services)
8. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## What is CI/CD?

**CI/CD** = **C**ontinuous **I**ntegration + **C**ontinuous **D**elivery/Deployment

### Why do we need it?

Imagine you're working on your HAR MLOps project:
- You write code for data preprocessing → **Does it break existing tests?**
- You update model training logic → **Does it still work with old data?**
- You modify Docker configuration → **Will the container still build?**

**Without CI/CD**: You manually test everything each time. Time-consuming and error-prone.

**With CI/CD**: Automatically tests, builds, and validates your code on every commit.

### The CI/CD Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR DEVELOPMENT CYCLE                      │
└─────────────────────────────────────────────────────────────────┘

1. Write Code           2. Commit & Push        3. Automated Tests
   (on laptop)             (to GitHub)             (in the cloud)
   
   📝 Edit files         ➜  git add .           ➜  ✅ Lint code
   🧪 Local test         ➜  git commit          ➜  ✅ Run tests
   🐳 Update Docker      ➜  git push            ➜  ✅ Build Docker
                                                  ➜  ✅ Validate model
                                                  
4. Get Feedback         5. Fix if Needed        6. Deploy (if passing)
   (GitHub shows           (make changes)          (automatically)
    results)
   
   ✅ All green?        ➜  🔧 Fix issues       ➜  🚀 Ready to use!
   ❌ Failed? Check     ➜  💾 Commit again     
      logs
```

---

## GitHub Actions Basics

### Core Concepts Explained

| Concept | What It Is | Example from Your Project |
|---------|-----------|---------------------------|
| **Workflow** | A complete automation process | Your entire CI/CD pipeline (lint → test → build → deploy) |
| **Trigger** | What starts the workflow | Push to `main` branch or opening a Pull Request |
| **Job** | A group of related tasks | "Run all unit tests" is one job |
| **Step** | A single command or action | "Install Python dependencies" is one step |
| **Runner** | The virtual machine that executes jobs | `ubuntu-latest`, `windows-latest`, `macos-latest` |
| **Action** | Reusable pre-built steps | `actions/checkout@v4` (downloads your code) |
| **Artifact** | Files produced during workflow | Test coverage reports, Docker images |
| **Secret** | Encrypted credentials | API keys, Docker Hub passwords |

### Where Does GitHub Actions Run?

**NOT on your computer!** GitHub provides free virtual machines (runners) in the cloud:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB'S CLOUD SERVERS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🖥️ Runner 1 (ubuntu-latest)                                   │
│     └─ Your workflow runs here (fresh VM each time)           │
│     └─ Has Python, Docker, Node.js, etc. pre-installed        │
│                                                                 │
│  ⏱️ FREE Tier Limits:                                          │
│     • Public repos: Unlimited minutes                          │
│     • Private repos: 2,000 minutes/month                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Understanding the Workflow Structure

### Anatomy of a `.yml` File

Every GitHub Actions workflow is a **YAML file** stored in `.github/workflows/`.

```
your-project/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml          ← Your main CI/CD pipeline
│       ├── deploy.yml         ← Deployment workflow (optional)
│       └── scheduled-tests.yml ← Nightly tests (optional)
├── src/
├── tests/
└── ...
```

### YAML Structure (Think of it like Python indentation)

```yaml
# PSEUDO-CODE STRUCTURE
name: "Workflow Name"          # What shows up in GitHub UI

on:                            # WHEN to run
  push:                        # On code push
    branches: [main]           # Only on main branch
  pull_request:                # On Pull Requests
  workflow_dispatch:           # Manual button

env:                           # GLOBAL variables
  PYTHON_VERSION: '3.11'

jobs:                          # WHAT to do
  job-name:                    # Job 1
    runs-on: ubuntu-latest     # WHERE to run
    steps:                     # HOW to do it
      - name: Step 1           # Checkout code
        uses: actions/checkout@v4
      - name: Step 2           # Run tests
        run: pytest tests/
```

---

## Building Your First CI/CD Pipeline

Let's build a simple workflow step-by-step for your HAR MLOps project.

### Step 1: Create the Workflow File

**File Location**: `.github/workflows/ci-cd.yml`

```yaml
# Minimal starter workflow
name: My First CI/CD Pipeline

on:
  push:
    branches: [main]

jobs:
  hello:
    runs-on: ubuntu-latest
    steps:
      - name: Say hello
        run: echo "Hello from GitHub Actions!"
```

**What happens when you commit this?**

1. GitHub detects the new file in `.github/workflows/`
2. On next push to `main`, GitHub:
   - Spins up an Ubuntu VM
   - Runs `echo "Hello from GitHub Actions!"`
   - Shows results in the "Actions" tab

---

### Step 2: Add Code Checkout

**Why?** The runner needs your project files to test them.

```yaml
name: CI Pipeline with Code Access

on:
  push:
    branches: [main]

jobs:
  test-project:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        # ↑ This downloads your code to the runner
      
      - name: List project files
        run: ls -la
        # ↑ You'll see: src/, tests/, config/, etc.
```

**`uses: actions/checkout@v4`** is a pre-built action that:
- Clones your repo
- Checks out the commit that triggered the workflow
- Makes files available to subsequent steps

---

### Step 3: Set Up Python Environment

Your project needs Python 3.11 with specific dependencies.

```yaml
name: CI with Python Setup

on:
  push:
    branches: [main]

jobs:
  test-python:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'  # ← Speeds up by caching dependencies
      
      - name: Verify Python version
        run: python --version
        # Output: Python 3.11.x
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r config/requirements.txt
      
      - name: Show installed packages
        run: pip list
```

**What's happening here?**

| Step | Action | Why |
|------|--------|-----|
| 1 | `uses: actions/setup-python@v5` | Installs Python 3.11 |
| 2 | `cache: 'pip'` | Saves time by reusing downloaded packages |
| 3 | `pip install -r config/requirements.txt` | Installs TensorFlow, pandas, MLflow, etc. |

---

### Step 4: Run Tests

Now let's actually test your code!

```yaml
name: CI with Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r config/requirements.txt
          pip install pytest pytest-cov  # Testing tools
      
      - name: Run unit tests
        run: pytest tests/unit -v
        # -v = verbose output (shows each test)
      
      - name: Run integration tests
        run: pytest tests/integration -v
      
      - name: Generate coverage report
        run: pytest tests/ --cov=src --cov-report=term-missing
        # Shows which lines of code are tested
```

**Understanding Test Outputs:**

```bash
tests/unit/test_preprocessing.py::test_load_data PASSED      [ 10%]
tests/unit/test_preprocessing.py::test_clean_data PASSED     [ 20%]
tests/unit/test_model.py::test_model_prediction FAILED       [ 30%]
                                                              ^^^^^^
                                                              This would FAIL the workflow!
```

---

### Step 5: Code Quality Checks (Linting)

Ensure code follows best practices before running tests.

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install linters
        run: pip install flake8 black isort
      
      - name: Check for syntax errors
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source
          # E9 = syntax errors
          # F63 = invalid syntax
      
      - name: Check code formatting
        run: black --check src/
        # Fails if code isn't formatted properly
      
      - name: Check import sorting
        run: isort --check-only src/
        # Ensures imports are organized
```

**Why lint before tests?**

- **No syntax errors** → Catches typos before wasting time on tests
- **Consistent style** → Easier to read and maintain
- **Fast feedback** → Linting takes seconds, tests take minutes

---

### Step 6: Build Docker Image

Your project has `docker/Dockerfile.inference` for model serving.

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    needs: test  # ← Only run if tests pass!
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        # Enables advanced Docker features
      
      - name: Build Docker image
        run: |
          docker build \
            -f docker/Dockerfile.inference \
            -t har-inference:latest \
            .
      
      - name: Test Docker container
        run: |
          docker run --rm har-inference:latest python --version
          # Quick smoke test: Does the container work?
      
      - name: Save image as artifact
        run: |
          docker save har-inference:latest -o har-image.tar
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: docker-image
          path: har-image.tar
```

**What's `needs: test`?**

Creates a **dependency chain**:

```
lint (runs first)
  ↓
test (runs after lint passes)
  ↓
build (runs after test passes)
```

If any job fails, subsequent jobs are skipped.

---

### Step 7: Job Dependencies & Parallel Execution

You can run multiple jobs simultaneously or in sequence.

```yaml
jobs:
  # These run in PARALLEL (faster!)
  lint:
    runs-on: ubuntu-latest
    steps: [...]
  
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run security checks
        run: pip install safety && safety check

  # This waits for BOTH to finish
  test:
    needs: [lint, security-scan]
    runs-on: ubuntu-latest
    steps: [...]
  
  # This waits for test
  build:
    needs: test
    runs-on: ubuntu-latest
    steps: [...]
```

**Execution Flow:**

```
        ┌─────────────┐
        │    PUSH     │
        └──────┬──────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌─────▼──────┐
│  LINT  │          │  SECURITY  │  ← Run in parallel
└───┬────┘          └─────┬──────┘
    │                     │
    └──────────┬──────────┘
               │
         ┌─────▼─────┐
         │   TEST    │  ← Wait for both
         └─────┬─────┘
               │
         ┌─────▼─────┐
         │   BUILD   │  ← Wait for test
         └───────────┘
```

---

## Complete Workflow Example

Here's a **complete production-ready workflow** for your HAR MLOps project:

```yaml
name: HAR MLOps CI/CD Pipeline

# ========================================
# WHEN TO RUN (Triggers)
# ========================================
on:
  push:
    branches: [main, develop]
    paths:  # Only run if these files change
      - 'src/**'
      - 'tests/**'
      - 'config/**'
      - 'docker/**'
      - '.github/workflows/**'
  
  pull_request:
    branches: [main]
  
  workflow_dispatch:  # Manual trigger button in GitHub UI

# ========================================
# GLOBAL CONFIGURATION
# ========================================
env:
  PYTHON_VERSION: '3.11'
  DOCKER_REGISTRY: ghcr.io  # GitHub Container Registry
  IMAGE_NAME: ${{ github.repository }}/har-inference

# ========================================
# JOBS
# ========================================
jobs:
  
  # ────────────────────────────────────
  # JOB 1: CODE QUALITY
  # ────────────────────────────────────
  lint:
    name: "🔍 Code Quality Check"
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install linting tools
        run: |
          python -m pip install --upgrade pip
          pip install flake8 black isort mypy
      
      - name: Run flake8 (syntax errors)
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
      
      - name: Check formatting (black)
        run: black --check --diff src/
        continue-on-error: true  # Don't fail workflow, just warn
      
      - name: Check import order (isort)
        run: isort --check-only --diff src/
        continue-on-error: true

  # ────────────────────────────────────
  # JOB 2: UNIT TESTS
  # ────────────────────────────────────
  test:
    name: "🧪 Run Tests"
    runs-on: ubuntu-latest
    needs: lint  # Wait for lint to pass
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'  # Cache dependencies
      
      - name: Install dependencies
        run: |
          pip install -r config/requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Run tests with coverage
        run: |
          pytest tests/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --cov-report=term-missing \
            --junitxml=test-results.xml
      
      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml
          fail_ci_if_error: false
      
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()  # Upload even if tests fail
        with:
          name: test-results
          path: |
            test-results.xml
            htmlcov/
          retention-days: 30

  # ────────────────────────────────────
  # JOB 3: BUILD DOCKER IMAGE
  # ────────────────────────────────────
  build:
    name: "🐳 Build Docker Image"
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name != 'pull_request'  # Skip on PRs
    
    permissions:
      contents: read
      packages: write  # Needed to push to GitHub Container Registry
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=raw,value=latest
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.inference
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ────────────────────────────────────
  # JOB 4: INTEGRATION TESTS
  # ────────────────────────────────────
  integration-test:
    name: "🔗 Integration Tests"
    runs-on: ubuntu-latest
    needs: build
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Pull Docker image
        run: |
          docker pull ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest
      
      - name: Run smoke test
        run: |
          docker run --rm \
            -v $(pwd)/data/samples_2005\ dataset:/data \
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest \
            python -c "import tensorflow; print('TensorFlow OK')"

  # ────────────────────────────────────
  # JOB 5: MODEL VALIDATION (Optional)
  # ────────────────────────────────────
  validate-model:
    name: "🤖 Validate ML Model"
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'  # Manual only
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: pip install -r config/requirements.txt
      
      - name: Run model validation
        run: |
          python -c "
          from src.models.train_model import validate_model
          metrics = validate_model('models/best_model.keras')
          assert metrics['accuracy'] > 0.85, 'Model accuracy too low!'
          print(f'Model validated: {metrics}')
          "

  # ────────────────────────────────────
  # JOB 6: NOTIFY ON FAILURE
  # ────────────────────────────────────
  notify:
    name: "📧 Notify on Failure"
    runs-on: ubuntu-latest
    needs: [lint, test, build, integration-test]
    if: failure()  # Only run if any previous job failed
    
    steps:
      - name: Send notification
        run: |
          echo "❌ Workflow failed! Check logs at:"
          echo "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
```

---

## Step-by-Step Implementation

Follow these steps to add CI/CD to your project:

### Step 1: Create Workflow Directory

```bash
# In your project root
mkdir -p .github/workflows
```

### Step 2: Create Workflow File

```bash
# Create the YAML file
touch .github/workflows/ci-cd.yml
```

### Step 3: Add Basic Workflow

Copy the **complete workflow example** above into `ci-cd.yml`.

### Step 4: Commit and Push

```bash
git add .github/workflows/ci-cd.yml
git commit -m "Add CI/CD pipeline"
git push origin main
```

### Step 5: View Results

1. Go to **GitHub.com → Your Repository**
2. Click **"Actions"** tab at the top
3. You'll see your workflow running!

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions Tab                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ▶ HAR MLOps CI/CD Pipeline                                │
│     └─ Add CI/CD pipeline (#123)                           │
│        ⏱️ Running... 2m 34s                                 │
│                                                             │
│        ✅ 🔍 Code Quality Check      (12s)                  │
│        ⏳ 🧪 Run Tests               (running...)           │
│        ⏸️ 🐳 Build Docker Image     (waiting...)           │
│        ⏸️ 🔗 Integration Tests      (waiting...)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 6: Configure Secrets (If Needed)

For pushing Docker images or deploying:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add:
   - `DOCKER_HUB_USERNAME` (your Docker Hub username)
   - `DOCKER_HUB_TOKEN` (access token from Docker Hub)

### Step 7: Protect Main Branch

1. **Settings** → **Branches** → **Add branch protection rule**
2. Pattern: `main`
3. Enable:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date
   - Select: `lint`, `test`, `build`

Now **no code can be merged** unless all checks pass! ✅

---

## Testing Without Cloud Services

Your project doesn't use AWS/Azure. Here's how to test effectively:

### Option 1: GitHub Container Registry (Free)

GitHub provides **free Docker image storage**:

```yaml
env:
  DOCKER_REGISTRY: ghcr.io  # GitHub Container Registry
  IMAGE_NAME: ${{ github.repository }}/har-inference

jobs:
  build:
    steps:
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided
      
      - name: Build and push
        run: |
          docker build -t ghcr.io/${{ github.repository }}/har-inference:latest .
          docker push ghcr.io/${{ github.repository }}/har-inference:latest
```

**Pull the image locally:**

```bash
docker pull ghcr.io/yourusername/masterarbeit_mlops/har-inference:latest
docker run -it ghcr.io/yourusername/masterarbeit_mlops/har-inference:latest
```

### Option 2: Local Testing with Act

Test workflows **on your machine** before pushing:

```bash
# Install Act (runs GitHub Actions locally)
choco install act  # Windows
brew install act   # Mac
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux

# Run workflow locally
act push
```

### Option 3: Artifact Storage

Store test results and models as **artifacts**:

```yaml
- name: Save trained model
  uses: actions/upload-artifact@v4
  with:
    name: trained-model
    path: models/best_model.keras

- name: Download in next job
  uses: actions/download-artifact@v4
  with:
    name: trained-model
```

Download artifacts from GitHub:

```
Actions → Workflow Run → Artifacts section → Download
```

### Option 4: Self-Hosted Runners

Run workflows **on your own machine** (advanced):

```bash
# In GitHub: Settings → Actions → Runners → New self-hosted runner
# Follow instructions to install runner on your PC

# Then in workflow:
jobs:
  train:
    runs-on: self-hosted  # Uses YOUR computer!
    steps:
      - name: Train model
        run: python src/models/train_model.py
```

**Pros:**
- Use your GPU for model training
- Access local data
- No time limits

**Cons:**
- Your computer must be on
- Security concerns (only use on private repos)

---

## Troubleshooting & Best Practices

### Common Issues

#### 1. **Workflow doesn't trigger**

**Problem:** You pushed code but nothing happens in Actions tab.

**Solutions:**
```yaml
# Check your triggers
on:
  push:
    branches: [main]  # ← Are you pushing to 'main'?
    paths:
      - 'src/**'      # ← Did you change files in src/?
```

```bash
# Verify you're on the right branch
git branch --show-current

# Check workflow file syntax
# (GitHub Actions tab will show YAML errors)
```

#### 2. **"Module not found" errors**

**Problem:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```yaml
# Make sure you install dependencies
- name: Install dependencies
  run: pip install -r config/requirements.txt
```

#### 3. **Tests pass locally but fail in CI**

**Possible causes:**

| Issue | Local | CI | Fix |
|-------|-------|-----|-----|
| **Missing files** | Has `data/raw/` | Empty runner | Use artifacts or download data |
| **Environment variables** | Set in `.env` | Not set | Add to workflow `env:` |
| **Different OS** | Windows | Ubuntu | Test on Ubuntu locally |
| **Cached data** | Old cache | Fresh run | Clear local cache |

**Solution:**
```yaml
- name: Download test data
  run: |
    mkdir -p data/raw
    # Download or use test fixtures
    cp tests/fixtures/sample_data.csv data/raw/
```

#### 4. **Docker build fails**

**Problem:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solution:**
```yaml
- name: Build Docker with verbose output
  run: |
    docker build \
      --progress=plain \  # See full output
      --no-cache \        # Force rebuild
      -f docker/Dockerfile.inference \
      -t har-inference:latest \
      .
```

#### 5. **Workflow takes too long**

**Current:**
```
Lint:  12s
Test:  5m 34s  ← Too slow!
Build: 8m 12s
Total: 13m 58s
```

**Optimization:**

```yaml
# 1. Cache dependencies
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'  # ← Saves 1-2 minutes

# 2. Parallel testing
- name: Run tests in parallel
  run: pytest tests/ -n auto  # Uses multiple CPU cores

# 3. Cache Docker layers
- name: Build Docker
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

**After optimization:**
```
Lint:  8s      (↓4s)
Test:  2m 10s  (↓3m 24s)
Build: 3m 45s  (↓4m 27s)
Total: 6m 3s   (↓7m 55s - 56% faster!)
```

### Best Practices

#### ✅ DO:

1. **Run lint before tests** (fast feedback)
   ```yaml
   jobs:
     lint:
       # ...
     test:
       needs: lint  # ← Don't waste time if lint fails
   ```

2. **Use caching** (faster builds)
   ```yaml
   - uses: actions/setup-python@v5
     with:
       cache: 'pip'
   
   - uses: actions/cache@v4
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
   ```

3. **Pin versions** (reproducible builds)
   ```yaml
   uses: actions/checkout@v4  # ← Specific version, not @latest
   ```

4. **Store secrets securely**
   ```yaml
   # ❌ DON'T
   env:
     API_KEY: "abc123"
   
   # ✅ DO
   env:
     API_KEY: ${{ secrets.API_KEY }}
   ```

5. **Run tests on multiple OS** (for Python packages)
   ```yaml
   jobs:
     test:
       strategy:
         matrix:
           os: [ubuntu-latest, windows-latest, macos-latest]
           python: ['3.9', '3.10', '3.11']
       runs-on: ${{ matrix.os }}
   ```

6. **Set timeouts** (prevent hung jobs)
   ```yaml
   jobs:
     test:
       timeout-minutes: 30  # Kill after 30 min
   ```

#### ❌ DON'T:

1. **Don't commit secrets**
   ```bash
   # Use GitHub Secrets instead!
   git add .env  # ❌ NO!
   ```

2. **Don't skip tests**
   ```yaml
   # Bad practice
   pytest tests/ || true  # ❌ Ignores failures
   ```

3. **Don't use `latest` tags**
   ```yaml
   # Can break unexpectedly
   uses: actions/checkout@latest  # ❌ Unstable
   ```

4. **Don't duplicate code**
   ```yaml
   # Use reusable workflows or composite actions
   # See: https://docs.github.com/en/actions/using-workflows/reusing-workflows
   ```

---

## Advanced Topics

### Reusable Workflows

Create **`.github/workflows/reusable-test.yml`**:

```yaml
name: Reusable Test Workflow

on:
  workflow_call:
    inputs:
      python-version:
        required: true
        type: string

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
      - run: pip install -r config/requirements.txt
      - run: pytest tests/
```

**Use it in other workflows:**

```yaml
jobs:
  python-3-11:
    uses: ./.github/workflows/reusable-test.yml
    with:
      python-version: '3.11'
  
  python-3-12:
    uses: ./.github/workflows/reusable-test.yml
    with:
      python-version: '3.12'
```

### Matrix Strategy (Test Multiple Configurations)

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ['3.9', '3.11']
        tensorflow: ['2.14', '2.15']
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install tensorflow==${{ matrix.tensorflow }}
      - run: pytest tests/
```

**Result:** Runs **8 parallel jobs** (2 OS × 2 Python × 2 TensorFlow = 8 combos)

### Scheduled Workflows (Nightly Builds)

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Every day at 2 AM UTC

jobs:
  nightly-test:
    runs-on: ubuntu-latest
    steps:
      - name: Run comprehensive tests
        run: pytest tests/ --slow
```

**Cron syntax:**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
│ │ │ │ │
* * * * *
```

Examples:
- `0 0 * * *` - Daily at midnight
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1` - Every Monday at 9 AM

---

## Summary: From Zero to CI/CD Hero

### What You've Learned

1. **What CI/CD is** and why it's essential for MLOps
2. **GitHub Actions architecture** (workflows, jobs, steps, runners)
3. **YAML syntax** for defining workflows
4. **Building a complete pipeline** (lint → test → build → deploy)
5. **Testing without cloud** using GitHub Container Registry
6. **Troubleshooting** common issues
7. **Best practices** for production-ready CI/CD

### Your HAR MLOps CI/CD Flow

```
Developer Workflow:
┌─────────────────────────────────────────────────────────────┐
│  1. Code locally           git add . && git commit          │
│  2. Push to GitHub         git push origin main             │
│  3. GitHub Actions runs:                                    │
│     ├─ 🔍 Lint code        (flake8, black, isort)          │
│     ├─ 🧪 Run tests        (pytest with coverage)          │
│     ├─ 🐳 Build Docker     (create inference image)        │
│     └─ ✅ Validate         (smoke tests)                   │
│  4. View results           Actions tab shows ✅ or ❌       │
│  5. Merge if passing       Or fix issues and retry         │
└─────────────────────────────────────────────────────────────┘
```

### Next Steps

1. **Start simple**: Add a basic workflow (just lint/test)
2. **Iterate**: Gradually add Docker build, integration tests
3. **Monitor**: Check execution times, optimize caching
4. **Expand**: Add model validation, performance benchmarks
5. **Automate more**: Deploy to production when ready

### Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Workflow Syntax**: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
- **Actions Marketplace**: https://github.com/marketplace?type=actions
- **Your workflow**: `.github/workflows/ci-cd.yml`

---

## Quick Reference

### Essential Commands

```bash
# Create workflow
mkdir -p .github/workflows
touch .github/workflows/ci-cd.yml

# Test locally (with Act)
act push

# Trigger manual workflow
gh workflow run ci-cd.yml  # Requires GitHub CLI

# View workflow runs
gh run list
gh run view <run-id> --log
```

### Workflow Template Checklist

```yaml
✅ name: "Workflow Name"
✅ on: [push, pull_request, workflow_dispatch]
✅ env: [Global variables]
✅ jobs:
   ✅ lint: [Code quality]
   ✅ test: [Unit tests with coverage]
   ✅ build: [Docker image]
   ✅ integration-test: [E2E tests]
   ✅ notify: [Failure alerts]
```

### GitHub Actions Status Badge

Add this to your `README.md`:

```markdown
![CI/CD](https://github.com/yourusername/masterarbeit_mlops/actions/workflows/ci-cd.yml/badge.svg)
```

Shows build status: ![CI/CD](https://img.shields.io/badge/build-passing-brightgreen)

---

**Congratulations! 🎉 You now understand GitHub Actions CI/CD from scratch!**

Start with the basic workflow, test it, and gradually expand. Every software project should have automated testing—you're now equipped to build professional-grade pipelines!

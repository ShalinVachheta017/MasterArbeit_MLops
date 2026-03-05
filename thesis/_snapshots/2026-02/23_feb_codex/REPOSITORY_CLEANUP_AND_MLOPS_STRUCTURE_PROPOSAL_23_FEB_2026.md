# Repository Cleanup and MLOps Structure Proposal (Production + Thesis) - 23 Feb 2026

Date: 2026-02-23  
Scope: Repo cleanup + target structure proposal tailored to current project state  
Method: Local repo inventory (top-level sizes, naming, tracked/generated separation) + external MLOps structure references (Azure/Microsoft, DVC, Cookiecutter Data Science, Google Rules of ML)

---

## 1. Executive Recommendation

Do **not** do a big-bang rewrite of the repository structure right now.

Use a **2-layer strategy**:

1. **Stabilize**: standardize naming, move obvious clutter, separate current-vs-archive docs, fix tracked binary/model debt
2. **Converge**: move to a production-ready structure with clear boundaries:
   - source code
   - configs
   - pipelines/scripts
   - data/model pointers (DVC)
   - generated runtime artifacts (ignored)
   - docs/runbooks/thesis
   - archival research/history

This gives you a better thesis repo immediately without breaking pipeline paths unnecessarily.

---

## 2. What the Current Repo Looks Like (practical audit)

### 2.1 Strong points (keep these)

- Good separation of `src/`, `tests/`, `scripts/`, `config/`, `docker/`, `docs/`
- DVC is already present (`.dvc`, `.dvc_storage`, `*.dvc`)
- Rich docs/thesis/research material (valuable for defense and writing)
- Multiple output channels exist (`outputs/`, `artifacts/`, `logs/`, `mlruns/`)

### 2.2 Main cleanup problems (high impact)

#### A. Root directory is overloaded

Examples of root clutter:

- ad hoc scripts (`batch_process_all_datasets.py`, `generate_summary_report.py`)
- standalone CSV research tables
- date-stamped codex folders (`22 feb codex`)
- planning folder with spaces (`things to do`)

Impact:

- harder navigation
- harder automation
- unclear source-of-truth documents

#### B. Generated runtime content is fragmented

You currently split generated artifacts across:

- `outputs/`
- `artifacts/`
- `logs/`
- `mlruns/`
- `reports/`
- `images/`

This is workable, but for production/reproducibility it becomes hard to answer:

- Which files are runtime outputs vs thesis figures vs final reports?
- Which folder should a new batch benchmark write into?

#### C. Naming consistency is weak

Examples:

- `22 feb codex` (spaces + mixed date style)
- `things to do` (spaces)
- `22Feb_Opus_Understanding` (camel+date)
- uppercase + lowercase + snake_case + kebab-case mixed across docs

Impact:

- link breakage risk
- script quoting issues (especially Windows + CI)
- cognitive overhead

#### D. Tracked binary model artifacts still exist in Git

`git ls-files` shows tracked `.keras` files under:

- `models/retrained/`
- `models/registry/`
- `models/archived_experiments/...`

Impact:

- repo bloat
- slower clone/fetch
- harder provenance/versioning compared to DVC/LFS

#### E. Historical documents are mixed with canonical docs

You have both:

- high-value current docs
- dated snapshot audits / handoff notes / superseded plans

in areas that look equally “current”.

Impact:

- duplicate truths
- stale guidance risk during thesis writing

---

## 3. Local Footprint Snapshot (why cleanup matters)

Large local directories (approximate current workspace sizes from scan):

- `data/` -> **~8.0 GB**
- `artifacts/` -> **~5.7 GB**
- `archive/` -> **~1.2 GB**
- `mlruns/` -> **~240 MB**
- `outputs/` -> **~49 MB**
- `logs/` -> **~47 MB**
- `models/` -> **~47 MB**
- `.mypy_cache/` -> **~175 MB** (safe local cache cleanup)

This is normal for ML work, but it makes structure discipline more important.

---

## 4. Production-Oriented Target Structure (tailored for your repo)

This is the structure I recommend as the **target state** (you can migrate in phases).

```text
MasterArbeit_MLops/
├── src/
│   └── har_mlops/
│       ├── api/
│       ├── components/
│       ├── pipelines/
│       ├── monitoring/
│       ├── domain_adaptation/
│       ├── utils/
│       └── ...
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── fixtures/
├── scripts/
│   ├── ops/
│   ├── batch/
│   ├── data/
│   ├── reporting/
│   └── dev/
├── config/
│   ├── pipeline/
│   ├── monitoring/
│   ├── mlflow/
│   ├── alerts/
│   └── environments/
├── data/
│   ├── README.md
│   ├── raw.dvc
│   ├── processed.dvc
│   ├── prepared.dvc
│   └── samples/                 # small tracked examples only
├── models/
│   ├── README.md
│   ├── pretrained.dvc
│   ├── registry/                # metadata only (JSON)
│   └── baselines/               # baseline metadata only (JSON)
├── runs/                        # gitignored: unify runtime outputs
│   ├── batch_analysis/
│   ├── evaluation/
│   ├── monitoring/
│   ├── calibration/
│   ├── sensor_placement/
│   ├── artifacts/
│   ├── logs/
│   └── mlflow/                  # or keep mlruns/ if easier
├── docs/
│   ├── thesis/
│   │   ├── chapters/
│   │   ├── figures/
│   │   └── tables/
│   ├── technical/
│   ├── runbooks/
│   ├── audits/
│   │   ├── 2026-02-22/
│   │   └── 2026-02-23/
│   ├── planning/
│   └── archive/
├── research/                    # optional: move reusable research indexes here
├── infra/
│   ├── docker/
│   ├── compose/
│   ├── monitoring/              # Prometheus/Grafana configs
│   └── ci/                      # docs/templates (actual GH Actions stays in .github/)
├── notebooks/
│   ├── exploratory/
│   ├── thesis_figures/
│   └── archive/
├── .github/
│   └── workflows/
├── pyproject.toml
├── README.md
└── Makefile / justfile (optional but recommended)
```

### Why this structure fits your project

- Keeps thesis docs and production pipeline work in one repo, but separates them cleanly
- Supports both offline batch evaluation and future online/near-real-time deployment
- Reduces root clutter
- Makes generated runtime content clearly disposable (`runs/`, gitignored)

---

## 5. What to Change First (Low-Risk, High-Return)

These are the best cleanup actions to do now without breaking your pipeline.

### Phase 1: Naming and organization (safe)

1. Create canonical docs buckets:
   - `docs/audits/`
   - `docs/planning/`
   - `docs/runbooks/`
2. Move date-stamped codex reports into dated audit folders
   - e.g. `22 feb codex/` -> `docs/audits/2026-02-22/`
   - new `23-5 codex/` content can later move to `docs/audits/2026-02-23/`
3. Rename `things to do/` -> `docs/planning/` (or `notes/planning/`)
4. Standardize new folder/file names to `lowercase-kebab-case` or `snake_case` (pick one and stick to it)

### Phase 2: Root cleanup (safe)

Move root-level utility files into focused locations:

- `batch_process_all_datasets.py` -> `scripts/batch/`
- `generate_summary_report.py` -> `scripts/reporting/`
- research summary CSVs -> `research/` or `docs/research/data/`

Also remove or archive empty/noise files:

- `monitoringans.txt` (empty)

### Phase 3: Runtime artifact consolidation (medium risk, high value)

Choose one canonical runtime root (recommended: `runs/`) and gradually migrate writes from:

- `outputs/`
- `artifacts/`
- `logs/`
- `reports/`
- `images/`

Migration approach (safe):

- keep old directories for compatibility
- add config aliases / env vars
- switch one producer at a time
- verify before deleting old path assumptions

### Phase 4: Model artifact hygiene (high impact)

Stop keeping large `.keras` binaries in normal Git history (where possible going forward).

Recommended options:

- DVC for model binaries + metadata in Git (best fit since DVC already exists)
- Git LFS if you prefer simpler model storage than DVC

Keep in Git:

- registry metadata (`model_registry.json`)
- label mapping
- scaler config
- training/eval reports

Move to DVC/LFS:

- `.keras`, `.h5`, large `.npy`, heavy experiment checkpoints

---

## 6. Output/Artifact Structure Recommendation (very important for your thesis)

Your thesis now depends heavily on generated evidence.
So the repo should distinguish **runtime evidence** from **thesis-ready assets**.

### Recommended split

#### Runtime evidence (gitignored)

- `runs/` (or keep `outputs/` + `artifacts/` short-term)
- Contains raw generated outputs from pipeline runs
- May be large, repetitive, and temporary

#### Thesis-ready exports (tracked)

- `docs/thesis/figures/`
- `docs/thesis/tables/`
- `reports/thesis/` (optional)

Rule:

- Generate many raw artifacts in `runs/`
- Curate only selected final figures/tables into thesis-tracked folders

This keeps the repo clean and makes your thesis reproducibility story stronger.

---

## 7. Suggested Canonical Conventions (so the repo stops drifting)

### 7.1 Naming conventions

Pick and document one standard:

- Folders: `lowercase-kebab-case`
- Python modules: `snake_case.py`
- Reports: `YYYY-MM-DD_topic.md`
- Run artifacts: timestamped (`YYYYMMDD_HHMMSS`) is fine

Examples:

- `docs/audits/2026-02-23/thesis-progress-outputs-audit.md`
- `docs/planning/remaining-work.md`
- `runs/batch-analysis/20260223_011207/`

### 7.2 “Source of truth” policy

For each category, designate one canonical file and mark others as archive/snapshot.

Examples:

- current progress -> one file in `docs/planning/`
- historical snapshots -> `docs/audits/YYYY-MM-DD/`
- current runbook -> one file in `docs/runbooks/`

### 7.3 Artifact metadata standard

Add a small metadata schema for batch runs:

- `run_id`
- `timestamp`
- `git_commit`
- `config_hash`
- `model_version`
- `dataset_count`
- `artifacts` (paths)
- `status`

This makes thesis tables and reproducibility claims much easier.

---

## 8. Minimal Migration Plan (1-2 days, no big breakage)

### Day 1 (safe organization)

1. Create `docs/audits/`, `docs/planning/`, `scripts/batch/`, `scripts/reporting/`
2. Move non-critical root files/folders into correct buckets
3. Update README links
4. Add naming convention note in `docs/README.md`

### Day 2 (artifact discipline)

1. Define canonical runtime output root in config (even if aliasing old paths)
2. Start writing new batch/evaluation outputs into the canonical path
3. Export one curated thesis figure/table set into `docs/thesis/figures` and `docs/thesis/tables`
4. Document retention policy (what to keep, what to delete)

---

## 9. Online Reference Basis (used for this recommendation)

I used these as design references (adapted to your repo, not copied blindly):

- **Cookiecutter Data Science**: emphasizes a standard project organization and reproducibility-friendly layout (`data`, `models`, `notebooks`, etc.)  
  https://cookiecutter-data-science.drivendata.org/
- **DVC Docs (Project Structure)**: clarifies separation between `dvc.yaml`, `params.yaml`, `data/`, `models/`, and reproducible pipelines  
  https://dvc.org/doc/user-guide/project-structure
- **Google Rules of ML**: emphasizes ML-specific engineering discipline (testing/infrastructure/iteration over one-off scripts)  
  https://developers.google.com/machine-learning/guides/rules-of-ml
- **Microsoft / Azure MLOps v2 guidance and template**: reinforces separation of environments, pipelines, components, deployment, and productionization workflow  
  https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2  
  https://github.com/Azure/mlops-v2

These align well with your thesis repo because you need both:

- research iteration speed, and
- production-style reproducibility evidence

---

## 10. Bottom Line (Pragmatic)

Your repo is already strong in content and implementation depth.
The cleanup task is mostly about **structure, naming, and artifact boundaries**, not missing capabilities.

If you apply only these three things, you’ll get most of the benefit:

1. Move dated/planning docs out of the root into `docs/audits` + `docs/planning`
2. Consolidate runtime outputs under one canonical gitignored root (gradually)
3. Stop tracking large model binaries directly in Git (use DVC/LFS)

That will make the repo more production-like and make your thesis evidence easier to defend.

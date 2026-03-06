# What to Do Next — HAR MLOps Thesis

**Updated:** 19 Feb 2026  
**Pipeline status:** ~97% complete (code done, thesis writing pending)  
**Current HEAD:** `e2bc784`

---

## Immediate (Next Session — Do First)

### 1. Re-run A4 to validate TENT confidence-drop rollback

The previous A4 showed Δconf = −0.079 which should now trigger rollback with the new gate (threshold −0.01). Run this to confirm the fix works:

```powershell
python run_pipeline.py --retrain --adapt adabn_tent --skip-ingestion 2>&1 | Tee-Object -FilePath "logs\a4_audit_v3.txt"
```

**Expected outcomes:**
- `tent_rollback = True` in MLflow `har-retraining` experiment
- Log: `"TENT rollback triggered (confidence dropped …)"`
- `tent_confidence_delta < −0.01`

---

## Short Term (This Week — 19–23 Feb)

### 2. Cross-dataset batch run

Run the full pipeline across all 26 Garmin sessions to get per-user statistics:

```powershell
python batch_process_all_datasets.py
```

Produces per-session inference CSVs → input for drift analysis and thesis Chapter 5.

### 3. Drift analysis across datasets

```powershell
python scripts/analyze_drift_across_datasets.py
```

Produces drift scores per session → shows which users need adaptation.

### 4. Generate thesis figures

```powershell
python scripts/generate_thesis_figures.py
```

Needed figures:
- Pipeline architecture diagram
- Ablation table plot (A1 vs A3 vs A4 vs A5)
- Confusion matrix (A3 best model)
- Drift score distribution across sessions
- TENT before/after confidence plot

### 5. A2 Audit Run (optional but useful for thesis)

Baseline *with* only AdaBN (no TENT) for a cleaner ablation:

```powershell
python run_pipeline.py --retrain --adapt adabn --skip-ingestion 2>&1 | Tee-Object -FilePath "logs\a2_audit_output.txt"
```

---

## Medium Term (23 Feb – 7 Mar) — Thesis Writing

### Chapter 3 — System Design & Architecture

Key points to cover:
- 14-stage pipeline design rationale
- Why 1D-CNN-BiLSTM (vs. alternatives)
- Test-time adaptation strategy (AdaBN → TENT)
- Governance design (baseline promotion, rollback gates)
- MLflow + DVC integration

**Start with:** Architecture diagram + stage-by-stage description using `docs/19_Feb/PIPELINE_RUNBOOK.md` as source material.

### Chapter 4 — Experiments & Results

Use the 4 audit run results:

| Section | Source data |
|---------|-------------|
| Inference baseline (A1) | `artifacts/20260219_115223/` |
| Supervised retraining (A3) | MLflow `har-retraining`, run A3 |
| AdaBN+TENT adaptation (A4) | MLflow `har-retraining`, run A4 |
| Pseudo-label self-training (A5) | MLflow `har-retraining`, run A5 |

Export all MLflow runs to CSV for writing:
```powershell
python scripts/export_mlflow_runs.py --experiment har-retraining
```

### Chapter 5 — Discussion

- Why TENT entropy rollback alone is insufficient (confidence drop example)
- Pseudo-label class imbalance and the entropy gate's effect
- Drift detection sensitivity (PSI 0.203 in A1 — borderline warning)
- Limitations: Windows-native GPU, offline pseudo-labels only

---

## Long Term (Mar – May 2026)

| Month | Task |
|-------|------|
| March | Chapters 3+4 first draft |
| April | Chapter 5 (discussion) + Chapter 2 (related work) |
| Late April | Full draft to supervisor |
| May | Revisions + final submission |

---

## Known Open Issues (Not Blocking)

| Issue | Severity | Notes |
|-------|:--------:|-------|
| A4 confidence drop: need validation re-run | 🟡 Medium | Run A4 again to confirm rollback fires |
| Prometheus / Grafana not live | 🟢 Low | Docker Compose ready; not needed for thesis |
| Class list in older docs shows 11 classes without "smoking" | 🟡 Medium | Update `TRAINING_RECIPE_MATRIX.md` class list from `models/retrained/label_mapping.json` |
| `models/training_baseline.json` in git may be stale | 🟢 Low | Remove from tracking or add `.gitignore` entry |
| `exit code 1` on PARTIAL runs is correct but confusing | 🟢 Low | Pipeline returns 1 when any stage fails; acceptable for thesis |

---

## Useful Commands Reference

```powershell
# Run inference only (no retrain)
python run_pipeline.py --skip-ingestion

# Run with supervised retrain
python run_pipeline.py --retrain --adapt none --skip-cv --epochs 10 --skip-ingestion

# Run AdaBN+TENT adaptation
python run_pipeline.py --retrain --adapt adabn_tent --skip-ingestion

# Run pseudo-label adaptation
python run_pipeline.py --retrain --adapt pseudo_label --skip-cv --epochs 10 --skip-ingestion

# Run with baseline promotion (careful!)
python run_pipeline.py --retrain --adapt none --skip-ingestion --update-baseline

# Export MLflow runs to CSV
python scripts/export_mlflow_runs.py --experiment har-retraining

# Open MLflow UI
mlflow ui --backend-store-uri mlruns/

# Check today's git log
git log --oneline --since="today"
```

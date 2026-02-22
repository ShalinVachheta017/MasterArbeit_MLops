# 26 — Thesis Figures and Tables Backlog

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22

---

## 1 Figures Backlog

| ID | Title | Type | Source | Chapter | Status | Notes |
|:--:|-------|------|--------|:-------:|:------:|-------|
| F-01 | End-to-End 14-Stage Pipeline | Mermaid flowchart | Diagram D-1 (File 22) | 3.1 | ✅ Mermaid ready | Render to SVG/PDF |
| F-02 | 3-Layer Monitoring Data Flow | Mermaid flowchart | Diagram D-2 (File 22) | 3.4 | ✅ Mermaid ready | Render to SVG/PDF |
| F-03 | Alert Escalation State Machine | Mermaid stateDiagram | Diagram D-3 (File 22) | 3.4 | ✅ Mermaid ready | Render to SVG/PDF |
| F-04 | Adaptation Decision Flow (4 Methods) | Mermaid flowchart | Diagram D-4 (File 22) | 3.6 | ✅ Mermaid ready | Render to SVG/PDF |
| F-05 | Drift → Adaptation Pipeline (Stages 6-10) | Mermaid flowchart | Diagram D-5 (File 22) | 3.5 | ✅ Mermaid ready | Render to SVG/PDF |
| F-06 | Trigger Policy State Machine | Mermaid stateDiagram | Diagram D-6 (File 22) | 3.5 | ✅ Mermaid ready | Render to SVG/PDF |
| F-07 | Full Retrain Governance Cycle | Mermaid flowchart | Diagram D-7 (File 22) | 3.7 | ✅ Mermaid ready | Render to SVG/PDF |
| F-08 | Docker Service Architecture | Mermaid flowchart | Diagram D-8 (File 22) | 4.3 | ✅ Mermaid ready | Render to SVG/PDF |
| F-09 | CI/CD Pipeline (7 Jobs) | Mermaid flowchart | Diagram D-9 (File 22) | 4.4 | ✅ Mermaid ready | Render to SVG/PDF |
| F-10 | Data Pipeline Flow (Stages 1-3) | Mermaid flowchart | Diagram D-10 (File 22) | 3.2 | ✅ Mermaid ready | Render to SVG/PDF |
| F-11 | 1D-CNN-BiLSTM Architecture Diagram | Custom diagram | `src/train.py:L300-450` | 3.3 | ❌ Needs creation | Draw.io or TikZ; show Conv1D → BiLSTM → Dense |
| F-12 | Confusion Matrix (Baseline 5-Fold CV) | Heatmap | Experiment E-1 output | 5.2 | ❌ Needs experiment | `seaborn.heatmap()` |
| F-13 | Drift Score vs Confidence Scatter Plot | Scatter | Experiment E-2, E-3 output | 5.3 | ❌ Needs experiment | 26 sessions; color by drift severity |
| F-14 | Monitoring Layer Activation Heatmap | Heatmap | Experiment E-4 output | 5.4 | ❌ Needs experiment | Sessions × Layers; color = PASS/WARN/ALERT |
| F-15 | Adaptation Method Comparison Bar Chart | Grouped bar | Experiment E-5 output | 5.5 | ❌ Needs experiment | 5 methods × Δconfidence, Δentropy |
| F-16 | Trigger Decision Distribution Pie/Bar | Bar chart | Experiment E-6 output | 5.6 | ❌ Needs experiment | NONE/MONITOR/QUEUE/TRIGGER counts |
| F-17 | Proxy vs True Accuracy Scatter + Pearson r | Scatter + fit line | Experiment E-7 output (if labeled data available) | 5.7 | ❌ Needs labeled data | May become theoretical figure |
| F-18 | Ablation: 1 vs 2 vs 3 Layers Performance | Grouped bar | Experiment E-8 output | 5.8 | ❌ Needs experiment | Show degradation detection rate |
| F-19 | Runtime Breakdown per Stage | Stacked bar | Experiment E-10 timing | 5.9 | ❌ Needs profiling | `time.time()` instrumentation |
| F-20 | Repository Module Dependency Map | Custom diagram | File inventory (File 02) | 4.1 | ❌ Optional | `pydeps` or manual Draw.io |

---

## 2 Tables Backlog

| ID | Title | Content | Chapter | Status | Data Source |
|:--:|-------|---------|:-------:|:------:|------------|
| T-01 | 14-Stage Pipeline Summary | Stage #, name, module, status, test count | 3.1 | ✅ Data in File 03 | Format from completion audit |
| T-02 | Dataset Characteristics | Sessions, samples, frequency, activities, sensor info | 5.1 | 🔶 Partial data | `batch_process_all_datasets.py` |
| T-03 | 1D-CNN-BiLSTM Layer Configuration | Layer, type, output shape, parameters, activation | 3.3 | ✅ Data in `train.py` | Extract from model summary |
| T-04 | Monitoring Thresholds | Layer, metric, threshold, action | 3.4 | ✅ Data in File 12 | `post_inference_monitoring.py` |
| T-05 | Adaptation Method Comparison | Method, requires labels?, safety gates, computational cost | 3.6 | ✅ Data in File 13 | Phase 2 analysis |
| T-06 | Trigger Policy Parameters (17 tunable) | Parameter, default, range, effect | 3.5 | ✅ Data in File 14 | `trigger_policy.py` |
| T-07 | Test Coverage Matrix | Module, unit tests, integration tests, total | 4.5 | ✅ Data in File 15 | `tests/` directory |
| T-08 | CI/CD Job Summary | Job #, name, trigger, dependencies, status | 4.4 | ✅ Data in File 15 | `ci-cd.yml` |
| T-09 | 5-Fold CV Baseline Results | Fold, accuracy, F1, Kappa, loss | 5.2 | ❌ Needs experiment | Experiment E-1 |
| T-10 | Per-Session Drift Scores | Session, Z-score mean, Z-score max, drift level | 5.3 | ❌ Needs experiment | Experiment E-2, E-3 |
| T-11 | Adaptation Results (5 methods × 5 sessions) | Method, session, Δconf, Δentropy, runtime | 5.5 | ❌ Needs experiment | Experiment E-5 |
| T-12 | Trigger Decision Distribution | Action, count, percentage | 5.6 | ❌ Needs experiment | Experiment E-6 |
| T-13 | Proxy Metric Reliability | Proxy metric, Pearson r, Spearman ρ, verdict | 5.7 | ❌ Needs labeled data | Experiment E-7 |
| T-14 | Monitoring Layer Ablation | Config, detection rate, false positive rate | 5.8 | ❌ Needs experiment | Experiment E-8 |
| T-15 | Runtime Breakdown per Stage | Stage, mean time, std, % of total | 5.9 | ❌ Needs profiling | Experiment E-10 |
| T-16 | Gap/Technical Debt Summary | Priority, count, key items | 4.7 | ✅ Data in File 24 | Phase 2 compilation |
| T-17 | Research Paper Theme Summary | Theme, papers, key finding | 2.2 | ✅ Available | `Summary_of_7_Research_Themes_in_HAR.csv` |

---

## 3 Existing Figures Inventory

| File | Type | Content | Reuse? |
|------|------|---------|:------:|
| `docs/figures/pipeline_architecture.png` | PNG | Pipeline overview | Review vs F-01 |
| `docs/figures/monitoring_layers.png` | PNG | 3-layer monitoring | Review vs F-02 |
| `docs/figures/confusion_matrix.png` | PNG | Training confusion matrix | May need update |
| `docs/figures/drift_analysis.png` | PNG | Drift visualization | Review vs F-13 |
| `docs/figures/adaptation_comparison.png` | PNG | Adaptation bar chart | Review vs F-15 |
| `docs/figures/model_architecture.png` | PNG | CNN-BiLSTM diagram | Review vs F-11 |
| `docs/figures/cicd_pipeline.png` | PNG | CI/CD flow | Review vs F-09 |

**Action:** Compare each existing PNG with the corresponding Mermaid diagram. Use whichever is higher quality. Re-render Mermaid versions for consistency.

---

## 4 Figure Generation Scripts

| Script | Generates | Status |
|--------|-----------|:------:|
| `scripts/generate_thesis_figures.py` | Multiple thesis figures | Exists — review and update |
| File 22 Mermaid diagrams | F-01 through F-10 | Ready to render via `mmdc` |
| New: experiment analysis script | F-12 through F-19 | **Needs creation** after experiments |

---

## 5 Priority Order for Figure Generation

| Phase | Figures | When |
|:-----:|---------|------|
| **Now** | F-01 through F-10 (Mermaid → SVG) | Can render immediately |
| **After experiments** | F-12, F-13, F-14, F-15, F-16 (core results) | Week 2-3 |
| **If time permits** | F-11 (architecture), F-17, F-18, F-19, F-20 | Week 4+ |

**Total figures needed:** ~20 | **Ready now:** 10 (Mermaid) | **Need experiments:** 8 | **Optional:** 2

> ⚠️ **ARCHIVED - SAFE TO DELETE**
> 
> **Reason:** Superseded by `docs/PATH_COMPARISON_ANALYSIS.md`
> 
> **Why not needed:** The two-pathway analysis has been consolidated into PATH_COMPARISON_ANALYSIS which has cleaner, corrected analysis.

---

# MLOps Project: Two Pathway To Do List

**Created:** December 11, 2025  
**Mentor Direction:** Focus on MLOps pipeline first; domain shift fixes deferred  
**Status:** Choose your path based on priorities

---

---

# 📊 SIDE-BY-SIDE COMPARISON: PATH A vs PATH B

```
┌─────────────────────────────────────────┬─────────────────────────────────────────┐
│         🛤️ PATH A: MLOps Only          │    🛤️ PATH B: MLOps + Domain Shift     │
│         (Mentor's Direction)            │         (Extended Plan)                 │
├─────────────────────────────────────────┼─────────────────────────────────────────┤
│ GOAL: Build MLOps infrastructure        │ GOAL: Fix model THEN build MLOps       │
│ around existing model (even if broken)  │ around a working model                 │
├─────────────────────────────────────────┼─────────────────────────────────────────┤
│ ⏱️ TIME: 2-3 weeks                      │ ⏱️ TIME: 4-5 weeks                      │
│ ✅ MENTOR: Approved                     │ ⚠️ MENTOR: Later                        │
│ 📊 ACCURACY: Low (100% hand_tapping)    │ 📊 ACCURACY: Higher (+20-50%)           │
│ 📚 THESIS: MLOps infrastructure focus   │ 📚 THESIS: Complete ML lifecycle        │
└─────────────────────────────────────────┴─────────────────────────────────────────┘
```

---

# 🗓️ WEEK-BY-WEEK SIDE-BY-SIDE

## WEEK 1

| PATH A (MLOps Only) | PATH B (MLOps + Domain Shift) |
|---------------------|-------------------------------|
| **A1.** Set up DVC for data versioning (2-3h) ⬜ | **B1.** Implement gravity removal filter (2-3h) ⬜ |
| **A2.** Set up MLflow for experiment tracking (2-3h) ⬜ | **B2.** Test gravity removal on production data (2-3h) ⬜ |
| **A3.** Create reproducible training script (3-4h) ⬜ | **B3.** Run inference with gravity-removed features (2h) ⬜ |
| **A4.** Document current pipeline architecture (2h) ⬜ | **B4.** Evaluate accuracy improvement (2h) ⬜ |
| | **B5.** Implement g-unit normalization (2h) ⬜ |
| | **B6.** Compare gravity removal vs g-unit (2h) ⬜ |
| **Deliverables:** | **Deliverables:** |
| • `dvc.yaml` with data pipeline | • Gravity removal code in preprocessing |
| • MLflow tracking server running | • Accuracy comparison table |
| • Training script with logging | • Best solution selected |
| • Architecture diagram | |

---

## WEEK 2

| PATH A (MLOps Only) | PATH B (MLOps + Domain Shift) |
|---------------------|-------------------------------|
| **A5.** Build FastAPI inference endpoint (3-4h) ⬜ | **B7.** Prepare user calibration protocol (2h) ⬜ |
| **A6.** Add input validation (2h) ⬜ | **B8.** Implement fine-tuning pipeline (4-6h) ⬜ |
| **A7.** Implement drift detection monitoring (3-4h) ⬜ | **B9.** Collect user calibration data (1-2 days) ⬜ |
| **A8.** Create Prometheus metrics exporter (2-3h) ⬜ | **B10.** Fine-tune model on user data (2-3h) ⬜ |
| **A9.** Build Grafana dashboard (2-3h) ⬜ | **B11.** Evaluate personalized model (2h) ⬜ |
| **Deliverables:** | **Deliverables:** |
| • `/predict` endpoint working | • Fine-tuning code ready |
| • Drift detection logging | • User calibration protocol doc |
| • Monitoring dashboard | • Personalized model (+30-50% acc) |

---

## WEEK 3

| PATH A (MLOps Only) | PATH B (MLOps + Domain Shift) |
|---------------------|-------------------------------|
| **A10.** Create Dockerfile (2-3h) ⬜ | **B12.** Implement MMD loss for domain alignment (4-6h) ⬜ |
| **A11.** Create docker-compose (2h) ⬜ | **B13.** Train with labeled + unlabeled data (3-4h) ⬜ |
| **A12.** Set up GitHub Actions CI (3-4h) ⬜ | **B14.** Evaluate domain-adapted model (2h) ⬜ |
| **A13.** Add automated tests (pytest) (3-4h) ⬜ | **B15.** Integrate best solution into preprocessing (2-3h) ⬜ |
| **A14.** Document deployment process (2h) ⬜ | **B16.** Update inference pipeline (2-3h) ⬜ |
| **Deliverables:** | **Deliverables:** |
| • Docker image for inference | • Domain-adapted model |
| • CI pipeline (lint, test, build) | • Updated preprocessing pipeline |
| • Deployment documentation | • Best solution integrated |

---

## WEEK 4

| PATH A (MLOps Only) | PATH B (MLOps + Domain Shift) |
|---------------------|-------------------------------|
| **A15.** Write MLOps thesis section (4-6h) ⬜ | **B17.** Add drift → recalibration trigger (3-4h) ⬜ |
| **A16.** Create architecture diagrams (2-3h) ⬜ | **B18.** Build full MLOps stack (1 week) ⬜ |
| **A17.** Document monitoring design (2-3h) ⬜ | (Same as Path A Week 2-3) |
| **A18.** Prepare demo for mentor (2h) ⬜ | |
| **Deliverables:** | **Deliverables:** |
| • 5-10 page thesis section | • MLOps + working model |
| • Architecture diagrams | • Auto-recalibration system |
| • Working demo | |

---

## WEEK 5 (Path B only)

| PATH A (MLOps Only) | PATH B (MLOps + Domain Shift) |
|---------------------|-------------------------------|
| ✅ **DONE** | **B19.** Write thesis (MLOps + domain shift) ⬜ |
| | **B20.** Create architecture diagrams ⬜ |
| | **B21.** Prepare demo for mentor ⬜ |
| | **Deliverables:** |
| | • Complete thesis section |
| | • Full working demo |

---

# 🎯 QUICK DECISION TABLE

```
┌─────────────────────────────┬───────────────┬───────────────┐
│ QUESTION                    │ PATH A        │ PATH B        │
├─────────────────────────────┼───────────────┼───────────────┤
│ Want results FAST?          │ ✅ YES        │ ❌ NO         │
│ Mentor approved?            │ ✅ YES        │ ⚠️ LATER      │
│ Model predictions work?     │ ❌ NO         │ ✅ YES        │
│ Need extra data?            │ ❌ NO         │ ⚠️ MAYBE      │
│ Full ML lifecycle demo?     │ ❌ NO         │ ✅ YES        │
│ Risk level?                 │ 🟢 LOW        │ 🟡 MEDIUM     │
│ Thesis pages?               │ 5-10 pages    │ 10-15 pages   │
└─────────────────────────────┴───────────────┴───────────────┘
```

---

# 📋 PAPER REFERENCES (Path B Only)

| Solution | Paper | Expected Impact |
|----------|-------|-----------------|
| Gravity Removal | Anguita et al. (2013), Yurtman et al. (2017) | +20-40% accuracy |
| g-Unit Normalization | Dhekane & Ploetz (2024), DAGHAR Benchmark | +10-20% accuracy |
| Personalization | Dey et al. (2015) | +30-50% accuracy |
| Domain Adaptation | Sanabria et al. (2021), Ganin & Lempitsky (2015) | +15-30% accuracy |

**Code Location:** `docs/THREE_SOLUTIONS_COMPLETE_CODE.md`

---

# 🎯 Recommended Approach

**Start with Path A** (mentor approved) but keep Path B solutions ready:

1. **Week 1-2:** Complete Path A Phase 1-2 (pipeline + monitoring)
2. **During monitoring:** You'll detect drift automatically (Az = -9.83 alert)
3. **Week 3:** Either continue Path A or pivot to Path B based on mentor feedback
4. **Path B code:** Already in `docs/archived/` folder, ready when needed

This way you:
- ✅ Follow mentor direction
- ✅ Build real MLOps infrastructure
- ✅ Can demonstrate drift detection working (thesis value!)
- ✅ Have solutions ready when domain shift work is approved

---

# 📁 File References

| File | Purpose |
|------|---------|
| `docs/CONCEPTS_EXPLAINED.md` | Technical background (units, windowing, etc.) |
| `docs/RESEARCH_PAPERS_ANALYSIS.md` | Paper methodology reference |
| `docs/SRC_FOLDER_ANALYSIS.md` | Codebase structure |
| `docs/UNIT_CONVERSION_SOLUTION.md` | Implemented conversion (milliG → m/s²) |
| `docs/archived/THREE_SOLUTIONS_COMPLETE_CODE.md` | Domain shift solutions (deferred) |
| `docs/archived/SOLUTION_IMPLEMENTATION_GUIDE.md` | Solution implementation guide (deferred) |
| `docs/archived/FINAL_PIPELINE_PROBLEMS_ANALYSIS.md` | Root cause analysis (deferred) |

---

**Last Updated:** December 11, 2025  
**Next Review:** After Phase 1 completion

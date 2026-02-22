# 22 — Figures and Diagrams (Mermaid Pack)

> **Status:** COMPLETE — Phase 3  
> **Repository Snapshot:** `168c05bb222b03e699acb7de7d41982e886c8b25`  
> **Auditor:** Claude Opus 4.6 | **Date:** 2026-02-22  
> **Usage:** Copy any diagram block into a Mermaid renderer (VS Code extension, Mermaid.live, LaTeX mermaid-filter) to generate thesis-ready SVG/PDF.

---

## Diagram Index

| # | Diagram | Type | Thesis Chapter | Source |
|--:|---------|------|:--------------:|--------|
| D-1 | End-to-End 14-Stage Pipeline | flowchart TB | Ch 3.1 | **NEW** |
| D-2 | 3-Layer Monitoring Data Flow | flowchart TB | Ch 3.4 | File 12 §5 (polished) |
| D-3 | Alert Escalation State Machine | stateDiagram-v2 | Ch 3.4 | File 12 §6 (polished) |
| D-4 | Adaptation Decision Flow | flowchart TB | Ch 3.6 | File 13 §4 (polished) |
| D-5 | Drift → Adaptation End-to-End | flowchart LR | Ch 3.5-3.6 | File 13 §5 (polished) |
| D-6 | Trigger Policy State Machine | stateDiagram-v2 | Ch 3.5 | File 14 §3 (polished) |
| D-7 | Full Retrain Governance Cycle | flowchart TB | Ch 3.7 | File 14 §8 (polished) |
| D-8 | Docker Service Architecture | flowchart LR | Ch 4.3 | **NEW** |
| D-9 | CI/CD Pipeline (7 Jobs) | flowchart LR | Ch 4.4 | **NEW** |
| D-10 | Data Pipeline Flow | flowchart TB | Ch 3.2 | **NEW** |

---

## D-1 — End-to-End 14-Stage Pipeline Architecture

> **Thesis placement:** Chapter 3, Section 3.1 — System Overview  
> **Interpretation:** Shows all 14 stages of the MLOps pipeline. Stages 1-10 (green) are wired into the orchestrator; stages 11-14 (orange) are implemented in code but not yet integrated.

```mermaid
flowchart TB
    subgraph Orchestrated["Orchestrated Stages (ProductionPipeline.ALL_STAGES)"]
        direction TB
        S1["Stage 1: Data Ingestion<br/>3 paths · merge_asof · manifest skip"]
        S2["Stage 2: Data Validation<br/>schema · bounds · NaN · sensor coverage"]
        S3["Stage 3: Data Transformation<br/>unit → g · gravity removal · domain calibration"]
        S4["Stage 4: Batch Inference<br/>sliding window · 200×6 → model → predictions CSV"]
        S5["Stage 5: Model Evaluation<br/>confidence stats · entropy · per-class metrics"]
        S6["Stage 6: Post-Inference Monitoring<br/>L1 confidence · L2 temporal · L3 drift"]
        S7["Stage 7: Trigger Evaluation<br/>3-signal voting · cooldown · action decision"]
        S8["Stage 8: Model Retraining<br/>AdaBN / TENT / pseudo-label"]
        S9["Stage 9: Model Registration<br/>SHA256 fingerprint · version · is_better gate"]
        S10["Stage 10: Baseline Update<br/>rebuild stats · promote_to_shared decision"]
    end

    subgraph NotWired["Implemented but NOT Orchestrated"]
        direction TB
        S11["Stage 11: Temperature Calibration<br/>src/calibration.py"]
        S12["Stage 12: Wasserstein Drift<br/>src/wasserstein_drift.py"]
        S13["Stage 13: Curriculum Pseudo-Labeling<br/>src/curriculum_pseudo_labeling.py"]
        S14["Stage 14: Model Rollback<br/>src/model_rollback.py"]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -- "should_retrain=True" --> S8 --> S9 --> S10
    S7 -- "action=MONITOR/NONE" --> DONE["Pipeline complete"]

    S10 -.-> S11
    S10 -.-> S12
    S10 -.-> S13
    S10 -.-> S14

    style S1 fill:#c8e6c9
    style S2 fill:#c8e6c9
    style S3 fill:#c8e6c9
    style S4 fill:#c8e6c9
    style S5 fill:#c8e6c9
    style S6 fill:#c8e6c9
    style S7 fill:#c8e6c9
    style S8 fill:#c8e6c9
    style S9 fill:#c8e6c9
    style S10 fill:#c8e6c9
    style S11 fill:#ffe0b2
    style S12 fill:#ffe0b2
    style S13 fill:#ffe0b2
    style S14 fill:#ffe0b2
```

---

## D-2 — 3-Layer Monitoring Data Flow

> **Thesis placement:** Chapter 3, Section 3.4 — Monitoring Framework  
> **Interpretation:** Three orthogonal monitoring layers process different input signals (confidence CSV, temporal sequence, drift from raw features vs baseline). Each layer independently produces PASS/ALERT/WARNING. Results aggregate into an overall status report.  
> **Evidence:** `scripts/post_inference_monitoring.py:L1-400`

```mermaid
flowchart TB
    subgraph Inputs
        CSV["predictions.csv<br/>(confidence, predicted_activity)"]
        NPY["production_X.npy<br/>(N × 200 × 6)"]
        BL["baseline.json<br/>(mean, std per channel)"]
    end

    subgraph Layer1["Layer 1 — Confidence"]
        L1A["Read max_prob per window"]
        L1B["Compute mean_confidence"]
        L1C["Count uncertain_pct<br/>(max_prob < 0.5)"]
        L1D{uncertain_pct > 10%?}
        L1P["PASS"]
        L1W["ALERT"]
    end

    subgraph Layer2["Layer 2 — Temporal"]
        L2A["Read predicted_activity sequence"]
        L2B["Count transitions<br/>(pred[i] ≠ pred[i-1])"]
        L2C["Compute transition_rate %"]
        L2D{transition_rate > 50%?}
        L2P["PASS"]
        L2W["WARNING"]
    end

    subgraph Layer3["Layer 3 — Drift"]
        L3A["Load baseline (mean, std)"]
        L3B["Compute per-channel<br/>Z-score distance"]
        L3C["max_drift = max(z_c)"]
        L3D{max_drift > 0.75?}
        L3P["PASS"]
        L3W["ALERT"]
        L3S["SKIPPED<br/>(no baseline)"]
    end

    subgraph Output
        AGG["_determine_overall_status()"]
        RPT["MonitoringReport JSON"]
    end

    CSV --> L1A --> L1B --> L1C --> L1D
    L1D -- No --> L1P
    L1D -- Yes --> L1W

    CSV --> L2A --> L2B --> L2C --> L2D
    L2D -- No --> L2P
    L2D -- Yes --> L2W

    NPY --> L3B
    BL --> L3A --> L3B --> L3C --> L3D
    L3D -- No --> L3P
    L3D -- Yes --> L3W
    BL -. "missing" .-> L3S

    L1P & L1W & L2P & L2W & L3P & L3W & L3S --> AGG --> RPT
```

---

## D-3 — Alert Escalation State Machine

> **Thesis placement:** Chapter 3, Section 3.4 — Monitoring Framework (subsection on escalation)  
> **Interpretation:** Shows how monitoring status transitions between PASS, WARNING, ALERT, and how ALERT feeds into Stage 7 trigger evaluation.  
> **Evidence:** `scripts/post_inference_monitoring.py:L300-350`

```mermaid
stateDiagram-v2
    [*] --> PASS

    PASS --> WARNING : Layer 2 fires (transition_rate > 50%)
    PASS --> ALERT : Layer 1 fires (uncertain > 10%)\nOR Layer 3 fires (drift > 0.75)

    WARNING --> ALERT : Layer 1 or 3 also fires
    WARNING --> PASS : Next run all clear

    ALERT --> PASS : Next run all clear
    ALERT --> TRIGGER : Monitoring feeds into\nStage 7 Trigger Evaluation

    TRIGGER --> RETRAIN : TriggerPolicyEngine\ndecides should_retrain=True
    TRIGGER --> MONITOR : TriggerPolicyEngine\ndecides action=MONITOR
```

---

## D-4 — Adaptation Decision Flow (4 Methods)

> **Thesis placement:** Chapter 3, Section 3.6 — Adaptation Methods  
> **Interpretation:** Four adaptation paths (AdaBN, TENT, combined, pseudo-labeling) with their internal steps, safety gates, and rollback conditions. TENT has an OOD guard + rollback check; pseudo-labeling has source-holdout catastrophic-forgetting check.  
> **Evidence:** `src/domain_adaptation/adabn.py`, `tent.py`, `src/components/model_retraining.py:L200+`

```mermaid
flowchart TB
    START["Trigger: should_retrain=True"]
    
    METHOD{"adaptation_method config"}
    
    subgraph AdaBN_Path["AdaBN Only"]
        AB1["Reset BN running stats"]
        AB2["Forward pass target data<br/>10 batches × 64 samples"]
        AB3["BN stats updated"]
    end
    
    subgraph TENT_Path["TENT Only"]
        T1["OOD Guard<br/>norm_entropy > 0.85?"]
        T2["SKIP — model unchanged"]
        T3["Snapshot BN γ/β + running stats"]
        T4["10× entropy minimization steps<br/>Adam lr=1e-4"]
        T5["Restore running stats each step"]
        T6{"Rollback check<br/>ΔH > 0.05 or Δconf < -0.01?"}
        T7["ROLLBACK to snapshot"]
        T8["Accept adapted model"]
    end
    
    subgraph Combined["AdaBN + TENT"]
        C1["Run AdaBN first"]
        C2["Run TENT second<br/>(preserves AdaBN stats)"]
    end
    
    subgraph PseudoLabel["Pseudo-Labeling"]
        PL1["Temperature calibration<br/>T ∈ [0.5, 3.0]"]
        PL2["Predict + entropy gate<br/>conf ≥ 0.70, H_norm ≤ 0.40"]
        PL3["Class-balanced top-k"]
        PL4["Fine-tune last 3 layers<br/>0.1× learning rate"]
        PL5{"Source holdout<br/>acc drop > 10%?"}
        PL6["ROLLBACK to base"]
        PL7["Accept fine-tuned model"]
    end
    
    SAVE["Save adapted model<br/>→ Stage 9 Registration"]
    
    START --> METHOD
    METHOD -- "adabn" --> AB1 --> AB2 --> AB3 --> SAVE
    METHOD -- "tent" --> T1
    T1 -- "Yes (OOD)" --> T2
    T1 -- "No" --> T3 --> T4 --> T5 --> T6
    T6 -- "Yes" --> T7
    T6 -- "No" --> T8 --> SAVE
    METHOD -- "adabn_tent" --> C1 --> C2 --> SAVE
    METHOD -- "pseudo_label" --> PL1 --> PL2 --> PL3 --> PL4 --> PL5
    PL5 -- "Yes" --> PL6
    PL5 -- "No" --> PL7 --> SAVE
```

---

## D-5 — Drift Detection → Adaptation End-to-End (Stages 6-10)

> **Thesis placement:** Chapter 3, Sections 3.5-3.6 — Overview of the reactive loop  
> **Interpretation:** End-to-end flow from monitoring (Stage 6) through trigger decision (Stage 7), adaptation (Stage 8), registration with proxy validation (Stage 9), and baseline update with promote-to-shared logic (Stage 10).  
> **Evidence:** `src/trigger_policy.py`, `src/components/model_registration.py`, `src/components/baseline_update.py`

```mermaid
flowchart LR
    subgraph Stage6["Stage 6: Monitoring"]
        MON["3-Layer Monitor"]
        L3["Layer 3: Z-score drift<br/>threshold 0.75"]
    end
    
    subgraph Stage7["Stage 7: Trigger"]
        WASS["Wasserstein drift<br/>(W₁ per channel)"]
        VOTE["3-signal voting<br/>(confidence, temporal, drift)"]
        COOL["Cooldown check<br/>(24h since last retrain?)"]
        DEC{"should_retrain?"}
    end
    
    subgraph Stage8["Stage 8: Retraining"]
        ADAPT["Adaptation method<br/>(AdaBN / TENT / Pseudo)"]
    end
    
    subgraph Stage9["Stage 9: Registration"]
        REG["Model Registry<br/>version + SHA256"]
        PROXY["Proxy validation<br/>(⚠ placeholder: is_better=True)"]
        DEPLOY["Auto-deploy if better"]
    end
    
    subgraph Stage10["Stage 10: Baseline Update"]
        BLINE["Rebuild baseline stats"]
        PROMOTE{"promote_to_shared?"}
        SHARED["Overwrite shared baseline"]
        LOCAL["Artifact-only baseline"]
    end
    
    MON --> L3 --> VOTE
    WASS --> VOTE --> COOL --> DEC
    DEC -- "Yes" --> ADAPT --> REG --> PROXY --> DEPLOY
    DEC -- "No" --> MONITOR_MORE["Continue monitoring"]
    DEPLOY --> BLINE --> PROMOTE
    PROMOTE -- "True" --> SHARED
    PROMOTE -- "False" --> LOCAL
```

---

## D-6 — Trigger Policy State Machine

> **Thesis placement:** Chapter 3, Section 3.5 — Trigger Policy  
> **Interpretation:** Shows how the trigger evaluates 3 signals into 5 action levels (NONE → MONITOR → QUEUE → TRIGGER → RETRAIN), including the escalation override (3 consecutive warnings forces retrain) and cooldown suppression (downgrades to QUEUE if last retrain < 24h).  
> **Evidence:** `src/trigger_policy.py:L1-822`

```mermaid
stateDiagram-v2
    [*] --> EVALUATE

    EVALUATE --> NONE : All 3 signals INFO
    EVALUATE --> MONITOR : Exactly 1 signal WARNING
    EVALUATE --> QUEUE_RETRAIN : 2+ signals WARNING
    EVALUATE --> TRIGGER_RETRAIN : Any signal CRITICAL

    MONITOR --> MONITOR : consecutive_warnings < 3
    MONITOR --> TRIGGER_RETRAIN : consecutive_warnings ≥ 3\n(escalation override)

    QUEUE_RETRAIN --> TRIGGER_RETRAIN : consecutive_warnings ≥ 3

    TRIGGER_RETRAIN --> COOLDOWN_CHECK : should_trigger = True

    COOLDOWN_CHECK --> RETRAIN : last_retrain > 24h ago
    COOLDOWN_CHECK --> QUEUE_RETRAIN : last_retrain < 24h ago\n(cooldown suppresses)

    RETRAIN --> REGISTRATION : Stage 9
    REGISTRATION --> BASELINE : Stage 10
    BASELINE --> [*]

    NONE --> [*]
```

---

## D-7 — Full Retrain Governance Cycle

> **Thesis placement:** Chapter 3, Section 3.7 — Model Governance & Rollback  
> **Interpretation:** Complete governance cycle from trigger → retraining → registration (SHA256 fingerprint + proxy validation, currently placeholder) → baseline update → rollback path. The rollback branch shows detection of quality regression → version rollback → shape/inference validation.  
> **Evidence:** `src/model_rollback.py`, `src/components/model_registration.py`

```mermaid
flowchart TB
    subgraph Stage7["Stage 7: Trigger"]
        EVAL["TriggerPolicyEngine.evaluate()"]
        VOTE["2-of-3 voting"]
        COOL["Cooldown (24h)"]
        DEC{should_retrain?}
    end
    
    subgraph Stage8["Stage 8: Retraining"]
        LOAD["Load base model<br/>fine_tuned_model_1dcnnbilstm.keras"]
        ADAPT["Run adaptation method"]
        SAVE["Save retrained model"]
    end
    
    subgraph Stage9["Stage 9: Registration"]
        REG["register_model()<br/>version + SHA256[:12]"]
        PROXY{"Proxy validation<br/>(⚠ PLACEHOLDER: is_better=True)"}
        DEPLOY["deploy_model()<br/>copy → current_model.keras"]
        HIST["Append to history<br/>(append-only audit trail)"]
    end
    
    subgraph Stage10["Stage 10: Baseline"]
        BUILD["BaselineBuilder.build_from_csv()"]
        PROM{"promote_to_shared?"}
        SHARED["Overwrite shared baseline<br/>+ versioned archive"]
        LOCAL["Artifact-only<br/>(safe default)"]
    end
    
    subgraph Rollback["Rollback Path"]
        DETECT["Quality regression detected"]
        RB["rollback(target_version)"]
        MARK["Mark current as 'rollback'"]
        RESTORE["Copy archived model<br/>→ current_model.keras"]
        VALIDATE["RollbackValidator<br/>shape + inference check"]
    end
    
    EVAL --> VOTE --> COOL --> DEC
    DEC -- "Yes" --> LOAD --> ADAPT --> SAVE
    DEC -- "No" --> MONITOR_MORE["Continue monitoring"]
    SAVE --> REG --> PROXY
    PROXY -- "True (always)" --> DEPLOY --> HIST
    HIST --> BUILD --> PROM
    PROM -- "True" --> SHARED
    PROM -- "False" --> LOCAL
    
    DEPLOY -.-> DETECT -.-> RB --> MARK --> RESTORE --> VALIDATE
```

---

## D-8 — Docker Service Architecture

> **Thesis placement:** Chapter 4, Section 4.3 — API & Containerization  
> **Interpretation:** Shows the 4-service Docker Compose setup: FastAPI inference service (port 8000), MLflow tracking server (port 5000), Prometheus (port 9090, config-only), and training service. Network connections and volume mounts.  
> **Evidence:** `docker-compose.yml:L1-143`, `docker/Dockerfile.inference`, `docker/Dockerfile.training`

```mermaid
flowchart LR
    subgraph DockerCompose["docker-compose.yml (4 services)"]
        direction TB
        subgraph InferenceService["inference-api"]
            FA["FastAPI App<br/>Dockerfile.inference<br/>Port 8000"]
            EP1["/predict"]
            EP2["/upload"]
            EP3["/health"]
            DASH["HTML Dashboard"]
        end
        
        subgraph MLflowService["mlflow-tracking"]
            ML["MLflow Server<br/>Port 5000<br/>SQLite backend"]
        end
        
        subgraph TrainingService["training"]
            TR["Training Container<br/>Dockerfile.training<br/>GPU passthrough"]
        end
        
        subgraph PrometheusService["prometheus (config-only)"]
            PM["Prometheus<br/>Port 9090<br/>⚠ Not in compose"]
        end
    end
    
    subgraph Volumes["Shared Volumes"]
        V1["./artifacts:/app/artifacts"]
        V2["./models:/app/models"]
        V3["./mlruns:/mlflow/mlruns"]
    end
    
    FA --> ML
    TR --> ML
    FA --> V1 & V2
    TR --> V1 & V2
    ML --> V3
    PM -.-> FA

    style PM fill:#fff3e0,stroke:#f57c00
```

---

## D-9 — CI/CD Pipeline (7 Jobs)

> **Thesis placement:** Chapter 4, Section 4.4 — CI/CD Workflow  
> **Interpretation:** GitHub Actions workflow with 7 jobs: lint, unit tests, integration tests, build Docker images, model validation (⚠ stub), notification, and manual deploy. Shows dependency chain.  
> **Evidence:** `.github/workflows/ci-cd.yml:L1-236`

```mermaid
flowchart LR
    PUSH["Push / PR to main"]
    
    subgraph CI["CI Jobs (parallel start)"]
        LINT["Job 1: Lint<br/>flake8 + black --check"]
        UNIT["Job 2: Unit Tests<br/>pytest -m 'not slow'<br/>markers: unit, integration, slow"]
    end
    
    subgraph Test["After CI"]
        INT["Job 3: Integration Tests<br/>pytest -m integration"]
    end
    
    subgraph Build["After Tests"]
        DOCKER["Job 4: Build Docker<br/>inference + training images"]
        MODEL["Job 5: Model Validation<br/>⚠ echo stubs only"]
    end
    
    subgraph Deploy["Final"]
        NOTIFY["Job 6: Notification<br/>Slack/email on failure"]
        DEPLOY["Job 7: Deploy<br/>manual trigger only"]
    end
    
    PUSH --> LINT & UNIT
    LINT --> INT
    UNIT --> INT
    INT --> DOCKER & MODEL
    DOCKER --> NOTIFY
    MODEL --> NOTIFY
    NOTIFY --> DEPLOY

    style MODEL fill:#fff3e0,stroke:#f57c00
```

---

## D-10 — Data Pipeline Flow (Stages 1-3)

> **Thesis placement:** Chapter 3, Section 3.2 — Data Pipeline  
> **Interpretation:** Detailed data flow through ingestion (3 paths), validation (5 checks), and transformation (4 steps). Shows how raw Garmin CSV data becomes model-ready windowed numpy arrays.  
> **Evidence:** `src/components/data_ingestion.py`, `src/preprocess_data.py`, `src/components/data_transformation.py`

```mermaid
flowchart TB
    subgraph Ingestion["Stage 1: Data Ingestion"]
        RAW["Raw Garmin CSVs<br/>(accelerometer, gyroscope, labels)"]
        P1["Path A: Single-file CSV"]
        P2["Path B: Multi-file merge<br/>(merge_asof 20ms tolerance)"]
        P3["Path C: Pre-merged data"]
        MAN["Manifest check<br/>(skip if already processed)"]
        RES["Resample to 50Hz"]
        OUT1["merged_sensor_data.csv"]
    end
    
    subgraph Validation["Stage 2: Data Validation"]
        V1["Schema check (required columns)"]
        V2["Bounds check (acc ∈ [-20g, +20g])"]
        V3["NaN fraction < threshold"]
        V4["Sensor coverage ≥ 4 of 6"]
        V5["Timestamp monotonicity"]
        VOUT{All pass?}
        WARN["⚠ Warning logged<br/>(does NOT halt pipeline)"]
        VPASS["Proceed"]
    end
    
    subgraph Transform["Stage 3: Data Transformation"]
        T1["UnitDetector<br/>(milliG → g conversion)"]
        T2["GravityRemover<br/>(Butterworth 0.3Hz highpass)"]
        T3["DomainCalibrator<br/>(per-session zero-mean)"]
        T4["SlidingWindowSegmenter<br/>(200 timesteps × 6 channels)"]
        OUT3["X.npy (N×200×6)<br/>y.npy (N,) labels"]
    end
    
    RAW --> P1 & P2 & P3
    P1 & P2 & P3 --> MAN --> RES --> OUT1
    OUT1 --> V1 --> V2 --> V3 --> V4 --> V5 --> VOUT
    VOUT -- "Fail" --> WARN --> VPASS
    VOUT -- "Pass" --> VPASS
    VPASS --> T1 --> T2 --> T3 --> T4 --> OUT3
```

---

## Rendering Notes for Thesis

1. **Mermaid → SVG/PDF:** Use `mmdc` CLI (`npm install -g @mermaid-js/mermaid-cli`) for batch rendering:
   ```bash
   mmdc -i diagram.mmd -o diagram.svg -t neutral --width 1200
   ```
2. **LaTeX integration:** Use `mermaid-filter` with pandoc, or render to PDF and include with `\includegraphics`.
3. **Color scheme:** Green (#c8e6c9) = operational; Orange (#ffe0b2) = placeholder/warning; Default = informational.
4. **Resolution:** Export at ≥ 300 DPI for print; SVG preferred for scalability.
5. **Existing PNGs:** 7 figures in `docs/figures/` — review for overlap with these Mermaid diagrams before thesis inclusion.

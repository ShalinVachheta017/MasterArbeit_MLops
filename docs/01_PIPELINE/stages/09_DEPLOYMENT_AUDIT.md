# Stage 8: Deployment & Audit Trail

> **⚠️ STATUS (2026-01-30): PARTIAL - Docker exists, NO CI/CD!**
> 
> **TODO:**
> - [ ] Create `.github/workflows/mlops.yml` (lint → test → build → deploy)
> - [ ] Create `tests/` folder with minimum 10 tests
> - [ ] Add pytest-cov for coverage reporting
> - [ ] Set up Grafana dashboard for monitoring
> - [ ] Configure manual deployment trigger (not auto on push)

**Pipeline Stage:** Deploy models safely, maintain audit logs  
**Input:** Trained models, validation results, deployment config  
**Output:** Deployed containers, audit logs, rollback capability

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │   MLflow    │────▶│   Docker    │────▶│  K8s/Edge   │               │
│  │  Registry   │     │   Build     │     │   Deploy    │               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│        │                    │                   │                       │
│        │                    │                   │                       │
│        ▼                    ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     AUDIT TRAIL                                  │   │
│  │  • Model version + hash                                         │   │
│  │  • Training data version (DVC)                                  │   │
│  │  • Config snapshot                                              │   │
│  │  • Validation metrics                                           │   │
│  │  • Deployment timestamp                                         │   │
│  │  • Deployer identity                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Container Structure

### Inference Container (docker/Dockerfile.inference)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/pretrained/ /app/models/
COPY models/normalized_baseline.json /app/models/
COPY src/run_inference.py /app/
COPY src/sensor_data_pipeline.py /app/

# Environment
ENV MODEL_PATH=/app/models/1DCNN_BiLSTM.h5
ENV BASELINE_PATH=/app/models/normalized_baseline.json
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import tensorflow; print('OK')" || exit 1

EXPOSE 8000

CMD ["python", "run_inference.py"]
```

### Build and Push

```bash
# Build image with version tag
docker build -f docker/Dockerfile.inference \
  -t har-inference:v1.2.0 \
  -t har-inference:latest .

# Push to registry
docker push registry.example.com/har-inference:v1.2.0
```

---

## Audit Trail Requirements

### Model Deployment Record

```json
{
  "deployment_id": "deploy-2026-01-15-001",
  "timestamp": "2026-01-15T10:30:00Z",
  "model": {
    "name": "1DCNN_BiLSTM",
    "version": "1.2.0",
    "mlflow_run_id": "abc123def456",
    "sha256": "a1b2c3d4e5f6...",
    "parameters": 499131
  },
  "training_data": {
    "dvc_version": "v1.0.0",
    "commit": "abc123",
    "n_samples": 450000,
    "n_users": 16
  },
  "validation": {
    "test_accuracy": 0.934,
    "macro_f1": 0.918,
    "passed_smoke_test": true
  },
  "config": {
    "window_size": 200,
    "sample_rate": 50,
    "n_classes": 11,
    "confidence_threshold": 0.70
  },
  "deployer": {
    "user": "oleh.pekh",
    "method": "CI/CD pipeline",
    "approved_by": "supervisor"
  },
  "environment": {
    "target": "production-cluster-01",
    "replicas": 3,
    "resources": {
      "cpu": "500m",
      "memory": "2Gi"
    }
  }
}
```

---

## Rollback Capability

### Version History

```
models/
├── v1.0.0/                    # Initial model
│   ├── model.h5
│   ├── config.json
│   └── deployment_record.json
│
├── v1.1.0/                    # First improvement
│   ├── model.h5
│   ├── config.json
│   └── deployment_record.json
│
├── v1.2.0/                    # Current production
│   ├── model.h5
│   ├── config.json
│   └── deployment_record.json
│
└── current -> v1.2.0          # Symlink to active version
```

### Rollback Procedure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ROLLBACK PROCEDURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DETECT ISSUE                                                        │
│     └── Critical alert OR manual decision                               │
│                                                                         │
│  2. IDENTIFY ROLLBACK TARGET                                            │
│     └── Usually previous version (v1.1.0 if current is v1.2.0)         │
│                                                                         │
│  3. VERIFY ROLLBACK TARGET                                              │
│     └── Check that target version has valid deployment record           │
│                                                                         │
│  4. EXECUTE ROLLBACK                                                    │
│     └── Update symlink/container tag                                    │
│     └── Restart inference service                                       │
│                                                                         │
│  5. VERIFY ROLLBACK SUCCESS                                             │
│     └── Run smoke tests                                                 │
│     └── Check monitoring metrics                                        │
│                                                                         │
│  6. LOG ROLLBACK                                                        │
│     └── Record in audit trail                                           │
│     └── Create incident report                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prometheus/Grafana Monitoring

### Metrics to Export

```python
# Prometheus metrics for inference service
from prometheus_client import Counter, Gauge, Histogram

# Request metrics
inference_requests_total = Counter(
    'har_inference_requests_total',
    'Total inference requests',
    ['status']  # success, error
)

inference_latency_seconds = Histogram(
    'har_inference_latency_seconds',
    'Inference latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Model metrics
model_confidence = Gauge(
    'har_model_confidence',
    'Mean prediction confidence (rolling window)'
)

model_entropy = Gauge(
    'har_model_entropy',
    'Mean prediction entropy (rolling window)'
)

drift_score = Gauge(
    'har_drift_score',
    'Current drift score',
    ['channel']  # Ax, Ay, Az, Gx, Gy, Gz
)

# Resource metrics
model_memory_bytes = Gauge(
    'har_model_memory_bytes',
    'Model memory usage'
)
```

### Grafana Dashboard Panels

| Panel | Metric | Purpose |
|-------|--------|---------|
| Request Rate | `rate(har_inference_requests_total[5m])` | Throughput |
| Latency P95 | `histogram_quantile(0.95, har_inference_latency_seconds)` | Performance |
| Confidence Trend | `har_model_confidence` | Model health |
| Drift per Channel | `har_drift_score{channel="*"}` | Data quality |
| Error Rate | `har_inference_requests_total{status="error"}` | Reliability |

---

## CI/CD Pipeline Integration

### Deployment Pipeline

```yaml
# .github/workflows/deploy.yml (conceptual)
name: Model Deployment

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Validate model version
        run: |
          # Check model exists in MLflow registry
          # Verify validation metrics passed
          
      - name: Build container
        run: |
          docker build -f docker/Dockerfile.inference \
            --build-arg MODEL_VERSION=${{ inputs.model_version }} \
            -t har-inference:${{ inputs.model_version }} .
            
      - name: Run smoke tests
        run: |
          docker run har-inference:${{ inputs.model_version }} \
            python scripts/inference_smoke.py
            
      - name: Deploy to environment
        run: |
          kubectl set image deployment/har-inference \
            inference=har-inference:${{ inputs.model_version }}
            
      - name: Verify deployment
        run: |
          # Wait for rollout
          # Check health endpoints
          # Verify metrics
          
      - name: Create audit record
        run: |
          # Log deployment to audit trail
```

---

## What to Do Checklist

- [ ] Create Dockerfile for inference service
- [ ] Set up container registry
- [ ] Define deployment record JSON schema
- [ ] Implement version tracking system
- [ ] Create rollback procedures and test them
- [ ] Set up Prometheus metrics export
- [ ] Create Grafana dashboard
- [ ] Document CI/CD pipeline
- [ ] Create runbook for deployment/rollback

---

## Evidence from Papers

**[MLOps: A Taxonomy and a Methodology | PDF: papers/mlops_production/MLOps_A_Taxonomy_and_a_Methodology 2022.pdf]**
- Importance of audit trails
- Version control for models

**[From Development to Deployment: MLOps Monitoring | PDF: papers/mlops_production/From_Development_to_Deployment_An_Approach_to_MLOps_Monitoring_for_Machine_Learning_Model_Operationalization 2023.pdf]**
- Containerization best practices
- Monitoring integration requirements

**[Serverless Machine Learning Inference | PDF: papers/mlops_production/serverless_machine_learning_inference_with_container_based_model_2020.pdf]**
- Container-based deployment patterns
- Scaling considerations

---

## Improvement Suggestions for This Stage

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| **HIGH** | Add automatic rollback on critical alerts | Medium | Safety |
| **HIGH** | Create comprehensive Grafana dashboard | Medium | Visibility |
| **MEDIUM** | Implement canary deployments | High | Safe rollout |
| **MEDIUM** | Add model signature verification | Low | Security |
| **LOW** | Implement blue-green deployments | High | Zero-downtime |

---

**Previous Stage:** [08_ALERTING_RETRAINING.md](08_ALERTING_RETRAINING.md)  
**Next Stage:** [10_IMPROVEMENTS_ROADMAP.md](10_IMPROVEMENTS_ROADMAP.md)

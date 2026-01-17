# Building a Scalable Offline MLOps Pipeline with Trained Models

A comprehensive guide to building a self-hosted, cloud-free MLOps pipeline using open-source tools and on-premise infrastructure.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Implementation Guide](#implementation-guide)
5. [Deployment Patterns](#deployment-patterns)
6. [Monitoring & Scaling](#monitoring--scaling)
7. [Best Practices](#best-practices)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE MLOps PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   DATA      │    │  EXPERIMENT  │    │  MODEL REGISTRY │   │
│  │ MANAGEMENT  │    │  TRACKING    │    │  & VERSIONING   │   │
│  │   (DVC)     │    │   (MLflow)   │    │   (MLflow)      │   │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│         ▲                    ▲                      ▲            │
│         └────────────────────┴──────────────────────┘            │
│                              │                                   │
│                    ┌─────────▼──────────┐                       │
│                    │  LOCAL STORAGE     │                       │
│                    │  (MinIO / S3-Compat)                       │
│                    └──────────────────────┘                       │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                   COMPUTE ORCHESTRATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          KUBERNETES (K8s / Minikube)                      │  │
│  │                                                            │  │
│  │  ┌─────────────────┐         ┌──────────────────────┐   │  │
│  │  │ TRAINING JOBS   │         │ SERVING INFERENCE    │   │  │
│  │  │ (PyTorch/TF)    │         │ (FastAPI/MLServer)   │   │  │
│  │  └─────────────────┘         └──────────────────────┘   │  │
│  │                                                            │  │
│  │  ┌─────────────────┐         ┌──────────────────────┐   │  │
│  │  │ ORCHESTRATION   │         │ AUTOSCALING          │   │  │
│  │  │ (Airflow/Kube   │         │ (HPA / Custom)       │   │  │
│  │  │  Scheduler)     │         │                      │   │  │
│  │  └─────────────────┘         └──────────────────────┘   │  │
│  │                                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                  MONITORING & OBSERVABILITY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ PROMETHEUS  │    │   GRAFANA    │    │  CUSTOM LOGGING │   │
│  │ (Metrics)   │    │  (Dashboard) │    │  (ELK/Loki)     │   │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│         ▲                    ▲                      ▲            │
│         └────────────────────┴──────────────────────┘            │
│                              │                                   │
│                   ┌──────────▼──────────┐                       │
│                   │  ALERTING (Alerts)  │                       │
│                   └──────────────────────┘                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **No Cloud Dependency**: All components run on-premise or self-hosted
- **Scalability**: Horizontal scaling via Kubernetes with GPU support
- **Open-Source**: Minimal licensing costs, full control over tooling
- **Production-Ready**: Enterprise patterns for reliability and observability
- **Modular**: Loosely coupled components for flexibility

---

## Core Components

### 1. **Data Management** (DVC + MinIO)

**Why DVC?**
- Version datasets and models like Git
- Track pipelines and dependencies
- No data bloat in Git repositories
- Reproducibility across teams

**Why MinIO?**
- S3-compatible object storage
- Self-hosted alternative to AWS S3
- Supports versioning and lifecycle policies
- Lower latency for local networks

**Installation:**

```bash
# Install DVC
pip install dvc

# Initialize DVC in your project
dvc init

# Configure MinIO as remote storage
dvc remote add -d myremote s3://your-bucket
dvc remote modify myremote url s3://your-bucket
dvc remote modify myremote endpointurl http://minio:9000
dvc remote modify myremote access_key_id minioadmin
dvc remote modify myremote secret_access_key minioadmin
```

**Usage:**

```bash
# Track large files and datasets
dvc add data/raw_dataset.csv

# Push to MinIO
dvc push

# Pull on another machine
dvc pull

# Create data pipeline
dvc stage add -n preprocess \
  -d data/raw.csv \
  -o data/processed.csv \
  python scripts/preprocess.py
```

---

### 2. **Experiment Tracking & Model Registry** (MLflow)

**Why MLflow?**
- Pure Python, works offline without server
- Tracks metrics, parameters, artifacts
- Built-in model registry for versioning
- Easy integration with training loops

**Installation & Setup:**

```bash
pip install mlflow

# Start MLflow server (runs on localhost:5000)
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns
```

**Training Integration:**

```python
import mlflow
import mlflow.pytorch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configure MLflow backend
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("time_series_forecasting")

# Start a run
with mlflow.start_run(run_name="lstm_v1"):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("config.yaml")

# Register model to registry
mlflow.register_model("runs:/<run_id>/model", "iris_classifier")
```

**Model Serving:**

```bash
# Serve model locally
mlflow models serve -m models:/iris_classifier/1 -p 5001 --no-conda

# For production: use MLServer
mlflow models serve -m models:/iris_classifier/1 \
                    -p 5001 \
                    --enable-mlserver
```

---

### 3. **Containerization** (Docker)

**Dockerfile for Training Job:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PYTHONUNBUFFERED=1

# Training entrypoint
ENTRYPOINT ["python", "train.py"]
```

**Dockerfile for Inference Server:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow mlserver

# Copy inference code
COPY serve.py .
COPY models/ models/

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PORT=5001

EXPOSE 5001

ENTRYPOINT ["mlflow", "models", "serve", "-m", "models:/model/1", "-p", "5001", "--enable-mlserver"]
```

**Build & Push to Local Registry:**

```bash
# Build image
docker build -t localhost:5000/mlops/training:v1.0 -f Dockerfile.train .
docker build -t localhost:5000/mlops/inference:v1.0 -f Dockerfile.serve .

# Push to local Docker registry
docker push localhost:5000/mlops/training:v1.0
docker push localhost:5000/mlops/inference:v1.0

# Verify images
docker images | grep mlops
```

---

### 4. **Orchestration** (Kubernetes + Kubeflow)

**Why Kubernetes?**
- Declarative infrastructure
- Horizontal & vertical scaling
- Resource management (CPU, GPU, Memory)
- Self-healing and high availability
- Native support for distributed training

**Local Setup with Minikube:**

```bash
# Install Minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster (with GPU support if available)
minikube start --cpus 8 --memory 16384 --gpus all --driver docker

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress
```

**Training Job Manifest (Kubernetes):**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
  namespace: mlops
spec:
  backoffLimit: 2
  template:
    spec:
      containers:
      - name: training
        image: localhost:5000/mlops/training:v1.0
        imagePullPolicy: Always
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"  # Request 1 GPU
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        - name: DATA_PATH
          value: "/mnt/data"
        volumeMounts:
        - name: data-storage
          mountPath: /mnt/data
        - name: models-storage
          mountPath: /mnt/models
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
      - name: models-storage
        persistentVolumeClaim:
          claimName: models-pvc
      restartPolicy: Never
```

**Inference Deployment Manifest:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  namespace: mlops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: inference
        image: localhost:5000/mlops/inference:v1.0
        imagePullPolicy: Always
        ports:
        - containerPort: 5001
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5001
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  namespace: mlops
spec:
  selector:
    app: inference
  ports:
  - protocol: TCP
    port: 5001
    targetPort: 5001
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: mlops
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Deploy to Kubernetes:**

```bash
# Create namespace
kubectl create namespace mlops

# Create persistent volumes
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: data-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/mnt/data"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: mlops
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
EOF

# Deploy training job
kubectl apply -f training-job.yaml

# Deploy inference server
kubectl apply -f inference-deployment.yaml

# Check status
kubectl get pods -n mlops
kubectl logs -n mlops -f deployment/inference-server
```

---

### 5. **Pipeline Orchestration** (Apache Airflow)

**Installation:**

```bash
pip install apache-airflow apache-airflow-providers-kubernetes

# Initialize Airflow
airflow db init
```

**DAG Definition for MLOps Pipeline:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta
import mlflow

default_args = {
    'owner': 'mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlops_training_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for time-series forecasting',
    schedule_interval='0 0 * * 0',  # Weekly
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

def data_validation(**context):
    """Validate data quality before training"""
    print("Validating data integrity...")
    # Add your data validation logic here
    pass

def train_model(**context):
    """Train model with MLflow tracking"""
    import subprocess
    result = subprocess.run(
        ["kubectl", "apply", "-f", "training-job.yaml"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return result.returncode

def evaluate_model(**context):
    """Evaluate model metrics"""
    mlflow.set_tracking_uri("http://localhost:5000")
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs:
        latest_run = runs[0]
        accuracy = latest_run.data.metrics.get("accuracy", 0)
        if accuracy > 0.95:
            return True
    return False

def deploy_model(**context):
    """Deploy model if evaluation passes"""
    print("Deploying model to production...")
    import subprocess
    subprocess.run(["kubectl", "apply", "-f", "inference-deployment.yaml"])

# Define tasks
task_validate = PythonOperator(
    task_id='validate_data',
    python_callable=data_validation,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

task_deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Define dependencies
task_validate >> task_train >> task_evaluate >> task_deploy
```

---

### 6. **Monitoring & Observability** (Prometheus + Grafana)

**Prometheus Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt

  - job_name: 'mlflow'
    static_configs:
      - targets: ['localhost:5000']

  - job_name: 'inference-server'
    static_configs:
      - targets: ['inference-service:5001']
```

**Deploy Prometheus + Grafana:**

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (includes Prometheus, Grafana, AlertManager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port-forward to access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Access at http://localhost:3000
```

**Custom Grafana Dashboard for Model Metrics:**

```json
{
  "dashboard": {
    "title": "MLOps Model Performance",
    "panels": [
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "mlflow_model_accuracy"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Inference Latency (p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_percent"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Data Drift Score",
        "targets": [
          {
            "expr": "model_data_drift_score"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## Infrastructure Setup

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU Cores | 8 | 16-32 | More for parallel training |
| RAM | 32 GB | 64-128 GB | GPU memory + system memory |
| GPU | None | 1-4 GPUs | NVIDIA V100/A100 for production |
| Storage | 1 TB | 5-10 TB | For datasets + models + artifacts |
| Network | 1 Gbps | 10 Gbps | For distributed training |

### Docker Registry Setup

```bash
# Run local Docker registry
docker run -d -p 5000:5000 --name registry registry:2

# Enable insecure registry in Docker daemon
cat > /etc/docker/daemon.json <<EOF
{
  "insecure-registries": ["localhost:5000"]
}
EOF

docker systemctl restart docker
```

### Network & Storage

**NFS Setup for Shared Storage:**

```bash
# On NFS server
sudo apt-get install nfs-kernel-server
sudo mkdir -p /mnt/nfs_storage
sudo chown nobody:nogroup /mnt/nfs_storage

# Edit /etc/exports
/mnt/nfs_storage *(rw,sync,no_subtree_check,no_root_squash)

sudo systemctl restart nfs-kernel-server

# On Kubernetes nodes
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 1Ti
  accessModes:
    - ReadWriteMany
  nfs:
    server: nfs-server-ip
    path: "/mnt/nfs_storage"
EOF
```

---

## Implementation Guide

### Phase 1: Local Development (Weeks 1-2)

```bash
# 1. Set up project structure
mlops-project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess.py
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── inference.py
├── tests/
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── training-job.yaml
│   └── inference-deployment.yaml
├── airflow/
│   └── dags/
│       └── mlops_pipeline.py
├── dvc.yaml
├── requirements.txt
└── docker-compose.yml

# 2. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Initialize DVC and MLflow locally
dvc init
mlflow ui

# 4. Train and log model
python scripts/train.py

# 5. Verify MLflow tracking
# Visit http://localhost:5000
```

### Phase 2: Containerization (Weeks 2-3)

```bash
# 1. Build training container
docker build -t mlops/training:v1.0 -f docker/Dockerfile.train .

# 2. Build inference container
docker build -t mlops/inference:v1.0 -f docker/Dockerfile.serve .

# 3. Test locally with Docker Compose
docker-compose up -d

# 4. Verify inference endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### Phase 3: Kubernetes Deployment (Weeks 3-4)

```bash
# 1. Start Minikube
minikube start --cpus 8 --memory 16384

# 2. Deploy MLflow server
kubectl apply -f k8s/mlflow-deployment.yaml

# 3. Deploy training job
kubectl apply -f k8s/training-job.yaml

# 4. Monitor job
kubectl logs -f job/model-training-job

# 5. Deploy inference service
kubectl apply -f k8s/inference-deployment.yaml

# 6. Test inference
kubectl port-forward svc/inference-service 8000:5001
curl http://localhost:8000/predict
```

### Phase 4: Production Deployment (Weeks 4+)

```bash
# 1. Set up monitoring
helm install prometheus prometheus-community/kube-prometheus-stack

# 2. Deploy Airflow
helm install airflow apache-airflow/airflow

# 3. Configure alerting
kubectl apply -f k8s/alertmanager-config.yaml

# 4. Enable CI/CD with GitOps (e.g., ArgoCD)
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

---

## Deployment Patterns

### 1. **Blue-Green Deployment**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: inference
    version: blue  # Switch between "blue" and "green"
  ports:
  - port: 5001
    targetPort: 5001
```

### 2. **Canary Deployment**

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: inference-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  progressDeadlineSeconds: 60
  service:
    port: 5001
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
```

### 3. **A/B Testing**

```python
# Custom routing logic in FastAPI
from fastapi import FastAPI
import random

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    if random.random() < 0.5:
        # Route to model A
        model = load_model("models:/model_a/1")
    else:
        # Route to model B
        model = load_model("models:/model_b/1")
    
    return model.predict(data)
```

---

## Monitoring & Scaling

### Model Performance Monitoring

```python
# Add to inference server
from prometheus_client import Counter, Histogram, Gauge
import time

prediction_counter = Counter('predictions_total', 'Total predictions', ['model_version'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

@app.post("/predict")
async def predict(data: dict):
    start_time = time.time()
    predictions = model.predict(data)
    
    latency = time.time() - start_time
    prediction_counter.labels(model_version="v1.0").inc()
    prediction_latency.observe(latency)
    
    return predictions
```

### Data Drift Detection

```python
import pandas as pd
from scipy import stats

def detect_drift(reference_data, current_data, threshold=0.05):
    """Detect statistical drift using KS test"""
    drifts = {}
    for column in reference_data.columns:
        ks_stat, p_value = stats.ks_2samp(reference_data[column], current_data[column])
        if p_value < threshold:
            drifts[column] = {'ks_stat': ks_stat, 'p_value': p_value}
    
    return drifts

# Log drift metrics to Prometheus/MLflow
if detect_drift(ref_data, curr_data):
    mlflow.log_metric("data_drift_detected", 1)
```

### Horizontal Auto-Scaling Configuration

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_latency_p95
      target:
        type: AverageValue
        averageValue: "1000m"
```

---

## Best Practices

### 1. **Version Everything**

```bash
# Model versioning
mlflow.register_model("runs:/<run_id>/model", "time_series_model")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage("time_series_model", 1, "Production")

# Data versioning
dvc add data/processed.csv
git add data/processed.csv.dvc
git commit -m "Update dataset to v2.0"
```

### 2. **Reproducibility**

```python
# Log everything for reproducibility
mlflow.log_param("seed", 42)
mlflow.log_param("train_test_split", 0.8)
mlflow.log_artifact("requirements.txt")
mlflow.log_artifact("config.yaml")

# Save training code snapshot
import shutil
shutil.copytree("src", "artifacts/src")
mlflow.log_artifact("artifacts/src")
```

### 3. **Testing**

```bash
# Unit tests
pytest tests/test_model.py -v

# Integration tests
pytest tests/test_inference.py -v

# E2E tests
pytest tests/test_pipeline.py -v

# Model validation
python scripts/validate_model.py
```

### 4. **Documentation**

```markdown
# Model Card
- **Model Name**: LSTM Time Series Forecaster
- **Version**: 1.0
- **Training Data**: Wearable sensor data (Jan-Dec 2024)
- **Performance**: MAE=0.15, RMSE=0.22
- **Limitations**: Only trained on healthy subjects
- **Bias**: May underperform for females <30 years
```

### 5. **Resource Management**

```yaml
# Always set resource limits
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
```

### 6. **Security**

```yaml
# Run containers as non-root
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true

# Use network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-ingress
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: mlops
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Pod OOMKilled | Insufficient memory | Increase memory limit in deployment |
| GPU not allocated | NVIDIA plugin missing | `kubectl apply -f nvidia-device-plugin.yaml` |
| Model not found | Wrong artifact path | Check MLflow artifact location |
| Slow inference | High latency network | Use local caching, batch inference |
| Data drift not detected | Wrong baseline | Retrain with recent data |

---

## Cost Analysis (On-Premise vs Cloud)

### Year 1 Costs

| Component | On-Premise | AWS | GCP | Savings |
|-----------|-----------|-----|-----|---------|
| Hardware | $30k | - | - | - |
| Licensing | $0 | - | - | - |
| Compute | $0 | $50k | $45k | $30k-45k/year |
| Storage | $2k | $12k | $10k | $8k-10k/year |
| **Total** | **$32k** | **$62k** | **$55k** | **23k-30k** |

### Break-Even Point: ~1.5 years

After 2 years, on-premise saves ~$50k+ while maintaining full control.

---

## Next Steps

1. **Week 1-2**: Set up local environment, train first model
2. **Week 2-3**: Containerize and test with Docker
3. **Week 3-4**: Deploy to Kubernetes with Minikube
4. **Week 4+**: Scale to production, add monitoring
5. **Month 2+**: Implement CI/CD, automated retraining

---

## Resources

- **MLflow Docs**: https://mlflow.org/docs
- **Kubernetes Docs**: https://kubernetes.io/docs
- **Kubeflow**: https://www.kubeflow.org
- **DVC Guide**: https://dvc.org/doc
- **ArgoCD**: https://argoproj.github.io/cd
- **PyCon DE 2024**: Self-Hosted MLOps with Kubernetes (Video)

---

## Community & Support

- **MLOps Community**: https://mlops.community
- **Kubernetes Slack**: kubernetes.slack.com
- **GitHub Issues**: Ask in tool repositories
- **Stack Overflow**: Tag: mlops, kubernetes, mlflow
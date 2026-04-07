# Real-Time Fraud Detection MLOps Pipeline

A production-grade, end-to-end MLOps system for detecting fraudulent transactions in real time. The platform ingests transaction events, enriches them with historical and real-time features from Feast, scores them with an XGBoost classifier served via FastAPI and KServe, and returns an **ALLOW**, **REVIEW**, or **BLOCK** decision -- all within a 50 ms p99 latency SLA.

## Why This Matters

- **$32 billion** in card fraud losses globally each year. Catching fraud early saves real money.
- **False positives cost revenue.** A legitimate customer blocked at checkout is a lost sale and a damaged relationship. This system optimises for high recall while maintaining precision >= 0.5, using Bayesian hyperparameter search (Optuna) with a custom objective that penalises precision drops.
- **Manual review does not scale.** The three-tier decision system (ALLOW / REVIEW / BLOCK) routes only ambiguous cases to human analysts, reducing review queue volume by 80%+ compared to binary models.
- **Silent model degradation is dangerous.** Grafana dashboards, Prometheus alerts, and Evidently drift detection ensure you know the moment model quality degrades -- before fraud losses spike.

## Architecture

```
                                    +---------------------+
                                    |   MLflow Registry   |
                                    |  (model artefacts,  |
                                    |   experiments,      |
                                    |   model versions)   |
                                    +----------+----------+
                                               |
                                       model loading
                                               |
+----------------+     +------------------+    |    +--------------------+     +------------------+
|  Transaction   |     |   Enrichment     |    v    |   Fraud Scoring    |     |    Decision       |
|  Event         +---->+   Service        +-------->+   Model            +---->+    Engine          |
|  (Kafka /      |     |   (FastAPI)      |         |   (FastAPI/KServe) |     |                   |
|   Event Hub)   |     |                  |         |                    |     |  ALLOW  (score<0.7)|
+----------------+     |  - Feast lookup  |         |  - XGBoost predict |     |  REVIEW (0.7-0.9) |
                       |  - velocity calc |         |  - warm-up on boot |     |  BLOCK  (score>0.9)|
                       |  - derived feats |         |  - audit trail log |     +--------+----------+
                       +--------+---------+         +--------+-----------+              |
                                |                            |                          v
                       +--------v---------+         +--------v-----------+     +--------+----------+
                       |   Redis          |         |   Prometheus       |     |   Downstream       |
                       |   (Feast online  |         |   Metrics          |     |   Systems           |
                       |    store +       |         |   /metrics endpoint|     |  (payment gateway,  |
                       |    velocity)     |         +--------+-----------+     |   case management)   |
                       +------------------+                  |                 +--------------------+
                                                             v
                                                    +--------+-----------+
                                                    |   Grafana          |
                                                    |   Dashboards       |
                                                    |   - fraud rate     |
                                                    |   - false positive |
                                                    |   - model latency  |
                                                    |   - drift detection|
                                                    +--------------------+
```

## Directory Structure

```
fraud-detection-mlops/
|
|-- data/
|   |-- generate_transactions.py   # Synthetic data generator (~200k txns, ~2% fraud rate)
|   |-- fraud_patterns.py          # Fraud pattern injection (velocity, geo mismatch, etc.)
|
|-- features/
|   |-- feast/
|   |   |-- feature_store.yaml     # Feast project configuration
|   |   |-- definitions.py         # Feature view and entity definitions
|   |-- enrichment_service/
|       |-- app.py                 # FastAPI enrichment service (Feast + real-time features)
|       |-- Dockerfile             # Container image for enrichment service
|
|-- training/
|   |-- train.py                   # XGBoost training with Optuna HPO (50 trials default)
|   |-- evaluate.py                # Model evaluation utilities
|   |-- Dockerfile                 # Training container image
|   |-- pipeline/
|       |-- training_pipeline.yaml # Kubeflow / Argo training pipeline definition
|
|-- serving/
|   |-- predict.py                 # FastAPI scoring service (single + batch endpoints)
|   |-- Dockerfile                 # Scoring container image
|   |-- inference_service.yaml     # KServe InferenceService manifest
|   |-- load_test/
|       |-- locustfile.py          # Locust load test (target: 1000 req/s, p99 < 50ms)
|
|-- deployment/
|   |-- helm/
|   |   |-- fraud-detection/
|   |       |-- Chart.yaml         # Helm chart metadata (v1.0.0)
|   |       |-- values.yaml        # Default values (replicas, thresholds, autoscaling)
|   |       |-- templates/
|   |           |-- deployment-enrichment.yaml
|   |           |-- deployment-scoring.yaml
|   |           |-- service.yaml
|   |           |-- hpa.yaml       # HorizontalPodAutoscaler
|   |           |-- _helpers.tpl
|   |-- shadow-deployment/
|   |   |-- shadow-mode.yaml       # Istio VirtualService for traffic mirroring
|   |-- Jenkinsfile                # CI/CD pipeline
|
|-- monitoring/
|   |-- dashboards/
|   |   |-- fraud_rate.json        # Grafana dashboard: fraud detection rate over time
|   |   |-- false_positive_rate.json  # Grafana dashboard: FP rate and trends
|   |   |-- model_latency.json    # Grafana dashboard: p50/p95/p99 scoring latency
|   |   |-- drift_detection.json  # Grafana dashboard: feature and prediction drift
|   |-- alerts/
|       |-- block_rate_spike.yaml  # Alert: block rate > 50% above 24hr average
|       |-- latency_breach.yaml    # Alert: p99 latency > 50ms for 5 minutes
|       |-- false_positive_spike.yaml  # Alert: FP rate > 10% for 1 hour
|
|-- tests/
|   |-- conftest.py                # Shared fixtures (mock Feast, mock model, TestClients)
|   |-- test_feature_enrichment.py # Enrichment service unit + integration tests
|   |-- test_model_accuracy.py     # Model quality assertions (precision, recall, F1)
|   |-- test_latency_sla.py        # Latency SLA smoke tests
|
|-- runbooks/
|   |-- model_rollback.md          # Step-by-step rollback procedure
|   |-- incident_high_block_rate.md  # Incident response for block rate spikes
|
|-- requirements.txt               # Python dependencies
|-- README.md                      # This file
```

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime for training and serving |
| Docker | 24+ | Container builds |
| kubectl | 1.28+ | Kubernetes cluster management |
| Helm | 3.14+ | Kubernetes deployment packaging |
| Redis | 7+ | Feast online store and velocity tracking |
| PostgreSQL | 15+ | MLflow backend store |
| MLflow | 2.14+ | Experiment tracking and model registry |
| Feast | 0.40+ | Feature store (online + offline) |

Optional:
- **Istio** 1.20+ for shadow deployments (traffic mirroring)
- **KServe** 0.12+ for autoscaling inference (alternative to plain Helm deployment)
- **Locust** 2.28+ for load testing
- **Grafana** 10+ and **Prometheus** 2.50+ for monitoring

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
cd data
python generate_transactions.py \
    --output-dir ./output \
    --n-transactions 200000 \
    --fraud-rate 0.02 \
    --seed 42
```

This produces `output/transactions.parquet` and `output/transactions.csv` with ~200k transactions containing realistic fraud patterns (velocity attacks, geographic mismatches, amount anomalies, account takeovers, and card testing).

### 3. Start MLflow Tracking Server

```bash
mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### 4. Train the Model

```bash
cd training
python train.py \
    --data-path ../data/output/transactions.parquet \
    --mlflow-uri http://localhost:5000 \
    --n-trials 50 \
    --experiment-name fraud-detection
```

Training performs:
1. Feature engineering (temporal, geographic match flags, log-transform amounts, label encoding)
2. SMOTE oversampling on the training set (sampling_strategy=0.3) to address the 2% fraud rate
3. Bayesian hyperparameter optimisation via Optuna (50 trials, 5-fold stratified CV)
4. Threshold tuning to maximise recall while keeping precision >= 0.5
5. Logging of all metrics, confusion matrix, ROC, PR curves, and feature importance to MLflow
6. Model registration in the MLflow Model Registry as `fraud-detection-xgb`

### 5. Run the Enrichment Service Locally

```bash
cd features/enrichment_service
REDIS_URL=redis://localhost:6379 \
FEAST_REPO_PATH=../feast \
python app.py
```

The service starts on port 8081 with endpoints:
- `POST /enrich` -- enrich a single transaction
- `POST /enrich/batch` -- enrich up to 100 transactions
- `GET /health` -- liveness probe
- `GET /metrics` -- Prometheus metrics

### 6. Run the Scoring Service Locally

```bash
cd serving
MLFLOW_TRACKING_URI=http://localhost:5000 \
MODEL_NAME=fraud-detection-xgb \
MODEL_VERSION=latest \
REVIEW_THRESHOLD=0.7 \
BLOCK_THRESHOLD=0.9 \
python predict.py
```

The service starts on port 8080. Test it:

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_test_001",
    "user_id": "user_42",
    "amount": 249.99,
    "merchant_id": "merch_1234",
    "merchant_category": "electronics",
    "card_type": "visa",
    "device_type": "mobile",
    "ip_country": "US",
    "billing_country": "US",
    "shipping_country": "US"
  }'
```

Response:

```json
{
  "transaction_id": "txn_test_001",
  "fraud_score": 0.023451,
  "action": "ALLOW",
  "confidence": 0.9765,
  "threshold_used": 0.7,
  "model_version": "latest",
  "scored_at": "2025-06-15T14:32:00.123456+00:00"
}
```

## Deployment to Kubernetes with Helm

### Build and Push Container Images

```bash
# Enrichment service
docker build -t gcr.io/fraud-detection-mlops/enrichment:latest \
    -f features/enrichment_service/Dockerfile .
docker push gcr.io/fraud-detection-mlops/enrichment:latest

# Scoring service
docker build -t gcr.io/fraud-detection-mlops/scoring:latest \
    -f serving/Dockerfile .
docker push gcr.io/fraud-detection-mlops/scoring:latest
```

### Deploy with Helm

```bash
# Default deployment (2 enrichment replicas, 3 scoring replicas, autoscaling enabled)
helm install fraud-detection deployment/helm/fraud-detection \
    --namespace fraud-production \
    --create-namespace

# With custom values
helm install fraud-detection deployment/helm/fraud-detection \
    --namespace fraud-production \
    -f deployment/helm/fraud-detection/values-prod.yaml \
    --set scoring.model.version=5 \
    --set scoring.thresholds.review=0.65 \
    --set scoring.thresholds.block=0.85
```

Key configuration in `values.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scoring.replicaCount` | 3 | Initial scoring service replicas |
| `scoring.autoscaling.maxReplicas` | 20 | Max scoring replicas under load |
| `scoring.thresholds.review` | 0.7 | Score threshold for REVIEW action |
| `scoring.thresholds.block` | 0.9 | Score threshold for BLOCK action |
| `scoring.model.version` | latest | MLflow model version to serve |
| `enrichment.feast.redisUrl` | redis://redis-master:6379 | Feast online store connection |

### Deploy with KServe (Alternative)

```bash
kubectl apply -f serving/inference_service.yaml -n fraud-production
```

This deploys the scoring service with KServe's autoscaling (concurrency-based, target: 10 concurrent requests per pod, min 2 / max 20 replicas) and built-in canary traffic splitting.

## Shadow Deployment Guide

Shadow (mirror) deployment lets you validate a new model version against live production traffic without affecting real outcomes. The candidate model receives copies of every request, but its responses are discarded by the Istio Envoy sidecar.

### Step 1: Deploy the Candidate

Update the `MODEL_VERSION` in `deployment/shadow-deployment/shadow-mode.yaml` and apply:

```bash
# Edit the candidate version
sed -i 's/REPLACE_WITH_CANDIDATE_VERSION/6/' \
    deployment/shadow-deployment/shadow-mode.yaml

# Apply the shadow deployment (creates candidate Service, Deployment, and VirtualService)
kubectl apply -f deployment/shadow-deployment/shadow-mode.yaml \
    -n fraud-production
```

### Step 2: Monitor Shadow Metrics

Both production and candidate models emit Prometheus metrics. Compare them side-by-side in Grafana:

- **Fraud score distribution:** Do the candidate's scores follow a similar distribution?
- **Block/review/allow rates:** Is the candidate more aggressive or lenient?
- **Latency:** Does the candidate meet the < 50ms p99 SLA?

Let the shadow run for at least **1 week** to capture weekly transaction patterns.

### Step 3: Promote or Discard

If shadow metrics are acceptable, proceed to a canary rollout. Otherwise, clean up:

```bash
kubectl delete virtualservice fraud-scoring-shadow -n fraud-production
kubectl delete deployment fraud-scoring-candidate -n fraud-production
kubectl delete service fraud-scoring-candidate -n fraud-production
```

## Monitoring Dashboards

Four pre-built Grafana dashboards are provided in `monitoring/dashboards/`:

| Dashboard | File | Key Panels |
|-----------|------|------------|
| **Fraud Rate** | `fraud_rate.json` | Fraud detection rate over time, block/review/allow breakdown, score distribution histogram |
| **False Positive Rate** | `false_positive_rate.json` | FP rate trend, precision over time, legitimate transactions blocked |
| **Model Latency** | `model_latency.json` | p50/p95/p99 scoring latency, latency heatmap, request throughput (req/s) |
| **Drift Detection** | `drift_detection.json` | Feature distribution drift (Evidently), prediction drift, data quality alerts |

Import them into Grafana:

```bash
for dashboard in monitoring/dashboards/*.json; do
    curl -X POST http://grafana:3000/api/dashboards/db \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -d "{\"dashboard\": $(cat $dashboard), \"overwrite\": true}"
done
```

### Alerts

Three Prometheus alerting rules are defined in `monitoring/alerts/`:

- **block_rate_spike.yaml** -- Fires when the block rate exceeds 50% above the 24-hour moving average for 15 minutes
- **latency_breach.yaml** -- Fires when p99 scoring latency exceeds 50ms for 5 consecutive minutes
- **false_positive_spike.yaml** -- Fires when the false positive rate exceeds 10% for 1 hour

## Load Testing with Locust

The Locust load test simulates realistic production traffic:

- 85% normal transactions, 10% suspicious transactions, 5% batch requests
- Target: **1000 req/s** sustained throughput
- SLA gates: **p99 < 50ms**, **error rate < 0.1%**

### Run Interactively

```bash
cd serving/load_test
locust -f locustfile.py --host http://localhost:8080
```

Open the Locust web UI at http://localhost:8089 to start the test.

### Run in CI/CD (Headless)

```bash
locust -f serving/load_test/locustfile.py \
    --host http://localhost:8080 \
    --headless \
    --users 200 \
    --spawn-rate 20 \
    --run-time 5m \
    --csv results/load_test
```

The test prints an SLA report on completion and exits with a non-zero code if any SLA check fails.

## Runbooks

Operational runbooks are in `runbooks/`:

| Runbook | When to Use |
|---------|------------|
| `model_rollback.md` | Any Prometheus alert fires, smoke tests fail, or business stakeholders report a surge in blocked legitimate transactions |
| `incident_high_block_rate.md` | Block rate spikes above 50% of the 24-hour average |

Rollback triggers include:
- `FraudBlockRateSpike` alert fires
- `FalsePositiveRateHigh` alert fires
- `FraudScoringLatencyHigh` alert fires
- Post-deployment smoke tests fail
- Shadow metrics show significant degradation

## Testing

### Run the Full Test Suite

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Individual Test Modules

```bash
# Feature enrichment service tests (unit + integration)
pytest tests/test_feature_enrichment.py -v

# Model accuracy assertions (precision, recall, F1, AUC thresholds)
pytest tests/test_model_accuracy.py -v

# Latency SLA smoke tests
pytest tests/test_latency_sla.py -v
```

### Linting and Type Checking

```bash
ruff check .
mypy . --strict
```

## Environment Variables

| Variable | Default | Service | Description |
|----------|---------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Scoring | MLflow server URI |
| `MODEL_NAME` | `fraud-detection-xgb` | Scoring | Registered model name |
| `MODEL_VERSION` | `latest` | Scoring | Model version to load |
| `REVIEW_THRESHOLD` | `0.7` | Scoring | Score >= this triggers REVIEW |
| `BLOCK_THRESHOLD` | `0.9` | Scoring | Score >= this triggers BLOCK |
| `MAX_BATCH_SIZE` | `200` | Scoring | Max transactions per batch request |
| `SERVICE_PORT` | `8080` / `8081` | Both | HTTP listen port |
| `FEAST_REPO_PATH` | `/app/features/feast` | Enrichment | Path to Feast feature repo |
| `REDIS_URL` | `redis://localhost:6379` | Enrichment | Redis connection for Feast + velocity |
| `LOG_LEVEL` | `INFO` | Both | Logging verbosity (DEBUG, INFO, WARNING) |

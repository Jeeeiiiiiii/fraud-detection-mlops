# Model Rollback Runbook

## When to Rollback

Initiate a rollback if **any** of the following conditions are met:

- **FraudBlockRateSpike** alert fires (block rate >50% above 24hr average for 15min)
- **FalsePositiveRateHigh** alert fires (FP rate > 10% for 1 hour)
- **FraudScoringLatencyHigh** alert fires (p99 > 50ms for 5 minutes)
- Post-deployment smoke tests fail
- Business stakeholders report a surge in customer complaints about blocked transactions
- Shadow metrics show significant degradation compared to production

## Prerequisites

- `kubectl` configured with cluster access to `fraud-production` namespace
- Helm v3 installed
- Access to the MLflow Model Registry UI or CLI
- Read access to Grafana dashboards

## Step 1: Identify the Current and Previous Model Versions

```bash
# Check what is currently deployed
kubectl get deployment fraud-detection-scoring -n fraud-production \
  -o jsonpath='{.spec.template.spec.containers[0].env}' | jq .

# Check the ConfigMap for current model version
kubectl get configmap fraud-detection-config -n fraud-production -o yaml

# List recent model versions in MLflow
# (via MLflow UI or CLI)
mlflow models list --name fraud-detection-xgb
```

Note the **current version** (failing) and the **previous version** (target).

## Step 2: Roll Back the Model Version

### Option A: Helm Rollback (preferred)

```bash
# List Helm release history
helm history fraud-detection -n fraud-production

# Roll back to the previous release
helm rollback fraud-detection <PREVIOUS_REVISION> -n fraud-production

# Wait for the rollout to complete
kubectl rollout status deployment/fraud-detection-scoring \
  -n fraud-production --timeout=300s
kubectl rollout status deployment/fraud-detection-enrichment \
  -n fraud-production --timeout=300s
```

### Option B: Update Model Version via ConfigMap

If only the model version changed (not the container image):

```bash
# Update the ConfigMap with the previous model version
kubectl patch configmap fraud-detection-config -n fraud-production \
  --type merge -p '{"data":{"model-version":"<PREVIOUS_VERSION>"}}'

# Restart the scoring pods to pick up the new version
kubectl rollout restart deployment/fraud-detection-scoring -n fraud-production

# Wait for rollout
kubectl rollout status deployment/fraud-detection-scoring \
  -n fraud-production --timeout=300s
```

### Option C: Kubernetes Deployment Rollback

If Helm history is not available:

```bash
# View deployment history
kubectl rollout history deployment/fraud-detection-scoring -n fraud-production

# Roll back to the previous revision
kubectl rollout undo deployment/fraud-detection-scoring -n fraud-production

# Wait for rollout
kubectl rollout status deployment/fraud-detection-scoring \
  -n fraud-production --timeout=300s
```

## Step 3: Verify the Rollback

```bash
# Verify pods are running
kubectl get pods -n fraud-production -l app.kubernetes.io/component=scoring

# Check health endpoint
kubectl exec -n fraud-production \
  $(kubectl get pod -n fraud-production -l app.kubernetes.io/component=scoring -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health | jq .

# Verify model version in health response
# Expected: model_version should match the rollback target version

# Score a test transaction
kubectl exec -n fraud-production \
  $(kubectl get pod -n fraud-production -l app.kubernetes.io/component=scoring -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "rollback_verify_001",
    "user_id": "user_1",
    "amount": 100.0,
    "merchant_id": "merch_1",
    "merchant_category": "grocery"
  }' | jq .
```

## Step 4: Monitor Post-Rollback Metrics

Open the Grafana dashboards and verify the following within 15 minutes:

1. **Fraud Rate Dashboard**: Block rate returns to normal range (1-5%)
2. **False Positive Dashboard**: FP rate drops below 10%
3. **Latency Dashboard**: p99 scoring latency < 50ms
4. **No new alerts firing** in the Prometheus Alertmanager

```bash
# Quick Prometheus query to check block rate
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -qO- 'http://localhost:9090/api/v1/query?query=sum(rate(block_rate[5m]))/(sum(rate(block_rate[5m]))+sum(rate(review_rate[5m]))+sum(rate(allow_rate[5m])))'
```

## Step 5: Clean Up Shadow/Canary Resources

If a shadow deployment or canary was active, clean it up:

```bash
# Remove shadow deployment
kubectl delete virtualservice fraud-scoring-shadow -n fraud-production --ignore-not-found
kubectl delete deployment fraud-scoring-candidate -n fraud-production --ignore-not-found
kubectl delete service fraud-scoring-candidate -n fraud-production --ignore-not-found
```

## Post-Rollback Checklist

- [ ] Alerts have cleared (no active critical/warning alerts)
- [ ] Block rate is within normal operating range (1-5%)
- [ ] False positive rate is below 5%
- [ ] p99 latency is below 50ms
- [ ] Health endpoints return `healthy` status
- [ ] Model version in health response matches the rollback target
- [ ] Customer support team has been notified of the resolution
- [ ] Incident ticket has been updated with rollback details
- [ ] A post-mortem has been scheduled (within 48 hours)

## Post-Mortem Investigation

After the rollback is confirmed stable, investigate the root cause:

1. **Compare model versions in MLflow**: Check evaluation metrics for the failing version vs the rollback target.
2. **Review training data**: Check for data quality issues, label noise, or distribution shift in the training set.
3. **Review feature pipeline**: Verify Feast feature freshness and correctness.
4. **Check evaluation report**: Review the `evaluation_report.json` artifact in MLflow for segment-level regressions.
5. **Review champion-challenger results**: If the champion-challenger gate passed but production failed, the evaluation dataset may not be representative.

## Escalation

If the rollback does not resolve the issue:

1. Page the ML Platform on-call: `@ml-platform-oncall`
2. Escalate to the Fraud Operations team: `#fraud-ops` Slack channel
3. If customer impact continues, consider temporarily raising the BLOCK_THRESHOLD to reduce false positives:

```bash
kubectl patch configmap fraud-detection-config -n fraud-production \
  --type merge -p '{"data":{"block-threshold":"0.95"}}'
kubectl rollout restart deployment/fraud-detection-scoring -n fraud-production
```

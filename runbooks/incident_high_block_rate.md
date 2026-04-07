# Incident Response: High Block Rate

## Alert Details

| Field | Value |
|-------|-------|
| **Alert Name** | FraudBlockRateSpike / FraudBlockRateHigh |
| **Severity** | Critical (spike >50%) / Warning (>20% absolute) |
| **Service** | fraud-detection-scoring |
| **Namespace** | fraud-production |
| **Dashboard** | [Fraud Rate](https://grafana.example.com/d/fraud-rate-overview/fraud-rate) |

## Impact

A high block rate means the fraud detection system is blocking an abnormal number of transactions. This can be caused by:

- **Model regression**: A newly deployed model has a higher false positive rate.
- **Feature pipeline failure**: Stale or corrupted features cause the model to over-predict fraud.
- **Genuine fraud attack**: A real surge in fraudulent transactions (less common as a sustained event).
- **Threshold misconfiguration**: Decision thresholds were changed incorrectly.

Customer impact depends on the root cause: if it is a model issue, legitimate customers are being blocked from completing purchases.

## Triage (First 5 Minutes)

### 1. Confirm the Alert

```bash
# Check current block rate
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -qO- 'http://localhost:9090/api/v1/query?query=sum(rate(block_rate[5m]))/(sum(rate(block_rate[5m]))+sum(rate(review_rate[5m]))+sum(rate(allow_rate[5m])))*100'
```

Open the [Fraud Rate Dashboard](https://grafana.example.com/d/fraud-rate-overview/fraud-rate) and verify:
- Is the block rate elevated across all segments or just specific categories?
- When did the spike begin?
- Is the review rate also elevated?

### 2. Check for Recent Deployments

```bash
# Check deployment history
kubectl rollout history deployment/fraud-detection-scoring -n fraud-production

# Check Helm release history
helm history fraud-detection -n fraud-production

# Check recent events
kubectl get events -n fraud-production --sort-by='.lastTimestamp' | tail -20
```

**Key question**: Was a new model version or container image deployed in the last few hours?

### 3. Check the Model Version

```bash
# Get current model version from the health endpoint
kubectl exec -n fraud-production \
  $(kubectl get pod -n fraud-production -l app.kubernetes.io/component=scoring -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health | jq .
```

Compare the deployed model version against MLflow:
- Was this model version properly evaluated via the champion-challenger pipeline?
- What were its evaluation metrics (recall, precision, FP rate)?

## Check Data Pipeline (Minutes 5-15)

### 4. Verify Feature Freshness

```bash
# Check Feast online store (Redis) connectivity
kubectl exec -n fraud-production \
  $(kubectl get pod -n fraud-production -l app.kubernetes.io/component=enrichment -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8081/health | jq .

# Check enrichment service logs for Feast errors
kubectl logs -n fraud-production \
  -l app.kubernetes.io/component=enrichment \
  --tail=100 | grep -i -E "feast|redis|error|warning"
```

If the Feast store is unreachable, the enrichment service falls back to default feature values, which may cause the model to behave differently.

### 5. Check Enrichment Service Errors

```bash
# Look for enrichment errors
kubectl logs -n fraud-production \
  -l app.kubernetes.io/component=enrichment \
  --tail=200 | grep -i error

# Check enrichment error metrics
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -qO- 'http://localhost:9090/api/v1/query?query=rate(enrichment_errors_total[5m])'
```

### 6. Check Scoring Service Logs

```bash
# Look for scoring anomalies
kubectl logs -n fraud-production \
  -l app.kubernetes.io/component=scoring \
  --tail=200 | grep -i -E "error|exception|fail"

# Check fraud score distribution -- is it skewed high?
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -qO- 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.50,sum(rate(fraud_score_bucket[5m]))by(le))'
```

## Decision: Rollback or Investigate Further

### If a recent model deployment caused the spike:

**ROLLBACK IMMEDIATELY.** Follow the [Model Rollback Runbook](./model_rollback.md).

```bash
# Quick rollback via Helm
helm rollback fraud-detection <PREVIOUS_REVISION> -n fraud-production

# Or via kubectl
kubectl rollout undo deployment/fraud-detection-scoring -n fraud-production
```

### If the feature pipeline is degraded:

1. Restart the enrichment service pods:

```bash
kubectl rollout restart deployment/fraud-detection-enrichment -n fraud-production
```

2. If Redis is down, check the Redis cluster:

```bash
kubectl get pods -n fraud-detection -l app=redis
kubectl logs -n fraud-detection -l app=redis --tail=50
```

3. If Feast materialization is stale, trigger a manual materialization:

```bash
kubectl create job --from=cronjob/feast-materialize feast-manual-$(date +%s) \
  -n ml-pipelines
```

### If this is a genuine fraud attack:

1. The elevated block rate is **expected behavior** -- the model is correctly detecting fraud.
2. Verify by checking the [False Positive Dashboard](https://grafana.example.com/d/fraud-fp-rate/false-positive-rate):
   - If the FP rate is normal (< 5%), the model is performing correctly.
   - If the FP rate is also elevated, it is likely a model issue, not a real attack.
3. Notify the Fraud Operations team via `#fraud-ops` Slack channel.

### If thresholds were misconfigured:

```bash
# Check current thresholds
kubectl get configmap fraud-detection-config -n fraud-production -o yaml

# Restore correct thresholds
kubectl patch configmap fraud-detection-config -n fraud-production \
  --type merge -p '{"data":{"review-threshold":"0.7","block-threshold":"0.9"}}'

# Restart scoring to pick up the change
kubectl rollout restart deployment/fraud-detection-scoring -n fraud-production
```

## Communication

### During the Incident

1. Post in `#fraud-detection-alerts`:
   ```
   INCIDENT: High fraud block rate detected.
   Impact: Elevated transaction blocking in production.
   Status: Investigating root cause.
   Lead: [Your Name]
   ```

2. If customer impact is confirmed, notify:
   - Customer Support lead via `#customer-support`
   - Product Manager via direct message
   - VP Engineering if block rate exceeds 50% for more than 30 minutes

### After Resolution

1. Update the incident channel:
   ```
   RESOLVED: High block rate incident resolved.
   Root cause: [brief description]
   Resolution: [rollback / fix / threshold adjustment]
   Duration: [X minutes]
   Customer impact: [estimated number of affected transactions]
   ```

2. Create a post-mortem ticket within 24 hours.
3. Schedule a post-mortem review within 48 hours.

## Post-Mortem Template

- **Incident Date/Time**: [timestamp]
- **Duration**: [minutes]
- **Severity**: [P1/P2]
- **Root Cause**: [description]
- **Detection**: How was the incident detected? (alert / customer report / manual)
- **Resolution**: What action resolved the issue?
- **Customer Impact**: Estimated number of affected transactions and customers
- **Revenue Impact**: Estimated blocked legitimate transaction volume
- **Timeline**: Minute-by-minute timeline of events
- **Action Items**:
  - [ ] [Specific improvement to prevent recurrence]
  - [ ] [Additional monitoring or alerting]
  - [ ] [Process changes]

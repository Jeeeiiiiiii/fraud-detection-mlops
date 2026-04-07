#!/bin/bash
# =============================================================================
# Deploy fraud-detection-mlops to the local kind cluster.
#
# Run AFTER payment-service/scripts/local-setup.sh (which creates the cluster).
#
# This deploys:
#   - Enrichment service (feature enrichment, port 8081)
#   - Scoring service (fraud scoring + /predict endpoint, port 8080)
#
# Both run in the ml-fraud namespace, matching the payment-service's
# FRAUD_SERVICE_URL configuration.
# =============================================================================

set -euo pipefail

CLUSTER_NAME="devops-test"
FRAUD_NS="ml-fraud"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$SCRIPT_DIR"

echo "=== Step 1: Verify kind cluster exists ==="
if ! kind get clusters 2>/dev/null | grep -q "$CLUSTER_NAME"; then
    echo "ERROR: Cluster '$CLUSTER_NAME' not found."
    echo "Run payment-service/scripts/local-setup.sh first to create it."
    exit 1
fi

echo "=== Step 2: Build Docker images ==="
docker build -t fraud-enrichment:latest -f features/enrichment_service/Dockerfile .
docker build -t fraud-scoring:latest -f serving/Dockerfile .

echo "=== Step 3: Load images into kind ==="
kind load docker-image fraud-enrichment:latest --name "$CLUSTER_NAME"
kind load docker-image fraud-scoring:latest --name "$CLUSTER_NAME"

echo "=== Step 4: Create namespace ==="
kubectl create namespace "$FRAUD_NS" --dry-run=client -o yaml | kubectl apply -f -

echo "=== Step 5: Deploy via Helm ==="
helm upgrade --install fraud-detection \
    deployment/helm/fraud-detection \
    -f deployment/helm/fraud-detection/values.yaml \
    -f deployment/helm/fraud-detection/values-dev.yaml \
    --namespace "$FRAUD_NS" \
    --wait --timeout 120s

echo "=== Step 6: Verify deployments ==="
echo ""
kubectl get pods -n "$FRAUD_NS"
echo ""
kubectl get svc -n "$FRAUD_NS"

echo ""
echo "=== Fraud Detection MLOps deployed ==="
echo ""
echo "Services running in namespace: $FRAUD_NS"
echo "  - Enrichment: fraud-detection-enrichment:8081"
echo "  - Scoring:    fraud-detection-scoring:8080"
echo ""
echo "To test the scoring endpoint directly:"
echo "  kubectl port-forward svc/fraud-detection-scoring -n $FRAUD_NS 8080:8080"
echo "  curl -X POST http://localhost:8080/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"amount\":150,\"merchant_category\":\"electronics\",\"customer_id\":\"c1\",\"billing_country\":\"US\",\"shipping_country\":\"US\"}'"
echo ""
echo "To test via payment-service (full flow):"
echo "  kubectl port-forward svc/payment-service -n payment 8000:8000"
echo "  curl -X POST http://localhost:8000/api/v1/transactions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"amount\":150,\"card_number_last4\":\"4242\",\"merchant_id\":\"m1\",\"merchant_category\":\"electronics\",\"customer_id\":\"c1\",\"customer_email\":\"test@test.com\"}'"
echo ""
echo "MCP agent queries to try:"
echo "  get_pods namespace:$FRAUD_NS"
echo "  get_pod_logs pod:fraud-detection-scoring-xxx namespace:$FRAUD_NS"
echo "  get_events namespace:$FRAUD_NS"

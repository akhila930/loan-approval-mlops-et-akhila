#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

APP_IMAGE="loan-approval-mlops:2.0"
GRAFANA_CONTAINER="grafana"

echo "=========================================="
echo "     Loan Approval MLOps Demo"
echo "=========================================="
echo ""

# --------------------------------------------------
# 1. Check required commands
# --------------------------------------------------

echo "[1/9] Checking required tools..."

for cmd in docker minikube kubectl curl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd is not available."
        exit 1
    fi
done

echo "✓ Docker"
echo "✓ Minikube"
echo "✓ kubectl"
echo "✓ curl"

# --------------------------------------------------
# 2. Start Minikube
# --------------------------------------------------

echo ""
echo "[2/9] Checking Minikube..."

MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || true)

if [ "$MINIKUBE_STATUS" != "Running" ]; then
    echo "Starting Minikube..."
    minikube start --driver=docker
else
    echo "✓ Minikube already running."
fi

# --------------------------------------------------
# 3. Verify Kubernetes node
# --------------------------------------------------

echo ""
echo "[3/9] Checking Kubernetes node..."

kubectl wait --for=condition=Ready node/minikube --timeout=120s

echo "✓ Kubernetes node is Ready."

# --------------------------------------------------
# 4. Load application image
# --------------------------------------------------

echo ""
echo "[4/9] Checking application image..."

if ! minikube image ls | grep -q "loan-approval-mlops:2.0"; then
    echo "Loading $APP_IMAGE into Minikube..."
    minikube image load "$APP_IMAGE"
else
    echo "✓ $APP_IMAGE already loaded."
fi

# Prometheus image
if ! minikube image ls | grep -q "prom/prometheus:latest"; then
    echo "Loading Prometheus image into Minikube..."
    minikube image load prom/prometheus:latest
else
    echo "✓ Prometheus image already loaded."
fi

# --------------------------------------------------
# 5. Deploy Kubernetes resources
# --------------------------------------------------

echo ""
echo "[5/9] Deploying Kubernetes resources..."

kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/prometheus.yaml

echo "✓ Kubernetes resources applied."

# --------------------------------------------------
# 6. Wait for deployments
# --------------------------------------------------

echo ""
echo "[6/9] Waiting for deployments..."

kubectl rollout status deployment/loan-approval-api --timeout=120s
kubectl rollout status deployment/prometheus --timeout=120s

echo "✓ Loan Approval API ready."
echo "✓ Prometheus ready."

# --------------------------------------------------
# 7. Start Prometheus port forwarding
# --------------------------------------------------

echo ""
echo "[7/9] Starting Prometheus access..."

if pgrep -f "kubectl port-forward svc/prometheus 9090:9090" >/dev/null 2>&1; then
    echo "✓ Prometheus port-forward already running."
else
    nohup kubectl port-forward svc/prometheus 9090:9090 \
        --address=0.0.0.0 \
        >/tmp/loan-prometheus.log 2>&1 &

    sleep 3
fi

# Verify Prometheus
if curl -fsS http://localhost:9090/-/healthy >/dev/null 2>&1; then
    echo "✓ Prometheus is healthy."
else
    echo "ERROR: Prometheus health check failed."
    echo "Check: cat /tmp/loan-prometheus.log"
    exit 1
fi

# --------------------------------------------------
# 8. Start Grafana
# --------------------------------------------------

echo ""
echo "[8/9] Starting Grafana..."

if docker ps --format '{{.Names}}' | grep -q "^${GRAFANA_CONTAINER}$"; then
    echo "✓ Grafana already running."

elif docker ps -a --format '{{.Names}}' | grep -q "^${GRAFANA_CONTAINER}$"; then
    docker start "$GRAFANA_CONTAINER" >/dev/null
    echo "✓ Existing Grafana container started."

else
    docker run -d \
        --name "$GRAFANA_CONTAINER" \
        -p 3000:3000 \
        --add-host=host.docker.internal:host-gateway \
        grafana/grafana:latest >/dev/null

    echo "✓ Grafana container created and started."
fi

# Give Grafana a moment to start
sleep 5

if curl -fsS http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "✓ Grafana is healthy."
else
    echo "WARNING: Grafana started but health endpoint is not ready yet."
fi

# --------------------------------------------------
# 9. Verify Loan API
# --------------------------------------------------

echo ""
echo "[9/9] Verifying Loan Approval API..."

API_URL=$(minikube service loan-approval-service --url 2>/dev/null | head -n 1)

if [ -z "$API_URL" ]; then
    echo "ERROR: Could not determine Loan Approval API URL."
    exit 1
fi

if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
    echo "✓ Loan Approval API is healthy."
else
    echo "ERROR: Loan Approval API health check failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "        DEMO READY"
echo "=========================================="
echo ""
echo "Loan Approval UI:"
echo "  ${API_URL}"
echo ""
echo "FastAPI Swagger:"
echo "  ${API_URL}/docs"
echo ""
echo "FastAPI Health:"
echo "  ${API_URL}/health"
echo ""
echo "Prometheus:"
echo "  http://localhost:9090"
echo ""
echo "Grafana:"
echo "  http://localhost:3000"
echo ""
echo "API URL:"
echo "  ${API_URL}"
echo ""
echo "=========================================="
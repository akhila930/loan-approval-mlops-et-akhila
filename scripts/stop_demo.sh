#!/usr/bin/env bash

echo "=========================================="
echo "Stopping Loan Approval MLOps Demo"
echo "=========================================="

echo "Stopping Prometheus port-forward..."
pkill -f "kubectl port-forward svc/prometheus 9090:9090" 2>/dev/null || true

echo "Stopping Grafana..."
docker stop grafana 2>/dev/null || true

echo ""
echo "Demo services stopped."
echo ""
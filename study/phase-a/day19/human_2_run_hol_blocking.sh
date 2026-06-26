#!/usr/bin/env bash
# Experiment 2: Head-of-Line Blocking
# Starts gateway with HOL budget (10K), runs exp2_runner.py, stops gateway.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Experiment 2: Head-of-Line Blocking ==="

# Kill any existing gateway on 8001
pkill -f "uvicorn gateway.*8001" 2>/dev/null && sleep 2 || true

# Start gateway with exp2 config
source ~/venv-vllm/bin/activate
ADMISSION_ENABLED=true \
BUDGET_TOKENS=10000 \
MAX_WAIT_SECONDS=5 \
RATE_LIMIT_REQUESTS=100000 \
RATE_LIMIT_TOKENS=100000000 \
  uvicorn gateway:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!

trap "kill $GATEWAY_PID 2>/dev/null || true" EXIT

echo "Waiting for gateway to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/debug/stats > /dev/null 2>&1; then
        echo "Gateway ready (PID $GATEWAY_PID)"
        break
    fi
    sleep 1
done

echo ""
python exp2_runner.py

echo ""
echo "=== Experiment 2 complete. Results in results/exp2_results.json ==="

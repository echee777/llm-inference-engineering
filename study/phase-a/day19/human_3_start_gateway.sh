#!/usr/bin/env bash
# Start gateway with default budget for Experiment 3 load sweep.
# Gateway runs in background. Use human_3_stop_gateway.sh to kill it.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Experiment 3: Starting gateway (default budget) ==="

# Kill any existing gateway on 8001
pkill -f "uvicorn gateway.*8001" 2>/dev/null && sleep 2 || true

# Copy locustfile if not present
if [ ! -f exp3_locustfile.py ]; then
    cp ../day18/locustfile.py exp3_locustfile.py
    echo "Copied locustfile from ../day18/"
fi

# Create results dir
mkdir -p results

# Start gateway with defaults (141K budget)
source ~/venv-vllm/bin/activate
ADMISSION_ENABLED=true uvicorn gateway:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!

echo "Waiting for gateway to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/debug/stats > /dev/null 2>&1; then
        echo "Gateway ready (PID $GATEWAY_PID)"
        break
    fi
    sleep 1
done

echo ""
echo "Gateway running in background (PID $GATEWAY_PID)."
echo "Run: ./human_3_run_level.sh <users> <label>"
echo "Stop: ./human_3_stop_gateway.sh"

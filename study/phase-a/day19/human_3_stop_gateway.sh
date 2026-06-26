#!/usr/bin/env bash
# Kill the Experiment 3 gateway.
set -euo pipefail

echo "Stopping gateway on port 8001..."
pkill -f "uvicorn gateway.*8001" 2>/dev/null && echo "Gateway stopped." || echo "No gateway process found."

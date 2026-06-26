#!/usr/bin/env bash
# Capture gateway and vLLM metrics. Prints to stdout.
set -euo pipefail

echo "=== Gateway metrics ==="
STATS=$(curl -s http://localhost:8001/debug/stats)
echo "$STATS" | python3 -m json.tool

echo ""
echo "=== vLLM metrics (filtered) ==="
curl -s http://localhost:8000/metrics | grep -E "num_requests_waiting|gpu_cache_usage|time_to_first_token" || echo "(no matching vLLM metrics found)"

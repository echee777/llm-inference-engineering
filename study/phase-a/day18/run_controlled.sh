#!/bin/bash
# run_controlled.sh — Run load test with admission control ENABLED.
#
# This is the "after" measurement. With admission control enforced,
# requests that would exceed the 65% KV budget are rejected (429/503).
# We expect to see:
#   - TTFT stays flat across ALL load levels (5-50 users)
#   - Rejection rate increases as load increases (excess load is shed)
#   - No preemption events in vLLM logs
#   - GPU KV cache utilization stays below ~70%
#   - Admitted throughput stays stable while total throughput grows
#
# The core tradeoff: admission control trades rejection rate for latency
# stability. Some requests get 429, but admitted requests have predictable,
# SLO-consistent latency. This is the correct tradeoff for any
# latency-SLO-bound serving system.

set -e
cd "$(dirname "$0")"

echo "=== Starting gateway with admission control ENABLED ==="
source ~/venv-vllm/bin/activate
# ADMISSION_ENABLED=true (or just omit it, since true is the default).
# The gateway enforces the budget: _active_tokens + cost > ADMISSION_BUDGET => reject.
ADMISSION_ENABLED=true uvicorn gateway:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!
sleep 3

echo "=== Starting Locust controlled test (50 users, 10 min) ==="
source ~/venv-locust/bin/activate
# IDENTICAL Locust config to baseline. Same users, same ramp, same duration,
# same traffic mix. The ONLY difference is the gateway's admission toggle.
# This makes the comparison scientifically valid: same independent variable
# (traffic), different treatment (admission on/off), measure the outcome (TTFT).
locust -f locustfile.py \
  --host http://localhost:8001 \
  --users 50 \
  --spawn-rate 2 \
  --run-time 10m \
  --headless \
  --csv controlled_results \
  --html controlled_report.html \
  --print-stats

echo "=== Controlled test complete ==="
kill $GATEWAY_PID 2>/dev/null || true

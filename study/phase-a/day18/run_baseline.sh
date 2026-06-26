#!/bin/bash
# run_baseline.sh — Run load test with admission control DISABLED.
#
# This is the "before" measurement. Without admission control, every request
# is admitted regardless of KV cache pressure. We expect to see:
#   - TTFT stable at low concurrency (5-25 users)
#   - TTFT hockey-stick at high concurrency (35-50 users) as KV cache fills
#     and preemption cascades begin
#   - Preemption events in vLLM logs
#   - GPU KV cache utilization approaching 100%
#
# The results from this test are directly compared against run_controlled.sh
# to prove that admission control protects TTFT under identical traffic.

# Exit on any error. Without this, a failed command would silently continue
# and you'd get confusing partial results.
set -e

# cd to the directory containing this script (day18/).
# This ensures relative paths (gateway.py, locustfile.py) work regardless
# of where you run the script from.
cd "$(dirname "$0")"

echo "=== Starting gateway with admission control DISABLED ==="
# Activate the venv that has FastAPI, httpx, transformers, prometheus_client.
source ~/venv-vllm/bin/activate
# ADMISSION_ENABLED=false is an env var prefix. It sets the variable for
# just this one command (uvicorn). The gateway reads it at startup and
# skips the budget check, admitting everything.
# --port 8001: gateway port (vLLM is on 8000).
# &: run in background so the script can continue to start Locust.
# $!: captures the PID of the background process so we can kill it later.
ADMISSION_ENABLED=false uvicorn gateway:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!
# Wait for the gateway to finish starting up (loading tokenizer, etc.)
sleep 3

echo "=== Starting Locust baseline test (50 users, 10 min) ==="
# Switch to the Locust venv (separate from vLLM to avoid dependency conflicts).
source ~/venv-locust/bin/activate
# Locust flags:
#   --host: base URL for all requests (Locust prepends this to "/v1/chat/completions")
#   --users 50: ramp up to 50 concurrent simulated users
#   --spawn-rate 2: add 2 new users per second during ramp-up (takes 25s to reach 50)
#   --run-time 10m: stop after 10 minutes
#   --headless: no web UI, just run and print results (for scripted/automated runs)
#   --csv baseline_results: write stats to baseline_results_stats.csv (for post-analysis)
#   --html baseline_report.html: write a visual HTML report with charts
#   --print-stats: print summary table to stdout when done
locust -f locustfile.py \
  --host http://localhost:8001 \
  --users 50 \
  --spawn-rate 2 \
  --run-time 10m \
  --headless \
  --csv baseline_results \
  --html baseline_report.html \
  --print-stats

echo "=== Baseline test complete ==="
# Clean up: kill the gateway process.
# 2>/dev/null suppresses "no such process" errors if it already exited.
# || true prevents set -e from failing the script if kill returns non-zero.
kill $GATEWAY_PID 2>/dev/null || true

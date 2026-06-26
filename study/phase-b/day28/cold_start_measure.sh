#!/bin/bash
# Day 28: Measure vLLM cold-start time (weights load, CUDA graph, first request)
# Run on GPU host. Outputs timestamps for each phase.

set -e

VENV=~/venv-vllm
MODEL="Qwen/Qwen2.5-3B-Instruct"
PORT=8000
LOG=/tmp/vllm_coldstart.log

echo "=== Cold Start Measurement ==="
echo "Model: $MODEL"
echo "Starting at: $(date +%s.%N)"

T0=$(date +%s.%N)

# Start vLLM in background, capture log
source $VENV/bin/activate
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype half \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    > $LOG 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready (poll health endpoint)
echo "Waiting for server ready..."
while ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 1
done
T_READY=$(date +%s.%N)

# Send first request and time it
echo "Sending first warmup request..."
T_REQ_START=$(date +%s.%N)
curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Say hello"}],"max_tokens":16}' \
    > /dev/null
T_REQ_END=$(date +%s.%N)

echo ""
echo "=== RESULTS ==="
echo "T0 (process start):     $T0"
echo "T_READY (health OK):    $T_READY"
echo "T_REQ_END (first resp): $T_REQ_END"

# Calculate durations
STARTUP=$(echo "$T_READY - $T0" | bc)
WARMUP=$(echo "$T_REQ_END - $T_READY" | bc)
TOTAL=$(echo "$T_REQ_END - $T0" | bc)

echo ""
echo "Startup (process -> health OK):  ${STARTUP}s"
echo "First request latency:           ${WARMUP}s"
echo "Total cold start:                ${TOTAL}s"
echo ""

# Extract weight loading timestamp from log
echo "=== Log phase markers ==="
grep -i "loading model" $LOG | head -3 || true
grep -i "loaded" $LOG | tail -3 || true
grep -i "ready" $LOG | tail -3 || true

echo ""
echo "vLLM still running as PID $VLLM_PID (leave running for next experiments)"

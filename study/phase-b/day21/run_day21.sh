#!/bin/bash
# Day 21 experiment runner - runs on GPU instance via SSM

VENV="/home/ssm-user/venv-vllm/bin"
WORKDIR="/home/ssm-user/day21"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export PATH="$VENV:$PATH"
export CUDA_HOME="/opt/pytorch/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"

# Kill any existing vLLM server
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

echo "=== STEP 2: STARTING VLLM SERVER ==="

# Start vLLM server, redirect all output to log file
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --port 8000 \
  > "$WORKDIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready (up to 3 min for model loading)
echo "Waiting for vLLM server..."
SERVER_READY=0
for i in $(seq 1 90); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after $((i*2))s"
        SERVER_READY=1
        break
    fi
    # Check if process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Log tail:"
        tail -30 "$WORKDIR/vllm_server.log"
        exit 1
    fi
    sleep 2
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: Server did not start in 180s. Log tail:"
    tail -30 "$WORKDIR/vllm_server.log"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Give it a moment to finish initialization logging
sleep 3

echo ""
echo "=== KV BUDGET INFO FROM STARTUP LOG ==="
grep -iE "KV cache|gpu_block|num.*block|block.size|cache_config|KVCacheConfig|num_gpu|memory|concurrency|Available" "$WORKDIR/vllm_server.log" || echo "(no block info found in log)"
echo ""

echo "=== GPU MEMORY ==="
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
echo ""

echo "=== PROMETHEUS METRICS AT STARTUP ==="
curl -s http://localhost:8000/metrics | grep -E "gpu_cache|prefix_cache" || echo "(no cache metrics found)"
echo ""

echo "=== STEP 1: SMOKE TEST (single request) ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hello world"}],"max_tokens":32}' | python3 -m json.tool 2>/dev/null | head -20
echo ""
echo "SMOKE TEST PASSED"
echo ""

echo "=== STEP 3: BASELINE ==="
python3 "$WORKDIR/benchmark.py" baseline 2>&1
echo ""

echo "=== STEP 4: RAMP TO FAILURE ==="
python3 "$WORKDIR/benchmark.py" ramp 2>&1
echo ""

echo "=== FINAL METRICS ==="
curl -s http://localhost:8000/metrics 2>/dev/null | grep -E "gpu_cache|num_requests|preemption" || echo "(server may have died)"
echo ""
nvidia-smi
echo ""

# Cleanup
kill $VLLM_PID 2>/dev/null || true
echo "=== DAY 21 EXPERIMENTS COMPLETE ==="

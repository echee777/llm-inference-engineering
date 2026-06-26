#!/bin/bash
# Day 13: Clean prefix caching experiment
# Restarts vLLM between each syslen to avoid cross-condition cache contamination.

set -e
source /home/ssm-user/venv-vllm/bin/activate

SYSLENS="100 500 1000 2000"
MODEL="Qwen/Qwen2.5-3B-Instruct"

wait_for_vllm() {
    echo "  Waiting for vLLM to become ready..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/v1/models | grep -q "$MODEL" 2>/dev/null; then
            echo "  vLLM ready."
            return 0
        fi
        sleep 3
    done
    echo "  ERROR: vLLM failed to start within 180s"
    return 1
}

kill_vllm() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
}

run_one_syslen() {
    local cache_flag="$1"
    local syslen="$2"
    local label="$3"

    echo ""
    echo "================================================================"
    echo "  $label | syslen=$syslen | cache=$cache_flag"
    echo "================================================================"

    kill_vllm

    echo "  Starting vLLM with $cache_flag..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --dtype half \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 32 \
        $cache_flag \
        --port 8000 > /tmp/vllm_${label}_${syslen}.log 2>&1 &

    wait_for_vllm

    # Verify config
    local pc_setting=$(grep "enable_prefix_caching" /tmp/vllm_${label}_${syslen}.log | head -1)
    echo "  Config: $pc_setting" | head -c 120
    echo ""

    echo "  Running benchmark..."
    python day13_prefix_cache.py --syslen $syslen
}

echo "========================================"
echo "  Day 13: Prefix Caching Experiment"
echo "  Fresh vLLM restart per syslen"
echo "========================================"

# --- RUN 1: Cache OFF ---
echo ""
echo "########################################"
echo "  RUN 1: CACHE OFF"
echo "########################################"
for sl in $SYSLENS; do
    run_one_syslen "--no-enable-prefix-caching" "$sl" "cache_off"
done

# --- RUN 2: Cache ON ---
echo ""
echo "########################################"
echo "  RUN 2: CACHE ON"
echo "########################################"
for sl in $SYSLENS; do
    run_one_syslen "--enable-prefix-caching" "$sl" "cache_on"
done

kill_vllm
echo ""
echo "========================================"
echo "  DONE. All runs complete."
echo "========================================"

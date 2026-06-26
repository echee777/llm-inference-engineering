#!/bin/bash
# Spec token sweep: restart server for each K value, run benchmark

for K in 3 5 7 10; do
    echo "=== K=$K ==="

    # Kill existing server
    pkill -f vllm 2>/dev/null
    sleep 5

    # Launch with ngram spec decode
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-3B-Instruct \
      --speculative-config "{\"method\": \"ngram\", \"num_speculative_tokens\": $K, \"prompt_lookup_max\": 4, \"prompt_lookup_min\": 2}" \
      --dtype half \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096 \
      --port 8000 > server_k${K}.log 2>&1 &

    # Wait for server
    echo "Waiting for server..."
    for i in $(seq 1 90); do
        curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && echo "Server ready" && break
        sleep 3
    done

    # Check if server is actually up
    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server failed to start for K=$K"
        tail -20 server_k${K}.log
        continue
    fi

    # Run benchmark
    python benchmark.py "ngram_k${K}"
    echo ""
done

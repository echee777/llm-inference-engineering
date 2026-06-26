# Day 23 Work: Prefill/Decode Interference Experiment

## Purpose

The Day 23 deliverable answers a specific staff-level interview question: **"Why do short requests get slow under mixed workloads, and what can you do about it?"**

To answer that with data, we needed to:

1. Quantify how much short requests suffer when co-scheduled with long requests (the interference penalty)
2. Show that chunked prefill reduces this penalty
3. Measure whether chunked prefill costs anything in throughput

## Procedure

### Step 1: Establish the four data points

We need a 2x2 matrix of runs to isolate two variables: (a) presence of long requests, and (b) chunked prefill on/off.

```
                    Chunked prefill ON       Chunked prefill OFF
Mixed traffic       Run A (Day 22)           Run C (Day 23)
Short-only          Run B (Day 22)           Run D (Day 23)
```

All four runs use identical hardware and settings: Qwen2.5-3B on T4, gpu_memory_utilization=0.90, max-model-len=4096, max-num-seqs=128, 96 concurrent requests, 10 minutes each.

### Step 2: Mixed traffic design

We split the 96 concurrent slots into two dedicated pools: 48 slots always run short requests (64 prompt tokens, 128 max_new_tokens), 48 slots always run long requests (2048 prompt tokens, 512 max_new_tokens). This guarantees exactly 50/50 in-flight ratio at all times. Without dedicated pools, the natural recycling rate difference (short requests complete in ~8s, long in ~33s) would skew the in-flight mix toward long requests.

### Step 3: Short-only control design

96 slots all running short requests, at the same concurrency as the mixed run. This isolates the variable we care about: "did the presence of long requests hurt short TTFT?" We matched concurrency rather than KV utilization because at these utilization levels (3-16%), we're well below the preemption cliff and KV pressure doesn't affect scheduler decisions.

### Step 4: Discovery that changed the experiment

When we checked the Day 22 vLLM server logs, we found:

```
Chunked prefill is enabled with max_num_batched_tokens=2048
```

vLLM 0.17.1 V1 enables chunked prefill by default. Our Day 22 data (Runs A and B) was already collected with chunking on. The syllabus was written for older vLLM where it was off by default. So we inverted the experiment: instead of enabling chunked prefill, we ran Day 23 with `--no-enable-chunked-prefill` to get the non-chunked baseline (Runs C and D).

### Step 5: Measurement

Each run logs per-request TTFT (time from HTTP request sent to first streaming token received), tagged as "short" or "long". The client measures this with `time.perf_counter()` before `session.post()` and on first non-empty SSE chunk. We also polled `vllm:kv_cache_usage_perc` from the metrics endpoint every 5 seconds during each run.

### Step 6: Profiling the mechanism

Beyond latency numbers, we wanted to understand *why* short requests slow down. We tried three profiling approaches:

1. **nvidia-smi polling** (1s interval): showed GPU at ~98.5% in both mixed and short-only. Conclusion: GPU is equally busy in both cases, so this tool can't explain the difference.

2. **Nsight Systems (nsys)**: failed because vLLM spawns the EngineCore as a separate subprocess. nsys only traces the launched process tree and couldn't capture the EngineCore's GPU kernels.

3. **vLLM Prometheus metrics**: succeeded. The `vllm:iteration_tokens_total` histogram tracks how many tokens are processed per engine step (forward pass). By capturing `_count` and `_sum` before and after a workload, we computed average tokens per iteration.

## Results

### Raw numbers

```
                          Non-Chunked                 Chunked (default)
                          Mixed     Short-only        Mixed     Short-only
Short TTFT p50            181.5ms   138.6ms           180.4ms   135.8ms
Short TTFT p99            1051.9ms  230.4ms           445.5ms   209.5ms

Interference penalty      4.57x                       2.13x
  (p99 mixed/isolated)

Avg tokens/iteration      411.5     ~32               275.9     31.8
Long throughput (tok/s)   778.2     n/a               778.2     n/a
Total throughput (req/s)  7.36      n/a               7.36      n/a
```

### What the numbers mean

**The p50 is nearly identical across all mixed runs (~181ms).** The median short request doesn't experience meaningful interference. This is because most of the time, a short request's prefill lands in an iteration that isn't dominated by a long prefill.

**The p99 is where interference shows up.** The unlucky short requests, those whose prefill or decode step coincides with a long request's prefill, get hit hard:
- Without chunking: 1052ms (4.57x the isolated baseline)
- With chunking: 446ms (2.13x the isolated baseline)

**Chunked prefill cut the tail by 57.6% at zero throughput cost.** Long-request throughput was 778.2 tok/s in both configurations. Total req/s was 7.36 in both. Chunked prefill didn't make anything slower, it just redistributed when the work happens within each iteration.

**The mechanism (from iteration profiling):**
- Without chunking, a 2048-token long prefill runs as a single forward pass. That iteration processes ~411 tokens on average, and every co-batched short request's decode step must wait for it to finish.
- With chunking (budget=2048), the scheduler splits the prefill into smaller chunks and interleaves decode tokens. Each iteration processes ~276 tokens, reducing the starvation window.
- Short-only runs process ~32 tokens per iteration (just 1-token decode steps), which is why they're fast.

**Zero preemptions, zero queuing in all runs.** This is not a memory or capacity problem. It is purely a scheduling fairness problem. The GPU is fully utilized in all cases. Standard observability (GPU %, memory %) would show a healthy system while short-request p99 is getting hammered.

## Key Numbers for Deliverable #6

```
Metric                                          Value
Short-request TTFT p99 (isolated, chunked)      209.5ms
Short-request TTFT p99 (mixed, non-chunked)     1051.9ms
Short-request TTFT p99 (mixed, chunked)         445.5ms
p99 degradation factor (non-chunked)            4.57x
p99 degradation factor (chunked)                2.13x
Chunked prefill short p99 improvement           57.6% (1052 -> 446ms)
Chunked prefill long throughput cost             0% (778.2 tok/s both)
```

## Data Files

```
File                               Description
day22/run_a_mixed.csv              Chunked mixed traffic (3,504 short + 912 long)
day22/run_b_short.csv              Chunked short-only control (9,312 short)
day23/run_nochunked_mixed.csv      Non-chunked mixed traffic (3,504 short + 912 long)
day23/run_nochunked_short.csv      Non-chunked short-only control (9,216 short)
day22/profiling-notes.md           Profiling methodology (nvidia-smi, nsys, vLLM metrics)
day23/experiment-results.md        Full results summary with server configurations
```

## Server Configurations

Non-chunked (Day 23 experiment):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-chunked-prefill \
  --max-num-seqs 128 \
  --port 8000
```

Chunked (Day 22, vLLM default):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --port 8000
# chunked prefill enabled by default in vLLM 0.17.1 V1
# max_num_batched_tokens=2048 (default)
```

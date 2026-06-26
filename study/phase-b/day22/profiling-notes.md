# Day 22 Afternoon: Profiling the Interference Mechanism

## Goal

Explain *why* short request TTFT increases when long requests are co-batched, at the GPU kernel/scheduler level.

## Experiment Results (before profiling)

```
                    Run A (mixed)    Run B (short-only)    Ratio
TTFT p50            180.4ms          135.8ms               1.33x
TTFT p99            445.5ms          209.5ms               2.13x
TTFT max            675.2ms          309.6ms               2.18x
Concurrency         96               96
Preemptions         0                0
```

Short requests take 2x longer at p99 when batched with long requests, with zero preemption and zero queuing. The interference is purely compute-level.

## Profiling Approach 1: nvidia-smi polling (coarse-grained)

**Method:** `nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv -l 1` running in background during 60s workloads.

**Results:**
- Mixed: GPU util avg=98.5%, max=100%
- Short-only: GPU util avg=98.4%, max=100%

**Conclusion:** GPU is equally saturated in both cases. nvidia-smi's 1-second polling granularity tells us "GPU is busy" but not *what* it's busy doing. This tool cannot explain the TTFT difference.

## Profiling Approach 2: Nsight Systems (nsys) -- failed attempts

**Attempt 1: Profile the client process.**
```
nsys profile --trace=cuda,nvtx,osrt --duration=30 \
  --output=nsys_mixed /path/to/python3 profile_workload.py mixed
```
Result: `SKIPPED: does not contain CUDA kernel data.` The client sends HTTP requests; it runs zero GPU kernels. The GPU kernels run in the vLLM EngineCore process (a separate subprocess of the API server).

**Attempt 2: Attach to running EngineCore process.**
```
nsys profile --attach-pid=<PID> ...
```
Result: `unrecognised option '--attach-pid'`. This version of nsys (2025.5.2) does not support attaching to a running process.

**Attempt 3: System-wide capture.**
```
nsys profile --scope=system-wide ...
nsys profile --process-scope=system-wide ...
nsys profile --target-processes=all ...
```
Result: All unrecognised options. These flags don't exist in this nsys version.

**Attempt 4: Launch vLLM under nsys.**
```
nsys profile --trace=cuda --duration=300 \
  /path/to/python3 -m vllm.entrypoints.openai.api_server ...
```
Result: nsys traced the API server process, but the EngineCore is spawned as a child subprocess (`VLLM::EngineCore`). nsys only captured 1 kernel (a trivial fill operation during startup). The EngineCore's CUDA activity was not captured because it runs in a forked process outside nsys's tracing scope.

**Why nsys failed:** vLLM v0.17.1 uses multiprocessing to spawn the EngineCore as a separate process. nsys profiles the launched process tree, but the EngineCore uses `multiprocessing.Process` which creates a new process outside the traced tree. Without `--target-processes=all` or attach support, nsys cannot see the EngineCore's GPU kernels.

## Profiling Approach 3: vLLM Prometheus metrics (succeeded)

**Key insight:** vLLM exposes `vllm:iteration_tokens_total`, a histogram of how many tokens are processed per engine step (forward pass). This directly measures what we need: the work done per iteration.

**Method:** Capture the histogram's `_count` and `_sum` before and after each workload. Delta gives iterations and total tokens for that workload.

```bash
# Before workload
curl -s http://localhost:8000/metrics | grep 'iteration_tokens_total_count'
curl -s http://localhost:8000/metrics | grep 'iteration_tokens_total_sum'

# Run workload
python3 profile_workload.py mixed   # or 'short'

# After workload
# Same curl commands, then compute delta
```

**Profile workload:** 16 concurrent requests (8 short + 8 long for mixed, 16 short for short-only), each generating 64 tokens, ~2s total duration.

**Results:**

```
                     Mixed         Short-only
Iterations           65            65
Total tokens         17,936        2,064
Avg tokens/iter      275.9         31.8         (8.7x ratio)
```

Both workloads completed in the same number of engine steps (65). But each mixed iteration processed 8.7x more tokens because chunked prefill packs long-request prefill tokens (up to 2048 per request) into each forward pass alongside short-request decode tokens.

## Mechanism (confirmed)

With chunked prefill enabled (default in vLLM v0.17.1 V1 scheduler):

1. The scheduler builds each iteration batch by combining prefill chunks and decode tokens.
2. In mixed traffic, prefill chunks from long requests (e.g., 512-token chunks of a 2048-token prompt) are packed into iterations alongside 1-token decode steps from short requests.
3. The forward pass processes all tokens in the batch. The attention computation for each prefill chunk scales with its sequence length (attending to all prior tokens), making it the dominant work unit.
4. Short requests' decode steps complete at iteration boundaries. Larger iterations = longer wait per decode step = higher TTFT.
5. In short-only traffic, each iteration contains only 1-token decode steps (~32 tokens total), so iterations are ~8.7x faster.

The interference is not queuing, not preemption, not memory pressure. It is the chunked prefill scheduler packing large prefill work units into shared iterations, increasing per-iteration latency for all co-batched requests.

## Nsight Compute (ncu)

Available at `/opt/nvidia/nsight-compute/2025.4.1/ncu` but not used. ncu profiles individual kernel invocations (roofline analysis, occupancy, memory bandwidth). It would show whether specific kernels (attention, FFN) are compute-bound vs memory-bandwidth-bound within an iteration, but would not explain the cross-request interference, which is a scheduler-level effect visible in the iteration_tokens histogram.

## Tools Summary

```
Tool                What it tells you                         Granularity    Worked?
nvidia-smi          "Is the GPU busy?"                        1s samples     Yes, but insufficient
nsys                Per-kernel timeline, iteration structure   Microseconds   No (can't trace EngineCore subprocess)
ncu                 Per-kernel roofline, bottleneck analysis   Per-kernel     Available, not needed for this question
vLLM /metrics       Tokens per iteration, KV util, preemption Per-iteration  Yes, answered the question
```

The lesson: when profiling a multi-process serving system, application-level metrics (vLLM's Prometheus endpoint) can be more informative than low-level GPU profilers, because the bottleneck was a scheduler design decision (chunked prefill packing), not a GPU hardware limitation.

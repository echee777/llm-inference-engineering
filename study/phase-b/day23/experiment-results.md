# Day 23 Experiment Results

## Setup

- Model: Qwen2.5-3B-Instruct on T4 (g4dn.xlarge)
- gpu_memory_utilization=0.90, max-model-len=4096, max-num-seqs=128
- Split concurrency pool: 48 short + 48 long slots (96 total)
- Short: 64 prompt tokens, 128 max_new_tokens
- Long: 2048 prompt tokens, 512 max_new_tokens
- Duration: 10 minutes per run
- Zero preemptions, zero queuing in all runs

## Critical Discovery

vLLM 0.17.1 V1 has chunked prefill enabled by default (max_num_batched_tokens=2048).
Day 22 data was collected WITH chunked prefill. The experiment was inverted:
we ran with `--no-enable-chunked-prefill` to get the non-chunked baseline.

## Results

```
                          Non-Chunked                 Chunked (default)
                          Mixed     Short-only        Mixed     Short-only
Short TTFT p50            181.5ms   138.6ms           180.4ms   135.8ms
Short TTFT p99            1051.9ms  230.4ms           445.5ms   209.5ms
Short TTFT max            1056.7ms  332.9ms           675.2ms   309.6ms

Interference penalty      4.57x                       2.13x
  (p99 mixed/isolated)

Avg tokens/iteration      411.5     ~32               275.9     31.8
Long throughput (tok/s)   778.2     n/a               778.2     n/a
Total throughput (tok/s)  1525.8    n/a               1525.8    n/a
Overall req/s             7.36      15.36             7.36      15.52
```

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

## Iteration-Level Profiling

```
Configuration             Avg tokens/iter    Mechanism
Non-chunked mixed         411.5              Full 2048-tok prefill monopolizes forward pass
Chunked mixed             275.9              Prefill split into chunks, interleaved with decode
Short-only (either)       ~32                Only 1-token decode steps, minimal per-iteration work
```

## Data Files

- day23/run_nochunked_mixed.csv - Non-chunked mixed traffic (3,504 short + 912 long)
- day23/run_nochunked_short.csv - Non-chunked short-only control (9,216 short)
- day22/run_a_mixed.csv - Chunked mixed traffic (3,504 short + 912 long)
- day22/run_b_short.csv - Chunked short-only control (9,312 short)
- day23/vllm_nochunked.log - vLLM server log with chunked prefill disabled

## Server Configurations

Non-chunked (day23 experiment):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-chunked-prefill \
  --max-num-seqs 128 \
  --port 8000
```

Chunked (day22, vLLM default):
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

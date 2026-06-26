# Mistakes Log

Running log of errors, misconceptions, and corrected assumptions. Day 38 selects Top 5 for Deliverable #12.

---

## Prior-Residency Candidates

1. **GQA num_kv_heads 8x error (Day 13)** -- miscalculated KV cache size by using num_attention_heads instead of num_kv_heads, overestimating by 8x.

2. **vLLM V0/V1 terminology drift** -- conflated V0 and V1 engine behaviors. V1 uses recompute-only preemption (no CPU swap). Documentation and blog posts mix terminology across versions.

3. **T4 BF16 incompatibility** -- attempted to run with --dtype bfloat16 on T4 (compute capability 7.5). T4 does not support BF16. Must use --dtype half (FP16).

4. **Cliff-as-divergence-ratio not raw-util (Day 16)** -- initially defined "cliff" as a KV utilization threshold. The actual cliff is defined by the TTFT divergence ratio (p99/p50 > 2.0), which is a latency behavior, not a utilization number. The utilization at which it occurs (87%) is workload-specific.

5. **GPU-util as scaling signal (Day 28)** -- assumed GPU utilization would differentiate healthy from degraded states. nvidia-smi GPU util reads ~98% in both. Even fine-grained DCGM metrics measure the wrong dimension (compute, not memory pressure).

---

## Day 28 Additions

6. **Signal lag v1 metric names wrong** -- script looked for `vllm:gpu_cache_usage_perc` but actual Prometheus metric includes labels `{engine="0",model_name="..."}`. Parsing by `startswith` failed. Fixed by searching for `kv_cache_usage_perc in line`.

7. **Signal lag v1 TTFT measurement wrong** -- measured total request time (~11s) instead of actual TTFT. Fixed by switching to streaming and capturing time-to-first-SSE-chunk.

8. **Signal lag v2 prompts too short** -- used natural language prompt (~15 tokens) instead of Day 24's 512-token prompts. KV never filled (peaked at 16.8%). Fixed in v3 by using `"x " * 500` (~530 tokens).

9. **Queue depth as leading indicator assumption** -- assumed queue depth would be a leading indicator universally. Day 28 showed queue depth stayed at 0 through the entire cliff event because max_num_seqs=160 > offered concurrency. Queue depth only works when max_num_seqs < cliff concurrency.

10. **SIGTERM behavior assumption** -- assumed vLLM would send connection reset on SIGTERM. Actual behavior: connections hang silently, clients wait full timeout (30s). Worse than immediate reset because it maximizes the timeout window.

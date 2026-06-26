# Frontier Lab Interview Brief — AI Inference Platform Residency, Phase A

## Top 6 Operating Insights

1. (Day 2/3) `gpu_memory_utilization` is applied to total VRAM (0.9 * 16GB = 14.4GB), not post-weights. Getting this wrong cascades into a wrong KV capacity and wrong admission budget. The derivation chain matters more than any single config value.

2. (Day 9) Continuous batching has a hard cliff, not graceful degradation. TTFT max was 671ms at c=12 (just under KV capacity), then 8,735ms at c=14 (13x spike). Two extra requests triggered 89 preemptions and 1,398 ALLOC_FAILs. That cliff is what admission control exists to prevent.

3. (Day 11) Quantization benefits are phase-dependent, confirmed by roofline. INT8-AWQ: 1.70x decode speedup (memory-bandwidth-bound) vs 1.26x prefill speedup (compute-bound). Mixed workloads won't see uniform gains.

4. (Day 17) Gateway overestimates token budget by +154% to +439% due to worst-case max_tokens charging and no visibility into prefix caching. Policy B progressive release recovered 11,672 tokens under sustained load by shrinking reservations as completions progressed.

5. (Day 18) AC is blind to vLLM scheduler internals. At 100 users: gateway queue was zero, all requests admitted within token budget, but TTFT p99 still hit 5,100ms from compute contention the gateway can't see.

6. (Day 19) AC must be token-based and operated below the p99 hockey stick. Identical token demand across three arrival orderings produced 20x TTFT difference (2,019ms vs 102ms). The hockey stick inflection hit at 70-80% utilization (p99/p50 ratio: 1.5x to 20x). Safe operating point: 60-65%.

## 3 Experiments I Would Present First

1. Day 9 TTFT cliff sweep: Concurrency sweep against Qwen2.5-3B on T4 with constrained KV capacity (1,021 blocks). Demonstrates that when KV memory approaches preemption cascade point, p99 tail latency TTFT explodes. The cliff between c=12 (671ms) and c=14 (8,735ms) motivates the entire admission control system.

2. Day 19 Exp 3 hockey stick sweep: 7-level KV utilization sweep (25-90%) to determine the AC operating threshold. p99 diverged at 70%, collapsed at 80% (p99/p50 ratio 1.5x to 20x). This experiment produced the operating point recommendation of 60-65%.

3. Day 17 reconciliation: Measured gateway token budget overestimation against actual vLLM block usage across three request shapes. Short prompts: +154% overestimate (max_tokens waste). Long shared prompts: +439% overestimate (max_tokens waste + prefix cache invisibility). Two measurement layers: vLLM usage response (+22% for Shape B) vs actual peak block allocation (+439%), with the gap being prefix caching.

## Top 3 Failure Modes Observed

```
Trigger                                → Mechanism                          → Observable Symptom
──────────────────────────────────────────────────────────────────────────────────────────────────────
KV memory exhaustion (c > capacity)    → preemption cascade                 → TTFT explosion
                                         (Day 9: c=12→c=14, 89 preemptions)  (671ms → 8,735ms)
                                         (Day 19: hockey stick at 70-80%)     (p99/p50: 1.5x → 20x)

Adversarial arrival order              → compute contention from large       → 20x TTFT disparity
  (Day 19 Exp 1: large-first)           prefill monopolizing GPU              (2,019ms vs 102ms)
                                                                               same demand, same budget

Over-conservative AC budget            → gateway queues/rejects traffic      → AC-induced degradation
  (Day 18: 65% of 217K token capacity)  that vLLM could serve                 p99: 5,100ms vs 340ms
                                                                               32% rejection rate
```

## 2 Mistakes Caught and Corrected

1. GQA head count error: Used num_kv_heads=16 (full attention heads) instead of num_kv_heads=2 (grouped query attention) for Qwen2.5-3B. This introduced an 8x overestimate of per-token KV memory cost, which would have invalidated every downstream capacity calculation. The error is common because older docs and tutorials assume full MHA.

2. Prefill/decode contention framing: Initially accepted that prefill/decode workload mix causes memory allocation contention. On closer analysis, if the full KV budget is allocated upfront in both cases, there is no memory pressure difference. The real blind spot is compute contention: prefill is compute-bound, decode is memory-bandwidth-bound, and the gateway can't distinguish between them.

## 2 vLLM V1 Mental Model Shifts

1. Preemption without swap: V0 preemption swapped KV cache to CPU RAM. V1 simply discards KV and recomputes from scratch when re-admitted. Simpler block ownership model, but recompute cost scales with prompt length, creating an implicit bias against long-context requests under memory pressure.

2. Try-and-fail allocation: V0 scheduler checked a watermark threshold before deciding whether to preempt (predictive). V1 just tries to allocate blocks and preempts only on actual failure (reactive). Simpler conceptually and reduces tuning parameters (no watermark to configure).

## 1 Production Recommendation

Set token-based admission budget to 60-65% of KV capacity. On our T4/Qwen2.5-3B setup, p99 TTFT diverged at 70% utilization and collapsed at 80% (p99/p50 ratio went from 1.5x to 20x). The threshold will vary by model architecture and GPU -- tighter KV capacity (e.g., Llama 70B on A100) likely requires a lower operating point because each request is a larger fraction of total budget. The operating point must be empirically validated per deployment.

# Phase A Experiments Summary

```
Day   Experiment                         What happened                                                Key numbers
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 3    Fragmentation simulation           Python sim: contiguous vs paged allocator.                   850MB free but unusable (contiguous) vs 0% waste (paged)
                                         Alternating frees create holes.                              Conceptual sim, not against real vLLM.

 9    Preemption spiral (Block 1)        80 concurrent requests, gpu_mem_util=0.45.                   89 preemptions, 1,398 ALLOC_FAILs
                                         KV pool exhausted at T+22.4s, cascade begins.                Worst request preempted 6x, lost 2,156 decode tokens
                                                                                                      Latency: 8.7s - 120s (baseline 43ms)

 9    TTFT cliff sweep (Block 2)         Concurrency sweep with long prompts.                        c=12: max TTFT 671ms (stable)
                                         ~88 blocks/request, capacity=11.6 requests.                  c=14: max TTFT 8,735ms (13x spike)
                                                                                                      c=20: max TTFT 11,674ms

11    Quantization benchmark             FP16 vs INT8-AWQ vs INT4-GPTQ,                              Decode: INT8 1.70x, INT4 2.44x speedup
                                         prefill-heavy vs decode-heavy workloads.                     Prefill: INT8 1.26x, INT4 1.60x speedup
                                                                                                      KV capacity: FP16 193K → INT8 268K → INT4 306K tokens

17    Reconciliation (overestimation)    3 request shapes, measured gateway charge                    Shape A: +154%, Shape B: +439%, Shape C: +155% overestimate
                                         vs actual vLLM block usage.                                  Sources: max_tokens waste + prefix cache invisibility

17    Policy B (progressive release)     Constrained budget=8K, 100 requests,                        77 tokens freed across 46 requests (negligible)
                                         short completions (~50-100 tokens).                          Short completions finish before first checkpoint

18    Load test: baseline vs AC (50u)    Locust 50 users, AC on vs off.                              No difference. System well within capacity.
                                                                                                      TTFT p50 ~160ms both modes. Zero rejections.

18    Load test: baseline vs AC (100u)   Locust 100 users, AC on vs off.                             AC OFF: TTFT p99 340-570ms, 0 rejections
                                         65% budget target too conservative for                       AC ON:  TTFT p99 5,100-5,200ms, 678 rejections (32%)
                                         Qwen 3B's 217K token KV capacity.                           Policy B freed 11,672 tokens under sustained load

18    Load test: 200u                    Both modes crashed.                                          Process-level overload, not KV exhaustion.
                                         AC cannot protect against this failure class.                 AC is not a general overload protector.

19    Exp 1: Adversarial starvation      Tenant A (5 short) + Tenant B (1 large),                    Case A (large first): Tenant A TTFT 2,019ms
                                         3 arrival orderings, budget=15,211.                          Case B (small first): Tenant A TTFT 102ms
                                         All admitted (no rejections in any case).                     20x TTFT difference, same total demand

19    Exp 2: HOL blocking                Budget=10K, large request=12,288 tokens                     Large request: rejected (queue timeout 5.0s)
                                         (exceeds budget). 10 shorts behind it.                       10 shorts: all admitted directly, TTFT 93-140ms
                                         Design flaw: shorts bypassed queue.                          HOL blocking not demonstrated

19    Exp 3: Hockey stick sweep          7 load levels (25%-90% budget util),                         70%: p50=180ms, p99=275ms, 6 rejections
                                         Locust + gateway + vLLM metrics.                             80%: p50=200ms, p99=3,980ms, 273 rejections (20x ratio)
                                                                                                      90%: p50=710ms, p99=5,050ms, collapse
```

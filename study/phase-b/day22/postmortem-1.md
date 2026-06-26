# Postmortem #1: KV Cache Exhaustion Under Concurrent Long-Context Requests

**System:** vLLM V1 serving Qwen2.5-3B-Instruct on T4 (g4dn.xlarge)
**Date:** 2026-04-07
**Model config:** `--max-model-len 4096`, `--max-num-seqs 64/128`
**Prompt spec:** 2,258 actual tokens, `max_new_tokens=512`

---

## Summary

<!-- TODO: You write this — one paragraph, system-level, no speculation -->

---

## Timeline

### Run A: Memory-Constrained Configuration (`gpu_memory_utilization=0.45`)

KV cache budget: 16,352 tokens (0.56 GiB). Effective capacity: ~5.9 concurrent requests at 2,770 tokens/request.

```
T=0:  c=1,  KV=14.0%, TTFT p99=582ms,   queue=0,  preempt=0    — healthy
T=1:  c=2,  KV=29.2%, TTFT p99=1,173ms,  queue=0,  preempt=0    — healthy
T=2:  c=4,  KV=60.0%, TTFT p99=2,255ms,  queue=0,  preempt=0    — healthy
T=3:  c=6,  KV=92.5%, TTFT p99=3,304ms,  queue=0,  preempt=48   — preemption onset
T=4:  c=8,  KV=94.4%, TTFT p99=3,494ms,  queue=1,  preempt=113  — preemption cascade
T=5:  c=10, KV=95.4%, TTFT p99=3,488ms,  queue=3,  preempt=178  — preemption cascade
T=6:  c=16, KV=95.4%, TTFT p99=3,536ms,  queue=9,  preempt=308  — preemption cascade
T=7:  c=32, KV=95.4%, TTFT p99=3,585ms,  queue=25, preempt=502  — preemption cascade
```

### Run B: Generous Memory Configuration (`gpu_memory_utilization=0.90`)

KV cache budget: 203,520 tokens (6.99 GiB). Theoretical capacity: ~73 concurrent requests.

```
T=0:  c=1,  KV=1.0%,  TTFT p99=611ms,    queue=0,  preempt=0   — healthy
T=1:  c=4,  KV=4.2%,  TTFT p99=1,183ms,  queue=0,  preempt=0   — healthy
T=2:  c=8,  KV=8.8%,  TTFT p99=3,993ms,  queue=0,  preempt=0   — compute saturation
T=3:  c=16, KV=17.5%, TTFT p99=9,341ms,  queue=0,  preempt=0   — compute saturation
T=4:  c=32, KV=34.6%, TTFT p99=17,910ms, queue=0,  preempt=0   — compute saturation
T=5:  c=48, KV=48.5%, TTFT p99=27,359ms, queue=3,  preempt=0   — queue onset
T=6:  c=64, KV=56.4%, TTFT p99=36,600ms, queue=11, preempt=0   — queue buildup
T=7:  c=96, KV=56.8%, TTFT p99=40,967ms, queue=44, preempt=0   — saturated
T=8:  c=128,KV=56.3%, TTFT p99=47,046ms, queue=76, preempt=0   — saturated
```

---

## Root Cause Analysis

### Trigger

Long-context requests (2,258 prompt tokens + 512 generation tokens) at increasing concurrency levels.

### Primary Root Cause

The system exhibits bimodal failure depending on the ratio of available KV memory to per-request compute cost:

**Mode A (memory-bound, `gpu_memory_utilization=0.45`):** KV cache memory exhausted under 6+ concurrent long-context requests. With only 16,352 tokens of KV budget, the system hits its memory ceiling at ~6 concurrent requests of 2,770 tokens each. Preemptions begin at 92.5% KV utilization with zero queue depth, confirming the bottleneck is memory, not compute.

**Mode B (compute-bound, `gpu_memory_utilization=0.90`):** Prefill compute saturates the GPU under 50+ concurrent requests. With 203,520 tokens of KV budget, the system has ample memory (KV util peaks at ~56%) but cannot process prefill fast enough. Prefill compute is proportional to prompt token count, so each 2,258-token prefill dominates GPU cycles and delays decode iterations for all co-batched requests.

### Contributing Root Cause

No backpressure at the API boundary. vLLM continues accepting HTTP requests regardless of internal queue depth or KV utilization. The scheduler provides internal backpressure by queuing requests it cannot immediately run, but this queue grows unboundedly because the API layer never rejects or sheds load. This is analogous to Kubernetes queue-based systems that lack natural backpressure and fail when queues overflow.

### Mechanism

**Memory-bound mechanism (Mode A):** Concurrent requests consume KV blocks. At c=6, KV utilization reaches 92.5% and block allocation begins to fail. The V1 scheduler preempts the lowest-priority running request, discarding its KV cache. The preempted request re-enters the wait queue and, when rescheduled, must recompute its entire prefill, consuming the same KV blocks it just freed. Under sustained concurrency this creates a positive feedback loop: preemption count escalates from 48 (c=6) to 502 (c=32) while throughput drops from 101 to 51 tok/s due to wasted recomputation. The diagnostic signal: queue depth = 0 with preemptions > 0.

**Compute-bound mechanism (Mode B):** Concurrent requests saturate GPU compute. Continuous batching processes a mix of requests in prefill and decode phases. Each 2,258-token prefill consumes ~2,258x the compute of a single decode step, monopolizing GPU cycles while all co-batched decode iterations wait. Running requests plateau at ~50 despite 64+ concurrent submissions, while the vLLM queue grows from 0 to 76 with zero preemptions. The diagnostic signal: queue depth > 0 with preemptions = 0 and KV utilization well below capacity.

---

## What vLLM's V1 Scheduler Did

### Memory-Constrained Run (0.45)

1. **Preemption onset KV%:** 92.5% (at c=6). Zero preemptions at c=4 (60% KV util), 48 preemptions at c=6.
2. **Did preemption stabilize?** No. Preemption worsened the system. Preemption count escalated from 48 (c=6) to 502 (c=32) while throughput dropped from 101 to 51 tok/s. Queue depth grew from 0 to 25 as preempted requests accumulated.
3. **Why?** Recompute-only preemption creates a positive feedback loop. Each preempted request loses its KV cache, re-enters the wait queue, and when rescheduled must recompute its entire prefill, consuming the same KV blocks it just freed. As more requests are preempted, a growing proportion of GPU compute is wasted on prefill that will be discarded at the next preemption, reducing effective throughput and increasing queue depth.

### Compute-Constrained Run (0.90)

1. **Running requests capped at:** ~50 concurrent running requests, regardless of submitted concurrency (50 at c=64, 50 at c=96, 50 at c=128).
2. **KV utilization at saturation:** ~56%. The system never approached its 203,520-token KV budget. Memory was not the bottleneck.
3. **What limited throughput?** Prefill compute. Each 2,258-token prefill consumes ~2,258x the compute of a single decode step. With ~50 concurrent requests cycling through prefill and decode, the GPU is fully saturated processing prefills. Additional requests queue at the scheduler level (queue depth grows from 0 at c=32 to 76 at c=128) but cannot be admitted because the GPU has no spare cycles for their prefill.

---

## Five Required Graphs

<!-- TODO: Generate from data -->

1. KV cache utilization % over time (both runs)
2. TTFT p50 and p99 over time (both runs)
3. Preemption event count over time (Run A)
4. Concurrency vs. KV utilization (scatter, both runs)
5. Queue depth / `num_waiting_seqs` over time (both runs)

---

## Lessons Learned

1. **LLM serving failure modes are regime-dependent, not workload-dependent alone.** The ratio of available KV memory to per-request compute requirement determines whether a system saturates in memory or compute first. In general, prefill-heavy request mixes (many concurrent requests in chunked prefill) exert O(P) greater compute per forward pass (P = prompt tokens), which can saturate GPU compute before exhausting KV memory. In our experiment, the same workload exhibited both failure modes: with `gpu_memory_utilization=0.90`, KV utilization plateaued at ~56% while the vLLM queue grew and TTFT exploded (compute-bound). With `gpu_memory_utilization=0.45`, the system hit 92% KV utilization with preemption cascades (memory-bound). The bottleneck shifted not because the workload changed, but because the KV budget did. Understanding which regime you're operating in determines whether the correct remediation is more memory, more compute, or admission control.

2. **Preemption is not a safety valve under sustained memory pressure.** In vLLM V1's recompute-only scheduler, preempted requests lose all KV state, re-enter the wait queue, and must rebuild their entire prefill when rescheduled, re-consuming the same KV blocks they just freed. Under sustained concurrency, this becomes a positive feedback loop: each preemption wastes compute on prefill that will be discarded at the next preemption, reducing effective throughput. The system enters a regime where an increasing proportion of total compute is churn (wasted recompute), net capacity decreases, queues grow, and latencies explode. Preemption count escalated from 48 to 502 while throughput dropped from 101 to 51 tok/s.

3. **Back-of-envelope capacity math predicts the failure point.** Total KV token budget = (VRAM x `gpu_memory_utilization`) minus model weights, activation buffers, and overhead. For uniform-length requests: `max_concurrent = budget_tokens / tokens_per_request`. In our memory-constrained run: 16,352 / 2,770 = 5.9 max concurrent requests. Failure occurred at c=6, exactly as predicted. In production, this ceiling is optimistic because request lengths vary and the temporal mix of requests at any instant can cause KV exhaustion earlier than the uniform-length estimate suggests. The lesson: always do this calculation before load testing, not after the incident.

---

## Remediation

1. Tighten admission control: enforce `max_prompt_length × max_concurrency` product limit
2. Add KV utilization as a real-time admission signal (not just a static token budget estimate)
3. Set `--max-model-len` tighter for high-concurrency deployments

---

## Why My Budget Was Wrong

Qwen2.5-3B uses Grouped Query Attention with 2 KV heads (vs. 16 query heads). Failing to account for GQA and using 16 heads would overestimate per-token KV cost by 8x, yielding a budget of ~2,044 tokens, which predicts max_concurrent = 2,044 / 2,770 = 0.7 requests. The model would say you can't serve a single request, which is obviously wrong. With the correct GQA correction (num_kv_heads=2), the budget is 16,352 tokens and max_concurrent = 5.9, matching observed failure at c=6.

**Corrected KV bytes per token (Qwen2.5-3B-Instruct):**
```
KV bytes per token per layer = 2 × num_kv_heads × head_dim × 2 bytes
                             = 2 × 2 × 128 × 2 = 1,024 bytes
Per token across all layers  = 1,024 × 36 = 36,864 bytes ≈ 36 KiB
```

---

## Narrative: "Tell Me About a Time a System You Owned Failed"

**Context:** A Kubernetes training cluster with Karpenter-autoscaled EC2 nodes. Users dynamically launched Ray clusters of arbitrary worker count and CPU/memory size via RayJobs. To prevent partial-worker deadlock, we installed Volcano as a gang scheduler. However, Volcano lacked first-class RayJob support: it gang-scheduled Ray workers but not the RayJob submitterPod (driver), leaving a resource gap in the admission decision.

**Trigger:** To avoid IP address exhaustion in the VPC, we capped EC2 node count via ResourceQuota. During product rollout, users launched hundreds of RayJobs, each requesting tens of nodes. The cluster hit the compute ceiling. RayClusters started and reserved resources, but their submitterPods couldn't be scheduled because Volcano didn't account for them in the gang. This led to idle clusters consuming resources without performing work. At saturation, deadlock occurred: no cluster could progress (waiting on its submitterPod) and no submitterPod could launch (no CPU available because idle clusters held all resources).

**Root cause:** Volcano's gang scheduling was incomplete. It ensured workers wouldn't start without full worker resources, but the submitterPod was outside the gang boundary. This created a failure mode worse than no gang scheduling at all: resources were fully committed to clusters that couldn't make progress, with no mechanism to release them.

**What I expected:** I foresaw this failure mode and raised it in design reviews with the team. A principal engineer decided to proceed with Volcano because we had already invested in the implementation. I disagreed but committed to the decision, documenting the discussion and the predicted starvation/deadlock scenario so we could revisit with evidence.

**What surprised me:** Deadlock manifested within 2-3 days of production rollout. The speed at which partial gang scheduling converted a "possible starvation" scenario into hard deadlock under real user behavior was faster than anticipated.

**What we changed:** After documenting the incident and communicating impact to users, I proceeded with my original recommendation: replace Volcano with Kueue. Kueue provides first-class RayJob support and will not launch any component of a Ray cluster (workers or driver) until all resources are available. This completely eliminated starvation and deadlock.

**Staff-level takeaway:** This is the same pattern as admission control in LLM serving. A proper queue must understand the full resource footprint of a unit of work, not just part of it. Partial admission (Volcano scheduling workers without the driver, or vLLM admitting requests without accounting for full KV budget) leads to resource commitment without progress, which under pressure becomes deadlock or cascading failure. The organizational lesson: when you disagree with an architectural decision, document your concerns with specific predicted failure modes so the team can revisit with data, not opinions.

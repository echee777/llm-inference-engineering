# Day 17 Work Log

## Morning: Reconciliation Experiment

### Setup

vLLM on T4 (g4dn.xlarge, 16 GiB), Qwen2.5-3B-Instruct FP16.
vLLM reported KV cache capacity: 207,440 tokens, 12,964 blocks (16 tokens/block).
Gateway ADMISSION_BUDGET: 124,150 tokens (191K * 0.65).

Note: vLLM reports 207K tokens, not the 191K from my Week 1 calculator. The gap is likely due to different overhead estimates. The gateway still uses 191K as the base, which is conservative and fine.

### Experiment Design

Three shapes, 5 concurrent requests each, max_tokens=512.
Prompts designed to produce SHORT completions (~40-150 tokens) to expose the max_completion overestimate. Each prompt is unique to reduce prefix cache conflation (partially, Shape B prompts share a common base).

- Shape A: ~200-token prompts (unique), short answers
- Shape B: ~2000-token prompts (shared base, unique suffixes), short answers
- Shape C: Mixed (3x A + 2x B)

### Results

```
Metric                                     Shape A    Shape B    Shape C
------------------------------------------------------------------------
Requests                                         5          5          5
Total prompt tokens                            959      10379       4756
Total completion tokens                        425        245        368

1. Gateway charge (prompt+max_compl)          3519      12939       7316
2. Actual tokens (prompt+actual_compl)        1384      10624       5124
3. vLLM peak blocks (x16)                72b=1152t 150b=2400t 179b=2864t

Overestimates:
  Shape A: gateway vs actual tokens = +154%
  Shape A: gateway vs vLLM blocks   = +205%

  Shape B: gateway vs actual tokens = +22%
  Shape B: gateway vs vLLM blocks   = +439%

  Shape C: gateway vs actual tokens = +43%
  Shape C: gateway vs vLLM blocks   = +155%

Divergence sources:
  Shape A: max_completion waste=2135 tokens, prefix cache savings=232 tokens
  Shape B: max_completion waste=2315 tokens, prefix cache savings=8224 tokens
  Shape C: max_completion waste=2192 tokens, prefix cache savings=2260 tokens
```

### Divergence Analysis

Three distinct layers of overestimation, each with a different cause:

1. max_completion overestimate (dominant for Shape A). Completions averaged ~50-150 tokens but 512 was reserved. This is ~2100-2300 tokens of phantom budget per shape. This is the gap Policy B is designed to close.

2. Prefix caching (dominant for Shape B). vLLM's prefix cache shares KV blocks across requests with identical prompt prefixes. My gateway charges full prompt_tokens per request. Shape B had 5 requests sharing ~2000 tokens of base prompt, so vLLM only allocated new blocks for the unique suffixes and completions. The gateway doesn't know about this at all.

3. Block rounding (minor). ceil(tokens/16) adds at most 15 tokens per request. Dwarfed by the other two effects.

The overestimate direction is always conservative: gateway thinks more KV is in use than actually is. This causes spurious 429s but never OOM. That's the safe direction to be wrong.

### Why Shape A shows higher % overestimate than Shape B

Shape A completions were short (~85 tokens average) against a small prompt (~190 tokens). The 512-token reservation is a huge fraction of the total charge (512/700 = 73% of the charge is reservation). Shape B has large prompts (~2000 tokens) that dominate the charge, so the 512 reservation is a smaller fraction (~20%). The absolute max_completion waste is similar (~2100-2300 tokens) but it's a bigger percentage of the smaller Shape A total.

### Prefix Caching: Empirical Confirmation

Prefix caching was already listed as one of the five Day 16 limitations of the gateway proxy. This experiment puts numbers on it. The gateway charges prompt_tokens independently per request, but vLLM deduplicates shared prefixes in the block table. For Shape B (5 requests sharing ~2000 tokens of base prompt), prefix cache savings were 8,224 tokens, dwarfing the max_completion overestimate.

For workloads with repeated or templated prompts (RAG with shared system prompts, chat with shared history), the gateway's estimate of KV block usage could be dramatically higher than reality. Still conservative (over-rejection, not over-admission). A production fix would be to expose vLLM's prefix cache hit rate and discount prompt tokens accordingly. Out of scope for Day 17.

### Trust Boundary Stance

The 15% threshold from the syllabus is for "gateway vs actual KV blocks." My numbers:
- Shape A: +205%
- Shape B: +439%
- Shape C: +155%

These are all way above 15%. But the overestimate is conservative (safe direction). The right interpretation: the gateway is dramatically over-conservative, not dangerously wrong. The fix isn't to tighten TARGET_UTILIZATION (that would make it worse). The fix is Policy B (release excess budget as tokens stream) and possibly prefix-cache-aware discounting.

Stance: Gateway is safe but wasteful. Policy B is essential for reasonable throughput. The 65% TARGET_UTILIZATION provides headroom for the remaining estimation errors after Policy B corrects the max_completion overestimate.

### What knobs can we turn?

Given the gross overconservatism (+154% to +439%), considered three fixes:

1. Lower the default max_completion_tokens (e.g. from 512 to 150). Rejected: actual completion length is situationally dependent. The client sets max_tokens as a ceiling, and the gateway can't predict how much of that ceiling gets used. Lowering the default risks under-reserving for requests that genuinely need long completions.

2. Increase TARGET_UTILIZATION (e.g. from 0.65 to 0.90). Tempting but blunt. The 0.65 came from Day 9's observation of preemption cascades at 65-70% actual KV block occupancy. The key insight: that threshold is about real KV block pressure, not the gateway's counter. With short completions, the counter might say 65% but actual blocks are at 20%. Bumping to 90% would help today's workload but could be too aggressive for a different workload where completions are longer and actual KV pressure tracks closer to the counter. It's a static guess that doesn't adapt.

3. Policy B: release excess budget dynamically as tokens stream back. This is the right fix because it makes the counter track reality in real time rather than picking a new static correction factor. The counter starts at the conservative worst case (prompt + max_completion) and ratchets down as actual completion length reveals itself. Adapts automatically to any workload.

Conclusion: Policy B is not an optimization. It's the necessary correction that makes the admission budget a useful proxy. The morning experiment proved this.

## Afternoon: Policy B + Rate Limiting + FIFO Queue

### What was built

Extended the Day 16 gateway (day17/gateway.py) with:

1. Policy B (periodic release with safety floor)
2. Per-API-key sliding window rate limiter (req/min + tokens/min)
3. Bounded FIFO queue (maxsize=50, 5s timeout), labeled as intentionally naive for Day 19 HOL blocking analysis
4. In-process counters (admitted, rejected_budget, rejected_rate_limit, rejected_queue_full, correction_delta_released)
5. try/finally budget release on all paths (normal, error, disconnect)

### Policy B mechanics

Reserve prompt_tokens + max_completion_tokens on admission. Every RELEASE_INTERVAL=50 generated tokens, check if we can release excess. The safety floor (SAFETY_MARGIN=64) prevents releasing too aggressively.

First release happens at token ~100 (not 50), because at token 50 the safe reservation (remaining + 64) still exceeds the original reservation. At token 100: remaining=412, safe=476, originally reserved=512, so release 512-476=36 tokens. After that, roughly 50 tokens freed every 50 generated.

Bug found and fixed during testing: original release logic used `settled_through_token` (a token position) to compute final release amount, but Policy B releases less than the full interval because of SAFETY_MARGIN. This leaked exactly SAFETY_MARGIN tokens per request. Fix: track cumulative `total_released` per request, final release = estimated_cost - total_released.

### Policy B experiment results

Constrained ADMISSION_BUDGET to 8,000 tokens, concurrency=50, 100 requests with max_tokens=512.

```
                            Baseline    Policy B
Admitted                          52          46
Rejected (budget+queue)           48          54
correction_delta_released          0          77
```

Policy B did not meaningfully improve admission rates here. Why: with short completions (~50-100 tokens), requests don't generate enough tokens to trigger many releases (first release at token ~100, many completions finish before that). The 77 tokens freed across 46 requests is ~1.7 tokens per request, barely enough to fit one additional request.

Policy B's value appears at longer completions (200+ tokens) where the gap between reserved and actual grows wider and persists longer. With short completions, most budget is freed at completion time anyway, and the queue timeout (5s) is often shorter than the time it takes for Policy B to accumulate meaningful releases.

This is a useful negative result: Policy B is necessary but not sufficient. The right fix depends on which overestimation source dominates your workload:
- Long shared prompts (RAG, chat with system prompts): prefix-cache-aware discounting (reduces prompt_tokens overcharge where vLLM deduplicates shared prefixes)
- Long completions (200+ tokens): Policy B (releases excess budget as completion length reveals itself)
- Short completions with short unique prompts: neither fix helps much, you're stuck with the max_completion overestimate and the 65% headroom absorbs it

### Rate limiting

Sliding window per API key, 60 req/min and 100K tokens/min. Tested: rate limit triggers at request 61, no budget charged for rejected requests.

During the Policy B experiment, the rate limiter was initially the bottleneck (40/100 rejected by rate limit). Loosened to 600 req/min for the experiment, then restored.

### FIFO queue

Bounded at 50 entries, 5s max wait. When budget is released, _drain_queue() tries to admit waiting requests. Queue full returns 503 immediately. Timeout returns 503 with budget never reserved.

Intentionally naive: FIFO has no priority, no fairness, no preemption awareness. This is the baseline for Day 19's HOL blocking analysis.

### Failure semantics checklist

```
Scenario                      Result
Normal completion             PASS: budget released to 0
Client disconnect mid-stream  PASS: budget released to 0
Queue timeout                 Document-only: 503, budget never reserved
Rate limit rejection          PASS: budget not charged
vLLM 500 error                Document-only: try/finally releases budget
Worker crash                  Not tested. Budget leaks. Blast radius: ~551 tokens
                              per in-flight request * active count. Recovery requires
                              process restart (budget resets to 0). Persistent budget
                              state is Phase C scope.
```

### Observation: no standardized backpressure interface

The gateway is entirely external to vLLM. It derives budget from static KV math, counts tokens via tokenizer, parses SSE chunks. It never queries vLLM's actual block utilization or prefix cache state. This is why overestimates are so large (morning reconciliation: +154% to +439%).

The product gap in the ecosystem isn't rate limiting or queuing (commodity via Envoy/Kong/LiteLLM). It's a standardized backpressure API between inference engines and admission controllers: actual KV block utilization, prefix cache hit rate, preemption count, scheduler queue depth. If engines exposed this uniformly, admission control could be a generic proxy product. That standard doesn't exist yet.

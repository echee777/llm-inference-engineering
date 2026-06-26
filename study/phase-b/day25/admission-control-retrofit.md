# Admission Control Retrofit Design

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

Feeds into D#9 and D#11.

---

## 1. Signal Change

```
FROM: static token budget estimate (computed at gateway from num_kv_heads * head_dim * layers)
TO:   real-time vllm:gpu_cache_usage_perc (Prometheus) as the primary admission signal
WHY:  the static estimate cannot see interference-driven KV retention or preemption
      feedback. Day 24 data showed the cliff at 87% KV utilization. A static gate
      calibrated to token budget alone would not have caught this because:
      (a) interference causes requests to hold blocks longer than the token math
          predicts (Day 23: 4.57x short-request p99 degradation at 3-16% KV)
      (b) preemption-driven recompute raises effective utilization above nominal
          (Day 24: recompute load reached 30% of forward progress at the cliff)
      The real-time metric captures both because they manifest directly as
      higher-than-predicted KV occupancy.
```

## 2. Hard Rejection Threshold

```
If vllm:gpu_cache_usage_perc >= 85% (cliff - 2%) --> reject all new requests (HTTP 429)
WHY:  past 85%, the system is within jitter range of the 87% cliff. Admitting more
      requests does not increase throughput. Day 24 showed throughput peaked at 879
      tok/s at 86% KV and dropped to 786 at 87%, never recovering. Admitting into
      the unstable regime reduces throughput for all requests, including those
      already in flight. The correct action is hard rejection, not queuing.
      Queuing defers admission to a system that drains slower under pressure,
      making the queue grow rather than shrink.
```

## 3. Tiered Admission Thresholds

```
Short requests  (prompt < 256 tokens):    admit up to 82% KV (cliff - 5pp)
Medium requests (prompt 256-1024 tokens): admit up to 77% KV (cliff - 10pp)
Long requests   (prompt > 1024 tokens):   admit up to 72% KV (cliff - 15pp)

WHY:  long requests carry asymmetric risk to system stability:
      (a) Direct: more KV blocks consumed per request (2048-token prompt uses ~128
          blocks vs ~4 blocks for a 64-token prompt)
      (b) Indirect: longer prefill starvation window (~504ms for a 2048-token
          prefill vs negligible for 64 tokens). This freezes co-scheduled decode
          steps, causing those requests to hold their KV blocks longer, raising
          effective utilization above the nominal level.
      The wider margins for longer requests account for both direct KV cost and
      indirect interference cost.
```

## 4. Chunked Prefill Gate

```
If incoming request prompt_tokens > max_num_batched_tokens (2048 default),
check current short-request queue depth before admitting.
If short-request queue depth > 10 --> defer or reject the long request.

WHY:  prevents a single large request from starving pending short requests.
      Day 23 data: short-request TTFT p99 degraded 4.57x (non-chunked) and 2.13x
      (chunked) when co-scheduled with 2048-token requests. Even with chunked
      prefill enabled, a residual 2.13x degradation remains. Gating long request
      admission on short-request queue depth protects short-request SLOs when
      the system is already under decode pressure.
```

## 5. Prometheus Scrape Interval Requirement

```
Scrape interval for vllm:gpu_cache_usage_perc must be < TTFT SLO / 2.
If TTFT SLO is 3 seconds, scrape every 1-1.5 seconds.

WHY:  the admission controller reacts to stale data. If the scrape interval is
      10 seconds and KV jumps from 75% to 90% in 3 seconds (observed: Day 24
      showed +-15pp jitter at fixed concurrency over 5-second intervals), the
      controller won't see the spike until it's too late. Faster scrape =
      faster reaction = tighter margin possible.
```

## 6. Explicit Tradeoff

```
We are sacrificing:   peak throughput (tokens/sec)
In exchange for:      stable p99 latency and system stability
Cost:                 8.6% throughput reduction operating at 79% vs 86% KV
                      (803 tok/s vs 879 tok/s)
Evidence:             Day 24 cliff graph. Past the cliff, throughput drops AND
                      latency explodes. There is no throughput to recover by
                      running hotter.
```

## 7. What This Does NOT Fix

```
- Does not eliminate preemption under bursty long-context load. Only reduces
  frequency by keeping sustained utilization below the cliff.
- Does not address retry amplification (Week 6 problem). Retry clients that
  resubmit on timeout will inject artificial concurrency that the admission
  controller sees as real load. Backoff policy is a client-side mitigation,
  not an admission control problem.
- Requires Prometheus scrape interval < TTFT SLO to be responsive. If scrape
  is too slow, the controller reacts to stale data and the effective margin
  shrinks.
- Does not prevent compound failure if all three conditions hit simultaneously.
  Only raises the utilization threshold at which compound failure is triggered.
- Does not handle the case where a single replica's cliff differs from the
  fleet average. Per-replica KV monitoring is required, not fleet-wide averages.
```

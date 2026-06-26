# Deliverable #8: Postmortem #2 -- Retry Cascade

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

Note: This postmortem is constructed from Day 24 cliff data and geometric series analysis rather than a live retry storm experiment. The mechanism, remediation, and narrative are derived from measured system behavior under KV pressure (Day 24) combined with the retry amplification model (Day 26 conceptual analysis). Specific timestamps are reasoned estimates, not observed values.

---

## Summary

System was healthy at ~60% KV utilization serving mixed traffic. A 30-second traffic burst of long-context requests elevated TTFT past the 2-second client timeout threshold, triggering retries. With an amplification factor of ~1.56 (40% timeout rate, 3 max attempts), effective load rose from 60% to ~94% of capacity, past the 87% KV cliff defined in Deliverable #7. By the time the burst ended at T+30s, retry-driven load had become self-sustaining. The system remained degraded for an estimated 2-3 minutes after the burst ended because retried requests already in-flight continued consuming KV blocks and iteration slots, preventing KV utilization from dropping below the cliff.

---

## Timeline (Reasoned)

```
T+0:00   Burst injection begins. Base load = 60% KV. TTFT p99 ~2115ms (healthy).
         Amplification factor = 1.0.

T+0:05   Long-context requests begin prefilling. TTFT p99 rises above 2000ms.
         First client timeouts fire. Amplification factor begins rising.

T+0:15   Amplification factor crosses 1.3. Effective load = 60% * 1.3 = 78% KV.
         Queue depth rising. System still below cliff.

T+0:30   Burst injection ends. But retried requests are already in-flight.
         Amplification factor ~1.5. Effective load = 60% * 1.5 = 90% KV.
         System has crossed the 87% cliff.

T+0:45   Amplification factor peaks. KV utilization above cliff. Preemption rate
         rising (>28/min based on Day 24 cliff data). TTFT p99 > 7500ms.
         Divergence ratio p99/p50 > 2.0.

T+1:00   Full cascade. Retries generating more retries. No new burst traffic
         but effective load remains above cliff from retry-driven concurrency.
         Burst ended 30 seconds ago.

T+2:00   System still degraded. In-flight retries draining slowly because
         preemption-driven recompute is consuming 30%+ of forward progress
         (Day 24: 30.1% recompute fraction at cliff).

T+3:00   Estimated recovery. Queue drains below cliff threshold as retried
         requests either complete or exhaust max attempts. KV utilization
         drops below 85%. Preemption rate returns to zero.
```

Recovery lag explanation: three conditions must clear before recovery begins. (1) Retried requests still in-flight must complete or be rejected. These hold KV blocks and iteration slots identical to real requests. (2) The queue is poisoned with backed-up work from the cascade period. Each queued request will consume KV blocks when admitted. (3) KV utilization must drop below the cliff (87%) and stay below it. While conditions 1 and 2 persist, condition 3 cannot be met, so recovery is blocked until the backlog drains.

---

## Amplification Graph (Description)

Dual-axis time series. Y-left: unique request rate (flat at base load, drops to zero when burst ends). Y-right: total request rate (unique + retries). X: time.

The two lines diverge at ~T+5s when first timeouts fire. The gap widens through T+30s. At T+30s, the unique rate drops (burst ends) but the total rate stays elevated because retries are self-generating. The burst-end marker sits left of the peak total-rate marker. The lines don't reconverge until ~T+3:00 when retry backlog drains.

---

## Root Cause

1. Positive feedback loop: elevated latency caused client timeouts, which generated retries, which increased effective load, which elevated latency further.
2. The triggering burst was transient and self-limiting; the retry-driven load was neither.

---

## Why Inference Is Especially Vulnerable

```
Service type                What retries hit             Effect
CPU-bound stateless fleet   Many instances via LB        Retries dilute across fleet
GPU inference (single)      One fixed KV cache pool      Retries concentrate into same
                                                         memory budget
```

The inference-specific consequence: there is no "scale out and absorb retries" path within a single serving instance. The KV pool is the hard ceiling. Every retry competes for the same blocks as the original request and all other in-flight requests. More retries do not find slack capacity; they consume capacity from requests that would have succeeded.

Additionally, GPU inference requests have large temporal surface area. A 2048-token prefill takes ~504ms. A full decode sequence takes seconds. During that entire window, the request is vulnerable to being "timed out" by the client while still actively consuming KV blocks on the server. In a stateless CPU service, request lifetimes are typically milliseconds, minimizing the window during which a timeout can create a duplicate in-flight request.

Cross-reference: the retry cascade pushed the system past the same 87% KV cliff measured in Deliverable #7. The cliff was not a new failure mode. It was the same cliff, reached via a different path (retry amplification rather than sustained high concurrency).

---

## Remediation

Five mechanisms, ordered from most to least impactful.

### 1. Admission control as first defense

The retry storm occurs because requests were admitted, queued, and then timed out at the client. A gateway enforcing KV utilization < 85% (cliff - 2pp) returns 429 immediately. No queue buildup, no timeout window, no retry trigger. This is the primary mitigation, not defense-in-depth.

Counterfactual: at the Day 24 measured cliff (87% KV, c=113 with 512-token prompts), the safe concurrency ceiling is ~95 requests. The cascade required effective concurrency above 113 to cross the cliff. A gateway enforcing the ceiling would have rejected excess requests at admission before the 2-second timeout window could generate the synchronized retry burst.

### 2. 503 + Retry-After (server-side)

When queue depth exceeds threshold: return 503 immediately with `Retry-After: N` header. Do not hold the connection until the client timeout fires. Holding until timeout is what generates the synchronized retry wave. Immediate 503 breaks synchronization because each client gets its rejection at a slightly different time based on connection arrival order.

Effect: reduces amplification factor by eliminating the timeout-driven synchronization window. At a 2-second client timeout, immediate 503 eliminates a 2-second window during which all concurrent requests would otherwise timeout simultaneously and retry in lockstep.

### 3. Exponential backoff with jitter (client-side)

`base=0.5s, max=10s, jitter=+/-50%`. Backoff reduces retry density (retries per second). Jitter desynchronizes retries across clients. Without jitter, all clients that timed out at the same moment retry at the same moment at exponentially increasing intervals (thundering herd at T=1s, T=2s, T=4s instead of T=0, T=0, T=0). Both mechanisms are required.

### 4. Retry budget (client + gateway)

`retry_budget = max(0.1 * successful_requests_per_second, 10)`. Caps system-wide retry rate to ~10% of successful throughput. Self-regulating: high success rate = generous budget, low success rate = budget shrinks automatically. Requires explicit coordination via X-Retry-Count header at the gateway. Bounds maximum amplification factor regardless of cascade depth.

### 5. Circuit breaker

If success rate < 50% over 10-second window, open circuit, all requests fail fast, system gets recovery headroom. Circuit breaker is the recovery mechanism once cascade has already started. Mechanisms 1-4 prevent reaching that state; the circuit breaker limits damage once there.

---

## Narrative

### What broke

The inference system stopped serving requests. From an outside observer's perspective, the service returned errors or timed out for approximately 3 minutes. Latency went from sub-2-second to over 7 seconds. Throughput dropped. No hardware failure, no deployment change, no config modification triggered it.

### What I expected

I expected the 30-second burst to cause temporary degradation while it was active, followed by recovery within seconds after it ended. The system was running at 60% KV utilization with plenty of headroom. A burst should push utilization up, cause some queueing, and then the queue would drain once the burst stopped.

### What surprised me

The system got worse after the burst ended. The retry mechanism converted a 30-second transient overload into a 3-minute sustained cascade. The burst was gone, but the retries it generated were self-sustaining: each retry that timed out generated another retry, maintaining effective load above the cliff independent of external traffic. The timeline gap between "burst ends" and "system recovers" is the core finding. The trigger was not the cause of the sustained failure. The trigger was the ignition source; the retry loop was the fuel.

### What I would change

Add admission control at the gateway enforcing KV utilization < 85%. Based on Day 24 data, this rejects requests before they can enter the 2-second timeout window that triggers retries. The cascade never starts because the precondition (requests sitting in queue long enough to timeout) is eliminated. The throughput cost is 8.6% (803 vs 879 tok/s) which is the explicit tradeoff documented in Deliverable #7.

---

## Decision Section

```
Given:
  Day 24 cliff at 87% KV utilization
  Retry amplification factor of 1.56 (40% timeout rate, 3 max attempts)
  Effective load = base_load * amplification_factor
  A 30-second burst can push effective load past the cliff via retries
  Recovery is blocked until retry backlog drains (estimated 2-3 minutes)

I would choose:
  Deploy admission control (gateway 429 at 85% KV) as primary mitigation,
  combined with client-side exponential backoff + jitter as secondary.

Because:
  AC prevents the retry loop from starting by rejecting before timeout.
  Backoff dampens any retries that do occur (e.g., from transient network
  issues unrelated to KV pressure).

What gets worse:
  Throughput reduced 8.6% (operating at 79% vs 86% KV).
  Some legitimate requests rejected during burst peaks (false positive 429s).

Not optimizing for:
  Peak throughput. Zero-rejection admission. Client simplicity (backoff
  requires client SDK changes).

What would change my decision:
  If retry budget infrastructure already existed at the gateway layer,
  I would rely on budget + backoff without hard KV-based rejection, since
  the budget directly caps amplification factor rather than using KV util
  as a proxy for retry pressure.
```

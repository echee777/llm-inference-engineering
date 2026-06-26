# Day 26 -- Retry Storm Mechanics (Conceptual)

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

---

## Decision: Skip Compute Experiments

The retry storm mechanism is fully deducible from Day 24 cliff data + the geometric series formula. The causal chain (retries add load, load pushes past cliff, cliff causes more timeouts, timeouts cause more retries) is not in question. Running GPU experiments to confirm the obvious is low-value when the time can be spent on higher-leverage AI concepts. The experiments would produce specific numbers (exact amplification factor, exact recovery time) but those numbers are workload-specific and not generalizable to production. The conceptual model is what transfers.

---

## 1. Retry Amplification as Second Feedback Loop

The preemption feedback loop (Day 21/24) is server-internal. The retry loop wraps around it from the client side.

```
Client timeout fires
  -> retry request sent (original still in-flight consuming KV)
  -> server sees 2 requests for 1 user intent
  -> more KV blocks consumed
  -> higher KV utilization
  -> more preemption / cliff behavior
  -> slower responses
  -> more timeouts
  -> more retries
```

The system cannot self-correct because clients have no visibility into server state. They are programmed to retry on timeout regardless of cause.

## 2. Amplification Factor

```
amplification_factor = total_attempts / unique_requests
```

Healthy system: ~1.0 (no retries firing).

With timeout_rate per attempt and max_attempts:

```
amplification_factor = sum(timeout_rate^i for i in range(max_attempts))

Example: 40% timeout rate, 3 max attempts
  = 1 + 0.4 + 0.16
  = 1.56
```

## 3. Effective Load Formula

```
effective_load = base_load * amplification_factor
```

If base load is at 60% of cliff capacity and amplification is 1.56, effective load is 93.6% -- past the cliff. A transient burst that pushes amplification above 1.0 can push effective load past the cliff even when base load is well below it.

## 4. Recovery Is a Drain Problem

When the trigger ends (burst stops, base load drops), the system does not recover immediately. The queue contains retried requests indistinguishable from real requests. Each must be admitted, prefilled, and decoded. If the backlog keeps KV utilization above the cliff, the system stays degraded while processing stale retries. Recovery time depends on backlog depth, not on when the trigger stopped.

## 5. Server Cannot Distinguish Retries

To vLLM's scheduler, a retry is identical to a fresh request. Same payload, same KV allocation, same iteration slot. No server-side component can differentiate retries without explicit client cooperation (e.g., X-Retry-Count header).

## 6. Defense Stack

```
Layer              Mechanism                       Knows about retries?
Client             Exponential backoff + jitter     Yes
Gateway/proxy      Retry budget (X-Retry-Count)     Yes (via headers)
Admission ctrl     KV-based hard rejection (85%)    No (protects by system state)
Server (vLLM)      Scheduler                        No
```

Each layer solves a different part:
- Backoff reduces retry density (retries per second)
- Jitter reduces retry synchronization (thundering herd)
- Retry budget caps system-wide retry rate to ~10% of successful throughput
- AC hard rejection prevents KV utilization from crossing the cliff regardless of cause

## 7. Backoff Policy Components

```
Exponential backoff:  delay = min(base * 2^attempt, max_delay)
                      base=0.5s, max=10s typical

Jitter:               delay += random(-50%, +50%) * delay
                      desynchronizes retries across clients

Max attempts cap:     bounds worst-case amplification to geometric series sum
```

Backoff without jitter creates synchronized waves at longer intervals (thundering herd at T=1, T=2, T=4 instead of T=0, T=0, T=0). Both mechanisms are required.

## 8. Circuit Breaker Pattern

```
CLOSED (normal):     requests flow, track failure rate
OPEN (tripped):      reject all immediately (503), start cooldown timer
HALF-OPEN (probe):   admit one request after cooldown
                     success -> CLOSED, failure -> OPEN with longer cooldown
```

Complements backoff: backoff slows retries but doesn't stop them. If system is past cliff, even slow retries prevent drain. Circuit breaker hard-stops all traffic to allow KV utilization to drop.

The hard rejection at 85% KV in admission-control-retrofit.md is functionally a KV-utilization-triggered circuit breaker.

## 9. Retry Budget

```
retry_budget = max(0.1 * successful_requests_per_second, 10)
```

Self-regulating: high success rate = generous budget, low success rate = budget shrinks. Unlike per-client backoff, this bounds aggregate retry load system-wide. Requires gateway coordination via X-Retry-Count headers or service mesh ownership of retries.

## 10. Detection

Alert on the amplification factor trend, not the absolute value.

```
amplification_factor rising + KV utilization rising = retry storm in progress
amplification_factor stable at 1.0                  = healthy
amplification_factor > 1.0 but stable               = retries present, not cascading
amplification_factor accelerating                    = active feedback loop
```

Production alert: `amplification_factor > 1.3 sustained for 30 seconds`.

## Key Formulas

```
amplification_factor = total_attempts / unique_requests
effective_load       = base_load * amplification_factor
backoff_delay        = min(base * 2^attempt, max_delay) + jitter
retry_budget         = max(0.1 * successful_rps, 10)
```

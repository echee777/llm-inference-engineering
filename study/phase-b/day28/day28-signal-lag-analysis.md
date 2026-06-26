# Day 28 -- Signal Lag Analysis + Scale-Down Hazard

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

---

## Cold-Start Measurement

```
Phase                              Duration
Weights load + CUDA graph + pool   93.7s
First-request warmup               0.5s
Total cold start (t_cold)          94.2s
```

Config: Qwen2.5-3B-Instruct, --dtype half, --max-model-len 4096, --gpu-memory-utilization 0.90, T4.

---

## Signal Lag Experiment

### Setup

Same config as Day 24 cliff sweep: gpu_memory_utilization=0.60, max_num_seqs=160, 512-token prompts, min_tokens=256.

Step-function ramp: c=50 (90s) -> c=95 (90s) -> c=110 (90s) -> c=130 (120s).

### Results

```
Step   Concurrency   KV util   Queue depth   TTFT p99
1      50            17.2%     0             221ms
2      95            33.1%     0             288ms
3      110           4.0%      0             15,841ms
4      130           7.6%      0             16,573ms
```

### Key Finding: Queue Depth Is Not a Leading Indicator With High max_num_seqs

Queue depth stayed at 0 throughout all concurrency levels, including c=130 which produced TTFT p99 of 16.5 seconds. With max_num_seqs=160, the scheduler admits all 130 requests to the "running" set immediately. There is no queueing at the admission boundary.

The degradation happens INSIDE the running set via preemption cycling:
- 130 requests need ~6,370 KV blocks but only 4,651 exist
- Scheduler preempts requests to free blocks, re-prefills them when blocks become available
- Instantaneous KV util oscillates rapidly (scrape catches troughs)
- TTFT explodes because requests are repeatedly preempted and restarted

`num_requests_waiting` captures requests NOT YET ADMITTED to the scheduler. It does NOT capture requests that are admitted but preempted within the running set. With max_num_seqs > offered concurrency, this metric is blind to KV-pressure degradation.

### Implication for Autoscaling Signal Selection

Queue depth is a leading indicator ONLY when max_num_seqs is set low enough to create queueing before KV exhaustion. The production fix:

1. Set max_num_seqs <= sustainable concurrency (e.g., ~95 for this config) so excess requests queue at the admission boundary, making queue depth a real signal
2. OR use admission control at the gateway (Day 25 retrofit) which acts on KV util directly and doesn't depend on vLLM's internal queue forming
3. OR use preemption rate as the signal (but this is lagging, not leading)

### nvidia-smi Trap (Confirmed)

nvidia-smi GPU memory utilization is useless for autoscaling vLLM. vLLM pre-allocates the entire KV block pool at startup. nvidia-smi shows the same memory usage at 0% and 100% KV occupancy. It is a flat constant that will never trigger an HPA threshold.

---

## Kill Observation (SIGTERM Behavior)

### Setup
3 concurrent streaming requests (256 max_tokens, actively decoding ~100 tokens received). Sent SIGTERM to vLLM process.

### Result

```
Request   Tokens received   Client error mode   Duration
0         101               timeout             30.2s
1         101               timeout             30.2s
2         102               timeout             30.2s
```

### Finding

On SIGTERM, vLLM does NOT:
- Immediately close connections (no "connection reset")
- Send a clean EOF or final SSE message
- Return an HTTP error

It simply stops sending data. The TCP connections remain open but silent. Clients wait their full timeout (30s in this test) before detecting failure.

### Implications for Scale-Down Hazard

1. Clients don't get an immediate error signal on pod termination
2. They wait the FULL timeout duration before retrying (delayed retry burst)
3. During that timeout window, the dead pod's "slot" in the load balancer is occupied but producing nothing
4. When retries finally fire, they arrive simultaneously (thundering herd, same as Day 26 analysis)

This is WORSE than an immediate connection reset because it maximizes the timeout window during which client resources are blocked without useful output.

---

## Three Timescales Model

```
Timescale   Value (measured)        Controlled by
Δ           INDETERMINATE           Queue depth signal doesn't work with max_num_seqs=160
            (queue never formed)    Would need max_num_seqs < cliff concurrency

t_cold      94 seconds              Model size, hardware, weight format

t_growth    Incident-specific       Traffic pattern (slow ramp vs sharp spike)
            (= headroom / dQ/dt)
```

### Stability Condition

```
Reactive scale-out viable  iff  Δ > t_cold (94s)  AND  t_growth > t_cold (94s)
```

With t_cold = 94s, the necessary condition requires >94 seconds of signal lead time. Queue depth on this config cannot provide that lead time (it doesn't form a queue at all). Therefore:

**Reactive autoscaling is structurally inadequate for this config.** Pre-warmed replicas are required.

### Coupling Note

Δ and t_growth are positively coupled. An earlier-firing signal (larger Δ) triggers at lower utilization, which means more headroom, which means larger t_growth. Choosing a more sensitive threshold simultaneously improves both conjuncts.

---

## Scale-Down Hazard Statement

If graceful drain is enabled with a 120-second drain window and new-request rejection at 503, the scale-down operation is safe provided all in-flight requests complete within 120 seconds. If drain timeout is exceeded, the remaining N in-flight requests are force-terminated.

Under the Day 26 retry configuration (2s timeout, 3 max retries, 1.56 amplification factor): force-terminating 20 requests produces 31.2 effective retry requests (20 * 1.56). At 1.05% KV per request (512 tokens, 4,651 block pool), this adds ~33pp of KV pressure to surviving replicas. At steady-state 79% KV, effective utilization rises to ~112%, well past the 87% cliff. A single failed drain operation triggers a retry cascade in the surviving pool.

---

## Scale-Up Miss Statement

With Δ indeterminate (queue depth signal does not form on this config) and t_cold = 94 seconds, the necessary condition for reactive autoscaling (Δ > t_cold) cannot be satisfied by queue-depth-based HPA alone.

Mitigation: pre-warmed replica pool (eliminates t_cold from the critical path) + admission control (bounds dQ/dt during spikes, keeping the surviving replica below the cliff while new capacity routes in).

---

## Graceful Drain State Machine

```
State         Enter condition           Exit condition              Actions on enter                    New requests
RUNNING       Process ready             Drain signal received       Normal serving                      Accept

DRAINING      Drain signal              running_requests == 0       1. Remove from LB endpoints         503 + Retry-After
              (autoscaler/SIGTERM)      OR drain_timer > 120s       2. Flip readiness probe unhealthy
                                                                    3. Start drain timer

TERMINATING   Exit from DRAINING        Process exit                1. Force-close remaining streams    Connection refused
                                                                    2. Emit forced_terminations metric
                                                                    3. Exit process
```

Drain window derivation:
```
TTFT_max = 37.6 + 0.228 * 530 = 158ms
decode_time_max = 256 tokens * ~12ms/token = 3,072ms
per_request_max = 158 + 3,072 = 3,230ms (~3.2s)
drain_window = 3.2s * safety_factor(3) + scheduling_overhead = ~10s per request
```

At the safe operating point (c=95), all in-flight requests should complete well within 120 seconds. The 120s bound is a safety valve for edge cases (requests that were preempted mid-decode and need full re-prefill).

---

## Symmetric Failure Pair

```
Failure mode        Timescales involved                     Stability condition
Scale-Up Miss       Δ, t_cold, t_growth                    Δ > t_cold AND t_growth > t_cold
Scale-Down Hazard   drain_window, amp_factor, cliff_gap     force_term * amp * kv_per_req < cliff - operating_point
```

---

## Production Translation

```
Theoretical element        Production infra pattern
Δ (signal lead time)       Custom HPA metrics: vllm_scheduler_waiting_queue_length
                           (requires max_num_seqs tuning to create observable queue)
                           OR gateway admission-control rejection rate as proxy signal

t_cold (scale-out latency) Pre-warmed replica pools (minReplicas > active demand)
                           Model weights on local SSD / shared memory
                           KubeRay: idleTimeoutSeconds tuned high to retain warm replicas

dQ/dt cap                  Admission control at gateway (Day 25 retrofit)
                           Returns 429 when instantaneous demand exceeds provisionable capacity
```

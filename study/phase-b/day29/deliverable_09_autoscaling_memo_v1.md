# Deliverable #9: Autoscaling Strategy Memo

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

---

## Section 1: Why GPU Inference Autoscaling Is Different

```
Dimension             CPU Web Service                GPU Inference
Cold start            seconds                        94.2s measured (weights + CUDA graph + pool) (Day 28)
State                 stateless                      stateful (KV cache mid-request)
Retry effect          dilutes across fleet via LB    concentrates into single KV pool
Correct scaling mode  reactive (cold start < SLO)    proactive / pre-warmed (cold start >> SLO)
```

The structural root is cold start. A CPU web service cold-starts in seconds, so reactive autoscaling (detect pressure, launch replica, route traffic) works. GPU inference cold-starts in 94.2 seconds (Day 28). During that window, the existing replicas must absorb all excess load with no relief. If the load crosses the KV cliff (87%, Day 24), the system enters a self-amplifying degradation loop before the new replica is ready.

The statefulness compounds the problem. Each in-flight request holds KV blocks that cannot be transferred to a new replica. Terminating a replica does not shed load; it destroys accumulated compute and forces clients to retry from scratch, adding load to surviving replicas (Day 28 kill observation).

---

## Section 2: Signal Selection

### Wrong Signals

**CPU utilization.** During KV exhaustion at the Day 24 cliff (87% KV, TTFT p99 = 7,587ms), CPU utilization remained at 20-40% (Day 27). CPU handles request parsing, tokenization, and KV block bookkeeping. The actual inference compute runs on GPU. CPU load correlates with request throughput, not GPU memory pressure. An HPA targeting CPU at 70% would never fire during a KV exhaustion event. Failure mode: no scale-out fires, full collapse under sustained load with autoscaler blind.

**GPU compute utilization.** GPU utilization in any form is the wrong signal because inference degradation under KV pressure is memory-bound. nvidia-smi GPU utilization (percentage of time any kernel is running) reads ~98% in both healthy and degraded states. Even fine-grained SM occupancy metrics from DCGM (DCGM_FI_PROF_SM_ACTIVE) measure the wrong dimension: KV exhaustion is a memory problem, not a compute problem. GPU compute may actually drop during the worst degradation as the scheduler cycles through preemption/re-prefill rather than running inference kernels. Failure mode: lagging trigger at best; scale-out fires after TTFT p99 has already breached SLO.

**nvidia-smi GPU memory utilization.** vLLM pre-allocates the entire KV block pool at startup (Day 28 confirmed). nvidia-smi reports the same memory usage at 0% and 100% KV occupancy. It is a flat constant that will never trigger an HPA threshold. The correct metric is `vllm:kv_cache_usage_perc`, which tracks logical block allocation within the pre-allocated pool.

### Right Signals

**Queue depth (`vllm:num_requests_waiting`) -- leading indicator.** Fires before SLO breach when requests back up at the admission boundary. Critical caveat: queue depth is only a leading indicator when `max_num_seqs` is set below cliff concurrency. Day 28 experiment: with `max_num_seqs=160` and offered concurrency of 130, queue depth stayed at zero throughout the entire cliff event (TTFT at 16,573ms, queue still 0). All requests were admitted to the running set immediately; degradation happened inside the scheduler via preemption cycling, invisible to external queue metrics. Production fix: set `max_num_seqs <= 95` (safe operating point) so excess requests queue at the admission boundary, making queue depth an observable signal.

**KV cache utilization (`vllm:kv_cache_usage_perc`) -- coincident indicator.** Directly measures the resource under pressure. Threshold: 76% (derived below). Caveat: during preemption cycling at/above the cliff, KV utilization oscillates rapidly as blocks are freed and re-consumed. Day 28 measured KV util at 4.0% during c=110 when TTFT was 15,841ms because metric scrapes catch troughs between preemption/re-prefill cycles. Reliable below the cliff as a warning signal; unreliable at or above it.

**TTFT p99 -- lagging indicator (SLO watchdog).** Directly measures SLO compliance with near-zero detection lag: if TTFT p99 > SLO target, the SLO is violated by definition. Useless as a scale-out trigger (users are already suffering when it fires). Essential as a validation signal: confirms whether the leading/coincident signals are actually protecting the SLO. At the Day 24 cliff, TTFT p99 divergence ratio (p99/p50) exceeded 2.0 at 87% KV.

### Composite Policy

```
Precondition: max_num_seqs <= 95 (safe operating point, Day 24)

scale_out iff queue_depth > 10 AND kv_util > 76% for > 30s
```

Threshold derivation:

- Q = 10: 10% of safe operating concurrency (c=95). Filters transient bursts (a spike of 5 requests that clears in seconds) while catching sustained pressure buildup.
- K = 76%: derived from cliff (87%) minus the KV headroom needed for Q queued requests. Each request consumes ~1.05% KV (530 tokens / 4,651 block pool, Day 24/28). 10 queued requests = 10.5% KV. 87% - 10.5% = 76.5%, rounded to 76%. When the trigger fires at KV=76% with 10 queued, worst-case admission pushes KV to ~86.5%, just under the 87% cliff.
- 30s sustained: filters transient noise without burning lead time. Combined with pre-warmed replica (t_cold = 0), total response time is 30s from first signal to traffic reroute.

Signal roles in the composite:

```
Signal            Role                  Justification
queue_depth > 10  leading trigger       pressure building at admission boundary
kv_util > 76%     coincident confirm    validates pressure is structural, not transient
TTFT p99 > SLO    watchdog              catches failures in the other two signals
```

---

## Section 3: Scale-Out Policy

### Trigger Budget Inequality

For reactive autoscaling to work:

```
t_headroom >= t_cold

where:
  t_headroom = time from trigger fire to SLO breach
  t_cold     = 94.2s (Day 28 measured cold start)
```

This inequality fails for this configuration. When the composite trigger fires (queue > 10 at KV 76%), 11pp of KV headroom remains before the 87% cliff. At the Day 28 step transition (c=95 to c=110), 15 additional concurrent requests pushed the system from healthy to full collapse. Each request admission takes milliseconds. 10 queued requests represent two-thirds of the 15 that caused collapse. The remaining 11pp of KV headroom is consumed in seconds, not the 94 seconds required by the inequality.

Therefore: **reactive autoscaling is structurally inadequate for this configuration. Pre-warmed replicas are required.**

### Pre-Warm Policy

Maintain one hot standby replica at all times. This eliminates t_cold from the critical path:

```
t_headroom >= t_cold
t_headroom >= 0s   (standby already warm)
-> inequality satisfied trivially
```

When the standby absorbs a burst, immediately begin cold-starting a replacement standby to replenish the pool. The system is exposed for 94.2s until the replacement is ready.

### Increment Policy

+1 replica per scale-out event. The queued 10 requests represent ~10% of one replica's safe capacity (c=95), so a single additional replica absorbs the overflow with substantial headroom. Over-provisioning (+N) creates a scale-down problem: more replicas to drain later, each drain carrying the hazard analyzed in Section 4.

---

## Section 4: Scale-Down Policy

### The Hazard: Naive Termination

Day 28 kill observation: on SIGTERM, vLLM does not immediately close connections, send a clean EOF, or return an HTTP error. Connections hang silently. Clients wait their full timeout (30s measured) before detecting failure. This is worse than an immediate connection reset because:

1. Clients don't get an immediate error signal. They block for 30s producing nothing.
2. When timeouts fire, all retries arrive simultaneously (thundering herd).
3. During the timeout window, the dead pod's slot in the load balancer is occupied but non-functional.

Quantified cascade: force-terminating 20 in-flight requests produces 31.2 effective retry requests (20 * 1.56 amplification factor, Day 26). At 1.05% KV per request, this adds ~33pp of KV pressure to surviving replicas. At steady-state 79% KV, effective utilization rises to ~112%, well past the 87% cliff (Day 28 analysis). A single failed drain operation triggers a retry cascade in the surviving pool.

### Graceful Drain Sequence

```
State         Enter condition           New requests     Actions
RUNNING       Process ready             Accept           Normal serving

DRAINING      Drain signal received     503 + Retry-After
              (autoscaler/SIGTERM)                       1. Remove from LB endpoints
                                                         2. Flip readiness probe unhealthy
                                                         3. Start drain timer

TERMINATING   running_requests == 0     Refused          1. Force-close remaining streams
              OR drain_timer > 120s                      2. Emit forced_terminations metric
                                                         3. Exit process
```

### Drain Window Derivation

```
TTFT_max       = 37.6 + 0.228 * 530 = 158ms   (Day 6 regression, 530 prompt tokens)
decode_max     = 256 tokens * 12ms/token = 3,072ms
per_request_max = 158ms + 3,072ms = 3,230ms (~3.2s)

drain_window   = per_request_max * safety_factor(3) = ~10s (soft target)
hard_kill      = 120s (safety valve for edge cases: preempted + requeued requests
                 needing full re-prefill under prefill/decode interference)
```

The safety factor of 3 covers prefill/decode interference inflating per-request time under concurrent load. The base estimate (3.2s) assumes clean execution; under drain conditions with multiple in-flight requests, concurrent prefills steal GPU time from active decodes, extending completion time.

At the safe operating point (c=95), all in-flight requests complete well within 120s. The 120s hard kill is a safety valve, not a routine expectation.

### Hysteresis

```
scale_out:  queue > 10 AND KV > 76% for 30s    (aggressive detection)
scale_down: queue = 0 AND KV < 10% for 10min   (patient confirmation)
```

The asymmetry reflects asymmetric risk. Scale-out mistakes cost money (one unnecessary replica for a few minutes). Scale-down mistakes cost availability (retry cascade, potential multi-replica collapse). 10 minutes of sustained low utilization confirms the traffic reduction is structural, not a transient dip.

---

## Section 5: Predictive Scaling

For workloads with diurnal traffic patterns, pre-warm replicas before predicted demand peaks rather than waiting for the composite signal to fire.

The cost-vs-reliability tradeoff becomes a business input, not a technical one: pre-warm cost (idle replica-hours) vs SLO miss cost (breached latency targets during the 30s confirmation window). An organization with strict SLO penalties pre-warms aggressively; one optimizing for cost tolerates occasional confirmation-window degradation.

This policy is not experimentally validated on this rig. Day 28 traffic patterns were synthetic step functions, not diurnal. The recommendation is based on the structural argument: with t_cold = 94.2s, any traffic spike faster than t_cold requires pre-positioned capacity. Predictive pre-warming converts that requirement from reactive (signal-driven, 30s response) to proactive (schedule-driven, 0s response).

---

## Decision Section

```
Given:
  - Qwen2.5-3B-Instruct on T4/g4dn.xlarge, single-replica baseline
  - t_cold = 94.2s (Day 28 measured)
  - KV cliff at 87% utilization, divergence ratio > 2.0 (Day 24)
  - Safe operating point: 79% KV, c=95, TTFT p99 = 2,115ms (Day 24)
  - Queue depth signal requires max_num_seqs <= 95 (Day 28)
  - SIGTERM causes silent connection hang, 30s client timeout (Day 28)
  - Retry amplification factor 1.56 at 40% timeout rate (Day 26)
  - Per-request KV cost: ~1.05% of pool (530 tokens / 4,651 blocks)

I would choose:
  Composite scale-out: queue_depth > 10 AND kv_util > 76% for > 30s
  Pre-warmed hot standby (eliminates t_cold from critical path)
  Graceful-drain scale-down with hysteresis (10min patience)
  +1 increment policy

Because:
  - Queue depth is a leading indicator that fires before SLO breach,
    but only with max_num_seqs constrained to create observable queueing (Day 28)
  - KV util as coincident confirmation filters transient bursts: queue > 10
    at KV 40% is not dangerous (headroom to absorb), queue > 10 at KV 76% is
    structural pressure approaching the cliff
  - Reactive scaling fails: t_headroom (seconds) << t_cold (94.2s). The
    trigger-to-cliff gap is consumed by ~10 request admissions at millisecond
    speed, not the 94s needed for cold start
  - Graceful drain prevents re-entry into retry cascade: 20 force-terminated
    requests * 1.56 amplification * 1.05% KV = 33pp added to survivors,
    pushing from 79% to 112%, well past cliff (Day 28 + Day 26)
  - Operating below cliff at 79% KV avoids divergence-ratio blowup while
    accepting 8.6% throughput reduction vs peak at 86% KV (Day 24)

What gets worse because of this decision:
  - Cost: hot standby increases fleet cost by 100/N%, where N = number of
    active replicas. Single-replica baseline = 100% cost overhead (2 replicas
    serving the load of 1).
  - Latency: 30s confirmation window where queued users experience elevated
    TTFT while the composite trigger confirms the signal is sustained, not
    transient. Users in this window see degraded but not catastrophic latency.

What I am explicitly NOT optimizing for:
  - Cost floor: accepting cost premium for SLO reliability
  - Burst-response latency minimization: not targeting sub-second scale-out;
    accepting 30s confirmation window as the tradeoff for false-positive filtering
  - Peak throughput: operating at 79% KV (803 tok/s) rather than 86% KV
    (879 tok/s) to maintain cliff margin

What would make me change this decision:
  - If t_cold drops to <10s (e.g., model preloading on local SSD, weight
    sharing across replicas), reactive scaling becomes viable and the hot
    standby cost can be eliminated
  - If workload shifts to steady-state with no diurnal pattern, predictive
    pre-warming loses value and the standby-only approach suffices
  - If chunked prefill reduces KV jitter (Day 23: 4.57x -> 2.13x interference
    reduction), the 76% threshold can move closer to the cliff, improving
    utilization
  - If retry budget infrastructure exists at the gateway, amplification factor
    is bounded independently and the drain hazard severity is reduced
```

---

## Track 1 Narrative

KV cache exhaustion is the central failure mode of GPU inference under load. When all KV blocks are consumed, the scheduler preempts running requests to free blocks, but preempted requests re-enter the queue and re-compete for the same blocks they just released, creating a self-amplifying feedback loop (Deliverable #5). The latency-vs-utilization curve (Deliverable #7) makes this visible: TTFT p99 remains stable up to 87% KV utilization, then divergence ratio explodes past 2.0 as preemption becomes self-sustaining. Prefill/decode interference (Deliverable #6) compounds the problem: concurrent prefills steal GPU time from active decodes, causing all requests to hold KV blocks longer and lowering the effective cliff point. The system is non-linearly fragile: each mechanism is manageable in isolation, but together they create a cliff that is sharper and lower than any single factor would predict.

Understanding the cliff drives every production mitigation. The retry cascade (Deliverable #8) demonstrates what happens when the cliff is crossed without protection: client retries amplify load by 1.56x, converting a transient 30-second burst into a 3-minute sustained collapse because retried requests are indistinguishable from real ones and hold KV blocks identically. This is why graceful drain matters in the autoscaling policy (Deliverable #9): naive termination of a replica force-terminates 20 requests whose retries add 33pp of KV pressure to survivors, pushing them past the same cliff. The composite autoscaling signal (queue depth > 10 AND KV > 76%) derives its thresholds directly from the cliff: 76% is the cliff (87%) minus the KV headroom needed for 10 queued requests at 1.05% each. The entire policy, from signal selection to drain window to hysteresis, is parameterized by the cliff measurement. Change the cliff and every downstream number changes with it.

---

## References

```
Day 6:   TTFT regression: TTFT = 37.6 + 0.228 * prompt_tokens
Day 24:  KV cliff at 87%, safe operating point 79% (c=95), TTFT p99 = 2,115ms
         Preemption onset at 84% KV (c=103)
         Peak throughput 879 tok/s at 86% KV (c=108)
         Divergence ratio > 2.0 at cliff
         KV jitter +/-15pp at fixed concurrency
Day 25:  Admission control retrofit: hard rejection at 85% KV
Day 26:  Retry amplification factor 1.56 (40% timeout, 3 max attempts)
         Effective load = base_load * amplification_factor
Day 28:  Cold start t_cold = 94.2s
         SIGTERM behavior: silent hang, 30s client timeout
         Queue depth blind with max_num_seqs=160 (stayed at 0 through cliff)
         KV util unreliable at cliff (read 4.0% during 15,841ms TTFT)
         Scale-down hazard: 20 * 1.56 * 1.05% = 33pp KV cascade
         Graceful drain state machine: RUNNING -> DRAINING -> TERMINATING
```

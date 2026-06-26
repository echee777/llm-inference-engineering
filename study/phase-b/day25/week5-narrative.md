# Week 5 Connection Narrative

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

---

## Unification Frame

All three Week 5 findings are instances of the same problem: memory scheduling under contention. KV cache exhaustion, prefill/decode interference, and the utilization cliff are not three separate failure modes. They are three manifestations of what happens when memory allocation, scheduling fairness, and system stability interact under load. vLLM V1 schedules memory-bound and compute-bound work through a single shared resource path, and every failure mode measured this week is a consequence of that coupling.

---

## Three Findings

### 1. KV cache is the hard ceiling, not compute (Day 21)

Under concurrent long-context load, KV blocks exhaust before compute saturates. At c=6 with 2048-token prompts, KV hit 92.5% and preemption began. Throughput halved from 101.6 to 50.7 tok/s at c=8 and stayed flat through c=32. The system entered a positive feedback loop: each preempted request discarded ~2,258 tokens of computed KV, re-entered the queue, and re-consumed the same blocks when rescheduled. The failure is memory-mediated, not queue-mediated. vLLM V1's recompute-only preemption means there is no CPU swap valve to buffer evicted requests.

### 2. Mixed request lengths create latency unfairness invisible in standard metrics (Day 23)

A 2048-token prefill blocks decode steps for co-scheduled short requests for ~504ms per iteration. Short-request TTFT p99 degraded 4.57x when co-scheduled with long requests under non-chunked prefill, dropping to 2.13x with chunked prefill enabled. This occurred at 3-16% KV utilization, well below any capacity alarm. GPU utilization read 98.5% in both mixed and isolated runs. Aggregate throughput was identical. The degradation is invisible to standard monitoring and only surfaces in per-request-type TTFT percentile tracking.

### 3. There is a non-linear utilization cliff that makes headroom non-negotiable (Day 24)

Above 87% KV utilization, p99 TTFT diverges non-linearly from p50. Operating 1.7 percentage points higher (85.6% to 87.3% KV) didn't cost 2% more latency. It cost 2.4x p99 TTFT (3200ms to 7587ms). Throughput peaked at 879 tok/s before the cliff and dropped to 786 at the cliff, never recovering. The cliff is the point where preemption-driven recompute work becomes a non-trivial fraction of forward progress (30.1% at the cliff, 74% past it). Adding load past this point reduces throughput, not just increases latency.

---

## Explicit Tradeoff

We operate at 79% KV utilization, which costs 8.6% throughput (803 vs 879 tok/s), because at 87% our p99 TTFT jumps 3.6x and throughput drops. We are not leaving capacity on the table. Past the cliff, there is no additional throughput to capture. The cliff graph is the evidence when leadership asks why we're not at 85%.

---

## Failure Interaction Table

```
Condition                        Isolated effect                    Combined effect
-------------------------------  ---------------------------------  ----------------------------------
High KV utilization only         Preemption onset, p99 begins       --
                                 rising. Throughput halves at
                                 exhaustion (Day 21: 101->50 tps)

Mixed request lengths only       Short-request TTFT degrades        --
                                 4.57x (non-chunked) at 3-16%
                                 KV. Pure scheduling unfairness.

Both simultaneously              Interference holds blocks          Cliff point shifts lower
                                 longer, raising effective KV       than homogeneous
                                 utilization. Preemption injects    measurement. 87% cliff
                                 more prefill work, extending       under homogeneous load
                                 starvation windows.                would be reached earlier
                                                                    under mixed traffic.

All three + retry clients        Retries add artificial             Compound failure: system
                                 concurrency proportional to        enters closed-loop
                                 failure severity. Each retry       instability faster with no
                                 consumes KV blocks and             self-stabilization. Retry
                                 iteration slots identical to       amplification converts a
                                 a real request. The retry          recoverable overload into
                                 client has no signal that the      a sustained one.
                                 system is in the unstable
                                 regime.
```

---

## Compound Failure Crystallization

At high utilization, prefill/decode interference increases effective concurrency, which accelerates KV exhaustion, which triggers recompute preemption, which re-consumes KV blocks and further increases concurrency, forming a closed-loop instability where the three failure modes amplify each other.

---

## The Production Argument

A static token budget admission gate (Phase A design) addresses KV exhaustion at design time but cannot respond to interference or the cliff in real time. It admits requests based on estimated KV consumption at the gateway. It cannot see that interference is raising effective utilization above what the token math predicts. It has no feedback signal when the system enters the unstable regime. The retrofit: switch the primary admission signal from static token budget to real-time `vllm:gpu_cache_usage_perc`, add tiered thresholds by request length, and hard-reject at the cliff boundary.

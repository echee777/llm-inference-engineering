# Autoscaling Strategy Memo (Deliverable #9 Skeleton)

Hardware: T4 (g4dn.xlarge) | Model: Qwen2.5-3B-Instruct | vLLM 0.17.1 (V1 engine)

---

## Section 1: Why GPU Inference Autoscaling Is Different

Claim: Model cold start takes ~90s on T4, making reactive autoscaling too slow for latency-sensitive workloads.
Data needed: model load time (measure or estimate from Day 21 startup logs) -- HAVE

Claim: KV cache is stateful mid-request. Terminating a pod drops all in-flight KV state. Unlike stateless services, you cannot terminate and reroute without wasting all accumulated compute.
Data: mechanical argument -- HAVE

Claim: The failure mode is memory exhaustion (KV cliff), not compute saturation. Standard CPU/GPU utilization signals do not detect it.
Data: Day 24 cliff data + Day 23 GPU utilization observation (98.5% in both healthy and degraded) -- HAVE

---

## Section 2: Wrong Signal -- CPU Utilization

Claim: CPU utilization stays at ~30-40% while KV cache is at 87% and TTFT p99 is 7587ms. CPU-based HPA at 70% would never fire during a KV exhaustion event.

Argument: At the Day 24 cliff (87% KV, TTFT p99 = 7587ms, throughput dropping), CPU utilization remained low because CPU only handles tokenization and scheduling. GPU compute utilization read 98.5% but was spending 30% of forward passes on recompute work from preempted requests. Neither CPU nor GPU utilization distinguished healthy from degraded state. A CPU-based HPA threshold at 70% would not have fired. A GPU-utilization-based HPA would see 98.5% in both states. The correct signal must reflect pending work (queue depth) or memory pressure (KV utilization), not compute busyness.

Data: Day 23 GPU utilization observation + Day 24 cliff data -- HAVE

---

## Section 3: Right Signals -- Comparison Table

```
Signal                          Type       Leads cliff by    Limitation
vllm:num_requests_waiting       Leading    seconds           Noisy under burst traffic
vllm:gpu_cache_usage_perc       Lagging    --                Subject to scheduler jitter (+-15pp)
CPU utilization                 Irrelevant --                Decoupled from memory bottleneck
GPU compute utilization         Irrelevant --                98.5% in both healthy and degraded
```

Claim: Queue depth leads KV utilization by N seconds because requests queue before being admitted and allocated KV blocks.
Data needed: Day 28 trigger lag measurement -- DO NOT HAVE

---

## Section 4: Scale-Out Policy

```
Scale-out:
  condition: vllm:num_requests_waiting > 5
             AND rate(vllm:num_requests_waiting[90s]) > 0
             sustained for 60s
  action: add 1 replica
```

Claim: 60s sustained window + 90s model load = ~2.5 minutes from first queue pressure to new replica ready. This beats the cliff onset timeline.
Data needed: Day 28 threshold validation -- DO NOT HAVE

---

## Section 5: Scale-Down Hazard + Drain Policy

Claim: Naive termination drops in-flight requests, wasting all accumulated KV state and triggering client retries that load remaining pods.

```
Scale-down:
  condition: vllm:num_requests_waiting == 0
             AND vllm:gpu_cache_usage_perc < 10%
             sustained for 10 minutes
  action:
    1. Remove pod from service endpoints (stop admission)
    2. Drain: wait for running requests to complete (max 120s)
    3. Terminate pod
```

Claim: During a retry storm, naive termination amplifies the cascade by adding terminated-request retries to already elevated retry load.
Data: mechanical argument from Day 27 conceptual analysis -- HAVE

Data needed: Day 28 graceful drain experiment (what breaks under naive termination) -- DO NOT HAVE

---

## Decision Section (Template)

```
Given:
  [all data from sections 1-5]

I would choose:
  Composite signal: queue depth (leading) + KV utilization (confirming)
  Scale-out threshold: queue > 5 sustained 60s with positive gradient
  Scale-down: 10-minute cooldown with drain policy

Because:
  [queue depth leads KV by N seconds -- fill from Day 28]
  [CPU/GPU utilization are decoupled from the actual bottleneck]

What gets worse:
  Cost: maintaining warm replicas during cooldown period
  Complexity: custom metrics pipeline (Prometheus -> HPA adapter)

Not optimizing for:
  Minimum replica count. Fast scale-down.

What would change my decision:
  [fill after Day 28 experiments]
```

---

## Data Gap Summary

```
Data point                                    Source          Status
Model cold start time                         Day 21 logs     HAVE (estimate)
CPU decoupled from KV bottleneck              Day 23/24       HAVE
KV cliff location and behavior                Day 24          HAVE
Queue depth trigger lag vs KV trigger lag      Day 28          NEED
Graceful drain vs naive termination behavior   Day 28          NEED
Composite signal threshold validation          Day 28          NEED
```

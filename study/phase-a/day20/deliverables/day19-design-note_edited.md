# Admission Control Design Note

Portfolio Deliverable #4, Phase A

---

## Section 1: Admission as KV Memory Budget

KV cache is the scarce resource behind inference serving. Admission control gates on it directly by maintaining a running token budget that tracks estimated KV consumption across all in-flight requests.

### Derivation: GPU VRAM to Token Capacity

Starting from hardware and working down to admission budget:

```
GPU VRAM (T4):                    ~15 GB usable
  x gpu_memory_utilization (0.9): ~13.5 GB (vLLM's total allocation cap)
  - model weights (Qwen2.5-3B FP16): ~6.5 GB
  - activations + runtime overhead:   ~1.0 GB
  = KV cache space:                   ~6.0 GB
```

gpu_memory_utilization is applied to total VRAM, not after subtracting weights. It caps everything vLLM is allowed to use, reserving 10% for CUDA context and temporary allocations.

### Per-Token KV Cost

For Qwen2.5-3B with Grouped Query Attention:

```
per_token_kv = 2 (K+V)
             x 2 bytes (FP16)
             x 36 layers
             x 2 KV heads (GQA, not 16 attention heads)
             x 128 head_dim
             = 36,864 bytes
             = 36 KiB per token
```

The GQA correction matters: using the full 16 attention heads instead of 2 KV heads would overestimate by 8x.

### Total Token Capacity

```
~6.0 GB / 36 KiB per token ≈ ~170,000 tokens  (manual derivation)

vLLM reported capacity: 217,312 tokens  (authoritative number)
```

The manual derivation underestimates by ~22% due to imprecise overhead estimates in the back-of-envelope VRAM subtraction. Use 217,312 as the true capacity.

### Fragmentation Floor

KV cache allocates in fixed-size blocks of 16 tokens (PagedAttention default). A request consuming 17 tokens occupies 2 blocks (32 tokens of memory). The gateway's token budget is a conservative upper bound, not an exact capacity ceiling.

Block-level waste per request: `(16 - (tokens mod 16)) mod 16` tokens. For short requests this overhead is proportionally larger. Consistent with the Day 3 fragmentation analysis.

### Admission Budget

```
ADMISSION_BUDGET = KV_CAPACITY_TOKENS x TARGET_UTILIZATION
                 = 217,312 x 0.65
                 = 141,252 tokens
```

The 35% headroom absorbs token estimation error (requests charged at max_tokens but may generate fewer) and buffers against fragmentation. The 65% target was calibrated empirically (see Section 5).

---

## Section 2: Why Token-Aware Beats Request-Count

Consider a concurrency cap of N=10 requests:

```
Scenario A: 10 requests, each prompt=50 + max_tokens=50
  Total KV demand: ~1,000 tokens
  Budget utilization: 0.5% of 217K capacity
  Result: 99.5% of KV cache sits idle

Scenario B: 10 requests, each prompt=8192 + max_tokens=4096
  Total KV demand: ~122,880 tokens
  Budget utilization: 57% of capacity
  11 concurrent requests of this size would exceed capacity → OOM risk
```

Same cap. Opposite failures. Request-count has no relationship to the resource it protects.

Token-aware admission budgets in the same unit as KV cache allocation. Each request is charged `prompt_tokens + max_tokens` against a shared pool. The budget naturally adapts to heterogeneous request sizes.

```
Admission check (one line):
  if active_tokens + (prompt_tokens + max_tokens) > ADMISSION_BUDGET:
      reject or queue
```

---

## Section 3: Token Budget Correction

The gateway charges `prompt_tokens + max_tokens` at admission time, a worst-case estimate. Two sources of overestimation:

- max_completion_tokens overcharge: requests charged for full max_tokens but completions often finish early. Day 17 reconciliation measured +154% overestimation.
- Prefix caching: vLLM deduplicates KV cache for shared prompt prefixes, but the gateway charges full prompt_tokens per request. Day 17 reconciliation measured +439% overestimation when active.

### Policy B: Progressive Release

Policy B corrects the max_completion_tokens overcharge by releasing excess budget as tokens stream back:

```
Every RELEASE_INTERVAL (50) generated tokens:
  remaining_needed = max_tokens - tokens_generated
  safe_remaining = remaining_needed + SAFETY_MARGIN (64)
  releasable = currently_reserved - safe_remaining
  if releasable > 0: release to pool
```

The SAFETY_MARGIN (64 tokens) prevents releasing too aggressively during coordinated burst releases.

### Measured Impact

Policy B's value is workload-dependent:

```
Day 17 experiment (short completions, ~50-100 tokens):
  Budget freed by Policy B: 77 tokens across 46 requests (~1.7 tokens/request)
  Admission rate improvement: negligible
  Why: completions finish before the first release checkpoint at ~100 tokens

Day 18 load test (long completions, 1500-2000 tokens via min_tokens):
  Budget freed by Policy B: 11,672 tokens across 100u controlled test
  Requests hit the 50-token release checkpoint ~30 times each
  Result: meaningful capacity recovery during sustained load
```

Policy B is not an optimization. It is the necessary correction that makes the token budget a useful proxy for KV usage. Without it, the budget overestimates by 2-5x and rejects traffic the GPU could serve. The 65% TARGET_UTILIZATION covers remaining errors (prefix caching, fragmentation) that Policy B does not address.

---

## Section 4: Adversarial Resilience

### Large-Request Starvation (Experiment 1)

Tenant A sends 5 short requests (prompt=100, max_tokens=100 each, ~1,000 tokens total). Tenant B sends 1 massive request (prompt=8192, max_tokens=4096, ~12,288 tokens). Budget set to 15,211 tokens to force contention.

All requests admitted in every case (13.3K < 15.2K budget). Starvation manifested as TTFT degradation, not rejection:

```
| Case | Order       | Tenant A avg TTFT | Tenant B TTFT | Budget peak |
|------|-------------|-------------------|---------------|-------------|
| A    | Large first | ~2,019ms          | 2,319ms       | ~79%        |
| B    | Small first | ~102ms            | 140ms         | ~7%         |
| C    | Interleaved | ~110ms            | 165ms         | ~79%        |
```

Case A: Tenant A's TTFT was 20x worse than Case B. The large prefill (8192 tokens) monopolized the GPU, forcing small requests to wait despite being admitted. Token budget reflects memory availability but has no visibility into compute contention from large prefills.

Arrival order is a policy lever. Identical total demand produced 20x TTFT differences. Token budget controls capacity but not its distribution.

### Head-of-Line Blocking (Experiment 2)

Budget set to 10,000 tokens. Large request (12,288 tokens) exceeds total budget. 10 short requests (200 tokens each) arrive behind it.

```
LARGE_8K:  rejected (queue timeout after 5.0s)
SHORT_0-9: all admitted directly, TTFT 93-140ms
```

Short requests passed try_admit (200 < 10,000) without entering the queue. The large request sat in the queue and timed out. This did not demonstrate HOL blocking because the shorts never queued.

However, the convoy effect is structurally present in _drain_queue: it processes FIFO and stops if the head request cannot be admitted. Any scenario with both large and small requests queued would exhibit blocking. The FIFO design was intentionally naive to motivate priority queuing in Phase C.

### Why FIFO Is Insufficient

FIFO creates two failure modes:
- Strict FIFO: convoy effect. Small requests starved behind large ones.
- Skip-ahead (shortest-job-first): large request starvation. Short requests continuously fit in budget, large requests never admitted.

The balanced solution requires priority with aging, or per-tenant budget caps. Both are Phase C scope.

### Limitations

1. Upfront charge vs gradual growth. Budget charged at admission as `prompt_tokens + max_tokens`, but KV memory grows one token per decode step. Policy B partially corrects this but cannot eliminate the gap.
2. Prefill/decode compute contention. Budget treats all tokens as equivalent, but prefill is compute-bound (high arithmetic intensity) and decode is memory-bandwidth-bound (AI ~1 FLOP/byte). Identical budget cost can produce completely different compute pressure. The budget tracks memory, not compute.
3. Scheduler independence. vLLM's scheduler controls batch composition, preemption, and decode prioritization independently. The gateway has no visibility into scheduler state. Two separate control loops with no shared state.
4. Recompute-only preemption. vLLM V1 has no swap path. Recompute cost scales with prompt length, so preemption disproportionately penalizes long-context requests under memory pressure.

Admission control prevents collapse. It does not guarantee efficiency or fairness. Full control requires scheduler-level integration.

---

## Section 5: Queue Depth to Latency Curves

Experiment 3 swept traffic load from 25% to 90% of budget utilization using the Day 18 Locust traffic mix (141K token budget, Policy B enabled). Each load level: 6 minutes (3 min warmup + 3 min recording).

```
| Load Level | Admitted | Rejected | TTFT p50 | TTFT p95  | TTFT p99    |
|------------|----------|----------|----------|-----------|-------------|
| 25%        | 161      | 0        | ~115ms   | ~175ms    | ~185ms      |
| 40%        | 222      | 0        | ~135ms   | ~195ms    | ~210ms      |
| 50%        | 308      | 0        | ~155ms   | ~195ms    | ~260ms      |
| 60%        | 391      | 0        | ~165ms   | ~215ms    | ~250ms      |
| 70%        | 443      | 6        | ~180ms   | ~250ms    | ~275ms      |
| 80%        | 717      | 273      | ~200ms   | ~1,260ms  | ~3,980ms    |
| 90%        | 706      | many     | ~710ms   | ~4,100ms  | ~5,050ms    |
```

### Shape

TTFT p50 rises gradually from 115ms to 180ms across 25-70% (1.6x over a 3x load increase). At 80%, p99 explodes to ~4s. At 90%, everything collapses.

### p99 Divergence

Below 70%, p99 tracks p50 within ~1.4x. At 70%, ratio is 1.5x with first rejections (6 total), confirming budget saturation onset. At 80%, the ratio explodes to ~20x (200ms vs 3,980ms).

### Operating Point

Recommended: 60-65% budget utilization.
- 5-10 points of headroom below the divergence threshold
- p99 TTFT under 260ms
- Zero rejections under steady-state load

The original 65% TARGET_UTILIZATION aligns with this empirical finding: upper bound of the stable operating zone.

---

## Section 6: With vs Without Admission Control

Day 18 load tests ran identical traffic against the gateway in two modes: admission disabled (baseline) and admission enabled (controlled).

### 50 Users

```
                       Baseline (AC OFF)    Controlled (AC ON)
TTFT p50               ~160ms               ~160ms
TTFT p99               ~210-230ms           ~210-270ms
Rejections             0                    0
Preemptions            0                    0
```

Both modes identical. System well within capacity. AC has no effect.

### 100 Users

```
                       Baseline (AC OFF)    Controlled (AC ON)
TTFT p50 (short)       210ms                210ms
TTFT p50 (long)        220ms                2,400ms
TTFT p95               260-290ms            4,600-4,900ms
TTFT p99               340-570ms            5,100-5,200ms
Rejections             0                    678 (32%)
```

AC hurt latency. The 65% budget was too conservative for this hardware/model combination. Qwen 3B has generous KV capacity (217K tokens) relative to load. No preemptions occurred even without AC. The gateway queued and rejected 32% of traffic that would have been served fine.

### 200 Users

vLLM crashed in both modes from process-level overload (connection handling, internal memory), not KV exhaustion. AC cannot protect against this failure class.

### GPU Efficiency Note

Overly conservative admission also imposes a throughput cost: fewer admitted sequences means smaller batch sizes, which reduces arithmetic intensity and moves the workload deeper into the memory-bandwidth-bound regime (from Day 2 roofline: decode at AI=1 FLOP/byte, far below the T4's ridge point of 203 FLOP/byte). The goal is to hold the system at the highest utilization that keeps p99 TTFT within SLO. The hockey-stick curve (Section 5) gives the empirical answer: ~60-65% on this T4/Qwen2.5-3B setup.

---

## Section 7: Dashboard

Grafana dashboard panels from Day 18, connected to Prometheus scraping the gateway (/prom) and vLLM (/metrics) endpoints.

```
Panel                        Metric                           Operator Action
─────────────────────────────────────────────────────────────────────────────────
Token Budget Utilization %   gateway_token_budget_used_pct    Primary signal. If sustained
                                                              >65%, tighten admission or
                                                              scale replicas.

Gateway TTFT (p50/p95/p99)   gateway_ttft_seconds             SLO metric. p99 divergence
                                                              from p50 is the early warning.

Model TTFT                   model_ttft_seconds               Isolates vLLM latency from
                                                              gateway queue wait. If model
                                                              TTFT rises while gateway TTFT
                                                              is flat, the model backend is
                                                              under pressure.

Queue Wait                   gateway_queue_wait_seconds       Leading indicator. Spikes
                                                              before TTFT spikes.

Queue Depth                  gateway_queue_depth              If sustained >0, budget is
                                                              saturated and requests are
                                                              backing up.

Request Disposition          gateway_requests_total            Rate of admitted vs rejected.
                             (by status label)                Rising rejection rate means
                                                              load exceeds capacity.

vLLM Waiting Sequences       vllm:num_requests_waiting        Scheduler-level contention.
                                                              Creep here with zero gateway
                                                              queue depth means scheduler
                                                              batches are filling up.

GPU KV Cache Usage           vllm:gpu_cache_usage_perc        Ground truth KV utilization.
                                                              Compare against gateway's
                                                              budget estimate to measure
                                                              estimation accuracy.
```

Gap between Gateway TTFT and Model TTFT equals queue wait plus gateway overhead. If Gateway TTFT rises but Model TTFT stays flat, the problem is queueing. If both rise, the model backend is under pressure.

---

## Section 8: Architecture Diagram

```
Client
  │
  ▼
Gateway (FastAPI, port 8001)
  │
  ├─ Tokenizer (count prompt tokens using model's tokenizer)
  │
  ├─ Rate Limiter (per API key, sliding window)
  │    └─ 429 if exceeded
  │
  ├─ Admission Check (token budget)
  │    ├─ active_tokens + estimated_cost > ADMISSION_BUDGET?
  │    ├─ YES → FIFO Queue (max 50, 5s timeout)
  │    │         ├─ Queue full → 503
  │    │         └─ Timeout → 503
  │    └─ NO → Admit
  │
  ├─ [TRUST BOUNDARY] ─────────────────────────────────
  │    The gateway trusts vLLM to handle admitted requests correctly.
  │    vLLM's scheduler operates independently: it controls batch
  │    composition, preemption, and decode prioritization with no
  │    coordination with the gateway. The gateway cannot enforce
  │    latency guarantees past this boundary.
  │
  ▼
vLLM V1 Engine (port 8000)
  │
  ├─ Scheduler (iteration-level batching, preemption)
  │
  ├─ KV Cache (PagedAttention blocks)
  │    └─ Block size: 16 tokens
  │    └─ Capacity: 217,312 tokens
  │
  └─ SSE Stream → Gateway → Client
       │
       └─ Policy B: every 50 tokens, release excess budget
          └─ On completion: release remaining budget
             └─ _drain_queue: try to admit waiting requests
```

### Budget Counter Lifecycle

```
Admission:   active_tokens += prompt_tokens + max_tokens
Streaming:   Policy B releases excess every 50 generated tokens
Completion:  active_tokens -= (estimated_cost - policy_b_released)
Post-release: _drain_queue() attempts to admit queued requests (FIFO)
```

---

> Staff-level insight: Admission control is a coarse approximation of a fine-grained scheduling problem. It prevents collapse, but optimal performance and fairness require integrating admission with the scheduler itself (token-aware scheduling, per-tenant quotas, priority-based iteration scheduling). That integration is Phase C scope.

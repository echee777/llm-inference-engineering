# Day 17 (v4) — Admission Control + Token Budget Correction
## AI Inference Platform Residency — Phase A, Week 4

> **Version:** v4 (upgraded from third reviewer feedback)
> **Document convention:** `[ADD]`, `[CHANGE]`, `[REJECT]` callouts mark diffs from v3.

---

## Correction Table (v3 → v4 Summary)

| # | Location | Change | Source | Rationale |
|---|---|---|---|---|
| 1 | Trust boundary table | Added reaction-speed sentence (config/control loop) | R3 (Accept) | Closes "how fast does the system adjust?" interviewer follow-up |
| 2 | Policy B, SAFETY_MARGIN | Tied to P95 completion distribution framing | R3 (Accept) | Statistical thinking signal; honest about why 64 is a stand-in |
| 3 | Known Limitations | Added latency vs. throughput tradeoff sentence | R3 (Accept) | Classic frontier-lab tradeoff; belongs adjacent to scheduler coupling |
| 4 | Trust boundary + Known Limitations | Added multi-tenant homogeneous-traffic assumption | R3 (Partial Accept) | One sentence connecting to Day 19/Phase C; not a new section |
| 5 | Trust boundary | Added control-loop evolution sentence | R3 (Partial Accept) | Wired into trust boundary where divergence is discussed; not a standalone callout |
| — | Mock interview script | Out of scope | R3 (Reject) | Separate artifact type; not a syllabus revision |

---

## Cumulative Correction Table (v1 → v4)

| # | Location | Change | Source | Rationale |
|---|---|---|---|---|
| 1 | Morning | Added reconciliation experiment | R1 (Accept) | Turns "I built a counter" into "I validated a proxy against engine reality" |
| 2 | Afternoon, budget correction | Sharpened to named policy + named tradeoffs | R1 (Partial Accept) | Full A/B/C too heavy; named-tradeoffs framing preserves interview value |
| 3 | Afternoon, queue | Added "intentionally naive" framing | R1 (Accept) | Sets up Day 19 HOL blocking analysis |
| 4 | Both blocks | Added lightweight in-process counters | R1 (Partial Accept) | Scoped to counters+stdout; full Prometheus stays on Day 18 |
| 5 | Watch-Outs | Elevated failure semantics to first-class checklist | R1 (Accept) | Budget leak is a real production bug |
| 6 | Watch-Outs | Reframed `max_completion_tokens` default as explicit policy | R1 (Accept) | "Configured reservation policy" is more deliberate than "default to 512" |
| 7 | Morning framing | Added "predictive vs. reactive control plane" note | R2 (Partial Accept) | Systems-layering signal; conceptual anchor for the week |
| 8 | Step 5 | Added proxy trust boundary stances | R2 (Accept) | Reconciliation operationalized, not just measured |
| 9 | Reconciliation divergence list | Named prefill/decode asymmetry as explicit limitation | R2 (Accept) | Explicit limitation stronger than implicit omission |
| 10 | Policy B code | Added SAFETY_MARGIN floor against cascading over-admission | R2 (Accept) | Closes coordinated burst release risk |
| 11 | New section | Added "Proxy Failure Modes" table | R2 (Accept) | Pre-answers hardest interviewer question |
| 12 | Known Limitations | Added scheduler coupling callout | R2 (Partial Accept) | Named only; Phase B measures it empirically |
| 13 | Trust boundary | Added reaction-speed + control loop evolution sentence | R3 (Accept + Partial) | Closes "how fast does it adjust?" and signals control-systems thinking |
| 14 | Policy B, SAFETY_MARGIN | Tied to P95 completion distribution | R3 (Accept) | Statistical grounding for an otherwise arbitrary constant |
| 15 | Known Limitations | Added latency/throughput tradeoff sentence | R3 (Accept) | Classic inference tradeoff; closes the scheduler coupling thought |
| 16 | Known Limitations | Added multi-tenant homogeneous-traffic assumption | R3 (Partial Accept) | Connects explicitly to Day 19 and Phase C |
| — | Full A/B/C policy comparison | R1 (Reject) | Too heavy for build day; named tradeoffs achieves same value |
| — | Full Prometheus export on Day 17 | R1 (Partial Reject) | Cannibalizes Day 18's dashboard block |
| — | Soft vs. hard admission queue redesign | R2 (Reject) | Day 19 Design Note future direction; not Day 17 build |
| — | Mock interview script | R3 (Reject) | Separate artifact type |

---

## Day 17 — Full Syllabus (v3)

**Theme:** Your admission policy is only as good as your budget accounting — and your budget accounting is only as good as your ability to prove it tracks real engine pressure.

**[ADD] Framing note:** This gateway is a *predictive* control plane: it makes admission decisions based on a token-budget proxy computed before execution. vLLM's scheduler is a *reactive* control plane: it responds to actual memory pressure and preempts when necessary. These two planes are coupled but not identical. Admission being granted does not guarantee immediate execution or latency. This distinction is the most important conceptual anchor for the week.

---

### Morning Block (4 hrs) — Token Budget Enforcement + Reconciliation

#### Step 1 — Atomic Token Budget Tracker

Your admission state is a single number: `active_token_budget`. Under concurrent requests, this counter must be thread-safe. A non-atomic counter will silently over-admit under load.

```python
import threading

active_token_budget = 0
budget_lock = threading.Lock()
```

#### Step 2 — Derive Your Budget from KV Math

Don't hardcode an arbitrary limit. Derive it from your Week 1 KV cache calculator. This derivation is what makes the number defensible in an interview — it's not a guess, it's a consequence of your memory model.

```python
# From your Week 1 KV cache calculator (Qwen2.5-3B at FP16 on T4)
KV_CAPACITY_TOKENS  = 52_000     # total KV slots (verify against your Day 3 measurement)
TARGET_UTILIZATION  = 0.65       # conservative safe zone; Phase B will pin the true cliff
ADMISSION_BUDGET    = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~33,800 tokens
```

**Why 0.65?** You're leaving headroom before the preemption cliff first observed in Day 9. Phase B will empirically measure the exact cliff on A10Gs. For now, 65% is a calibrated conservative estimate, not a guess.

#### Step 3 — Admission Check + Release

```python
def try_admit(prompt_tokens: int, max_completion_tokens: int) -> bool:
    estimated_cost = prompt_tokens + max_completion_tokens
    with budget_lock:
        global active_token_budget
        if active_token_budget + estimated_cost > ADMISSION_BUDGET:
            return False   # → 429 with Retry-After header
        active_token_budget += estimated_cost
        return True

def release_budget(amount: int):
    with budget_lock:
        global active_token_budget
        active_token_budget = max(0, active_token_budget - amount)
```

#### Step 4 — Hard Concurrency Cap as Safety Net

```python
MAX_CONCURRENT  = 20  # secondary guardrail; should almost never fire in normal operation
active_requests = 0   # protected by budget_lock or a separate semaphore
```

**Interview framing:** The token budget is the *primary* control, derived from memory math. The concurrency cap is *defense-in-depth* — it guards against malformed requests (e.g., `max_completion_tokens=0`) that would confuse the budget accounting.

---

#### Step 5 — Reconciliation Experiment (45 min)

*This is the most interview-critical step. It turns the day from "I built a counter" into "I measured whether my counter is an honest proxy for engine pressure."*

After implementing Steps 1–4, run requests of several shapes and compare three numbers side by side:

| Metric | Source | How to read it |
|---|---|---|
| `active_token_budget` | Your gateway counter | Your proxy estimate |
| Observed KV block occupancy | Your Day 8 instrumentation patch | Engine's actual block allocation |
| GPU memory delta | `nvidia-smi` / `torch.cuda.memory_allocated()` | Physical memory consumed |

**Experiment design:**

```python
# Shape A: 5 requests × (200 prompt + 512 max_completion) = 3,560 tokens budgeted
# Shape B: 5 requests × (2000 prompt + 512 max_completion) = 12,560 tokens budgeted
# Shape C: Mixed — 3× Shape A + 2× Shape B
```

For each shape, record:
- Gateway `active_token_budget` at peak concurrency
- Actual KV blocks allocated (from your instrumentation)
- `torch.cuda.memory_allocated()` delta (MB)

**Expected divergence sources (document all of these explicitly):**

1. **Block rounding:** vLLM allocates KV in fixed-size blocks (e.g., 16 tokens). A 200-token prompt uses 13 blocks × 16 = 208 tokens of capacity — your gateway sees 200.
2. **Fragmentation:** Short completions leave partial blocks; your gateway counts tokens, not blocks.
3. **Prefill/decode dynamics:** During prefill, KV allocation grows in large bursts (whole prompt at once). During decode, it grows one token per step. Your proxy treats `prompt_tokens + max_completion_tokens` as a single homogeneous cost — it does not distinguish bursty prefill allocation from incremental decode growth. This is a known simplification: a production system would weight prefill tokens more heavily in the admission budget, since they land all at once and stress the allocator differently. At Day 17 scope, this simplification is acceptable and defensible as long as you name it.
4. **Scheduler execution delay:** Even after admission, vLLM's scheduler may not execute the request immediately under load (prefill queue, iteration batching). Your proxy charges budget at admission time, not at execution time. This means your proxy can over-charge during scheduler delay — a conservative bias that is safe but slightly wastes capacity.

**Goal:** You don't need perfect agreement. You need to know the error bars. Typical finding: gateway over-estimates budget consumption by 5–15% due to block rounding, which means your 65% target is slightly conservative. That is fine and explainable.

**[ADD] Proxy trust boundary — operationalizing the measurement:**

The reconciliation table tells you *how much* the proxy diverges. These three stances tell you *what to do* about it:

| Condition | Stance | Rationale |
|---|---|---|
| Proxy error ≤ 15% | Gateway is authoritative | Block rounding and fragmentation absorbed by the 35% headroom in the 65% target |
| Proxy error > 15% (measured under your traffic mix) | Tighten `TARGET_UTILIZATION` until proxy is within 15% | Conservative but safe; headroom is cheap on a single-tenant system |
| Engine-side OOM or preemption spike observed (Phase B) | Fall back to engine-side backpressure signals; treat gateway as a pre-filter only | Scheduler is the ground truth; gateway is the first line of defense |

**[ADD] Reaction speed and control loop thinking:** In the current Day 17 implementation, adjusting `TARGET_UTILIZATION` is a manual, per-deployment config change. In production, this would be driven by a feedback control loop: observe proxy error over a rolling window (e.g., 5 minutes), and automatically tighten or relax the target to keep divergence within the 15% bound. That evolution — from static threshold to adaptive control loop — is the natural Phase C upgrade path for this system.

Document which stance applies to your Day 17 measurements. This is the operationalized answer to "why should I trust your token counter?"

**Record:** Gateway tokens vs. actual KV tokens vs. GPU memory at each shape. This table, plus the trust-boundary stance you selected, goes in your Day 19 Admission Control Design Note as "proxy validation."

---

### Afternoon Block (4 hrs)

#### Part 1 — Token Budget Correction (1.5 hrs)

**The problem:**

Every request reserves `prompt_tokens + max_completion_tokens` on admission. Most requests complete well below `max_completion_tokens`. If `max_completion=512` but average actual output is ~150 tokens, you lock up ~362 tokens of budget per request that will never be used. This causes spurious 429s — rejections of requests that would have fit.

**Implement one named policy, then measure and name the tradeoffs:**

**Policy implemented (Policy B — Periodic Release with Safety Floor):**

```python
RELEASE_INTERVAL   = 50    # release excess budget every N generated tokens
SAFETY_MARGIN      = 64    # never release below this many output tokens of remaining headroom

def on_request_admit(request):
    request.estimated_cost    = request.prompt_tokens + request.max_completion_tokens
    request.tokens_generated  = 0
    request.settled_through_token   = 0
    reserve(request.estimated_cost)

def on_token_generated(request, tokens_generated):
    request.tokens_generated = tokens_generated
    if tokens_generated - request.settled_through_token >= RELEASE_INTERVAL:
        remaining_needed  = request.max_completion_tokens - tokens_generated
        # [ADD] Safety invariant: never release if doing so would drop headroom below SAFETY_MARGIN.
        # This prevents cascading over-admission under synchronized bursts where many requests
        # simultaneously release budget and new requests rush in before any of them complete.
        safe_remaining    = remaining_needed + SAFETY_MARGIN
        reserved_output   = request.max_completion_tokens - request.settled_through_token
        releasable        = reserved_output - safe_remaining
        if releasable > 0:
            release_budget(releasable)
            stats["correction_delta_released"] += releasable
            request.settled_through_token = tokens_generated

def on_request_complete(request, actual_completion_tokens):
    actual_cost           = request.prompt_tokens + actual_completion_tokens
    remaining_reservation = request.estimated_cost - actual_cost
    release_budget(max(0, remaining_reservation))
```

**Why SAFETY_MARGIN matters:** Without the floor, a burst of 20 requests could all release budget simultaneously at token 50, admitting 20 more requests before any of the originals complete. The safety margin ensures that even if a request runs long (to `max_completion_tokens`), the budget has not been prematurely freed. **[CHANGE] Ideal value:** `SAFETY_MARGIN` should be derived from the tail of your workload's completion length distribution — specifically the P95 of `actual_completion_tokens` — so it covers realistic long-completion cases without being excessively conservative. It is fixed at 64 here because per-model completion statistics are not available at Day 17 scope. Document this: "SAFETY_MARGIN=64 is a stand-in for a P95-derived value; revisit once completion distribution data is available from Day 18 load tests."

**Measure the impact (required — this data goes into the Design Note):**

```
Experiment: 100 requests
  max_completion_tokens = 512
  actual average completion = ~150 tokens

Baseline (reserve max, release only at completion):
  - Record: admitted, rejected, tokens freed mid-stream = 0.

Policy B with safety floor (release every 50 tokens, floor at 64):
  - Record: admitted, rejected, Δ_released mid-stream.

Report:
  - Δ_admitted = admitted(B) - admitted(baseline)
  - Capacity improvement = Δ_admitted / total_requests × 100%
  - Tokens released mid-stream (correction_delta_released from your counter)
```

**Named tradeoffs to document (even if not implemented):**

| Policy | Description | Tradeoff |
|---|---|---|
| Baseline | Reserve max, release at completion | Safe; wastes capacity; causes spurious 429s under sustained load |
| Policy B (implemented) | Release excess every N tokens, with safety floor | Better throughput; floor prevents cascading over-admission; slightly more complex |
| Policy C | Reserve prompt + capped estimate + guarded incremental expansion | Highest throughput; requires per-model completion distribution; most implementation risk |

Document why you implemented B and not C: B is implementable without per-model completion statistics. C requires a distribution model you don't have at Day 17. This is the kind of explicit tradeoff reasoning frontier-lab interviewers look for.

---

#### Part 2 — Rate Limiting + Queue (2.5 hrs)

**Per-API-key rate limits:**

```python
from collections import defaultdict
import time

rate_limits = defaultdict(lambda: {"requests": [], "tokens": []})

def check_rate_limit(api_key: str, tokens: int,
                     req_limit=60, token_limit=100_000,
                     window_sec=60) -> bool:
    now = time.time()
    rl  = rate_limits[api_key]
    rl["requests"] = [t for t in rl["requests"] if now - t < window_sec]
    rl["tokens"]   = [(t, n) for t, n in rl["tokens"] if now - t < window_sec]
    if len(rl["requests"]) >= req_limit:
        return False
    if sum(n for _, n in rl["tokens"]) + tokens > token_limit:
        return False
    rl["requests"].append(now)
    rl["tokens"].append((now, tokens))
    return True
```

**Bounded FIFO queue — intentional naive baseline:**

```python
import queue

# INTENTIONAL DESIGN CHOICE: FIFO is a known-naive baseline.
# It will be stress-tested on Day 19 for head-of-line blocking pathologies.
# Phase C will replace this with weighted fair queuing.
# Note: a production system might also distinguish "soft admit" (queue/deprioritize)
# from "hard admit" (immediate execution) — that distinction is deferred to Phase C.
request_queue = queue.Queue(maxsize=50)
MAX_WAIT_SECONDS = 5.0

def enqueue_or_reject(request):
    try:
        request_queue.put_nowait(request)
        return True
    except queue.Full:
        return False   # → 503 immediately
```

**Why this framing matters:** When an interviewer asks "why FIFO?", the answer is: "FIFO is a deliberately naive baseline to expose head-of-line blocking under adversarial traffic on Day 19. That measurement will motivate the fair-queuing design in Phase C."

---

#### Part 3 — Lightweight In-Process Counters (wire during implementation)

Don't wait for Day 18 dashboards. Wire counters as you build:

```python
import time

stats = {
    "admitted":                  0,
    "rejected_budget":           0,
    "rejected_rate_limit":       0,
    "rejected_queue_full":       0,
    "tokens_admitted":           0,
    "tokens_rejected":           0,
    "correction_delta_released": 0,
}

def log_stats():
    utilization = active_token_budget / ADMISSION_BUDGET * 100
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"budget={utilization:.1f}% "
          f"admitted={stats['admitted']} "
          f"rejected_budget={stats['rejected_budget']} "
          f"correction_freed={stats['correction_delta_released']} tokens")
```

Log every 10 seconds during your correction experiment. Real-time feedback that Policy B is firing — without needing Grafana yet.

---

### [ADD] Proxy Failure Modes

*Force yourself to answer the interviewer's hardest question before they ask it: "When does your admission controller become wrong?"*

Document each failure mode with a "how bad can it get?" estimate:

| Failure mode | Trigger | Error direction | How bad? |
|---|---|---|---|
| Bursty long outputs | Adversarial `max_completion_tokens=8192` requests | Under-charges actual pressure | One such request could consume ~8K tokens of budget against ~8K actual KV — accurate, but stresses block allocator with bursty prefill |
| Synchronized burst release | Many requests release safety floor simultaneously | Over-admits | SAFETY_MARGIN=64 limits blast radius to ~64 × N_requests tokens of extra exposure |
| Fragmentation-heavy workload | Many short completions, high request churn | Over-estimates actual KV in use | Gateway sees more "budget used" than actual KV blocks — conservative, causes spurious 429s, does not cause OOM |
| Prefill/decode asymmetry at scale | Long-prompt-heavy traffic mix | Under-weights bursty prefill cost | Block allocator stress not captured in token counter; mitigated by 35% headroom |
| Scheduler execution delay | High-load batching, scheduler queuing | Over-charges (conservative) | Budget held during delay period; wastes some capacity but does not cause over-admission |

**Key framing for interviews:** Every failure mode in this table is conservative (risks spurious 429s) except synchronized burst release, which the SAFETY_MARGIN addresses. An admission controller that fails by over-rejecting is much safer than one that fails by over-admitting into OOM. Design conservatism is a deliberate choice, not a limitation to apologize for.

---

### [ADD] Known Limitations (v3)

**Prefill vs. decode cost symmetry.** The admission budget treats `prompt_tokens + max_completion_tokens` as a flat token cost. In reality, prefill tokens land in one burst (bursty allocator pressure) while decode tokens arrive one per step (incremental pressure). A production system would weight these differently — for example, applying a prefill multiplier to account for the bursty stress on the block allocator. Day 17's flat model is a reasonable first-order approximation at single-tenant T4 scale; it becomes a real gap under multi-tenant, mixed-workload traffic in Phase C.

**Scheduler coupling and the latency/throughput tradeoff.** This gateway operates as a predictive control plane upstream of vLLM. The scheduler is a reactive control plane inside vLLM. Admission is *necessary* but not *sufficient* for latency guarantees. More conservative admission (lower `TARGET_UTILIZATION`) improves latency stability at the cost of throughput — fewer requests compete for scheduler time. More aggressive admission improves utilization but risks tail latency spikes as scheduler contention increases. Phase B will expose this tradeoff empirically by pushing past the admission limit and observing preemption behavior.

**Homogeneous traffic assumption.** The current model assumes a single tenant with roughly uniform request characteristics. Multi-tenant traffic introduces fairness and isolation problems: one tenant sending adversarial large requests can starve others even within budget. This assumption is named here because it is the direct motivation for per-tenant budgets and weighted fair queuing in Phase C. Day 19's adversarial testing will begin to surface this.

---

### Failure Semantics — First-Class Validation Checklist

Budget leaks are silent production bugs. Before calling the day done, explicitly test each failure path:

| Failure scenario | Expected behavior | How to test |
|---|---|---|
| vLLM returns 500 before first token | Budget released; no leak | Kill vLLM mid-request; verify counter decrements |
| Client disconnects mid-stream | Budget released; no leak | `Ctrl+C` on curl during active stream |
| Request timeout (>5s wait in queue) | 503 returned; budget never reserved | Fill queue past capacity, observe response |
| Gateway worker crash during request | Budget leaked (acknowledged); document blast radius | Kill worker process; count leaked tokens on restart |

**For the crash case:** Full recovery (persistent budget state) is Phase C scope. Day 17 requirement: document the failure mode, estimate the blast radius (leaked tokens × recovery time), and describe the recovery mechanism you'd add.

---

### Watch-Outs (v3)

**1. Budget release must be in `try/finally`.**

```python
async def handle_request(request):
    estimated_cost = compute_cost(request)
    if not try_admit(...):
        return 429
    try:
        async for token in stream_from_vllm(request):
            on_token_generated(request, ...)
            yield token
    finally:
        release_budget(estimated_cost)  # fires on: completion, error, client disconnect
```

**2. Token counting must match vLLM.**
Use vLLM's tokenizer (or tiktoken with the same model). `len(prompt.split())` is off by 20–30% for code or multilingual content.

**3. `max_completion_tokens` missing or zero — explicit reservation policy.**
```python
DEFAULT_OUTPUT_RESERVATION = 512   # configured; a deliberate policy choice

max_completion = request.get("max_completion_tokens") or DEFAULT_OUTPUT_RESERVATION
# If the client omits this field, the gateway applies its configured default reservation
# to protect the backend from underestimated output costs.
```

**4. Reconciliation error grows with mixed workloads.**
Longer-prompt traffic (code generation, RAG) increases block rounding error. Your 65% utilization target absorbs this margin. Document as a known approximation; tighten `TARGET_UTILIZATION` if reconciliation error exceeds 15%.

---

### End-of-Day Output

| Artifact | What it demonstrates |
|---|---|
| Working admission gateway with token budget enforcement | KV-derived admission policy from memory math |
| Reconciliation table + trust boundary stance | Proxy validated against engine reality; operationalized, not just measured |
| Correction experiment: baseline vs. Policy B with safety floor | Named tradeoffs, quantified capacity improvement, burst-safety invariant |
| Proxy failure modes table | Pre-answered the interviewer's hardest question |
| Known limitations section (prefill/decode asymmetry, scheduler coupling) | Honest about approximation boundaries; defensible framing |
| Rate limiter + bounded FIFO queue (labeled as naive baseline) | Sets up Day 19 HOL blocking stress test |
| In-process counters + stdout log | Budget utilization visible without Day 18 dashboards |
| Failure semantics checklist | Budget leak tested as first-class production concern |

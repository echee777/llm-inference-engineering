# Day 16 (Mon) — KV-Memory-Driven Admission Design

**Phase A · Week 4 · Admission Control**

**Engine note:** This day targets the vLLM V1 engine. Admission policy is implemented in a gateway layer external to vLLM — the gateway enforces a conservative memory budget derived from KV math, not from engine internals. The engine's internal scheduler state is not directly observable from the gateway; the relationship between them is explicitly addressed in the design section below.

---

## Goals

By end of day you will have:
- Derived your admission budget from the KV cache math you built in Week 1
- Understood exactly what the gateway estimate does and does not tell you about engine state
- Built a working SSE-proxying gateway with live token counting
- Demonstrated one real 429 rejection path (not just a stub)
- Written an admission policy design document with all the nuances that matter for interview

---

## Morning (4 hrs) — Concepts + Design from Memory Math

---

### Block 1 — Read (1 hr)

Three concepts to internalize before designing anything.

**Little's Law: L = λW**

- L = average number of requests in the system
- λ = arrival rate (requests/sec)
- W = average time a request spends in the system

The key inference: if your service rate can't keep up with arrival rate, W grows, and therefore L grows — your queue depth becomes unbounded. This is the mathematical reason you need admission control at all: without it, an arrival spike that exceeds serving capacity produces unbounded queue growth and eventual latency explosion. You observed this empirically in the mini-collapse experiment on Day 9. Little's Law is why that was inevitable, not accidental.

Keep this brief — this is an admission-control day, not a queueing-theory day. One paragraph of intuition is enough.

**Load shedding: fail-fast vs. queue-and-wait**

- **Fail-fast (HTTP 429 + `Retry-After` header):** Reject immediately when over budget. Client is informed explicitly and must retry. Tail latency remains predictable because requests never pile up waiting.
- **Queue-and-wait:** Hold the request in a bounded queue and hope capacity frees. Appropriate only if bursts are short and completion times are highly predictable. For LLM inference — where completion time varies with sequence length and preemption — queue-and-wait is the wrong default. It converts admission pressure into latency spikes that are harder to observe and harder to explain.

**Default for this system: fail-fast.** Queue is a design-note item, not a Day 16 build item (see Block 2).

**HTTP SSE for streaming tokens**

vLLM streams tokens via Server-Sent Events. Your gateway must proxy bytes as they arrive — not buffer the full response. This has two consequences:

1. The client sees the first token sooner (TTFT is not gateway-gated).
2. SSE streaming exposes token generation events, which enables progressive budget release. Without streaming visibility, budget release must remain coarse-grained (on full completion). SSE is not just a UX choice — it is control-plane signal that makes Day 17's token correction possible.

---

### Block 2 — Derive Admission Budget from KV Math (1.5 hrs)

Open your KV cache calculator from Week 1. Work through the derivation chain explicitly:

```
Total HBM (GPU)
  → minus model weights (loaded at startup)
  → minus runtime overhead (CUDA context, allocator metadata, activation buffers)
  → practical KV pool
  → cliff is BELOW practical KV pool (vLLM's scheduler degrades before full exhaustion)
  → gateway budget is set BELOW that cliff
```

This is more precise than "HBM minus weights." There is allocator and runtime headroom that is not available to KV storage. Your Week 1 calculator already accounted for this implicitly; make it explicit in today's design document.

**Admission budget:**

```
ADMISSION_BUDGET = KV_CACHE_CAPACITY_TOKENS × TARGET_UTILIZATION
```

- `KV_CACHE_CAPACITY_TOKENS`: from your Week 1 calculator, using practical KV pool (not raw HBM)
- `TARGET_UTILIZATION`: set to 0.65 today. This is not an arbitrary heuristic — it is derived from your Day 9 collapse experiment, where TTFT p99 began spiking at ~65–70% KV pool utilization under concurrent load. The gateway enforces a buffer below that observed instability boundary. Phase B will measure the exact cliff position systematically and may refine this number.

**Per-request cost:**

```
estimated_cost = prompt_tokens + max_completion_tokens
```

This is conservative — it budgets for the worst case on every request. Day 17 introduces token budget correction to recover the capacity wasted by this conservatism.

**Policy pseudocode:**

```
on_request(request):
  estimated_cost = request.prompt_tokens + request.max_completion_tokens
  if active_token_budget + estimated_cost > ADMISSION_BUDGET:
    return HTTP 429, Retry-After: 5
  else:
    active_token_budget += estimated_cost
    forward to vLLM

on_request_complete(request):
  active_token_budget -= estimated_cost   # release full reservation (coarse-grained)
  # Day 17 introduces progressive release: budget freed incrementally as tokens stream,
  # so capacity is recovered before the request completes entirely.
```

**Critical framing to write down:** This is not a concurrency cap. It is a memory budget, expressed in tokens. A flat concurrency cap of 10 requests treats 10 × 100-token requests identically to 10 × 8,000-token requests. The token budget does not.

---

### Block 3 — Design Gateway Architecture + Write the Limitations Section (1.5 hrs)

#### Gateway architecture

```
HTTP Request
    ↓
[Gateway]
  ├─ Tokenize: count prompt_tokens using model tokenizer
  ├─ Admission check: active_token_budget + estimated_cost ≤ ADMISSION_BUDGET?
  │     NO  → HTTP 429 + Retry-After header
  │     YES → active_token_budget += estimated_cost
  ↓
[vLLM API Server]
    ↓
[SSE stream — proxied back byte-by-byte, never buffered]
    ↓
on_complete: active_token_budget -= estimated_cost
```

**Bounded queue (design note only — do not build today):** A bounded FIFO queue (max depth 20, max wait 5s) can absorb short bursts without immediate 429s. Document it in the architecture. Implement it in Day 17 or 18 if time permits. If you build the queue today, the day scope-creeps into "build mini API gateway product" and the memory-budget insight gets diluted.

#### Limitations section — write this explicitly

This is the most interview-critical section of the design document. If you omit it, a strong interviewer will expose it.

**Gateway estimate ≠ engine state**

The gateway's `active_token_budget` counter is a conservative external estimate of engine memory pressure. It is not a direct mirror of vLLM's internal allocator state. The relationship:

| | Gateway Budget | Engine State |
|---|---|---|
| What it tracks | Reserved token cost per request | Actual physical KV blocks allocated |
| Accuracy | Conservative (estimates at max_completion) | Ground truth |
| Staleness | Updated on request arrival/completion | Updated every scheduler step |
| Visibility | Observable directly | Not exposed externally (Day 16) |

**Why the estimate is still useful:** It prevents the engine from being driven into known unsafe regions. The engine's scheduler never has to deal with a load it can't handle, because the gateway pre-filters. The estimate is not exact truth — it is a policy fence around the region where you observed collapse.

**What the estimate does not account for (known limitations — state these in your doc):**

1. **Prefix caching:** vLLM's prefix cache means some prompt tokens are served from cached KV blocks and impose no marginal allocation cost. Day 16 treats all prompt tokens as uniformly expensive. This is intentionally conservative and will be revisited if prefix-cache hit rates are high.

2. **Prefill vs. decode memory pressure difference:** The gateway collapses temporal memory behavior into a static reservation. In reality, a request with a 4K prompt allocates KV blocks immediately during prefill, while a request with a 4K completion grows KV incrementally over many decode steps. This means instantaneous engine memory pressure can differ significantly from the gateway's flat estimate — two requests with identical `estimated_cost` can produce very different memory pressure curves. Both cost `prompt_tokens + max_completion_tokens` in the budget, but the risk profile is not identical. This connects directly to Phase B's prefill/decode interference analysis.

3. **Engine divergence under recompute:** If vLLM preempts a sequence and recomputes, the gateway's budget counter still reflects the original reservation. It does not detect that a sequence was temporarily evicted. This is acceptable for a first-pass gateway; Phase C can add engine-exported metrics for tighter control loops.

4. **Single-instance assumption:** The `_active_tokens` counter is in-process memory. This design assumes a single gateway instance. Distributed deployment (multiple gateway pods) requires shared budget state — e.g., a Redis counter with atomic increment/decrement, or a shard-per-pod model with approximate aggregation across shards. This is a Phase C extension.

**Framing for interview:** "The gateway enforces a conservative external memory budget derived from KV math. It is not a perfect mirror of engine internals, but it prevents the engine from being driven into known unsafe regions. We then refine the estimate with correction on Day 17, and can tighten the control loop further with engine-exported metrics in Phase C."

---

## Afternoon (4 hrs) — Build

---

### Build the gateway (3.5 hrs)

Build with FastAPI. Your goals for today:

1. HTTP endpoint accepting chat completion requests (OpenAI-compatible schema)
2. Token counting with the **model's actual tokenizer** — not `cl100k_base`
3. Active token budget counter with admission enforcement
4. Forward to vLLM, proxy SSE stream byte-by-byte (no buffering)
5. At least one real 429 path — see smoke test below

**On tokenizer choice:** Use the model's own tokenizer (`AutoTokenizer` from `transformers` or the tokenizer vLLM loads internally). Using `tiktoken`'s `cl100k_base` for a Qwen-family model produces a systematic estimation error because token boundaries differ between tokenizer families. If you use tiktoken as a temporary bootstrap, document the approximation explicitly and measure the error (spot-check: count the same prompt with both tokenizers and record the delta). For interview: "We use the model's tokenizer to avoid systematic estimation error. tiktoken is only used as a bootstrap approximation and is replaced with the model-native tokenizer before production."

**On chat template:** Token counting must match the exact prompt format actually sent to vLLM, which applies the chat template internally before tokenizing. If your gateway counts raw message content without applying the template, the estimation error is systematic and directional — the gateway will consistently undercount prompt tokens (because the template adds special tokens, role markers, and formatting). Undercounting biases toward over-admission, which is the dangerous direction. Apply the chat template in `count_prompt_tokens` before encoding.

**Skeleton:**

```python
# gateway.py
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- KV budget derived from Week 1 calculator ---
# T4 (16GB): ~12GB usable after weights + runtime overhead
# bytes_per_token = 2 (layers) * 2 (heads) * head_dim * 2 (FP16) * num_layers
# Plug in your own Week 1 numbers here
KV_CAPACITY_TOKENS = 52_000        # from your calculator
TARGET_UTILIZATION = 0.65
ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)   # ~33,800

# asyncio.Lock — correct primitive for async FastAPI context.
# threading.Lock() would work but could block the event loop under contention.
# Note: assumes single-process deployment. Distributed deployment requires
# shared budget state (e.g., Redis or shard-per-pod with approximate aggregation).
_lock = asyncio.Lock()
_active_tokens = 0

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def count_prompt_tokens(messages: list[dict]) -> int:
    # Apply the model's chat template before tokenizing.
    # This matches vLLM's internal formatting path and avoids systematic
    # undercounting from template expansion (role markers, special tokens).
    # Undercounting biases toward over-admission — the dangerous direction.
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(formatted))

@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    global _active_tokens

    body = await request.json()
    messages = body.get("messages", [])
    max_completion = body.get("max_tokens", 512)

    prompt_tokens = count_prompt_tokens(messages)
    estimated_cost = prompt_tokens + max_completion

    # --- Admission check ---
    async with _lock:
        if _active_tokens + estimated_cost > ADMISSION_BUDGET:
            return JSONResponse(
                status_code=429,
                content={"error": "token budget exceeded", "retry_after": 5},
                headers={"Retry-After": "5"}
                # Retry-After is currently static (5s). In production this would be
                # derived from observed budget recovery velocity — e.g., average
                # completion time × (estimated_cost / ADMISSION_BUDGET).
            )
        _active_tokens += estimated_cost

    print(f"[gateway] ADMIT  prompt={prompt_tokens} max_completion={max_completion} "
          f"estimated_cost={estimated_cost} active={_active_tokens}/{ADMISSION_BUDGET}")

    async def stream_and_release():
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", VLLM_URL, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        finally:
            async with _lock:
                global _active_tokens
                _active_tokens -= estimated_cost
            print(f"[gateway] RELEASE estimated_cost={estimated_cost} active={_active_tokens}")

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")

@app.get("/metrics")
def metrics():
    return {
        "active_tokens": _active_tokens,
        "admission_budget": ADMISSION_BUDGET,
        "budget_utilization_pct": round(100 * _active_tokens / ADMISSION_BUDGET, 1)
    }
```

---

### Smoke test validation matrix (30 min)

Run these three tests before calling Day 16 done. Record results in your design document.

| Test | Setup | Expected Result | Actual Result |
|---|---|---|---|
| Small request admitted | `max_tokens=50`, short prompt (~20 tokens) | 200, response streams | |
| Two medium requests concurrent | `max_tokens=256` each, send simultaneously | Both 200 | |
| Oversized request rejected | Manually set `ADMISSION_BUDGET=100`, send request with `max_tokens=200` | 429 + Retry-After header | |

For test 3, temporarily hardcode a tiny budget to force a rejection. Verify the 429 response and that `_active_tokens` is not incremented. Then restore the real budget.

**What you're proving:** The admit/reject path is live, not a stub. The budget counter increments on admission and decrements on completion. A budget breach produces a 429 with the right headers.

---

## End-of-Day Output

**1. Working gateway** (`gateway.py`) with:
- Model-native tokenizer (not cl100k_base)
- Live admission check with real 429 path
- SSE streaming without buffering
- `/metrics` endpoint showing active budget utilization
- Smoke test 3 verified: forced 429 observed

**2. Admission Control Design Document** (markdown) with:
- KV budget derivation chain (HBM → weights → runtime overhead → practical KV pool → cliff → gateway budget)
- Policy pseudocode
- Gateway architecture diagram (text)
- **Limitations section** (gateway estimate vs. engine truth, prefix caching, prefill/decode asymmetry, recompute divergence)
- Smoke test results table

---

## Interview Self-Test

Before starting Day 17, answer these from memory:

**Q: How does your gateway decide whether to admit a request?**

Target: "We compute `prompt_tokens + max_completion_tokens` for each request and check it against a token budget derived from KV cache math. The budget is: practical KV pool × 65% utilization target, where practical KV pool is total HBM minus model weights and runtime overhead. If the active budget plus estimated cost exceeds the limit, we return 429 immediately. This is a memory budget expressed in tokens, not a concurrency cap."

**Q: Is your gateway's token count the same as what vLLM actually allocates?**

Target: "No. The gateway estimate is a conservative external proxy for engine memory pressure. It uses `max_completion_tokens` as the reservation, which overestimates for requests that terminate early — that's what Day 17's token correction addresses. It also ignores prefix-cache hits, which reduce marginal KV cost for shared-prefix requests. And it doesn't observe recompute events. The estimate prevents the engine from being driven into unsafe regions; it's not a perfect mirror of engine state. Phase C can tighten this with engine-exported metrics."

**Q: How would you make this work across 5 gateway instances?**

Target: "The current design stores `_active_tokens` in process memory — it's a single-instance design. Across 5 pods you'd need shared budget state. The simplest approach is a Redis counter: `INCRBY` on admission, `DECRBY` on completion, both atomic. The tradeoff is a network round-trip on every request for the admission check. An alternative is shard-per-pod budgets (each pod owns 1/5 of the total budget) which avoids coordination but requires load-balancing to be budget-aware. In practice, for a first multi-instance deployment I'd start with Redis and profile the added latency against the admission check path."

If you can answer all three cleanly, Day 16 is done.

---

## What's Ahead on Day 17

Day 17 is about correctness and refinement on top of today's working foundation:
- Token budget correction: measure how much capacity is wasted by budgeting at max_completion, and implement progressive release
- Per-API-key rate limiting
- Bounded queue as a build item (now that the core is solid)
- Measure: X% more requests admitted with correction enabled (this number goes in Deliverable #4)

---

## Change Log

| Version | Change | Source | Rationale |
|---|---|---|---|
| v1→v2 | Replaced tiktoken with model-native tokenizer | Reviewer R1 | cl100k_base produces systematic estimation error for Qwen-family models |
| v1→v2 | Added gateway estimate vs. engine state limitations section | Reviewer R1 | Most interview-critical gap; gateway count ≠ allocator truth |
| v1→v2 | Added prefix-caching conservatism note | Reviewer R1 | Day 16 ignores cache hits; intentional but must be stated |
| v1→v2 | Sharpened KV derivation chain to include runtime overhead | Reviewer R1 | "HBM minus weights" skips allocator/runtime headroom |
| v1→v2 | Added smoke test validation matrix with forced 429 | Reviewer R1 | Day must end with real rejection path, not stub |
| v1→v2 | Downgraded bounded queue to design-note item | Reviewer R1 | Build focus stays on memory-budget insight, not API gateway product |
| v1→v2 | Refined "earliest point" framing to "earliest point with enough signal" | Reviewer R1 | More mature; acknowledges gateway estimate is useful but imperfect |
| v2→v3 | Applied chat template in `count_prompt_tokens` | Reviewer R2 | Raw content join undercounts systematically; template expansion adds role markers and special tokens; undercounting biases toward over-admission |
| v2→v3 | Added note on systematic error direction (undercounting = over-admission) | Reviewer R2 | Direction of error matters: over-admission is the dangerous side |
| v2→v3 | Replaced `threading.Lock()` with `asyncio.Lock()` | Reviewer R2 | threading.Lock can block event loop under contention in async context; asyncio.Lock is the correct primitive |
| v2→v3 | Added single-instance assumption to limitations section | Reviewer R2 | Distributed deployment requires shared state (Redis / shard-per-pod); explicit limitation prevents naive horizontal scaling |
| v2→v3 | Added progressive release foreshadow to pseudocode comment | Reviewer R2 | Connects SSE streaming to Day 17 correction story earlier |
| v2→v3 | Added third interview self-test: multi-instance budget state | Reviewer R2 | Reviewer identified this as a likely follow-up; answer prepared |
| v3→v4 | Tied TARGET_UTILIZATION to Day 9 empirical collapse observation | Reviewer R3 | 65% was reading as arbitrary heuristic; must read as measured boundary |
| v3→v4 | Replaced "different timescales" with precise temporal pressure formulation | Reviewer R3 | "Gateway collapses temporal behavior into static reservation" is the exact insight; instantaneous pressure ≠ flat estimate |
| v3→v4 | Elevated SSE to control-plane signal (not just UX) | Reviewer R3 | SSE exposes token events that make progressive release possible; without it, release is coarse-grained — this is the motivation for Day 17 |
| v3→v4 | Added dynamic Retry-After note as code comment | Reviewer R3 | Static 5s is interview-visible; production value derived from budget recovery velocity |

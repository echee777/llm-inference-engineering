# Day 16 - KV-Memory-Driven Admission Design

## Morning: Concepts + Design

### Little's Law

L = λW. L is average requests in system, λ is arrival rate, W is average time in system. If arrival rate exceeds service rate, W grows without bound. Queue depth increases by the difference every second. There's no self-correcting mechanism. Without admission control, the system doesn't stabilize at some higher latency, it just keeps getting worse until something breaks. This is why Day 9's collapse was inevitable, not accidental.

### Fail-fast vs queue-and-wait

Default for LLM inference: fail-fast with HTTP 429 + Retry-After header.

Queue-and-wait is wrong for LLM inference because:
- LLM completion times are highly variable (depends on sequence length, preemption). Wait time in a queue is unpredictable.
- Deep queues cause client timeouts. Timed-out clients retry. Now you have the original request still queued plus a retry, doubling load. This is a retry storm.
- With 429, the client knows immediately and can back off. The system never accumulates hidden pressure. Tail latency stays predictable.

Queue-and-wait is only appropriate if bursts are short and completion times are highly predictable. Neither is true for LLM inference.

### Admission budget derivation

The primary constraint is KV cache memory.

```
Total GPU HBM
  - model weights (loaded at startup)
  - runtime overhead (CUDA context, activations, sampler buffers)
  = practical KV pool
  x gpu_memory_utilization (0.9, vLLM's knob)
  = KV capacity in tokens
  x 0.65 (gateway target utilization, from Day 9 collapse data)
  = ADMISSION_BUDGET (in tokens)
```

Two separate multipliers, two separate reasons. gpu_memory_utilization (0.9) is vLLM's parameter controlling how much VRAM it uses. The 0.65 is the gateway's safety margin below the observed instability boundary from Day 9 where TTFT p99 started spiking.

Why 65%? On Day 9 without admission control, TTFT p99 spiked at ~65-70% KV pool utilization. The root cause: as the free block pool shrinks, running decode requests need new blocks to grow but fewer are available. The scheduler preempts (evicts) running sequences to free blocks. Preempted sequences need recomputation later, adding more load, potentially triggering more preemption. This cascade is what makes the latency curve nonlinear.

Note: PagedAttention solves external fragmentation (blocks don't need to be contiguous). The cliff is about running out of free blocks for decode growth, not about contiguous allocation.

The 0.65 is not a universal constant. It's empirical and workload-specific. Different models, hardware, or traffic patterns would hit the cliff at different utilization levels. Phase B will characterize the exact cliff systematically.

### Per-request cost

```
estimated_cost = prompt_tokens + max_completion_tokens
```

This is conservative. max_completion_tokens may be much larger than actual completion length, leaving capacity on the table. Day 17 introduces progressive release to recover this wasted capacity as tokens stream back via SSE.

### This is a token budget, not a concurrency cap

A flat concurrency cap of 10 treats 10 x 100-token requests identically to 10 x 8000-token requests. A token budget does not.

### Policy pseudocode

```
on_request(request):
  estimated_cost = request.prompt_tokens + request.max_completion_tokens
  if active_token_budget + estimated_cost > ADMISSION_BUDGET:
    return HTTP 429, Retry-After: 5
  else:
    active_token_budget += estimated_cost
    forward to vLLM

on_request_complete(request):
  active_token_budget -= estimated_cost
```

### Gateway architecture

```
HTTP Request
    |
[Gateway]
  - Tokenize: count prompt_tokens using model's own tokenizer + chat template
  - Admission check: active_token_budget + estimated_cost <= ADMISSION_BUDGET?
      NO  -> HTTP 429 + Retry-After header
      YES -> active_token_budget += estimated_cost
    |
[vLLM API Server]
    |
[SSE stream - proxied byte-by-byte, never buffered]
    |
on_complete: active_token_budget -= estimated_cost
```

Tokenizer choice matters. Must use the model's own tokenizer, not tiktoken's cl100k_base. Different tokenizer families produce different token boundaries, creating systematic estimation error. Must also apply the chat template before tokenizing, because the template adds role markers and special tokens. Without template application, the gateway systematically undercounts prompt tokens. Undercounting biases toward over-admission, which is the dangerous direction.

SSE streaming is not just a UX choice. It exposes token generation events, which enables progressive budget release on Day 17. Without streaming visibility, budget release is coarse-grained (only on full completion).

### Limitations: gateway estimate vs engine state

The gateway's active_token_budget is a conservative external estimate of engine memory pressure, not a direct mirror of vLLM's internal block allocator. Ways they diverge:

1. max_completion_tokens overestimate. Request reserves for worst case but may generate far fewer tokens. This is too conservative, leaves capacity on the table. Day 17 fixes this.

2. Prefix caching. vLLM's prefix cache means some prompt tokens are served from cached KV blocks at no marginal allocation cost. The gateway treats all prompt tokens as uniformly expensive. Too conservative.

3. Prefill vs decode temporal mismatch. The gateway collapses temporal memory behavior into a static reservation. A 4K-token prompt allocates KV blocks immediately during prefill. A 4K-token completion grows KV incrementally over many decode steps. Two requests with identical estimated_cost produce very different memory pressure curves. The gateway treats them the same.

4. Preemption divergence. If vLLM preempts a sequence and recomputes, the gateway's counter still reflects the original reservation. It does not detect temporary eviction.

5. Single-instance assumption. The active_tokens counter is in-process memory. Distributed deployment (multiple gateway pods) requires either shared state via Redis (simple, adds network round-trip per admission check) or sharded budgets (each pod owns 1/Nth of total budget, avoids coordination but requires budget-aware load balancing to prevent hot spots).

Despite the imprecision, the estimate is still useful. It prevents the engine from being driven into known unsafe regions. The estimate is a policy fence, not a perfect mirror.

Interview framing: "The gateway enforces a conservative external memory budget derived from KV math. It is not a perfect mirror of engine internals, but it prevents the engine from being driven into known unsafe regions. We refine the estimate with progressive release on Day 17, and can tighten the control loop further with engine-exported metrics in Phase C."

## Afternoon: Build + Smoke Tests

### Gateway implementation

Built `gateway.py` with:
- Qwen2.5-3B tokenizer with chat template applied before encoding (not tiktoken)
- asyncio.Lock for budget counter (not threading.Lock)
- SSE stream proxied byte-by-byte via httpx.AsyncClient.stream(), never buffered
- Admission check: estimated_cost = prompt_tokens + max_completion_tokens, reject with 429 if over budget
- Budget released in finally block of the stream generator
- /metrics endpoint exposing active_tokens, admission_budget, budget_utilization_pct

Budget constants:
```
KV_CAPACITY_TOKENS = 191,000
TARGET_UTILIZATION = 0.65
ADMISSION_BUDGET   = 124,150
```

### Smoke test results

```
Test                            Setup                                    Expected                    Actual
Small request admitted          max_tokens=50, short prompt              200, response streams       200, 12 chunks streamed, active_tokens back to 0 after
Two medium requests concurrent  max_tokens=256 each, sent simultaneously Both 200                    Both 200 (req-A and req-B both admitted and completed)
Oversized request rejected      ADMISSION_BUDGET=100, max_tokens=200     429 + Retry-After header    429, Retry-After: 5, body: {"error": "token budget exceeded"}, active_tokens stayed at 0
```

All three tests pass. The admit/reject path is live. Budget counter increments on admission, decrements on completion. Rejected requests do not increment the counter. 429 includes Retry-After header.

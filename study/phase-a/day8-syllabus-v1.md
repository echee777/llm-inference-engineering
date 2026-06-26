# Day 8 — Instrumentation Patch (v1 Architecture)

> **Rewritten for vLLM v1 (main branch).** The original syllabus targeted v0.6.x.
> v1 removed SequenceGroup, BlockSpaceManager, SchedulingBudget, and swap-to-CPU.
> This version uses the correct v1 file paths, class names, and concepts.

---

## Context

Day 8 operationalizes the Week 2 instrumentation milestone. You've traced the full
request lifecycle on Day 7. Today you harden that understanding into a diagram you
can explain from memory, then build a small logging patch that makes KV cache block
events observable. Day 9 will use this patch to run the mini-collapse experiment.

---

## Morning Block (4 hrs)

### Block 1: Refine the Architecture Diagram (2 hrs)

Take the Day 7 diagram and harden it. The test: can you explain every arrow and every
decision point without consulting the code?

Walk through each traced step and fill any gaps:

- **`entrypoints/openai/chat_completion/serving.py`** — What does the HTTP handler create?
  - Parse JSON → apply chat template (`render_chat_request()`, line 305) → tokenize
  - Build `SamplingParams` (`request.to_sampling_params()`, line 354)
  - Call `engine_client.generate(prompt, sampling_params, request_id)` (line 387)

- **`v1/engine/async_llm.py`** — Where does `generate()` enqueue?
  - `generate()` (line 529) calls `add_request()` (line 563)
  - `add_request()` calls `engine_core.add_request_async()` (line 417)
  - Request sent via **ZMQ ROUTER socket** to EngineCore in a separate process
  - Results arrive in a `RequestOutputCollector` (per-request mailbox, `asyncio.Event`)

- **`v1/core/sched/scheduler.py`** — Can you trace `schedule()`?
  - No `SchedulingBudget` class — just `token_budget = self.max_num_scheduled_tokens` (line 341)
  - For each RUNNING request: schedule decode tokens, allocate new KV blocks if needed
  - For each WAITING request (while budget > 0 and running < max):
    - `kv_cache_manager.allocate_slots()` — returns blocks or None (line 437)
    - If None: stop admitting
    - If success: `request.status = RequestStatus.RUNNING` (line 812)

- **`v1/core/kv_cache_manager.py` + `v1/core/block_pool.py`** — Walk the allocation chain:
  - `allocate_slots()` → `coordinator.get_num_blocks_to_allocate()` → check free count
  - If `needed > free`: return None (line 348)
  - Otherwise: `block_pool.get_new_blocks(n)` → pop from free queue, `ref_cnt += 1` (line 341)
  - `free_blocks()` → `ref_cnt -= 1` → back to free queue when ref_cnt == 0 (lines 420-421)
  - No `can_allocate()` pre-check — v1 uses try-and-fail pattern
  - No watermark — relies on preemption to recover

- **`v1/worker/gpu/model_runner.py`** — Prefill vs decode in `execute_model()`:
  - Both handled in the **same call**, batch can mix prefill and decode
  - `prepare_inputs()` (line 605) sorts decode-first (line 614)
  - Detection: `num_computed_tokens < prompt_length` = prefill, `>=` = decode
  - Prefill: many tokens, KV cache populated fresh, logits at every position
  - Decode: 1 token per request, KV cache appended, logit at last position only

For each state transition, annotate the specific function responsible:

**State machine (v1 — no SWAPPED state):**

```
              ┌──────────┐
   arrival →  │ WAITING  │ ←──────────────────┐
              └────┬─────┘                    │
                   │ allocate_slots() succeeds │ prepend to waiting queue
                   │ + budget + running < max  │ (scheduler.py:951)
                   │ (scheduler.py:812)        │
                   ▼                           │
              ┌──────────┐                     │
              │ RUNNING  │ ────── preempt ─────┘
              └────┬─────┘    _preempt_request()
                   │          (scheduler.py:931)
                   │          → kv_cache_manager.free()
                   │          → num_computed_tokens = 0
                   │          → status = PREEMPTED
                   ▼
              ┌──────────┐
              │ FINISHED │
              └──────────┘
```

**Preemption cost (v1 — recompute only, no swap):**

v1 eliminated swap-to-CPU entirely. When preempted:
- KV cache is **freed**, not swapped (line 940)
- `num_computed_tokens` reset to 0 (line 943)
- Request returns to the waiting queue (line 951)
- When rescheduled, it **re-prefills from scratch**

Cost is proportional to prompt length. A preempted 4000-token request must
re-run the entire prefill. With prefix caching (v1 default), common prefixes
survive eviction — only the unique suffix is recomputed.

Why v1 dropped swap: prefix caching made it redundant, recompute is simpler,
and swap contradicted v1's "near-zero CPU overhead" design goal.

---

### Block 2: Choose and Begin the Instrumentation Patch (2 hrs)

**What you're building:** Add logging to the KV cache manager and scheduler so that
every block allocation, deallocation, and preemption event emits a structured log line.

Target output format:

```
[BLOCK_ALLOC]   ts=1234567 req=abc123 alloc=8  free=104/128
[BLOCK_FREE]    ts=1234590 req=abc123 freed=8  free=112/128
[BLOCK_PREEMPT] ts=1234601 req=def456 freed=4  free=116/128 computed_tokens_lost=500
```

Note: `computed_tokens_lost` is meaningful in v1 because preemption means recompute.
This number tells you how much work is thrown away.

**Setup:**

```bash
cd ~/Documents/code/vllm
git checkout -b day8-block-instrumentation
```

No need to check out v0.6.6 — instrument the v1 code directly.

**Where to add logging:**

- **ALLOC** — `v1/core/kv_cache_manager.py` inside `allocate_slots()` (line 334+),
  after blocks are successfully allocated. Log the request ID, number of blocks
  allocated, and free block count.

- **FREE** — `v1/core/kv_cache_manager.py` inside `free()`, after blocks are
  returned to the pool. Log the request ID, number of blocks freed, and free
  block count.

- **PREEMPT** — `v1/core/sched/scheduler.py` inside `_preempt_request()` (line 931).
  This is the scheduler decision point — log here because you have access to the
  request context: which request, how many computed tokens are being discarded,
  and the memory state.

  The distinction matters: state transitions live in the scheduler;
  memory consequences live in the KV cache manager.

The patch should be **20–50 lines** of `logging.info()` calls in the right places.
Don't add metrics libraries, structured JSON output, or a sidecar process.

---

## Afternoon Block (4 hrs)

### Block 3: Implement and Test (3 hrs)

Instrument the three event types as described above. Then start the server and
filter for your log lines:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  2>&1 | grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT"
```

Send a single request and verify the log output is visible. Then move to validation.

---

### Block 4: Validation Experiments (1 hr)

#### Experiment 1 — Sequential baseline

Send 5 requests one at a time (wait for each to finish before sending the next).

Do not expect exactly one BLOCK_ALLOC and one BLOCK_FREE per request. During decode,
as a sequence grows beyond its initially allocated blocks, additional allocations are
triggered. A single request may emit multiple ALLOC events.

Validate that the **pattern is directionally correct and the totals reconcile**:
- Free count decreases on ALLOC events, increases on FREE events
- After each request completes, free count returns to baseline
- No overlap between sequential requests

Expected pattern:
- ALLOC events cluster at request start (prefill), more during decode growth
- FREE events appear at request completion
- No PREEMPT events (sequential requests don't compete for memory)

#### Experiment 2 — Concurrent allocation

Send 5 requests simultaneously (use an async client or `asyncio.gather` script).

Expected pattern:
- ALLOC events cluster as requests arrive
- Free count drops with each allocation
- FREE events arrive as requests finish, free count recovering
- This is continuous batching made visible: batch composition changes each step

**Occupancy sanity check:**

At prefill, each request should allocate approximately:

```
ceil(prompt_tokens / block_size) blocks
```

This is a lower-bound estimate. Block usage grows during decode as new tokens
fill blocks. The formula is a sanity check on prefill allocation, not a strict
invariant for the full request lifetime.

Record peak block usage during the concurrent run and check whether it's broadly
consistent with your Day 3 KV cache math.

---

## End-of-Day Output Checklist

1. **Refined architecture diagram** — all steps with v1 file paths, function names,
   data structures in/out; state machine with code-referenced transitions and
   preemption cost annotations (recompute only, no swap)
2. **Working instrumentation patch** — committed branch; ALLOC/FREE/PREEMPT log events
   visible; PREEMPT logged from `_preempt_request()` in the scheduler
3. **Validation data** — sequential and concurrent baseline runs recorded; totals
   reconcile; prefill-phase block counts checked against KV cache math

---

## v0 → v1 Changes Summary

| v0 Concept | v1 Equivalent | Why |
|---|---|---|
| `SequenceGroup` | `Request` (`v1/request.py:295`) | Simplified — no beam search in core |
| `BlockSpaceManager` | `KVCacheManager` + `BlockPool` | Prefix caching integrated, ref-counted |
| `can_allocate()` → OK/LATER/NEVER | `allocate_slots()` → blocks or None | Try-and-fail is simpler |
| `SchedulingBudget` class | `token_budget` int (line 341) | No need for a class |
| SWAPPED state + swap-to-CPU | PREEMPTED + recompute | Prefix caching made swap redundant |
| Watermark reserve | No watermark | Relies on preemption to recover |
| `async_llm_engine.py` (full impl) | Alias → `v1/engine/async_llm.py` | v0 engine removed |

---

## What's Ahead: Day 9

Day 9 is where the patch pays off. You'll run the **mini-collapse experiment**:
push `--gpu-memory-utilization` to 0.98, hammer the server with 20 concurrent 4K-token
requests, and watch your block instrumentation show free blocks dropping toward 0,
PREEMPT events firing from the scheduler (with `computed_tokens_lost` showing the
recompute penalty), and TTFT spiking. This is the latency cliff you annotated on
Day 7's state machine — now you'll observe it live with real numbers.

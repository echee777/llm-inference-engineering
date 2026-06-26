# Day 8 — Instrumentation Patch (Corrected)

> **Corrections applied from reviewer feedback.** Four specific fixes are incorporated below;
> each is marked with a `[FIX]` callout so the reasoning is visible alongside the instruction.

---

## Context

Day 8 operationalizes the Week 2 instrumentation milestone. You've traced the full
request lifecycle on Day 7 (five steps, architecture diagram, SequenceGroup state machine).
Today you harden that understanding into a diagram you can explain from memory, then build
a small logging patch in your vLLM fork that makes KV cache block events observable.
Day 9 will use this patch to run the mini-collapse experiment.

---

## Morning Block (4 hrs)

### Block 1: Refine the Architecture Diagram (2 hrs)

Take the Day 7 diagram and harden it. The test: can you explain every arrow and every
decision point without consulting the code?

Walk through each of the five traced steps and fill any gaps:

- **`api_server.py`** — What does the HTTP handler create? (`SamplingParams` + prompt → `SequenceGroup`)
- **`async_llm_engine.py`** — Where exactly does `generate()` enqueue the request? What queue?
- **`scheduler.py`** — Can you trace the full `schedule()` call start to output? What is `SchedulingBudget`? Where does the WAITING→RUNNING transition happen in code?
- **`block_manager.py`** — Can you walk `can_allocate()` → `allocate()` → `free()` with block math?
- **`model_runner.py`** — What is different between the prefill and decode execution paths in `execute_model()`?

For each SequenceGroup state transition, annotate the specific `scheduler.py` function
responsible. Include the preemption cost annotations from Day 7:

- **Swap path cost:** CPU↔GPU transfer time, proportional to KV cache size of the evicted sequence
- **Recompute path cost:** full prefill re-execution, proportional to prompt length

These annotations connect directly to the Day 9 mini-collapse experiment. You'll observe
the latency explosion live tomorrow.

---

### Block 2: Choose and Begin the Instrumentation Patch (2 hrs)

**Use Option A: KV Cache Block Events.** This is the highest-value instrumentation for
Phase B collapse engineering.

**What you're building:** Add logging to the block manager and scheduler so that every
block allocation, deallocation, and preemption event emits a structured log line.

Target output format:

```
[BLOCK_ALLOC]   ts=1234567 req=abc123 alloc=8  freed=0  used=24/128 free=104/128
[BLOCK_FREE]    ts=1234590 req=abc123 alloc=0  freed=8  used=16/128 free=112/128
[BLOCK_PREEMPT] ts=1234601 req=def456 alloc=0  freed=4  used=12/128 free=116/128 reason=memory_pressure
```

**Setup:**

```bash
cd ~
git clone https://github.com/vllm-project/vllm.git vllm-instrumented
cd vllm-instrumented
git checkout v0.6.6
git checkout -b day8-block-instrumentation
pip install -e . --break-system-packages
```

**Where to add logging:**

> [FIX #3 — File paths are starting points; the actual call graph is the source of truth.]
>
> Start with `vllm/core/block_manager.py` for v0.6.6. If the code path has shifted from
> what the syllabus describes, follow the actual call graph rather than the label.
> Use `grep -r "def allocate" vllm/` and `grep -r "def free" vllm/` to orient yourself.

- **ALLOC** — instrument inside `allocate()` after blocks are assigned
- **FREE** — instrument inside `free()` after blocks are returned to the free pool
- **PREEMPT** — see below

> [FIX #4 — PREEMPT is scheduler-originated, not block-manager-originated.]
>
> Preemption is a *scheduler decision*. `scheduler.py`'s `_preempt()` (or `preempt()`)
> decides to evict a sequence; the block manager then executes the memory consequence
> (swap-out or free). Log the PREEMPT event at the scheduler eviction call site, where
> you have access to the decision context: which sequence, why, what was the memory state.
> You may optionally also log the block-manager memory consequence separately.
>
> This distinction reinforces the Day 7 lesson: state transitions live in the scheduler;
> memory consequences live in the block manager.

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

> [FIX #1 — Validate by reconciling totals, not by counting events per request.]
>
> Do not expect exactly one BLOCK_ALLOC and one BLOCK_FREE per request. During decode,
> as a sequence grows beyond its initially allocated blocks, `append_slot()` can trigger
> additional allocations. A single request may emit multiple ALLOC events.
>
> Instead, validate that the **pattern is directionally correct and the totals reconcile**:
> - `used` increases on ALLOC events, decreases on FREE events
> - After each request completes, `used` returns to baseline
> - `used + free == total_blocks` holds throughout

Expected pattern for sequential requests:
- ALLOC events cluster at request start (prefill), possibly more during decode growth
- FREE events appear at request completion
- No overlap in `used` between sequential requests (one finishes before next starts)

#### Experiment 2 — Concurrent allocation

Send 5 requests simultaneously (use an async client or a quick `asyncio.gather` script).

Expected pattern:
- ALLOC events cluster as requests arrive
- `used` climbs with each allocation
- FREE events arrive as requests finish, `used` declining back toward baseline
- This is continuous batching made visible: the running batch composition changes each step

**Occupancy sanity check:**

> [FIX #2 — Use the ceil formula only as a prefill-phase occupancy estimate, not as a
> full-run invariant.]
>
> At the moment a request begins executing (immediately after prefill), the blocks
> allocated should be broadly consistent with:
>
> ```
> ceil(prompt_tokens / block_size) blocks per request
> ```
>
> This is your Day 3 KV cache math applied to the block manager. Treat it as an
> approximate lower-bound on occupancy at prefill start. Block usage will grow during
> decode as new tokens are generated, and rounding effects will add at least one block
> per request. The formula is a sanity check on prefill allocation, not a strict
> invariant for the full request lifetime.

Record peak `used` during the concurrent run and note whether it is broadly consistent
with your Day 3 math at prefill, adjusted for decode growth.

---

## End-of-Day Output Checklist

1. ✅ **Refined architecture diagram** — all 5 steps with file paths, function names, data structures in/out; state machine with code-referenced transitions and preemption cost annotations
2. ✅ **Working instrumentation patch** — committed branch in your vLLM fork; ALLOC/FREE/PREEMPT log events visible, PREEMPT logged from the scheduler eviction path
3. ✅ **Validation data** — sequential and concurrent baseline runs recorded; totals reconcile; prefill-phase block counts checked against KV cache math

---

## Summary of Corrections Applied

| # | Original claim | Corrected version |
|---|---|---|
| 1 | "Verify exactly one BLOCK_ALLOC and one BLOCK_FREE per request" | Verify the pattern is directionally correct and totals reconcile; multiple ALLOCs per request are expected during decode growth |
| 2 | "Block count should match `ceil(prompt_tokens / block_size)` — It should" | Use formula as a prefill-phase lower-bound estimate; block usage grows during decode and rounding adds at least one block |
| 3 | "Use `block_manager.py`, not `v2`" | Start with `block_manager.py` for v0.6.6; follow the actual call graph if paths differ |
| 4 | PREEMPT logged from block manager side | PREEMPT logged from `scheduler.py` eviction path where decision context is available; block manager logs the memory consequence separately if needed |

---

## What's Ahead: Day 9

Day 9 is where the patch pays off. You'll run the **mini-collapse experiment**:
push `--gpu-memory-utilization` to 0.98, hammer the server with 20 concurrent 4K-token
requests, and watch your block instrumentation show `used/total` climbing toward 100%,
PREEMPT events firing from the scheduler, and TTFT spiking. This is the latency cliff
you annotated on Day 7's state machine — now you'll observe it live with real numbers.

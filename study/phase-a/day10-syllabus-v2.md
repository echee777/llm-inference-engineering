# Day 10 (Fri) — Week 2 Deliverable + Buffer

## Revised Syllabus v2

**Theme:** Finalize. Polish. Self-test. Prep Week 3.

**Output:** Finalized **Annotated vLLM Architecture Diagram** (Portfolio Deliverable #2)

---

## Changes from Original

| What Changed                                                             | Why                                                                                           |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Removed SWAPPED state from scheduler decision tree                       | vLLM V1 has no swap path — preemption = recompute only                                        |
| Updated file paths to V1 (`kv_cache_manager.py`, `v1/core/scheduler.py`) | Original used V0 paths (`block_manager_v2.py`); student traced V1 on Days 7–9                 |
| Reframed as "finalize Day 9 draft" not "write from scratch"              | Day 9 already produces a rough draft; Day 10 polishes it                                      |
| Added explicit block math (`ceil(n/block_size)`) in Section 2            | Makes KV math understanding concrete and demonstrable                                         |
| Added Section 6 "What Surprised Me"                                      | Distinguishes document written by someone who ran experiments from someone who just read docs |
| Added Orca citation with code-level proof requirement in Section 5       | Closes the loop on Day 7 pre-reading revision                                                 |
| Added ASCII architecture diagram requirement                             | Produces interview-ready whiteboard artifact                                                  |
| Added V1 preemption cost model as Section 4                              | Connects scheduler logic to empirical Day 9 data                                              |

---

## 🌅 Morning Block (4 hrs) — Finalize the Deliverable

### Step 1 — Finalize the Annotated vLLM Architecture Diagram (3 hrs)

Day 9 produced a rough draft. Today you polish it into a document you would be comfortable sharing with an ML infra hiring manager. Six sections:

---

### Section 1 — Request Lifecycle (V1 File Paths)

The lifecycle table must use V1 paths — these are what you actually traced:

| Step            | File                                    | Key Function               | What Flows Through                                     |
| --------------- | --------------------------------------- | -------------------------- | ------------------------------------------------------ |
| 1. HTTP Ingress | `vllm/entrypoints/openai/api_server.py` | `create_chat_completion()` | OpenAI request → `SamplingParams` + tokenized prompt   |
| 2. AsyncLLM     | `vllm/v1/engine/async_llm.py`           | `generate()`               | InputProcessor tokenizes → EngineCoreClient sends over |

→ returns async generator |
| 3. EngineCore | `vllm/v1/engine/core.py` | `step()` | Receives request via ZMQ, runs scheduler + model, returns outputs via ZMQ |
| 4. Scheduler | `vllm/v1/core/sched/scheduler.py` | `schedule()` | Per-iteration batch selection: Phase 1 RUNNING, Phase 2 WAITING promotion |
| 5. KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots()` / `free()` | Physical KV blocks assigned/released per request |
| 6. GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` | `execute_model()` + `sample_tokens()` | Batch → GPU forward pass → logits → sampled tokens |
| 7. OutputProcessor | `vllm/v1/engine/async_llm.py` | detokenize | Tokens → text, yielded via async generator → SSE stream to client |

V1 key design: AsyncLLM and EngineCore run in **separate processes** connected by ZMQ + msgpack. InputProcessor (tokenization) and OutputProcessor (detokenization) run on the API server side. EngineCore owns the GPU.

For each row write 2–3 sentences: _what_ the component does, _why_ it exists, and _what would break if it were removed._

---

### Section 2 — KV Cache Block Allocation

Two parts: the math, then your instrumentation data.

**Block math (show this explicitly):**

```
block_size = 16 tokens

prompt = 100 tokens  → blocks_needed = ceil(100  / 16) =  7 blocks
prompt = 1000 tokens → blocks_needed = ceil(1000 / 16) = 63 blocks
prompt = 2000 tokens → blocks_needed = ceil(2000 / 16) = 125 blocks
prompt = 4000 tokens → blocks_needed = ceil(4000 / 16) = 250 blocks
```

Connect this to V1's block pool structure:

- `kv_cache_manager.py` maintains a `KVCacheCoordinator` → `BlockPool` → `FreeKVCacheBlockQueue` (doubly-linked list) — physical blocks pre-allocated at server startup
- `allocate_slots(request)` pops blocks from the pool; `free(request)` returns them
- When `allocate_slots()` returns `None`, the scheduler either preempts a RUNNING request (Phase 1) or stops promoting WAITING requests (Phase 2)

**Instrumentation data:**

Pull your Day 9 experiment results here:

- Observed blocks for 100-token, 2K-token, 4K-token prompts vs. theoretical (`ceil(n / block_size)`)
- At what concurrent request count did block exhaustion first occur?
- Did observed numbers match theoretical? If not — why?

---

### Section 3 — Scheduler Decision Logic (V1)

The V1 scheduler has **no SWAPPED state and no swap path.** The decision tree is:

```
Per iteration (one call to schedule()):

  Phase 1 — Service RUNNING requests:
    for each request in self.running:
      result = kv_cache_manager.allocate_slots(request)
      if result is None:
           # memory pressure — preempt lowest-priority RUNNING request
           while allocation fails:
               victim = self.running.pop()
               _preempt_request(victim):
                   kv_cache_manager.free(victim)
                   victim.num_computed_tokens = 0
                   self.waiting.prepend(victim)  # front of queue
           retry allocation for the request that triggered preemption

  Phase 2 — Promote from WAITING (only if no preemptions in Phase 1):
    for each request in self.waiting:
      if kv_cache_manager.allocate_slots(request) succeeds
         AND token budget has capacity:
           request → RUNNING
      else:
           break  # FCFS: don't skip, maintain order

  emit SchedulerOutputs → GPUModelRunner
```

**Request State Machine (V1):**

```
              ┌──────────┐
   arrival →  │ WAITING  │ ◄──────────────────────┐
              └────┬─────┘                         │
                   │ can_allocate() == True         │ preempt:
                   │ AND budget available           │   free() KV blocks
                   ▼                               │   num_computed_tokens = 0
              ┌──────────┐                         │
              │ RUNNING  │ ────── memory pressure ──┘
              └────┬─────┘
                   │ EOS token OR max_tokens reached
                   ▼
              ┌──────────┐
              │ FINISHED │
              └──────────┘
```

> **Note:** There is no SWAPPED state in V1. The RUNNING → SWAPPED → RUNNING path present in V0 was removed. Preemption in V1 always returns the request to WAITING with `num_computed_tokens = 0` — full re-prefill from scratch.

---

### Section 4 — Preemption Cost Model

| Preemption Type        | V0                               | V1                                   |
| ---------------------- | -------------------------------- | ------------------------------------ |
| Swap to CPU            | ✅ Available                     | ❌ Removed                           |
| Recompute (re-prefill) | ✅ Available                     | ✅ Only option                       |
| Cost driver            | CPU↔GPU transfer ∝ KV cache size | Prefill re-execution ∝ prompt length |
| Latency impact         | Proportional to sequence KV size | Proportional to prompt token count   |

**V1 preemption path in code (`scheduler.py`, `_preempt_request()`):**

```python
_preempt_request(request):
    kv_cache_manager.free(request)      # blocks returned to pool immediately
    request.num_computed_tokens = 0     # reset — must re-prefill from scratch
    self.waiting.prepend(request)       # front of WAITING queue
```

**Preemption is memory-neutral, compute-wasteful:** Freeing N blocks from the victim makes exactly N blocks available. When the victim is rescheduled it consumes at most N blocks. The block pool doesn't shrink. The waste is entirely compute — re-prefill discards all prior decode work.

Write 1 paragraph: _Why did V1 remove swap?_ Reference your Day 9 data — did you observe preemption-induced latency spikes? What was the magnitude?

---

### Section 5 — Continuous Batching (Orca Connection)

> The vLLM scheduler does not schedule **requests**. It schedules **iterations of requests.**

Explain the mechanism in your own words, then anchor it in code:

1. **Orca insight:** Naive batching locks a batch until the longest sequence finishes. Iteration-level scheduling re-evaluates every forward pass — finished sequences leave immediately, waiting sequences join immediately.

2. **Why the KV cache makes this possible:** Without the KV cache, every token would require recomputing the full prompt — iteration-level scheduling would be prohibitively expensive. The KV cache stores prior computation, so each iteration only pays the cost of one new token per running sequence.

3. **Code proof (from your Day 7 trace):** Cite the specific function and line number in `scheduler.py` where a finished sequence is removed from the running set AND a waiting sequence is promoted — both within a single `schedule()` call. This is continuous batching in code.

**ASCII diagram:**

```
Request A (long)  ████████████████████████░░░░░░░░░░
Request B (short) ████████
Request C                   ░░░░░░░████████████████
                            ↑
                     B finishes in update_from_output()
                     → blocks freed → C promoted next schedule() call

█ = actively generating    ░ = waiting
```

**Timing detail:** Finish-freed blocks (from `update_from_output()` → `_free_request()`) are available in the **next** scheduler iteration, because `update_from_output()` runs after `schedule()` in the same step. Preemption-freed blocks are available in the **same** step.

**Parameter sensitivity table** (from Day 6 experiments):

| Parameter                  | Values Tested   | Key Finding                                                          |
| -------------------------- | --------------- | -------------------------------------------------------------------- |
| `--max-num-seqs`           | 1, 4, 8, 16, 32 | [fill from your data]                                                |
| `--gpu-memory-utilization` | 0.5 → 0.95      | [fill from your data]                                                |
| `--max-num-batched-tokens` | 512 → 4096      | TTFT = 37.6 + 0.228 × prompt_tokens, R²=0.9966, ~4,386 tok/s prefill |

> The TTFT regression from Exp 3e confirms prefill is compute-bound and linear in prompt length. This is the quantitative baseline for Phase B interference reasoning.

---

### Section 6 — What Surprised Me

3–5 bullets. Write from your actual experiment experience — not what you expected, but what the data actually showed. Examples of the kind of observation that belongs here:

- "Preemption cost was larger than expected — a 2K-token sequence being preempted and recomputed added ~X ms TTFT to the re-queued request"
- "Block allocation matched theoretical `ceil(n/16)` exactly — the KV math from Day 3 was precise to the block"
- "The TTFT regression (R²=0.9966) was much tighter than expected — prefill is genuinely linear in prompt length with minimal noise"
- "The TTFT cliff at c=12→14 was binary, not gradual — TTFT jumped 13x in two concurrency steps"
- "Preemption is memory-neutral — freeing N blocks and consuming at most N blocks on reschedule. The waste is entirely compute, not memory"

This section is what distinguishes a document written by someone who ran the experiments from someone who just read the code.

---

### Step 2 — Self-Test (1 hr)

Close the document. Answer all four in writing from memory. If any answer has a gap, fix it before calling the deliverable done:

**Q1.** What happens when a request arrives at vLLM?
_(Name the 5 components in order, with V1 file names.)_

**Q2.** How does the V1 scheduler decide what runs each iteration?
_(Cover: Phase 1 services RUNNING first with preemption if allocate_slots() fails, Phase 2 promotes WAITING only if no preemptions occurred. Do not mention swap.)_

**Q3.** When and why does preemption occur in V1, and what does it cost?
_(Cover: block exhaustion trigger, recompute-only path, cost proportional to prompt length.)_

**Q4.** What is PagedAttention and why does it matter?
_(Your answer must mention: logical vs. physical blocks, fragmentation elimination, copy-on-write for parallel sampling.)_

---

## 🌆 Afternoon Block (4 hrs) — Buffer + Week 3 Prep

### Step 3 — Catch-Up (variable, up to 3 hrs)

Work through anything open from Days 6–9. Common gaps at this point:

- Instrumentation patch producing unexpected output?
- Day 9 concurrent block competition experiment missing some concurrency levels (1 / 2 / 4 / 8 / 16)?
- Section 4 (preemption cost) thin because preemption was not directly observable?
- Day 6 parameter sensitivity table incomplete?

> **Preemption was directly observed** with Qwen2.5-3B at 0.45 utilization: 89 preemptions, 1,398 ALLOC_FAIL events, 11 timeouts. Request `9474796f` was preempted 6 times, wasting 2,156 tokens of decode work. This data anchors Section 4.

---

### Step 4 — Week 3 Prep (1 hr)

Read **abstract + introduction only** for two papers. You are building a mental map, not studying:

**AWQ — Activation-aware Weight Quantization**
`https://arxiv.org/abs/2306.00978`

Key idea: not all weights are equally important. Weights that correspond to large activations are salient — quantizing them aggressively causes disproportionate error. AWQ identifies these weights and scales them before quantization to preserve accuracy.

**GPTQ — Accurate Post-Training Quantization**
`https://arxiv.org/abs/2210.17323`

Key idea: layer-by-layer quantization using second-order information (Hessian approximation) to minimize reconstruction error per layer. Requires a small calibration dataset but no retraining.

After reading, write **two sentences per paper** answering:

1. What problem does it solve?
2. What is the core mechanism?

Compress aggressively — this forces gaps to surface immediately.

---

## End-of-Day Checklist

- [ ] All 6 sections present; each has numbers or code references (no prose-only sections)
- [ ] V1 file paths used throughout (`kv_cache_manager.py`, `v1/core/scheduler.py`)
- [ ] SWAPPED state and swap path do **not** appear anywhere — preemption = recompute only
- [ ] Block math (`ceil(n / block_size)`) shown explicitly in Section 2
- [ ] Instrumentation data from Day 9 incorporated in Sections 2 and 6
- [ ] Day 6 Exp 3e regression line (`TTFT = 37.6 + 0.228 × prompt_tokens`, R²=0.9966) appears in Section 5
- [ ] Orca insight cited by name in Section 5 with code-level proof (function + line number)
- [ ] ASCII architecture diagram present in Section 5
- [ ] V1 state machine diagram present in Section 3 (no SWAPPED state)
- [ ] Self-test Q1–Q4 answered in writing with no open gaps
- [ ] AWQ + GPTQ 2-sentence summaries written

---

## Difficulty & Time Estimates

| Task                           | Estimated   | Realistic |
| ------------------------------ | ----------- | --------- |
| Finalize document (6 sections) | 3 hrs       | 2–4 hrs   |
| Self-test (written answers)    | 1 hr        | 30–60 min |
| Catch-up from Days 6–9         | up to 3 hrs | variable  |
| Week 3 prep (AWQ + GPTQ)       | 1 hr        | accurate  |

Day 10 difficulty: **Medium** (synthesis, not discovery)

| Day | Difficulty |
| --- | ---------- |
| 6   | Medium     |
| 7   | Hard       |
| 8   | Hard       |
| 9   | Hard       |
| 10  | Medium     |

---

## What Day 11 Expects From You

Monday's quantization experiments use FP16 as the baseline. Your Day 6 parameter sensitivity table and TTFT regression are that baseline. If those numbers are not clean and documented today, you will be interpolating from memory on Monday.

Day 11 will load FP16, INT8-AWQ, and INT4-GPTQ variants of the same model and benchmark them at concurrency levels 1 / 4 / 8 / 16. The regression line from Exp 3e is the anchor point for interpreting those results.

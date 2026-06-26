# Day 7 (Tue) — Bounded Code Trace: Part 1 (Revised)

## Changes from Original

| What Changed | Why |
|---|---|
| Added **Pre-Reading Block** (30 min) on Orca / iteration-level scheduling before touching code | The scheduler code is much harder to parse without the right mental model. Reading "the scheduler schedules *iterations*, not requests" first makes `step()` and `schedule()` immediately legible. |
| Added sub-task in Trace Step 3: find the **continuous batching join/leave path** in code | This is the concrete code proof of the Orca insight. Without it, the trace is descriptive but not explanatory. |
| Added **SequenceGroup state machine diagram** as explicit afternoon deliverable | The original punted this to Day 8 refinement. It belongs here — the state transitions are freshest while reading `scheduler.py`. |
| Added **preemption cost annotation** on the state machine | Connects the "hidden performance cliff" (KV cache full → preemption → latency explosion) to the Day 9 mini-collapse experiment. |
| Added **interview answer self-test** as end-of-day gate | Forces synthesis. If you can't articulate "why is vLLM faster than naive batching?" in 3 sentences from memory, you have a gap. |

---

## Pre-Reading (30 min) — The Orca Mental Model

**Do this before opening any code.** The goal is to load the correct mental model so the code trace is legible on first read.

### Core Insight

> The vLLM scheduler does not schedule requests. It schedules **iterations of requests**.

This comes from the **Orca paper** and is the foundation of **continuous batching**.

### What to internalize

- **Naive batching** locks a batch for the entire generation. Short requests pad to the longest. New requests wait. GPU utilization collapses.
- **Iteration-level scheduling** re-evaluates the batch every single forward pass. Finished sequences leave immediately. Waiting sequences can join immediately. The batch stays full.
- The `step()` loop you're about to trace does exactly this: one call to `schedule()` + one call to `execute_model()` = one iteration across a dynamic set of sequences.
- The scheduler's real job is deciding **which sequences participate in the next iteration**, balancing three competing resources: GPU compute (too few sequences → GPU idle), GPU memory (too many sequences → KV cache OOM), and latency (too many sequences → per-step latency increases).
- Continuous batching only works **because of the KV cache**. Without it, every token would require recomputing the entire prompt. With it, each iteration only computes the next token, making iteration-level scheduling feasible.

### What to read

If you have the reviewer's "Orca insight" document, read sections 1–6. Otherwise, read the Orca paper abstract + Section 3 (iteration-level scheduling) — 15 minutes max.

**Checkpoint:** Before proceeding, you should be able to answer: *"Why can't you do continuous batching without a KV cache?"* If not, re-read.

---

## Morning (4 hrs) — Trace: Request Ingress → Scheduler

### Setup (~15 min)

- **Clone vLLM source** and pin to v0.6.6 (your known-working version from Day 6):
  ```bash
  git clone https://github.com/vllm-project/vllm.git
  cd vllm
  git checkout v0.6.6
  ```
- Open in your editor with Python navigation (VS Code + Pylance, or PyCharm). You'll be doing heavy grep and symbol-jumping today.

### Trace Step 1: HTTP Request Ingress (~45 min)

- Start at `vllm/entrypoints/openai/api_server.py`
- Follow the request: how does it get from the OpenAI-compatible endpoint to the engine?
- Find the route handler for `/v1/chat/completions` and `/v1/completions`
- Trace the data transformation: OpenAI request format → `SamplingParams` + tokenized prompt
- Find the handoff point where the request enters the engine
- **Document:** file name, function name, key data structures created, handoff target

Expected trace shape:
```
POST /v1/chat/completions
  → create_chat_completion()
  → parse request → SamplingParams constructed
  → prompt tokenized via tokenizer
  → engine.generate(prompt, sampling_params, request_id)
```

**Tip:** If you can't find something where expected, use `grep -r "def generate" --include="*.py"` or your editor's symbol search. The logical flow is stable even if file paths shifted between versions.

### Trace Step 2: AsyncLLMEngine (~1 hr)

- Find `vllm/engine/async_llm_engine.py`
- **Key questions:**
  - What does `generate()` actually do? (It doesn't run inference immediately — it adds the request to a queue. Find where.)
  - What is `RequestOutput`? Find its definition. Note its fields.
  - What is `SamplingParams`?
  - Find the engine's main loop — the `step()` method.
  - **Connect to pre-reading:** Each call to `step()` is one scheduler iteration: pick sequences, run one forward pass, return outputs. This is the Orca loop in code.
  - How does the engine know when a request is done?

**Document:**
```
AsyncLLMEngine.generate():
  - Creates SequenceGroup from prompt + params
  - Adds to scheduler's waiting queue
  - Returns async generator that yields RequestOutput

Engine loop (step()):
  - Calls scheduler.schedule() → get batch for this iteration
  - Calls model_runner.execute_model() → one forward pass
  - Processes outputs → update sequence states
  - Yield results to waiting generators
```

### Trace Step 3: Scheduler (~1.5 hrs)

- Find `vllm/core/scheduler.py`
- **This is the most important file in vLLM for your purposes.**

#### 3a: SequenceGroup States

Find the state enum. You're looking for:
- `WAITING` — request arrived, not yet allocated KV cache blocks
- `RUNNING` — actively generating tokens (participating in iterations)
- `SWAPPED` — evicted to CPU memory due to memory pressure
- `FINISHED` — generation complete (EOS or max length)

#### 3b: Read `schedule()` Carefully

The logic goes roughly:
1. Try to move SWAPPED sequences back to GPU (if there's space)
2. Try to move WAITING sequences to RUNNING (if there's space)
3. If not enough space, preempt RUNNING sequences (move to SWAPPED)
4. Return the batch of sequences to run this iteration

For each decision, note:
- What determines "enough space"? (This connects to the block manager)
- How does priority work? (FCFS? Something else?)
- What's the preemption policy? (Swap to CPU vs. recompute?)

**Document as pseudocode:**
```
schedule():
  budget = SchedulingBudget(token_budget, max_num_seqs)

  # Phase 1: Try to resume swapped sequences
  for seq_group in swapped_queue:
    if block_manager.can_swap_in(seq_group):
      swap_in(seq_group)

  # Phase 2: Try to start waiting sequences
  for seq_group in waiting_queue:
    if block_manager.can_allocate(seq_group):
      allocate(seq_group)
      move to running

  # Phase 3: If over budget, preempt
  while over_budget:
    victim = running_queue.pop_last()
    preempt(victim)  # swap out or mark for recompute
```

#### 3c: Find the Continuous Batching Path (NEW)

This is the sub-task that proves the Orca insight in code:

- **Find where a finished sequence leaves the running batch.** What function detects EOS / max length? How does that sequence get removed from the running set within the same `schedule()` call?
- **Find where a waiting sequence joins the running batch.** Within the same `schedule()` call, after a slot opens, how does a WAITING sequence get promoted to RUNNING?
- **Annotate the exact code path** where both happen in a single iteration. This is continuous batching — the batch composition changes every step.

**What to write down:** The function names and line numbers where (a) a finished sequence is removed and (b) a new sequence is admitted, both within one `schedule()` invocation.

---

## Afternoon (4 hrs) — Block Manager → Model Runner + State Machine

### Trace Step 4: BlockSpaceManager (~1.5 hrs)

- Find `vllm/core/block_manager.py` (in v0.6.6 it may be `block_manager.py` rather than `block_manager_v2.py`)
- This is PagedAttention's implementation — the solution to your Day 3 fragmentation problem.

**Key things to find and document:**
- **Block size** — how many tokens per block? (typically 16)
- **`can_allocate(seq_group)`** — what does it check? How many free blocks needed?
- **`allocate(seq_group)`** — how does it assign physical blocks to a sequence's logical blocks?
- **`free(seq_group)`** — how are blocks returned to the free pool?
- **Block table** — the mapping from logical block index → physical block index. Find it.

**Connection to Day 3:** In your fragmentation simulation, contiguous allocation wasted memory. Here, the block table lets a sequence's KV cache live in non-contiguous physical blocks. That's the whole trick of PagedAttention.

**Document:**
```
BlockSpaceManager:
  total_blocks = GPU_KV_MEMORY / (block_size * kv_per_token)
  free_blocks: list of available physical block IDs
  block_tables: dict[seq_id] → list[physical_block_id]

  can_allocate(seq_group):
    needed = ceil(num_tokens / block_size)
    return len(free_blocks) >= needed

  allocate(seq_group):
    for each logical block needed:
      physical = free_blocks.pop()
      block_table[seq_id].append(physical)

  free(seq_group):
    for physical_block in block_table[seq_id]:
      free_blocks.append(physical_block)
```

### Trace Step 5: ModelRunner → Token Streaming (~1 hr)

- Find `vllm/worker/model_runner.py`
- You don't need to go kernel-deep. Understand the handoff:
  - How does `execute_model()` take the scheduled batch and run inference?
  - How are prefill and decode handled differently? (Prefill processes all prompt tokens at once; decode processes one token per sequence)
  - How do generated tokens flow back to the engine?

**Document:** The handoff chain from scheduler output → GPU forward pass → new tokens returned to engine.

### Build the Architecture Diagram (~30 min)

Synthesize Steps 1–5 into a single diagram showing the full request lifecycle with actual file paths and function names:

```
[api_server.py] POST /v1/chat/completions
  → parse request, create SamplingParams
  → [async_llm_engine.py] generate() → add SequenceGroup to waiting queue
    → [scheduler.py] schedule() → select sequences for this iteration
      → [block_manager.py] can_allocate() / allocate() / free()
    → [model_runner.py] execute_model() → GPU forward pass
  → stream tokens back via SSE
```

For each box: key function names, data structures in, data structures out. Accuracy matters more than aesthetics — you'll refine on Day 8.

### Draw the SequenceGroup State Machine (NEW — 45 min)

This is high-value interview material. Draw it as a formal state diagram:

```
              ┌──────────┐
   arrival →  │ WAITING  │
              └────┬─────┘
                   │ can_allocate() == True
                   ▼
              ┌──────────┐
              │ RUNNING  │ ←──────────────┐
              └────┬─────┘                │
                   │                      │ swap_in (blocks available)
            ┌──────┴──────┐               │
            │             │          ┌────┴─────┐
            ▼         preempt →      │ SWAPPED  │
      ┌──────────┐  (memory pressure)└──────────┘
      │ FINISHED │
      └──────────┘
```

**For each transition, document:**
- **WAITING → RUNNING:** Scheduler has budget AND `block_manager.can_allocate()` returns True
- **RUNNING → FINISHED:** EOS token generated or `max_tokens` reached
- **RUNNING → SWAPPED:** Memory pressure — scheduler preempts this sequence, KV cache blocks swapped to CPU
- **SWAPPED → RUNNING:** Space freed up, KV cache blocks swapped back to GPU
- **SWAPPED → WAITING (recompute path):** Alternative preemption policy — discard KV cache entirely, re-prefill when resumed

**Annotate preemption costs (NEW):**
- Swap path cost: CPU↔GPU memory transfer time (proportional to KV cache size of evicted sequence)
- Recompute path cost: full prefill re-execution (proportional to prompt length)
- Note: this cost annotation connects directly to the Day 9 "mini-collapse" experiment, where you'll observe the latency explosion when preemption starts

**Cite the code:** For each transition, note the function in `scheduler.py` that handles it.

---

## End-of-Day Output

1. ✅ Written trace of all 5 steps with file names and function names
2. ✅ Architecture diagram showing the full request lifecycle (code-referenced)
3. ✅ SequenceGroup state machine with transition conditions, preemption costs, and code references
4. ✅ Annotated continuous batching path: where a finished sequence leaves and a new sequence joins within one `schedule()` call
5. ✅ Interview self-test (below)

### Interview Self-Test (NEW — 15 min gate)

Close your notes. Answer this from memory in ≤3 sentences:

> **"Why is vLLM faster than naive batching?"**

Your answer should hit these three points:
1. Iteration-level scheduling (batch changes every forward pass)
2. Finished sequences leave immediately, new sequences join immediately
3. This keeps GPU batch size consistently high

If you can't do it cleanly, re-read your scheduler trace and the pre-reading until you can. This is a near-guaranteed interview question.

---

## What's Ahead

- **Day 8:** Refine the diagram, choose and implement your instrumentation patch (block events, preemption logging, or prefill/decode timing)
- **Day 9:** Run instrumented experiments — the "mini-collapse" experiment will show the preemption → latency explosion cliff you annotated today
- **Later (Day 9+ or Phase B):** The *second* hidden insight — scheduler limits vs. KV cache limits, and why confusing them causes production outages — connects to admission control. Don't chase it today.

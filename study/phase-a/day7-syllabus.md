Let's walk through Day 7 — **Bounded Code Trace: Part 1**. This is where you go from "vLLM is running" to "I can explain exactly what happens at every layer, citing specific files and functions." It's one of the highest-value days in the whole residency for interview prep.

Here's your step-by-step plan:

---

## Morning Block (4 hrs) — Trace: Request Ingress → Scheduler

### Step 1: Clone and Pin vLLM Source (~15 min)

Before anything else, pin to a specific release so the code doesn't shift under you:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git tag -l  # pick a recent stable release
git checkout v0.6.x  # or whatever recent stable tag you see
```

Open the repo in your editor of choice. You'll be doing a lot of grep and file-jumping today — having good code navigation (VS Code with Python extension, or PyCharm) will save you time.

### Step 2: Trace HTTP Request Ingress (~45 min)

Start at the entrypoint — this is where an OpenAI-compatible API request first hits vLLM:

**File:** `vllm/entrypoints/openai/api_server.py`

What to look for:
- Find the route handler for `/v1/chat/completions` and `/v1/completions`
- Follow how the incoming JSON request gets parsed into vLLM's internal representation
- Note the data transformation: OpenAI request format → vLLM's `SamplingParams` + tokenized prompt
- Look for where the request gets handed off to the engine

**What to document:** Write down the function names, the key data structures created, and the handoff point. Something like:

```
1. POST /v1/chat/completions → create_chat_completion()
2. Request parsed → SamplingParams constructed
3. Prompt tokenized via tokenizer
4. Handed to engine.generate(prompt, sampling_params, request_id)
```

**Tip:** vLLM's codebase has evolved significantly across versions. If you can't find something where the syllabus says, use `grep -r "def generate" --include="*.py"` or your editor's symbol search. The architecture may have been refactored into slightly different file paths, but the logical flow is the same.

### Step 3: Trace AsyncLLMEngine (~1 hr)

**File:** `vllm/engine/async_llm_engine.py` (or potentially `vllm/engine/async_llm_engine.py` wrapping `llm_engine.py`)

Key questions to answer as you read:
- What does `generate()` actually do? It doesn't run inference immediately — it adds the request to a queue. Find where.
- What is `RequestOutput`? Find its definition and note its fields — this is what gets returned to the caller.
- Find the engine's main loop — the `step()` method. Each call to `step()` is one scheduler iteration: pick sequences, run one forward pass, return outputs.
- How does the engine know when a request is done?

**What to document:**
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

### Step 4: Trace the Scheduler (~1.5 hrs)

**File:** `vllm/core/scheduler.py`

This is the most important file for your purposes. Take your time here.

**SequenceGroup states** — Find the state enum. You're looking for states like:
- `WAITING` — request arrived, not yet running
- `RUNNING` — actively generating tokens
- `SWAPPED` — evicted to CPU memory due to memory pressure
- `FINISHED` — generation complete

**Read `schedule()` carefully.** The logic goes roughly:
1. Try to move SWAPPED sequences back to GPU (if there's space)
2. Try to move WAITING sequences to RUNNING (if there's space)
3. If not enough space, preempt RUNNING sequences (move to SWAPPED)
4. Return the batch of sequences to run this iteration

For each decision, note:
- What determines "enough space"? (This connects to the block manager)
- How does priority work? (FCFS? Something else?)
- What's the preemption policy? (Swap to CPU vs. recompute?)

**What to document:** Write the decision tree as pseudocode:

```
schedule():
  budget = SchedulingBudget(token_budget, max_num_seqs)
  
  # Phase 1: Try to resume swapped sequences
  for seq_group in swapped_queue:
    if block_manager.can_swap_in(seq_group):
      swap_in(seq_group)  # move KV cache back to GPU
  
  # Phase 2: Try to start waiting sequences  
  for seq_group in waiting_queue:
    if block_manager.can_allocate(seq_group):
      allocate(seq_group)  # reserve KV cache blocks
      move to running
  
  # Phase 3: If over budget, preempt
  while over_budget:
    victim = running_queue.pop_last()  # lowest priority
    preempt(victim)  # swap out or mark for recompute
```

The actual code will be more nuanced, but capture the logical flow.

---

## Afternoon Block (4 hrs) — Block Manager → Model Runner + State Machine

### Step 5: Trace the BlockSpaceManager (~1.5 hrs)

**File:** `vllm/core/block_manager.py` (might be `block_manager_v2.py` in newer versions)

This is PagedAttention's implementation — the solution to your Day 3 fragmentation problem.

Key things to find and document:
- **Block size** — how many tokens per block? (typically 16)
- **`can_allocate(seq_group)`** — what does it check? How many free blocks needed?
- **`allocate(seq_group)`** — how does it assign physical blocks to a sequence's logical blocks?
- **`free(seq_group)`** — how are blocks returned to the free pool?
- **Block table** — this maps logical block index → physical block index. Find it.

**Connect to Day 3:** In your fragmentation simulation, contiguous allocation wasted memory. Here, the block table lets a sequence's KV cache live in non-contiguous physical blocks. That's the whole trick.

**What to document:**
```
BlockSpaceManager:
  - total_blocks = GPU_KV_MEMORY / (block_size * kv_per_token)
  - free_blocks: list of available physical block IDs
  - block_tables: dict[seq_id] → list[physical_block_id]
  
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

### Step 6: Trace ModelRunner → Token Output (~1 hr)

**File:** `vllm/worker/model_runner.py`

This is where the actual GPU forward pass happens. You don't need to go kernel-deep, but understand:
- How does `execute_model()` take the scheduled batch and run inference?
- How are prefill and decode handled differently? (Prefill processes all prompt tokens at once; decode processes one token per sequence)
- How do generated tokens flow back to the engine?

**What to document:** The handoff chain from scheduler output → GPU → new tokens.

### Step 7: Build the Architecture Diagram (~45 min)

Now synthesize everything into a clean diagram. This should show the full request lifecycle with actual file paths:

```
[api_server.py] POST /v1/chat/completions
  → parse request, create SamplingParams
  → [async_llm_engine.py] generate() → add SequenceGroup to waiting queue
    → [scheduler.py] schedule() → select sequences for this iteration
      → [block_manager.py] can_allocate() / allocate() / free()
    → [model_runner.py] execute_model() → GPU forward pass
  → stream tokens back via SSE
```

For each box, note the key function names and what data flows between them.

### Step 8: Draw the SequenceGroup State Machine (~45 min)

This is excellent interview material. Draw it formally:

```
              ┌──────────┐
   arrival →  │ WAITING  │
              └────┬─────┘
                   │ block_manager.can_allocate() == True
                   ▼
              ┌──────────┐
              │ RUNNING  │ ←──────────────┐
              └────┬─────┘                │
                   │                      │ swap_in (blocks available)
            ┌──────┴──────┐               │
            │             │          ┌────┴─────┐
            ▼             ▼          │ SWAPPED  │
      ┌──────────┐   preempt →      └──────────┘
      │ FINISHED │   (memory pressure)
      └──────────┘
```

For each transition, document:
- **WAITING → RUNNING:** Scheduler has budget, block manager can allocate blocks
- **RUNNING → FINISHED:** EOS token generated or max length reached
- **RUNNING → SWAPPED:** Memory pressure, scheduler preempts this sequence (KV cache swapped to CPU)
- **SWAPPED → RUNNING:** Space freed up, KV cache swapped back to GPU
- **SWAPPED → WAITING (recompute path):** Alternative preemption strategy — discard KV cache, re-prefill when resumed

Note which code path handles each transition — cite the function in `scheduler.py`.

---

## End-of-Day Checklist

By the end of Day 7, you should have:

1. A written trace of all 5 steps with file names and function names
2. The architecture diagram showing the full request lifecycle
3. The SequenceGroup state machine with transition conditions and code references
4. Clear understanding of the scheduler's decision logic (what runs, what waits, what gets preempted)
5. Clear understanding of how the block manager implements PagedAttention

**Practical tip:** Don't try to understand every line of code. The vLLM codebase is large and actively evolving. Your goal is the logical flow and the key decision points. If you find yourself deep in utility code or GPU kernel details, zoom back out.

Want me to help you with any specific step — for instance, navigating a particular part of the vLLM source, or working through the scheduler logic in more detail?
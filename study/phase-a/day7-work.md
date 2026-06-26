# Trace Step 2: AsyncLLM (~1 hr)

## Find `vllm/v1/engine/async_llm.py`

- **Key questions:**
  - What does `generate()` actually do? (It doesn't run inference immediately — it adds the request to a queue. Find where.)
    - shallow function that sends the request to EngineCore via ZMQ

```
  generate() is a consumer, not a producer. It never touches the model. It just:
  1. Passes the request through InputProcessor (tokenization, multimodal processing)
  2. Sends an EngineCoreRequest to EngineCore via EngineCoreClient (ZMQ)
  3. An output_handler asyncio task polls for outputs and dispatches to per-request async generators
  4. generate() yields whatever shows up in its async generator

```

All the actual work — scheduling, GPU forward pass, sampling — happens in the EngineCore subprocess, completely decoupled. AsyncLLM never blocks on GPU execution. The output path is: EngineCore → ZMQ → OutputProcessor (detokenization) → per-request async generator → generate() yields.

This is why vLLM can handle thousands of concurrent generate() calls without blocking: they're all just lightweight async generators waiting on their individual output streams.

### ZeroMQ (ZMQ) — a messaging library for sending data between processes. Think of it as a socket on steroids.

Why vLLM needs it

vLLM V1 runs the API server (AsyncLLM) and the EngineCore in separate processes. The API server handles HTTP and async Python. The EngineCore runs the GPU-heavy step loop. They need to talk to each other without sharing memory.

Options for inter-process communication:

- Python multiprocessing.Queue — works but slow, pickle overhead, no fan-out
- gRPC — full HTTP/2 protocol, overkill for same-machine communication
- Unix sockets — fast but you manage framing/serialization yourself
- ZMQ — fast, handles framing, supports patterns like pub/sub and router/dealer out of the box

What vLLM uses it for

```
  API Process (AsyncLLM)               EngineCore Process
  (FastAPI, async)                     (scheduler, GPU, step loop)
       │                                      │
       │ InputProcessor                       │
       │   tokenize, multimodal → EngineCoreRequest
       │                                      │
       │── ZMQ input socket ──────────────────▶│  "here's a new request"
       │                                      │
       │◀── ZMQ output socket ────────────────│  "here are new tokens"
       │                                      │
       │ OutputProcessor                      │
       │   detokenize, stream → RequestOutput │
```

Two sockets, two directions. The API process sends EngineCoreRequest objects in (via EngineCoreClient), the EngineCore process sends EngineCoreOutputs objects back. Messages are serialized with msgpack. The API process runs InputProcessor (tokenization, multimodal) before sending requests, and OutputProcessor (detokenization, streaming) after receiving outputs.

Why not just run everything in one process?

Python's GIL. The API server needs to handle thousands of concurrent HTTP connections (async I/O). The EngineCore needs to do CPU-intensive scheduling and coordinate GPU work. In one process, they'd fight over the GIL. Separate processes with ZMQ as the bridge lets both run at full speed. V1 explicitly isolates the EngineCore loop so it "focuses exclusively on the scheduler and model executor" — allowing overlap of CPU-intensive tasks (tokenization, detokenization, structured output) with the core execution loop.

## What is `RequestOutput`? Find its definition. Note its fields.

- output of a completion request
- prompt, prompt_token_ids
- outputs
- finished
- metrics

```
RequestOutput — vllm/outputs.py

  The receipt you get back for each request. Contains everything the caller needs:

  RequestOutput:
    request_id          — which request this is
    prompt              — the original text
    prompt_token_ids    — tokenized prompt
    outputs             — list[CompletionOutput], one per beam/sample
      └─ CompletionOutput:
          text            — generated text so far
          token_ids       — generated token IDs
          logprobs        — per-token log probabilities (if requested)
          finish_reason   — "stop" (EOS), "length" (max tokens), or None (still going)
    finished            — bool, is this request done?
    metrics             — timing: arrival, first token, last token latencies

  One RequestOutput is yielded per engine step per request. In streaming mode, each one contains the incremental delta. The finished flag is what tells generate()'s while-loop to stop.
```

## What is `SamplingParams`?

- sampling_params for the request
- an arg to generate()

```
  SamplingParams — vllm/sampling_params.py

  The knobs that control token selection. Passed as an argument to generate():

  SamplingParams:
    temperature         — scales logits before softmax (0 = greedy)
    top_k               — keep only top K candidates
    top_p               — keep smallest set of candidates summing to P probability
    max_tokens           — stop after this many generated tokens
    stop                — stop strings (e.g., ["\n\n", "END"])
    repetition_penalty  — penalize already-seen tokens
    n                   — how many completions to generate per prompt
    logprobs            — how many per-token logprobs to return
    presence_penalty    — penalize tokens that appeared at all
    frequency_penalty   — penalize tokens proportional to their count

  Think of it as the user's "how creative and how long" instruction, translated into math. The model produces logits; SamplingParams determines how those logits become a token.
```

## Find the engine's main loop — the `step()` method.

- `vllm/v1/engine/core.py` — `EngineCore.step()`
- **Connect to pre-reading:** Each call to `step()` is one scheduler iteration: pick requests, run one forward pass, sample tokens, return outputs. This is the Orca loop in code.

```
  step():  (vllm/v1/engine/core.py:376-405)
    1. scheduler.schedule()                          → pick which requests run this iteration
    2. executor.execute_model(scheduler_output)       → GPU forward pass (non-blocking, returns Future)
    3. scheduler.get_grammar_bitmask(scheduler_output) → structured output bitmask (computed while GPU runs)
    4. future.result()                                → wait for forward pass to finish
    5. executor.sample_tokens(grammar_output)          → sample from hidden states
    6. scheduler.update_from_output(scheduler_output, model_output) → append new tokens, mark finished
    → return EngineCoreOutputs
```

Two-phase design: `execute_model()` runs the forward pass and stores hidden states on GPU, then `sample_tokens()` samples from them. Grammar bitmask computation overlaps with the GPU forward pass.

Runs continuously in the EngineCore subprocess. Every request in the system — prefilling, decoding, just arrived — competes for the same batch every step. V1's unified scheduler treats prompt tokens and output tokens identically via a `{request_id: num_tokens}` budget. No separate prefill and decode phases — just one loop that processes whatever the scheduler picks.

## How does the engine know when a request is done?

```
  How the Engine Knows a Request Is Done

  Multiple finish signals, checked in scheduler.update_from_output() after each step:

  1. EOS token — the model sampled the end-of-sequence token → FINISHED_STOPPED
  2. Max length — len(generated_tokens) >= sampling_params.max_tokens → FINISHED_LENGTH_CAPPED
  3. Stop strings (e.g., "\n\n") — checked against detokenized output → FINISHED_STOPPED
  4. Repetition limit — excessive token repetition → FINISHED_REPETITION
  5. Client abort — request cancelled by caller → FINISHED_ABORTED
  6. Error — runtime failure (e.g., KV transfer error) → FINISHED_ERROR
```

When finished:

- Scheduler calls `_free_request(request)` which removes the request from RUNNING
- KV cache manager frees its blocks via `kv_cache_manager.free(request)`
- EngineCoreOutputs sent back through ZMQ → OutputProcessor (detokenization) → per-request async generator → generate() yields the final RequestOutput

Per-request completion is determined by the token-level and status signals above. The scheduler's `has_requests()` check (any WAITING or RUNNING) tells whether there's any work left globally.

## Trace (V1)

```
@router.post("/v1/chat/completions")
  → api_server::create_chat_completion()
  → OpenAiServingChat.create_chat_completion()
  → AsyncLLM.generate()                               (vllm/v1/engine/async_llm.py)
    → InputProcessor.process_inputs()                  tokenize, multimodal processing
    → EngineCoreClient.add_request(EngineCoreRequest)  send over ZMQ to EngineCore subprocess
                                                       ──── process boundary (ZMQ) ────
    → EngineCore receives request                      (vllm/v1/engine/core.py)
    → Scheduler.add_request(Request)                   add to waiting queue
    → EngineCore.step() loop picks it up               schedule → execute_model → sample_tokens
```

# 3a: Request States (V1)

V1 replaced `SequenceGroup` with a flat `Request` class (`vllm/v1/request.py`). One Request = one token stream. V0's `SequenceGroup` held multiple `Sequence` objects for beam search and `best_of` sampling — V1 removed both features, eliminating the need for the grouping abstraction.

Find `RequestStatus` (`vllm/v1/request.py:295`):

**Core states:**
- `WAITING` — request arrived, in waiting queue, not yet allocated KV cache blocks
- `RUNNING` — actively generating tokens (holding KV blocks, participating in iterations)
- `PREEMPTED` — was running, blocks freed, requeued to front of waiting queue for re-prefill

**Blocked-waiting sub-states** (in `skipped_waiting` queue, not `waiting`):
- `WAITING_FOR_FSM` — structured output grammar not yet compiled
- `WAITING_FOR_REMOTE_KVS` — P/D disaggregation: waiting for async KV transfer
- `WAITING_FOR_STREAMING_REQ` — streaming input session paused (no next chunk yet)

**Finished states** (all states > PREEMPTED):
- `FINISHED_STOPPED` — EOS token or stop string matched
- `FINISHED_LENGTH_CAPPED` — hit max_tokens
- `FINISHED_ABORTED` — client abort
- `FINISHED_IGNORED` — prompt too long
- `FINISHED_ERROR` — runtime error
- `FINISHED_REPETITION` — repetition stop

**No SWAPPED state.** V1 removed GPU↔CPU KV cache swapping entirely. Preemption = discard all KV blocks + requeue for full re-prefill. Simpler design, but higher preemption cost (proportional to prompt length).

### State Machine Diagram (V1)

```
                 ┌───────────┐
   arrival ───▶  │  WAITING  │ ◀─────────────────────────────┐
                 └─────┬─────┘                               │
                       │                                     │
                       │ allocate_slots() succeeds           │ _preempt_request()
                       │ + token_budget available            │   (scheduler.py:931)
                       │ + running < max_num_seqs            │
                       │   (scheduler.py:812)                │ → kv_cache_manager.free()
                       │                                     │ → num_computed_tokens = 0
                       ▼                                     │ → status = PREEMPTED
                 ┌───────────┐                               │ → prepend to waiting queue
                 │  RUNNING  │ ──── memory pressure ─────────┘
                 └─────┬─────┘
                       │
                       │ EOS / max_tokens / stop string /
                       │ abort / error / repetition
                       │   (scheduler.py: update_from_output)
                       │
                       │ → _free_request()
                       │ → kv_cache_manager.free()
                       ▼
                 ┌───────────┐
                 │ FINISHED  │
                 └───────────┘
                   (one of: FINISHED_STOPPED, FINISHED_LENGTH_CAPPED,
                    FINISHED_ABORTED, FINISHED_IGNORED,
                    FINISHED_ERROR, FINISHED_REPETITION)
```

**Transition costs:**
- **WAITING → RUNNING:** Full prefill cost ∝ prompt length (or partial if chunked)
- **RUNNING → FINISHED:** Free. Blocks released, request done.
- **RUNNING → WAITING (preempt):** All KV discarded. Re-prefill cost = original prefill time + wait time in queue. This is the TTFT spike you see during collapse.
- **WAITING → WAITING (alloc fail):** Zero cost, but the request stays queued. This is the steady-state under pressure — `BLOCK_ALLOC_FAIL` in Day 8 logs.

# 3b Read `schedule()` Carefully

## V1 Unified Scheduler (`vllm/v1/core/sched/scheduler.py`)

V1 has a single `schedule()` method — no separate `_schedule_default()` or `_schedule_chunked_prefill()`. There’s no prefill/decode distinction. Each request simply tracks `num_computed_tokens` and the scheduler advances it toward `num_tokens` by allocating a token budget per step.

**Key state:**
```python
self.waiting: RequestQueue          # New / preempted requests not yet running
self.skipped_waiting: RequestQueue  # Requests blocked on async deps (FSM, remote KV, streaming)
self.running: list[Request]         # Requests currently holding KV cache blocks
# NO self.swapped — V1 has no swap-to-CPU mechanism
```

```python
def schedule():
    token_budget = max_num_batched_tokens
    request_budget = max_num_seqs

    # ─── Phase 1: Schedule RUNNING requests ───
    # Iterate self.running in order. For each request:
    for request in self.running:
        num_new_tokens = request.num_tokens - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        # Try to allocate KV cache blocks for the new tokens
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

        if new_blocks is not None:
            # Success — request continues this iteration
            scheduled_running_reqs.append(request)
            token_budget -= num_new_tokens
        else:
            # Allocation failed — preemption loop
            while not enough free blocks:
                victim = self.running.pop()  # FCFS: evict lowest-priority (last)
                _preempt_request(victim)
                # victim.status = PREEMPTED
                # kv_cache_manager.free(victim) — all blocks immediately freed
                # victim.num_computed_tokens = 0 — must re-prefill from scratch
                # self.waiting.prepend(victim) — front of queue for priority resumption

            # Retry allocation with freed blocks
            new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

    # ─── Phase 2: Schedule WAITING requests (only if no preemptions occurred) ───
    if no_preemptions and not paused:
        for request in self.waiting + self.skipped_waiting:
            # Check for blocked sub-states (FSM not compiled, remote KVs, etc.)
            if request.status in blocked_states:
                skipped_waiting.append(request)
                continue

            # Check prefix cache hits
            computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)

            # Compute tokens needed this step (may be chunked)
            num_new_tokens = request.num_tokens - num_cached_tokens
            num_new_tokens = min(num_new_tokens, token_budget)

            # Try to allocate
            new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, computed_blocks)

            if new_blocks is not None:
                # Promote: WAITING → RUNNING
                request.status = RUNNING
                self.running.append(request)
                scheduled_new_reqs.append(request)
                token_budget -= num_new_tokens
            else:
                break  # No more blocks — stop scheduling new requests (no preemption from waiting)

    # ─── Phase 3: Build output ───
    # Advance num_computed_tokens for all scheduled requests
    # Assemble SchedulerOutput with scheduled_new_reqs, scheduled_running_reqs,
    # finished_req_ids, preempted_req_ids, num_scheduled_tokens per request
    return SchedulerOutput(...)
```

**Key V1 design differences from V0:**
- **Unified loop** — no `_schedule_prefills()`, `_schedule_running()`, `_schedule_swapped()` split
- **No swap path** — preemption = free blocks + requeue (no CPU memory management)
- **Token-uniform** — prompt tokens and decode tokens budgeted identically via `{request_id: num_tokens}`
- **Chunked prefill built-in** — a new request may receive fewer tokens than its full prompt (chunked), handled naturally by the `num_computed_tokens` tracking
- **Preempted requests go to front** of waiting queue via `prepend_request()`, ensuring priority resumption

# 3c: Find the Continuous Batching Path (V1)

## Waiting → Running

In V1’s unified `schedule()`, Phase 2 handles WAITING → RUNNING promotion:

```
schedule() Phase 2:
  for request in self.waiting:
      computed_blocks = kv_cache_manager.get_computed_blocks(request)  # prefix cache hits
      new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, computed_blocks)
      if new_blocks is not None:
          request.status = RUNNING
          self.running.append(request)
```

No separate `_schedule_prefills()` — it’s just a WAITING request that gets blocks allocated. Whether it’s doing a fresh prefill or resuming after preemption is handled uniformly via `num_computed_tokens`.

## Releasing finished requests

`vllm/v1/core/sched/scheduler.py` — `update_from_output()`:

After each step, `update_from_output()` checks stop conditions for every scheduled request:
- EOS token sampled
- max_tokens reached
- Stop string matched
- Repetition limit hit

When finished:

```
update_from_output():
  for request in scheduled_requests:
      if stop_condition_met:
          request.status = FINISHED_STOPPED (or FINISHED_LENGTH_CAPPED, etc.)
          _free_request(request)
            → self.running.remove(request)
            → _free_blocks(request)
              → kv_cache_manager.free(request)  # blocks returned to BlockPool
          self.finished_req_ids.add(request.request_id)
```

**Continuous batching timing:** Blocks freed by finished requests in step N are immediately available for `schedule()` in step N+1. A new WAITING request can be promoted to RUNNING using those exact blocks. The freed blocks go to the tail of `free_block_queue` (most-recently-used position), and prefix-cached blocks with `ref_cnt == 0` can be evicted LRU-first if needed.

# AFTERNOON

## Step 5: Trace the KVCacheManager (~1.5 hrs)

### Summary

V1 replaced V0’s `BlockSpaceManager` with `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`). The key architectural change: V0 used a `BlockTable` per sequence with a `_free_ids` deque. V1 uses a `KVCacheCoordinator` → `BlockPool` architecture with a doubly-linked `FreeKVCacheBlockQueue`, `ref_cnt`-based lifetime management, and built-in prefix cache support via content hashing.

**Component hierarchy:**
```
KVCacheManager                          (vllm/v1/core/kv_cache_manager.py)
  └─ KVCacheCoordinator                 (vllm/v1/core/kv_cache_coordinator.py)
       ├─ BlockPool                     (vllm/v1/core/block_pool.py)
       │    ├─ blocks: list[KVCacheBlock]           — flat array, indexed by block_id
       │    ├─ free_block_queue: FreeKVCacheBlockQueue — doubly-linked list, LRU-ordered
       │    ├─ cached_block_hash_to_block: dict      — hash → block for prefix cache
       │    └─ null_block (block_id=0)               — placeholder for sliding window gaps
       └─ SingleTypeKVCacheManager(s)   — one per KV cache group (attention, Mamba, cross-attn)
            └─ req_to_blocks: dict[request_id → list[KVCacheBlock]]
```

**KVCacheBlock** (`vllm/v1/core/kv_cache_utils.py`):
```
KVCacheBlock:
    block_id: int
    ref_cnt: int = 0               # 0 = free/evictable, >0 = in use
    _block_hash: BlockHashWithGroupId | None  # set when block is full and cached
    prev_free_block / next_free_block         # doubly linked list pointers
```

### Key Methods

1. `allocate_slots(request, num_new_tokens, ...)` — central allocation

   - Calls `coordinator.get_num_blocks_to_allocate()` — how many new blocks needed
   - If `num_blocks_to_allocate > block_pool.get_num_free_blocks()`: returns `None` (triggers preemption in scheduler)
   - Registers prefix cache hits: `coordinator.allocate_new_computed_blocks()`
   - Allocates fresh blocks: `coordinator.allocate_new_blocks()` — pops from `free_block_queue`
   - Caches full blocks: `coordinator.cache_blocks()` — sets `block_hash`, inserts into hash map
   - Returns `KVCacheBlocks | None`

2. `get_computed_blocks(request)` — prefix cache lookup

   - Calls `coordinator.find_longest_cache_hit(request.block_hashes)`
   - Returns `(KVCacheBlocks, num_new_computed_tokens)`
   - `max_cache_hit_length = request.num_tokens - 1` (must recompute last token for logits)

3. `free(request)` — block release

   - Calls `coordinator.free(request.request_id)`
   - Decrements `ref_cnt` on each block
   - Blocks with `ref_cnt == 0` return to `free_block_queue` at **tail** (most-recently-used = last to be evicted)
   - Blocks with `ref_cnt > 0` stay allocated (shared by other requests via prefix cache)
   - Freed in reverse order so prefix-hot blocks (early in sequence) survive longer

### V0 → V1 Comparison

**V0 (BlockSpaceManager):**
- Allocation unit: SequenceGroup (multiple sequences)
- Block tracking: BlockTable per sequence
- Free pool: `_free_ids` deque
- Prefix caching: CachedBlockAllocator
- can_allocate result: OK / LATER / NEVER (three-way)
- Swap to CPU: Yes (SWAPPED state)
- Multi-type support: Not native

**V1 (KVCacheManager):**
- Allocation unit: Request (single sequence)
- Block tracking: `req_to_blocks` per request per KV cache group
- Free pool: `FreeKVCacheBlockQueue` (doubly-linked list, LRU-ordered)
- Prefix caching: BlockPool hash map + `ref_cnt`
- allocate_slots result: returns blocks or `None` (binary)
- Swap to CPU: No — blocks always on GPU, preempt = free
- Multi-type support: KVCacheCoordinator manages multiple groups (attention, Mamba, cross-attn)

## Step 6: Trace ModelRunner → Token Output (~1 hr)

**File:** `vllm/v1/worker/gpu/model_runner.py`

This is where the actual GPU forward pass happens. You don't need to go kernel-deep, but understand:

- How does `execute_model()` take the scheduled batch and run inference?
- How are prefill and decode handled differently? (Prefill processes all prompt tokens at once; decode processes one token per sequence)
- How do generated tokens flow back to the engine?

**What to document:** The handoff chain from scheduler output → GPU → new tokens.

ModelRunner Trace: Step-by-Step

The Key Files

- **Engine loop** — `vllm/v1/engine/core.py:390-414`
- **Executor** — `vllm/v1/executor/abstract.py:210-236`
- **Worker** — `vllm/v1/worker/gpu_worker.py:752-840`
- **GPUModelRunner** — `vllm/v1/worker/gpu/model_runner.py`
- **Input batch** — `vllm/v1/worker/gpu/input_batch.py`
- **Sampler output** — `vllm/v1/worker/gpu/sample/output.py`
- **Final output** — `vllm/v1/outputs.py:211-249`

### Step 0: How the Engine Calls In

```
vllm/v1/engine/core.py:390-414 — the step() method:

scheduler.schedule() → SchedulerOutput
↓
executor.execute_model(scheduler_output) # non-blocking
↓ via collective_rpc
Worker.execute_model(scheduler_output)
↓
GPUModelRunner.execute_model(scheduler_output)
↓ returns None (signals "now sample")
GPUModelRunner.sample_tokens(grammar_output)
↓
scheduler.update_from_output(scheduler_output, model_output)

Two-phase design: execute_model() runs the forward pass and stores hidden states, then sample_tokens() samples from them.
```

### Step 1: Request State Management (execute_model lines 862-868)

Before any GPU work, the model runner updates its internal request tracking:

finish_requests(scheduler_output) → remove done/preempted requests
free_states(scheduler_output) → free encoder cache entries
add_requests(scheduler_output) → initialize new requests (tokens, sampling params, KV blocks)
update_requests(scheduler_output) → allocate additional KV blocks for continuing requests
block_tables.apply_staged_writes() → commit block table changes to GPU

Early exit if nothing to do:
if scheduler_output.total_num_scheduled_tokens == 0:
return self.kv_connector.no_forward(scheduler_output)

### Step 2: CUDA Graph Dispatch (lines 874-894)

batch_desc = cudagraph_manager.dispatch(num_reqs, num_tokens, max_query_len)

Selects one of three execution modes:

- FULL — pre-recorded CUDA graphs for fixed batch sizes (lowest overhead)
- PIECEWISE — dynamic batch + graph replay (moderate)
- EAGER — pure PyTorch eager (most flexible, highest overhead)

If data parallelism is enabled, batch descriptors are synced across DP ranks here.

---

### Step 3: Prepare Model Inputs — prepare_inputs() (lines 605-742)

Transforms SchedulerOutput into GPU-friendly InputBatch:

1. Sort requests — decode-first ordering (fewer tokens first) for efficiency
2. Build index mapping — idx_mapping[batch_idx] → req_state_idx
3. Handle speculative decode tokens — expand logit indices if draft tokens present
4. Compute positions and seq_lens — via Triton kernel prepare_pos_seq_lens()
5. Prefill inputs — if any requests in prefill phase, copy next chunk of prompt tokens
6. Combine sampled + draft tokens — rewrite input_ids with last sampled token + drafts

Returns InputBatch:

```
InputBatch(
    req_ids, # [num_reqs]
    input_ids, # [num_tokens_after_padding] — GPU tensor
    positions, # [num_tokens_after_padding] — absolute position per token
    seq_lens, # [num_reqs] — current sequence length
    logits_indices, # [total_num_logits] — which positions produce logits
    query_start_loc, # [num_reqs+1] — cumsum of query lengths
    idx_mapping, # [num_reqs] — batch→state mapping
    num_scheduled_tokens, # [num_reqs] — tokens per request this iteration
...
)
```

---

### Step 4: Prepare Attention — prepare_attn() (lines 744-760)

Builds KV cache metadata for the attention layers:

block_tables = self.block_tables.gather_block_tables(idx_mapping, num_reqs_padded)

#### → tuple of tensors [num_reqs_padded, max_num_blocks], one per KV cache group

```
slot_mappings = self.block_tables.compute_slot_mappings(
    idx_mapping, query_start_loc, positions, num_tokens_padded
)
```

#### → [num_kv_cache_groups, num_tokens_padded]

#### Maps each token to its physical KV cache slot

This is where PagedAttention's block table is wired into the forward pass — each token knows exactly which physical memory slot to read/write its KV cache.

---

### Step 5: Build Attention Metadata (lines 929-943)

slot_mappings_by_layer = {layer: slot_mappings for ...} # per-layer mappings
attn_metadata = model_state.prepare_attn(batch_desc, block_tables, slot_mappings)

Creates the backend-specific attention metadata (FlashAttention, FlashInfer, etc.) that gets passed to every attention layer during the forward pass.

---

### Step 6: Multimodal Embeddings (lines 945-955)

If the model supports multimodal inputs (images, audio) and this is the first pipeline-parallel rank:

inputs_embeds = self.\_get_multimodal_embeddings(scheduler_output, input_batch)

Runs the vision encoder (or audio encoder) to produce inputs_embeds that replace raw token IDs at multimodal positions.

---

### Step 7: GPU Forward Pass (lines 957-1005)

Assemble inputs:

```
model_inputs = {
    "input_ids": input_batch.input_ids, # [num_tokens]
    "positions": input_batch.positions, # [num_tokens]
    "inputs_embeds": inputs_embeds, # optional multimodal
    \*\*model_state.prepare_inputs(input_batch), # LoRA, spec decode state, etc.
}
```

Execute:

#### FULL CUDA graph mode:

```
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
```

#### PIECEWISE or EAGER mode:

```
else:
    with set_forward_context(attn_metadata, ...):
    model_output = self.model(\*\*model_inputs)
```

```
model_output = hidden_states tensor of shape [num_tokens, hidden_size].
```

---

### Step 8: Store State for Sampling (lines 1007-1025)

```
self.execute_model_state = ExecuteModelState(
    input_batch=input_batch,
    attn_metadata=attn_metadata,
    hidden_states=hidden_states, # [num_tokens, hidden_size]
    aux_hidden_states=aux_hidden_states, # for EAGLE3 spec decode
    kv_connector_output=kv_connector_output,
    ...
)
```

For pipeline parallelism: non-last ranks return IntermediateTensors to send to next rank. Last rank returns None → triggers sample_tokens().

---

### Step 9: Token Sampling — sample_tokens() (lines 1028-1121)

1. Extract logits at sampling positions:
   sample_hidden_states = hidden_states[input_batch.logits_indices]

   #### → [total_num_logits, hidden_size]

2. Compute vocabulary logits:
   logits = model.compute_logits(sample_hidden_states)

   #### → [total_num_logits, vocab_size]

3. Run sampler (applies temperature, top-k, top-p, penalties):
   sampler_output, num_sampled, num_rejected = self.sample(
   hidden_states, input_batch, grammar_output
   )

# sampler_output.sampled_token_ids → [num_reqs, max_num_generated_tokens]

4. Prompt logprobs (if requested):
   prompt_logprobs_dict = self.prompt_logprobs_worker.compute(...)

5. Build final output:

```
   ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=[[token_ids...]], # per request
        logprobs=logprobs,
        prompt_logprobs_dict=prompt_logprobs_dict,
        kv_connector_output=kv_connector_output,
   )
```

### Step 10: Postprocess — postprocess() (lines 825-852)

Updates internal state after sampling:

num_computed_tokens += num_sampled # track generation progress
total_len += num_sampled # cumulative sequence length
all_token_ids.append(new_tokens) # full token history
last_sampled_tokens = new_tokens # for next iteration's input

### Prefill vs Decode — How They Differ

**Prefill:**
- Tokens per step: Many (chunked prompt)
- Input source: Prompt token IDs (copied via Triton)
- KV cache: Populated fresh
- Logit positions: Every token produces logits
- Detection: `num_computed_tokens < prompt_length`

**Decode:**
- Tokens per step: 1 per request (+ speculative drafts)
- Input source: Last sampled token
- KV cache: Appended incrementally
- Logit positions: Only last token
- Detection: `num_computed_tokens >= prompt_length`

**Both are handled in the same `execute_model()` call — the batch can contain a mix of prefill and decode requests simultaneously.**

---

### Complete Flow Diagram

```
  SchedulerOutput
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │  execute_model()                            │
  │                                             │
  │  1. finish/free/add/update request states   │
  │  2. cudagraph_manager.dispatch() → mode     │
  │  3. prepare_inputs() → InputBatch           │
  │     - sort requests (decode first)          │
  │     - compute positions, seq_lens           │
  │     - handle prefill tokens                 │
  │  4. prepare_attn() → block_tables, slots    │
  │  5. build attention metadata                │
  │  6. get multimodal embeddings (optional)    │
  │  7. model(**inputs) → hidden_states         │
  │     [num_tokens, hidden_size]               │
  │  8. store ExecuteModelState                 │
  └─────────────────┬───────────────────────────┘
                    │ returns None
                    ▼
  ┌─────────────────────────────────────────────┐
  │  sample_tokens()                            │
  │                                             │
  │  9. hidden_states[logits_indices]           │
  │     → compute_logits() → [N, vocab_size]   │
  │     → sampler (temp, top_k, top_p)         │
  │     → sampled_token_ids                    │
  │ 10. postprocess() — update tracking state   │
  │ 11. (optional) speculator → draft tokens    │
  └─────────────────┬───────────────────────────┘
                    │
                    ▼
            ModelRunnerOutput
            → scheduler.update_from_output()
```

## CUDA Graph

A GPU doesn't run one big operation — it runs hundreds of small operations (kernels) per forward pass: matrix multiplies, activations, normalizations, etc. Each kernel launch
has CPU overhead: the CPU tells the GPU "run this kernel with these args," waits for scheduling, then launches the next one.

A CUDA Graph records the entire sequence of kernel launches once, then replays it as a single unit. The GPU gets the whole execution plan upfront — no CPU round-trips between
kernels. It's called a "graph" because it's a DAG of operations: some kernels depend on others' outputs, some can run in parallel. CUDA captures this dependency structure.

The tradeoff: graphs require fixed tensor shapes. That's why vLLM pads batches to predetermined sizes and pre-records graphs for each size.

## Logit

A logit is the raw, unnormalized score the model assigns to each token in the vocabulary before any probability conversion. After the forward pass, you get a vector of
~32K-128K floats (one per vocab token). The highest logit means "the model thinks this token is most likely next."

To get actual probabilities you'd run softmax on the logits, but sampling algorithms (top-k, top-p, temperature) work directly on logits for numerical stability, so vLLM often
skips explicit softmax.

The name comes from statistics — "log-odds" → "logit."

Why "Forward Pass"

Neural networks have two directions:

- Forward pass — input flows forward through layers to produce output
- Backward pass — gradients flow backward through layers during training

During inference there's no backward pass, but the term "forward pass" persists because it describes the direction of computation through the network. "LLM run" is ambiguous —
does it mean one token? The whole generation? The forward pass is precisely one trip through all transformer layers, producing logits for the next token position(s). A full
generation requires many forward passes in a loop.

## Why Sampling Is Separate From the Forward Pass

The forward pass is pure deterministic math — matrix multiplications through transformer layers, producing logits. It's the same output every time for the same input.

Sampling is where randomness and policy enter. Given the logits, you decide:

- Temperature scaling (sharpen or flatten the distribution)
- Top-k/top-p filtering (restrict candidate tokens)
- Repetition penalties
- Grammar-constrained decoding (force valid JSON, etc.)
- Greedy vs random selection

Separating them is both conceptually clean and practically useful — vLLM exploits this split for pipeline parallelism (the last PP rank samples while other ranks already start
the next forward pass) and async scheduling (hidden states can be on GPU while sampling logic runs partially on CPU).

❯ so you[re saying we get around 32K-128K floats spanning the entire vocabulary out? and only ONE is selected? Does that mean LLMs cannot handle too-large vocabs?

## ASIDE: Crossing from Python into GPU Execution

There are three distinct mechanisms where Python crosses into GPU execution:

1. PyTorch Built-in Ops (implicit)

Most of the model is standard PyTorch — nn.Linear, nn.LayerNorm, matrix multiplies. These cross into CUDA invisibly:

### llama.py:227 — looks like Python, runs on GPU

qkv, \_ = self.qkv_proj(hidden_states) # nn.Linear → cuBLAS GEMM kernel

PyTorch's dispatcher routes torch.matmul, torch.softmax, tensor.argmax etc. to pre-compiled CUDA kernels under the hood. No explicit boundary in code — it's the tensor's
.device that determines where computation runs.

---

2. Custom CUDA Kernels (explicit C++/CUDA)

vLLM writes its own CUDA kernels for performance-critical operations. The boundary has three layers:

Python side — calls a registered torch op:

### attention.py:461

torch.ops.vllm.unified_attention_with_output(query, key, value, output)

C++ binding — csrc/torch_bindings.cpp:40:
ops.def("paged_attention_v1(...)");
ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

CUDA kernel launch — csrc/attention/paged_attention_v1.cu:
// The actual GPU launch — the triple-chevron syntax
paged_attention_v1_kernel<<<grid, block, shared_mem, stream>>>(
out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, ...
);

The <<<grid, block>>> syntax is the boundary. Everything before it is CPU setup. Everything inside that kernel function runs massively parallel on GPU cores.

Custom kernels exist for: paged attention, rotary embeddings, fused activations (SiLU), sampling penalties.

---

3. Triton Kernels (Python that compiles to GPU)

Triton lets you write GPU kernels in Python syntax. The @triton.jit decorator compiles them to GPU machine code at runtime:

# vllm/v1/attention/ops/triton_decode_attention.py:52

@triton.jit
def \_fwd_kernel_stage1(Q, K, V, ...):
pid = tl.program_id(0) # which GPU block am I?
offs = pid \* BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
q = tl.load(Q + offs) # load from GPU memory # ... compute attention ...
tl.store(Out + offs, result) # write to GPU memory

This looks like Python but it's not — Triton compiles it to PTX (NVIDIA's GPU assembly). Used for position encoding, slot mapping preparation, and some attention variants.

---

The Full Boundary Chain

LlamaModel.forward() ← Python, but tensors live on GPU
│
├─ self.qkv_proj(x) ← nn.Linear → PyTorch dispatches to cuBLAS
├─ self.rotary_emb(pos, q, k) ← Triton @jit kernel
├─ self.attn(q, k, v) ← torch.ops.vllm.\* → C++ → <<<CUDA kernel>>>
├─ self.mlp(x) ← nn.Linear + fused SiLU CUDA kernel
└─ ... repeat per layer ...
│
▼
hidden_states ← GPU tensor [num_tokens, hidden_size]
│
model.compute_logits(hidden_states) ← nn.Linear → cuBLAS (matmul against vocab)
│
▼
logits ← GPU tensor [num_tokens, vocab_size]
│
sampler(logits) ← torch.argmax / FlashInfer sampling kernel
│
▼
token_ids ← GPU tensor → copied to CPU (D2H transfer)

The key insight: Python never "runs" on the GPU. It orchestrates — setting up tensor shapes, choosing which kernels to launch, managing memory. The actual math always happens
in compiled code (cuBLAS, custom CUDA, or Triton-compiled PTX). PyTorch just makes the boundary invisible for standard ops.

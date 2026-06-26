# Day 8 Block 1: Architecture Diagram Answers (v1 validated)

1. What does the HTTP handler create?

Files: entrypoints/openai/chat_completion/api_router.py:47 → serving.py:279

The handler takes raw JSON and produces two things the engine needs:

1. Tokenized prompt — parse JSON → apply chat template via render_chat_request() (line 305) → tokenize
2. SamplingParams — request.to_sampling_params() (line 354) — temperature, top_k, top_p, max_tokens, stop strings

These get passed to engine_client.generate(prompt, sampling_params, request_id) at line 387. The HTTP handler's job is translation: OpenAI wire format → vLLM internal representation. It never touches the model.

---

2. Where does generate() enqueue? What queue?

File: v1/engine/async_llm.py:529

generate() is a consumer, not a producer. It:

1. Calls add_request() (line 563) → which calls engine_core.add_request_async() (line 417)
2. That sends the request over a ZMQ ROUTER socket to the EngineCore in a separate process
3. Creates a RequestOutputCollector (line 381) — per-request mailbox backed by asyncio.Event (output_processor.py:58)
4. Sits in a while-loop, yielding RequestOutput objects as they arrive in the mailbox

ZMQ is used (instead of multiprocessing.Queue or gRPC) because it's fast, handles framing, and avoids GIL contention between the async HTTP server and the GPU-heavy engine loop.

---

3. Can you trace schedule()? What is SchedulingBudget? Where does WAITING→RUNNING happen?

File: v1/core/sched/scheduler.py:322

SchedulingBudget

No class in v1. Just an integer:

token_budget = self.max_num_scheduled_tokens # line 341

Each admitted request eats tokens from the budget. When it hits 0, scheduling stops for this step.

schedule() flow

schedule(): 1. For each RUNNING request: schedule decode tokens, allocate new KV blocks if needed 2. For each WAITING/PREEMPTED request (while token_budget > 0): - Check len(running) < max_num_running_reqs (line 544) - Try kv_cache_manager.allocate_slots() → blocks or None (line 437) - If None: stop admitting - If success: request.status = RequestStatus.RUNNING (line 812) 3. Return SchedulerOutput

WAITING → RUNNING

Three gates, all must pass:

- token_budget > 0 — room in this step's batch
- len(self.running) < max_num_running_reqs — not too many concurrent requests
- allocate_slots() returned blocks, not None — enough KV cache memory

---

4. Block allocation: allocate → free with block math

Files: v1/core/kv_cache_manager.py, v1/core/block_pool.py

Allocate (try-and-fail, no pre-check)

scheduler calls kv_cache_manager.allocate_slots(request) # line 437
→ coordinator.get_num_blocks_to_allocate() # how many needed?
→ if needed > block_pool.get_num_free_blocks(): return None # line 348 — can't fit
→ block_pool.get_new_blocks(n) # line 320
→ free_block_queue.popleft() for each block
→ block.ref_cnt += 1 # line 341
→ return allocated blocks

Free

block_pool.free_blocks(blocks) # line 409
→ block.ref_cnt -= 1 # line 420
→ if ref_cnt == 0: return to free_block_queue # line 421

Block math

A 1000-token prompt with block_size=16 needs ceil(1000/16) = 63 blocks at prefill. Each 16 new decode tokens triggers one more allocation. 1000 prompt + 200 generated = ceil(1200/16) = 75 blocks total.

No watermark in v1 — relies on preemption to recover from memory pressure.

---

5. Prefill vs decode in execute_model()

File: v1/worker/gpu/model_runner.py:855

Both run through the same execute_model() call. The batch can mix prefill and decode requests simultaneously.

**Prefill:**
- Tokens per step: Many (chunked prompt)
- Input source: Prompt token IDs
- KV cache: Populated fresh
- Logit positions: Every token produces logits
- Detection: `num_computed_tokens < prompt_length`

**Decode:**
- Tokens per step: 1 per request
- Input source: Last sampled token
- KV cache: Appended incrementally
- Logit positions: Only last token
- Detection: `num_computed_tokens >= prompt_length`

prepare_inputs() (line 605) sorts requests decode-first (line 614: "Decode first, then prefill") by ascending token count. The GPU sees one flat batch — it doesn't know which tokens are prefill vs decode.

---

6.  State Machine (V1)

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

Key v1 difference from v0: no SWAPPED state

v1 eliminated swap-to-CPU entirely. When a request is preempted:

- KV cache is freed, not swapped (kv_cache_manager.free() — line 940)
- num_computed_tokens reset to 0 (line 943)
- Request goes back to the waiting queue (line 951)
- When rescheduled, it re-prefills from scratch

This is the recompute path only. Simpler design — no CPU block pool, no swap coordination, no tracking of CPU-side blocks. The tradeoff: preemption is more expensive (full re-prefill vs just copying KV cache back),
but the implementation is far simpler.

Preemption cost

Recompute cost: proportional to prompt length. A preempted request with a 4000-token prompt must re-run the entire prefill when it gets rescheduled. This is what makes the Day 9 mini-collapse experiment visible — at
high utilization, preemptions spike, and each one adds a full prefill's worth of latency.

# CPU swap was removed!

1. Official v1 User Guide (docs/usage/v1_guide.md)

Direct statement: "GPU <> CPU KV Cache Swapping: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping to handle request preemptions."

---

2. Metrics Design Doc (docs/design/metrics.md, lines 502-533)

The historical rationale:

- CPU swapping was originally designed for beam search, where multiple sequences shared KV cache blocks
- v1 removed SequenceGroup and moved beam search out of the core — eliminating the original use case
- Prefix caching replaced it as the default memory optimization: blocks can be evicted incrementally on demand, and the evicted portion gets recomputed when needed
- Key quote from the doc: with prefix caching enabled by default, "the preemption and recompute strategy should work better"

---

3. Recent Commit (March 7, 2026)

PR #36216: "[V0 Deprecation] Remove unused swap_space parameter" — removed all references to swap_space config, confirming the feature is fully gone.

---

Why recompute beat swap — the intuition

**Swap to CPU (V0):**
- CPU memory overhead: Need a CPU block pool sized to handle worst-case swaps
- CPU coordination: Track CPU↔GPU block mappings, async copies, swap-in scheduling
- Code complexity: BlockSpaceManager needed swap_in/swap_out/can_swap_in/can_swap_out
- With prefix caching: Redundant — cached prefix blocks survive eviction anyway
- V1 design goal: Contradicts "near-zero CPU overhead"

**Recompute (V1):**
- CPU memory overhead: Zero
- CPU coordination: None — just free and re-prefill
- Code complexity: `free()` + reset `num_computed_tokens = 0`
- With prefix caching: Only recompute the non-cached suffix
- V1 design goal: Aligned

The killer insight: prefix caching makes swap mostly redundant. If 80% of a preempted request's prompt is a common system prompt, those blocks are still cached. On re-prefill, only the unique 20% needs
recomputation. Swapping the whole KV cache to CPU and back is more work than just recomputing the small non-cached portion.

# Block 2

## Changes

4 log events, 3 locations:

- **[BLOCK_ALLOC]** — `kv_cache_manager.py`, after `allocate_new_blocks()` succeeds. Logs: req_id, blocks allocated, free/total.
- **[BLOCK_ALLOC_FAIL]** — `kv_cache_manager.py`, when needed > free. Logs: req_id, blocks needed, free/total.
- **[BLOCK_FREE]** — `kv_cache_manager.py`, after `coordinator.free()`. Logs: req_id, blocks freed, free/total.
- **[BLOCK_PREEMPT]** — `scheduler.py`, in `_preempt_request()` before freeing. Logs: req_id, computed_tokens_lost, preemption count, free/total.

Design choices:

- PREEMPT is logged before free() so computed_tokens_lost captures the work being thrown away and free count shows the pressure state at the moment of the decision
- ALLOC_FAIL is separate from ALLOC — lets you see when requests can't be admitted (Day 9 collapse signal)
- All events include free/total so you can track occupancy over time
- num_gpu_blocks - 1 accounts for the null block

To filter during testing:
2>&1 | grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT"

## Why is total blocks -1 (self.block_pool.num_gpu_blocks - 1)

⏺ The block pool reserves block 0 as a null block — a placeholder that represents "no block assigned."

From block_pool.py:175:
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True

It's popped from the free queue at initialization and never returned. It's used wherever the code needs a sentinel value in block tables (e.g., padding positions that don't store real KV cache data). Since it's
permanently occupied and never holds actual data, reporting it as part of usable capacity would be misleading.

So num_gpu_blocks - 1 = actual usable blocks. The get_usage() method on line 494 does the same thing:

total_gpu_blocks = self.num_gpu_blocks - 1 # Subtract 1 to account for null block

# Block 3

- I installed vllm 0.17.x which implement v1
- Then patched the print commands directly into the pip installed files

```
❯ ((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ curl -s http://localhost:8000/v1/chat/completions     -H "Content-Type: application/json"     -d
  '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}' | python3 -m json.tool
  {
      "id": "chatcmpl-b230ad9a85deb569",
      "object": "chat.completion",
      "created": 1773354177,
      "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "choices": [
          {
              "index": 0,
              "message": {
                  "role": "assistant",
                  "content": "As Business Process Management takes hold of a growing number of organizations, new technology solutions are emerging and available to help meet the
  demands of modern business processes. The following are some of the latest BPM technology trends:\n\n1. The",
                  "refusal": null,
                  "annotations": null,
                  "audio": null,
                  "function_call": null,
                  "tool_calls": [],
                  "reasoning": null
              },
              "logprobs": null,
              "finish_reason": "length",
              "stop_reason": null,
              "token_ids": null
          }
      ],
      "service_tier": null,
      "system_fingerprint": null,
      "usage": {
          "prompt_tokens": 17,
          "total_tokens": 67,
          "completion_tokens": 50,
          "prompt_tokens_details": null
      },
      "prompt_logprobs": null,
      "prompt_token_ids": null,
      "kv_transfer_params": null
  }
```

grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT" /tmp/vllm-server.log

```
((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT" /tmp/vllm-server.log
(EngineCore_DP0 pid=9357) INFO 03-12 22:22:57 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354177423 req=chatcmpl-b230ad9a85deb569-8900286a alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:22:57 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354177578 req=chatcmpl-b230ad9a85deb569-8900286a alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:22:57 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354177724 req=chatcmpl-b230ad9a85deb569-8900286a alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:22:57 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354177871 req=chatcmpl-b230ad9a85deb569-8900286a alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:22:57 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354177899 req=chatcmpl-b230ad9a85deb569-8900286a freed=5 free=30278/30278
```

The instrumentation is working. Let's read the data:

- Initial ALLOC: 2 blocks for prefill (17 prompt tokens, block_size=16 → ceil(17/16) = 2 blocks)
- 3 more ALLOCs: 1 block each as decode tokens fill up blocks (50 generated tokens need ~3 more blocks)
- FREE: 5 blocks freed at completion → 2 + 1 + 1 + 1 = 5 — totals reconcile
- Free count: 30278 → 30273 → back to 30278 — clean round trip

This matches the Day 8 syllabus predictions exactly. You're ready for Block 4 validation experiments (5 sequential, then 5 concurrent).

# Block 4

## Sequential Experiment

### Code

```
((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ for i in 1 2 3 4 5; do
echo "=== Request $i ==="
curl -s http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"
echo ""
done
```

### Stdout

```
=== Request 1 ===
{"id":"chatcmpl-9b0488459b278b31","object":"chat.completion","created":1773354770,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"The number 1 is a massive, existential force that defines the Universe itself. It began as a simple symbol, given to the ancient Greeks by the sun god Apollo. From that one number grew an unfathomable lineage of mathematical concepts, geometric forms, mathematical sequences, and mathematical endurance, all of which are intertwined and aligned with one another. It has been through countless iterations, changes, and adaptations to shape our material world into what it is","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
=== Request 2 ===
{"id":"chatcmpl-afeaa81f6df72f90","object":"chat.completion","created":1773354771,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"The number 2 is a prime number, meaning it is not divisible by any non-negative integer less than itself. This implies that it will always have exactly two divisors, or integers equal to it: 1 (if 2 is odd) and itself. Other than the need for proof, this property makes prime numbers temptingly shiny and irresistible.\n\nIt's easy to see why some cultures hold the belief that a prime number symbolizes greatness","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
=== Request 3 ===
{"id":"chatcmpl-9f9c2826ee126b1e","object":"chat.completion","created":1773354772,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"As we journey through life’s three seasons, we take a break from studying and integrating knowledge to embark on uplifting new paths while allowing for growth, patience, and divine timing. Every passing quarter is a new year in the calendar, and fresh starts inspire us to patiently navigate life’s interior promises. The unfolding of an individual’s life journey is a journey of discovery, and appreciating our surroundings, making new memories and relationships, learning","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
=== Request 4 ===
{"id":"chatcmpl-8df7ed8091f00cc9","object":"chat.completion","created":1773354773,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"4 is a common number that serves as a basis for a variety of mathematical computations and operations. One of the most commonly used numbers in everyday life, it has many practical applications from accounting to architecture to quantum computing. The primary mathematical operation carried out with 4 is addition, which is used to add two numbers together, even if they [],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokensare different from each other. It is the basis f     def remove_or many other operations like subtraction, multiplication, division, and exponents. As such,","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
=== Request 5 ===
{"id":"chatcmpl-bd53edc013227315","object":"chat.completion","created":1773354774,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"As the fifth leaf on your doorstep, written in bright yellow, greets your eye every morning, you're reminded of the astronomical marvel that is the体育花字节 calendar. With 52 weeks packed into each season, this special fifth leaf is a testament to the beauty and intricacy of the seasons, a reminder of the interconnectedness and cyclical nature of life. The fifth leaf marks a turning point between the grasses","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

### vllm Logs

((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT" /tmp/vllm-server.log

```
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770408 req=chatcmpl-9b0488459b278b31-b602974d alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770477 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770623 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770770 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770919 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771068 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30271/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771216 req=chatcmpl-9b0488459b278b31-b602974d alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354771338 req=chatcmpl-9b0488459b278b31-b602974d freed=8 free=30278/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771350 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771420 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771566 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771713 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:51 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354771862 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772011 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30271/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772160 req=chatcmpl-afeaa81f6df72f90-a1e6e493 alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354772281 req=chatcmpl-afeaa81f6df72f90-a1e6e493 freed=8 free=30278/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772293 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772363 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772510 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772656 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772805 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:52 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354772954 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30271/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773103 req=chatcmpl-9f9c2826ee126b1e-990d5e88 alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354773224 req=chatcmpl-9f9c2826ee126b1e-990d5e88 freed=8 free=30278/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773237 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773306 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773452 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773600 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773748 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:53 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354773897 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30271/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774046 req=chatcmpl-8df7ed8091f00cc9-a401637b alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354774168 req=chatcmpl-8df7ed8091f00cc9-a401637b freed=8 free=30278/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774180 req=chatcmpl-bd53edc013227315-afd64d95 alloc=2 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774250 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30275/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774397 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774544 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30273/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774692 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774841 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30271/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:54 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354774990 req=chatcmpl-bd53edc013227315-afd64d95 alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:32:55 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354775112 req=chatcmpl-bd53edc013227315-afd64d95 freed=8 free=30278/30278
```

## Concurrent

### Command

```
((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ for i in 1 2 3 4 5; do
curl -s http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}" &
done
wait
```

### Stdout

```
[1] 9993
[2] 9994
[3] 9995
[4] 9996
[5] 9997
{"id":"chatcmpl-b9bb1df86e69a2e5","object":"chat.completion","created":1773354960,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"Many people know the number 1 as a symbol of the universe, the start of a sentence, or even as a symbol for status. However, according to ancient Greek mythology, the number 1 was revered as the firstborn offspring of the gods. It represented the beginning of all things and creation, and was seen as a sacred number. This belief is evident in popular culture today, with musicians and artists using the number 1 to symbolize their contributions to society or their","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}[1]   Done                    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"
{"id":"chatcmpl-a02fed4434b4e333","object":"chat.completion","created":1773354960,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"There are only two numbers in the alphabet, both starting with the letter 'A': 1 and 2. 2 is known as the second smallest number in the English language, after one. It is often assumed that since two numerals cannot be equivalent because they start with the same letter, we can safely conclude that there can only be two numbers in a regular sequence. However, a thorough examination of the history and origins of this idea leads to the conclusion that we should not allow","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_t{"id":"chatcmpl-a8a8f4286ebeab4e","object":"chat.completion","created":1773354960,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"The number four is often associated with balance, continuity, and symmetry in the natural world. It is also a significant figure in mathematics and science, representing the four corners of a square, for example. Despite its ubiquity in various fields, it has an unexpected significance in another area—photography.\n\nThere is a photograph by English photographer Simon Cox called \"Omkosine,\" which features a set of columns that are arranged in the shape of a dove.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_to{"id":"chatcmpl-84cbc859d00919f0","object":"chat.completion","created":1773354960,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"The number 5 has long intrigue assorted cultures, nativity traditions, and modern technical calculations. It is a significant number with significant significance, as it can be plainly perceived in different circumstances of life, both organic and philosophic. In organic exercises, number five is crucial as it apportions the following qualities: crimson, spring, birth, IMHOTEP (indigenous magnetic hexagram), and the star of Bethle","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":nuken_ids":null,"kv_transfer_params":null}{"id":"chatcmpl-89c9e31738b0fd95","object":"chat.completion","created":1773354960,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"The number 3 is one of the most recognizable and commonly used integers in mathematics. It is the third largest positive integer, and its digits are traditionally shown as an arabic numeral (I, II, III) in the mobile-assembly hexadecimal number system. The reason behind 3's popularity is its embedded significance in logic, algebra, mathematics, and several sciences. 3 is the number of the three-body problem, for example. Moreover, it","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":25,"total_tokens":125,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params"ll}oken_ids":null,"kv_transfer_params":null}:null}[2]   Done                    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"
[3]   Done                    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"
[4]-  Done                    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"
[5]+  Done                    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the number $i\"}],\"max_tokens\":100}"

```

### vllm logs

((venv-vllm) ) ((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ grep -E "BLOCK_ALLOC|BLOCK_FREE|BLOCK_PREEMPT" /tmp/vllm-server.log | tail -60

```
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960507 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30276/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960515 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30274/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960515 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30272/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960515 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960516 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30268/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960584 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30267/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960594 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30266/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960594 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30265/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960594 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30264/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960594 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30263/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960748 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30262/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960758 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30261/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960758 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30260/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960758 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30259/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960758 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30258/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960912 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30257/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960922 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30256/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960922 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30255/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960922 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30254/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:00 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354960922 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30253/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961077 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30252/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961087 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30251/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961087 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30250/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961087 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30249/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961087 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30248/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961242 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30247/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961252 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30246/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961252 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30245/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961253 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30244/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961253 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30243/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961408 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 alloc=1 free=30242/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961418 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 alloc=1 free=30241/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961419 req=chatcmpl-84cbc859d00919f0-ad96b0e8 alloc=1 free=30240/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961419 req=chatcmpl-a8a8f4286ebeab4e-b245e870 alloc=1 free=30239/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354961419 req=chatcmpl-a02fed4434b4e333-99efec97 alloc=1 free=30238/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354961551 req=chatcmpl-b9bb1df86e69a2e5-808a80e8 freed=8 free=30246/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354961560 req=chatcmpl-89c9e31738b0fd95-bd19e1f9 freed=8 free=30254/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354961560 req=chatcmpl-84cbc859d00919f0-ad96b0e8 freed=8 free=30262/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354961560 req=chatcmpl-a8a8f4286ebeab4e-b245e870 freed=8 free=30270/30278
(EngineCore_DP0 pid=9357) INFO 03-12 22:36:01 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773354961560 req=chatcmpl-a02fed4434b4e333-99efec97 freed=8 free=30278/30278
```

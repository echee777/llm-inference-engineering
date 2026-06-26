# Day 10 Final Artifact

## Request flow and Architecture

Http Ingress. vllm/entrypoints/openai/api_server.py

- FastAPI server.
- Receives prompt

AsyncLLM engine.generate() vllm/v1/engine/async_llm.py

- Enqueues request to async process via zeroMQ
- Decouples engine from fastAPI layer for fault tolerance and scalability
- generate() prepares a RequestOutput which marshalls incoming result
  and yields it back to the client
- Prepare request
  - Use InputProcessor to tokenize.
  - Prepare SamplingParams.
- Call engine.generate(request)
- RequestOutput -> use output detokenizer to convert tokens to words

CoreEngine (server) vllm/v1/engine/core.py

- calls scheduler()

Scheduler vllm/v1/core/sched/scheduler.py

- Operates on iterations
- An iteration handles multiple concurrent batched requests
- A request can be in decode or chunked prefill. Scheduler is agnostic.
- For each request, scheduler
  - delegates memory reservation to KVCacheManager, supplying it required new blocks
  - delegates forward pass execution to worker via ZeroMQ

KVCacheManager vllm/v1/core/kv_cache_manager.py

- memory reservation is attempted based on required blocks
- on failure return None

Worker

- Delegates to GPUModelRunner vllm/v1/worker/gpu_model_runner.py
- model(inputs)
- prefill: Computes KV cache
- decode:
  - Calculates next token using KVCache and MLP
  - Produces logits (width = vocabulary)
  - Runs sampling on the logits to choose the token
  - Streams token back to scheduler
    - scheduler -> engine -> async_llm

## KVCache

- 1 block = 16 token KV cache
- 1 token KV cache size = 2 x dtype_bytes x num_layers x num_heads_per_layer x head_dimension
- free blocks are maintained in a double linked list data structure (FreeKVCacheBlockQueue)
- Each request has a page-table like mapping of logical blocks to physical addresses. This is called PagedAttention.
  - this mapping is maintained in KVCacheManager's `req_to_blocks` dict -maps request_id to a list of physical block IDs
  - the block table is passed to the attention kernel so it can gather the right KV data from non-contiguous physical memory
- For Qwen2.5-3B on T4: 2 x 2bytes x 36 layers x 2 kv_heads x 128 dim = 36KB per token, x16 = ~576KB per block

## Scheduler decision logic

- Iteration loop / request states
  - States: WAITING, RUNNING, FINISHED.
  - Phase 1 (service RUNNING):
    - Existing RUNNING requests get priority.
    - First check if preemption is needed to allow all RUNNING to advance (i.e. cannot allocate KV block for next decode token).
    - If preempted, discard the lowest priority RUNNING's KV cache and move it to head of WAITING queue.
    - Keep doing this for as long as we cannot allocate() new KV block for ALL running requests to advance.
  - Phase 2 (promote WAITING):
    - Only if NO preemption occurred.
    - Try to allocate large enough KV for chunked prefill. If fail, do not schedule anything. If succeed, promote head of WAITING queue to RUNNING.
  - Run the forward pass on all RUNNING.
  - At the end of the iteration, free the KV cache for all completed sequences.
    - completion can occur due to EOS token, max tokens reached, client abort, etc. One of 6 or 7 reasons.
    - NB: blocks freed here are available _next_ iteration, not this one. Preemption-freed blocks are available same iteration.

- State machine
  - WAITING --> RUNNING
    - if threshold criteria (e.g. max concurrent requests and max tokens per batch) are met
    - and if no preemption occurred
    - then attempt to promote the head of the WAITING queue
  - RUNNING --> WAITING
    - the lowest priority running request is preempted
      thus freeing its KV memory
  - RUNNING --> FINISHED
    - eos, max tokens outputted, client cancellation, various other reasons

## Preemption cost model

- There is no swap in v1
- cpu ram mgmt was becoming bottleneck. PCIe at ~9GB/s vs HBM at ~217GB/s, so recompute from HBM is faster than swapping over PCIe for most prompt lengths.
- KV prefix cache ameliorates prefill recompute. If the prefix blocks haven't been evicted from the hash map, the re-prefill can skip those tokens.
- Preemption is memory-neutral but regressed in terms of compute advancement as it throws away completed work.
- Observed: request 9474796f was preempted 6 times, generated 2,156 tokens of decode work that got thrown away. More than the ~1,900 tokens needed to finish. It timed out.

## Continous batching

- The Orca insight is that naive batching wastes many GPU threads' worth compute capacity when shorter requests complete and the system has to wait for the longest request to complete.
- Therefore, make a batch step a single decode token compute or chunked prefill. The batch latency is constrained by the slowest operation across the batch. However, requests that complete allow other new requests to be admitted into the continuous batch work.
- Day 9 Block 3 confirmed this: 10 staggered requests, 344 scheduler iterations over 19.6s. Batch composition changed every single iteration. Requests joined on arrival, left on completion independently.

## What surprised me

- Had to lower gpu-memory-utilization to 0.45 to trigger preemption. At 0.90 and 0.55, the block pool was big enough that --max-num-seqs 32 throttled admission before memory ran out. I was trying to stress memory but the concurrency gate kept firing first. Had to shrink the pool to force the memory gate.

- The TTFT cliff was binary not gradual. At c=12 max TTFT was 671ms. At c=14 it jumped to 8735ms. Thats a 13x spike in two concurrency steps. Not a smooth curve -a step function. The formula floor(total_blocks / blocks_per_request) predicted c=11 as the safe max, and the cliff showed up right there. The implication is that capacity planning can't rely on "add one more request and see if latency is still ok" -by the time you see degradation you've already fallen off the cliff. You need to compute the hard limit from block math upfront: total_blocks / blocks_per_request. Below that number TTFT is flat. Above it TTFT explodes. There's no middle ground to autoscale through gracefully.

- Free block count oscillated between 0 and ~51 during the thrashing phase. Exactly one victim's worth of blocks freed then immediately consumed by the next prefill. The system was stuck in this 51-block oscillation for ~78 seconds. Why ~51? Probably the typical decode progress of the victim at preemption time. With 1,034 blocks and ~32 running requests, average is ~32 blocks each, but the victim (lowest priority, been running longest) would have accumulated more. Those ~51 blocks get freed, the triggering request grabs 1, a WAITING request gets promoted, then the pool drains back to 0 as running requests each grab 1 block per decode step. Would need to cross-check against the actual BLOCK_FREE logs per victim to confirm the exact number.

- 11 out of 80 requests timed out at 120s. Not really surprising -vLLM should be coded to not crash under pressure, and it didn't. Good to confirm the system prioritizes robustness over latency guarantees. No OOM, no segfault, just degraded performance. The interesting part is that from the outside (HTTP status codes) every request looked fine -you'd only catch the problem by monitoring TTFT or p99 latency.

# Quantization

## AWQ (Activation weight quantization)

Notes:

- Addresses the problem of weights taking a large chunk of GPU ram
- E.g. 70B weight model = 140GB; Quantization of weights can reduce by 3/4 to ~35GB
- However, quantization loses precision.
- AWQ's main insight is that not all weights matter. For a representative training set
  only about 1% of weights have activations that are large enough that quantization will
  reduce performance (e.g. hallucinations etc). The remaining 99% can be aggressively quantized e.g. from FP16 --> INT4 without noticeable penalty.
- IOW AWQ seeks to reduce activation-aware error in layer outputs Yerr = minimize |X(W - Wq)|

Note: AWQ primarily benefits prefill not decode; for decode KV cache memory access predominates weight access.

## GPTQ

Key idea: Quantize activation-aware error in layer outputs using second-order information (Hessian approximation) to minimize layer output error. Requires a small calibration dataset but no retraining. IOW the quantization is based on activation x weight and requires more compute.

In contrast, AWQ identifies 1% of high-activations, then pre/post adjusts only those weights by multiplying each weight by a scaleup factor before quantization
and then dividing them by the same factor after quantization. Factor differs per weight.

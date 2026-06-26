# Final Artifact

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

Scheduler vllm1/v1/core/sched/scheduler.py

- Operates on iterations
- An iteration handles multiple concurrent batched requests
- A request can be in decode or chunked prefill. Scheduler is agnostic.
- For each request, scheduler
  - delegates memory reservation to KVCacheManager, supplying it required new blocks
  - delegates forward pass execution to worker via ZeroMQ

KVCacheManager vllm1/v1/core/kv_cache_manager.py

- memory reservation is attempted based on required blocks
- on failure return None

Worker

- Delegates to GPUModelRunner vllm1/v1/worker/gpu_model_runner.py
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
- free blocks are maintained in a double linked list data structure
- Each request has a page-table like mapping of logical blocks to physical addresses. This is called PagedAttention.
  - this mapping is maintained in <TODO>

## Scheduler decision logic

- Iteration loop / request states
  - States: WAITING, RUNNING, FINISHED.
  - Phase 1 (decode):
    - Prioritize decode
    - First check if preemption is needed to allow all RUNNING to advance (i.e. cannot allocate KV block for decode).
    - If preempted, discard the lowest priority RUNNING's KV cache and move it to head of WAITING queue.
    - Keep doing this for as long as we cannot allocate() new KV block for ALL running requests to advance.
  - Phase 2 (prefill):
    - Only if NO preemption occurred.
    - Try to allocate large enough KV for chunked prefill. If fail, do not schedule anything. If succeed, promote head of WAITING queue to RUNNING.
  - Run the forward pass on all RUNNING.
  - At the end of the iteration, free the KV cache for all completed sequences.
    - completion can occur due to EOS token, max tokens reached, client abort, etc. One of 6 or 7 reasons.

- State machine
  - WAITING --> RUNNING
    - if threshold criteria (e.g. max concurrent requests and max tokens per batch) are met
    - and if no preemption occurred
    - then attempt to promote the head of the WAITING queue
  - RUNNING --> WAITING
    - the lowest priority running request is preempted
      thus feeing it's KV memory
  - RUNNING --> FINISHED
    - eos, max tokens outputed, client cancellation, various other reasons

## Preemption code model

- There is no swap in v1
- cpu ram mgmt was becoming bottleneck.
- KV prefix cache ameliorates prefill recompute.
- Preemption is memory-neutral but regressed in terms of compute advancement as it throws away completed work.

## Continous batching

- The Orca insight is that naive batching wastes many GPU threads' worth compute capacity when shorter requests complete and the system has to wait for the longest request to complete.
- Therefore, make a batch step a single decode token compute or chunked prefill. The batch latency is constrained by the slowest operation across the batch. However, requests that complete allow other new requests to be admitted into the continuous batch work.

## What surprised me

<TODO: I can't remember>

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

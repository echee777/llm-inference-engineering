# AI Inference Platform Residency — Phase A (v3 Definitive)

## Inference Foundations: Day-by-Day (Weeks 1–4)

**Schedule:** 4 weeks × 5 days × 8 hours = 160 hours\
**Daily Structure:** Morning block (4 hrs) + Afternoon block (4 hrs)\
**Week 1 Time Split:** 20% reading / 55% hands-on / 15% profiling / 10% writing\
**Weeks 2–4 Time Split:** 10% reading / 60% hands-on / 20% profiling+debugging / 10% writing

### Rules

1. Never read for more than 90 minutes without switching to hands-on work
2. Every experiment must produce a number, a graph, or a table
3. Every written memo gets adversarial review before moving on
4. If stuck for more than 2 hours, write down exactly what you're stuck on and move to the next task — return later
5. GPU time is expensive. Plan experiments before running them. Know what you're measuring before you start the instance.

### Cloud Cost Estimate (Phase A)

- Instance: 1× A100-40G spot (~$1.50–2.50/hr) or A10G fallback (~$0.75–1.25/hr)
- ~50% of daily hours need GPU = ~80 GPU-hours over 4 weeks
- Estimated cost: $120–200 (spot A100) or $60–100 (spot A10G)

---

# ═══════════════════════════════════
# WEEK 1: GPU Architecture & Memory Model
# ═══════════════════════════════════

**Goal:** Build the hardware mental model. After this week, you think in terms of HBM bandwidth, KV cache blocks, occupancy, memory fragmentation, and memory-bound vs. compute-bound operations.

**This is the heaviest reading week (~20%). Every week after this drops to ≤10%.**

---

## Day 1 (Mon) — GPU Hardware Mental Model

### Morning (4 hrs) — Focused Reading + Orientation

- **Read (1.5 hrs):** NVIDIA CUDA C Programming Guide — **targeted sections only.** You are not writing kernels. You need the hardware mental model.
  - **Read these:** Memory hierarchy (global/shared/local/registers), memory coalescing, warp execution model (32 threads in lockstep), occupancy concept (warps per SM)
  - **Skip these:** Cooperative groups, advanced memory scopes, thread block configuration details, CUDA API specifics
  - Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Read (1 hr):** One GPU architecture overview. Recommended:
  - "Making Deep Learning Go Brrrr From First Principles" by Horace He
  - OR the NVIDIA A100/H100 whitepaper for your target GPU
  - Focus on: SM count, HBM capacity, HBM bandwidth, compute TFLOPS, Tensor Core specs
- **Exercise (30 min):** Fill out this table from documentation:

  | Spec | Value |
  |---|---|
  | HBM capacity (GB) | |
  | HBM bandwidth (GB/s, theoretical) | |
  | FP16 TFLOPS (Tensor Core) | |
  | INT8 TOPS (Tensor Core) | |
  | SM count | |
  | Max warps per SM | |
  | Max threads per SM | |
  | Register file size per SM (KB) | |
  | Shared memory per SM (KB) | |
  | L2 cache size (MB) | |
  | PCIe bandwidth (GB/s) | |
  | NVLink bandwidth (GB/s, if applicable) | |

- **Warp utilization microbenchmark (1 hr):**
  - Run a simple matrix multiply at varying batch sizes to observe how small batches underutilize the GPU:
    ```python
    import torch, time
    # Simulate decode-like workload: tiny batch matmul
    W = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    for batch in [1, 4, 16, 64, 256, 1024]:
        x = torch.randn(batch, 4096, device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            y = x @ W.T
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = (2 * batch * 4096 * 4096 * 200) / (elapsed * 1e12)
        print(f"batch={batch:4d}: {tflops:.1f} TFLOPS ({tflops/312*100:.0f}% of A100 peak)")
    ```
  - **Key observation:** batch=1 (like a single decode step) achieves a tiny fraction of peak TFLOPS. batch=256+ approaches peak. This is why decode is memory-bound: the batch size is too small to saturate compute.
  - Record the results. This is empirical proof of the roofline concept before you even learn roofline theory.

### Afternoon (4 hrs) — First Hands-On

- **Spin up your GPU instance.** Get comfortable with the environment.
- **Run `nvidia-smi` (30 min):** Document every field. Critical distinction:
  - "GPU Util" = % of time at least one kernel is running. Not how "busy" the GPU is. A single lightweight kernel running continuously shows 100% utilization. **This is why GPU utilization is misleading.**
  - "Memory Usage" = allocated HBM. More useful, but still doesn't tell you about bandwidth saturation.
- **Run `nvidia-smi dmon` (30 min):** Watch metrics change in real-time.
- **Install and run `nvtop` (30 min):** Visual GPU monitoring.
- **HBM bandwidth benchmark (2.5 hrs):**
  ```python
  import torch, time
  for size_mb in [1, 10, 100, 1000]:
      n = size_mb * 1024 * 1024 // 4  # float32 elements
      a = torch.randn(n, device='cuda')
      b = torch.empty_like(a)
      torch.cuda.synchronize()
      start = time.perf_counter()
      for _ in range(100):
          b.copy_(a)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - start
      bw = (2 * size_mb * 100) / (elapsed * 1024)  # GB/s (read + write)
      print(f"{size_mb}MB: {bw:.1f} GB/s")
  ```
  - Also measure CPU↔GPU transfer bandwidth (PCIe)
  - Record results. You should see ~60–80% of theoretical HBM peak for large on-device transfers.

**End-of-day output:** GPU spec table + bandwidth benchmark results + warp utilization table (batch size vs. TFLOPS achieved).

---

## Day 2 (Tue) — Memory Bandwidth, Arithmetic Intensity, and Occupancy

### Morning (4 hrs) — Arithmetic Intensity + Roofline

- **Read (1 hr):** Roofline model concept.
  - Key insight: every operation is either **memory-bandwidth-bound** or **compute-bound**, determined by arithmetic intensity (FLOPs per byte of memory accessed).
  - LLM decode has very low arithmetic intensity → memory-bound.
  - Good source: "Demystifying the Roofline Model" or original Williams et al. paper (skim).
- **Hands-on (3 hrs):** Compute arithmetic intensity of a single transformer decode step:
  - For a 7B parameter model at FP16:
    - Model weights: ~14 GB
    - Each decode step: read all weights once, produce one token
    - FLOPs per decode step: ~2 × 7B = 14 GFLOPs
    - Bytes read: ~14 GB
    - Arithmetic intensity: 14 GFLOPs / 14 GB ≈ **1 FLOP/byte**
  - Compare to your GPU's ridge point:
    - A100: ~2 TB/s bandwidth, ~312 TFLOPS FP16 → ridge point ≈ 156 FLOPs/byte
    - Your decode step at 1 FLOP/byte is **deep in memory-bandwidth territory**
  - Now do the same for **prefill** (processing a 2048-token prompt):
    - Prefill processes many tokens in parallel → batch matmul → higher arithmetic intensity
    - Prefill is closer to compute-bound (especially for long prompts)
  - **Connect to your Day 1 microbenchmark:** Your batch=1 TFLOPS was low because arithmetic intensity was low. Your batch=256 TFLOPS was high because arithmetic intensity was high. Same principle.
  - **Write this out clearly.** The prefill/decode asymmetry is the single most important insight in inference engineering.

### Afternoon (4 hrs) — Profiling: torch.profiler + Nsight

- **Profile with torch.profiler (1.5 hrs):**
  ```python
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA],
      with_stack=True
  ) as prof:
      model.generate(input_ids, max_new_tokens=1)  # single prefill
  ```
  - Analyze: % time in attention vs. MLP vs. other?
  - Run again with `max_new_tokens=50`. Compare first token (prefill) vs. subsequent (decode).
  - They should look very different — prefill is compute-heavy, decode is memory-read-heavy.

- **Profile with Nsight Systems or Nsight Compute (2.5 hrs):**
  ```bash
  # Nsight Systems (timeline profiling)
  nsys profile -o inference_trace python your_inference_script.py

  # Nsight Compute for a specific kernel
  ncu --target-processes all --set full -o kernel_profile python your_script.py
  ```
  - For at least one kernel (an attention or MLP kernel), capture:
    - **Achieved occupancy %** — what fraction of the SM's max warps are active?
    - **SM active %** — what fraction of SMs have at least one active warp? (Occupancy ≠ SM utilization: you can have high occupancy on few SMs, or low occupancy across all SMs.)
    - **Memory throughput** — how close to peak HBM bandwidth?
    - **Compute throughput** — how close to peak TFLOPS?
    - **Stall reasons** — is the kernel stalling on memory? On compute? On synchronization?
  - Fill out:

    | Phase | Achieved Occupancy | SM Active % | Memory Throughput (% peak) | Compute Throughput (% peak) | Primary Stall Reason |
    |---|---|---|---|---|---|
    | Prefill (2K tokens) | | | | | |
    | Decode (single token) | | | | | |

  - **Why this matters for interviews:** When someone asks "GPU utilization is only 40%, why aren't we using the GPU more efficiently?" you now answer: "Utilization is 40% because we're memory-bandwidth-stalled at Y% of peak HBM bandwidth. Achieved occupancy is X%, SM active is Z%. We can't do more compute because we're waiting for weight data from HBM. This is fundamental to decode — increasing utilization requires fundamentally different approaches like batching more requests together."

**End-of-day output:** Arithmetic intensity calculations (prefill + decode) + torch.profiler traces + Nsight profiling table with occupancy, SM active %, throughput, and stall reasons. One paragraph explaining why decode is memory-bound, with your profiler data as evidence.

---

## Day 3 (Wed) — KV Cache: The Binding Constraint

### Morning (4 hrs) — KV Cache Math

- **Read (1 hr):** How the KV cache works in transformer inference.
  - Each layer stores key and value tensors for every token in the sequence
  - As sequence length grows, KV cache grows linearly
  - Source: vLLM paper Section 2 (background) or any good KV cache explainer
- **Calculate (2 hrs):** For your model on your GPU:
  - KV cache size per token per layer = 2 × num_kv_heads × head_dim × dtype_size
  - For Llama 2 7B (32 layers, 32 KV heads, head_dim 128, FP16):
    - Per token per layer: 2 × 32 × 128 × 2 bytes = 16,384 bytes
    - Per token all layers: 16,384 × 32 = 524,288 bytes ≈ **0.5 MB per token**
  - Capacity planning:

    | Sequence Length | KV Cache Per Request | Max Concurrent @ 26GB Free |
    |---|---|---|
    | 512 | 256 MB | ~101 |
    | 2,048 | 1 GB | ~26 |
    | 4,096 | 2 GB | ~13 |
    | 8,192 | 4 GB | ~6 |
    | 16,384 | 8 GB | ~3 |

  - **This is why KV cache is the binding constraint.** At 8K sequence length, you can only serve 6 concurrent requests on a 40GB GPU. Not because of compute — because of memory.

- **Build a KV cache calculator (1 hr):**
  ```python
  def max_concurrent_requests(
      gpu_memory_gb, model_size_gb, num_layers, num_kv_heads,
      head_dim, seq_length, dtype_bytes=2, overhead_pct=0.10
  ):
      available = (gpu_memory_gb - model_size_gb) * (1 - overhead_pct) * 1e9
      kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
      kv_per_request = kv_per_token * seq_length
      return int(available // kv_per_request)
  ```

### Afternoon (4 hrs) — Memory Fragmentation + KV Precision Impact

- **Memory fragmentation simulation (1.5 hrs):**
  This experiment builds intuition for *why PagedAttention exists.* You'll read the PagedAttention paper in Week 2 — this gives you the problem before the solution.
  - Write a simple Python simulation of two KV cache allocation strategies:
    - **Strategy A: Contiguous allocation** — each request gets a contiguous block of memory for its max possible sequence length (allocated upfront)
    - **Strategy B: Paged allocation** — each request gets small fixed-size blocks (e.g., 16 tokens per block), allocated on demand as the sequence grows
  - Simulate this scenario:
    1. Start with 26 GB free (your KV cache budget)
    2. Allocate 10 requests with varying sequence lengths: 500, 1200, 800, 3000, 400, 2500, 600, 1800, 1000, 2000
    3. Free requests 2, 4, 6, 8 (leaving holes in contiguous allocation)
    4. Try to allocate 3 new requests of length 2500, 3500, 4000
  - **Under contiguous allocation:** even though total free memory is sufficient, some new requests can't fit because free space is fragmented into non-contiguous chunks
  - **Under paged allocation:** all new requests fit because blocks don't need to be contiguous
  - Calculate: **fragmentation waste %** = (total free memory - largest allocatable request) / total free memory
  - **Write one paragraph:** "Without paging, X% of my KV cache memory is wasted due to fragmentation under realistic mixed-length workloads. This is why PagedAttention matters — it eliminates external fragmentation by decoupling logical sequence position from physical memory location."
  - You don't need GPU time for this — run it on your laptop in pure Python.

- **KV cache precision impact (30 min):**
  - Re-run your KV cache calculator at FP16, INT8, and FP8 KV cache precision
  - At INT8 KV cache: bytes per token halves → max concurrent requests roughly doubles
  - Record the numbers. You'll connect this to quantization experiments in Week 3.

- **Hardware multi-tenancy overview (1 hr):**
  - Read about NVIDIA MPS, MIG, and time-slicing (high-level overview, not hands-on):
    - MPS: shares one GPU across processes
    - MIG: hard-partitions GPU (A100/H100 only)
    - Time-slicing: round-robin access
  - Document the tradeoffs in a few sentences each. MIG hands-on is not worth the setup time — understanding when you'd use each option is sufficient.

- **Write (1 hr):** 1-page memo: **"GPU Memory Budget for [Your Model] on [Your GPU]"**
  - KV cache calculator output (table for multiple sequence lengths and precisions)
  - Bandwidth benchmark numbers from Day 1
  - Occupancy/throughput numbers from Day 2
  - Fragmentation simulation results and why paging matters
  - Key insight: "My GPU can serve N concurrent requests at sequence length L, limited by HBM capacity for KV cache, not by compute."

**End-of-day output:** KV cache calculator script + fragmentation simulation script + GPU Memory Budget memo

---

## Day 4 (Thu) — Interconnect Topology + Week 1 Synthesis

### Morning (4 hrs) — PCIe, NVLink, and TP Cost Preview

- **Read (1 hr):** Interconnect architecture:
  - PCIe Gen4: ~32 GB/s bidirectional
  - PCIe Gen5: ~64 GB/s bidirectional
  - NVLink (A100): 600 GB/s total
  - NVLink (H100): 900 GB/s total
  - NVSwitch: full-bisection bandwidth between all GPUs
  - **Key insight: PCIe is 10–30× slower than NVLink.**
- **Hands-on (1.5 hrs):**
  - Run `nvidia-smi topo -m` — document your topology
  - If multi-GPU: measure GPU-to-GPU transfer bandwidth with PyTorch
  - If single GPU: measure CPU↔GPU (PCIe) bandwidth
- **TP communication cost preview (1.5 hrs):**
  - This is a mental exercise for now — hands-on TP comes in Phase B. But you need the cost model in your head early.
  - For a 7B model with TP=2:
    - Each transformer layer requires an all-reduce after the attention and MLP blocks
    - All-reduce message size ≈ 2 × hidden_dim × batch_size × dtype_bytes
    - For hidden_dim=4096, batch=1, FP16: ~16 KB per all-reduce
    - 2 all-reduces per layer × 32 layers = 64 all-reduces per decode step
    - Over PCIe (~32 GB/s): latency per all-reduce ≈ 16KB / 32 GB/s + overhead ≈ ~5–10 μs each → ~320–640 μs total per step
    - Over NVLink (~600 GB/s): ~0.5 μs each → ~32 μs total per step
  - **Key question for interviews:** "When does tensor parallel over PCIe make things *worse*?"
    - Answer: when communication overhead per step exceeds the time saved by splitting compute. For small models on PCIe, TP=2 can be slower than single-GPU because the communication tax exceeds the compute savings.
  - Write out this calculation. You'll validate it with real NCCL measurements in Phase B.

### Afternoon (4 hrs) — Week 1 Deliverable

- **Write (3 hrs):** **GPU Architecture & Memory Budget Document** (Week 1 deliverable):
  - **Section 1: Memory Hierarchy** — GPU specs with measured bandwidths (Day 1)
  - **Section 2: Arithmetic Intensity** — prefill vs. decode roofline analysis with your numbers. Connection to warp utilization microbenchmark. (Day 1–2)
  - **Section 3: Occupancy & Throughput** — Nsight profiling results. Achieved occupancy, SM active %, memory throughput, compute throughput, stall reasons for both prefill and decode. (Day 2)
  - **Section 4: KV Cache Budget** — calculator output for multiple sequence lengths and precisions (Day 3)
  - **Section 5: Memory Fragmentation** — simulation results showing why paged allocation matters (Day 3)
  - **Section 6: Interconnect & TP Cost Preview** — topology, bandwidth, TP communication cost calculation (Day 4)
  - **Section 7: Key Takeaways:**
    1. Inference decode is memory-bandwidth-bound (proved with roofline + warp microbench + Nsight)
    2. KV cache is the binding constraint for concurrency (proved with calculator)
    3. Memory fragmentation wastes capacity under mixed workloads (proved with simulation)
    4. GPU utilization % is misleading — memory throughput and occupancy tell the real story (proved with profiler)
    5. TP over PCIe has a quantifiable communication tax that can make things worse (proved with calculation)
- **Review (1 hr):** Does every claim have a number? If not, go get the number.

**End-of-day output:** Complete GPU Architecture & Memory Budget Document (v1)

---

## Day 5 (Fri) — Buffer + Adversarial Review

### Morning (4 hrs) — Catch Up + Polish

- Finish anything incomplete from Days 1–4
- Polish the GPU Architecture document
- Re-run any experiments where results were unclear or noisy
- Ensure your KV cache calculator is clean, commented, reusable

### Afternoon (4 hrs) — Self-Review + Week 2 Prep

- **Adversarial self-review (2 hrs):** Answer from memory. If you can't, go back.
  1. What is the theoretical HBM bandwidth of your GPU? What did you actually achieve?
  2. Why does batch=1 matmul achieve only X% of peak TFLOPS? (Reference your warp microbenchmark.)
  3. What is the arithmetic intensity of a decode step for a 7B FP16 model? Where does it sit on the roofline?
  4. What was the achieved occupancy during decode? SM active %? Primary stall reason?
  5. How many concurrent 4K-token requests can your GPU serve? Show the math.
  6. How does INT8 KV cache change that number?
  7. What is memory fragmentation and why does it waste KV cache capacity?
  8. Why is "GPU utilization" (nvidia-smi) misleading? What metrics are more informative?
  9. What's the communication overhead of TP=2 over PCIe for your model? When does TP make things worse?
- **Week 2 prep (2 hrs):** Skim the vLLM paper (https://arxiv.org/abs/2309.06180). Read abstract and Sections 1–3. Don't go deep — build a mental map. **Note:** your fragmentation simulation from Day 3 is the *problem* that Section 3 (PagedAttention) solves. Read it with that context.

**End-of-day output:** Finalized Week 1 deliverable. Readiness for vLLM internals.

---

# ═══════════════════════════════════
# WEEK 2: Model Serving Internals (vLLM)
# ═══════════════════════════════════

**Goal:** Deploy vLLM, trace the request lifecycle through actual source code, produce one instrumentation patch, and experience your first system collapse. After this week, you can explain what happens at every step from HTTP request to generated token — citing specific files and functions — and you've *seen* what happens when it breaks.

**Reading load:** ~15% (front-loaded on Day 6)

**Note on scheduler modification:** This week focuses on tracing, instrumenting, and observing. You will modify a scheduler heuristic in Phase B (Week 5), when you understand the scheduler deeply enough to change it safely and measure the impact against your collapse experiments.

---

## Day 6 (Mon) — vLLM Concepts + Deployment

### Morning (4 hrs) — Core Papers

- **Read (2 hrs):** vLLM PagedAttention paper (https://arxiv.org/abs/2309.06180), full paper.
  - Focus on:
    - Section 3: PagedAttention — KV cache managed like virtual memory pages
    - Section 4: KV Cache Manager — logical blocks, physical blocks, block table
    - Section 5: Scheduling — preemption policies (swap vs. recomputation)
  - Key concepts:
    - Logical blocks vs. physical blocks (your Day 3 fragmentation simulation is the *problem*; this is the *solution*)
    - Block table maps sequence → physical memory blocks (non-contiguous is fine)
    - Copy-on-write for parallel sampling
    - Preemption: when memory is full, evict lower-priority sequences
- **Read (1 hr):** Orca paper (https://arxiv.org/abs/2206.06063) — iteration-level scheduling
  - Foundational idea behind continuous batching
  - Key insight: schedule at the iteration (step) level, not the batch level
- **Deploy vLLM (1 hr):**
  ```bash
  pip install vllm
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 4096
  ```
  - Send a test request via curl. Confirm tokens come back.

### Afternoon (4 hrs) — Parameter Sensitivity Experiments

- **Experiment 1: `--max-num-seqs` (1.5 hrs):**
  - Values: 1, 4, 8, 16, 32
  - For each: send 16 concurrent requests (identical prompt, 512 tokens input, 256 output)
  - Measure: TTFT p50/p99, throughput (tokens/sec), per-request completion time
- **Experiment 2: `--gpu-memory-utilization` (1.5 hrs):**
  - Values: 0.5, 0.7, 0.85, 0.95
  - For each: ramp concurrent requests until preemption or rejection begins
  - Record: max concurrent requests before degradation
- **Experiment 3: `--max-num-batched-tokens` (1 hr):**
  - Values: 512, 1024, 2048, 4096
  - Send a long prompt (2K tokens). Measure TTFT.

  | Parameter | Values Tested | Key Observation |
  |---|---|---|
  | `--max-num-seqs` | 1, 4, 8, 16, 32 | |
  | `--gpu-memory-utilization` | 0.5, 0.7, 0.85, 0.95 | |
  | `--max-num-batched-tokens` | 512, 1024, 2048, 4096 | |

**End-of-day output:** vLLM running + parameter sensitivity table (3 experiments)

---

## Day 7 (Tue) — Bounded Code Trace: Part 1

### Morning (4 hrs) — Trace: Request Ingress → Scheduler

- **Clone vLLM source** (pin to a specific release tag for stability)
- **Trace Step 1: HTTP Request Ingress**
  - Start at `vllm/entrypoints/openai/api_server.py`
  - Follow the request: OpenAI-compatible endpoint → engine
  - Document: file name, function name, data transformations
- **Trace Step 2: AsyncLLMEngine**
  - Find `vllm/engine/async_llm_engine.py`
  - How does `generate()` work? What gets queued?
  - What is `RequestOutput`? `SamplingParams`?
  - Engine loop: `step()` runs in a loop, each step = one scheduler iteration
- **Trace Step 3: Scheduler**
  - Find `vllm/core/scheduler.py`
  - **This is the most important file in vLLM for your purposes.**
  - Read `Scheduler.schedule()`:
    - What states can a SequenceGroup be in? (WAITING, RUNNING, SWAPPED, FINISHED)
    - How does the scheduler decide what to run?
    - How does it decide what to preempt?
    - Preemption policy: swap to CPU vs. recompute from scratch?

### Afternoon (4 hrs) — Trace: Block Manager → Model Runner + State Machine

- **Trace Step 4: BlockSpaceManager**
  - Find `vllm/core/block_manager.py` (or `block_manager_v2.py`)
  - How are KV cache blocks allocated?
  - Block size? How does it map to tokens?
  - When does `can_allocate()` return False?
  - How does `free()` work?
  - **Connection to Day 3:** This is PagedAttention's implementation. Compare to your fragmentation simulation — paged allocation eliminates the fragmentation you demonstrated.
- **Trace Step 5: ModelRunner → Token Streaming**
  - Find `vllm/worker/model_runner.py`
  - How does `execute_model()` work?
  - Prefill vs. decode execution paths?
  - How do tokens get streamed back?
- **Architecture diagram + SequenceGroup state machine (1 hr):**
  - Map all 5 steps with actual file paths and function names:
    ```
    [api_server.py] HTTP request
      → [async_llm_engine.py] generate() → add to waiting queue
        → [scheduler.py] schedule() → select sequences for this iteration
          → [block_manager.py] allocate/free KV cache blocks
        → [model_runner.py] execute_model() → GPU forward pass
      → stream tokens back via SSE
    ```
  - **Draw a formal state machine diagram for SequenceGroup transitions:**
    ```
    WAITING → RUNNING → FINISHED
       ↑          ↓
       ↑      SWAPPED
       ↑          ↓
       └──────────┘ (recompute path)
    ```
    - What triggers each transition? What code path executes it?
    - This is excellent interview material. When asked "how does vLLM handle memory pressure?" you draw this diagram.

**End-of-day output:** Draft architecture diagram with 5 traced steps + source references + SequenceGroup state machine diagram with transition conditions

---

## Day 8 (Wed) — Instrumentation Patch

### Morning (4 hrs) — Implement Block Allocation Instrumentation

- **Use Option A: KV Cache Block Events.** This is the highest-value instrumentation for the rest of the program.
  - Add logging to `BlockSpaceManager` that records:
    - timestamp, request_id, event_type (ALLOC/FREE/PREEMPT)
    - blocks_allocated, blocks_freed
    - total_used, total_free, total_blocks
  - Example output:
    ```
    [BLOCK_ALLOC]   ts=1234567 req=abc123 alloc=8  freed=0 used=24/128 free=104/128
    [BLOCK_FREE]    ts=1234590 req=abc123 alloc=0  freed=8 used=16/128 free=112/128
    [BLOCK_PREEMPT] ts=1234601 req=def456 alloc=0  freed=4 used=12/128 free=116/128 reason=memory_pressure
    ```
- **Fork vLLM, create branch, implement (3 hrs):**
  - The patch should be 20–50 lines. Don't over-engineer.
  - Focus on the block manager's `allocate()`, `free()`, and any preemption-related code paths.

### Afternoon (4 hrs) — Validate + Initial Experiments

- **Validate (1.5 hrs):**
  - Send 5 sequential requests. Verify block logs make sense:
    - Blocks allocate on request arrival, free on completion
    - No overlap for sequential requests
  - Send 5 concurrent requests. Verify:
    - Blocks accumulate (concurrent allocation)
    - Total used increases with each new request
- **Experiment 1: Prompt Length → Block Allocation (1.5 hrs):**
  - Send prompts: 100, 500, 1000, 2000, 4000 tokens
  - Record blocks allocated per request
  - Calculate: blocks × block_size = actual KV cache bytes
  - Compare to your Day 3 theoretical math. Should match (within block rounding).
- **Experiment 2: Concurrent Block Competition (1 hr):**
  - Send 1, 2, 4, 8, 12, 16 concurrent 2K-token requests
  - Record total blocks used at each concurrency
  - At what concurrency does used approach total? (Capacity ceiling)
  - At what point do you see PREEMPT events?

**End-of-day output:** Working instrumentation patch + validation data + block allocation experiments

---

## Day 9 (Thu) — Mini Collapse + Architecture Document

### Morning (4 hrs) — Mini Collapse Experiment + Continuous Batching

- **Mini collapse experiment (2 hrs):**
  You need one moment in Phase A where the system breaks in front of you. This emotional anchor makes Phase B collapse engineering hit harder.
  - Set `--gpu-memory-utilization 0.98` (dangerously high — almost no headroom)
  - Send 20 concurrent requests with 4K-token prompts
  - **Watch what happens:**
    - Your block allocation logs will show used/total approaching 100%
    - Preemption events will start firing
    - TTFT will spike for all requests (preempted requests need to redo prefill)
    - Some requests may fail entirely
    - If you push hard enough: OOM crash
  - Record:
    - Timeline: when did preemption start? When did TTFT spike? When did failures begin?
    - Block utilization at each stage
    - TTFT before, during, and after preemption cascade
  - **You don't need to write a full postmortem yet** (that's Phase B). Just record the data and write 2–3 sentences: what happened, when, and how it felt watching it.
  - Reset `--gpu-memory-utilization` to a safe value when done.

- **Continuous batching observation (30 min):**
  - Send 10 requests staggered by 200ms intervals
  - In your block logs, observe: new requests get blocks allocated while earlier requests are still generating
  - This is continuous batching made visible.

- **Buffer (1.5 hrs):** Catch up on anything from Day 7–8 that's incomplete. Refine your architecture diagram.

### Afternoon (4 hrs) — Write Annotated Architecture Diagram

- **Write (4 hrs):** The Week 2 deliverable:
  - **Section 1: Request Lifecycle** — 5-step trace with source file and function references
  - **Section 2: SequenceGroup State Machine** — formal state diagram with transition conditions and code references. When does RUNNING → SWAPPED? When does SWAPPED → WAITING? What triggers FINISHED?
  - **Section 3: Scheduler Deep-Dive** — decision tree for `schedule()`: what gets run, what gets preempted, what stays waiting
  - **Section 4: Block Manager** — how blocks are allocated, freed, capacity tracked. Block size, total blocks, connection to your fragmentation simulation from Day 3.
  - **Section 5: Parameter Sensitivity** — table from Day 6
  - **Section 6: Instrumentation Patch** — what you added, where (file + function), example output, and findings:
    - Block allocation matches theoretical KV math
    - Concurrency level where capacity saturates
    - What preemption looks like in the logs
  - **Section 7: Mini Collapse Observation** — what happened when you pushed to 98% memory utilization. Timeline, block data, TTFT spike. 2–3 paragraphs. This previews Phase B.
  - **Section 8: Architecture Diagram** — clean version with all source references

**End-of-day output:** Draft Annotated vLLM Architecture Diagram document

---

## Day 10 (Fri) — Week 2 Deliverable + Buffer

### Morning (4 hrs) — Complete + Self-Test

- **Finish the document (2 hrs):** Polish all sections.
- **Self-test (2 hrs):** Answer from memory:
  1. What happens when a request arrives at vLLM? Walk through each step with file names.
  2. Draw the SequenceGroup state machine. What triggers each transition?
  3. When and why does preemption occur? What are the strategies?
  4. What is PagedAttention and how does it solve the fragmentation problem you demonstrated in Week 1?
  5. How many blocks does a 2K-token request use? (Your instrumentation data.)
  6. At what concurrency does your system hit capacity? (Your instrumentation data.)
  7. What happened during your mini collapse? Describe the timeline.
  8. Describe your instrumentation patch: what you added, what it revealed.

### Afternoon (4 hrs) — Buffer + Week 3 Prep

- Catch up on anything incomplete
- **Prep for Week 3 (1.5 hrs):** Read abstracts + introductions of:
  - AWQ paper (https://arxiv.org/abs/2306.00978)
  - GPTQ paper (https://arxiv.org/abs/2210.17323)
- **Note for Phase B:** In Week 5, you'll return to the scheduler and modify a heuristic (preemption threshold or block allocation strategy) to observe how it shifts your collapse point. You now have the trace knowledge and instrumentation to do this safely. Deferred intentionally — not forgotten.

**End-of-day output:** Finalized Week 2 deliverable (Annotated vLLM Architecture Diagram)

---

# ═══════════════════════════════════
# WEEK 3: Quantization & Model Optimization
# ═══════════════════════════════════

**Goal:** Serve the same model at multiple precisions. Measure quality, throughput, latency, AND capacity tradeoffs — split by prefill-heavy and decode-heavy workloads. Calculate $/million tokens. After this week, you can recommend a precision for a specific use case with numbers, memory math, and cost data.

**Reading load:** ~10%

---

## Day 11 (Mon) — Quantization Theory + Setup

### Morning (4 hrs) — Concepts + Model Preparation

- **Read (1.5 hrs):**
  - Quantization fundamentals: PTQ vs. QAT, weight-only vs. weight+activation, why weight-only works for LLMs
  - AWQ (Sections 1–3): activation-aware weight preservation
  - GPTQ (Sections 1–3): layer-by-layer quantization with calibration data
- **Prepare quantized models (2.5 hrs):**
  - FP16 baseline, INT8-AWQ, INT4-GPTQ — download and verify each runs in vLLM

### Afternoon (4 hrs) — Throughput + Latency Benchmarks (Split Workloads)

- **Benchmark setup (1 hr):**
  - Write a benchmarking script with **two workload profiles:**
    - **Prefill-heavy:** long prompt (2K tokens), short completion (32 tokens) — exercises compute path
    - **Decode-heavy:** short prompt (64 tokens), long completion (512 tokens) — exercises memory-bandwidth path
  - Metrics: throughput (tokens/sec), TTFT (p50/p95/p99), inter-token latency (p50/p95/p99)
- **Run benchmarks (3 hrs):** For EACH precision × EACH workload profile:
  - Concurrency=1, 4, 8, 16
  - Record all metrics
  - **Key observation to look for:** INT8 should improve decode throughput more than prefill throughput, because decode is memory-bound (smaller weights = faster memory reads) while prefill is compute-bound (quantized compute may not be proportionally faster, depending on Tensor Core INT8 support).

  | Precision | Workload | Conc | Throughput (tok/s) | TTFT p99 | ITL p99 |
  |---|---|---|---|---|---|
  | FP16 | Prefill-heavy | 8 | | | |
  | FP16 | Decode-heavy | 8 | | | |
  | INT8-AWQ | Prefill-heavy | 8 | | | |
  | INT8-AWQ | Decode-heavy | 8 | | | |
  | INT4-GPTQ | Prefill-heavy | 8 | | | |
  | INT4-GPTQ | Decode-heavy | 8 | | | |

**End-of-day output:** Benchmark results split by workload type (prefill-heavy vs. decode-heavy) across precisions

---

## Day 12 (Tue) — Quality + Capacity + Cost

### Morning (4 hrs) — Quality Evaluation

- **Setup and run quality evals (2 hrs):**
  - Perplexity on WikiText-2 for each precision
  - At least one downstream task (HellaSwag, ARC-Easy, or MMLU subset)
- **Qualitative error analysis (1 hr):**
  - 10 diverse prompts: 2 Q&A, 2 creative, 2 code, 2 reasoning, 2 long-context
  - Generate at all 3 precisions with identical sampling params
  - Side-by-side read. Flag cases where INT4 noticeably degrades.
  - 10 examples, 1 hour. Build taste, don't chase rabbits.
- **Connection check (1 hr):** Do your quality results align with the workload benchmarks?
  - If INT4 degrades quality on code prompts but not Q&A, does that match what you'd expect from the quantization approach (less precision = more impact on tasks requiring exact reasoning)?

### Afternoon (4 hrs) — Capacity Impact + Cost Model

- **Memory footprint + KV concurrency (1.5 hrs):**
  - Measure model weight size per precision (from nvidia-smi after loading)
  - Compute available KV cache memory for each
  - Run your KV cache calculator:

    | Precision | Model Size (GB) | Free for KV (GB) | Max Conc @ 2K | Max Conc @ 4K | Max Conc @ 8K |
    |---|---|---|---|---|---|
    | FP16 | | | | | |
    | INT8-AWQ | | | | | |
    | INT4-GPTQ | | | | | |

  - **State the insight:** "INT8-AWQ reduces model size by X GB, freeing Y GB for KV cache. This increases max concurrent requests at 4K sequence length from A to B — a Z% improvement. Quantization's real value is capacity, not just speed."

- **$/million tokens cost model (1.5 hrs):**
  - Get your GPU's hourly cost (e.g., A100-40G spot: ~$1.80/hr)
  - For each precision at concurrency=8 (moderate load):
    - Throughput = T tokens/sec from your benchmarks
    - Tokens per hour = T × 3600
    - $/million tokens = (hourly cost / tokens per hour) × 1,000,000

    | Precision | Throughput @ conc=8 | Tokens/hr | $/M tokens |
    |---|---|---|---|
    | FP16 | | | |
    | INT8-AWQ | | | |
    | INT4-GPTQ | | | |

  - This is operator thinking. "INT8-AWQ reduces our serving cost from $X/M tokens to $Y/M tokens — a Z% reduction — with W% perplexity increase."

- **Complete tradeoff table (1 hr):**

  | Metric | FP16 | INT8-AWQ | INT4-GPTQ |
  |---|---|---|---|
  | Model size (GB) | | | |
  | Free memory for KV (GB) | | | |
  | Max concurrent @ 4K seq | | | |
  | Throughput (decode-heavy, conc=8) | | | |
  | Throughput (prefill-heavy, conc=8) | | | |
  | TTFT p99 (conc=8) | | | |
  | $/M tokens (conc=8) | | | |
  | Perplexity (WikiText-2) | | | |
  | Downstream task score | | | |
  | Qualitative issues | | | |

**End-of-day output:** Complete tradeoff table with quality + performance + capacity + cost

---

## Day 13 (Wed) — Prefix Caching + FP8 KV Cache

### Morning (4 hrs) — Prefix Caching

- **Read (30 min):** Prefix caching in vLLM
- **Experiment (3.5 hrs):**
  - 100 requests sharing a 500-token system prompt + unique 100-token user messages
  - Without prefix caching: measure TTFT
  - With prefix caching (`--enable-prefix-caching`): measure TTFT
  - Vary system prompt length: 100, 500, 1000, 2000 tokens
  - At what length does prefix caching become a significant win?

### Afternoon (4 hrs) — FP8 KV Cache + Documentation

- **FP8 KV Cache (2 hrs):**
  - If H100: enable `--kv-cache-dtype fp8`, measure max concurrent increase + quick quality check
  - If no FP8: calculate theoretical benefit with your KV calculator, document
- **Update tradeoff analysis (2 hrs):** Add prefix caching and FP8 KV cache results.

**End-of-day output:** Prefix caching results + FP8 KV cache analysis

---

## Day 14 (Thu) — Speculative Decoding

### Morning (4 hrs) — Setup + Concepts

- **Read (1 hr):** Speculative decoding: draft model generates candidates, target model verifies in one pass.
- **Setup (3 hrs):** Configure vLLM with a draft model. Verify it works.

### Afternoon (4 hrs) — Experiments

- **Vary speculative tokens (1.5 hrs):** 3, 5, 7, 10 — measure throughput, TTFT, acceptance rate
- **Vary task type (1.5 hrs):** Q&A, creative writing, code — measure per-task speedup
- **Concurrency impact (30 min):** Does spec decode help or hurt at high concurrency?
- **Build decision table (30 min):**

  | Scenario | Speedup | Acceptance Rate | Use Spec Decode? |
  |---|---|---|---|
  | Simple Q&A, conc=1 | | | |
  | Simple Q&A, conc=8 | | | |
  | Creative writing, conc=1 | | | |
  | Code generation, conc=4 | | | |

**End-of-day output:** Spec decode results + decision table

---

## Day 15 (Fri) — Week 3 Deliverable

### Morning (4 hrs) — Write Quantization & Optimization Tradeoff Analysis

- **Write (4 hrs):**
  - **Section 1: Approaches** — AWQ, GPTQ, FP8 KV cache (brief)
  - **Section 2: Performance by Workload Type** — prefill-heavy vs. decode-heavy results per precision. State: "INT8 improves decode throughput by X% but prefill by only Y%, because decode is memory-bound and prefill is compute-bound."
  - **Section 3: Quality** — Perplexity + task score + qualitative observations
  - **Section 4: Capacity** — Quantization as KV cache capacity lever. Exact numbers.
  - **Section 5: Cost** — $/million tokens per precision. ROI calculation.
  - **Section 6: Prefix Caching** — When to use, measured benefits
  - **Section 7: Speculative Decoding** — When it helps, when it hurts
  - **Section 8: Recommendations per use case:**
    - Latency-sensitive chat: recommend X, because... (cost + quality + capacity)
    - Throughput-maximizing batch: recommend Y, because...
    - Quality-critical (code, medical): recommend Z, because...

### Afternoon (4 hrs) — Buffer + Self-Test

- **Self-test:** Answer with your own numbers:
  1. What throughput improvement does INT8 give for decode-heavy vs. prefill-heavy workloads? Why the difference?
  2. What's the perplexity cost? Did you see qualitative degradation?
  3. How many additional concurrent 4K requests with INT8 vs. FP16? (Exact number.)
  4. What's your $/million tokens at FP16 vs. INT8?
  5. When does spec decode help vs. hurt?
  6. If leadership says "just use INT4 for everything," what's your response? (Quality data + cost data.)

**End-of-day output:** Finalized Week 3 deliverable

---

# ═══════════════════════════════════
# WEEK 4: Admission Control & Request Routing
# ═══════════════════════════════════

**Goal:** Build a token-aware API gateway where admission decisions are derived directly from KV cache memory budgets. Include realistic token estimation correction. Test with adversarial traffic. After this week, your system models inference capacity in memory terms and handles hostile request patterns.

**Reading load:** ~10%

**Core framing:** Admission control is not "how many requests can I allow." It is "how many KV cache tokens can I have active before I hit my cliff." Your Week 1 KV cache calculator IS your admission policy — the gateway enforces it in real-time.

---

## Day 16 (Mon) — KV-Memory-Driven Admission Design

### Morning (4 hrs) — Concepts + Design from Memory Math

- **Read (1 hr):**
  - Little's Law: L = λW
  - Load shedding: fail-fast vs. queue-and-wait
  - HTTP SSE for streaming tokens
- **Derive admission policy from KV math (1.5 hrs):**
  - Open your KV cache calculator from Week 1
  - Your GPU has T total KV cache tokens capacity
  - Cliff at ~C% utilization (estimate 65–70%, precise measurement in Phase B)
  - **Admission budget = T × C%**
  - Each request costs: prompt_tokens + max_completion_tokens
  - Policy:
    ```
    ADMISSION_BUDGET = KV_CACHE_CAPACITY_TOKENS × TARGET_UTILIZATION

    on_request(request):
      estimated_cost = request.prompt_tokens + request.max_completion_tokens
      if active_token_budget + estimated_cost > ADMISSION_BUDGET:
        return 429 with Retry-After header
      else:
        active_token_budget += estimated_cost
        forward to vLLM

    on_request_complete(request):
      active_token_budget -= estimated_cost  # release full estimate
    ```
  - **This is not a concurrency cap. It is a memory budget, expressed in tokens.**
- **Design gateway architecture (1.5 hrs):**
  - HTTP → token counter → admission check → vLLM → SSE streaming
  - Per-API-key rate limiting
  - Bounded queue for burst absorption

### Afternoon (4 hrs) — Start Building

- **Implement (4 hrs):** Build gateway (FastAPI or Go):
  - HTTP endpoint for chat completion requests
  - Token counting (tiktoken or vLLM tokenizer)
  - Forward to vLLM, stream response via SSE
  - Get end-to-end flow working (admission logic comes tomorrow)

**End-of-day output:** Working proxy gateway + admission policy design document

---

## Day 17 (Tue) — Admission Control + Token Budget Correction

### Morning (4 hrs) — Token Budget Enforcement

- **Implement (4 hrs):**
  - Active token budget tracker (atomic counter)
  - Admission check: active_tokens + estimated_cost ≤ ADMISSION_BUDGET
  - Hard concurrency cap as safety net
  - Budget configuration from your KV calculator:
    ```python
    KV_CAPACITY_TOKENS = 52_000
    TARGET_UTILIZATION = 0.65
    ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~33,800
    ```

### Afternoon (4 hrs) — Token Estimation Correction + Rate Limiting

- **Implement token budget correction (1.5 hrs):**
  In reality, most requests terminate well below `max_completion_tokens`. If you always budget at max, you waste capacity (reject requests that would have fit). If you budget optimistically, you risk OOM.
  - **Solution: Budget at max on admission, release the delta as tokens stream.**
    ```python
    on_request(request):
      estimated_cost = prompt_tokens + max_completion_tokens
      active_token_budget += estimated_cost  # reserve max

    on_token_generated(request):
      # As each token streams, actual cost becomes clearer
      # Optionally: release excess budget periodically
      pass

    on_request_complete(request):
      actual_cost = prompt_tokens + actual_completion_tokens
      overestimate = estimated_cost - actual_cost
      active_token_budget -= estimated_cost  # release full reservation
      # Net effect: budget freed = estimated_cost
      # But during the request, we held estimated_cost worth of budget
    ```
  - **Measure the impact:** Send 100 requests where max_completion=512 but actual average completion=150.
    - Without correction: how many requests get rejected due to budget exhaustion?
    - With periodic budget release (release excess every 50 tokens): how many more requests are admitted?
    - This quantifies the capacity improvement from smarter budget tracking.

- **Rate limiting + queue (2.5 hrs):**
  - Per-API-key: requests/min and tokens/min limits
  - Bounded FIFO queue: max depth, max wait time, fail-fast on overflow
  - Manual testing: verify admission, rate limiting, queue behavior

**End-of-day output:** Working admission control with token budget correction + rate limiting + queue

---

## Day 18 (Wed) — Load Testing + Dashboards

### Morning (4 hrs) — Load Tests

- **Setup (1 hr):** Locust or k6 script with realistic traffic:
  - 20% short (100 tokens), 50% medium (500 tokens), 30% long (2000 tokens)
  - Ramp from 1 to 50 concurrent users
- **Test 1: No admission control (1.5 hrs):**
  - Push until system breaks. Record: when does TTFT degrade? Preemption? OOM?
- **Test 2: With admission control (1.5 hrs):**
  - Same traffic, token budget enforced at 65%
  - Record: rejection rate, TTFT for admitted requests
  - Admitted requests should have stable latency — system never approaches cliff

### Afternoon (4 hrs) — Dashboards

- **Setup Prometheus + Grafana (2 hrs):**
  - Export: request_count, rejection_rate, queue_depth, active_token_budget_utilization (%)
  - Export from vLLM: GPU memory, TTFT, throughput
  - Export GPU metrics
- **Build dashboard (2 hrs).** Must include:
  - Active token budget utilization (%) — primary operational metric
  - Queue depth
  - GPU memory utilization
  - TTFT p50/p95/p99
  - Rejection rate
  - Side-by-side: token budget utilization vs. TTFT

**End-of-day output:** Load test results (with vs. without admission) + working dashboard

---

## Day 19 (Thu) — Adversarial Testing + Design Note

### Morning (4 hrs) — Adversarial Experiments

- **Adversarial request simulation (1.5 hrs):**
  - Send 5 short requests (100 tokens each) from Tenant A
  - Send 1 massive request (8K tokens prompt + 4K completion) from Tenant B
  - Observe: does the massive request starve Tenant A?
  - Under your token budget system: the massive request consumes ~12K tokens of budget. Does this leave enough for the short requests?
  - If not: this previews why per-tenant budgets matter (Phase C)
  - Record: per-tenant TTFT, budget consumption, rejection patterns

- **Head-of-line blocking experiment (1 hr):**
  - Fill the queue with requests
  - Place a very long request (8K tokens) at the front of the queue
  - Send 10 short requests (100 tokens each) behind it
  - Observe: do short requests wait for the long request to be admitted?
  - Does your queue discipline need priority or reordering?
  - Record: wait time for short requests vs. long requests
  - **Insight:** FIFO queuing for inference is naive. In Phase C, you'll implement weighted fair queuing. This experiment shows why.

- **Queue depth → latency curves (1.5 hrs):**
  - Hold traffic at 25%, 40%, 50%, 60%, 70%, 80%, 90% of admission budget
  - At each level: stabilize 3 minutes, record token budget %, queue depth, TTFT p50/p95/p99
  - Plot the hockey stick

### Afternoon (4 hrs) — Write Admission Control Design Note

- **Write (4 hrs):**
  - **Section 1: Admission as KV Memory Budget**
    - Derivation: GPU memory → model size → KV capacity → admission budget
    - "Admission is not a concurrency cap. It is a memory budget, expressed in tokens."
  - **Section 2: Why Token-Aware Beats Request-Count**
    - Your data: 10 short requests use ~1,000 tokens. 10 long requests use ~80,000. A flat cap of 10 either wastes capacity or causes OOM.
  - **Section 3: Token Budget Correction**
    - Why budgeting at max is conservative, how periodic release improves capacity
    - Your measured improvement: X% more requests admitted with correction
  - **Section 4: Adversarial Resilience**
    - Malicious large-request results
    - Head-of-line blocking results
    - Why FIFO is insufficient and per-tenant budgets + priority queuing are needed (preview of Phase C)
  - **Section 5: Queue Depth → Latency Curves** (your graphs)
  - **Section 6: With vs. Without Admission Control** (load test comparison)
  - **Section 7: Dashboard** — screenshots with metric explanations
  - **Section 8: Architecture Diagram** — gateway design showing token budget flow

**End-of-day output:** Admission Control Design Note (complete)

---

## Day 20 (Fri) — Phase A Exit

### Morning (4 hrs) — Finalize + Exit Assessment

- Polish all deliverables
- **Phase A Exit Self-Assessment (2.5 hrs):** Write 1–2 paragraphs each:
  1. **Request lifecycle:** Full path from HTTP to GPU to streamed token. Cite vLLM source files.
  2. **SequenceGroup state machine:** Draw it, explain each transition.
  3. **PagedAttention:** What problem does it solve? Reference your fragmentation simulation.
  4. **Quantization tradeoffs:** Present your data. How does INT8 affect decode vs. prefill differently? Why? State KV capacity impact. State $/M tokens impact.
  5. **GPU utilization is misleading:** What metrics are more informative? (Memory throughput, achieved occupancy, SM active % — reference your Nsight data.)
  6. **Concurrent request capacity:** How many concurrent 4K requests at FP16? INT8? Math.
  7. **10 simultaneous long-context requests:** Walk through in memory terms. When does preemption start? What happens to latency? (Reference your mini collapse from Day 9.)
  8. **Admission control as memory budget:** Why derived from KV math, not concurrency count. How does token budget correction improve capacity.
  9. **Adversarial requests:** What happens when one client sends a massive request? Why is FIFO queuing insufficient?
  10. **Your instrumentation patch:** What you added, what it revealed.

### Afternoon (4 hrs) — Phase B Preparation

- Review complete system: instrumented vLLM + admission gateway + dashboards
- Ensure everything is stable and repeatable
- **Phase B preview:** In Week 5, you will:
  - Intentionally push past your admission limits to observe systematic collapse
  - Modify a scheduler heuristic to shift the collapse point
  - Produce your first full postmortem
  - Break everything you just built, methodically

**End-of-day output:** Phase A complete. All 4 deliverables finalized:

1. ✅ GPU Architecture & Memory Budget Document (with Nsight profiling, fragmentation simulation, TP cost preview)
2. ✅ Annotated vLLM Architecture Diagram (with instrumentation patch, state machine, mini collapse observation)
3. ✅ Quantization & Optimization Tradeoff Analysis (with prefill/decode split, KV capacity impact, $/M tokens)
4. ✅ Admission Control Design Note (KV-memory-derived, token correction, adversarial testing)

---

# Appendix: Phase A Reading List

| Resource | Day | Time |
|---|---|---|
| CUDA C Programming Guide (targeted sections) | Day 1 | 1.5 hrs |
| GPU architecture overview (Horace He or whitepaper) | Day 1 | 1 hr |
| Roofline model primer | Day 2 | 1 hr |
| vLLM paper (PagedAttention) — full | Day 6 | 2 hrs |
| Orca paper (iteration-level scheduling) — skim | Day 6 | 1 hr |
| AWQ paper — Sections 1–3 | Day 10 | 30 min |
| GPTQ paper — Sections 1–3 | Day 10 | 30 min |
| Speculative decoding overview | Day 14 | 1 hr |
| Admission control / Little's Law | Day 16 | 1 hr |
| **Total** | | **~9.5 hrs** |

---

# Appendix: Phase A Deliverables Checklist

| # | Deliverable | Due | Key Features |
|---|---|---|---|
| 1 | GPU Architecture & Memory Budget Document | Day 4 | Nsight occupancy + SM active %, warp microbench, fragmentation simulation, TP cost preview |
| 2 | Annotated vLLM Architecture Diagram | Day 10 | Bounded code trace, SequenceGroup state machine, block allocation instrumentation, mini collapse observation |
| 3 | Quantization & Optimization Tradeoff Analysis | Day 15 | Prefill/decode workload split, KV capacity calculation, $/M tokens cost model, qualitative error analysis |
| 4 | Admission Control Design Note | Day 19 | KV-memory-derived admission, token budget correction, adversarial request testing, HOL blocking analysis |

---

# Appendix: Full Change Log

| Version | Change | Source | Rationale |
|---|---|---|---|
| v1→v2 | Added Nsight occupancy profiling (Day 2) | LLM2 R1 | Distinguishes memory-stalled vs. underloaded |
| v1→v2 | Added KV precision preview (Day 3) | LLM2 R1 | Connects quantization to capacity early |
| v1→v2 | Added KV→concurrency calculation (Day 12) | LLM2 R1 | Staff-level insight: quantization is a capacity lever |
| v1→v2 | Added qualitative error analysis (Day 12) | LLM2 R1 (scoped) | Builds taste for quantization failures |
| v1→v2 | Reframed Day 16 as KV-memory-derived admission | LLM2 R1 | More engine-native than concurrency cap |
| v1→v2 | Deferred scheduler modification to Phase B | Pushback on LLM2 R1 | Week 2 too dense; Phase B has better context |
| v2→v3 | Trimmed CUDA reading, added warp/shape microbench (Day 1) | LLM2 R2 | 30 min saved on reading → empirical proof of roofline before theory |
| v2→v3 | Added SM Active % to Nsight table (Day 2) | LLM2 R2 | Occupancy ≠ SM utilization; interviewers test this |
| v2→v3 | Added memory fragmentation simulation (Day 3) | LLM2 R2 | Visceral understanding of why PagedAttention matters |
| v2→v3 | Compressed MIG to overview-only (Day 3) | Time rebalance | MIG hands-on not worth setup time; overview sufficient |
| v2→v3 | Added TP communication cost preview (Day 4) | LLM2 R2 | "When does TP over PCIe make things worse?" — staff-level mental model |
| v2→v3 | Added SequenceGroup state machine diagram (Day 7) | LLM2 R2 | Authoritative interview artifact; 20 min investment |
| v2→v3 | Forced Option A for instrumentation (Day 8) | LLM2 R2 | Block allocation is highest-value instrumentation for Phase B |
| v2→v3 | Added mini collapse experiment (Day 9) | LLM2 R2 | Emotional anchor needed before Phase B systematic collapse |
| v2→v3 | Added prefill/decode workload split to benchmarks (Day 11) | LLM2 R2 | Shows roofline awareness: INT8 helps decode more than prefill |
| v2→v3 | Added $/million tokens cost model (Day 12) | LLM2 R2 | Operator thinking: quantization as cost lever with actual numbers |
| v2→v3 | Added token budget correction (Day 17) | LLM2 R2 | Real-world realism: max_completion overestimates waste capacity |
| v2→v3 | Added adversarial request simulation (Day 19) | LLM2 R2 | Tests policy robustness; previews multi-tenant isolation |
| v2→v3 | Added head-of-line blocking experiment (Day 19) | LLM2 R2 | Shows why FIFO is insufficient; previews fair queueing |

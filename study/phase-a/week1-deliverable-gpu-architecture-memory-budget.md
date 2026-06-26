# Week 1 Deliverable: GPU Architecture & Memory Budget Document

**Phase A - GPU Architecture**
**Hardware:** Tesla T4 (15,360 MiB VRAM, Turing sm75)
**Models tested:** microsoft/Phi-2 (2.7B), Qwen2.5-3B-Instruct

---

## Section 1: Memory Hierarchy

### GPU Specifications — Tesla T4

- **Architecture:** Turing (sm75, compute capability 7.5)
- **VRAM:** 15,360 MiB (15 GB) GDDR6
- **FP16 Tensor Core peak:** 65 TFLOPS
- **Memory bandwidth (theoretical):** 320 GB/s
- **TDP:** 70W
- **SMs:** 40
- **Persistence mode:** On (reduces startup latency)

### Measured Bandwidths

**GPU-to-GPU (HBM on-device):**

| Transfer Size | Bandwidth | % of Theoretical |
|---|---|---|
| 1 MB | 179.0 GB/s | 56% |
| 10 MB | 213.7 GB/s | 67% |
| 100 MB | 216.7 GB/s | 68% |
| 1000 MB | 217.0 GB/s | 68% |

On-device HBM achieves ~217 GB/s at large transfers (68% of 320 GB/s theoretical). Expected range is 60-80%.

**GPU-to-CPU (PCIe):**

| Transfer Size | Bandwidth |
|---|---|
| 1 MB | 5.0 GB/s |
| 10 MB | 8.6 GB/s |
| 100 MB | 9.2 GB/s |
| 1000 MB | 9.4 GB/s |

**CPU-to-GPU (PCIe):**

| Transfer Size | Bandwidth |
|---|---|
| 1 MB | 5.1 GB/s |
| 10 MB | 8.9 GB/s |
| 100 MB | 8.9 GB/s |
| 1000 MB | 8.9 GB/s |

**Key ratio:** HBM is ~23x faster than PCIe for large transfers (217 GB/s vs 9.4 GB/s). This is why V1's decision to eliminate GPU<->CPU KV swap in favor of recompute makes sense — the swap path was bottlenecked by PCIe, not HBM.

### nvidia-smi dmon Observations Under Load

- **Idle:** 9-13W, 24-30C, P8 state, clocks at 405/300 MHz (mclk/pclk)
- **Active burst:** 68W (near 70W cap), 30-37C, sm=89-100%, mem=37-100%, clocks at 5000/1590 MHz
- **Key pattern:** High sm% + high mem% = both compute and memory controllers active (typical of transformer workloads)

**Caveat on sm%:** This metric reports the percentage of time any SM had at least one warp executing. Both fully saturated SMs and a single active warp per SM can show sm% ~ 100%. It is a necessary but not sufficient indicator of compute utilization.

---

## Section 2: Arithmetic Intensity — Prefill vs. Decode Roofline

### Ridge Point Calculation (T4)

```
Ridge point = Peak TFLOPS / Memory Bandwidth
            = 65 TFLOPS / 320 GB/s
            = 203 FLOPs/byte
```

### Decode Arithmetic Intensity (7B model, FP16)

```
Weights = 7B parameters x 2 bytes = 14 GB
FLOPs per decode step = 2 x 7B = 14 GFLOPs  (2 FLOPs/parameter for matmul)
AI = 14 GFLOPs / 14 GB = 1 FLOP/byte
```

Since 1 << 203 (ridge point), decode is **deeply memory-bandwidth-bound**. The GPU can achieve at most ~0.5% of peak compute during decode. The intuition: decode re-reads the entire model weights from HBM for every single output token.

### Prefill Arithmetic Intensity (7B model, FP16, 2048-token prompt)

```
FLOPs = 2 x 7B x 2048 = 28.7 TFLOPs
AI = 28.7 TFLOPs / 14 GB = 2048 FLOPs/byte
```

Since 2048 >> 203 (ridge point), prefill is **firmly compute-bound**. The intuition: prefill amortizes a single read of the model weights across all prompt tokens, turning the same memory read into 2048x more useful work.

### Measured TFLOPS vs. Batch Size

Benchmark: `(batch, 4096) @ (4096, 4096)` FP16 matmul, 200 rounds:

| Batch | TFLOPS | % of T4 Peak |
|---|---|---|
| 1 | 0.0 | 0% |
| 4 | 0.4 | 1% |
| 16 | 3.5 | 5% |
| 64 | 12.0 | 19% |
| 256 | 19.4 | 30% |
| 1024 | 23.0 | 35% |

The T4 peaks at ~23 TFLOPS FP16 (35% of theoretical 65 TFLOPS). This confirms: at small batch sizes the GPU is severely underutilized — not because it's hitting memory bandwidth limits, but because there isn't enough parallel work to occupy all SMs. The theoretical peak assumes perfectly saturated tensor cores.

**Power-limited note:** Extended runs (rounds=5000) showed the T4 was power-limited at 70W TDP, only reaching 23 TFLOPS at batch=4096.

---

## Section 3: Occupancy & Throughput — Nsight Profiling Results

Profiled microsoft/Phi-2 (2.7B) on T4 using Nsight Systems (nsys) and Nsight Compute (ncu). Clean capture using `cudaProfilerStart/Stop` to exclude model load.

### Prefill Kernel Characterization

**Kernel mix by GPU time:**
- `turing_fp16_s1688gemm` (256x128): 39.2% — Tensor Core FP16 GEMM
- `turing_fp16_s1688gemm` (128x256): 21.9% — Tensor Core FP16 GEMM
- `fmha_cutlassF_f16`: 9.3% — fused multi-head attention
- Remaining ~30%: elementwise ops (layernorm, tanh, add, pow, cat copies)

**Combined GEMMs = ~61% of GPU time.** Prefill is GEMM-dominated.

**ncu roofline for dominant prefill GEMM:**
- Compute (SM) Throughput: **83.87%** of peak
- DRAM Throughput: 24.59% of peak
- SM Active: ~98.8%
- Memory Throughput: 38.58%

**Conclusion:** The dominant GEMM is **compute-bound** and operating near Tensor Core saturation. Nsight's own recommendation: "shift work from compute to another unit."

**ncu roofline for attention (FMHA):**
- Compute (SM) Throughput: **27.82%** of peak
- SM Active: ~95%
- DRAM Throughput: 6.50%
- L1/TEX Throughput: **83.40%**

**Conclusion:** Attention is **not compute-saturated** and **not DRAM-bound**. It is **latency-limited and cache-heavy**, relying on L1/TEX paths. High SM active cycles (~95%) with low compute throughput (~28%) indicates dependency chains, warp synchronization, and limited instruction-level parallelism.

### Decode Kernel Characterization

**Kernel launch profile:**
- 53,000 `cudaLaunchKernel` calls — launch-heavy (scales with tokens x layers x ops/layer)
- Dominant kernels: `gemvx` (matrix-vector) + small elementwise ops
- Significant `cudaStreamSynchronize` time — CPU waits between GPU segments
- Minimal PCIe traffic (bottleneck is internal HBM weight streaming)

**ncu roofline for dominant decode kernel (`cublasGemvParamsEx`):**
- Memory (DRAM) Throughput: **96.10%** of peak
- Compute (SM) Throughput: **58.47%**
- SM Active: ~99.5%
- L1/TEX Throughput: 87.05%

**Conclusion:** Decode is **memory-bandwidth-bound at 96% DRAM throughput**. The GPU is spending almost all its time streaming weights from HBM. Nsight's recommendation: "shift work from memory to another unit."

**Warp stall analysis (decode GEMV):**
- Warp cycles per issued instruction: 14.39
- Primary stall: 47.3% waiting for L1 instruction queue (LG memory operations)
- Avg active threads per warp: 31.22 (near-full warps)

### Summary Table

| Metric | Prefill (GEMM) | Decode (GEMV) |
|---|---|---|
| Compute throughput | 83.87% | 58.47% |
| DRAM throughput | 24.59% | 96.10% |
| SM Active | ~98.8% | ~99.5% |
| Primary bottleneck | Compute (Tensor Core) | Memory (DRAM bandwidth) |
| Primary stall | N/A (near saturation) | LG memory queue (47%) |
| Kernel type | Large GEMM (tiles) | GEMV (vector ops) |

This confirms the roofline prediction from Section 2: prefill is compute-bound (AI >> ridge point), decode is memory-bound (AI << ridge point).

---

## Section 4: KV Cache Budget

*Partially complete — no Day 3 KV cache calculator artifact found. Calculations reconstructed from Day 6 experiment notes.*

### Per-Token KV Cache Size (Qwen2.5-3B-Instruct)

```
Architecture:
  num_layers   = 36
  num_kv_heads = 2  (GQA: 2 KV heads, 16 attention heads)
  head_dim     = 128
  dtype        = FP16 (2 bytes)

Per-token KV size = 2 (K+V) x 36 layers x 2 heads x 128 dim x 2 bytes
                  = 2 x 36 x 2 x 128 x 2
                  = 36,864 bytes
                  = 36 KB per token
```

### Block-Level KV Cache (vLLM configuration)

vLLM uses paged KV cache with configurable block size (default: 16 tokens/block).

```
Per-block KV size = 36 KB/token x 16 tokens/block = 576 KB per block

Block size formula (from vllm/v1/kv_cache_interface.py):
  real_page_size_bytes = 2 x block_size x num_kv_heads x head_size x dtype_size
  Per layer group: 2 x 16 x 2 x 128 x 2 = 16,384 bytes
  x 36 layers = 589,824 bytes (~576 KB)
```

### KV Cache Pool at Various Utilizations (Qwen2.5-3B, T4 15GB)

Model weights overhead: ~8.44 GB (for Qwen2.5-3B FP16 + CUDA context + activations).

| gpu_memory_utilization | Total GPU budget | KV cache budget | Approx blocks | Token capacity |
|---|---|---|---|---|
| 0.45 | 6.91 GB | ~1.1 GB | ~1,021 | ~16,336 |
| 0.70 | 10.75 GB | ~1.75 GB | ~3,100 | ~49,600 |
| 0.85 | 13.06 GB | ~3.94 GB | ~6,800 | ~108,800 |
| 0.90 | 13.82 GB | ~5.38 GB | ~9,300 | ~148,800 |
| 0.95 | 14.59 GB | ~5.39 GB | ~9,400 | ~150,400 |

### Capacity Planning

```
max_concurrent_requests = total_blocks / blocks_per_request

where blocks_per_request = ceil(prompt_tokens / 16) + ceil(max_output_tokens / 16)
```

Example: 1,203-token prompt + 200-token output = 75 + 13 = 88 blocks/request.
At 0.45 utilization (1,021 blocks): max_concurrent = floor(1021/88) = **11 requests**.

---

## Section 5: Memory Fragmentation

*Incomplete — no Day 3 fragmentation simulation artifact found.*

### Why Paged Allocation Matters (Conceptual)

Traditional KV cache allocation (pre-PagedAttention) pre-allocates contiguous memory for each sequence's maximum possible length. This creates two problems:

- **Internal fragmentation:** A sequence that generates 100 tokens but was allocated for 2048 wastes 95% of its reservation
- **External fragmentation:** Released contiguous chunks create holes that can't be reused by sequences needing different sizes

PagedAttention (the core vLLM innovation) eliminates both by allocating KV cache in fixed-size blocks (16 tokens each), mapped via a block table. Blocks can be non-contiguous in physical memory but appear contiguous to the attention kernel via indirection.

**Measured impact (from Week 2 experiments):** At 0.45 utilization with 1,034 blocks, the block pool was fully utilized across 32 concurrent sequences — each holding a different number of blocks depending on their decode progress. Without paging, the same 32 sequences at max_tokens=1900 would require 32 x ceil(1900/16) = 32 x 119 = 3,808 blocks, exceeding the pool by 3.7x.

---

## Section 6: Interconnect & TP Cost Preview

*Incomplete — no Day 4 interconnect/TP artifact found. The T4 is a single-GPU setup.*

### Topology

The Tesla T4 connects via PCIe (measured at ~9 GB/s bidirectional). In a multi-GPU setup, tensor parallelism (TP) requires all-reduce operations after each transformer layer's attention and MLP computations.

### TP Communication Cost (Conceptual)

For TP degree N across PCIe:
```
All-reduce volume per layer = 2 x (N-1)/N x hidden_size x seq_len x dtype_bytes
Bandwidth cost = volume / PCIe_bandwidth
```

On T4 with PCIe at ~9 GB/s, TP across multiple GPUs would be severely bandwidth-limited compared to NVLink-equipped GPUs (A100: 600 GB/s, H100: 900 GB/s). This is why T4 is a single-GPU inference card — TP across PCIe would negate throughput gains from additional compute.

---

## Section 7: Key Takeaways

### 1. Decode is memory-bound, prefill is compute-bound

Measured with Nsight Compute: decode GEMV hits 96% DRAM throughput while prefill GEMM hits 84% compute throughput. This matches the roofline prediction (AI=1 for decode vs AI=2048 for prefill at 2048-token prompt).

### 2. The T4 peaks at ~35% of theoretical compute

23 TFLOPS achieved vs 65 TFLOPS theoretical. Power-limited at 70W TDP. Real-world inference workloads should plan for this ceiling, not the spec sheet.

### 3. HBM is 23x faster than PCIe

On-device bandwidth (~217 GB/s) vs PCIe (~9.4 GB/s). This ratio explains why vLLM V1 dropped GPU<->CPU KV swap — recomputing KV from HBM is faster than swapping via PCIe for most prompt lengths.

### 4. Batch size is the primary lever for GPU utilization

At batch=1, the GPU achieves 0% of peak. At batch=1024, it achieves 35%. Continuous batching's value is keeping the batch size large enough to amortize memory access across multiple requests.

### 5. KV cache is the binding constraint for concurrent serving

At 36 KB/token for Qwen2.5-3B, a 2048-token sequence consumes ~72 MB of KV cache. With only ~1-5 GB available for KV (depending on utilization setting), the system can serve 11-80 concurrent sequences before exhausting the block pool.

### 6. Attention is neither compute-bound nor memory-bound

Nsight Compute shows FMHA at 28% compute throughput and 6.5% DRAM throughput, but 83% L1/TEX throughput. It's latency-limited with dependency chains, not throughput-limited. This is why Flash Attention's algorithmic improvements (tiling, reduced HBM reads) matter more than raw bandwidth.

### 7. sm% is misleading without ncu

nvidia-smi dmon showed sm=89-100% during workloads, but Nsight Compute revealed the decode kernel is actually memory-stalled 47% of the time. sm% only measures "was any warp active" — it doesn't distinguish between fully saturated SMs and a single active warp per SM.

---

*Phase A - Week 1 of 4 - GPU Architecture*
*Note: Sections 4-6 are partially reconstructed from later experiments. No dedicated Day 3-4 artifacts were found.*

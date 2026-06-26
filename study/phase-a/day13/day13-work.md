# Day 13 Work Log

## Step 1: Prefix Caching - Key Concepts

### How it works

Prefix caching reuses KV cache blocks across requests that share identical token prefixes. vLLM hashes token IDs within each block (block_size=16) and looks up existing cached blocks before allocating new ones. The lookup happens in `KVCacheManager.get_computed_blocks()`, which calls `coordinator.find_longest_cache_hit()` against the request's block hashes.

### What it saves

Cached prefill tokens don't need to be recomputed or re-scheduled. In vLLM V1's unified scheduler, this frees token budget for other requests. The benefit depends on workload:

- **Prefill-heavy** (long system prompts, RAG): improves both TTFT and effective throughput, because freed scheduler budget lets new requests start sooner.
- **Decode-heavy**: primarily TTFT reduction. Decode work for generated tokens is unchanged.

### Constraints

- Only **full blocks** are cached. A 483-token prefix caches 480 tokens (30 blocks of 16). The remaining 3 tokens always recompute.
- Even on a full cache hit, the **last token is always recomputed** to produce logits. This can force an entire block to recompute since `num_computed_tokens` must be block-size aligned.
- Cache hits require **exact token-level matches** at block boundaries. One different token means a full hash mismatch.

### Four failure modes

1. **Near-miss prefixes.** System prompt differs by a single token (session ID, timestamp). Hash mismatch causes a full cache miss. No error, no warning, just silently slower.
2. **`prompt_logprobs` bypass.** Requests with `prompt_logprobs` set skip the cache entirely. The model needs to run a forward pass on every prompt token to produce logprobs, so cached KV blocks can't help. Only affects those specific requests, not other requests sharing the same prefix.
3. **Non-deterministic prefix construction.** Any upstream code injecting variable data into a "fixed" system prompt (timestamps, request IDs, A/B variants) destroys hit rate without producing visible errors.
4. **Partial block waste.** A 500-token system prompt caches 496 tokens. The last 4 recompute every time. For short prefixes, this overhead is non-negligible relative to the benefit.

### Interview framing

"Prefix caching eliminates redundant prefill computation for shared prefixes using hash-based block matching. It is a no-quality-loss optimization. In prefill-heavy workloads, it frees both compute and V1 scheduler token budget, improving TTFT and potentially effective throughput. In decode-dominated workloads, the benefit is primarily TTFT. It does not reduce decode cost. Production value depends on cache hit rate, block alignment, and deterministic prompt construction."

---

## Steps 2-4: Experiment Results

### Gotcha: vLLM V1 enables prefix caching by default

Our first run used `--enable-prefix-caching` to turn it ON, but omitted the flag to turn it OFF. Both runs showed identical results because **vLLM V1 defaults to `enable_prefix_caching=True`**. The correct flag to disable it is `--no-enable-prefix-caching`. This itself is an important operational detail: if you think caching is off but haven't explicitly disabled it, you're benchmarking cache-vs-cache.

### Experiment setup

```
vLLM 0.17.1, Qwen2.5-3B-Instruct, FP16, T4 GPU
Concurrency=8, max-num-seqs=32, output_tokens=64, 40 measured requests + 8 warmup per condition
Fresh vLLM restart between every syslen to prevent cross-syslen cache contamination
Unique variant IDs per hit-ratio condition to prevent cross-condition cache contamination
Prompts shuffled within each condition to avoid ordering artifacts
Run 1: --no-enable-prefix-caching  (4 vLLM restarts, one per syslen)
Run 2: --enable-prefix-caching     (4 vLLM restarts, one per syslen)
```

### The bug that kept coming back: cache contamination

This experiment went through four iterations, all caused by the same fundamental mistake in different forms: **allowing cached KV blocks from one experimental condition to leak into another**.

vLLM's prefix cache is hash-based and persistent within a process. Any prompt that runs once gets its KV blocks cached. If a later condition uses the same prompt text (even if that condition is supposed to measure a different hit ratio), the "miss" request silently becomes a hit. There's no error, no warning. The data just looks too good.

**Iteration 1:** Omitted `--no-enable-prefix-caching` for the cache-OFF run. vLLM V1 defaults to `enable_prefix_caching=True`, so both runs had caching enabled. Symptom: cache OFF and ON produced identical numbers. Fix: use `--no-enable-prefix-caching` explicitly.

**Iteration 2:** Ran all syslens sequentially in the same vLLM process. The syslen=500 canonical prompt ("You are a highly capable assistant...") shared the same first N blocks as the syslen=1000 canonical prompt. Cached blocks from syslen=500 leaked into syslen=1000's 0%-hit condition. Symptom: 0%-hit at larger syslens appeared faster than cache-OFF baseline. Fix: restart vLLM between every syslen.

**Iteration 3:** Hit-ratio conditions ran sequentially within the same vLLM process, and variant IDs overlapped between conditions. At 0% hit, the script created variants 1-40. At 25% hit, the "unique" prompts were variants 11-40, which were byte-for-byte identical to prompts already cached from the 0% run. So at 25% "hit ratio," all 40 requests were actually cache hits (10 canonical + 30 from the previous condition's cache). Symptom: a step function where 25% hit looked identical to 100% hit, with tight p99 (~101ms) proving that every single request got a cache hit. This was initially misattributed to a "prefill contention relief" scheduling effect. Fix: offset variant IDs per condition (`variant_base = condition_idx * num_requests`) so no two conditions ever generate the same prompt text.

The meta-lesson: in any benchmark that sweeps conditions sequentially in a shared process, you must ensure **complete isolation of cached state between conditions**. Hash-based caches make this especially treacherous because contamination is silent and the resulting data often looks plausible enough to explain away.

### Final clean results (unique variants per condition, fresh vLLM per syslen)

Cache OFF (`--no-enable-prefix-caching`):

```
| SysLen | HitRatio | TTFT p50 | TTFT p99  | E2E p99   | tok/s |
|--------|----------|----------|-----------|-----------|-------|
|    100 |       0% |    544ms |     551ms |   2403ms  | 216.8 |
|    100 |      25% |    546ms |     548ms |   2409ms  | 216.0 |
|    100 |      50% |    552ms |     556ms |   2422ms  | 215.2 |
|    100 |      75% |    554ms |     560ms |   2426ms  | 214.7 |
|    100 |     100% |    556ms |     564ms |   2433ms  | 214.2 |
|    500 |       0% |   1207ms |    2353ms |   4805ms  | 122.9 |
|    500 |      25% |   1227ms |    2409ms |   4872ms  | 121.4 |
|    500 |      50% |   1239ms |    2474ms |   4951ms  | 120.3 |
|    500 |      75% |   1247ms |    2490ms |   4969ms  | 120.0 |
|    500 |     100% |   1242ms |    2495ms |   4976ms  | 119.9 |
|   1000 |       0% |   2193ms |    4861ms |   9498ms  |  76.8 |
|   1000 |      25% |   2243ms |    4986ms |   9693ms  |  75.8 |
|   1000 |      50% |   2229ms |    4944ms |   9628ms  |  76.2 |
|   1000 |      75% |   2238ms |    4956ms |   9649ms  |  76.1 |
|   1000 |     100% |   2238ms |    4958ms |   9660ms  |  76.1 |
|   2000 |       0% |   2915ms |   10056ms |  19290ms  |  43.2 |
|   2000 |      25% |   2915ms |   10201ms |  19505ms  |  43.0 |
|   2000 |      50% |   2932ms |   10278ms |  19648ms  |  42.8 |
|   2000 |      75% |   2933ms |   10317ms |  19693ms  |  42.7 |
|   2000 |     100% |   2935ms |   10311ms |  19682ms  |  42.7 |
```

Cache ON (`--enable-prefix-caching`):

```
| SysLen | HitRatio | TTFT p50 | TTFT p99  | E2E p99   | tok/s |
|--------|----------|----------|-----------|-----------|-------|
|    100 |       0% |    554ms |     564ms |   2428ms  | 215.0 |
|    100 |      25% |    467ms |     568ms |   2435ms  | 224.5 |
|    100 |      50% |    268ms |     474ms |   2340ms  | 237.2 |
|    100 |      75% |    197ms |     410ms |   2279ms  | 250.9 |
|    100 |     100% |     93ms |      97ms |   1960ms  | 266.0 |
|    500 |       0% |   1215ms |    2400ms |   4851ms  | 122.5 |
|    500 |      25% |    924ms |    2145ms |   4594ms  | 139.4 |
|    500 |      50% |    691ms |    1325ms |   3509ms  | 163.6 |
|    500 |      75% |    409ms |    1015ms |   3136ms  | 201.3 |
|    500 |     100% |    127ms |     134ms |   2046ms  | 255.3 |
|   1000 |       0% |   2192ms |    4903ms |   9530ms  |  76.8 |
|   1000 |      25% |   1758ms |    4387ms |   8611ms  |  91.9 |
|   1000 |      50% |   1307ms |    2513ms |   6131ms  | 116.4 |
|   1000 |      75% |    730ms |    1893ms |   4390ms  | 158.1 |
|   1000 |     100% |    124ms |     154ms |   2116ms  | 247.6 |
|   2000 |       0% |   2899ms |    9953ms |  19130ms  |  43.4 |
|   2000 |      25% |   2597ms |    8908ms |  15897ms  |  54.0 |
|   2000 |      50% |   1828ms |    5144ms |   8870ms  |  72.4 |
|   2000 |      75% |    814ms |    3372ms |   7171ms  | 109.6 |
|   2000 |     100% |    237ms |     239ms |   2326ms  | 225.0 |
```

### Side-by-side comparison (cache OFF vs ON)

```
| SysLen | HitRatio | TTFT p50 OFF | TTFT p50 ON | Speedup | tok/s OFF | tok/s ON |
|--------|----------|-------------|-------------|---------|-----------|----------|
|    100 |       0% |       544ms |       554ms |   0.98x |     216.8 |    215.0 |
|    100 |      50% |       552ms |       268ms |   2.06x |     215.2 |    237.2 |
|    100 |     100% |       556ms |        93ms |   5.98x |     214.2 |    266.0 |
|    500 |       0% |      1207ms |      1215ms |   0.99x |     122.9 |    122.5 |
|    500 |      50% |      1239ms |       691ms |   1.79x |     120.3 |    163.6 |
|    500 |     100% |      1242ms |       127ms |   9.78x |     119.9 |    255.3 |
|   1000 |       0% |      2193ms |      2192ms |   1.00x |      76.8 |     76.8 |
|   1000 |      50% |      2229ms |      1307ms |   1.71x |      76.2 |    116.4 |
|   1000 |     100% |      2238ms |       124ms |  18.05x |      76.1 |    247.6 |
|   2000 |       0% |      2915ms |      2899ms |   1.01x |      43.2 |     43.4 |
|   2000 |      50% |      2932ms |      1828ms |   1.60x |      42.8 |     72.4 |
|   2000 |     100% |      2935ms |       237ms |  12.38x |      42.7 |    225.0 |
```

### Key findings

**1. Prefix caching benefit scales smoothly with hit ratio.**

With correctly isolated conditions, the hit-ratio sweep shows a clean gradient, not a step function. At syslen=1000: 0% hit = 2192ms, 25% = 1758ms, 50% = 1307ms, 75% = 730ms, 100% = 124ms. Each increment of cache hit ratio delivers additional TTFT reduction and throughput improvement. This is the expected behavior: the p50 TTFT reflects a mix of cache-hit requests (fast) and cache-miss requests (slow), shifting smoothly as the ratio changes.

**2. At 100% hit, TTFT speedups are dramatic: 6x to 18x.**

At syslen=1000, TTFT drops from 2238ms (cache OFF) to 124ms (cache ON, 100% hit), an 18x improvement. Throughput jumps from 76 tok/s to 248 tok/s, a 3.3x improvement. The benefit grows with system prompt length because longer prompts mean more prefill work to skip.

**3. At 50% hit, the benefit is moderate but real.**

At syslen=1000, TTFT drops from 2229ms to 1307ms (1.7x) and throughput rises from 76 to 116 tok/s (1.5x). The p50 is pulled toward the miss requests' latency because half the batch still needs full prefill. The p99 reflects the slowest requests, which are always the misses.

**4. Zero overhead at 0% hit rate.**

Cache OFF and cache ON at 0% hit are indistinguishable across all syslens (within 1%). The hash lookup and block management add negligible cost. This confirms vLLM V1's decision to enable prefix caching by default is sound.

**5. Cache OFF is perfectly flat across hit ratios, as expected.**

Hit ratio should not affect TTFT when caching is disabled, because every request gets full prefill regardless. The data confirms this: cache-OFF TTFT varies by less than 2% across hit ratios at every syslen. This serves as a sanity check that the experiment is measuring what it claims to measure.

**6. Longer system prompts amplify the benefit at every hit ratio.**

At 100% hit: 6.0x speedup at syslen=100, 9.8x at 500, 18.1x at 1000, 12.4x at 2000. The slight drop at syslen=2000 likely reflects the remaining non-cached tokens (partial block, user message, template) becoming a larger fixed cost, plus decode overhead being constant regardless of prefix length.

### Experiment design lessons

1. Always verify vLLM config in logs. Check `enable_prefix_caching=` in the engine init line.
2. Use `--no-enable-prefix-caching` to explicitly disable, not just omit the flag.
3. Restart vLLM between conditions that could share cached blocks. Running multiple syslens in the same process is a contamination risk.
4. **Give each experimental condition its own namespace for unique prompts.** If conditions A and B both generate "variant 15," that's the same prompt text, and condition B inherits condition A's cached KV blocks. This was our most insidious bug: it produced a plausible-looking step function that we spent hours trying to explain with scheduling theory before realizing it was just data leakage.
5. If cache-OFF shows variation across hit ratios, something is wrong with the experimental setup, not with caching theory. Cache-OFF should be flat.
6. If your hit-ratio sweep shows a step function (25% looks identical to 100%), suspect cache contamination before reaching for exotic explanations.

---

## Step 6: FP8 KV Cache Capacity Modeling

T4 doesn't support FP8 (needs Hopper SM90 or Ada SM89). This is a theoretical exercise.

### Part A: KV bytes per token from first principles

Qwen2.5-3B uses GQA with `num_kv_heads=2` (not 16).

```
FP16 KV per token per layer:
  2 (K+V) x 2 KV heads x 128 head_dim x 2 bytes = 1,024 bytes

FP16 KV per token all layers:
  1,024 x 36 layers = 36,864 bytes = 36 KiB/token

FP8 KV per token all layers:
  512 x 36 = 18,432 bytes = 18 KiB/token
```

Without GQA correction (num_kv_heads=16), you get ~288 KiB/token. Off by 8x. Those familiar with modern LLM architecture (Llama 3, Mistral, Qwen2.5, Gemma 2 all use GQA) would catch this immediately.

### Capacity estimate

Assuming ~8.5 GB available for KV cache on T4:

```
FP16: 8.5E9 / 36,864 bytes = ~230K tokens
  At 2048 tokens/seq: ~112 concurrent sequences

FP8:  8.5E9 / 18,432 bytes = ~461K tokens
  At 2048 tokens/seq: ~224 concurrent sequences (2x FP16)
```

### Part B: Six assumptions that break this in production

1. **Runtime overhead.** CUDA context, activation buffers, sampler workspace all eat into the memory we assumed was available for KV.
2. **Compute saturation / max_num_seqs.** Even with KV capacity for 112 sequences, compute or the `--max-num-seqs` cap binds first. Saw this empirically on Day 6.
3. **Block fragmentation.** PagedAttention uses blocks of 16 tokens. Partially-filled final blocks waste memory. Many short or variable-length sequences amplify this.
4. **Mixed sequence lengths.** Calculator assumes uniform 2048-token sequences. Real traffic has a distribution. Peak memory is driven by the longest active sequences, not the average.
5. **gpu_memory_utilization.** vLLM constrains KV pool to this fraction of VRAM (default 0.9, used 0.85 on T4). Further reduces available memory below the naive estimate.
6. **Scheduler headroom.** Admission control and preemption logic reserve margin beyond raw KV capacity.

### Part C: FP8 backend and calibration nuance

Two execution paths for FP8 KV, with different implications:

**Standard backend:** K/V stored as FP8, dequantized to FP16 before the attention matmul. Compute is still FP16, FLOPs unchanged. The win is purely memory capacity (2x KV tokens).

**Flash Attention 3 (H100):** Attention runs natively in FP8. Queries also quantized to FP8, dot products happen in FP8 domain. This gives 2x compute throughput on H100 tensor cores in addition to the memory savings. But accuracy impact is different because rounding error accumulates in the attention scores themselves, not just in stored values.

Calibration also matters. vLLM supports no calibration, random-token calibration, and dataset-based calibration. Quality impact varies by method.

FP8 KV cache is a capacity lever that roughly doubles KV token capacity, enabling longer contexts or more concurrent sequences. On H100 with FA3, it also provides a compute speedup. The quality tradeoff depends on which backend is active and what calibration method is used. I would not ship it without empirical validation on the target workload's traffic distribution.

---

## Step 7: Master Tradeoff Table (Week 3, After Day 13)

All numbers are empirical from T4 (g4dn.xlarge, 16 GiB) unless marked.

### Weight precision (Days 11-13)

```
Metric                                     FP16        INT8-AWQ     INT4-GPTQ
Model size (GiB)                           5.79        3.23         1.94
Free memory for KV (GiB)                   6.73        9.29         10.58
Max concurrent @ 4K, FP16 KV               47          66           75
Max concurrent @ 4K, FP8 KV (theoretical)  94          132          150
Throughput decode-heavy conc=8 (tok/s)     264.90      355.12       480.86
Throughput prefill-heavy conc=8 (tok/s)    222.49      128.41*      343.21
TTFT p99 decode-heavy conc=8 (ms)          101.1       95.3         114.8
ITL p99 decode-heavy conc=8 (ms)           31.2        23.6         17.5
$/M tokens conc=8 (on-demand)              $0.552      $0.411       $0.304
Perplexity (WikiText-2)                    11.66       11.68        12.57
HellaSwag acc_norm (0-shot, 25%)           64.20%      64.20%       63.36%
Qualitative failures                       baseline    0/10         5/10
```

*INT8-AWQ prefill-heavy is anomalous (compressed-tensors kernel issue on T4). Not representative.

FP8 KV theoretical row = 2x the FP16 KV row. FP8 halves KV bytes/token (18 KiB vs 36 KiB).

### Prefix caching (Day 13, FP16 baseline, conc=8)

TTFT p50, cache OFF (Cond A) vs cache ON at 100% hit (Cond F):

```
SysLen   TTFT p50 OFF   TTFT p50 ON   Speedup
100T     556ms          93ms          5.98x
500T     1242ms         127ms         9.78x
1000T    2238ms         124ms         18.05x
2000T    2935ms         237ms         12.38x
```

Inflection: speedup is significant (>20% TTFT reduction) at all tested syslens even at 50% hit ratio. Speedup grows with syslen up to 1000T, then drops at 2000T because uncached tokens (partial block + user message + template) pay O(seq_len) attention against all cached entries.

### Hit ratio sweep (FP16, syslen=1000, conc=8)

```
Condition   Hit Ratio    TTFT p50    TTFT p99    tok/s
A           N/A (OFF)    2193ms      4861ms      76.8
B           0%           2192ms      4903ms      76.8
C           25%          1758ms      4387ms      91.9
D           50%          1307ms      2513ms      116.4
E           75%          730ms       1893ms      158.1
F           100%         124ms       154ms       247.6
```

A vs B: indistinguishable. Prefix caching has zero overhead at 0% hit. Validates V1's decision to default it on.

Benefit becomes significant (>20% TTFT reduction) at 25% hit ratio (1758ms vs 2193ms = 20% reduction). Scales smoothly from there.

---

## Step 8: Interview Insight Paragraphs

**On prefix caching:**

Prefix caching eliminates redundant prefill computation for shared prefixes using hash-based block matching. It is a no-quality-loss optimization. Production value depends on cache hit rate, block alignment, and scheduler interactions. With a 1000-token system prompt at 50% hit rate, I measured a 2.4s TTFT reduction at p99 (4861ms to 2513ms), a 48% improvement. The optimization is most impactful in chat, agent, and RAG workloads with standardized prompt scaffolding. Key failure modes: non-deterministic prefix construction silently destroys hit rate; `prompt_logprobs` requests bypass the cache entirely in V1.

**On FP8 KV cache:**

FP8 KV cache is a memory-capacity lever. For Qwen2.5-3B with 2 GQA KV heads (~36 KiB/token FP16), theoretical capacity doubles to ~18 KiB/token under FP8. With 6.73 GiB measured available for KV on T4, this would increase theoretical max concurrency at 4K sequences from 47 to 94. Actual achievable concurrency must be measured. Six assumptions break the theoretical model in production: runtime overhead, compute saturation, block fragmentation, mixed sequence lengths, gpu_memory_utilization cap, and scheduler headroom. The quality tradeoff is backend- and calibration-dependent (standard path dequantizes to FP16 for attention; FA3 on H100 runs attention natively in FP8 with different accuracy characteristics). I would not ship FP8 KV cache without workload-specific validation.

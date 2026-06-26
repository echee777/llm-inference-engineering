# Quantization & Optimization Tradeoff Analysis

Portfolio Artifact #3. All data from Qwen2.5-3B-Instruct on T4 (g4dn.xlarge, 16 GiB) unless noted.

---

## 1. Approaches

AWQ identifies the ~1% of weights that matter most for minimizing layer output error (activation-aware). These weights are scaled up before quantization, then scaled back down at runtime. The scaling preserves signal for the critical weights. Naive quantization treats all weights equally and loses more signal at the same bit width.

GPTQ quantizes all weights using second-order activation-aware error minimization, layer by layer. It doesn't preferentially scale any subset. More compute-intensive calibration than AWQ, but designed for INT4 where you need every bit of accuracy recovery.

FP8 KV cache is separate from weight quantization. It compresses KV cache values from FP16 to FP8, halving memory per cached token. Does not touch model weights. Requires Hopper (H100) or later. AWQ/GPTQ and FP8 KV are orthogonal and can be combined. Both dequantize at compute time.

---

## 2. Performance by Workload Type

Full 24-cell benchmark matrix: 3 precisions x 2 workloads x 4 concurrency levels. 64 prompts per cell. Decode-heavy: 64-token input, 512-token output. Prefill-heavy: 1900-token input, 32-token output.

```
Precision    Workload        Conc   Throughput (tok/s)   TTFT p99 (ms)   ITL p99 (ms)
FP16         Prefill-heavy   1      24.6                 1587.0          N/A
FP16         Prefill-heavy   4      112.1                5506.0          N/A
FP16         Prefill-heavy   8      222.5                7291.0          N/A
FP16         Prefill-heavy   16     279.3                10541.0         N/A
FP16         Decode-heavy    1      37.0                 50.0            27.5
FP16         Decode-heavy    4      146.4                57.0            30.6
FP16         Decode-heavy    8      264.9                101.1           31.2
FP16         Decode-heavy    16     495.1                216.0           32.2
INT8-AWQ     Prefill-heavy   1      31.1                 1257.0          N/A
INT8-AWQ     Prefill-heavy   4      199.3                3239.0          N/A
INT8-AWQ     Prefill-heavy   8      128.4*               9095.0*         N/A
INT8-AWQ     Prefill-heavy   16     330.0                8040.0          N/A
INT8-AWQ     Decode-heavy    1      62.8                 29.0            16.2
INT8-AWQ     Decode-heavy    4      218.6                37.0            19.8
INT8-AWQ     Decode-heavy    8      355.1                95.3            23.6
INT8-AWQ     Decode-heavy    16     639.5                193.0           24.7
INT4-GPTQ    Prefill-heavy   1      39.5                 990.0           N/A
INT4-GPTQ    Prefill-heavy   4      161.9                3808.0          N/A
INT4-GPTQ    Prefill-heavy   8      343.2                3581.0          N/A
INT4-GPTQ    Prefill-heavy   16     433.6                6155.0          N/A
INT4-GPTQ    Decode-heavy    1      90.1                 22.0            11.5
INT4-GPTQ    Decode-heavy    4      297.2                27.0            14.7
INT4-GPTQ    Decode-heavy    8      480.9                114.8           17.5
INT4-GPTQ    Decode-heavy    16     600.4                116.0           18.5
```

*INT8-AWQ prefill-heavy conc=8 is anomalous (compressed-tensors kernel issue on T4). Not representative.

INT8-AWQ improves decode throughput by 70% over FP16 (at conc=1: 62.8 vs 37.0 tok/s) but prefill throughput by only 26% (31.1 vs 24.6 tok/s). Decode is memory-bandwidth-bound (smaller weight tensors = faster HBM reads per token). Prefill is compute-dominated under these workload parameters (matmul throughput does not improve proportionally with INT8 quantization on T4, consistent with limited INT8 Tensor Core utilization on this hardware, though this is inferred from the throughput asymmetry rather than directly profiled at the kernel level).

---

## 3. Quality

```
Metric                                  FP16      INT8-AWQ    INT4-GPTQ
Perplexity (WikiText-2, lower=better)   11.66     11.68       12.57
HellaSwag acc_norm (0-shot, 25%)        64.20%    64.20%      63.36%
Qualitative failures                    baseline  0/10        5/10
```

INT8-AWQ: +0.17% perplexity, zero accuracy drop, zero qualitative failures. Indistinguishable from FP16 across all 10 test prompts. Not "close", literally the same content with trivial wording differences.

INT4-GPTQ: +7.8% perplexity, -1.31% accuracy drop, 5/10 qualitative failures. The failures were not subtle:

- Creative writing (2/2 failed): severe repetition loops, repeated stanzas verbatim
- Factual Q&A (1/2 failed): divergent hallucination, different wrong answer path than FP16
- Code generation (1/2 failed): syntax error in test harness
- Long-context (1/2 failed): mischaracterized the source question
- Multi-step reasoning (0/2 failed): survived intact

The pattern: INT4 holds up on tasks with constrained token distributions (reasoning, where each step has high conditional probability) but breaks on tasks with flatter distributions (creative writing, open-ended generation). The model is confidently generating the wrong tokens, not uncertain ones. Perplexity catches the aggregate distribution shift. Task accuracy only catches shifts that cross a decision boundary. You need both.

---

## 4. Capacity Impact

```
Precision    Model Size (GiB)   Free for KV (GiB)   Max Conc @ 2K   Max Conc @ 4K   Max Conc @ 8K
FP16         5.79               6.73                 95              47              23
INT8-AWQ     3.23               9.29                 132             66              33
INT4-GPTQ    1.94               10.58                150             75              37
```

KV bytes per token: 2 (K+V) x 36 layers x 2 kv_heads x 128 head_dim x 2 bytes (FP16) = 36 KiB. KV cache is always FP16 regardless of weight quantization.

INT8-AWQ reduces model size by 2.56 GiB (44% reduction), freeing that for KV cache. At 4K sequence length, max concurrent requests goes from 47 to 66, a 40% capacity improvement. For throughput-maximizing deployments, this capacity gain is more operationally significant than the per-request latency reduction.

---

## 5. Cost Model ($/Million Tokens)

Instance: g4dn.xlarge, on-demand $0.526/hr. Using decode-heavy throughput at conc=8 as the reference operating point.

```
Precision    Throughput @ conc=8 (tok/s)   Tokens/hr     $/M tokens
FP16         264.90                         953,640       $0.552
INT8-AWQ     355.12                         1,278,432     $0.411
INT4-GPTQ    480.86                         1,731,096     $0.304
```

INT8-AWQ reduces serving cost from $0.552/M tokens to $0.411/M tokens, a 25.4% reduction, with +0.17% perplexity increase. INT4-GPTQ reduces cost to $0.304/M tokens, a 44.9% reduction, with +7.8% perplexity increase and qualitative degradation on creative writing, code, and factual Q&A.

---

## 6. Prefix Caching

From Day 13 experiments. FP16 baseline, conc=8. Cache OFF vs cache ON at 100% hit rate.

```
SysLen    TTFT p50 OFF   TTFT p50 ON   Speedup
100T      556ms          93ms          5.98x
500T      1242ms         127ms         9.78x
1000T     2238ms         124ms         18.05x
2000T     2935ms         237ms         12.38x
```

Speedup grows with system prompt length up to 1000T (18x), then drops at 2000T (12.4x). At 2000T, uncached tokens still pay O(sequence_length) attention against all cached KV entries, and that cross-attention cost starts to dominate.

At 0% hit rate, prefix caching has zero overhead. TTFT was indistinguishable from cache-off (2192ms vs 2193ms at syslen=1000). This validates V1's decision to default it on.

Benefit becomes significant (>20% TTFT reduction) at 25% hit ratio (1758ms vs 2193ms at syslen=1000).

When it matters: chatbots with shared system prompts, RAG pipelines with common document prefixes, agent frameworks with standardized scaffolding. When it doesn't: per-user custom prompts with low reuse.

---

## 7. Speculative Decoding

From Day 14 experiments. Tested both n-gram (prompt pattern matching) and draft model (Qwen2.5-0.5B-Instruct) paths.

Compatibility check: TinyLlama (vocab 32,000) failed hard against Qwen (vocab 151,936). vLLM rejected at startup. Qwen2.5-0.5B shares the same tokenizer and launched successfully.

```
Scenario                       Speedup   Acceptance Rate   VRAM Cost     Use Spec Decode?
Q&A, ngram, conc=1             0.92x     56.7%             ~0 MB         No
Q&A, ngram, conc=8             0.86x     56.7%             ~0 MB         No
Creative writing, ngram, c=1   0.93x     4.2%              ~0 MB         No
Code generation, ngram, c=1    0.93x     24.7%             ~0 MB         No
Q&A, draft model, conc=1       0.40x     39.3%             +0.94 GiB     No
```

Every configuration was net negative. N-gram was 7-8% slower than baseline across all tasks. The draft model path was worse: throughput collapsed to 15.1 tok/s (less than half of baseline 37.5) despite higher acceptance (39.3%). The 3B target is only 6x larger than the 0.5B draft, so draft forward passes are a significant fraction of total cost. On hardware with a 70B target and 1B draft (70x ratio), draft overhead would be negligible.

Speedup degrades with concurrency (0.92x at conc=1 to 0.86x at conc=8). Continuous batching already raises GPU utilization during decode, closing the gap that spec decode was designed to fill.

Spec decode is a low-concurrency, memory-bound-regime optimization for larger models on more powerful hardware. On T4 with a 3B model, individual decode steps are already fast (~27ms ITL), leaving no room for spec decode to improve.

---

## 8. Recommendations

Every recommendation names its target SLO. The four axes are: latency (p99 TTFT/ITL), throughput (tok/s), cost ($/M tokens), and quality floor (perplexity ceiling, task accuracy minimum).

### By use case

```
Use Case                            Recommendation              Key Rationale
Latency-sensitive chat              INT8-AWQ + prefix caching   Capacity gain (S4) without quality hit; prefix caching
                                                                cuts TTFT on repeat system prompts (S6)
Throughput-maximizing batch          INT4-GPTQ                  Lowest $/M tokens (S5), highest concurrency ceiling
                                                                (S4); acceptable quality for non-critical tasks only
Quality-critical (code, med, legal) FP16                        INT4 qualitative degradation unacceptable (S3); INT8 is
                                                                a judgment call depending on task
```

### By fleet type

T4-class cost-constrained fleet (my hardware): INT8-AWQ is the default. 40% more concurrent capacity (S4) at 25.4% lower $/M tokens (S5) with 0.17% perplexity increase (S3). Prefix caching on by default for any deployment with repeated system prompts. Spec decode off.

H100-class latency-sensitive fleet (extrapolated, not measured): FP16 or BF16 as baseline. Higher HBM bandwidth (3.35 TB/s vs 320 GB/s) reduces the urgency of weight quantization for decode throughput, but INT8-AWQ still helps for capacity and cost. FP8 KV cache becomes the primary new capacity lever (S9, theoretical). Spec decode viable at low concurrency with appropriate draft model sizing (target-to-draft ratio needs to be large).

### Operational simplicity

INT8-AWQ over INT4-GPTQ not just on quality grounds but because it requires less validation surface. Fewer edge cases, lower risk of silent quality regressions, simpler rollback.

Prefix caching is high-upside, low-risk. No new model variant, no calibration, trivially reversible. Enable by default for repeated-prefix workloads.

Spec decode adds a draft model dependency, makes latency behavior harder to predict under load, and complicates capacity planning. Acceptance rate is workload-sensitive so speedup is not stable across traffic changes. Not worth it unless the latency win is critical and the workload is predictable.

### Non-recommendations

Do not deploy INT4-GPTQ for quality-critical workloads. The +7.8% perplexity and 5/10 qualitative failure rate are unacceptable for code generation, medical summarization, or multi-step reasoning where creative/fluency outputs matter. Cost savings do not justify shipping degraded outputs.

Do not deploy speculative decoding at high concurrency (conc >= 8 in my experiments). Draft overhead competes with live serving. At the concurrency levels where throughput matters most, spec decode hurts.

Do not treat prefix caching wins as universal. TTFT reduction is real only at high hit rates. For deployments with diverse system prompts, the cache provides no benefit.

Do not apply T4 throughput numbers directly to A100/H100 fleet sizing. See S9.

### What I would ship Monday morning

Given a T4-class fleet, general-purpose chat workload, moderate quality requirements:

INT8-AWQ + prefix caching on + spec decode off.

INT8-AWQ gives 40% more concurrent capacity than FP16 (S4) at 25.4% lower $/M tokens (S5) with +0.17% perplexity (S3). Quality cost is within bounds for general-purpose chat. Prefix caching cuts TTFT on system-prompt-heavy traffic by up to 18x (S6). Spec decode stays off because it was net negative at all concurrency levels and task types on this hardware (S7).

---

## 9. Generalizability Limits

```
Finding                                       Status                      Likely change on H100
INT8-AWQ decode throughput gain (1.7x)        Measured on T4              Expected but not measured. Decode stays
                                                                          bandwidth-bound (AI~1 vs ridge ~295).
                                                                          Relative gain should hold. Absolute
                                                                          TPOT drops from ~27ms to ~3ms.
INT8 prefill gain smaller than decode gain    Measured on T4              Pattern holds. Prefill is compute-dominated
                                                                          under these workload parameters.
Capacity expansion from INT8 (KV freed)       Measured on T4              Math holds exactly. Weight size reduction
                                                                          is hardware-independent.
$/M tokens improvement from INT8              Measured on T4              Direction holds. Magnitude changes because
                                                                          hourly rates and throughput differ.
FP8 KV cache (~2x capacity)                  Theoretical, not measured   Measurable on H100. 50% KV memory
                                                                          reduction = ~2x concurrency ceiling.
                                                                          With FA3: also 2x compute throughput
                                                                          on FP8 native matmul.
Prefix caching TTFT reduction                Measured on T4              Pattern holds on all hardware. Absolute
                                                                          ms values differ.
Spec decode hurts at high concurrency        Measured on T4              Pattern holds. But "high concurrency"
                                                                          threshold shifts. With 70B target and
                                                                          1B draft (70x ratio), draft overhead
                                                                          becomes negligible. Likely viable at
                                                                          low concurrency on H100.
INT4 qualitative degradation                 Measured on T4              Hardware-independent. Quantization error
                                                                          is a weight property, not a GPU property.
```

What I would measure first on H100:

1. INT8-AWQ decode gain: does the ~1.7x still hold with 11x more bandwidth?
2. FP8 KV cache: does the theoretical 2x capacity actually deliver?
3. Spec decode: at what concurrency does it flip from benefit to harm with a real 70x-ratio draft model?

Falsification condition: if INT8-AWQ decode gain on H100 is significantly below 1.7x, the throughput justification weakens. FP8 KV cache delivers the capacity benefit without touching weights, so FP16 + FP8 KV becomes the simpler path with less validation surface. That said, decode stays bandwidth-bound on all current hardware (AI~1, ridge point ~200-300), so the gain is unlikely to drop much in practice.

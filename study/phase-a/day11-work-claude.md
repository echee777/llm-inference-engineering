# Day 11

## Quantization - self-quantization and downloads

Self-quantized Qwen2.5-3B-Instruct to INT8-AWQ using llm-compressor (autoawq is deprecated).

- Used `oneshot()` with `AWQModifier(scheme="W8A16")`, 256 calibration samples from ultrachat_200k
- Took ~2.25 hours (8085s) for 36 layers on T4. Each layer does calibrate -> grid search smoothing (~3.5 min) -> propagate.
- Saved to `./qwen2.5-3b-int8-awq`

Downloaded pre-quantized GPTQ-Int4 from Qwen org: `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`

Note on loading: the AWQ model saved by llm-compressor uses `compressed-tensors` format internally. Don't pass `quantization='awq'` - vLLM auto-detects it. GPTQ still needs `quantization='gptq'`.

## VRAM footprint

All loaded with `max_num_seqs=32, max_model_len=2048`:

```
Metric                   FP16        INT8-AWQ     INT4-GPTQ
-----------------------  ----------  -----------  -----------
Model weights            5.79 GiB    3.23 GiB     1.95 GiB
Free for KV cache        6.64 GiB    9.2 GiB      10.51 GiB
KV cache tokens          193,360     268,000      306,160
Max concurrency @ 2048   94x         131x         149x
nvidia-smi total         ~14.0 GB    ~13.7 GB     ~13.9 GB
Load time                46.1s       25.0s        14.8s
```

nvidia-smi totals are all ~14GB because vLLM backfills freed weight VRAM with more KV cache blocks. The savings show up as more blocks, not less total VRAM.

Max concurrency = total KV tokens / max_model_len. This is worst case assuming every request fills 2048 tokens.

Per-token KV cache size: 2 (K+V) x 2 bytes (FP16) x 36 layers x 2 kv_heads x 128 dim = 36KB.
KV cache is always FP16 regardless of weight quantization - weight-only quantization doesn't touch KV.

## Benchmark setup

Ran 24-cell matrix: 3 precisions x 2 workloads x 4 concurrency levels.

`vllm bench serve` is a client-side load generator. It sends HTTP requests to a running vLLM server, streams the response via SSE, and records timestamps for each token. TTFT, TPOT, ITL are all measured client-side - they reflect what a real caller would see.

The two flags that control workload shape are `--random-input-len` and `--random-output-len`. There's no explicit "prefill mode" - the input/output ratio determines which phase dominates. The benchmark generates random token IDs of the specified input length, wraps them in the Qwen chat template (adds ~20-25 tokens overhead), and sets `max_tokens` in the request body to the output length.

Workloads:

- prefill-heavy: `--random-input-len 1900 --random-output-len 32`. Server spends nearly all time on prefill - processing 1900 tokens through attention in parallel. Only 32 decode steps after. Tests compute-bound regime (high arithmetic intensity).
- decode-heavy: `--random-input-len 64 --random-output-len 512`. Prefill is negligible (64 tokens). Server spends nearly all time decoding - 512 sequential steps, each reading all weights from HBM to produce one token. Tests bandwidth-bound regime (AI ~1).

Input was set to 1900 not 2048 because chat template overhead pushes total tokens past max_model_len=2048. First run at 2048 failed with "Bad Request" on all 64 prompts.

Concurrency: 1, 4, 8, 16. 64 prompts per cell.

Per precision, started the server in one terminal, ran `python benchmark_matrix.py --precision fp16`, killed the server, restarted with the next model, repeated. `benchmark_matrix.py` loops through both workloads x 4 concurrency levels = 8 `vllm bench serve` calls per precision.

## Benchmark results - decode-heavy

```
Precision    c=1 TPOT (ms)   c=1 tok/s   c=16 tok/s   Speedup vs FP16 (c=1)
-----------  -------------   ---------   ----------   ---------------------
FP16         27.0            37.0        495.1        -
INT8-AWQ     15.9            62.8        639.5        1.70x
INT4-GPTQ    11.1            90.1        600.4        2.44x
```

TPOT tracks weight size ratio: INT8 ~1.7x faster (weights ~1.8x smaller), INT4 ~2.4x faster (weights ~3x smaller). Decode is bandwidth-bound, fewer bytes = proportionally faster.

TPOT barely degrades with concurrency (27ms -> 32ms from c=1 to c=16 for FP16). Weights are read once per step regardless of batch size - batching amortizes the weight read across more tokens. KV cache reads add minimal bandwidth at these concurrency levels (~3% of total traffic).

## Benchmark results - prefill-heavy

```
Precision    c=1 tok/s   Speedup vs FP16 (c=1)
-----------  ---------   ---------------------
FP16         24.6        -
INT8-AWQ     31.1        1.26x
INT4-GPTQ    39.5        1.60x
```

Smaller speedup than decode for both. Weight-only quantization dequantizes to FP16 before matmul - FLOPs don't change. The modest gains come from less weight data to load initially.

## Roofline confirmation

decode_speedup > prefill_speedup for both:

- INT8-AWQ: 1.70x decode vs 1.26x prefill
- INT4-GPTQ: 2.44x decode vs 1.60x prefill

This confirms the roofline model. Decode sits at AI ~1 (deep in bandwidth territory, ridge point is 203). Prefill sits at AI ~input_len (closer to compute-bound). Quantization reduces bytes read from HBM - that directly speeds up decode but barely helps prefill since compute is the bottleneck there.

The concurrency sweep didn't add much signal here. Weights dominate bandwidth at all concurrency levels (97% of HBM traffic is weights, 3% is KV cache reads). The prefill-vs-decode split is what shows the roofline effect, not concurrency.

## Anomaly

INT8-AWQ prefill-heavy c=8: 128.4 tok/s (lower than c=4 at 199.3), TTFT p99 = 9095ms. Transient server issue. Did not re-run.

## What I learned

- Quantization's primary production value is concurrency capacity, not speed. INT4-GPTQ gives 149 max concurrent requests vs FP16's 94 - 58% more capacity from the same T4. Speed improvement (2.4x decode) is a secondary benefit.

- AWQ self-quantization is slow and the tooling is messy. AutoAWQ is deprecated, llm-compressor is the replacement. The calibration itself took 2.25 hours for a 3B model on T4. In production you'd just download pre-quantized checkpoints when available.

- The "weight-only" part matters. Both AWQ and GPTQ store weights as INT4/INT8 but dequantize to FP16 for the actual matmul. There's no integer compute happening. The speedup comes entirely from reading less data from HBM, not from faster math. This is why the TPOT improvement tracks the weight size ratio so cleanly.

- vLLM sampler warmup OOM'd at default max_num_seqs=256 on T4. Each seq needs ~304KB for logits (vocab_size=151936 x 2 bytes). 256 seqs = ~78MB just for logits, plus sampling workspace. Had to set max_num_seqs=32.

## Self-Test Gate — Answer From Memory Before Tomorrow

1. Why does weight-only quantization help decode more than prefill? (Connect to arithmetic intensity and roofline.)

- Decode is memory bound as AI is 1 FLOP/Byte because all weights must be read and streamed over the entire model.
- Prefill reads the weights once and then amortizes over all tokens' KV cache computation. AI is far right of the ridge point in Roofline.
- Therefore, weight-only quantization heavily benefits decode more than prefill (even though both derive some benefit) because decode is memory bound and reducing by a factor of 2 or 4 the total amount of weights needing to be read from GPU RAM translates largely into commensurate decode throughput increase.

2. What is the runtime execution path for INT4-GPTQ weights in vLLM? (Hint: dequantize → FP16 matmul, not INT4 matmul.)

- Quantized weights are read as INT4 from GPU RAM into GPU SM
- Then dequantized into FP16 (happens in GPU stream multiprocessors)
- Then used in matrix multiplications.

3. What is the key difference between AWQ and GPTQ's quantization strategy?

- AWQ tries to identify the 1% weights that matter (large enough corresponding activations) then scales them up. After applying a global quantization, the 1% weights' are protected by the scale up (in groups/channels of size 128). At runtime, the quantized INT are dequantized and scaled down. In contrast, GPTQ does not prioritize a subset of weights. It uses second order information to reduce output layer error and quantizes all weights, layer by layer. GPTQ is more compute intensive. AWQ is easy to implement and has become widely used.
- The commonalities are both methods are activation-aware and post-training.

4. If INT8-AWQ gives 2× memory reduction, what is the theoretical maximum decode throughput improvement on a memory-bandwidth-bound system?

- Throughput can at best improve by a factor of 2x insofar as memory bw remains the gating factor. Real sytems are a mix of compute and memory so the actual improvement will not be 2x but lower.

5. You see decode ITL improve by 40% with INT8-AWQ. A colleague says "great, let's use INT4-GPTQ everywhere." What do you ask before agreeing?

- Ask what is the tradeoff increase in perplexity.
- Ask what is the predominant nature of the workload. If prefill it's not going to help much while increasing perplexity.

---

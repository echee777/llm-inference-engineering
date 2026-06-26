# Day 11

## AWQ

Done. ~2.25 hours (8085s) for the 36-layer calibration. The model is saved at ./qwen2.5-3b-int8-awq.

You can verify it loads:

from vllm import LLM
llm = LLM('./qwen2.5-3b-int8-awq', max_num_seqs=32, max_model_len=2048)

[ssm-user@ip-10-99-0-199 ~]$ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
13693

## GPTQ download

> llm = LLM('Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', quantization='gptq', max_num_seqs=32, max_model_len=2048)

[ssm-user@ip-10-99-0-199 ~]$ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
13857

⏺ ~13.9GB out of 16GB used. Here's the breakdown:

```
┌────────────────────────┬────────────────┬────────────────────────────────────────────────────┐
│ Component              │ Estimated Size │ How                                                │
├────────────────────────┼────────────────┼────────────────────────────────────────────────────┤
│ Model weights (INT4)   │ ~1.8GB         │ 3B params × 4 bits / 8 = ~1.5GB + overhead         │
├────────────────────────┼────────────────┼────────────────────────────────────────────────────┤
│ KV cache pool          │ ~10.5GB        │ vLLM fills up to 90% of remaining VRAM with blocks │
├────────────────────────┼────────────────┼────────────────────────────────────────────────────┤
│ CUDA context + kernels │ ~1.0GB         │ Driver, cuBLAS, compiled kernels                   │
├────────────────────────┼────────────────┼────────────────────────────────────────────────────┤
│ Warmup/runtime buffers │ ~0.5GB         │ Logits, sampling tensors, activations              │
├────────────────────────┼────────────────┼────────────────────────────────────────────────────┤
│ Total                  │ ~13.8GB        │                                                    │
└────────────────────────┴────────────────┴────────────────────────────────────────────────────┘
```

The KV cache pool is the dominant consumer. vLLM's logic is roughly:

1. Load model weights → ~1.8GB used
2. Measure remaining free VRAM → ~13.2GB free
3. Multiply by gpu_memory_utilization (0.9) → ~11.9GB budget for KV cache
4. Allocate as many blocks as fit in that budget
5. Remaining ~1.3GB left for warmup + runtime

## KV / concurrency calculation

```
  ┌────────────────────────┬─────────────────┬──────────────────────────┬──────────────────────────┐
  │         Metric         │ FP16 (baseline) │         INT8-AWQ         │        INT4-GPTQ         │
  ├────────────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Model weight memory    │ 5.79 GiB        │ 3.23 GiB (44% reduction) │ 1.95 GiB (66% reduction) │
  ├────────────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Available for KV cache │ 6.64 GiB        │ 9.2 GiB                  │ 10.51 GiB                │
  ├────────────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┤
  │ KV cache tokens        │ 193,360         │ 268,000                  │ 306,160                  │
  ├────────────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Max concurrency @ 2048 │ 94.41x          │ 130.86x                  │ 149.49x                  │
  ├────────────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Load time              │ 46.1s           │ 25.0s                    │ 14.8s                    │
  └────────────────────────┴─────────────────┴──────────────────────────┴──────────────────────────┘
```

The story is clear:

- FP16 → INT8 saves 2.56 GiB of weights, which becomes 74,640 more KV cache tokens (+39%), or ~36 more concurrent requests
- FP16 → INT4 saves 3.84 GiB of weights, which becomes 112,800 more KV cache tokens (+58%), or ~55 more concurrent requests
- INT8 → INT4 saves another 1.28 GiB, gaining ~19 more concurrent requests

On a memory-constrained GPU like the T4, quantization's main value isn't inference speed — it's freeing VRAM for KV cache, which directly pushes out the TTFT cliff you observed on Day 9.

## Note: Max concurrency calculation

FP16: 193,360 tokens / 2,048 = 94.41x
INT8-AWQ: 268,000 tokens / 2,048 = 130.86x
INT4-GPTQ: 306,160 tokens / 2,048 = 149.49x

It's just total_kv_cache_tokens / max_model_len. That's the theoretical max — assumes every request fills the full 2048 tokens. In practice requests vary in length, so actual concurrency will be higher than
this for shorter requests.

# AFTERNOON

Start the servers like so:

```
# FP16
python -m vllm.entrypoints.openai.api_server       --model Qwen/Qwen2.5-3B-Instruct       --max-num-seqs 32 --max-model-len 2048 --port 8000

# INT8 awq
python -m vllm.entrypoints.openai.api_server       --model ./day11/qwen2.5-3b-int8-awq       --quantization awq       --max-num-seqs 32 --max-model-len 2048 --port 8000

# INT4 GPTQ (downloaded snapshot)
python -m vllm.entrypoints.openai.api_server \
  --model /home/ssm-user/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GPTQ-Int4/snapshots/68c4276063d1496cdf13d0b8e8221f899bfa77f7 \
  --quantization gptq \
  --max-num-seqs 32 \
  --max-model-len 2048 \
  --port 8000
```

## Result

Roofline hypothesis confirmed for both quantizations. Here's the analysis:

Decode-heavy (bandwidth-bound regime)

```
  ┌───────────┬───────────────┬────────────────┬─────────────────┐
  │ Precision │ c=1 TPOT (ms) │ c=1 Throughput │ Speedup vs FP16 │
  ├───────────┼───────────────┼────────────────┼─────────────────┤
  │ FP16      │ 27.0          │ 37.0 tok/s     │ baseline        │
  ├───────────┼───────────────┼────────────────┼─────────────────┤
  │ INT8-AWQ  │ 15.9          │ 62.8 tok/s     │ 1.70x           │
  ├───────────┼───────────────┼────────────────┼─────────────────┤
  │ INT4-GPTQ │ 11.1          │ 90.1 tok/s     │ 2.44x           │
  └───────────┴───────────────┴────────────────┴─────────────────┘
```

TPOT tracks the weight size ratio closely: INT8 is ~1.7x faster (weights ~1.8x smaller), INT4 is ~2.4x faster (weights ~3x smaller). This is the roofline prediction — decode is bandwidth-bound, fewer bytes
to read = proportionally faster.

Prefill-heavy (compute-bound regime)

```
  ┌───────────┬────────────────┬─────────────────┐
  │ Precision │ c=1 Throughput │ Speedup vs FP16 │
  ├───────────┼────────────────┼─────────────────┤
  │ FP16      │ 24.6 tok/s     │ baseline        │
  ├───────────┼────────────────┼─────────────────┤
  │ INT8-AWQ  │ 31.1 tok/s     │ 1.26x           │
  ├───────────┼────────────────┼─────────────────┤
  │ INT4-GPTQ │ 39.5 tok/s     │ 1.60x           │
  └───────────┴────────────────┴─────────────────┘
```

Prefill speedup is smaller than decode speedup for both. Prefill is compute-bound — weight-only quantization still dequantizes to FP16 for matmul, so FLOPs don't change. The modest improvement comes from
less weight data to initially load.

Key finding

Decode speedup > prefill speedup for both INT8-AWQ (1.70x vs 1.26x) and INT4-GPTQ (2.44x vs 1.60x). This empirically confirms the roofline model: quantization's primary benefit is in the memory-bandwidth-bound decode phase.

Anomaly

INT8-AWQ prefill-heavy c=8 shows only 128.4 tok/s (lower than c=4 at 199.3) and a TTFT p99 of 9095ms — something went wrong in that run. Likely a transient issue on the server. Worth re-running that single
cell.

## Final artifacts

INT8-AWQ improved decode throughput by 70% vs FP16, but improved prefill throughput by only 26%. This is consistent with decode being memory-bandwidth-bound (fewer bytes read per weight) while prefill is compute-bound. INT4-GPTQ showed 70% additional improvement over INT8-AWQ in decode, at the cost of [quality observations on Day 12].

# gpu-memory-utilization sweep
# Shows: How KV memory budget constrains max RPS (Decode throughput)


((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ python day6-exp2.py
================================================================================
Day 6 — Experiment 2: --gpu-memory-utilization Parameter Sensitivity
Model:          TinyLlama/TinyLlama-1.1B-Chat-v1.0
Sweep values:   [0.5, 0.7, 0.85, 0.95]
max-num-seqs:   256 (fixed — not under test)
Concurrency ramp: [1, 4, 8, 12, 16, 20, 24, 32, 40, 48]
Requests/level: 30
Degradation:    error_rate>10% OR TTFT p99>2.5× baseline
================================================================================

────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.5
────────────────────────────────────────────────────────────

  Starting vLLM: --gpu-memory-utilization 0.5
  Log: vllm_gmem_0p50.log
  Server ready after 64s
  Warming up (2 requests)...
  → concurrency=  1  ✓ clean     TTFT p99=0.09s  throughput=83 tok/s  errors=0/30
  → concurrency=  4  ✗ DEGRADED  TTFT p99=0.31s  throughput=233 tok/s  errors=0/30  [ttft_spike=3.4x baseline (>2.5x)]
  Degradation onset at concurrency=4. Stopping ramp.
  Preemption events detected in vLLM log: 1

  Ramp detail for gpu_memory_utilization=0.50:
  concurrency    Throughput    TTFT p50    TTFT p99    Latency p50    Latency p99    Errors                                       Status
                    (tok/s)         (s)         (s)            (s)            (s)
-------------  ------------  ----------  ----------  -------------  -------------  --------  -------------------------------------------
            1          82.9       0.083       0.09           0.429          2.807      0/30                                        clean
            4         232.9       0.085       0.309          0.734          4.477      0/30  DEGRADED [ttft_spike=3.4x baseline (>2.5x)]
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.7
────────────────────────────────────────────────────────────

  Starting vLLM: --gpu-memory-utilization 0.7
  Log: vllm_gmem_0p70.log
  Server ready after 43s
  Warming up (2 requests)...
  → concurrency=  1  ✓ clean     TTFT p99=0.09s  throughput=83 tok/s  errors=0/30
  → concurrency=  4  ✗ DEGRADED  TTFT p99=0.31s  throughput=220 tok/s  errors=0/30  [ttft_spike=3.5x baseline (>2.5x)]
  Degradation onset at concurrency=4. Stopping ramp.
  Preemption events detected in vLLM log: 1

  Ramp detail for gpu_memory_utilization=0.70:
  concurrency    Throughput    TTFT p50    TTFT p99    Latency p50    Latency p99    Errors                                       Status
                    (tok/s)         (s)         (s)            (s)            (s)
-------------  ------------  ----------  ----------  -------------  -------------  --------  -------------------------------------------
            1          82.8       0.084       0.09           0.429          2.819      0/30                                        clean
            4         219.9       0.084       0.314          0.683          4.09       0/30  DEGRADED [ttft_spike=3.5x baseline (>2.5x)]
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.85
────────────────────────────────────────────────────────────

  Starting vLLM: --gpu-memory-utilization 0.85
  Log: vllm_gmem_0p85.log
  Server ready after 43s
  Warming up (2 requests)...
  → concurrency=  1  ✓ clean     TTFT p99=0.09s  throughput=83 tok/s  errors=0/30
  → concurrency=  4  ✗ DEGRADED  TTFT p99=0.31s  throughput=232 tok/s  errors=0/30  [ttft_spike=3.4x baseline (>2.5x)]
  Degradation onset at concurrency=4. Stopping ramp.
  Preemption events detected in vLLM log: 1

  Ramp detail for gpu_memory_utilization=0.85:
  concurrency    Throughput    TTFT p50    TTFT p99    Latency p50    Latency p99    Errors                                       Status
                    (tok/s)         (s)         (s)            (s)            (s)
-------------  ------------  ----------  ----------  -------------  -------------  --------  -------------------------------------------
            1          82.8       0.083       0.092          0.43           2.822      0/30                                        clean
            4         232.3       0.085       0.309          0.733          4.488      0/30  DEGRADED [ttft_spike=3.4x baseline (>2.5x)]
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.95
────────────────────────────────────────────────────────────

  Starting vLLM: --gpu-memory-utilization 0.95
  Log: vllm_gmem_0p95.log
  Server ready after 44s
  Warming up (2 requests)...
  → concurrency=  1  ✓ clean     TTFT p99=0.09s  throughput=83 tok/s  errors=0/30
  → concurrency=  4  ✗ DEGRADED  TTFT p99=0.31s  throughput=233 tok/s  errors=0/30  [ttft_spike=3.4x baseline (>2.5x)]
  Degradation onset at concurrency=4. Stopping ramp.
  Preemption events detected in vLLM log: 1

  Ramp detail for gpu_memory_utilization=0.95:
  concurrency    Throughput    TTFT p50    TTFT p99    Latency p50    Latency p99    Errors                                       Status
                    (tok/s)         (s)         (s)            (s)            (s)
-------------  ------------  ----------  ----------  -------------  -------------  --------  -------------------------------------------
            1          82.8       0.084       0.09           0.429          2.813      0/30                                        clean
            4         233.3       0.085       0.309          0.731          4.467      0/30  DEGRADED [ttft_spike=3.4x baseline (>2.5x)]
  Stopping vLLM...
  Waiting 5s for GPU memory release...

================================================================================
EXPERIMENT 2 RESULTS: --gpu-memory-utilization Parameter Sensitivity
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  |  max-num-seqs: 256 (uncapped)
Prompt: ~512 tokens  |  Max completion: 256 tokens
Degradation thresholds: error_rate>10%  OR  TTFT p99>2.5× baseline
================================================================================
+----------------+---------------+---------------+---------------+--------------+---------------+--------------+
|     gpu_memory |      KV cache |     Max clean |   Degradation |     TTFT p99 |      TTFT p99 |   Preemption |
|   _utilization |   (~GB avail) |   concurrency |   onset conc. |   @ baseline |   @ max clean |       events |
+================+===============+===============+===============+==============+===============+==============+
|           0.5  |          ~5.8 |             1 |             4 |       0.090s |        0.090s |            1 |
+----------------+---------------+---------------+---------------+--------------+---------------+--------------+
|           0.7  |          ~9.0 |             1 |             4 |       0.090s |        0.090s |            1 |
+----------------+---------------+---------------+---------------+--------------+---------------+--------------+
|           0.85 |         ~11.4 |             1 |             4 |       0.092s |        0.092s |            1 |
+----------------+---------------+---------------+---------------+--------------+---------------+--------------+
|           0.95 |         ~13.0 |             1 |             4 |       0.090s |        0.090s |            1 |
+----------------+---------------+---------------+---------------+--------------+---------------+--------------+

Raw results saved to day6_exp2_results.json
Done. Review the tables above and record observations in your notes.

Key questions to answer:
  • How does max_clean_concurrency scale with gpu_memory_utilization?
    Is it roughly linear with KV cache GB, or does it plateau?
  • At which value does preemption first appear in the logs?
  • Does the degradation reason change across values?
    (Error-rate vs. TTFT spike tell you different things about failure mode.)
  • Compare max_clean_concurrency at 0.85 vs. 0.95 — is the extra headroom
    worth the reduced safety margin before OOM?
  • Cross-check: does your max_clean_concurrency match your Day 3 KV cache
    calculator prediction for the same sequence length and precision?
	
	
# exp2c

- Qwen2.5-3B
- vLLM configuration 1 block = 16 tokens
	- at startup vLLM reports 533 blocks => 533 x 16 tokens = 8528 tokens
	- TOTAL kv capacity is 8528 tokens
- 1 token requires
	- From Qwen2.5-3B's architecture:
		- num_layers   = 36
		- num_kv_heads = 8
		- head_dim     = 128
		- dtype        = FP16 = 2 bytes per element
		- so 2 (KV) * 36 * 8 * 128 * 2 = 2^12 * 36 = 4K * 36 = 144 KB
- So total KV memory
		- 144 KB & 8528
	

## Analysis
- Tried different strategy to surface the KV cache limitation at different utilizations
- What we expected to see was topping out at increasing RPS for increasing allowed memory.
- Unfortunately we see same RPS degradation for different memory utilizations
	- this is because the limiting factor is still PREFILL
	- short responses (not prompts) means each request spends little time in decode
	- so the system is prefilling all the time (and we aren't activating the optimizations for prefill)
	- prefill as you know is compute bound

====================================================================================================
EXPERIMENT 2c RESULTS: --gpu-memory-utilization (Option C — Sustained Arrival Rate)
Model: Qwen/Qwen2.5-3B-Instruct  |  max-num-seqs: 256 (uncapped)
Level: 150s total | 60s warmup | 90s measurement
Degradation: error_rate>10%  OR  TTFT>3.0× baseline  OR  TTFT trend>1.5× within window
====================================================================================================

  ── gpu_memory_utilization=0.7 ──────────────────────
    RPS   n_ok  n_tot   err%   TTFT p50   TTFT p99   lat p50   tok/s  act_rps    trend  Status
  ────────────────────────────────────────────────────────────────────────────────────────────
    0.5     46     46  0.0%     0.061s     0.073s     7.79s     131      0.5    0.99×  clean
    1.0     91     91  0.0%     0.060s     0.075s     8.36s     259      1.0    0.96×  clean
    2.0    181    181  0.0%     0.066s     0.086s    10.99s     515      2.0    0.94×  clean
    3.0    271    271  0.0%     0.069s     0.089s    12.49s     771      3.0    1.00×  clean
    4.0    361    361  0.0%     0.072s     0.096s    14.71s    1027      4.0    1.00×  clean
    5.0    451    451  0.0%     0.082s     0.253s    19.54s    1283      5.0    1.03×  clean
    6.0    541    541  0.0%     0.119s     0.310s    38.07s    1539      6.0    1.07×  clean
    8.0    742    742  0.0%     6.141s     9.838s    43.86s    2111      8.2    1.18×  DEGRADED [ttft_p50=100.4× baseline (>3.0×)]
   10.0   1007   1007  0.0%     6.552s    11.921s    43.76s    2864     11.2    1.00×  DEGRADED [ttft_p50=107.1× baseline (>3.0×)]
   12.0   1256   1256  0.0%     5.101s    17.128s    41.96s    3573     14.0    0.82×  DEGRADED [ttft_p50=83.4× baseline (>3.0×)]

  ── gpu_memory_utilization=0.85 ──────────────────────
    RPS   n_ok  n_tot   err%   TTFT p50   TTFT p99   lat p50   tok/s  act_rps    trend  Status
  ────────────────────────────────────────────────────────────────────────────────────────────
    0.5     46     46  0.0%     0.058s     0.073s     7.78s     131      0.5    0.99×  clean
    1.0     91     91  0.0%     0.060s     0.075s     8.35s     259      1.0    0.99×  clean
    2.0    181    181  0.0%     0.064s     0.085s    10.97s     515      2.0    0.96×  clean
    3.0    271    271  0.0%     0.068s     0.090s    12.45s     771      3.0    1.02×  clean
    4.0    361    361  0.0%     0.072s     0.096s    14.68s    1027      4.0    0.98×  clean
    5.0    451    451  0.0%     0.082s     0.239s    19.54s    1283      5.0    0.97×  clean
    6.0    541    541  0.0%     0.118s     0.252s    37.73s    1539      6.0    1.10×  clean
    8.0    739    739  0.0%     6.322s     9.902s    43.97s    2102      8.2    1.15×  DEGRADED [ttft_p50=108.4× baseline (>3.0×)]
   10.0   1008   1008  0.0%     6.636s    12.450s    43.42s    2867     11.2    0.98×  DEGRADED [ttft_p50=113.8× baseline (>3.0×)]
   12.0   1258   1258  0.0%     4.884s    17.290s    41.82s    3578     14.0    0.85×  DEGRADED [ttft_p50=83.7× baseline (>3.0×)]

  ── gpu_memory_utilization=0.95 ──────────────────────
    RPS   n_ok  n_tot   err%   TTFT p50   TTFT p99   lat p50   tok/s  act_rps    trend  Status
  ────────────────────────────────────────────────────────────────────────────────────────────
    0.5     46     46  0.0%     0.060s     0.074s     7.78s     131      0.5    1.03×  clean
    1.0     91     91  0.0%     0.060s     0.074s     8.35s     259      1.0    0.96×  clean
    2.0    181    181  0.0%     0.067s     0.084s    10.97s     515      2.0    1.01×  clean
    3.0    271    271  0.0%     0.068s     0.088s    12.46s     771      3.0    0.96×  clean
    4.0    361    361  0.0%     0.072s     0.096s    14.67s    1027      4.0    1.03×  clean
    5.0    451    451  0.0%     0.082s     0.204s    19.55s    1283      5.0    1.01×  clean
    6.0    541    541  0.0%     0.119s     0.302s    37.90s    1539      6.0    1.09×  clean
    8.0    741    741  0.0%     6.288s     9.697s    44.16s    2108      8.2    1.13×  DEGRADED [ttft_p50=105.5× baseline (>3.0×)]
   10.0   1008   1008  0.0%     6.671s    12.011s    44.09s    2867     11.2    1.00×  DEGRADED [ttft_p50=111.9× baseline (>3.0×)]
   12.0   1253   1253  0.0%     4.896s    17.106s    41.73s    3564     13.9    0.85×  DEGRADED [ttft_p50=82.1× baseline (>3.0×)]
   
# Attempted Fix

ab: ok going back to the calculation so conceptually 

Little's law states C (concurrency) = Lambda (rps  or arrival rate) x W (average residency)

For tokens, we know that max completion size is 256 tokens and it takes 7.8 seconds to generate 256 tokens for a given a single stream request

I.e. token generation rate = 256/7 = 33 tokens/s

ConcurrencyKV = token capacity / per-request tokens

Token capacity is determined by utilization (60% eg) of leftovers

Per request tokens can be calculated from  -> prompt tokens + completion max tokens = (e.g. 80+256 = 336)

RPS_kv = ConcurrencyKV / ResidencyKV

We know the single stream decode rate of 256maxtoken/7.8s = 33 tokens/s

For other maxtoken sizes we can get maxtoken/33 = residency (single stream)

So the single-stream RPS_kv (token_capacity × 33) / (tokens_per_req × max_tokens)
Single-stream RPS_kv = LOWER BOUND

# Experiment 2d

Still compute bound.  Throughput is the same in all so we are still COMPUTE bound not MEMORY bound

Here are the things that were fixed vs 2c

--dtype half — fixed the BF16 crash on T4 (compute capability 7.5)
--max-model-len 2048 — fixed the KV cache floor check crash at util=0.60
- GPU_MEM_UTIL_VALUES = [0.70, 0.85, 0.95] — dropped 0.60 since it would still crash without max-model-len, and was nearly useless anyway (~6 KV slots)
- WARMUP_PER_LEVEL_S 30→60, DURATION_PER_LEVEL_S 90→150 — fixed the backward trend ratio artifact at saturation

MAX_TOKENS was never changed — still 256 throughout.

I think we basically need longer decode output (increase MAX_TOKENS)

## exp2d Results	

((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ python day6-exp2d.py
====================================================================================================
Day 6 — Experiment 2c: --gpu-memory-utilization (Option C: Sustained Arrival Rate)
Model:           Qwen/Qwen2.5-3B-Instruct
dtype:           half  (T4 compute capability 7.5 — no BF16)
max-model-len:   2048  (capped from 32768 to allow low-util startup)
GPU util sweep:  [0.7, 0.85, 0.95]
RPS sweep:       [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24]
Level timing:    150s total  |  60s warmup  |  90s measurement
Max outstanding: 300 (semaphore safety valve)

Expected capacity ceilings (corrected overhead floor: 8.44 GB for Qwen2.5-3B):
  Little's Law: max_RPS = KV_slots / W,  W ≈ 7s avg latency
  util=0.7 → KV pool ~1.75 GB → ~ 37 concurrent → max stable ≈ 5.3 RPS
  util=0.85 → KV pool ~3.94 GB → ~ 83 concurrent → max stable ≈ 11.9 RPS
  util=0.95 → KV pool ~5.39 GB → ~114 concurrent → max stable ≈ 16.3 RPS
====================================================================================================

──────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.7
──────────────────────────────────────────────────────────────
  Starting vLLM: --gpu-memory-utilization 0.7
  Log: vllm_gmem_2c_0p70.log
  Server ready after 96s
  Warming up (5 sequential requests)...
  Warmup complete.

  RPS=  0.5  [60s warmup + 90s measurement]
    ✓ clean     n=46/46  TTFT p50=0.059s  actual_rps=0.5  throughput=131 tok/s  err=0.00  trend=1.00×

  RPS=  1.0  [60s warmup + 90s measurement]
    ✓ clean     n=91/91  TTFT p50=0.060s  actual_rps=1.0  throughput=259 tok/s  err=0.00  trend=1.00×

  RPS=  2.0  [60s warmup + 90s measurement]
    ✓ clean     n=181/181  TTFT p50=0.067s  actual_rps=2.0  throughput=515 tok/s  err=0.00  trend=1.04×

  RPS=  3.0  [60s warmup + 90s measurement]
    ✓ clean     n=271/271  TTFT p50=0.069s  actual_rps=3.0  throughput=771 tok/s  err=0.00  trend=1.04×

  RPS=  4.0  [60s warmup + 90s measurement]
    ✓ clean     n=361/361  TTFT p50=0.073s  actual_rps=4.0  throughput=1027 tok/s  err=0.00  trend=1.05×

  RPS=  5.0  [60s warmup + 90s measurement]
    ✓ clean     n=451/451  TTFT p50=0.083s  actual_rps=5.0  throughput=1283 tok/s  err=0.00  trend=1.03×

  RPS=  6.0  [60s warmup + 90s measurement]
    ✓ clean     n=541/541  TTFT p50=0.141s  actual_rps=6.0  throughput=1539 tok/s  err=0.00  trend=1.27×

  RPS=  8.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=743/743  TTFT p50=6.467s  actual_rps=8.3  throughput=2113 tok/s  err=0.00  trend=1.12×  [ttft_p50=109.0× baseline (>3.0×)]
  ↳ Degradation onset at RPS=8. Continuing 2 more level(s) to characterise cliff.

  RPS= 10.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=1016/1016  TTFT p50=6.614s  actual_rps=11.3  throughput=2890 tok/s  err=0.00  trend=0.97×  [ttft_p50=111.5× baseline (>3.0×)]

  RPS= 12.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=1264/1264  TTFT p50=4.917s  actual_rps=14.0  throughput=3595 tok/s  err=0.00  trend=0.86×  [ttft_p50=82.9× baseline (>3.0×)]
  ↳ Confirmed degradation. Stopping ramp.

  Stopping vLLM...
  Waiting 12s for GPU memory release...
  Preemption events in log: 1

──────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.85
──────────────────────────────────────────────────────────────
  Starting vLLM: --gpu-memory-utilization 0.85
  Log: vllm_gmem_2c_0p85.log
  Server ready after 51s
  Warming up (5 sequential requests)...
  Warmup complete.

  RPS=  0.5  [60s warmup + 90s measurement]
    ✓ clean     n=46/46  TTFT p50=0.063s  actual_rps=0.5  throughput=131 tok/s  err=0.00  trend=0.85×

  RPS=  1.0  [60s warmup + 90s measurement]
    ✓ clean     n=91/91  TTFT p50=0.061s  actual_rps=1.0  throughput=259 tok/s  err=0.00  trend=0.99×

  RPS=  2.0  [60s warmup + 90s measurement]
    ✓ clean     n=181/181  TTFT p50=0.068s  actual_rps=2.0  throughput=515 tok/s  err=0.00  trend=0.94×

  RPS=  3.0  [60s warmup + 90s measurement]
    ✓ clean     n=271/271  TTFT p50=0.069s  actual_rps=3.0  throughput=771 tok/s  err=0.00  trend=0.99×

  RPS=  4.0  [60s warmup + 90s measurement]
    ✓ clean     n=361/361  TTFT p50=0.074s  actual_rps=4.0  throughput=1027 tok/s  err=0.00  trend=1.00×

  RPS=  5.0  [60s warmup + 90s measurement]
    ✓ clean     n=451/451  TTFT p50=0.086s  actual_rps=5.0  throughput=1283 tok/s  err=0.00  trend=1.01×

  RPS=  6.0  [60s warmup + 90s measurement]
    ✓ clean     n=541/541  TTFT p50=0.140s  actual_rps=6.0  throughput=1539 tok/s  err=0.00  trend=1.36×

  RPS=  8.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=746/746  TTFT p50=6.389s  actual_rps=8.3  throughput=2122 tok/s  err=0.00  trend=1.18×  [ttft_p50=100.9× baseline (>3.0×)]
  ↳ Degradation onset at RPS=8. Continuing 2 more level(s) to characterise cliff.

  RPS= 10.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=1008/1008  TTFT p50=6.263s  actual_rps=11.2  throughput=2867 tok/s  err=0.00  trend=1.00×  [ttft_p50=98.9× baseline (>3.0×)]

  RPS= 12.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=1259/1259  TTFT p50=4.933s  actual_rps=14.0  throughput=3580 tok/s  err=0.00  trend=0.81×  [ttft_p50=77.9× baseline (>3.0×)]
  ↳ Confirmed degradation. Stopping ramp.

  Stopping vLLM...
  Waiting 12s for GPU memory release...
  Preemption events in log: 1

──────────────────────────────────────────────────────────────
  RUN: --gpu-memory-utilization=0.95
──────────────────────────────────────────────────────────────
  Starting vLLM: --gpu-memory-utilization 0.95
  Log: vllm_gmem_2c_0p95.log
  Server ready after 48s
  Warming up (5 sequential requests)...
  Warmup complete.

  RPS=  0.5  [60s warmup + 90s measurement]
    ✓ clean     n=46/46  TTFT p50=0.060s  actual_rps=0.5  throughput=131 tok/s  err=0.00  trend=1.00×

  RPS=  1.0  [60s warmup + 90s measurement]
    ✓ clean     n=91/91  TTFT p50=0.060s  actual_rps=1.0  throughput=259 tok/s  err=0.00  trend=0.96×

  RPS=  2.0  [60s warmup + 90s measurement]
    ✓ clean     n=181/181  TTFT p50=0.066s  actual_rps=2.0  throughput=515 tok/s  err=0.00  trend=1.04×

  RPS=  3.0  [60s warmup + 90s measurement]
    ✓ clean     n=271/271  TTFT p50=0.070s  actual_rps=3.0  throughput=771 tok/s  err=0.00  trend=1.01×

  RPS=  4.0  [60s warmup + 90s measurement]
    ✓ clean     n=361/361  TTFT p50=0.074s  actual_rps=4.0  throughput=1027 tok/s  err=0.00  trend=1.03×

  RPS=  5.0  [60s warmup + 90s measurement]
    ✓ clean     n=451/451  TTFT p50=0.084s  actual_rps=5.0  throughput=1283 tok/s  err=0.00  trend=1.00×

  RPS=  6.0  [60s warmup + 90s measurement]
    ✓ clean     n=541/541  TTFT p50=0.125s  actual_rps=6.0  throughput=1539 tok/s  err=0.00  trend=1.13×

  RPS=  8.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=742/742  TTFT p50=6.311s  actual_rps=8.2  throughput=2111 tok/s  err=0.00  trend=1.17×  [ttft_p50=105.6× baseline (>3.0×)]
  ↳ Degradation onset at RPS=8. Continuing 2 more level(s) to characterise cliff.

  RPS= 10.0  [60s warmup + 90s measurement]
    ✗ DEGRADED  n=1011/1011  TTFT p50=6.606s  actual_rps=11.2  throughput=2876 tok/s  err=0.00  trend=1.00×  [ttft_p50=110.5× baseline (>3.0×)]

  RPS= 12.0  [60s warmup + 90s measurement]
^A    ✗ DEGRADED  n=1259/1259  TTFT p50=4.912s  actual_rps=14.0  throughput=3581 tok/s  err=0.00  trend=0.85×  [ttft_p50=82.2× baseline (>3.0×)]
  ↳ Confirmed degradation. Stopping ramp.

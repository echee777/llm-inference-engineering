# --max-num-batched-token sweep
#     this param is the max number of tokens in a forward pass
#     does not include queue memory
# Shows: How --max-num-batched-token constraints COMPUTE (prefill throughput)

Day 6 — Experiment 3: --max-num-batched-tokens Parameter Sensitivity
This experiment probes how vLLM's per-iteration token budget affects prefill throughput and latency. 
--max-num-batched-tokens caps the total tokens processed in a single forward pass.
Without chunked prefill enabled, this parameter also acts as an implicit ceiling on prompt length — a prompt longer than the budget cannot be scheduled.
Setup conditions:

Chunked prefill disabled (default)
Fixed model: TinyLlama-1.1B on T4
Fixed --max-num-seqs and --gpu-memory-utilization (held constant from Exp 1/2 sweet spots)
Prompt length varied across short (~128 tokens) and long (~1024 tokens) to stress both regimes
Sweep: --max-num-batched-tokens ∈ {512, 1024, 2048, 4096}

Inquiry goals:

- How does the token budget affect prefill throughput for long prompts — does doubling the budget halve prefill time?
- At what budget value does a long prompt fail to schedule, confirming the implicit length ceiling?
- What happens to TTFT and throughput for short concurrent requests when a large prefill batch claims the full budget — can you observe the iteration latency spike?

The third goal is the most interesting signal: it's a controlled preview of the prefill/decode interference problem that becomes a central focus in Phase B.

## Explanation of max-num-batched-tokens w.r.t. requests

--max-num-batched-tokens = 512 total across all 50 requests combined in that one forward pass.
It's a global budget for the iteration, not a per-request budget. The scheduler has to decide how to allocate those 512 slots across whatever requests are active. In decode mode, each of the 50 requests contributes exactly 1 token to the batch (their next decode step), so 50 decode tokens fit easily within 512. In prefill mode, a single 400-token prompt consumes 400 of the 512 slots, leaving only 112 for anything else.
That's precisely why the parameter creates the prefill/decode tension. Decode is cheap per slot (1 token per request per iteration). Prefill is hungry (potentially hundreds of tokens per request per iteration). A large prefill can crowd out decode steps for all other concurrent requests within a single iteration's budget.

## Explanation of max-num-batched-tokens vs gpu-memory-utilization

"remind me again max-num-batched-tokens is the max number of tokens allowed globally in a forward pass which excludes req sitting in the queue .  So this basically constraints the KV cache you can build in prefill and decode across all requests"

The first part is right — it's the token count cap for a single forward pass, queue is excluded.

The second part isn't right. `--max-num-batched-tokens` does not constrain the KV cache. Those are independent dimensions.

The KV cache accumulates **across iterations over time** — every token that has ever been processed for an active sequence has an entry in the KV cache, and it stays there until the request completes. `--max-num-batched-tokens` only controls how many tokens get processed **in one iteration**. It has no bearing on how large the KV cache grows.

The concrete distinction: you could have a budget of 512 tokens per forward pass, but 50 concurrent requests each with 2000 tokens of KV cache already built up — that's 100,000 tokens resident in KV cache simultaneously. The budget only governs the next 512 tokens of work the GPU will do this iteration.

What actually constrains the KV cache is `--gpu-memory-utilization`, which sets how much VRAM vLLM reserves for KV storage. That's the parameter that caps how many tokens can be **resident** across all active requests at once.

So the two parameters control orthogonal things:
- `--gpu-memory-utilization` → how much KV cache can exist in memory simultaneously
- `--max-num-batched-tokens` → how many tokens the GPU processes in one forward pass


## Deliverable (from 3e)

**Day 6 — Experiment 3: `--max-num-batched-tokens` Parameter Sensitivity**

*Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 | GPU: T4 (g4dn.xlarge) | Fixed: `--max-num-seqs=16`, `--gpu-memory-utilization=0.85`, chunked prefill OFF*

| Budget (tokens) | SHORT throughput (tok/s) | SHORT TTFT p50 (s) | LONG throughput (tok/s) | LONG TTFT p50 (s) | Ceiling test (≈938 tok prompt) |
|---|---|---|---|---|---|
| 512 | 817 | 0.205 | — (invalid) | — | FAIL |
| 1024 | 826 | 0.210 | 313 | 0.659 | FAIL |
| 2048 | 836 | 0.232 | 335 | 0.674 | PASS |
| 4096 | 838 | 0.249 | 334 | 0.675 | PASS |

*Note: LONG prompt (≈678 tokens) and CEILING prompt (≈938 tokens) fail at budget=512 because vLLM enforces `max_num_batched_tokens >= max_model_len` as a startup contract, requiring `effective_max_model_len` to be clamped to the budget value. Rejection occurs at request validation, not the scheduler.*

---

Sweeping `--max-num-batched-tokens` from 1024 to 4096 produced no meaningful change in throughput or TTFT for either prompt regime on this hardware. SHORT throughput moved from 826 to 838 tok/s (+1.4%) and LONG TTFT p50 from 0.659s to 0.675s — well within noise. The parameter is not the binding constraint for TinyLlama on a T4: the GPU processes even a 678-token prefill fast enough that serializing across budget values produces indistinguishable forward pass times.

The parameter's actual purpose is **iteration latency bounding**, not throughput control. By capping the tokens processed per forward pass, it limits how long a single large prefill can monopolize the GPU — protecting p99 TTFT for concurrent short requests when long-context requests are in flight. This matters at production scale with larger models, where uncapped prefill batches can cause forward passes lasting hundreds of milliseconds. A follow-on prompt-length sweep (Exp 3e, Qwen2.5-3B) confirmed the underlying mechanic directly: prefill TTFT scales linearly with prompt length at R²=0.9966, with a measured slope of 0.228 ms/token and implied prefill throughput of 4,386 tok/s. At that rate, a 2,000-token prompt occupies the GPU for approximately 493ms — the duration every concurrent request must wait. This is the quantitative foundation for the prefill/decode interference experiments in Phase B.

The ceiling test result is the most mechanistically instructive finding from this experiment: vLLM enforces `max_num_batched_tokens >= max_model_len` as a hard startup contract in `SchedulerConfig._verify_args`, not as a per-request scheduler decision. Prompts exceeding `effective_max_model_len` are rejected at request validation before reaching the scheduler. Without chunked prefill, `--max-num-batched-tokens` therefore acts as an implicit prompt length ceiling — a configuration contract rather than a runtime guard.

=====================================================================================
EXPERIMENT 3e RESULTS: Prefill Compute Boundness — Prompt Length Sweep
Model:  Qwen/Qwen2.5-3B-Instruct
Budget: 2048 (fixed, no chunking)  |  max_completion=1  |  repeats=7  |  concurrency=1
=====================================================================================
+----------+----------+------------+------------+----------------+----------+
|   Target |   Actual |   TTFT p50 |   TTFT p50 |       ms/token |   Errors |
|   Tokens |   Tokens |        (s) |       (ms) |   (slope est.) |          |
+==========+==========+============+============+================+==========+
|      128 |      130 |      0.071 |       70.7 |              — |        0 |
+----------+----------+------------+------------+----------------+----------+
|      256 |      260 |      0.099 |       99.5 |          0.222 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      384 |      390 |      0.127 |      126.6 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      512 |      520 |      0.152 |      151.9 |          0.194 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      640 |      650 |      0.18  |      179.7 |          0.214 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      768 |      780 |      0.212 |      212.2 |          0.250 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      896 |      884 |      0.248 |      247.7 |          0.341 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1024 |     1014 |      0.275 |      274.9 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1280 |     1274 |      0.312 |      312.1 |          0.143 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1536 |     1534 |      0.394 |      394.2 |          0.316 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1792 |     1794 |      0.449 |      448.7 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+

Linear fit: TTFT(ms) = 37.6 + 0.2280 × prompt_tokens
  Intercept A = 37.6 ms  (fixed overhead: scheduler + 1 decode token)
  Slope     B = 0.2280 ms/token
  R²          = 0.9966
  Implied prefill throughput = 4386 tokens/s


## NOTES

- Experiment 3d had fixed FLOPS even though chunked prefill size varied.  This means total prefill was approximately the same because for this machine/model, the overhead per chunk was unnoticeable.

- Experiment 3e removed chunked prefill and simply used different prompt lengths, keeping them all below the max-num-batch-tokens


## day6-exp3a.py

[ssm-user@ip-10-99-0-199 ~]$ jobs
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ !em
emacs -nw day6-exp3.py
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ mv day6-exp3.py day6-exp3b^C
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ rm day6-exp3.py
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ emacs -nw ^C
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ emacs -nw day6-exp3.py
(failed reverse-i-search)`souce': ^Curce ~/venv-vllm/bin/activate
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3.py
====================================================================================================
Day 6 — Experiment 3: --max-num-batched-tokens Parameter Sensitivity
Model:           TinyLlama/TinyLlama-1.1B-Chat-v1.0
Sweep values:    [512, 1024, 2048, 4096]
max-num-seqs:    16  (fixed)
gpu-mem-util:    0.85  (fixed)
chunked-prefill: OFF  (default — not passed to server)

Prompt token estimates:
  SHORT    ≈  102 tokens  (6 × base phrase)
  LONG     ≈  442 tokens  (26 × base phrase)  — fits every budget ≥ 512
  OVERFLOW ≈  612 tokens  (36 × base phrase)  — fails at 512, passes at ≥ 1024

Effective max_model_len per run (= min(MAX_MODEL_LEN, budget)):
  vLLM enforces max_num_batched_tokens >= max_model_len at startup.
  We satisfy this by clamping max_model_len to the budget value.
  The OVERFLOW prompt (612 tokens) is rejected at request time when
  effective_max_model_len < 612, preserving the ceiling signal.

  budget= 512: effective_max_model_len= 512  OVERFLOW → FAIL (612 > eff_len=512)
  budget=1024: effective_max_model_len=1024  OVERFLOW → PASS (612 <= eff_len)
  budget=2048: effective_max_model_len=2048  OVERFLOW → PASS (612 <= eff_len)
  budget=4096: effective_max_model_len=2048  OVERFLOW → PASS (612 <= eff_len)
====================================================================================================

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=512
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 512  --max-model-len 512  (effective ceiling for this run)
  Log: vllm_batched_512.log
  Server ready after 48s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   605.6 tok/s  TTFT p50: 0.375s  TTFT p99: 0.458s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ⚠  32 error(s):
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
  ✗ All LONG requests failed.
  [CEILING] Single request, prompt≈612 tokens vs budget=512...
  ✗ [CEILING] FAIL  (HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 1066 to)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=1024
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 1024  --max-model-len 1024  (effective ceiling for this run)
  Log: vllm_batched_1024.log
  Server ready after 39s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   813.9 tok/s  TTFT p50: 0.239s  TTFT p99: 0.276s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   307.4 tok/s  TTFT p50: 0.668s  TTFT p99: 1.162s  Success: 32/32
  [CEILING] Single request, prompt≈612 tokens vs budget=1024...
  ✗ [CEILING] FAIL  (HTTP 400: {"object":"error","message":"This model's maximum context length is 1024 tokens. However, you requested 1066 t)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=2048
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 2048  --max-model-len 2048  (effective ceiling for this run)
  Log: vllm_batched_2048.log
  Server ready after 32s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   832.5 tok/s  TTFT p50: 0.232s  TTFT p99: 0.255s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   331.7 tok/s  TTFT p50: 0.676s  TTFT p99: 1.047s  Success: 32/32
  [CEILING] Single request, prompt≈612 tokens vs budget=2048...
  ✓ [CEILING] PASS  (TTFT: 0.096s)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=4096
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 4096  --max-model-len 2048  (effective ceiling for this run)
  Log: vllm_batched_4096.log
  Server ready after 29s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   844.1 tok/s  TTFT p50: 0.245s  TTFT p99: 0.253s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   329.7 tok/s  TTFT p50: 0.877s  TTFT p99: 1.058s  Success: 32/32
  [CEILING] Single request, prompt≈612 tokens vs budget=4096...
  ✓ [CEILING] PASS  (TTFT: 0.087s)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

====================================================================================================
EXPERIMENT 3 RESULTS: --max-num-batched-tokens Parameter Sensitivity
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Fixed: --max-num-seqs=16  --gpu-memory-utilization=0.85  chunked-prefill=OFF
SHORT prompt ≈ 102 tokens | LONG prompt ≈ 442 tokens | OVERFLOW prompt ≈ 612 tokens | max_completion=128
====================================================================================================
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|     Budget |   Regime |     Prompt |   Throughput |   TTFT p50 |   TTFT p99 |   Latency p50 |   Latency p99 |   Success |   Ceiling |
|   (tokens) |          |   (tokens) |      (tok/s) |        (s) |        (s) |           (s) |           (s) |           |      Test |
+============+==========+============+==============+============+============+===============+===============+===========+===========+
|        512 |    SHORT |        102 |        605.6 |      0.375 |      0.458 |         1.03  |         1.032 |     48/48 |    FAIL ✗ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        442 |          0   |      0     |      0     |         0     |         0     |      0/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       1024 |    SHORT |        102 |        813.9 |      0.239 |      0.276 |         0.766 |         0.772 |     48/48 |    FAIL ✗ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        442 |        307.4 |      0.668 |      1.162 |         1.615 |         1.618 |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       2048 |    SHORT |        102 |        832.5 |      0.232 |      0.255 |         0.748 |         0.753 |     48/48 |    PASS ✓ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        442 |        331.7 |      0.676 |      1.047 |         1.499 |         1.502 |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       4096 |    SHORT |        102 |        844.1 |      0.245 |      0.253 |         0.74  |         0.744 |     48/48 |    PASS ✓ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        442 |        329.7 |      0.877 |      1.058 |         1.509 |         1.512 |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+

Preemption events per run:
  budget=512: 1 preemption(s) in vLLM log
  budget=1024: 1 preemption(s) in vLLM log
  budget=2048: 1 preemption(s) in vLLM log
  budget=4096: 1 preemption(s) in vLLM log

Key questions to answer:
  • Does LONG TTFT p50 decrease as budget increases?
    If yes: larger budget allows more co-batching of LONG prefills.
    If flat: something else is the bottleneck (compute, max-num-seqs).

  • Does SHORT TTFT p50 also decrease with larger budget?
    If yes: head-of-line blocking by LONG prefills is reduced.
    This is Goal 3 — a preview of prefill/decode interference.

  • At what budget does the CEILING test transition from FAIL to PASS?
    Expected: FAIL at budget=512 (effective_max_model_len=512 < 612 tokens).
    Expected: PASS at budget≥1024 (effective_max_model_len=1024 or 2048 ≥ 612).
    Note: rejection here is at request validation (prompt > max_model_len),
    not the scheduler. vLLM enforces max_num_batched_tokens >= max_model_len
    as a startup contract — this is earlier and harder than a per-request check.

  • Does throughput scale roughly linearly with budget for the LONG regime?
    If 2× budget ≈ 2× throughput: batching efficiency is roughly linear.
    If sublinear: other bottlenecks are entering (memory bandwidth, decode compute).

Raw results saved to day6_exp3_results.json
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$


## day6-exp3b.py

### Analysis

The SHORT TTFT is now flat: 0.205, 0.210, 0.232, 0.249 — no meaningful trend. The earlier run's drop from 0.375 to 0.232 was a warmup artifact, not a real signal.
LONG TTFT is also flat: 0.659, 0.674, 0.675.
The real finding from this experiment is that --max-num-batched-tokens is not the binding constraint on this hardware. Here's why. At budget=512 with 156-token SHORT prompts, only ~3 requests fit per iteration. The queue-wait theory predicts meaningful TTFT inflation — but you don't see it. That means the iteration latency at budget=512 is so small that even serializing prefills across multiple iterations adds negligible wall-clock time. TinyLlama on a T4 processes a 512-token forward pass fast enough that the queuing effect is below measurement noise.
The binding constraint is decode throughput with max-num-seqs=16 all generating tokens simultaneously. That's what's setting your ~0.2s TTFT floor and your ~330-840 tok/s throughput ceiling — not the prefill budget.
This is actually the more instructive finding for your notes: --max-num-batched-tokens matters when prefill is expensive relative to decode, which happens with larger models and longer prompts at production scale. On TinyLlama it's a no-op. The parameter becomes meaningful when you scale up — which is exactly the regime Phase B's prefill/decode interference experiments are designed to expose.

### Console log

[ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3.py
====================================================================================================
Day 6 — Experiment 3: --max-num-batched-tokens Parameter Sensitivity
Model:           TinyLlama/TinyLlama-1.1B-Chat-v1.0
Sweep values:    [512, 1024, 2048, 4096]
max-num-seqs:    16  (fixed)
gpu-mem-util:    0.85  (fixed)
chunked-prefill: OFF  (default — not passed to server)

Prompt token counts (Llama tokenizer, ~26 tokens/repeat — confirmed from
vLLM HTTP 400 error messages during first run):
  SHORT    ≈  156 tokens  (6 × base phrase)   fits all budgets   (156+128=284 < 512)
  LONG     ≈  678 tokens  (26 × base phrase)  fits budgets ≥ 1024 (678+128=806 > 512)
  OVERFLOW ≈  938 tokens  (36 × base phrase)  fits budgets ≥ 2048 (938+128=1066 > 1024)

Effective max_model_len per run (= min(MAX_MODEL_LEN, budget)):
  vLLM enforces max_num_batched_tokens >= max_model_len at startup.
  We satisfy this by clamping max_model_len to the budget value.

  budget= 512: eff_len= 512  LONG → FAIL (806 > 512)      CEILING → FAIL (1066 > 512)
  budget=1024: eff_len=1024  LONG → PASS                  CEILING → FAIL (1066 > 1024)
  budget=2048: eff_len=2048  LONG → PASS                  CEILING → PASS
  budget=4096: eff_len=2048  LONG → PASS                  CEILING → PASS
====================================================================================================

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=512
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 512  --max-model-len 512  (effective ceiling for this run)
  Log: vllm_batched_512.log
  Server ready after 29s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   817.2 tok/s  TTFT p50: 0.205s  TTFT p99: 0.282s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ⚠  32 error(s):
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
     HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 806 tokens (678 in the messages, 128 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}
  ✗ All LONG requests failed.
  [CEILING] Single request, prompt≈938 tokens vs budget=512...
  ✗ [CEILING] FAIL  (HTTP 400: {"object":"error","message":"This model's maximum context length is 512 tokens. However, you requested 1066 to)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=1024
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 1024  --max-model-len 1024  (effective ceiling for this run)
  Log: vllm_batched_1024.log
  Server ready after 29s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   826.2 tok/s  TTFT p50: 0.210s  TTFT p99: 0.264s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   313.5 tok/s  TTFT p50: 0.659s  TTFT p99: 1.126s  Success: 32/32
  [CEILING] Single request, prompt≈938 tokens vs budget=1024...
  ✗ [CEILING] FAIL  (HTTP 400: {"object":"error","message":"This model's maximum context length is 1024 tokens. However, you requested 1066 t)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=2048
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 2048  --max-model-len 2048  (effective ceiling for this run)
  Log: vllm_batched_2048.log
  Server ready after 29s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   835.9 tok/s  TTFT p50: 0.232s  TTFT p99: 0.256s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   335.3 tok/s  TTFT p50: 0.674s  TTFT p99: 1.025s  Success: 32/32
  [CEILING] Single request, prompt≈938 tokens vs budget=2048...
  ✓ [CEILING] PASS  (TTFT: 0.095s)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

──────────────────────────────────────────────────────────────────────
  SWEEP: --max-num-batched-tokens=4096
──────────────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-batched-tokens 4096  --max-model-len 2048  (effective ceiling for this run)
  Log: vllm_batched_4096.log
  Server ready after 29s
  Warming up (2 requests)...
  [SHORT] Sending 48 requests @ concurrency=16...
  ✓ [SHORT] Throughput:   838.2 tok/s  TTFT p50: 0.249s  TTFT p99: 0.255s  Success: 48/48
  Warming up (2 requests)...
  [LONG] Sending 32 requests @ concurrency=16...
  ✓ [LONG] Throughput:   333.5 tok/s  TTFT p50: 0.675s  TTFT p99: 1.034s  Success: 32/32
  [CEILING] Single request, prompt≈938 tokens vs budget=4096...
  ✓ [CEILING] PASS  (TTFT: 0.097s)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

====================================================================================================
EXPERIMENT 3 RESULTS: --max-num-batched-tokens Parameter Sensitivity
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Fixed: --max-num-seqs=16  --gpu-memory-utilization=0.85  chunked-prefill=OFF
SHORT prompt ≈ 156 tokens | LONG prompt ≈ 678 tokens | OVERFLOW prompt ≈ 938 tokens | max_completion=128
====================================================================================================
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|     Budget |   Regime |     Prompt |   Throughput |   TTFT p50 |   TTFT p99 |   Latency p50 |   Latency p99 |   Success |   Ceiling |
|   (tokens) |          |   (tokens) |      (tok/s) |        (s) |        (s) |           (s) |           (s) |           |      Test |
+============+==========+============+==============+============+============+===============+===============+===========+===========+
|        512 |    SHORT |        156 |        817.2 |      0.205 |      0.282 |         0.764 |         0.773 |     48/48 |    FAIL ✗ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        678 |          0   |      0     |      0     |         0     |         0     |      0/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       1024 |    SHORT |        156 |        826.2 |      0.21  |      0.264 |         0.757 |         0.762 |     48/48 |    FAIL ✗ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        678 |        313.5 |      0.659 |      1.126 |         1.58  |         1.582 |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       2048 |    SHORT |        156 |        835.9 |      0.232 |      0.256 |         0.744 |         0.755 |     48/48 |    PASS ✓ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        678 |        335.3 |      0.674 |      1.025 |         1.477 |         1.48  |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|       4096 |    SHORT |        156 |        838.2 |      0.249 |      0.255 |         0.742 |         0.748 |     48/48 |    PASS ✓ |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+
|            |     LONG |        678 |        333.5 |      0.675 |      1.034 |         1.485 |         1.489 |     32/32 |           |
+------------+----------+------------+--------------+------------+------------+---------------+---------------+-----------+-----------+

Preemption events per run:
  budget=512: 1 preemption(s) in vLLM log
  budget=1024: 1 preemption(s) in vLLM log
  budget=2048: 1 preemption(s) in vLLM log
  budget=4096: 1 preemption(s) in vLLM log
  
# day6-exp3c.py

BudgetChunksPredicted ratioTTFT p501281313×0.279s25677×0.234s51244×0.206s
The TTFT is barely moving. Going from 4 chunks to 13 chunks should produce a 3.25× TTFT increase. You're seeing 0.206 → 0.279, which is only 1.35×. The predicted ratios are not tracking at all.
TinyLlama is too small to be compute-bound on prefill. A 1.1B parameter model processes even 128-token chunks so fast that the per-pass time is negligible — dominated instead by fixed overhead: kernel launch latency, Python/C++ scheduler loop, CUDA synchronization between chunks. Adding 9 more chunks of 128 tokens adds 9× the overhead, not 9× the GPU compute time, because the actual matrix multiply for 128 tokens at this model size is nearly instantaneous on a T4.
The experiment is working correctly — it's just revealing that compute-boundness on prefill requires either a large model or long chunks. With a 7B+ model, each 512-token chunk would take meaningfully longer to process and the ratios would track.
Let it finish running for completeness, but the conclusion is already clear: this is the wrong model to demonstrate this effect. The finding itself is worth noting.

[ssm-user@ip-10-99-0-199 ~]$ !em
emacs -nw day6-exp3.py
[ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3c.py
==========================================================================================
Day 6 — Experiment 3c: Prefill Compute Boundness
Model:           TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompt:          ≈1664 tokens  (8064 chars)
max_completion:  1 token  (TTFT only — minimize decode contamination)
max-num-seqs:    1  (single request — no queuing noise)
Chunked prefill: ON
Repeats:         5 per budget
Baseline budget: 2048  (prompt fits in 1 chunk)

Predicted chunk counts and TTFT ratios vs baseline:
  budget= 128: 13 chunks → predicted TTFT ratio = 13.00×
  budget= 256:  7 chunks → predicted TTFT ratio = 7.00×
  budget= 512:  4 chunks → predicted TTFT ratio = 4.00×
  budget=1024:  2 chunks → predicted TTFT ratio = 2.00×
  budget=2048:  1 chunks → predicted TTFT ratio = 1.00×

If measured ratios ≈ predicted ratios: prefill is compute-bound.
==========================================================================================

────────────────────────────────────────────────────────────
  SWEEP: budget=128
────────────────────────────────────────────────────────────

  Starting vLLM: budget=128  predicted_chunks=13
  Log: vllm_exp3c_128.log
  Server ready after 27s
  Warming up...
  repeat 1/5: TTFT = 0.279s
  repeat 2/5: TTFT = 0.279s
  repeat 3/5: TTFT = 0.279s
  repeat 4/5: TTFT = 0.279s
  repeat 5/5: TTFT = 0.280s
  → TTFT p50: 0.279s  (predicted chunks=13  errors=0)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=256
────────────────────────────────────────────────────────────

  Starting vLLM: budget=256  predicted_chunks=7
  Log: vllm_exp3c_256.log
  Server ready after 27s
  Warming up...
  repeat 1/5: TTFT = 0.235s
  repeat 2/5: TTFT = 0.234s
  repeat 3/5: TTFT = 0.237s
  repeat 4/5: TTFT = 0.224s
  repeat 5/5: TTFT = 0.233s
  → TTFT p50: 0.234s  (predicted chunks=7  errors=0)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=512
────────────────────────────────────────────────────────────

  Starting vLLM: budget=512  predicted_chunks=4
  Log: vllm_exp3c_512.log
  Server ready after 28s
  Warming up...
  repeat 1/5: TTFT = 0.205s
  repeat 2/5: TTFT = 0.210s
  repeat 3/5: TTFT = 0.206s
  repeat 4/5: TTFT = 0.206s
  repeat 5/5: TTFT = 0.205s
  → TTFT p50: 0.206s  (predicted chunks=4  errors=0)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=1024
────────────────────────────────────────────────────────────

  Starting vLLM: budget=1024  predicted_chunks=2
  Log: vllm_exp3c_1024.log
  Server ready after 27s
  Warming up...
  repeat 1/5: TTFT = 0.194s
  repeat 2/5: TTFT = 0.190s
  repeat 3/5: TTFT = 0.190s
  repeat 4/5: TTFT = 0.192s
  repeat 5/5: TTFT = 0.192s
  → TTFT p50: 0.192s  (predicted chunks=2  errors=0)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=2048
────────────────────────────────────────────────────────────

  Starting vLLM: budget=2048  predicted_chunks=1
  Log: vllm_exp3c_2048.log
  Server ready after 28s
  Warming up...
  repeat 1/5: TTFT = 0.179s
  repeat 2/5: TTFT = 0.183s
  repeat 3/5: TTFT = 0.181s
  repeat 4/5: TTFT = 0.182s
  repeat 5/5: TTFT = 0.186s
  → TTFT p50: 0.182s  (predicted chunks=1  errors=0)
  Stopping vLLM...
  Waiting 8s for GPU memory release...

==========================================================================================
EXPERIMENT 3c RESULTS: Prefill Compute Boundness
Model:  TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompt: ≈1664 tokens  |  max_completion=1  |  repeats=5  |  concurrency=1
Chunked prefill: ON  |  max-num-seqs=1  |  baseline budget=2048
Baseline TTFT (budget=2048): 0.182s  (1 chunk = 1 forward pass)
==========================================================================================
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|     Budget |   Predicted |   Predicted |   TTFT p50 |   Measured |   Deviation |            Verdict |
|   (tokens) |      Chunks |       Ratio |        (s) |      Ratio |         (%) |                    |
+============+=============+=============+============+============+=============+====================+
|        128 |          13 |      13.00× |      0.279 |      1.53× |       88.2% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|        256 |           7 |       7.00× |      0.234 |      1.28× |       81.6% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|        512 |           4 |       4.00× |      0.206 |      1.13× |       71.8% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|       1024 |           2 |       2.00× |      0.192 |      1.05× |       47.3% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|       2048 |           1 |       1.00× |      0.182 |      1.00× |        0.0% |    ✓ compute-bound |
+------------+-------------+-------------+------------+------------+-------------+--------------------+


## day6-exp3d.py


### CHATGPT

The main issue is that the interpretation is built on a faulty expected curve.

Your experiment does **not** actually test “compute-boundness implies TTFT scales like chunk count.” In your script, you explicitly assume both:

1. `TTFT ≈ ceil(prompt_tokens / budget) × T_per_pass`, and
2. the predicted ratio is just the chunk-count ratio `[9×, 5×, 3×, 2×, 1×]` for a fixed ~1040-token prompt. 

That is the wrong null hypothesis.

For a **single isolated request**, if prefill is compute-bound, the dominant compute term is much closer to:

`TTFT(b) ≈ total_prefill_compute(prompt_len) / effective_throughput(shape=b) + decode_1tok + per_chunk_overhead × num_chunks`

not `num_chunks × constant_pass_time`.

Because the **total number of prompt tokens is fixed**, shrinking the chunk budget creates **more passes but smaller passes**. The total prefill FLOPs do not blow up by 9× just because you split 1040 tokens into 9 chunks. Your own script even prints chunk times that scale with chunk size (`~22, 44, 89, 178, 356 ms`), which already contradicts the later use of pure chunk-count ratios as the expected TTFT ratio. 

So the flat curve is not strong evidence that “33 ms of scheduler overhead dominates everything.” It is more consistent with:

* total prefill work staying roughly constant,
* smaller chunks being somewhat less efficient,
* and a modest extra per-chunk fixed cost.

That is exactly the kind of curve you measured.

### CLAUDE:

The other LLM is right on the central point, and it's a clean logical error worth owning directly.
The expected ratio model was wrong from the start. The prediction of 9×/5×/3×/2×/1× assumed that TTFT = num_chunks × T_per_pass, where T_per_pass is constant. But T_per_pass is not constant — it scales with chunk size. A 128-token chunk takes roughly 1/8 the compute of a 1024-token chunk. So when you halve the budget, you double the chunk count but also halve the per-chunk time. The total prefill FLOPs stay fixed because the prompt length is fixed. You were never going to see 9× TTFT at budget=128 — that would require 9× more total compute, which isn't there.
What the experiment actually measured was chunking overhead and kernel efficiency degradation at small batch shapes, which is a real and interesting signal — just not the one the framing claimed.
The correct experiment to demonstrate TTFT ∝ prompt_tokens is the one suggested: fix the budget, sweep prompt length. That keeps the chunk shape constant while scaling total FLOPs linearly. If TTFT grows linearly with prompt length at a fixed budget, you've demonstrated compute-boundness directly and correctly.
So the redesigned experiment is:

One vLLM instance, budget fixed at 2048, chunked prefill on
Single request, no concurrency, max_completion=1
Prompt lengths: 256, 512, 768, 1024, 1280, 1536, 1792, 2048
Measure TTFT at each length, fit a line, check linearity

### LOG

ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3d.py
===============================================================================================
Day 6 — Experiment 3d: Prefill Compute Boundness (Qwen2.5-3B-Instruct)
Model:           Qwen/Qwen2.5-3B-Instruct
Prompt:          ≈1040 tokens  (5040 chars)
max_completion:  1 token  (TTFT only)
max-num-seqs:    1  (single request — no queuing noise)
Chunked prefill: ON
Repeats:         5 per budget  (p50 reported)
Baseline budget: 2048  (prompt fits in 1 chunk)

Why Qwen2.5-3B instead of TinyLlama:
  TinyLlama Exp 3c result: budget=128 measured 1.35× vs predicted 13×.
  Per-chunk overhead (~8ms) >> compute time (~2ms at 1.1B params, 128 tokens).
  Qwen2.5-3B: ~89ms per 512-token chunk vs ~8ms overhead → 10:1 signal/noise.

Predicted chunk counts and TTFT ratios vs baseline:
  budget= 128:  9 chunks  predicted ratio=9.00×  est. chunk time≈22ms
  budget= 256:  5 chunks  predicted ratio=5.00×  est. chunk time≈44ms
  budget= 512:  3 chunks  predicted ratio=3.00×  est. chunk time≈89ms
  budget=1024:  2 chunks  predicted ratio=2.00×  est. chunk time≈178ms
  budget=2048:  1 chunks  predicted ratio=1.00×  est. chunk time≈356ms

If measured ratios ≈ predicted ratios (deviation < 15%): prefill is compute-bound.
===============================================================================================

────────────────────────────────────────────────────────────
  SWEEP: budget=128
────────────────────────────────────────────────────────────

  Starting vLLM: budget=128  predicted_chunks=9
  Log: vllm_exp3d_128.log
  Server ready after 45s
  Warming up...
  repeat 1/5: TTFT = 0.546s
  repeat 2/5: TTFT = 0.538s
  repeat 3/5: TTFT = 0.539s
  repeat 4/5: TTFT = 0.540s
  repeat 5/5: TTFT = 0.538s
  → TTFT p50: 0.539s  (predicted chunks=9  errors=0)
  Stopping vLLM...
  Waiting 10s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=256
────────────────────────────────────────────────────────────

  Starting vLLM: budget=256  predicted_chunks=5
  Log: vllm_exp3d_256.log
  Server ready after 38s
  Warming up...
  repeat 1/5: TTFT = 0.318s
  repeat 2/5: TTFT = 0.313s
  repeat 3/5: TTFT = 0.309s
  repeat 4/5: TTFT = 0.315s
  repeat 5/5: TTFT = 0.314s
  → TTFT p50: 0.314s  (predicted chunks=5  errors=0)
  Stopping vLLM...
  Waiting 10s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=512
────────────────────────────────────────────────────────────

  Starting vLLM: budget=512  predicted_chunks=3
  Log: vllm_exp3d_512.log
  Server ready after 34s
  Warming up...
  repeat 1/5: TTFT = 0.283s
  repeat 2/5: TTFT = 0.283s
  repeat 3/5: TTFT = 0.291s
  repeat 4/5: TTFT = 0.286s
  repeat 5/5: TTFT = 0.284s
  → TTFT p50: 0.284s  (predicted chunks=3  errors=0)
  Stopping vLLM...
  Waiting 10s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=1024
────────────────────────────────────────────────────────────

  Starting vLLM: budget=1024  predicted_chunks=2
  Log: vllm_exp3d_1024.log
  Server ready after 36s
  Warming up...
  repeat 1/5: TTFT = 0.274s
  repeat 2/5: TTFT = 0.276s
  repeat 3/5: TTFT = 0.285s
  repeat 4/5: TTFT = 0.272s
  repeat 5/5: TTFT = 0.273s
  → TTFT p50: 0.274s  (predicted chunks=2  errors=0)
  Stopping vLLM...
  Waiting 10s for GPU memory release...

────────────────────────────────────────────────────────────
  SWEEP: budget=2048
────────────────────────────────────────────────────────────

  Starting vLLM: budget=2048  predicted_chunks=1
  Log: vllm_exp3d_2048.log
  Server ready after 34s
  Warming up...
  repeat 1/5: TTFT = 0.275s
  repeat 2/5: TTFT = 0.272s
  repeat 3/5: TTFT = 0.283s
  repeat 4/5: TTFT = 0.275s
  repeat 5/5: TTFT = 0.274s
  → TTFT p50: 0.275s  (predicted chunks=1  errors=0)
  Stopping vLLM...
  Waiting 10s for GPU memory release...

===============================================================================================
EXPERIMENT 3d RESULTS: Prefill Compute Boundness — Qwen2.5-3B-Instruct
Model:  Qwen/Qwen2.5-3B-Instruct
Prompt: ≈1040 tokens  |  max_completion=1  |  repeats=5  |  concurrency=1
Chunked prefill: ON  |  max-num-seqs=1  |  baseline budget=2048
Baseline TTFT (budget=2048): 0.275s  (1 chunk = 1 forward pass)
===============================================================================================
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|     Budget |   Predicted |   Predicted |   TTFT p50 |   Measured |   Deviation |            Verdict |
|   (tokens) |      Chunks |       Ratio |        (s) |      Ratio |         (%) |                    |
+============+=============+=============+============+============+=============+====================+
|        128 |           9 |       9.00× |      0.539 |      1.96× |       78.2% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|        256 |           5 |       5.00× |      0.314 |      1.14× |       77.1% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|        512 |           3 |       3.00× |      0.284 |      1.03× |       65.6% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|       1024 |           2 |       2.00× |      0.274 |      1.00× |       50.2% | ✗ other bottleneck |
+------------+-------------+-------------+------------+------------+-------------+--------------------+
|       2048 |           1 |       1.00× |      0.275 |      1.00× |        0.0% |    ✓ compute-bound |
+------------+-------------+-------------+------------+------------+-------------+--------------------+


## day6-exp3e.py


### CONFIRMED WITH LINEAR REGRESSION 
### linear relationship between prompt length & TTFT

That's a clean result. R² = 0.9966 is about as linear as you'll see on real hardware.
The three numbers worth recording:
Slope: 0.228 ms/token — this is the marginal cost of one additional prompt token on Qwen2.5-3B on a T4. Every token you add to a prompt costs 0.228ms of TTFT, deterministically.
Intercept: 37.6ms — the fixed overhead that exists regardless of prompt length. This matches the ~33ms residual estimated from Exp 3d, now properly identified as fixed overhead rather than per-chunk scheduler cost. It includes the Python scheduler loop, CUDA kernel launch setup, KV cache initialization, and the single decode step.
Implied prefill throughput: 4,386 tokens/s — this is your T4's measured prefill rate for Qwen2.5-3B at FP16 under single-request conditions. You can use this number directly in Phase B when reasoning about how long a given prefill holds the forward pass.
The ms/token column has some noise (0.143 to 0.341) but averages to ~0.22 consistently, which is expected — incremental slopes between adjacent points are noisy; the global fit is what matters.
The Phase B implication is now concrete and quantified: a 2000-token prompt occupies the GPU for approximately 37.6 + (2000 × 0.228) = 493ms before the first token. Any concurrent request waits that entire duration. That's the mechanistic foundation for the interference experiments, now backed by your own measured numbers.

### log

[ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3e.py
=====================================================================================
Day 6 — Experiment 3e: Prefill Compute Boundness (Prompt Length Sweep)
Model:           Qwen/Qwen2.5-3B-Instruct
Budget:          2048 (fixed — no chunking occurs at any tested length)
max_completion:  1 token  (TTFT only)
max-num-seqs:    1  (single request — no queuing noise)
Repeats:         7 per length  (p50 reported)

Why this formulation is correct (unlike Exp 3c/3d):
  Exp 3d swept BUDGET with fixed prompt → fixed total FLOPs → flat TTFT.
  This experiment sweeps PROMPT LENGTH with fixed budget → FLOPs scale
  linearly with L → TTFT should also scale linearly if compute-bound.

Prompt lengths to test:
  target= 128  actual≈ 130  repeats=5
  target= 256  actual≈ 260  repeats=10
  target= 384  actual≈ 390  repeats=15
  target= 512  actual≈ 520  repeats=20
  target= 640  actual≈ 650  repeats=25
  target= 768  actual≈ 780  repeats=30
  target= 896  actual≈ 884  repeats=34
  target=1024  actual≈1014  repeats=39
  target=1280  actual≈1274  repeats=49
  target=1536  actual≈1534  repeats=59
  target=1792  actual≈1794  repeats=69

One vLLM instance runs for the entire sweep (no restarts).
=====================================================================================
  Starting vLLM (single instance for full sweep)
  budget=2048  max-num-seqs=1  gpu-mem=0.85
  Log: vllm_exp3e.log
  Server ready after 31s


  Stopping vLLM...
  Waiting 10s for GPU memory release...
Traceback (most recent call last):
  File "/home/ssm-user/day6-exp3e.py", line 535, in <module>
    main()
  File "/home/ssm-user/day6-exp3e.py", line 508, in main
    connector = aiohttp.TCPConnector(limit=2)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ssm-user/venv-vllm/lib64/python3.12/site-packages/aiohttp/connector.py", line 993, in __init__
    super().__init__(
  File "/home/ssm-user/venv-vllm/lib64/python3.12/site-packages/aiohttp/connector.py", line 313, in __init__
    loop = loop or asyncio.get_running_loop()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: no running event loop
[ssm-user@ip-10-99-0-199 ~]$ %
-bash: fg: %: no such job
[ssm-user@ip-10-99-0-199 ~]$ !em
emacs -nw day6-exp3e.py
[ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp3e.py
=====================================================================================
Day 6 — Experiment 3e: Prefill Compute Boundness (Prompt Length Sweep)
Model:           Qwen/Qwen2.5-3B-Instruct
Budget:          2048 (fixed — no chunking occurs at any tested length)
max_completion:  1 token  (TTFT only)
max-num-seqs:    1  (single request — no queuing noise)
Repeats:         7 per length  (p50 reported)

Why this formulation is correct (unlike Exp 3c/3d):
  Exp 3d swept BUDGET with fixed prompt → fixed total FLOPs → flat TTFT.
  This experiment sweeps PROMPT LENGTH with fixed budget → FLOPs scale
  linearly with L → TTFT should also scale linearly if compute-bound.

Prompt lengths to test:
  target= 128  actual≈ 130  repeats=5
  target= 256  actual≈ 260  repeats=10
  target= 384  actual≈ 390  repeats=15
  target= 512  actual≈ 520  repeats=20
  target= 640  actual≈ 650  repeats=25
  target= 768  actual≈ 780  repeats=30
  target= 896  actual≈ 884  repeats=34
  target=1024  actual≈1014  repeats=39
  target=1280  actual≈1274  repeats=49
  target=1536  actual≈1534  repeats=59
  target=1792  actual≈1794  repeats=69

One vLLM instance runs for the entire sweep (no restarts).
=====================================================================================
  Starting vLLM (single instance for full sweep)
  budget=2048  max-num-seqs=1  gpu-mem=0.85
  Log: vllm_exp3e.log
  Server ready after 31s

  Global warmup (2 requests at mid-range length)...

  Prompt length: target=128  actual≈130 tokens
  repeat 1/7: TTFT = 0.070s
  repeat 2/7: TTFT = 0.071s
  repeat 3/7: TTFT = 0.071s
  repeat 4/7: TTFT = 0.071s
  repeat 5/7: TTFT = 0.071s
  repeat 6/7: TTFT = 0.071s
  repeat 7/7: TTFT = 0.071s
  → p50: 0.071s  (errors=0)

  Prompt length: target=256  actual≈260 tokens
  repeat 1/7: TTFT = 0.101s
  repeat 2/7: TTFT = 0.096s
  repeat 3/7: TTFT = 0.101s
  repeat 4/7: TTFT = 0.099s
  repeat 5/7: TTFT = 0.099s
  repeat 6/7: TTFT = 0.100s
  repeat 7/7: TTFT = 0.099s
  → p50: 0.099s  (errors=0)

  Prompt length: target=384  actual≈390 tokens
  repeat 1/7: TTFT = 0.127s
  repeat 2/7: TTFT = 0.133s
  repeat 3/7: TTFT = 0.124s
  repeat 4/7: TTFT = 0.125s
  repeat 5/7: TTFT = 0.124s
  repeat 6/7: TTFT = 0.127s
  repeat 7/7: TTFT = 0.129s
  → p50: 0.127s  (errors=0)

  Prompt length: target=512  actual≈520 tokens
  repeat 1/7: TTFT = 0.153s
  repeat 2/7: TTFT = 0.151s
  repeat 3/7: TTFT = 0.154s
  repeat 4/7: TTFT = 0.152s
  repeat 5/7: TTFT = 0.152s
  repeat 6/7: TTFT = 0.153s
  repeat 7/7: TTFT = 0.150s
  → p50: 0.152s  (errors=0)

  Prompt length: target=640  actual≈650 tokens
  repeat 1/7: TTFT = 0.181s
  repeat 2/7: TTFT = 0.179s
  repeat 3/7: TTFT = 0.179s
  repeat 4/7: TTFT = 0.180s
  repeat 5/7: TTFT = 0.181s
  repeat 6/7: TTFT = 0.179s
  repeat 7/7: TTFT = 0.180s
  → p50: 0.180s  (errors=0)

  Prompt length: target=768  actual≈780 tokens
  repeat 1/7: TTFT = 0.212s
  repeat 2/7: TTFT = 0.212s
  repeat 3/7: TTFT = 0.211s
  repeat 4/7: TTFT = 0.211s
  repeat 5/7: TTFT = 0.213s
  repeat 6/7: TTFT = 0.212s
  repeat 7/7: TTFT = 0.215s
  → p50: 0.212s  (errors=0)

  Prompt length: target=896  actual≈884 tokens
  repeat 1/7: TTFT = 0.246s
  repeat 2/7: TTFT = 0.249s
  repeat 3/7: TTFT = 0.248s
  repeat 4/7: TTFT = 0.247s
  repeat 5/7: TTFT = 0.245s
  repeat 6/7: TTFT = 0.248s
  repeat 7/7: TTFT = 0.250s
  → p50: 0.248s  (errors=0)

  Prompt length: target=1024  actual≈1014 tokens
  repeat 1/7: TTFT = 0.273s
  repeat 2/7: TTFT = 0.275s
  repeat 3/7: TTFT = 0.275s
  repeat 4/7: TTFT = 0.275s
  repeat 5/7: TTFT = 0.272s
  repeat 6/7: TTFT = 0.275s
  repeat 7/7: TTFT = 0.277s
  → p50: 0.275s  (errors=0)

  Prompt length: target=1280  actual≈1274 tokens
  repeat 1/7: TTFT = 0.310s
  repeat 2/7: TTFT = 0.315s
  repeat 3/7: TTFT = 0.313s
  repeat 4/7: TTFT = 0.312s
  repeat 5/7: TTFT = 0.311s
  repeat 6/7: TTFT = 0.313s
  repeat 7/7: TTFT = 0.312s
  → p50: 0.312s  (errors=0)

  Prompt length: target=1536  actual≈1534 tokens
  repeat 1/7: TTFT = 0.395s
  repeat 2/7: TTFT = 0.384s
  repeat 3/7: TTFT = 0.384s
  repeat 4/7: TTFT = 0.397s
  repeat 5/7: TTFT = 0.394s
  repeat 6/7: TTFT = 0.395s
  repeat 7/7: TTFT = 0.393s
  → p50: 0.394s  (errors=0)

  Prompt length: target=1792  actual≈1794 tokens
  repeat 1/7: TTFT = 0.447s
  repeat 2/7: TTFT = 0.449s
  repeat 3/7: TTFT = 0.447s
  repeat 4/7: TTFT = 0.453s
  repeat 5/7: TTFT = 0.446s
  repeat 6/7: TTFT = 0.449s
  repeat 7/7: TTFT = 0.451s
  → p50: 0.449s  (errors=0)

  Stopping vLLM...
  Waiting 10s for GPU memory release...

=====================================================================================
EXPERIMENT 3e RESULTS: Prefill Compute Boundness — Prompt Length Sweep
Model:  Qwen/Qwen2.5-3B-Instruct
Budget: 2048 (fixed, no chunking)  |  max_completion=1  |  repeats=7  |  concurrency=1
=====================================================================================
+----------+----------+------------+------------+----------------+----------+
|   Target |   Actual |   TTFT p50 |   TTFT p50 |       ms/token |   Errors |
|   Tokens |   Tokens |        (s) |       (ms) |   (slope est.) |          |
+==========+==========+============+============+================+==========+
|      128 |      130 |      0.071 |       70.7 |              — |        0 |
+----------+----------+------------+------------+----------------+----------+
|      256 |      260 |      0.099 |       99.5 |          0.222 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      384 |      390 |      0.127 |      126.6 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      512 |      520 |      0.152 |      151.9 |          0.194 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      640 |      650 |      0.18  |      179.7 |          0.214 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      768 |      780 |      0.212 |      212.2 |          0.250 |        0 |
+----------+----------+------------+------------+----------------+----------+
|      896 |      884 |      0.248 |      247.7 |          0.341 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1024 |     1014 |      0.275 |      274.9 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1280 |     1274 |      0.312 |      312.1 |          0.143 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1536 |     1534 |      0.394 |      394.2 |          0.316 |        0 |
+----------+----------+------------+------------+----------------+----------+
|     1792 |     1794 |      0.449 |      448.7 |          0.209 |        0 |
+----------+----------+------------+------------+----------------+----------+

Linear fit: TTFT(ms) = 37.6 + 0.2280 × prompt_tokens
  Intercept A = 37.6 ms  (fixed overhead: scheduler + 1 decode token)
  Slope     B = 0.2280 ms/token
  R²          = 0.9966
  Implied prefill throughput = 4386 tokens/s

  Verdict: ✓ COMPUTE-BOUND — TTFT scales linearly with prompt length.

How to read the ms/token column:
  Each row shows the incremental ms per token relative to the previous
  prompt length. If prefill is compute-bound, this should be approximately
  constant across all rows. If it grows: attention or memory costs are
  increasing. If it shrinks: kernel tiling efficiency is improving.

Context for prior experiments:
  Exp 3c (TinyLlama, budget sweep): flat TTFT — overhead dominated compute.
  Exp 3d (Qwen2.5-3B, budget sweep): flat TTFT — expected ratio model was
    wrong. Fixed prompt = fixed FLOPs. Chunk count scaling was never valid.
  Exp 3e (Qwen2.5-3B, length sweep): correct formulation. TTFT ∝ L if
    compute-bound. R² measures how cleanly the linear relationship holds.
Raw results saved to day6_exp3e_results.json
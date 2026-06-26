# Perplexity

Captures the uncertainty (low confidence / low probability) of each successive token generated.

# Why is perplexity measured with log(P)

## Reason 1

If you just averaged the probabilities, you'd get misleading results.

Consider two models over 2 tokens:

Model A: P = 0.99, P = 0.01 (nails one, totally wrong on the other)
Model B: P = 0.5, P = 0.5 (mediocre on both)

Average probability: both get 0.50. They look identical.

But Model A is catastrophically wrong on token 2 — it essentially failed. You want that to dominate the score. Log does this because it heavily penalizes near-zero probabilities:

Model A: log(0.99) + log(0.01) = -0.01 + -4.61 = -4.62 → perplexity = exp(2.31) = 10.05
Model B: log(0.5) + log(0.5) = -0.69 + -0.69 = -1.38 → perplexity = exp(0.69) = 2.0

Model B wins — and it should. A model that's consistently decent is better than one that's great half the time and completely lost the other half.

The log turns multiplication into addition. What you're really computing is the geometric mean of the probabilities, not the arithmetic mean. The geometric mean punishes low values much more harshly, which matches the intuition that a
single confident wrong prediction is worse than being moderately uncertain everywhere.

## Reason 2:

P("They then pour") = P(They) × P(then|They) × P(pour|They then)

That's a product. log(a × b × c) = log(a) + log(b) + log(c). Just a property of logarithms — it turns products into sums.

Raw probabilities quickly asymptote to zero and lose signal.

# MORNING

## Step 1: Perplexity on WikiText-2

### Methodology

Used `lm_eval` with the `wikitext` task (WikiText-2 raw v1, 62 samples, 0-shot). All runs on T4 GPU with `dtype=float16`.

Working command:

```bash
lm_eval --model vllm \
  --model_args pretrained=<model>,dtype=float16,gpu_memory_utilization=0.6,max_model_len=2048,enforce_eager=True \
  --tasks wikitext \
  --device cuda \
  --output_path ./results/<output_dir>
```

### Results

| Precision | Model                              | Word Perplexity |
| --------- | ---------------------------------- | --------------- |
| FP16      | Qwen/Qwen2.5-3B-Instruct           | 11.66           |
| INT8-AWQ  | qwen2.5-3b-int8-awq                | 11.68           |
| INT4-GPTQ | Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 | 12.57           |

**Observations:**

- Ordering matches expectation: FP16 (11.66) < INT8 (11.68) < INT4 (12.57)
- INT8-AWQ perplexity increase: +0.17% vs FP16 — negligible distribution shift
- INT4-GPTQ perplexity increase: +7.8% vs FP16 — meaningful distribution shift

### Lessons Learned

Getting lm_eval running on T4 took some iteration. The main issues were around memory and backend compatibility.

On the memory side, just tuning `gpu_memory_utilization` wasn't enough — I kept OOM'ing even at 0.5. The fix was adding `max_model_len=2048` (caps KV cache allocation) and `enforce_eager=True` (disables CUDA graph capture, which eats a lot of memory on T4). Needed all three knobs together. This is T4-specific — wouldn't be an issue on A100 with 80GB.

For INT8-AWQ, the vLLM backend produced NaN logprobs with the local checkpoint. Tried the Hub AWQ checkpoint (`Qwen/Qwen2.5-3B-Instruct-AWQ`) through vLLM and it worked, but that's actually a 4-bit AWQ model (perplexity 12.45), not 8-bit
— the naming is misleading. Ended up switching to the HF backend (`--model hf`) for the local INT8 checkpoint, which worked fine. Perplexity is a model-intrinsic measurement so the serving backend doesn't matter — it's fine to use HF for eval even though Day 11 benchmarks used vLLM.

## Step 2 — ACCURACY -- Downstream Task Eval: HellaSwag (0-shot)

HellaSwag is similar to perplexity in that it tests probability of a completion but this time with MCQ adversarial choices to pick from.

It measures 2 metrics -- accuracy and normalized accuracy

- Accuracy is the rate it chooses the correct/human answer
- Normalized acc is accuracy divided by number of tokens in the choice

  Request 1:
  context: "Making a cake: The person cracks eggs into a bowl And stirs the mixture."
  continuation: " They then pour the batter into a greased pan and place it in the oven."
  → model returns: logprob = -12.3

  Request 2:
  context: "Making a cake: The person cracks eggs into a bowl And stirs the mixture."
  continuation: " The person then jumps on a trampoline while holding the bowl."
  → model returns: logprob = -28.7

  Request 3:
  context: "Making a cake: The person cracks eggs into a bowl And stirs the mixture."
  continuation: " A dog runs across the kitchen and knocks over a chair."
  → model returns: logprob = -31.2

  Request 4:
  context: "Making a cake: The person cracks eggs into a bowl And stirs the mixture."
  continuation: " The narrator describes the history of ancient Rome."
  → model returns: logprob = -35.1

  The model never sees the choices listed together. It just gets the same context 4 times, each with a different continuation appended. For each one, it computes "how likely would I be to generate this continuation given this context?" —
  the sum of per-token log-probs across the continuation tokens.

  Then lm-eval takes the argmax:
  - acc: argmax of raw log-probs → picks request 1 (-12.3) → correct
  - acc_norm: argmax of (log-prob / token count per continuation) → same pick, but removes length bias

  model_pick = argmax(logprobs across 4 continuations) → e.g. choice 0
  gold = dataset's label field → e.g. choice 0

  acc = 1.0 if model_pick == gold else 0.0

  So for each example it's a binary 1 or 0 — did the model's highest-likelihood continuation match the human-labeled correct one? The reported accuracy is the mean across all examples.

  That's the whole thing. No "A/B/C/D" framing, no instruction to "pick the best answer." Just raw next-token likelihood, four times.

### Results

| Precision | Model                                   | Accuracy | Acc (norm) | Backend |
| --------- | --------------------------------------- | -------- | ---------- | ------- |
| FP16      | Qwen/Qwen2.5-3B-Instruct                | 50.34%   | 64.20%     | vllm    |
| INT8-AWQ  | qwen2.5-3b-int8-awq (local, HF backend) | 50.54%   | 64.20%     | hf      |
| INT4-GPTQ | Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4      | 49.14%   | 63.36%     | vllm    |

**Observations:**

- INT8-AWQ acc_norm identical to FP16 (64.20% both) — zero decision-level degradation, consistent with the near-zero perplexity shift from Step 1
- INT4-GPTQ acc_norm drops by 0.84 pp (−1.31%) — same direction as the +7.8% perplexity increase, both metrics degrading together strengthens the signal
- Interesting gap: INT4's perplexity shifted 7.8% but accuracy only dropped 1.31%. The distribution moved a lot, but most of that shift didn't cross HellaSwag's decision boundaries. This is why you need both metrics — perplexity catches all probability mass shifts, task accuracy only catches the ones that flip an answer

### Lessons Learned

Same INT8-AWQ backend issue as Step 1 — vLLM produces NaN logprobs with the local checkpoint, so I used the HF backend again. HellaSwag is likelihood-based (comparing log-probabilities of completion options), so the backend doesn't matter — same weights, same tokenizer, same numbers.

Ran a 25% subset (`--limit 0.25`) to keep runtimes sane on T4. The vLLM runs took ~6 min each, but the HF backend run for INT8-AWQ took ~64 min — big difference. Standard errors came out to ±1.00 pp on accuracy and ±0.96 pp on acc_norm. That's tight enough to see INT4-GPTQ's degradation direction clearly, but not tight enough to call it statistically significant on this task alone. Between perplexity (Step 1) and task accuracy both pointing the same way, the picture is pretty clear though.

### Methodology

Used `lm_eval` with the `hellaswag` task (0-shot, 25% subset = 2,511 of 10,042 validation samples). All runs on T4 GPU with `dtype=float16`.

Working command:

```bash
lm_eval --model vllm \
  --model_args pretrained=<model>,dtype=float16,gpu_memory_utilization=0.6,max_model_len=2048,enforce_eager=True \
  --tasks hellaswag \
  --num_fewshot 0 \
  --limit 0.25 \
  --device cuda \
  --output_path ./results/<output_dir>
```

#### FB16

```
  # INT4-GPTQ
  lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4,dtype=float16,gpu_memory_utilization=0.6,max_model_len=2048,enforce_eager=True \
    --tasks hellaswag \
    --num_fewshot 0 \
    --limit 0.25 \
    --device cuda \
    --output_path ./results/int4_gptq_hellaswag
```

Output:

```
(EngineCore_DP0 pid=6678) INFO 03-18 05:47:37 [scheduler.py:875] [SCHED_STEP] ts=1773812857616 active=[] new=0 running=0 resumed=0 finished=[] preempted=[] total_tokens=0
(EngineCore_DP0 pid=6678) INFO 03-18 05:47:37 [kv_cache_manager.py:411] [BLOCK_FREE] ts=1773812857616 req=10043-bd248159 freed=2 free=11156/11156
(EngineCore_DP0 pid=6678) INFO 03-18 05:47:37 [scheduler.py:875] [SCHED_STEP] ts=1773812857617 active=[] new=0 running=0 resumed=0 finished=[bd248159] preempted=[] total_tokens=0
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████| 10044/10044 [05:38<00:00, 29.64it/s]
fatal: not a git repository (or any of the parent directories): .git
2026-03-18:05:47:40 INFO     [loggers.evaluation_tracker:247] Saving results aggregated
vllm ({'pretrained': 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', 'dtype': 'float16', 'gpu_memory_utilization': 0.6, 'max_model_len': 2048, 'enforce_eager': True}), gen_kwargs: ({}), limit: 0.25, num_fewshot: 0, batch_size: 1
|  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------|------:|------|-----:|--------|---|-----:|---|-----:|
|hellaswag|      1|none  |     0|acc     |↑  |0.4914|±  |0.0100|
|         |       |none  |     0|acc_norm|↑  |0.6336|±  |0.0096|
```

#### INT8-AWQ (HF backend)

```
  lm_eval --model hf \
    --model_args pretrained=/home/ssm-user/day11/qwen2.5-3b-int8-awq,dtype=float16 \
    --tasks hellaswag \
    --num_fewshot 0 \
    --limit 0.25 \
    --device cuda \
    --output_path ./results/int8_awq_hellaswag
```

#### FP16

```
  lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,gpu_memory_utilization=0.6,max_model_len=2048,enforce_eager=True \
    --tasks hellaswag \
    --num_fewshot 0 \
    --limit 0.25 \
    --device cuda \
    --output_path ./results/fp16_hellaswag
```

# STEP 3

## Methodology

- Generate the prompts
- Write a driver script that launches headless vllm engine

## Side learnings on vllm

vllm supports headless for batched operations (no need request serving but want all benefits of batching/pagedAttention and pipeline/tensor parallelism)

```
  from vllm import LLM, SamplingParams

  llm = LLM(model="...", quantization="...")          # instantiates LLMEngine
  outputs = llm.generate(prompts, SamplingParams(...)) # runs inference
```

vllm also provides pipelined/tensor parallelism

```
llm = LLM(model="...", tensor_parallel_size=2) # split layers across GPUs width-wise
llm = LLM(model="...", pipeline_parallel_size=2) # split layers across GPUs depth-wise
```

Not relevant to you on a single T4, but it's how you'd serve a 70B model across 4×A100s. Tensor parallelism is the more common one — it shards the weight matrices within each layer so all GPUs participate in every token. Pipeline
parallelism splits by layer groups, which has less communication overhead but introduces pipeline bubbles.

## Side learnings on intq-awq quantized checkpoint

## Note — Why the INT8-AWQ Checkpoint Doesn't Work in vLLM

This has been a recurring issue across Steps 1, 2, and 3, so worth writing up properly.

When I self-quantized Qwen2.5-3B to INT8 using AutoAWQ on Day 11, the checkpoint got saved in `compressed-tensors` format (you can see `"quant_method": "compressed-tensors"` in its config.json). This is different from the native AWQ form at that vLLM's AWQ kernels expect.

The result: vLLM loads the model without error, but the inference output is garbage. In Steps 1–2 (lm_eval), this showed up as NaN logprobs. In Step 3 (generation), it showed up as the model outputting solid walls of `!!!!!!!!!!` — it gets stuck repeating one high-probability token forever.

**Why it works through HuggingFace but not vLLM:**

This comes down to how the two inference stacks execute a forward pass. At the HuggingFace/transformers level, inference is generic PyTorch — the weights get dequantized back to FP16, then it's standard `torch.matmul`, `F.softmax`, etc. The same code path works for any quantization format because by the time the matmul runs, everything is FP16. It's slow but it always works.

vLLM replaces those generic ops with specialized CUDA kernels — fused attention, paged KV management, and critically, quantization-aware matmul that operates on the packed INT4/INT8 weights directly without fully dequantizing. These kernels are fast but format-specific. The `compressed-tensors` kernel expects weights packed a certain way, and if anything is off — layout, scaling factors, zero-point encoding — you get garbage out.

So same weights, same math conceptually, but completely different code executing the forward pass. The bug is in vLLM's `compressed-tensors` kernel (or in how AutoAWQ serialized the checkpoint for that format), not in the weights themselves.

**Practical impact:** For eval (Steps 1–2) this didn't matter — perplexity and accuracy are model-intrinsic measurements, so using the HF backend is fine. For Step 3 (qualitative generation), I used `transformers.generate()` directly for INT8-AWQ while keeping vLLM for FP16 and INT4-GPTQ. The outputs are comparable since we're evaluating quality, not speed. For actual serving benchmarks (Day 11), this checkpoint can't be used through vLLM — would need to re-quantize in native AWQ format.

**Lesson:** The quantization format and the inference runtime are tightly coupled. A checkpoint that works in one stack can silently produce garbage in another without any load-time error. Always sanity-check generation output after load\
ing a quantized model in a new runtime — don't assume "it loaded, so it works."

## Step 3 — Qualitative Error Analysis

### Results

Wrote 10 prompts — two per category (Q&A, creative writing, code generation, multi-step reasoning, long-context) — and ran all of them through each precision with `temperature=0`, `max_tokens=256`. Same INT8-AWQ backend workaround as bef\
ore: `transformers.generate()` for INT8, vLLM for FP16 and INT4-GPTQ.

| Category             | Prompts | INT8-AWQ Failures | INT4-GPTQ Failures |
| -------------------- | ------- | ----------------- | ------------------ |
| Q&A (factual)        | 2       | 0/2               | 1/2                |
| Creative writing     | 2       | 0/2               | 2/2                |
| Code generation      | 2       | 0/2               | 1/2                |
| Multi-step reasoning | 2       | 0/2               | 0/2                |
| Long-context         | 2       | 0/2               | 1/2                |
| **Total**            | **10**  | **0/10**          | **5/10**           |

## Observations

INT8-AWQ was indistinguishable from FP16 across all 10 prompts. Not "close" — literally the same content with trivial wording differences. Completely consistent with the +0.17% perplexity number from Step 1. There's nothing to flag here.

INT4-GPTQ is a different story. It degraded on 5 of 10 prompts, and the failures weren't subtle:

- **Creative writing (2/2 failed):** This was the worst category. The poem prompt (creative_2) hit a severe repetition loop — INT4 wrote 6 lines, then repeated the entire stanza verbatim 4 times until it ran out of tokens. The prose prompt (creative_1) had repeated phrases ("fur as white as the snow that blankets the ground" appeared twice) and ran way past the requested 4-5 sentences. Repetition is clearly INT4's dominant failure mode.
- **Factual Q&A (1/2 failed):** On the speed-of-light question, all three precisions hallucinated the scientist attribution (none got it right), but FP16 and INT8 hallucinated _the same wrong answer_ (Michelson-Morley), while INT4 went off on a rambling detour through Huygens, Rømer, and Laplace — internally contradictory and much more confused. The probability mass has shifted enough to land on entirely different tokens, not just slightly worse versions of the same answer.
- **Code generation (1/2 failed):** The balanced-parentheses function itself was fine, but INT4's accompanying test cases had a syntax error — `("{[)", False]` with a `]` closing what should be a `)`. Would crash at parse time. The two-sum prompt was correct across all three precisions.
- **Long-context (1/2 failed):** On the memory wall question, INT4 got the facts right (2:1 in 1980, 200:1 today, three techniques) but opened with "To what ratio does the passage refer to as the 'memory wall'?" — restating a different question than what was asked, and slightly mischaracterizing what the passage calls the memory wall. Minor, but it's the kind of comprehension slip you'd notice.
- **Reasoning (0/2 failed):** Both multi-step reasoning prompts — wheat production and the handshake problem — produced identical correct reasoning chains at all three precisions. The math and logic survived INT4 quantization without any issues.

### Lessons Learned

The interesting thing is that INT4's failures aren't uniform across task types. Repetition and fluency tasks (creative writing) broke hard. Precision tasks (code) broke on the edges (test harness, not the algorithm itself). But structure\
d reasoning held up perfectly. This makes intuitive sense — reasoning follows a fairly constrained chain of tokens where each step has high conditional probability, so the distribution shift doesn't push it off track. Creative writing ha\
s much flatter distributions where INT4's probability mass shift is more likely to land on a repetition attractor.

The other thing worth noting: INT4 hallucination is qualitatively _different_ from FP16 hallucination, not just worse. FP16 and INT8 both confidently give the same wrong scientist name — they're sampling from the same region of the distr\
ibution. INT4 samples from a different region entirely and produces a confused narrative. That's the +7.8% perplexity in action — the distribution has shifted enough that you're not getting a slightly degraded version of the same output,\
 you're getting a fundamentally different generation path.

Full prompt-by-prompt breakdown with raw responses in `step3_qualitative_analysis.md` and `step3_results.json`.

# Step 4

## Step 4 — Connection Check

This is synthesis, not new experiments. The goal is to connect today's quality findings (Steps 1–3) back to the Day 11 performance data and the roofline model from Day 2.

### Do the qualitative failure modes align with the benchmark results?

Not in the way the syllabus hints at. INT4-GPTQ's quality failures were in creative writing (repetition loops), factual Q&A (divergent hallucination), and code (syntax error) — all decode-time generation failures. But INT4's Day 11 _deco\
de-heavy_ benchmarks actually looked great: 540 tok/s at concurrency 8 vs FP16's 298 tok/s, with lower ITL p99 (17.5ms vs 31.2ms). The model was generating faster and more consistently — it was just generating worse content. This is an i\
mportant distinction: throughput and latency measure how fast tokens come out, not whether the tokens are right. INT4's quality degradation is invisible to the serving infrastructure.

On the prefill side, INT4's TTFT p99 didn't show more variance than FP16 — it was actually comparable or better at most concurrency levels (e.g., 215ms vs 190ms at conc=8, 350ms vs 333ms at conc=16). So the quality degradation I saw in S\
tep 3 doesn't correlate with prefill instability. The degradation is in the model weights themselves, not in how the serving system handles the workload.

### Does the perplexity delta feel proportional to the decode throughput gain?

INT4-GPTQ: +7.8% perplexity, +81% decode throughput at conc=8 (540 vs 298 tok/s). That's a lot of throughput for a meaningful but not catastrophic quality hit. Whether it's "worth it" depends entirely on the use case — for a chatbot doin\
g creative writing, the repetition loops make INT4 a non-starter. For a summarization pipeline where reasoning accuracy matters more than fluency, INT4's 0/2 reasoning failures suggest it might be fine.

INT8-AWQ: +0.17% perplexity, +34% decode throughput at conc=8 (399 vs 298 tok/s). This is the easy call — essentially free quality with a solid throughput bump. The perplexity increase is noise-level, the qualitative eval showed zero deg\
radation, and you get a meaningful speed improvement.

### Does INT8 helping decode more than prefill make sense?

Yes, and this is where the roofline model ties everything together. From Day 11, INT8-AWQ's gains were much larger on decode-heavy workloads (~34% throughput increase at conc=8) than on prefill-heavy workloads (where it was actually inco\
nsistent — it even showed an anomalous TTFT spike at conc=8). This is exactly what the roofline model predicts:

- **Decode is memory-bandwidth-bound.** Each decode step reads the full model weights to produce one token. INT8 cuts the bytes read per step roughly in half, so decode throughput scales almost linearly with the compression ratio. On T4 \
  with 320 GB/s bandwidth, halving the weight reads directly translates to faster token generation.
- **Prefill is compute-bound.** Prefill processes many tokens in parallel — it's doing large matrix multiplications that saturate the GPU's compute units, not the memory bus. Reducing weight size doesn't help much because the bottleneck \
  is FLOPS, not memory bandwidth. INT8's prefill gains are modest because the operation was already compute-limited.

### The through-line: hardware constraints → quantization mechanism → observed behavior

The T4's roofline has two regimes: memory-bound (low arithmetic intensity, like decode) and compute-bound (high arithmetic intensity, like prefill). Quantization reduces model weight size, which shifts the memory-bound regime's ceiling u\
pward — more tokens per second per byte of bandwidth. This is why INT8 and INT4 show their biggest throughput gains on decode-heavy workloads. But quantization also shifts the model's probability distribution (measured as +0.17% perplexi\
ty for INT8, +7.8% for INT4), and that distributional shift manifests as task-dependent quality degradation: invisible for constrained tasks like arithmetic reasoning, but severe for open-ended generation where the model explores flatter\
 regions of the distribution. The hardware story and the quality story are two sides of the same coin — you're trading bits of weight precision for bits of memory bandwidth, and the quality cost of that trade is non-uniform across tasks.

# AFTERNOON

## Step 5 -- Memory Footprint + KV Concurrency

### How I measured this

I loaded each model one at a time into vLLM (v0.17.1) on the T4, waited for the server to finish starting up, then grabbed the memory numbers from vLLM's startup logs. The "model loading took X GiB" and "Available KV cache memory: Y GiB"\
 lines give you exactly what you need. I also ran `nvidia-smi` after each load to confirm total GPU allocation, then killed the server before loading the next model.

All three runs used the same flags:

- `--gpu-memory-utilization 0.90` (vLLM's default, lets it use up to 90% of GPU memory)
- `--max-model-len 8192`
- `--enforce-eager` (turns off CUDA graph capture, which eats extra memory on T4)
- `--dtype float16`

### Commands

**FP16:**

```bash
source /home/ssm-user/venv-vllm/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enforce-eager \
  --port 8000
```

**INT8-AWQ:**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/ssm-user/day11/qwen2.5-3b-int8-awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enforce-eager \
  --port 8000
```

**INT4-GPTQ:**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enforce-eager \
  --port 8000
```

After each server printed "Application startup complete", I ran `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` and then shut it down.

### Memory footprint

These are empirical values straight from vLLM's engine logs, not theoretical estimates:

```
Precision    Model Size (GiB)   Free for KV (GiB)   nvidia-smi Used (MiB)
FP16         5.79               6.73                 14,089
INT8-AWQ     3.23               9.29                 14,093
INT4-GPTQ    1.94               10.58                14,057
```

The nvidia-smi column looks almost identical for all three (~14 GiB) because vLLM pre-allocates 90% of GPU memory no matter what. The difference is how that 90% gets split: smaller model means more room for KV cache. That shows up in the\
 "Free for KV" column.

### KV cache concurrency calculation

Architecture parameters (from `config.json`):

- `num_hidden_layers` = 36
- `num_key_value_heads` = 2 (GQA)
- `head_dim` = 128 (hidden_size 2048 / num_attention_heads 16)
- KV cache dtype = FP16 (2 bytes per element)

```
kv_bytes_per_token = 2 (K+V) x 36 (layers) x 2 (kv_heads) x 128 (head_dim) x 2 (bytes) = 36,864 bytes (36 KiB)
kv_per_request     = kv_bytes_per_token x seq_len
max_concurrent     = floor(free_for_kv_bytes / kv_per_request)
```

```
Precision    Model Size (GiB)   Free for KV (GiB)   Max Conc @ 2K   Max Conc @ 4K   Max Conc @ 8K
FP16         5.79               6.73                 95           47              23
INT8-AWQ     3.23               9.29                 132          66              33
INT4-GPTQ    1.94               10.58                150          75              37
```

As a sanity check, vLLM's own concurrency estimates at 8K were 23.94x (FP16), 33.04x (INT8-AWQ), 37.62x (INT4-GPTQ). Those match our floor calculations. The fractional part comes from vLLM's block-level allocation granularity.

### What this means

INT8-AWQ cuts model size by 2.56 GiB (5.79 to 3.23, a 44% reduction). That freed memory goes straight to KV cache. At 4K sequence length, max concurrent requests goes from 47 to 66, a 40% increase. At 8K: 23 to 33, a 43% increase.

INT4-GPTQ cuts model size by 3.85 GiB (5.79 to 1.94, a 66% reduction). At 4K sequence length, max concurrent requests goes from 47 to 75, a 60% increase. At 8K: 23 to 37, a 61% increase.

> "INT8-AWQ reduces model size by 2.56 GiB, freeing 2.56 GiB for KV cache. This increases max concurrent requests at 4K sequence length from 47 to 66, a 40% improvement. Quantization's primary production value is capacity headroom, not raw throughput."

### Mental model progression

- **Naive:** "INT8 is faster"
- **Mid:** "INT8 reduces memory bandwidth pressure, so decode gets faster"
- **Staff:** "INT8 frees KV cache, which means more concurrency, which means lower $/token"

The naive framing isn't wrong, it just misses the bigger picture. On T4, INT8-AWQ's decode throughput gain was about 34% (Day 11 data). That's nice, but not transformative. The capacity gain (40% more concurrent users at 4K, 43% at 8K) is the real production lever. A 3B model on T4 is tight enough on memory that every GiB you free from weights turns into almost exactly 1 GiB of KV headroom. At higher concurrency, throughput per user might stay flat, but the system's total throughput goes up because you're serving more users at once. That's how quantization reduces $/token: not by making individual requests faster, but by fitting more requests on the same GPU.

## Step 6 -- $/Million Tokens Cost Model

### GPU cost

Instance: g4dn.xlarge (NVIDIA T4, 16 GiB)

- On-demand: $0.526/hr
- Spot: ~$0.16/hr (variable)

I'm using on-demand pricing for the cost model. Spot is cheaper but not guaranteed, so on-demand is the right conservative baseline for production capacity planning.

### Throughput data

Output throughput numbers come from the Day 11 decode-heavy benchmarks at concurrency=8 (`/home/ssm-user/day11/benchmark_results/`). I'm using output tokens/sec specifically because output tokens are the work product you're paying for in a serving scenario.

### Cost calculation

```
tokens_per_hour   = output_throughput_tok_per_sec x 3600
cost_per_M_tokens = (hourly_rate_usd / tokens_per_hour) x 1,000,000
```

```
Precision    Output Throughput @ conc=8 (tok/s)   Tokens/hr     $/M tokens
FP16         264.90                                953,640       $0.552
INT8-AWQ     355.12                                1,278,432     $0.411
INT4-GPTQ    480.86                                1,731,096     $0.304
```

### Cost impact

> "INT8-AWQ reduces serving cost from $0.552/M tokens to $0.411/M tokens, a 25.4% reduction, with +0.17% perplexity increase."

> "INT4-GPTQ reduces serving cost from $0.552/M tokens to $0.304/M tokens, a 44.9% reduction, with +7.8% perplexity increase."

The cost reduction comes from two things that compound:

1. **Throughput gain** -- INT8 generates tokens about 34% faster per request (memory bandwidth savings on decode)
2. **Capacity gain** -- INT8 fits about 40% more concurrent requests (smaller model, more KV headroom)

At concurrency=8, only the throughput effect shows up in the numbers above. At higher concurrency where FP16 would run out of KV cache and start queueing, the capacity effect would widen the gap further. INT8-AWQ can sustain 33 concurrent requests at 8K seq_len where FP16 tops out at 23.

### Spot pricing comparison

At spot rates (~$0.16/hr), same throughput numbers give:

```
Precision    $/M tokens (spot)
FP16         $0.168
INT8-AWQ     $0.125
INT4-GPTQ    $0.092
```

All three are pretty cheap at spot rates. Even FP16 at $0.17/M output tokens is competitive with hosted API pricing for a 3B model. The quantization savings still matter in absolute terms ($0.076/M cheaper for INT8 vs FP16), but the relative reduction stays the same (25.4% / 44.9%) regardless of the hourly rate. It's a function of throughput, not price.

# Step 7 -- Complete Tradeoff Table + Production Recommendation

## Tradeoff Table

All numbers are empirical measurements from T4 (g4dn.xlarge, 16 GiB) unless marked `(calc)`.

```
Metric                                    FP16        INT8-AWQ     INT4-GPTQ
Model size (GiB)                          5.79        3.23         1.94
Free memory for KV (GiB)                  6.73        9.29         10.58
Max concurrent @ 4K seq (calc)            47          66           75
Throughput, decode-heavy, conc=8 (tok/s)  264.90      355.12       480.86
Throughput, prefill-heavy, conc=8 (tok/s) 222.49      128.41*      343.21
TTFT p99, decode-heavy, conc=8 (ms)       101.1       95.3         114.8
ITL p99, decode-heavy, conc=8 (ms)        31.2        23.6         17.5
$/M tokens, conc=8                        $0.552      $0.411       $0.304
Perplexity (WikiText-2)                   11.66       11.68        12.57
HellaSwag acc_norm (0-shot, 25%)          64.20%      64.20%       63.36%
Qualitative failures observed             (baseline)  0/10         5/10
```

\*INT8-AWQ prefill-heavy throughput is anomalously low (128.41 tok/s vs FP16's 222.49). This is the same anomalous TTFT spike noted in Step 4 (9,096ms TTFT p99 on prefill-heavy at conc=8). Likely a vLLM `compressed-tensors` kernel issue on T4, not representative of INT8-AWQ's actual prefill capability. All other INT8-AWQ metrics are consistent and healthy.

## Production Recommendation

### A. Failure Budget

I'm adopting a conservative threshold of ≤2% perplexity increase and ≤3% accuracy drop as a proxy for acceptable quality degradation. These thresholds are assumptions, not derived from this data. In production they'd come from user research or SLAs.

INT8-AWQ: +0.17% perplexity, 0% accuracy drop, 0/10 qualitative failures, 25.4% cost reduction.
Acceptable. Comfortably within both thresholds. Zero observable degradation across every evaluation method I ran. This is the easy recommendation.

INT4-GPTQ: +7.8% perplexity, -1.31% accuracy drop, 5/10 qualitative failures, 44.9% cost reduction.
Not acceptable as a general-purpose deployment. The +7.8% perplexity blows past the 2% threshold by nearly 4x. The HellaSwag accuracy drop (1.31%) technically passes the 3% accuracy threshold, but that's misleading. The qualitative eval tells the real story: 5 out of 10 prompts degraded, including 100% failure on creative writing. The accuracy benchmark didn't catch this because HellaSwag tests commonsense completion, not fluency or generation quality. This is exactly why you need both quantitative benchmarks and qualitative spot checks.

### B. Derived Use Case Mapping

Based on my Step 3 observation counts:

- INT8-AWQ is acceptable for all tested use cases (Q&A, creative writing, code generation, reasoning, long-context). 0/10 failures across the board, identical acc_norm to FP16. I'd deploy this as a drop-in replacement for FP16 without qualification.
- INT4-GPTQ is acceptable for structured reasoning tasks only. Both multi-step reasoning prompts (2/2) produced identical correct outputs to FP16. The math and logic chains survived quantization.
- INT4-GPTQ is not acceptable for creative writing (2/2 failed, severe repetition loops), code generation (1/2 failed, syntax errors in test harness), factual Q&A (1/2 failed, divergent hallucination), or long-context comprehension (1/2 failed, mischaracterized source material).

The pattern: INT4 holds up on tasks with constrained token distributions (reasoning, where each step has high conditional probability) but breaks on tasks with flatter distributions (creative writing, open-ended generation) where the shifted probability mass lands on repetition attractors or divergent paths.

### C. T4 Hardware Caveat

Results obtained on NVIDIA T4 (g4dn.xlarge, 16 GiB). AWQ kernel optimizations primarily target A100/H100 architecture. The syllabus predicted INT8 decode speedup on T4 would be modest (~10-15% rather than ~30-50% on A100). I actually measured ~34% decode throughput gain, which is better than expected for T4, though this may partly reflect vLLM-specific optimizations rather than pure hardware scaling.

The INT8-AWQ prefill-heavy anomaly (9,096ms TTFT p99 at conc=8) is almost certainly a T4-specific `compressed-tensors` kernel issue, not a fundamental INT8 limitation. Don't generalize that data point.

The capacity argument (model size to KV headroom to concurrency) is hardware-agnostic and generalizes to any GPU. The throughput numbers are T4-specific and would look different on A100/H100.

```
Day 11 decode-heavy, conc=8 reference:
Precision    Output tok/s   Total tok/s   TPOT (ms)
FP16         264.90         298.02        30.07
INT8-AWQ     355.12         399.51        22.41
INT4-GPTQ    480.86         540.97        16.47
```

### D. Mental Model Progression

Naive framing: INT8 is faster.

Mid-level framing: INT8 reduces bytes read per decode step, improving memory-bound decode throughput.

Staff-level framing: INT8 reduces model footprint, freeing KV cache capacity, increasing max concurrency, and directly lowering $/token. Quantization is a capacity lever, not just a performance optimization. On T4 with Qwen2.5-3B, INT8-AWQ's 34% throughput gain is nice, but the 40% increase in max concurrent users at 4K (47 to 66) is the number that actually changes your capacity plan and your cost model.

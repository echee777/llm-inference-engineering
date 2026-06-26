# Day 11 — Quantization Theory + Setup

**Week 3 · Day 1 | Theme: Precision as an inference systems lever**

> **Career anchor:** Every question you answer today with data — "does INT8 help decode more than prefill?" — is an interview question at Anthropic/OpenAI. You are not running benchmarks. You are building the empirical foundation for a recommendation you will defend with numbers.

---

## Ground Truth: What You Are Measuring

The curriculum specifies **three precision variants**:

| Variant       | Quantization        | Notes                      |
| ------------- | ------------------- | -------------------------- |
| FP16          | None (baseline)     | What you've been running   |
| **INT8-AWQ**  | AWQ, 8-bit weights  | Half the memory of FP16    |
| **INT4-GPTQ** | GPTQ, 4-bit weights | Quarter the memory of FP16 |

> ⚠️ **Do not substitute INT4-AWQ for INT8-AWQ.** The curriculum's choice is deliberate: it isolates two variables simultaneously — quantization method (AWQ vs GPTQ) and precision level (INT8 vs INT4). This lets Day 12's tradeoff analysis compare both dimensions.

---

## Mental Models to Lock In Before Running Anything

### 1. PTQ vs QAT — why it matters for deployment

**Post-Training Quantization (PTQ):** Quantize after training using calibration data. No retraining. Fast to deploy. This is what AWQ and GPTQ do, and it's what virtually every production inference platform uses for open-weight models.

**Quantization-Aware Training (QAT):** Simulate quantization noise during training itself. Better accuracy at extreme precision, but requires access to the training pipeline. Not relevant to you as a serving engineer unless you work on model teams.

**Interview framing:** "PTQ is the operator's tool. QAT is the model team's tool. As an inference engineer I care about PTQ."

---

### 2. Weight-only vs Weight+Activation — the runtime difference

**Weight-only (AWQ, GPTQ):**

```
Stored as INT4/INT8 → dequantize to FP16 → FP16 matmul
```

Memory savings come from storage. The matmul itself still runs in FP16.

**Weight + Activation (INT8 TensorCores, FP8 Hopper):**

```
Stored as INT8/FP8 → INT8 matmul on TensorCores
```

True compute savings. Requires hardware with integer TensorCore support. Harder to deploy without quality loss due to activation outliers.

**Why this matters for your T4:** T4 has limited INT8 TensorCore throughput. Expect weight-only AWQ and GPTQ to show memory-bandwidth gains in decode, but do not expect INT8 matmul compute gains. This sets a realistic hypothesis before you run.

---

### 3. Why quantization helps decode more than prefill — the roofline connection

You measured this in Days 1–2. Now you're going to see it in production:

| Phase       | Bottleneck                                                           | What quantization changes                                                              |
| ----------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Prefill** | Compute (parallel matmuls over many tokens)                          | Less weight memory, but FLOPs roughly unchanged for weight-only. Smaller benefit.      |
| **Decode**  | HBM bandwidth (reading weights once per token, batch=1 per sequence) | INT8 = half the bytes read. INT4 = quarter the bytes read. Direct latency improvement. |

**Arithmetic intensity reminder (Day 2):**

- Prefill: high arithmetic intensity → closer to compute ridge point
- Decode: ~1 FLOP/byte → deep in memory-bandwidth territory

Every byte of weight reduction hits decode directly. This is the hypothesis you are testing today.

**Interview answer you are building:** "Quantization's primary inference benefit is decode latency and throughput, not prefill. Prefill is compute-bound so reducing memory footprint doesn't proportionally improve speed. Decode is memory-bandwidth-bound so halving weight size directly reduces the time to read weights per token."

---

### 4. AWQ vs GPTQ — why two methods

**AWQ (Activation-Aware Weight Quantization):**

- Key insight: channels with large activations cause disproportionate quantization error when their weights are rounded
- AWQ finds these channels and scales them before quantization to protect the signal
- Fast to quantize (minutes), popular for production deployment
- Paper: https://arxiv.org/abs/2306.00978 — read Sections 1–3

**GPTQ (Second-Order Post-Training Quantization):**

- Key insight: minimize per-layer output reconstruction error using an approximation of the Hessian
- Weights that most strongly affect the layer output are quantized more carefully
- Slower to quantize than AWQ (hours for large models), but high quality at INT4
- Paper: https://arxiv.org/abs/2210.17323 — read Sections 1–3

**What to watch for in your results:** AWQ (INT8) vs GPTQ (INT4) comparison in Day 12's quality eval will show whether the more aggressive INT4 compression of GPTQ costs quality vs AWQ's INT8 precision. That's the tradeoff table frontier labs care about.

---

## Morning Block (4 hrs) — Reading + Model Preparation

### Step 1: Reading (1.5 hrs)

Work through:

1. Quantization fundamentals: PTQ vs QAT, weight-only vs weight+activation (use your mental models above as a pre-read frame)
2. AWQ paper Sections 1–3: focus on the activation-channel insight, skip implementation details
3. GPTQ paper Sections 1–3: focus on the layer-by-layer Hessian framing, skip the math derivation

Target: after reading, you can explain both methods in 2 sentences each without looking at notes.

---

### Step 2: Model Preparation (2.5 hrs)

**Find pre-quantized checkpoints on HuggingFace:**

```bash
# Search for these (exact repo names may vary):
# Qwen2.5-3B-Instruct-AWQ        → INT8 or INT4, pick INT8 if available
# Qwen2.5-3B-Instruct-GPTQ-Int4  → INT4

# Common publishers: Qwen official org, Qwen2.5 community repos
# Avoid TheBloke for Qwen2.5 — check the Qwen org first
```

**Verify each model loads cleanly:**

```python
from vllm import LLM

# FP16 baseline
llm = LLM("Qwen/Qwen2.5-3B-Instruct")
print("FP16 OK")
del llm

# INT8-AWQ
llm = LLM("<awq-checkpoint>", quantization="awq")
print("INT8-AWQ OK")
del llm

# INT4-GPTQ
llm = LLM("<gptq-checkpoint>", quantization="gptq")
print("INT4-GPTQ OK")
del llm
```

**Record model footprint immediately after each load:**

```bash
# After each LLM() call, in a second terminal:
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

````
| Metric | FP16 | INT8-AWQ | INT4-GPTQ |
|--------|------|----------|-----------|
| Checkpoint size on disk (GB)     | | | |
| VRAM used after load (GB) |      | | |
| Theoretical ratio vs FP16 | 1.0× | ~2.0× | ~4.0× |
| Measured ratio vs FP16 | —       | | |
`

> If your measured ratio deviates significantly from theory, understand why before moving on. Packing overhead, calibration data, or metadata can add 5–10%. Large deviations indicate a problem.

### INT8-AWQ Checkpoint Availability — Fallback Decision Tree

AWQ checkpoints in the wild are predominantly INT4. If you cannot find a Qwen2.5-3B INT8-AWQ checkpoint, do not substitute arbitrarily. Use this hierarchy to preserve experimental integrity:

**Option 1 (preferred): Self-quantize to INT8-AWQ using AutoAWQ — ~10 minutes**
```bash
pip install autoawq
````

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 8,          # ← INT8, not INT4
    "version": "GEMM"
}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./qwen2.5-3b-int8-awq")
```

Then load with `LLM("./qwen2.5-3b-int8-awq", quantization="awq")`.

**Option 2: Switch base model to one with INT8-AWQ checkpoints available**
Mistral-7B has broader INT8-AWQ community coverage. Note: 7B changes your VRAM math.

**Option 3 (last resort): Use INT4-AWQ instead, document the confound**
If you must use INT4-AWQ, you lose the clean precision-vs-method isolation. Document it explicitly:

> "Note: both INT8-AWQ and INT4-GPTQ vary both method and precision. INT8-AWQ checkpoint unavailable for this model. The Day 12 tradeoff analysis controls for this by comparing the precision dimension separately."

**What NOT to do:** Do not substitute `bitsandbytes` INT8 for INT8-AWQ. `bitsandbytes` uses mixed-precision outlier handling (LLM.int8()) — a completely different mechanism from activation-aware weight scaling. Swapping it in changes _both_ the quantization method _and_ the precision simultaneously, which is the same variable-conflation error as INT4-AWQ, just less obviously.

---

## Afternoon Block (4 hrs) — Throughput + Latency Benchmarks (Split Workloads)

### Step 3: Benchmark Design (1 hr)

**Why split workloads — the one thing interviewers check:**

A naive benchmarker runs one workload and reports one number. A systems engineer runs two workloads that isolate different bottlenecks. Your results will show that quantization helps decode more than prefill — but only if you designed the benchmark to expose that.

**Two profiles:**

| Profile       | Input tokens | Output tokens | Dominant phase                             | Expected quantization benefit              |
| ------------- | ------------ | ------------- | ------------------------------------------ | ------------------------------------------ |
| Prefill-heavy | 2,048        | 32            | Compute (parallel matmul)                  | Smaller (weight-only doesn't reduce FLOPs) |
| Decode-heavy  | 64           | 512           | Memory bandwidth (sequential weight reads) | Larger (fewer bytes read per token)        |

**Three metrics — all required:**

| Metric             | What it measures                                      | Why it matters                                    |
| ------------------ | ----------------------------------------------------- | ------------------------------------------------- |
| Throughput (tok/s) | Total tokens generated per second across all requests | System capacity                                   |
| TTFT p99 (ms)      | Time to first token at 99th percentile                | User-facing latency for interactive use           |
| ITL p99 (ms)       | Inter-token latency at 99th percentile                | Streaming smoothness; decode bottleneck indicator |

> Collapsing these into one number loses the ability to distinguish "fast prefill, slow decode" from "slow prefill, fast decode." Frontier labs ask about all three.

**Synthetic prompt construction — do this correctly:**

```python
import random

# BAD: "hello " * 2048 — one token repeated, no attention diversity
# GOOD: varied vocabulary approximating real prompts

def make_synthetic_prompt(target_tokens: int, tokenizer) -> str:
    """Build a varied prompt that hits the target token count."""
    words = [
        "the", "model", "generates", "text", "by", "predicting", "next",
        "token", "given", "context", "attention", "weights", "compute",
        "memory", "cache", "batch", "sequence", "layer", "embedding",
        "inference", "latency", "throughput", "request", "response",
        "system", "user", "assistant", "language", "neural", "network",
    ]
    prompt = ""
    while True:
        chunk = " ".join(random.choices(words, k=20))
        candidate = prompt + " " + chunk
        if len(tokenizer.encode(candidate)) >= target_tokens:
            break
        prompt = candidate
    # Trim to exact target
    tokens = tokenizer.encode(prompt)[:target_tokens]
    return tokenizer.decode(tokens)
```

---

### Step 4: Benchmark Script

Use vLLM's built-in benchmark tooling where possible — it handles concurrency, TTFT tracking, and percentile math correctly.

**Option A — vLLM's benchmark_serving.py (preferred for TTFT/ITL accuracy):**

```bash
# Start vLLM server first
python -m vllm.entrypoints.openai.api_server \
    --model <model_path> \
    --quantization awq \
    --port 8000

# Then in a second terminal, run benchmark
python benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:8000 \
    --model <model_path> \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 32 \
    --num-prompts 64 \
    --request-rate 8          # concurrency proxy; adjust per run
```

**Option B — offline LLM class (simpler, loses per-request TTFT):**

```python
from vllm import LLM, SamplingParams
import time, statistics, random

def benchmark(
    model_path: str,
    quantization: str | None,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    num_batches: int = 8,
    tokenizer=None,
) -> dict:

    llm = LLM(model=model_path, quantization=quantization)
    params = SamplingParams(max_tokens=output_tokens, temperature=0)

    # Build prompts
    prompts = [make_synthetic_prompt(input_tokens, tokenizer)
               for _ in range(concurrency)]

    # Warmup — one batch, not measured
    llm.generate(prompts, params)

    # Measured batches
    throughputs = []
    wall_times = []

    for _ in range(num_batches):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        end = time.perf_counter()

        elapsed = end - start
        total_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs
        )
        throughputs.append(total_output_tokens / elapsed)
        wall_times.append(elapsed)

    return {
        "throughput_p50": statistics.median(throughputs),
        "throughput_p99": sorted(throughputs)[int(0.99 * len(throughputs))],
        "wall_time_p99": sorted(wall_times)[int(0.99 * len(wall_times))],
        # Note: TTFT/ITL require AsyncEngine or serving endpoint for accuracy
    }
```

> **Measurement discipline:** Warmup once. Temperature=0 for reproducibility. Identical prompts across precision variants. Report medians, not means. ≥8 repetitions per cell.

---

### Step 5: Run the Full Matrix (3 hrs)

24 configurations: 3 precisions × 2 workloads × 4 concurrency levels.

**Fill this table as you go — this is your Day 12 and Day 15 input:**

```
| Precision | Workload      | Concurrency | Throughput tok/s | TTFT p99 ms | ITL p99 ms |
| --------- | ------------- | ----------- | ---------------- | ----------- | ---------- |
| FP16      | Prefill-heavy | 1           |                  |             |            |
| FP16      | Prefill-heavy | 4           |                  |             |            |
| FP16      | Prefill-heavy | 8           |                  |             |            |
| FP16      | Prefill-heavy | 16          |                  |             |            |
| FP16      | Decode-heavy  | 1           |                  |             |            |
| FP16      | Decode-heavy  | 4           |                  |             |            |
| FP16      | Decode-heavy  | 8           |                  |             |            |
| FP16      | Decode-heavy  | 16          |                  |             |            |
| INT8-AWQ  | Prefill-heavy | 1           |                  |             |            |
| INT8-AWQ  | Prefill-heavy | 4           |                  |             |            |
| INT8-AWQ  | Prefill-heavy | 8           |                  |             |            |
| INT8-AWQ  | Prefill-heavy | 16          |                  |             |            |
| INT8-AWQ  | Decode-heavy  | 1           |                  |             |            |
| INT8-AWQ  | Decode-heavy  | 4           |                  |             |            |
| INT8-AWQ  | Decode-heavy  | 8           |                  |             |            |
| INT8-AWQ  | Decode-heavy  | 16          |                  |             |            |
| INT4-GPTQ | Prefill-heavy | 1           |                  |             |            |
| INT4-GPTQ | Prefill-heavy | 4           |                  |             |            |
| INT4-GPTQ | Prefill-heavy | 8           |                  |             |            |
| INT4-GPTQ | Prefill-heavy | 16          |                  |             |            |
| INT4-GPTQ | Decode-heavy  | 1           |                  |             |            |
| INT4-GPTQ | Decode-heavy  | 4           |                  |             |            |
| INT4-GPTQ | Decode-heavy  | 8           |                  |             |            |
| INT4-GPTQ | Decode-heavy  | 16          |                  |             |            |
```

---

### What to Watch For — the Hypothesis Check

After filling the table, answer these before writing up:

**Roofline validation:**

```
decode_throughput_speedup = (INT8_decode_tok_s) / (FP16_decode_tok_s)
prefill_throughput_speedup = (INT8_prefill_tok_s) / (FP16_prefill_tok_s)

Expected: decode_speedup > prefill_speedup
```

If this holds: you've empirically confirmed memory-bandwidth-bound decode behavior with real model weights and real vLLM scheduling.

If it doesn't hold (decode speedup ≤ prefill speedup): diagnose before moving on. Possible causes on a T4:

- T4 INT8 kernel not activating (check vLLM logs)
- Model is small enough that overhead dominates at low concurrency
- Prefill chunking behavior hiding the effect

**ITL vs TTFT split:**

- Quantization should compress ITL more than TTFT for decode-heavy workloads (fewer bytes per weight read = faster per-token generation)
- If TTFT improves substantially for prefill-heavy workloads, that's unexpected — note it

---

## End-of-Day Outputs

Checklist before closing:

- [ ] All 3 model variants load cleanly in vLLM
- [ ] VRAM footprint table complete with measured vs theoretical ratios
- [ ] Full 24-cell benchmark matrix filled
- [ ] Roofline hypothesis confirmed or disconfirmed (with diagnosis if disconfirmed)
- [ ] 2–3 sentence written observation: "INT8-AWQ improved decode throughput by X% vs FP16, but improved prefill throughput by only Y%. This is consistent with decode being memory-bandwidth-bound (fewer bytes read per weight) while prefill is compute-bound. INT4-GPTQ showed Z% additional improvement over INT8-AWQ in decode, at the cost of [quality observations on Day 12]."

---

## Self-Test Gate — Answer From Memory Before Tomorrow

1. Why does weight-only quantization help decode more than prefill? (Connect to arithmetic intensity and roofline.)
2. What is the runtime execution path for INT4-GPTQ weights in vLLM? (Hint: dequantize → FP16 matmul, not INT4 matmul.)
3. What is the key difference between AWQ and GPTQ's quantization strategy?
4. If INT8-AWQ gives 2× memory reduction, what is the theoretical maximum decode throughput improvement on a memory-bandwidth-bound system?
5. You see decode ITL improve by 40% with INT8-AWQ. A colleague says "great, let's use INT4-GPTQ everywhere." What do you ask before agreeing?

---

## KV-Capacity Preview — Why Quantization's Real Value Is Concurrency

Speed improvements are real, but they're not why frontier labs quantize in production. The deeper reason is concurrency capacity: smaller weights leave more VRAM for KV cache, which means more simultaneous requests, which means higher throughput under real multi-tenant load.

Fill this table using your VRAM measurements from Step 2. The math uses your Day 3 KV cache block formula.

**T4 VRAM budget (16 GB):**

```
KV cache available = Total VRAM × gpu_memory_utilization − model weights
KV bytes per token = 2 layers × n_heads × head_dim × 2 bytes (FP16) × num_layers
```

For Qwen2.5-3B: 28 layers, 16 KV heads, head_dim=128. KV bytes per token ≈ 28 × 16 × 128 × 2 × 2 = 229,376 bytes ≈ 0.22 MB/token.

| Precision | Model weights (GB) | Free for KV @ 0.90 util (GB) | Max tokens in KV  | Max concurrent @ 4K seq |
| --------- | ------------------ | ---------------------------- | ----------------- | ----------------------- |
| FP16      | (measure)          | (16 × 0.90) − weights        | free_GB / 0.00022 | max_tokens / 4096       |
| INT8-AWQ  | (measure)          |                              |                   |                         |
| INT4-GPTQ | (measure)          |                              |                   |                         |

> This is the table Day 12 builds on. If you fill it now, tomorrow you're validating against live measurements, not computing from scratch.

**Interview framing this builds:** "We chose INT8-AWQ because it increased our max concurrent 4K-token requests from N to M — a 2× capacity improvement — with less than 1% perplexity increase. Decode throughput improved by ~30% as a secondary benefit. The primary driver was capacity, not speed."

---

## Forward Links

```
| Day                      | Depends on today's output                                                      |
| ------------------------ | ------------------------------------------------------------------------------ |
| **Day 12 (tomorrow)**    | VRAM table → KV capacity calculation; throughput table → $/M tokens cost model |
| **Day 15 (deliverable)** | Full benchmark matrix → Quantization Tradeoff Analysis document                |
| **Phase A Exit Q4**      | "INT8 affects decode vs prefill differently — why?"                            |
| **Interview**            | "Recommend a precision for our use case" — you answer with data, not opinion   |
```

---

## Errata — Deviations from Revised Syllabus

```
| Item               | Revised syllabus               | This version                                                | Why                                                                                    |
| ------------------ | ------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Precision variants | INT4-AWQ, INT4-GPTQ            | **INT8-AWQ, INT4-GPTQ**                                     | Curriculum specifies INT8-AWQ; losing INT8 loses the INT8 vs INT4 comparison on Day 12 |

| Benchmark script   | Returns single wall-clock time | Returns throughput, notes TTFT/ITL require serving endpoint | Curriculum requires TTFT p50/p95/p99 and ITL p50/p95/p99 separately                    |

| Synthetic prompt   | `"hello " * prompt_tokens`     | Varied vocabulary construction                              | Repeated single token produces unrealistic KV cache patterns and attention behavior    |

| Hypothesis framing | Stated once upfront            | Woven through mental models + roofline connection           | Roofline was measured empirically on Day 2; this experiment is the payoff              |
```

---

## Reviewer Assessment — What to Accept vs. Reject

A second LLM reviewed v2 of this document. Decision log:

| Reviewer claim                                               | Accept?      | Reasoning                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INT8-AWQ correct (not INT4-AWQ)                              | ✅ Confirmed | Validates curriculum design; don't change                                                                                                                                                                                                                                                                                                                                   |
| Synthetic prompt fix correct                                 | ✅ Confirmed |                                                                                                                                                                                                                                                                                                                                                                             |
| Metric separation (TTFT/ITL/throughput) correct              | ✅ Confirmed |                                                                                                                                                                                                                                                                                                                                                                             |
| Roofline integration strong                                  | ✅ Confirmed |                                                                                                                                                                                                                                                                                                                                                                             |
| INT8-AWQ checkpoints are rare → use bitsandbytes as fallback | ❌ Reject    | **bitsandbytes INT8 uses a different quantization mechanism (outlier-handling mixed precision) vs AWQ (activation-aware weight scaling). Substituting it changes both precision and method simultaneously — the same variable-conflation error the reviewer correctly identified earlier. Correct fallback: self-quantize with AutoAWQ to INT8, or document the confound.** |
| KV-capacity framing is "missing"                             | ⚠️ Partial   | It's not missing — it's Day 12's core content (curriculum is explicit). But a preview table in Day 11 bridges the two days usefully. Added above as a preview, not a full analysis.                                                                                                                                                                                         |

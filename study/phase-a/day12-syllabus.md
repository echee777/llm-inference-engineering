# Day 12 — Quality + Capacity + Cost

**Week 3 | Phase A**  
**Theme:** Close the loop on quantization. Move from benchmark runner to decision-maker. By end of day you have a single tradeoff table — quality + performance + capacity + cost — that you can defend number by number, with an explicit production recommendation backed by your own data.

**Prerequisite:** Day 11 benchmark results in hand — throughput and latency numbers for FP16 / INT8-AWQ / INT4-GPTQ across prefill-heavy and decode-heavy workloads at concurrency 1, 4, 8, 16.

---

## Morning Block (4 hrs) — Quality Evaluation

### Step 1 — Perplexity on WikiText-2 (45 min)

Perplexity is your quantitative, distribution-level quality signal. Run for all three precisions:

```bash
pip install lm-eval
lm_eval --model vllm \
  --model_args pretrained=<model-path>,dtype=float16 \
  --tasks wikitext \
  --device cuda
# Repeat for INT8-AWQ and INT4-GPTQ model paths
```

Record one number per precision. Lower = better. Expected ordering: FP16 < INT8 < INT4.

**Framing to internalize:** Perplexity is a distribution-level signal — it captures how much the model's probability mass has shifted. It can move without immediate user-visible impact. That's why you also need Step 2.

---

### Step 2 — Downstream Task Eval (1 hr 15 min)

Pick **one** task: HellaSwag, ARC-Easy, or a MMLU subset (5-shot). Don't run all three — you'll burn the morning.

```bash
lm_eval --model vllm \
  --model_args pretrained=<model-path> \
  --tasks arc_easy \
  --num_fewshot 0 \
  --device cuda
```

Run for all three precisions. Record accuracy scores.

**Framing to internalize:** Task accuracy is a decision-level signal — it captures whether the model's output crosses a correctness threshold that a user would actually notice. You need both perplexity and task accuracy because a model can have elevated perplexity but still pass accuracy thresholds, or vice versa. Running both lets you say: "distribution shifted, but decision-level impact was minimal" or "distribution shifted AND accuracy dropped — this is user-visible."

---

### Step 3 — Qualitative Error Analysis (1 hr)

**Highest ROI exercise of the morning. Build taste, not a benchmark suite.**

Construct exactly 10 prompts, two from each category:

- Q&A (factual)
- Creative writing
- Code generation
- Multi-step reasoning
- Long-context (paste a passage, ask a question about it)

Run all 10 through FP16, INT8-AWQ, and INT4-GPTQ with **identical sampling parameters** (`temperature=0`, `max_tokens=256`). Read responses side by side.

**What to flag:**

- Does INT4 hallucinate on factual Q&A?
- Does code have syntax errors or wrong logic?
- Does reasoning lose coherence mid-chain?
- Does creative writing become noticeably less coherent?

Record your observations with counts: "INT4 failed 3/4 code prompts (syntax error in 2, wrong logic in 1)." Counts matter — they're what you cite later when deriving the use case mapping.

**One hour, 10 examples, notes only.** No rabbit holes. The goal is to generate your own evidence base for which failure modes appear at which precision, so your deployment recommendation in Step 7 is derived, not imported from priors.

---

### Step 4 — Connection Check (1 hr)

Synthesis, not more experiments. Ask:

- Do your qualitative failure modes align with your Day 11 benchmark results? (e.g., if INT4 degrades on reasoning prompts — does TTFT p99 on prefill-heavy workloads also show more variance at INT4?)
- Does the perplexity delta feel proportional to the decode throughput gain from Day 11?
- Does INT8 helping decode more than prefill (from Day 11) make sense given that decode is memory-bound and INT8 reduces the bytes read per token?

Write 2–3 sentences explicitly connecting your quality observations to the roofline model from Day 2. This is the through-line: hardware constraints → quantization mechanism → observed behavior.

---

## Afternoon Block (4 hrs) — Capacity Impact + Cost Model

### Step 5 — Memory Footprint + KV Concurrency (1.5 hrs)

Load each model into vLLM and immediately record actual GPU memory from `nvidia-smi`:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

Record model weight size in GB (memory used before any requests arrive). Then compute available KV cache and max concurrency:

```
free_for_kv_GB   = gpu_total_GB × gpu_memory_utilization − model_size_GB
kv_bytes_per_tok = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
kv_per_request   = kv_bytes_per_tok × seq_len
max_concurrent   = floor(free_for_kv_GB × 1e9 / kv_per_request)
```

Fill in from empirical measurements (not theoretical estimates — T4 is tight enough that the difference matters):

| Precision | Model Size (GB) | Free for KV (GB) | Max Conc @ 2K | Max Conc @ 4K | Max Conc @ 8K |
| --------- | --------------- | ---------------- | ------------- | ------------- | ------------- |
| FP16      |                 |                  |               |               |               |
| INT8-AWQ  |                 |                  |               |               |               |
| INT4-GPTQ |                 |                  |               |               |               |

**State the staff-level insight explicitly in your document:**

> "INT8-AWQ reduces model size by X GB, freeing Y GB for KV cache. This increases max concurrent requests at 4K sequence length from A to B — a Z% improvement. Quantization's primary production value is capacity headroom, not raw throughput."

**Mental model progression to write down explicitly:**

- Naive: "INT8 is faster"
- Mid: "INT8 reduces memory bandwidth pressure → faster decode"
- Staff: "INT8 frees KV cache → more concurrency → lower $/token"

This progression belongs in your deliverable. It signals depth in interviews.

---

### Step 6 — $/Million Tokens Cost Model (1.5 hrs)

Get your GPU's actual hourly cost. For g4dn.xlarge (T4): ~$0.526/hr on-demand, ~$0.16/hr spot. Use whichever you're actually paying.

For each precision at `concurrency=8`:

```
tokens_per_hour     = throughput_tok_per_sec × 3600
cost_per_M_tokens   = (hourly_rate_usd / tokens_per_hour) × 1_000_000
```

| Precision | Throughput @ conc=8 (tok/s) | Tokens/hr | $/M tokens |
| --------- | --------------------------- | --------- | ---------- |
| FP16      |                             |           |            |
| INT8-AWQ  |                             |           |            |
| INT4-GPTQ |                             |           |            |

**Write the operator framing:**

> "INT8-AWQ reduces serving cost from $X/M tokens to $Y/M tokens — a Z% reduction — with W% perplexity increase."

This is the sentence that belongs in your tradeoff analysis document and in any interview answer about optimization tradeoffs.

---

### Step 7 — Complete Tradeoff Table + Production Recommendation (1 hr)

This is the Day 12 deliverable. Pull every measured number into a single table:

| Metric                                     | FP16       | INT8-AWQ | INT4-GPTQ |
| ------------------------------------------ | ---------- | -------- | --------- |
| Model size (GB)                            |            |          |           |
| Free memory for KV (GB)                    |            |          |           |
| Max concurrent @ 4K seq                    |            |          |           |
| Throughput — decode-heavy, conc=8 (tok/s)  |            |          |           |
| Throughput — prefill-heavy, conc=8 (tok/s) |            |          |           |
| TTFT p99, conc=8 (ms)                      |            |          |           |
| ITL p99, conc=8 (ms)                       |            |          |           |
| $/M tokens, conc=8                         |            |          |           |
| Perplexity (WikiText-2)                    |            |          |           |
| Downstream task accuracy                   |            |          |           |
| Qualitative failures observed              | (baseline) |          |           |

Leave nothing blank. If a cell is theoretically derived rather than directly measured, note it with `(calc)`.

---

#### Production Recommendation Section

After the table, write the following four sub-sections explicitly.

**A. Failure Budget**

State your quality threshold as an explicit assumption (not derived from this data — it would be circular; in production this comes from user research or SLAs):

> "Adopting a conservative threshold of ≤2% perplexity increase and ≤3% task accuracy drop as a proxy for acceptable quality degradation. This threshold requires validation against real user feedback before production use."

Then apply it per precision:

```
INT8-AWQ:  +W% perplexity, −Z% cost
           → Acceptable / Not acceptable (state which, with justification)

INT4-GPTQ: +W% perplexity, −Z% cost
           → Acceptable / Not acceptable (state which, with justification)
```

**B. Derived Use Case Mapping**

Derived from your Step 3 observation counts and Step 2 accuracy data — not from priors:

```
Based on observed failures:
- [precision] → acceptable for [use case] because [your data]
- [precision] → required for [use case] because [your data, e.g., "INT4 failed 3/4 code prompts"]
```

Do not write this table before running Step 3. It must be populated from your evidence, not from prior knowledge.

**C. Hardware Caveat (T4-specific)**

> "Results obtained on NVIDIA T4 (g4dn.xlarge, 16GB). AWQ kernel optimizations are primarily targeting A100/H100 architecture. Observed INT8 decode speedup on T4 may be lower than expected (~10–15% rather than ~30–50% on A100). Results — especially throughput gains — may not generalize to A100/H100. The capacity argument (model size → KV headroom → concurrency) is hardware-agnostic and generalizes."

**D. Mental Model Progression (one paragraph)**

State the three-tier framing in your own words:

> "Naive framing: INT8 is faster. Mid-level framing: INT8 reduces bytes read per decode step, improving memory-bound decode throughput. Staff-level framing: INT8 reduces model footprint, freeing KV cache capacity, increasing max concurrency, and directly lowering $/token — quantization is a capacity lever, not just a performance optimization."

---

## End-of-Day Output

A single document containing:

1. Quality evaluation results (perplexity, task accuracy, qualitative observation counts)
2. Complete tradeoff table (all cells filled)
3. Production recommendation with: failure budget (with explicit threshold assumption), derived use case mapping, T4 hardware caveat, mental model progression

This document is the foundation of the Week 3 deliverable (Quantization & Optimization Tradeoff Analysis, due Day 15).

---

## Key Mental Models

| Concept                               | One-sentence form                                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Quantization ≠ just speed             | Smaller model → more KV headroom → more concurrent users                                                 |
| INT8 helps decode more than prefill   | Decode is memory-bound; INT8 reduces bytes read. Prefill is compute-bound; gains are smaller.            |
| Quality degradation is not uniform    | INT4 fails more on precision-requiring tasks (code, reasoning) than fluency tasks (creative writing)     |
| $/M tokens is the operator metric     | Throughput improvement has a dollar value you can calculate and state with precision                     |
| Perplexity vs. task accuracy          | Perplexity = distribution-level signal. Task accuracy = decision-level signal. Run both.                 |
| Failure budgets need external anchors | Thresholds come from user research or SLAs — you cannot derive them from the same data you're evaluating |

---

## Watch-Outs for T4 Setup

- **INT8 kernel performance:** vLLM's AWQ INT8 kernels are optimized for A100/H100. On T4, INT8 throughput gains may be modest. If INT8 decode speedup is ~10% rather than ~30%, that's a hardware finding, not a quantization failure — state it as such.
- **Memory is tight:** Qwen2.5-3B FP16 takes ~7–8GB on T4. INT8-AWQ saves ~3.5GB. Use actual `nvidia-smi` readings for all capacity calculations, not theoretical values.
- **BF16 not supported:** T4 is compute capability 7.5. Use `--dtype half` (FP16), not BF16.
- **INT4-GPTQ checkpoint availability:** If a pre-quantized INT4-GPTQ checkpoint is unavailable for your model, use AutoGPTQ to quantize locally. Budget ~30 min for this.

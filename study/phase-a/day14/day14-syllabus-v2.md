# DAY 14 INTUITION

The main concepts for Day 14 are really these:

### 1. Speculative decoding is a **systems decision**, not a feature demo

You are not trying to prove “spec decode is cool.”
You are trying to answer:

- should we deploy it
- for this workload
- on this hardware
- at this concurrency level

So the whole day is about building a **decision memo**, not a benchmark scrapbook.

### 2. **Acceptance rate** is the core organizing variable

This is probably the single biggest concept.

You want to internalize:

- high acceptance rate → more draft tokens survive → bigger speedup
- low acceptance rate → lots of rejected work → little benefit or even regression

So acceptance rate is the bridge between:

- workload characteristics
- draft/target compatibility
- actual latency/throughput outcome

It is the main “why” metric.

### 3. Spec decode mainly helps **decode**, not prefill

This should connect back to everything you already learned.

The core mental model is:

- target-model decode is often **HBM / weight-loading bound**
- speculative decoding can let one expensive target pass validate multiple tokens
- that increases effective arithmetic intensity of decode

So the optimization target is the **decode path**.

### 4. It often helps most at **low concurrency**

Another very important Day 14 concept.

At low concurrency:

- the target model is underutilized during decode
- spec decode can fill that gap

At higher concurrency:

- continuous batching already raises utilization
- so the extra benefit of speculative decoding shrinks
- sometimes it can even hurt because of overhead

So one big lesson is:

> spec decode is not universally good; it is regime-dependent

### 5. **Continuous batching competes with speculative decoding**

This is a staff-level idea, not just a benchmark observation.

You want to learn that there are multiple ways to improve decode efficiency:

- continuous batching
- larger concurrency
- speculative decoding

And these are not independent.
If batching already keeps the GPU busy, speculative decoding may add less value.

### 6. **Draft-model compatibility is a real production constraint**

This is not just theory.

If the draft model and target model are poorly matched, then:

- acceptance rate drops
- speedup drops
- engineering complexity may not be worth it

And even more basic:

- tokenizer / vocab compatibility can be a hard blocker

So one of the Day 14 lessons is that:

> “Can we even do this cleanly with this draft model?”
> is a first-class engineering question.

That is why the syllabus emphasizes checking something like Qwen vs TinyLlama compatibility before you get too deep.

### 7. There is a real **VRAM / deployability tradeoff**

The draft model is not free.

It costs:

- extra VRAM
- extra serving complexity
- extra operational surface area

So you are learning to ask:

- does the speedup justify the memory cost?
- does the draft model reduce KV headroom too much?
- is the net deployability actually positive?

This is why the syllabus includes VRAM deltas and a deployability summary table.

### 8. Task type affects acceptance rate

This is another core concept.

Different workloads have different predictability:

- simple Q&A often has higher acceptance
- code may vary
- creative writing often has lower acceptance

So Day 14 is teaching you that speculative decoding performance is **workload-shaped**, not just model-shaped.

### 9. You need to think in terms of **production recommendation**

By the end, you should be able to say something like:

- deploy for low-concurrency factual chat
- do not deploy for high-concurrency serving where batching already saturates decode
- be cautious for creative workloads because acceptance may be too low
- do not deploy if tokenizer/model compatibility is messy or VRAM overhead is too costly

That kind of recommendation is the real output.

---

So if I compress Day 14 into the few concepts you are really supposed to come away with, it is this:

1. **Spec decode is a decode optimization**
2. **Acceptance rate determines whether it wins**
3. **It tends to help more at low concurrency than high concurrency**
4. **Continuous batching can reduce its marginal value**
5. **Draft-target compatibility is crucial**
6. **VRAM overhead matters**
7. **The final output is a deploy / don’t deploy decision by workload regime**

That’s the spine of the whole day.

A nice interview-grade one-liner would be:

> “Day 14 is really about learning that speculative decoding is not a blanket speedup trick; it is a conditional decode optimization whose value depends on acceptance rate, concurrency regime, draft-target compatibility, and memory overhead.”

# Day 14 (Thu) — Speculative Decoding

## AI Inference Platform Residency — Phase A, Week 3

### Syllabus v2 (Revised: incorporates Reviewer 1 + Reviewer 2 feedback)

---

## Change Log (v1 → v2)

```
#   Change                                                                                                  Source    Rationale
1   Reframed daily goal to production-decision framing, added "hardware budget"                             R1 + R2   Operator mindset, not feature tourism
2   VRAM cost column added to all benchmark tables                                                          R1 + R2   Memory is a first-class constraint on T4
3   Added deployability summary table as end-of-day artifact                                                R2        Scannable production recommendation format
4   Acceptance rate promoted to organizing variable with 3 explicit analysis questions                       R1 + R2   Core lever; without it, tables are incomplete
5   Added systems paragraph: arithmetic intensity → continuous batching → spec decode concurrency failure    R1 + R2   Closes curriculum thread from Day 1 microbenchmark
6   Success criterion redefined as decision memo, not setup success                                         R1 + R2   De-risks day from V1 integration flakiness
7   Compatibility failure elevated to first-class finding, not setup nuisance                               R2        More staff-level than "it worked" or "I used ngram"
```

---

## Daily Goal

> **Decide whether speculative decoding should be deployed for a target workload, hardware budget, and concurrency regime.**

This is not a feature benchmark. The output is a production decision memo — backed by your numbers — that a staff engineer could use to argue for or against deploying spec decode in a serving stack. Setup friction, compatibility failures, and fallback choices are data, not noise.

**Success criterion:** A defensible decision memo. Not "draft model path happened to work."

---

## Systems Framing (Read Before Starting)

Speculative decoding addresses a fundamental inefficiency: during decode, the target model processes one token per forward pass, leaving the GPU underutilized. Your Day 1 microbenchmark showed this directly — at batch=1, achieved TFLOPS was a small fraction of peak. The theoretical fix:

1. A small **draft model** autoregressively generates K candidate tokens cheaply
2. The **target model** verifies all K in a single forward pass (parallel → higher arithmetic intensity)
3. Accepted tokens are committed; first rejected token triggers rollback

The key insight: the target model's decode bottleneck is HBM weight loading, not compute. Verifying K tokens costs approximately the same as verifying 1, because weights are loaded once regardless. So if draft acceptance rate is high, you get K tokens for the cost of ~1.

### Why Concurrency Degrades This

At low concurrency (1–2 sequences), decode underutilization is real — spec decode addresses a genuine gap. But your Phase A work has shown that **continuous batching increases arithmetic intensity** by raising the effective batch size processed per decode step. At concurrency=8, the target model is handling 8 sequences per step — closer to the batch=64–256 range from your Day 1 curve, where GPU utilization is already high. The draft model's overhead now competes with genuinely useful target model work. This is the mechanism behind spec decode hurting at high concurrency: not because the algorithm changes, but because the underutilization gap it was filling no longer exists.

This connects directly to:

- Day 1: batch size → TFLOPS curve
- Day 2: arithmetic intensity → roofline → decode as memory-bound
- Day 6–9: continuous batching, scheduler behavior, batched token processing

Spec decode is a **low-concurrency, memory-bound-regime optimization**. Keep that frame for the entire day.

---

## Morning Block (4 hrs) — Concepts + Setup

---

### Step 1 — Read: Speculative Decoding (1 hr)

**What to understand (not memorize):**

- Draft model proposes K tokens; target model verifies in one parallel pass
- **Acceptance rate** is the primary variable. Everything else follows from it.
- Acceptance rate is determined by how well the draft distribution matches the target distribution — which is task-dependent
- Greedy/repetitive tasks (formulaic Q&A, structured code syntax) → higher acceptance
- Creative/unpredictable tasks → lower acceptance, spec decode may be net negative

**Key questions to hold while reading:**

1. What determines acceptance rate? (vocab overlap, task predictability, temperature)
2. What acceptance rate threshold is needed for spec decode to be net positive?
3. When does high acceptance fail to translate into speedup? (GPU already saturated, draft overhead dominates)

**Sources (skim, 1 hr total):**

- Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding" — Sections 1–3: https://arxiv.org/abs/2211.17192
- vLLM spec decode docs: https://docs.vllm.ai/en/latest/features/spec_decode.html

---

### Step 2 — Setup: Configure vLLM with a Draft Model (3 hrs)

#### VRAM Budget Check (do this first)

```bash
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

On T4 16GB with Qwen2.5-3B-Instruct:

```
Component                           Estimated VRAM
Target model (FP16)                 ~6 GB
Draft model TinyLlama-1.1B (FP16)  ~2.2 GB
KV cache (remaining)               ~7–8 GB
```

This is tight. Record actual measured values after loading — these will feed into your deployability summary table.

#### Compatibility Check (run before assuming it works)

TinyLlama and Qwen2.5 have **different tokenizers and vocab sizes**. vLLM V1 spec decode support is experimental. Before launching, check:

```bash
# Inspect tokenizer vocab sizes
python3 -c "
from transformers import AutoTokenizer
t1 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
t2 = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('Qwen vocab:', t1.vocab_size)
print('TinyLlama vocab:', t2.vocab_size)
print('Match:', t1.vocab_size == t2.vocab_size)
"
```

**Record this output.** If vocab sizes differ, draft-model spec decode may fail or silently produce incorrect behavior. This is not a setup nuisance — it is the lesson: **spec decode's theoretical gains are gated by cross-model compatibility, which is non-trivial in practice and often invisible until you test it.** Document what you find as a first-class finding.

#### Launch: Draft Model Path (attempt first)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-speculative-tokens 5 \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

If this fails (tokenizer mismatch, V1 incompatibility, OOM), **do not spend more than 30 minutes debugging**. Proceed immediately to the ngram fallback and document the failure as a compatibility finding.

#### Fallback: N-gram Speculative Decoding

No draft model required — uses n-gram matching from the prompt context:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --speculative-model [ngram] \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4 \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

N-gram fallback: VRAM overhead is negligible (no draft model weights), acceptance rate is lower and task-dependent, but the concurrency and task-type dynamics are the same. The decision table is still valid. **Document which path you're on and why.**

#### Verification

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

resp = client.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    prompt="The capital of France is",
    max_tokens=20,
    temperature=0
)
print(resp)
```

Check `localhost:8000/metrics` for `vllm:spec_decode_acceptance_rate` — if this metric appears, spec decode is active.

---

## Afternoon Block (4 hrs) — Experiments + Decision Memo

---

### Step 3 — Spec Token Sweep (1.5 hrs)

**Workload:** decode-heavy (64-token prompt, 256-token completion), concurrency=1, temperature=0

Test `num-speculative-tokens` ∈ {baseline/none, 3, 5, 7, 10}

```python
import time, statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
PROMPT = "Explain the history of the Roman Empire in detail. " + "history " * 20

def run_benchmark(n_requests=40):
    ttfts, itls, throughputs = [], [], []
    for _ in range(n_requests):
        start = time.perf_counter()
        first_token_time = None
        tokens = 0
        for chunk in client.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            prompt=PROMPT,
            max_tokens=256,
            temperature=0,
            stream=True
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if chunk.choices[0].text:
                tokens += 1
        end = time.perf_counter()
        ttfts.append((first_token_time - start) * 1000)
        total_time = end - start
        if tokens > 1:
            itls.append((total_time - (first_token_time - start)) / (tokens - 1) * 1000)
        throughputs.append(tokens / total_time)
    return {
        "ttft_p50": statistics.median(ttfts),
        "ttft_p99": sorted(ttfts)[int(0.99 * len(ttfts))],
        "itl_p50": statistics.median(itls),
        "throughput_mean": statistics.mean(throughputs)
    }
```

Check `localhost:8000/metrics` after each run for `vllm:spec_decode_acceptance_rate`.

**Record:**

```
Spec Tokens       Throughput (tok/s)   TTFT p50 (ms)   TTFT p99 (ms)   ITL p50 (ms)   Acceptance Rate   VRAM Delta (MB)
Baseline (none)                                                                        N/A               0
3
5
7
10
```

**Key observation:** There is a sweet spot. Too few spec tokens → overhead with minimal benefit. Too many → acceptance rate drops, wasted draft compute. The optimal K is task-dependent; this sweep finds your hardware's optimum for this workload.

---

### Step 4 — Task Type Sweep (1.5 hrs)

Fix `num-speculative-tokens` = best K from Step 3. Concurrency=1, temperature=0.

**Prompts (30 requests each):**

```python
tasks = {
    "qa":       "What is the boiling point of water at sea level? Answer in detail:",
    "creative": "Write a short story about a dragon who discovers jazz music.",
    "code":     "Write a Python function implementing binary search on a sorted list. Include docstring and type hints."
}
```

Also run each task at **baseline (no spec decode)** for the speedup calculation.

**Record:**

```
Task               Baseline tok/s   Spec tok/s   Speedup   Acceptance Rate   VRAM Delta (MB)
Q&A
Creative writing
Code generation
```

**Expected pattern:**

- Q&A → highest acceptance (formulaic, predictable tokens) → biggest speedup
- Code → medium acceptance (syntax predictable, logic varies)
- Creative → lowest acceptance (unpredictable distribution) → may be net negative

If this pattern doesn't hold, investigate: mismatched tokenizers (draft/target vocab mismatch artificially suppresses acceptance), or ngram fallback behavior differs from draft model.

---

### Step 5 — Concurrency Impact (30 min)

Fix `num-speculative-tokens` = best K, task = Q&A (best case). Run at concurrency ∈ {1, 4, 8}.

```python
import concurrent.futures

def single_request():
    # same as benchmark function above, single call
    pass

for concurrency in [1, 4, 8]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(concurrency * 10)]
        results = [f.result() for f in futures]
    # aggregate throughput across all futures
```

**Record:**

```
Concurrency   Baseline tok/s   Spec tok/s   Speedup   Acceptance Rate
1
4
8
```

**What to look for:** Speedup should decrease as concurrency increases. Connect this to the systems framing: continuous batching at concurrency=8 already raises GPU utilization, closing the underutilization gap that spec decode was designed to fill. If acceptance rate also drops with concurrency, note that too — it may reflect scheduling pressure on the draft model.

---

### Step 6 — Decision Table + Acceptance Rate Analysis (30 min)

#### Decision Table

Fill with your actual numbers:

```
Scenario                       Speedup   Acceptance Rate   VRAM Cost   Use Spec Decode?   Reasoning
Simple Q&A, conc=1
Simple Q&A, conc=8
Creative writing, conc=1
Code generation, conc=4
[Add one row from your data]
```

#### Acceptance Rate Analysis

Answer these three questions in writing (2–3 sentences each):

1. **What determines acceptance rate in your experiments?**
   - Which task types had high vs. low acceptance, and why? Was this consistent with vocab/distribution reasoning?
   - If using ngram fallback: how does n-gram length affect acceptance?

2. **Where does acceptance rate collapse?**
   - At what concurrency level did acceptance degrade? At what K (spec token count)?
   - What does that imply about the operational regime where spec decode is viable?

3. **When does high acceptance fail to translate into speedup?**
   - If you observed a case where acceptance was ≥70% but speedup was weak, explain the mechanism.
   - Hypothesis: draft model overhead cost vs. batch arithmetic intensity.

---

### Step 7 — Deployability Summary Table

This is the final artifact. Fill it in after completing Steps 3–6.

```
Dimension                                 Finding
VRAM overhead (draft model path)          ___ MB additional
VRAM overhead (ngram path)                ~0 MB
Compatibility risk (TinyLlama/Qwen)       Low / Medium / High — [state what you observed]
Implementation complexity                 Low / Medium / High
Effective concurrency range               conc ≤ ___ (spec decode net positive above this)
Best task type                            ___ (acceptance rate: __%)
Worst task type                           ___ (acceptance rate: __%)
Optimal spec token count (K)              ___
Net recommendation                        Deploy / Deploy with conditions / Do not deploy
Conditions (if conditional)               ___
```

**Write one paragraph (3–5 sentences) justifying the net recommendation.** Cite at least two numbers from your experiments. Frame it in terms of: workload type, target concurrency, hardware budget, and operational cost.

---

## End-of-Day Output

A single markdown document containing:

1. **Compatibility check results** — vocab sizes, what path you used, why (first-class finding)
2. **Spec token sweep table** (Step 3) with sweet-spot observation
3. **Task type table** (Step 4) with acceptance rate interpretation
4. **Concurrency impact table** (Step 5) with systems-framing explanation
5. **Decision table** (Step 6) — feeds directly into Day 15 Section 7
6. **Acceptance rate analysis** (Step 6) — 3 questions answered
7. **Deployability summary table** (Step 7) with written justification

This document is Section 7 of your Week 3 deliverable (Day 15). Write it at that level of quality today.

---

## Key Mental Models to Cement

```
Concept                            Interview-grade framing
Why spec decode works              Target model's decode bottleneck is HBM weight loading, not compute.
                                   Verifying K tokens costs ~same as verifying 1. High acceptance →
                                   K tokens for the price of ~1.
Why it fails at high concurrency   Continuous batching already raises arithmetic intensity. Draft
                                   overhead competes with genuinely useful target model work. The gap
                                   spec decode fills no longer exists.
Why creative tasks hurt            Low acceptance → most spec tokens rejected → pure overhead with no
                                   throughput gain.
The real lever                     Acceptance rate. Below ~70%, spec decode is likely net negative on
                                   a T4-class GPU.
Cross-model compatibility          Theoretical elegance doesn't guarantee deployability.
                                   Tokenizer/vocab compatibility is a hard constraint, not a detail.
                                   Production viability is part of the engineering judgment.
```

---

## Time Pressure Triage

If behind schedule:

- **Skip Step 5** (concurrency impact) — reason through it from your roofline mental model; note it as untested
- **Reduce n_requests** from 40 → 20 per cell — pattern will still be visible
- **Ngram fallback is acceptable** — document the compatibility path, don't lose time on model setup
- Day 15 needs Section 7 (spec decode), but it is one section of eight. Don't let Day 14 experiments bleed into Day 15 writing time.

---

## Connection to Phase A Curriculum

```
Prior Day   Connection
Day 1       Batch size → TFLOPS curve. Batch=1 decode underutilization is what spec decode addresses.
Day 2       Roofline model: decode is memory-bound, low arithmetic intensity. Spec decode raises
            effective intensity by verifying K tokens per pass.
Day 6–8     Continuous batching raises effective batch size → raises GPU utilization → shrinks the gap
            spec decode fills.
Day 11      Prefill vs. decode workload split. Spec decode is decode-phase only — irrelevant to
            prefill-heavy workloads.
Day 15      This day's output is Section 7 of the Week 3 deliverable.
```

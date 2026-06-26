# Day 14 - Speculative Decoding Decision Memo

## Morning Block: Concepts + Setup

### Step 1 - Conceptual Understanding

Speculative decoding addresses a specific inefficiency: during autoregressive decode, the target model processes one token per forward pass, and the GPU sits underutilized. The fix is simple in theory:

1. A small draft model (or n-gram proposer) generates K candidate tokens cheaply
2. The target model verifies all K in a single forward pass (parallel, so higher arithmetic intensity)
3. Accepted tokens get committed. First rejected token triggers rollback.

The key insight is that the target model's decode bottleneck is HBM weight loading, not compute. Verifying K tokens costs roughly the same as verifying 1, because weights get loaded once regardless. So if draft acceptance rate is high, you get K tokens for the cost of roughly 1.

**Acceptance rate** is the organizing variable for the whole day. It determines whether spec decode is a net win or pure overhead. It depends on:

- How well draft distribution matches target distribution
- Task predictability (formulaic Q&A = high acceptance, creative writing = low)
- Temperature setting
- Vocab/tokenizer compatibility between draft and target

**Why it fails at high concurrency:** continuous batching already raises arithmetic intensity by increasing effective batch size per decode step. At concurrency=8, the GPU is already busy. The underutilization gap that spec decode was designed to fill no longer exists. The draft model's overhead now competes with genuinely useful target work.

---

### Step 2 - Setup

#### VRAM Budget Check

```bash
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

Result:
```
memory.total [MiB], memory.free [MiB]
15360 MiB, 14912 MiB
```

T4 with 16GB total, nearly all free before loading.

#### Compatibility Check - Tokenizer Vocab Sizes

```bash
python3 -c "
from transformers import AutoTokenizer
t1 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
t2 = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('Qwen vocab:', t1.vocab_size)
print('TinyLlama vocab:', t2.vocab_size)
print('Match:', t1.vocab_size == t2.vocab_size)
"
```

Result:
```
Qwen vocab: 151643
TinyLlama vocab: 32000
Match: False
```

Nearly 5x difference in vocab size. This is not a soft performance concern. It is a hard structural blocker. The verification step requires comparing P_target(token) vs P_draft(token) for the same token. If the two models use completely different tokenizers, token ID 5000 in TinyLlama maps to a different string than token ID 5000 in Qwen. And Qwen has ~120K tokens that TinyLlama cannot even represent. The algorithm is undefined when the models don't share a token space.

**Finding:** Draft-target compatibility is a first-class engineering constraint. You have to check this before anything else. "Can we even do this cleanly with this draft model?" is the first question, not an afterthought.

#### Draft Model Launch Attempt (failed as expected)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --speculative-config '{"method": "draft_model", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "num_speculative_tokens": 5}' \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
```

**Note on vLLM 0.17.1:** The syllabus uses the older `--speculative-model` and `--num-speculative-tokens` CLI flags. These no longer exist in v0.17.1. The new interface is `--speculative-config` which takes a JSON string. All spec decode parameters go inside that JSON object.

Result: hard failure at startup with a Pydantic validation error:

```
Value error, Target and draft model should have the same vocabulary size.
Target model vocab_size=151936. Draft model vocab_size=32000.
```

vLLM enforces vocab size equality via `verify_equal_vocab_size_if_draft_model()`. It does not silently produce incorrect behavior. It refuses to start. This is the right engineering choice.

**Note:** The model reports vocab_size=151936 (not 151643 from the tokenizer check). The difference is because model embedding layers are often padded for alignment. The tokenizer's functional vocab is 151643 but the model's embedding matrix is sized to 151936.

#### N-gram Fallback Launch (success)

Since no compatible draft model is available for Qwen2.5-3B, I switched to n-gram speculative decoding. N-gram spec decode does not use a separate draft model. Instead, it proposes candidate tokens by matching n-gram patterns from the existing prompt/context.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4, "prompt_lookup_min": 2}' \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
```

Key startup logs:
```
Model loading took 5.79 GiB memory and 47.906125 seconds
Available KV cache memory: 6.15 GiB
GPU KV cache size: 179,216 tokens
Maximum concurrency for 4,096 tokens per request: 43.75x
```

```
VRAM after loading (from nvidia-smi):
memory.used: 14,583 MiB
memory.free: 329 MiB
memory.total: 15,360 MiB
```

VRAM overhead from n-gram spec decode vs a baseline server: effectively 0 MB for draft model weights (there is no draft model). The VRAM cost is only the target model + KV cache, same as without spec decode.

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
print(resp.choices[0].text)
```

Output: `Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain`

#### Spec Decode Metrics Verification

```bash
curl -s http://localhost:8000/metrics | grep -i "spec_decode"
```

Key metrics from this single test request:
```
spec_decode_num_drafts_total: 3
spec_decode_num_draft_tokens_total: 15 (5 per draft)
spec_decode_num_accepted_tokens_total: 4
```

Acceptance rate: 4/15 = ~26.7%

Per-position acceptance breakdown:
```
position 0: 2 accepted
position 1: 2 accepted
position 2: 0 accepted
position 3: 0 accepted
position 4: 0 accepted
```

Acceptance drops to zero after position 1. This makes sense for n-gram on a short, non-repetitive prompt. The n-gram proposer needs matching patterns in the existing context to propose good candidates. With a tiny prompt like "The capital of France is", there is very little context to match against.

---

### Morning Summary

```
Component                           Finding
Target model                        Qwen/Qwen2.5-3B-Instruct (vocab 151,936)
Draft model attempted               TinyLlama-1.1B-Chat-v1.0 (vocab 32,000)
Compatibility result                FAILED - hard vocab mismatch, vLLM rejected at startup
Spec decode path used               N-gram fallback (no draft model weights)
VRAM used (target model)            ~5.79 GiB
VRAM overhead (ngram)               ~0 MB
KV cache available                  6.15 GiB (179,216 tokens)
Server status                       Running on port 8000, spec decode metrics active
```

**Path decision:** Using n-gram spec decode for the afternoon experiments. The compatibility failure with TinyLlama/Qwen is itself a finding, not a setup nuisance. The n-gram approach avoids the compatibility problem entirely but has a different acceptance rate profile since it depends on prompt repetition rather than a learned language model.

#### Draft Model Launch Attempt #2: Qwen2.5-0.5B-Instruct (tested after ngram experiments)

After the ngram experiments showed consistently net-negative results, I tried the same-family draft model path. Qwen2.5-0.5B-Instruct shares the same tokenizer as Qwen2.5-3B-Instruct (both 151,643 vocab). This is the "right" pairing for draft-model spec decode.

Compatibility check:
```
Qwen 3B vocab: 151643
Qwen 0.5B vocab: 151643
Match: True
```

Launch:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 3}' \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000
```

Server started successfully. Resource impact:

```
Metric                          Without draft    With draft     Delta
Model weights in VRAM           5.79 GiB         6.73 GiB       +0.94 GiB
Available KV cache              6.15 GiB         4.49 GiB       -1.66 GiB
KV cache tokens                 179,216          98,112         -81,104
Max concurrency (4096 ctx)      43.75x           23.95x         -19.80x
```

The draft model costs 0.94 GiB in weights, but the KV cache impact is worse: it dropped from 6.15 to 4.49 GiB. Max concurrency nearly halved. This is the static VRAM tradeoff the syllabus warns about.

Benchmark result (same Step 3 workload, K=3, concurrency=1):

```
Metric              Baseline    Ngram K=3    Draft K=3
Throughput (tok/s)  37.51       34.81        15.10
TTFT p50 (ms)       38.7        37.8         59.8
TTFT p99 (ms)       44.3        43.8         146.2
ITL p50 (ms)        26.6        28.7         65.5
Acceptance Rate     N/A         18.6%        39.3%
```

Draft model acceptance rate (39.3%) is higher than ngram (18.6%) as expected, since a learned language model can propose better tokens than n-gram pattern matching. But throughput collapsed to 15.1 tok/s, less than half of baseline. ITL more than doubled from 26.6ms to 65.5ms.

**Why is the draft model path so much worse despite higher acceptance?** The draft model generates K tokens autoregressively: 3 sequential forward passes through the 0.5B model, then 1 verification pass through the 3B target. On a T4, the 0.5B model's forward passes aren't cheap enough relative to the 3B target. The target is only 6x larger than the draft, so draft overhead is a large fraction of total cost. On an A100 with a 70B target and 1B draft, the ratio would be 70x, making draft overhead proportionally tiny. This is a hardware-dependent constraint: spec decode with a draft model needs the target-to-draft size ratio to be large enough that draft passes are negligible. On a T4 with a 3B target, even a 0.5B draft is too expensive.

Higher K values would only make this worse: more sequential draft passes, and acceptance rate drops at later positions.

**Finding:** On T4 with Qwen2.5-3B as target, neither spec decode path (ngram or draft model) produces a speedup. The draft model path is actually worse than ngram despite higher acceptance, because the draft model's forward pass cost is too high relative to the target model on this hardware.

---

## Afternoon Block: Experiments + Decision Memo

All experiments use n-gram speculative decoding (method=ngram, prompt_lookup_max=4, prompt_lookup_min=2). 20 requests per cell, temperature=0, max_tokens=256.

### Step 3 - Spec Token Sweep

Workload: decode-heavy (64-token prompt via "Explain the history of the Roman Empire in detail." + "history " * 20), concurrency=1.

```
Spec Tokens       Throughput (tok/s)   TTFT p50 (ms)   TTFT p99 (ms)   ITL p50 (ms)   Acceptance Rate   VRAM Delta (MB)
Baseline (none)   37.51               38.7            44.3            26.6           N/A               0
3                 34.81               37.8            43.8            28.7           18.6%             ~0
5                 34.68               37.9            42.1            28.8           11.2%             ~0
7                 34.78               37.9            42.1            28.7           8.0%              ~0
10                34.64               37.7            41.8            28.8           5.6%              ~0
```

**Observation:** There is no sweet spot. Every K value is slower than baseline by 7-8%. Acceptance rate drops monotonically as K increases (18.6% at K=3 down to 5.6% at K=10), which makes sense: the n-gram proposer has to find longer matching sequences in the prompt, and there are fewer of those.

The prompt is the problem. It's padded with "history history history..." which the n-gram proposer matches against, but the target model generates a detailed essay about Rome. The proposer and the target fundamentally disagree about what comes next. N-gram spec decode needs the completion to reuse token sequences from the input (summarization, extraction, copy-heavy tasks). This prompt doesn't give it that.

K=3 is the "least bad" option, so I use it for the remaining experiments, but it's still a net negative.

---

### Step 4 - Task Type Sweep

Fixed K=3, concurrency=1, temperature=0.

```
Task               Baseline tok/s   Spec tok/s   Speedup   Acceptance Rate   VRAM Delta (MB)
Q&A                37.54            34.53        0.92x     56.7%             ~0
Creative writing   37.43            34.78        0.93x     4.2%              ~0
Code generation    37.41            34.67        0.93x     24.7%             ~0
```

**Observation:** The acceptance rate pattern matches expectations: Q&A (56.7%) > Code (24.7%) > Creative (4.2%). Q&A is the most formulaic and predictable, creative writing is the least. But even Q&A at 56.7% acceptance still produces a net slowdown.

Why doesn't 56.7% acceptance translate to a speedup? At K=3 with 56.7% acceptance, I'm committing roughly 1.7 tokens per verification cycle. Each cycle costs one target forward pass over K+1 positions (larger than a normal decode step). The question is whether one bigger pass is cheaper than 1.7 regular decode steps. On a T4, apparently not. The verification pass isn't free, it processes a larger sequence which means more attention computation and KV cache writes. At 56.7%, the overhead of the bigger pass outweighs the savings from batching 1.7 tokens into one step.

The syllabus cites ~70% as a rough threshold for net positive on T4-class hardware. My data is consistent with that.

Also worth noting: these are short, single-sentence prompts with very little context for the n-gram proposer to match against. With a draft model (e.g., Qwen2.5-0.5B sharing the same tokenizer), acceptance rates would likely be much higher since the draft model uses a learned language model rather than pattern matching. But we can't test that path due to the vocab mismatch.

---

### Step 5 - Concurrency Impact

Fixed K=3, task=Q&A (best case), temperature=0. Aggregate throughput (total tokens / wall time) is the production-relevant metric here.

```
Concurrency   Baseline tok/s   Spec tok/s   Speedup   Acceptance Rate
1             37.56            34.63        0.92x     56.7%
4             140.15           125.77       0.90x     56.7%
8             271.53           234.47       0.86x     56.7%
```

**Observation:** The speedup ratio degrades from 0.92x at concurrency=1 to 0.86x at concurrency=8. Spec decode gets worse as concurrency increases, not better.

The systems reasoning: at concurrency=1, the target model is underutilized during decode (memory-bound, low arithmetic intensity, the Day 1/Day 2 story). There's at least some theoretical headroom for spec decode to exploit. At concurrency=8, continuous batching has already raised the effective batch size, pushing decode closer to compute-bound territory. The GPU is already busier. Spec decode's verification overhead now competes with genuinely useful target model work, and the gap it was designed to fill is smaller.

Acceptance rate stays flat at 56.7% across all concurrency levels. This is expected for n-gram since the proposer's quality depends only on the prompt content, not on how many other sequences are running. Literature suggests that even with draft models, per-token acceptance rate doesn't actually degrade with concurrency. What degrades is the cost-effectiveness of exploiting that acceptance rate, due to (a) the regime transition from memory-bound to compute-bound, and (b) post-verification alignment overhead (reconciling ragged acceptance counts across sequences into rectangular tensors) that scales superlinearly with batch size.

---

### Step 6 - Decision Table + Acceptance Rate Analysis

#### Decision Table

```
Scenario                       Speedup   Acceptance Rate   VRAM Cost   Use Spec Decode?   Reasoning
Q&A, conc=1                    0.92x     56.7%             ~0 MB       No                 Net negative even at best acceptance rate
Q&A, conc=8                    0.86x     56.7%             ~0 MB       No                 Worse at high concurrency, batching already fills GPU
Creative writing, conc=1       0.93x     4.2%              ~0 MB       No                 Near-zero acceptance, pure overhead
Code generation, conc=1        0.93x     24.7%             ~0 MB       No                 Low acceptance, no benefit
Q&A with draft model, conc=1   0.40x     39.3%             +0.94 GiB   No                 Draft passes too expensive relative to 3B target on T4
```

#### Acceptance Rate Analysis

**1. What determines acceptance rate in my experiments?**

Two factors dominate. First, task predictability: Q&A had the highest acceptance (56.7%) because the output follows formulaic patterns that partially overlap with prompt tokens. Creative writing had the lowest (4.2%) because the output is novel and unpredictable. Second, and more importantly for n-gram specifically, the mechanism is prompt-output overlap. The n-gram proposer can only propose sequences it finds in the existing context. Short prompts with little repetition give it almost nothing to work with. The Step 3 prompt with "history" repeated 20 times had only 18.6% acceptance because the target model doesn't generate "history" repeatedly in the completion. N-gram acceptance is fundamentally about whether the output copies from the input.

**2. Where does acceptance rate collapse?**

Acceptance rate doesn't collapse with concurrency in my experiments. It stayed flat at 56.7% for Q&A across conc=1, 4, 8. This is consistent with what the literature says: acceptance rate is a property of how well the proposer matches the target distribution for a given input, not a function of how many other sequences are in flight. What collapses is the cost-effectiveness, from 0.92x speedup at conc=1 to 0.86x at conc=8.

Acceptance rate does collapse with K (spec token count): 18.6% at K=3 down to 5.6% at K=10 on the history prompt. For n-gram, longer match sequences are exponentially rarer in the prompt context. This implies that for n-gram, K should be kept small (3-4), and even then it only pays off if the task involves significant prompt-to-output copying.

**3. When does high acceptance fail to translate into speedup?**

Q&A at 56.7% acceptance is the clearest example. That's a decent rate, yet throughput is 8% worse than baseline. The mechanism: at K=3 with 56.7% acceptance, each verification cycle commits ~1.7 tokens but costs one forward pass over 4 positions (K+1). That pass involves more attention computation than 1.7 individual decode steps would. On a T4, the crossover point where the batched verification becomes cheaper than sequential single-token decode appears to be somewhere above 70% acceptance. Below that, the overhead of the larger verification pass dominates.

---

### Step 7 - Deployability Summary Table

```
Dimension                                 Finding
VRAM overhead (draft model path)          +0.94 GiB weights, -1.66 GiB KV cache (Qwen2.5-0.5B-Instruct)
VRAM overhead (ngram path)                ~0 MB (no draft model weights)
Compatibility risk (TinyLlama/Qwen)       High. Hard failure: vocab sizes 32,000 vs 151,936. vLLM rejected at startup.
Compatibility risk (Qwen 0.5B/3B)         None. Same tokenizer, same vocab (151,643). Server launched successfully.
Implementation complexity                 Low for ngram (just config flags), Medium for draft model (requires same-vocab model)
Effective concurrency range               N/A. Both paths were net negative at all concurrency levels tested.
Best task type                            Q&A (ngram acceptance: 56.7%, draft acceptance: 39.3%), still net negative on both paths
Worst task type                           Creative writing (ngram acceptance: 4.2%)
Optimal spec token count (K)              3 (least bad), but all K values tested were net negative
Net recommendation                        Do not deploy. Neither path produces a speedup on T4 with Qwen2.5-3B.
```

**Justification:** Speculative decoding should not be deployed for Qwen2.5-3B on T4 hardware, regardless of the spec decode method. I tested both paths.

N-gram spec decode was consistently 7-8% slower than baseline. Even in the best case (Q&A, concurrency=1, K=3), throughput dropped from 37.54 to 34.53 tok/s (0.92x), worsening to 0.86x at concurrency=8. The acceptance rate ceiling for generative tasks was ~57%, below the ~70% threshold needed to offset verification overhead on T4. N-gram proposals depend on prompt-to-output token overlap, which is minimal for most generative workloads.

The draft model path (Qwen2.5-0.5B-Instruct) was even worse: 15.1 tok/s, less than half of baseline, despite a compatible tokenizer and 39.3% acceptance rate. The problem is the target-to-draft size ratio. The 3B target is only 6x larger than the 0.5B draft, so draft forward passes are a significant fraction of total cost rather than negligible overhead. The draft model also consumed 0.94 GiB of VRAM and reduced KV cache headroom by 1.66 GiB, nearly halving max concurrency from 44x to 24x.

Spec decode's value proposition depends on two conditions that don't hold here: (1) the target model's decode step being expensive enough that verification of K tokens is negligible by comparison, and (2) acceptance rate being high enough to amortize the draft/proposal overhead. On a T4 with a 3B model, individual decode steps are already fast (~26ms ITL), leaving little room for spec decode to improve. This optimization is better suited to larger models (30B+) on more powerful hardware where single-token decode is genuinely slow and the target-to-draft ratio provides real leverage.

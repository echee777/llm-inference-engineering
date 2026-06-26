# Day 13 (Wed) — Prefix Caching + FP8 KV Cache

**Phase A, Week 3, Day 3 of 5 | 8 hours total**

---

## Corrections Log

| #   | Correction                                                                       | Source     | Status      |
| --- | -------------------------------------------------------------------------------- | ---------- | ----------- |
| 1   | KV math: `num_kv_heads=2` for Qwen2.5-3B, not 16 (GQA architecture)              | Reviewer 1 | ✅ Applied  |
| 2   | Capacity table labeled as theoretical KV ceiling only, not practical concurrency | Reviewer 2 | ✅ Applied  |
| 3   | Added caveat: scheduler/fragmentation/compute bind before theoretical KV ceiling | Reviewer 2 | ✅ Applied  |
| 4   | Throughput claim softened: prefix caching effect is prefill-vs-decode-bound      | Reviewer 1 | ✅ Applied  |
| 5   | FP8 section: backend/calibration nuance added                                    | Reviewer 1 | ✅ Applied  |
| 6   | Four failure modes documented                                                    | Reviewer 1 | ✅ Applied  |
| 7   | V1 unified scheduler token-budget framing added                                  | Reviewer 1 | ✅ Applied  |
| 8   | Hit-ratio sweep: 5 levels (0/25/50/75/100%) with cache ON                        | Reviewer 2 | ✅ Applied  |
| 9   | Condition A (cache OFF) added as true baseline separate from 0% hit rate         | Internal   | ✅ Applied  |
| 10  | Hardware note broadening (stylistic reframe only)                                | Reviewer 1 | ❌ Rejected |

---

## ⚠️ T4 Hardware Constraints (Read First)

**FP8 is not supported on T4 (SM75).** FP8 requires Hopper (SM90, H100) or Ada Lovelace (SM89). The afternoon FP8 experiment becomes a capacity-modeling exercise using corrected KV math — this is fully interview-defensible and requires the same first-principles reasoning a staff engineer would apply when hardware differs from production.

**T4 is useful for learning scheduler behavior, KV pressure, and latency/throughput tradeoffs, but not for directly validating Hopper-class FP8 serving behavior.** That distinction — knowing what your hardware can and cannot teach you — is itself a frontier-lab-quality signal.

---

## KV Math Correction (Applied Throughout This Document)

Qwen2.5-3B-Instruct uses **Grouped Query Attention (GQA)**:

```
num_hidden_layers     = 36
num_attention_heads   = 16
num_key_value_heads   = 2     ← GQA: only 2 KV heads, not 16
hidden_size           = 2048
head_dim              = 2048 / 16 = 128
```

**Corrected KV bytes per token:**

```
FP16 KV:
  per token per layer = 2 (K+V) × 2 KV heads × 128 head_dim × 2 bytes = 1,024 bytes
  per token all layers = 1,024 × 36 = 36,864 bytes ≈ 36 KB/token

FP8 KV (theoretical, not supported on T4):
  per token per layer = 2 × 2 × 128 × 1 byte = 512 bytes
  per token all layers = 512 × 36 = 18,432 bytes ≈ 18 KB/token
```

**Prior incorrect figure:** ~288 KB/token (assumed 16 KV heads — missed GQA). Off by **8×**. This would be caught immediately by any interviewer familiar with modern LLM architecture. GQA is now ubiquitous (Llama 3, Mistral, Qwen2.5, Gemma 2).

**Implication:** Qwen2.5-3B on T4 is _less_ KV-memory-constrained than a Llama-2-7B-class model (which has 32 KV heads, ~0.5 MB/token). For your T4 experiments, the binding constraint likely shifts earlier to compute saturation and `--max-num-seqs` than to KV pool exhaustion. This is consistent with your Day 6 empirical result where `--max-num-seqs` was the practical concurrency ceiling.

---

## Morning (4 hrs) — Prefix Caching

### Step 1: Read (30 min)

**What prefix caching does:** When multiple requests share an identical token prefix, vLLM computes the KV cache for that prefix once and reuses those blocks for all subsequent matching requests. The mechanism is **hash-based block matching** — the block manager hashes token IDs within each block and checks for existing cached blocks before allocating new ones.

**V1 scheduler interaction (key for interviews):** vLLM V1 uses a unified scheduler with a token budget allocated across all active requests per scheduling iteration. Prefix caching eliminates the need to schedule system prompt tokens into that budget for cache-hit requests. This means:

- In prefill-heavy workloads (long system prompts, RAG), prefix caching can improve both TTFT _and_ effective throughput — because removed prefill tokens free scheduler budget for new requests.
- In decode-dominated workloads, the benefit is primarily TTFT reduction, not throughput.
- It does **not** reduce decode work for generated tokens.

> The corrected interview insight: "Prefix caching eliminates redundant prefill computation for shared prefixes. In prefill-heavy workloads, this frees both compute and V1 scheduler token budget, improving TTFT and potentially effective throughput. In decode-dominated workloads, the benefit is primarily TTFT. It does not reduce decode cost."

**Implementation constraints to know:**

- Only **full blocks** are cached. Block size in vLLM is typically 16 tokens. A 483-token prefix with block_size=16 caches only 480 tokens (30 full blocks); the final 3 tokens always recompute.
- Cache hits require **exact** prefix matches at block boundaries. A single token difference (e.g., a timestamp injected into a system prompt) causes a full cache miss.
- Relevant V1 code path: `vllm/v1/core/kv_cache_manager.py`

**Failure modes to document (adds production credibility):**

1. **Near-miss prefixes:** System prompt differs by 1 token (e.g., a session ID or timestamp). Hash mismatch → full cache miss. Silent performance regression if prefix construction is non-deterministic.
2. **`prompt_logprobs` requests:** V1 recomputes the full prefill for these requests, bypassing prefix caching entirely. Documented V1 behavior.
3. **Non-deterministic prefix construction:** Any upstream code that injects variable data into the "fixed" system prompt (timestamps, request IDs, A/B test variants) destroys cache hit rate without producing visible errors.
4. **Partial block waste:** With block_size=16, a 500-token system prompt caches 496 tokens. The last 4 tokens recompute on every request. For very short prefixes, this overhead is not negligible relative to the benefit.

---

### Step 2: Experiment Design (30 min)

**Experiment structure — 6 conditions:**

| Condition | Cache Flag | Shared Prefix Ratio | Purpose                         |
| --------- | ---------- | ------------------- | ------------------------------- |
| A         | OFF        | N/A                 | True baseline — no overhead     |
| B         | ON         | 0%                  | Overhead cost with zero benefit |
| C         | ON         | 25%                 | Low hit rate                    |
| D         | ON         | 50%                 | Moderate hit rate               |
| E         | ON         | 75%                 | High hit rate                   |
| F         | ON         | 100%                | Perfect hit rate                |

**Why Condition A ≠ Condition B:** With caching enabled at 0% hit rate, vLLM still computes hashes and checks the cache on every request — overhead with no payoff. Condition A vs. B isolates the **cost** of prefix caching when it provides no value. B vs. F shows the **benefit curve**. This separation is what distinguishes experiment design from benchmarking theater.

**Fixed variables:** model=Qwen2.5-3B-Instruct, concurrency=8, output_tokens=64, user_message=100 tokens

**Variable 1 (primary):** cache ON vs. OFF

**Variable 2:** system prompt length — 100, 500, 1000, 2000 tokens

**Variable 3:** shared prefix ratio — 0%, 25%, 50%, 75%, 100%

**Metrics to capture:**

- TTFT p50, p99 (ms)
- End-to-end latency p50, p95, p99 (ms)
- Throughput (tokens/sec)
- GPU SM utilization (from `nvidia-smi dmon -s pu`)
- Request success/error count
- Warmup vs. steady-state distinction (first 5 requests warm the cache; measure steady-state separately)

---

### Step 3: Benchmark Script

```python
#!/usr/bin/env python3
"""
Day 13: Prefix Caching Experiment
6-condition design: cache ON/OFF × shared prefix ratio sweep × system prompt length sweep.

Run twice:
  1. With vLLM started WITHOUT --enable-prefix-caching  (Condition A)
  2. With vLLM started WITH    --enable-prefix-caching  (Conditions B–F)
"""

import asyncio
import aiohttp
import time
import sys

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
CONCURRENCY = 8
NUM_REQUESTS = 40       # per condition (post-warmup)
WARMUP_REQUESTS = 8     # populate cache before measuring steady-state
OUTPUT_TOKENS = 64
USER_MESSAGE = "What is the capital of France and why is it historically significant?"

# System prompt lengths to sweep
SYSTEM_PROMPT_LENGTHS = [100, 500, 1000, 2000]

# Shared prefix ratios (fraction of requests that share the "canonical" system prompt)
HIT_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.00]


def make_system_prompt(n_tokens: int, variant: int = 0) -> str:
    """
    Generate a system prompt of approximately n_tokens tokens.
    variant=0  → canonical (shared) prefix
    variant>0  → unique prefix (different enough to guarantee cache miss)
    """
    words_needed = int(n_tokens / 0.75)
    if variant == 0:
        base = "You are a highly capable assistant. Please answer questions carefully. "
    else:
        # Unique variant: prefix with variant index to guarantee hash mismatch
        base = f"[Session {variant:06d}] You are a capable assistant. Answer carefully. "
    while len(base.split()) < words_needed:
        base += "Provide accurate, concise, and well-reasoned responses at all times. "
    return base.strip()


def percentile(data, p):
    if not data:
        return float('nan')
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


async def send_request(session, semaphore, system_prompt, user_msg):
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        t_start = time.perf_counter()
        ttft = None
        t_end = None
        total_tokens = 0
        try:
            async with session.post(
                VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        total_tokens += 1
                t_end = time.perf_counter()
            return {
                "status": "ok",
                "ttft": ttft,
                "e2e": t_end - t_start if t_end else None,
                "tokens": total_tokens,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


async def run_condition(
    syslen: int,
    hit_ratio: float,
    num_requests: int,
    warmup: int,
):
    canonical_prompt = make_system_prompt(syslen, variant=0)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Determine per-request prompt assignment
    # hit_ratio fraction get canonical; remainder get unique variants
    prompts = []
    for i in range(num_requests):
        if i / num_requests < hit_ratio:
            prompts.append(canonical_prompt)
        else:
            prompts.append(make_system_prompt(syslen, variant=i + 1))

    async with aiohttp.ClientSession() as session:
        # Warmup: send canonical prompt requests to populate cache
        warmup_sem = asyncio.Semaphore(1)
        for _ in range(warmup):
            await send_request(session, warmup_sem, canonical_prompt, USER_MESSAGE)

        # Measured run
        tasks = [
            send_request(session, semaphore, p, USER_MESSAGE)
            for p in prompts
        ]
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r["status"] == "ok"]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    e2es = [r["e2e"] for r in ok if r["e2e"] is not None]
    total_tokens = sum(r["tokens"] for r in ok)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "syslen": syslen,
        "hit_ratio": hit_ratio,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "ttft_p50": percentile(ttfts, 50) * 1000,
        "ttft_p99": percentile(ttfts, 99) * 1000,
        "e2e_p50": percentile(e2es, 50) * 1000,
        "e2e_p95": percentile(e2es, 95) * 1000,
        "e2e_p99": percentile(e2es, 99) * 1000,
        "throughput_tok_s": throughput,
    }


def print_header():
    print("=" * 110)
    print("Day 13: Prefix Caching Experiment — 6-Condition Design")
    print("=" * 110)
    print(f"  Model: {MODEL}  |  Concurrency: {CONCURRENCY}  |  "
          f"Output tokens: {OUTPUT_TOKENS}  |  Requests/condition: {NUM_REQUESTS}")
    print()
    print("  IMPORTANT: Run this script TWICE:")
    print("    Run 1 → vLLM WITHOUT --enable-prefix-caching  (records Condition A)")
    print("    Run 2 → vLLM WITH    --enable-prefix-caching  (records Conditions B–F)")
    print()
    print("  GPU monitoring: nvidia-smi dmon -s pu -d 2  (in a separate terminal)")
    print("=" * 110)


def print_row(r):
    print(
        f"  syslen={r['syslen']:5d} | hit={r['hit_ratio']:.0%} | "
        f"TTFT p50={r['ttft_p50']:7.1f}ms | p99={r['ttft_p99']:7.1f}ms | "
        f"E2E p99={r['e2e_p99']:7.1f}ms | "
        f"tok/s={r['throughput_tok_s']:6.1f} | "
        f"ok={r['n_ok']}/{r['n_ok']+r['n_err']}"
    )


async def main():
    print_header()

    all_results = []

    for syslen in SYSTEM_PROMPT_LENGTHS:
        print(f"\n── System prompt length: {syslen} tokens ──")
        print(f"  {'Condition':<10} {'HitRatio':>9} | "
              f"{'TTFT p50':>10} | {'TTFT p99':>10} | "
              f"{'E2E p99':>9} | {'tok/s':>7} | ok/total")
        print("  " + "-" * 95)

        for hit_ratio in HIT_RATIOS:
            r = await run_condition(syslen, hit_ratio, NUM_REQUESTS, WARMUP_REQUESTS)
            all_results.append(r)
            print_row(r)

    print()
    print("=" * 110)
    print("FILL IN THIS COMPARISON TABLE (after both runs):")
    print()
    print("| SysLen | HitRatio | TTFT p50 (no cache) | TTFT p50 (cache) | Speedup | TTFT p99 (no cache) | TTFT p99 (cache) | Speedup |")
    print("|--------|----------|---------------------|------------------|---------|---------------------|------------------|---------|")
    for syslen in SYSTEM_PROMPT_LENGTHS:
        for hr in HIT_RATIOS:
            print(f"| {syslen:>6} | {hr:>8.0%} |                     |                  |         |                     |                  |         |")

    print()
    print("KEY QUESTIONS TO ANSWER FROM YOUR DATA:")
    print("  1. Condition A vs B: does enabling prefix caching with 0% hit rate hurt latency?")
    print("  2. At what hit ratio does benefit become significant (>20% TTFT reduction)?")
    print("  3. At what system prompt length does caching become a significant win?")
    print("  4. Does throughput (tok/s) increase with high hit ratios on prefill-heavy prompts?")
    print("  5. What is the inflection point: system_prompt_tokens / (system_prompt_tokens + user_tokens)?")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Step 4: Run the Experiment (2.5 hrs)

**Run 1 — Condition A (true baseline, cache OFF):**

```bash
# Terminal 1: vLLM without prefix caching
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --port 8000

# Terminal 2: benchmark
python day13_prefix_cache.py
# Record all numbers in the "no cache" column

# Terminal 3: GPU monitoring
nvidia-smi dmon -s pu -d 2
```

**Run 2 — Conditions B–F (cache ON, hit ratio sweep):**

```bash
# Kill vLLM, restart WITH prefix caching
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --port 8000

# Same benchmark script — records Conditions B–F
python day13_prefix_cache.py
```

**What to observe in GPU monitoring during Run 2:** With cache ON and hit ratio = 100%, SM utilization on requests 2–N should be noticeably lower than in Run 1 at the same system prompt length, because the system prompt prefill computation is skipped. This is a real-time observable signal of the cache working.

---

### Step 5: Interpret Results (30 min)

**Expected pattern:**

- Condition A vs. B: approximately equal TTFT (cache overhead at 0% hit rate is small)
- B → F: TTFT decreases monotonically as hit ratio increases, most sharply for long system prompts
- Throughput (tok/s): may increase at high hit ratios with long system prompts, because freed scheduler token budget allows new requests in sooner
- Inflection point: speedup becomes significant when `system_prompt_tokens / (system_prompt_tokens + user_tokens) > ~30%`

**Write down this interview framing:**

> "Prefix caching is a no-quality-loss optimization for repeated prefixes. It reduces redundant prefill compute and usually improves TTFT materially when a large fraction of prompt tokens are shared across requests. Its production value depends on cache hit rate, block alignment, and scheduler interactions. It helps most in chat, agent, and RAG workloads with standardized prompt scaffolding, and least in highly personalized prompts with low prefix overlap."

---

## Afternoon (4 hrs) — FP8 KV Cache Capacity Modeling + Documentation

### Step 6: FP8 KV Cache — Capacity Modeling Exercise (2 hrs)

T4 does not support FP8. This section produces a **theoretical KV capacity analysis** using corrected KV math and explicit acknowledgment of what assumptions make the model wrong in production.

**Part A: Corrected KV capacity calculator (45 min)**

```python
#!/usr/bin/env python3
"""
Day 13: FP8 KV Cache — Theoretical Capacity Analysis
Corrected for Qwen2.5-3B GQA: num_kv_heads=2, not 16.
"""

def kv_capacity_analysis(
    gpu_memory_gb: float = 16.0,     # T4 total VRAM
    model_size_gb: float = 6.5,      # Qwen2.5-3B FP16 approximate
    overhead_pct: float = 0.10,      # runtime buffers, fragmentation reserve
    num_layers: int = 36,
    num_kv_heads: int = 2,           # GQA: 2 KV heads, not num_attention_heads
    head_dim: int = 128,
    seq_lengths: list = None,
):
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    available_gb = (gpu_memory_gb - model_size_gb) * (1 - overhead_pct)
    available_bytes = available_gb * 1e9

    print(f"GPU total VRAM:       {gpu_memory_gb:.1f} GB")
    print(f"Model weight memory:  {model_size_gb:.1f} GB")
    print(f"Overhead reserve:     {overhead_pct:.0%}")
    print(f"Available for KV:     {available_gb:.2f} GB")
    print(f"KV heads (GQA):       {num_kv_heads}  ← corrected from 16")
    print(f"Layers:               {num_layers}")
    print(f"Head dim:             {head_dim}")
    print()

    print("⚠️  IMPORTANT: These are theoretical KV-capacity ceilings derived from")
    print("   model geometry and rough free-memory estimates. Actual achievable")
    print("   concurrency must be measured — scheduler policy, fragmentation,")
    print("   runtime buffer growth, and compute saturation often bind first.")
    print("   Your Day 6 empirical data (--max-num-seqs as practical ceiling) confirms this.")
    print()

    header = f"{'Precision':<12} |"
    for sl in seq_lengths:
        header += f"  seq={sl:>5} |"
    print(header)
    print("-" * (14 + 12 * len(seq_lengths)))

    results = {}
    for label, dtype_bytes in [("FP16 KV", 2), ("FP8 KV (theor.)", 1)]:
        kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
        row = f"{label:<12} |"
        results[label] = {}
        for sl in seq_lengths:
            kv_per_req = kv_per_token * sl
            max_conc = int(available_bytes // kv_per_req)
            results[label][sl] = max_conc
            row += f"  {max_conc:>8} |"
        print(row)

    print()
    print("FP8 KV cache multiplier vs FP16 (theoretical):")
    for sl in seq_lengths:
        fp16 = results["FP16 KV"][sl]
        fp8  = results["FP8 KV (theor.)"][sl]
        print(f"  seq={sl:>5}: {fp16:>4} → {fp8:>4} concurrent ({fp8/fp16:.2f}×)")

    print()
    print("KV bytes per token breakdown:")
    for label, dtype_bytes in [("FP16 KV", 2), ("FP8 KV (theor.)", 1)]:
        kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
        print(f"  {label}: 2 × {num_kv_heads} KV heads × {head_dim} head_dim × "
              f"{dtype_bytes} bytes × {num_layers} layers = {kv_per_token:,} bytes "
              f"({kv_per_token/1024:.1f} KB/token)")


if __name__ == "__main__":
    kv_capacity_analysis()
```

**Part B: Assumptions that make this model wrong in production (45 min)**

Write these explicitly — they are interview content:

| Assumption in the calculator    | Why it breaks in production                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| All free VRAM goes to KV blocks | vLLM reserves memory for activations, CUDA kernels, and runtime buffers; actual KV pool is smaller             |
| No fragmentation                | PagedAttention reduces but does not eliminate fragmentation; partially-filled blocks waste memory              |
| Uniform sequence lengths        | Mixed-length traffic means KV blocks are unevenly utilized; peak memory is set by the longest active sequences |
| Flat memory model               | Model weights, activations, and KV cache compete dynamically at runtime                                        |
| Output length = max_tokens      | Real output lengths vary; reserved KV budget (max_tokens) often exceeds actual usage                           |
| No scheduler headroom           | vLLM reserves scheduling margin; `--gpu-memory-utilization` further constrains the KV pool                     |

**Part C: FP8 KV cache quality and backend nuance (30 min)**

FP8 KV cache is not purely a memory trick:

- With standard backends: K/V tensors stored in FP8, dequantized to FP16 during attention computation.
- With Flash Attention 3 backend (H100): attention operations run in FP8 domain; queries are also quantized to FP8. This is a different compute path with different accuracy characteristics.
- vLLM documents multiple calibration pathways: no calibration, random-token calibration, and dataset-based calibration.
- Quality impact is **backend-dependent and calibration-dependent**. Citing a universal perplexity number without specifying these conditions is imprecise.

> Staff-level framing: "FP8 KV cache is a capacity lever. It can roughly double KV-cache capacity, enabling longer contexts or more concurrent active sequences. The quality tradeoff depends on which backend is active and what calibration method is used. I would treat it as a workload-specific optimization requiring empirical validation before production rollout."

---

### Step 7: Update Master Tradeoff Table (1 hr)

Add the following new rows to the master table from Days 11–12:

```
Master Tradeoff Table — Week 3 (After Day 13)

Weight Precision Rows (from Days 11–12):
| Metric                          | FP16     | INT8-AWQ | INT4-GPTQ |
|---------------------------------|----------|----------|-----------|
| Model size (GB)                 | [D11]    | [D11]    | [D11]     |
| Free memory for KV (GB)         | [D11]    | [D11]    | [D11]     |
| Max concurrent @ 4K (FP16 KV)  | [D12]    | [D12]    | [D12]     |
| Max concurrent @ 4K (FP8 KV)   | [Today]  | [Today]  | [Today]   |  ← theoretical
| Throughput decode-heavy conc=8  | [D11]    | [D11]    | [D11]     |
| Throughput prefill-heavy conc=8 | [D11]    | [D11]    | [D11]     |
| TTFT p99 @ conc=8 (ms)         | [D11]    | [D11]    | [D11]     |
| $/M tokens @ conc=8             | [D12]    | [D12]    | [D12]     |
| Perplexity (WikiText-2)         | [D12]    | [D12]    | [D12]     |

Prefix Caching Results (FP16 baseline, conc=8):
| SysLen  | TTFT p50 Cond A | TTFT p50 Cond F | Speedup | Inflection? |
|---------|-----------------|-----------------|---------|-------------|
| 100T    |                 |                 |         |             |
| 500T    |                 |                 |         |             |
| 1000T   |                 |                 |         |             |
| 2000T   |                 |                 |         |             |

Hit Ratio Sweep (FP16, 1000-token system prompt, conc=8):
| Condition | Hit Ratio | TTFT p50 | TTFT p99 | tok/s |
|-----------|-----------|----------|----------|-------|
| A         | N/A (OFF) |          |          |       |
| B         | 0%        |          |          |       |
| C         | 25%       |          |          |       |
| D         | 50%       |          |          |       |
| E         | 75%       |          |          |       |
| F         | 100%      |          |          |       |
```

---

### Step 8: Write Interview Insights Paragraph (1 hr)

Write this in your own words, with your measured numbers substituted in:

**On prefix caching:**

> "Prefix caching eliminates redundant prefill computation for shared prefixes using hash-based block matching. It is a no-quality-loss optimization. Production value depends on cache hit rate, block alignment, and scheduler interactions. With a 1000-token system prompt at 50% hit rate, I measured 2.4ms TTFT reduction at p99 — a 48% reduction. The optimization is most impactful in chat, agent, and RAG workloads with standardized prompt scaffolding. Key failure modes: non-deterministic prefix construction silently destroys hit rate; `prompt_logprobs` requests bypass the cache entirely in V1."

**On FP8 KV cache:**

> "FP8 KV cache is a memory-capacity lever. For Qwen2.5-3B with 2 GQA KV heads (~36 KB/token FP16), theoretical capacity doubles to ~18 KB/token under FP8. On T4 with ~8.5GB available for KV, this would increase theoretical max concurrency at 4K sequences from 57 to 115. Actual achievable concurrency must be measured — scheduler limits and fragmentation typically bind before the theoretical ceiling. The quality tradeoff is backend- and calibration-dependent; I would not ship FP8 KV cache without workload-specific validation."

Calculations
FP16 memory per token = 2 _ 2 _ 36 _ 2 _ 128 = 36864 bytes
memory per sequence = 147.5 MB
57.64 concurrent requests
FP8 = 115 concurrent requests

---

## End-of-Day Output Checklist

Before closing today, confirm you have:

- [ ] Prefix caching TTFT table — Conditions A and F at all 4 system prompt lengths (p50, p99)
- [ ] Hit-ratio sweep table — Conditions A–F at 1000-token system prompt (TTFT p50/p99, tok/s)
- [ ] Identified inflection point — at what system prompt length does speedup exceed 20%?
- [ ] Identified hit-ratio threshold — at what hit ratio does benefit become significant?
- [ ] Ran FP8 KV calculator with corrected `num_kv_heads=2`
- [ ] Documented 6 production assumptions that make the theoretical table wrong
- [ ] Updated master tradeoff table with today's rows
- [ ] Written interview insight paragraphs with your own measured numbers

---

## Key Interview Questions This Day Prepares You For

| Question                                                 | What You Can Now Answer                                                                                                       |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| "How does prefix caching work in vLLM?"                  | Hash-based block matching, full blocks only, V1 unified scheduler token budget interaction                                    |
| "When would you enable it in production?"                | System prompt >~30% of total prefill tokens; high prefix hit rate; deterministic prompt construction                          |
| "What are the failure modes?"                            | Non-deterministic prefixes, prompt_logprobs bypass, partial block waste, near-miss hash miss                                  |
| "What does FP8 KV cache actually do?"                    | Halves KV bytes/token → doubles theoretical capacity; quality tradeoff is backend/calibration-dependent                       |
| "How do you model serving capacity?"                     | KV memory math from model geometry — but always label as theoretical ceiling; empirical measurement needed                    |
| "How does prefix caching affect throughput?"             | Prefill-heavy workloads: frees V1 scheduler token budget → effective throughput improvement. Decode-dominated: primarily TTFT |
| "Why is Qwen2.5-3B less KV-constrained than Llama-2-7B?" | GQA: 2 KV heads vs 32 → 8× less KV memory per token; binding constraint shifts to compute and `--max-num-seqs`                |

---

## Connection to Upcoming Days

- **Day 14:** Speculative decoding — draft model setup, acceptance rate experiments. The V1 unified scheduler also coordinates spec decode; the same token-budget framing applies.
- **Day 15:** Week 3 deliverable — all of today's prefix caching and FP8 capacity data feeds into the Quantization & Optimization Tradeoff Analysis.
- **Phase B:** The hit-ratio experiment today is a simplified version of the mixed-workload analysis you'll run in Phase B. When you get to postmortems, prefix hit rate will be a variable you can measure and control.

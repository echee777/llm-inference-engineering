# Day 23 — Prefill/Decode Interference Analysis
## AI Inference Platform Residency — Phase B, Track 1
**Version:** v1
**Deliverable:** #6 — Prefill/Decode Interference Analysis
**Hardware:** T4 (g4dn.xlarge)
**Model:** Qwen2.5-3B-Instruct

---

## Prerequisites

- Day 22 afternoon mixed-length experiment data collected:
  - Traffic mix: 50% short (prompt=64 tokens, max_new_tokens=128) + 50% long (prompt=2048 tokens, max_new_tokens=512)
  - System held at ~65% KV utilization during run
  - Per-request latency tagged by type (`short` / `long`)
  - Isolated short-only control run at same concurrency recorded
- Postmortem #1 (Deliverable #5) complete

---

## Objectives

1. Quantify the TTFT penalty short requests suffer when co-scheduled with long-context requests
2. Measure chunked prefill's effect on short-request latency and long-request throughput
3. Produce Deliverable #6: Prefill/Decode Interference Analysis

---

## Morning (4 hrs) — Analyze + Chunked Prefill Experiment

### Step 1 — Analyze Mixed-Length Data (2 hrs)

Produce three artifacts from Day 22 raw data:

**a) Dual-CDF Plot**
- Two curves on one graph: `TTFT(short, mixed)` vs. `TTFT(short, isolated)`
- X-axis: latency (ms), Y-axis: cumulative fraction (0 → 1)
- The gap between curves at p99 is your measured penalty

**b) Bimodal TTFT Distribution**
- Histogram of TTFT for *all* requests in the mixed run
- Expected: two humps — short requests cluster low, long requests cluster high

**c) p99 Degradation Factor**
```
degradation_factor = TTFT_p99(short, mixed) / TTFT_p99(short, isolated)
```
Write this number down explicitly. It is the headline stat for the deliverable.

**vLLM V1 scheduling model:**
```
Each iteration = one forward pass
That pass is either:
  - prefill (processes N input tokens, populates KV cache for all N tokens)
  - decode  (generates 1 token per active sequence, reads KV cache)

vLLM does not preempt within an iteration.
Scheduling decisions occur only between iterations.
=> Prefill monopolizes the entire forward pass.
=> No intra-iteration interleaving of prefill and decode.
```
Note: during prefill, KV cache is populated for all N tokens simultaneously, increasing both compute duration and memory pressure for that iteration — further extending the starvation window for co-scheduled decode requests.

**Mechanism to internalize:**
Starvation occurs because a prefill iteration consumes the entire forward pass, and decode cannot progress until that iteration completes.

Long 2048-token prefill operations run to completion before the scheduler advances to the next iteration. During that prefill pass, decode steps for all concurrent short requests are starved — they cannot advance even one token. The GPU is fully utilized but fully occupied by the long prefill. This is **decode starvation**: a scheduling fairness problem, not a capacity problem.

> **Scheduler policy, not hardware limit:** vLLM's scheduler does not mix prefill and decode requests within the same forward pass. This is a design choice — the GPU is capable of running different kernel types sequentially; the scheduler simply does not interleave them. Chunked prefill is the mechanism that relaxes this policy.

**Why short requests are especially sensitive:** a short request's expected decode step is ~tens of ms. A 504ms starvation window represents an order-of-magnitude slowdown relative to that baseline — regardless of how small the short request itself is.

**Blocking duration estimate (use your Day 21 empirical fit):**
From Day 21: `TTFT(ms) = 37.6 + 0.228 × prompt_tokens` (R²=0.9966, T4/Qwen2.5-3B)
At 2048-token prefill: starvation window ≈ 37.6 + 0.228 × 2048 ≈ **504ms**
Every co-scheduled short request is starved from its next decode step for ~504ms — regardless of how short it is or how lightly loaded the system is. This is the number to cite in interviews.

---

### Step 2 — Chunked Prefill Experiment (2 hrs)

Restart vLLM with chunked prefill enabled:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096
```

> **V1 startup note:** `--max-num-batched-tokens` must satisfy `>= max_model_len` (4096). Chunk size is controlled separately — confirm the server starts cleanly before running the experiment.

Re-run the **identical mixed traffic workload** from Day 22:
- Same concurrency level
- Same 50/50 short/long mix
- Same 10-minute duration

Record:
| Metric | Non-Chunked | Chunked |
|---|---|---|
| Short-request TTFT p50 (ms) | | |
| Short-request TTFT p99 (ms) | | |
| Long-request throughput (tok/s) | | |
| Overall system throughput (req/s) | | |

**Expected result:** Short-request p99 drops meaningfully. Long-request throughput may dip slightly due to chunking overhead. Both directions matter for the tradeoff analysis.

**Mental model to carry forward:**
- Non-chunked prefill: maximizes throughput for long requests, starves short ones
- Chunked prefill: improves scheduling fairness, adds per-chunk overhead, slightly reduces long-request throughput

Chunked prefill trades global efficiency for fairness. It is not a free throughput upgrade.

**Chunk size is the control knob:**
```
Smaller chunks (e.g., 256 tokens):
  + better decode fairness (shorter starvation windows)
  - more scheduling overhead per prefill

Larger chunks (e.g., 1024 tokens):
  + better long-request throughput
  - longer decode starvation per chunk
```
At `--max-num-batched-tokens 512` your 2048-token prefill splits into 4 chunks, each causing a ~120ms starvation window instead of one 504ms window. Measure whether this is perceptible in your short-request p99.

---

## Afternoon (4 hrs) — Write Deliverable #6

### Deliverable #6: Prefill/Decode Interference Analysis

Write the full document using your morning experimental data. Required sections:

---

**Section 1: The Problem**

Explain prefill/decode asymmetry:
- **Prefill** is compute-bound and bursty: processes all N input tokens in a single (or chunked) forward pass through all attention heads and FFN layers
- **Decode** is memory-bandwidth-bound and sustained: generates one token per forward pass, bounded by KV cache read bandwidth
- They compete for the same iteration scheduling slot — the GPU cannot interleave them within a single forward pass
- Quantify blocking: one 2048-token prefill blocks approximately `prompt_tokens / tokens_per_decode_batch` decode steps for co-scheduled requests

**Section 2: Measured Interference**

Include:
- The dual-CDF plot (short, mixed vs. short, isolated)
- Your degradation factor (the concrete number from Step 1c)
- The bimodal TTFT histogram
- Narrative: what the graphs show, in plain language

**Section 3: Chunked Prefill Results**

Include:
- Short-request TTFT p50/p99 before vs. after chunked prefill
- Long-request throughput before vs. after chunked prefill
- Explicit tradeoff statement: "Chunked prefill improved short-request p99 by X% at the cost of Y% long-request throughput degradation"
- Frame as a scheduling fairness vs. throughput tradeoff — not a free lunch

**Section 4: Production Implications**

Staff-level insight: in a multi-tenant system, one tenant sending long-context requests can silently degrade another tenant's short-request SLO. This failure mode is **invisible in average GPU utilization metrics** — GPU util reads high and healthy while short-request p99 is getting hammered. Standard observability (GPU%, memory%) will not surface this. Per-request-type latency distributions are required to detect it.

**Concurrency sensitivity:** this problem compounds as concurrency increases. With more short requests co-scheduled against the same blocking prefill, more requests accumulate the ~504ms decode starvation window. The cumulative latency damage scales with concurrency — which directly connects to why the Day 24 cliff steepens under mixed-length traffic. Admission control that ignores request-type composition will underestimate this effect.

**Section 5: Mitigations**

**Why continuous batching alone does not fix this:** A common misconception is that continuous batching eliminates prefill blocking. It does not. Continuous batching allows new requests to join the batch between iterations — but within each forward pass, the scheduler still executes either a prefill chunk or decode steps. A large un-chunked prefill still dominates a full forward pass. Continuous batching improves GPU utilization across requests; it does not change the within-iteration serialization of prefill and decode.

Beyond chunked prefill, three mitigations to document:

1. **Priority queues / sequence-length tiering** — separate queue lanes for short and long requests; long requests cannot block the short queue
2. **Per-tenant prefill budget caps** — cap the number of long-context (high-token-count) requests allowed in-flight simultaneously per tenant
3. **Disaggregated prefill/decode** — architectural fix (preview of Phase C): route prefill and decode to separate GPU pools, eliminating the scheduling competition entirely. Note: this is the production-grade solution at scale; chunked prefill is an operational band-aid

---

## End-of-Day Completion Gate

Deliverable #6 is complete when all of the following are true:

- [ ] Dual-CDF plot produced (short mixed vs. short isolated)
- [ ] Bimodal TTFT histogram produced
- [ ] p99 degradation factor computed — a specific number, not a range
- [ ] Chunked prefill experiment run with same workload parameters as baseline
- [ ] Chunked prefill table complete: short-request improvement AND long-request throughput cost both quantified
- [ ] All 5 deliverable sections written with your actual experimental numbers — no placeholders
- [ ] Section 4 production implications paragraph written in declarative staff-level prose (not "I learned that..." — state the operational risk directly)

---

## Key Numbers to Carry Forward

These values feed directly into Deliverable #7 (Latency vs. Utilization Curve), the Phase B Key Numbers table (Appendix C), and Deliverable #11 (End-to-End Platform Design):

| Metric | Your Value |
|---|---|
| Short-request TTFT p99 (isolated) | |
| Short-request TTFT p99 (mixed, no chunked prefill) | |
| Short-request TTFT p99 (mixed, chunked prefill) | |
| p99 degradation factor (mixed / isolated) | |
| Chunked prefill long-request throughput cost (%) | |

---

## Concepts: vLLM V1 Scheduling Behavior

| Concept | V1 Behavior |
|---|---|
| Prefill scheduling | Runs to completion per iteration (or per chunk if chunked-prefill enabled) before decode resumes |
| Chunked prefill | Splits large prefill across multiple iterations; each chunk interleaved with decode steps |
| Preemption | Recompute-only (no CPU swap). Preempted requests re-enter the waiting queue. |
| Iteration budget | Controlled by `--max-num-batched-tokens` (per-forward-pass token budget, not a memory constraint) |

---

## Interview Signal

Deliverable #6 answers: **"Why do short requests get slow under mixed workloads?"** and **"How do you handle mixed-length traffic?"**

Staff signal demonstrated:
- Hardware-level scheduling intuition (prefill/decode asymmetry is a GPU iteration scheduling problem, not just a latency problem)
- Understands why GPU utilization is a misleading metric for this failure mode
- Can propose mitigations at three levels: scheduling (queuing), operational (chunked prefill), and architectural (disaggregation)

---

*Phase B Track 1 | Day 23 of 45 | Residency v4 | Syllabus v4*

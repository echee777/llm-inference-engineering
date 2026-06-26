# Phase B — Day 21: KV Cache Exhaustion: Instrument + Induce
## v5 — Reviewer changes applied

**Track:** Track 1 — Collapse Engineering (T4 / g4dn.xlarge)
**Goal:** Establish a clean healthy baseline, then ramp concurrency until the system collapses. Produce a collapse timeline with real numbers.
**Feeds into:** Postmortem #1 (Deliverable #5), due Day 22.

---

## Reviewer Correction Table (v1 → v2)

| # | Change | Disposition | Applied |
|---|---|---|---|
| 1 | Add queue depth / waiting requests to ramp table | Accept | Step 4 metrics table |
| 2 | Record throughput (tok/s) at every ramp level | Accept | Step 4 metrics table |
| 3 | Explicitly label preemption onset vs. cliff onset as separate events | Accept | Step 4 + Step 5 timeline |
| 4 | Promote budget delta to named required writeup | Accept | Step 2 — new "Why My Budget Was Wrong" section |
| 5 | Add short-prompt/long-generation control experiment | Partial accept | Added as optional extension after Step 5; not required |
| 6 | Add effective_concurrency = running + waiting derived metric | Accept | Makes feedback loop numerically visible in ramp table |
| 7 | Add divergence_ratio = TTFT_p99 / TTFT_p50 column | Accept | Clean numeric cliff definition; quotable interview anchor |
| 8 | Add required core insight takeaway at end of day | Accept | Locks in causal mental model before Day 22 postmortem writing |
| 9 | Add sentence tying divergence ratio → admission control | Accept | Makes Phase A gateway → Day 21 measurement → Phase B autoscaling through-line explicit |
| 10 | Add explicit anti-pattern: 85% KV util unsafe even if throughput is high | Accept | Strengthens Decision Section framing for Deliverables #7 and #9 |
| 11 | Replace Step 2 math re-derivation with Phase A calculator reference | Accept | KV formula was already derived in Phase A Day 3 — re-doing it is rote; the high-value work is the estimate-vs-actual delta and the writeup |

---

## Schedule Overview

| Block | Duration | Activity |
|---|---|---|
| Morning Step 1 | 30 min | Instance check + smoke test |
| Morning Step 2 | 30 min | KV budget calibration + budget discrepancy writeup |
| Morning Step 3 | 3 hrs | Baseline run |
| Afternoon Step 4 | 3 hrs | Ramp to failure |
| Afternoon Step 5 | 30 min | Collapse timeline construction |
| Optional | 30 min | Control experiment: decode-retention pressure profile |

---

## Step 1 — Instance Check (30 min)

Confirm your Phase A instrumented vLLM fork is live and your KV block allocation patch fires.

```bash
# Quick smoke test — 2 concurrent requests
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', gpu_memory_utilization=0.90)
params = SamplingParams(max_tokens=64)
outputs = llm.generate(['hello world'] * 2, params)
print([o.outputs[0].text[:50] for o in outputs])
"
```

**Pass criteria:** KV block allocation logs fire for both requests.

If they don't fire, your Phase A patch isn't hooked in — fix before proceeding.

**V1 code reference:**
- KV Cache Manager: `vllm/core/kv_cache_manager.py` (`KVCacheManager` + `BlockPool`)
- Request object: `vllm/sequence.py` → `Request` class (not `SequenceGroup`)
- Preemption: recompute-only — no CPU swap path in V1

---

## Step 2 — Calibrate Your KV Budget (30 min)

Run your Phase A KV cache calculator for Qwen2.5-3B-Instruct on T4. Record the estimated token capacity it outputs. If you no longer have the calculator script, re-derive from first principles — but do not re-derive if the script is available. The derivation was Phase A work; the value here is the comparison to reality.

### Record estimated vs. actual

After vLLM starts, read `num_gpu_blocks` from the startup log.

| Metric | Your Value |
|---|---|
| Estimated KV token capacity | |
| vLLM reported `num_gpu_blocks` | |
| Block size (tokens per block) | |
| Derived actual KV token capacity | |
| Delta (estimated − actual) | |

### Required writeup: "Why My Theoretical Budget Was Wrong"

Write 3–5 sentences explaining the delta — not just the number, but the cause. Cover the applicable factors:

- **Runtime overhead:** CUDA context, PyTorch allocator, driver reservations
- **Block rounding:** vLLM allocates in fixed-size blocks; fractional blocks are wasted
- **Reserved headroom:** vLLM withholds a small buffer from the stated `gpu_memory_utilization` fraction
- **Activations and ephemeral buffers:** attention computation and sampling retain memory during a forward pass
- **Fragmentation-like effects:** not all free blocks are contiguous or schedulable for any given request

This writeup is the raw material for the postmortem's Root Cause Analysis and is exactly the kind of grounded discrepancy analysis that credible interview answers are built from.

---

## Step 3 — Baseline Run (3 hrs)

**Purpose:** Establish a healthy-system reference. Every failure metric in Phase B is expressed relative to this baseline.

### Start vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

### Workload

- **Prompt:** 512 tokens
- **max_new_tokens:** 256
- **Concurrency:** 4 steady concurrent requests

### Healthy-system baseline

| Metric | Value |
|---|---|
| TTFT p50 | ___ ms |
| TTFT p99 | ___ ms |
| TPS — output tokens/sec | ___ |
| Completed requests/min | ___ |
| KV blocks allocated | ___ |
| KV blocks free | ___ |
| Queue depth (waiting requests) | 0 (expected) |
| GPU memory (nvidia-smi) | ___ MB |
| Preemption count | 0 (expected) |

Store this before starting Step 4.

---

## Step 4 — Ramp to Failure (3 hrs)

**Key change from baseline:** 2,048-token prompts force high KV block consumption per request and drive exhaustion at achievable concurrency levels on T4.

### Workload parameters

| Parameter | Value |
|---|---|
| Prompt length | 2,048 tokens |
| max_new_tokens | 512 |
| Concurrency sweep | 1, 2, 4, 6, 8, 10, 12, 16 |
| Stabilization time per level | 2 min |

### Per-level metrics table

| Concurrency | KV Util % | Queue Depth | Running Reqs | Effective Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | Divergence Ratio (p99/p50) | Throughput (tok/s) | Preemptions | Event Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | — | | 0 | healthy | |
| 2 | | | | | | | | | | | |
| 4 | | | | | | | | | | | |
| 6 | | | | | | | | | | | |
| 8 | | | | | | | | | | | |
| 10 | | | | | | | | | | | |
| 12 | | | | | | | | | | | |
| 16 | | | | | | | | | | | |

**Event Label column:** Mark each row as one of: `healthy` / `preemption onset` / `cliff onset` / `terminal failure`.

These are distinct events — do not conflate them:
- **Preemption onset:** first preemption log event appears. System may still be stable.
- **Cliff onset:** TTFT p99 begins diverging from p50 non-linearly. Leading indicator that the positive-feedback loop has begun.
- **Terminal failure:** vLLM refuses all requests, crashes, or OOM. Record the error.

### Effective concurrency

```
effective_concurrency = running_requests + waiting_requests
```

Track this at every level. It makes the recompute feedback loop directly observable: as preempted requests re-enter the waiting queue, effective concurrency climbs even without new arrivals. The system "feels overloaded" before GPU saturation — because it is. This is also why GPU utilization is a misleading signal for autoscaling; effective concurrency rises first.

### Divergence ratio

```
divergence_ratio = TTFT_p99 / TTFT_p50
```

At healthy utilization this should be near 1.0–1.5. Define your cliff as the concurrency level where divergence_ratio first exceeds 2.0. This gives you a clean, numeric, reproducible cliff definition you can reference in the postmortem and in interview answers: "I define the cliff as divergence ratio > 2×, which occurred at N concurrent requests / X% KV utilization."

> **Connection to admission control:** This divergence threshold — not raw KV utilization — is the correct basis for admission control rejection. Your Phase A gateway enforced a token budget; Day 21 gives you the empirical evidence for where that budget's limit should sit.

### Throughput degradation signal

Track output tok/s at every level. Watch for the point where throughput **flattens or degrades despite increasing offered load** — this is the moment useful work stops growing even as the system appears "busy." This pattern is distinct from latency explosion and often appears first.

### Anti-pattern to internalize

> Operating at 85% KV utilization is unsafe even if throughput appears high, because divergence_ratio may already indicate impending instability. High throughput and low divergence_ratio are both required conditions for a safe operating point — neither alone is sufficient.

This is the empirical argument you will build in Deliverable #7 (Latency vs. Utilization Curve). Start forming it now.

### Rules during the ramp

1. **When preemption starts — do not stop.** The point is to observe V1 recompute-only preemption under increasing pressure.
2. **When vLLM fails:** capture the exact error message and last metrics, then stop.
3. **Redirect stderr to a log file** so the error is preserved: `2>> ramp_errors.log`

### What to observe about V1 preemption

In vLLM V1, a preempted request:
1. Loses its KV blocks
2. Re-enters the waiting queue
3. Must re-prefill from scratch when scheduled again
4. Re-consumes KV blocks that other requests need

This creates a **positive feedback loop**: more preemption → longer queue → higher effective concurrency → more preemption. Watch whether queue depth growth precedes TTFT p99 explosion — if it does, queue depth is the better leading signal (relevant for Day 28 autoscaling work).

**Questions to answer from your observations:**
- At what KV util % does preemption first appear?
- Does preemption stabilize the system or amplify failure?
- Does throughput flatten before or after TTFT p99 explodes?
- Is the cliff sharp or gradual?
- What is the exact error at terminal failure?

---

## Step 5 — Collapse Timeline Construction (30 min)

Fill in with your real numbers. Label preemption onset and cliff onset as separate rows — they may not coincide.

```
T=0: Concurrency=___, KV util=___%, Queue depth=___, TTFT p99=___ms, Throughput=___tok/s — HEALTHY

T=1: Concurrency=___, KV util=___%, PREEMPTION ONSET
     First preemption log event. TTFT p99=___ms (delta from baseline: +___ms)
     Queue depth=___. Throughput still growing? [yes/no]

T=2: Concurrency=___, KV util=___%, CLIFF ONSET
     TTFT p99 diverges from p50. p50=___ms, p99=___ms (ratio=___)
     Queue depth=___. Throughput peaked at ___tok/s and is now [flat/declining].
     Preemption rate=___ events/min.

T=3: Concurrency=___, KV util=___%, POSITIVE FEEDBACK VISIBLE
     Preemption rate=___ events/min (vs. T=1: ___)
     Queue depth=___ (growing? [yes/no])

T=4: TERMINAL FAILURE
     Error: ___________________________
     Last KV util: ___%, Last TTFT p99: ___ms, Last throughput: ___tok/s
```

---

## Optional Extension — Decode-Retention Pressure Profile (30 min)

**Run only if you finish Step 5 with time remaining. Else defer to Day 25 buffer.**

Repeat one concurrency level that caused preemption, but flip the pressure profile:

| Parameter | Ramp workload | Control workload |
|---|---|---|
| Prompt | 2,048 tokens (prefill-heavy) | 128 tokens (light prefill) |
| max_new_tokens | 512 | 2,048 (decode-retention-heavy) |
| Concurrency | Same failing level | Same failing level |

**What to observe:** Does the system collapse at the same concurrency? Earlier or later? The KV blocks consumed are similar in total, but the temporal pattern differs — prefill-heavy bursts blocks early; decode-retention holds blocks for the duration of the generation.

This is the first data point for the intuition that "collapse" is reachable through different pressure profiles, not just high concurrency. It directly seeds Day 23's prefill/decode interference analysis.

---

## Required Core Insight (write before Day 22)

Before starting the Day 22 postmortem, write this takeaway in your own words — 1–2 sentences, no hedging:

> "The system did not fail when KV cache was full. It failed when the rate of recompute-driven re-entry exceeded the system's ability to drain the queue — a positive feedback loop with no self-correcting mechanism."

Your version must reference your own numbers: the concurrency level, the KV utilization %, the point where effective concurrency began growing independently of new arrivals.

This is the mental model that separates "I observed collapse" from "I understand why it collapsed." The postmortem's Root Cause Analysis section is built on this sentence.

---

## End-of-Day Output Checklist

- [ ] Phase A instrumentation patch confirmed firing (Step 1)
- [ ] KV budget calibration completed with estimated vs. actual delta recorded (Step 2)
- [ ] "Why My Theoretical Budget Was Wrong" writeup complete — 3–5 sentences with named causes (Step 2)
- [ ] Healthy-system baseline stored with all columns filled (Step 3)
- [ ] Ramp table complete — queue depth, throughput, and preemption columns filled (Step 4)
- [ ] Preemption onset and cliff onset identified and labeled as separate events (Step 4)
- [ ] Terminal failure error message captured (Step 4)
- [ ] Collapse timeline complete with all four labeled stages (Step 5)
- [ ] Optional control experiment run (if time permitted)

---

## Connection to Phase B Key Numbers (Appendix C)

| Metric | Your Value | Status |
|---|---|---|
| T4 KV cache capacity (tokens) | | Fill today (Step 2) |
| Preemption onset (KV util %) | | Fill today (Step 4) |
| Cliff point (KV util %) | | Day 24 |
| Cliff safe operating point | | Day 24 |

---

## Common Mistakes

| Mistake | Consequence | Prevention |
|---|---|---|
| Using `num_kv_heads=16` instead of 2 | 8× capacity overestimate | Confirm GQA config for Qwen2.5-3B |
| Conflating preemption onset with cliff onset | Miss the leading-signal window | Label them separately in the table |
| Stopping the ramp at first preemption | Miss the full collapse curve | Keep going until terminal failure |
| Not tracking throughput during ramp | Lose the flattening signal | Add tok/s column before starting |
| Not recording queue depth | Lose the leading indicator data needed for Day 28 | Add queue depth column before starting |
| Not capturing error message at failure | Postmortem Root Cause is incomplete | Redirect stderr to log file |

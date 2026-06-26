# Day 9 (Thu) — Mini Collapse + Architecture Document

**Phase A · Week 2 · vLLM Internals**

**Prerequisites:** Day 7 code trace complete, Day 8 instrumentation patch deployed and validated.

**Engine note:** This syllabus targets **vLLM V1**. V1 removed CPU swap entirely. Preemption is **recompute-only**: KV blocks are discarded and the sequence is requeued for re-prefill. Any reference in older materials to a `SWAPPED` state or swap-path cost (`∝ KV size`) describes V0 behavior and does not apply here. Your logs will show no swap events — only recompute-driven TTFT spikes.

---

## Goals

By end of day you will have:
- Watched the system break under memory pressure — in real time, in your own logs
- Understood *why* it breaks (memory scheduling, not compute)
- Finalized the Week 2 deliverable: Annotated vLLM Architecture Diagram

---

## Morning (4 hrs) — Mini Collapse + Continuous Batching

---

### Block 1 — Mini Collapse Experiment (2 hrs)

This is the most important experiment in Week 2. You need one moment where the system breaks in front of you. The emotional anchor this creates makes Phase B collapse engineering hit harder.

#### Setup

Start vLLM with dangerously high memory utilization — almost no KV headroom:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.98 \
  --dtype half \
  --max-model-len 4096
```

#### Load Script

Fire 20 concurrent requests, each with a long prompt:

```python
import asyncio
import aiohttp
import time

async def send_request(session, i):
    prompt = "Summarize the following document:\n" + ("word " * 1000)  # ~4K tokens
    t0 = time.time()
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 200,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            latency = time.time() - t0
            print(f"[req {i}] latency={latency:.2f}s status={resp.status}")
            return latency
    except Exception as e:
        print(f"[req {i}] FAILED: {e}")
        return None

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(20)]
        latencies = await asyncio.gather(*tasks)
    print("\nLatency summary:", [f"{l:.2f}s" if l else "FAILED" for l in latencies])

asyncio.run(main())
```

**Note on workload design:** This script generates mostly prefill pressure (long prompt, short generation). If you want to observe longer decode-phase block retention — which holds blocks longer and makes preemption more likely mid-decode — use shorter prompts with longer generation instead:

```python
"prompt": "Tell me a very detailed story.",
"max_tokens": 1000,
```

Both variants are valid. The long-prompt version is simpler to reason about. The short-prompt/long-generation variant is closer to a real production collapse pattern.

#### What to Watch

Monitor your instrumentation logs in one terminal and vLLM's output in another. You are looking for this sequence:

1. `blocks_used / blocks_total` climbing toward 100%
2. Preemption events firing (look for your patch's log lines or vLLM's internal warnings)
3. TTFT spiking — preempted sequences are requeued and must re-prefill from scratch (V1 recompute path)
4. Requests stalling in `WAITING` as the scheduler backlog grows
5. Possible OOM or outright rejection if you push hard enough

#### Additional Metrics to Log

Block counts alone are insufficient to diagnose collapse. Also capture:

```
num_running_seqs   — how many sequences are actively being decoded
num_waiting_seqs   — how many are queued, waiting for blocks
scheduler_budget   — token budget consumed this iteration
GPU memory (MB)    — from nvidia-smi, correlated with block counts
```

Correlate block counts against raw GPU memory:

```bash
# In a second terminal while the experiment runs
nvidia-smi dmon -s mu -d 1
```

Example correlation you want to see in your notes:

```
blocks_used: 124 / 128
GPU memory:  15.7 GB / 16.0 GB
```

This builds intuition for how logical KV blocks map to physical HBM.

#### Record This Timeline

Use this format now — it's the same structure you'll use for Phase B postmortems:

| Elapsed | Event | blocks_used/total | num_running | num_waiting | TTFT (approx) |
|---------|-------|-------------------|-------------|-------------|---------------|
| T+0s    | Requests dispatched | ~10% | — | — | baseline |
| T+Xs    | Block util >80%, preemption begins | ~85% | — | — | rising |
| T+Ys    | TTFT spike, waiting queue grows | ~98% | — | — | >> baseline |
| T+Zs    | Failures / OOM / rejection | 100% | — | — | — |

Fill in the actual numbers from your logs.

#### Anchor Note

After the experiment, write 2–3 sentences: what happened, when it happened, and how it felt watching it. This is not a postmortem (that's Phase B). It is a personal record. Keep it honest.

#### Reset

```bash
# Restart with a safe value before continuing
--gpu-memory-utilization 0.90
```

---

### Block 2 — TTFT vs. Concurrency Curve (20 min)

Generate the single most useful graph in inference infrastructure.

Run the same load script at varying concurrency levels: 1, 2, 4, 8, 12, 16, 20 concurrent requests. Record mean TTFT at each level. Plot:

```
x-axis: concurrency (number of simultaneous requests)
y-axis: TTFT (ms)
```

You will see:

```
flat → slight slope → vertical cliff
```

The cliff is not gradual. It is abrupt. This is the capacity boundary of your system.

Save this graph. It will be referenced in your Phase A exit assessment and in Phase B.

---

### Block 3 — Continuous Batching Observation (30 min)

Send 10 requests staggered 200ms apart. In your instrumentation logs, observe the active sequence set changing across iterations:

```
iteration 1:  seqs [A]
iteration 2:  seqs [A, B]
iteration 3:  seqs [A, B, C]
iteration 4:  seqs [B, C, D]       ← A finished, D joined
```

If your patch logs `active_seq_ids` per iteration, this becomes directly visible. If not, you can infer it from block allocation/free events.

This is the Orca insight made concrete: **batch composition changes every forward pass**. New sequences join the moment a slot opens; they don't wait for the whole batch to finish. This is why continuous batching outperforms static batching at any meaningful load level.

---

### Block 4 — Buffer / Catch-Up (1 hr 10 min)

Use this time to:
- Finish anything incomplete from Days 7–8
- Refine your architecture diagram now that you have observed the preemption path firing in real logs
- Re-examine the scheduler trace from Day 7 with the collapse in mind — the `can_allocate()` path should now feel concrete

---

## Afternoon (4 hrs) — Finalize Week 2 Deliverable

---

### Gate: Interview Self-Test (15 min)

Do this before writing anything. Close your notes. Answer from memory in ≤ 3 sentences:

> **"Walk me through what happens when vLLM runs out of KV cache blocks under concurrent load."**

A complete staff-level answer covers:

1. Scheduler detects `can_allocate()` returns False for running or waiting sequences
2. Preemption policy triggers — in V1, this means KV blocks are discarded and the sequence is requeued for re-prefill (no swap path)
3. TTFT spikes for preempted requests (full re-prefill cost); `num_waiting_seqs` grows; new requests stall behind the backlog
4. Under sustained pressure the waiting queue grows faster than it drains — latency cascades system-wide

If you cannot cover all four points from memory, re-read your scheduler trace before continuing.

---

### Write: Annotated vLLM Architecture Diagram (3 hrs 45 min)

This is the Week 2 deliverable. It must be complete and self-contained — written as if an interviewer will read it without you present.

Required sections:

---

#### Section 1: Request Lifecycle

Trace the full path with actual file paths and key function names at each step:

```
[api_server.py]       POST /v1/chat/completions
                        → parse request, build SamplingParams
[async_llm_engine.py] generate()
                        → create SequenceGroup, add to waiting queue
[scheduler.py]        schedule()
                        → select sequences for this iteration
                        → can_allocate() check
[block_manager.py]    allocate() / free()
                        → assign physical KV blocks to sequence
[model_runner.py]     execute_model()
                        → GPU forward pass
                        → tokens returned to engine → streamed to client
```

For each node: key function names, data structures in, data structures out.

---

#### Section 2: SequenceGroup State Machine (V1)

Draw the formal state diagram. V1 has no SWAPPED state — the state machine is:

```
              ┌──────────┐
   arrival →  │ WAITING  │ ←──────────────────────────┐
              └────┬─────┘                             │
                   │ can_allocate() == True             │ preemption:
                   │ + scheduler budget available       │ blocks discarded,
                   ▼                                    │ sequence requeued
              ┌──────────┐                             │
              │ RUNNING  │ ──── memory pressure ───────┘
              └────┬─────┘
                   │ EOS token or max_tokens reached
                   ▼
              ┌──────────┐
              │ FINISHED │
              └──────────┘
```

For each transition, document:
- **WAITING → RUNNING:** `can_allocate()` returns True AND scheduler token budget is available
- **RUNNING → FINISHED:** EOS generated or `max_tokens` reached; blocks freed via `free()`
- **RUNNING → WAITING (recompute):** Memory pressure — scheduler discards KV blocks, sequence requeued for full re-prefill. Cost: proportional to prompt length.
- **WAITING → WAITING (allocation failed):** Scheduler tried to promote but `can_allocate()` returned False — sequence stays queued. This is the common steady-state under pressure.

Cite the function in `scheduler.py` responsible for each transition.

**Note:** The SWAPPED state and swap-path cost (`∝ KV size`) exist in vLLM V0 but are absent in V1. Do not include SWAPPED in your diagram. If you see it in older architecture references, treat it as V0-specific.

---

#### Section 3: Preemption Cost Annotation

In V1, preemption has one cost model:

```
Recompute path cost ∝ prompt length
  → the entire prefill must be re-executed
  → TTFT for the requeued request = original prefill time + wait time in queue
```

Annotate your state machine diagram with this cost. Relate it to the TTFT spike you observed in the mini-collapse experiment.

---

#### Section 4: Continuous Batching Path

Annotate where, within a single `schedule()` call, a finished sequence leaves and a new one joins. Specifically:
- When does a FINISHED sequence's blocks get freed?
- When does the scheduler see those blocks as available for a WAITING sequence?
- Does this happen within the same iteration or the next?

This is what makes continuous batching efficient: the GPU is never waiting for a full batch to drain before adding new work.

---

#### Section 5: Instrumentation Hook Points

Mark on the diagram where your Day 8 patch fires. For each hook point:
- File and function where the log line is inserted
- What it emits (fields, format)
- Example output line from your mini-collapse experiment

Then state your findings:
- Blocks allocated per request matches your theoretical KV math from Day 3 (within block-rounding)
- Concurrency level at which `blocks_used` approaches `blocks_total`
- What preemption events looked like in the logs

---

#### Section 6: Mini Collapse Observation

2–3 paragraphs. Cover:
- Setup: model, `--gpu-memory-utilization 0.98`, 20 concurrent 4K-token requests
- Timeline: when did blocks saturate, when did preemption fire, when did TTFT spike
- The TTFT vs. concurrency cliff graph (embed or reference it)
- What the `num_waiting_seqs` counter did during collapse
- One sentence on what you would add to prevent this (preview of Phase B admission control)

This section previews Phase B Postmortem #1. The data you collected today is the seed.

---

## End-of-Day Outputs

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Mini-collapse timeline table with block util, num_running, num_waiting, TTFT | ✅ |
| 2 | 2–3 sentence anchor note (what happened, how it felt) | ✅ |
| 3 | TTFT vs. concurrency cliff graph | ✅ |
| 4 | Continuous batching iteration log showing batch composition change | ✅ |
| 5 | **Week 2 Deliverable: Annotated vLLM Architecture Diagram** — all 6 sections complete | ✅ |

---

## What This Day Is Actually Teaching

The real lesson underneath the experiment:

> **LLM inference capacity is a memory scheduling problem, not a compute problem.**

Your collapse was caused by HBM exhaustion → scheduler preemption → growing WAITING queue → TTFT cascade. Compute was never the bottleneck. This is the core insight behind PagedAttention, continuous batching, KV quantization, and admission control — every major vLLM innovation addresses a memory scheduling constraint, not a FLOP constraint.

---

## What's Ahead

- **Day 10:** Polish the Week 2 deliverable. Interview self-assessment across all of Week 2. Buffer. Prep for Week 3 (quantization).
- **Phase B, Week 5:** You will return to this collapse — but systematically. Concurrency sweep at multiple levels, full postmortem write-up, scheduler heuristic modification to shift the cliff point. Today's data is the baseline.

---

*Phase A · Week 2 of 4 · vLLM Internals*

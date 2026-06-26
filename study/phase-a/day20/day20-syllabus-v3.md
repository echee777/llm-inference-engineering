# Day 20 — Phase A Exit
## AI Inference Platform Residency | Phase A v3

## Correction Table (v2 → v3)

| # | Location | Change | Source | Rationale |
|---|---|---|---|---|
| 1 | Q2 | Added explicit V1 behavioral shift description: try-and-fail allocation, no watermark threshold, token budget as first-class primitive | Reviewer R2 | Naming changes are cosmetic; behavioral changes are what interviewers probe |
| 2 | Step 5 Interview Brief | Added "Top 3 failure modes observed" item (trigger → mechanism → symptom format); adjusted structure to 5/3/3/2/2/1 | Reviewer R2 | Failure mode taxonomy is a canonical frontier-lab interview question; implicit coverage is not enough |
| 3 | Step 5 Interview Brief | Tightened production recommendation prompt: requires measurable threshold tied to specific experimental data, with explicit bad/good example | Reviewer R2 | Generic recommendation = observer; data-tied threshold = operator |
| 4 | Optional Step 6 | Added "Mock Staff Interview (30 min)" as optional afternoon step | Reviewer R2 | Verbal fluency under pressure is separate from written articulation; record + listen is the correct method |

## Correction Table (v1 → v2)

| # | Location | Change | Source | Rationale |
|---|---|---|---|---|
| 1 | Q1 | Replaced `AsyncLLMEngine` / `BlockManager` chain with V1-correct path: `LLMEngine` (V1) → `KVCacheManager` / `BlockPool` | Reviewer R1 | V0 vocabulary in a V1 context is an immediate credibility failure in a frontier-lab interview |
| 2 | Q2 | Renamed "SequenceGroup State Machine" → "Request State Machine in vLLM V1"; removed `SWAPPED` state; explicitly called out V1 removals (`SequenceGroup`, `BlockSpaceManagerV2`, `SchedulingBudget`, CPU swap) | Reviewer R1 | `SequenceGroup` does not exist in V1; Day 8 already established this |
| 3 | Q6 | Removed pre-filled shortcut (`~112MB/request`); replaced with first-principles derivation prompt requiring measured free-VRAM as input | Reviewer R1 | Memorized number does not survive interviewer follow-up; derivation chain is the actual answer |
| 4 | Q10 | Replaced `BlockManager` with `KVCacheManager` / `BlockPool`; added note to flag any V0 class names used in the patch | Reviewer R1 | Consistent with V1 terminology fix across Q1/Q2 |
| 5 | Afternoon Step 5 | Added Frontier Lab Interview Brief as new Step 5 deliverable (5/3/2/2/1 structure) | Reviewer R1 | Converts curriculum closure into recruiting-ready packet |
| 6 | End-of-Day Output | Updated deliverable table to include Deliverables 5 and 6 | Reviewer R1 | Reflects new Step 5 output |

---

**Date:** Day 20 (Friday)
**Theme:** Synthesis, closure, and Phase B preparation
**Structure:** Morning (4 hrs) polish + exit self-assessment | Afternoon (4 hrs) system review + Phase B prep

---

## Overview

No new experiments. No new code. Day 20 is where you prove to yourself (and prospective interviewers) that you actually understood the last four weeks — not just that you ran the experiments. The exit self-assessment is the highest-leverage thing you'll produce today.

---

## Morning Block (4 hrs) — Polish + Exit Self-Assessment

### Step 1 — Polish All Deliverables (1.5 hrs)

Before writing the self-assessment, do a final pass on all four deliverables:

| # | Deliverable | Check |
|---|---|---|
| 1 | GPU Architecture & Memory Budget Document | Nsight tables populated, fragmentation sim output included, TP cost preview section present |
| 2 | Annotated vLLM Architecture Diagram | State machine diagram finalized, instrumentation patch described, mini-collapse data cited |
| 3 | Quantization & Optimization Tradeoff Analysis | Prefill/decode split data present, KV capacity table correct (GQA-corrected), $/M tokens model complete |
| 4 | Admission Control Design Note | KV-math-derived budget, token correction improvement quantified, adversarial + HOL results included |

**Critical check on Deliverable #3:** Confirm your KV capacity figures used `num_kv_heads=2` (Qwen2.5-3B GQA). If any table used 16, patch it now.

---

### Step 2 — Phase A Exit Self-Assessment (2.5 hrs)

Write **1–2 paragraphs per question**. These are interview simulation prompts — answer as if you're at a whiteboard in front of a staff engineer at a frontier lab. Cite your own data and experiments, not generic facts.

---

#### Q1 — Request Lifecycle
> Full path from HTTP request → GPU → streamed token back to client. Cite vLLM V1 source files.

Trace the V1 path you actually walked in Days 6–8: `api_server.py` → `LLMEngine` (V1) → `Scheduler` (V1 redesigned) → `KVCacheManager` / `BlockPool` → forward pass → output processor → streamed token. Name the actual source files you opened. Explicitly note where V1 diverges from V0 (e.g., no `AsyncLLMEngine` in the same form, no `BlockSpaceManagerV2`, no CPU swap path). If you can cite a specific line number or function name from your Day 8 trace, do so.

---

#### Q2 — Request State Machine in vLLM V1
> Draw it, explain each transition. Be explicit about what V1 removed vs. V0 — both naming and behavior.

V1 states operate on `Request` objects, not `SequenceGroup`. Draw: `WAITING → RUNNING → PREEMPTED (recompute) → FINISHED_*`. Explicitly call out what does **not** exist in V1: no `SWAPPED` state, no CPU swap path, no `BlockSpaceManagerV2`, no `SchedulingBudget` object.

Then go one level deeper — explain the **behavioral** differences, not just the naming differences:
- V1 uses recompute-only preemption (simpler block ownership model, no pinned CPU memory)
- V1 scheduler uses try-and-fail block allocation rather than pre-check watermarks
- No swap watermark threshold — preemption is triggered by allocation failure
- Token budget is a first-class scheduling primitive in V1

Reference the state machine diagram you built on Day 7 — if it used V0 vocabulary, note the correction here. An interviewer will follow up on any behavioral claim, so only assert what you can defend from your Day 6–8 code trace.

---

#### Q3 — PagedAttention
> What problem does it solve? Reference your fragmentation simulation.

Lead with the problem: pre-PagedAttention KV cache is statically allocated per sequence at max length → internal fragmentation wastes 30–60% of HBM. PagedAttention borrows the virtual memory page table abstraction. Reference your Day 3 fragmentation simulation output — actual % waste under naïve vs. paged allocation.

---

#### Q4 — Quantization Tradeoffs
> Present your data. How does INT8 affect decode vs. prefill differently? Why? State KV capacity and $/M tokens impact.

The "why" is roofline: decode is memory-bandwidth-bound → INT8 halves weight bytes → nearly linear throughput gain. Prefill is compute-bound → weight size matters less → INT8 gain is smaller. Cite your Day 11 prefill/decode split data. Cite your Day 12 KV capacity table and $/M tokens model.

---

#### Q5 — GPU Utilization Is Misleading
> What metrics are more informative?

`nvidia-smi` GPU Util = "is any kernel running" — a single lightweight kernel pegs it at 100%. More informative: **Memory Throughput %** (are you saturating HBM bandwidth?), **Achieved Occupancy** (warps active vs. max), **SM Active %** (fraction of SMs doing useful work). Reference your Day 2 Nsight data.

---

#### Q6 — Concurrent Request Capacity
> How many concurrent 4K requests at FP16? INT8? Derive from first principles using your measured free-VRAM.

Do not recall a pre-computed number. Show the full derivation chain:

1. **Measured free HBM** after model weights + runtime overhead (your observed value from T4 experiments)
2. **Per-layer KV footprint** formula: `2 × num_kv_heads × head_dim × seq_len × bytes_per_element`
   — plug in Qwen2.5-3B architectural constants (`num_kv_heads=2`, `head_dim=128`) and your target seq_len
3. **Per-request total** across all 28 layers
4. **FP16 vs. INT8** comparison: show the bytes_per_element change and resulting capacity delta
5. **Utilization target**: state your assumption (e.g., 85%) and why (empirical collapse threshold from Day 17/19)
6. **Final answer**: concurrent capacity at FP16, concurrent capacity at INT8

An interviewer can verify every step. A memorized number cannot be verified and will not survive follow-up questions.

---

#### Q7 — 10 Simultaneous Long-Context Requests
> Walk through in memory terms. When does preemption start? What happens to latency?

When KV cache pool is exhausted, the vLLM V1 scheduler preempts via recompute (no CPU swap). The preempted sequence loses its KV cache blocks, must be re-prefilled when re-admitted → TTFT for the preempted sequence spikes. Reference your Day 9 mini-collapse observation: at what queue depth / token count did you see TTFT degrade?

---

#### Q8 — Admission Control as Memory Budget
> Why derived from KV math, not concurrency count. How does token budget correction improve capacity.

Flat concurrency cap (e.g., "max 10 requests") is wrong because request sizes vary by 10–80×. A 100-token request and an 8K-token request are not equivalent. The correct primitive is: **how many KV tokens can HBM hold simultaneously?** Token budget correction: instead of reserving `max_completion_tokens` per request upfront, release budget as tokens are actually generated. Cite your Day 17 measured improvement: X% more requests admitted with correction.

---

#### Q9 — Adversarial Requests
> What happens when one client sends a massive request? Why is FIFO queuing insufficient?

From your Day 19 experiments: a single 8K+4K request can consume the entire token budget, causing Tenant A's short requests to queue or be rejected. FIFO is insufficient because it provides no isolation — a pathological request at the front of the queue starves all subsequent requests regardless of their size. This previews Phase C: weighted fair queuing and per-tenant budgets.

---

#### Q10 — Your Instrumentation Patch
> What you added, what it revealed. Use V1-correct component names.

Reference your Day 8 Option A patch: block allocation logging in the V1 `KVCacheManager` / `BlockPool` layer (not V0's `BlockManager`). Describe: what function or event you hooked, what the log output showed (block allocation and free events, fragmentation visibility across concurrent requests), and what it revealed that `nvidia-smi` alone cannot show — specifically, the difference between "GPU is busy" and "KV cache pool is under pressure." If your patch used any V0 class names, note the correction and what the V1 equivalent is.

---

## Afternoon Block (4 hrs) — System Stability Review + Phase B Prep

### Step 3 — System Review (2 hrs)

Ensure the full stack is stable and repeatable:

- `vLLM` V1 serving Qwen2.5-3B on T4 — cold-start works cleanly
- Admission gateway (FastAPI) — token budget enforcement, per-API-key rate limiting, FIFO queue all functional
- Locust load test config saved and re-runnable
- Prometheus + Grafana dashboards — metrics flowing, no stale data

Document any known brittleness (e.g., T4 OOM edge cases, gateway startup ordering) — these become Phase B experiment targets.

---

### Step 4 — Phase B Mental Preparation (2 hrs)

Phase B shifts the posture from **build** to **break and analyze**:

| Phase A | Phase B |
|---|---|
| Build understanding | Systematically break what you built |
| Instrument and observe | Induce failures and measure collapse |
| Derive admission budget | Push past it, postmortem why |
| Single GPU (T4) | Multi-GPU (A10G-class, RunPod/Vast.ai) |
| Steady-state behavior | Collapse behavior and recovery |

Phase B Week 5 will have you intentionally exceed your admission limits, modify a scheduler heuristic to shift the collapse point, and produce your first full postmortem. Your Day 9 mini-collapse is the emotional anchor — Phase B is that, but systematic and documented.

**Hardware note:** Phase B Track 2 requires multi-GPU (A10G-class). If you haven't already, now is the time to set up RunPod or Vast.ai accounts and confirm you can spin up an A10G instance.

### Step 5 — Frontier Lab Interview Brief (1 hr)

Distill the self-assessment into a **1-page recruiting-ready artifact**. This is what you reference before a phone screen. Write it in your own voice — these are your conclusions from four weeks of measurement, not generic inference theory.

Structure:

**Top 5 operating insights** (one sentence each — things you now know empirically that you didn't before)

**3 experiments you would present first** (the ones with the clearest data and most defensible conclusions)

**Top 3 failure modes observed**, each in the format: `trigger → mechanism → observable symptom`
- e.g., memory exhaustion → preemption cascade → TTFT spike
- e.g., queue saturation → HOL blocking → p99 latency amplification
- e.g., adversarial large request → budget starvation → Tenant A rejection

**2 mistakes you caught and corrected** (e.g., GQA head-count error, V0 vs. V1 nomenclature — being explicit about corrections demonstrates rigor, not weakness)

**2 places where vLLM V1 architecture changed your mental model** (relative to what you assumed going in or what older docs describe)

**1 crisp production recommendation you can defend** — must be measurable, tied to your data, not generic.
- Bad: "Set a conservative admission budget"
- Good: "Set admission budget to X% of KV capacity based on observed latency cliff at Y% utilization in Day 17/19 experiments"
- The number must come from your data. If you don't have a specific threshold, state what experiment would produce it.

This artifact is not for the curriculum. It is for the interview.

---


### Optional Step 6 — Mock Staff Interview (30 min)

Answer Q1–Q10 out loud, no notes, record yourself. Listen once. You will catch: gaps in clarity, overlong answers, missing transitions between concepts. This is the last mile — the self-assessment is preparation for being asked these questions under pressure, not a substitute for it.

---

## End-of-Day Output

Phase A complete. All four deliverables finalized, plus two synthesis artifacts:

| # | Deliverable | Status |
|---|---|---|
| 1 | GPU Architecture & Memory Budget Document | ✅ |
| 2 | Annotated vLLM Architecture Diagram | ✅ |
| 3 | Quantization & Optimization Tradeoff Analysis | ✅ |
| 4 | Admission Control Design Note | ✅ |
| 5 | Phase A Exit Self-Assessment (10 questions, 1–2 paragraphs each) | ✅ |
| 6 | Frontier Lab Interview Brief (1 page, 5/3/2/2/1 structure) | ✅ |

---

*AI Inference Platform Residency — Phase A v3 | day20-syllabus-v3.md*

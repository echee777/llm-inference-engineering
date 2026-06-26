# Day 22 — Postmortem #1 + Mixed-Length Setup

**Phase B / Track 1 — Collapse Engineering**
**Deliverable Due:** Postmortem #1 (Deliverable #5)
**Hardware:** T4 / g4dn.xlarge

---

## Before You Write Anything — Required Gate

Before drafting, state your core insight in one sentence:

> *"vLLM V1's recompute-only preemption creates a positive feedback loop — preempted requests re-enter the queue, re-consume KV blocks, and amplify the exhaustion they caused."*

This prevents writing a symptom description instead of a root cause analysis. If you cannot state the mechanism in one sentence, go back to your Day 21 data before proceeding.

Also prepare: **"Why My Budget Was Wrong"** — a required named section. What did your theoretical token ceiling predict vs. what you actually observed? The Qwen2.5-3B GQA correction (`num_kv_heads=2` not 16) is the likely culprit if your budget diverged.

---

## Morning (4 hrs) — Write Postmortem #1

**Required gate: no prose drafting until your five graphs and timeline table exist with real numbers.** Writing narrative before the evidence is assembled produces elegant postmortems disconnected from data. Build the evidence base first.

Work through each section in order. Do not skip ahead.

---

### Step 1 — Summary (1 paragraph)

System-level, no speculation. Only what your metrics showed.

Template:
> *"Under [N concurrent requests / prompt length X], KV cache utilization reached Y%. vLLM V1 began preempting at Z% utilization. Preemption did not stabilize the system — it accelerated failure because [recompute mechanism]. Total failure at [concurrency/KV util level]."*

Fill in real numbers from Day 21.

---

### Step 2 — Timeline

Pull your Day 21 collapse timeline and populate with actual measurements:

```
T=0: Concurrency=N, KV util=X%, TTFT p99=Y ms — healthy
T=1: First preemption event at KV util=Z%
T=2: TTFT p99 begins rising non-linearly
T=3: Preemption rate accelerates, queue builds
T=4: [OOM / total rejection / crash — with final KV util % and concurrency]
```

Every line requires a real number. No placeholder values in the final document.

---

### Step 3 — Root Cause Analysis

Three-part structure:

- **Trigger:** The chosen prompt-length × concurrency pattern — long-context requests at concurrency N.
- **Primary root cause:** KV cache memory exhausted under concurrent long-context requests
- **Contributing root cause:** No admission control tuned to this specific prompt-length × concurrency product
- **Mechanism:** As concurrent requests grew, KV blocks were consumed faster than freed. V1 recompute-only preemption (no CPU swap path) means preempted requests re-enter the waiting queue and re-consume the same blocks — creating a positive feedback loop.

*Separating trigger from root cause is required. The trigger is what you chose to do; the root cause is why the system had no capacity to absorb it. This distinction will matter again in Postmortem #2.*

---

### Step 4 — What vLLM's V1 Scheduler Did

Answer these three questions with your actual data:

1. At what KV utilization % did preemption begin?
2. Did preemption stabilize the system, or did it worsen it?
3. Why? *(The mechanism: recompute re-queues the request → it competes for the same blocks it just released → net effect is zero or negative under high concurrency.)*

**Note:** Preemption begins when block allocation fails for new or running requests — not when memory is fully exhausted. This means the system enters an unstable regime before hitting the hard ceiling. Your preemption-onset KV utilization % (from question 1) is the evidence for this: if preemption started at 72% rather than 100%, the system was already degrading with ~28% of KV capacity nominally available. This pre-cliff instability is what Day 24's utilization cliff experiment will characterize precisely.

---

### Step 5 — Four Required Graphs

All five are required. Pull from Day 21 data:

1. KV cache utilization % over time
2. TTFT p50 and p99 over time (same time axis as graph 1)
3. Preemption event count over time
4. Concurrency vs. KV utilization (scatter)
5. Queue depth / waiting requests (`num_waiting_seqs`) over time — this is the clearest evidence for the recompute feedback loop. Preemption count shows events; queue depth shows the amplification effect those events produce.

If any of these are missing from your Day 21 data, re-run the relevant segment before proceeding with the writeup.

---

### Step 6 — Lessons Learned

Three required bullets:

1. KV cache capacity is the hard ceiling, not GPU compute.
2. In V1, recompute-only preemption can exacerbate exhaustion under high concurrency.
3. **Show the math:** at your measured token budget, `max_concurrent_requests = budget_tokens / prompt_tokens`. If you'd enforced that ceiling, the system would likely have rejected or shed load earlier instead of entering collapse. Use your actual numbers to show where that ceiling sits relative to the concurrency level where failure occurred.

---

### Step 7 — Remediation

Three concrete items:

- Tighten admission control: enforce `max_prompt_length × max_concurrency` product limit
- Add KV utilization as a real-time admission signal (not just a static token budget estimate)
- Set `--max-model-len` tighter for high-concurrency deployments

---

### Step 8 — "Why My Budget Was Wrong" (named section, required)

Compare your theoretical capacity ceiling from Day 21 morning (the `~180,000 token` estimate at `--gpu-memory-utilization 0.90`) against what vLLM actually allocated.

If your estimate diverged from observed behavior, explain why. If you used `num_kv_heads=16` anywhere in your estimate instead of the GQA-correct value of `2` for Qwen2.5-3B, that is an 8× overestimate of KV heads — document it explicitly here with the corrected math.

**Corrected KV bytes per token (Qwen2.5-3B-Instruct):**
```
KV bytes per token per layer = 2 × num_kv_heads × head_dim × 2 bytes
                             = 2 × 2 × 128 × 2 = 1,024 bytes
Per token across all layers  = 1,024 × 36 = 36,864 bytes ≈ 36 KB
```

---

## Afternoon (4 hrs) — Mixed-Length Experiment Setup

This feeds directly into Deliverable #6 (Day 23). Collect clean data today; analyze tomorrow.

---

### Step 1 — Experiment Design (1 hr)

**Traffic mix:**
| Type | Prompt Tokens | max_new_tokens | Share |
|---|---|---|---|
| Short | 64 | 128 | 50% |
| Long | 2,048 | 512 | 50% |

**Target KV utilization:** ~65% — deliberately below your cliff point. You are isolating *scheduling interference*, not re-inducing exhaustion.

**Tagging:** Each request must carry a `short` / `long` tag so latency distributions can be split post-run.

**Key question:** Do short requests suffer disproportionately when batched with long requests?

---

### Step 2 — Run A: Mixed Traffic (part of 3-hr block)

- Duration: 10 minutes
- Log per-request latency tagged by type
- Record TTFT p50/p99 for short requests and long requests separately
- Save raw per-request log to CSV

---

### Step 3 — Run B: Short-Only Control (part of 3-hr block)

- Short requests only (no long requests)
- **Control variable: target the same measured KV utilization as Run A (~65%), not just the same request concurrency.** Short requests consume far fewer KV blocks per request — matching concurrency without matching KV utilization puts the two runs in different scheduler regimes and confounds the interference penalty number.
- Record TTFT p50/p99

**Target metric:** `p99_TTFT(short, mixed) / p99_TTFT(short, isolated)` — the interference penalty. This is the number you will report in Deliverable #6.

---

## End-of-Day Checklist

- [ ] Postmortem #1 complete with all 8 sections (trigger/root-cause distinction present)
- [ ] All five required graphs present (including queue depth / `num_waiting_seqs`)
- [ ] "Why My Budget Was Wrong" section named and filled in
- [ ] Mixed-length Run A raw data saved (tagged per request type)
- [ ] Short-only Run B control data saved
- [ ] Interference penalty number calculated: `p99_TTFT(short, mixed) / p99_TTFT(short, isolated)`

---

## What Day 23 Expects

You arrive with:

1. **Postmortem #1 complete** (Deliverable #5 done)
2. **Mixed-length raw latency data** from Run A (tagged per request type)
3. **Short-only control baseline** from Run B

Day 23 morning begins immediately with CDF plotting and the chunked prefill experiment. No data collection catch-up on Day 23.

---

## Interview Signal

Deliverable #5 answers these Staff-level questions:

| Question | Your Answer |
|---|---|
| "Why did latency spike?" | KV exhaustion → recompute preemption → positive feedback loop |
| "What metric do you check first?" | KV utilization %, not GPU compute utilization |
| "What's the real bottleneck in LLM serving?" | Memory capacity, not compute throughput |
| "Tell me about a system that failed." | Postmortem narrative: what broke, what you expected, what surprised you, what you changed |

The Narrative Section (what broke / what you expected / what surprised you / what you changed) is the raw material for behavioral interview answers. Write it as if you are answering "tell me about a time a system you owned failed."


---

## v1 → v2 Correction Table

| # | Reviewer Suggestion | Decision | Rationale |
|---|---|---|---|
| 1 | Soften "rejected at N rather than collapsing at N+4" | **Accept** | Phrasing implied measured precision not in the data. Reworded to "rejected or shed load earlier instead of entering collapse." |
| 2 | Add queue depth / `num_waiting_seqs` as 5th required graph | **Accept** | `num_waiting_seqs` is the direct evidence for the recompute feedback loop — preemption count shows events, queue depth shows amplification. Omitting it left the feedback loop asserted rather than demonstrated. |
| 3 | Control Run B by KV utilization, not concurrency | **Accept** | Short-only traffic at the same concurrency as mixed traffic lands at a substantially lower KV utilization. Different scheduler regime contaminates the interference penalty number. Controlling for KV utilization isolates the right variable. |
| 4 | Add trigger vs. root cause distinction in RCA | **Accept** | Trigger = concurrency/prompt-length pattern chosen; root cause = memory exhaustion + recompute-loop amplification. Staff-level framing habit, and sets up Postmortem #2 cleanly. |
| 5 | Enforce "graphs before prose" explicitly | **Partial Accept** | Added as a bold gate at the top of the morning block. Kept existing "re-run if graphs missing" guidance — both serve different purposes (one prevents premature writing, the other handles missing data). |

---

## v2 → v3 Correction Table

| # | Suggestion | Decision | Rationale |
|---|---|---|---|
| 1 | Add note to Step 4: preemption begins before full memory exhaustion | **Accept (optional upgrade)** | Technically correct and high-signal. Connects Step 4's scheduler behavior directly to Day 24's cliff experiment. The example numbers (72% onset vs. 100% hard ceiling) make the pre-cliff instability regime concrete rather than abstract. |

# Day 24 — Latency vs. Utilization Cliff
## v5 Corrections (applied on top of v4)

| # | Change | Reviewer Suggestion | Decision |
|---|--------|-------------------|----------|
| 9 | Tightened cliff confirmation: preemption_rate must show monotonic increase across adjacent utilization points; single noisy point does not qualify | Micro-adjustment 1: define "rising" more precisely | Accept |
| 10 | Added synthesis sentence at end of Step 4 grounding the regime-transition framing in measured signals | Micro-adjustment 3: connecting sentence | Accept — "comparable to" replaced with "non-trivial fraction of" for internal consistency |
| — | Cliff point definition: (p99/p50) increases by ≥X% vs previous point | Micro-adjustment 2 | Reject — X left undefined; contradicts established divergence ratio >2× definition from Day 16; pseudo-precision |

## v4 Corrections (applied on top of v3)

| # | Change | Reviewer Suggestion | Decision |
|---|--------|-------------------|----------|
| 7 | Tightened "sustained" definition to ΔQueue/Δt > 0 over ≥80% of sampling window | Gap 1: Define "sustained" more precisely | Accept |
| 8 | Added explicit margin decomposition: cliff uncertainty (from ±std), traffic variance, scheduler jitter — margin should be ≥ dominant term | Gap 3: Tie cliff → decision margin explicitly | Partial accept — named three components; rejected numeric formula (denominator for recompute_fraction not directly observable) |
| — | Normalize recompute load as fraction of total_tokens_generated | Gap 2: recompute_fraction metric | Reject — divides one estimate by another unobservable; "30–40%" threshold is fabricated; absolute metric already tells the story |

## v3 Corrections (applied on top of v2)

| # | Change | Reviewer Suggestion | Decision |
|---|--------|-------------------|----------|
| 4 | Added explicit queue instability criterion: ΔQueue/Δt > 0 sustained over sampling window; compound condition with preemption_rate rising as cliff confirmation | Gap 1: Missing queue stability criterion | Accept |
| 5 | Added recompute load estimate: preemption_rate × avg_tokens_lost (estimated from request logs, flagged as derived not directly measured) | Gap 2: Recompute cost measurement | Partial accept — formula valid; flagged as estimate since V1 does not expose tokens_lost natively |
| 6 | Added variance treatment for repeat runs: mean ± 1 std for TTFT p50/p99 in 60–75% zone, plotted as shaded band on primary graph | Gap 3: Error bars / variance | Accept |
| — | "Elite-level" / "top 1%" framing and closing offer | — | Enthusiasm, not analysis. No changes. |

## v2 Corrections

| # | Change | Reviewer Suggestion | Decision |
|---|--------|-------------------|----------|
| 1 | Added actual vs. target KV util verification requirement (>3% divergence → re-stabilize) | Pitfall 1: "Fake" utilization control | Accept |
| 2 | Added queue growth rate (ΔQueue/Δt) as explicit metric in transition zone | Pitfall 2: Queue growth rate as instability signal | Partial accept — depth already listed; growth rate added as derived signal |
| 3 | Added 2–3 repeat runs requirement for 60–75% transition zone | Pitfall 3: Single-run cliff estimates unreliable due to T4 scheduling jitter | Accept |
| — | "Recompute work > forward progress" framing | Pitfall 4 | Reject — already covered in three-regime explanation; phrasing is interview delivery, not a syllabus gap |
| — | Quantified strong-answer example | "Strong answer" section | Reject — numbers are fabricated pre-experiment; Decision Section template already forces real data |

---



**Deliverable:** #7 — Latency vs. Utilization Curve
**Hardware:** T4 / g4dn.xlarge

The syllabus flags this as **one of the two most important experiments in the entire program.** The output — your cliff graph — becomes the evidence you cite in every downstream conversation about operating point selection, retry budgets, autoscaling triggers, and the cost/reliability tradeoff.

---

## Morning (4 hrs) — Cliff Experiment

### Step 1 — Design (30 min)

You're doing a controlled utilization sweep, not a ramp-to-failure. The key differences:

- **Workload:** homogeneous, medium prompts (~512 tokens). This isolates KV utilization as the independent variable. Don't mix lengths here — that was Day 22/23. You want one clean curve.
- **Independent variable:** KV utilization %, controlled by adjusting request rate at each step. Use your Phase A gateway + instrumentation patch to verify actual utilization. If actual KV util diverges from target by >3%, adjust rate and re-stabilize before sampling — do not record a point where target and actual disagree. Your x-axis is only valid if it reflects measured utilization, not intended utilization.
- **Points:** 10 operating points — 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%
- **Stabilization:** 3 minutes at each point before you sample. Don't shortcut this — transients will contaminate your cliff shape.
- **Sample window:** 2 minutes of metric collection per point after stabilization.
- **Metrics to record at each point:**
  - TTFT p50 / p95 / p99
  - Inter-token latency p50 / p99
  - Queue depth
  - Queue growth rate (ΔQueue/Δt) — especially near the transition zone; sustained positive growth rate is a cleaner instability marker than absolute depth alone. **Instability criterion: ΔQueue/Δt > 0 over ≥80% of the 2-minute sampling window (this threshold avoids false positives from short bursts). Cliff confirmation: ΔQueue/Δt > 0 (sustained) AND preemption_rate shows monotonic increase across adjacent utilization points — either condition alone may be transient; a single noisy point does not qualify.**
  - Preemption rate (events/min)
  - KV util % (confirm from instrumentation, not just target)

### Step 2 — Run (3.5 hrs)

Mechanically: set rate → wait 3 min → sample 2 min → record → increment.

What to watch for:
- **40–65%:** Should be relatively flat. TTFT p50 and p99 track closely. Preemption is rare.
- **65–75%:** This is the transition zone. Density here matters. Don't skip points. **Run 2–3 repeats at each point in this range** — scheduling jitter on the T4 is sufficient to shift an apparent cliff by ±5–10%. Single-run cliff estimates in this zone are unreliable. Outside this zone (flat 40–60%, clearly degraded 80–85%), single runs are sufficient.

- **75–85%:** Non-linear explosion expected. p99 begins diverging hard from p50. Preemption rate climbs.

The cliff shape is the finding. Your specific cliff point — the utilization % where p99 starts rising *faster* than p50 — is the number that goes into Appendix C and gets cited for the rest of Phase B and Phase C.

---

## Afternoon (4 hrs) — Graph + Write Cliff Document

### Step 3 — Plot (1.5 hrs)

Two required graphs:

1. **Primary:** KV utilization % (x-axis) vs. TTFT p50/p95/p99 (y-axis, three lines). Mark your cliff point explicitly on the graph.
2. **Secondary:** KV utilization % vs. preemption events per minute.

The two graphs together tell the mechanistic story: the preemption rate curve will inflect at the same point the TTFT curves diverge. That's not coincidence — it's the causal link.

**Variance treatment for repeat runs:** For the 60–75% transition zone where you ran 2–3 repeats, compute mean ± 1 std for TTFT p50 and p99 across runs. Plot as a shaded band around each line in that region of the primary graph. Outside the transition zone, single-run points are sufficient. This makes your cliff point estimate defensible: you can state the cliff at X% ± Y% rather than a point estimate that may reflect a single noisy run.

### Step 4 — Mechanistic Analysis (1 hr)

Write the three-regime explanation:

- **Below the cliff:** Occasional preemption, manageable queue, recompute cost is rare. Feedback loop is damped.
- **At the cliff:** Preemption frequency crosses a threshold where recomputed requests re-compete for the same blocks — the positive feedback loop activates.
- **Above the cliff:** Self-amplifying — more preemption → longer queue → higher effective concurrency → more preemption. p99 becomes non-linear because you're now measuring the tail of a biased queue, not just service time variance.

This is the V1-specific dynamic: recompute-only preemption means preempted requests re-enter the *waiting queue* and re-consume KV blocks when rescheduled. There's no CPU swap valve to buffer them. You observed the onset of this feedback loop on Day 21 — today you're mapping its shape precisely.

**Recompute load estimate (derived metric):** For each operating point in the 60–80% range, compute:
```
recompute_load ≈ preemption_rate (events/min) × avg_prompt_tokens_of_preempted_requests
```
`avg_prompt_tokens_of_preempted_requests` is not directly exposed by vLLM V1 — estimate it from your request logs (you know the prompt length distribution you sent). Flag this as an estimate in your deliverable. The value is directional: it shows that at the cliff, recompute work becomes a non-trivial fraction of total forward pass work, which explains mechanistically *why* the feedback loop activates rather than damping out.

**Synthesis:** The cliff corresponds to the point where recompute load becomes a non-trivial fraction of forward progress, causing the system to transition from a stable to an unstable queueing regime. This framing is now grounded in your measured signals — preemption rate, queue growth rate, and recompute load estimate — not just the shape of the TTFT curve.

### Step 5 — Write Deliverable #7 (1.5 hrs)

Structure:

1. **Cliff graph** (both graphs)
2. **Cliff point statement** — e.g., "Cliff observed at 68% KV utilization on T4/Qwen2.5-3B-Instruct. Below 68%, p99 TTFT is within 1.4× of p50. Above 68%, the ratio exceeds 2× and grows non-linearly."
3. **Mechanistic explanation** — 1–2 paragraphs, the three regimes above
4. **Recommended operating point** — cliff minus safety margin. If cliff is at 68%, recommend ≤65%. The margin is not arbitrary — name its three components explicitly: (a) **cliff uncertainty**: your ±std from repeat runs in the transition zone defines the lower bound here; if your cliff is 68% ± 4%, a 3% margin is insufficient; (b) **traffic variance**: peak/sustained ratio from your Day 21 data — a bursty workload can overshoot the steady-state operating point transiently; (c) **scheduler jitter**: vLLM's batch composition varies iteration-to-iteration, causing KV util to fluctuate even at fixed request rate. Your margin should be at least as large as the dominant of these three terms.
5. **Decision Section (required for this deliverable)**

---

## Decision Section — Required Format

This is not optional for Deliverable #7. Fill it with your actual numbers:

```
Given:
  [measured cliff at X% KV utilization on T4/Qwen2.5-3B-Instruct]
  [p99 TTFT at X% vs. p99 at X+5%: specific ms values from your table]

I would choose:
  Operate at ≤(X-3)% KV utilization

Because:
  1. p99 TTFT at X% is [Y ms]. At X+5%, it is [Z ms] — a [ratio]× increase.
  2. Preemption rate at X% is [N events/min]. The recompute cost adds [M ms] avg per affected request.
  3. Safety margin covers traffic variance: peak/average ratio from your Day 21 data was [ratio].

What gets worse because of this decision:
  Cost per token increases ~[%] due to operating at [X-3]% vs. [X+5]% utilization.
  GPU idle headroom is [%] — hardware is underutilized vs. theoretical max.

What I am explicitly NOT optimizing for:
  Peak throughput — accepting [N]% lower token/sec vs. theoretical max
  Hardware utilization — accepting idle GPU cycles as insurance against tail latency

What would make me change this decision:
  Admission control with real-time KV utilization feedback (can tighten the margin)
  Chunked prefill uniformly enabled (reduces variance, may shift cliff point right)
```

The strength of the deliverable is the specificity. Weak version: "operate conservatively below the cliff." Strong version: "operate at ≤65%, because at 70% my p99 triples and the feedback loop becomes self-sustaining — here is the graph."

---

## End-of-Day Output Checklist

- [ ] 10-point metrics table (KV util % → TTFT p50/p95/p99, preemption rate, queue depth)
- [ ] Primary cliff graph with cliff point marked
- [ ] Secondary preemption rate graph
- [ ] Cliff point number entered in **Appendix C** (cited on Days 26, 28, 29, and throughout Phase C)
- [ ] Deliverable #7 written with Decision Section complete

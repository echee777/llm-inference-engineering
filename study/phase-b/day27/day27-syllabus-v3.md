# Day 27 — Postmortem #2: Retry Cascade + Autoscaling Signal Prep
**Phase B Track 1 | Week 6 | T4/g4dn.xlarge**
**Deliverable:** #8 — Postmortem #2: Retry Cascade
**Version:** v1

---

## Prerequisite Check

Day 26 retry storm experiment must be complete before starting. Confirm you have:

- [ ] Amplification factor time series (unique request rate vs. total request rate over time)
- [ ] Collapse and recovery timeline with real timestamps
- [ ] TTFT p50/p99 at each phase of the cascade
- [ ] The exact timestamp when the burst ended vs. when collapse completed (these must differ)
- [ ] Queue depth and KV utilization readings through the cascade

If any of these are missing, finish Day 26 data collection first. Postmortem #2 cannot be written without them.

---

## Morning (4 hrs) — Write Postmortem #2 (Deliverable #8)

**Objective:** Produce a staff-grade postmortem on a qualitatively distinct failure mode from Postmortem #1. The key distinction: the trigger was transient; the collapse was self-sustaining. Every section must make that argument with your data.

---

### Required Sections

#### Summary (1 paragraph)

Template — fill in from your Day 26 data:

> System was healthy at ~60% KV utilization. A 30-second traffic burst triggered client retries that amplified effective load to [X]% — past the cliff defined in Deliverable #7 (divergence ratio p99/p50 > 2×). By the time the burst ended at T=[timestamp], the retry-driven load had become self-sustaining. Full cascade occurred at T=[timestamp], [N] seconds after the original burst had already stopped.

The last sentence is the thesis. If your timeline doesn't show this, re-examine the data — it's there.

---

#### Timeline

Minute-by-minute. Include real metric values at each row. Mandatory annotations:

```
T+0:00  Burst injection begins. Unique req/s = [N]. KV util = ~60%. TTFT p99 = [X]ms (healthy).
T+0:30  Burst ends. Total req/s = [M] (retries already in flight). KV util = [Y]%.
T+X:XX  Amplification factor crosses 1.5×. Queue depth = [N]. TTFT p99 = [Z]ms.
T+X:XX  Amplification factor = [peak]. Divergence ratio (p99/p50) > 2× — cliff crossed.
T+X:XX  Full cascade. System rejecting / crashing. KV util = [Z]%.
T+X:XX  Burst injection already ended [N] seconds ago.
T+X:XX  Recovery begins (after load shedding / manual intervention).
T+X:XX  System returns to baseline TTFT.
```

The gap between "burst ends" and "collapse occurs" is your empirical evidence for self-sustaining cascade. Annotate it explicitly.

**Recovery lag annotation (required):** The recovery segment is not just a timestamp — annotate it with the mechanical reason recovery is slow. The forcing function (the burst) is gone, but three conditions persist: (1) retried requests are still in flight and consuming KV blocks, (2) the queue is already poisoned with backed-up work, (3) the system is still operating above the stable region defined by your Day 24 cliff. Recovery cannot begin until all three conditions clear, which takes longer than the burst duration. If your data shows recovery took [N] seconds after burst-end, that duration should be explainable by these three factors — not left as an observation without a cause.

---

#### Amplification Graph (required)

Dual-axis time series. This is the central evidence artifact for this postmortem.

- Y-left: unique request rate (original client requests, excluding retries)
- Y-right: total request rate (unique + retries hitting the server)
- X: time
- Annotate: divergence point (amplification factor > 1.0), cliff crossing, collapse, burst-end marker

The visual argument: the two lines separate and don't reconverge until after collapse. The burst-end marker sits left of the collapse marker.

---

#### Root Cause

Two sentences. No hedging:

1. Positive feedback loop: elevated latency → client timeouts → retries → increased effective load → further latency elevation.
2. The triggering burst was transient and self-limiting; the retry-driven load was neither.

---

#### Why Inference Is Especially Vulnerable

This section earns interview points by distinguishing GPU inference from the general distributed systems case. The argument:

| Service type | What retries hit | Effect |
|---|---|---|
| CPU-bound stateless | Many instances across a fleet | Retries dilute — each retry lands on a different instance, spreading load |
| GPU inference (single instance) | One fixed KV cache pool | Retries concentrate — every retry competes for the same memory budget |

The inference-specific consequence: there is no "scale out and absorb retries" path within a single serving instance. The KV pool is the hard ceiling. More retries do not find slack capacity — they consume capacity from requests that would have succeeded.

Cross-reference: your cliff point from Deliverable #7. The retry cascade pushed the system past that exact threshold. The cliff was not a new failure — it was the same cliff, reached via a different path.

---

#### Remediation

Five mechanisms, ordered from most to least impactful. Each must be specific enough to implement.

**1. Admission control as first defense**
The retry storm only occurs because requests were admitted, queued, and then timed out at the client. Your Phase A gateway enforcing the cliff threshold (KV utilization < cliff point, divergence ratio < 2×) would have returned 503 immediately — no queue buildup, no timeout, no retry trigger. This is not a retrofit — it's the primary mitigation. The other four are defense-in-depth.

**Counterfactual (fill in from your data):** At your measured token budget limit, max safe concurrency = budget_tokens / prompt_tokens. Your Day 26 cascade required concurrency N+K above that limit to generate the initial timeout wave. A gateway enforcing that ceiling would have rejected the K excess requests at admission — no timeout, no retry, no amplification. The cascade is not survivable by the serving layer alone; it requires the admission layer to hold the line. This is the direct link from Phase A design to Phase B failure analysis.

**2. 503 + Retry-After (server-side)**
When queue is full: return 503 immediately with `Retry-After: [N]` seconds. Do not hold the connection until the client timeout fires. Holding until timeout is what generates the retry wave — clients all time out at the same moment and retry synchronously. Immediate 503 breaks that synchronization. Effect: reduces amplification factor. Rough bound: at 5s timeout with immediate 503, you eliminate the 5-second window of in-flight requests that generate the synchronized retry burst.

**3. Exponential backoff with jitter (client-side)**
`min=0.5s, max=10s, jitter=±50%`. The jitter is load-bearing — without it, all clients back off for the same duration and retry simultaneously, reproducing the wave. With jitter, retries spread across the backoff window. This is the standard fix for thundering herd and it applies directly here.

**4. Retry budget (client + server)**
Hard cap: retries cannot exceed [N]% of total traffic (e.g., 10%). Enforced client-side via token bucket; server-side via a retry count header that clients decrement. If budget is exhausted, requests fail fast rather than retrying. This bounds the maximum amplification factor regardless of cascade depth.

**5. Circuit breaker**
If success rate < 50% over a 10-second window → open circuit → all requests fail fast → system gets recovery headroom. Circuit breakers are the recovery mechanism once cascade has already started. The other four mechanisms prevent reaching that state; the circuit breaker limits damage once there.

---

#### Narrative Section

*(Required for all Phase B postmortems — this is your behavioral interview answer.)*

Four elements. Write each as a paragraph, not bullets.

1. **What broke** — system-level description, no jargon. What did an outside observer see?
2. **What you expected** — your mental model before running the experiment. What did you think would happen when the burst ended?
3. **What surprised you** — where reality diverged from your model. The most important element. For this failure mode, the likely surprise is the self-sustaining nature: the cascade continued and worsened after the trigger was gone. If you expected the system to recover when the burst stopped, and it didn't, that's the learning.
4. **What you changed** — one concrete design decision you would make differently, backed by a number from your data.

**Quality gate:** "What surprised you" cannot be something derivable from reading a textbook. If you find yourself writing something generic, look harder at your timeline data.

---

## Afternoon (4 hrs) — Autoscaling Signal Analysis + Day 28 Prep

**The CPU HPA experiment from the original syllabus is cut.** The conclusion is predetermined by construction — CPU and KV cache utilization are decoupled because the GPU does the work and the CPU orchestrates. Observing CPU at 30% while KV is maxed calibrates nothing that first principles don't already tell you. The graph would confirm an expectation, not challenge one.

Replace it with:

---

### Block 1 (1.5 hrs) — Harden Postmortem #2

Postmortem #2 is your most conceptually complex deliverable so far. The morning session produces a complete draft; this block locks it.

Specific focus areas:

**Amplification graph:** Must be publication-quality. Both axes labeled with units. Burst-end marker and cliff-crossing marker visible and annotated. If the graph requires explanation to read, it is not done.

**Narrative "What surprised you":** This is the hardest sentence in the document and the most valuable in an interview. Draft it, then ask: could someone have predicted this from documentation? If yes, rewrite. The target is a statement that required running the experiment to know.

**Remediation ordering:** Confirm the five mechanisms are ordered correctly (admission control first — it's the primary, not an afterthought). The 503 + Retry-After mechanism should quantitatively tie to your timeline: "at [N]s client timeout, immediate 503 eliminates a [N]-second window of synchronized retries."

---

### Block 2 (1 hr) — Write the CPU Signal Argument Analytically

For the Autoscaling Memo (Deliverable #9), you need one crisp paragraph explaining why CPU is the wrong signal. Write it now from mechanical reasoning — no experiment needed:

Target structure:
> At KV utilization = [measured peak]%, CPU was processing ~[N] tokens/sec, which scales with throughput, not memory pressure. GPU memory saturation is invisible to CPU-based HPA because the bottleneck is KV block allocation, not compute throughput or kernel scheduling. A CPU threshold at 70% would not have triggered during your Day 27 cascade because CPU remained at [estimated range]% while the system was fully degraded. The correct signal is queue depth: it reflects pending work the GPU has not yet started, making it a leading indicator of SLO breach rather than a lagging confirmation of one.

Fill in your numbers. This paragraph, with those numbers, is the Autoscaling Memo Section 2 core argument. One page of mechanical reasoning replaces two hours of running a predetermined experiment.

---

### Block 3 (1.5 hrs) — Autoscaling Memo Skeleton + Day 28 Design

**Memo skeleton:** Write section headers and claim placeholders for Deliverable #9. For each claim, note which data it requires and whether you have it. Designing toward the deliverable now tells you exactly what Day 28's queue depth lag experiment must produce.

Skeleton structure:
```
Section 1: Why GPU inference autoscaling is different
  Claim: Model cold start = [X]s → reactive autoscaling is too slow
  Claim: KV cache is stateful mid-request → cannot terminate without drain
  Data needed: model load time (measure or estimate) ✓/✗

Section 2: Wrong signal — CPU utilization
  Claim: CPU stays at [X]% while KV is maxed
  Data: mechanical argument (written above) — no experiment needed ✓

Section 3: Right signals — comparison table
  Claim: Queue depth leads KV utilization by [N] seconds
  Data needed: Day 28 trigger lag measurement ✗ → must produce tomorrow

Section 4: Scale-out policy
  Claim: Composite signal — queue_depth > [Q] AND kv_util > [K] for > 30s
  Data needed: thresholds from Day 28 ✗

Section 5: Scale-down hazard + drain policy
  Claim: Naive termination drops in-flight requests
  Data: Day 28 graceful drain experiment ✗

Final Decision section (required format for Deliverable #9)
  Given / I would choose / Because / What gets worse / Not optimizing for / What would change my decision
  Data needed: all of the above ✗
```

**Day 28 design:** From the skeleton, your Day 28 must produce:
1. Queue depth trigger lag vs. KV utilization trigger lag (the lead time delta — quantified in seconds)
2. Graceful drain behavior under naive termination (what breaks, specifically)
3. Your composite signal thresholds

Write these three as explicit questions Day 28 must answer. This is the output of this block.

---

## End-of-Day Outputs

| Output | Status at EOD |
|---|---|
| **Deliverable #8: Postmortem #2 — Retry Cascade** | Locked |
| CPU signal argument paragraph (for Autoscaling Memo §2) | Written |
| Autoscaling Memo skeleton with data gaps identified | Written |
| Day 28 questions (3 explicit) | Written |

---

## Interview Signal This Day Builds

**Deliverable #8 answers:** "Why do retries make things worse?" / "How do you prevent cascading failure?"

**Staff-level signal demonstrated:** You understand retry cascade as a feedback loop problem, not a load problem. The fix is not "more capacity" — it's breaking the loop at the client (backoff/jitter), at the server (503 + Retry-After), and at the admission layer (reject before timeout). You have empirical data showing the amplification factor and the lag between burst-end and collapse. You can draw the dual-axis graph from memory and explain each annotation. That combination — mechanism + data + remediation ordered by impact — is what separates this answer from a candidate who read about retry storms.

---

## Version History

| Version | Changes |
|---|---|
| v1 | Initial. Afternoon CPU HPA experiment cut (predetermined outcome, zero calibration value). Replaced with postmortem hardening, analytical CPU signal argument, and Day 28 design. |
| v3 | Timeline section: added recovery lag annotation prompt with three-factor mechanical explanation (in-flight retries, poisoned queue, above-cliff state). Reviewer suggestion — Partial Accept (recovery mechanism non-obvious, passes filter). Remainder of reviewer feedback rejected: experiment critique targets Day 26 not Day 27, re-narration of existing content, synthetic interview story offer. |

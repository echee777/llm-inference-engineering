# Day 29 (Thu) — Autoscaling Strategy Memo

**Phase B Compressed Syllabus (v3) — Execution Guide**

**Version:** v3 (Reviewer Pass 2 applied — see correction tables below)

---

## Correction Table — v2 → v3 (Reviewer Pass 2)

| # | Reviewer Claim | Verdict | Reasoning |
|---|---|---|---|
| 1 | v2 is "A+ / near-optimal / elite signal / top-tier portfolio artifact" | Ignored | Grade framing. Not actionable. |
| 2 | Correction table filtering discipline is "staff-level thinking" | Ignored | Meta-commentary on prior filtering. Not a change request. |
| 3 | Section 2 correctly treated as core highest-signal section | Ignored | Affirmation of existing spec. Not a change. |
| 4 | Failure-mode bullets are "causal not descriptive reasoning" | Ignored | Affirmation of v1 → v2 partial accept. Not a change. |
| 5 | Decision Section is properly constrained | Ignored | Affirmation. Not a change. |
| 6 | Empirical grounding gate is critical | Ignored | Affirmation. Not a change. |
| 7 | Track 1 narrative block is "underrated but important" | Ignored | Affirmation. Not a change. |
| 8 | Execution-failure warnings: "writing instead of thinking" / "weak what-gets-worse" / "over-explaining basics" | Rejected | Already enforced: empirical-grounding gate kills generic writing; Decision Section format gate demands quantified tradeoffs; Section 1 signal-density gate kills HPA basics. Duplicate enforcement. |
| 9 | Add closing line to Decision Section: *"If this system were deployed at scale, the first thing I would instrument to validate this policy is: [X metric]"* | Rejected (Day 29 scope) | Two reasons. (a) **Format drift:** v4 Decision Section format is standardized across Deliverables #7, #9, #10, #11. A one-off addition to Day 29 breaks consistency. If adopted, it must be a Phase B–wide format extension (compressed syllabus v3 → v4), not a Day 29 local change. (b) **Partial subsumption:** the existing "What would make me change this decision" field demands production telemetry to detect flip conditions, so some validation instrumentation is already mechanically enforced. Note: "validation-that-confirms-success" vs "flip-trigger-that-invalidates" are genuinely distinct concerns, so the subsumption is only partial — the suggestion has merit as a systematic spec extension, but is rejected at Day 29 scope. Defer to Phase B spec-level discussion if wanted. |

**Net schedule/content impact:** None. All v2 body content unchanged.

---

## Correction Table — v1 → v2 (Reviewer Pass 1)

| # | Reviewer Claim | Verdict | Reasoning |
|---|---|---|---|
| 1 | Day 29 is "A-tier / critical / highest-signal day / staff transition" | Ignored | Grade-level framing. Not actionable. No syllabus change. |
| 2 | Memo must include hard numbers from experiments | Rejected | Already enforced by Section 2 empirical-grounding gate and end-of-day checklist item 2. Duplicate enforcement. |
| 3 | Memo must include explicit thresholds (not "high queue depth") | Rejected | Already enforced by Section 2 composite policy spec: `queue_depth > Q AND kv_util > K` with measured Q, K values. |
| 4 | Memo must include tradeoff with specific example (e.g., "operating at 65% vs. 80% → ~18% cost increase") | Rejected | Pre-fabricated example. The 18% figure is the v4 Decision Section spec's own illustrative example — re-injecting it here adds no structure. The format gate ("what gets worse" line must be quantified) already enforces the requirement; the actual number must come from my own data, not a reviewer hand-off. |
| 5 | Memo must include a firm decision, not "it depends" | Rejected | Already enforced by v4 Decision Section format: `I would choose: [explicit decision, no hedging]`. |
| 6 | Practice frontier-lab interview questions on Day 29 ("Why not scale on GPU util?", "What signal would you scale on?", "Why did your system collapse even with autoscaling?") | Rejected | Out of scope. Day 38 exit self-assessment already assigns Q2 (retry storm walkthrough) and Q4 ("Why is queue depth better than GPU utilization %?"). Duplicating on Day 29 creates drift and pre-empties Day 38's hard gate. |
| 7 | Add a "Failure if wrong signal is used" section to the memo | Partial accept | Useful negative-space framing. Implemented as a one-line *failure mode* addition to each wrong-signal bullet in Section 2 — not a standalone new section. Data already in hand (Day 27 CPU-HPA trace showing no trigger during overload; Day 28 GPU-util lag measured in seconds). Cost: ~2 sentences. |
| 8 | "Treat it like a real design doc at OpenAI / top-5% candidate framing" | Rejected | Already enforced by staff-level tone requirement in afternoon polish block and Decision Section format. Duplicate. |

**Net schedule impact:** None. Two sentences added to Section 2 wrong-signals bullets. All other suggestions rejected or ignored.

---

**Status:** Day 29 is **pure writing**. Every piece of data you need is already collected from Days 27–28 (CPU-HPA wrong-signal, GPU-util lag, queue-depth lead time, scale-down hazard). No experiments to run.

**Experiment-value-filter application:** Mostly dormant today — no experiments to tag as RUN / CONCEPTUAL. Forward application kicks in on Day 30's TP=2 environment validation. What it *does* enforce today: cut rote SRE-canon summarization, cite specific measured numbers, reject any section that isn't empirically grounded.

**Deliverable:** #9 Autoscaling Strategy Memo (due end of Day 29).

---

## Pre-Day Setup (10 min)

Before you open the memo file:

1. **Start `mistakes_log.md`** (per v3 running-practice header). Seed with the known candidates from the residency so far:
   - GQA `num_kv_heads` 8× error (Day 13)
   - vLLM V0/V1 terminology drift
   - T4 BF16 incompatibility
   - Cliff-as-divergence-ratio not raw-util (Day 16)
   - GPU-util as scaling signal (Day 28)

   You'll add entries Day 29 → Day 37 as they surface. Day 38 selects the Top 5 from this log for Deliverable #12.

2. **Gather Day 27–28 artifacts** in one place — you'll cite them continuously:
   - Day 27: CPU-HPA "wrong signal" trace (CPU 20–40% while GPU maxed)
   - Day 28: GPU-util HPA trigger lag (seconds of degradation before fire)
   - Day 28: queue-depth HPA trigger lag (measured lead time vs. GPU-util)
   - Day 28: signal comparison table
   - Day 28: scale-down hazard trace + graceful-drain design

3. **File naming convention:**
   - This execution log: `day29-syllabus-v1.md`
   - Deliverable #9 itself: `deliverable_09_autoscaling_memo_v1.md`

---

## Morning (4 hrs) — Write Deliverable #9

The memo has 5 content sections plus a mandatory Decision Section. Rough budget: ~40 min per content section + ~40 min Decision Section.

### Section 1: Why GPU Inference Autoscaling Is Different (~30 min)

Core contrast table — write this as a structured side-by-side, not prose:

| Dimension | CPU Web Service | GPU Inference |
|---|---|---|
| Cold start | seconds | 30–120s (model weight load) |
| State | stateless | stateful (KV cache mid-request) |
| Retry effect | amplification small, spread across fleet | amplification compounds on scarce KV |
| Correct autoscaling mode | reactive | proactive / predictive |

**Signal-density gate:** Keep this section short. It's orienting context, not the main work. One page max. If you find yourself explaining HPA basics, cut it — the reader is a staff-level interviewer, not onboarding.

---

### Section 2: Signal Selection (~45 min) — *highest-signal section*

This is where your Day 27–28 data lives. Structure:

**Wrong signals (with your trace):**
- CPU utilization: cite your measured CPU% during GPU overload (Day 27). The point is *the autoscaler literally cannot see the overload* — quantify it. **Failure mode:** no scale-out fires → full collapse under sustained load.
- GPU compute %: explain why memory-bound inference decouples compute-% from load pressure. **Failure mode:** lagging trigger — scale-out fires *after* TTFT p99 has already degraded (cite Day 28 measured lag in seconds).

**Right signals (with your Day 28 data):**
- Queue depth — leading indicator. Cite your measured lead time vs. GPU-util trigger.
- KV memory utilization — coincident. Ties to your Day 24 cliff (87% threshold, divergence ratio > 2×).
- TTFT p99 — lagging but direct SLO. Near-zero detection lag.

**Composite policy:**
```
scale_out iff queue_depth > Q AND kv_util > K for > 30s
```
with your measured Q, K values.

**Empirical-grounding gate:** Every number here must trace to a specific Day N experiment. If you catch yourself writing "typically around X" or "in the range of Y" — stop. Go find the measurement or remove the claim.

---

### Section 3: Scale-Out Policy (~30 min)

- **Trigger budget math:** If your measured model load time is `T_load` seconds, trigger must fire ≥ `T_load` seconds before predicted SLO breach. Write this as an inequality with your actual `T_load` value.
- **Predictive pre-warm:** Time-of-day pattern identification. Acknowledge this is *policy*, not experimentally validated on your rig — say so explicitly.
- **Increment policy:** +1, not +N. One-sentence justification (burst-recovery over-provision risk).

---

### Section 4: Scale-Down Policy (~30 min)

This section is carried by your Day 28 hazard experiment. Structure:

- **The hazard itself:** cite your Day 28 `kill -15` trace — what happened to the 20 in-flight requests, whether clients retried and re-entered the cascade pattern.
- **Graceful drain sequence** (from your Day 28 design):
  1. Mark instance draining → return 503 on new requests
  2. Wait for in-flight (bounded ≤ 5 min)
  3. Terminate
- **Drain-time estimate formula:** `T_drain ≈ T_p99 + padding`, with your measured T_p99.
- **Hysteresis:** scale-out threshold ≠ scale-down threshold. Specify both.

---

### Section 5: Predictive Scaling (~20 min)

Kept short. The interview-relevant insight is the **cost-vs-reliability tradeoff becomes a business input, not a technical one** — pre-warm cost vs. SLO miss cost. Don't over-engineer. One page.

---

### Decision Section (~40 min) — *mandatory, format-gated*

Use the exact v4 structure (no improvisation):

```
Given:
  - Qwen2.5-3B on T4/g4dn.xlarge, single-replica baseline
  - [your measured T_load seconds]
  - [queue-depth lead time from Day 28]
  - KV cliff point 87% from Day 24, divergence ratio > 2×
  - [your SLO target, e.g., TTFT p99 < X ms]

I would choose:
  Composite scale-out: queue_depth > Q AND kv_util > K for > 30s
  Graceful-drain scale-down with hysteresis
  Predictive pre-warm for known traffic patterns

Because:
  - [tradeoff 1 with data: queue depth leads GPU-util by N seconds (Day 28)]
  - [tradeoff 2: operating below cliff (~85% kv_util) avoids divergence-ratio blowup (Day 24)]
  - [tradeoff 3: graceful drain prevents re-entry into retry cascade (Day 28 + Day 26 data)]

What gets worse because of this decision:
  - [Quantified. e.g., "Pre-warm cost increases fleet capacity by ~X% over reactive baseline"]
  - [Queue-depth trigger adds Y seconds of additional latency headroom that is unused ~Z% of the time]

What I am explicitly NOT optimizing for:
  - Cost floor — accepting cost premium for SLO reliability
  - Burst-response latency minimization — not targeting sub-second scale-out

What would make me change this decision:
  - If T_load drops to <10s (e.g., via model preloading tricks), reactive scaling becomes viable
  - If workload shifts to steady-state with no diurnal pattern, predictive pre-warm loses value
```

**Format gate:** The "what gets worse" line must be **quantified**, not qualitative. This is the line interviewers push on. If you can't put a number on it, your data isn't rich enough — go back to your Day 27–28 traces.

---

## Afternoon (4 hrs) — Polish + Track 1 Narrative + Finalization

### Block 1: Polish Postmortem #2 + Autoscaling Memo (~1.5 hrs)

- Read both end-to-end. Cross-check numbers for consistency (especially anything citing the cliff — 87%, divergence ratio > 2×, TTFT regression `37.6 + 0.228 × prompt_tokens`).
- Graph check: axes labeled, units shown, legends present, titles informative.
- Writing voice check: declarative, not "I learned that…". Staff-level tone.

---

### Block 2: Phase B Track 1 Narrative (~1 hr) — *portfolio connective tissue*

Re-read all 5 Track 1 deliverables in order:

1. #5 Postmortem — KV Cache Exhaustion
2. #6 Prefill/Decode Interference Analysis
3. #7 Latency vs. Utilization Curve
4. #8 Postmortem — Retry Cascade
5. #9 Autoscaling Strategy Memo

Write **2 paragraphs**, format: `problem → discovery → production implication`.

The underlying thesis: *GPU inference systems are non-linearly fragile under load, and the mitigations are interdependent.* Your narrative should make the interdependence explicit. Connective claims to draw out:

- The cliff (#7) defines the admission control threshold (#9 signal selection).
- The retry cascade (#8) is why graceful drain matters (#9 scale-down).
- Prefill/decode interference (#6) is why queue depth alone is insufficient as a signal — you need KV-util coincident too.

This narrative becomes the Phase B Track 1 portfolio cover entry.

---

### Block 3: Final Track 1 Polish (~1.5 hrs)

Finalize all 5 Track 1 deliverables to portfolio quality. Cross-reference pass — make sure:

- #9 cites #7's cliff point in its KV-util threshold justification.
- #9 cites #8's retry amplification in its graceful-drain rationale.
- #7 has a forward reference stub noting it will be cited in #9 and in Phase C Cost vs. Reliability Memo.

---

## End-of-Day Gate

Before closing the day, confirm:

- [ ] Deliverable #9 complete with all 5 sections + Decision Section in the v4 format
- [ ] Every claim in the memo traces to a specific Day N experimental number (no "typically", no "generally")
- [ ] Decision Section "what gets worse" line is quantified
- [ ] Track 1 narrative written (2 paragraphs)
- [ ] All 5 Track 1 deliverables portfolio-final
- [ ] `mistakes_log.md` initialized, seeded with prior-residency candidates
- [ ] Files saved to `/mnt/user-data/outputs/` with versioned naming

If any of the first 4 isn't clean by end of day, **don't roll it into Day 30** — Day 30 is a gating checkpoint for Track 2 environment validation and cannot absorb slip. Use Day 30 afternoon flex time only if absolutely necessary; otherwise extend today.

---

## What's Next — Day 30 Preview

Day 30 is the Track 2 gate (NCCL/TP smoke test + reading). Prerequisites to handle by end of Day 29:

- **Provision A10G × 2 environment** (RunPod or Vast.ai). It needs to be smoke-tested by end of Day 30, or you slip half a day before Day 31.
- **Reading queue for Day 30 afternoon (1.5 hrs):** TP concept (weight-matrix split + all-reduce), PP concept (layer split + pipeline bubbles), NCCL primitives (all-reduce, all-gather, broadcast).

---

## References

- **Parent syllabus:** `phase_b_day29_38_compressed_v3.md` (supersedes `AI_Inference_Platform_Residency_PhaseB_v4.md` from Day 29 onward)
- **Decision Section spec:** `AI_Inference_Platform_Residency_PhaseB_v4.md` (Decision Section Requirement block)
- **Experiment value filter:** `experiment-value-filter.md` (applied forward from Day 30)
- **Input data sources:** Day 27 (CPU-HPA), Day 28 (GPU-util HPA, queue-depth HPA, scale-down hazard), Day 24 (cliff at 87%), Day 26 (retry amplification), Day 6 Exp 3e (TTFT regression)

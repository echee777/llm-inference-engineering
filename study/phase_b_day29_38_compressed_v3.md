# Phase B — Compressed Syllabus (Days 29–38)
## v3 — Reviewer Pass 2: Interpretation Discipline

**Status:** Supersedes `AI_Inference_Platform_Residency_PhaseB_v4.md` from Day 29 onward. Days 21–28 already executed under the v4 plan and are unchanged.

**New Phase B end point:** Day 38 (Wed of Week 8). Phase C begins Day 39.

**Days cut from v4:** 7 days (Day 35 buffer, Days 38–43 Week-8/Week-9 deepening and Phase C previews, Days 44–45 portfolio audit + exit assessment folded into new Day 38).

**Days remaining from Day 29:** 10.

**Running practice (new in v2):** Maintain a `mistakes_log.md` from Day 29 forward. Every time an experimental result contradicts your mental model, add an entry that day — prior mental model, data that contradicted it, concrete change. Day 38 finalizes to Top 5; the log ensures the sharpest entries aren't lost to memory decay.

---

## Correction Table — v4 → v1 Compressed

| # | Change | Rationale |
|---|---|---|
| 1 | Day 35 (Track 2 buffer) removed | Polish folds into Day 34 afternoon. No dedicated buffer day. Residual risk: no recovery slack if TP=2 data is noisy — accepted. |
| 2 | Day 36 (TP=2 cliff re-run) → new Day 35 | Only Week-8 content with unique interview signal (TP=1 vs. TP=2 failure-mode comparison). Kept. |
| 3 | Day 37 (TP=2 retry storm re-run) → new Day 36 | Same rationale as above. Kept. |
| 4 | Day 38 (straggler detection design + Amdahl TP scaling analysis) cut | Straggler content covered Day 33. Amdahl analysis is conceptual and can be folded into Day 36 afternoon write-up if desired. |
| 5 | Day 40 (Week 8 integration + DistServe pre-read) cut | Phase C content — DistServe reading belongs in Phase C Day 50 prep, not Phase B. |
| 6 | Days 41–42 (disaggregated prototype) cut | Exact duplicate of Phase C Deliverable #14 (Day 54). Running it twice adds no signal. |
| 7 | Day 43 (multi-tenant design sketch) cut | Exact duplicate of Phase C Deliverable #11 (Day 47). |
| 8 | Days 44–45 (portfolio audit + exit assessment) compressed into new Day 38 | Each was a full day of re-reading and reflection. The re-read is ~2 hrs; exit questions are ~2 hrs; portfolio cross-references are ~2 hrs. Fits in 8 hrs. |
| 9 | Deliverable #11 (End-to-End Platform Design) moved to new Day 37 | Was Day 35 in v4 (crowded into final buffer day). Given its importance as the Staff synthesis artifact, it gets a full dedicated day. |
| 10 | Deliverable #12 (Top 5 Mistakes) moved to new Day 38 morning | Written last, alongside exit self-assessment. |

---

## Correction Table — v1 → v2 (Reviewer Pass)

| # | Change | Rationale |
|---|---|---|
| 1 | Day 34 morning: Amdahl / TP scaling limits analysis promoted from "optional Section 7" to **mandatory 30–45 min block integrated into Deliverable #10 Section 6** | Reviewer accept. "At what TP degree does scaling break?" is a standard frontier-lab interview question. Data already in hand (Day 31 NCCL bandwidth, Day 32 comm fraction). Cost: 30–45 min. Optional framing was too soft — under pressure, "optional" means "cut". |
| 2 | Day 34 afternoon relabeled: "Polish + Consolidation" → **"Flex Buffer / Data Recovery / Polish"** with explicit instruction to use for Day 31–33 data re-runs if any result is noisy or suspect | Reviewer accept. Compression has zero buffer elsewhere; protecting Deliverable #10 input quality here avoids cascading to #11. Uses existing time, no schedule change. |
| 3 | Day 36 afternoon: added **30-min end-of-day task** "Assemble Deliverable #11 Number Sheet" with specific fields | Reviewer accept. Pre-filling the key numbers (KV capacity, TP=1 and TP=2 cliff points, retry amplification, comm fraction, TP decision) turns Day 37 into assembly rather than re-derivation. Mitigates Day 37 overflow risk. |
| 4 | Running `mistakes_log.md` practice added to header | Reviewer accept (practice, not schedule). Sharpest mistakes decay in memory by Day 38 if batched. Daily add-as-you-go entries are ~5 min. Day 38 still finalizes to the Top 5 with quality gates. |
| — | Reviewer's enthusiasm framing ("~90–95% signal preserved", "strong compression", "I would ship this") | Ignored. Not actionable. |

---

## Correction Table — v2 → v3 (Reviewer Pass 2)

| # | Change | Rationale |
|---|---|---|
| 1 | Day 32 afternoon: comparison table rows must name the limiting factor (compute / comm / memory-BW / KV) per configuration | Reviewer partial accept. Prevents the weak "comm fraction increases with concurrency" observation in favor of the stronger "iteration time bounded by X, dominated by Y" framing. One-line addition, no schedule change. |
| — | Reviewer Risk 2: "write a unified failure model sentence for Day 35–36 synthesis" | **Rejected.** Already enforced by existing Day 36 afternoon instruction: "Combine... into a single appendix" + staff-signal sentence template ("Under TP=2, the cliff shifted from X% to Y% because [mechanism], and retry amplification changed from Zx to Wx because [mechanism]."). Reviewer's proposed sentence is a wordier duplicate of what's already there. |
| — | Reviewer Risk 3: "Deliverable #11 must commit, not hedge" | **Rejected.** Already enforced by Decision Section format (inherited from v4 spec) which requires "explicit decision, no hedging" and a structured "I would choose: [...]" field. This is a format gate, not an aspiration. |
| — | Reviewer's quality-tier language ("interview-grade", "production-ready", "~95% correct"), offer to "simulate a real interview loop" | Ignored / deferred. Interview simulation is post-execution, not a v3 syllabus change. |

---

## Phase B Deliverables — Revised Due Dates

| # | Deliverable | v4 Due | v1 Due |
|---|---|---|---|
| 5 | Postmortem #1 — KV Cache Exhaustion | Day 22 | Day 22 ✓ already done |
| 6 | Prefill/Decode Interference Analysis | Day 23 | Day 23 ✓ already done |
| 7 | Latency vs. Utilization Curve | Day 24 | Day 24 ✓ already done |
| 8 | Postmortem #2 — Retry Cascade | Day 27 | Day 27 |
| 9 | Autoscaling Strategy Memo | Day 29 | Day 29 |
| 10 | Multi-GPU Serving Architecture Document | Day 34 | Day 34 |
| 11 | End-to-End Inference Platform Design | Day 35 | **Day 37** |
| 12 | Top 5 Mistakes I Made | Day 35 | **Day 38** |

---

## Signal Preserved / Signal Lost

**Preserved (unchanged):**
- All 8 deliverables.
- Full Track 2 multi-GPU week (NCCL profiling, TP=2 deployment, PP analysis, straggler experiment, Multi-GPU Architecture doc).
- TP=1 vs. TP=2 failure-mode comparison (cliff + retry storm on multi-GPU) — the only Week-8 content with unique interview signal.
- All Decision Sections on Deliverables #7, #9, #10, #11.
- Postmortem narrative sections on #5 and #8.

**Lost:**
- No recovery buffer if Track 2 provisioning is slow or TP=2 data is noisy. Mitigation: pre-validate environment on Day 30 afternoon (Appendix B of v4). If NCCL/TP smoke test doesn't pass by end of Day 30, add a half-day slip rather than starting Day 31 broken.
- Deeper-dive straggler detection design doc (v4 Day 38) — not a deliverable, won't show in portfolio. Concept still covered in Day 33 experiment and Day 34 write-up.
- Amdahl's Law analysis of TP scaling limits (v4 Day 38 afternoon) — **no longer lost in v2.** Promoted to mandatory 30–45 min block in Day 34 morning, integrated into Deliverable #10 Section 6.

---

# Day-by-Day Syllabus (Days 29–38)

---

## Day 29 (Thu) — Autoscaling Strategy Memo

**Unchanged from v4. Retained verbatim for continuity.**

### Morning (4 hrs) — Write Deliverable #9

Write **Autoscaling Strategy Memo**, five sections:

1. **Why GPU inference autoscaling is different**
   - CPU services: stateless, fast cold start, retries safe → reactive autoscaling works
   - GPU inference: 30–120s cold start, stateful KV mid-request, retries amplify → proactive/predictive required

2. **Signal selection (with your data from Day 28)**
   - Wrong: CPU utilization, GPU compute %
   - Right: queue depth (leading), KV memory utilization (coincident), TTFT p99 (lagging, direct SLO)
   - Composite: scale-out when queue_depth > Q AND kv_util > K for > 30s
   - Cite your measured threshold values

3. **Scale-out policy**
   - Trigger = "model_load_time before predicted SLO breach" (e.g., 90s before)
   - Predictive pre-warm on time-of-day patterns
   - +1 replica increments, not +N

4. **Scale-down policy (with Day 28 hazard analysis)**
   - Never scale down with in-flight requests
   - Graceful drain: mark → stop new traffic → wait ≤5 min → terminate
   - Hysteresis: scale-out threshold ≠ scale-down threshold

5. **Advanced — predictive scaling**
   - Traffic pattern identification
   - Pre-warm 5–10 min before predicted spikes
   - Cost vs. reliability tradeoff — business input required

**Must include Decision Section** per v4 format (Given / I would choose / Because / What gets worse / Not optimizing for / What would change the decision).

### Afternoon (4 hrs) — Track 1 Narrative + Polish

- Polish Postmortem #2 and Autoscaling Memo.
- **Phase B Track 1 narrative (1 hr):** Re-read all 5 Track 1 deliverables in sequence. Write 2 paragraphs summarizing what they collectively say about GPU inference fragility. Format: problem → discovery → production implication.
- Remaining time: finalize all Track 1 deliverables to portfolio quality.

**End-of-day output:** Deliverable #9 complete with Decision Section. Track 1 narrative written. All 5 Track 1 deliverables (#5–#9) finalized.

---

## Day 30 (Fri) — Track 1 Polish + Track 2 Environment Validation

**Unchanged from v4 in structure; this day is a gating checkpoint, not compressible.**

### Morning (4 hrs) — Final Track 1 Polish

- Finish any incomplete Track 1 deliverables.
- Re-run any noisy experiments.
- All graphs: publication-quality (axes, units, legends, titles).
- Cross-check numbers for consistency across Deliverables #5–#9.

### Afternoon (4 hrs) — Track 2 Environment Validation

**Do not skip. This is the gate.**

- **Read (1.5 hrs):**
  - TP concept: weight-matrix split per layer, all-reduce combine.
  - PP concept: layer split per GPU, sequential dataflow, bubbles.
  - NCCL primitives: all-reduce, all-gather, broadcast.

- **Provisioning + smoke test (2.5 hrs) — Hardware Provisioning Checklist from v4 Appendix B:**
  - [ ] 2× A10G on g5.12xlarge (or 2× A100) provisioned
  - [ ] `nvidia-smi topo -m` → record NVLink vs. PCIe
  - [ ] `nccl-tests` installed, `all_reduce_perf` smoke test passing
  - [ ] vLLM installed, TP=1 smoke test passing
  - [ ] TP=2 smoke test: single request returns tokens
  - [ ] TP=1 baseline recorded (TTFT, throughput) — comparison anchor for Day 31+

**Hard gate:** If TP=2 smoke test does not return tokens by end of day, add a half-day slip rather than starting Day 31 broken. The compression has no buffer — don't eat into Day 31 with environment debugging.

**End-of-day output:** Track 2 environment validated. TP=1 baseline numbers recorded. Cost envelope confirmed (~$3–5/hr spot, plan ~40 GPU-hrs across Days 31–36).

---

## Day 31 (Mon) — NCCL Microbenchmarks + TP Deployment

**Unchanged from v4.**

### Morning (4 hrs) — NCCL Communication Profiling

- **all-reduce sweep (2 hrs):**
  ```bash
  ./build/all_reduce_perf -b 1K -e 512M -f 2 -g 2 -n 50
  ```
  Record bandwidth (GB/s) and latency (μs) at: 1K, 8K, 64K, 512K, 4M, 32M, 256M, 512M.
  Compare to theoretical peak (NVLink 3.0 on A10G ~300 GB/s; PCIe Gen4 ~32 GB/s).

- **all-gather sweep (1 hr):** Same message sizes. all-reduce is the TP per-layer op — all-gather is for context.

- **Interpret (1 hr):** At what message size does bandwidth saturate? Below that size, latency dominates (small-message regime); above, bandwidth dominates.

### Afternoon (4 hrs) — TP=2 Deployment + Baseline

- Deploy vLLM with `--tensor-parallel-size 2` on Qwen2.5-3B-Instruct (or larger if memory allows with TP=2 headroom).
- Send test requests, verify output quality matches TP=1.
- **Single-request baseline:** TTFT and TPS at concurrency 1 under TP=2.
- Compare to TP=1 baseline from Day 30. Expected: TP=2 single-request TTFT slightly higher due to all-reduce per-layer cost; throughput similar or slightly lower at concurrency 1.

**End-of-day output:** NCCL bandwidth table. TP=2 deployment working. TP=1 vs. TP=2 single-request comparison recorded.

---

## Day 32 (Tue) — TP Performance Profiling Under Load

**Unchanged from v4.**

### Morning (4 hrs) — Communication Fraction Under Load

- **torch.profiler trace (2 hrs):** Run with profiling enabled, send 10 concurrent requests. Extract NCCL time vs. compute time per iteration.
- **Communication fraction = NCCL time / total iteration time.** Record at concurrency 1, 4, 8, 16.
- Hypothesis: comm fraction rises with concurrency as batches get larger but comm latency is fixed per iteration. Test it.

### Afternoon (4 hrs) — Load Sweep + TP vs. Single-GPU Comparison

- **Load sweep (2 hrs):** TP=2 at concurrency 1, 4, 8, 16, 24. Measure TTFT p50/p99 and TPS at each level.

- **Comparison table (2 hrs):**

  | Config | TTFT p50 | TTFT p99 | Throughput | Comm Fraction | Limiting Factor |
  |---|---|---|---|---|---|
  | 1-GPU quantized (from Phase A) | | | | N/A | |
  | TP=2 (your data) | | | | | |

  Key question: does TP=2 beat single-GPU-quantized at your hardware? At what concurrency?

  **Interpretation discipline:** The "Limiting Factor" column is not optional. For each row, name the dominant bottleneck (compute / comm / memory-BW / KV). Do not write "comm fraction increases with concurrency" — write "at concurrency 16, iteration time is bounded by [X], and scaling is dominated by [Y] because [mechanism]." This is the difference between reporting data and analyzing systems.

**End-of-day output:** Comm fraction table (concurrency 1→16). TP load sweep. TP=2 vs. single-GPU-quantized comparison with your data.

---

## Day 33 (Wed) — Pipeline Parallelism + Straggler Experiment

**Unchanged from v4.**

### Morning (4 hrs) — PP Analysis (or Theoretical Comparison)

- If vLLM PP available: deploy `--pipeline-parallel-size 2`, run single-request and load sweep, compare bubble time vs. TP comm overhead.
- If not available (likely): theoretical analysis using your TP numbers.
  - TP advantage: parallelism every layer, no bubbles.
  - TP disadvantage: all-reduce per layer — bandwidth-hungry.
  - PP advantage: communication only at layer boundaries — PCIe-friendly.
  - PP disadvantage: pipeline bubbles reduce utilization.
  - Crossover: PP wins when NVLink unavailable and PCIe bandwidth is the binding constraint.

### Afternoon (4 hrs) — Straggler Experiment

- **Clock-throttle one GPU:**
  ```bash
  nvidia-smi -i 1 -lgc 900   # Lock GPU 1 to 900 MHz
  ```
- Send 20 concurrent requests, measure TTFT and TPS.
- Sweep throttle: 90%, 80%, 70%, 60% of normal clock.
- Plot: straggler GPU clock speed vs. system-wide TTFT. Expect near-linear degradation (convoy effect).
- **Reset GPU clocks:**
  ```bash
  nvidia-smi -i 1 -rgc
  ```
- Write implications:
  - Never mix GPU generations within a TP group.
  - Per-GPU health monitoring required (not per-node).
  - Straggler detection is a production requirement, not optional.

**End-of-day output:** PP analysis (measured or theoretical). Straggler clock-vs-TTFT curve. Reset confirmed.

---

## Day 34 (Thu) — Multi-GPU Serving Architecture Document

**Unchanged from v4 morning. Afternoon adjusted: no longer starts Deliverable #11 — that gets its own Day 37.**

### Morning (4 hrs) — Write Deliverable #10

Write **Multi-GPU Serving Architecture Document**, six sections:

1. **NCCL Communication Profiles** — all-reduce/all-gather tables, bandwidth vs. theoretical, saturation message size.
2. **Tensor Parallelism — Performance Under Load** — load sweep, comm fraction curve, dominance threshold.
3. **TP=2 vs. Single-GPU Trade Study** — your comparison table, which wins for which use case, justification for TP vs. quantization choice.
4. **Pipeline Parallelism Analysis** — measured or theoretical, TP vs. PP decision matrix, NVLink vs. PCIe dimension.
5. **Straggler Impact** — clock-vs-TTFT curve, convoy effect mechanism, production implications (fleet homogeneity, health monitoring, canaries, circuit-breakers).
6. **Recommendations — including Max Efficient TP Degree (Amdahl analysis).** **Mandatory 30–45 min block.** Using your Day 31 NCCL all-reduce bandwidth and Day 32 comm fraction data:
   - Per-layer compute time estimate (FLOPs per layer / GPU TFLOPS).
   - Per-layer all-reduce time estimate (message size / NCCL bandwidth) at TP=2, TP=4, TP=8 hypotheticals.
   - Crossover point: at what TP degree does communication time exceed compute time?
   - State the answer as a single number for your hardware ("beyond TP=N, the system is communication-bound").
   - This is the answer to the interview question "at what TP degree does scaling break, and why?" Do not skip.
   - Follow with TP-vs-PP decision rule and straggler mitigations.

**Must include Decision Section.** Example: "Given Qwen2.5-3B workload on A10G×2, I would choose TP=2 over single-GPU INT8 because [data-backed tradeoffs]. What gets worse: cost-per-token ~X% higher. Not optimizing for: peak single-request TTFT. What would change this: workload shifts to latency-sensitive short-prompt-heavy traffic."

### Afternoon (4 hrs) — Flex Buffer / Data Recovery / Polish

**Order of operations — do in priority order:**

1. **First: Data quality check on Day 31–33 outputs.** Re-examine NCCL bandwidth table, TP load sweep, comm fraction trace, straggler clock-vs-TTFT plot. Any high variance across runs? Any numbers that contradict intuition without a mechanism? Any traces that didn't capture what they should have?

2. **If data is shaky: re-run.** This afternoon is the only recovery slot before Deliverable #10 locks. Use it. Better to re-run a noisy comm fraction measurement here than to build #10 and #11 on suspect numbers.

3. **If data is clean: polish Deliverable #10.** All 6 sections complete with experimental data. Cross-check numbers against Deliverables #5–#9 for consistency. Finalize Decision Section.

4. **Do not start Deliverable #11 here** — it gets Day 37. Use any leftover time for polish across all Track 1 + Track 2 artifacts, or banking polish time for later.

**End-of-day output:** Deliverable #10 complete with Decision Section and Amdahl-derived max efficient TP degree. Track 2 core experimentation done. Track 2 data confirmed clean (or re-run).

---

## Day 35 (Fri) — Utilization Cliff on TP=2

**Was v4 Day 36.**

### Morning (4 hrs) — TP=2 Cliff Experiment

- Repeat Day 24 methodology, now on TP=2 deployment.
- KV utilization sweep: 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%.
- At each level, measure: TTFT p50/p95/p99, preemption rate, queue depth, divergence ratio (p99/p50).
- Apply the same cliff criteria established Day 16/Day 24 (divergence ratio > 2× is the cliff, not raw utilization).

### Afternoon (4 hrs) — TP=1 vs. TP=2 Cliff Comparison

- Side-by-side plot: TP=1 cliff (from Day 24) vs. TP=2 cliff (from this morning).
- Answer:
  - Does the cliff point shift? Per-GPU KV budget differs under TP — does that change the cliff location?
  - Is the cliff steeper or shallower? Why?
  - Under TP=2, preemption requires all-reduce synchronization — does preemption cost change, and does that affect the cliff shape?
- Write 1-page comparison note. This is the interview-signal artifact: "Here's how TP changes the failure mode." File as Appendix to Deliverable #10.

**End-of-day output:** TP=2 cliff curve with 10 data points. TP=1 vs. TP=2 comparison note (~1 page).

---

## Day 36 (Mon) — Retry Storm on TP=2 + Synthesis

**Was v4 Day 37. Afternoon adjusted: consolidates both TP=2 re-run comparisons into a single multi-GPU-failure-modes appendix rather than being a standalone note.**

### Morning (4 hrs) — TP=2 Retry Storm

- Repeat Day 26 methodology on TP=2 serving.
- Same setup: 60% load, 2s timeout, 3 retries, burst trigger.
- Capture: amplification factor peak, recovery curve, KV cascade chain.

### Afternoon (4 hrs) — Multi-GPU Failure Modes Appendix + Deliverable #11 Number Sheet

**Main work (3.5 hrs):**

- Combine Day 35 (TP=2 cliff) + Day 36 morning (TP=2 retry storm) into a single appendix for Deliverable #10.
- Answer:
  - Retry amplification factor under TP=2 vs. TP=1 — different?
  - Cascade speed — faster/slower? (More memory headroom before cliff, but same communication patterns.)
  - Recovery time — faster/slower?
- Staff-signal sentence: "Under TP=2, the cliff shifted from X% to Y% because [mechanism], and retry amplification changed from Zx to Wx because [mechanism]."
- Add appendix to Deliverable #10. Update cross-references.

**End-of-day prep for Deliverable #11 (30 min):**

Create `deliverable_11_numbers.md`. Fill in every field from your prior experiments. This turns Day 37 into assembly, not re-derivation.

| Number | Your Value | Source |
|---|---|---|
| KV cache capacity (tokens, T4) | | Day 21 |
| KV cache capacity (tokens, A10G × 2) | | Day 31 |
| TP=1 cliff point (KV util %) | | Day 24 |
| TP=1 safe operating point | | Day 24 |
| TP=2 cliff point (KV util %) | | Day 35 |
| TP=2 safe operating point | | Day 35 |
| Retry storm peak amplification (TP=1) | | Day 26 |
| Retry storm peak amplification (TP=2) | | Day 36 |
| NCCL all-reduce bandwidth | | Day 31 |
| TP=2 comm fraction @ concurrency 16 | | Day 32 |
| Max efficient TP degree (Amdahl) | | Day 34 |
| Straggler: 70% clock → system TTFT change | | Day 33 |
| Queue depth vs. GPU util trigger lag | | Day 28 |
| TTFT regression (prefill scaling) | 37.6 + 0.228 × prompt_tokens | Day 6 Exp 3e |
| TP=2 vs. single-GPU INT8 decision | [which wins, for what workload] | Day 32 |
| Autoscaling signal choice | [primary + secondary + thresholds] | Day 29 |

**End-of-day output:** TP=2 retry storm data. Multi-GPU failure modes appendix integrated into Deliverable #10. Deliverable #11 Number Sheet complete — all 16 fields filled.

---

## Day 37 (Tue) — End-to-End Platform Design (Deliverable #11)

**Was v4 Day 35 (crowded into final buffer day). Now gets a full dedicated day.**

**Prerequisite from Day 36 afternoon:** `deliverable_11_numbers.md` must be complete before starting. Day 37 is assembly, not derivation — if you find yourself re-computing numbers here, stop and pull them from the Number Sheet.

### Morning (4 hrs) — Write Sections 1–4

Write **End-to-End Inference Platform Design**, all 7 sections (v4 spec). Every number must come from your Phase A+B experiments. No hypotheticals.

1. **Workload Model** — short/medium/long request mix (%), peak/sustained concurrency, TTFT p99 SLOs per bucket.
2. **Capacity Model** — KV cache budget math for Qwen2.5-3B on T4 (and TP=2 on A10G if applicable). Safe operating point = cliff − safety margin. Concurrent request ceiling at each SLO target.
3. **Admission Control Design** — token budget derivation from Phase A + Phase B work. Fail-fast vs. bounded-queue rejection. Why token budget > concurrency cap (cite Day 17 data).
4. **Serving Architecture Decision** — single-GPU-quantized vs. TP=2 vs. disaggregated. TP cost/benefit from Day 32 + Day 36 data. Decision for each of: latency-sensitive / throughput-optimized / cost-optimized.

### Afternoon (4 hrs) — Write Sections 5–7 + Decision Section

5. **Autoscaling Design** — signal selection with your Day 28 lag data. Pre-warming cost math. Scale-down drain policy.
6. **Failure Handling** — retry budget (cite Day 26 amplification data), circuit breaker spec (cite Day 36 TP=2 retry data if applicable), graceful drain, KV-cliff-approach response.
7. **Known Failure Modes and Mitigations** — KV exhaustion, retry cascade, prefill/decode interference, TP straggler. Each: symptoms, detection, mitigation. One paragraph each.

**Must end with full Decision Section.** "Given X workload and Y hardware, I would choose [architecture]. Because [3 tradeoffs with data]. What gets worse: [quantified]. Not optimizing for: [with accepted cost]. What would change this: [2 conditions]."

- Final polish: 4–6 pages, every claim cited to a prior deliverable.

**End-of-day output:** Deliverable #11 complete with full Decision Section.

---

## Day 38 (Wed) — Top 5 Mistakes + Portfolio Audit + Exit Assessment

**Final day of Phase B. Consolidates v4 Days 44 + 45 + Deliverable #12.**

### Morning (4 hrs) — Deliverable #12 + Portfolio Audit

- **Deliverable #12: Top 5 Mistakes I Made (1.5 hrs).** Five numbered entries, each:
  ```
  Mistake N: [one-sentence description]
  What I assumed: [prior mental model]
  What the data showed: [specific experimental result that contradicted it]
  What I changed: [concrete design or approach decision]
  ```
  Quality gates (all three required per entry):
  1. Not predictable from documentation — required running the experiment.
  2. Specific number/measurement, not vague observation.
  3. Resulted in a concrete design or mental-model change.

  Candidate mistakes from the residency so far: GQA num_kv_heads misread (Day 13, 8× error); vLLM V0/V1 terminology drift; T4 BF16 incompatibility; assuming cliff == raw utilization rather than divergence ratio (Day 16); assuming GPU utilization is a useful scaling signal (Day 28). **Plus all entries added to `mistakes_log.md` during Days 29–36.** Pick the five with strongest specific-data + concrete-change pairings. Day 38 work is selection and refinement, not recall from scratch.

- **Portfolio audit (2.5 hrs):** Read all 8 Phase B deliverables in sequence. For each:
  - Every claim has a number backing it up?
  - Graphs publication-quality?
  - Cross-references to earlier deliverables where appropriate?
  - Writing declarative and evidence-backed, not "I learned that..."?
  - Add forward-reference stubs where Phase C will cite (e.g., cliff graph → Cost vs. Reliability Memo; retry analysis → Safety Cascade Postmortem; multi-GPU doc → Disaggregated Serving Note).

### Afternoon (4 hrs) — Exit Self-Assessment + Phase B Narrative + Phase C Orientation

- **Exit self-assessment (2 hrs).** Answer in writing (1–2 paragraphs each):
  1. Why does p99 TTFT explode before average GPU utilization looks concerning? Use your cliff graph.
  2. Walk through a retry storm with your actual data. Quantify the amplification factor.
  3. Why do GPUs under TP behave like a convoy? What is the mechanical cause?
  4. Why is queue depth a better autoscaling signal than GPU utilization %?
  5. Explain prefill/decode interference. What mitigation did you test?
  6. TP vs. PP: when would you choose each? Defend with your numbers.
  7. What NCCL operation does TP use? What was your measured comm fraction?
  8. Straggler at TP=4: one GPU at 70% clock → what happens system-wide?
  9. Walk through your Deliverable #11 design. Where does it break first? What are you NOT optimizing for?

  **Hard gate:** If you cannot answer any of these confidently, do not proceed to Phase C. Close the gap first.

- **Phase B narrative (1 hr).** 3-paragraph executive summary of Phase B findings for portfolio cover entry. Format: problem → what you discovered → what it means for production design.

- **Phase C orientation (1 hr).** Review Phase C goals (Weeks 10–13): multi-tenant isolation, SLO design, model lifecycle, disaggregated serving, safety cascade, cost vs. reliability. Identify which Phase B artifacts Phase C builds on:
  - Cliff curve → SLO design + Cost vs. Reliability Memo
  - Retry storm → Safety cascade postmortem
  - Multi-GPU doc → Disaggregated serving architecture
  - Interference analysis → Multi-tenant isolation

**End-of-day output:** All 8 Phase B deliverables complete, portfolio-audited, cross-referenced. Exit assessment answered in writing. Phase B narrative written. **Phase B complete.** Phase C begins Day 39.

---

# Appendix: Phase B Key Numbers Reference (updated)

Fill in as experiments complete. Cited throughout Phase B and Phase C documents.

| Metric | Value | Day Measured |
|---|---|---|
| T4 KV cache capacity (tokens) | | Day 21 |
| Preemption onset (KV util %) | | Day 21 |
| TP=1 cliff point (KV util %) | | Day 24 |
| TP=1 cliff safe operating point | | Day 24 |
| Retry storm peak amplification factor (TP=1) | | Day 26 |
| CPU utilization at GPU saturation | | Day 27 |
| Queue depth trigger lag vs. GPU util trigger lag | | Day 28 |
| NCCL all-reduce bandwidth (your hardware) | | Day 31 |
| TP=2 communication fraction @ concurrency 16 | | Day 32 |
| Straggler impact: GPU at 70% clock → system TTFT change | | Day 33 |
| TP=2 cliff point | | Day 35 |
| TP=2 retry storm amplification factor | | Day 36 |

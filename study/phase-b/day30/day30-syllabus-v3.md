# Day 30 (Fri) — Track 1 Polish + Track 2 Environment Validation — v3

**Source:** Derived from `AI_Inference_Platform_Residency_PhaseB_v4.md` Day 30 and `phase_b_day29_38_compressed_v3.md` Day 30. Experiment Value Filter applied per-task.

**Status per v3 compressed:** *"Unchanged from v4 in structure; this day is a gating checkpoint, not compressible."* Afternoon gate carries the full weight of the compression — v3 has zero buffer elsewhere, so a broken environment here cascades into Day 31.

---

## Correction Table — v1 → v2 (Reviewer Pass)

| # | Reviewer Suggestion | Verdict | Rationale |
|---|---|---|---|
| 1 | Add post-smoke-test "What would break in production?" 5-bullet section, examples: NCCL mismatch, PCIe-vs-NVLink 10× slowdown, uneven GPU clocks → straggler, CUDA_VISIBLE_DEVICES mis-set, memory fragmentation | **Partial accept** | Accept the *structural concept* — a failure-mode crystallization step at the end of Block 2 is a cheap way to convert mechanical validation into interview-grade systems-thinking content. Reject the specific 5 pre-fabricated bullets: 2 duplicate content already in v1 (PCIe/NVLink flag in topology block + Pre-Start Flag; CUDA_VISIBLE_DEVICES in NCCL-failure common-issues list), 1 (uneven GPU clocks → straggler) is Day 33 scope. Accepted version: 3–5 bullets, written *after* validation based on what was actually observed, each specific to the topology and versions in use, with Day 31–33 scope explicitly excluded. |
| 2 | Add hypothesis log: predicted NCCL bandwidth from topology + hypothesis for Day 31 comm results | **Accept** | Fits the empirical-grounding principle (hypothesis-first measurement). Converts Day 31 from "run and report" to "predict, measure, explain the delta." Cost ~10 min. Added to Block 3 (Day 31 prep). |
| — | Framing: "Day 30 is gating + risk-elimination, not learning" | Rejected as additive | Already in v1 Framing section ("Day 30 is a hinge day, not an execution day"). Reviewer is restating. |
| — | Re-analysis through Experiment Value Filter arriving at RUN verdict | Rejected as additive | v1 applies the filter per-task and arrives at RUN for every task. Reviewer duplicates the analysis. |
| — | Commentary: "What this day is REALLY training" (infra instinct, environment correctness, baseline discipline, hidden bottlenecks) | Rejected as additive | Commentary on existing content, not proposals. The topology check, baseline capture, and gate discipline are already concrete in v1. |
| — | Enthusiasm framing ("high leverage", "correctly minimal", "non-compressible", one-line takeaway) | Ignored | Not actionable. |
| — | Offers to "list top 10 NCCL/TP failure modes" or "design a golden validation checklist" | Ignored | Not proposals. v1 already has a gated checklist. |

## Correction Table — v2 → v3 (Reviewer Pass 2)

| # | Reviewer Suggestion | Verdict | Rationale |
|---|---|---|---|
| 1 | Add 4-bullet sanity-check block after TP=2 smoke test to catch silent misconfigurations: both GPUs allocated, both GPUs utilized, latency not >2–3× TP=1, no NCCL warnings in logs | **Partial accept** | Accept the core point: "returns tokens" is necessary but not sufficient — a silently-misconfigured TP (wrong NCCL algorithm, PCIe fallback, one GPU carrying most load) can still return tokens. Reject the 4-bullet list as written: 2 of 4 bullets (memory allocation, GPU utilization) duplicate v2's TP=2 smoke test step; the "2–3× latency" threshold is too loose — for Qwen2.5-3B at concurrency 1, healthy TP=2/TP=1 TTFT ratio is expected at 1.1–1.4× on NVLink, so a 3× threshold would miss most silent misconfigurations. Accepted version: dedicated **Silent-Failure Gate** subsection after smoke tests, with (a) TP=2/TP=1 TTFT ratio at concurrency 1 recorded as a data point, flag-for-debug threshold set at >2× (not a hard gate — genuine slowdowns can be physical), (b) runtime NCCL warnings check (net-new; v2 only covered init errors), (c) existing memory+compute checks restated as explicit gate criteria so they can't be skipped. |
| — | Commentary on v2 training objectives, alignment with end goal, Experiment Value Filter re-verification, strengths/tweak summary | Rejected as additive | All commentary, no new content. v2 already does what this section describes. |
| — | Enthusiasm framing ("frontier-lab signal optimal", "extremely high signal", "staff-level trait", "interview gold") | Ignored | Not actionable. |
| — | Offers to "pre-review Day 31" and "show a perfect Day 30 output artifact" (pre-filled failure-mode catalog + hypothesis log) | Ignored | Not proposals. Pre-fabricated example artifacts are explicitly excluded per prior pattern — the artifacts are meant to be written *after* validation using actual observed data, not filled in from a template. |

---

## Framing

Day 30 is a hinge day, not an execution day. Two distinct objectives:

1. **Morning:** Close out Track 1. Deliverables #5–#9 need to be portfolio-shippable before you context-switch to multi-GPU. Anything unfinished here gets harder to return to after Track 2 starts.
2. **Afternoon:** Validate the Track 2 environment end-to-end. This is the **hard gate**. If TP=2 doesn't return tokens by EOD, slip half a day rather than starting Day 31 broken.

No experiments on Day 30 in the signal-producing sense — everything in the afternoon is infrastructure smoke-testing and baseline capture. The Experiment Value Filter still applies, but with a different verdict pattern (see section below).

---

## Morning (4 hrs) — Final Track 1 Polish

Work through Deliverables #5–#9 in order. For each, run the same pass:

1. **Graph audit (per deliverable):** axes labeled, units present, legend present, title present, cliff point / regime boundaries annotated where relevant. Anything that fails is not "90% done" — it's not shippable.
2. **Number consistency across deliverables:** the same quantity should have the same value everywhere it appears. Common drift points to check:
   - KV capacity number (Day 13 corrected value with `num_kv_heads=2`) — must match in #5, #7, #9
   - Cliff point (Day 24 result) — must match in #7 Decision Section and #9 signal selection section
   - Retry amplification factor (Day 26) — must match #8 narrative and #9 reference
   - `TTFT = 37.6 + 0.228 × prompt_tokens` (Day 6 Exp 3e) — referenced in #6 starvation-window derivation
3. **Re-run noisy experiments:** if any run showed >20% variance across repeats, re-run it now. Not worth carrying noise into the portfolio.
4. **Cross-references:** Deliverable #9 should cite the cliff curve from #7 explicitly. #8 should cite #7's safe operating point as the pre-storm baseline. These aren't decorative — they're the evidence chain interviewers will trace.
5. **Writing style sweep:** every claim backed by a number or citation. Strip any "I learned that…" / "it seems like…" constructions. Declarative, evidence-backed.

**Not in scope this morning:** new analysis, new sections, scope expansion. If a new idea surfaces, log it to `mistakes_log.md` or a "Phase C followups" stub and move on.

---

## Afternoon (4 hrs) — Track 2 Environment Validation

Per v3: **"Do not skip. This is the gate."**

### Block 1 (1.5 hrs) — Read: TP / PP / NCCL Concepts

v3 added this explicitly (wasn't in v4 afternoon). Purpose: you cannot interpret Day 31 NCCL numbers without the mental model in place. Target depth — enough to predict shape of results, not enough to implement.

- **Tensor parallelism:** weight matrices split column-wise (or row-wise) per layer; each GPU computes a partial; `all-reduce` combines. Per-layer `all-reduce` on the hidden-dim activation. This is the "tax" per transformer layer.
- **Pipeline parallelism:** layers partitioned across GPUs; microbatches flow through the pipeline; bubbles (idle time) at pipeline fill/drain. Sensitive to slowest stage.
- **NCCL primitives:** `all-reduce` (sum across ranks, result to all), `all-gather` (concatenate across ranks), `broadcast` (one-to-all). Ring vs. tree algorithms — NCCL picks based on message size and topology.

Don't read beyond the point where you can answer: *"Why does TP use all-reduce and not all-gather per layer?"* and *"Where would you expect bandwidth-bound vs. latency-bound regimes?"*

### Block 2 (2.5 hrs) — Provisioning + Smoke Tests

**Hardware note:** RunPod/Vast.ai 2× A10G is the cost-efficient path (~$0.60–1.00/hr/GPU spot) vs. AWS g5.12xlarge (~$3–5/hr spot total, but includes 4 GPUs you're not using). Either works. On RunPod, confirm the GPUs share a host (not network-attached) — otherwise NCCL bandwidth collapses.

Run this checklist in order. Do not skip a step because the previous one "looked fine."

- [ ] **2× A10G instance up.** `nvidia-smi` shows both GPUs, both at 0% util, full memory free.
- [ ] **Topology documented:**
  ```bash
  nvidia-smi topo -m
  ```
  Record: NVLink, PCIe, or something else? This determines your NCCL bandwidth ceiling (A10G NVLink 3.0 peak ≈ 300 GB/s; PCIe Gen4 x16 ≈ 32 GB/s — **an order of magnitude difference**). If it's PCIe, all your Day 31 numbers will interpret against PCIe peak, not NVLink peak. Flag this now.
- [ ] **`nccl-tests` built:**
  ```bash
  git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make
  ```
- [ ] **NCCL all-reduce smoke test:**
  ```bash
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
  ```
  You're not analyzing this output yet — that's Day 31. You just want confirmation that NCCL can communicate across both GPUs without errors. If this fails, stop and debug. Common failures: NCCL/CUDA version mismatch, `CUDA_VISIBLE_DEVICES` not set, container missing IPC/SHM.
- [ ] **vLLM TP=1 smoke test on the new instance:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096
  ```
  One request, verify tokens stream. Confirms the instance has a working vLLM install before you add the complication of TP.
- [ ] **vLLM TP=2 smoke test:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 4096
  ```
  One request. `nvidia-smi` should show **both GPUs** with allocated memory and activity. If only GPU 0 shows load, TP didn't actually distribute — check logs for NCCL init errors. Verify output quality against TP=1 (same prompt → substantively same content; not bit-identical due to nondeterministic reductions).

- [ ] **Silent-failure gate (5 min).** "Returns tokens" is necessary but not sufficient — a silently-misconfigured TP can still pass the smoke test. Run the following checks before accepting TP=2 as validated. A single-request TP=2 run is sufficient to collect all four.

  | Check | Pass criterion | Failure interpretation |
  |---|---|---|
  | GPU memory allocation | Both GPUs show non-trivial allocated memory (not just GPU 0) | TP didn't distribute the model — likely a vLLM config issue or single-GPU fallback |
  | GPU compute activity during request | `nvidia-smi dmon -c 20` during the request shows compute % >0 on both GPUs | Only one GPU actually computing — silent TP misconfig |
  | TP=2/TP=1 TTFT ratio at concurrency 1 | Record the ratio. Expected range on NVLink: 1.1–1.4×. Flag for debug if >2×. | >2× indicates PCIe topology, NCCL falling back to a slow algorithm, or cross-NUMA comm — either fix now or flag as a known limitation affecting all Day 31+ numbers |
  | NCCL warnings in logs | `grep -i 'nccl\|warn' vllm.log` returns no warnings during startup *or* request handling | Runtime warnings (e.g., "NCCL IB disabled", "falling back to ring algorithm", "P2P not available") can silently degrade bandwidth by 5–10×. Distinct from init errors — a job that starts clean can still emit warnings on first comm. |

  Soft gate (not hard): if the TP=2/TP=1 ratio is >2× but everything else passes, proceed but document in `mistakes_log.md` and factor into Day 31 NCCL-peak interpretation. If any of the hard checks (memory, compute, NCCL warnings) fails, debug before moving on.

- [ ] **TP=1 baseline captured on the new instance.** Not the T4 baseline — A10G is a different SKU with different FP16 throughput and memory bandwidth. You need *this instance's* TP=1 numbers as the comparison anchor for Day 31+:
  - TTFT at concurrency 1, using your standard prompt distribution
  - Throughput (tokens/sec) at concurrency 1
  - Record which A10G SKU (PCIe vs. NVLink variant), CUDA version, vLLM version, `--dtype` flag

**Model choice note:** On 2× A10G (48 GB aggregate HBM), you're no longer VRAM-bound at 3B. You could push to 7B or 13B for a more interesting TP=2 comparison. **Don't switch models today** — keep Qwen2.5-3B-Instruct as the anchor so TP=1 A10G ↔ T4 ↔ TP=2 comparisons are clean. Track 2 Week 7 can revisit model-size choice once the environment is validated.

### Block 2.5 (15 min) — Failure-Mode Catalog (post-gate)

Only run this *after* the gate has cleared. Purpose: crystallize the environment-fragility thinking into a portable artifact before moving on. Becomes interview-grade material for *"walk me through debugging a misbehaving multi-GPU inference job."*

Write 3–5 bullets, each in the format:

```
If <specific condition from my validation had been X instead>:
  Failure mode: <what breaks>
  Detection signal: <what you'd see in nvidia-smi / vllm logs / NCCL output>
  Mitigation: <concrete action>
```

**Hard constraints:**
- Each bullet must be specific to *your* observed topology, driver version, and config — not generic NCCL textbook failures.
- **Explicitly excluded from scope (reserved for later days):**
  - GPU clock variance / straggler effects → Day 33 experiment
  - NCCL bandwidth-vs-latency regime analysis → Day 31 analysis
  - Comm fraction modeling → Day 32 analysis
- If your validation was entirely clean and you have no real near-misses to draw on, write the bullets in "what would I have seen if X had failed" form — still based on the specific versions/topology you actually have.

This is ~15 min of work that produces a reusable artifact. If it takes longer, you're over-scoping it.

### Block 3 (0.5 hrs, minimum — can expand if time remains after gate clears) — Day 31 Prep + Hypothesis Log

- Plan Day 31's NCCL sweep as a single experiment run, not an interactive session. Multi-GPU instance cost is 4–6× a T4. Script the sweep, know every message size you'll test, know your logging format. You want to be measuring, not debugging, on Day 31 morning.
- Add one entry to `mistakes_log.md` if anything in the environment validation surprised you (driver versions, topology vs. what you expected, container IPC issue, etc.). Day 30 is exactly the kind of day where small surprises get lost by Day 38.
- **Hypothesis log (10 min):** Write down your predictions *before* Day 31 measurements, based on the topology you observed in Block 2. Deltas between hypothesis and measurement are Day 31's actual signal — matches confirm the mental model, gaps are the learning.

  ```
  Observed topology (from nvidia-smi topo -m): ___
  Predicted NCCL all-reduce peak bandwidth (GB/s): ___
    Derivation: ___ (e.g., NVLink 3.0 spec → ~300 GB/s; PCIe Gen4 x16 → ~32 GB/s)
  Predicted message size at which bandwidth saturates (bytes): ___
    Reasoning: ___ (small-message regime → latency-dominated; above saturation → bandwidth-dominated)
  Predicted Day 32 TP=2 comm fraction at concurrency 16 (rough range OK): ___
    Reasoning: ___
  Confidence (low/med/high) and why: ___
  ```

  On Day 31 end-of-day, compare measurements to these predictions. Any delta >20% is a candidate for the `mistakes_log.md` and potentially Deliverable #12.

---

## Experiment Value Filter — Day 30 Application

Day 30 is unusual in that most tasks are *infrastructure*, not experiments in the "produces an analytical number" sense, but the filter still tags each correctly:

| Task | Verdict | Reasoning |
|---|---|---|
| Graph/number consistency sweep (AM) | **RUN** | Criterion 4: produces portfolio-quality artifacts that later deliverables reference. Also criterion 2: the audit itself is the skill. |
| Re-run experiments with >20% variance | **RUN** | Criterion 4: downstream deliverables cite these numbers. Noisy data corrupts every downstream inference. |
| TP/PP/NCCL reading block | **RUN (conceptual)** | Gate for interpreting Day 31 data. Without it you can measure NCCL bandwidth but can't reason about where it falls on the bandwidth-vs-latency-bound regime split. |
| NCCL all-reduce smoke test | **RUN** | Criterion 2: skill-building (you'll run variants of this command all of Day 31). Criterion 4: gates Day 31 execution. |
| vLLM TP=2 smoke test | **RUN** | Criterion 4: gates everything. Non-negotiable. |
| TP=1 baseline on A10G | **RUN** | Criterion 4: anchor number for Days 31–36. Cannot be reconstructed later without re-provisioning. |
| Re-deriving NCCL bandwidth theoretical peak from NVLink spec | **CONCEPTUAL** | Deducible from published spec; 1 line of arithmetic; no new signal. Just look it up. |

No tasks recommended for cut. Day 30 is already a compressed version of v4's Day 30 (the reading block is the main v3 addition).

---

## Hard Gate (end of afternoon)

Per v3: *"If TP=2 smoke test does not return tokens by end of day, add a half-day slip rather than starting Day 31 broken. The compression has no buffer — don't eat into Day 31 with environment debugging."*

**Gate pass criteria (all six):**
1. Both GPUs visible and healthy
2. `nccl-tests` all-reduce runs without error
3. vLLM TP=1 smoke test passes on new instance
4. vLLM TP=2 smoke test passes and shows activity on both GPUs
5. Silent-failure gate: memory+compute on both GPUs, no NCCL warnings, TP=2/TP=1 ratio recorded (debugged if >2×)
6. TP=1 baseline TTFT/throughput numbers recorded

If any of these fails at 5pm: stop, take the half-day slip, don't push through. An extra 4 hrs on Saturday morning is cheaper than eating Day 31 morning (which is the NCCL sweep — the single highest-information experiment in Week 7).

---

## End-of-Day Output Checklist

- [ ] Deliverables #5–#9 portfolio-final (graphs audited, numbers cross-checked, writing swept)
- [ ] Track 2 hardware provisioned, topology documented
- [ ] NCCL smoke test passing
- [ ] vLLM TP=1 and TP=2 smoke tests passing
- [ ] Silent-failure gate cleared: both GPUs compute-active, no NCCL warnings, TP=2/TP=1 TTFT ratio recorded
- [ ] TP=1 baseline (TTFT, throughput) recorded for A10G instance, with SKU/driver/vLLM version noted
- [ ] Failure-mode catalog written (3–5 topology-specific bullets, Day 31–33 scope excluded)
- [ ] Hypothesis log filled: predicted NCCL bandwidth, saturation threshold, Day 32 comm fraction
- [ ] Day 31 NCCL sweep scripted and ready to run
- [ ] Any surprises logged to `mistakes_log.md`
- [ ] Cost envelope for Days 31–36 confirmed (~40 GPU-hrs planned)

---

## Pre-Start Flag

Before you start the afternoon: confirm whether your RunPod/Vast.ai 2× A10G option supports NVLink or is PCIe-only. Some providers list "2× A10G" where the cards are on separate PCIe roots, which wrecks NCCL performance and will make your Day 31 NCCL numbers look nothing like the published NVLink peak. If that's the case, you want to know *before* you've paid for provisioning.

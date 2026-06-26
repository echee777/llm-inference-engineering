# Phase C Planning Notes

**Status:** Holding document for Phase C modifications decided during Phase B v4 planning but deferred for execution. Not a syllabus. To be reconciled with `AI_Inference_Platform_Residency_Final.md` when Phase C execution begins (Day 41).

**Why this file exists:** Decisions made now have full rationale fresh. By Day 41, the Phase B v4 context will have decayed. This captures the *what* and *why* so future-me isn't re-deriving the choice.

---

## Modification 1 — Replace Phase C Days 41–42 PD-disagg prototype with vLLM v1 native PD-disagg deployment

### Context

Original master syllabus (`AI_Inference_Platform_Residency_Final.md`) has a "build a disaggregated serving prototype" task on Days 41–42 — hands-on construction of a toy disaggregated system using shared memory or similar, intended to teach the prefill/decode separation pattern. v3 Phase B compressed syllabus cut a Day 38 duplicate of this exercise; the Phase C version is still scheduled.

### Decision

Replace the build-a-prototype task with a **deploy-and-instrument vLLM v1 native PD-disagg** task. vLLM v1 ships a production PD-disaggregation path with a pluggable KV-transfer architecture (verify the supported backend set at execution time — was moving fast in early 2026; LMCache, NIXL, and Mooncake-derived implementations were candidates).

### Rationale

- **Portfolio impact:** Deploying and analyzing the production path produces an artifact that maps directly onto frontier-lab PD-disagg work. A toy prototype produces "I built something for myself" at best, and worse, it leaves me without hands-on experience of the actual serving-engine surface area I'd be expected to know in interviews.
- **Time/effort:** Production deployment + instrumentation is ~1.5–2 days. A from-scratch toy is also ~2 days. Net wash on time, large win on output quality.
- **Already covered conceptually:** Day 54 Phase C Deliverable #14 is the architectural *design doc* for disaggregated serving. Days 41–42 should produce *measured data* feeding into #14, not a redundant design exercise.
- **Reusable infra:** The KubeRay cluster from Phase B v4 Day 30 can serve as the deployment substrate. Two-pod deployment (prefill / decode) maps cleanly onto existing manifests.

### What to do at Phase C execution time

1. **Verify the current vLLM v1 PD-disagg API.** Backend set, config-flag names, and stability tier may have shifted from Phase B planning. Read `docs/source/serving/disagg_prefill.md` (or equivalent path) before committing to a backend.
2. **Pick the cleanest config-driven backend.** Do not fight build issues — if the chosen backend requires a custom Triton kernel build or a from-source NCCL patch, switch backends. Goal is measurement, not binary compatibility heroics.
3. **Two-instance deployment:**
   - 1× prefill instance, 1× decode instance.
   - Reuse Phase B Track 2 A10G provisioning if still available; otherwise re-provision 2× A10G.
   - Verify NCCL P2P or chosen KV transport works between the two pods at the K8s layer (cross-pod comm path may differ from within-pod TP).
4. **Instrumentation targets:**
   - KV transfer latency per request (p50, p99)
   - Throughput delta vs single-instance baseline (TP=2 from Phase B Day 33)
   - Prefill instance utilization vs decode instance utilization (the *whole point* of PD-disagg — measure that the bottleneck shifted as expected)
   - Failure modes: what happens when KV transfer stalls? When the prefill instance scales up but decode is saturated?
5. **Output:** Day 41–42 produces a `pd_disagg_measurement_notes.md` artifact feeding Day 54 Deliverable #14. Decision section: "Given my workload, would I deploy PD-disagg in production? Under what conditions does single-instance TP=2 win?"

### Open questions (resolve at execution time, not now)

- Which KV transfer backend is most stable at execution time?
- Does the KubeRay topology from Phase B Day 30 support cross-pod NCCL P2P, or does PD-disagg need a different K8s networking config (host networking, specific CNI features)?
- Does the chosen backend need shared filesystem / object storage for KV blocks, or is in-memory transfer sufficient at single-node-pair scale?
- Is there a vLLM v1 metric for KV transfer latency exposed natively, or does this need custom instrumentation?

---

## Modification 2 — Phase B → Phase C data handoff

The following Phase B v4 outputs will be cited or extended in Phase C. Do not re-derive these in Phase C — pull from the artifact named.

| Phase C work | Phase B v4 source |
|---|---|
| SLO design + Cost vs. Reliability Memo | Deliverable #7 cliff curve (Day 24) |
| Safety cascade postmortem | Deliverable #8 retry storm (Day 27) + Day 38 TP=2 retry data |
| Multi-tenant isolation design | Day 30 KubeRay addendum + Day 23 prefill/decode interference (Deliverable #6) |
| Disaggregated serving architecture (#14) | Deliverable #10 — esp. Section 6 (Amdahl, two-regime) and Section 7 (H100 contrast) |
| Cost vs. Reliability Memo | Day 35 FP8 throughput delta + H100 NCCL bandwidth |
| End-to-end Phase C synthesis | Deliverable #11 (Day 39) |
| Top 5 mistakes carry-forward | Deliverable #12 (Day 40) — extend with Phase C entries |

**Number Sheet:** Phase C deliverables that need Phase B numbers should pull from the Day 38 EOD `deliverable_11_numbers.md` (21 fields) — not from individual day notes. Single source of truth.

---

## Phase D handoff — MoE source path-tracing (deferred)

**Decision:** Deferred from Phase B v4 to Phase D. Captured here for continuity until `phase_d_notes.md` exists.

**What it is:** A ~4-hour source-reading exercise tracing how MoE / Expert Parallelism is implemented in vLLM and SGLang. No measurement, no benchmarks — purely understanding all-to-all comm pattern, expert routing, capacity-factor handling at the code level.

**Why deferred:** Lowest-leverage of the four candidate gap-closers from the frontier-lab interview-surface analysis. Source reading without measurement produces no portfolio artifact. Phase D ("model lifecycle, advanced architectures") is a more natural home if MoE topics surface there.

**Re-trigger condition:** If Phase D includes any MoE-adjacent design (mixture-of-experts inference cost model, expert-routing scheduling, heterogeneous expert placement), pull this in then. Otherwise, document as out-of-scope in the Day 63 portfolio summary's "what I didn't do and why" section.

**Move this section to `phase_d_notes.md`** once that file is created.

---

## Out-of-scope decisions (do NOT re-trigger in Phase C without explicit cause)

These were considered during Phase B v4 gap analysis and explicitly *not* added. Listed so future-me doesn't burn cycles reconsidering:

- **Multi-node NCCL / RDMA / IB / RoCE** — requires hardware I don't have provisioned. If Phase C work surfaces a multi-node design question, answer it from literature + Phase B Section 7 contrast data, not from new measurement. Do **not** attempt fake-multi-node experiments over TCP — low signal, high frustration, looks worse on portfolio than honestly scoping out.
- **Sequence Parallelism / Context Parallelism** — only relevant for >32K context workloads. Out of residency scope unless Phase D adds long-context work.
- **Heterogeneous inference engines (SGLang, TensorRT-LLM, lmdeploy)** — engine-level diversity is breadth without depth. Stick with vLLM v1 throughout. The exception is the vLLM v1 PD-disagg work above, which stays within vLLM.
- **Read-only concept summaries of frontier-lab topics** — violates the signal-density rule. If a topic is worth covering, it gets measured. If it's not measurable in the available environment, it's named as out-of-scope, not summarized.

---

## Phase C readiness gate (re-read at Day 40 EOD)

Before starting Day 41, confirm:

- [ ] Phase B v4 Deliverables #5–#12 all complete and portfolio-audited
- [ ] `deliverable_11_numbers.md` complete (21 fields filled)
- [ ] `mistakes_log.md` finalized to Top 5
- [ ] Day 40 exit self-assessment answered in writing
- [ ] This document re-read; Modification 1 backend choice researched
- [ ] Phase B → Phase C data handoff table reviewed; gaps escalated before Day 41 starts

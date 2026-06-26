# Phase B — Compressed Syllabus (Days 29–40)
## v4 — KubeRay Autoscaling + H100 Hardware Contrast Inserts

**Status:** Supersedes `phase_b_day29_38_compressed_v3.md` from Day 30 onward. Day 29 already executed under v3 plan; unchanged.

**Filename note:** Originally requested as `phase_b_day29_38_compressed_v4.md`. Renamed to `phase_b_day29_40_compressed_v4.md` to reflect actual scope — Phase B end shifts from Day 38 (v3) to Day 40 (v4) due to two full-day inserts (Day 30 KubeRay, Day 35 H100 contrast).

**New Phase B end point:** Day 40. Phase C begins Day 41 (was Day 39 in v3).

**Net schedule change vs v3:** +2 days.

**Days remaining from Day 29:** 12.

**Recommendations from prior gap analysis NOT included in v4:**
- **MoE source path-tracing** — deferred to Phase D. Lowest-leverage of the four candidate gap-closers (source reading, no measurement, no portfolio artifact). Document as out-of-scope in Day 63 portfolio summary.
- **vLLM v1 PD-disagg upgrade** — deferred to separate Phase C document modification. Phase C territory (modifies Days 41–42 of master schedule), out of v4 scope.

**Running practice (carried from v2):** Maintain `mistakes_log.md` from Day 29 forward. Every time an experimental result contradicts your mental model, add an entry that day. Day 40 finalizes to Top 5.

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

## Correction Table — v3 → v4 (Hardware Contrast + KubeRay Inserts)

| # | Change | Rationale |
|---|---|---|
| 1 | NEW Day 30 — KubeRay autoscaling re-execution as addendum to Deliverable #9 | Converts Deliverable #9 from a generic memo into a portfolio piece tied to your `infra_kuberay` repo. CPU-first plumbing strategy (fake-vLLM exposing the same metric name on a CPU node pool) de-risks the entire K8s/Prometheus/HPA pipeline before paying for GPU. ~1 day with Claude Code + your existing Terraform skill base. |
| 2 | v3 Day 30 (Track 1 polish + Track 2 env validation) → v4 Day 31 | Content preserved verbatim. Day shifted +1 to absorb new Day 30. |
| 3 | v4 Day 33 morning (was v3 Day 32) — torch.profiler block extended by 1 hour to capture NCCL_DEBUG=INFO subsystem traces and DCGM tensor-core utilization | Almost free — profiler is already wired up. Closes the "show me a flame graph" interview surface that Phase A's Nsight work didn't quite reach. Tensor-core utilization is the metric frontier-lab interviewers ask about by name. |
| 4 | NEW Day 35 — H100 hardware contrast (1 day; ~$60–100 spot on RunPod 2× H100 SXM) | Adds Section 7 to Deliverable #10. Measures FP8 throughput on Hopper tensor cores and NVLink-fast TP comm fraction, side-by-side with your A10G PCIe baseline. Closes hardware-class interview gap with measured data, not concept-only summaries. **Critical: use Llama 3.1 8B or larger model, NOT Qwen2.5-3B.** FP8 throughput delta won't show on a 3B model. |
| 5 | v4 Day 36 (was v3 Day 34) Deliverable #10 — now requires Section 7 "Hardware Contrast: A10G PCIe vs H100 NVLink" | Adds ~1 page to Deliverable #10. Two hardware regimes side-by-side using same NCCL bandwidth and comm-fraction methodology. Section 6 Amdahl analysis gains a second real data point, not just extrapolated. |
| 6 | v3 Days 35–38 → v4 Days 37–40 | Content preserved verbatim. All days shifted +2 to absorb Day 30 (KubeRay) and Day 35 (H100 contrast) inserts. |
| 7 | Internal cross-references throughout (Number Sheet, deliverable instructions, appendix) | Updated to v4 numbering. Day 21–28 references unchanged (locked, already executed). |
| — | MoE source path-tracing | **Deferred to Phase D.** Lowest-leverage of the four gap-closers (source reading, no measurement, no portfolio artifact). |
| — | vLLM v1 PD-disagg upgrade | **Deferred to separate Phase C document modification.** Out of v4 scope (modifies Days 41–42 of Phase C). |

---

## Phase B Deliverables — Revised Due Dates

| # | Deliverable | Original v4 | v3 Compressed | v4 (this doc) |
|---|---|---|---|---|
| 5 | Postmortem #1 — KV Cache Exhaustion | Day 22 | Day 22 ✓ | Day 22 ✓ |
| 6 | Prefill/Decode Interference Analysis | Day 23 | Day 23 ✓ | Day 23 ✓ |
| 7 | Latency vs. Utilization Curve | Day 24 | Day 24 ✓ | Day 24 ✓ |
| 8 | Postmortem #2 — Retry Cascade | Day 27 | Day 27 ✓ | Day 27 ✓ |
| 9 | Autoscaling Strategy Memo (+ KubeRay addendum) | Day 29 | Day 29 | Day 29 (memo) + **Day 30 (addendum)** |
| 10 | Multi-GPU Serving Architecture Document (+ Section 7 H100 contrast) | Day 34 | Day 34 | **Day 36** |
| 11 | End-to-End Inference Platform Design | Day 35 | Day 37 | **Day 39** |
| 12 | Top 5 Mistakes I Made | Day 35 | Day 38 | **Day 40** |

---

## Signal Preserved / Signal Lost (vs v3)

**Preserved (unchanged from v3):**
- All 8 deliverables, content unchanged.
- Full Track 2 multi-GPU week structure.
- TP=1 vs. TP=2 failure-mode comparison.
- All Decision Sections.
- v3's compression rationale (cuts vs original v4) all still valid.

**Added (new in v4):**
- Empirical KubeRay autoscaling data (Day 30) → Deliverable #9 addendum.
- Empirical H100 NVLink + FP8 contrast data (Day 35) → Deliverable #10 Section 7.
- Subsystem-level NCCL trace + DCGM tensor-core profiling (Day 33 morning).

**Lost (vs v3):**
- 2 calendar days of total schedule (29→40 vs 29→38).
- ~$60–100 of GPU spend for the H100 day (acceptable given portfolio impact).
- No recovery buffer if Track 2 provisioning is slow. Mitigation unchanged: pre-validate environment on Day 31 afternoon (was Day 30 in v3). If TP=2 smoke test doesn't pass by EOD, slip rather than starting Day 32 broken.

**Documented out-of-scope (per Day 63 portfolio summary):**
- Multi-node NCCL / RDMA / IB / RoCE.
- MoE / Expert Parallelism / all-to-all comm patterns.
- Sequence Parallelism / Context Parallelism.
- Heterogeneous inference engines (SGLang, TensorRT-LLM, lmdeploy).

---

# Day-by-Day Syllabus (Days 29–40)

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

## Day 30 (Fri) — KubeRay Autoscaling Re-Execution (NEW in v4)

**Adds an addendum to Deliverable #9 (Autoscaling Strategy Memo).** Converts the autoscaling deliverable from a standalone-vLLM memo into a real-cluster artifact tied to your `infra_kuberay` repo. Two-phase plan: de-risk the K8s/metrics pipeline cheaply on CPU before paying for GPU.

### Morning (4 hrs) — Phase 1: CPU Plumbing

**Goal:** Validate the entire HPA → Prometheus → adapter → HPA loop on a CPU node pool, with no GPU confounders.

- **Build fake-vLLM (1 hr) — Claude Code-generated:** FastAPI service exposing
  - `/metrics` endpoint with a Prometheus gauge named identically to real vLLM (`vllm_gpu_cache_usage_perc`), same units (percent 0–100), same label keys
  - `/v1/completions` endpoint that sleeps proportional to in-flight count and bumps the gauge accordingly
  - **Verify metric semantics first:** hit real vLLM's `/metrics` endpoint locally, mirror the exact metric name, type (gauge), units, and label keys. Mismatch here breaks the swap on Day 30 PM.

- **Terraform the cluster (2 hrs) — Claude Code generates 80%+, you tweak:**
  - KubeRay operator install (Helm)
  - RayService manifest pointing at fake-vLLM container
  - kube-prometheus-stack with ServiceMonitor for the workload
  - Prometheus Adapter rules mapping the metric to `external.metrics.k8s.io/v1beta1`
  - HPA manifests for three policies: CPU-based, "GPU util" (faked via the same gauge for symmetry), queue-depth (custom metric)
  - CPU node pool only — no GPU resource requests yet

- **Validate end-to-end (1 hr):** Send traffic via Locust at increasing rates. Verify gauge climbs, HPA reads it (no `<unknown>` values), replica count scales. **Hard gate:** all five links of the metric pipeline (workload exposes → ServiceMonitor scrapes → Prometheus stores → Adapter exposes → HPA references) must work cleanly before Phase 2. If anything is broken at lunch, debug here — not on a GPU meter.

### Afternoon (4 hrs) — Phase 2: GPU Swap + Three-Policy Experiments

**Goal:** Swap fake-vLLM for real vLLM on a GPU node pool. Re-run the Day 28 three-policy comparison on a real cluster.

- **GPU swap (1 hr):**
  - Provision GPU node pool (g5.xlarge or equivalent, ~$1–2/hr spot)
  - Update Helm values: workload image fake-vLLM → real vLLM with Qwen2.5-3B-Instruct, add GPU resource request, add node selector / tolerations
  - Verify metric pipeline still working (real vLLM exposes the same metric name fake-vLLM mocked)

- **Re-run three policies (2 hrs):**
  - CPU-based HPA: traffic ramp 30% → 90% over 10 min, observe CPU stays low, no scale event
  - GPU-util-based HPA: same ramp, observe lagging-indicator behavior
  - Queue-depth HPA (using `vllm_gpu_cache_usage_perc` or vLLM queue metric): same ramp, observe leading-indicator behavior
  - Capture per policy: trigger time, replica count over time, model load time per pod cold-start, request error rate during scale events

- **Cluster-level signals (NEW vs Day 28 standalone) (1 hr):**
  - Pod cold-start time vs in-process replica add (the K8s-specific timing finding)
  - HPA controller loop latency (stabilization window vs reaction time)
  - In-flight request handling on scale-down (graceful drain vs request loss)
  - Whether RayService head-vs-worker scaling semantics introduce surprises

### Day 30 Decision Section addition for Deliverable #9

Append to Deliverable #9: "On a real KubeRay cluster, the queue-depth signal triggered scale-up X seconds before GPU-util-based HPA. Pod cold-start added Y seconds of physical scale-up time after the controller decision. The control-loop reaction time and physical scale-up time are distinct components of total scaling latency; only the latter is hardware-bound. What gets worse on K8s vs standalone vLLM: [your data]. What would change this: [your data]."

**End-of-day output:** KubeRay cluster running, three-policy experiment data on real cluster, draft addendum to Deliverable #9.

**Cost envelope:** ~$0.10/hr CPU node pool (negligible), ~$1–2/hr × ~2 hrs GPU node pool ≈ $5 total GPU spend.

**Risk buffer:** +0.5 day if first-time wiring of Prometheus Adapter custom metric. The Ray side won't be your bottleneck; the metrics plumbing will. If Phase 1 doesn't complete by lunch, do not start Phase 2.

---

## Day 31 (Mon) — Track 1 Polish + Track 2 Environment Validation

**Was v3 Day 30. Content preserved verbatim. Day shifted +1 to absorb new Day 30.**

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
  - [ ] TP=1 baseline recorded (TTFT, throughput) — comparison anchor for Day 32+

**Hard gate:** If TP=2 smoke test does not return tokens by end of day, add a half-day slip rather than starting Day 32 broken. The compression has no buffer — don't eat into Day 32 with environment debugging.

**End-of-day output:** Track 2 environment validated. TP=1 baseline numbers recorded. Cost envelope confirmed (~$3–5/hr spot, plan ~40 GPU-hrs across Days 31–36).

---

## Day 32 (Tue) — NCCL Microbenchmarks + TP Deployment

**Was v3 Day 31. Content preserved verbatim. Day shifted +1.**

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
- Compare to TP=1 baseline from Day 31. Expected: TP=2 single-request TTFT slightly higher due to all-reduce per-layer cost; throughput similar or slightly lower at concurrency 1.

**End-of-day output:** NCCL bandwidth table. TP=2 deployment working. TP=1 vs. TP=2 single-request comparison recorded.

---

## Day 33 (Wed) — TP Performance Profiling Under Load + Subsystem Profiling

**Was v3 Day 32. Day shifted +1. Morning extended +1 hour for NCCL_DEBUG and DCGM tensor-core capture (NEW in v4).**

### Morning (4 hrs) — Communication Fraction Under Load + Subsystem Profiling

- **torch.profiler trace (1.5 hrs):** Run with profiling enabled, send 10 concurrent requests. Extract NCCL time vs. compute time per iteration.
- **Communication fraction = NCCL time / total iteration time.** Record at concurrency 1, 4, 8, 16.
- Hypothesis: comm fraction rises with concurrency as batches get larger but comm latency is fixed per iteration. Test it.

- **NCCL subsystem trace (30 min) — NEW in v4:** Re-run torch.profiler with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` env vars. Capture: which NCCL primitives are invoked per layer, transport selection (P2P / SHM / NET), and topology decisions logged at init. Save to `nccl_debug_<concurrency>.log`. Cross-reference with the torch.profiler trace.

- **DCGM tensor-core utilization (30 min) — NEW in v4:** Run DCGM during the same load. Extract `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` per request burst. Tensor-core util tells you whether the SMs are busy on the cores that matter for inference, vs busy on something else. Compare to compute-fraction from torch.profiler.

- **Interpret (1.5 hrs):** Comm fraction trend with concurrency + NCCL primitive identification + tensor-core utilization. Three lines on the same chart: comm fraction, compute fraction, tensor-core active fraction. State which is bottleneck at each concurrency level.

### Afternoon (4 hrs) — Load Sweep + TP vs. Single-GPU Comparison

- **Load sweep (2 hrs):** TP=2 at concurrency 1, 4, 8, 16, 24. Measure TTFT p50/p99 and TPS at each level.

- **Comparison table (2 hrs):**

  | Config | TTFT p50 | TTFT p99 | Throughput | Comm Fraction | Limiting Factor |
  |---|---|---|---|---|---|
  | 1-GPU quantized (from Phase A) | | | | N/A | |
  | TP=2 (your data) | | | | | |

  Key question: does TP=2 beat single-GPU-quantized at your hardware? At what concurrency?

  **Interpretation discipline:** The "Limiting Factor" column is not optional. For each row, name the dominant bottleneck (compute / comm / memory-BW / KV). Do not write "comm fraction increases with concurrency" — write "at concurrency 16, iteration time is bounded by [X], and scaling is dominated by [Y] because [mechanism]." This is the difference between reporting data and analyzing systems.

**End-of-day output:** Comm fraction table (concurrency 1→16). NCCL_DEBUG subsystem trace. DCGM tensor-core utilization curve. TP load sweep. TP=2 vs. single-GPU-quantized comparison with your data.

---

## Day 34 (Thu) — Pipeline Parallelism + Straggler Experiment

**Was v3 Day 33. Content preserved verbatim. Day shifted +1.**

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

## Day 35 (Fri) — Hardware Contrast: A10G PCIe vs H100 NVLink (NEW in v4)

**Adds Section 7 to Deliverable #10.** One-day H100 contrast experiment producing measured FP8 throughput and NVLink-fast TP comm fraction data, side-by-side with your A10G PCIe baseline.

### Pre-flight Checklist (do BEFORE spinning up H100)

Cost discipline: H100 is $3–5/hr per GPU spot, $60–100/day for 2× H100 SXM. Every minute the meter runs on prep work is wasted budget. Have all of these ready before pressing "deploy":

- [ ] Terraform / pod spec for 2× H100 SXM RunPod (or equivalent)
- [ ] Model identified — **Llama 3.1 8B or larger, NOT Qwen2.5-3B.** FP8 delta will not show on a 3B model.
- [ ] Model weights URL ready; HF token if gated
- [ ] vLLM args ready: `--tensor-parallel-size 2 --dtype fp8_e4m3` (Hopper FP8) and BF16 baseline command
- [ ] NCCL test scripts (same as Day 32, parameter-identical)
- [ ] Locust traffic config (same as Day 33, parameter-identical)
- [ ] Comparison memo template (Section 7 outline)

### Morning (4 hrs) — H100 Spin-up + NCCL + Comm Fraction

- **Provision + smoke (30 min):** 2× H100 SXM, verify NVLink topology with `nvidia-smi topo -m`. Expect NV12 / NV18 (NVLink fast path), not PHB.
- **NCCL bandwidth sweep (1 hr):** Same `all_reduce_perf` sweep as Day 32 (1K → 512M message sizes). Record bandwidth at each. Expect 5–10× higher bandwidth than A10G PCIe at message sizes >1MB.
- **vLLM TP=2 deploy + comm fraction (2 hrs):**
  - Deploy Llama 3.1 8B with `--tensor-parallel-size 2 --dtype bfloat16` (BF16 baseline)
  - Run torch.profiler trace at concurrency 1 and concurrency 16
  - Extract comm fraction at both points
  - Compare to A10G PCIe comm fraction (Day 33 data)
- **Capture (30 min):** Tabulate side-by-side: NCCL all-reduce bandwidth at 1MB and 32MB, comm fraction at concurrency 1 and 16, single-request TTFT at TP=2. Two rows: A10G PCIe / H100 NVLink.

### Afternoon (4 hrs) — FP8 Throughput Delta + Section 7 Memo

- **FP8 inference (2 hrs):**
  - Same Llama 3.1 8B, same TP=2, switch `--dtype fp8_e4m3`
  - Run throughput sweep: concurrency 1, 8, 16, 32. Measure tokens/sec.
  - Compare to BF16 numbers from morning. Expected: ~1.5–2× throughput gain on FP8 at higher concurrency. Report your actual measured delta.
  - Note any output quality degradation observed (FP8 calibration matters; flag if completions degrade visibly).
- **Spin down (15 min):** Tear down H100 instance. Stop the meter.
- **Write Section 7 (1.75 hrs):** "Hardware Contrast: A10G PCIe vs H100 NVLink"
  - **7.1:** NCCL bandwidth side-by-side (table + delta)
  - **7.2:** TP comm fraction at concurrency 1 and 16 (table + delta)
  - **7.3:** FP8 vs BF16 throughput on Hopper Llama 3.1 8B (your measured delta)
  - **7.4:** Implication for the Section 6 Amdahl analysis — recompute "max efficient TP degree" with H100 NVLink bandwidth as a second data point. Does the TP=2 vs TP=4 vs TP=8 crossover shift between hardware regimes?
  - **7.5:** Honest scope note. NOT closed by this experiment: multi-node, MoE/all-to-all, MIG, full-mesh NVSwitch (you only have 2× H100 SXM, not 8× with full crossbar).

**End-of-day output:** Section 7 written with measured data. Ready to integrate into Deliverable #10 on Day 36.

**Cost envelope:** $60–100 actual spend if everything goes right. Add $30–60 buffer for a re-run. **Hard cap: spin down by EOD regardless of progress.** Do not let an H100 leak overnight.

---

## Day 36 (Mon) — Multi-GPU Serving Architecture Document

**Was v3 Day 34. Day shifted +2. Section list extended to seven sections (added Section 7 — Hardware Contrast from Day 35).**

### Morning (4 hrs) — Write Deliverable #10

Write **Multi-GPU Serving Architecture Document**, **seven** sections:

1. **NCCL Communication Profiles** — all-reduce/all-gather tables, bandwidth vs. theoretical, saturation message size. (Day 32 data.)
2. **Tensor Parallelism — Performance Under Load** — load sweep, comm fraction curve, dominance threshold. (Day 33 data, including new NCCL subsystem and DCGM tensor-core traces.)
3. **TP=2 vs. Single-GPU Trade Study** — your comparison table, which wins for which use case, justification for TP vs. quantization choice.
4. **Pipeline Parallelism Analysis** — measured or theoretical, TP vs. PP decision matrix, NVLink vs. PCIe dimension.
5. **Straggler Impact** — clock-vs-TTFT curve, convoy effect mechanism, production implications (fleet homogeneity, health monitoring, canaries, circuit-breakers). (Day 34 data.)
6. **Recommendations — including Max Efficient TP Degree (Amdahl analysis).** **Mandatory 30–45 min block.** Using your Day 32 NCCL all-reduce bandwidth, Day 33 comm fraction data, **and Day 35 H100 contrast data**:
   - Per-layer compute time estimate (FLOPs per layer / GPU TFLOPS).
   - Per-layer all-reduce time estimate (message size / NCCL bandwidth) at TP=2, TP=4, TP=8 hypotheticals — now using TWO bandwidth data points (A10G PCIe AND H100 NVLink) to bound the crossover.
   - Crossover point: at what TP degree does communication time exceed compute time? State the answer for each hardware regime.
   - This is the answer to the interview question "at what TP degree does scaling break, and why does the answer depend on interconnect?" Do not skip.
   - Follow with TP-vs-PP decision rule and straggler mitigations.
7. **Hardware Contrast: A10G PCIe vs H100 NVLink** (NEW in v4, ~1 page) — uses Day 35 measured data:
   - NCCL bandwidth side-by-side
   - TP comm fraction at concurrency 1 and 16
   - FP8 vs BF16 throughput delta on Hopper Llama 3.1 8B
   - Implication for Section 6 Amdahl crossover (refer back)
   - Honest scope note: multi-node, MoE/all-to-all, MIG, full-mesh NVSwitch are NOT in scope.

**Must include Decision Section.** Example: "Given Qwen2.5-3B workload on A10G×2, I would choose TP=2 over single-GPU INT8 because [data-backed tradeoffs]. What gets worse: cost-per-token ~X% higher. Not optimizing for: peak single-request TTFT. What would change this: workload shifts to latency-sensitive short-prompt-heavy traffic OR migration to H100 NVLink hardware (cite Section 7 numbers)."

### Afternoon (4 hrs) — Flex Buffer / Data Recovery / Polish

**Order of operations — do in priority order:**

1. **First: Data quality check on Day 32–35 outputs.** Re-examine NCCL bandwidth table (A10G PCIe), TP load sweep, comm fraction trace, straggler clock-vs-TTFT plot, H100 contrast numbers. Any high variance across runs? Any numbers that contradict intuition without a mechanism?

2. **If data is shaky: re-run.** This afternoon is the only recovery slot before Deliverable #10 locks. Use it. Better to re-run a noisy comm fraction measurement here than to build #10 and #11 on suspect numbers. (Note: re-running H100 numbers means another spin-up; cost-budget accordingly.)

3. **If data is clean: polish Deliverable #10.** All 7 sections complete with experimental data. Cross-check numbers against Deliverables #5–#9 for consistency. Finalize Decision Section.

4. **Do not start Deliverable #11 here** — it gets Day 39. Use any leftover time for polish across all Track 1 + Track 2 artifacts.

**End-of-day output:** Deliverable #10 complete with all 7 sections, Decision Section, and Amdahl-derived max efficient TP degree (now bounded by two hardware regimes).

---

## Day 37 (Tue) — Utilization Cliff on TP=2

**Was v3 Day 35. Day shifted +2.**

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

## Day 38 (Wed) — Retry Storm on TP=2 + Synthesis

**Was v3 Day 36. Day shifted +2. Afternoon consolidates both TP=2 re-run comparisons into a single multi-GPU-failure-modes appendix.**

### Morning (4 hrs) — TP=2 Retry Storm

- Repeat Day 26 methodology on TP=2 serving.
- Same setup: 60% load, 2s timeout, 3 retries, burst trigger.
- Capture: amplification factor peak, recovery curve, KV cascade chain.

### Afternoon (4 hrs) — Multi-GPU Failure Modes Appendix + Deliverable #11 Number Sheet

**Main work (3.5 hrs):**

- Combine Day 37 (TP=2 cliff) + Day 38 morning (TP=2 retry storm) into a single appendix for Deliverable #10.
- Answer:
  - Retry amplification factor under TP=2 vs. TP=1 — different?
  - Cascade speed — faster/slower? (More memory headroom before cliff, but same communication patterns.)
  - Recovery time — faster/slower?
- Staff-signal sentence: "Under TP=2, the cliff shifted from X% to Y% because [mechanism], and retry amplification changed from Zx to Wx because [mechanism]."
- Add appendix to Deliverable #10. Update cross-references.

**End-of-day prep for Deliverable #11 (30 min):**

Create `deliverable_11_numbers.md`. Fill in every field from your prior experiments. This turns Day 39 into assembly, not re-derivation.

| Number | Your Value | Source |
|---|---|---|
| KV cache capacity (tokens, T4) | | Day 21 |
| KV cache capacity (tokens, A10G × 2) | | Day 32 |
| TP=1 cliff point (KV util %) | | Day 24 |
| TP=1 safe operating point | | Day 24 |
| TP=2 cliff point (KV util %) | | Day 37 |
| TP=2 safe operating point | | Day 37 |
| Retry storm peak amplification (TP=1) | | Day 26 |
| Retry storm peak amplification (TP=2) | | Day 38 |
| NCCL all-reduce bandwidth (A10G PCIe) | | Day 32 |
| NCCL all-reduce bandwidth (H100 NVLink) | | Day 35 |
| TP=2 comm fraction @ concurrency 16 (A10G) | | Day 33 |
| TP=2 comm fraction @ concurrency 16 (H100) | | Day 35 |
| Tensor-core utilization @ concurrency 16 | | Day 33 |
| FP8 vs BF16 throughput delta (Llama 3.1 8B) | | Day 35 |
| Max efficient TP degree (Amdahl, two regimes) | | Day 36 |
| Straggler: 70% clock → system TTFT change | | Day 34 |
| Queue depth vs. GPU util trigger lag | | Day 28 |
| KubeRay pod cold-start vs HPA reaction time | | Day 30 |
| TTFT regression (prefill scaling) | 37.6 + 0.228 × prompt_tokens | Day 6 Exp 3e |
| TP=2 vs. single-GPU INT8 decision | [which wins, for what workload] | Day 33 |
| Autoscaling signal choice | [primary + secondary + thresholds] | Day 29 |

**End-of-day output:** TP=2 retry storm data. Multi-GPU failure modes appendix integrated into Deliverable #10. Deliverable #11 Number Sheet complete — all 21 fields filled.

---

## Day 39 (Thu) — End-to-End Platform Design (Deliverable #11)

**Was v3 Day 37. Day shifted +2. Was original v4 Day 35 (crowded into final buffer day in original v4); now gets a full dedicated day.**

**Prerequisite from Day 38 afternoon:** `deliverable_11_numbers.md` must be complete before starting. Day 39 is assembly, not derivation — if you find yourself re-computing numbers here, stop and pull them from the Number Sheet.

### Morning (4 hrs) — Write Sections 1–4

Write **End-to-End Inference Platform Design**, all 7 sections (v4 spec). Every number must come from your Phase A+B experiments. No hypotheticals.

1. **Workload Model** — short/medium/long request mix (%), peak/sustained concurrency, TTFT p99 SLOs per bucket.
2. **Capacity Model** — KV cache budget math for Qwen2.5-3B on T4 (and TP=2 on A10G if applicable). Safe operating point = cliff − safety margin. Concurrent request ceiling at each SLO target.
3. **Admission Control Design** — token budget derivation from Phase A + Phase B work. Fail-fast vs. bounded-queue rejection. Why token budget > concurrency cap (cite Day 17 data).
4. **Serving Architecture Decision** — single-GPU-quantized vs. TP=2 vs. disaggregated. TP cost/benefit from Day 33 + Day 38 data, with hardware-regime contrast from Day 35. Decision for each of: latency-sensitive / throughput-optimized / cost-optimized.

### Afternoon (4 hrs) — Write Sections 5–7 + Decision Section

5. **Autoscaling Design** — signal selection with your Day 28 lag data. Pre-warming cost math. Scale-down drain policy.
6. **Failure Handling** — retry budget (cite Day 26 amplification data), circuit breaker spec (cite Day 38 TP=2 retry data if applicable), graceful drain, KV-cliff-approach response.
7. **Known Failure Modes and Mitigations** — KV exhaustion, retry cascade, prefill/decode interference, TP straggler. Each: symptoms, detection, mitigation. One paragraph each.

**Must end with full Decision Section.** "Given X workload and Y hardware, I would choose [architecture]. Because [3 tradeoffs with data]. What gets worse: [quantified]. Not optimizing for: [with accepted cost]. What would change this: [2 conditions]."

- Final polish: 4–6 pages, every claim cited to a prior deliverable.

**End-of-day output:** Deliverable #11 complete with full Decision Section.

---

## Day 40 (Fri) — Top 5 Mistakes + Portfolio Audit + Exit Assessment

**Was v3 Day 38. Day shifted +2. Final day of Phase B.**

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

  Candidate mistakes from the residency so far: GQA num_kv_heads misread (Day 13, 8× error); vLLM V0/V1 terminology drift; T4 BF16 incompatibility; assuming cliff == raw utilization rather than divergence ratio (Day 16); assuming GPU utilization is a useful scaling signal (Day 28). **Plus all entries added to `mistakes_log.md` during Days 29–38.** Pick the five with strongest specific-data + concrete-change pairings. Day 40 work is selection and refinement, not recall from scratch.

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

**End-of-day output:** All 8 Phase B deliverables complete, portfolio-audited, cross-referenced. Exit assessment answered in writing. Phase B narrative written. **Phase B complete.** Phase C begins Day 41.

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
| KubeRay pod cold-start time | | Day 30 |
| KubeRay HPA controller loop latency | | Day 30 |
| NCCL all-reduce bandwidth (A10G PCIe) | | Day 32 |
| TP=2 communication fraction @ concurrency 16 (A10G) | | Day 33 |
| Tensor-core utilization @ concurrency 16 | | Day 33 |
| Straggler impact: GPU at 70% clock → system TTFT change | | Day 34 |
| NCCL all-reduce bandwidth (H100 NVLink) | | Day 35 |
| TP=2 communication fraction @ concurrency 16 (H100) | | Day 35 |
| FP8 vs BF16 throughput delta (Llama 3.1 8B) | | Day 35 |
| Max efficient TP degree — A10G regime | | Day 36 |
| Max efficient TP degree — H100 regime | | Day 36 |
| TP=2 cliff point | | Day 37 |
| TP=2 retry storm amplification factor | | Day 38 |

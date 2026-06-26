# Day 28 (Wed) — Autoscaling: Right Signals + Scale-Down Hazard

**Version:** v4 (reviewer feedback round 3 applied — final polish)
**Phase B, Week 6, Day 3 of 5** — Track 1 (T4 / g4dn.xlarge)
**Prerequisite days:** 26 (retry storm data), 27 (Postmortem #2 + CPU-HPA "wrong signal" run)
**Feeds:** Day 29 (Autoscaling Strategy Memo — Deliverable #9), Day 35 (End-to-End Platform Design — Deliverable #11)
**Memory tracker update this day:** `Queue depth trigger lag vs. GPU util trigger lag | Day 28`

---

## Correction Table (v1 → v2 → v3 → v4)

### Round 1 (v1 → v2)

| # | Source | Reviewer point | Verdict | v2 change |
|---|---|---|---|---|
| 1 | Reviewer round 1 | Keep queue-vs-GPU-vs-CPU signal comparison | Accept (already present) | No change |
| 2 | Reviewer round 1 | Restore full 2-hr `kill -15` scale-down experiment as "Tier-1" | Reject | Meets all four skip criteria in `experiment-value-filter.md`: outcome deducible from Day 26, industry-established (K8s/Ray drain), numbers non-transferable, time redirectable. v1 already captures the one non-deducible detail (client-side stream-abort mode + V1 SIGTERM behavior) in 20-min observation + source dive. Hold. |
| 3 | Reviewer round 1 | Trim signal comparison table — "ordering matters, not numbers" | Reject | Contradicts memory tracker row `Queue depth trigger lag vs. GPU util trigger lag | Day 28` (asks for a number) and Day 29 §2 ("measured threshold values"). Without measured Δ, the Δ-vs-cold-start insight collapses to textbook ordering. |
| 4 | Reviewer round 1 | Reduce "long sweeps" | Partial reject | v1 specifies 2–3 × ~15 min ramps. Median lag requires ≥3 samples. Already minimal. |
| 5 | Reviewer round 1 | **Add cold-start latency measurement** | Accept | v1 had `[measure or estimate]` placeholder. v2 promotes to explicit 30-min measurement block in Morning. This second number is what makes the Δ-vs-cold-start insight quantitative. |
| 6 | Reviewer round 1 | Add "missed scaling window" scenario | Partial accept via reframe | Cannot be run empirically on single T4 (no replica to scale to). v2 adds a **Scale-Up Miss Statement** — derived analytically from measured Δ + measured cold-start, paired symmetrically with the Scale-Down Hazard Statement. Day now produces two composed failure statements (scale-up miss, scale-down drain-overrun). |

### Round 2 (v2 → v3)

| # | Source | Reviewer point | Verdict | v3 change |
|---|---|---|---|---|
| 7 | Reviewer round 2 | Add transferability caveat — absolute lag numbers are hardware/workload-specific; the reusable artifact is the inequality, not raw Δ | Accept | Added one sentence to the Δ-vs-cold-start paragraph in Morning Block 4. Hardens against "you overfit to your setup" critique without changing the measurement. |
| 8 | Reviewer round 2 | **Elevate dQ/dt (queue growth rate) to a third named timescale** alongside Δ and t_cold | Accept | Genuine structural upgrade. v2 buried dQ/dt inside Scale-Up Miss Statement as a computational detail. v3 promotes it to a named third timescale (`t_growth`). Scale-Up Miss Statement rewritten in three-timescale form. Symmetric pair summary updated. Stability condition becomes `Δ > t_cold AND capacity-gap/dQ/dt > t_cold − Δ`. |
| 9 | Reviewer round 2 | Add one qualitative-observation line to the 20-min kill run | Accept | Added to Afternoon Block 2 deliverables. Anchors the interview instinct ("scale-down is often more dangerous than scale-up") that pure derivation can't produce. |
| 10 | Reviewer round 2 | Add explicit "Final Insight: Three Timescales" closing block | Accept | New §7 at end of document. Crystallizes the day's mental model into one interview-ready summary. Natural closing given point 8. |

### Round 3 (v3 → v4)

| # | Source | Reviewer point | Verdict | v4 change |
|---|---|---|---|---|
| 11 | Reviewer round 3 | Stability condition `Δ > t_cold AND t_growth > t_cold` is technically coupled — t_growth depends on Δ via capacity_headroom at trigger | Accept | One-sentence clarification added to §7 stability condition. Defends against interview probe: "Is t_growth independent of Δ?" |
| 12 | Reviewer round 3 | Admission control is the first line of defense at frontier labs; underplayed | Accept | §7 updated to show admission control as the mechanism that parameterizes initial capacity_headroom and caps dQ/dt. Links back to Day 16 (divergence ratio admission control) and Day 25 (admission control retrofit narrative). |
| 13 | Reviewer round 3 | Reframe scale-down symmetrically as timescale problem (t_drain vs t_retry) | Partial accept | Reviewer's specific formulation conflates drain duration with retry amplitude — less rigorous than current amplitude-based condition. v4 adds a timescale-framing sentence for interview-delivery symmetry while **keeping** the amplitude-based inequality as the operational check. |
| 14 | Reviewer round 3 | Missing production hook — connect to K8s HPA custom metrics, Ray Serve, pre-provisioned replicas | Accept | New closing subsection in §7 ("Production Translation") maps the three-timescale theory to concrete production infra patterns. |

---

## 0. Filter Verdicts (applied before scheduling the day)

Per `experiment-value-filter.md`:

| Block (as written in Phase B v4) | Verdict | Reasoning |
|---|---|---|
| Morning: queue depth vs. GPU util trigger lag measurement | **RUN (tightened)** | Directions are deducible (queue leads, GPU util lags — industry-established). Magnitudes on this T4 + Qwen2.5-3B config are not, and the lag delta is a load-bearing input for Days 29 and 35. Skill: instrumenting V1 leading-indicator signals is production-relevant. |
| Afternoon: `kill -15` during in-flight requests (full 2 hrs) | **CONCEPTUAL (replaced)** | Outcome fully deducible from Day 26: killed streams → client errors → retries → the retry cascade already characterized in Postmortem #2. Well-established in SRE literature. No downstream deliverable requires a measured drain-time number. |

**Replacement for the afternoon kill experiment:** source dive (vLLM V1 SIGTERM behavior) + 20-min single kill observation (client-side stream-abort mode is the one non-deducible detail) + analytical drain protocol design + cross-linked hazard statement composing Day 24 cliff data with Day 26 amplification data.

**Cost of skip:** 2 hrs of confirming textbook retry-cascade dynamics.
**Value gained:** a quantified hazard statement of the form "force-terminating N requests at M% load produces an effective load transient past the cliff" — compositional reasoning across prior empirical results, which is the staff-level signal.

---

## 1. Morning (4 hrs) — Signal Lag Measurement + Cold-Start Measurement

### Time budget

| Block | Duration | Content |
|---|---|---|
| 1 | 45 min | Setup: instrumentation confirm + threshold definitions + ramp profile |
| 2 | 1 hr 45 min | Run ramp (2–3 reps) + signal time-series capture |
| 3 | 30 min | **Cold-start measurement (new in v2)** |
| 4 | 1 hr | Analysis + comparison table + Δ-vs-cold-start paragraph |

### Reframe up front

This is a **single-instance** setup. No HPA is actually firing. No second replica exists.
What's being measured: **trigger condition onset timing** — how early does each candidate signal cross its threshold relative to client-visible SLO breach.

**Precise definition of trigger lag:**

```
lag(signal) = t(signal crosses threshold) − t(p99 TTFT crosses SLO)
```

Negative lag ⇒ leading indicator. Positive ⇒ lagging. Zero ⇒ coincident.

State this framing explicitly in the writeup. It's more honest than claiming to have tested "HPA policies" on a single node, and the measured quantity is still what feeds the Autoscaling Memo.

### Block 1 — Setup (45 min)

**1. Confirm vLLM V1 queue-depth instrumentation.**

In V1 the scheduler maintains a **waiting queue** of `Request` objects not yet admitted to the `running` set. The relevant metric is the length of that waiting list per scheduler step. Verify your Day 8/9 instrumentation exposes it; if not, add a log line per scheduler step:

```python
# Somewhere in the V1 scheduler step path
logger.info(f"SCHED_STEP waiting={len(self.waiting)} running={len(self.running)} "
            f"kv_blocks_used={self.kv_cache_manager.num_used_blocks}")
```

Confirm the log line emits at ~scheduler tick rate, not per request.

**2. Define thresholds explicitly with justification.** No plucked-from-air numbers.

| Threshold | Value | Justification |
|---|---|---|
| KV block util (leading candidate) | Day 24 cliff − 3pp | Fire before the cliff, not at it |
| Waiting queue depth (candidate A) | 5 | Conservative |
| Waiting queue depth (candidate B) | 15 | Aggressive — lower false-positive rate |
| `nvidia-smi` GPU memory util | Any — included to demonstrate it's useless (see Block 3) | N/A |
| TTFT p99 SLO (reference) | 2 × below-cliff baseline p99 | Reasonable breach definition |

**3. Ramp profile.**

Use a **step function**, not a smooth ramp. Step functions make crossing timestamps unambiguous; smooth ramps make lag measurements noisy.

- 5 min at 40% of admission limit (baseline, well below cliff)
- Step to 60% — hold 3 min
- Step to Day 24 cliff point − 2pp — hold 3 min
- Step to cliff + 3pp — hold until all signals cross or system collapses

Total: ~15 min per run.

### Block 2 — Run the ramp (1 hr 45 min)

Run **2–3 repetitions**. Single-run lag numbers are noise. Report the median.

Log these four time series at 1 Hz:

| Stream | Source |
|---|---|
| `kv_util_pct` | V1 `KVCacheManager.num_used_blocks / num_total_blocks` from your instrumentation |
| `waiting_queue_len` | `len(scheduler.waiting)` from your instrumentation |
| `ttft_p99_60s_window` | Rolling 60s window from your Locust metrics |
| `nvsmi_gpu_mem_util` | `nvidia-smi --query-gpu=memory.used,memory.total` |

**Subtle V1-specific trap to document during the run:**

`nvidia-smi`'s `memory.used` is **near-pinned at vLLM startup**, because vLLM pre-allocates the KV cache block pool up front. It will read a flat high value regardless of actual KV load. **If your HPA is wired to `nvidia-smi` GPU memory util, it will never fire correctly for vLLM.** This is not a hypothetical — this is a real production misconfiguration. Catching it is a staff-level signal.

The only meaningful "GPU memory" signal is **active KV block occupancy** from inside vLLM, not from `nvidia-smi`.

### Block 3 — Cold-start measurement (30 min) **[new in v2]**

The Δ-vs-cold-start insight requires a second number: how long does a cold vLLM replica take to become ready to serve? This block produces that number for the current config (Qwen2.5-3B-Instruct on T4, `--dtype half`).

**Measurement protocol:**

1. Shut down the running vLLM instance cleanly.
2. Record wall-clock timestamp `t_0`.
3. Start vLLM with the same flags used throughout Phase B.
4. Record timestamps at three phase boundaries (parse from stderr log lines):
   - `t_1`: weights loaded into GPU memory (look for log line at end of HF model loading)
   - `t_2`: CUDA graph capture complete / block pool allocated (look for scheduler-ready log)
   - `t_3`: first request served end-to-end (send a single short warmup request, time from send to last token)
5. Report:

   | Phase | Duration |
   |---|---|
   | Weights load (`t_1 − t_0`) | |
   | CUDA graph + block pool (`t_2 − t_1`) | |
   | First-request warmup (`t_3 − t_2`) | |
   | **Total cold start (`t_3 − t_0`)** | |

Run twice; take the median. Single-run cold-start numbers can be skewed by page cache state (first run after reboot is slower because weights aren't in page cache).

**Why this block is worth 30 min and not more:**

- Not non-obvious enough for a full sweep — we're not varying model size or hardware
- But it's not deducible either — actual T4 + Qwen2.5-3B cold start could plausibly be anywhere from 15s to 90s depending on page cache, graph capture cost, and weight format
- The number is **required** to make the Δ-vs-cold-start argument quantitative rather than hand-wavy

### Block 4 — Analysis + comparison table (1 hr)

Produce the table with measured values:

| Signal | Trigger lag vs. SLO breach (median, s) | False-positive risk | Notes |
|---|---|---|---|
| CPU utilization | N/A | — | Day 27 data: CPU = [X]% at GPU saturation. Unusable signal. |
| `nvidia-smi` GPU memory util | ≈ 0 (pinned at startup) | — | **Wrong metric.** vLLM pre-allocates block pool; nvsmi memory util is flat. |
| KV block utilization (internal) | [your value] | Low | Coincident-to-slightly-leading. Fires near cliff. |
| Waiting queue depth = 5 | [your value] | Medium-high | Aggressive; may fire on transient bursts that self-resolve |
| Waiting queue depth = 15 | [your value] | Medium | Conservative; firm signal |
| TTFT p99 (direct) | 0 by definition | Low | Reference signal; not useful for scale-out because it is the breach |

### The differentiating insight

Add this paragraph to your memo — this is the line that turns "queue depth leads GPU util" (textbook) into a staff-level claim:

> The measured lead time of waiting-queue depth over SLO breach is **Δ = [your value from Block 2] seconds**. For this to function as a useful reactive scale-out signal, Δ must exceed the model cold-start time. Measured cold start on this config (Qwen2.5-3B-Instruct, T4, `--dtype half`) is **t_cold = [your value from Block 3] seconds**. Therefore:
>
> - If **Δ > t_cold**: queue depth is a sufficient reactive scale-out signal on this workload.
> - If **Δ ≤ t_cold**: reactive autoscaling is fundamentally inadequate for this workload regardless of signal choice — the queue-depth lead time is consumed by model load before the new replica is ready to serve. The system requires **predictive / pre-warmed capacity**. This is a capacity-planning conclusion, not a signal-selection conclusion.
>
> Measured result: **[state which side your numbers land on]**.

This framing is what the memo actually needs. "Queue depth is a leading indicator" is not interview-differentiating. "Measured Δ = X, measured cold-start = Y, here is why that forces [pre-warming / is acceptable reactive-only]" is.

**Transferability caveat (new in v3):** The absolute values of Δ and t_cold are hardware- and workload-specific (T4, Qwen2.5-3B, this batch config). On H100 + 70B, both numbers change substantially — t_cold likely grows faster than Δ. The reusable artifacts from this measurement are (a) the **inequality** `Δ vs. t_cold`, not the raw values, and (b) the **methodology** for measuring both on any given deployment. The memo should present the numbers as an illustrative instance, not as recommended thresholds for other configurations.

### Morning deliverables

1. Time-series data (CSV or JSON) from 2–3 ramp runs, four signals each
2. Comparison table with filled-in median lag values
3. **Cold-start measurement: 4-row table with phase durations + total (new in v2)**
4. The Δ-vs-cold-start paragraph with both measured numbers filled in and the sufficient/insufficient verdict stated
5. One-paragraph note on the `nvidia-smi` misconfiguration trap

---

## 2. Afternoon (4 hrs) — Scale-Down Hazard (analytical + narrow empirical)

### Block 1 — vLLM V1 SIGTERM source dive (30 min)

Read the V1 shutdown path. Specifically:

- `vllm/entrypoints/openai/api_server.py` — does the FastAPI/Uvicorn app install a SIGTERM handler that triggers ordered shutdown?
- `vllm/v1/engine/core.py` (or equivalent `EngineCore` module in your installed version) — is there a graceful drain path?

**Question to answer in writing:** On SIGTERM, does V1 (a) continue serving in-flight requests to completion before exiting, (b) abort all in-flight requests immediately, or (c) something in between? Cite file and function.

**Expected finding:** minimal or no application-level drain logic — the process relies on the orchestrator to handle drain externally (mark unhealthy → wait → SIGTERM). This is the factual gap that makes the hazard real: **out of the box, vLLM V1 is not safe to terminate under load.**

### Block 2 — Single 20-min kill observation (20 min)

One run only. This is the one detail not fully deducible: the exact client-visible error mode.

- Start a streaming completion request with a long `max_tokens` (so decode is ongoing)
- Wait until ~20 tokens have streamed
- From another shell: `kill -15 <vllm_pid>`
- Record what the HTTP client sees: connection reset? clean EOF with partial content? HTTP-level error? Mid-stream SSE truncation?
- **(v3) Additional observation with retry-enabled client:** run a second variant with 2–3 concurrent streaming requests and a retry-enabled client (same config as Day 26). After `kill -15`, record whether the retry burst is observable in client logs. Expected: immediate retry burst consistent with Day 26 cascade dynamics, delivered into the surviving replica pool (which in this single-T4 setup is nothing — but the retry-burst shape itself is the observable).

Document the exact error surface in one paragraph. This matters for retry policy design: if the client sees a clean EOF, it may treat the response as complete and **not retry**; if it sees a reset, it **will retry**. The two scenarios have opposite implications for scale-down hazard severity.

**Qualitative observation line for deliverable (v3):** After observing the retry burst, write one sentence of the form: *"Forced termination of [N] concurrent streaming requests produced [observed retry pattern] in the client logs, consistent with the amplification factor characterized in Postmortem #2."* This anchors the intuition "scale-down is often more dangerous than scale-up" with a direct experiential observation rather than pure derivation.

### Block 3 — Graceful drain protocol design (1.5 hrs)

Write the explicit state machine. Not prose — a table.

| State | Enter condition | Exit condition | Actions on enter | New-request handling |
|---|---|---|---|---|
| `RUNNING` | Process start | Drain signal received (orchestrator or SIGTERM) | Normal serving | Accept |
| `DRAINING` | Drain signal | `running == []` OR drain timer expires | Flip health probe to unhealthy; begin tracking in-flight count | **Reject with 503 + `Retry-After: <drain_window>`** |
| `TERMINATING` | Exit from `DRAINING` | Process exit | Force-close remaining streams; emit metric for forced-terminations | Reject (connection refused / 503) |

**Key design decisions to resolve explicitly:**

1. **Rejection code during `DRAINING`: 503 with `Retry-After` vs. 429.**
   - 503 signals "try elsewhere" — correct semantics for a draining instance
   - `Retry-After` value: set equal to drain window so retry hits a different (presumably not-draining) instance
   - Rationale: 429 implies rate limit on the client, which is wrong

2. **Drain window bound (numerical, for this config).**

   Use Day 6 Exp 3e regression and current concurrency:
   ```
   TTFT_max = 37.6 + 0.228 × prompt_max_tokens  (ms)
   decode_time_max = (completion_max_tokens × inter_token_latency_p99)  (ms)
   per_request_max = TTFT_max + decode_time_max
   drain_window = per_request_max + padding
   ```
   Plug in your measured inter-token latency from Day 23/24 at the chosen operating point. Produce a single number in seconds for your config.

3. **What if drain exceeds window?**
   Force-terminate remaining in-flight requests. They become client retries. This is where the cross-link to Day 26 lands (Block 4).

### Block 4 — Cross-linked hazard statement composing Days 24 + 26 (45 min)

This is the first of two composed failure statements. Instead of re-measuring, **compose** prior empirical findings.

Template sentence, fill in with your measured numbers:

> **Scale-Down Hazard Statement.** If graceful drain is enabled with a drain window of [T] seconds and new-request rejection at 503, the scale-down operation is safe provided all in-flight requests complete within T. If drain timeout is exceeded, the remaining N in-flight requests are force-terminated. Under the retry client configuration characterized in Postmortem #2 (2s timeout, 3 retries), force-termination produces an effective load transient of N × [Day 26 peak amplification factor] = [computed value] additional requests into the surviving replica pool. At steady-state [M]% load, this transient raises effective load to [M × amp]%. Compared to the Day 24 cliff at [cliff]%, this is **[below / at / above]** the cliff. Therefore, acceptable drain timeout for this config is [bounded] seconds; exceeding that bound re-triggers the retry cascade in the surviving pool and converts a single-instance scale-down into a fleet-wide collapse.

### Block 5 — Scale-Up Miss Statement (15 min) **[new in v2, expanded in v3]**

Symmetric to the scale-down hazard: derive the missed-scaling-window condition analytically from Morning measurements, expressed in three-timescale form.

**Reviewer proposed this as an experiment**; rejected as experiment (no second replica to scale to on single T4), accepted as derived statement.

**Three timescales involved (v3 — elevated from v2's two-timescale form):**

| Symbol | Name | Source | What it represents |
|---|---|---|---|
| Δ | Signal lead time | Morning Block 2 | How early the queue-depth signal fires before SLO breach |
| t_cold | Cold-start time | Morning Block 3 | How long a new replica takes to become ready to serve |
| **t_growth** | **Queue growth rate (dQ/dt)** | **Incident-specific input** | **How fast demand increases once pressure begins** |

**Why three, not two (v3 rationale):** Two systems with identical Δ and t_cold can still behave differently under different traffic patterns. A slow ramp may give the surviving replica enough headroom to absorb demand while the new replica loads; a sharp spike may cross the cliff before t_cold elapses. The viability of reactive scaling depends on all three jointly, not on Δ and t_cold alone.

Template sentence, fill in with your measured numbers:

> **Scale-Up Miss Statement (v3).** Let Δ be the measured lead time of queue-depth signal over SLO breach (Morning Block 2), t_cold be the measured cold-start time (Morning Block 3), and dQ/dt be the queue growth rate during the incident. A scale-out triggered at queue-depth threshold crossing produces a new serving replica at time (t_cold − Δ) after the SLO-breach moment — i.e., the new replica is late by (t_cold − Δ) unless t_cold < Δ.
>
> **Necessary condition for reactive scaling:** Δ > t_cold.
> **Sufficient condition for reactive scaling:** Δ > t_cold **AND** the surviving replica has enough capacity headroom to absorb the additional demand that accrues during t_cold seconds of growth at rate dQ/dt. Formally:
>
> ```
> (capacity_headroom_at_trigger) > dQ/dt × t_cold
> ```
>
> On this config: Δ = [value] s, t_cold = [value] s, necessary condition **[holds / fails]**. The sufficient condition is incident-specific — if dQ/dt exceeds `(KV_capacity − KV_at_trigger) / t_cold`, the surviving replica crosses the cliff before the new replica arrives. This is the **missed scaling window** regime.
>
> **Mitigation in each failure case:**
>
> - Necessary condition fails (Δ ≤ t_cold) → reactive autoscaling structurally inadequate; requires predictive / pre-warmed capacity regardless of traffic pattern
> - Necessary condition holds but sufficient condition may fail (fast spikes) → queue-depth reactive scaling works for slow ramps but must be supplemented with burst-absorbing buffers (admission-control rejection, warm pool of N pre-loaded replicas, etc.) to handle sharp spikes

### The symmetric pair (v3: three-timescale form)

Day 28 produces two composed failure statements that pair symmetrically, expressed in three-timescale form:

| Failure mode | Timescales involved | Composed from | Stability condition |
|---|---|---|---|
| **Scale-Up Miss** | Δ, t_cold, t_growth (dQ/dt) | Day 28 Morning measurements | Δ > t_cold **AND** dQ/dt × t_cold < capacity headroom at trigger |
| **Scale-Down Hazard** | drain_window T, amplification factor, cliff | Day 24 cliff + Day 26 amp + Day 28 drain derivation | All in-flight requests complete within T seconds **OR** forced-termination amp × steady-state load < cliff |

Both are the kind of compositional argument that separates strong-debugger from staff-designer: prior empirical results → derivation → design constraint with a number on it.

Write both as closing sections of the Scale-Down Hazard Analysis + Scale-Up Miss Analysis, which together become a section of the Day 29 Autoscaling Memo.

### Afternoon deliverables

1. V1 SIGTERM source-dive note with file/function references and factual finding
2. One-paragraph client-side stream-abort error-mode observation
3. Graceful drain state-machine table with drain-window derivation
4. Scale-Down Hazard Statement (composed from Days 24 + 26, with your numbers plugged in)
5. **Scale-Up Miss Statement (composed from Morning Δ + t_cold, with your numbers plugged in) — new in v2**
6. **Symmetric pair summary table — new in v2**

---

## 3. End-of-Day Deliverables Checklist

- [ ] Signal lag time series (2–3 ramp runs, 4 signals)
- [ ] Comparison table with measured median lag values
- [ ] Cold-start measurement table (4 rows: weights load, CUDA graph + block pool, warmup, total)
- [ ] Δ-vs-cold-start paragraph with **both** measured numbers and sufficient/insufficient verdict
- [ ] **Transferability caveat appended to Δ-vs-cold-start paragraph (new in v3)**
- [ ] `nvidia-smi` memory-util misconfiguration note
- [ ] V1 SIGTERM source-dive note
- [ ] Client-side stream-abort error-mode observation (1 paragraph)
- [ ] **Qualitative retry-burst observation from concurrent-kill variant (new in v3)**
- [ ] Graceful drain state machine with numerical drain-window bound
- [ ] Scale-Down Hazard Statement composing Days 24 + 26
- [ ] **Scale-Up Miss Statement in three-timescale form (Δ, t_cold, t_growth) — upgraded in v3**
- [ ] Symmetric failure-pair summary table (three-timescale form, updated in v3)
- [ ] **Final Insight: Three Timescales — §7 written (new in v3)**
- [ ] Memory tracker row filled: `Queue depth trigger lag vs. GPU util trigger lag | Day 28`

---

## 4. Forward References (to be cited in Day 29)

The Autoscaling Memo (Deliverable #9) will cite from this day:

- The signal comparison table → Section 2 (signal selection)
- The Δ-vs-cold-start paragraph → Section 3 (scale-out policy) — justifies pre-warming requirement (or its absence, depending on which side Δ vs. t_cold lands on)
- The cold-start measurement → Section 3 (scale-out policy) — provides the t_cold number that the memo's pre-warming recommendation pivots on
- **Transferability caveat → Section 2 disclaimer — makes explicit that numbers are config-specific, inequality is the portable artifact (new in v3)**
- The `nvidia-smi` misconfiguration note → Section 2 (what not to use)
- The graceful drain state machine → Section 4 (scale-down policy)
- The Scale-Down Hazard Statement → Section 4 (scale-down policy) — the quantified hazard that justifies conservative drain windows
- The Scale-Up Miss Statement → Section 3 (scale-out policy) — the quantified condition under which reactive scaling is insufficient
- **§7 Three-Timescale Final Insight → Memo executive summary / conclusion (new in v3) — the headline mental model**

---

## 5. V1 Correctness Checks (per standing policy)

Terms used in this document and verified V1-correct:

- ✅ `waiting` queue of `Request` objects in the scheduler
- ✅ `running` set of `Request` objects
- ✅ `KVCacheManager` / `BlockPool` for KV block accounting
- ✅ Recompute-only preemption (no swap, no `SWAPPED` state)
- ✅ No `AsyncLLMEngine`, no `SequenceGroup`, no `BlockManager` / `block_manager_v2.py` references

Terms explicitly **not** used (V0 residue to avoid): `SWAPPED`, `BlockManager`, `SequenceGroup`, `AsyncLLMEngine`, CPU swap path.

---

## 6. Filter Filter — what's NOT in this day

Things the v4 spec suggested, or that reviewer LLMs proposed, that were explicitly rejected:

| Proposed | Source | Verdict | Reasoning |
|---|---|---|---|
| Full 2-hr `kill -15` experiment with multiple variations | Phase B v4 spec | Reject | Outcome deducible; replaced with 20-min observation + source dive + composed hazard analysis |
| Re-run CPU-based HPA experiment | Phase B v4 spec | Reject | Already done Day 27 afternoon; cite that data, don't re-measure |
| Simulated second vLLM process for actual HPA firing | Phase B v4 spec | Reject | Single-T4 constraint; "trigger condition onset timing" reframe captures the scientific content without requiring multi-replica |
| Measure drain time empirically with varying concurrency | Phase B v4 spec | Reject | Derivable from Day 6 TTFT regression + current inter-token latency; empirical sweep adds no non-deducible signal |
| Test exponential backoff variants during scale-down hazard | Phase B v4 spec | Reject | Per `experiment-value-filter.md` example: backoff direction is known, exact delta is workload-specific, no downstream deliverable requires the number |
| **Restore full 2-hr `kill -15` as "Tier-1" experiment** | Reviewer round 1 | Reject | Meets all four skip criteria; reviewer cited "K8s drain, Ray Serve drain" as *reason to keep*, which is exactly the industry-established criterion for skipping. v1 already captures the one non-deducible detail (client-side error surface + V1 SIGTERM behavior) in 20-min observation + source dive |
| **Trim signal comparison table to ordering only** | Reviewer round 1 | Reject | Contradicts memory tracker (asks for a number) and Day 29 §2 (cites measured values). Without measured Δ, the Δ-vs-cold-start insight collapses to textbook ordering |
| **Run "missed scaling window" as an experiment** | Reviewer round 1 | Reject as experiment | No second replica exists on single T4 — cannot empirically test scale-out. Accepted as **derived** Scale-Up Miss Statement instead, composed from measured Δ + measured t_cold |

### What was added from reviewer round 1 (accepted)

| Added | Source | Location in v2 |
|---|---|---|
| Cold-start latency measurement (30 min, explicit) | Reviewer round 1 | Morning Block 3 (new) |
| Scale-Up Miss Statement (analytical, paired with Scale-Down Hazard) | Reviewer round 1 reframed | Afternoon Block 5 (new) + symmetric pair summary |

### What was added from reviewer round 2 (accepted)

| Added | Source | Location in v3 |
|---|---|---|
| Transferability caveat (numbers are config-specific; reusable artifact is the inequality) | Reviewer round 2 | Morning Block 4 (end of Δ-vs-cold-start paragraph) |
| t_growth (dQ/dt) elevated to named third timescale | Reviewer round 2 | Afternoon Block 5 (rewritten) + symmetric pair summary (updated) |
| Qualitative retry-burst observation during kill run | Reviewer round 2 | Afternoon Block 2 (added variant + deliverable line) |
| Three-Timescale Final Insight block | Reviewer round 2 | §7 (new) |

### What was added from reviewer round 3 (accepted)

| Added | Source | Location in v4 |
|---|---|---|
| Coupling clarification: Δ and t_growth not orthogonal | Reviewer round 3 | §7 stability condition (new "Coupling note") |
| Admission control elevated to first line of defense; links to Day 16 / Day 25 | Reviewer round 3 | §7 new subsection ("The role of admission control") |
| Timescale-framing restatement of scale-down (t_drain, t_request, t_retry) for symmetry | Reviewer round 3 | §7 symmetric scale-down (added paragraph; amplitude condition retained as operational check) |
| Production translation table — K8s / Ray Serve / admission control patterns | Reviewer round 3 | §7 new closing subsection ("Production translation") |

---

## 7. Final Insight — Three Timescales (new in v3)

The day's composed failure statements reduce to a single mental model, suitable for interview delivery.

### The mental model

Autoscaling in GPU inference is constrained by three interacting timescales. System stability under load requires a specific relationship among them.

| Timescale | What it is | Typical magnitude (this config) | Controlled by |
|---|---|---|---|
| **Δ** | Signal lead time — how early the queue-depth signal fires before SLO breach | [Morning Block 2 value] s | Signal choice + threshold tuning |
| **t_cold** | Scale-out latency — how long a new replica takes to become ready to serve | [Morning Block 3 value] s | Model size, hardware, warmup strategy, pre-loading |
| **t_growth** | Queue growth timescale — how fast demand accrues once pressure begins (≈ capacity_headroom / dQ/dt) | Incident-specific | Traffic pattern (slow ramp vs. sharp spike) |

### Stability condition

```
Reactive scale-out is viable  ⟺  Δ > t_cold  AND  t_growth > t_cold
```

- **If Δ ≤ t_cold:** signal fires too late regardless of traffic pattern → requires predictive / pre-warmed capacity
- **If Δ > t_cold but t_growth < t_cold:** reactive scaling keeps up with slow ramps but fails on sharp spikes → requires burst absorption (admission control rejection, warm pool, etc.)
- **Both hold:** reactive queue-depth scaling is sufficient

**Coupling note (v4):** The two conjuncts are not independent. `t_growth ≈ capacity_headroom_at_trigger / (dQ/dt)`, and `capacity_headroom_at_trigger` is itself a function of Δ — an earlier-firing signal (larger Δ) triggers at a lower utilization and therefore leaves more headroom, which increases t_growth. So `Δ` and `t_growth` are positively coupled: choosing an earlier signal simultaneously improves both conjuncts. Interview-grade phrasing: **"The three timescales decompose reactive scaling viability, but they're not orthogonal — Δ and t_growth are jointly tuned by the signal threshold choice."** This is the answer to the interview probe "Is t_growth independent of Δ?"

### The role of admission control (v4)

Admission control is the mechanism that **parameterizes** the stability condition. Specifically:

- It sets **initial capacity_headroom** at the moment the signal fires, by defining the operating ceiling (e.g., reject when KV utilization approaches cliff minus safety margin — see Day 25 retrofit narrative)
- It caps **dQ/dt** during spikes, by rejecting requests rather than queueing them past a threshold

Consequently, at frontier labs admission control is typically the **first line of defense**, not autoscaling. The sequence is:

1. Admission control absorbs bursts → keeps dQ/dt bounded
2. Queue depth signal fires → autoscaler initiates scale-out
3. New replica arrives before t_growth elapses → load redistributes

Autoscaling-without-admission-control is fragile because an unbounded dQ/dt can violate `t_growth > t_cold` for any finite t_cold, regardless of signal choice. This is the concrete linkage from Day 28 back to Day 16 (divergence-ratio admission control) and Day 25 (admission control retrofit).

### The symmetric scale-down failure

Scale-down has its own timescale inequality, composed from Day 24 + Day 26:

```
Scale-down is safe  ⟺  drain_window > max_request_duration  AND
                        forced_termination_count × amp < (cliff − steady_state_load)
```

Violating either clause converts a single-instance scale-down into a fleet-wide retry cascade.

**Timescale-framing restatement (v4, for symmetry with scale-up):** Scale-down also involves three competing timescales — `t_drain` (actual time to complete in-flight requests), `t_request` (max per-request duration), and `t_retry` (retry amplification timescale, governed by client timeout × amplification factor). The timescale view: the drain window must exceed t_request (first conjunct), and if it doesn't, force-termination must not produce a retry burst that outruns the cliff (second conjunct). **Both scale-up and scale-down reduce to: the system must react faster than demand changes, in both directions.** The operational check remains amplitude-based — the timescale view is for mental-model coherence, not operational substitution.

### Interview-delivery form

> "Autoscaling in LLM inference is constrained by three timescales: Δ, the queue-depth signal lead time; t_cold, the cold-start latency of a new replica; and t_growth, the queue growth timescale during an incident. Reactive autoscaling is viable only when Δ exceeds t_cold and t_growth exceeds t_cold. On our config, we measured Δ = [X] s and t_cold = [Y] s, which [satisfies / violates] the necessary condition. Under sharp spikes — even when the necessary condition holds — reactive scaling still fails unless augmented with a warm pool sized to absorb the burst during t_cold. The symmetric failure on scale-down is governed by a drain-window condition composed from our measured retry amplification and cliff location: force-terminating in-flight requests at steady-state load produces an effective transient of `N × amp`, which must remain below the cliff. Both sides reduce to: the system must react faster than the demand changes, in both directions."

That is the interview answer. It's not memorizable in that exact form — but if you internalize the three timescales and both stability conditions, you can reconstruct it from the mental model alone.

### What's NOT in this mental model (and why)

The mental model deliberately excludes signal-type comparisons (CPU vs. GPU vs. queue vs. TTFT). Those are a subordinate question: given that you've chosen the best available leading signal, the three-timescale analysis tells you whether reactive scaling works at all. Signal choice is necessary but not sufficient; the timescale analysis is the sufficient test.

This reframing — from "which signal?" to "which timescales, and in what relationship?" — is the Day 28 headline contribution to the portfolio.

### Production translation (v4)

The three-timescale theory maps to three concrete infra patterns:

| Theoretical element | Production translation |
|---|---|
| **Δ (signal lead time)** | Custom autoscaling metrics exposing scheduler-internal queue depth. In K8s: custom metrics API (Prometheus adapter) publishing `vllm_scheduler_waiting_queue_length` as an HPA target, replacing or supplementing GPU-memory targets (which are unusable per the `nvidia-smi` trap). In Ray Serve: `num_ongoing_requests_per_replica` with low autoscaling thresholds. |
| **t_cold (scale-out latency)** | Pre-warmed replica pools sized to cover t_cold. In K8s: `minReplicas` > active demand, with Karpenter/Cluster Autoscaler provisioning replacement capacity ahead of spikes. In KubeRay: `idleTimeoutSeconds` tuned high to retain warm replicas. Model weights pre-cached to local SSD to reduce t_cold. |
| **dQ/dt cap (admission control)** | Token-budget or divergence-ratio admission control at the gateway layer (per Day 16 / Day 25), returning 429 or 503 when instantaneous demand would exceed the capacity the autoscaler can provision within t_cold. Admission control is the first line of defense; autoscaling is the second. |

Staff-level statement of what a production autoscaling stack actually looks like:

> "Reactive HPA on queue-depth custom metrics, backed by a pre-warmed replica pool sized to absorb t_cold of demand growth, fronted by admission control that caps dQ/dt. The three components correspond one-to-one to the three timescales. Remove any one and you expose a failure regime."

This is the shape of the actual recommendation the Day 29 Autoscaling Memo will make.


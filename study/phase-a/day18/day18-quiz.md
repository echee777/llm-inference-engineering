# Day 18 Quiz Session — Load Testing + Admission Control Validation

## Concepts Covered

### 1. Prefill vs. Decode Asymmetry [QUIZ — PASSED]
- Long prompts = prefill-heavy = compute-bound (GEMM workload)
- Long completions = decode-heavy = memory-bound (KV cache)
- Same token budget, completely different GPU resource stress
- Admission control only gates on memory budget, blind to compute
- Key insight: `max_completion_tokens` overestimates because it's a ceiling, not a prediction. Day 17's progressive release addresses this.
- Temporal mismatch: decode-heavy requests ramp KV usage over time, so point-in-time budget is more conservative than reality for decode-heavy mixes.

### 2. Two-Dimensional Traffic Matrix [TEACH]
- Vary both prompt length AND max_tokens independently
- Include prefill-heavy bucket (long-prompt/short-output) explicitly to probe AC's compute-bound blind spot
- Traffic weights reflect production distributions
- Interview framing: "I designed the experiment, not just ran a load test"

### 3. Token Budget Admission Control [QUIZ — PASSED]
- `active_token_budget + estimated_cost > ADMISSION_BUDGET` => 429
- Why not request count: blind to request heterogeneity (100 short = 1 long)
- Why not GPU util%: blunt metric, single lightweight kernel shows 100%. True compute profiling requires nsight (Nsight Systems/Compute)
- Active token budget directly tracks the resource that causes cascading failure (KV memory)

### 4. Recompute-Only Preemption in vLLM V1 [TEACH]
- V1 removed swap path entirely. No SWAPPED state.
- Preemption = KV blocks freed, request re-enters WAITING, must re-prefill from scratch
- Creates positive feedback loop: preemption adds new prefill work on top of existing queue
- Even though scheduler checks capacity before re-admitting, the system wastes compute on redundant re-prefills, reducing effective throughput, keeping occupancy pinned at ceiling
- This is the mechanism behind the TTFT hockey stick, not gradual degradation
- AC prevents this by keeping KV below the fragmentation threshold

### 5. stream=True and Valid TTFT [QUIZ — PASSED]
- `stream=False` measures total generation latency, not TTFT
- For a 2000-token completion, the difference is seconds
- TTFT matters for two reasons: (1) operational health signal for system stability, (2) user experience metric (perceived responsiveness, parallel consumption of streamed output)

### 6. Gateway TTFT vs. Model TTFT [TEACH]
- Gateway TTFT = arrival_time to first_token_time (what client experiences)
- Model TTFT = admitted_time to first_token_time (pure backend performance)
- Delta = queue wait + gateway overhead
- Diagnostic patterns:
  - Gateway rising, model flat => queue building, need to scale
  - Both rising => backend under pressure despite AC, check KV/preemption
  - Both flat under load => success state, AC is working

### 7. The 65% Target and Estimation Error [QUIZ — PASSED]
- Overestimation (safe): wasted capacity, spurious 429s
- Underestimation (unsafe): AC fails to prevent preemption cascades, defeats its purpose
- 65% is deliberately conservative, biased toward safety
- Headroom absorbs token estimation bias

### 8. Prometheus histogram_quantile [TEACH]
- Histograms store cumulative bucket counters, not individual values
- `rate(bucket[1m])` converts cumulative counters into a recent-window snapshot (NOT the rate of change of the percentile)
- `histogram_quantile(0.99, rate(...))` finds which bucket p99 falls in, linearly interpolates within it
- Bucket boundaries matter: dense buckets around SLO threshold for accuracy
- Low traffic can produce NaN from near-zero rates

### 9. Dashboard Causal Story (7 Panels) [QUIZ — PASSED]
Panels and their purpose in incident diagnosis:

1. **Active Token Budget %** — First look. Are we at the admission limit?
2. **Queue Depth** — Leading indicator of pressure. With queue wait, shows requests piling up and user wait time.
3. **GPU KV Cache Utilization** — Compare against Panel 1 to validate estimation accuracy. Divergence = miscalibrated budget.
4. **TTFT p50/p95/p99** — Percentile values over sliding window. Tail divergence (p99 >> p50) is early warning.
5. **Admitted/Rejected/Queued Rates** — Cost of protection. Rejections rise while admits stabilize under load. Queued shows burst absorption.
6. **Token Budget vs TTFT p99 (dual Y-axis)** — The money panel. With AC: both flat. Without AC: budget climbs, p99 lags then explodes.
7. **Gateway TTFT vs Model TTFT** — Isolates queue delay from backend degradation. Both stable and close = healthy.

### 10. Compute-Bound vs. Memory-Bound Failure [QUIZ — PASSED]
- AC does not protect against prefill compute saturation
- Dashboard signature: Panel 3 KV flat + Panel 7 model TTFT rising + Panel 4 p99 diverging from p50
- Production mitigations: prefill rate limiting, prompt length caps, separate admission budgets for prefill-heavy vs decode-heavy traffic

---

## Interview Round Results

```
Q1 (max_tokens vs avg estimation)     Strong. Safety argument + progressive release.
Q2 (budget vs KV mismatch)            Strong. Both failure sources identified,
                                       including preemption + release interaction.
Q3 (GPU util% as signal)              Clean, concise.
Q4 (reducing rejection rate)          Good. Progressive release + queuing.
                                       Could mention scaling sooner.
Q5 (AC limitations)                   Got the concept. Needed prompting for
                                       specific multi-panel dashboard signature.
Q6 (queue wait vs gateway TTFT)       Clean.
Q7 (429 as success in Locust)         Nailed reasoning despite not knowing Locust.
```

**Key coaching point:** On dashboard questions, lead with the multi-panel signature (e.g., "Panel 3 KV flat, Panel 7 model TTFT rising, Panel 4 p99 diverging from p50") before explaining the reasoning. Interviewers want correlated signals, not single metrics.

---

## Hands-On Execution Plan

**Approach:** Claude runs the baseline test (AC disabled), debugs any issues, and validates the pipeline end-to-end. Then the user runs the controlled test (AC enabled) for hands-on experience observing admission control in action. Claude handles Prometheus/Grafana setup and dashboard construction.

**Rationale:** The learning value is in observing system behavior under load and interpreting the results, not in debugging setup issues or repeating identical commands. One hands-on run gives enough visceral feel; the second run is mechanically identical.

# Day 26 — Retry Storm Setup + Baseline
## v3

**Week 6 goal:** Produce Postmortem #2 (Retry Cascade) and the Autoscaling Strategy Memo.
Today is setup + storm induction. Deliverable #8 is written Day 27.

---

## Correction Table

| # | Change | Rationale |
|---|---|---|
| 1 | Control baseline compressed 2 hrs → 30 min | Day 24 10-point sweep already established healthy-system behavior; confirmation only |
| 2 | Recovery observation compressed 30 min → folded into storm run | Insight (in-flight retries prevent fast recovery) is conceptual; 10 min passive tail sufficient |
| 3 | Storm run padded time reduced; recovered time reallocated to backoff variant | Option A below — turns Day 27 remediation from prescriptive to evidence-backed |
| 4 | Added effective load formula to Step 3 | Directly connects Day 24 cliff to Day 26 cascade — produces a cross-deliverable data point during the run |
| 5 | Added explicit KV cascade chain to Step 3 | Forces correct instrumentation during the run rather than post-hoc reconstruction |
| 6 | Added amplification threshold detection to Step 6 | Instability threshold is a design input for Day 27 retry budget spec, not just an observation |
| 7 | Added drain mechanism to recovery tail (Step 4) | Completes the mental model: collapse = feedback loop, recovery = drain problem; not just convergence |
| 8 | Upgraded KV utilization tracking to coupling point language (Step 4) | Causal anchor — point where KV util crosses cliff AND amplification accelerates simultaneously |

---

## Morning (3 hrs)

### Step 1 — Build retry client (2 hrs)

Modify your existing Locust script. Produce **two clients**: retry-enabled and no-retry control (identical otherwise).

```python
# Per-request config
TIMEOUT_SECS = 2.0
MAX_RETRIES = 3

# Log per request:
#   original_request_id
#   attempt_number
#   outcome: success / timeout / error

# Compute continuously:
# amplification_factor = total_requests_sent / unique_requests_originated
# Healthy system = ~1.0
```

**Why amplification factor is non-negotiable:** It is the single metric that makes a retry storm visible in real time. It also answers the staff-level interview question: "how do you detect a retry storm before it completes?" You need the time series, not just the peak value.

---

### Step 2 — Control baseline confirm (30 min)

- No-retry client, **60% of your Day 24 admission limit**
- Run 5 minutes. Confirm: TTFT stable, queue depth low, amplification = 1.0
- Save metrics. This is your pre-storm reference. Do not over-invest here — you know what healthy looks like from Day 24.

---

## Afternoon (4 hrs)

### Step 3 — Experiment design (20 min)

Scenario:
1. Start retry-enabled clients at 60% load
2. Stabilize 2 minutes
3. Inject **30-second burst**: long-context requests (4K tokens) at **2× normal rate** — the incident trigger
4. Stop burst. Observe whether retry behavior sustains the collapse after the trigger ends

The core hypothesis: **the trigger is transient; the cascade is self-sustaining.** Your data will confirm or deny this.

**Effective load formula — compute during the run, compare to your Day 24 cliff:**
```
effective_load = base_load × amplification_factor
# e.g., base=0.60, amplification=2.0 → effective=1.20 → beyond cliff → collapse
```

**KV cascade chain — what you're actually instrumenting:**
```
retries → more concurrent requests
       → more KV block allocations
       → faster KV exhaustion
       → V1 recompute-only preemption triggers
       → TTFT spike
       → more timeouts → more retries
```
Track KV utilization % alongside amplification factor. The moment both start rising together, the loop is live.

---

### Step 4 — Storm run (2 hrs)

Watch for four stages. Record exact timestamps:

| Stage | Observable |
|---|---|
| 1 | Burst → latency spike → requests exceed 2s timeout |
| 2 | Clients retry → effective load crosses admission limit |
| 3 | Higher load → more latency → more timeouts → more retries |
| 4 | Queue builds → KV util hits cliff → collapse |

**Track continuously:**
- Amplification factor time series (primary metric)
- Unique request rate vs. total request rate
- TTFT p50/p99
- Queue depth
- KV utilization % — identify the point where KV util crosses your Day 24 cliff threshold *and* amplification factor begins accelerating simultaneously; this is the causal coupling point, not just correlation

**Let the system collapse. Do not intervene.**

After collapse: stop burst injection, note timestamp, run 10 more minutes passively. Record when TTFT normalizes. This is your recovery tail — do not over-watch it. Note the mechanism: recovery is slow because retried requests remain in-flight and queued, continuing to consume KV cache and scheduler budget even after the trigger ends. This is a drain problem, not a convergence problem.

---

### Step 5 — Option A: Backoff variant (1.5 hrs) ← HIGH PRIORITY

**If Step 4 completes cleanly, run this before stopping for the day.**

Re-run the identical storm scenario with **exponential backoff + jitter** on the retry client:

```python
# Backoff config
BASE_DELAY_S = 0.5
MAX_DELAY_S = 10.0
JITTER_FRACTION = 0.5  # ±50%

import random, time
def backoff_delay(attempt):
    delay = min(BASE_DELAY_S * (2 ** attempt), MAX_DELAY_S)
    jitter = delay * JITTER_FRACTION * (random.random() * 2 - 1)
    time.sleep(delay + jitter)
```

Record the same metrics. Compare amplification curves: **unmitigated vs. mitigated**.

**Why this is high signal:** Day 27's Postmortem #2 remediation section becomes evidence-backed ("backoff reduced peak amplification factor from X to Y") rather than prescriptive ("you should use backoff"). That's the difference between a staff-level postmortem and a blog post.

---

### Step 6 — Note capture (30 min)

While data is fresh:
- Write stage timestamps into a timeline skeleton (you'll finalize Day 27)
- Note: at what amplification factor did the cascade become self-sustaining? (i.e., the loop continued rising after the burst ended — this is your instability threshold, and it becomes the design input for Day 27's retry budget spec)
- Note: did the burst end before collapse completed? (It should — that's the key finding)
- Sketch the amplification curve from memory; compare to actual trace

---

## End-of-Day Outputs

- Retry storm data: **amplification factor time series**, collapse/recovery timeline with stage timestamps
- (If Option A complete) Backoff variant amplification curve for comparison
- Pre-storm baseline metrics saved separately
- Timeline skeleton ready for Day 27 postmortem

---

## Appendix C Entry (fill after run)

| Metric | Your Value | Day Measured |
|---|---|---|
| Retry storm peak amplification factor | ___ | Day 26 |

---

## Day 27 Preview

Postmortem #2 (Deliverable #8). Required:
- Dual-axis amplification graph: unique req rate vs. total req rate
- Narrative: what broke / what you expected / what surprised you / what you'd change
- Remediation: retry budgets, backoff spec, circuit breaker — backed by your Day 26 data if Option A was run
- Decision Section (required per Phase B spec)

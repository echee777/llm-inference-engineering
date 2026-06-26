# Day 25 — Week 5 Buffer + Consolidation
## AI Inference Platform Residency — Phase B, Track 1
**Version:** v7
**Hardware:** T4 / g4dn.xlarge
**Model:** Qwen2.5-3B-Instruct

---

## Purpose

Quality gate and conversion day. Four outputs: (1) D#5–D#7 internally consistent and finalized, (2) a connection narrative that frames all three Week 5 findings under a single systems principle, (3) an admission control retrofit design that is interview-quotable, (4) Week 6 retry client implemented and smoke-tested.

---

## Morning (4 hrs) — Consistency Audit + Targeted Re-Runs

### Step 1: Cross-Reference Consistency Check (1 hr)

The three deliverables share experimental data. Inconsistencies are interview liabilities — an interviewer who asks you to reconcile two numbers you can't reconcile will downgrade your credibility on the entire portfolio.

**Check these specific cross-references:**

| Claim | Must be consistent across |
|---|---|
| KV utilization % at first preemption event | D#5 timeline and D#7 preemption-rate graph |
| Short-request TTFT degradation factor | D#6 must state the KV utilization level at which it was measured (target ~65%) |
| Cliff point (KV util %) | D#7 identifies it; D#5 should corroborate that this is where the preemption positive-feedback loop accelerated |
| Divergence ratio (p99/p50 ≥ 2×) as cliff definition | Must be consistently applied in both D#5 and D#7 — not visual judgment |

If any of these are inconsistent: fix the prose, not the data. If the data itself is contradictory, flag it in the deliverable's limitations section with a likely cause.

### Step 2: Targeted Re-Runs if Noise Flagged (up to 3 hrs, conditional)

**Only proceed if Step 1 flagged specific noisy data points.** If data is clean, skip to the afternoon.

Signs a data point is suspect:
- p99/p50 divergence ratio flipped direction between runs at the same utilization level
- Preemption counts differed by >20% across identical concurrency levels
- TTFT p99 at a given KV utilization is out of place relative to the surrounding curve shape

Re-run only the flagged points. If Step 2 completes early, use remaining time to polish graph labels, axis ranges, and legends on D#5–D#7. Unlabeled axes are credibility hits in interviews.

---

## Afternoon (4 hrs) — Synthesis + Week 6 Setup

### Step 3: Week 5 Connection Narrative (1 hr)

**Output:** One page. Three findings → one unifying principle → one production argument → one concrete design change.

This is not a summary of what you did. It is an argument about what it means for a production system.

---

**Unification frame — write this first, before the three findings:**

> All three Week 5 findings are instances of the same problem: **memory scheduling under contention**. KV cache exhaustion, prefill/decode interference, and the utilization cliff are not three separate failure modes — they are three manifestations of what happens when memory allocation, scheduling fairness, and system stability interact under load. This is the class of problem inference infrastructure engineers spend their careers on.

This sentence is what separates "I ran some experiments" from "I understand what class of system this is."

---

**The three findings (one paragraph each):**

1. *KV cache is the hard ceiling, not compute.* Under concurrent long-context load, KV blocks exhaust before compute saturates. In V1, recompute-only preemption amplifies the problem: preempted requests re-enter the queue and re-compete for the same blocks. The system does not fail gracefully — it enters a positive-feedback loop that is memory-mediated, not queue-mediated.

2. *Mixed request lengths create latency unfairness invisible in standard metrics.* A long-prefill request (2K tokens) blocks decode steps for co-scheduled short requests. This does not appear in GPU utilization dashboards. At the concurrency levels you tested, short-request TTFT degraded by [your measured factor]× when co-scheduled with long requests — silently, without any capacity alarm firing.

3. *There is a non-linear utilization cliff that makes headroom non-negotiable.* Above [your cliff point]% KV utilization, p99 TTFT diverges non-linearly from p50. Operating at 80% does not cost 15% more latency — it costs [your measured multiple]× p99 TTFT.

**Explicit tradeoff statement — required:**

> We are explicitly sacrificing peak throughput (tokens/sec) in exchange for stable p99 latency. At [cliff − margin]% vs [cliff]% operating point, throughput loss ≈ [X]%. This is the cost of stability. The cliff graph is the evidence when leadership asks why we're not at 85%.

**Compound failure crystallization — required single sentence:**

> At high utilization, prefill/decode interference increases effective concurrency, which accelerates KV exhaustion, which triggers recompute preemption, which re-consumes KV blocks and further increases concurrency — forming a closed-loop instability where the three failure modes amplify each other.

---

**Failure interaction table — write this as part of the narrative:**

| Condition | Isolated effect | Combined effect |
|---|---|---|
| High KV utilization only | Preemption onset, p99 begins rising | — |
| Mixed request lengths only | Short-request TTFT degrades [X]× | — |
| Both simultaneously | Preemption more frequent (long requests consume more blocks), short requests suffer compound degradation | Cliff point shifts lower than homogeneous measurement |
| All three + retry clients | Retry amplification adds artificial concurrency load on top of preemption load | **Compound failure: system enters closed-loop instability faster, with no self-stabilization** |

> The compound failure row is your new insight artifact. Most candidates can describe individual failure modes. Explaining what happens when all three interact simultaneously — and why the system cannot self-stabilize — is staff-level signal.

---

**The production argument:**
A static token budget admission gate (your Phase A design) addresses constraint #1 at design time but cannot respond to constraints #2 and #3 in real time. It will admit a mix that creates interference and has no feedback signal when KV utilization approaches the cliff.

### Step 4: Admission Control Retrofit Design (1 hr)

**Output:** A numbered, defensible design decision list. Feeds directly into D#9 and D#11. Write it so you can quote it verbatim.

```
Signal change:
  FROM: static token budget estimate (computed at gateway from num_kv_heads × head_dim × layers)
  TO:   real-time vllm_gpu_cache_usage_perc (Prometheus) as the primary admission signal
  WHY:  estimated budget and actual vLLM allocation diverge under mixed-length traffic;
        real-time signal captures the actual cliff; your Day 24 data showed the cliff
        at [X]% — a static gate calibrated to token budget alone would not have caught this

Hard rejection threshold:
  If KV utilization ≥ cliff_point - 2% → reject all new requests immediately (HTTP 429)
  WHY:  past cliff_point - 2%, the system is in the unstable regime. Admitting more requests
        does not increase throughput — it accelerates the feedback loop. The correct action
        is hard rejection, not queuing. This is a decisive policy, not a conservative one.

Tiered admission threshold:
  Short requests (prompt < 256 tokens):   admit up to cliff_point - 5%
  Medium requests (prompt 256–1024):      admit up to cliff_point - 10%
  Long requests (prompt > 1024 tokens):   admit up to cliff_point - 15%
  WHY:  long requests consume disproportionate KV blocks and trigger longer prefill
        operations that degrade co-scheduled short requests; the asymmetric safety
        margins reflect asymmetric risk to system stability

Chunked prefill gate:
  If incoming request prompt_tokens > max_num_batched_tokens,
  check current short-request queue depth before admitting.
  If short-request queue depth > [threshold], defer or reject the long request.
  WHY:  prevents a single large request from starving pending short requests;
        your Day 23 data quantified this penalty at [X]× p99 TTFT

Explicit tradeoff:
  We are sacrificing: peak throughput (tokens/sec)
  In exchange for:   stable p99 latency and system stability
  Cost:              ~[X]% throughput reduction operating at cliff - margin vs. cliff

What this does NOT fix:
  - Does not eliminate preemption under bursty long-context load — only reduces frequency
  - Does not address retry amplification (Week 6 problem)
  - Requires Prometheus scrape interval < your TTFT SLO to be responsive
  - Does not prevent compound failure if all three conditions hit simultaneously —
    only raises the threshold at which that compound failure is triggered
```

Fill in every `[X]` with a real value from your data before finalizing.

### Step 5: Week 6 Retry Client Implementation (1 hr)

**Implementation only — not reading, not design.** Build it now so Day 26 starts with a working client.

```python
import time
import uuid

TIMEOUT_SECS = 2.0
MAX_RETRIES = 3
# No backoff — observe the unmitigated storm first.
# Backoff is a mitigation added in Day 27 to measure the delta.

attempt_log = []  # (request_id, attempt_number, outcome, latency_ms)

def send_with_retry(client, payload):
    request_id = str(uuid.uuid4())
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.post("/v1/completions", json=payload, timeout=TIMEOUT_SECS)
            latency_ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                attempt_log.append((request_id, attempt, "success", latency_ms))
                return resp
            else:
                attempt_log.append((request_id, attempt, f"http_{resp.status_code}", latency_ms))
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            attempt_log.append((request_id, attempt, "timeout_or_error", latency_ms))
    return None

def amplification_factor():
    total_attempts = len(attempt_log)
    unique_requests = len(set(r[0] for r in attempt_log))
    return total_attempts / unique_requests if unique_requests > 0 else 1.0
```

**Smoke test before Day 26:**
- Run at 20% of your admission limit (well below cliff)
- Confirm: amplification factor ≈ 1.0, no timeouts, all attempts logged as "success"
- If amplification > 1.05 at low load: timeout is misconfigured or vLLM is unresponsive — fix before Day 26

---

## End-of-Day Outputs

| Output | Interview Signal |
|---|---|
| D#5, D#6, D#7 cross-referenced and finalized | Defensible under "walk me through your data" questioning |
| Connection narrative with unification frame | "What class of problem is this?" — answered at the right level of abstraction |
| Failure interaction table | "What happens when multiple things go wrong at once?" — most candidates fail here |
| Admission control retrofit (numbered, with hard rejection threshold) | Decisive policy with data backing; directly quotable in D#9 and D#11 |
| Retry client implemented and smoke-tested | Day 26 starts with working tooling |

---

## This Day Directly Answers These Interview Questions

- "Why does latency suddenly spike at high utilization?"
- "Why not run at 90% utilization?" — one-line answer backed by your cliff graph
- "How do different workloads interact under load?"
- "What would you change in your Phase A admission control design?"
- "What happens when multiple failure modes hit simultaneously?"
- "What did you give up to get stable p99 latency?" — answered with a number, not a shrug

---

## Correction Table

| Version | Change | Rationale |
|---|---|---|
| v1 | Initial version | — |
| v2 | Removed standalone completeness-audit step; folded into cross-reference check | Checklist review is low signal; cross-reference consistency check is high signal |
| v2 | Upgraded admission control retrofit from "one paragraph" to numbered decision list | One paragraph is not interview-defensible |
| v2 | Cut Week 6 prep from 2 hrs to 1 hr; removed "review retry storm scenario" reading time | Implementation + smoke test is the signal |
| v2 | Made re-runs explicitly conditional on noise found in Step 1 | Unconditional re-runs are busy work if data is clean |
| v3 | Added unification frame: "memory scheduling under contention" | Frames all three findings as one class of problem; moves from benchmark-runner to systems thinker |
| v3 | Added failure interaction table with compound failure row | Forces causal cross-effect reasoning; directly answers "what happens when multiple things go wrong?" |
| v3 | Added hard rejection threshold to admission control retrofit | Implied in v2; must be explicit — decisive policy at the cliff is staff-level signal |
| v3 | Added explicit latency vs. throughput tradeoff statement with cost quantification | Leadership question "why not 85%?" requires a one-line answer backed by a number, not a philosophy |
| v3 | Added compound failure crystallization sentence to narrative | Closed-loop instability description was present but not crystallized into a single quotable sentence |
| v3 | Added interview question mapping section | Connects portfolio artifact to interview usage; prevents "portfolio-only" syndrome |

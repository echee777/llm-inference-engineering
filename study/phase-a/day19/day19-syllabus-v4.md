# Day 19 — Adversarial Testing + Design Note

**Theme:** Put your admission control system under adversarial pressure, then synthesize everything into the final portfolio deliverable — the Admission Control Design Note.

---

## Morning Block (4 hrs) — Adversarial Experiments

Three experiments in sequence. Each generates data you'll pull directly into the afternoon's Design Note.

---

### Experiment 1: Adversarial Request Simulation (1.5 hrs)

**Setup:**
- Tenant A fires 5 short requests concurrently: `prompt_tokens=100, max_tokens=100` each → ~500 tokens total budget draw
- Tenant B fires 1 massive request: `prompt_tokens=8192, max_tokens=4096` → ~12,288 tokens budget draw

**What to observe:**
- Does Tenant B's single request consume most of your token budget, leaving Tenant A queued or rejected?
- On a T4 with Qwen2.5-3B, your total KV budget is roughly 8–12K tokens (depending on your Day 16/17 derivation). A 12K token request alone can saturate it entirely.
- Record: per-tenant TTFT, budget % at time of each request, rejection decisions

**Expected result:** Tenant A's short requests get starved or rejected because the single large request monopolizes the budget. This is the correct failure mode to observe — it's the motivation for per-tenant budget caps in Phase C.

**What to write down:** "A single 12K-token request from Tenant B consumed X% of budget, causing Y% of Tenant A's requests to be rejected/queued for Z seconds."

**Arrival-order extension (~20 min):** Repeat with 3 orderings to show that FIFO amplifies unfairness:

| Case | Order | Outcome |
|---|---|---|
| A | Large request first | |
| B | Small requests first | |
| C | Interleaved (alternating) | |

Record starvation outcome (rejections, queue wait) for each case. Expected finding: Case A is worst for Tenant A; Case B allows small requests to drain budget before the large request is admitted; Case C is intermediate. This is your empirical argument that arrival order is a policy lever, not an accident. This demonstrates that token budget alone does not enforce fairness across tenants — the budget controls collapse, but not the distribution of capacity.

---

### Experiment 2: Head-of-Line (HOL) Blocking (1 hr)

**Setup:**
1. Fill the queue to near capacity with pending requests
2. Submit a massive request (8K prompt tokens) as the **first** item in the FIFO queue
3. Immediately behind it, submit 10 short requests (100 tokens each)

**What to observe:**
- Do the 10 short requests wait for the large request to clear admission, even though they'd each consume <1% of budget?
- FIFO forces them to queue behind the 8K request regardless of their size
- Record: median wait time for short requests vs. the long request's admission latency

**Why this matters for interviews:** This is a precise articulation of why FIFO is insufficient for inference. The fix — weighted fair queuing, shortest-job-first, or priority lanes — is what Phase C implements. You can now say: "I observed X seconds of unnecessary wait for 100-token requests blocked behind an 8K request. That's the empirical case for priority queuing." This is equivalent to the convoy effect in OS scheduling: a slow process at the head of a FIFO queue forces all faster processes behind it to wait, regardless of their own resource needs.

---

### Experiment 3: Queue Depth → Latency Curves (1.5 hrs)

**Setup:** Drive traffic at 25 / 40 / 50 / 60 / 70 / 80 / 90% of your token budget ceiling, using Locust from Day 18.

At each load level:
1. Let it stabilize for **3 minutes** before recording
2. Capture: token budget utilization %, queue depth, TTFT p50 / p95 / p99, `num_waiting_seqs` (from vLLM V1 metrics)

**Expected shape:** Flat TTFT up to ~60–70% utilization, then sharp upturn (the hockey stick). The p99 will diverge from p50 well before p50 starts rising — that's the early warning signal your dashboard should expose. `num_waiting_seqs` connects the latency spike to scheduler state: when it climbs, the scheduler has more sequences queued than it can batch in one iteration.

**Record as a table:**

| Budget % | Queue Depth | num_waiting_seqs | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---|---|---|---|---|
| 25% | | | | | |
| 40% | | | | | |
| 50% | | | | | |
| 60% | | | | | |
| 70% | | | | | |
| 80% | | | | | |
| 90% | | | | | |

This table and its plot become **Section 5** of the Design Note.

---

## Afternoon Block (4 hrs) — Write Admission Control Design Note

This is **Portfolio Deliverable #4**. It's a complete engineering design document — the kind you'd author at a Staff level to justify an architectural decision to a skeptical peer or interview panel.

Eight sections. Here's what to put in each:

---

**Section 1: Admission as KV Memory Budget**

The derivation chain:
```
GPU VRAM (T4: 16GB)
  - model weights (Qwen2.5-3B FP16: ~6.5GB)
  - activations + runtime overhead (~1GB)
  = KV cache budget (~8.5GB)

KV cache budget → blocks → tokens per block → total token capacity
(Use your GQA-correct math: num_kv_heads=2, not 16)
```

Lead with the thesis: *"Admission is not a concurrency cap. It is a memory budget, expressed in tokens."*

> **Fragmentation floor:** KV cache allocates in fixed-size blocks (default 16 tokens/block), not continuous token increments. A request consuming 17 tokens occupies 2 blocks. This means effective token capacity is slightly less than the theoretical budget — consistent with your Day 3 fragmentation simulation. The gateway token budget is therefore a conservative upper bound, not an exact capacity ceiling.

---

**Section 2: Why Token-Aware Beats Request-Count**

Use your data to make the case. A flat concurrency cap of N either:
- Wastes capacity when requests are short (N slots occupied by tiny requests)
- Causes OOM when requests are long (N slots occupied by 8K requests)

The token budget adapts naturally to both cases.

---

**Section 3: Token Budget Correction**

From Day 17: you budgeted conservatively at `max_tokens` (the worst case), which held budget even after completions finished early. Periodic release corrected this.

Write: what was the measured improvement in admitted request rate before vs. after correction? (Even an approximate number is fine — "~X% more throughput at equivalent TTFT.")

---

**Section 4: Adversarial Resilience**

Pull directly from your morning experiments:
- Large-request starvation: your numbers from Experiment 1
- HOL blocking: your wait-time data from Experiment 2
- Why FIFO is insufficient: the explanation
- Why per-tenant budgets + priority queuing are the fix (Phase C preview)

This section is where the morning's work pays off. Don't editorialize — just state what you observed and what it implies.

**Limitations paragraph (required — admission control is necessary but not sufficient):**

The gateway token budget is an approximation of actual KV usage, not a precise measurement. Three known gaps: (1) the budget is charged at `prompt_tokens + max_tokens` on admission, but actual KV growth is gradual during decode — the real memory footprint at any instant is lower than the charged budget; (2) prefill is a bursty, allocation-heavy operation while decode holds KV blocks for the full completion duration — the budget model treats both identically and cannot detect contention between co-running prefill and decode sequences; (3) vLLM's scheduler operates at iteration granularity and can produce TTFT variance even within "safe" admission bounds, because it controls batch composition, preemption, and decode prioritization independently of the gateway. Additionally, vLLM V1 uses recompute-only preemption (no swap path) — recompute cost scales with prompt length, so preemption disproportionately penalizes long-context requests under memory pressure, introducing an implicit fairness bias against them even when admission was nominally equal. Admission control prevents collapse — it does not guarantee efficiency or fairness. Full control requires scheduler-level integration (Phase C).

---

**Section 5: Queue Depth → Latency Curves**

Your hockey stick table/plot from Experiment 3. Annotate: where does p99 first diverge from p50? That's your practical admission threshold recommendation. Close the loop with an explicit operating point: "Based on observed p99 divergence at ~[your measured threshold]%, we target admission budget ≤ [threshold - 5–10]% as the stable operating point to maintain latency SLO with headroom for burst."

---

**Section 6: With vs. Without Admission Control**

Pull from Day 18's Locust load test. Two curves on the same axes: TTFT vs. RPS with admission enabled vs. disabled. The "without" line should show early TTFT collapse; the "with" line should hold flat until the budget limit.

**GPU efficiency note (add one paragraph):** The comparison is not purely about latency stability. Overly conservative admission also imposes a throughput cost: fewer admitted sequences per time window means smaller batch sizes per scheduler iteration, which reduces arithmetic intensity and moves the workload deeper into the memory-bandwidth-bound regime (Day 2 roofline). The goal is not to minimize admission — it is to hold the system at the highest token budget utilization that keeps p99 TTFT within SLO. Your hockey-stick curve (Section 5) gives the empirical answer for where that point is on your T4 / Qwen2.5-3B setup.

---

**Section 7: Dashboard**

Screenshot(s) from your Grafana dashboard. For each panel, one sentence: what does this metric tell an operator and what action does it trigger?

---

**Section 8: Architecture Diagram**

A clean diagram showing the request flow:

```
Client → Gateway (FastAPI)
           ├─ Token Budget Check → [admit / queue / reject]
           ├─ Per-API-Key Rate Limiter
           ├─ Bounded FIFO Queue
           └─ Proxy → vLLM V1 Engine
                           └─ KV Cache (PagedAttention blocks)
```

Label the trust boundary (Day 17's proxy trust analysis). Show where the budget counter lives and how periodic release feeds back into it.

---

> **Staff-level insight:**
> Admission control is a *coarse-grained approximation* of a fine-grained scheduling problem.
> It is necessary to prevent collapse, but optimal performance and fairness require integrating admission with the scheduler itself — e.g., token-aware scheduling, per-tenant quotas, or priority-based iteration scheduling.

---

## End-of-Day Checklist

- [ ] Experiment 1 data: starvation numbers recorded (budget %, TTFT deltas, rejection count)
- [ ] Experiment 1 arrival-order table: all 3 cases (large-first, small-first, interleaved) recorded
- [ ] Experiment 2 data: HOL blocking wait times for short vs. long requests
- [ ] Experiment 3 data: hockey stick table complete with p50/p95/p99 + num_waiting_seqs at all 7 load levels
- [ ] Design Note: all 8 sections drafted
- [ ] Section 1 KV math uses `num_kv_heads=2` (GQA-correct)
- [ ] Section 1 includes fragmentation floor caveat
- [ ] Section 4 cites actual measured numbers, not hypotheticals
- [ ] Section 4 includes Limitations paragraph (approximation gaps, prefill/decode contention, scheduler independence)
- [ ] Architecture diagram shows trust boundary

---

## v3 → v4 Correction Table

| Location | Change | Source | Decision |
|---|---|---|---|
| After Section 8 | Added Staff Insight Box: admission is a coarse-grained approximation of a fine-grained scheduling problem; optimal performance and fairness require scheduler integration | Reviewer final 1% upgrade | Accept |

| Location | Change | Source | Decision |
|---|---|---|---|
| Experiment 2 (HOL) | Added convoy effect analogy to OS scheduling | Reviewer optional suggestion | Accept |
| Experiment 1 arrival-order extension | Added "token budget alone does not enforce fairness" sentence | Reviewer optional suggestion | Accept |
| Section 4 Limitations | Added preemption cost asymmetry: recompute scales with prompt length → implicit bias against long-context requests under memory pressure; V1 recompute-only path noted | Reviewer Upgrade 2 | Accept |
| Section 5 | Added explicit safe operating point recommendation: set admission target to observed p99 divergence threshold minus headroom | Reviewer Upgrade 3 | Accept |
| Section 6 | Added GPU efficiency paragraph: conservative admission → smaller batches → lower arithmetic intensity → throughput penalty; cites Day 2 roofline | Reviewer Upgrade 1 | Accept |

| Location | Change | Source | Decision |
|---|---|---|---|
| Section 1 derivation | Added fragmentation floor caveat: block-granular allocation means effective capacity < theoretical budget | Reviewer Upgrade 2 | Accept |
| Section 4 | Added Limitations paragraph: budget approximation gaps, prefill/decode contention, scheduler independence | Reviewer Upgrades 1 & 3 (scoped down) | Partial Accept — absorbed into limitations paragraph, not a new section |
| Experiment 1 | Extended to 3 arrival-order cases (large-first, small-first, interleaved) | Reviewer Exp 1 suggestion | Accept |
| Experiment 3 table | Added `num_waiting_seqs` column | Reviewer Exp 3 suggestion | Accept |
| Reviewer Upgrade 4 ("what would you change in vLLM?") | Not added | Out of scope | Reject — deferred to Day 20 exit Q10 and Phase B; requires scheduler modification data to avoid speculation |

---

## Looking Ahead

Tomorrow is **Day 20 — Phase A Exit**. You'll polish deliverables and write the 10-question exit self-assessment covering everything from GPU architecture through admission control. The Design Note you write today is the anchor for exit question #8 ("admission control as memory budget") and #9 ("adversarial requests").

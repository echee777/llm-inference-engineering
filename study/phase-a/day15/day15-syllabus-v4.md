# Day 15 — Week 3 Deliverable: Quantization & Optimization Tradeoff Analysis (v4)

> **Revision note — v4:** Final patch release. Three nit-level edits from third external LLM review.
> All three accepted. No structural changes from v3.
> Nit 1: hedged INT8 Tensor Core claim in §2 required sentence (inferred from throughput, not profiled).
> Nit 2: softened "compute-bound on all hardware" to "compute-dominated under these workload parameters."
> Nit 3: added explicit SLO-naming requirement to §8 recommendations.
> See correction table at end of document.
>
> **Revision note — v2:** Incorporated first external LLM review.
> Accepted: explicit non-recommendations, generalizability limits section, fleet-type differentiation.
> Partially accepted: "what I would ship" merged into Section 8 (not a standalone section).
> Rejected: four-subsection bolt-on structure (bloats the memo; staff-level docs should be tight).

---

## Context

Day 15 is the Week 3 capstone. No new experiments — all data should exist from Days 11–14.
This is a **4-hour writing block + 4-hour buffer/self-test**. The deliverable is Portfolio Artifact #3.

The document you produce today should read like something a staff inference engineer would send
to a team lead before a model serving decision — not a school report, not a literature review.
Tight, data-backed, with explicit calls and explicit rejections.

---

## Morning Block (4 hrs) — Write the Tradeoff Analysis

### What you are writing

A **9-section recommendation memo** (~2,000–3,000 words + tables). Every claim cites your data.
No hedging in the recommendations. Staff engineers make calls.

---

### Section 1 — Approaches (~0.5 pages)

Cover AWQ, GPTQ, and FP8 KV cache briefly. One paragraph each. Orient the reader — do not
recap the papers.

**AWQ:** Activation-aware weight quantization. Not all weights affect output equally — salient
weights are preserved at higher precision. This reduces quantization error relative to uniform
weight quantization that treats all weights as equally important, which is why AWQ degrades
less at the same bit width.

**GPTQ:** Layer-by-layer post-training quantization with calibration data. Designed for INT4.
The calibration step is what makes it accurate enough to be viable at 4 bits.

**FP8 KV cache:** Distinct from weight quantization. Compresses the KV cache entries themselves,
not model weights. Reduces KV memory footprint by ~50% vs FP16 KV. Hardware-gated: requires
Hopper (H100) or later. If on T4: document theoretical benefit, do not fabricate measured numbers.

> **Framing note:** These three techniques operate at different layers of the stack. AWQ/GPTQ
> target weight memory (model size). FP8 KV targets activation memory (KV cache size). They
> compound — you can run AWQ weights with FP8 KV cache simultaneously on supported hardware.

---

### Section 2 — Performance by Workload Type

This is the core empirical section. Present your full benchmark table.

| Precision | Workload | Conc | Throughput (tok/s) | TTFT p99 (ms) | ITL p99 (ms) |
|-----------|----------|------|-------------------|---------------|--------------|
| FP16 | Prefill-heavy | 1 | | | |
| FP16 | Prefill-heavy | 4 | | | |
| FP16 | Prefill-heavy | 8 | | | |
| FP16 | Prefill-heavy | 16 | | | |
| FP16 | Decode-heavy | 1 | | | |
| FP16 | Decode-heavy | 4 | | | |
| FP16 | Decode-heavy | 8 | | | |
| FP16 | Decode-heavy | 16 | | | |
| INT8-AWQ | Prefill-heavy | 1 | | | |
| INT8-AWQ | Prefill-heavy | 4 | | | |
| INT8-AWQ | Prefill-heavy | 8 | | | |
| INT8-AWQ | Prefill-heavy | 16 | | | |
| INT8-AWQ | Decode-heavy | 1 | | | |
| INT8-AWQ | Decode-heavy | 4 | | | |
| INT8-AWQ | Decode-heavy | 8 | | | |
| INT8-AWQ | Decode-heavy | 16 | | | |
| INT4-GPTQ | Prefill-heavy | 1 | | | |
| INT4-GPTQ | Prefill-heavy | 4 | | | |
| INT4-GPTQ | Prefill-heavy | 8 | | | |
| INT4-GPTQ | Prefill-heavy | 16 | | | |
| INT4-GPTQ | Decode-heavy | 1 | | | |
| INT4-GPTQ | Decode-heavy | 4 | | | |
| INT4-GPTQ | Decode-heavy | 8 | | | |
| INT4-GPTQ | Decode-heavy | 16 | | | |

**Required analytical sentence — write this explicitly with your numbers:**

> "INT8-AWQ improves decode throughput by X% over FP16 but prefill throughput by only Y%,
> because decode is memory-bandwidth-bound (smaller weight tensors = faster HBM reads per
> token) while prefill is compute-dominated under these workload parameters (matmul throughput
> does not improve proportionally with INT8 quantization on T4 — consistent with limited INT8
> Tensor Core utilization on this hardware, though this is inferred from the throughput
> asymmetry rather than directly profiled at the kernel level)."

If you cannot write this sentence with real numbers, your Day 11 data is incomplete. Do not
move to Section 3 until this sentence is written.

---

### Section 3 — Quality

| Metric | FP16 | INT8-AWQ | INT4-GPTQ |
|--------|------|----------|-----------|
| Perplexity (WikiText-2) ↓ better | | | |
| Downstream task score (name task) ↑ better | | | |
| Qualitative degradation observed? | No | ? | ? |

**Qualitative error analysis:** List any prompt categories where INT4 visibly degraded vs FP16.
Examples to check: multi-step reasoning, code generation, long-context summarization.
One concrete example per failure mode is sufficient — do not pad.

**Analytical framing to include:**

> "The perplexity gap between FP16 and INT8-AWQ is X points; between INT8-AWQ and INT4-GPTQ
> is Y points. Qualitatively, INT4 degraded on [task types], consistent with the expectation
> that lower-precision representations struggle more on tasks requiring exact numerical reasoning
> or long dependency chains."

---

### Section 4 — Capacity Impact

The staff-level insight: quantization's primary production value is often KV cache capacity,
not per-request speedup. Smaller weights → more free VRAM → more concurrent requests.

| Precision | Model Size (GB) | Free for KV (GB) | Max Conc @ 2K seq | Max Conc @ 4K seq | Max Conc @ 8K seq |
|-----------|-----------------|------------------|--------------------|--------------------|--------------------|
| FP16 | | | | | |
| INT8-AWQ | | | | | |
| INT4-GPTQ | | | | | |

**Required analytical sentence:**

> "INT8-AWQ reduces model weight size by X GB, freeing Y GB for KV cache. At 4K sequence
> length, this raises max concurrent requests from A to B — a Z% capacity improvement.
> For throughput-maximizing deployments, this capacity gain is more operationally significant
> than the per-request latency reduction."

---

### Section 5 — Cost Model ($/Million Tokens)

Formula: `$/M tokens = (GPU hourly rate) / (throughput_tok_s × 3600) × 1,000,000`

Use your actual g4dn.xlarge rate. Use concurrency=8 as the reference operating point.

| Precision | Throughput @ conc=8 (tok/s) | Tokens/hr | $/M tokens |
|-----------|----------------------------|-----------|------------|
| FP16 | | | |
| INT8-AWQ | | | |
| INT4-GPTQ | | | |

**Required framing:**

> "INT8-AWQ reduces serving cost from $X/M tokens to $Y/M tokens — a Z% reduction — with a
> ΔPPl perplexity increase of W points. INT4-GPTQ reduces cost further to $V/M tokens, but
> with qualitative degradation on [task types] that makes it unsuitable for quality-critical
> workloads."

---

### Section 6 — Prefix Caching

From Day 13 experiments:

| System Prompt Length (tokens) | TTFT without cache (ms) | TTFT with cache (ms) | Reduction (%) |
|-------------------------------|------------------------|-----------------------|---------------|
| 100 | | | |
| 500 | | | |
| 1000 | | | |
| 2000 | | | |

**When it matters:** Prefix caching pays off when the shared-prefix ratio is high (many requests
sharing the same system prompt). For a deployment where every request has a unique system prompt,
prefix caching provides no benefit. State the threshold from your data — at what prompt length
does the TTFT reduction become operationally significant?

**When it doesn't:** Single-turn requests with no shared prefix, or deployments where system
prompt variety is high (e.g., per-user custom prompts with low reuse).

---

### Section 7 — Speculative Decoding

From Day 14 experiments:

| Scenario | Draft Tokens | Acceptance Rate | Latency vs Baseline | Use Spec Decode? |
|----------|-------------|-----------------|---------------------|------------------|
| Simple Q&A, conc=1 | | | | |
| Simple Q&A, conc=8 | | | | |
| Creative writing, conc=1 | | | | |
| Code generation, conc=4 | | | | |

**Key analytical framing to include:**

> "Speculative decoding improved throughput by X% at concurrency=1 on [task type]. At
> concurrency=8, it degraded throughput by Y% because draft model execution consumed
> GPU compute that would otherwise serve real requests. Spec decode is a low-concurrency,
> predictable-output optimization — it is not a general-purpose throughput lever."

---

### Section 8 — Recommendations (expanded from v1)

**Do not hedge. Make calls. Cite your section numbers.**

Every recommendation must name its target SLO. A recommendation without a named SLO is not
a recommendation — it is a preference. The four axes are: **latency** (p99 TTFT / ITL),
**throughput** (tokens/sec, requests/sec), **cost** ($/M tokens), and **quality floor**
(perplexity ceiling, task score minimum). State which axis you are optimizing and which
you are treating as a constraint.

#### By use case

| Use Case | Recommendation | Key Rationale |
|----------|---------------|---------------|
| Latency-sensitive chat | INT8-AWQ + prefix caching | Capacity gain (§4) without quality hit; prefix caching cuts TTFT on repeat system prompts (§6) |
| Throughput-maximizing batch | INT4-GPTQ | Lowest $/M tokens (§5), highest concurrency ceiling (§4); acceptable quality for non-critical tasks |
| Quality-critical (code, medical, legal reasoning) | FP16 | INT4 qualitative degradation on [tasks] is unacceptable; INT8 is a judgment call depending on task |

#### Operational simplicity as a decision criterion

Cost, capacity, and quality are not the only axes. Frontier labs also weight **rollout risk,
validation burden, and operational complexity** — and so should this memo.

- **INT8-AWQ is preferred over INT4-GPTQ** not just on quality grounds, but because it
  requires less validation surface: fewer edge cases to catch, lower risk of silent quality
  regressions in production, simpler rollback story.
- **Prefix caching is high-upside, low-risk.** It adds no new model variant, no calibration
  step, and is trivially reversible. For any deployment with repeated system prompts, enable
  it by default.
- **Speculative decoding is operationally trickier than its headline speedup suggests.** It
  adds a draft model dependency, makes latency behavior harder to predict under load, and
  complicates capacity planning. Its acceptance rate is workload-sensitive, which means its
  speedup is not stable across traffic changes. The operational overhead is often not worth
  it unless the latency win is critical and the workload is predictable.

Prefer the simplest deployment that meets the SLO. Add complexity only when the SLO
requires it and the data supports it.

#### By fleet type

This is the production realism layer. The answer is not "one precision for everything."

**T4-class cost-constrained fleet (your hardware):**
INT8-AWQ is the default. It provides meaningful capacity expansion (§4) at acceptable quality
cost (§3), and the $/M tokens improvement (§5) is significant on expensive on-demand instances.
INT4-GPTQ only if the workload is clearly non-quality-critical and throughput is the primary SLO.
Prefix caching enabled by default for any deployment with repeated system prompts.
Spec decode off — concurrency is typically low enough to matter, but T4 compute limits make
draft model overhead expensive.

**H100-class latency-sensitive fleet (extrapolated, not measured on T4):**
FP16 or BF16 is the baseline comparison point on H100-class fleets. Higher HBM bandwidth
reduces the relative urgency of weight quantization for decode throughput versus T4, but
quantization can still matter for capacity, cost efficiency, and larger-model serving —
it does not become irrelevant. FP8 KV cache becomes the primary new capacity lever (§9 —
theoretical). Spec decode viable at low concurrency with appropriate draft model sizing.
Prefix caching still valuable for shared-prompt workloads.

#### Explicit non-recommendations

These are as important as the recommendations.

- **Do not deploy INT4-GPTQ for quality-critical workloads** (code generation, medical
  summarization, multi-step reasoning). The perplexity cost (§3) is acceptable in aggregate;
  the qualitative degradation on specific task types is not. Cost savings do not justify
  shipping degraded outputs in high-stakes contexts.

- **Do not deploy speculative decoding at high concurrency** (conc ≥ 8 in your experiments).
  Draft model compute competes with live request serving. At the concurrency levels where
  throughput matters most, spec decode hurts. Turn it on only for latency-critical,
  low-concurrency deployments with predictable output distributions.

- **Do not treat prefix caching wins as universal.** The TTFT reduction is real only when
  shared-prefix ratios are high. For deployments with high system-prompt diversity, the
  cache hit rate approaches zero and the overhead of maintaining the cache provides no benefit.
  Measure your actual shared-prefix ratio before enabling.

- **Do not apply T4 throughput numbers directly to A100/H100 fleet sizing.** See Section 9.

#### What I would ship on Monday morning

Given a T4-class fleet, general-purpose chat workload, moderate quality requirements:

**INT8-AWQ + prefix caching enabled + spec decode off.**

Rationale: INT8-AWQ gives X% more concurrent capacity than FP16 (§4) at Y% lower $/M tokens (§5)
with Z point perplexity increase (§3) — quality cost is within acceptable bounds for
general-purpose chat. Prefix caching cuts TTFT on system-prompt-heavy traffic by W% (§6).
Spec decode stays off until concurrency drops below [threshold] or a dedicated low-latency
tier is carved out.

This is the call I can defend with data. It is not the call for every workload.

---

### Section 9 — Generalizability Limits [NEW in v2]

> This section is required because the T4 is not a production frontier-lab GPU. Interviewers
> will ask: "your numbers are from a T4 — what changes on H100?" You need a prepared answer.

| Finding | Status | Likely change on A100/H100 |
|---------|--------|---------------------------|
| INT8-AWQ decode throughput gain | **Measured on T4** | **Expected but not measured:** relative decode gain may shrink on A100/H100 because higher memory bandwidth reduces the pressure that makes quantization especially valuable on T4. Direction plausible; magnitude unconfirmed. |
| INT8 prefill gain is smaller than decode gain | **Measured on T4** | Pattern holds — prefill is compute-dominated under these workload parameters on all tested hardware; INT8 Tensor Core utilization improves more on A100/H100 but prefill remains compute-dominated relative to decode |
| Capacity expansion from INT8 (KV VRAM freed) | **Measured on T4** | Math holds exactly — weight size reduction is hardware-independent. KV capacity gain is deterministic. |
| $/M tokens improvement from INT8 | **Measured on T4** | Direction holds; magnitude changes because A100/H100 on-demand rates differ and throughput at each precision differs |
| FP8 KV cache capacity benefit (~2x concurrent requests) | **Theoretical — not measured** | Would be measurable on H100. Estimated from: 50% KV memory reduction → ~2x concurrency ceiling at fixed sequence length |
| Prefix caching TTFT reduction | **Measured on T4** | Pattern holds on all hardware; absolute ms values will differ |
| Spec decode hurts at high concurrency | **Measured on T4** | Pattern holds — but "high concurrency" threshold shifts on more powerful hardware. The mechanism (draft model compute competes with live serving) is hardware-independent. |
| Qualitative INT4 degradation on reasoning tasks | **Measured on T4** | Hardware-independent — quantization error is a property of the weight representation, not the GPU executing it |

**What I would measure first on H100:**
1. FP8 KV cache: does it actually deliver the theoretical 2x concurrency gain?
2. INT8 decode gain: is it smaller on H100 as expected from the bandwidth argument?
3. Spec decode viability: at what concurrency does it flip from benefit to harm on H100?

---

## Afternoon Block (4 hrs) — Buffer + Self-Test

### Buffer (use as needed)

Finish any section that's incomplete. Every table should have numbers. Every analytical
sentence should have X%, Y points, A→B format — not "improved significantly."

Ensure cross-references are correct: Section 8 should cite §3, §4, §5, §6 by number.

### Self-Test — 6 questions, close your notes

You must answer these with your own numbers. Vague answers fail.

1. **What throughput improvement does INT8 give for decode-heavy vs. prefill-heavy workloads?
   Why the difference at the hardware level?**

2. **What's the perplexity cost of INT4? Did you observe qualitative degradation?
   On which task types?**

3. **How many additional concurrent 4K-token requests can you serve with INT8 vs. FP16?
   Exact number from your capacity table.**

4. **What's your $/million tokens at FP16 vs. INT8 on g4dn.xlarge?**

5. **When does speculative decoding help vs. hurt? What is the mechanism for the hurt case?**

6. **[v2] An interviewer says: "Your T4 numbers show INT8-AWQ is worth it. Why should I
   believe that holds on our H100 fleet?"**

   Your answer must cover three things:
   - What stays the same (capacity math, qualitative degradation, prefill/decode asymmetry
     direction)
   - What changes (absolute throughput gains, FP8 KV availability, spec decode thresholds)
   - What you'd measure first if handed an H100 (from §9)

   **[v3 addition] Follow-up: "What result on H100 would falsify your recommendation?"**

   You must name a specific outcome that would cause you to change the call — e.g., "If
   INT8-AWQ decode gain on H100 is < Z%, the capacity argument from §4 may still hold but
   the throughput justification weakens; I'd re-evaluate whether FP16 + FP8 KV cache is
   the better path." If you cannot name a falsification condition, you are extrapolating,
   not reasoning. Revise §9 until you can.

   If you can't answer this cold, your Section 9 is not tight enough. Revise before moving on.

---

## End-of-Day Output

✅ **Finalized Quantization & Optimization Tradeoff Analysis** — Portfolio Deliverable #3

Nine sections. All tables populated with real numbers. Explicit non-recommendations in §8.
Generalizability limits in §9. One concrete "what I would ship" call in §8.

---

## Correction Table (v1 → v2 → v3 → v4)

| # | Change | Source | Rationale |
|---|--------|--------|-----------|
| v1→v2 | Expanded §8 with fleet-type differentiation (T4-class vs H100-class) | Reviewer 1 (accepted) | Production reality: one precision for everything is wrong; fleet type changes the answer |
| v1→v2 | Added explicit non-recommendations to §8 | Reviewer 1 (accepted) | Rejecting bad defaults is a staff-level signal; interviewers test this |
| v1→v2 | Merged "what I would ship" into §8 conclusion | Reviewer 1 (partially accepted) | Concept valid; standalone section would bloat memo. Merged instead. |
| v1→v2 | Added §9 — Generalizability Limits | Reviewer 1 (accepted) | T4 results do not transfer directly to H100; epistemic discipline reads senior |
| v1→v2 | Added self-test Q6 (H100 extrapolation challenge) | Derived from §9 | Prepares for the obvious interviewer follow-up on hardware gap |
| v1→v2 | **Rejected** four-subsection bolt-on structure | Reviewer 1 (rejected) | Adds bloat; staff-level memos should be tight. Ideas integrated into existing sections instead. |
| v2→v3 | AWQ wording in §1: "naive INT8" → "uniform weight quantization that ignores activation salience" | Reviewer 2 (partially accepted) | Removes imprecision; reviewer's full jargon-dense phrasing avoided in favor of readable middle ground |
| v2→v3 | H100 fleet framing in §8 softened: removed implication that INT8 becomes broadly uninteresting on H100 | Reviewer 2 (accepted) | Original wording overstated the case; quantization still matters on H100 for capacity and cost |
| v2→v3 | §9 INT8 decode gain row relabeled "expected but not measured" | Reviewer 2 (accepted) | Was stated as likely conclusion; correctly labeled as inference from bandwidth argument |
| v2→v3 | Added operational simplicity dimension to §8 recommendations | Reviewer 2 (accepted) | Rollout risk, validation burden, and debugging surface are real decision factors at frontier labs |
| v2→v3 | Added falsification condition to self-test Q6 | Reviewer 2 (accepted) | Forces experimentalist thinking; extrapolation without a falsification condition is not reasoning |
| v3→v4 | §2 required sentence: hedged INT8 Tensor Core claim as inference from throughput asymmetry, not validated hardware fact | Reviewer 3 (accepted) | "Limited INT8 Tensor Core acceleration on T4" was stated as fact; it was inferred. Interview risk if probed at kernel level. |
| v3→v4 | §9 prefill row: "compute-bound on all hardware" → "compute-dominated under these workload parameters" | Reviewer 3 (accepted) | Absolute claim; softer phrasing is defensible without losing the point |
| v3→v4 | §8 added explicit SLO-naming requirement before recommendation table | Reviewer 3 (accepted) | Recommendations without named SLOs are preferences. Four axes stated: latency, throughput, cost, quality floor. |

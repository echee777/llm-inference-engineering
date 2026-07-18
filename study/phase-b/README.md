# Phase B: Failure modes and operating at scale (Days 21-29, plus a planned Day 30)

Phase B breaks the system on purpose and engineers the controls: failure-mode analysis,
the latency cliff, retry storms, and an autoscaling policy. Same hardware as Phase A: a
single NVIDIA T4 serving Qwen2.5-3B-Instruct through vLLM V1.

For the plain-English tour of every day see [../progress_summary.md](../progress_summary.md);
for full technical depth see [../phase_a_b_day1_29_summaries_v1.md](../phase_a_b_day1_29_summaries_v1.md).

Each day folder holds the syllabus, the scripts that ran the experiment, the raw results,
and the writeup.

## Days

- Day 21 — [KV cache exhaustion is bimodal](day21)
- Day 22 — [postmortem 1 and the GQA correction](day22)
- Day 23 — [prefill/decode interference is a tail problem](day23)
- Day 24 — [the latency cliff and the divergence ratio](day24)
- Day 25 — [consolidation and admission control retrofit](day25)
- Day 26 — [retry storm, derived on paper](day26)
- Day 27 — [postmortem 2 and the wrong-signal proof](day27)
- Day 28 — [three timescales and the queue-depth surprise](day28)
- Day 29 — [autoscaling memo and mistakes log](day29)
- Day 30 — [multi-GPU track (planned, syllabus only)](day30)

## Key deliverables

- [Prefill/decode interference](day23/deliverable-6-prefill-decode-interference.md)
- [The latency cliff](day24/deliverable-7-cliff.md)
- [Postmortem: KV exhaustion](day22/postmortem-1.md)
- [Postmortem: retry cascade](day27/deliverable-8-postmortem-retry-cascade.md)
- [Autoscaling policy](day29/deliverable_09_autoscaling_memo_v1.md)
- [Mistakes log](day29/mistakes_log.md) — every time a measurement contradicted an assumption

## Scope note

Days 1-29 are the portion that was fully executed and measured. Day 30 (multi-GPU tensor
parallelism on A10G, with an H100 contrast) has a detailed syllabus prepared but was not
executed.

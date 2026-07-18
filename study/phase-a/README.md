# Phase A: Fundamentals, vLLM internals, quantization, admission control (Days 1-20)

Phase A builds the fundamentals from the hardware up: the GPU memory model, vLLM V1
internals, quantization as a capacity lever, and a first admission-control gateway.
All work ran on a single NVIDIA T4 (g4dn.xlarge spot), serving Qwen2.5-3B-Instruct and
TinyLlama through vLLM (V0 early, V1 from Day 7).

For the plain-English tour of every day see [../progress_summary.md](../progress_summary.md);
for full technical depth see [../phase_a_b_day1_29_summaries_v1.md](../phase_a_b_day1_29_summaries_v1.md).

Each day folder (or file, for the early days) holds the syllabus, the scripts that ran
the experiment, the raw results, and the writeup.

## Days

- Day 1 — [GPU memory hierarchy and the bandwidth gap](day1-step4-results.md)
- Day 2 — [roofline: prefill compute-bound, decode memory-bound](day2-morning-artifacts.md)
- Day 3 — [KV cache as the binding constraint, why paging matters](day3-kv-cache-analysis.py)
- Day 4 — [interconnect topology and the TP-over-PCIe tax](day4-interconnect.py)
- Day 5 — [Week 1 deliverable and adversarial review](week1-deliverable-gpu-architecture-memory-budget.md)
- Day 6 — [vLLM parameter sensitivity and the prefill regression](day6-morning.md)
- Day 7 — [vLLM V1 request lifecycle and the unified scheduler](day7-work.md)
- Day 8 — [instrumenting the KV block manager](day8-work.md)
- Day 9 — [driving the system into collapse on purpose](day9-work.md)
- Day 10 — [Week 2 annotated architecture deliverable](day10-work.md)
- Day 11 — [quantization as a capacity lever](day11-work.md)
- Day 12 — [quality, capacity, and cost of quantization](day12-work.md)
- Day 13 — [prefix caching and the GQA correction](day13)
- Day 14 — [speculative decoding as a regime-dependent decision](day14)
- Day 15 — [quantization tradeoff memo](day15)
- Day 16 — [admission gateway and the cliff as divergence ratio](day16)
- Day 17 — [reconciliation and progressive budget release](day17)
- Day 18 — [load test where admission control made things worse](day18)
- Day 19 — [adversarial testing and the design note](day19)
- Day 20 — [Phase A capstone and interview prep](day20)

## Key deliverables

- [Week 1: GPU architecture and memory budget](week1-deliverable-gpu-architecture-memory-budget.md)
- [Week 2: annotated vLLM architecture](week2-deliverable-annotated-architecture.md)
- [Day 15: quantization tradeoff memo](day15/day15-tradeoff-analysis.md)
- [Day 19: admission control design note](day19/design-note.md)

## Reproducing

Dependencies are in [requirements.txt](requirements.txt). Most experiments are a single
script plus a runner in their day folder; the writeup in each folder states the exact
command and the expected numbers. Some day folders have a [Makefile](Makefile) target.

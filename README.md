# GPU Inference Engineering Residency

A self-directed, hands-on engineering deep dive into how LLM serving systems behave under load and
where they break. Every finding here came from running real experiments on a live GPU and measuring
the result, not from reading papers or coursework.

The work is documented day by day. If you are reviewing this as part of a hiring decision, the two
fastest entry points are:

- study/progress_summary.md — plain-English summary, one entry per day, each leading with the finding
- study/phase_a_b_day1_29_summaries_v1.md — the same days at full technical depth

## Why this exists

I am a senior systems engineer with a background in cloud and ML infrastructure. I run this kind of
residency to stay hands-on at the hardware level rather than only directing from above. The goal was
operational mastery of GPU-constrained inference: capacity modeling, failure modes, and the control
systems that keep a server on the safe side of collapse.

What it demonstrates to an employer:

- I still read framework source, profile kernels, and instrument internals, not just architect.
- I measure before I claim. Every number below is reproducible from the scripts in this repo.
- I think at the altitude the role needs (cost, capacity, reliability) while staying in the code.

## Setup

```
Hardware:  single NVIDIA T4 (g4dn.xlarge spot instance)
Models:    Qwen2.5-3B-Instruct, TinyLlama
Engine:    vLLM (V0 early on, V1 from Day 7)
Tooling:   PyTorch, Nsight Compute/Systems, nvidia-smi/dmon, torch.profiler,
           llm-compressor (AWQ/GPTQ), FastAPI, Locust, Prometheus/Grafana
```

## Headline findings

```
KV cache is the binding constraint    single-GPU concurrency is capped by memory, not FLOPS
Latency cliff at 87% KV utilization   p99 TTFT jumps 2.4x (3,200ms -> 7,587ms) over ~2% load change
Quantization is a capacity lever      INT8-AWQ: 1.70x decode, +0.17% perplexity, +40% concurrency
Retry storms self-amplify ~1.56x      a transient trigger drives a 2-3 minute cascade past the cliff
Reactive autoscaling is too slow      94.2s cold start vs seconds of headroom -> hot standby required
Prefill time is linear in prompt len  TTFT(ms) = 37.6 + 0.228 * prompt_tokens, R-squared 0.997
```

## How the repo is organized

Phase A (Days 1-20) builds the fundamentals: GPU memory model, vLLM internals, quantization, and a
first admission-control gateway. Phase B (Days 21-29) breaks the system on purpose and engineers the
controls: failure-mode analysis, the latency cliff, retry storms, and an autoscaling policy.

```
study/phase-a/   Days 1-20: fundamentals, vLLM internals, quantization, admission control
study/phase-b/   Days 21-29: failure modes, the cliff, retry storms, autoscaling
study/*.md       cross-phase summaries and primers
docs/            infra roadmap assessment (separate Kubernetes/KubeRay platform work)
```

Each day folder holds the syllabus, the scripts that ran the experiment, the raw results, and a
writeup. Raw server logs and profiler binary traces are gitignored as regenerable noise; the analysis
and the data tables they produced are kept.

## Worth reading first

- [Capacity and memory model](study/phase-a/week1-deliverable-gpu-architecture-memory-budget.md)
- [vLLM internals, annotated](study/phase-a/week2-deliverable-annotated-architecture.md)
- [Quantization tradeoff memo](study/phase-a/day15/day15-tradeoff-analysis.md)
- [Admission control design](study/phase-a/day19/design-note.md)
- [Prefill/decode interference](study/phase-b/day23/deliverable-6-prefill-decode-interference.md)
- [The latency cliff](study/phase-b/day24/deliverable-7-cliff.md)
- [Postmortem: KV exhaustion](study/phase-b/day22/postmortem-1.md)
- [Postmortem: retry cascade](study/phase-b/day27/deliverable-8-postmortem-retry-cascade.md)
- [Autoscaling policy](study/phase-b/day29/deliverable_09_autoscaling_memo_v1.md)

Two artifacts I would point to specifically:

- study/phase-a/day8-scripts/apply_block_instrumentation.py — a patch to vLLM V1's KV cache manager
  and scheduler that makes every block alloc/free/preempt event observable. This is the microscope the
  rest of the residency depends on.
- study/phase-a/day16/gateway.py (evolves through day19) — a FastAPI admission-control gateway that
  gates on a token-denominated KV memory budget rather than a flat concurrency cap.

A running log of every time a measurement contradicted my assumptions is kept in
study/phase-b/day29/mistakes_log.md, including an 8x KV-sizing error I caught and corrected.

## Methodology note

This residency was run AI-assisted: I used an LLM as a demanding instructor and pair (the prompts and
scaffolding live in study/.claude/) to structure the days and pressure-test my reasoning. The
experiments, the hardware runs, the instrumentation, and the measurements are mine. I am keeping the
scaffolding in the repo because the process is part of the work.

## Reproducing

Dependencies for the Phase A experiments are in study/phase-a/requirements.txt. Most experiments are a
single script plus a shell runner in their day folder; the writeup in each folder states the exact
command and the expected numbers.

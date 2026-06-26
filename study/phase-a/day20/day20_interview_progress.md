# Day 20 Interview Progress — AI Inference Platform Residency

## Context
This document captures the user's live mock interview progress for Day 20.
It is intended for continuation by another LLM or reviewer.

---

# Q1 — Request Lifecycle (vLLM V1)

### Answer Summary
- Request enters FastAPI API server
- API layer calls async client → engine.generate()
- Communication via ZeroMQ to separate engine process
- Engine runs scheduler loop (step-based execution)
- Scheduler batches requests across different stages (prefill + decode)
- GPU worker executes forward pass via model
- Logits returned → sampling applied → tokens generated
- Tokens streamed back via RequestOutput abstraction
- Tokenization at ingress, detokenization at egress

### Key Understanding
- Execution is **iteration-based**, not request-based
- Continuous batching mixes requests at different lifecycle stages

---

# Q2 — Request State Machine (vLLM V1)

### States
- WAITING
- RUNNING (prefill or decode)
- PREEMPTED (recompute-only)
- FINISHED

### Behavior
- WAITING → RUNNING when scheduled
- RUNNING → PREEMPTED when KV memory exhausted
- PREEMPTED → WAITING (recompute required)
- RUNNING → FINISHED when complete

### Key V1 Differences
- No SWAPPED state
- No CPU KV offload
- Preemption = discard KV + recompute
- Simpler model, higher recompute cost

---

# Q3 — PagedAttention

### Problem
- Naive KV allocation requires contiguous memory
- Leads to severe fragmentation (30–60% waste)

### Solution
- KV cache divided into pages (e.g., 16 tokens)
- Logical → physical mapping via block tables
- Enables non-contiguous allocation

### Insight
- Similar to virtual memory paging
- Enables efficient continuous batching with variable-length requests

---

# Q4 — Quantization Tradeoffs

### Core Insight
- Decode = memory-bandwidth bound
- Prefill = compute bound

### Effect of INT8
- Halves weight size → improves decode throughput significantly
- Limited effect on prefill (compute-bound)

### Capacity Impact
- Smaller weights → more VRAM for KV cache
- Higher concurrency

### Cost Impact
- Higher throughput → lower $/token

---

# Q5 — GPU Utilization

### Problem
- nvidia-smi GPU util = "any kernel active"
- Does NOT reflect actual utilization

### Better Metrics
- Memory throughput %
- Achieved occupancy
- SM active %

### Tools
- Nsight Compute / Nsight Systems

---

# Q6 — KV Capacity Derivation (First Principles)

### Steps
1. Total GPU memory × utilization (e.g., 0.9)
2. Subtract:
   - Model weights
   - Runtime overhead
3. Remaining = KV budget

### Per-token KV formula
2 × num_kv_heads × head_dim × bytes × num_layers

### Per-request
Multiply per-token KV by sequence length (e.g., 4K)

### Capacity
KV budget / per-request KV

### INT8 Effect
- Reduces weight footprint → increases KV capacity

---

# Q7 — Preemption & Latency

### Trigger
- KV memory exhaustion

### Behavior
- Lowest priority request preempted
- KV discarded
- Request requeued

### Latency Impact
- TTFT spike (full recompute)
- Initially affects tail latency (p99)
- Can cascade into system-wide degradation

---

# Q8 — Admission Control

### Key Principle
- Requests are NOT equal → tokens matter

### Token Budget
prompt_tokens + max_completion_tokens

### Why better than concurrency
- Accounts for request size variability

### Token Budget Correction
- Reduce reserved budget as tokens are generated
- Logical release, not physical KV release

---

# Q9 — Adversarial Requests

### Scenario
- One large request (e.g., 12K tokens)

### Problem
- Consumes full token budget
- Blocks smaller requests

### FIFO Failure
- Head-of-line blocking
- Small requests starved despite available capacity

---

# Q10 — Instrumentation Patch

### What was implemented
- Logging KV block allocation / free events

### What it revealed
- Real-time KV memory pressure
- Which requests consume memory
- When system approaches preemption

### Key Insight
- nvidia-smi ≠ memory pressure visibility
- Need fine-grained KV observability

---

# Clarification — Token Budget Correction

### Important Distinction
- KV cache is NOT freed mid-request
- Must retain all tokens (prefill + decode)

### What is "released"
- Logical reservation only
- Allows more requests to be admitted

### Insight
- This is a **prediction-based admission optimization**

---

# Additional Concepts Discussed

## Fair Scheduling
- Not implemented in vLLM by default
- FIFO → unfair under mixed workloads

### Strategies
- Weighted fair queuing
- Deficit round robin
- Per-tenant budgets

### Future Work
- Phase C focuses on fairness

---

# Meta Insight — Value of Phase B/C

User insight:
- True value = **operator intuition**
- Example:
  - Preemption cascade threshold (~65–70% KV usage)
  - Latency cliff behavior

---

# Next Step (Not Completed Yet)

## Frontier Lab Interview Brief

To be written:

1. Top 5 operating insights
2. 3 key experiments
3. Top 3 failure modes (trigger → mechanism → symptom)
4. 2 mistakes corrected
5. 2 mental model shifts
6. 1 production recommendation

---

# Notes for Next LLM

- User has strong intuition on:
  - KV memory as primary constraint
  - Scheduler behavior
  - Continuous batching
  - Admission control correctness

- Weak / incomplete areas:
  - Exact vLLM class names
  - Formal articulation (needs tightening)
  - Fair scheduling implementation details

- Recommended next step:
  Continue by building **Interview Brief (Step 5)**


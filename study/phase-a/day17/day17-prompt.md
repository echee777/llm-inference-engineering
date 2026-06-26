# Day 17 Continuation Prompt

## Context

This is Day 17 of a 20-day vLLM inference study plan. The user is building a KV-memory-driven admission gateway in front of vLLM on a T4 GPU (g4dn.xlarge, 16 GiB).

Day 16 morning concepts are complete and written up at `../day16/day16-work.md`. Day 16 afternoon build (the base gateway) may or may not be complete. Check with the user.

The full Day 17 syllabus is at `day17-syllabus-v4.md` in this directory. Read it first.

## What the user has already internalized (do not re-teach)

- Little's Law and why admission control is necessary
- Fail-fast (429) over queue-and-wait for LLM inference (retry storm risk)
- Admission budget derivation: `(total HBM - weights - overhead) / kv_bytes_per_token * gpu_memory_utilization * 0.65 = ADMISSION_BUDGET in tokens`
- The 0.65 comes from Day 9 empirical collapse observation at ~65-70% KV utilization. It's workload-specific, not a universal constant.
- Token budget vs concurrency cap distinction
- Per-request cost = prompt_tokens + max_completion_tokens (conservative)
- Five limitations of gateway estimate vs engine state: max_completion overestimate, prefix caching, prefill/decode temporal mismatch, preemption divergence, single-instance assumption
- PagedAttention solves external fragmentation. The cliff is about running out of free blocks for decode growth, triggering preemption cascades.
- Gateway is a predictive control plane. vLLM scheduler is a reactive control plane. Coupled but not identical.
- "Reconciliation" means comparing gateway counter vs actual KV block allocation vs GPU memory side by side to quantify proxy error.

## What needs to happen on Day 17

### If Day 16 build is not done

Build the base gateway first. See `../day16/day16-afternoon-prompt.md` for the full spec. Summarized:
- FastAPI gateway with Qwen tokenizer + chat template, admission check, SSE proxy, /metrics endpoint
- Three smoke tests (small admit, concurrent admit, forced 429)
- Deploy on T4 against vLLM running Qwen/Qwen2.5-3B-Instruct

### Day 17 morning: reconciliation experiment

1. Run 3 request shapes through the gateway against live vLLM:
   - Shape A: 5 requests x (200 prompt + 512 max_completion)
   - Shape B: 5 requests x (2000 prompt + 512 max_completion)
   - Shape C: Mixed (3x Shape A + 2x Shape B)

2. For each shape, record side by side:
   - Gateway `active_token_budget` at peak concurrency
   - Actual KV blocks allocated (from Day 8 scheduler instrumentation in vLLM)
   - GPU memory delta (nvidia-smi or torch.cuda.memory_allocated)

3. Quantify the divergence. Expected: gateway overestimates by 5-15%.

4. Select a trust boundary stance:
   - <= 15% error: gateway is authoritative
   - > 15% error: tighten TARGET_UTILIZATION
   - OOM/preemption observed: fall back to engine-side signals

The Day 8 instrumentation patch is in the vLLM source at `vllm/v1/core/sched/scheduler.py` and `vllm/v1/core/kv_cache_manager.py`. The user modified these files to log block allocation events. Check `git diff` on those files for details.

### Day 17 afternoon: token budget correction + rate limiting + queue

Part 1 (1.5 hrs): Implement Policy B (periodic release with safety floor)
- Release excess budget every 50 generated tokens
- SAFETY_MARGIN=64 floor to prevent cascading over-admission
- Requires counting tokens as they stream back via SSE
- Measure: 100 requests, compare baseline vs Policy B. How many more admitted?

Part 2 (2.5 hrs): Per-API-key rate limiting + bounded FIFO queue
- Sliding window rate limiter (requests/min + tokens/min per key)
- FIFO queue (maxsize=50, max wait 5s). Intentionally naive. Sets up Day 19 HOL blocking analysis.
- In-process counters (admitted, rejected_budget, rejected_rate_limit, correction_delta_released)

Part 3: Failure semantics checklist
- Test: vLLM 500 before first token -> budget released
- Test: client disconnect mid-stream -> budget released
- Test: queue timeout -> 503, budget never reserved
- Document: worker crash -> budget leaked, estimate blast radius

## Infrastructure

- T4 instance: `i-009226e4c86d676fa` (gpu-workbench-dev), currently stopped
- Start: `aws ec2 start-instances --instance-ids i-009226e4c86d676fa`
- Stop when done: `aws ec2 stop-instances --instance-ids i-009226e4c86d676fa`
- vLLM server: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --max-num-seqs 32 --max-model-len 4096 --port 8000`
- Gateway: `uvicorn gateway:app --host 0.0.0.0 --port 8080`

## KV math reference

- 36 KiB per token (FP16 KV, 36 layers, 2 GQA heads, 128 head_dim)
- FP16 model weights: 5.79 GiB
- Available for KV: 6.73 GiB (empirical from vLLM logs)
- KV capacity: ~191K tokens
- ADMISSION_BUDGET = 191,000 * 0.65 = ~124,150 tokens

## Writing style

All notes go in `day17-work.md`. Sound like concise human notes. No em-dashes. No bolding. No LLM-ish phrasing. All tables in ``` code blocks. Keep it minimal.

## Working style

The user prefers being quizzed interview-style on concepts before building. Challenge them on reasoning rather than giving answers. After the quiz/discussion, write up the artifact in their voice.

# Day 16 Afternoon: Build KV-Memory-Driven Admission Gateway

## Context

This is Day 16 of a 20-day vLLM study plan. The morning was spent on concepts and design (see `day16-work.md` in this directory). The afternoon is the build phase.

The full syllabus is at `day16-syllabus-v4.md` in this directory. Read it first for the complete specification including the code skeleton and smoke test matrix.

## What exists

- T4 GPU instance (g4dn.xlarge, 16 GiB) on AWS. Instance ID: `i-009226e4c86d676fa`, name: `gpu-workbench-dev`. Currently stopped. Start it before testing.
- Model: Qwen/Qwen2.5-3B-Instruct (FP16). Start vLLM with: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --max-num-seqs 32 --max-model-len 4096 --port 8000`
- KV math from previous days: 36 KiB per token (FP16 KV, 36 layers, 2 GQA heads, 128 head_dim). With FP16 weights, ~6.73 GiB available for KV cache = ~191K tokens capacity.
- Morning design doc written at `day16-work.md`.

## What to build

A FastAPI gateway (`gateway.py`) that sits in front of vLLM and enforces a KV-memory-based token budget. Requirements:

1. OpenAI-compatible `/v1/chat/completions` POST endpoint
2. Token counting using Qwen's own tokenizer (`AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")`), with chat template applied before encoding. Do not use tiktoken.
3. Active token budget counter protected by `asyncio.Lock()` (not threading.Lock). Per-request cost = `prompt_tokens + max_completion_tokens`.
4. Admission check: if `active_tokens + estimated_cost > ADMISSION_BUDGET`, return 429 with `Retry-After: 5` header. Do not admit.
5. Forward admitted requests to vLLM at `http://localhost:8000/v1/chat/completions`. Proxy the SSE stream byte-by-byte using `httpx.AsyncClient.stream()`. Never buffer the full response.
6. On request completion (in `finally` block of the stream generator), decrement `active_tokens` by `estimated_cost`.
7. `/metrics` GET endpoint returning `active_tokens`, `admission_budget`, and `budget_utilization_pct`.

Budget constants:
```
KV_CAPACITY_TOKENS = 191_000  # from empirical vLLM logs (6.73 GiB / 36 KiB per token)
TARGET_UTILIZATION = 0.65     # from Day 9 collapse observation
ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~124,150
```

The syllabus has a code skeleton in the "Build the gateway" section. Use it as a starting point but verify correctness.

## Smoke tests (must pass before calling the day done)

Run these three tests. Record results in `day16-work.md`.

```
Test                            Setup                                              Expected
Small request admitted          max_tokens=50, short prompt (~20 tokens)            200, response streams
Two medium requests concurrent  max_tokens=256 each, send simultaneously            Both 200
Oversized request rejected      Temporarily set ADMISSION_BUDGET=100, send          429 + Retry-After header
                                request with max_tokens=200
```

For test 3, temporarily hardcode a tiny budget to force the rejection. Verify the 429 response body contains `"error": "token budget exceeded"` and the `Retry-After` header is present. Verify that `_active_tokens` is NOT incremented for rejected requests. Then restore the real budget.

## Deployment

1. Start the T4: `aws ec2 start-instances --instance-ids i-009226e4c86d676fa`
2. SSH in, start vLLM server (command above)
3. SCP `gateway.py` to the instance
4. Install deps: `pip install fastapi uvicorn httpx transformers`
5. Run: `uvicorn gateway:app --host 0.0.0.0 --port 8080`
6. Run smoke tests (can use curl or a short Python script with httpx/aiohttp)
7. Record results in `day16-work.md`
8. Stop the T4 when done: `aws ec2 stop-instances --instance-ids i-009226e4c86d676fa`

## Writing style

All notes appended to `day16-work.md` should sound like concise human notes. No em-dashes. No bolding. No LLM-ish phrasing. All tables in ``` code blocks. Keep it minimal.

## What NOT to do

- Do not build a bounded queue. That's a Day 17/18 item.
- Do not implement progressive token release. That's Day 17.
- Do not add per-API-key rate limiting. That's Day 17.
- Keep scope tight: admission check, SSE proxy, one real 429 path.

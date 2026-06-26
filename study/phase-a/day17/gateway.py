# gateway.py — Day 17 afternoon: Policy B + rate limiting + FIFO queue
import asyncio
import json
import time
from collections import defaultdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- KV budget derived from Week 1 calculator ---
KV_CAPACITY_TOKENS = 191_000
TARGET_UTILIZATION = 0.65
ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~124,150

# Policy B parameters
RELEASE_INTERVAL = 50     # release excess budget every N generated tokens
SAFETY_MARGIN = 64        # never release below this many output tokens of headroom
# SAFETY_MARGIN=64 is a stand-in for a P95-derived value from the completion length
# distribution. Revisit once completion distribution data is available from Day 18 load tests.

DEFAULT_OUTPUT_RESERVATION = 512  # applied when client omits max_completion_tokens

# Rate limiting defaults (per API key)
RATE_LIMIT_REQUESTS = 60       # requests per window
RATE_LIMIT_TOKENS = 100_000    # tokens per window
RATE_LIMIT_WINDOW = 60         # window in seconds

# Queue parameters (intentionally naive FIFO, sets up Day 19 HOL blocking analysis)
QUEUE_MAXSIZE = 50
MAX_WAIT_SECONDS = 5.0

# --- State ---
_lock = asyncio.Lock()
_active_tokens = 0

stats = {
    "admitted": 0,
    "rejected_budget": 0,
    "rejected_rate_limit": 0,
    "rejected_queue_full": 0,
    "tokens_admitted": 0,
    "tokens_rejected": 0,
    "correction_delta_released": 0,
}

rate_limits = defaultdict(lambda: {"requests": [], "tokens": []})

_admission_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def count_prompt_tokens(messages: list[dict]) -> int:
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(formatted))


def check_rate_limit(api_key: str, tokens: int) -> bool:
    now = time.time()
    rl = rate_limits[api_key]
    rl["requests"] = [t for t in rl["requests"] if now - t < RATE_LIMIT_WINDOW]
    rl["tokens"] = [(t, n) for t, n in rl["tokens"] if now - t < RATE_LIMIT_WINDOW]
    if len(rl["requests"]) >= RATE_LIMIT_REQUESTS:
        return False
    if sum(n for _, n in rl["tokens"]) + tokens > RATE_LIMIT_TOKENS:
        return False
    rl["requests"].append(now)
    rl["tokens"].append((now, tokens))
    return True


async def try_admit(estimated_cost: int) -> bool:
    global _active_tokens
    async with _lock:
        if _active_tokens + estimated_cost > ADMISSION_BUDGET:
            return False
        _active_tokens += estimated_cost
        return True


async def release_budget(amount: int):
    global _active_tokens
    async with _lock:
        _active_tokens = max(0, _active_tokens - amount)


class RequestState:
    """Tracks per-request state for Policy B budget correction."""
    def __init__(self, prompt_tokens: int, max_completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.estimated_cost = prompt_tokens + max_completion_tokens
        self.tokens_generated = 0
        self.settled_through_token = 0
        self.total_released = 0  # cumulative budget released by Policy B


async def on_token_generated(req_state: RequestState):
    """Policy B: periodically release excess budget as tokens stream back."""
    req_state.tokens_generated += 1
    if req_state.tokens_generated - req_state.settled_through_token >= RELEASE_INTERVAL:
        remaining_needed = req_state.max_completion_tokens - req_state.tokens_generated
        if remaining_needed < 0:
            remaining_needed = 0
        safe_remaining = remaining_needed + SAFETY_MARGIN
        reserved_output = req_state.max_completion_tokens - req_state.settled_through_token
        releasable = reserved_output - safe_remaining
        if releasable > 0:
            await release_budget(releasable)
            stats["correction_delta_released"] += releasable
            req_state.total_released += releasable
            req_state.settled_through_token = req_state.tokens_generated


def count_sse_completion_tokens(chunk_bytes: bytes) -> int:
    """Parse SSE chunk to count completion tokens generated.

    Each SSE data line with a delta content token represents one generated token.
    This is approximate: some chunks may contain multiple tokens or metadata-only events.
    """
    count = 0
    text = chunk_bytes.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    # Each content delta is roughly one token from vLLM streaming.
                    # Not perfectly 1:1 but close enough for budget correction.
                    count += 1
        except (json.JSONDecodeError, KeyError):
            pass
    return count


@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_completion = body.get("max_tokens") or DEFAULT_OUTPUT_RESERVATION
    api_key = request.headers.get("authorization", "default").removeprefix("Bearer ").strip() or "default"

    prompt_tokens = count_prompt_tokens(messages)
    estimated_cost = prompt_tokens + max_completion

    # Rate limit check (before admission, before queue)
    if not check_rate_limit(api_key, estimated_cost):
        stats["rejected_rate_limit"] += 1
        stats["tokens_rejected"] += estimated_cost
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded", "retry_after": 10},
            headers={"Retry-After": "10"},
        )

    # Admission check
    admitted = await try_admit(estimated_cost)

    if not admitted:
        # Try queue with bounded wait
        try:
            waiter = asyncio.get_event_loop().create_future()
            _admission_queue.put_nowait((estimated_cost, waiter))
        except asyncio.QueueFull:
            stats["rejected_queue_full"] += 1
            stats["tokens_rejected"] += estimated_cost
            return JSONResponse(
                status_code=503,
                content={"error": "queue full", "retry_after": 5},
                headers={"Retry-After": "5"},
            )

        # Wait for budget to become available (up to MAX_WAIT_SECONDS)
        try:
            admitted = await asyncio.wait_for(waiter, timeout=MAX_WAIT_SECONDS)
        except asyncio.TimeoutError:
            # Remove from queue if still there, budget was never reserved
            stats["rejected_budget"] += 1
            stats["tokens_rejected"] += estimated_cost
            return JSONResponse(
                status_code=503,
                content={"error": "queue timeout", "retry_after": 5},
                headers={"Retry-After": "5"},
            )

        if not admitted:
            stats["rejected_budget"] += 1
            stats["tokens_rejected"] += estimated_cost
            return JSONResponse(
                status_code=429,
                content={"error": "token budget exceeded", "retry_after": 5},
                headers={"Retry-After": "5"},
            )

    stats["admitted"] += 1
    stats["tokens_admitted"] += estimated_cost

    req_state = RequestState(prompt_tokens, max_completion)

    print(f"[gateway] ADMIT  prompt={prompt_tokens} max_completion={max_completion} "
          f"cost={estimated_cost} active={_active_tokens}/{ADMISSION_BUDGET}")

    async def stream_and_release():
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", VLLM_URL, json=body) as resp:
                    if resp.status_code >= 500:
                        # vLLM error before first token, release full budget
                        yield b"data: " + json.dumps({"error": "upstream error"}).encode() + b"\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        # Count tokens in this SSE chunk for Policy B
                        token_count = count_sse_completion_tokens(chunk)
                        for _ in range(token_count):
                            await on_token_generated(req_state)
                        yield chunk
        finally:
            # Release whatever budget is still held: original charge minus Policy B releases
            final_release = req_state.estimated_cost - req_state.total_released
            await release_budget(final_release)

            print(f"[gateway] RELEASE prompt={prompt_tokens} generated={req_state.tokens_generated} "
                  f"policy_b_freed={stats['correction_delta_released']} "
                  f"final_release={final_release} active={_active_tokens}")

            # Try to admit queued requests after releasing budget
            await _drain_queue()

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")


async def _drain_queue():
    """After budget is released, try to admit queued waiters."""
    while not _admission_queue.empty():
        try:
            estimated_cost, waiter = _admission_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if waiter.done():
            continue
        admitted = await try_admit(estimated_cost)
        if admitted:
            waiter.set_result(True)
        else:
            # Put back and stop draining, not enough budget
            try:
                _admission_queue.put_nowait((estimated_cost, waiter))
            except asyncio.QueueFull:
                waiter.set_result(False)
            break


@app.get("/metrics")
def metrics():
    return {
        "active_tokens": _active_tokens,
        "admission_budget": ADMISSION_BUDGET,
        "budget_utilization_pct": round(100 * _active_tokens / ADMISSION_BUDGET, 1),
        **stats,
    }


# Periodic stats logging
async def _stats_logger():
    while True:
        await asyncio.sleep(10)
        utilization = _active_tokens / ADMISSION_BUDGET * 100
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"budget={utilization:.1f}% "
              f"admitted={stats['admitted']} "
              f"rejected_budget={stats['rejected_budget']} "
              f"rejected_rate={stats['rejected_rate_limit']} "
              f"correction_freed={stats['correction_delta_released']} tokens")


@app.on_event("startup")
async def startup():
    asyncio.create_task(_stats_logger())

# gateway.py — Day 18: Day 17 gateway + Prometheus instrumentation + TTFT tracking
#
# This file is a reverse proxy that sits BETWEEN the client (Locust) and
# the model server (vLLM). Every request flows: Client -> Gateway -> vLLM.
# The gateway's job is to decide whether to ADMIT a request (let it through
# to vLLM) or REJECT it (return 429/503) based on how much KV cache memory
# is currently reserved by in-flight requests.
#
# What's new in Day 18 vs Day 17:
#   1. Prometheus metrics (counters, gauges, histograms) so Grafana can plot
#      real-time dashboards.
#   2. Three-timestamp TTFT instrumentation: arrival_time, admitted_time,
#      first_token_time. These let us separately measure gateway TTFT
#      (what the user experiences) vs model TTFT (how fast vLLM responds
#      after admission).
#   3. ADMISSION_ENABLED env flag so we can run the same gateway in two
#      modes: baseline (no gating, everything admitted) vs controlled
#      (admission enforced). Same code, same metrics, directly comparable.

import asyncio
import json
import os
import time
from collections import defaultdict

# httpx: async HTTP client. We use it to forward requests to vLLM with
# streaming support (like requests, but async-native).
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# prometheus_client: Python library that exposes metrics in the format
# Prometheus expects to scrape.
#   Counter: only goes up (total requests served)
#   Gauge: goes up and down (current queue depth)
#   Histogram: records observations into buckets (TTFT latency distribution)
#   make_asgi_app: creates a mini ASGI app that serves the /metrics page
#     Prometheus scrapes
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# AutoTokenizer: we use the model's actual tokenizer to count prompt tokens
# accurately. This matters because admission cost = prompt_tokens + max_completion_tokens.
from transformers import AutoTokenizer

app = FastAPI()

# Mount a Prometheus-format metrics endpoint at /prom.
# Why /prom and not /metrics? Because vLLM already serves its own metrics at
# /metrics on port 8000. Using /prom avoids confusion. Prometheus is configured
# in prometheus.yml to scrape this path.
# make_asgi_app() returns a tiny ASGI application that, when hit, serializes
# all registered Counter/Gauge/Histogram objects into Prometheus text format.
metrics_app = make_asgi_app()
app.mount("/prom", metrics_app)

# vLLM runs on port 8000. The gateway proxies every admitted request here.
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- KV budget parameters (from Day 17 / Week 1 KV calculator) ---
# KV_CAPACITY_TOKENS: the total number of tokens the GPU's KV cache can hold.
# This was calculated from GPU memory, model dimensions, and dtype.
KV_CAPACITY_TOKENS = 217_312

# TARGET_UTILIZATION: we only use 65% of KV capacity. The 35% headroom absorbs
# token estimation error. If we estimated perfectly, we could go higher. But
# max_completion_tokens is a ceiling (requests often generate fewer tokens),
# so our budget overestimates real usage. 65% is the safe operating point
# calibrated on Day 17.
TARGET_UTILIZATION = 0.65
_DEFAULT_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)
ADMISSION_BUDGET = int(os.environ.get("BUDGET_TOKENS", _DEFAULT_BUDGET))

# --- Policy B: progressive budget release (from Day 17) ---
# Instead of holding the full max_completion_tokens reservation until the
# request finishes, we periodically release excess budget as tokens stream
# back and we learn the actual completion length.
# RELEASE_INTERVAL: check every 50 generated tokens whether we can release some budget.
# SAFETY_MARGIN: always keep at least 64 tokens of headroom beyond what's still needed.
RELEASE_INTERVAL = 50
SAFETY_MARGIN = 64

# If the client doesn't send max_completion_tokens, assume 512.
# This prevents unbounded reservations from blowing the budget.
DEFAULT_OUTPUT_RESERVATION = 512

# --- Rate limiting (per API key, separate from admission control) ---
# Rate limiting caps individual users. Admission control caps the system.
# They serve different purposes: rate limiting is fairness, admission is protection.
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 10_000))   # raised for load testing (default 60)
RATE_LIMIT_TOKENS = int(os.environ.get("RATE_LIMIT_TOKENS", 50_000_000)) # raised for load testing (default 100k)
RATE_LIMIT_WINDOW = 60         # sliding window in seconds

# --- Queue parameters ---
# When a request can't be immediately admitted, it waits in this queue.
# If the queue is full (50 items), new requests get 503 immediately.
# If a queued request waits longer than MAX_WAIT_SECONDS, it gets 503.
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", 50))
MAX_WAIT_SECONDS = float(os.environ.get("MAX_WAIT_SECONDS", 5.0))

# --- Admission toggle ---
# Set ADMISSION_ENABLED=false as an environment variable to run in baseline
# mode (no gating). The gateway still tracks _active_tokens so the Prometheus
# gauges show what budget pressure WOULD have been without admission control.
# This lets us overlay baseline and controlled dashboards for direct comparison.
ADMISSION_ENABLED = os.getenv("ADMISSION_ENABLED", "true").lower() in ("true", "1", "yes")

# ============================================================================
# PROMETHEUS METRIC DEFINITIONS
# ============================================================================
# Each of these creates a metric that Prometheus will scrape every 5 seconds.
# The metric names here (e.g. 'gateway_requests_total') are what you'll
# reference in PromQL queries on the Grafana dashboard.

# Counter: monotonically increasing count of requests by outcome.
# The ['status'] label lets us query admitted vs rejected separately:
#   rate(gateway_requests_total{status="admitted"}[1m])  -> admitted requests/sec
#   rate(gateway_requests_total{status="rejected_budget"}[1m])  -> rejected/sec
REQUEST_COUNTER = Counter(
    'gateway_requests_total', 'Requests by disposition',
    ['status']
)

# Gauge: current value that goes up and down.
# Queue depth is a leading indicator: it spikes BEFORE TTFT spikes.
QUEUE_DEPTH = Gauge('gateway_queue_depth', 'Current queue depth')

# Token budget utilization as a percentage. This is Dashboard Panel 1,
# the primary operational metric. Target operating range: 55-70%.
TOKEN_BUDGET_USED = Gauge('gateway_token_budget_used_pct', 'Token budget utilization %')

# Histogram: records individual observations into predefined buckets.
# Gateway TTFT: measures arrival_time -> first_token_time.
# This is what the CLIENT experiences (includes queue wait + model processing).
# The buckets define resolution: [50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s].
# Dense buckets in the sub-second range where our SLO lives gives better
# accuracy when Prometheus computes percentiles via linear interpolation.
GATEWAY_TTFT = Histogram(
    'gateway_ttft_seconds', 'Gateway TTFT (arrival to first token)',
    buckets=[.05, .1, .25, .5, 1, 2, 5, 10]
)

# Model TTFT: measures admitted_time -> first_token_time.
# This is purely how fast vLLM responds AFTER we forward the request.
# The gap between Gateway TTFT and Model TTFT = queue wait + gateway overhead.
# If Gateway TTFT rises but Model TTFT stays flat, the problem is queueing,
# not the model. If both rise, the model backend is under pressure.
MODEL_TTFT = Histogram(
    'model_ttft_seconds', 'Model TTFT (post-admission to first token)',
    buckets=[.05, .1, .25, .5, 1, 2, 5, 10]
)

# Queue wait: arrival_time -> admitted_time.
# How long requests wait before the admission decision is made.
# Buckets are much smaller (1ms to 5s) because admission decisions should
# be fast. If queue wait p95 is high, requests are backing up.
QUEUE_WAIT = Histogram(
    'gateway_queue_wait_seconds', 'Queue wait before admission',
    buckets=[.001, .005, .01, .05, .1, .5, 1, 5]
)

# ============================================================================
# INTERNAL STATE
# ============================================================================

# asyncio.Lock ensures only one coroutine modifies _active_tokens at a time.
# Without this, concurrent requests could read-modify-write _active_tokens
# and produce a race condition (e.g., two requests both see 120k, both add
# their cost, both get admitted, actual total exceeds budget).
_lock = asyncio.Lock()

# _active_tokens: the sum of estimated_cost for all currently in-flight requests.
# This is THE admission signal. When _active_tokens + new_request_cost > ADMISSION_BUDGET,
# the new request is rejected.
_active_tokens = 0

# Internal stats dict for the /debug/stats JSON endpoint (human debugging).
# These are SEPARATE from Prometheus metrics. Prometheus metrics are for
# dashboards; this dict is for quick curl checks during development.
stats = {
    "admitted": 0,
    "rejected_budget": 0,
    "rejected_rate_limit": 0,
    "rejected_queue_full": 0,
    "tokens_admitted": 0,
    "tokens_rejected": 0,
    "correction_delta_released": 0,
}

# Per-API-key rate limit tracking. Each key gets its own sliding window
# of recent requests and token counts.
rate_limits = defaultdict(lambda: {"requests": [], "tokens": []})

# The admission queue. When a request can't be admitted immediately,
# it's placed here and waits for budget to free up (when another request
# completes and releases its tokens).
_admission_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

# Load the model's tokenizer once at startup. This is used to accurately
# count prompt tokens. We need the real tokenizer (not a heuristic) because
# different models tokenize differently, and our budget math depends on
# accurate token counts.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def count_prompt_tokens(messages: list[dict]) -> int:
    """Count exact prompt tokens using the model's tokenizer.

    apply_chat_template formats the messages into the model's expected format
    (with special tokens, role markers, etc.), then we encode and count.
    This gives us the real prompt token count, not an estimate.
    """
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(formatted))


def check_rate_limit(api_key: str, tokens: int) -> bool:
    """Sliding window rate limiter per API key.

    Checks two limits: requests per window AND tokens per window.
    Both must be within limits to pass. The sliding window works by
    keeping a list of timestamps and pruning entries older than
    RATE_LIMIT_WINDOW seconds.
    """
    now = time.time()
    rl = rate_limits[api_key]
    # Prune expired entries from the sliding window
    rl["requests"] = [t for t in rl["requests"] if now - t < RATE_LIMIT_WINDOW]
    rl["tokens"] = [(t, n) for t, n in rl["tokens"] if now - t < RATE_LIMIT_WINDOW]
    if len(rl["requests"]) >= RATE_LIMIT_REQUESTS:
        return False
    if sum(n for _, n in rl["tokens"]) + tokens > RATE_LIMIT_TOKENS:
        return False
    # Record this request in the window
    rl["requests"].append(now)
    rl["tokens"].append((now, tokens))
    return True


async def try_admit(estimated_cost: int) -> bool:
    """The core admission control check.

    This is the single most important function in the gateway.
    It atomically checks whether there's enough budget headroom for
    this request. If yes, it reserves the tokens and returns True.
    If no, it returns False without modifying state.

    The check: _active_tokens + estimated_cost > ADMISSION_BUDGET
    If this would exceed the budget, reject. Otherwise, add to active.
    """
    global _active_tokens
    async with _lock:  # atomic read-check-write
        if _active_tokens + estimated_cost > ADMISSION_BUDGET:
            return False
        _active_tokens += estimated_cost
        return True


async def force_admit(estimated_cost: int):
    """Baseline mode: always admit, but still track token usage.

    When ADMISSION_ENABLED=false, we skip the budget check but still
    increment _active_tokens. This way the Prometheus gauge shows what
    the budget pressure IS, even though we're not gating on it.
    In baseline mode, _active_tokens can exceed ADMISSION_BUDGET
    (utilization > 100%), which is exactly what we want to see on
    the dashboard to demonstrate the need for admission control.
    """
    global _active_tokens
    async with _lock:
        _active_tokens += estimated_cost


async def release_budget(amount: int):
    """Return tokens to the available pool when a request completes
    or when Policy B progressively releases excess reservation.

    The max(0, ...) prevents underflow from rounding/timing edge cases.
    """
    global _active_tokens
    async with _lock:
        _active_tokens = max(0, _active_tokens - amount)


def _update_gauges():
    """Push current state to Prometheus gauges.

    Called after every admission/release/queue change so the dashboard
    reflects near-real-time state. Prometheus scrapes every 5s, but
    we update the gauge values continuously so any scrape gets a
    fresh reading.
    """
    TOKEN_BUDGET_USED.set(round(100 * _active_tokens / ADMISSION_BUDGET, 1))
    QUEUE_DEPTH.set(_admission_queue.qsize())


class RequestState:
    """Tracks per-request state for Policy B budget correction.

    Each admitted request gets one of these. It tracks:
    - The original reservation (estimated_cost = prompt + max_completion)
    - How many tokens have actually been generated so far
    - How much budget has been released back via Policy B
    This lets us progressively release excess budget as the real
    completion length becomes known, instead of holding the full
    max_completion_tokens reservation until the end.
    """
    def __init__(self, prompt_tokens: int, max_completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.estimated_cost = prompt_tokens + max_completion_tokens
        self.tokens_generated = 0       # actual tokens generated so far
        self.last_release_at = 0        # token count at last Policy B release
        self.total_released = 0         # cumulative budget released by Policy B


async def on_token_generated(req_state: RequestState):
    """Policy B: periodically release excess budget as tokens stream back.

    Every RELEASE_INTERVAL tokens (50), we calculate how much of the original
    max_completion_tokens reservation is no longer needed:
      - remaining_needed: how many more tokens could the model still generate?
      - safe_remaining: remaining_needed + SAFETY_MARGIN (keep a buffer)
      - releasable: what we're still holding minus what we safely need

    Example: request reserved 2000 max_completion. After 500 tokens generated:
      remaining_needed = 2000 - 500 = 1500
      safe_remaining = 1500 + 64 = 1564
      reserved_output = 2000 - 0 = 2000  (if first release)
      releasable = 2000 - 1564 = 436 tokens returned to the pool

    This is how we avoid the decode-heavy overestimation problem: we don't
    wait until the request finishes to learn it only used 200 of 2000 tokens.
    """
    req_state.tokens_generated += 1
    if req_state.tokens_generated - req_state.last_release_at >= RELEASE_INTERVAL:
        remaining_needed = req_state.max_completion_tokens - req_state.tokens_generated
        if remaining_needed < 0:
            remaining_needed = 0
        safe_remaining = remaining_needed + SAFETY_MARGIN
        reserved_output = req_state.max_completion_tokens - req_state.last_release_at
        releasable = reserved_output - safe_remaining
        if releasable > 0:
            await release_budget(releasable)
            stats["correction_delta_released"] += releasable
            req_state.total_released += releasable
            req_state.last_release_at = req_state.tokens_generated
            _update_gauges()


def count_sse_completion_tokens(chunk_bytes: bytes) -> int:
    """Parse an SSE (Server-Sent Events) chunk to count generated tokens.

    vLLM streams responses as SSE: each generated token produces a line like:
      data: {"choices": [{"delta": {"content": "Hello"}}]}

    We count lines that have delta content. This is approximate (some chunks
    may batch multiple tokens or contain metadata), but close enough for
    Policy B budget correction. Off-by-a-few-tokens doesn't matter when
    the release interval is 50.
    """
    count = 0
    text = chunk_bytes.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":  # vLLM sends this as the final SSE event
            continue
        try:
            obj = json.loads(data)
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    count += 1
        except (json.JSONDecodeError, KeyError):
            pass
    return count


def _has_sse_content(chunk_bytes: bytes) -> bool:
    """Check if an SSE chunk contains actual model output (not just keepalive/headers).

    We use this for TTFT measurement. We don't want to measure time-to-first-keepalive,
    we want time-to-first-actual-token. A chunk with real content has a "data: " line
    that isn't empty or "[DONE]".
    """
    text = chunk_bytes.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
            return True
    return False


# ============================================================================
# MAIN REQUEST HANDLER
# ============================================================================
# This is the core request lifecycle. Every inference request flows through
# this function. The flow is:
#
#   1. Parse request, count prompt tokens, compute estimated cost
#   2. Rate limit check (per-user fairness)
#   3. Admission check (system-wide protection)
#      - If admitted: proceed to vLLM
#      - If not admitted: try the queue (wait up to MAX_WAIT_SECONDS)
#      - If queue full or timeout: reject with 429/503
#   4. Stream response from vLLM back to client
#   5. On completion: release budget, drain queue (let waiting requests in)
#
# THREE TIMESTAMPS are captured for metrics:
#   arrival_time  -> when the request first hits the gateway
#   admitted_time -> when the admission decision completes (after any queue wait)
#   first_token_time -> when the first real token arrives from vLLM
#
# From these we derive:
#   Queue Wait   = admitted_time - arrival_time
#   Model TTFT   = first_token_time - admitted_time
#   Gateway TTFT = first_token_time - arrival_time  (= Queue Wait + Model TTFT)

@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    # TIMESTAMP 1: when the request arrives at the gateway.
    # time.monotonic() is used instead of time.time() because it's immune to
    # system clock adjustments. We only care about elapsed durations, not
    # wall-clock time.
    arrival_time = time.monotonic()

    body = await request.json()
    messages = body.get("messages", [])
    # Use client's max_tokens if provided, otherwise default to 512.
    max_completion = body.get("max_tokens") or DEFAULT_OUTPUT_RESERVATION
    api_key = request.headers.get("authorization", "default").removeprefix("Bearer ").strip() or "default"

    # Count prompt tokens using the real tokenizer (not a heuristic).
    prompt_tokens = count_prompt_tokens(messages)
    # estimated_cost is the admission signal: how many KV cache tokens this
    # request will reserve. prompt_tokens are consumed during prefill (all at once),
    # max_completion is reserved for decode (consumed incrementally).
    estimated_cost = prompt_tokens + max_completion

    # --- GATE 1: Rate limiting (per-user, before admission) ---
    # Rate limiting runs BEFORE admission so a single abusive user can't
    # consume the entire admission queue.
    if not check_rate_limit(api_key, estimated_cost):
        stats["rejected_rate_limit"] += 1
        stats["tokens_rejected"] += estimated_cost
        REQUEST_COUNTER.labels(status="rejected_rate_limit").inc()
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded", "retry_after": 10},
            headers={"Retry-After": "10"},
        )

    # --- GATE 2: Admission control ---
    if not ADMISSION_ENABLED:
        # BASELINE MODE: skip the budget check, admit everything.
        # We still track _active_tokens so the dashboard shows what budget
        # pressure WOULD have been. This is how we prove admission control
        # is necessary: the baseline dashboard shows utilization > 100%
        # while the controlled dashboard shows it capped at ~65%.
        await force_admit(estimated_cost)
        admitted = True
        admitted_time = time.monotonic()  # TIMESTAMP 2 (immediate, no wait)
        REQUEST_COUNTER.labels(status="admitted").inc()
        QUEUE_WAIT.observe(admitted_time - arrival_time)
    else:
        # CONTROLLED MODE: enforce the budget.
        admitted = await try_admit(estimated_cost)

        if not admitted:
            # Request exceeds budget. Instead of rejecting immediately,
            # try the queue: maybe an in-flight request will finish soon
            # and free up budget.
            try:
                # Create a Future that will be resolved when budget frees up.
                # _drain_queue() (called when requests complete) will check
                # this queue and set the Future's result to True if it can
                # admit the waiting request.
                waiter = asyncio.get_event_loop().create_future()
                _admission_queue.put_nowait((estimated_cost, waiter))
                _update_gauges()
            except asyncio.QueueFull:
                # Queue itself is full (50 items). No room to wait.
                stats["rejected_queue_full"] += 1
                stats["tokens_rejected"] += estimated_cost
                REQUEST_COUNTER.labels(status="rejected_queue_full").inc()
                return JSONResponse(
                    status_code=503,
                    content={"error": "queue full", "retry_after": 5},
                    headers={"Retry-After": "5"},
                )

            # Wait for up to MAX_WAIT_SECONDS for budget to free up.
            try:
                admitted = await asyncio.wait_for(waiter, timeout=MAX_WAIT_SECONDS)
            except asyncio.TimeoutError:
                # Waited too long. Budget never freed up in time.
                stats["rejected_budget"] += 1
                stats["tokens_rejected"] += estimated_cost
                REQUEST_COUNTER.labels(status="rejected_queue_timeout").inc()
                return JSONResponse(
                    status_code=503,
                    content={"error": "queue timeout", "retry_after": 5},
                    headers={"Retry-After": "5"},
                )

            if not admitted:
                # _drain_queue set the Future to False (couldn't admit).
                stats["rejected_budget"] += 1
                stats["tokens_rejected"] += estimated_cost
                REQUEST_COUNTER.labels(status="rejected_budget").inc()
                return JSONResponse(
                    status_code=429,
                    content={"error": "token budget exceeded", "retry_after": 5},
                    headers={"Retry-After": "5"},
                )

        # TIMESTAMP 2: admission decision complete.
        # For immediately-admitted requests, this is ~instant after arrival.
        # For queued requests, this includes the queue wait time.
        admitted_time = time.monotonic()
        REQUEST_COUNTER.labels(status="admitted").inc()
        # Queue Wait = how long the request waited before being admitted.
        # For immediately-admitted requests this is near-zero.
        QUEUE_WAIT.observe(admitted_time - arrival_time)

    # --- Request is admitted. Forward to vLLM. ---
    stats["admitted"] += 1
    stats["tokens_admitted"] += estimated_cost
    _update_gauges()

    req_state = RequestState(prompt_tokens, max_completion)

    print(f"[gateway] ADMIT  prompt={prompt_tokens} max_completion={max_completion} "
          f"cost={estimated_cost} active={_active_tokens}/{ADMISSION_BUDGET} "
          f"admission={'ON' if ADMISSION_ENABLED else 'OFF'}")

    async def stream_and_release():
        """Stream vLLM's response back to the client, measuring TTFT and
        running Policy B budget correction along the way.

        This is an async generator: each `yield chunk` sends a piece of
        the SSE stream to the client. FastAPI's StreamingResponse consumes
        this generator.

        The `finally` block is critical: it runs whether the request
        completes normally, errors out, or the client disconnects. It
        releases the budget reservation so other requests can be admitted.
        """
        first_token_seen = False
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Stream the request to vLLM. client.stream() returns chunks
                # as they arrive rather than buffering the full response.
                async with client.stream("POST", VLLM_URL, json=body) as resp:
                    if resp.status_code >= 500:
                        # vLLM returned an error. Forward it and bail out.
                        # The finally block will still release the budget.
                        yield b"data: " + json.dumps({"error": "upstream error"}).encode() + b"\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        # --- TIMESTAMP 3: first token detection ---
                        # _has_sse_content checks that this chunk contains
                        # actual model output (a "data:" line with content),
                        # not just a keepalive or metadata frame.
                        if not first_token_seen and _has_sse_content(chunk):
                            first_token_time = time.monotonic()
                            # Gateway TTFT: what the client experiences.
                            # arrival -> first token. Includes queue wait.
                            GATEWAY_TTFT.observe(first_token_time - arrival_time)
                            # Model TTFT: how fast vLLM responded.
                            # admission -> first token. Excludes queue wait.
                            MODEL_TTFT.observe(first_token_time - admitted_time)
                            first_token_seen = True

                        # Count tokens in this chunk for Policy B progressive release.
                        token_count = count_sse_completion_tokens(chunk)
                        for _ in range(token_count):
                            await on_token_generated(req_state)

                        # Forward the chunk to the client.
                        yield chunk
        finally:
            # BUDGET RELEASE: return whatever tokens are still reserved.
            # estimated_cost was reserved at admission. Policy B may have
            # already released some (total_released). Release the remainder.
            # This is what makes room for the next request to be admitted.
            final_release = req_state.estimated_cost - req_state.total_released
            await release_budget(final_release)
            _update_gauges()

            print(f"[gateway] RELEASE prompt={prompt_tokens} generated={req_state.tokens_generated} "
                  f"policy_b_freed={req_state.total_released} "
                  f"final_release={final_release} active={_active_tokens}")

            # After releasing budget, check if any queued requests can now
            # be admitted. This is the feedback loop: request completes ->
            # budget frees -> queued request admitted -> repeat.
            await _drain_queue()

    # StreamingResponse consumes the async generator and sends chunks to
    # the client as they arrive. media_type="text/event-stream" tells the
    # client this is an SSE stream (same format vLLM uses).
    return StreamingResponse(stream_and_release(), media_type="text/event-stream")


async def _drain_queue():
    """After budget is released, try to admit queued waiters.

    This is called every time a request completes (in the finally block).
    It walks the queue front-to-back, attempting to admit each waiter.
    If a waiter can be admitted (enough budget), its Future is resolved
    with True, which unblocks the waiting coroutine in proxy_completions.
    If there's not enough budget for the next waiter, we stop (FIFO order
    is preserved, no queue jumping).
    """
    while not _admission_queue.empty():
        try:
            estimated_cost, waiter = _admission_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if waiter.done():
            # Waiter already timed out or was cancelled. Skip it.
            continue
        admitted = await try_admit(estimated_cost)
        if admitted:
            waiter.set_result(True)  # unblocks the waiting request
        else:
            # Not enough budget. Put this waiter back at the front and stop.
            # We can't skip it (FIFO), so no point checking further.
            try:
                _admission_queue.put_nowait((estimated_cost, waiter))
            except asyncio.QueueFull:
                waiter.set_result(False)
            break
    _update_gauges()


@app.get("/debug/stats")
def debug_stats():
    """JSON stats endpoint for quick debugging via curl.

    This is NOT what Prometheus scrapes (that's /prom). This is for
    human-readable inspection during development:
      curl http://localhost:8001/debug/stats | python -m json.tool
    """
    return {
        "active_tokens": _active_tokens,
        "admission_budget": ADMISSION_BUDGET,
        "budget_utilization_pct": round(100 * _active_tokens / ADMISSION_BUDGET, 1),
        "admission_enabled": ADMISSION_ENABLED,
        **stats,
    }


async def _stats_logger():
    """Print stats to stdout every 10 seconds for terminal monitoring.

    Also updates Prometheus gauges on the same interval as a safety net
    (gauges are also updated on every admission/release event, but this
    ensures they're fresh even during quiet periods).
    """
    while True:
        await asyncio.sleep(10)
        _update_gauges()
        utilization = _active_tokens / ADMISSION_BUDGET * 100
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"admission={'ON' if ADMISSION_ENABLED else 'OFF'} "
              f"budget={utilization:.1f}% "
              f"admitted={stats['admitted']} "
              f"rejected_budget={stats['rejected_budget']} "
              f"rejected_rate={stats['rejected_rate_limit']} "
              f"correction_freed={stats['correction_delta_released']} tokens")


@app.on_event("startup")
async def startup():
    """Run once when the gateway starts. Launches the background stats logger."""
    asyncio.create_task(_stats_logger())
    print(f"[gateway] Admission control: {'ENABLED' if ADMISSION_ENABLED else 'DISABLED (baseline mode)'}")
    print(f"[gateway] Budget: {ADMISSION_BUDGET:,} tokens (KV capacity: {KV_CAPACITY_TOKENS:,})")

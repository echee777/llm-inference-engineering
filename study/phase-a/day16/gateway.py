"""
Day 16 admission-control gateway (FastAPI, sits in front of a vLLM server).

What it does: gates incoming requests on a memory budget expressed in TOKENS, not a
flat concurrency cap, because ten short requests and ten long ones are not the same
load on the KV cache. It counts each request's prompt tokens with the model's own
tokenizer (chat template applied), admits while the running total stays under the
budget, and returns a real HTTP 429 with Retry-After when it would not. Admitted
traffic is proxied to vLLM with byte-by-byte SSE streaming so latency is untouched.

Honest limitation (see day16-work.md): the budget is an external proxy for KV usage.
It cannot see prefix-cache hits, preemption, or the prefill/decode timing asymmetry,
so it deliberately errs conservative. Day 17-18 quantify and partly correct this.

How to run:
    # 1. start a vLLM OpenAI-compatible server on port 8000 (separate terminal):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-3B-Instruct --max-model-len 2048 --port 8000
    # 2. start this gateway on port 8080:
    uvicorn gateway:app --host 0.0.0.0 --port 8080
    # 3. validate:
    python smoke_test.py
"""
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- KV budget derived from Week 1 calculator ---
# T4 16 GiB, Qwen2.5-3B FP16: ~6.73 GiB available for KV cache
# 36 KiB per token (36 layers, 2 GQA heads, 128 head_dim, FP16 K+V)
# 6.73 GiB / 36 KiB = ~191K tokens capacity
KV_CAPACITY_TOKENS = 191_000
TARGET_UTILIZATION = 0.65          # from Day 9 collapse observation
ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~124,150

# asyncio.Lock is the correct primitive for async FastAPI context.
# threading.Lock() would block the event loop under contention.
# Single-process assumption. Distributed deployment needs shared state (Redis or shard-per-pod).
_lock = asyncio.Lock()
_active_tokens = 0

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def count_prompt_tokens(messages: list[dict]) -> int:
    """Count prompt tokens using the model's own tokenizer with chat template applied.

    Applying the chat template before tokenizing matches vLLM's internal formatting path.
    Without it, the gateway undercounts (missing role markers, special tokens), which
    biases toward over-admission, the dangerous direction.
    """
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(formatted))


@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    global _active_tokens

    body = await request.json()
    messages = body.get("messages", [])
    max_completion = body.get("max_tokens", 512)

    prompt_tokens = count_prompt_tokens(messages)
    estimated_cost = prompt_tokens + max_completion

    # --- Admission check ---
    async with _lock:
        if _active_tokens + estimated_cost > ADMISSION_BUDGET:
            return JSONResponse(
                status_code=429,
                content={"error": "token budget exceeded", "retry_after": 5},
                headers={"Retry-After": "5"},
                # Retry-After is static (5s). Production value would be derived from
                # observed budget recovery velocity.
            )
        _active_tokens += estimated_cost

    print(f"[gateway] ADMIT  prompt={prompt_tokens} max_completion={max_completion} "
          f"estimated_cost={estimated_cost} active={_active_tokens}/{ADMISSION_BUDGET}")

    async def stream_and_release():
        global _active_tokens
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", VLLM_URL, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        finally:
            async with _lock:
                _active_tokens -= estimated_cost
            print(f"[gateway] RELEASE estimated_cost={estimated_cost} active={_active_tokens}")

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    return {
        "active_tokens": _active_tokens,
        "admission_budget": ADMISSION_BUDGET,
        "budget_utilization_pct": round(100 * _active_tokens / ADMISSION_BUDGET, 1),
    }

# gateway.py
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

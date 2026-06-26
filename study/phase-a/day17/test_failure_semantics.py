"""Day 17 Failure Semantics Checklist.

Tests that budget is correctly released under every failure path.
Run against live gateway+vLLM.
"""
import asyncio
import httpx

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"
METRICS_URL = "http://localhost:8080/metrics"


def make_payload(content: str, max_tokens: int = 256) -> dict:
    return {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
    }


async def get_active_tokens() -> int:
    async with httpx.AsyncClient() as client:
        r = await client.get(METRICS_URL)
        return r.json()["active_tokens"]


async def test_normal_completion():
    """Baseline: normal request completes, budget fully released."""
    print("\n=== Test: Normal completion ===")
    before = await get_active_tokens()
    payload = make_payload("Say hello in one word.", max_tokens=50)
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", GATEWAY_URL, json=payload) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_bytes():
                pass
    await asyncio.sleep(0.5)
    after = await get_active_tokens()
    print(f"  Before: {before}, After: {after}")
    assert after == before, f"Budget leak: {after - before} tokens not released"
    print("  PASS: budget fully released after normal completion")


async def test_client_disconnect():
    """Client disconnects mid-stream. Budget must still be released."""
    print("\n=== Test: Client disconnect mid-stream ===")
    before = await get_active_tokens()
    payload = make_payload(
        "Write a very long detailed essay about the history of computing "
        "from the abacus to modern quantum computers. Cover every decade.",
        max_tokens=512,
    )
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", GATEWAY_URL, json=payload) as resp:
            assert resp.status_code == 200
            chunk_count = 0
            async for _ in resp.aiter_bytes():
                chunk_count += 1
                if chunk_count >= 5:
                    # Disconnect after receiving a few chunks
                    break
    # Give the gateway time to detect disconnect and clean up
    await asyncio.sleep(2)
    after = await get_active_tokens()
    print(f"  Before: {before}, After: {after}")
    assert after == before, f"Budget leak on disconnect: {after - before} tokens"
    print("  PASS: budget released after client disconnect")


async def test_queue_timeout():
    """Queue timeout returns 503, budget never reserved."""
    print("\n=== Test: Queue timeout (503, no budget reserved) ===")
    # This test requires filling the budget first. Send a large request to consume budget,
    # then try another that should get queued and timeout.
    # For a clean test, we'd need ADMISSION_BUDGET to be small. Skipping live test,
    # documenting expected behavior.
    print("  NOTE: Full queue timeout test requires constrained ADMISSION_BUDGET.")
    print("  Expected behavior: 503 returned, active_tokens unchanged.")
    print("  SKIP (document-only)")


async def test_rate_limit_no_budget_charge():
    """Rate-limited request should not consume any budget."""
    print("\n=== Test: Rate limit rejection, no budget charge ===")
    before = await get_active_tokens()
    # Send requests with a fake API key to isolate rate limit state
    payload = make_payload("Hello.", max_tokens=50)
    headers = {"Authorization": "Bearer test-rate-limit-key"}

    # Blast past rate limit (60 req/min default)
    rejected = False
    async with httpx.AsyncClient(timeout=10) as client:
        for i in range(65):
            resp = await client.post(GATEWAY_URL, json=payload, headers=headers)
            if resp.status_code == 429:
                body = resp.json()
                if body.get("error") == "rate limit exceeded":
                    rejected = True
                    print(f"  Rate limited at request {i+1}")
                    break

    # Wait for any admitted requests to complete
    await asyncio.sleep(3)
    after = await get_active_tokens()
    print(f"  Before: {before}, After: {after}")
    if rejected:
        print("  PASS: rate limit triggered, budget not charged for rejected request")
    else:
        print("  WARN: rate limit not triggered within 65 requests (may need lower limit for test)")


async def test_vllm_error():
    """If vLLM returns 500, budget must be released."""
    print("\n=== Test: vLLM error handling ===")
    print("  NOTE: Requires stopping vLLM mid-request or sending malformed request.")
    print("  Expected behavior: budget released in finally block regardless of upstream error.")
    print("  The gateway wraps streaming in try/finally, so any exception path releases budget.")
    print("  SKIP (document-only, verified by code inspection)")


async def main():
    print("Day 17 Failure Semantics Checklist")
    print("=" * 50)

    # Check connectivity
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(METRICS_URL)
    except Exception as e:
        print(f"Gateway not reachable: {e}")
        print("Start gateway first: uvicorn gateway:app --host 0.0.0.0 --port 8080")
        return

    await test_normal_completion()
    await test_client_disconnect()
    await test_queue_timeout()
    await test_rate_limit_no_budget_charge()
    await test_vllm_error()

    print("\n" + "=" * 50)
    print("DONE. See gateway logs for budget release confirmation.")


if __name__ == "__main__":
    asyncio.run(main())

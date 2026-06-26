"""Smoke tests for the KV-memory admission gateway."""
import asyncio
import httpx
import sys

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"
METRICS_URL = "http://localhost:8080/metrics"


def make_payload(content: str, max_tokens: int) -> dict:
    return {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
    }


async def test_small_request():
    """Test 1: Small request admitted, response streams back."""
    print("\n=== Test 1: Small request admitted ===")
    payload = make_payload("Say hello.", max_tokens=50)
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", GATEWAY_URL, json=payload) as resp:
            print(f"Status: {resp.status_code}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            chunks = []
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
            print(f"Received {len(chunks)} chunks, total {sum(len(c) for c in chunks)} bytes")
    # Check metrics after completion
    async with httpx.AsyncClient() as client:
        m = (await client.get(METRICS_URL)).json()
        print(f"Metrics after: active_tokens={m['active_tokens']}")
    print("PASS")


async def test_two_concurrent():
    """Test 2: Two medium requests admitted concurrently."""
    print("\n=== Test 2: Two concurrent medium requests ===")
    payload = make_payload("Write a short poem.", max_tokens=256)

    async def send_one(label: str):
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", GATEWAY_URL, json=payload) as resp:
                print(f"  {label}: status={resp.status_code}")
                assert resp.status_code == 200, f"{label}: Expected 200, got {resp.status_code}"
                async for _ in resp.aiter_bytes():
                    pass
        return True

    results = await asyncio.gather(send_one("req-A"), send_one("req-B"))
    assert all(results)
    print("PASS")


async def test_oversized_rejected():
    """Test 3: Oversized request rejected with 429.

    IMPORTANT: Before running this test, temporarily set ADMISSION_BUDGET=100 in gateway.py
    and restart the gateway. Restore after.
    """
    print("\n=== Test 3: Oversized request rejected (requires ADMISSION_BUDGET=100) ===")
    payload = make_payload("Tell me a long story.", max_tokens=200)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(GATEWAY_URL, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Headers Retry-After: {resp.headers.get('retry-after')}")
        print(f"Body: {resp.json()}")
        assert resp.status_code == 429, f"Expected 429, got {resp.status_code}"
        assert resp.headers.get("retry-after") == "5", "Missing Retry-After header"
        body = resp.json()
        assert body.get("error") == "token budget exceeded", f"Wrong error: {body}"

    # Verify active_tokens was NOT incremented
    async with httpx.AsyncClient() as client:
        m = (await client.get(METRICS_URL)).json()
        print(f"Metrics after rejection: active_tokens={m['active_tokens']}")
        assert m["active_tokens"] == 0, f"active_tokens should be 0, got {m['active_tokens']}"
    print("PASS")


async def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "all"
    if test in ("1", "all"):
        await test_small_request()
    if test in ("2", "all"):
        await test_two_concurrent()
    if test in ("3",):
        await test_oversized_rejected()
    if test == "all":
        print("\nTests 1 and 2 passed. Run test 3 separately with ADMISSION_BUDGET=100.")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

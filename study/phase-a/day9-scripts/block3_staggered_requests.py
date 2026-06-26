import asyncio
import aiohttp
import time


async def send_request(session, request_id, delay):
    """Send a request after a staggered delay. Short prompt, moderate decode."""
    await asyncio.sleep(delay)
    prompt = f"Tell me a short story about character {request_id}"
    t0 = time.time()
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 300,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            latency = time.time() - t0
            print(f"[req {request_id}] latency={latency:.2f}s status={resp.status}")
            return latency
    except Exception as e:
        print(f"[req {request_id}] FAILED: {e}")
        return None


async def main():
    num_requests = 10
    stagger_ms = 200  # 200ms apart

    print(f"Staggered requests: {num_requests} requests, {stagger_ms}ms apart")
    print(f"max_tokens=300, short prompts")
    print()

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, i, i * stagger_ms / 1000)
            for i in range(num_requests)
        ]
        latencies = await asyncio.gather(*tasks)

    print("\nLatency summary:",
          [f"{l:.2f}s" if l else "FAILED" for l in latencies])


asyncio.run(main())

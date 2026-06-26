import asyncio
import aiohttp
import time


async def send_request(session, i):
    prompt = "Tell me a very detailed story about adventure number " + str(i)
    t0 = time.time()
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 1900,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            latency = time.time() - t0
            print(f"[req {i}] latency={latency:.2f}s status={resp.status}")
            return latency
    except Exception as e:
        print(f"[req {i}] FAILED: {e}")
        return None


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(80)]
        latencies = await asyncio.gather(*tasks)
    print("\nLatency summary:", [f"{l:.2f}s" if l else "FAILED" for l in latencies])


asyncio.run(main())

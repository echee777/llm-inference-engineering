"""
Experiment 2: Head-of-Line (HOL) Blocking

Demonstrates the convoy effect: a large request at the FIFO queue head
blocks small requests that would individually fit in budget.

TODO: First run did not produce HOL blocking. The short requests bypassed
the queue entirely because try_admit() succeeded (budget was empty, the
queued large request doesn't hold budget). Fix: add a warm-up phase that
pre-fills ~90% of budget with medium requests before sending the test burst.
This forces both the large AND short requests to fail try_admit() and enter
the queue. When warm-up requests complete and _drain_queue runs, the large
request at the queue head blocks the short ones behind it (convoy effect).

Current setup (does NOT produce HOL blocking):
- Budget at 10,000 tokens
- Large request: 8K prompt + 4K max = 12,288 tokens (cannot be admitted)
- 10 short requests: 100 prompt + 100 max = 200 tokens each (easily fit)

The large request fires first, queues at the head. 10 short requests
queue behind it. FIFO prevents them from being admitted even though
each one fits in budget individually.

Writes raw JSON to results/exp2_results.json.
"""
import asyncio
import json
import os
import time

import httpx

GATEWAY_URL = "http://localhost:8001/v1/chat/completions"
METRICS_URL = "http://localhost:8001/debug/stats"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_prompt(n_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    words = max(1, int(n_tokens * 0.75))
    return (base * (words // 9 + 1))[:words * 5]


async def send_request(
    client: httpx.AsyncClient, name: str, prompt_tokens: int, max_tokens: int
) -> dict:
    """Send one request, measure time-to-admission and TTFT."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(prompt_tokens)}],
        "max_tokens": max_tokens,
        "min_tokens": max(1, max_tokens - 20),
        "stream": True,
    }
    start = time.monotonic()

    try:
        async with client.stream(
            "POST", GATEWAY_URL, json=payload, timeout=30
        ) as resp:
            status = resp.status_code
            if status in (429, 503):
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                elapsed = time.monotonic() - start
                error_msg = ""
                try:
                    error_msg = json.loads(body).get("error", "")
                except Exception:
                    pass
                return {
                    "name": name,
                    "status": status,
                    "outcome": f"rejected ({error_msg})",
                    "ttft_ms": None,
                    "total_s": round(elapsed, 3),
                    "submit_time": start,
                }

            ttft = None
            async for chunk in resp.aiter_bytes():
                if ttft is None:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        if line.startswith("data: ") and line[6:].strip() not in (
                            "",
                            "[DONE]",
                        ):
                            ttft = (time.monotonic() - start) * 1000
                            break
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "status": status,
                "outcome": "admitted",
                "ttft_ms": round(ttft, 1) if ttft else None,
                "total_s": round(elapsed, 3),
                "submit_time": start,
            }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "status": None,
            "outcome": f"error: {e}",
            "ttft_ms": None,
            "total_s": round(elapsed, 3),
            "submit_time": start,
        }


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("HOL Blocking Experiment")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            metrics = await client.get(METRICS_URL)
            print(f"Starting budget: {metrics.json()}")
        except Exception as e:
            print(f"Warning: could not fetch starting metrics: {e}")

        tasks = []

        # Large request at t=0
        async def send_large():
            return await send_request(
                client, "LARGE_8K", prompt_tokens=8192, max_tokens=4096
            )

        tasks.append(asyncio.create_task(send_large()))

        # 10 short requests starting 50ms later, spaced 20ms apart
        for i in range(10):

            async def send_short(idx=i):
                await asyncio.sleep(0.05 + idx * 0.02)
                return await send_request(
                    client, f"SHORT_{idx}", prompt_tokens=100, max_tokens=100
                )

            tasks.append(asyncio.create_task(send_short()))

        results = await asyncio.gather(*tasks)

    # Print results
    print(
        f"\n{'Name':<15} {'Status':<8} {'Outcome':<25} {'TTFT(ms)':<12} {'Total(s)':<10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<15} {str(r['status']):<8} {r['outcome']:<25} "
            f"{str(r['ttft_ms']):<12} {str(r['total_s']):<10}"
        )

    # Analysis
    large = [r for r in results if r["name"].startswith("LARGE")]
    shorts = [r for r in results if r["name"].startswith("SHORT")]
    admitted_shorts = [r for r in shorts if r["outcome"] == "admitted"]
    rejected_shorts = [r for r in shorts if r["outcome"] != "admitted"]

    print(f"\nSummary:")
    print(f"  Large request: {large[0]['outcome']}, TTFT={large[0]['ttft_ms']}ms")
    print(f"  Short requests admitted: {len(admitted_shorts)}/10")
    print(f"  Short requests rejected: {len(rejected_shorts)}/10")
    if admitted_shorts:
        ttfts = [r["ttft_ms"] for r in admitted_shorts if r["ttft_ms"]]
        if ttfts:
            print(
                f"  Short request TTFT: median={sorted(ttfts)[len(ttfts)//2]}ms, "
                f"max={max(ttfts)}ms"
            )
        totals = [r["total_s"] for r in admitted_shorts]
        print(
            f"  Short request total time: median={sorted(totals)[len(totals)//2]}s, "
            f"max={max(totals)}s"
        )

    out_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

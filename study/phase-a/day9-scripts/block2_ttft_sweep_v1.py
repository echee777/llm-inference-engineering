import asyncio
import aiohttp
import time
import json


async def measure_ttft(session, request_id, concurrency_label):
    """Send one streaming request, return TTFT in milliseconds.

    Hits /v1/completions with stream=true. Measures wall time from
    request send to first SSE 'data:' chunk containing token output.
    """
    prompt = f"Tell me a very detailed story about adventure number {request_id}"
    t0 = time.time()
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 3500,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            async for chunk in resp.content:
                line = chunk.decode("utf-8").strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    ttft_ms = (time.time() - t0) * 1000
                    print(
                        f"  [c={concurrency_label} req={request_id}] "
                        f"TTFT={ttft_ms:.0f}ms"
                    )
                    # Drain remaining stream without blocking other coroutines
                    async for _ in resp.content:
                        pass
                    return ttft_ms
            print(f"  [c={concurrency_label} req={request_id}] NO DATA CHUNKS")
            return None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(
            f"  [c={concurrency_label} req={request_id}] "
            f"FAILED after {elapsed:.0f}ms: {e}"
        )
        return None


async def sweep_concurrency_level(concurrency):
    """Fire `concurrency` simultaneous requests, return list of TTFTs."""
    print(f"\n{'='*50}")
    print(f"Concurrency level: {concurrency}")
    print(f"{'='*50}")

    async with aiohttp.ClientSession() as session:
        tasks = [
            measure_ttft(session, i, concurrency)
            for i in range(concurrency)
        ]
        ttfts = await asyncio.gather(*tasks)
    return [t for t in ttfts if t is not None]


def compute_stats(ttfts):
    """Compute mean, min, max, p99 from a list of TTFT values in ms."""
    if not ttfts:
        return {"mean": None, "min": None, "max": None, "p99": None, "n": 0}
    sorted_t = sorted(ttfts)
    n = len(sorted_t)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "mean": sum(sorted_t) / n,
        "min": sorted_t[0],
        "max": sorted_t[-1],
        "p99": sorted_t[p99_idx],
        "n": n,
    }


def print_summary_table(results):
    """Print formatted summary table of all concurrency levels."""
    print(f"\n{'='*70}")
    print("TTFT vs Concurrency — Summary")
    print(f"{'='*70}")
    print(
        f"{'Concurrency':>12} {'N':>4} {'Mean(ms)':>10} {'Min(ms)':>10} "
        f"{'Max(ms)':>10} {'P99(ms)':>10}"
    )
    print("-" * 70)
    for conc, stats in results:
        if stats["mean"] is not None:
            print(
                f"{conc:>12} {stats['n']:>4} {stats['mean']:>10.0f} "
                f"{stats['min']:>10.0f} {stats['max']:>10.0f} "
                f"{stats['p99']:>10.0f}"
            )
        else:
            print(f"{conc:>12} {'—':>4} {'FAILED':>10}")


def print_ascii_chart(results):
    """Print ASCII bar chart of mean TTFT per concurrency level."""
    print(f"\n{'='*70}")
    print("TTFT vs Concurrency — Chart (mean TTFT)")
    print(f"{'='*70}")

    means = [
        (c, s["mean"]) for c, s in results if s["mean"] is not None
    ]
    if not means:
        print("No data to chart.")
        return

    max_mean = max(m for _, m in means)
    bar_width = 40

    for conc, mean in means:
        bar_len = int((mean / max_mean) * bar_width) if max_mean > 0 else 0
        bar = "#" * bar_len
        print(f"  c={conc:>3} | {bar:<{bar_width}} | {mean:.0f}ms")


async def main():
    concurrency_levels = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56]
    drain_delay = 5  # seconds between levels

    print("TTFT vs Concurrency Sweep")
    print(f"Levels: {concurrency_levels}")
    print(f"max_tokens=3500, drain_delay={drain_delay}s")
    print(f"Server: http://localhost:8000")

    results = []
    for conc in concurrency_levels:
        ttfts = await sweep_concurrency_level(conc)
        stats = compute_stats(ttfts)
        results.append((conc, stats))

        if stats["mean"] is not None:
            print(
                f"  >> mean={stats['mean']:.0f}ms  "
                f"min={stats['min']:.0f}ms  "
                f"max={stats['max']:.0f}ms"
            )

        print(f"  Draining {drain_delay}s before next level...")
        await asyncio.sleep(drain_delay)

    print_summary_table(results)
    print_ascii_chart(results)


asyncio.run(main())

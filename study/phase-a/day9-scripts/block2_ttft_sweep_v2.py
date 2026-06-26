import asyncio
import aiohttp
import time


async def background_load(session, request_id):
    """Send a long-decode request that holds blocks for its full duration.

    Does NOT measure anything — just keeps the server busy and KV cache full.
    Returns when the request completes or times out.
    """
    prompt = f"Write an extremely long and detailed essay about topic {request_id}"
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 3500,
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            await resp.read()
    except Exception:
        pass


async def measure_probe_ttft(session, probe_id, bg_level):
    """Send a single streaming probe request, return TTFT in milliseconds.

    This is the request we actually measure — it arrives into a system
    already loaded with background decode work.
    """
    prompt = f"Hello, tell me a quick fact number {probe_id}"
    t0 = time.time()
    try:
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": prompt,
                "max_tokens": 50,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for chunk in resp.content:
                line = chunk.decode("utf-8").strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    ttft_ms = (time.time() - t0) * 1000
                    print(f"  [bg={bg_level} probe={probe_id}] TTFT={ttft_ms:.0f}ms")
                    async for _ in resp.content:
                        pass
                    return ttft_ms
            print(f"  [bg={bg_level} probe={probe_id}] NO DATA CHUNKS")
            return None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(f"  [bg={bg_level} probe={probe_id}] FAILED after {elapsed:.0f}ms: {e}")
        return None


async def sweep_level(bg_count, settle_time=30, num_probes=3):
    """Run one level of the sweep.

    1. Launch bg_count background requests (long decode, hold blocks)
    2. Wait settle_time seconds for blocks to fill
    3. Send num_probes fresh probe requests and measure their TTFT
    4. Wait for everything to finish before returning
    """
    print(f"\n{'='*60}")
    print(f"Background load: {bg_count} requests  |  Settle: {settle_time}s")
    print(f"{'='*60}")

    async with aiohttp.ClientSession() as session:
        # 1. Launch background load
        print(f"  Launching {bg_count} background requests...")
        bg_tasks = [
            asyncio.create_task(background_load(session, i))
            for i in range(bg_count)
        ]

        # 2. Wait for blocks to fill up
        print(f"  Settling {settle_time}s for decode to consume blocks...")
        await asyncio.sleep(settle_time)

        # 3. Send probe requests
        print(f"  Sending {num_probes} probe requests...")
        probe_tasks = [
            measure_probe_ttft(session, i, bg_count)
            for i in range(num_probes)
        ]
        ttfts = await asyncio.gather(*probe_tasks)

        # 4. Wait for background to finish (or cancel if taking too long)
        if bg_tasks:
            print(f"  Waiting for background requests to drain...")
            done, pending = await asyncio.wait(bg_tasks, timeout=300)
            for t in pending:
                t.cancel()
            if pending:
                print(f"  Cancelled {len(pending)} background tasks (timeout)")

    return [t for t in ttfts if t is not None]


def compute_stats(ttfts):
    """Compute mean, min, max from a list of TTFT values in ms."""
    if not ttfts:
        return {"mean": None, "min": None, "max": None, "n": 0}
    sorted_t = sorted(ttfts)
    n = len(sorted_t)
    return {
        "mean": sum(sorted_t) / n,
        "min": sorted_t[0],
        "max": sorted_t[-1],
        "n": n,
    }


def print_summary_table(results):
    """Print formatted summary table."""
    print(f"\n{'='*60}")
    print("TTFT vs Background Load — Summary")
    print(f"{'='*60}")
    print(f"{'BG Load':>8} {'Probes':>7} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    print("-" * 60)
    for bg, stats in results:
        if stats["mean"] is not None:
            print(
                f"{bg:>8} {stats['n']:>7} {stats['mean']:>10.0f} "
                f"{stats['min']:>10.0f} {stats['max']:>10.0f}"
            )
        else:
            print(f"{bg:>8} {'—':>7} {'FAILED':>10}")


def print_ascii_chart(results):
    """Print ASCII bar chart of mean TTFT."""
    print(f"\n{'='*60}")
    print("TTFT vs Background Load — Chart (mean TTFT)")
    print(f"{'='*60}")

    means = [(bg, s["mean"]) for bg, s in results if s["mean"] is not None]
    if not means:
        print("No data to chart.")
        return

    max_mean = max(m for _, m in means)
    bar_width = 40

    for bg, mean in means:
        bar_len = int((mean / max_mean) * bar_width) if max_mean > 0 else 0
        bar = "#" * bar_len
        print(f"  bg={bg:>3} | {bar:<{bar_width}} | {mean:.0f}ms")


async def main():
    # Background load levels to sweep
    # With 12,950 blocks and ~220 blocks per request at 3500 tokens:
    #   20 bg = ~4,400 blocks (34%)
    #   40 bg = ~8,800 blocks (68%)
    #   50 bg = ~11,000 blocks (85%)
    #   55 bg = ~12,100 blocks (93%)
    #   58 bg = ~12,760 blocks (99%) -- near cliff
    bg_levels = [0, 10, 20, 30, 40, 45, 50, 53, 56, 58]
    settle_time = 30  # seconds to let decode fill blocks
    num_probes = 3    # probe requests per level

    print("TTFT vs Background Load Sweep (v2)")
    print(f"BG levels: {bg_levels}")
    print(f"BG requests: max_tokens=3500 (hold ~220 blocks each)")
    print(f"Probe requests: max_tokens=50 (measure TTFT only)")
    print(f"Settle time: {settle_time}s per level")
    print(f"Probes per level: {num_probes}")
    print(f"Server: http://localhost:8000")
    print(f"Expected block pool: ~12,950 blocks")

    results = []
    for bg in bg_levels:
        ttfts = await sweep_level(bg, settle_time, num_probes)
        stats = compute_stats(ttfts)
        results.append((bg, stats))

        if stats["mean"] is not None:
            print(
                f"  >> mean={stats['mean']:.0f}ms  "
                f"min={stats['min']:.0f}ms  "
                f"max={stats['max']:.0f}ms"
            )
        else:
            print("  >> ALL PROBES FAILED")

    print_summary_table(results)
    print_ascii_chart(results)


asyncio.run(main())

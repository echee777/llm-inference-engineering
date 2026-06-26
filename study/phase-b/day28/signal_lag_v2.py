"""
Day 28: Signal lag measurement v2.
Fixed: metric parsing (labels), TTFT via streaming, shorter steps.
"""
import asyncio
import aiohttp
import time
import json
from collections import deque

VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Thresholds
QUEUE_THRESHOLD_A = 5
QUEUE_THRESHOLD_B = 15
KV_THRESHOLD = 0.84       # cliff (87%) - 3pp
TTFT_SLO_MS = 4230.0      # 2x Day 24 baseline p99 (2115ms)

# Shorter ramp: we know cliff is at c=113, just need to measure signal timing
RAMP_STEPS = [
    (50, 120),    # baseline, hold 2 min
    (80, 120),    # medium load, hold 2 min
    (108, 120),   # pre-cliff, hold 2 min
    (125, 120),   # past cliff, hold 2 min
]

metrics_log = []
ttft_buffer = deque(maxlen=200)


async def single_request_stream(session, sem):
    """Send one request via streaming, measure actual TTFT."""
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Explain quantum computing in detail with examples."}],
            "max_tokens": 256,
            "min_tokens": 256,
            "stream": True,
        }
        t0 = time.monotonic()
        ttft = None
        try:
            async with session.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    return
                async for line in resp.content:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                        if ttft is None:
                            ttft = (time.monotonic() - t0) * 1000
                            ttft_buffer.append(ttft)
                        # Keep reading to complete the request
                # Wait for full response to complete (so KV is freed)
        except Exception:
            pass


async def scrape_metrics():
    """Scrape vLLM Prometheus metrics."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{VLLM_BASE}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                text = await resp.text()
                kv_util = None
                queue_depth = None
                for line in text.split('\n'):
                    if 'kv_cache_usage_perc' in line and not line.startswith('#'):
                        # Parse: vllm:kv_cache_usage_perc{...} 0.123
                        kv_util = float(line.split()[-1])
                    elif 'num_requests_waiting' in line and not line.startswith('#'):
                        queue_depth = int(float(line.split()[-1]))
                return kv_util, queue_depth
    except Exception:
        return None, None


async def metrics_collector(stop_event):
    """Collect metrics at 1Hz."""
    while not stop_event.is_set():
        kv_util, queue_depth = await scrape_metrics()
        ttft_p99 = None
        if len(ttft_buffer) >= 5:
            sorted_ttft = sorted(ttft_buffer)
            p99_idx = min(int(len(sorted_ttft) * 0.99), len(sorted_ttft) - 1)
            ttft_p99 = sorted_ttft[p99_idx]

        metrics_log.append({
            "t": time.monotonic(),
            "kv_util": kv_util,
            "queue_depth": queue_depth,
            "ttft_p99": ttft_p99,
        })
        await asyncio.sleep(1)


async def run_ramp():
    print("=== Signal Lag Experiment v2 ===")
    print(f"Queue thresholds: A={QUEUE_THRESHOLD_A}, B={QUEUE_THRESHOLD_B}")
    print(f"KV threshold: {KV_THRESHOLD}")
    print(f"TTFT SLO: {TTFT_SLO_MS}ms")
    print(f"Ramp steps: {RAMP_STEPS}")
    print()

    stop_event = asyncio.Event()
    collector_task = asyncio.create_task(metrics_collector(stop_event))

    t_start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        for step_idx, (concurrency, duration) in enumerate(RAMP_STEPS):
            print(f"Step {step_idx+1}: concurrency={concurrency}, hold={duration}s")
            sem = asyncio.Semaphore(concurrency)

            step_start = time.monotonic()
            request_count = 0

            async def worker():
                nonlocal request_count
                while time.monotonic() - step_start < duration:
                    await single_request_stream(session, sem)
                    request_count += 1

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.gather(*workers)

            elapsed = time.monotonic() - step_start
            print(f"  Completed: {request_count} requests in {elapsed:.1f}s")

            if metrics_log:
                latest = metrics_log[-1]
                print(f"  KV util: {latest['kv_util']:.3f}" if latest['kv_util'] else "  KV util: N/A")
                print(f"  Queue depth: {latest['queue_depth']}")
                print(f"  TTFT p99: {latest['ttft_p99']:.1f}ms" if latest['ttft_p99'] else "  TTFT p99: N/A")
            print()

    stop_event.set()
    await collector_task

    # Analysis
    t_base = metrics_log[0]["t"] if metrics_log else t_start

    t_queue_a = None
    t_queue_b = None
    t_kv = None
    t_ttft_slo = None

    for m in metrics_log:
        if t_queue_a is None and m["queue_depth"] is not None and m["queue_depth"] >= QUEUE_THRESHOLD_A:
            t_queue_a = m["t"] - t_base
        if t_queue_b is None and m["queue_depth"] is not None and m["queue_depth"] >= QUEUE_THRESHOLD_B:
            t_queue_b = m["t"] - t_base
        if t_kv is None and m["kv_util"] is not None and m["kv_util"] >= KV_THRESHOLD:
            t_kv = m["t"] - t_base
        if t_ttft_slo is None and m["ttft_p99"] is not None and m["ttft_p99"] >= TTFT_SLO_MS:
            t_ttft_slo = m["t"] - t_base

    print("=" * 60)
    print("SIGNAL CROSSING TIMES (seconds from start)")
    print("=" * 60)
    print(f"Queue depth >= {QUEUE_THRESHOLD_A}:   t = {t_queue_a:.1f}s" if t_queue_a else f"Queue depth >= {QUEUE_THRESHOLD_A}:   NEVER CROSSED")
    print(f"Queue depth >= {QUEUE_THRESHOLD_B}:  t = {t_queue_b:.1f}s" if t_queue_b else f"Queue depth >= {QUEUE_THRESHOLD_B}:  NEVER CROSSED")
    print(f"KV util >= {KV_THRESHOLD:.0%}:     t = {t_kv:.1f}s" if t_kv else f"KV util >= {KV_THRESHOLD:.0%}:     NEVER CROSSED")
    print(f"TTFT p99 >= {TTFT_SLO_MS:.0f}ms: t = {t_ttft_slo:.1f}s" if t_ttft_slo else f"TTFT p99 >= {TTFT_SLO_MS:.0f}ms: NEVER CROSSED")
    print()

    if t_ttft_slo:
        print("TRIGGER LAG vs TTFT SLO breach (negative = leading)")
        print("-" * 60)
        if t_queue_a is not None:
            lag_a = t_queue_a - t_ttft_slo
            print(f"Queue(>={QUEUE_THRESHOLD_A}) lag:  {lag_a:+.1f}s  {'LEADING' if lag_a < 0 else 'LAGGING'}")
        if t_queue_b is not None:
            lag_b = t_queue_b - t_ttft_slo
            print(f"Queue(>={QUEUE_THRESHOLD_B}) lag: {lag_b:+.1f}s  {'LEADING' if lag_b < 0 else 'LAGGING'}")
        if t_kv is not None:
            lag_kv = t_kv - t_ttft_slo
            print(f"KV util lag:       {lag_kv:+.1f}s  {'LEADING' if lag_kv < 0 else 'LAGGING'}")

        # The key number
        if t_queue_a is not None:
            delta = t_ttft_slo - t_queue_a
            print(f"\nΔ (queue lead time over SLO breach) = {delta:.1f}s")
    else:
        print("TTFT SLO never breached. System did not reach cliff.")

    # Save
    output = {
        "config": {
            "queue_threshold_a": QUEUE_THRESHOLD_A,
            "queue_threshold_b": QUEUE_THRESHOLD_B,
            "kv_threshold": KV_THRESHOLD,
            "ttft_slo_ms": TTFT_SLO_MS,
            "ramp_steps": RAMP_STEPS,
        },
        "crossings": {
            "t_queue_a": t_queue_a,
            "t_queue_b": t_queue_b,
            "t_kv": t_kv,
            "t_ttft_slo": t_ttft_slo,
        },
        "metrics_count": len(metrics_log),
    }
    with open("/tmp/signal_lag_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to /tmp/signal_lag_results.json")


if __name__ == "__main__":
    asyncio.run(run_ramp())

"""
Day 28: Signal lag measurement v3.
Fixed: uses 512-token prompts + min_tokens=256 to match Day 24 cliff config.
Uses non-streaming to measure actual TTFT from usage stats.
"""
import asyncio
import aiohttp
import time
import json
from collections import deque

VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 512-token prompt (matches Day 24)
PROMPT = "x " * 500  # ~530 tokens

# Thresholds
QUEUE_THRESHOLD_A = 5
QUEUE_THRESHOLD_B = 15
KV_THRESHOLD = 0.84       # cliff (87%) - 3pp
TTFT_SLO_MS = 4230.0      # 2x Day 24 baseline p99 (2115ms)

# Step ramp matching Day 24 concurrency levels
# Day 24: cliff at c=113, preemption onset at c=103
RAMP_STEPS = [
    (50, 90),     # well below cliff, hold 90s
    (95, 90),     # Day 24 safe operating point, hold 90s
    (110, 90),    # just below cliff, hold 90s
    (130, 120),   # past cliff, hold 2 min
]

metrics_log = []
ttft_buffer = deque(maxlen=200)


async def single_request(session, sem):
    """Send one request, measure TTFT via streaming first-chunk timing."""
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
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
                # Read to completion so KV is freed properly
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
    print("=== Signal Lag Experiment v3 (512-tok prompts) ===")
    print(f"Prompt tokens: ~530")
    print(f"Queue thresholds: A={QUEUE_THRESHOLD_A}, B={QUEUE_THRESHOLD_B}")
    print(f"KV threshold: {KV_THRESHOLD:.0%}")
    print(f"TTFT SLO: {TTFT_SLO_MS:.0f}ms")
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
                    await single_request(session, sem)
                    request_count += 1

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.gather(*workers)

            elapsed = time.monotonic() - step_start
            print(f"  Completed: {request_count} requests in {elapsed:.1f}s")
            if metrics_log:
                latest = metrics_log[-1]
                kv_str = f"{latest['kv_util']:.3f}" if latest['kv_util'] is not None else "N/A"
                q_str = f"{latest['queue_depth']}" if latest['queue_depth'] is not None else "N/A"
                t_str = f"{latest['ttft_p99']:.1f}ms" if latest['ttft_p99'] is not None else "N/A"
                print(f"  KV util: {kv_str}, Queue: {q_str}, TTFT p99: {t_str}")
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
    print(f"Queue depth >= {QUEUE_THRESHOLD_A}:   t = {t_queue_a:.1f}s" if t_queue_a is not None else f"Queue depth >= {QUEUE_THRESHOLD_A}:   NEVER CROSSED")
    print(f"Queue depth >= {QUEUE_THRESHOLD_B}:  t = {t_queue_b:.1f}s" if t_queue_b is not None else f"Queue depth >= {QUEUE_THRESHOLD_B}:  NEVER CROSSED")
    print(f"KV util >= {KV_THRESHOLD:.0%}:     t = {t_kv:.1f}s" if t_kv is not None else f"KV util >= {KV_THRESHOLD:.0%}:     NEVER CROSSED")
    print(f"TTFT p99 >= {TTFT_SLO_MS:.0f}ms: t = {t_ttft_slo:.1f}s" if t_ttft_slo is not None else f"TTFT p99 >= {TTFT_SLO_MS:.0f}ms: NEVER CROSSED")
    print()

    if t_ttft_slo is not None:
        print("TRIGGER LAG vs TTFT SLO breach (negative = leading)")
        print("-" * 60)
        if t_queue_a is not None:
            lag_a = t_queue_a - t_ttft_slo
            print(f"Queue(>={QUEUE_THRESHOLD_A}) lag:  {lag_a:+.1f}s  {'LEADING' if lag_a < 0 else 'LAGGING'}")
            delta_a = t_ttft_slo - t_queue_a
            print(f"  -> Δ = {delta_a:.1f}s")
        if t_queue_b is not None:
            lag_b = t_queue_b - t_ttft_slo
            print(f"Queue(>={QUEUE_THRESHOLD_B}) lag: {lag_b:+.1f}s  {'LEADING' if lag_b < 0 else 'LAGGING'}")
        if t_kv is not None:
            lag_kv = t_kv - t_ttft_slo
            print(f"KV util lag:       {lag_kv:+.1f}s  {'LEADING' if lag_kv < 0 else 'LAGGING or COINCIDENT'}")
    else:
        print("TTFT SLO never breached. Need higher concurrency or lower SLO threshold.")

    # Peak values
    max_kv = max((m["kv_util"] for m in metrics_log if m["kv_util"] is not None), default=0)
    max_q = max((m["queue_depth"] for m in metrics_log if m["queue_depth"] is not None), default=0)
    max_ttft = max((m["ttft_p99"] for m in metrics_log if m["ttft_p99"] is not None), default=0)
    print(f"\nPeak values: KV={max_kv:.3f}, Queue={max_q}, TTFT p99={max_ttft:.0f}ms")

    # Save
    with open("/tmp/signal_lag_v3_results.json", "w") as f:
        json.dump({
            "crossings": {"t_queue_a": t_queue_a, "t_queue_b": t_queue_b, "t_kv": t_kv, "t_ttft_slo": t_ttft_slo},
            "peaks": {"max_kv": max_kv, "max_queue": max_q, "max_ttft_p99": max_ttft},
            "metrics_count": len(metrics_log),
        }, f, indent=2)
    print(f"\nResults saved to /tmp/signal_lag_v3_results.json")


if __name__ == "__main__":
    asyncio.run(run_ramp())

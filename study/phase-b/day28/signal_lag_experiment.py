"""
Day 28: Signal lag measurement.
Step-function load ramp, recording queue depth + KV util + TTFT at 1Hz.
Measures how many seconds queue depth crosses threshold BEFORE TTFT crosses SLO.

Uses same config as Day 24 (gpu_memory_utilization=0.60) so we can reach cliff.
"""
import asyncio
import aiohttp
import time
import json
import statistics
from collections import deque

VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Thresholds from Day 24 + syllabus
QUEUE_THRESHOLD_A = 5    # Conservative
QUEUE_THRESHOLD_B = 15   # Aggressive
KV_THRESHOLD = 0.84      # cliff (87%) - 3pp
TTFT_SLO_MS = 4230.0     # 2x Day 24 baseline p99 (2115ms)

# Ramp steps: concurrency levels matching Day 24 sweep
# 40% of cliff(113) = ~45, 60% = ~68, cliff-2pp = ~108, cliff+3pp = ~120
RAMP_STEPS = [
    (45, 300),    # 40% of cliff, hold 5 min (baseline)
    (68, 180),    # 60%, hold 3 min
    (108, 180),   # cliff - 2pp, hold 3 min
    (120, 180),   # cliff + 3pp, hold 3 min (or until collapse)
]

# Tracking
metrics_log = []  # (timestamp, kv_util, queue_depth, ttft_latest, concurrency)
ttft_buffer = deque(maxlen=100)  # last 100 TTFT measurements for rolling p99


async def single_request(session, sem):
    """Send one request, record TTFT, return immediately for next."""
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Explain quantum computing in detail."}],
            "max_tokens": 256,
            "min_tokens": 256,
            "stream": False,
        }
        t0 = time.monotonic()
        try:
            async with session.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Extract TTFT from usage if available, else use total time as proxy
                    ttft_ms = (time.monotonic() - t0) * 1000
                    # vLLM returns time_to_first_token in some versions
                    usage = data.get("usage", {})
                    ttft_buffer.append(ttft_ms)
                    return True
                else:
                    return False
        except Exception:
            return False


async def scrape_metrics():
    """Scrape vLLM Prometheus metrics for KV util and queue depth."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{VLLM_BASE}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                text = await resp.text()
                kv_util = None
                queue_depth = None
                for line in text.split('\n'):
                    if line.startswith('vllm:gpu_cache_usage_perc'):
                        kv_util = float(line.split()[-1])
                    elif line.startswith('vllm:num_requests_waiting'):
                        queue_depth = int(float(line.split()[-1]))
                return kv_util, queue_depth
    except Exception:
        return None, None


async def metrics_collector(stop_event):
    """Collect metrics at 1Hz."""
    while not stop_event.is_set():
        kv_util, queue_depth = await scrape_metrics()
        ttft_p99 = None
        if len(ttft_buffer) >= 10:
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
    """Run the step-function ramp and record all signals."""
    print("=== Signal Lag Experiment ===")
    print(f"Queue threshold A: {QUEUE_THRESHOLD_A}")
    print(f"Queue threshold B: {QUEUE_THRESHOLD_B}")
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

            # Keep firing requests at this concurrency for `duration` seconds
            async def worker():
                nonlocal request_count
                while time.monotonic() - step_start < duration:
                    await single_request(session, sem)
                    request_count += 1

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.gather(*workers)

            elapsed = time.monotonic() - step_start
            print(f"  Completed: {request_count} requests in {elapsed:.1f}s")

            # Print current state
            if metrics_log:
                latest = metrics_log[-1]
                print(f"  KV util: {latest['kv_util']}, Queue: {latest['queue_depth']}, TTFT p99: {latest['ttft_p99']}")
            print()

    stop_event.set()
    await collector_task

    # Analysis: find crossing timestamps
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

    print("=== SIGNAL CROSSING TIMES (seconds from start) ===")
    print(f"Queue depth >= {QUEUE_THRESHOLD_A}:  t = {t_queue_a:.1f}s" if t_queue_a else f"Queue depth >= {QUEUE_THRESHOLD_A}:  NEVER CROSSED")
    print(f"Queue depth >= {QUEUE_THRESHOLD_B}: t = {t_queue_b:.1f}s" if t_queue_b else f"Queue depth >= {QUEUE_THRESHOLD_B}: NEVER CROSSED")
    print(f"KV util >= {KV_THRESHOLD}:        t = {t_kv:.1f}s" if t_kv else f"KV util >= {KV_THRESHOLD}:        NEVER CROSSED")
    print(f"TTFT p99 >= {TTFT_SLO_MS}ms:    t = {t_ttft_slo:.1f}s" if t_ttft_slo else f"TTFT p99 >= {TTFT_SLO_MS}ms:    NEVER CROSSED")
    print()

    if t_ttft_slo:
        print("=== TRIGGER LAG (negative = leading) ===")
        if t_queue_a:
            print(f"Queue(>=5) lag:   {t_queue_a - t_ttft_slo:.1f}s {'(LEADING)' if t_queue_a < t_ttft_slo else '(LAGGING)'}")
        if t_queue_b:
            print(f"Queue(>=15) lag:  {t_queue_b - t_ttft_slo:.1f}s {'(LEADING)' if t_queue_b < t_ttft_slo else '(LAGGING)'}")
        if t_kv:
            print(f"KV util lag:      {t_kv - t_ttft_slo:.1f}s {'(LEADING)' if t_kv < t_ttft_slo else '(LAGGING)'}")
    else:
        print("TTFT SLO never breached -- system did not reach cliff. Increase load or lower SLO.")

    # Save full metrics to file
    with open("/tmp/signal_lag_metrics.json", "w") as f:
        json.dump(metrics_log, f, default=str)
    print(f"\nFull metrics saved to /tmp/signal_lag_metrics.json ({len(metrics_log)} samples)")


if __name__ == "__main__":
    asyncio.run(run_ramp())

"""
Day 25 -- Retry client for Week 6 retry storm experiments.

Deliberately has NO backoff. Day 26 measures unmitigated retry amplification.
Day 27 adds backoff to measure the delta.
"""

import time
import uuid
import aiohttp
import asyncio
import json
import statistics

VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

TIMEOUT_SECS = 2.0
MAX_RETRIES = 3

attempt_log = []  # (request_id, attempt_number, outcome, latency_ms)


async def send_with_retry(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 64,
) -> dict | None:
    """Send a request with up to MAX_RETRIES attempts. No backoff.

    Returns the parsed response on success, None if all attempts fail.
    Each attempt is logged regardless of outcome.
    """
    request_id = str(uuid.uuid4())
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            async with session.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS),
            ) as resp:
                latency_ms = (time.monotonic() - t0) * 1000
                if resp.status == 200:
                    data = await resp.json()
                    attempt_log.append((request_id, attempt, "success", latency_ms))
                    return data
                else:
                    body = await resp.text()
                    attempt_log.append(
                        (request_id, attempt, f"http_{resp.status}", latency_ms)
                    )
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - t0) * 1000
            attempt_log.append((request_id, attempt, "timeout", latency_ms))
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            attempt_log.append(
                (request_id, attempt, f"error:{str(e)[:80]}", latency_ms)
            )

    return None


def amplification_factor() -> float:
    """Total attempts / unique requests. 1.0 means no retries fired."""
    total_attempts = len(attempt_log)
    unique_requests = len(set(r[0] for r in attempt_log))
    return total_attempts / unique_requests if unique_requests > 0 else 1.0


def print_summary():
    """Print retry statistics."""
    total = len(attempt_log)
    unique = len(set(r[0] for r in attempt_log))
    successes = sum(1 for r in attempt_log if r[2] == "success")
    timeouts = sum(1 for r in attempt_log if r[2] == "timeout")
    errors = sum(1 for r in attempt_log if r[2].startswith("error:"))
    http_errors = sum(1 for r in attempt_log if r[2].startswith("http_"))

    print(f"\n{'='*60}")
    print("RETRY CLIENT SUMMARY")
    print(f"{'='*60}")
    print(f"Unique requests:     {unique}")
    print(f"Total attempts:      {total}")
    print(f"Amplification factor: {amplification_factor():.3f}")
    print(f"Successes:           {successes}")
    print(f"Timeouts:            {timeouts}")
    print(f"HTTP errors:         {http_errors}")
    print(f"Other errors:        {errors}")

    success_latencies = [r[3] for r in attempt_log if r[2] == "success"]
    if success_latencies:
        print(f"Latency p50:         {statistics.median(success_latencies):.1f} ms")
        sorted_lat = sorted(success_latencies)
        p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        print(f"Latency p99:         {sorted_lat[p99_idx]:.1f} ms")
    print(f"{'='*60}")


async def smoke_test(concurrency: int = 4, num_requests: int = 20):
    """Run a low-load smoke test. Amplification should be ~1.0."""
    print(f"Smoke test: {num_requests} requests at concurrency={concurrency}")
    print(f"Timeout: {TIMEOUT_SECS}s, Max retries: {MAX_RETRIES}")

    sem = asyncio.Semaphore(concurrency)

    async def worker(i: int):
        async with sem:
            prompt = f"Say hello. Request number {i}."
            await send_with_retry(session, prompt, max_tokens=16)

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)

    print_summary()

    af = amplification_factor()
    if af > 1.05:
        print(f"\nWARNING: amplification factor {af:.3f} > 1.05 at low load.")
        print("Timeout is misconfigured or vLLM is unresponsive. Fix before Day 26.")
    else:
        print(f"\nPASS: amplification factor {af:.3f} <= 1.05")


if __name__ == "__main__":
    asyncio.run(smoke_test())

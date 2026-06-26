#!/usr/bin/env python3
"""
Day 22 Afternoon Experiment: Mixed-Length Scheduling Interference

Run A: 50/50 short (64-tok prompt, 128 max_new) + long (2048-tok prompt, 512 max_new)
Run B: Short-only control, matched to same KV utilization as Run A

Target: ~65% KV utilization at gpu_memory_utilization=0.90
Metric: interference_penalty = p99_TTFT(short, mixed) / p99_TTFT(short, isolated)
"""

import asyncio
import aiohttp
import json
import time
import csv
import argparse
import random
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

# --- Request specs ---
SHORT_PROMPT_TOKENS = 64
SHORT_MAX_NEW = 128
LONG_PROMPT_TOKENS = 2048
LONG_MAX_NEW = 512


def make_prompt(num_tokens: int) -> str:
    """Generate a prompt of approximately num_tokens tokens.

    Uses repeated simple words (~1 token each) to get close to target length.
    """
    # "hello " is reliably 1-2 tokens. Overshoot slightly and let the tokenizer handle it.
    return "hello " * num_tokens


async def send_request(
    session: aiohttp.ClientSession,
    req_type: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict:
    """Send a streaming request to vLLM and measure TTFT.

    Returns dict with: request_id, type, ttft_ms, tokens_received, total_ms, error
    """
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    t_first_token = None
    tokens_received = 0

    try:
        async with session.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "request_id": request_id,
                    "type": req_type,
                    "ttft_ms": None,
                    "tokens_received": 0,
                    "total_ms": (time.perf_counter() - t_start) * 1000,
                    "error": f"HTTP {resp.status}: {body[:200]}",
                }

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text and t_first_token is None:
                        t_first_token = time.perf_counter()
                    if text:
                        tokens_received += 1
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return {
            "request_id": request_id,
            "type": req_type,
            "ttft_ms": None,
            "tokens_received": 0,
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "error": str(e),
        }

    t_end = time.perf_counter()
    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None

    return {
        "request_id": request_id,
        "type": req_type,
        "ttft_ms": ttft_ms,
        "tokens_received": tokens_received,
        "total_ms": (t_end - t_start) * 1000,
        "error": None,
    }


async def run_experiment(
    mode: str,
    concurrency: int,
    duration_s: int,
    output_csv: str,
):
    """Run the experiment for `duration_s` seconds at `concurrency` in-flight requests."""

    short_prompt = make_prompt(SHORT_PROMPT_TOKENS)
    long_prompt = make_prompt(LONG_PROMPT_TOKENS)

    results = []
    request_counter = 0
    counter_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    async def typed_worker(session, req_type, prompt, max_tokens):
        """Each worker is dedicated to one request type and loops until stopped."""
        nonlocal request_counter
        while not stop_event.is_set():
            async with counter_lock:
                req_id = request_counter
                request_counter += 1

            result = await send_request(session, req_type, prompt, max_tokens, req_id)
            results.append(result)

            if result["error"]:
                print(f"  [!] req {req_id} ({req_type}): {result['error']}")
            elif req_id % 20 == 0:
                print(
                    f"  req {req_id} ({req_type}): "
                    f"TTFT={result['ttft_ms']:.0f}ms, "
                    f"tokens={result['tokens_received']}, "
                    f"total={result['total_ms']:.0f}ms"
                )

    if mode == "mixed":
        # Split concurrency pool: half short, half long
        # Guarantees 50/50 in-flight at all times
        short_slots = concurrency // 2
        long_slots = concurrency - short_slots
    else:
        short_slots = concurrency
        long_slots = 0

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _ in range(short_slots):
            tasks.append(asyncio.create_task(
                typed_worker(session, "short", short_prompt, SHORT_MAX_NEW)
            ))
        for _ in range(long_slots):
            tasks.append(asyncio.create_task(
                typed_worker(session, "long", long_prompt, LONG_MAX_NEW)
            ))

        # Background KV utilization poller
        kv_samples = []

        async def poll_kv_util():
            while not stop_event.is_set():
                try:
                    async with session.get(
                        f"{BASE_URL}/metrics", timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        text = await resp.text()
                        for line in text.split("\n"):
                            if line.startswith("vllm:kv_cache_usage_perc{"):
                                val = float(line.split()[-1])
                                kv_samples.append(val)
                                break
                except Exception:
                    pass
                await asyncio.sleep(5)

        kv_task = asyncio.create_task(poll_kv_util())

        print(f"Running {mode} experiment: concurrency={concurrency}, duration={duration_s}s")
        print(f"  Short slots: {short_slots}, Long slots: {long_slots}")
        print(f"  Short: {SHORT_PROMPT_TOKENS} prompt tok, {SHORT_MAX_NEW} max_new")
        if mode == "mixed":
            print(f"  Long:  {LONG_PROMPT_TOKENS} prompt tok, {LONG_MAX_NEW} max_new")

        await asyncio.sleep(duration_s)
        stop_event.set()
        kv_task.cancel()

        # Wait for in-flight requests to finish
        print("Stopping... waiting for in-flight requests")
        await asyncio.gather(*tasks, return_exceptions=True)

    # Write CSV
    csv_path = Path(output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["request_id", "type", "ttft_ms", "tokens_received", "total_ms", "error"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"Results: {len(results)} total requests -> {csv_path}")

    for req_type in ["short", "long"]:
        typed = [r for r in results if r["type"] == req_type and r["ttft_ms"] is not None]
        if not typed:
            continue
        ttfts = sorted([r["ttft_ms"] for r in typed])
        p50 = ttfts[len(ttfts) // 2]
        p99 = ttfts[int(len(ttfts) * 0.99)]
        print(f"\n  {req_type} requests: n={len(typed)}")
        print(f"    TTFT p50={p50:.1f}ms  p99={p99:.1f}ms")
        print(f"    TTFT min={ttfts[0]:.1f}ms  max={ttfts[-1]:.1f}ms")

    if kv_samples:
        avg_kv = sum(kv_samples) / len(kv_samples) * 100
        max_kv = max(kv_samples) * 100
        print(f"\n  KV utilization: avg={avg_kv:.1f}%, max={max_kv:.1f}% ({len(kv_samples)} samples)")

    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\n  Errors: {len(errors)}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Mixed-length scheduling interference experiment")
    parser.add_argument("--mode", choices=["mixed", "short-only"], required=True,
                        help="'mixed' for Run A (50/50), 'short-only' for Run B (control)")
    parser.add_argument("--concurrency", type=int, required=True,
                        help="Number of concurrent in-flight requests")
    parser.add_argument("--duration", type=int, default=600,
                        help="Test duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: run_a_mixed.csv or run_b_short.csv)")
    args = parser.parse_args()

    if args.output is None:
        if args.mode == "mixed":
            args.output = "day22/run_a_mixed.csv"
        else:
            args.output = "day22/run_b_short.csv"

    asyncio.run(run_experiment(args.mode, args.concurrency, args.duration, args.output))


if __name__ == "__main__":
    main()

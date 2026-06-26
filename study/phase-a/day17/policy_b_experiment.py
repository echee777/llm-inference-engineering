"""Day 17: Policy B correction experiment.

Compares baseline (reserve max, release at completion) vs Policy B (periodic release
with safety floor). Sends 100 requests and measures admission rate improvement.

Usage:
    python policy_b_experiment.py baseline   # run with Policy B disabled
    python policy_b_experiment.py policyb    # run with Policy B enabled
    python policy_b_experiment.py compare    # run both back-to-back
"""
import asyncio
import sys
import time

import httpx

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"
METRICS_URL = "http://localhost:8080/metrics"

# Prompts that produce ~100-200 token completions against max_tokens=512.
# This maximizes the gap between reserved and actual, which is what Policy B corrects.
PROMPTS = [
    "Explain what a hash table is in 2-3 sentences.",
    "What is the difference between a stack and a queue? Brief answer.",
    "Describe TCP three-way handshake in one paragraph.",
    "What is virtual memory? Answer briefly.",
    "Explain what a mutex is in simple terms.",
    "What does DNS do? One paragraph.",
    "Describe the difference between processes and threads.",
    "What is a B-tree used for? Brief answer.",
    "Explain eventual consistency in 2 sentences.",
    "What is a page fault? Brief answer.",
]

N_REQUESTS = 100
MAX_TOKENS = 512
CONCURRENCY = 50  # max concurrent requests in flight (high to stress budget)


async def send_request(client, prompt, max_tokens, semaphore):
    async with semaphore:
        body = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            async with client.stream("POST", GATEWAY_URL, json=body) as resp:
                status = resp.status_code
                chunks = 0
                async for _ in resp.aiter_bytes():
                    chunks += 1
                return {"status": status, "chunks": chunks}
        except Exception as e:
            return {"status": "error", "error": str(e)}


async def run_experiment(label: str, n_requests: int):
    print(f"\n{'='*50}")
    print(f"Running: {label} ({n_requests} requests, concurrency={CONCURRENCY})")
    print(f"{'='*50}")

    # Reset stats via metrics read (just record starting point)
    async with httpx.AsyncClient(timeout=5) as c:
        before = (await c.get(METRICS_URL)).json()
    print(f"  Before: {before}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    start = time.time()

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for i in range(n_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(send_request(client, prompt, MAX_TOKENS, semaphore))
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    # Tally results
    admitted = sum(1 for r in results if r["status"] == 200)
    rejected_429 = sum(1 for r in results if r["status"] == 429)
    rejected_503 = sum(1 for r in results if r["status"] == 503)
    errors = sum(1 for r in results if r["status"] == "error")

    async with httpx.AsyncClient(timeout=5) as c:
        after = (await c.get(METRICS_URL)).json()

    print(f"\n  Results ({label}):")
    print(f"    Admitted:     {admitted}")
    print(f"    Rejected 429: {rejected_429}")
    print(f"    Rejected 503: {rejected_503}")
    print(f"    Errors:       {errors}")
    print(f"    Elapsed:      {elapsed:.1f}s")
    print(f"    Metrics after: {after}")

    return {
        "label": label,
        "admitted": admitted,
        "rejected_429": rejected_429,
        "rejected_503": rejected_503,
        "errors": errors,
        "elapsed": elapsed,
        "correction_delta_released": after.get("correction_delta_released", 0),
    }


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"

    # Check connectivity
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(METRICS_URL)
    except Exception as e:
        print(f"Gateway not reachable: {e}")
        return

    if mode == "baseline":
        await run_experiment("Baseline (no Policy B)", N_REQUESTS)
    elif mode == "policyb":
        await run_experiment("Policy B (release every 50, floor 64)", N_REQUESTS)
    elif mode == "compare":
        print("NOTE: For a true A/B comparison, run baseline with RELEASE_INTERVAL=999999")
        print("(effectively disabling Policy B), then restart gateway with normal settings")
        print("and run policyb. This script runs both sequentially for convenience,")
        print("but the gateway config doesn't change between runs.\n")
        r1 = await run_experiment("Run 1", N_REQUESTS)
        await asyncio.sleep(5)
        r2 = await run_experiment("Run 2", N_REQUESTS)

        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")
        print(f"  Run 1: admitted={r1['admitted']}, rejected={r1['rejected_429']+r1['rejected_503']}, "
              f"correction_freed={r1['correction_delta_released']}")
        print(f"  Run 2: admitted={r2['admitted']}, rejected={r2['rejected_429']+r2['rejected_503']}, "
              f"correction_freed={r2['correction_delta_released']}")
    else:
        print(f"Unknown mode: {mode}. Use: baseline, policyb, or compare")


if __name__ == "__main__":
    asyncio.run(main())

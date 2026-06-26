#!/usr/bin/env python3
"""
Day 13: Prefix Caching Experiment
6-condition design: cache ON/OFF x shared prefix ratio sweep x system prompt length sweep.

Usage:
  python day13_prefix_cache.py                  # run all syslens
  python day13_prefix_cache.py --syslen 1000    # run single syslen (use between vLLM restarts)
"""

import asyncio
import aiohttp
import time
import sys
import argparse
import random

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
CONCURRENCY = 8
NUM_REQUESTS = 40       # per condition (post-warmup)
WARMUP_REQUESTS = 8     # populate cache before measuring steady-state
OUTPUT_TOKENS = 64
USER_MESSAGE = "What is the capital of France and why is it historically significant?"

# System prompt lengths to sweep
ALL_SYSTEM_PROMPT_LENGTHS = [100, 500, 1000, 2000]

# Shared prefix ratios (fraction of requests that share the "canonical" system prompt)
HIT_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.00]


def make_system_prompt(n_tokens: int, variant: int = 0) -> str:
    """
    Generate a system prompt of approximately n_tokens tokens.
    variant=0  -> canonical (shared) prefix
    variant>0  -> unique prefix (different enough to guarantee cache miss)
    """
    words_needed = int(n_tokens / 0.75)
    if variant == 0:
        base = "You are a highly capable assistant. Please answer questions carefully. "
    else:
        # Unique variant: prefix with variant index to guarantee hash mismatch
        base = f"[Session {variant:06d}] You are a capable assistant. Answer carefully. "
    while len(base.split()) < words_needed:
        base += "Provide accurate, concise, and well-reasoned responses at all times. "
    return base.strip()


def percentile(data, p):
    if not data:
        return float('nan')
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


async def send_request(session, semaphore, system_prompt, user_msg):
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        t_start = time.perf_counter()
        ttft = None
        t_end = None
        total_tokens = 0
        try:
            async with session.post(
                VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        total_tokens += 1
                t_end = time.perf_counter()
            return {
                "status": "ok",
                "ttft": ttft,
                "e2e": t_end - t_start if t_end else None,
                "tokens": total_tokens,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


async def run_condition(
    syslen: int,
    hit_ratio: float,
    num_requests: int,
    warmup: int,
    condition_idx: int = 0,
):
    canonical_prompt = make_system_prompt(syslen, variant=0)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Determine per-request prompt assignment
    # hit_ratio fraction get canonical; remainder get unique variants
    # Offset variant IDs by condition index so no two conditions share prompts
    variant_base = condition_idx * num_requests
    prompts = []
    for i in range(num_requests):
        if i / num_requests < hit_ratio:
            prompts.append(canonical_prompt)
        else:
            prompts.append(make_system_prompt(syslen, variant=variant_base + i + 1))
    random.shuffle(prompts)

    async with aiohttp.ClientSession() as session:
        # Warmup: send canonical prompt requests to populate cache
        warmup_sem = asyncio.Semaphore(1)
        for _ in range(warmup):
            await send_request(session, warmup_sem, canonical_prompt, USER_MESSAGE)

        # Measured run
        tasks = [
            send_request(session, semaphore, p, USER_MESSAGE)
            for p in prompts
        ]
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r["status"] == "ok"]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    e2es = [r["e2e"] for r in ok if r["e2e"] is not None]
    total_tokens = sum(r["tokens"] for r in ok)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "syslen": syslen,
        "hit_ratio": hit_ratio,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "ttft_p50": percentile(ttfts, 50) * 1000,
        "ttft_p99": percentile(ttfts, 99) * 1000,
        "e2e_p50": percentile(e2es, 50) * 1000,
        "e2e_p95": percentile(e2es, 95) * 1000,
        "e2e_p99": percentile(e2es, 99) * 1000,
        "throughput_tok_s": throughput,
    }


def print_header(syslens):
    print("=" * 110)
    print("Day 13: Prefix Caching Experiment - 6-Condition Design")
    print("=" * 110)
    print(f"  Model: {MODEL}  |  Concurrency: {CONCURRENCY}  |  "
          f"Output tokens: {OUTPUT_TOKENS}  |  Requests/condition: {NUM_REQUESTS}")
    print(f"  System prompt lengths: {syslens}")
    print("=" * 110)


def print_row(r):
    print(
        f"  syslen={r['syslen']:5d} | hit={r['hit_ratio']:.0%} | "
        f"TTFT p50={r['ttft_p50']:7.1f}ms | p99={r['ttft_p99']:7.1f}ms | "
        f"E2E p99={r['e2e_p99']:7.1f}ms | "
        f"tok/s={r['throughput_tok_s']:6.1f} | "
        f"ok={r['n_ok']}/{r['n_ok']+r['n_err']}"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syslen", type=int, default=None,
                        help="Run a single syslen (for use with vLLM restarts between syslens)")
    args = parser.parse_args()

    if args.syslen:
        syslens = [args.syslen]
    else:
        syslens = ALL_SYSTEM_PROMPT_LENGTHS

    print_header(syslens)

    all_results = []

    for syslen in syslens:
        print(f"\n-- System prompt length: {syslen} tokens --")
        print(f"  {'Condition':<10} {'HitRatio':>9} | "
              f"{'TTFT p50':>10} | {'TTFT p99':>10} | "
              f"{'E2E p99':>9} | {'tok/s':>7} | ok/total")
        print("  " + "-" * 95)

        for ci, hit_ratio in enumerate(HIT_RATIOS):
            r = await run_condition(syslen, hit_ratio, NUM_REQUESTS, WARMUP_REQUESTS, condition_idx=ci)
            all_results.append(r)
            print_row(r)

    print()


if __name__ == "__main__":
    asyncio.run(main())

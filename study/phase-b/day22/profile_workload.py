#!/usr/bin/env python3
"""Short workload for nsys/ncu profiling. Sends a small burst of requests."""

import asyncio
import aiohttp
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def make_prompt(num_tokens):
    return "hello " * num_tokens

async def send_one(session, prompt, max_tokens, label):
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    t_first = None
    tokens = 0
    async with session.post(f"{BASE_URL}/v1/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)) as resp:
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
                if text and t_first is None:
                    t_first = time.perf_counter()
                if text:
                    tokens += 1
            except json.JSONDecodeError:
                continue
    ttft = (t_first - t0) * 1000 if t_first else 0
    total = (time.perf_counter() - t0) * 1000
    print(f"  {label}: TTFT={ttft:.0f}ms, tokens={tokens}, total={total:.0f}ms")

async def run(mode):
    short_prompt = make_prompt(64)
    long_prompt = make_prompt(2048)

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        if mode == "mixed":
            # 8 short + 8 long concurrent
            tasks = []
            for i in range(8):
                tasks.append(send_one(session, short_prompt, 64, f"short-{i}"))
                tasks.append(send_one(session, long_prompt, 64, f"long-{i}"))
            await asyncio.gather(*tasks)
        else:
            # 16 short concurrent
            tasks = [send_one(session, short_prompt, 64, f"short-{i}") for i in range(16)]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    print(f"Running {mode} profile workload...")
    asyncio.run(run(mode))

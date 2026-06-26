"""
Day 21 KV Cache Exhaustion Benchmark
Runs baseline and ramp-to-failure experiments against vLLM serving Qwen2.5-3B-Instruct on T4.
Collects TTFT, throughput, KV utilization, queue depth, preemption count per concurrency level.
"""

import asyncio
import aiohttp
import json
import time
import sys
import statistics
import argparse
from dataclasses import dataclass, field


VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class RequestResult:
    ttft_ms: float
    total_ms: float
    output_tokens: int
    error: str = ""


@dataclass
class LevelResult:
    concurrency: int
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    throughput_tps: float = 0.0
    kv_util_pct: float = 0.0
    queue_depth: int = 0
    running_reqs: int = 0
    effective_concurrency: int = 0
    preemptions: int = 0
    divergence_ratio: float = 0.0
    errors: list = field(default_factory=list)
    completed: int = 0


_prompt_counter = 0


def make_prompt(token_count: int, unique: bool = False) -> str:
    """Generate a prompt that approximates the target token count.
    Uses diverse sequential numbers to resist BPE compression.
    Qwen tokenizer: each number averages ~3.5 tokens (varies with digit count).
    If unique=True, offset the number range to defeat prefix caching.
    """
    global _prompt_counter
    num_count = max(1, int(token_count / 3.5))
    if unique:
        _prompt_counter += 1
        offset = _prompt_counter * num_count
        numbers = [str(i + offset) for i in range(num_count)]
    else:
        numbers = [str(i) for i in range(num_count)]
    return " ".join(numbers)


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a streaming chat completion request, measure TTFT and output tokens."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    start = time.monotonic()
    ttft = 0.0
    output_tokens = 0

    try:
        async with session.post(
            f"{VLLM_BASE}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()  # get all response
                return RequestResult(
                    ttft_ms=0,
                    total_ms=0,
                    output_tokens=0,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        if ttft == 0.0:
                            ttft = (time.monotonic() - start) * 1000
                        output_tokens += 1
                except (
                    json.JSONDecodeError,
                    KeyError,
                    IndexError,
                ):  # exception handling
                    continue

        total_ms = (time.monotonic() - start) * 1000
        return RequestResult(
            ttft_ms=ttft, total_ms=total_ms, output_tokens=output_tokens
        )

    except asyncio.TimeoutError:
        return RequestResult(
            ttft_ms=0, total_ms=0, output_tokens=0, error="timeout_300s"
        )
    except Exception as e:
        return RequestResult(ttft_ms=0, total_ms=0, output_tokens=0, error=str(e)[:200])


async def fetch_metrics() -> dict:
    """Scrape vLLM /metrics endpoint for KV cache and scheduler stats."""
    metrics = {
        "kv_util_pct": 0.0,
        "queue_depth": 0,
        "running_reqs": 0,
        "preemptions": 0,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{VLLM_BASE}/metrics", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                text = await resp.text()
                for line in text.split("\n"):
                    if line.startswith("#"):
                        continue
                    # KV cache utilization (vLLM 0.17+ metric name)
                    if "vllm:kv_cache_usage_perc{" in line:
                        try:
                            val = float(line.split()[-1])
                            metrics["kv_util_pct"] = val * 100
                        except (ValueError, IndexError):
                            pass
                    # Waiting requests (queue depth)
                    if "vllm:num_requests_waiting{" in line:
                        try:
                            metrics["queue_depth"] = int(float(line.split()[-1]))
                        except (ValueError, IndexError):
                            pass
                    # Running requests
                    if "vllm:num_requests_running{" in line:
                        try:
                            metrics["running_reqs"] = int(float(line.split()[-1]))
                        except (ValueError, IndexError):
                            pass
                    # Preemption count
                    if "vllm:num_preemptions_total{" in line:
                        try:
                            metrics["preemptions"] = int(float(line.split()[-1]))
                        except (ValueError, IndexError):
                            pass
    except Exception:
        pass
    return metrics


async def run_level(
    concurrency: int,
    prompt_tokens: int,
    max_new_tokens: int,
    duration_secs: int = 120,
    warmup_secs: int = 10,
) -> LevelResult:
    """Run a concurrency level for the specified duration, collecting metrics."""
    prompt = make_prompt(prompt_tokens)
    result = LevelResult(concurrency=concurrency)

    all_results: list[RequestResult] = []
    stop_event = asyncio.Event()

    async def worker():
        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                r = await send_request(session, prompt, max_new_tokens)
                all_results.append(r)
                if r.error:
                    # Small delay on error to avoid tight loop
                    await asyncio.sleep(1)

    # Start workers
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

    # Warmup period
    print(f"  [c={concurrency}] warming up {warmup_secs}s...", flush=True)
    await asyncio.sleep(warmup_secs)

    # Mark measurement start
    measurement_start_idx = len(all_results)
    measurement_start_time = time.monotonic()

    # Collect metrics samples during the measurement window
    metrics_samples = []
    elapsed = 0
    sample_interval = 5
    while elapsed < duration_secs:
        await asyncio.sleep(sample_interval)
        elapsed += sample_interval
        m = await fetch_metrics()
        metrics_samples.append(m)
        running = m["running_reqs"]  # number of running requests
        waiting = m["queue_depth"]  # queue depth
        kv = m["kv_util_pct"]
        print(
            f"  [c={concurrency}] t={elapsed}s running={running} waiting={waiting} kv={kv:.1f}% preempt={m['preemptions']}",
            flush=True,
        )

    measurement_end_time = time.monotonic()
    measurement_duration = measurement_end_time - measurement_start_time

    # Stop workers
    stop_event.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Compute stats from measurement window only
    measured = all_results[measurement_start_idx:]
    successful = [r for r in measured if not r.error]
    errors = [r for r in measured if r.error]

    result.completed = len(successful)
    result.errors = [r.error for r in errors]

    if successful:
        ttfts = sorted([r.ttft_ms for r in successful if r.ttft_ms > 0])
        if ttfts:
            result.ttft_p50 = ttfts[len(ttfts) // 2]
            p99_idx = min(int(len(ttfts) * 0.99), len(ttfts) - 1)
            result.ttft_p99 = ttfts[p99_idx]
            if result.ttft_p50 > 0:
                result.divergence_ratio = result.ttft_p99 / result.ttft_p50

        total_output_tokens = sum(r.output_tokens for r in successful)
        result.throughput_tps = total_output_tokens / measurement_duration

    # Average metrics across samples
    if metrics_samples:
        result.kv_util_pct = statistics.mean(s["kv_util_pct"] for s in metrics_samples)
        result.queue_depth = int(
            statistics.mean(s["queue_depth"] for s in metrics_samples)
        )
        result.running_reqs = int(
            statistics.mean(s["running_reqs"] for s in metrics_samples)
        )
        result.preemptions = max(s["preemptions"] for s in metrics_samples)
        result.effective_concurrency = result.running_reqs + result.queue_depth

    return result


def classify_event(r: LevelResult, prev_preemptions: int) -> str:
    """Classify the event label for a ramp level."""
    if r.errors and any("OOM" in e or "CUDA" in e for e in r.errors):
        return "terminal failure"
    if r.divergence_ratio > 2.0:
        return "cliff onset"
    if r.preemptions > prev_preemptions:
        return "preemption onset"
    return "healthy"


async def run_baseline():
    """Step 3: Baseline at concurrency=4, 512-token prompts."""
    print(
        "\n=== STEP 3: BASELINE (c=4, 512-token prompts, max_new=256) ===\n", flush=True
    )
    result = await run_level(
        concurrency=4,
        prompt_tokens=512,
        max_new_tokens=256,
        duration_secs=120,
        warmup_secs=15,
    )
    print(f"\n--- Baseline Results ---")
    print(f"TTFT p50:        {result.ttft_p50:.1f} ms")
    print(f"TTFT p99:        {result.ttft_p99:.1f} ms")
    print(f"Throughput:      {result.throughput_tps:.1f} tok/s")
    print(f"KV util:         {result.kv_util_pct:.1f}%")
    print(f"Queue depth:     {result.queue_depth}")
    print(f"Running reqs:    {result.running_reqs}")
    print(f"Preemptions:     {result.preemptions}")
    print(f"Completed reqs:  {result.completed}")
    print(f"Errors:          {len(result.errors)}")

    # Final metrics snapshot
    final_metrics = await fetch_metrics()
    print(f"\nFinal KV util:   {final_metrics['kv_util_pct']:.1f}%")
    print(f"Final queue:     {final_metrics['queue_depth']}")

    return result


async def run_ramp():
    """Step 4: Ramp to failure with 2048-token prompts."""
    print(
        "\n=== STEP 4: RAMP TO FAILURE (2048-token prompts, max_new=512) ===\n",
        flush=True,
    )

    # Verify prompt token count via a quick non-streaming request
    prompt = make_prompt(2048)
    print(f"  Prompt length (chars): {len(prompt)}", flush=True)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
            }
            async with session.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "unknown")
                print(f"  Actual prompt tokens: {prompt_tokens}", flush=True)
    except Exception as e:
        print(f"  Token count check failed: {e}", flush=True)

    levels = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    results = []
    prev_preemptions = 0

    for conc in levels:
        print(f"\n--- Concurrency = {conc} ---", flush=True)
        try:
            r = await run_level(
                concurrency=conc,
                prompt_tokens=2048,
                max_new_tokens=512,
                duration_secs=60,
                warmup_secs=10,
            )
        except Exception as e:
            print(f"  LEVEL FAILED: {e}", flush=True)
            r = LevelResult(concurrency=conc, errors=[str(e)])

        event = classify_event(r, prev_preemptions)
        prev_preemptions = r.preemptions

        print(f"  KV util:       {r.kv_util_pct:.1f}%")
        print(f"  Queue depth:   {r.queue_depth}")
        print(f"  Running:       {r.running_reqs}")
        print(f"  Eff conc:      {r.effective_concurrency}")
        print(f"  TTFT p50:      {r.ttft_p50:.1f} ms")
        print(f"  TTFT p99:      {r.ttft_p99:.1f} ms")
        print(f"  Div ratio:     {r.divergence_ratio:.2f}")
        print(f"  Throughput:    {r.throughput_tps:.1f} tok/s")
        print(f"  Preemptions:   {r.preemptions}")
        print(f"  Event:         {event}")
        print(f"  Completed:     {r.completed}")
        print(f"  Errors:        {len(r.errors)}")

        results.append((r, event))

        # Stop if terminal failure
        if event == "terminal failure" or (r.errors and len(r.errors) > r.completed):
            print(f"\n  STOPPING: too many errors at c={conc}", flush=True)
            break

    # Print summary table
    print("\n\n=== RAMP SUMMARY TABLE ===\n")
    print(
        f"{'Conc':>4} {'KV%':>6} {'Queue':>5} {'Run':>4} {'EffC':>5} {'p50ms':>7} {'p99ms':>7} {'DivR':>6} {'tok/s':>7} {'Prempt':>6} {'Event':<20} {'Errors'}"
    )
    print("-" * 110)
    for r, event in results:
        err_str = f"{len(r.errors)}" if r.errors else "0"
        print(
            f"{r.concurrency:>4} {r.kv_util_pct:>6.1f} {r.queue_depth:>5} {r.running_reqs:>4} {r.effective_concurrency:>5} {r.ttft_p50:>7.1f} {r.ttft_p99:>7.1f} {r.divergence_ratio:>6.2f} {r.throughput_tps:>7.1f} {r.preemptions:>6} {event:<20} {err_str}"
        )

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["baseline", "ramp", "all"])
    args = parser.parse_args()

    if args.step in ("baseline", "all"):
        baseline = await run_baseline()

    if args.step in ("ramp", "all"):
        ramp_results = await run_ramp()


if __name__ == "__main__":
    asyncio.run(main())

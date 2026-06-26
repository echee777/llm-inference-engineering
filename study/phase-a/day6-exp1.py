"""
Day 6 — Experiment 1: --max-num-seqs Parameter Sensitivity
===========================================================
Measures how max-num-seqs affects throughput, TTFT, and latency
when sending 16 concurrent requests against vLLM on a T4.

Usage:
    python day6_experiment1_max_num_seqs.py

Requirements:
    pip install aiohttp tiktoken tabulate
"""

import asyncio
import json
import os
import signal
import subprocess
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import aiohttp
    from tabulate import tabulate
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "aiohttp", "tabulate", "--break-system-packages", "-q"])
    import aiohttp
    from tabulate import tabulate


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL = os.environ.get("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://localhost:{PORT}"
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.85  # safe default for T4

# Experiment parameters
MAX_NUM_SEQS_VALUES = [32] # [1, 4, 8, 16, 32]
NUM_REQUESTS = 66          # total requests per run
# One thing to consider: 33 requests gives you noisy percentiles. p99 from 33 samples is literally just your single worst request. If you want stable p99 numbers, bump num_requests to 66 or 99 while keeping concurrency=33. The first 33 all fire simultaneously, then as slots free up the remaining requests fill in — you still maintain constant pressure on the server cap throughout.

CONCURRENCY = 33           # all 16 sent concurrently
MAX_COMPLETION_TOKENS = 256

# Build a ~512-token prompt (repetition is fine — we're testing the engine, not the model)
PROMPT = (
    "Explain the complete history of computing from the abacus to modern "
    "supercomputers, covering every major milestone in detail. "
) * 30  # ~500-520 tokens with most tokenizers

STARTUP_TIMEOUT_S = 120    # max seconds to wait for vLLM to start
SHUTDOWN_WAIT_S = 5        # seconds after kill before restarting
WARMUP_REQUESTS = 2        # throwaway requests to warm up the engine


# ─────────────────────────────────────────────
# vLLM Process Management
# ─────────────────────────────────────────────
def start_vllm(max_num_seqs: int) -> subprocess.Popen:
    """Launch vLLM server with the given --max-num-seqs value."""
    impocmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(max_num_seqs),
        "--dtype", "half",
        "--port", str(PORT),
        "--disable-log-requests",  # keep output clean
    ]
    print(f"\n  Starting vLLM: --max-num-seqs {max_num_seqs}")
    print(f"  Command: {' '.join(cmd)}")

    log_path = Path(f"vllm_seqs_{max_num_seqs}.log")
    log_file = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # so we can kill the whole process group
    )

    # Poll health endpoint until ready
    import urllib.request
    import urllib.error

    for i in range(STARTUP_TIMEOUT_S):
        if proc.poll() is not None:
            print(f"  ERROR: vLLM exited with code {proc.returncode}. Check {log_path}")
            sys.exit(1)
        try:
            resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=1)
            if resp.status == 200:
                print(f"  Server ready after {i + 1}s")
                return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)

    proc.kill()
    print(f"  ERROR: vLLM failed to start within {STARTUP_TIMEOUT_S}s. Check {log_path}")
    sys.exit(1)


def stop_vllm(proc: subprocess.Popen):
    """Gracefully stop vLLM and wait for GPU memory release."""
    print("  Stopping vLLM...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()

    print(f"  Waiting {SHUTDOWN_WAIT_S}s for GPU memory release...")
    time.sleep(SHUTDOWN_WAIT_S)


# ─────────────────────────────────────────────
# Benchmark Client (streaming for TTFT)
# ─────────────────────────────────────────────
    error_detail: str = ""


async def send_streaming_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """Send one streaming request, measuring TTFT and total latency."""
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0.7,
        "stream": True,
    }

    async with semaphore:
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
                    return RequestResult(
                        status="error",
                        total_latency=time.perf_counter() - t_start,
                        error_detail=f"HTTP {resp.status}: {body[:200]}",
                    )

                # Read SSE stream
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get("choices", [{}])[0]
                        text = choice.get("text", "")
                        if text and t_first_token is None:
                            t_first_token = time.perf_counter()
                        if text:
                            tokens_received += 1  # approximate: 1 chunk ≈ 1 token
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            return RequestResult(
                status="error",
                total_latency=time.perf_counter() - t_start,
                error_detail=str(e)[:200],
            )

        t_end = time.perf_counter()
        return RequestResult(
            status="ok",
            ttft=(t_first_token - t_start) if t_first_token else (t_end - t_start),
            total_latency=t_end - t_start,
            completion_tokens=tokens_received,
        )


@dataclass
class BenchmarkResult:
    max_num_seqs: int = 0
    throughput_tok_s: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    wall_time: float = 0.0


def percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


async def run_benchmark(max_num_seqs: int) -> BenchmarkResult:
    """Send NUM_REQUESTS concurrent streaming requests, collect metrics."""
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup: send a couple of throwaway requests sequentially
        print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
        warmup_sem = asyncio.Semaphore(1)
        for _ in range(WARMUP_REQUESTS):
            await send_streaming_request(session, warmup_sem)

        # Actual benchmark
        print(f"  Sending {NUM_REQUESTS} requests at concurrency={CONCURRENCY}...")
        tasks = [
            send_streaming_request(session, semaphore)
            for _ in range(NUM_REQUESTS)
        ]
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        t_wall_end = time.perf_counter()

    wall_time = t_wall_end - t_wall_start
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]

    if errors:
        print(f"  ⚠ {len(errors)} request(s) failed:")
        for e in errors[:3]:
            print(f"    {e.error_detail}")

    if not ok:
        print(f"  ✗ All requests failed!")
        return BenchmarkResult(max_num_seqs=max_num_seqs, error_count=len(errors))

    ttfts = [r.ttft for r in ok]
    latencies = [r.total_latency for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)

    result = BenchmarkResult(
        max_num_seqs=max_num_seqs,
        throughput_tok_s=total_tokens / wall_time if wall_time > 0 else 0,
        ttft_p50=percentile(ttfts, 50),
        ttft_p99=percentile(ttfts, 99),
        latency_p50=percentile(latencies, 50),
        latency_p99=percentile(latencies, 99),
        success_count=len(ok),
        error_count=len(errors),
        total_tokens=total_tokens,
        wall_time=wall_time,
    )

    print(f"  ✓ Throughput:  {result.throughput_tok_s:>8.1f} tok/s")
    print(f"    TTFT p50:    {result.ttft_p50:>8.3f}s")
    print(f"    TTFT p99:    {result.ttft_p99:>8.3f}s")
    print(f"    Latency p50: {result.latency_p50:>8.3f}s")
    print(f"    Latency p99: {result.latency_p99:>8.3f}s")
    print(f"    Total tokens:{result.total_tokens:>8d}")
    print(f"    Wall time:   {result.wall_time:>8.1f}s")
    print(f"    Success:     {result.success_count}/{result.success_count + result.error_count}")

    return result


# ─────────────────────────────────────────────
# Main: Sweep + Summary Table
# ─────────────────────────────────────────────

def print_summary(results: list[BenchmarkResult]):
    """Print a formatted summary table."""
    headers = [
        "max-num-seqs",
        "Throughput\n(tok/s)",
        "TTFT p50\n(s)",
        "TTFT p99\n(s)",
        "Latency p50\n(s)",
        "Latency p99\n(s)",
        "Success\nRate",
    ]
    rows = []
    for r in results:
        total = r.success_count + r.error_count
        rate = f"{r.success_count}/{total}"
        rows.append([
            r.max_num_seqs,
            f"{r.throughput_tok_s:.1f}",
            f"{r.ttft_p50:.3f}",
            f"{r.ttft_p99:.3f}",
            f"{r.latency_p50:.3f}",
            f"{r.latency_p99:.3f}",
            rate,
        ])

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 RESULTS: --max-num-seqs Parameter Sensitivity")
    print(f"Model: {MODEL}")
    print(f"Prompt: ~512 tokens | Max completion: {MAX_COMPLETION_TOKENS} tokens")
    print(f"Requests per run: {NUM_REQUESTS} | Concurrency: {CONCURRENCY}")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))
    print()


def save_results(results: list[BenchmarkResult]):
    """Save raw results to JSON for later analysis."""
    output = {
        "experiment": "day6_exp1_max_num_seqs",
        "config": {
            "model": MODEL,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "num_requests": NUM_REQUESTS,
            "concurrency": CONCURRENCY,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
        "results": [
            {
                "max_num_seqs": r.max_num_seqs,
                "throughput_tok_s": r.throughput_tok_s,
                "ttft_p50": r.ttft_p50,
                "ttft_p99": r.ttft_p99,
                "latency_p50": r.latency_p50,
                "latency_p99": r.latency_p99,
                "success_count": r.success_count,
                "error_count": r.error_count,
                "total_tokens": r.total_tokens,
                "wall_time": r.wall_time,
            }
            for r in results
        ],
    }
    out_path = Path("day6_exp1_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Raw results saved to {out_path}")


def main():
    print("=" * 80)
    print("Day 6 — Experiment 1: --max-num-seqs Parameter Sensitivity")
    print(f"Model:       {MODEL}")
    print(f"GPU Mem:     {GPU_MEMORY_UTILIZATION}")
    print(f"Sweep:       {MAX_NUM_SEQS_VALUES}")
    print(f"Requests:    {NUM_REQUESTS} @ concurrency={CONCURRENCY}")
    print("=" * 80)

    all_results: list[BenchmarkResult] = []

    for seqs_value in MAX_NUM_SEQS_VALUES:
        print(f"\n{'─' * 60}")
        print(f"  RUN: --max-num-seqs={seqs_value}")
        print(f"{'─' * 60}")

        proc = start_vllm(seqs_value)
        try:
            result = asyncio.run(run_benchmark(seqs_value))
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            stop_vllm(proc)
            break
        finally:
            stop_vllm(proc)

    if all_results:
        print_summary(all_results)
        save_results(all_results)

    print("Done. Review the table above and record observations in your notes.")
    print("\nKey questions to answer:")
    print("  • At what --max-num-seqs does throughput plateau on your T4?")
    print("  • How does TTFT p99 change as you allow more concurrent sequences?")
    print("  • Did any runs hit errors (rejected/preempted requests)?")
    print("  • What's the sweet spot for your T4 + TinyLlama setup?")


if __name__ == "__main__":
    main()
"""
Day 6 — Experiment 3: --max-num-batched-tokens Parameter Sensitivity
=====================================================================
Sweeps the per-iteration token budget across [512, 1024, 2048, 4096]
with chunked prefill DISABLED (the vLLM default). Runs three sub-tests
per budget value:

  SHORT run  (~100-token prompt, concurrency=16):
    Baseline throughput and TTFT. Larger budget allows more SHORT requests
    to be co-batched per iteration, reducing queue wait.

  LONG run   (~440-token prompt, concurrency=16):
    TTFT here is dominated by prefill time and queue wait. With a small
    budget, concurrent LONG requests are forced toward serial execution
    (~1 per iteration). With a large budget they can be co-batched.
    Expected: higher budget → lower LONG TTFT, higher throughput.

  CEILING test (~610-token prompt, single request):
    Without chunked prefill, any prompt longer than the budget cannot be
    scheduled. At budget=512 this request should fail. At budget≥1024 it
    should succeed. Directly demonstrates the implicit prompt-length ceiling.

Fixed parameters (held constant across the sweep to isolate the variable):
  --max-num-seqs          16    (T4 sweet spot from Exp 1)
  --gpu-memory-utilization 0.85
  --max-model-len         2048
  chunked prefill         disabled (default — not passed)
  --dtype                 half   (T4: no bfloat16)

Token estimates (TinyLlama tokenizer, ~17 tokens per base phrase repeat):
  SHORT prompt:    6 repeats  ≈  102 tokens
  LONG prompt:    26 repeats  ≈  442 tokens   (fits in every budget ≥ 512)
  OVERFLOW prompt: 36 repeats ≈  612 tokens   (fails at 512, passes at ≥1024)

Inquiry goals:
  1. How does the token budget affect aggregate prefill throughput?
     (Larger budget → more co-batching → lower TTFT for LONG regime.)
  2. At what budget does the OVERFLOW prompt transition from fail to pass?
     (The implicit length ceiling, with chunked prefill off.)
  3. Does SHORT regime TTFT improve with larger budget as HOL blocking
     by LONG prefills is reduced?
     (Preview of the prefill/decode interference problem in Phase B.)

Usage:
    python day6_exp3.py

Requirements:
    pip install aiohttp tabulate
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    import aiohttp
    from tabulate import tabulate
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install",
         "aiohttp", "tabulate", "--break-system-packages", "-q"]
    )
    import aiohttp
    from tabulate import tabulate


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL = os.environ.get("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://localhost:{PORT}"
MAX_MODEL_LEN = 2048

# ── Variable under test ──────────────────────
MAX_NUM_BATCHED_TOKENS_VALUES = [512, 1024, 2048, 4096]

# ── Fixed parameters ─────────────────────────
MAX_NUM_SEQS = 16          # held constant; isolate the batched-token budget
GPU_MEMORY_UTILIZATION = 0.85
MAX_COMPLETION_TOKENS = 128

# ── Prompt construction ──────────────────────
# Base phrase: ~17 tokens per repeat (measured against TinyLlama tokenizer)
_BASE = (
    "Explain the complete history of computing from the abacus to modern "
    "supercomputers, covering every major milestone in detail. "
)
SHORT_PROMPT    = _BASE * 6    # ≈  102 tokens  — fits in every sweep value
LONG_PROMPT     = _BASE * 26   # ≈  442 tokens  — fits in every sweep value ≥ 512
OVERFLOW_PROMPT = _BASE * 36   # ≈  612 tokens  — fails at 512, passes at ≥ 1024

# ── Load parameters ──────────────────────────
SHORT_CONCURRENCY  = 16
SHORT_NUM_REQUESTS = 48   # 3 × concurrency: stable p50/p99 without excessive runtime

LONG_CONCURRENCY   = 16
LONG_NUM_REQUESTS  = 32   # 2 × concurrency

WARMUP_REQUESTS    = 2
STARTUP_TIMEOUT_S  = 180
SHUTDOWN_WAIT_S    = 8


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class RequestResult:
    status: str = "ok"         # "ok" | "error"
    ttft: float = 0.0          # time-to-first-token (s)
    total_latency: float = 0.0
    completion_tokens: int = 0
    error_detail: str = ""


@dataclass
class RegimeResult:
    """Results for one regime (SHORT or LONG) at one budget value."""
    budget: int = 0
    regime: str = ""           # "SHORT" | "LONG"
    prompt_tokens: int = 0
    throughput_tok_s: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    wall_time: float = 0.0


@dataclass
class CeilingResult:
    """Result of the single-request overflow ceiling test."""
    budget: int = 0
    prompt_tokens: int = 0     # approximate
    passed: bool = False
    ttft: float = 0.0
    error_detail: str = ""


@dataclass
class BudgetSweepResult:
    budget: int = 0
    short: RegimeResult = field(default_factory=RegimeResult)
    long: RegimeResult = field(default_factory=RegimeResult)
    ceiling: CeilingResult = field(default_factory=CeilingResult)
    preemption_count: int = 0


# ─────────────────────────────────────────────
# vLLM Process Management
# ─────────────────────────────────────────────

def start_vllm(max_num_batched_tokens: int) -> subprocess.Popen:
    """Launch vLLM with the given --max-num-batched-tokens. Chunked prefill off (default)."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--dtype", "half",
        "--port", str(PORT),
        "--disable-log-requests",
        # NOTE: --enable-chunked-prefill intentionally NOT passed.
        # Default is off. Without it, prompts > max-num-batched-tokens fail.
    ]

    log_path = Path(f"vllm_batched_{max_num_batched_tokens}.log")
    print(f"\n  Starting vLLM: --max-num-batched-tokens {max_num_batched_tokens}")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    for i in range(STARTUP_TIMEOUT_S):
        if proc.poll() is not None:
            print(f"  ERROR: vLLM exited (code {proc.returncode}). Check {log_path}")
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
    print(f"  ERROR: vLLM failed to start within {STARTUP_TIMEOUT_S}s.")
    sys.exit(1)


def stop_vllm(proc: subprocess.Popen):
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


def count_preemptions(budget: int) -> int:
    log_path = Path(f"vllm_batched_{budget}.log")
    if not log_path.exists():
        return 0
    return len(re.findall(r"(?i)preempt", log_path.read_text(errors="replace")))


# ─────────────────────────────────────────────
# Async Request Client
# ─────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
) -> RequestResult:
    """Send one streaming request; measure TTFT and total latency."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0.0,   # greedy — removes sampling variance from timing
        "stream": True,
    }

    async with semaphore:
        t_start = time.perf_counter()
        t_first_token: float | None = None
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
                        error_detail=f"HTTP {resp.status}: {body[:300]}",
                    )

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

        except Exception as exc:
            return RequestResult(
                status="error",
                total_latency=time.perf_counter() - t_start,
                error_detail=str(exc)[:300],
            )

        t_end = time.perf_counter()
        return RequestResult(
            status="ok",
            ttft=(t_first_token - t_start) if t_first_token else (t_end - t_start),
            total_latency=t_end - t_start,
            completion_tokens=tokens_received,
        )


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


async def run_warmup(session: aiohttp.ClientSession):
    """Fire a few throwaway requests to warm up the engine."""
    print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
    sem = asyncio.Semaphore(1)
    for _ in range(WARMUP_REQUESTS):
        await send_request(session, sem, SHORT_PROMPT)


# ─────────────────────────────────────────────
# Regime Runners
# ─────────────────────────────────────────────

async def run_regime(
    budget: int,
    regime: str,
    prompt: str,
    prompt_tokens: int,
    num_requests: int,
    concurrency: int,
) -> RegimeResult:
    """Run one load regime (SHORT or LONG) against the live server."""
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        await run_warmup(session)
        sem = asyncio.Semaphore(concurrency)
        print(f"  [{regime}] Sending {num_requests} requests @ concurrency={concurrency}...")
        tasks = [send_request(session, sem, prompt) for _ in range(num_requests)]
        t0 = time.perf_counter()
        raw = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    ok = [r for r in raw if r.status == "ok"]
    errors = [r for r in raw if r.status != "ok"]

    if errors:
        print(f"  ⚠  {len(errors)} error(s):")
        for e in errors[:3]:
            print(f"     {e.error_detail}")

    if not ok:
        print(f"  ✗ All {regime} requests failed.")
        return RegimeResult(
            budget=budget, regime=regime, prompt_tokens=prompt_tokens,
            error_count=len(errors),
        )

    ttfts     = [r.ttft for r in ok]
    latencies = [r.total_latency for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)
    tps = total_tokens / wall_time if wall_time > 0 else 0.0

    result = RegimeResult(
        budget=budget,
        regime=regime,
        prompt_tokens=prompt_tokens,
        throughput_tok_s=tps,
        ttft_p50=percentile(ttfts, 50),
        ttft_p99=percentile(ttfts, 99),
        latency_p50=percentile(latencies, 50),
        latency_p99=percentile(latencies, 99),
        success_count=len(ok),
        error_count=len(errors),
        total_tokens=total_tokens,
        wall_time=wall_time,
    )

    print(f"  ✓ [{regime}] Throughput: {result.throughput_tok_s:>7.1f} tok/s  "
          f"TTFT p50: {result.ttft_p50:.3f}s  TTFT p99: {result.ttft_p99:.3f}s  "
          f"Success: {result.success_count}/{result.success_count + result.error_count}")
    return result


async def run_ceiling_test(budget: int, prompt: str, prompt_tokens: int) -> CeilingResult:
    """Send a single request whose prompt length exceeds the current budget.

    Without chunked prefill, vLLM cannot schedule a prompt longer than
    --max-num-batched-tokens. The server will return an error or the request
    will hang then timeout. Either outcome is a FAIL for this test.
    At budget values >= prompt_tokens this should succeed.
    """
    print(f"  [CEILING] Single request, prompt≈{prompt_tokens} tokens vs budget={budget}...")
    connector = aiohttp.TCPConnector(limit=2)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(1)
        result = await send_request(session, sem, prompt)

    if result.status == "ok" and result.completion_tokens > 0:
        c = CeilingResult(
            budget=budget,
            prompt_tokens=prompt_tokens,
            passed=True,
            ttft=result.ttft,
        )
        print(f"  ✓ [CEILING] PASS  (TTFT: {c.ttft:.3f}s)")
    else:
        c = CeilingResult(
            budget=budget,
            prompt_tokens=prompt_tokens,
            passed=False,
            error_detail=result.error_detail[:120],
        )
        print(f"  ✗ [CEILING] FAIL  ({c.error_detail})")
    return c


# ─────────────────────────────────────────────
# Summary + Persistence
# ─────────────────────────────────────────────

def print_summary(sweep_results: list[BudgetSweepResult]):
    print("\n" + "=" * 100)
    print("EXPERIMENT 3 RESULTS: --max-num-batched-tokens Parameter Sensitivity")
    print(f"Model: {MODEL}")
    print(f"Fixed: --max-num-seqs={MAX_NUM_SEQS}  --gpu-memory-utilization={GPU_MEMORY_UTILIZATION}"
          f"  chunked-prefill=OFF")
    print(f"SHORT prompt ≈ 102 tokens | LONG prompt ≈ 442 tokens | "
          f"OVERFLOW prompt ≈ 612 tokens | max_completion={MAX_COMPLETION_TOKENS}")
    print("=" * 100)

    # Throughput + TTFT table
    headers = [
        "Budget\n(tokens)",
        "Regime",
        "Prompt\n(tokens)",
        "Throughput\n(tok/s)",
        "TTFT p50\n(s)",
        "TTFT p99\n(s)",
        "Latency p50\n(s)",
        "Latency p99\n(s)",
        "Success",
        "Ceiling\nTest",
    ]
    rows = []
    for sr in sweep_results:
        for regime_result, is_first in [(sr.short, True), (sr.long, False)]:
            total = regime_result.success_count + regime_result.error_count
            ceiling_str = ""
            if is_first:
                ceiling_str = "PASS ✓" if sr.ceiling.passed else "FAIL ✗"
            rows.append([
                sr.budget if is_first else "",
                regime_result.regime,
                regime_result.prompt_tokens,
                f"{regime_result.throughput_tok_s:.1f}",
                f"{regime_result.ttft_p50:.3f}",
                f"{regime_result.ttft_p99:.3f}",
                f"{regime_result.latency_p50:.3f}",
                f"{regime_result.latency_p99:.3f}",
                f"{regime_result.success_count}/{total}" if total else "—",
                ceiling_str,
            ])

    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))

    print("\nPreemption events per run:")
    for sr in sweep_results:
        print(f"  budget={sr.budget}: {sr.preemption_count} preemption(s) in vLLM log")

    print("\nKey questions to answer:")
    print("  • Does LONG TTFT p50 decrease as budget increases?")
    print("    If yes: larger budget allows more co-batching of LONG prefills.")
    print("    If flat: something else is the bottleneck (compute, max-num-seqs).")
    print()
    print("  • Does SHORT TTFT p50 also decrease with larger budget?")
    print("    If yes: head-of-line blocking by LONG prefills is reduced.")
    print("    This is Goal 3 — a preview of prefill/decode interference.")
    print()
    print("  • At what budget does the CEILING test transition from FAIL to PASS?")
    print("    Expected: FAIL at 512 (612 > 512), PASS at ≥ 1024.")
    print("    If unexpected: check whether chunked prefill got enabled by default in your vLLM version.")
    print()
    print("  • Does throughput scale roughly linearly with budget for the LONG regime?")
    print("    If 2× budget ≈ 2× throughput: batching efficiency is roughly linear.")
    print("    If sublinear: other bottlenecks are entering (memory bandwidth, decode compute).")


def save_results(sweep_results: list[BudgetSweepResult]):
    output = {
        "experiment": "day6_exp3_max_num_batched_tokens",
        "config": {
            "model": MODEL,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "chunked_prefill": False,
            "short_prompt_tokens_approx": 102,
            "long_prompt_tokens_approx": 442,
            "overflow_prompt_tokens_approx": 612,
        },
        "results": [],
    }
    for sr in sweep_results:
        def regime_dict(r: RegimeResult) -> dict:
            return {
                "regime": r.regime,
                "prompt_tokens": r.prompt_tokens,
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
        output["results"].append({
            "budget": sr.budget,
            "short": regime_dict(sr.short),
            "long": regime_dict(sr.long),
            "ceiling": {
                "prompt_tokens": sr.ceiling.prompt_tokens,
                "passed": sr.ceiling.passed,
                "ttft": sr.ceiling.ttft,
                "error_detail": sr.ceiling.error_detail,
            },
            "preemption_count": sr.preemption_count,
        })

    out_path = Path("day6_exp3_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nRaw results saved to {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 100)
    print("Day 6 — Experiment 3: --max-num-batched-tokens Parameter Sensitivity")
    print(f"Model:           {MODEL}")
    print(f"Sweep values:    {MAX_NUM_BATCHED_TOKENS_VALUES}")
    print(f"max-num-seqs:    {MAX_NUM_SEQS}  (fixed)")
    print(f"gpu-mem-util:    {GPU_MEMORY_UTILIZATION}  (fixed)")
    print(f"chunked-prefill: OFF  (default — not passed to server)")
    print()
    print("Prompt token estimates:")
    print(f"  SHORT    ≈  102 tokens  (6 × base phrase)")
    print(f"  LONG     ≈  442 tokens  (26 × base phrase)  — fits every budget ≥ 512")
    print(f"  OVERFLOW ≈  612 tokens  (36 × base phrase)  — fails at 512, passes at ≥ 1024")
    print()
    print("Expected ceiling boundary:")
    for b in MAX_NUM_BATCHED_TOKENS_VALUES:
        verdict = "PASS" if b >= 612 else "FAIL"
        print(f"  budget={b:4d}: OVERFLOW test → {verdict}")
    print("=" * 100)

    all_sweep_results: list[BudgetSweepResult] = []

    for budget in MAX_NUM_BATCHED_TOKENS_VALUES:
        print(f"\n{'─' * 70}")
        print(f"  SWEEP: --max-num-batched-tokens={budget}")
        print(f"{'─' * 70}")

        proc = start_vllm(budget)
        try:
            short_result = asyncio.run(run_regime(
                budget=budget,
                regime="SHORT",
                prompt=SHORT_PROMPT,
                prompt_tokens=102,
                num_requests=SHORT_NUM_REQUESTS,
                concurrency=SHORT_CONCURRENCY,
            ))
            long_result = asyncio.run(run_regime(
                budget=budget,
                regime="LONG",
                prompt=LONG_PROMPT,
                prompt_tokens=442,
                num_requests=LONG_NUM_REQUESTS,
                concurrency=LONG_CONCURRENCY,
            ))
            ceiling_result = asyncio.run(run_ceiling_test(
                budget=budget,
                prompt=OVERFLOW_PROMPT,
                prompt_tokens=612,
            ))
            preemptions = count_preemptions(budget)

            all_sweep_results.append(BudgetSweepResult(
                budget=budget,
                short=short_result,
                long=long_result,
                ceiling=ceiling_result,
                preemption_count=preemptions,
            ))

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            stop_vllm(proc)
            break
        finally:
            stop_vllm(proc)

    if all_sweep_results:
        print_summary(all_sweep_results)
        save_results(all_sweep_results)


if __name__ == "__main__":
    main()

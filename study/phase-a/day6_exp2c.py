#!/usr/bin/env python3
"""
Day 6 — Experiment 2c: --gpu-memory-utilization Parameter Sensitivity
Method: Sustained Arrival Rate (Option C)

WHY OPTION C OVER 2B:
  Burst loading (Exp 2b) fired degradation at concurrency=8 for all utilization
  values with near-identical numbers (0.219–0.220s TTFT, 3.1× baseline). Root
  cause: FIFO prefill queuing inflates TTFT before KV pressure is ever reached.
  Degradation threshold fired on the wrong phenomenon.

  Option C sends requests at a fixed arrival rate and discards the first 30s of
  results so the system can reach steady state. At steady state, TTFT reflects
  actual scheduling pressure. KV pool exhaustion manifests as a lower maximum
  *sustainable RPS* — which differs predictably across utilization values.

EXPECTED DIFFERENTIATION (Little's Law: N = λ × W, W ≈ 7s avg latency):
  Overhead floor for Qwen2.5-3B on T4: 8.44 GB
    (weights 5.79 GB + activations 2.52 GB + other 0.13 GB — measured from vLLM log)
  Note: 0.60 excluded — KV pool (~0.30 GB) too small to satisfy vLLM's max_seq_len
    floor check (32768 tokens required, only 8528 storable). Fix: --max-model-len 2048.

  gpu_memory_utilization=0.70 → KV pool ~1.75 GB → ~37 concurrent → max stable ≈  5.3 RPS
  gpu_memory_utilization=0.85 → KV pool ~3.94 GB → ~83 concurrent → max stable ≈ 11.9 RPS
  gpu_memory_utilization=0.95 → KV pool ~5.39 GB → ~114 concurrent → max stable ≈ 16.3 RPS

DEGRADATION DETECTION (three signals):
  1. error_rate > 10%
  2. TTFT p50 > 3× baseline (less hair-trigger than burst version's 2×)
  3. TTFT trending up within measurement window: second-half p50 > 1.5× first-half
     → queue still growing, system hasn't reached steady state = overloaded

STRUCTURE PER LEVEL:
  90s total → 30s warmup (discard) + 60s measurement window
  Uniform inter-arrival spacing (1/RPS seconds between dispatches)
  asyncio.Semaphore(MAX_OUTSTANDING) caps concurrent HTTP connections
"""

import asyncio
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import statistics

import aiohttp

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these constants, not the logic below
# ═══════════════════════════════════════════════════════════════════════

MODEL_ID            = "Qwen/Qwen2.5-3B-Instruct"
HOST                = "127.0.0.1"
PORT                = 8000
BASE_URL            = f"http://{HOST}:{PORT}"

# Fixed vLLM parameter (not under test)
MAX_NUM_SEQS        = 256

# T4 dtype constraint: BF16 requires compute capability 8.0+; T4 is Turing (7.5)
DTYPE               = "half"

# Qwen2.5-3B default max_seq_len is 32768. At low gpu_memory_utilization the KV
# pool cannot hold 32768 tokens and vLLM refuses to start. Cap to 2048 — well
# above our request size (~336 tokens) and satisfiable at all util values.
MAX_MODEL_LEN       = 2048

# Parameter under test — 0.60 excluded: KV pool too small to start (see docstring)
GPU_MEM_UTIL_VALUES = [0.70, 0.85, 0.95]

# RPS sweep — spans well below to well above expected capacity ceilings.
# Corrected via measured overhead floor (8.44 GB for Qwen2.5-3B, not 7.33 GB
# from TinyLlama). Little's Law, W ≈ 7s:
#   0.70 → ceil ≈ 5.3 RPS,  0.85 → 11.9,  0.95 → 16.3
RPS_RAMP            = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24]

# Timing per RPS level
#
# WARMUP rationale: at high RPS the queue fills *during* warmup. If warmup is
# too short, measurement starts with the queue already at depth — early
# measurement requests see peak TTFT and later ones see it drain, producing a
# trend ratio < 1.0 (backward) even when the system is saturated. The trend
# detector needs the queue to be in steady state at the *start* of the window.
#
# At W ≈ 7s avg latency and RPS=8 (observed cliff for util=0.70), the queue
# can absorb ~56 request-seconds of backlog before measurement begins. 60s
# warmup gives the system time to either reach steady state or fully saturate
# before counting — both are valid start conditions for the trend detector.
#
# Rule of thumb: WARMUP_PER_LEVEL_S >= 2× W (avg latency).
DURATION_PER_LEVEL_S = 150       # total seconds at each RPS (was 90)
WARMUP_PER_LEVEL_S   = 60        # discard first N seconds (was 30 — too short at saturation)
MEASUREMENT_WINDOW_S = DURATION_PER_LEVEL_S - WARMUP_PER_LEVEL_S   # 90s

# Safety valve: caps concurrent HTTP connections; prevents client-side OOM
# when RPS far exceeds server capacity
MAX_OUTSTANDING      = 300

# Gather timeout: how long after dispatch loop ends to wait for in-flight requests.
# At very high RPS on an overloaded server, requests can queue for a long time.
GATHER_TIMEOUT_S     = 180

# Degradation criteria
DEGRADE_ERROR_RATE   = 0.10   # 10% request failures
DEGRADE_TTFT_MULT    = 3.0    # 3× baseline TTFT p50
DEGRADE_TREND_RATIO  = 1.5    # second-half TTFT p50 > 1.5× first-half within window

# Stop ramp after this many consecutive degraded levels
MAX_CONSECUTIVE_DEGRADED = 3

# Prompt (kept identical to Exp 2b for comparability)
PROMPT = (
    "You are a helpful assistant. "
    "Explain the key tradeoffs between DRAM bandwidth and compute throughput "
    "in GPU inference serving systems, focusing on the memory-bound nature of "
    "autoregressive decode. Include discussion of KV cache memory pressure."
)
MAX_TOKENS           = 256

# vLLM lifecycle
STARTUP_TIMEOUT_S    = 180
POLL_INTERVAL_S      = 3.0
TEARDOWN_WAIT_S      = 12
WARMUP_REQUESTS      = 5

OUTPUT_DIR           = Path(".")
LOG_PREFIX           = "vllm_gmem_2c"
JSON_OUTPUT          = OUTPUT_DIR / "exp2c_results.json"


# ═══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RequestResult:
    dispatch_offset_s: float    # seconds from level start (at actual dispatch)
    ttft: Optional[float]       # time-to-first-token (s); None on error
    e2e_latency: float          # end-to-end wall-clock time (s)
    tokens_generated: int
    success: bool
    error_msg: str = ""


@dataclass
class LevelResult:
    rps: float
    measurement_window_s: float
    results: list[RequestResult] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────
    @property
    def n(self):          return len(self.results)
    @property
    def n_success(self):  return sum(1 for r in self.results if r.success)
    @property
    def error_rate(self):
        return 1.0 - self.n_success / self.n if self.n else 0.0

    def _ttfts(self):
        return sorted(r.ttft for r in self.results if r.success and r.ttft is not None)

    def _lats(self):
        return sorted(r.e2e_latency for r in self.results if r.success)

    @staticmethod
    def _pct(vals: list[float], p: float) -> Optional[float]:
        if not vals:
            return None
        idx = max(0, int(math.ceil(p * len(vals))) - 1)
        return vals[idx]

    @property
    def ttft_p50(self):  return self._pct(self._ttfts(), 0.50)
    @property
    def ttft_p99(self):  return self._pct(self._ttfts(), 0.99)
    @property
    def lat_p50(self):   return self._pct(self._lats(),  0.50)
    @property
    def lat_p99(self):   return self._pct(self._lats(),  0.99)

    @property
    def throughput_tok_s(self):
        total = sum(r.tokens_generated for r in self.results if r.success)
        return total / self.measurement_window_s if self.measurement_window_s > 0 else 0.0

    @property
    def actual_rps(self):
        return self.n_success / self.measurement_window_s if self.measurement_window_s > 0 else 0.0

    def ttft_trend_ratio(self) -> Optional[float]:
        """
        Splits the measurement window in half by dispatch time, computes TTFT
        p50 for each half.  Ratio > 1 means latency was still growing — the
        queue had not stabilised, i.e. the system was overloaded at this RPS.

        Returns None if fewer than 10 successful results (insufficient sample).
        """
        good = sorted(
            [r for r in self.results if r.success and r.ttft is not None],
            key=lambda r: r.dispatch_offset_s,
        )
        if len(good) < 10:
            return None
        mid = len(good) // 2
        first_p50  = statistics.median(r.ttft for r in good[:mid])
        second_p50 = statistics.median(r.ttft for r in good[mid:])
        return second_p50 / first_p50 if first_p50 > 0 else None


def check_degradation(
    level: LevelResult,
    baseline_ttft_p50: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    # Signal 1 — error rate
    if level.error_rate > DEGRADE_ERROR_RATE:
        reasons.append(
            f"error_rate={level.error_rate:.2f} (>{DEGRADE_ERROR_RATE:.2f})"
        )

    # Signal 2 — absolute TTFT multiplier
    if level.ttft_p50 is not None and baseline_ttft_p50 > 0:
        mult = level.ttft_p50 / baseline_ttft_p50
        if mult > DEGRADE_TTFT_MULT:
            reasons.append(
                f"ttft_p50={mult:.1f}× baseline (>{DEGRADE_TTFT_MULT:.1f}×)"
            )

    # Signal 3 — TTFT trend within window (queue still growing)
    trend = level.ttft_trend_ratio()
    if trend is not None and trend > DEGRADE_TREND_RATIO:
        reasons.append(
            f"ttft_trend={trend:.2f}× within window (>{DEGRADE_TREND_RATIO:.2f}×)"
        )

    return bool(reasons), reasons


# ═══════════════════════════════════════════════════════════════════════
#  vLLM LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════

def start_vllm(gpu_mem_util: float) -> tuple[subprocess.Popen, Path]:
    tag = f"{gpu_mem_util:.2f}".replace(".", "p")
    log_path = OUTPUT_DIR / f"{LOG_PREFIX}_{tag}.log"
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", HOST,
        "--port", str(PORT),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--disable-log-requests",
    ]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_path


async def wait_for_ready(timeout_s: float = STARTUP_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout_s
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    f"{BASE_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(POLL_INTERVAL_S)
    return False


def stop_vllm(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def count_preemptions(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return len(re.findall(r"preempt", log_path.read_text(errors="replace"), re.IGNORECASE))


# ═══════════════════════════════════════════════════════════════════════
#  REQUEST WORKER
# ═══════════════════════════════════════════════════════════════════════

async def send_request(
    session: aiohttp.ClientSession,
) -> tuple[Optional[float], float, int, bool, str]:
    """
    Sends one streaming chat completion.
    Returns: (ttft, e2e_latency, tokens_generated, success, error_msg)

    Token count: one SSE chunk ≈ one token (approximate, consistent with Exp 2b).
    """
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    t0    = time.perf_counter()
    ttft  = None
    tokens = 0

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120, connect=10),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return None, time.perf_counter() - t0, 0, False, \
                    f"HTTP {resp.status}: {body[:120]}"

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                content = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1

    except asyncio.CancelledError:
        raise
    except Exception as e:
        return None, time.perf_counter() - t0, tokens, False, str(e)[:120]

    return ttft, time.perf_counter() - t0, tokens, True, ""


# ═══════════════════════════════════════════════════════════════════════
#  WARMUP
# ═══════════════════════════════════════════════════════════════════════

async def run_warmup(n: int = WARMUP_REQUESTS):
    print(f"  Warming up ({n} sequential requests)...", flush=True)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(n):
            await send_request(session)
    print("  Warmup complete.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  SUSTAINED LOAD DRIVER  — core of Option C
# ═══════════════════════════════════════════════════════════════════════

async def run_level(rps: float) -> LevelResult:
    """
    Drives uniform-spaced arrivals at `rps` requests/second for
    DURATION_PER_LEVEL_S total seconds.

    Only results whose dispatch happened after WARMUP_PER_LEVEL_S are recorded
    (the warmup window lets the server absorb the initial burst and reach
    steady-state queue depth before we start measuring).

    A semaphore of size MAX_OUTSTANDING prevents unbounded connection growth
    when RPS exceeds server capacity.  When all slots are taken, new workers
    block inside the semaphore — the dispatch loop continues on schedule,
    creating lightweight coroutines that wait for a slot.  This is intentional:
    at overload, actual dispatch lags behind scheduled dispatch, and the TTFT
    trend detector catches the growing queue.
    """
    interval         = 1.0 / rps
    results: list[RequestResult] = []
    tasks:   list[asyncio.Task]  = []
    semaphore = asyncio.Semaphore(MAX_OUTSTANDING)

    connector = aiohttp.TCPConnector(limit=MAX_OUTSTANDING + 20)
    session   = aiohttp.ClientSession(connector=connector)
    level_start = time.perf_counter()

    async def worker(scheduled_offset: float):
        # Acquire semaphore before dispatching — blocks if MAX_OUTSTANDING in flight.
        # Record ACTUAL dispatch time (after semaphore acquisition) for warmup filter.
        async with semaphore:
            dispatch_offset = time.perf_counter() - level_start
            ttft, e2e, tokens, ok, err = await send_request(session)

        # Only record results from the measurement window (post-warmup dispatches)
        if dispatch_offset >= WARMUP_PER_LEVEL_S:
            results.append(RequestResult(
                dispatch_offset_s=dispatch_offset,
                ttft=ttft,
                e2e_latency=e2e,
                tokens_generated=tokens,
                success=ok,
                error_msg=err,
            ))

    # Dispatch loop: schedule arrivals at uniform intervals for DURATION_PER_LEVEL_S
    i = 0
    while True:
        scheduled_at = level_start + i * interval
        if scheduled_at > level_start + DURATION_PER_LEVEL_S:
            break

        # Sleep until next scheduled arrival time
        sleep_for = scheduled_at - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        task = asyncio.create_task(worker(scheduled_at - level_start))
        tasks.append(task)
        i += 1

    # Wait for all in-flight requests with a hard timeout to prevent hanging
    # at overload levels where requests queue indefinitely
    try:
        done, pending = await asyncio.wait(tasks, timeout=GATHER_TIMEOUT_S)
        if pending:
            print(
                f"    [warn] {len(pending)} requests still in-flight after "
                f"{GATHER_TIMEOUT_S}s — cancelling (system overloaded)",
                flush=True,
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        pass

    await session.close()

    return LevelResult(
        rps=rps,
        measurement_window_s=MEASUREMENT_WINDOW_S,
        results=results,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PER-UTILIZATION RUN
# ═══════════════════════════════════════════════════════════════════════

async def run_utilization(
    gpu_mem_util: float,
) -> list[tuple[LevelResult, bool, list[str]]]:
    """
    Full RPS ramp for a single gpu_memory_utilization value.
    Stops after MAX_CONSECUTIVE_DEGRADED consecutive degraded levels.
    Returns list of (LevelResult, degraded, reasons).
    """
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  RUN: --gpu-memory-utilization={gpu_mem_util}")
    print(sep)

    proc, log_path = start_vllm(gpu_mem_util)
    print(f"  Starting vLLM: --gpu-memory-utilization {gpu_mem_util}")
    print(f"  Log: {log_path.name}")

    t_start = time.time()
    ready = await wait_for_ready()
    elapsed = int(time.time() - t_start)

    if not ready:
        print(f"  ERROR: vLLM failed to start within {STARTUP_TIMEOUT_S}s. Aborting.")
        stop_vllm(proc)
        return []

    print(f"  Server ready after {elapsed}s")
    await run_warmup()

    level_results: list[tuple[LevelResult, bool, list[str]]] = []
    baseline_ttft:       Optional[float] = None
    consecutive_degraded = 0

    for rps in RPS_RAMP:
        print(
            f"\n  RPS={rps:5.1f}  "
            f"[{WARMUP_PER_LEVEL_S}s warmup + {MEASUREMENT_WINDOW_S}s measurement]",
            flush=True,
        )

        level = await run_level(rps)

        # Use the lowest-RPS level as baseline
        if baseline_ttft is None and level.ttft_p50 is not None:
            baseline_ttft = level.ttft_p50

        degraded, reasons = check_degradation(level, baseline_ttft or 0.001)
        level_results.append((level, degraded, reasons))

        # Live summary line
        ttft_str  = f"{level.ttft_p50:.3f}s" if level.ttft_p50 is not None else "N/A"
        trend     = level.ttft_trend_ratio()
        trend_str = f"  trend={trend:.2f}×" if trend is not None else ""
        status    = "✗ DEGRADED" if degraded else "✓ clean   "
        reason_str = f"  [{', '.join(reasons)}]" if reasons else ""

        print(
            f"    {status}  n={level.n_success}/{level.n}"
            f"  TTFT p50={ttft_str}"
            f"  actual_rps={level.actual_rps:.1f}"
            f"  throughput={level.throughput_tok_s:.0f} tok/s"
            f"  err={level.error_rate:.2f}"
            f"{trend_str}"
            f"{reason_str}",
            flush=True,
        )

        # Stop condition: enough consecutive degraded levels to characterise the cliff
        if degraded:
            consecutive_degraded += 1
            if consecutive_degraded == 1:
                print(
                    f"  ↳ Degradation onset at RPS={rps}. "
                    f"Continuing {MAX_CONSECUTIVE_DEGRADED - 1} more level(s) to characterise cliff.",
                    flush=True,
                )
            if consecutive_degraded >= MAX_CONSECUTIVE_DEGRADED:
                print(f"  ↳ Confirmed degradation. Stopping ramp.", flush=True)
                break
        else:
            consecutive_degraded = 0

    # Teardown
    print(f"\n  Stopping vLLM...")
    stop_vllm(proc)
    print(f"  Waiting {TEARDOWN_WAIT_S}s for GPU memory release...")
    await asyncio.sleep(TEARDOWN_WAIT_S)
    print(f"  Preemption events in log: {count_preemptions(log_path)}")

    return level_results


# ═══════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════

COL_W = 100


def print_summary_table(
    all_runs: dict[float, list[tuple[LevelResult, bool, list[str]]]],
):
    print()
    print("=" * COL_W)
    print("EXPERIMENT 2c RESULTS: --gpu-memory-utilization (Option C — Sustained Arrival Rate)")
    print(f"Model: {MODEL_ID}  |  max-num-seqs: {MAX_NUM_SEQS} (uncapped)")
    print(
        f"Level: {DURATION_PER_LEVEL_S}s total | "
        f"{WARMUP_PER_LEVEL_S}s warmup | "
        f"{MEASUREMENT_WINDOW_S}s measurement"
    )
    print(
        f"Degradation: error_rate>{DEGRADE_ERROR_RATE:.0%}  OR  "
        f"TTFT>{DEGRADE_TTFT_MULT}× baseline  OR  "
        f"TTFT trend>{DEGRADE_TREND_RATIO}× within window"
    )
    print("=" * COL_W)

    header = (
        f"  {'RPS':>5}  {'n_ok':>5}  {'n_tot':>5}  {'err%':>5}  "
        f"{'TTFT p50':>9}  {'TTFT p99':>9}  "
        f"{'lat p50':>8}  {'tok/s':>6}  "
        f"{'act_rps':>7}  {'trend':>7}  Status"
    )
    divider = "  " + "─" * (len(header) - 2)

    for util, levels in all_runs.items():
        if not levels:
            continue
        print(f"\n  ── gpu_memory_utilization={util} ──────────────────────")
        print(header)
        print(divider)
        for level, degraded, reasons in levels:
            t50    = f"{level.ttft_p50:.3f}s" if level.ttft_p50 is not None else "N/A"
            t99    = f"{level.ttft_p99:.3f}s" if level.ttft_p99 is not None else "N/A"
            l50    = f"{level.lat_p50:.2f}s"  if level.lat_p50  is not None else "N/A"
            trend  = level.ttft_trend_ratio()
            t_str  = f"{trend:.2f}×" if trend is not None else "N/A"
            status = "DEGRADED" if degraded else "clean"
            reason_str = f" [{', '.join(reasons)}]" if reasons else ""
            print(
                f"  {level.rps:>5.1f}  {level.n_success:>5}  {level.n:>5}  "
                f"{level.error_rate:>4.1%}  "
                f"{t50:>9}  {t99:>9}  "
                f"{l50:>8}  {level.throughput_tok_s:>6.0f}  "
                f"{level.actual_rps:>7.1f}  {t_str:>7}  "
                f"{status}{reason_str}"
            )


def save_json(
    all_runs: dict[float, list[tuple[LevelResult, bool, list[str]]]],
    path: Path,
):
    out: dict = {}
    for util, levels in all_runs.items():
        out[str(util)] = [
            {
                "rps":               lvl.rps,
                "n":                 lvl.n,
                "n_success":         lvl.n_success,
                "error_rate":        lvl.error_rate,
                "ttft_p50":          lvl.ttft_p50,
                "ttft_p99":          lvl.ttft_p99,
                "lat_p50":           lvl.lat_p50,
                "lat_p99":           lvl.lat_p99,
                "throughput_tok_s":  lvl.throughput_tok_s,
                "actual_rps":        lvl.actual_rps,
                "ttft_trend_ratio":  lvl.ttft_trend_ratio(),
                "degraded":          degraded,
                "degradation_reasons": reasons,
            }
            for lvl, degraded, reasons in levels
        ]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    print("=" * COL_W)
    print("Day 6 — Experiment 2c: --gpu-memory-utilization (Option C: Sustained Arrival Rate)")
    print(f"Model:           {MODEL_ID}")
    print(f"dtype:           {DTYPE}  (T4 compute capability 7.5 — no BF16)")
    print(f"max-model-len:   {MAX_MODEL_LEN}  (capped from 32768 to allow low-util startup)")
    print(f"GPU util sweep:  {GPU_MEM_UTIL_VALUES}")
    print(f"RPS sweep:       {RPS_RAMP}")
    print(
        f"Level timing:    {DURATION_PER_LEVEL_S}s total  |  "
        f"{WARMUP_PER_LEVEL_S}s warmup  |  "
        f"{MEASUREMENT_WINDOW_S}s measurement"
    )
    print(f"Max outstanding: {MAX_OUTSTANDING} (semaphore safety valve)")
    print()
    print("Expected capacity ceilings (corrected overhead floor: 8.44 GB for Qwen2.5-3B):")
    print("  Little's Law: max_RPS = KV_slots / W,  W ≈ 7s avg latency")
    for util, kv_gb, conc, ceil_rps in [
        (0.70, 1.75, 37,  5.3),
        (0.85, 3.94, 83,  11.9),
        (0.95, 5.39, 114, 16.3),
    ]:
        print(f"  util={util} → KV pool ~{kv_gb:.2f} GB → ~{conc:3d} concurrent → max stable ≈ {ceil_rps:.1f} RPS")
    print("=" * COL_W)

    all_runs: dict[float, list] = {}

    for util in GPU_MEM_UTIL_VALUES:
        all_runs[util] = await run_utilization(util)

    print_summary_table(all_runs)
    save_json(all_runs, JSON_OUTPUT)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

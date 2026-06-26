"""
Day 6 — Experiment 2: --gpu-memory-utilization Parameter Sensitivity
=====================================================================
For each gpu_memory_utilization value, ramps concurrent requests until
the system degrades. Records max concurrent requests before degradation onset.

Key insight: gpu_memory_utilization controls total VRAM budget. The KV pool
is the residual after ALL non-KV memory costs are subtracted:

    kv_pool = (total_vram × gpu_memory_utilization)
              - model_weights
              - pytorch_activation_peak
              - non_torch_overhead

Why short prompts (~80 tokens, not ~1000 tokens):
  With a 1000-token prompt, each prefill takes ~0.293s on the T4. At
  concurrency=8, the median request waits for ~4 prefills ahead of it,
  inflating TTFT p50 to ~3.6× baseline — before KV memory is even stressed.
  Prefill compute becomes the binding constraint, and the KV pool size never
  matters because the system can't admit enough concurrent sequences to exhaust it.

  With an 80-token prompt, each prefill takes ~15-25ms. The scheduler can
  admit many sequences before compute becomes the wall, allowing concurrency
  to ramp high enough that different KV pool sizes produce different ceilings.

KV footprint with short prompts (80 prompt + 256 max_completion = 336 tokens):
  Per request: 336 tokens × 144 KB/token ≈ 48 MB

  gpu_memory_utilization=0.60 → KV pool ~1.4 GB → ~29 concurrent requests
  gpu_memory_utilization=0.70 → KV pool ~2.9 GB → ~61 concurrent requests
  gpu_memory_utilization=0.85 → KV pool ~5.0 GB → ~106 concurrent requests
  gpu_memory_utilization=0.95 → KV pool ~6.5 GB → ~137 concurrent requests

  These exhaustion points are spaced far enough apart that the sweep should
  produce clearly different degradation onsets across utilization values.

Qwen2.5-3B-Instruct on T4 (14.56 GB usable VRAM):
  Measured overhead floor (from vLLM startup profiling):
    Weights:               5.79 GB
    Activation peak:       1.39 GB
    Non-torch overhead:    0.15 GB
    Total floor:           7.33 GB
  Minimum viable utilization: 7.33 / 14.56 ≈ 0.503 → sweep starts at 0.60

Degradation detection — TTFT p50 (not p99):
  p99 inflates with concurrency from FIFO queuing alone (worst request waits
  for N-1 prefills). p50 stays flat under normal queuing and only rises under
  genuine KV pressure: preemption cascades, block exhaustion, scheduler stalls.

Prerequisites:
  pip install aiohttp tabulate

Usage:
  python day6_exp2_gpu_memory_utilization.py
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
# Configuration — edit these constants
# ─────────────────────────────────────────────

MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://localhost:{PORT}"
MAX_MODEL_LEN = 2048

# ── The variable under test ──────────────────
# 0.50 is below the viability floor (~0.503). Sweep starts at 0.60.
GPU_MEM_UTIL_VALUES = [0.60, 0.70, 0.85, 0.95]

# Fixed — high enough to never be the bottleneck here.
MAX_NUM_SEQS = 256

# ── Concurrency ramp ─────────────────────────
# Needs to reach ~150 to cover the 0.95 exhaustion point (~137 concurrent).
# Dense at low end to catch 0.60 exhaustion (~29 concurrent) precisely.
CONCURRENCY_RAMP = [1, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 144, 160]

# 30 requests per step gives stable p50 estimates.
REQUESTS_PER_LEVEL = 30

MAX_COMPLETION_TOKENS = 256

# ── Prompt (~80 tokens) ──────────────────────
# Short prompt keeps per-prefill time ~15-25ms so the scheduler can admit
# many sequences before compute saturates. This lets the KV pool become
# the binding constraint rather than prefill throughput.
#
# At 80 prompt + 256 max_completion = 336 total tokens per request:
#   KV footprint ≈ 336 × 144 KB ≈ 48 MB per request
#   Pool at 0.60 (~1.4 GB) → ~29 concurrent before exhaustion
#   Pool at 0.95 (~6.5 GB) → ~137 concurrent before exhaustion
PROMPT = (
    "Summarize the key differences between supervised and unsupervised "
    "machine learning, including examples of each approach and their "
    "typical use cases in industry. "
) * 3  # ~75-85 tokens

# ── Degradation detection ────────────────────
# TTFT p50 > 2× baseline OR error rate > 10%.
# With short prompts the baseline TTFT p50 will be much lower (~50-100ms),
# so 2× is a tighter absolute threshold than it was with long prompts.
# If this fires too early due to residual queuing noise, raise to 3×.
DEGRADATION_ERROR_RATE_THRESHOLD = 0.10
DEGRADATION_TTFT_P50_MULTIPLIER = 2.0

STARTUP_TIMEOUT_S = 180
SHUTDOWN_WAIT_S = 8
WARMUP_REQUESTS = 3


# ─────────────────────────────────────────────
# Qwen2.5-3B KV geometry + pool estimates
# ─────────────────────────────────────────────
# Measured from vLLM startup log (not guessed):
GPU_TOTAL_GB = 14.56      # T4 usable VRAM
OVERHEAD_GB = 7.33        # weights (5.79) + activation peak (1.39) + non-torch (0.15)

# Qwen2.5-3B architecture
KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 36
DTYPE_BYTES = 2           # FP16

# Expected tokens per request under short-prompt workload:
# 80 prompt + 256 max_completion = 336 total
TOKENS_PER_REQUEST = 80 + MAX_COMPLETION_TOKENS  # 336


def kv_bytes_per_request(tokens: int) -> float:
    per_token = 2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS
    return per_token * tokens


def estimated_max_concurrent(gpu_mem_util: float) -> int:
    """Theoretical max concurrent requests before KV pool exhaustion."""
    kv_pool_bytes = max(0.0, GPU_TOTAL_GB * gpu_mem_util - OVERHEAD_GB) * (1024 ** 3)
    per_req = kv_bytes_per_request(TOKENS_PER_REQUEST)
    return int(kv_pool_bytes / per_req) if per_req > 0 else 0


def kv_pool_gb(gpu_mem_util: float) -> float:
    return max(0.0, GPU_TOTAL_GB * gpu_mem_util - OVERHEAD_GB)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class RequestResult:
    status: str = "ok"
    ttft: float = 0.0
    total_latency: float = 0.0
    completion_tokens: int = 0
    error_detail: str = ""


@dataclass
class RampStepResult:
    concurrency: int = 0
    throughput_tok_s: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    wall_time: float = 0.0
    degraded: bool = False
    degradation_reason: str = ""


@dataclass
class SweepResult:
    gpu_memory_utilization: float = 0.0
    ramp_steps: list = field(default_factory=list)
    baseline_ttft_p50: float = 0.0
    max_clean_concurrency: int = 0
    degradation_onset: int = 0        # 0 = never reached within ramp
    preemption_count: int = 0


# ─────────────────────────────────────────────
# vLLM Process Management
# ─────────────────────────────────────────────

def start_vllm(gpu_memory_utilization: float) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--dtype", "half",
        "--port", str(PORT),
        "--disable-log-requests",
    ]

    label = f"{gpu_memory_utilization:.2f}".replace(".", "p")
    log_path = Path(f"vllm_gmem_{label}.log")

    print(f"\n  Starting vLLM: --gpu-memory-utilization {gpu_memory_utilization}")
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
    print(f"  ERROR: vLLM failed to start within {STARTUP_TIMEOUT_S}s. Check {log_path}")
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


def count_preemptions_in_log(gpu_memory_utilization: float) -> int:
    label = f"{gpu_memory_utilization:.2f}".replace(".", "p")
    log_path = Path(f"vllm_gmem_{label}.log")
    if not log_path.exists():
        return 0
    return len(re.findall(r"(?i)preempt", log_path.read_text(errors="replace")))


# ─────────────────────────────────────────────
# Async Request Client
# ─────────────────────────────────────────────

async def send_streaming_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0.7,
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
                        error_detail=f"HTTP {resp.status}: {body[:200]}",
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
                error_detail=str(exc)[:200],
            )

        t_end = time.perf_counter()
        return RequestResult(
            status="ok",
            ttft=(t_first_token - t_start) if t_first_token else (t_end - t_start),
            total_latency=t_end - t_start,
            completion_tokens=tokens_received,
        )


# ─────────────────────────────────────────────
# Metrics Helpers
# ─────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


async def run_ramp_step(concurrency: int) -> RampStepResult:
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_streaming_request(session, semaphore)
                 for _ in range(REQUESTS_PER_LEVEL)]
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    total_tokens = sum(r.completion_tokens for r in ok)

    return RampStepResult(
        concurrency=concurrency,
        throughput_tok_s=total_tokens / wall_time if wall_time > 0 else 0.0,
        ttft_p50=percentile([r.ttft for r in ok], 50),
        ttft_p99=percentile([r.ttft for r in ok], 99),
        latency_p50=percentile([r.total_latency for r in ok], 50),
        latency_p99=percentile([r.total_latency for r in ok], 99),
        success_count=len(ok),
        error_count=len(errors),
        total_tokens=total_tokens,
        wall_time=wall_time,
    )


# ─────────────────────────────────────────────
# Degradation Detection
# ─────────────────────────────────────────────

def check_degradation(
    step: RampStepResult,
    baseline_ttft_p50: float,
) -> tuple[bool, str]:
    """
    TTFT p50 is the primary signal. With short prompts (~80 tokens) the
    baseline TTFT p50 is low (~50-100ms), so p50 queuing inflation is
    proportionally smaller and the 2× threshold is less likely to false-fire
    from pure prefill queuing. If it still false-fires, raise the multiplier
    to 3× — the key signal is a sharp nonlinear spike, not a gentle ramp.
    """
    total = step.success_count + step.error_count
    error_rate = step.error_count / total if total > 0 else 0.0

    if error_rate > DEGRADATION_ERROR_RATE_THRESHOLD:
        return True, f"error_rate={error_rate:.2f} (>{DEGRADATION_ERROR_RATE_THRESHOLD:.2f})"

    if baseline_ttft_p50 > 0:
        multiplier = step.ttft_p50 / baseline_ttft_p50
        if multiplier > DEGRADATION_TTFT_P50_MULTIPLIER:
            return True, f"ttft_p50={multiplier:.1f}x baseline (>{DEGRADATION_TTFT_P50_MULTIPLIER:.1f}x)"

    return False, ""


# ─────────────────────────────────────────────
# Per-Value Sweep
# ─────────────────────────────────────────────

async def warmup(n: int = WARMUP_REQUESTS):
    connector = aiohttp.TCPConnector(limit=2)
    sem = asyncio.Semaphore(1)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(n):
            await send_streaming_request(session, sem)


async def run_sweep(gpu_memory_utilization: float) -> SweepResult:
    sweep = SweepResult(gpu_memory_utilization=gpu_memory_utilization)

    print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
    await warmup()

    baseline_ttft_p50 = 0.0
    max_clean = 0
    degradation_onset = 0

    for conc in CONCURRENCY_RAMP:
        print(f"  → concurrency={conc:4d}  ", end="", flush=True)
        step = await run_ramp_step(conc)

        if conc == 1:
            baseline_ttft_p50 = step.ttft_p50
            sweep.baseline_ttft_p50 = baseline_ttft_p50

        degraded, reason = check_degradation(step, baseline_ttft_p50)
        step.degraded = degraded
        step.degradation_reason = reason

        total = step.success_count + step.error_count
        icon = "✗ DEGRADED" if degraded else "✓ clean   "
        print(
            f"{icon}  "
            f"TTFT p50={step.ttft_p50:.3f}s  "
            f"TTFT p99={step.ttft_p99:.3f}s  "
            f"throughput={step.throughput_tok_s:.0f} tok/s  "
            f"errors={step.error_count}/{total}"
            + (f"  [{reason}]" if reason else "")
        )

        sweep.ramp_steps.append(step)

        if not degraded:
            max_clean = conc
        elif degradation_onset == 0:
            degradation_onset = conc
            print(f"  Degradation onset at concurrency={conc}. Stopping ramp.")
            break

    sweep.max_clean_concurrency = max_clean
    sweep.degradation_onset = degradation_onset
    sweep.preemption_count = count_preemptions_in_log(gpu_memory_utilization)

    if sweep.preemption_count > 0:
        print(f"  Preemption events in vLLM log: {sweep.preemption_count}")

    return sweep


# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────

def print_summary(sweeps: list[SweepResult]):
    headers = [
        "gpu_memory\n_utilization",
        "KV pool\n(~GB)",
        "Theoretical\nmax conc.",
        "Max clean\nconcurrency",
        "Degradation\nonset",
        "TTFT p50\n@ baseline",
        "TTFT p50\n@ max clean",
        "Preemption\nevents",
    ]
    rows = []
    for s in sweeps:
        clean_steps = [st for st in s.ramp_steps if not st.degraded]
        ttft_at_max_clean = clean_steps[-1].ttft_p50 if clean_steps else 0.0
        onset_label = str(s.degradation_onset) if s.degradation_onset > 0 else "not reached"
        rows.append([
            f"{s.gpu_memory_utilization:.2f}",
            f"~{kv_pool_gb(s.gpu_memory_utilization):.1f}",
            estimated_max_concurrent(s.gpu_memory_utilization),
            s.max_clean_concurrency,
            onset_label,
            f"{s.baseline_ttft_p50:.3f}s",
            f"{ttft_at_max_clean:.3f}s",
            s.preemption_count,
        ])

    print("\n" + "=" * 90)
    print("EXPERIMENT 2 RESULTS: --gpu-memory-utilization Parameter Sensitivity")
    print(f"Model: {MODEL}  |  max-num-seqs: {MAX_NUM_SEQS} (uncapped)")
    print(f"Prompt: ~80 tokens  |  Max completion: {MAX_COMPLETION_TOKENS} tokens")
    print(
        f"Degradation: error_rate>{DEGRADATION_ERROR_RATE_THRESHOLD:.0%}  OR  "
        f"TTFT p50>{DEGRADATION_TTFT_P50_MULTIPLIER:.1f}× baseline"
    )
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))
    print()
    print("Note: 'Theoretical max conc.' uses measured overhead floor of "
          f"{OVERHEAD_GB} GB ({GPU_TOTAL_GB} GB usable VRAM).")
    print(f"      KV geometry: {KV_HEADS} KV heads × {HEAD_DIM} head_dim × "
          f"{NUM_LAYERS} layers × FP16 = "
          f"{2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS // 1024} KB/token")
    print(f"      Assumes {TOKENS_PER_REQUEST} tokens/request "
          f"(~80 prompt + {MAX_COMPLETION_TOKENS} max completion).")
    print()


def print_ramp_detail(sweep: SweepResult):
    headers = [
        "concurrency",
        "Throughput\n(tok/s)",
        "TTFT p50\n(s)",
        "TTFT p99\n(s)",
        "Latency p50\n(s)",
        "Latency p99\n(s)",
        "Errors",
        "Status",
    ]
    rows = []
    for st in sweep.ramp_steps:
        rows.append([
            st.concurrency,
            f"{st.throughput_tok_s:.1f}",
            f"{st.ttft_p50:.3f}",
            f"{st.ttft_p99:.3f}",
            f"{st.latency_p50:.3f}",
            f"{st.latency_p99:.3f}",
            f"{st.error_count}/{st.success_count + st.error_count}",
            f"DEGRADED [{st.degradation_reason}]" if st.degraded else "clean",
        ])
    print(f"\n  Ramp detail — gpu_memory_utilization={sweep.gpu_memory_utilization:.2f}  "
          f"(baseline TTFT p50={sweep.baseline_ttft_p50:.3f}s):")
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))


def save_results(sweeps: list[SweepResult]):
    output = {
        "experiment": "day6_exp2_gpu_memory_utilization",
        "config": {
            "model": MODEL,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "concurrency_ramp": CONCURRENCY_RAMP,
            "requests_per_level": REQUESTS_PER_LEVEL,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "prompt_approx_tokens": 80,
            "tokens_per_request_assumed": TOKENS_PER_REQUEST,
            "degradation_error_rate_threshold": DEGRADATION_ERROR_RATE_THRESHOLD,
            "degradation_ttft_p50_multiplier": DEGRADATION_TTFT_P50_MULTIPLIER,
            "gpu_constants": {
                "gpu_total_gb": GPU_TOTAL_GB,
                "overhead_gb": OVERHEAD_GB,
                "note": "overhead = weights(5.79) + activation_peak(1.39) + non_torch(0.15)",
            },
            "kv_geometry": {
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": NUM_LAYERS,
                "dtype_bytes": DTYPE_BYTES,
                "kb_per_token": 2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS // 1024,
            },
        },
        "sweeps": [
            {
                "gpu_memory_utilization": s.gpu_memory_utilization,
                "kv_pool_gb": kv_pool_gb(s.gpu_memory_utilization),
                "theoretical_max_concurrent": estimated_max_concurrent(s.gpu_memory_utilization),
                "baseline_ttft_p50": s.baseline_ttft_p50,
                "max_clean_concurrency": s.max_clean_concurrency,
                "degradation_onset": s.degradation_onset,
                "preemption_count": s.preemption_count,
                "ramp_steps": [
                    {
                        "concurrency": st.concurrency,
                        "throughput_tok_s": st.throughput_tok_s,
                        "ttft_p50": st.ttft_p50,
                        "ttft_p99": st.ttft_p99,
                        "latency_p50": st.latency_p50,
                        "latency_p99": st.latency_p99,
                        "success_count": st.success_count,
                        "error_count": st.error_count,
                        "total_tokens": st.total_tokens,
                        "wall_time": st.wall_time,
                        "degraded": st.degraded,
                        "degradation_reason": st.degradation_reason,
                    }
                    for st in s.ramp_steps
                ],
            }
            for s in sweeps
        ],
    }
    out_path = Path("day6_exp2_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Raw results saved to {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Day 6 — Experiment 2: --gpu-memory-utilization Parameter Sensitivity")
    print(f"Model:            {MODEL}")
    print(f"Sweep values:     {GPU_MEM_UTIL_VALUES}")
    print(f"max-num-seqs:     {MAX_NUM_SEQS} (fixed — not under test)")
    print(f"Concurrency ramp: {CONCURRENCY_RAMP}")
    print(f"Requests/level:   {REQUESTS_PER_LEVEL}")
    print(
        f"Degradation:      error_rate>{DEGRADATION_ERROR_RATE_THRESHOLD:.0%}  OR  "
        f"TTFT p50>{DEGRADATION_TTFT_P50_MULTIPLIER:.1f}× baseline"
    )
    print()
    print(f"T4 usable VRAM: {GPU_TOTAL_GB} GB  |  "
          f"Overhead floor: {OVERHEAD_GB} GB  |  "
          f"KV geometry: {2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS // 1024} KB/token")
    print(f"Prompt: ~80 tokens  |  "
          f"Max completion: {MAX_COMPLETION_TOKENS} tokens  |  "
          f"Tokens/request assumed: {TOKENS_PER_REQUEST}")
    print()
    print("Theoretical KV exhaustion points:")
    for v in GPU_MEM_UTIL_VALUES:
        print(f"  gpu_memory_utilization={v:.2f} → KV pool ~{kv_pool_gb(v):.1f} GB "
              f"→ ~{estimated_max_concurrent(v)} concurrent requests")
    print("=" * 90)

    all_sweeps: list[SweepResult] = []

    for gmu in GPU_MEM_UTIL_VALUES:
        print(f"\n{'─' * 60}")
        print(f"  RUN: --gpu-memory-utilization={gmu}")
        print(f"{'─' * 60}")

        proc = start_vllm(gmu)
        try:
            sweep = asyncio.run(run_sweep(gmu))
            print_ramp_detail(sweep)
            all_sweeps.append(sweep)
        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            stop_vllm(proc)
            break
        finally:
            stop_vllm(proc)

    if all_sweeps:
        print_summary(all_sweeps)
        save_results(all_sweeps)

    print("Key questions to answer:")
    print("  • Do the degradation onsets differ across utilization values this time?")
    print("    If yes: KV pool is now the binding constraint — the experiment worked.")
    print("    If still uniform: something else is saturating first (decode compute,")
    print("    max-num-seqs, or the degradation threshold needs adjusting).")
    print("  • Does max_clean_concurrency track the theoretical exhaustion points?")
    print("    The ratio (observed / theoretical) tells you how accurate your")
    print("    Day 3 KV calculator is against the real system.")
    print("  • At what concurrency do preemption events appear in the log?")
    print("    Preemptions should appear at or just before the degradation onset.")
    print("  • How does TTFT p50 evolve through the ramp?")
    print("    Gradual linear rise = queuing pressure (compute bound).")
    print("    Flat then sharp spike = KV exhaustion triggering preemption cascade.")
    print("  • Compare the p50/p99 gap at high concurrency.")
    print("    Converging upward together = genuine system pressure.")
    print("    p99 much higher than p50 = tail from queuing, not memory.")


if __name__ == "__main__":
    main()

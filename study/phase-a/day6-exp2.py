"""
Day 6 — Experiment 2: --gpu-memory-utilization Parameter Sensitivity
=====================================================================
For each gpu_memory_utilization value, ramps concurrent requests until
the system degrades. Records max concurrent requests before degradation onset.

Key insight: gpu_memory_utilization controls total VRAM budget. The KV pool
is the residual after model weights are loaded:

    kv_pool_bytes = (total_vram × gpu_memory_utilization) - model_weights_vram

Llama 3.2 3B on T4 (16 GB):
  Weights (FP16):     ~6.0 GB
  KV per token:       2 × 8 heads × 128 head_dim × 2 bytes × 28 layers = 114 KB
  Per request @ 1024 tokens: ~117 MB

  gpu_memory_utilization=0.50 → budget= 8.0 GB → KV pool ~2.0 GB → ~17 concurrent
  gpu_memory_utilization=0.70 → budget=11.2 GB → KV pool ~5.2 GB → ~44 concurrent
  gpu_memory_utilization=0.85 → budget=13.6 GB → KV pool ~7.6 GB → ~65 concurrent
  gpu_memory_utilization=0.95 → budget=15.2 GB → KV pool ~9.2 GB → ~79 concurrent

These exhaustion points fall within a reachable concurrency ramp, so the
four values should produce clearly different degradation onsets — which is
what this experiment is designed to surface.

Why Llama 3.2 3B instead of TinyLlama:
  TinyLlama uses GQA with only 4 KV heads, making its KV footprint ~22 KB/token.
  At that size, even gpu_memory_utilization=0.50 holds hundreds of concurrent
  requests before exhaustion — far beyond any practical concurrency ramp.
  The pool size never becomes the bottleneck, so the sweep produces uniform
  results regardless of the parameter value. Llama 3.2 3B has 8 KV heads and
  ~5× the per-token KV footprint, bringing the exhaustion thresholds into range.

Degradation detection — TTFT p50 (not p99):
  p99 TTFT naturally inflates with concurrency just from FIFO queuing: the worst
  request waits for N-1 prefills ahead of it. This produces false positives at
  low concurrency that have nothing to do with memory pressure.
  p50 TTFT stays flat under normal queuing because most requests are served
  promptly. It only inflates when the system is genuinely under pressure:
  preemption cascades, KV pool exhaustion, or scheduler backpressure.

Prerequisites:
  huggingface-cli login   # Llama 3.2 is gated; token must have access approved
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

MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://localhost:{PORT}"

# Llama 3.2 3B fits comfortably in 16 GB at FP16 with room for KV cache.
# max_model_len=2048 keeps per-request KV footprint bounded while still
# being long enough that the pool exhaustion thresholds are meaningful.
MAX_MODEL_LEN = 2048

# ── The variable under test ──────────────────
GPU_MEM_UTIL_VALUES = [0.50, 0.70, 0.85, 0.95]

# Fixed — high enough that it is never the bottleneck in this experiment.
# Exp1 measured max-num-seqs; here it must stay out of the way.
MAX_NUM_SEQS = 256

# ── Concurrency ramp ─────────────────────────
# Spans from 1 (baseline) up through ~128, covering the estimated
# exhaustion points for all four utilization values.
# Steps are denser at low concurrency (where the 0.50 ceiling sits)
# and sparser at high concurrency (where 0.95 exhaustion sits).
CONCURRENCY_RAMP = [1, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 112, 128]

# Requests per ramp step. 30 gives stable p50 without excessive runtime.
REQUESTS_PER_LEVEL = 30

MAX_COMPLETION_TOKENS = 256

# ── Prompt ───────────────────────────────────
# ~1000 tokens. Longer prompts increase per-request KV footprint, which
# brings pool exhaustion down to reachable concurrency levels.
# At 114 KB/token × 1024 tokens ≈ 117 MB per request.
PROMPT = (
    "Explain the complete history of computing from the abacus to modern "
    "supercomputers, covering every major milestone in detail. Include "
    "discussion of Babbage's difference engine, Ada Lovelace, the invention "
    "of the transistor, the development of UNIX, the rise of personal "
    "computers, the Internet, mobile computing, and the current era of "
    "large-scale machine learning and GPU-accelerated workloads. "
) * 12  # ~960-1020 tokens depending on tokenizer

# ── Degradation detection ────────────────────
# Trigger if EITHER condition is met at a given ramp step:
#   1. Error rate > DEGRADATION_ERROR_RATE_THRESHOLD
#      Catches hard failures: rejections, timeouts, OOM-induced crashes.
#   2. TTFT p50 > DEGRADATION_TTFT_P50_MULTIPLIER × baseline (measured at conc=1)
#      Catches soft degradation: KV exhaustion causing preemption, scheduler
#      backpressure, or sequence swapping inflating median latency.
#      p50 is used (not p99) to filter out FIFO queuing tail noise.
DEGRADATION_ERROR_RATE_THRESHOLD = 0.10    # 10% hard failure rate
DEGRADATION_TTFT_P50_MULTIPLIER = 2.0      # p50 TTFT > 2× baseline

STARTUP_TIMEOUT_S = 180   # Llama 3.2 3B takes longer to load than TinyLlama
SHUTDOWN_WAIT_S = 8       # extra time for 6 GB weight unload vs TinyLlama
WARMUP_REQUESTS = 3       # throwaway requests before ramping


# ─────────────────────────────────────────────
# Llama 3.2 3B KV geometry (for summary table)
# ─────────────────────────────────────────────
# Used only for the estimated pool/exhaustion display in the summary.
# These are architectural constants, not measured at runtime.

KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 28
DTYPE_BYTES = 2          # FP16
TOKENS_PER_REQUEST = MAX_MODEL_LEN // 2   # conservative: half of max_model_len
GPU_TOTAL_GB = 16.0
MODEL_WEIGHTS_GB = 6.0   # approximate FP16 weight footprint

def kv_bytes_per_request(tokens: int) -> float:
    """KV cache bytes for one request at `tokens` total tokens."""
    per_token = 2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS
    return per_token * tokens

def estimated_max_concurrent(gpu_mem_util: float) -> int:
    """Rough theoretical max concurrent requests before KV pool exhaustion."""
    kv_pool_bytes = (GPU_TOTAL_GB * gpu_mem_util - MODEL_WEIGHTS_GB) * (1024 ** 3)
    kv_pool_bytes = max(0.0, kv_pool_bytes)
    per_req = kv_bytes_per_request(TOKENS_PER_REQUEST)
    return int(kv_pool_bytes / per_req) if per_req > 0 else 0


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class RequestResult:
    status: str = "ok"           # "ok" | "error"
    ttft: float = 0.0            # time-to-first-token (seconds)
    total_latency: float = 0.0   # wall time to last token (seconds)
    completion_tokens: int = 0   # approximate tokens received
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
    ramp_steps: list = field(default_factory=list)   # list[RampStepResult]
    baseline_ttft_p50: float = 0.0    # TTFT p50 at concurrency=1
    max_clean_concurrency: int = 0    # last step with degraded=False
    degradation_onset: int = 0        # first step with degraded=True (0=never)
    preemption_count: int = 0


# ─────────────────────────────────────────────
# vLLM Process Management
# ─────────────────────────────────────────────

def start_vllm(gpu_memory_utilization: float) -> subprocess.Popen:
    """Launch vLLM with the given gpu_memory_utilization and wait until ready."""
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
    """Terminate vLLM and wait for GPU memory to drain."""
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
    text = log_path.read_text(errors="replace")
    return len(re.findall(r"(?i)preempt", text))


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
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


async def run_ramp_step(concurrency: int) -> RampStepResult:
    """Fire REQUESTS_PER_LEVEL requests at the given concurrency and collect metrics."""
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_streaming_request(session, semaphore)
            for _ in range(REQUESTS_PER_LEVEL)
        ]
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    total_tokens = sum(r.completion_tokens for r in ok)
    ttfts = [r.ttft for r in ok]
    latencies = [r.total_latency for r in ok]

    return RampStepResult(
        concurrency=concurrency,
        throughput_tok_s=total_tokens / wall_time if wall_time > 0 else 0.0,
        ttft_p50=percentile(ttfts, 50),
        ttft_p99=percentile(ttfts, 99),
        latency_p50=percentile(latencies, 50),
        latency_p99=percentile(latencies, 99),
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
    Returns (degraded, reason).

    Uses TTFT p50 (not p99) as the latency signal. p99 inflates naturally
    with concurrency due to FIFO queuing — the worst request always waits
    for N-1 prefills ahead of it. p50 stays flat under normal queuing and
    only inflates when the system is genuinely under pressure: KV pool
    exhaustion forcing preemption, sequence swapping, or scheduler stalls.
    """
    total = step.success_count + step.error_count
    error_rate = step.error_count / total if total > 0 else 0.0

    if error_rate > DEGRADATION_ERROR_RATE_THRESHOLD:
        return True, f"error_rate={error_rate:.2f} (>{DEGRADATION_ERROR_RATE_THRESHOLD:.2f})"

    if baseline_ttft_p50 > 0:
        multiplier = step.ttft_p50 / baseline_ttft_p50
        if multiplier > DEGRADATION_TTFT_P50_MULTIPLIER:
            return True, f"ttft_p50_spike={multiplier:.1f}x baseline (>{DEGRADATION_TTFT_P50_MULTIPLIER:.1f}x)"

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
        kv_gb = max(0.0, GPU_TOTAL_GB * s.gpu_memory_utilization - MODEL_WEIGHTS_GB)
        theoretical = estimated_max_concurrent(s.gpu_memory_utilization)
        clean_steps = [st for st in s.ramp_steps if not st.degraded]
        ttft_at_max_clean = clean_steps[-1].ttft_p50 if clean_steps else 0.0
        onset_label = str(s.degradation_onset) if s.degradation_onset > 0 else "not reached"
        rows.append([
            f"{s.gpu_memory_utilization:.2f}",
            f"~{kv_gb:.1f}",
            theoretical,
            s.max_clean_concurrency,
            onset_label,
            f"{s.baseline_ttft_p50:.3f}s",
            f"{ttft_at_max_clean:.3f}s",
            s.preemption_count,
        ])

    print("\n" + "=" * 90)
    print("EXPERIMENT 2 RESULTS: --gpu-memory-utilization Parameter Sensitivity")
    print(f"Model: {MODEL}  |  max-num-seqs: {MAX_NUM_SEQS} (uncapped)")
    print(f"Prompt: ~1000 tokens  |  Max completion: {MAX_COMPLETION_TOKENS} tokens")
    print(
        f"Degradation: error_rate>{DEGRADATION_ERROR_RATE_THRESHOLD:.0%}  OR  "
        f"TTFT p50>{DEGRADATION_TTFT_P50_MULTIPLIER:.1f}× baseline"
    )
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))
    print()
    print("Note: 'Theoretical max conc.' is derived from KV geometry math, not measured.")
    print(f"      Assumes {TOKENS_PER_REQUEST} tokens/request average "
          f"({KV_HEADS} KV heads × {HEAD_DIM} head_dim × {NUM_LAYERS} layers × FP16).")
    print("      Compare against 'Max clean concurrency' to validate the Day 3 calculator.")
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
            "degradation_error_rate_threshold": DEGRADATION_ERROR_RATE_THRESHOLD,
            "degradation_ttft_p50_multiplier": DEGRADATION_TTFT_P50_MULTIPLIER,
            "kv_geometry": {
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": NUM_LAYERS,
                "dtype_bytes": DTYPE_BYTES,
                "tokens_per_request_assumed": TOKENS_PER_REQUEST,
            },
        },
        "sweeps": [
            {
                "gpu_memory_utilization": s.gpu_memory_utilization,
                "baseline_ttft_p50": s.baseline_ttft_p50,
                "max_clean_concurrency": s.max_clean_concurrency,
                "degradation_onset": s.degradation_onset,
                "preemption_count": s.preemption_count,
                "theoretical_max_concurrent": estimated_max_concurrent(s.gpu_memory_utilization),
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
    # Preflight: warn if no HuggingFace token is present.
    # Llama 3.2 is gated — vLLM will fail silently at startup without access.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_cache = Path.home() / ".cache" / "huggingface" / "token"
    if not hf_token and not hf_cache.exists():
        print("WARNING: No HuggingFace token found.")
        print("  Llama 3.2 is a gated model. Run: huggingface-cli login")
        print("  or set HF_TOKEN in your environment before continuing.")
        print()

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
    print("Theoretical KV exhaustion points (Llama 3.2 3B, ~1024 tokens/request):")
    for v in GPU_MEM_UTIL_VALUES:
        kv_gb = max(0.0, GPU_TOTAL_GB * v - MODEL_WEIGHTS_GB)
        print(f"  gpu_memory_utilization={v:.2f} → KV pool ~{kv_gb:.1f} GB "
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
    print("  • Does max_clean_concurrency track the theoretical exhaustion points?")
    print("    If not — what is the actual binding constraint?")
    print("  • At what utilization value do preemption events first appear in the log?")
    print("  • Does TTFT p50 stay flat until near the degradation onset, then spike?")
    print("    That shape confirms KV exhaustion as the cause, not compute saturation.")
    print("  • Compare TTFT p50 vs p99 at the same concurrency level.")
    print("    A large p50/p99 gap at moderate concurrency = queuing noise.")
    print("    p50 and p99 both spiking together = genuine system pressure.")
    print("  • Cross-check: does your Day 3 KV cache calculator predict the same")
    print("    exhaustion point as max_clean_concurrency for each utilization value?")


if __name__ == "__main__":
    main()

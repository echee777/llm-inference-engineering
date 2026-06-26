"""
Day 24 — Latency vs. Utilization Cliff Experiment

Concurrency-driven sweep that maps actual KV utilization to TTFT and preemption.
For each concurrency level, runs warmup + stabilization, then samples a fixed
window. Records per-point metrics (CSV) and per-request streams (CSV).

Why concurrency-controlled instead of rate-controlled:
The syllabus says "rate-controlled" but in practice, near the cliff, request
latency explodes and rate-controlled experiments diverge (queue grows
unboundedly). Concurrency-controlled experiments converge to a steady state
because each completed request is replaced by a new one, keeping in-flight
load constant. The reported x-axis is the *measured* KV util, not target —
this is exactly what the syllabus requires ("x-axis is only valid if it
reflects measured utilization").

Per-point metrics captured:
  - TTFT p50 / p95 / p99 (ms)
  - Inter-token latency p50 / p99 (ms)
  - Queue depth (mean over sample window)
  - Queue growth rate (ΔQueue/Δt) — fraction of samples with positive slope
  - Preemption rate (events/min)
  - KV utilization % (mean over sample window)
  - Throughput (output tok/s)
"""

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp


VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Workload spec — homogeneous medium prompts (Day 24 syllabus Step 1)
PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 256

# Timing — syllabus Step 1 (defaults; overridable from CLI)
DEFAULT_WARMUP_SECS = 60          # Allow workers to fill the pipe
DEFAULT_STABILIZATION_SECS = 180  # 3 min before sampling
DEFAULT_SAMPLE_SECS = 120         # 2 min sample window
SAMPLE_INTERVAL_SECS = 5          # 5 s metric polling cadence

# Mutable runtime values populated from CLI in main()
WARMUP_SECS = DEFAULT_WARMUP_SECS
STABILIZATION_SECS = DEFAULT_STABILIZATION_SECS
SAMPLE_SECS = DEFAULT_SAMPLE_SECS


@dataclass
class RequestRecord:
    """One completed request's measurements."""
    request_id: int
    concurrency: int
    repeat: int
    sent_at: float          # monotonic seconds since run start
    ttft_ms: float
    total_ms: float
    output_tokens: int
    inter_token_ms: float   # avg ms per token after first
    error: str = ""


@dataclass
class MetricsSample:
    """One snapshot from /metrics during sampling window."""
    elapsed: float
    kv_util_pct: float
    queue_depth: int
    running_reqs: int
    preemptions_total: int


@dataclass
class LevelSummary:
    """Aggregated stats for one concurrency level run."""
    concurrency: int
    repeat: int
    target_label: str
    duration: float
    completed: int
    errors: int
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    ttft_max: float = 0.0
    itl_p50: float = 0.0
    itl_p99: float = 0.0
    throughput_tps: float = 0.0
    kv_util_pct_mean: float = 0.0
    kv_util_pct_max: float = 0.0
    queue_depth_mean: float = 0.0
    queue_depth_max: int = 0
    queue_growth_frac: float = 0.0  # fraction of intervals with ΔQueue > 0
    preemption_rate_per_min: float = 0.0
    divergence_ratio: float = 0.0   # p99 / p50


def make_prompt(token_count: int) -> str:
    """Build a deterministic prompt of approximately `token_count` Qwen tokens.

    Diverse sequential numbers resist prefix caching and BPE compression.
    Roughly 3.5 tokens per number for Qwen2.5.
    """
    num_count = max(1, int(token_count / 3.5))
    return " ".join(str(i) for i in range(num_count))


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    request_id: int,
    concurrency: int,
    repeat: int,
    run_start: float,
) -> RequestRecord:
    """Send one streaming chat completion. Capture TTFT and inter-token latency."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "min_tokens": MAX_NEW_TOKENS,  # forces full decode regardless of EOS
        "stream": True,
        "temperature": 0.7,
    }

    sent_at = time.monotonic() - run_start
    start = time.monotonic()
    ttft = 0.0
    last_token_time = 0.0
    inter_token_total_ms = 0.0
    output_tokens = 0

    try:
        async with session.post(
            f"{VLLM_BASE}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestRecord(
                    request_id=request_id,
                    concurrency=concurrency,
                    repeat=repeat,
                    sent_at=sent_at,
                    ttft_ms=0,
                    total_ms=0,
                    output_tokens=0,
                    inter_token_ms=0,
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
                        now = time.monotonic()
                        if ttft == 0.0:
                            ttft = (now - start) * 1000
                            last_token_time = now
                        else:
                            inter_token_total_ms += (now - last_token_time) * 1000
                            last_token_time = now
                        output_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except asyncio.TimeoutError:
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            repeat=repeat,
            sent_at=sent_at,
            ttft_ms=0,
            total_ms=0,
            output_tokens=0,
            inter_token_ms=0,
            error="timeout_600s",
        )
    except Exception as e:
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            repeat=repeat,
            sent_at=sent_at,
            ttft_ms=0,
            total_ms=0,
            output_tokens=0,
            inter_token_ms=0,
            error=str(e)[:200],
        )

    total_ms = (time.monotonic() - start) * 1000
    inter_token_ms = (
        inter_token_total_ms / (output_tokens - 1) if output_tokens > 1 else 0.0
    )
    return RequestRecord(
        request_id=request_id,
        concurrency=concurrency,
        repeat=repeat,
        sent_at=sent_at,
        ttft_ms=ttft,
        total_ms=total_ms,
        output_tokens=output_tokens,
        inter_token_ms=inter_token_ms,
    )


async def fetch_metrics(session: aiohttp.ClientSession) -> dict:
    """Pull a single snapshot from vLLM's /metrics endpoint."""
    out = {
        "kv_util_pct": 0.0,
        "queue_depth": 0,
        "running_reqs": 0,
        "preemptions_total": 0,
    }
    try:
        async with session.get(
            f"{VLLM_BASE}/metrics", timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            text = await resp.text()
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    try:
                        out["kv_util_pct"] = float(line.split()[-1]) * 100
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("vllm:num_requests_waiting{"):
                    try:
                        out["queue_depth"] = int(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("vllm:num_requests_running{"):
                    try:
                        out["running_reqs"] = int(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("vllm:num_preemptions_total{"):
                    try:
                        out["preemptions_total"] = int(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass
    return out


def percentile(data: list, p: float) -> float:
    """Inclusive percentile over a sorted list (handles empty)."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[idx]


async def run_level(
    concurrency: int,
    repeat: int,
    target_label: str,
    run_csv_writer,
    request_csv_writer,
) -> LevelSummary:
    """Run one concurrency level: warmup → stabilization → 2 min sample window.

    Workers loop continuously sending requests until stop_event is set.
    During the sample window, metrics are polled and per-request results
    captured.
    """
    prompt = make_prompt(PROMPT_TOKENS)
    summary = LevelSummary(
        concurrency=concurrency,
        repeat=repeat,
        target_label=target_label,
        duration=0.0,
        completed=0,
        errors=0,
    )

    all_results: list[RequestRecord] = []
    request_counter = [0]
    stop_event = asyncio.Event()
    sample_start_time: list[float] = []  # set when sampling begins

    run_start = time.monotonic()

    async def worker(session: aiohttp.ClientSession):
        while not stop_event.is_set():
            request_counter[0] += 1
            req_id = request_counter[0]
            r = await send_request(
                session, prompt, req_id, concurrency, repeat, run_start
            )
            all_results.append(r)
            if r.error:
                # Small backoff on error to avoid tight loop
                await asyncio.sleep(1)

    connector = aiohttp.TCPConnector(limit=concurrency + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(concurrency)]

        # Phase 1: warmup
        print(
            f"  [c={concurrency} r{repeat}] warmup {WARMUP_SECS}s...",
            flush=True,
        )
        await asyncio.sleep(WARMUP_SECS)

        # Phase 2: stabilization (no sampling, just let things settle)
        print(
            f"  [c={concurrency} r{repeat}] stabilize {STABILIZATION_SECS}s...",
            flush=True,
        )
        # Periodic status during stabilization
        elapsed = 0
        check_interval = 30
        async with aiohttp.ClientSession() as ms:
            while elapsed < STABILIZATION_SECS:
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                m = await fetch_metrics(ms)
                print(
                    f"  [c={concurrency} r{repeat}] stab t={elapsed}s "
                    f"kv={m['kv_util_pct']:.1f}% run={m['running_reqs']} "
                    f"wait={m['queue_depth']} preempt={m['preemptions_total']}",
                    flush=True,
                )

        # Phase 3: sampling window
        print(
            f"  [c={concurrency} r{repeat}] SAMPLE {SAMPLE_SECS}s...",
            flush=True,
        )
        sample_start_time.append(time.monotonic())
        measurement_start_idx = len(all_results)
        samples: list[MetricsSample] = []

        elapsed = 0
        async with aiohttp.ClientSession() as ms:
            while elapsed < SAMPLE_SECS:
                await asyncio.sleep(SAMPLE_INTERVAL_SECS)
                elapsed += SAMPLE_INTERVAL_SECS
                m = await fetch_metrics(ms)
                samples.append(
                    MetricsSample(
                        elapsed=elapsed,
                        kv_util_pct=m["kv_util_pct"],
                        queue_depth=m["queue_depth"],
                        running_reqs=m["running_reqs"],
                        preemptions_total=m["preemptions_total"],
                    )
                )
                print(
                    f"  [c={concurrency} r{repeat}] samp t={elapsed}s "
                    f"kv={m['kv_util_pct']:.1f}% run={m['running_reqs']} "
                    f"wait={m['queue_depth']} preempt={m['preemptions_total']}",
                    flush=True,
                )

        # Stop workers
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate
    measured = all_results[measurement_start_idx:]
    successful = [r for r in measured if not r.error and r.ttft_ms > 0]
    errors = [r for r in measured if r.error]
    summary.duration = SAMPLE_SECS
    summary.completed = len(successful)
    summary.errors = len(errors)

    if successful:
        ttfts = [r.ttft_ms for r in successful]
        itls = [r.inter_token_ms for r in successful if r.inter_token_ms > 0]
        summary.ttft_p50 = percentile(ttfts, 50)
        summary.ttft_p95 = percentile(ttfts, 95)
        summary.ttft_p99 = percentile(ttfts, 99)
        summary.ttft_max = max(ttfts)
        summary.itl_p50 = percentile(itls, 50)
        summary.itl_p99 = percentile(itls, 99)
        if summary.ttft_p50 > 0:
            summary.divergence_ratio = summary.ttft_p99 / summary.ttft_p50
        total_out = sum(r.output_tokens for r in successful)
        summary.throughput_tps = total_out / summary.duration

    if samples:
        kvs = [s.kv_util_pct for s in samples]
        qds = [s.queue_depth for s in samples]
        summary.kv_util_pct_mean = statistics.mean(kvs)
        summary.kv_util_pct_max = max(kvs)
        summary.queue_depth_mean = statistics.mean(qds)
        summary.queue_depth_max = max(qds)

        # Queue growth fraction: # of intervals where ΔQueue > 0 / total intervals
        # Syllabus: ΔQueue/Δt > 0 over ≥80% of sampling window
        if len(qds) > 1:
            deltas = [qds[i + 1] - qds[i] for i in range(len(qds) - 1)]
            summary.queue_growth_frac = sum(1 for d in deltas if d > 0) / len(deltas)

        # Preemption rate (events/min) from start to end of sample window
        preempts = [s.preemptions_total for s in samples]
        delta_preempts = preempts[-1] - preempts[0]
        sample_minutes = (samples[-1].elapsed - samples[0].elapsed) / 60.0
        if sample_minutes > 0:
            summary.preemption_rate_per_min = delta_preempts / sample_minutes

    # Persist
    run_csv_writer.writerow(
        {
            "concurrency": summary.concurrency,
            "repeat": summary.repeat,
            "target_label": summary.target_label,
            "completed": summary.completed,
            "errors": summary.errors,
            "ttft_p50_ms": f"{summary.ttft_p50:.1f}",
            "ttft_p95_ms": f"{summary.ttft_p95:.1f}",
            "ttft_p99_ms": f"{summary.ttft_p99:.1f}",
            "ttft_max_ms": f"{summary.ttft_max:.1f}",
            "itl_p50_ms": f"{summary.itl_p50:.2f}",
            "itl_p99_ms": f"{summary.itl_p99:.2f}",
            "divergence_ratio": f"{summary.divergence_ratio:.3f}",
            "throughput_tps": f"{summary.throughput_tps:.1f}",
            "kv_util_pct_mean": f"{summary.kv_util_pct_mean:.2f}",
            "kv_util_pct_max": f"{summary.kv_util_pct_max:.2f}",
            "queue_depth_mean": f"{summary.queue_depth_mean:.2f}",
            "queue_depth_max": summary.queue_depth_max,
            "queue_growth_frac": f"{summary.queue_growth_frac:.3f}",
            "preemption_rate_per_min": f"{summary.preemption_rate_per_min:.2f}",
        }
    )

    for r in successful + errors:
        request_csv_writer.writerow(
            {
                "concurrency": r.concurrency,
                "repeat": r.repeat,
                "request_id": r.request_id,
                "sent_at": f"{r.sent_at:.2f}",
                "ttft_ms": f"{r.ttft_ms:.1f}",
                "total_ms": f"{r.total_ms:.1f}",
                "output_tokens": r.output_tokens,
                "inter_token_ms": f"{r.inter_token_ms:.2f}",
                "error": r.error,
            }
        )

    print(
        f"\n  === c={summary.concurrency} r{summary.repeat} ({summary.target_label}) ===",
        flush=True,
    )
    print(
        f"  KV util mean={summary.kv_util_pct_mean:.1f}% max={summary.kv_util_pct_max:.1f}%",
        flush=True,
    )
    print(
        f"  TTFT  p50={summary.ttft_p50:.0f}ms p95={summary.ttft_p95:.0f}ms "
        f"p99={summary.ttft_p99:.0f}ms div={summary.divergence_ratio:.2f}",
        flush=True,
    )
    print(
        f"  Queue mean={summary.queue_depth_mean:.1f} max={summary.queue_depth_max} "
        f"growth_frac={summary.queue_growth_frac:.2f}",
        flush=True,
    )
    print(
        f"  Preempt={summary.preemption_rate_per_min:.1f}/min  "
        f"thr={summary.throughput_tps:.0f}tok/s  "
        f"completed={summary.completed} errors={summary.errors}",
        flush=True,
    )
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Day 24 cliff sweep")
    parser.add_argument(
        "--concurrencies",
        type=str,
        required=True,
        help="Comma-separated concurrency levels, e.g. 38,43,48,53,58,63,68,73,78,83",
    )
    parser.add_argument(
        "--zone-repeats",
        type=str,
        default="",
        help=(
            "Comma-separated concurrency levels in 60-75% transition zone "
            "to repeat. Each is run REPEAT_COUNT additional times."
        ),
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Extra repeats per zone level (default 2 → 3 total runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Where to write CSVs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="day24_cliff",
        help="Filename prefix for output CSVs",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_SECS,
                        help="Warmup seconds per level")
    parser.add_argument("--stabilization", type=int, default=DEFAULT_STABILIZATION_SECS,
                        help="Stabilization seconds per level (no sampling)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_SECS,
                        help="Sample window seconds per level")
    args = parser.parse_args()

    # Override module-level timing constants from CLI
    global WARMUP_SECS, STABILIZATION_SECS, SAMPLE_SECS
    WARMUP_SECS = args.warmup
    STABILIZATION_SECS = args.stabilization
    SAMPLE_SECS = args.sample

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    zone = set(int(x) for x in args.zone_repeats.split(",") if x.strip())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    requests_path = out_dir / f"{args.prefix}_requests.csv"

    summary_fields = [
        "concurrency",
        "repeat",
        "target_label",
        "completed",
        "errors",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "ttft_max_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "divergence_ratio",
        "throughput_tps",
        "kv_util_pct_mean",
        "kv_util_pct_max",
        "queue_depth_mean",
        "queue_depth_max",
        "queue_growth_frac",
        "preemption_rate_per_min",
    ]
    request_fields = [
        "concurrency",
        "repeat",
        "request_id",
        "sent_at",
        "ttft_ms",
        "total_ms",
        "output_tokens",
        "inter_token_ms",
        "error",
    ]

    print(
        f"Day 24 cliff sweep: concurrencies={concurrencies} "
        f"zone_repeats={sorted(zone)} repeat_count={args.repeat_count}",
        flush=True,
    )
    print(
        f"  Per point: {WARMUP_SECS}s warmup + {STABILIZATION_SECS}s stab + "
        f"{SAMPLE_SECS}s sample",
        flush=True,
    )
    base_total_min = (
        len(concurrencies) * (WARMUP_SECS + STABILIZATION_SECS + SAMPLE_SECS) / 60
    )
    repeat_total_min = (
        len(zone)
        * args.repeat_count
        * (WARMUP_SECS + STABILIZATION_SECS + SAMPLE_SECS)
        / 60
    )
    print(
        f"  Estimated wall time: {base_total_min:.0f} min base + "
        f"{repeat_total_min:.0f} min repeats = "
        f"{base_total_min + repeat_total_min:.0f} min",
        flush=True,
    )

    with open(summary_path, "w", newline="") as sf, open(
        requests_path, "w", newline=""
    ) as rf:
        s_writer = csv.DictWriter(sf, fieldnames=summary_fields)
        s_writer.writeheader()
        sf.flush()
        r_writer = csv.DictWriter(rf, fieldnames=request_fields)
        r_writer.writeheader()
        rf.flush()

        for c in concurrencies:
            label = f"c{c}"
            try:
                await run_level(
                    concurrency=c,
                    repeat=0,
                    target_label=label,
                    run_csv_writer=s_writer,
                    request_csv_writer=r_writer,
                )
            except Exception as e:
                print(f"  LEVEL FAILED c={c}: {e}", flush=True)
            sf.flush()
            rf.flush()

            if c in zone:
                for rep in range(1, args.repeat_count + 1):
                    try:
                        await run_level(
                            concurrency=c,
                            repeat=rep,
                            target_label=f"{label}_rep{rep}",
                            run_csv_writer=s_writer,
                            request_csv_writer=r_writer,
                        )
                    except Exception as e:
                        print(
                            f"  LEVEL FAILED c={c} rep={rep}: {e}", flush=True
                        )
                    sf.flush()
                    rf.flush()

    print(f"\nWrote: {summary_path}", flush=True)
    print(f"Wrote: {requests_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

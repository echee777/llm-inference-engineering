#!/usr/bin/env python3
"""
Day 11 Step 3+5: Full 24-cell benchmark matrix.

3 precisions × 2 workloads × 4 concurrency levels = 24 runs.
Uses vLLM's benchmark_serving.py against vLLM OpenAI-compatible server.

Usage:
    # 1. Start the vLLM server for a model (in a separate terminal):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-3B-Instruct \
        --max-num-seqs 32 --max-model-len 2048 --port 8000

    # 2. Run benchmarks for that model:
    python benchmark_matrix.py --precision fp16 --port 8000

    # Repeat for each precision variant (restart server with correct model).
    # After all 3 runs, combine results:
    python benchmark_matrix.py --summarize
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VLLM_BENCH_CMD = "vllm bench serve"  # replaces deprecated benchmark_serving.py
RESULTS_DIR = Path(__file__).parent / "benchmark_results"

# --- Model configs ---
MODELS = {
    "fp16": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "quantization": None,
        "server_args": [],
    },
    "int8-awq": {
        "model": "./day11/qwen2.5-3b-int8-awq",
        "quantization": "awq",
        "server_args": [],
    },
    "int4-gptq": {
        "model": "/home/ssm-user/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GPTQ-Int4/snapshots/68c4276063d1496cdf13d0b8e8221f899bfa77f7",
        "quantization": "gptq",
        "server_args": [],
    },
}

# --- Workload profiles ---
WORKLOADS = {
    "prefill-heavy": {"input_len": 1900, "output_len": 32},  # ~2048 with chat template overhead
    "decode-heavy":  {"input_len": 64,   "output_len": 512},
}

CONCURRENCY_LEVELS = [1, 4, 8, 16]
NUM_PROMPTS = 64  # per run; ≥8 batches worth at highest concurrency


def run_benchmark(
    precision: str,
    workload_name: str,
    concurrency: int,
    port: int,
    num_prompts: int = NUM_PROMPTS,
) -> Path:
    """Run one cell of the benchmark matrix."""
    model_cfg = MODELS[precision]
    workload = WORKLOADS[workload_name]

    result_filename = f"{precision}_{workload_name}_c{concurrency}.json"
    result_path = RESULTS_DIR / result_filename

    cmd = [
        *VLLM_BENCH_CMD.split(),
        "--backend", "openai-chat",
        "--base-url", f"http://localhost:{port}",
        "--endpoint", "/v1/chat/completions",
        "--model", model_cfg["model"],
        "--dataset-name", "random",
        "--random-input-len", str(workload["input_len"]),
        "--random-output-len", str(workload["output_len"]),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--percentile-metrics", "ttft,tpot,itl",
        "--metric-percentiles", "50,95,99",
        "--save-result",
        "--result-dir", str(RESULTS_DIR),
        "--result-filename", result_filename,
        "--seed", "42",
    ]

    print(f"\n{'='*70}")
    print(f"  {precision} | {workload_name} | concurrency={concurrency}")
    print(f"  input={workload['input_len']} output={workload['output_len']}")
    print(f"{'='*70}")
    print(f"  cmd: {' '.join(cmd[-16:])}")  # show tail of command

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  *** FAILED (exit {result.returncode}) after {elapsed:.1f}s ***")
    else:
        print(f"  completed in {elapsed:.1f}s → {result_path.name}")

    return result_path


def run_precision(precision: str, port: int, num_prompts: int):
    """Run all 8 cells for one precision variant."""
    RESULTS_DIR.mkdir(exist_ok=True)

    for workload_name in WORKLOADS:
        for concurrency in CONCURRENCY_LEVELS:
            run_benchmark(precision, workload_name, concurrency, port, num_prompts)


def summarize():
    """Parse all result JSONs and print the 24-cell table."""
    if not RESULTS_DIR.exists():
        print("No results directory found. Run benchmarks first.")
        return

    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  skipping {path.name} (invalid JSON)")
            continue

        # Parse filename: precision_workload_cN.json
        parts = path.stem.rsplit("_c", 1)
        if len(parts) != 2:
            continue
        concurrency = int(parts[1])
        # Split precision from workload
        prefix = parts[0]
        for wname in WORKLOADS:
            if prefix.endswith(wname):
                precision = prefix[: -(len(wname) + 1)]  # strip _workload
                workload = wname
                break
        else:
            continue

        throughput = data.get("output_throughput", data.get("request_throughput", 0))
        ttft_p99 = data.get("ttft_p99", 0)
        itl_p99 = data.get("itl_p99", 0)

        rows.append({
            "precision": precision,
            "workload": workload,
            "concurrency": concurrency,
            "throughput": throughput,
            "ttft_p99": ttft_p99,
            "itl_p99": itl_p99,
        })

    if not rows:
        print("No valid results found.")
        return

    # Print table
    print(f"\n{'Precision':<12} {'Workload':<16} {'Conc':>4} {'Throughput tok/s':>18} {'TTFT p99 ms':>13} {'ITL p99 ms':>12}")
    print("-" * 80)

    rows.sort(key=lambda r: (r["precision"], r["workload"], r["concurrency"]))
    for r in rows:
        ttft = f"{r['ttft_p99']*1000:.1f}" if r["ttft_p99"] else "—"
        itl = f"{r['itl_p99']*1000:.1f}" if r["itl_p99"] else "—"
        throughput = f"{r['throughput']:.1f}" if r["throughput"] else "—"
        print(f"{r['precision']:<12} {r['workload']:<16} {r['concurrency']:>4} {throughput:>18} {ttft:>13} {itl:>12}")

    # Roofline hypothesis check
    print(f"\n{'='*70}")
    print("ROOFLINE HYPOTHESIS CHECK")
    print("="*70)
    fp16_decode = [r for r in rows if r["precision"] == "fp16" and r["workload"] == "decode-heavy"]
    fp16_prefill = [r for r in rows if r["precision"] == "fp16" and r["workload"] == "prefill-heavy"]

    for quant in ["int8-awq", "int4-gptq"]:
        q_decode = [r for r in rows if r["precision"] == quant and r["workload"] == "decode-heavy"]
        q_prefill = [r for r in rows if r["precision"] == quant and r["workload"] == "prefill-heavy"]

        if not (fp16_decode and fp16_prefill and q_decode and q_prefill):
            print(f"  {quant}: insufficient data (need all cells)")
            continue

        # Average across concurrency levels
        fp16_d_avg = sum(r["throughput"] for r in fp16_decode) / len(fp16_decode)
        fp16_p_avg = sum(r["throughput"] for r in fp16_prefill) / len(fp16_prefill)
        q_d_avg = sum(r["throughput"] for r in q_decode) / len(q_decode)
        q_p_avg = sum(r["throughput"] for r in q_prefill) / len(q_prefill)

        decode_speedup = q_d_avg / fp16_d_avg if fp16_d_avg else 0
        prefill_speedup = q_p_avg / fp16_p_avg if fp16_p_avg else 0

        verdict = "CONFIRMED" if decode_speedup > prefill_speedup else "DISCONFIRMED"
        print(f"\n  {quant} vs fp16:")
        print(f"    decode  speedup: {decode_speedup:.2f}×")
        print(f"    prefill speedup: {prefill_speedup:.2f}×")
        print(f"    decode_speedup > prefill_speedup? → {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Day 11 benchmark matrix runner")
    parser.add_argument("--precision", choices=list(MODELS.keys()),
                        help="Which precision variant to benchmark")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM server port (default: 8000)")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS,
                        help=f"Prompts per run (default: {NUM_PROMPTS})")
    parser.add_argument("--summarize", action="store_true",
                        help="Parse results and print the 24-cell table")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    elif args.precision:
        run_precision(args.precision, args.port, args.num_prompts)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

import time, statistics, sys, json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
PROMPT = "Explain the history of the Roman Empire in detail. " + "history " * 20
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_REQUESTS = 20  # reduced from 40 per syllabus triage guidance

def run_benchmark():
    ttfts, itls, throughputs = [], [], []
    for i in range(N_REQUESTS):
        start = time.perf_counter()
        first_token_time = None
        tokens = 0
        for chunk in client.completions.create(
            model=MODEL,
            prompt=PROMPT,
            max_tokens=256,
            temperature=0,
            stream=True
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if chunk.choices[0].text:
                tokens += 1
        end = time.perf_counter()
        if first_token_time is None:
            continue
        ttfts.append((first_token_time - start) * 1000)
        total_time = end - start
        if tokens > 1:
            itls.append((total_time - (first_token_time - start)) / (tokens - 1) * 1000)
        throughputs.append(tokens / total_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{N_REQUESTS} requests", file=sys.stderr)

    results = {
        "ttft_p50": round(statistics.median(ttfts), 1),
        "ttft_p99": round(sorted(ttfts)[int(0.99 * len(ttfts))], 1),
        "itl_p50": round(statistics.median(itls), 1) if itls else 0,
        "throughput_mean": round(statistics.mean(throughputs), 2),
        "n_requests": N_REQUESTS,
    }
    return results

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print(f"Running benchmark: {label}", file=sys.stderr)

    # Warmup
    print("  Warmup...", file=sys.stderr)
    resp = client.completions.create(model=MODEL, prompt=PROMPT, max_tokens=32, temperature=0)

    results = run_benchmark()
    results["label"] = label

    # Try to get spec decode metrics
    try:
        import urllib.request
        raw = urllib.request.urlopen("http://localhost:8000/metrics").read().decode()
        accepted = 0
        drafted = 0
        for line in raw.split("\n"):
            if "spec_decode_num_accepted_tokens_total" in line and not line.startswith("#"):
                accepted = float(line.split()[-1])
            if "spec_decode_num_draft_tokens_total" in line and not line.startswith("#"):
                drafted = float(line.split()[-1])
        if drafted > 0:
            results["acceptance_rate"] = round(accepted / drafted * 100, 1)
        else:
            results["acceptance_rate"] = "N/A"
    except Exception:
        results["acceptance_rate"] = "N/A"

    print(json.dumps(results, indent=2))

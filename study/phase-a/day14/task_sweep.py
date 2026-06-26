import time, statistics, sys, json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_REQUESTS = 20

tasks = {
    "qa":       "What is the boiling point of water at sea level? Answer in detail:",
    "creative": "Write a short story about a dragon who discovers jazz music.",
    "code":     "Write a Python function implementing binary search on a sorted list. Include docstring and type hints."
}

def run_benchmark(prompt):
    ttfts, itls, throughputs = [], [], []
    for i in range(N_REQUESTS):
        start = time.perf_counter()
        first_token_time = None
        tokens = 0
        for chunk in client.completions.create(
            model=MODEL,
            prompt=prompt,
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

    return {
        "ttft_p50": round(statistics.median(ttfts), 1),
        "itl_p50": round(statistics.median(itls), 1) if itls else 0,
        "throughput_mean": round(statistics.mean(throughputs), 2),
    }

def get_acceptance_rate():
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
        return accepted, drafted
    except Exception:
        return 0, 0

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print(f"Running task sweep: {label}", file=sys.stderr)

    # Warmup
    resp = client.completions.create(model=MODEL, prompt="hello", max_tokens=16, temperature=0)

    for task_name, prompt in tasks.items():
        # Get metrics before
        acc_before, draft_before = get_acceptance_rate()

        print(f"  Running {task_name}...", file=sys.stderr)
        results = run_benchmark(prompt)

        # Get metrics after
        acc_after, draft_after = get_acceptance_rate()
        drafted_delta = draft_after - draft_before
        accepted_delta = acc_after - acc_before
        if drafted_delta > 0:
            task_acceptance = round(accepted_delta / drafted_delta * 100, 1)
        else:
            task_acceptance = "N/A"

        results["task"] = task_name
        results["acceptance_rate"] = task_acceptance
        results["label"] = label
        print(json.dumps(results))

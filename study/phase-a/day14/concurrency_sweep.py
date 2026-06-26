import time, statistics, sys, json, concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROMPT = "What is the boiling point of water at sea level? Answer in detail:"

def single_request():
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
        return None
    return {
        "ttft": (first_token_time - start) * 1000,
        "total_time": end - start,
        "tokens": tokens,
        "throughput": tokens / (end - start)
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
    print(f"Running concurrency sweep: {label}", file=sys.stderr)

    # Warmup
    client.completions.create(model=MODEL, prompt="hello", max_tokens=16, temperature=0)

    for concurrency in [1, 4, 8]:
        n_total = concurrency * 10
        acc_before, draft_before = get_acceptance_rate()

        print(f"  Concurrency={concurrency}, total requests={n_total}...", file=sys.stderr)

        all_results = []
        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(single_request) for _ in range(n_total)]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r:
                    all_results.append(r)
        wall_end = time.perf_counter()

        acc_after, draft_after = get_acceptance_rate()
        drafted_delta = draft_after - draft_before
        accepted_delta = acc_after - acc_before
        task_acceptance = round(accepted_delta / drafted_delta * 100, 1) if drafted_delta > 0 else "N/A"

        total_tokens = sum(r["tokens"] for r in all_results)
        wall_time = wall_end - wall_start
        # aggregate throughput = total tokens across all requests / wall clock time
        agg_throughput = round(total_tokens / wall_time, 2)
        # per-request throughput median
        per_req_throughput = round(statistics.median([r["throughput"] for r in all_results]), 2)

        output = {
            "concurrency": concurrency,
            "agg_throughput": agg_throughput,
            "per_req_throughput": per_req_throughput,
            "ttft_p50": round(statistics.median([r["ttft"] for r in all_results]), 1),
            "acceptance_rate": task_acceptance,
            "label": label,
            "n_requests": n_total
        }
        print(json.dumps(output))

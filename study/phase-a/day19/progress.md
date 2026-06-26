# Day 19 Morning Experiments — GPU Host Runbook

## Prerequisites

- vLLM running on port 8000 with Qwen/Qwen2.5-3B-Instruct (same setup as Day 18)
- Day 18 gateway.py and locustfile.py available in ../day18/
- Python venvs: ~/venv-vllm (FastAPI, httpx, transformers, prometheus_client) and ~/venv-locust (locust)

## Verify vLLM is running

```bash
curl http://localhost:8000/v1/models
```

If not running:
```bash
source ~/venv-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dtype half \
  --enforce-eager \
  --port 8000 &
```

---

## Experiment 1: Adversarial Request Simulation (1.5 hrs)

### Goal
Show that a single large request from one tenant can starve another tenant's small requests. Then show that arrival order changes the outcome even when total demand is identical.

### What to record
- Per-tenant TTFT
- Budget % at time of each request
- Rejection decisions (which requests got rejected/queued)
- For each of the 3 arrival-order cases: starvation outcome (rejections, queue wait)

### Setup

Use the Day 18 gateway on port 8001. For this experiment, we need a TIGHTER budget to force contention. The default 141K budget is too generous for 13K tokens of demand. Lower it to force the system into rejection territory.

Create `day19/exp1_gateway.py` — a modified copy of the Day 18 gateway with these changes:

```python
# OVERRIDE: tight budget to force adversarial contention
# Set to 15,000 tokens so that Tenant B's 12K request + Tenant A's 5x200 = ~13K
# forces admission decisions. At 15K budget, both CAN'T fit simultaneously.
KV_CAPACITY_TOKENS = 217_312
TARGET_UTILIZATION = 0.07  # ~15,000 tokens — forces contention
ADMISSION_BUDGET = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~15,211

# Disable rate limiting for this experiment
RATE_LIMIT_REQUESTS = 100_000
RATE_LIMIT_TOKENS = 100_000_000

# Shorter queue timeout so we see rejections faster
MAX_WAIT_SECONDS = 3.0
```

Everything else stays the same as the Day 18 gateway (Policy B, FIFO queue, Prometheus metrics).

Create `day19/exp1_runner.py` — the experiment driver:

```python
"""
Experiment 1: Adversarial Request Simulation

Sends requests from two tenants to the gateway and measures starvation.
- Tenant A: 5 short requests (prompt=100, max_tokens=100 each) = ~1000 tokens total
- Tenant B: 1 massive request (prompt=8192, max_tokens=4096) = ~12,288 tokens total

Runs 3 arrival-order cases:
  Case A: Large request first, then small requests
  Case B: Small requests first, then large request
  Case C: Interleaved (large, small, small, large-remainder, small, small, small)
          — since there's only 1 large, interleave means: small, LARGE, small, small, small, small

Records: per-request TTFT, admission decision (admitted/rejected/queued), budget % at admission time.
"""
import asyncio
import time
import json
import httpx

GATEWAY_URL = "http://localhost:8001/v1/chat/completions"
METRICS_URL = "http://localhost:8001/debug/stats"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def build_prompt(n_tokens: int) -> str:
    """Generate filler text approximately n_tokens long."""
    base = "The quick brown fox jumps over the lazy dog. "
    words = max(1, int(n_tokens * 0.75))
    return (base * (words // 9 + 1))[:words * 5]


def make_request(name: str, prompt_tokens: int, max_tokens: int, tenant: str) -> dict:
    """Build a request payload."""
    return {
        "name": name,
        "tenant": tenant,
        "payload": {
            "model": MODEL,
            "messages": [{"role": "user", "content": build_prompt(prompt_tokens)}],
            "max_tokens": max_tokens,
            "min_tokens": max(1, max_tokens - 20),  # force near-full completion
            "stream": True,
        },
    }


async def send_request(client: httpx.AsyncClient, req: dict) -> dict:
    """Send one request and measure TTFT + outcome."""
    name = req["name"]
    tenant = req["tenant"]
    start = time.monotonic()

    # Check budget before sending
    try:
        metrics_resp = await client.get(METRICS_URL)
        budget_before = metrics_resp.json().get("budget_utilization_pct", "?")
    except Exception:
        budget_before = "?"

    try:
        async with client.stream("POST", GATEWAY_URL, json=req["payload"], timeout=30) as resp:
            status = resp.status_code
            if status in (429, 503):
                # Rejected
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                elapsed = time.monotonic() - start
                return {
                    "name": name,
                    "tenant": tenant,
                    "status": status,
                    "outcome": "rejected",
                    "ttft_ms": None,
                    "elapsed_s": round(elapsed, 3),
                    "budget_before_pct": budget_before,
                }

            # Admitted — measure TTFT (time to first SSE content chunk)
            ttft = None
            async for chunk in resp.aiter_bytes():
                if ttft is None:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                            ttft = (time.monotonic() - start) * 1000
                            break
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "tenant": tenant,
                "status": status,
                "outcome": "admitted",
                "ttft_ms": round(ttft, 1) if ttft else None,
                "elapsed_s": round(elapsed, 3),
                "budget_before_pct": budget_before,
            }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "tenant": tenant,
            "status": None,
            "outcome": f"error: {e}",
            "ttft_ms": None,
            "elapsed_s": round(elapsed, 3),
            "budget_before_pct": budget_before,
        }


async def run_case(case_name: str, requests_in_order: list[dict]):
    """Run a single arrival-order case."""
    print(f"\n{'='*60}")
    print(f"CASE {case_name}")
    print(f"{'='*60}")
    print(f"Order: {[r['name'] for r in requests_in_order]}")

    # Wait for budget to clear from previous case
    await asyncio.sleep(5)

    async with httpx.AsyncClient(timeout=60) as client:
        # Check starting budget
        try:
            metrics = await client.get(METRICS_URL)
            print(f"Budget before: {metrics.json().get('budget_utilization_pct')}%")
        except Exception:
            pass

        # Fire all requests concurrently but with staggered start times
        # to preserve arrival order (100ms gap between launches)
        tasks = []
        for i, req in enumerate(requests_in_order):
            async def send_with_delay(r, delay):
                await asyncio.sleep(delay)
                return await send_request(client, r)
            tasks.append(asyncio.create_task(send_with_delay(req, i * 0.1)))

        results = await asyncio.gather(*tasks)

    # Print results
    print(f"\nResults for Case {case_name}:")
    print(f"{'Name':<20} {'Tenant':<10} {'Status':<8} {'Outcome':<12} {'TTFT(ms)':<12} {'Elapsed(s)':<12} {'Budget%':<10}")
    print("-" * 94)
    for r in results:
        print(f"{r['name']:<20} {r['tenant']:<10} {str(r['status']):<8} {r['outcome']:<12} "
              f"{str(r['ttft_ms']):<12} {str(r['elapsed_s']):<12} {str(r['budget_before_pct']):<10}")

    return results


async def main():
    # Define the requests
    tenant_a_reqs = [
        make_request(f"A_short_{i}", prompt_tokens=100, max_tokens=100, tenant="A")
        for i in range(5)
    ]
    tenant_b_req = make_request("B_large_0", prompt_tokens=8192, max_tokens=4096, tenant="B")

    all_results = {}

    # Case A: Large request first
    case_a_order = [tenant_b_req] + tenant_a_reqs
    all_results["A"] = await run_case("A (large first)", case_a_order)

    # Wait for all requests to complete and budget to drain
    print("\nWaiting 30s for budget to fully drain...")
    await asyncio.sleep(30)

    # Case B: Small requests first
    case_b_order = tenant_a_reqs + [tenant_b_req]
    all_results["B"] = await run_case("B (small first)", case_b_order)

    await asyncio.sleep(30)

    # Case C: Interleaved
    case_c_order = [
        tenant_a_reqs[0], tenant_b_req, tenant_a_reqs[1],
        tenant_a_reqs[2], tenant_a_reqs[3], tenant_a_reqs[4],
    ]
    all_results["C"] = await run_case("C (interleaved)", case_c_order)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*60}")
    for case_name, results in all_results.items():
        rejected_a = sum(1 for r in results if r["tenant"] == "A" and r["outcome"] == "rejected")
        rejected_b = sum(1 for r in results if r["tenant"] == "B" and r["outcome"] == "rejected")
        ttfts_a = [r["ttft_ms"] for r in results if r["tenant"] == "A" and r["ttft_ms"] is not None]
        ttfts_b = [r["ttft_ms"] for r in results if r["tenant"] == "B" and r["ttft_ms"] is not None]
        avg_ttft_a = sum(ttfts_a) / len(ttfts_a) if ttfts_a else None
        avg_ttft_b = sum(ttfts_b) / len(ttfts_b) if ttfts_b else None
        print(f"\nCase {case_name}:")
        print(f"  Tenant A: {rejected_a}/5 rejected, avg TTFT={avg_ttft_a}ms")
        print(f"  Tenant B: {rejected_b}/1 rejected, avg TTFT={avg_ttft_b}ms")

    # Dump raw results to JSON
    with open("exp1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRaw results written to exp1_results.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### How to run

```bash
cd ~/day19

# Terminal 1: Start the tight-budget gateway
source ~/venv-vllm/bin/activate
ADMISSION_ENABLED=true uvicorn exp1_gateway:app --host 0.0.0.0 --port 8001

# Terminal 2: Run the experiment
source ~/venv-vllm/bin/activate
python exp1_runner.py
```

### Expected results

```
Case A (large first):  Tenant B admitted, most of Tenant A rejected (budget consumed by B)
Case B (small first):  Tenant A all admitted, Tenant B rejected (budget consumed by A's 5 requests)
Case C (interleaved):  Mixed — first small admitted, then large admitted, remaining smalls rejected
```

### What to record in day19-work.md

Fill in this table:

```
| Case | Order       | Tenant A rejected | Tenant A avg TTFT | Tenant B rejected | Tenant B TTFT | Budget at peak |
|------|-------------|-------------------|-------------------|-------------------|---------------|----------------|
| A    | Large first |                   |                   |                   |               |                |
| B    | Small first |                   |                   |                   |               |                |
| C    | Interleaved |                   |                   |                   |               |                |
```

Write one sentence: "A single 12K-token request from Tenant B consumed X% of budget, causing Y of Tenant A's 5 requests to be rejected."

---

## Experiment 2: Head-of-Line (HOL) Blocking (1 hr)

### Goal
Show that FIFO queuing forces small requests to wait behind a large request at the queue head, even when there's budget available for them.

### What to record
- Median wait time for the 10 short requests
- Admission latency for the large request
- How long short requests waited despite fitting in budget

### Setup

Use the same `exp1_gateway.py` but with a slightly different budget. We want the large request to NOT be immediately admittable (so it sits at the queue head), but we want enough budget for the small requests IF they could skip ahead.

Modify the budget for this experiment:
```python
# Budget just barely too small for the 8K request, but plenty for 100-token requests
# 8K prompt + 4K max_tokens = 12,288 tokens. Set budget to 10,000 so large request queues.
# Each small request = 100 + 100 = 200 tokens. 10 of them = 2,000 tokens. Easily fits in 10K.
TARGET_UTILIZATION = 0.046  # ~10,000 tokens
```

Create `day19/exp2_runner.py`:

```python
"""
Experiment 2: Head-of-Line (HOL) Blocking

Setup:
1. Pre-fill budget to near capacity with a warm-up request
2. Submit a massive request (8K prompt) as FIRST in queue
3. Immediately behind it, submit 10 short requests (100 tokens each)

The large request can't be admitted (not enough budget). In strict FIFO,
the 10 short requests wait behind it even though they'd each fit easily.

This demonstrates the convoy effect: a slow process at the head of a FIFO
queue blocks all faster processes behind it.
"""
import asyncio
import time
import json
import httpx

GATEWAY_URL = "http://localhost:8001/v1/chat/completions"
METRICS_URL = "http://localhost:8001/debug/stats"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def build_prompt(n_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    words = max(1, int(n_tokens * 0.75))
    return (base * (words // 9 + 1))[:words * 5]


async def send_request(client: httpx.AsyncClient, name: str, prompt_tokens: int,
                       max_tokens: int) -> dict:
    """Send one request, measure time-to-admission and TTFT."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(prompt_tokens)}],
        "max_tokens": max_tokens,
        "min_tokens": max(1, max_tokens - 20),
        "stream": True,
    }
    start = time.monotonic()

    try:
        async with client.stream("POST", GATEWAY_URL, json=payload, timeout=30) as resp:
            status = resp.status_code
            if status in (429, 503):
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                elapsed = time.monotonic() - start
                error_msg = ""
                try:
                    error_msg = json.loads(body).get("error", "")
                except Exception:
                    pass
                return {
                    "name": name,
                    "status": status,
                    "outcome": f"rejected ({error_msg})",
                    "ttft_ms": None,
                    "total_s": round(elapsed, 3),
                    "submit_time": start,
                }

            ttft = None
            async for chunk in resp.aiter_bytes():
                if ttft is None:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                            ttft = (time.monotonic() - start) * 1000
                            break
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "status": status,
                "outcome": "admitted",
                "ttft_ms": round(ttft, 1) if ttft else None,
                "total_s": round(elapsed, 3),
                "submit_time": start,
            }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "status": None,
            "outcome": f"error: {e}",
            "ttft_ms": None,
            "total_s": round(elapsed, 3),
            "submit_time": start,
        }


async def main():
    print("HOL Blocking Experiment")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60) as client:
        # Check starting budget
        metrics = await client.get(METRICS_URL)
        print(f"Starting budget: {metrics.json()}")

        # Step 1: Send large request FIRST (should queue because budget is tight)
        # Step 2: Immediately send 10 short requests behind it
        # Use staggered timing: large at t=0, shorts starting at t=0.05s
        tasks = []

        # Large request at t=0
        async def send_large():
            return await send_request(client, "LARGE_8K", prompt_tokens=8192, max_tokens=4096)
        tasks.append(asyncio.create_task(send_large()))

        # 10 short requests starting 50ms later, spaced 20ms apart
        for i in range(10):
            async def send_short(idx=i):
                await asyncio.sleep(0.05 + idx * 0.02)
                return await send_request(client, f"SHORT_{idx}", prompt_tokens=100, max_tokens=100)
            tasks.append(asyncio.create_task(send_short()))

        results = await asyncio.gather(*tasks)

    # Print results
    print(f"\n{'Name':<15} {'Status':<8} {'Outcome':<25} {'TTFT(ms)':<12} {'Total(s)':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<15} {str(r['status']):<8} {r['outcome']:<25} "
              f"{str(r['ttft_ms']):<12} {str(r['total_s']):<10}")

    # Analysis
    large = [r for r in results if r["name"].startswith("LARGE")]
    shorts = [r for r in results if r["name"].startswith("SHORT")]
    admitted_shorts = [r for r in shorts if r["outcome"] == "admitted"]
    rejected_shorts = [r for r in shorts if r["outcome"] != "admitted"]

    print(f"\nSummary:")
    print(f"  Large request: {large[0]['outcome']}, TTFT={large[0]['ttft_ms']}ms")
    print(f"  Short requests admitted: {len(admitted_shorts)}/10")
    print(f"  Short requests rejected: {len(rejected_shorts)}/10")
    if admitted_shorts:
        ttfts = [r["ttft_ms"] for r in admitted_shorts if r["ttft_ms"]]
        if ttfts:
            print(f"  Short request TTFT: median={sorted(ttfts)[len(ttfts)//2]}ms, "
                  f"max={max(ttfts)}ms")
    if admitted_shorts:
        totals = [r["total_s"] for r in admitted_shorts]
        print(f"  Short request total time: median={sorted(totals)[len(totals)//2]}s, "
              f"max={max(totals)}s")

    # Dump raw results
    with open("exp2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nRaw results written to exp2_results.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### How to run

```bash
cd ~/day19

# Terminal 1: Start gateway with HOL-blocking budget
# IMPORTANT: kill the previous gateway first!
# Modify exp1_gateway.py TARGET_UTILIZATION to 0.046 (~10K tokens) and save as exp2_gateway.py
# OR create exp2_gateway.py as a copy with the budget change
source ~/venv-vllm/bin/activate
ADMISSION_ENABLED=true uvicorn exp2_gateway:app --host 0.0.0.0 --port 8001

# Terminal 2: Run experiment
source ~/venv-vllm/bin/activate
python exp2_runner.py
```

### Expected results

The large 8K request either:
- Gets rejected (budget too small) or queues and times out
- While it sits at the queue head, the 10 small requests (200 tokens each) queue behind it
- Even though 10 x 200 = 2,000 tokens easily fits in 10K budget, FIFO blocks them

Key number to extract: How long did short requests wait (total_s) compared to if they'd been admitted directly (~0.2s TTFT for 100-token requests)?

### What to record in day19-work.md

```
Large request: [admitted/rejected/timed out], TTFT=[X]ms
Short requests admitted: [N]/10
Short request median wait: [X]s vs expected ~0.2s without HOL blocking
Short request max wait: [X]s
```

Write: "10 short requests (100 tokens each) waited X seconds behind a single 8K request that couldn't be admitted. FIFO created [X]s of unnecessary wait for requests consuming <2% of budget."

---

## Experiment 3: Queue Depth to Latency Curves (1.5 hrs)

### Goal
Sweep traffic load from 25% to 90% of budget utilization. Find the hockey-stick inflection point where TTFT spikes. Identify where p99 first diverges from p50.

### What to record
At each load level: token budget utilization %, queue depth, TTFT p50/p95/p99, num_waiting_seqs from vLLM metrics.

### Setup

Use the Day 18 gateway (full 141K budget, Policy B enabled) — NOT the tight budgets from Experiments 1-2. We want realistic operating conditions.

Use the Day 18 locustfile.py traffic mix.

The approach: run 7 separate Locust runs at different user counts, calibrated to hit target budget utilization levels. Each run stabilizes for 3 minutes then records for 3 minutes.

The mapping from users to budget utilization depends on your specific system. Start with these estimates and adjust:

```
Target Budget %    Estimated Users    Notes
25%                10-15              very light load
40%                20-25              moderate
50%                30-35              medium
60%                40-45              approaching pressure
70%                50-60              near threshold
80%                65-75              past threshold
90%                80-100             heavy pressure
```

These are rough. After the first run, check actual budget utilization from gateway metrics and adjust user counts.

Create `day19/exp3_runner.sh`:

```bash
#!/bin/bash
# Experiment 3: Queue Depth -> Latency Curves
# Runs Locust at 7 load levels, 6 minutes each (3 min warmup + 3 min recording)
# Uses the Day 18 gateway with standard budget (141K tokens)
set -e
cd "$(dirname "$0")"

# Kill any existing gateway
pkill -f "uvicorn.*gateway.*8001" 2>/dev/null || true
sleep 2

# Start gateway with standard budget
echo "=== Starting Day 18 gateway (standard budget) ==="
source ~/venv-vllm/bin/activate
cp ../day18/gateway.py ./exp3_gateway.py
cp ../day18/locustfile.py ./exp3_locustfile.py
ADMISSION_ENABLED=true uvicorn exp3_gateway:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!
sleep 3

source ~/venv-locust/bin/activate

# Run at each load level
# ADJUST THESE USER COUNTS based on observed budget utilization after the first run
USER_COUNTS=(12 22 32 42 55 70 90)
LABELS=("25pct" "40pct" "50pct" "60pct" "70pct" "80pct" "90pct")

for i in "${!USER_COUNTS[@]}"; do
    USERS=${USER_COUNTS[$i]}
    LABEL=${LABELS[$i]}

    echo ""
    echo "============================================"
    echo "=== Load level: ${LABEL} (${USERS} users) ==="
    echo "============================================"

    # Run for 6 minutes total (first 3 min is warmup/stabilization)
    locust -f exp3_locustfile.py \
        --host http://localhost:8001 \
        --users $USERS \
        --spawn-rate 5 \
        --run-time 6m \
        --headless \
        --csv "exp3_${LABEL}" \
        --html "exp3_${LABEL}_report.html" \
        --print-stats

    # Capture gateway metrics at this load level
    echo "--- Gateway metrics at ${LABEL} ---"
    curl -s http://localhost:8001/debug/stats | python -m json.tool

    # Capture vLLM metrics (num_waiting_seqs)
    echo "--- vLLM metrics at ${LABEL} ---"
    curl -s http://localhost:8000/metrics | grep -E "num_waiting|gpu_cache_usage|time_to_first_token"

    # Wait 30s between runs for system to drain
    echo "Waiting 30s for system to drain..."
    sleep 30

    # Reset gateway stats between runs
    # (If gateway doesn't support reset, just note the delta)
done

echo ""
echo "=== All load levels complete ==="
kill $GATEWAY_PID 2>/dev/null || true
```

### How to run

```bash
cd ~/day19
chmod +x exp3_runner.sh
./exp3_runner.sh
```

This takes ~50 minutes (7 levels x 6 min + 30s gaps). Let it run unattended.

### Extracting results

After all runs complete, extract TTFT percentiles from the Locust CSV files:

```python
"""Extract TTFT p50/p95/p99 from Experiment 3 Locust CSV files."""
import csv
import glob

print(f"{'Label':<10} {'Users':<8} {'TTFT p50':<12} {'TTFT p95':<12} {'TTFT p99':<12}")
print("-" * 54)

for f in sorted(glob.glob("exp3_*_stats.csv")):
    label = f.replace("exp3_", "").replace("_stats.csv", "")
    with open(f) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "TTFT" in row.get("Name", "") and "Aggregated" in row.get("Name", ""):
                # Locust CSV columns: 50%, 95%, 99%
                # Column names may vary by Locust version
                p50 = row.get("50%", row.get("Median Response Time", "?"))
                p95 = row.get("95%", "?")
                p99 = row.get("99%", "?")
                print(f"{label:<10} {'?':<8} {p50:<12} {p95:<12} {p99:<12}")
```

For vLLM's num_waiting_seqs, you'll need to capture it from the /metrics endpoint. The exp3_runner.sh script dumps this at the end of each load level. Look for `vllm:num_requests_waiting` or `vllm:num_waiting_seqs` in the output.

### What to record in day19-work.md

Fill in this table:

```
| Budget % | Queue Depth | num_waiting_seqs | TTFT p50 | TTFT p95 | TTFT p99 |
|----------|-------------|------------------|----------|----------|----------|
| 25%      |             |                  |          |          |          |
| 40%      |             |                  |          |          |          |
| 50%      |             |                  |          |          |          |
| 60%      |             |                  |          |          |          |
| 70%      |             |                  |          |          |          |
| 80%      |             |                  |          |          |          |
| 90%      |             |                  |          |          |          |
```

Note: "Budget %" is the ACTUAL observed budget utilization from gateway metrics, not the target. Adjust the table labels to match reality.

Identify and write: "p99 first diverged from p50 at ~[X]% utilization. Operating point recommendation: [X - 8]% to maintain SLO with burst headroom."

---

## Troubleshooting

### Gateway won't start
- Check that port 8001 is free: `lsof -i :8001`
- Kill stale processes: `pkill -f "uvicorn.*8001"`

### Requests all succeed in Experiment 1 (no rejections)
- Budget is too high. Lower TARGET_UTILIZATION further.
- Check actual budget usage: `curl http://localhost:8001/debug/stats`

### Requests all rejected in Experiment 2
- Budget is too low. The large request needs to be AT the queue, not instantly rejected.
- Increase QUEUE_MAXSIZE and MAX_WAIT_SECONDS to give it time to queue.

### Experiment 3 user counts don't map to expected budget %
- Check actual utilization after the first run and adjust USER_COUNTS array.
- The mapping depends on request mix, completion speed, and Policy B release rate.

### vLLM not responding
- Check it's still running: `curl http://localhost:8000/v1/models`
- Check GPU memory: `nvidia-smi`
- If crashed, restart: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --dtype half --enforce-eager --port 8000`

---

## File checklist (create these on GPU host)

```
day19/
  exp1_gateway.py    — Day 18 gateway.py with TARGET_UTILIZATION=0.07
  exp1_runner.py     — Experiment 1 driver (adversarial simulation)
  exp2_gateway.py    — Day 18 gateway.py with TARGET_UTILIZATION=0.046
  exp2_runner.py     — Experiment 2 driver (HOL blocking)
  exp3_gateway.py    — Copy of Day 18 gateway.py (standard budget)
  exp3_locustfile.py — Copy of Day 18 locustfile.py
  exp3_runner.sh     — Experiment 3 sweep script
  exp3_extract.py    — TTFT percentile extractor (optional, run after exp3)
```

For exp1_gateway.py and exp2_gateway.py: copy ../day18/gateway.py and change only TARGET_UTILIZATION. Everything else (Policy B, FIFO queue, Prometheus, rate limits) stays the same.

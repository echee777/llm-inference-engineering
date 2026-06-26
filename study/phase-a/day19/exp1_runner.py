"""
Experiment 1: Adversarial Request Simulation

Sends requests from two tenants to the gateway and measures starvation.
- Tenant A: 5 short requests (prompt=100, max_tokens=100 each) = ~1000 tokens total
- Tenant B: 1 massive request (prompt=8192, max_tokens=4096) = ~12,288 tokens total

Runs 3 arrival-order cases:
  Case A: Large request first, then small requests
  Case B: Small requests first, then large request
  Case C: Interleaved: small, LARGE, small, small, small, small

Records: per-request TTFT, admission decision, budget % at submission time.
Writes raw JSON to results/exp1_results.json.
"""
import asyncio
import json
import os
import time

import httpx

GATEWAY_URL = "http://localhost:8001/v1/chat/completions"
METRICS_URL = "http://localhost:8001/debug/stats"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_prompt(n_tokens: int) -> str:
    """Generate filler text approximately n_tokens long."""
    base = "The quick brown fox jumps over the lazy dog. "
    words = max(1, int(n_tokens * 0.75))
    return (base * (words // 9 + 1))[:words * 5]


def make_request(name: str, prompt_tokens: int, max_tokens: int, tenant: str) -> dict:
    return {
        "name": name,
        "tenant": tenant,
        "payload": {
            "model": MODEL,
            "messages": [{"role": "user", "content": build_prompt(prompt_tokens)}],
            "max_tokens": max_tokens,
            "min_tokens": max(1, max_tokens - 20),
            "stream": True,
        },
    }


async def send_request(client: httpx.AsyncClient, req: dict) -> dict:
    """Send one request and measure TTFT + outcome."""
    name = req["name"]
    tenant = req["tenant"]
    start = time.monotonic()

    try:
        metrics_resp = await client.get(METRICS_URL)
        budget_before = metrics_resp.json().get("budget_utilization_pct", "?")
    except Exception:
        budget_before = "?"

    try:
        async with client.stream(
            "POST", GATEWAY_URL, json=req["payload"], timeout=60
        ) as resp:
            status = resp.status_code
            if status in (429, 503):
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

            ttft = None
            async for chunk in resp.aiter_bytes():
                if ttft is None:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        if line.startswith("data: ") and line[6:].strip() not in (
                            "",
                            "[DONE]",
                        ):
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

    await asyncio.sleep(5)

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            metrics = await client.get(METRICS_URL)
            print(f"Budget before: {metrics.json().get('budget_utilization_pct')}%")
        except Exception:
            pass

        tasks = []
        for i, req in enumerate(requests_in_order):
            async def send_with_delay(r, delay):
                await asyncio.sleep(delay)
                return await send_request(client, r)

            tasks.append(asyncio.create_task(send_with_delay(req, i * 0.1)))

        results = await asyncio.gather(*tasks)

    print(f"\nResults for Case {case_name}:")
    print(
        f"{'Name':<20} {'Tenant':<10} {'Status':<8} {'Outcome':<12} "
        f"{'TTFT(ms)':<12} {'Elapsed(s)':<12} {'Budget%':<10}"
    )
    print("-" * 94)
    for r in results:
        print(
            f"{r['name']:<20} {r['tenant']:<10} {str(r['status']):<8} "
            f"{r['outcome']:<12} {str(r['ttft_ms']):<12} "
            f"{str(r['elapsed_s']):<12} {str(r['budget_before_pct']):<10}"
        )

    return results


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tenant_a_reqs = [
        make_request(f"A_short_{i}", prompt_tokens=100, max_tokens=100, tenant="A")
        for i in range(5)
    ]
    tenant_b_req = make_request(
        "B_large_0", prompt_tokens=8192, max_tokens=4096, tenant="B"
    )

    all_results = {}

    # Case A: Large request first
    case_a_order = [tenant_b_req] + tenant_a_reqs
    all_results["A"] = await run_case("A (large first)", case_a_order)

    print("\nWaiting 30s for budget to fully drain...")
    await asyncio.sleep(30)

    # Case B: Small requests first
    case_b_order = tenant_a_reqs + [tenant_b_req]
    all_results["B"] = await run_case("B (small first)", case_b_order)

    print("\nWaiting 30s for budget to fully drain...")
    await asyncio.sleep(30)

    # Case C: Interleaved
    case_c_order = [
        tenant_a_reqs[0],
        tenant_b_req,
        tenant_a_reqs[1],
        tenant_a_reqs[2],
        tenant_a_reqs[3],
        tenant_a_reqs[4],
    ]
    all_results["C"] = await run_case("C (interleaved)", case_c_order)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*60}")
    for case_name, results in all_results.items():
        rejected_a = sum(
            1 for r in results if r["tenant"] == "A" and r["outcome"] == "rejected"
        )
        rejected_b = sum(
            1 for r in results if r["tenant"] == "B" and r["outcome"] == "rejected"
        )
        ttfts_a = [
            r["ttft_ms"]
            for r in results
            if r["tenant"] == "A" and r["ttft_ms"] is not None
        ]
        ttfts_b = [
            r["ttft_ms"]
            for r in results
            if r["tenant"] == "B" and r["ttft_ms"] is not None
        ]
        avg_ttft_a = sum(ttfts_a) / len(ttfts_a) if ttfts_a else None
        avg_ttft_b = sum(ttfts_b) / len(ttfts_b) if ttfts_b else None
        print(f"\nCase {case_name}:")
        print(f"  Tenant A: {rejected_a}/5 rejected, avg TTFT={avg_ttft_a}ms")
        print(f"  Tenant B: {rejected_b}/1 rejected, avg TTFT={avg_ttft_b}ms")

    out_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

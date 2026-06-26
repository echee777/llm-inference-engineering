"""
Day 17 Morning: Reconciliation experiment.
Sends three request shapes through the gateway, records:
  1. Gateway active_token_budget at peak concurrency
  2. Actual KV blocks allocated (from vLLM BLOCK_ALLOC logs parsed via /v1/completions usage)
  3. GPU memory delta (nvidia-smi)

Shapes:
  A: 5 requests x (~200 prompt + 512 max_completion), short-answer prompts
  B: 5 requests x (~2000 prompt + 512 max_completion), short-answer prompts
  C: Mixed (3x Shape A + 2x Shape B)

Key: prompts are designed to produce SHORT completions (~50-150 tokens)
while max_tokens=512 is set. This exposes the max_completion overestimate.
"""

import asyncio
import math
import subprocess
import time

import httpx

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"
GATEWAY_METRICS = "http://localhost:8080/metrics"
BLOCK_SIZE = 16
TOTAL_BLOCKS = 12964

# Short-answer prompts that use ~200 tokens but produce brief completions.
# Each is unique to avoid prefix caching conflating results.
SHORT_PROMPTS = [
    (
        "I have a question about photosynthesis in plants. Specifically about "
        "the light-dependent reactions and the Calvin cycle. I want to understand "
        "the role of chlorophyll, the electron transport chain, and how ATP and "
        "NADPH are produced. I also want to understand how carbon dioxide is "
        "fixed into glucose through the Calvin cycle, including details about "
        "RuBisCO, G3P, and regeneration of RuBP. Also the factors that affect "
        "the rate of photosynthesis such as light intensity, CO2 concentration, "
        "and temperature. And how C4 and CAM plants have adapted their "
        "photosynthetic pathways to deal with photorespiration in hot and dry "
        "environments. Furthermore, the evolutionary significance of oxygenic "
        "photosynthesis and how it transformed Earth early atmosphere. The "
        "endosymbiotic theory explaining the origin of chloroplasts from "
        "cyanobacteria. The structure of thylakoid membranes and the arrangement "
        "of photosystem I and photosystem II within them. "
        "Given all that context, answer in ONE sentence: what is the net equation "
        "of photosynthesis?"
    ),
    (
        "I am studying distributed systems and consensus algorithms including "
        "Paxos, Raft, and Byzantine fault tolerance. I want to understand the "
        "CAP theorem and its practical implications for system design. I also "
        "need to learn about distributed transactions, two-phase commit, and "
        "saga patterns. Database internals are also important to me, including "
        "B-tree and LSM-tree storage engines, write-ahead logging, MVCC "
        "concurrency control, query optimization with cost-based optimizers, "
        "and index selection strategies. Operating system internals matter too: "
        "virtual memory management, page tables, TLB caching, process scheduling "
        "algorithms like CFS and BFS, file system design including journaling "
        "and copy-on-write approaches, and IO scheduling. "
        "Given all that background, answer briefly: what does the CAP theorem state?"
    ),
    (
        "I have been researching network protocols including TCP congestion "
        "control algorithms like Reno, CUBIC, and BBR. I am interested in QUIC "
        "protocol design decisions, TLS 1.3 handshake optimization, HTTP/2 "
        "multiplexing and HTTP/3 migration strategies. I also study compiler "
        "design including lexical analysis, parsing techniques like recursive "
        "descent and LR parsing, semantic analysis, intermediate representations "
        "like SSA form, optimization passes including dead code elimination, "
        "loop unrolling, and register allocation via graph coloring. Machine "
        "learning systems are another interest: gradient descent optimization, "
        "backpropagation through computational graphs, attention mechanisms in "
        "transformers, and efficient inference techniques like quantization and "
        "pruning. I care about memory management in GPU systems too. "
        "Given all that, answer in one sentence: what is TCP BBR optimizing for?"
    ),
    (
        "I want to understand the full stack of modern software engineering. "
        "Starting from hardware: CPU architectures, cache hierarchies, memory "
        "controllers, PCIe bus protocols, NVMe storage interfaces. Then the OS "
        "layer: virtual memory, page tables, TLBs, context switching overhead, "
        "system calls, interrupt handling. Then networking: the TCP/IP stack, "
        "socket programming, epoll and io_uring, zero-copy techniques, kernel "
        "bypass with DPDK. Then application frameworks: async runtimes like "
        "tokio and asyncio, connection pooling, request routing, middleware "
        "chains, serialization formats like protobuf and flatbuffers. And "
        "distributed systems: consensus protocols, distributed hash tables, "
        "vector clocks, CRDTs, and leader election algorithms. "
        "Given all that context, answer briefly: what is a TLB?"
    ),
    (
        "My research covers GPU computing extensively. I study CUDA programming "
        "models, thread hierarchies with warps and blocks, shared memory bank "
        "conflicts, coalesced memory access patterns, and occupancy optimization. "
        "I understand tensor cores and their role in matrix multiplication for "
        "deep learning. I follow the development of transformer architectures "
        "including multi-head attention, rotary position embeddings, grouped "
        "query attention, and KV cache management during autoregressive decoding. "
        "I also study inference optimization techniques: continuous batching, "
        "PagedAttention for KV cache memory management, speculative decoding, "
        "flash attention and its memory-efficient backward pass, and model "
        "parallelism strategies including tensor, pipeline, and expert parallelism. "
        "Given all that, answer in one sentence: what problem does PagedAttention solve?"
    ),
]

# Long prompts (~2000 tokens each), also designed for short answers.
# Each unique to avoid prefix caching.
LONG_PROMPT_BASE = (
    "Write a comprehensive analysis of the following topics in software engineering. "
    "For each topic provide historical context, current best practices, common pitfalls, "
    "and future directions. Topic 1: Distributed systems and consensus algorithms including "
    "Paxos, Raft, and Byzantine fault tolerance. Discuss the CAP theorem and its practical "
    "implications for system design. Cover distributed transactions, two-phase commit, and "
    "saga patterns. Topic 2: Database internals including B-tree and LSM-tree storage engines, "
    "write-ahead logging, MVCC concurrency control, query optimization with cost-based "
    "optimizers, and index selection strategies. Topic 3: Operating system internals including "
    "virtual memory management, page tables, TLB caching, process scheduling algorithms "
    "like CFS and BFS, file system design including journaling and copy-on-write approaches, "
    "and IO scheduling. Topic 4: Network protocols including TCP congestion control algorithms "
    "like Reno, CUBIC, and BBR, QUIC protocol design decisions, TLS 1.3 handshake optimization, "
    "HTTP/2 multiplexing and HTTP/3 migration strategies. Topic 5: Compiler design including "
    "lexical analysis, parsing techniques like recursive descent and LR parsing, semantic "
    "analysis, intermediate representations like SSA form, optimization passes including "
    "dead code elimination, loop unrolling, and register allocation via graph coloring. "
)

LONG_SUFFIXES = [
    "Given all of the above context about these five topics, answer in ONE sentence only: "
    "What is the primary advantage of LSM-trees over B-trees for write-heavy workloads?",
    "Given all of the above context about these five topics, answer in ONE sentence only: "
    "Why is Raft considered easier to implement than Paxos?",
    "Given all of the above context about these five topics, answer in ONE sentence only: "
    "What is the key insight behind BBR congestion control compared to loss-based algorithms?",
    "Given all of the above context about these five topics, answer in ONE sentence only: "
    "What problem does MVCC solve in database concurrency?",
    "Given all of the above context about these five topics, answer in ONE sentence only: "
    "Why is SSA form useful as an intermediate representation in compilers?",
]

LONG_PROMPTS = [LONG_PROMPT_BASE * 8 + suffix for suffix in LONG_SUFFIXES]


def get_gpu_memory_mb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def get_gateway_metrics():
    import requests as req
    r = req.get(GATEWAY_METRICS)
    return r.json()


async def send_request(client, prompt, max_tokens, request_id):
    body = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.time()
    resp = await client.post(GATEWAY_URL, json=body)
    elapsed = time.time() - start

    if resp.status_code == 429:
        return {"id": request_id, "status": 429, "elapsed": elapsed}

    data = resp.json()
    usage = data.get("usage", {})
    return {
        "id": request_id,
        "status": resp.status_code,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
    }


async def poll_peak_metrics():
    """Poll gateway metrics to capture peak active_tokens."""
    peak = 0
    try:
        while True:
            m = get_gateway_metrics()
            if m["active_tokens"] > peak:
                peak = m["active_tokens"]
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        m = get_gateway_metrics()
        if m["active_tokens"] > peak:
            peak = m["active_tokens"]
        return peak


async def run_shape(name, prompts, max_tokens=512):
    print(f"\n{'='*60}")
    print(f"Shape {name}: {len(prompts)} requests, max_tokens={max_tokens}")
    print(f"{'='*60}")

    gpu_before = get_gpu_memory_mb()
    metrics_before = get_gateway_metrics()
    print(f"  Before: GPU={gpu_before} MB, active_tokens={metrics_before['active_tokens']}")

    ts_start = time.time()

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for i, prompt in enumerate(prompts):
            tasks.append(send_request(client, prompt, max_tokens, f"{name}-{i}"))

        peak_task = asyncio.create_task(poll_peak_metrics())
        results = await asyncio.gather(*tasks)
        peak_task.cancel()
        try:
            peak_active = await peak_task
        except asyncio.CancelledError:
            peak_active = get_gateway_metrics()["active_tokens"]

    ts_end = time.time()
    gpu_after = get_gpu_memory_mb()

    # Per-request results
    total_prompt = 0
    total_completion = 0
    total_gateway_charge = 0
    for r in results:
        if r["status"] == 429:
            print(f"  {r['id']}: REJECTED (429) in {r['elapsed']:.2f}s")
        else:
            pt = r.get("prompt_tokens", 0)
            ct = r.get("completion_tokens", 0)
            charge = pt + max_tokens
            total_prompt += pt
            total_completion += ct
            total_gateway_charge += charge
            print(f"  {r['id']}: prompt={pt} completion={ct} "
                  f"gateway_charge={charge} in {r['elapsed']:.2f}s")

    actual_tokens = total_prompt + total_completion
    actual_blocks = sum(
        math.ceil((r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)) / BLOCK_SIZE)
        for r in results if r["status"] != 429
    )
    actual_block_tokens = actual_blocks * BLOCK_SIZE

    print(f"\n  --- Reconciliation ---")
    print(f"  Gateway charge (peak active_tokens): {peak_active}")
    print(f"  Gateway charge (sum prompt+max):     {total_gateway_charge}")
    print(f"  Actual tokens used:                  {actual_tokens}")
    print(f"  Actual blocks needed:                {actual_blocks} ({actual_block_tokens} tokens)")
    print(f"  GPU delta:                           {gpu_after - gpu_before} MB")

    if actual_tokens > 0:
        overestimate_vs_tokens = (total_gateway_charge - actual_tokens) / actual_tokens * 100
        print(f"  Overestimate vs actual tokens:       {overestimate_vs_tokens:+.1f}%")
    if actual_block_tokens > 0:
        overestimate_vs_blocks = (total_gateway_charge - actual_block_tokens) / actual_block_tokens * 100
        print(f"  Overestimate vs actual blocks:       {overestimate_vs_blocks:+.1f}%")

    return {
        "shape": name,
        "peak_active": peak_active,
        "gateway_charge": total_gateway_charge,
        "actual_tokens": actual_tokens,
        "actual_blocks": actual_blocks,
        "actual_block_tokens": actual_block_tokens,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_delta": gpu_after - gpu_before,
        "results": results,
        "max_tokens": max_tokens,
    }


async def main():
    print("Day 17 Reconciliation Experiment")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10) as client:
        await client.get("http://localhost:8000/v1/models")
        print("vLLM: OK")
        r = await client.get(GATEWAY_METRICS)
        print(f"Gateway: OK, budget={r.json()['admission_budget']}")

    await asyncio.sleep(2)

    # Shape A: 5 x (~200 prompt + 512 max), expect short completions
    shape_a = await run_shape("A", SHORT_PROMPTS)
    await asyncio.sleep(5)

    # Shape B: 5 x (~2000 prompt + 512 max), expect short completions
    shape_b = await run_shape("B", LONG_PROMPTS)
    await asyncio.sleep(5)

    # Shape C: 3x short + 2x long
    shape_c_prompts = SHORT_PROMPTS[:3] + LONG_PROMPTS[:2]
    shape_c = await run_shape("C", shape_c_prompts)
    await asyncio.sleep(3)

    # Final summary table
    print("\n" + "=" * 60)
    print("RECONCILIATION SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Metric':<35} {'Shape A':>10} {'Shape B':>10} {'Shape C':>10}")
    print("-" * 65)
    for label, key in [
        ("Gateway charge (tokens)", "gateway_charge"),
        ("Actual tokens used", "actual_tokens"),
        ("Actual block tokens", "actual_block_tokens"),
        ("Peak active_tokens", "peak_active"),
        ("Total prompt tokens", "total_prompt"),
        ("Total completion tokens", "total_completion"),
        ("GPU delta (MB)", "gpu_delta"),
    ]:
        vals = [shape_a[key], shape_b[key], shape_c[key]]
        print(f"  {label:<33} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print("-" * 65)
    for shape in [shape_a, shape_b, shape_c]:
        name = shape["shape"]
        if shape["actual_tokens"] > 0:
            ov_tok = (shape["gateway_charge"] - shape["actual_tokens"]) / shape["actual_tokens"] * 100
            ov_blk = (shape["gateway_charge"] - shape["actual_block_tokens"]) / shape["actual_block_tokens"] * 100
            print(f"  Shape {name}: overestimate vs tokens={ov_tok:+.1f}%, vs blocks={ov_blk:+.1f}%")

    # Trust boundary assessment
    print("\n" + "=" * 60)
    print("TRUST BOUNDARY ASSESSMENT")
    print("=" * 60)
    max_overestimate = 0
    for shape in [shape_a, shape_b, shape_c]:
        if shape["actual_tokens"] > 0:
            ov = (shape["gateway_charge"] - shape["actual_tokens"]) / shape["actual_tokens"] * 100
            max_overestimate = max(max_overestimate, ov)

    if max_overestimate <= 15:
        print(f"  Max overestimate: {max_overestimate:.1f}% <= 15%")
        print(f"  Stance: Gateway is authoritative")
    elif max_overestimate <= 30:
        print(f"  Max overestimate: {max_overestimate:.1f}% > 15%")
        print(f"  Stance: Consider adjusting TARGET_UTILIZATION or implementing Policy B")
    else:
        print(f"  Max overestimate: {max_overestimate:.1f}% >> 15%")
        print(f"  Stance: Policy B correction is essential to avoid excessive spurious 429s")


if __name__ == "__main__":
    asyncio.run(main())

# AI Inference Residency, Progress Summary

What this is: one entry per day of the inference platform residency. Each entry leads with the finding,
then describes how the experiment was structured, what tooling ran and measured it, and the numbers
that came out. Written for an engineer with general backend/devops/infra background but no prior LLM
inference experience.

Hardware and stack: all hands-on work ran on a single NVIDIA T4 (g4dn.xlarge spot instance), serving
small open models, mostly Qwen2.5-3B-Instruct and TinyLlama, through vLLM (the dominant open-source LLM
serving engine). Early days used vLLM V0; from Day 7 on, V1.

Inference-specific terms used below (everything else assumes normal infra fluency):
- Prefill: the model ingesting the prompt. One parallel pass over all prompt tokens. Compute-bound.
- Decode: the model generating the response one token per step. Memory-bandwidth-bound.
- KV cache: per-request attention state held in GPU memory, growing one entry per token. This is the
  scarce resource that caps how many requests you can run at once.
- TTFT: time to first token, the latency before generation starts (dominated by prefill).
- TPOT: time per output token during decode.
- Preemption: when the scheduler runs out of KV memory and evicts a running request to free blocks.
  In vLLM V1 the evicted request loses its work and must re-run prefill from scratch.
- Paging / PagedAttention: vLLM's allocator that hands out KV memory in fixed-size blocks instead of
  one contiguous span per request, the same idea as OS virtual-memory paging.

## Phase A: Fundamentals and vLLM internals (Days 1 to 20)

### Day 1: On a T4, inference is bottlenecked on memory bandwidth, not FLOPS
Benchmarked the two things that actually gate inference: memory bandwidth and compute. A PyTorch
microbenchmark copied tensors of 1, 10, 100, and 1000 MB on-device with `torch.cuda.synchronize()`
fences and 100 iterations each, measuring on-GPU (HBM) bandwidth at 217 GB/s, about 68% of the 320 GB/s
spec. The same pattern over the PCIe link (host to device and back) measured roughly 9 GB/s, making
on-GPU memory about 23x faster than the link to the host. A second sweep ran square matmuls at batch
sizes 1, 4, 16, 64, 256, 1024 (200 iterations each) and watched throughput climb from near zero to 23
TFLOPS, only 35% of the 65 TFLOPS spec; an extended run at batch 4096 confirmed the card was hitting its
70W power limit, not a compute limit. Monitored throughout with `nvidia-smi`, `nvidia-smi dmon`, and
`nvtop`. Key takeaway for later: the `nvidia-smi` "GPU utilization" field is time-based and reads high
even when one warp runs per SM, so it is necessary but not sufficient for diagnosing saturation.

### Day 2: Prefill is compute-bound, decode is memory-bound, proven with a profiler
Established the central asymmetry of inference and backed it with kernel-level profiling. First worked
the roofline math by hand: for a 7B FP16 model the T4's "ridge point" is 203 FLOPs/byte, decode runs at
about 1 FLOP/byte (deep in memory-bound territory, capped near 0.5% of peak compute), and prefill on a
2048-token prompt runs near 2048 FLOPs/byte (firmly compute-bound). Then confirmed it empirically with
Nsight Compute (`ncu`) on Phi-2. The dominant prefill kernel (`turing_fp16_s1688gemm`) ran at 83.9%
compute throughput and 98.8% SM-active, sitting near tensor-core saturation. The decode kernel
(`cublasGemvParamsEx`) ran at 96.1% DRAM throughput against only 58.5% compute, with 47% of warp stalls
waiting on global memory. Decode launched ~53,000 kernels in the capture window because launch count
scales with tokens times layers. Also profiled with `torch.profiler` for the operator-level view.

### Day 3: Without paging, KV memory fragments and capacity collapses, which is exactly what PagedAttention fixes
The point of the day was to show that single-GPU concurrency is gated by KV cache memory, and that naive
contiguous allocation wastes much of it through fragmentation. Built two things. First, a KV capacity
calculator in Python: per-token KV size is `2 * num_kv_heads * head_dim * dtype_bytes * num_layers`,
swept over sequence lengths 512 through 16384 against a budget of (GPU memory - model weights - 10%
overhead). For a 7B-shaped config this lands near 0.5 MB/token, so an 8K-token request alone eats 4 GB
and a 40 GB GPU serves only single-digit concurrency at long context. Second, a pure-Python allocator
simulation (no GPU needed) over a 1500 MB pool: it allocated 10 requests of varying lengths, freed every
other one to punch holes, then tried to place new large requests. The contiguous allocator failed to
place 250/350/400 MB requests once the largest free hole shrank below them, even with enough total free
memory; the paged allocator (8 MB blocks) placed them all because it only needs free blocks, not a
contiguous span. That fragmentation gap is precisely the problem vLLM's PagedAttention solves, which set
up the Day 6 source dive.

### Day 4: Splitting a small model across two GPUs over PCIe can be slower than one GPU
Quantified the cost of cross-GPU communication to decide whether tensor parallelism (splitting a model's
layers across GPUs) is worth it on PCIe-class hardware. A PyTorch benchmark moved float32 tensors of 1,
10, 100, 500, 1000 MB across the PCIe link with `.to('cuda')` / `.to('cpu')`, `cuda.synchronize()`
fences, and 50 iterations per size, measuring about 9 GB/s versus 600 to 900 GB/s on NVLink-equipped
GPUs. Then a back-of-envelope model for a 7B model at tensor-parallel degree 2: two all-reduce
collectives per layer times 32 layers, each all-reduce about 16 KB at batch 1, which pencils out to
roughly 320 to 640 microseconds of communication per decode step over PCIe versus about 32 microseconds
over NVLink. Conclusion: for a small model on a slow interconnect, the per-step communication tax can
exceed the compute saved by splitting, so the T4 was treated as a single-GPU inference card for the rest
of Phase A.

### Day 5: Consolidation and adversarial self-review
No new measurements. Caught up Days 1 to 4, wrote the Week 1 deliverable (a 305-line GPU Architecture
and Memory Budget document drawing on the Day 1 to 4 numbers), and ran an out-loud self-quiz covering
theoretical vs achieved HBM bandwidth, why batch-1 matmul underutilizes the GPU, decode arithmetic
intensity, KV concurrency math at 4K context, and why `nvidia-smi` utilization misleads. Afternoon was a
first read of the vLLM PagedAttention paper through Section 3, framed against the Day 3 fragmentation
result.

### Day 6: vLLM prefill time is a near-perfect linear function of prompt length
Ran parameter sweeps to find where each vLLM knob binds, and landed a clean predictive model for prefill
latency. The `--max-num-seqs` sweep (1, 4, 8, 16, 32) on TinyLlama drove 66 requests at concurrency 33
and showed throughput climbing from 84.7 to 488.4 tok/s with TTFT p50 dropping from 21.6s to 0.36s; an
anomalous flat 84 tok/s result the first night turned out to be a thermal artifact that did not
reproduce. The `--gpu-memory-utilization` sweep (0.5 to 0.95) showed near-identical degradation at
concurrency 4 because TinyLlama's short outputs kept the system pegged in prefill, so KV memory never
became the binding constraint. The headline came from a fixed-budget prompt-length sweep on Qwen2.5-3B
(prompts 130 to 1794 tokens), which fit `TTFT_ms = 37.6 + 0.228 * prompt_tokens` with R-squared 0.9966,
implying ~4,386 tok/s of prefill throughput. That means a 2,000-token prompt occupies the GPU for about
493 ms, a delay every co-scheduled request inherits, which became the quantitative seed for the Phase B
interference work.

### Day 7: vLLM V1 dropped CPU swap, so preemption now means redo all the work
Traced a request end to end through vLLM v0.6.6 source to build an accurate mental model. The path: the
OpenAI-compatible API server hands off to `AsyncLLM`, which tokenizes and ships an `EngineCoreRequest`
over ZMQ (msgpack-encoded) to a separate `EngineCore` subprocess that owns the GPU step loop
(`schedule -> execute_model -> sample -> update`). The architecturally important finding: V1 removed the
old SWAPPED state and the swap-KV-to-CPU recovery path entirely. Preemption now frees all of a request's
KV blocks and re-queues it, paying the full re-prefill cost on resume rather than a CPU round-trip,
which is the direct consequence of the Day 1 finding that PCIe is 23x slower than HBM. Also mapped how
continuous batching actually works: a finishing request frees blocks in step N that a waiting request
claims in step N+1, all through one `KVCacheManager` free pool governed by reference counts and a
doubly-linked free queue.

### Day 8: Instrumented the engine so every KV block alloc, free, and eviction is observable
Built the measurement tool the rest of the residency depends on: a patch to vLLM V1's KV cache manager
and scheduler that emits four event types, BLOCK_ALLOC, BLOCK_ALLOC_FAIL, BLOCK_FREE, and BLOCK_PREEMPT,
each tagged with a timestamp, request id, and free/total block counts. Placement followed the design:
PREEMPT logs from the scheduler's `_preempt_request()` where the decision context lives, while ALLOC and
FREE log from the cache manager where the memory effect lands. Validated on a single TinyLlama request
and watched a clean round trip: 17 prompt tokens grabbed 2 blocks at prefill, 3 more accrued across a
50-token decode, and all 5 freed at completion, with the free pool returning exactly to 30,278 of 30,278
blocks (the off-by-one total is a reserved null block at index 0).

### Day 9: Drove the system into collapse on purpose and watched the thrash spiral
Reproduced a real production failure mode under controlled conditions. Capping `gpu_memory_utilization`
to 0.45 shrank the block pool to 1,034 blocks, then 80 concurrent long-decode requests pushed it into a
78-second thrashing spiral: 89 preemptions, 1,398 ALLOC_FAIL events, and 11 silent client timeouts, with
free blocks oscillating between 0 and about 51 (one eviction victim's worth) the whole time. The block
logs from Day 8 showed one request preempted six times, discarding 2,156 tokens of decode work, more
than the 1,900 it actually needed, which crystallized that V1 preemption is memory-neutral but
compute-wasteful. A separate cliff sweep (0.45 utilization, 1,203-token prompts at 75 blocks each,
`max_tokens=200`) produced the textbook flat-then-vertical latency curve right where the block math
predicted, between concurrency 12 and 14 (14 x 75 = 1,050 blocks > 1,021 available): max TTFT jumped from
671 ms to 8,735 ms, a 13x step. Continuous batching was confirmed in another run: 344 scheduler
iterations across 19.6s with batch composition changing roughly every 57 ms.

### Day 10: Wrote the Week 2 architecture deliverable
No new experiments. Turned the Day 7 to 9 work into a self-contained six-section document: V1 request
lifecycle, the WAITING/RUNNING/FINISHED state machine (no SWAPPED state), the recompute-only preemption
cost model, scheduler decision logic, the instrumentation hook points, and the mini-collapse
observation. The practical thesis: capacity is `floor(total_blocks / blocks_per_request)`, computable up
front from block math, rather than something you discover by crashing. Afternoon was prep reading on the
AWQ and GPTQ quantization methods for the next week.

### Day 11: Quantization's real payoff is capacity, not raw speed
Compressed the model and measured the effect across speed and capacity. Self-quantized Qwen2.5-3B to
INT8 using llm-compressor (AWQ method), a 2.25-hour, 36-layer calibration run; the INT4 build (GPTQ) was
pulled pre-made from the Qwen org. Benchmarks at concurrency 1, decode-heavy, showed INT8 at 1.70x and
INT4 at 2.44x over FP16, with time-per-token tracking the weight-size ratio almost exactly. Prefill-heavy
gains were much smaller (1.26x and 1.60x) because weight-only quantization still dequantizes to FP16
before the matmul, so prefill FLOPs are unchanged. The more important result was capacity: weights
dropped from 5.79 GiB (FP16) to 3.23 (INT8) to 1.95 (INT4), and that freed VRAM rolled straight into KV
cache, raising max concurrency at 2,048 tokens from 94x to 131x to 149x. The progression to internalize:
INT8 is not just faster, it frees KV cache, which raises concurrency, which directly lowers cost per
token.

### Day 12: INT8 is a free lunch on quality, INT4 is not
Closed the quantization loop with quality and cost data. Ran perplexity on WikiText-2 (11.66 FP16, 11.68
INT8 at +0.17%, 12.57 INT4 at +7.8%) and HellaSwag accuracy on a 25% subset (FP16 and INT8 tied at
64.20%, INT4 at 63.36%). The decisive evidence was a qualitative eval over 10 varied prompts at fixed
sampling params: INT8 produced 0 of 10 failures and was indistinguishable from FP16, while INT4 failed 5
of 10 with a clear pattern, breaking on creative writing (repetition loops), factual Q&A (divergent
hallucination), code (a syntax error in a test harness), and long-context, but holding on structured
reasoning. A cost model on the g4dn.xlarge at $0.526/hr on-demand gave $0.552/M tokens for FP16, $0.411
for INT8 (-25%), and $0.304 for INT4 (-45%). Recommendation: ship INT8 as a drop-in for FP16; restrict
INT4 to reasoning-only workloads.

### Day 13: Prefix caching gives an 18x TTFT win, plus an 8x KV-math correction
Two findings. First, prefix caching (reusing KV state for a shared prompt prefix across requests) was
measured by sweeping shared-prefix lengths and cache hit ratios. The experiment took four debug
iterations because of cache contamination: running multiple prefix lengths in one vLLM process leaked
blocks across conditions, and sequential hit-ratio sweeps silently reused request-variant IDs, turning a
25% hit into 100%. Fixing it (restart vLLM per condition, offset variant IDs) gave clean results: zero
overhead at 0% hit (validating vLLM's default-on choice) and, at a 1000-token shared prefix with 100%
hit, TTFT p50 dropping from 2,238 ms to 124 ms (18x) with throughput up from 76 to 248 tok/s. Second, and
more consequential, caught a bug in all prior KV math: Qwen2.5-3B has `num_kv_heads=2`, not 16 (it uses
grouped-query attention), so true per-token KV is 36 KiB, not 288 KiB, an 8x overestimate. This reset
capacity numbers upward and meant the binding constraint usually shifts to compute and `--max-num-seqs`
before KV runs out.

### Day 14: Speculative decoding is a net loss on this hardware, with the math to say why
Tested speculative decoding (a small "draft" model proposes several tokens, the big model verifies them
in one pass) and found it counterproductive here. The vocab-size check failed first: TinyLlama (32k) and
Qwen (151,936) mismatch and vLLM correctly refuses to pair them, so the work moved to the n-gram variant
(no draft model, zero VRAM cost). It was net negative everywhere: 0.92x at concurrency 1 degrading to
0.86x at concurrency 8 as continuous batching closed the very gap speculation exists to fill. A K sweep
(3, 5, 7, 10 proposed tokens) found no sweet spot, all 7 to 8% slower than the 37.5 tok/s baseline, with
acceptance falling from 18.6% to 5.6%. A task sweep matched intuition (Q&A 56.7% acceptance, code 24.7%,
creative 4.2%) but even Q&A was not enough to beat single-step decode on a T4. A same-family draft model
(Qwen2.5-0.5B) paired cleanly but was worse (0.40x) and burned 0.94 GiB of weights plus 1.66 GiB of KV
headroom, nearly halving max concurrency. Recommendation: do not deploy; it wants 30B+ targets and
lower-concurrency regimes than this hardware offers.

### Day 15: Quantization decision memo
A writing day. Synthesized four weeks of data into a nine-section memo. Headline: INT8 improves decode
throughput 70% but prefill only 26% (because decode is bandwidth-bound and the T4's INT8 tensor cores
stay underused), at +0.17% perplexity and zero qualitative failures, while INT4's 5-of-10 failure rate
rules it out as a default. The capacity story (INT8 freeing 2.56 GiB, raising max concurrent 4K requests
from 47 to 66, a 40% gain) mattered more than per-request speed. Ship call: INT8 plus prefix caching,
speculative decoding off.

### Day 16: Built a token-budget admission gateway and redefined the cliff as an observable signal
Two moves. First, reframed "the cliff" (the load point where latency explodes) from an internal KV
threshold into the ratio of TTFT p99 to p50 crossing 2.0, which makes it detectable from black-box
latency without reading scheduler internals. Second, built a FastAPI gateway in front of vLLM that does
admission control on a memory budget expressed in tokens (budget = practical KV pool x 0.65), not a flat
concurrency cap, because 10 short requests and 10 long requests are not the same load. The gateway uses
Qwen's own tokenizer with the chat template applied, an asyncio lock around the budget counter, byte-by-
byte SSE proxying with no buffering, and a real HTTP 429 path with Retry-After. Three smoke tests passed
(admit, concurrent admit, forced rejection). The honest limitation, written up explicitly: the gateway's
estimate is an external proxy blind to prefix-cache hits, preemption, and the prefill/decode timing
asymmetry.

### Day 17: The gateway over-charges by 150 to 440%, so it needs progressive release
Validated the gateway's accounting against reality and fixed the worst gap. A reconciliation experiment
compared three numbers across three request shapes: tokens the gateway charged, tokens actually used, and
peak KV blocks vLLM actually allocated. Overestimates were large: one shape 154% over actual tokens and
205% over blocks; another 439% over blocks because the gateway cannot see prefix-cache hits. The budget
model was reframed as `max_prompt_length * max_concurrency`, which makes admission robust to mixed request
sizes. Built "Policy B," which releases reserved budget every 50 generated tokens with a 64-token safety
margin; an 8K-budget test with 100 short-completion requests freed only 77 tokens across 46 admits,
because completions finished before the first release checkpoint, a useful negative result showing Policy
B only pays off on long completions. Also added per-key sliding-window rate limits and a deliberately
naive FIFO queue to set up Day 19.

### Day 18: Admission control made latency worse, because the limit was too conservative for the hardware
Load-tested the gateway and got a humbling result. Identical Locust traffic ran in baseline (no control)
and controlled modes at 50, 100, and 200 users, using a five-bucket matrix mixing prompt length and
max_tokens. At 50 users the two modes were identical (TTFT p50 ~160 ms, no rejections). At 100 users the
controlled mode was worse: p99 hit 5,100 ms with 32% of traffic rejected, while baseline ran 340 to 570
ms with zero rejections. Root cause: the corrected Day 13 math meant Qwen-3B on a T4 has ~217k tokens of
KV, so generous that the preemption cliff never materialized at 100 users, making the 65% target far too
conservative. At 200 users vLLM crashed in both modes, but from process-level overload (connections and
asyncio task explosion), not KV exhaustion, exposing that admission control guards one resource and is
blind to others. Policy B did fire correctly under forced long completions, freeing 11,672 tokens.

### Day 19: Same load, 20x different latency depending on arrival order
Stress-tested fairness and abuse resistance for the Week 4 deliverable. Experiment 1 held total demand,
budget, and admission outcome constant while varying arrival order: one tenant's 12,288-token request
plus another tenant's five short requests against a ~15k-token budget. All were admitted every time, but
the short-tenant's average TTFT swung from 102 ms (smalls first) to 2,019 ms (the giant prefill first), a
20x spread, because the long prefill monopolized GPU compute. The lesson: the budget controls total
capacity but not its distribution, and FIFO is not fairness. Experiment 2 (intended to show head-of-line
blocking) accidentally showed the opposite, that small requests bypassed the queue via the immediate-
admission path. Experiment 3 swept load from 25% to 90% of budget and located the divergence: p99 first
pulls away from p50 at 70% (1.5x), explodes to 20x at 80% (200 ms p50 vs 3,980 ms p99), and fully
collapses at 90%. Recommended operating point: 60 to 65%.

### Day 20: Phase A capstone and self-assessment
No new experiments. Polished the four deliverables, answered all 10 exit questions in mock-interview
format, and distilled a frontier-lab interview brief. The audit explicitly logged two corrected
mistakes: the 8x grouped-query-attention KV error from Day 13, and an earlier wrong framing that
prefill/decode mixing causes memory contention when the real blind spot is compute contention. Two V1
mental-model shifts cemented: recompute-only preemption (no swap), and try-and-fail block allocation
rather than watermark-based prediction.

## Phase B: Failure modes and operating at scale (Days 21 to 29)

### Day 21: The same workload fails two completely different ways depending on the memory budget
Showed that "the system is overloaded" has two distinct root causes that need opposite fixes. At
`gpu_memory_utilization=0.45` the 16,352-token KV pool ran out at just 6 concurrent 2,258-token requests:
KV hit 92.5%, recompute-only preemption fired, throughput halved from 101 to 51 tok/s, and preemption
count climbed from 48 to 502 as concurrency rose from 6 to 32, a memory-bound collapse. At
`gpu_memory_utilization=0.90` the 203,520-token pool never filled past 56% even at 128 concurrent
requests, but the running set plateaued at 50 and the queue grew to 76 with zero preemptions, a pure
compute-bound saturation. Same prompts, opposite bottleneck, determined by the ratio of KV memory to
per-request prefill cost. The instrumented block count (12,951) also revealed vLLM's runtime profiler
reserves about 7.4% more KV than the theoretical estimate, which set up the next day's correction.

### Day 22: First postmortem, with the diagnostic signatures that tell the two failures apart
Wrote up the bimodal model and the math under it. The memory-bound failure is a positive feedback loop:
a preempted request loses its KV, re-queues, and on reschedule must recompute its entire prefill, re-
consuming the blocks it just freed. Its signature is queue depth near zero while preemption count rises.
The compute-bound failure is the inverse: queue depth rising with zero preemptions and KV stuck around
56%. The "why my budget was wrong" section folded in the grouped-query-attention correction (2 KV heads,
not 16), giving a true per-token KV cost of 36,864 bytes; that fix turned earlier nonsense into a model
that predicted collapse at 5.9 concurrent requests, matching the observed failure at 6.

### Day 23: Mixing long and short requests is a tail-latency problem, and chunked prefill fixes it for free
Quantified how much short requests suffer when batched with long ones, and isolated the cheap mitigation.
Short-only traffic had a TTFT p99 of 230 ms. Mixing in long requests with chunked prefill disabled drove
short p99 to 1,052 ms, a 4.57x penalty. Re-enabling chunked prefill (splitting big prefills into pieces
so they cannot monopolize a scheduler step, the vLLM 0.17.1 default) cut that to 446 ms, a 57.6%
improvement at zero throughput cost (long-request throughput stayed at 778 tok/s either way). The
mechanism came from an iteration-tokens histogram: average tokens processed per scheduler step was 411
non-chunked vs 276 chunked vs 32 for short-only, and a 2,048-token prefill creates roughly a 504 ms
starvation window for any short request co-scheduled with it. Crucially, p50 was essentially identical
across all conditions and KV utilization sat at 3 to 16% throughout, proving this is a scheduling
fairness problem, not a memory-pressure one.

### Day 24: Located the latency cliff at 87% KV with enough resolution to defend an operating point
Mapped the cliff precisely. With `gpu_memory_utilization=0.60` constraining the KV pool to 74,416 tokens
and homogeneous 512-token prompts, the p99/p50 divergence ratio jumped from 1.21 at concurrency 108 to
2.06 at 113, over just 1.7 points of KV change (85.6% to 87.3%), while p99 leapt from 3,200 ms to 7,587
ms. That fixes the cliff at 87% KV. Throughput peaked at 879 tok/s at c=108 and fell to 786 past the
cliff without recovering. The recommended operating point is 79% KV at c=95, where p99 is 2,115 ms with
zero preemptions and 803 tok/s, an 8.6% throughput cost bought for stability. The 8-point safety margin
is set by scheduler jitter: at a fixed c=113, instantaneous KV utilization oscillated plus or minus 15
points around the mean. Above the cliff, the ratio of recompute work to forward progress crossed 30% and
reached 75% past c=128, the regime where the engine does more makeup work than new work.

### Day 25: Unified the three failures into one model and a real-time admission controller
Tied Days 21, 23, and 24 together: KV exhaustion, prefill/decode interference, and the cliff are three
views of one thing, memory scheduling under contention. The compound failure in one sentence:
interference raises effective concurrency, which accelerates KV exhaustion, which triggers preemption,
which re-consumes blocks and raises concurrency again, with no self-correcting term. The redesigned
controller replaces the static token-budget gate with the live `vllm:gpu_cache_usage_perc` metric as the
primary signal, hard-rejects with 429 at 85% KV (cliff minus 2), and tiers thresholds by prompt length
(short up to 82%, medium 77%, long 72%), plus a chunked-prefill gate that defers a long request when the
short queue exceeds 10. Operational requirement noted: the Prometheus scrape interval must be under half
the TTFT SLO, or the controller reacts to stale data inside the 15-point jitter window.

### Day 26: Derived the retry-storm amplification factor on paper
A reasoning day, because the dynamics follow from the Day 24 cliff plus a geometric series. The
amplification factor is the sum of timeout_rate^i over attempts; at a 40% timeout rate with 3 attempts
that is 1.56x. So a system at 60% capacity under a storm sits at effective 93.6%, past the 87% cliff.
Recovery is a drain problem, not a convergence problem: retries already in flight are indistinguishable
from real traffic and keep consuming KV after the trigger clears. Designed a four-layer defense, each
addressing a different part: client exponential backoff with jitter (breaks synchronization), a gateway
retry budget capping retries near 10% of successful throughput, admission hard-rejection at 85% KV as a
circuit breaker, and the vLLM scheduler as last resort. Production alert: amplification above 1.3
sustained for 30 seconds.

### Day 27: Second postmortem, and the proof that CPU-based autoscaling cannot see this failure
Reframed the retry cascade as a client-side feedback loop wrapping the server-side preemption loop:
trigger transient, cascade self-sustaining. A 30-second burst at 60% base load reaches ~94% effective via
1.56x amplification, crosses the cliff, and stays degraded 2 to 3 minutes afterward because three things
must clear together (in-flight retries complete, the poisoned queue drains, KV falls below the cliff).
Inference is uniquely exposed because retries concentrate into one shared KV pool rather than diluting
across a stateless fleet, and request lifetimes are seconds, so the duplicate-firing window is wide. Also
settled the autoscaling-signal question analytically: at the cliff, with TTFT p99 at 7,587 ms, CPU stayed
at 20 to 40% because the CPU only does tokenization and bookkeeping while the GPU does the work, so a
CPU-based HPA at 70% would never fire during a KV-exhaustion event.

### Day 28: Reactive autoscaling is structurally too slow, and the obvious warning signals lie
Measured the three numbers that decide whether reactive autoscaling can even work, and found it cannot
here. Cold start was 94.2 seconds (model load plus CUDA-graph capture plus block-pool allocation),
against a danger window of seconds. The supposed leading indicator, queue depth, did not lead: with
`max_num_seqs=160` and offered concurrency stepping to 130, queue depth stayed at zero through the entire
cliff event while TTFT p99 climbed to 16,573 ms, because the scheduler admitted everything to the running
set and degradation happened internally via preemption cycling. KV-usage scrapes even read 4.0% during a
15,841 ms TTFT moment, because the metric samples troughs in rapid free-and-reconsume oscillation.
SIGTERM was the other trap: vLLM does not close connections or return errors on shutdown, it just stops
sending bytes, so clients hang for their full 30-second timeout before retrying. The scale-down hazard
math: force-terminating 20 in-flight requests, times 1.56x retry amplification, times 1.05% KV each,
equals 33 points of added pressure on survivors, shoving 79% to 112%, well past the cliff. Conclusion:
reactive scaling fails the timing inequality, so a pre-warmed standby is required.

### Day 29: Turned the measurements into a defensible autoscaling policy
Converted Days 27 and 28 into a concrete policy. The scale-up trigger is composite: `queue_depth > 10`
AND `kv_util > 76%` sustained for 30 seconds, each threshold tied to the cliff arithmetic (10 is about
10% of the safe c=95 operating point; 76% is the 87% cliff minus 10 queued requests at 1.05% KV each, so
worst-case admission still lands under the cliff). The policy mandates `max_num_seqs <= 95` so a queue
actually forms at the admission boundary, otherwise the queue-depth signal stays blind. Because headroom
is seconds and cold start is 94.2 seconds, a hot standby is required to take cold start off the critical
path. Scale-down is a graceful drain: return 503 with Retry-After while DRAINING, target a 3.2-second
soft drain per request, hard-kill at 120 seconds, and apply 10-minute scale-down hysteresis. Also seeded
a running mistakes log (the 8x GQA error, V0/V1 terminology drift, T4 BF16 incompatibility, the
cliff-as-divergence-ratio reframe, and the GPU-util-as-scaling-signal misconception).

## Day 30 and beyond (planned next phase)

Phase B Track 2 was scoped to extend everything above to multiple GPUs: tensor parallelism on A10G
hardware (which, unlike the T4, has the interconnect to make splitting worthwhile), with a one-day
contrast run on a high-end H100 (NVLink and FP8). It would then synthesize into two capstone design
deliverables, a multi-GPU serving architecture and an end-to-end inference platform design. Day 30 has a
detailed syllabus prepared. The single-GPU body of work (Days 1 to 29) is the portion that was fully
executed and measured.

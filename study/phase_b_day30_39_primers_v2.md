# Phase B Remaining Schedule with Conversational Primers

Days 30 to 39 plus an opportunistic parallel track for KubeRay. v2 is voiced for a 9-hour driving conversation with ChatGPT: mechanisms, intuitions, tradeoffs, and failure-mode reasoning anchored to my actual measured numbers from Days 1 to 29. Tools surface only as one-liners; commands and flags live in v1 and the v4 syllabus.

## Schedule

```
Day 30  Multi-GPU environment up
Day 31  How fast two GPUs talk
Day 32  Talking vs working under load
Day 33  Alternative split + breaking a GPU
Day 34  Premium-hardware contrast
Day 35  Multi-GPU serving doc (synthesis)
Day 36  Where does multi-GPU break?
Day 37  Retry storm under multi-GPU
Day 38  Master platform design
Day 39  Mistakes, audit, exit
.................................................
Day 40+ KubeRay autoscaling re-run (parallel/opportunistic)
```

## Day 30, get on the multi-GPU box and prove it works

The link between two GPUs determines what every later measurement means. NVLink, PCIe across the same root complex, and PCIe across separate roots span an order of magnitude in bandwidth, so the same TP=2 deployment will look fast or painful depending on which one I got. The wires are not interchangeable from the model's perspective. The model performs a per-layer all-reduce, and the per-layer cost is bounded by the slowest path in the topology graph.

The interesting failure mode is silent. A TP=2 deployment can return tokens with one GPU doing all the work. NCCL can fall back to a slow algorithm with a warning that nobody reads. The cards can sit on different PCIe roots and route GPU-to-GPU traffic through host RAM. None of these surface as errors. They surface as 5x to 10x slower than expected, and standard observability calls the deployment healthy.

The reason TP=1 baseline matters more than it sounds is that A10G is a different SKU than the T4 that anchored Phase A. T4 prefill on Qwen2.5-3B fit `TTFT = 37.6 + 0.228 × prompt_tokens` with R² = 0.9966, putting a 530-token prompt at 158 ms TTFT and per-token slope at 0.228 ms. The A10G slope will be different. Without the new per-instance baseline locked, every Day 31+ comparison is a measurement without a control. The TP=1 baseline becomes the denominator in the comm-fraction analysis, the Amdahl crossover derivation, and the hardware-contrast story. Skip it and the rest of the week spends compute producing numbers that cannot be interpreted.

The wider question lurking under Day 30 is what "correctly configured" means for multi-GPU. The wrong answer is "it returns tokens." The right answer names the topology, confirms both GPUs carry non-trivial memory and compute, places the comm fraction at concurrency 1 in the expected band for the model and link, and shows zero NCCL warnings on first-request load.

Tools: `nvidia-smi topo -m` for topology; `nccl-tests` for collective sanity; `nvidia-smi dmon` for whether both GPUs do work during a request.

Discussion seeds:
- If a TP=2 deployment returns tokens with `nvidia-smi` showing only GPU 0 active, what failure modes match that signature and how would comm fraction at concurrency 1 disambiguate them?
- Why does cross-root PCIe drop NCCL bandwidth 5-10x versus same-root, and what does this mean for choosing 2 of the 4 GPUs on a g5.12xlarge?
- The Phase A T4 baseline is `158 ms` TTFT at 530 tokens. Why is that not a usable anchor for the A10G TP=2 comparison, and what specifically would I be lying to myself about if I tried to use it?

## Day 31, how fast two GPUs talk

NCCL all-reduce bandwidth lives in two regimes. Small messages: launch overhead and ring synchronization dominate, the wire idles, and the bandwidth number is poor. Large messages: the wire saturates and the number reflects physical link capacity. The transition is not gradual. Below the transition the curve is roughly flat-and-low; above it the curve is roughly flat-and-high; the elbow itself is sharp because the cost structure changes from "synchronization-dominated" to "transfer-dominated."

The shape matters because TP=2 inference does not get to choose where it sits on the curve. The per-layer all-reduce in a transformer moves something on the order of `2 × seq_len × hidden_dim × bytes` per call. For Qwen2.5-3B at hidden_dim 2048, BF16, single-request prefill on a short prompt, the message size is small and the call sits firmly in the latency-bound regime. At high concurrency with longer prompts the message size grows and the call slides into the bandwidth-bound regime. Two different operating points, two different cost structures, one model.

The TP=1 baseline anchor from Day 30 turns into a sanity check here. For a 530-token prompt the T4 TTFT was 158 ms (the 0.228 ms/token slope tells me the prefill itself was about 121 ms; the 37.6 ms intercept covers scheduler overhead and one decode token). On A10G TP=2 at concurrency 1, the per-layer comm tax shows up as a small TTFT increase, typically 5-15% over the matched-hardware TP=1 number on PCIe-class links. If the TP=2 single-request TTFT is 1.5x the TP=1 number, the wire is cross-root, NCCL has fallen back to a slow algorithm, or the configuration is wrong. The shape of the failure tells me which.

The mechanically interesting question this day pre-empts: why does splitting work across more GPUs make a single request slower? Each GPU does roughly half the matmul, but neither can finish a layer alone. They have to combine partials with an all-reduce, and the all-reduce sits on the critical path of every layer. At concurrency 1 the matmul is small and the all-reduce overhead is large relative to it, so the comm tax is visible. At higher concurrency the matmul grows linearly in batch but the per-layer message size only grows linearly in the activation slice, and amortization eventually wins.

Tools: `nccl-tests all_reduce_perf` for the bandwidth-vs-message-size curve; vLLM serving for the deployment baseline.

Discussion seeds:
- TP=2 single-request TTFT is *higher* than TP=1 even though the model is split across more compute. Why? At what concurrency does TP=2 catch up, and what does the answer depend on?
- The NCCL curve has a transition message size where bandwidth saturates. What does that transition tell me about the link, and how would I predict it for PCIe Gen4 versus NVLink 4.0 from spec alone?
- For a transformer with hidden_dim H, N layers, and seq_len S in TP=2, what is the per-request communication volume, and how does it scale with batch?

## Day 32, talking vs working under load

Tensor parallelism levies a per-layer tax: every layer requires an all-reduce to sum partial activations across GPUs before the next layer can start. The communication fraction of an iteration is `NCCL_time / total_iteration_time`. At concurrency 1 the matmul is small and comm is a meaningful fraction of the iteration. As concurrency rises the matmul grows with batch (more tokens going through the same GEMM at once) while the all-reduce message size grows only with the activation slice, so comm fraction drops. This is the amortization story, and it has a specific shape.

The headline interview question this day trains me to answer: "at concurrency N, what is the iteration bounded by, and what dominates the scaling?" Bad answer: "comm fraction goes up as concurrency goes up." Good answer: "at concurrency 1, iteration time is bounded by per-layer compute on a small batch, with comm tax visible at roughly X%; at concurrency 16, iteration time is bounded by KV cache pressure or memory bandwidth on the larger batch, and comm has been amortized below Y%; at concurrency 24, the system is preemption-cycling and the iteration time tells me nothing about steady-state comm." The "limiting factor" framing forces me to name the bottleneck rather than describe the trend.

Phase A gave me the anchor for the compute side. Day 6 measured Qwen2.5-3B prefill on T4 at 0.228 ms per prompt token, with a 37.6 ms fixed overhead (scheduler plus one decode token), R² = 0.9966 over the 130-to-1794 token range. That is the compute-time number that goes into the comm-versus-compute ratio. On A10G the slope will differ but the structure of the analysis is identical: prefill compute scales linearly in prompt tokens, and comm scales with hidden-dim activations per layer. Comm fraction is the ratio of the two.

What the profiler shows beyond comm fraction is which compute units are busy. A GPU can be at 98% utilization on `nvidia-smi` while the tensor cores sit idle because the kernel is bandwidth-bound on HBM rather than compute-bound on tensor cores. DCGM exposes the metric `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` that pulls this apart. This is the same flavor of distinction Day 28 made about GPU utilization in the autoscaling context: the headline number can stay flat while the system silently changes regimes underneath. Tensor-core activity tells me whether the matmul is actually using the matmul hardware. Bandwidth activity tells me whether HBM is the binding constraint. Neither is visible in the aggregate utilization number.

The NCCL subsystem trace adds a third dimension: which collective algorithm did NCCL pick, and over which transport. Ring on NVLink versus tree on PCIe versus direct P2P versus host-staged each behave differently and pick different message-size transitions. If my comm fraction at small message looks 3x the spec band, the trace tells me whether the wire was wrong or the algorithm was wrong.

Tools: torch.profiler for comm fraction breakdown; DCGM for tensor-core activity; NCCL subsystem trace for algorithm and transport selection.

Discussion seeds:
- Why does comm fraction *fall* with concurrency under TP=2 even though more concurrent requests means more all-reduces per second? What is amortizing what?
- GPU utilization reads 98% during the Day 24 cliff and during healthy operation. Tensor-core activity might tell a different story. What would it look like under healthy decode versus during cliff cycling, and why?
- If the NCCL subsystem trace shows tree algorithm at message size 256 KB, but ring would have been faster, what does that tell me about the topology assumptions NCCL made?

## Day 33, alternative split + breaking a GPU

There are two ways to split a model across GPUs. Tensor parallelism splits the work *within* each layer: every GPU computes a slice of every layer, then all-reduces partials, then proceeds. Pipeline parallelism splits *between* layers: GPU 0 owns the first half of the layers, GPU 1 owns the second half, microbatches flow through the pipeline. The communication patterns are completely different. TP requires bandwidth-hungry all-reduce on every layer. PP requires only point-to-point send/receive at stage boundaries, which is friendly to PCIe and even tolerable across nodes.

The reason production TP shops do not just switch to PP on slow interconnect is that PP introduces idle time. The first microbatch has to traverse all stages before the last stage produces output, and during that fill phase later stages sit idle. The drain phase at the end is symmetric. The bigger the pipeline depth and the smaller the microbatch count, the more bubble time as a fraction of the run. PP wins when the link is too slow for TP to amortize and the workload is large enough that bubbles do not dominate. TP wins when the link is fast and the workload is latency-sensitive.

The straggler experiment makes the convoy mechanism mechanical and inarguable. Lock one GPU's clock to a fraction of normal (90%, 80%, 70%, 60%) and run the same load. System-wide TTFT degrades roughly linearly with the slowest GPU's clock, not the average. This is a direct consequence of synchronous all-reduce: every GPU has to finish its slice before the all-reduce can complete, and the all-reduce sits on the critical path of every layer. The system runs at the speed of the slowest GPU, every layer, every iteration, every request.

The production implication compounds the failure modes already known. Day 24 showed the system is non-linearly fragile near the KV cliff at 87% utilization on Qwen2.5-3B. Add a straggler and the cliff shifts left, because every request now holds KV blocks longer (the convoy slows the iteration), so KV utilization at any fixed concurrency rises, and the cliff fires earlier. A fleet that has 1% of GPUs running at 70% clock has not lost 1% of capacity; it has potentially halved the safe operating point of every TP group containing one of those GPUs. This is why production fleets enforce homogeneity within TP groups, monitor per-GPU clocks, and treat a thermally throttling GPU as a circuit-breaker event rather than a performance issue.

Tools: vLLM with TP and PP for the comparison; `nvidia-smi -lgc` for forced clock throttle; per-GPU clock telemetry for the production analog.

Discussion seeds:
- Under what hardware regime would PP beat TP, and why? What would tell me, before measuring, that I am about to be in that regime?
- The convoy effect comes from synchronous all-reduce. Suppose I had asynchronous TP. What would break, and why is no production system doing this?
- A straggler at 70% clock on one GPU in a TP=4 group degrades system-wide TTFT by what factor, and how does that interact with the Day 24 cliff at 87% KV?

## Day 34, premium hardware contrast

Every Phase A and Phase B Track 1 measurement sat on cost-conscious hardware: T4 single-GPU, then A10G TP=2 with PCIe Gen4 between cards, GDDR6 memory, no FP8 support. The cheap interconnect makes the multi-GPU tax painful, which is the most instructive way to learn it: comm fraction is high enough to force me to think about whether TP is worth it. On Hopper-class hardware (H100 with NVLink 4.0 and HBM3) the comm tax shrinks to a footnote and the bottleneck moves elsewhere. Both stories are true; both stories are different.

Two specific contrasts surface only with the hardware swap. The first is NVLink-fast TP. PCIe Gen4 between A10G cards offers around 32 GB/s; NVLink 4.0 between H100s offers around 450 GB/s. That is a 14x bandwidth ratio. At small messages the difference is muted (both regimes are latency-bound, and launch overhead does not scale with link speed), but above the bandwidth-bound transition the difference is the full ratio. Comm fraction at concurrency 16 under TP=2 on H100 NVLink should be a small fraction of the A10G PCIe number for the same workload. The shape of the curve looks similar; the magnitude is dramatically different.

The second contrast is FP8. Hopper added FP8 tensor cores that double the throughput of BF16 for the same kernel shape, modulo the calibration question. Ampere (A10G) cannot run FP8 at all; the kernel falls back to BF16 or fails to compile. The expected throughput gain in production for Llama 3.1 8B-class models is roughly 1.5-2x at higher concurrency where the compute side is the binding constraint. Below that concurrency, the FP8 advantage is masked by KV pressure, prefill compute saturation, or memory bandwidth, depending on regime. This is why the Phase A and Phase B portfolio cannot show the FP8 number on Qwen2.5-3B, which is too small for the FP8/BF16 throughput delta to surface cleanly.

The point of running a 13B-class model for the contrast (Llama 3.1 8B at minimum) is that smaller models are not compute-bound enough to expose the FP8 advantage. On Qwen2.5-3B with 2 GPUs of HBM3 (160 GB aggregate), I am running a model into 5% of the available memory and the per-layer compute is fast enough that other costs dominate. With Llama 3.1 8B I am closer to a regime where the matmul is large enough that doubling the tensor-core throughput is visible in the headline tokens-per-second number.

The honest scope note this day forces is what the contrast does *not* close. Multi-node NCCL behavior, MoE all-to-all comm patterns, MIG slicing, full-mesh NVSwitch (only relevant at 4-GPU and above), and rack-scale NVLink fabric (NVL72) all remain out of scope. The contrast closes the per-link bandwidth question and the per-precision throughput question. It does not close the cross-node or cross-precision-tier portfolio gaps.

Tools: vLLM with `--dtype` switching for the BF16/FP8 baseline; same NCCL and profiler stack as Days 31-33 for parameter-identical sweeps.

Discussion seeds:
- The A10G PCIe comm fraction at concurrency 16 was X%. The H100 NVLink number is much smaller. What changes about the recommended TP-versus-quantization decision when comm becomes a footnote rather than a tax?
- FP8 doubles tensor-core throughput on paper. On a 3B model the doubling does not show up cleanly. Why? What does that tell me about the regimes where FP8 actually wins?
- A10G has no NVLink at any scope, regardless of provider. What is the mechanical reason, and what does it say about how NVIDIA segments the cloud GPU market?

## Day 35, multi-GPU serving doc as synthesis

Synthesis day pulls every number into one architecture document. The headline analysis is the Amdahl crossover: at what TP degree does communication time exceed compute time, and how does the answer depend on the link? The mechanism is straightforward. Per-layer compute time for a transformer scales roughly with `FLOPs_per_layer / GPU_TFLOPS` and is largely insensitive to TP degree at fixed batch (every GPU does its share faster as TP grows). Per-layer all-reduce time scales with `message_size / NCCL_bandwidth` and is largely insensitive to TP degree at fixed batch (the message size grows with the activation slice in some directions but the bandwidth scales with the link, not with how many ranks). At some TP degree the comm time crosses over the compute time, and beyond that point adding more GPUs makes the system slower, not faster.

Phase A and Phase B Track 1 give me the compute-side anchor. The Day 6 prefill regression (`TTFT = 37.6 + 0.228 × prompt_tokens`, R² = 0.9966) tells me that Qwen2.5-3B prefill on T4 runs at 4,386 tokens per second per GPU under single-request conditions. A 2,000-token prompt occupies the GPU for 493 ms. The slope, 0.228 ms per token, is the marginal compute cost. On A10G the slope is different (faster per-token, lower memory bandwidth), and the H100 contrast from Day 34 gives a third data point. Three slopes, three regimes.

The A10G PCIe NCCL bandwidth from Day 31 gives me one comm-side anchor. The H100 NVLink NCCL bandwidth from Day 34 gives me a second. With both, I can plot "compute time per layer at TP=K" against "all-reduce time per layer at TP=K" for K in {2, 4, 8, 16}, find the crossover for each link, and read off the maximum efficient TP degree per regime. The answer is not a single number; it is a pair of numbers, one for PCIe and one for NVLink, and the gap between them is the headline finding.

The interview-grade framing of the Decision Section forces me to commit. "Given Qwen2.5-3B workload on A10G PCIe, I would choose TP=2 over single-GPU INT8 because comm fraction at concurrency 16 is X% and the TP=2 throughput advantage is Y% per dollar even after the comm tax. What gets worse: cost-per-token Z% higher than single-GPU INT8. Not optimizing for: peak single-request TTFT, which favors single-GPU. What would change this: workload shifting toward latency-sensitive short-prompt-heavy traffic (favors single-GPU), or migration to H100 NVLink hardware where the comm tax disappears (changes the calculus entirely)." The structure forces tradeoffs into prose; the data forces the tradeoffs to have specific numbers.

The Section 7 hardware contrast is what makes Deliverable #10 portfolio-credible rather than parochial. Without it, the document reads as "what I learned on the cheapest hardware available." With it, the document reads as "what the cost-conscious analysis is, what the frontier-lab analysis is, and how the conclusions diverge between regimes." The latter is a staff-level artifact.

Discussion seeds:
- Walk me through Amdahl crossover analysis for TP. What scales with TP degree on the compute side? What scales on the comm side? Where do they cross for PCIe versus NVLink, and what is the mechanical reason?
- The Decision Section commits to TP=2 over single-GPU INT8 with Y% advantage. What workload shift would flip this decision back to single-GPU, and how would I know it was happening from telemetry?
- Section 7 contrasts A10G PCIe with H100 NVLink. Both numbers are real measurements. What is the *third* hardware regime I am explicitly not measuring (multi-node), and why does naming the gap matter for portfolio credibility?

## Day 36, where does multi-GPU break

The Day 24 cliff is the anchor: at 87% KV utilization on T4 single-GPU, the TTFT p99/p50 divergence ratio jumps from 1.21 (c=108) to 2.06 (c=113) over a 1.7-percentage-point KV change, with TTFT p99 going from 3,200 ms to 7,587 ms. The cliff is defined by the divergence ratio crossing 2.0, not by a raw utilization number. The mechanism is vLLM V1's recompute-only preemption: when KV blocks run out, the scheduler preempts running requests, the preempted requests lose their KV state and re-enter the wait queue, when rescheduled they re-prefill the same blocks they just freed, and under sustained pressure this creates a positive feedback loop where more compute is spent on recompute than on forward progress. The Day 24 data shows recompute fraction crossing 30% at the cliff and reaching 75% above it.

Under TP=2 the cliff's location, sharpness, and recovery shape can all shift, and the question is which direction. Three first-principles arguments compete. First, TP=2 splits the model weights across two GPUs, leaving more aggregate HBM available for KV cache; per-GPU KV budget can grow, pushing the cliff right. Second, TP=2 introduces synchronous all-reduce on every layer, making preemption more expensive: a preemption now requires both GPUs to roll back coherently, the recompute on re-prefill incurs the all-reduce cost again, and the per-event recompute time is higher. Third, the NCCL synchronization tightens the iteration timing, potentially making the system more or less sensitive to KV jitter (Day 24 measured ±15pp jitter at fixed concurrency on T4) depending on how the all-reduce interacts with batch composition.

The *direction* of the shift is not derivable from first principles; the relative magnitudes of these three effects determine the answer, and the magnitudes are workload-, model-, and link-dependent. The experiment gives me a second cliff curve to plot side-by-side. The comparison note becomes the interview artifact: "TP=2 shifted the cliff from 87% to Y% because [mechanism]. The recompute cost per preemption changed from Z to W because the all-reduce now sits on the critical path of every recompute-prefill iteration. The cliff sharpness [steepened/softened] because [mechanism]." Each blank fills with a measured number plus a mechanical explanation.

The non-obvious aspect this day pre-empts is that the cliff is not a property of the hardware. It is a property of the scheduler's preemption policy under a memory constraint. vLLM V1 chose recompute-only preemption deliberately (V0 had CPU swap; V1 removed it because swap-in latency dominated under high KV churn). The cliff exists because of this design choice, and it would have a different shape under a different scheduler. Knowing this is a marker between "I have used vLLM" and "I understand why vLLM behaves the way it does."

Tools: same Day 24 sweep methodology adapted to TP=2 deployment; vLLM metrics for per-iteration preemption count and KV utilization.

Discussion seeds:
- The cliff at 87% on T4 was about the scheduler's preemption policy, not the hardware. If TP=2 shifts the cliff, what would that tell me about how the policy and the hardware interact?
- Three competing effects move the cliff in opposite directions under TP=2: more aggregate HBM per GPU, more expensive preemption, tighter iteration timing. Which effect would I expect to dominate on PCIe versus NVLink, and why?
- vLLM V1 dropped V0's CPU swap. What does that decision say about the design trade between memory pressure and recompute cost, and what would change if HBM were 4x larger?

## Day 37, retry storm under multi-GPU

Day 26 worked the retry storm conceptually rather than experimentally: amplification factor follows the geometric series `sum(timeout_rate^i for i in range(max_attempts))`, so a 40% timeout rate with 3 max attempts gives 1.56. The Day 26 decision was to skip the GPU experiment because the mechanism is fully deducible from the cliff data plus the formula, and the specific workload-dependent numbers do not transfer. That decision is worth remembering when I plan Day 37: the goal under TP=2 is not to re-derive the formula but to find what changes mechanistically about how the system absorbs and recovers from the storm.

Three things plausibly change. First, KV pressure during the storm. With TP=2's larger aggregate HBM, the per-replica KV pool is bigger, so the same retry rate consumes a smaller fraction of pool capacity, and the cliff fires later. Second, recovery speed. Day 26's analysis identified recovery as a drain problem, not a convergence problem: the queue accumulates retry-stale requests indistinguishable from real ones, and recovery time is set by how fast the queue drains, not by when the trigger ends. Under TP=2, drain throughput depends on per-iteration cost, which now includes the all-reduce tax. Drain might be slower, even though the cliff is later. Third, cascade speed. The same KV-cliff feedback loop drives the cascade in both cases, but the rate at which crossing the cliff degrades each subsequent request depends on how preemption interacts with the all-reduce synchronization. This is the same "preemption is more expensive under TP" question from Day 36, applied to the cascade phase rather than the steady-state cliff.

The Day 28 SIGTERM observation matters here. On a graceful drain, in-flight requests block silently for the full client timeout (30 seconds was the measured number) before retrying, and the retries arrive in a synchronized burst. If the surviving replicas are at 79% steady-state KV (the Day 24 safe operating point), 20 force-terminated requests at 1.56 amplification add 33pp of KV pressure to survivors, taking them past the 87% cliff. This was the cascade math in Deliverable #9. Under TP=2 with larger per-replica KV budget, the same 33pp pressure surge consumes a smaller fraction of the new pool, possibly leaving headroom; or possibly not, depending on how the new safe operating point sits. The number changes; the cascade structure does not.

Synthesis for the appendix in Deliverable #10 is two paragraphs at most. Staff-signal sentence: "Under TP=2, the cliff shifted from 87% to Y% because [larger per-GPU KV budget, modulated by more expensive preemption], and retry amplification under burst behaved differently because [drain throughput under TP=2 reflects the all-reduce tax]." The portfolio reader gets one paragraph of mechanism plus measured numbers, not an experiment write-up.

The Day 38 Number Sheet pre-fill is the operational discipline: every cell that will appear in Deliverable #11 has a value before Day 38 morning, sourced from a specific prior day. This is what turns Day 38 from derivation into assembly.

Discussion seeds:
- Under TP=2 the cliff might move to a higher KV percentage because the per-GPU KV pool is larger, but recovery might be slower because the all-reduce tax slows drain. How would I tell, from a retry-storm trace, which effect dominated?
- Day 26 deduced amplification factor from the geometric series rather than measuring it. Why is this a defensible choice for an experiment-heavy program, and what would the measured number have added that the formula did not?
- The cascade math says 20 force-terminated requests at 1.56 amplification, 1.05% KV per request, gives 33pp of pressure. Under TP=2 the per-request KV cost might change. Why, and which direction?

## Day 38, master platform design

Deliverable #11 is the document a hiring manager reads first. It pulls every measurement from Phase A and Phase B into a coherent system design with no hypotheticals. The structure is workload, capacity, admission, architecture, autoscaling, failure handling, known failure modes. Each section is parameterized by a specific measured number.

The capacity section is where the GQA correction shows up. Qwen2.5-3B uses Grouped Query Attention with `num_kv_heads=2`, not 16. The KV cost per token is `2 × num_kv_heads × head_dim × bytes_per_element × layers` = `2 × 2 × 128 × 2 × 36` = 36,864 bytes, around 36 KiB. Day 13 caught this 8x error: using 16 query heads would have predicted max concurrent at 0.7 requests, which is nonsensical and would have failed the back-of-envelope sanity check. With the correct number the predicted max concurrent matches the observed Day 21 failure point exactly (5.9 predicted, c=6 observed at gpu_memory_utilization=0.45). The capacity model is the place this prediction comes home; the number propagates into admission control, autoscaling thresholds, and the cost model.

The admission control section uses the Day 17 token-budget reasoning. The token budget is `max_prompt_length × max_concurrency`, not just concurrency, because two requests with 2,000-token prompts consume the same KV as four requests with 1,000-token prompts. This is why bounded-queue rejection beats fail-fast at the admission boundary: the gateway can shed load smoothly without per-request retry cascades. The 85% KV hard rejection from the Day 25 admission-control retrofit is the fall-through: when the gateway smoothing fails or an instantaneous burst exceeds it, the engine itself refuses new requests at 85% KV to keep the system below the 87% cliff.

The autoscaling section comes from Deliverable #9: composite trigger of `queue_depth > 10 AND kv_util > 76% for > 30s`, with thresholds derived from the cliff. K = 76% is `87% (cliff) − 10 × 1.05% (per-request KV) ≈ 76.5%`, rounded down. Q = 10 is 10% of safe operating concurrency at c=95. The signal hierarchy is queue depth as leading, KV utilization as coincident confirmation, TTFT p99 as lagging SLO watchdog, with CPU utilization and GPU compute % explicitly excluded as wrong signals (Day 27 and Day 28: CPU was 20-40% during the cliff event, and `nvidia-smi` GPU utilization read 98% in both healthy and degraded states). The signal hierarchy is the place where Phase A and Phase B Track 1 converge into a single design rule.

The failure-handling section pulls from Day 26 (retry budget bounds amplification to a geometric-series ceiling, jitter desynchronizes thundering herd), Day 28 (graceful drain state machine RUNNING → DRAINING → TERMINATING with 120s hard kill), and Day 23 (chunked prefill bounds per-iteration starvation, not throughput-free above ~70% KV utilization). Each subsystem has a specific named mechanism, a specific measured cost, and a specific failure mode that surfaces if the mechanism is missing.

The known-failure-modes section is the closer. KV exhaustion: cliff at 87% (Day 24), recompute-only preemption feedback loop, mitigation via admission control and operating below 79% mean KV. Retry cascade: amplification 1.56 at 40% timeout/3 attempts, drain hazard 33pp KV from 20 force-terminated requests, mitigation via graceful drain and retry budget. Prefill/decode interference: 4.57x p99 penalty non-chunked, 2.13x chunked, mitigation via chunked prefill or queue-tiering or PD-disagg. TP straggler: convoy effect, system-wide TTFT degrades with slowest GPU's clock, mitigation via fleet homogeneity and per-GPU health monitoring. Each one paragraph, each one measurement-anchored.

The Decision Section commits to a specific architecture for a specific workload and quantifies what gets worse. Day 38 is assembly, not derivation; the discipline is that the Number Sheet from Day 37 contains every cell, and Day 38 prose pulls them into structure.

Discussion seeds:
- The capacity model uses `num_kv_heads=2` for Qwen2.5-3B. What would change if the model used MQA (1 KV head) or full multi-head attention (16 KV heads), and how would the cliff and the autoscaling threshold move?
- The composite autoscaling trigger is `queue > 10 AND KV > 76% for 30s`. Why is the AND important rather than either signal alone? What would each fail to catch on its own?
- Walk me through the failure-mode section. Pick one failure mode and explain the mechanism, the mitigation, and what gets worse if the mitigation is not in place.

## Day 39, mistakes, audit, exit

Deliverable #12 picks the five times during the residency where my mental model was wrong and a measurement corrected me. The discipline is not "things I got wrong" in general; it is "things I would have gotten wrong without running the experiment." Three quality gates per entry: it required running the experiment (not predictable from documentation), it has a specific number, and it caused a concrete design or mental-model change.

The seeded candidates from `mistakes_log.md` already have the structure. The GQA `num_kv_heads` 8x error (Day 13) is the cleanest: the documentation describes Grouped Query Attention but does not flag that the KV cost calculation must use `num_kv_heads`, not `num_attention_heads`. The error compresses the predicted KV budget by 8x, predicting that Qwen2.5-3B cannot serve a single 2,770-token request on T4 (16,352 / 2,770 = 5.9 with GQA, or 0.7 without). The corrected number matches observed failure at c=6, exactly. This is a measurement correcting a back-of-envelope, with a specific number and a concrete change in how I size every model.

The cliff-as-divergence-ratio mistake (Day 16) is a deeper one. The early intuition was that "cliff" means "KV utilization above some threshold." The data showed the cliff is defined by the TTFT p99/p50 divergence ratio crossing 2.0, and the KV utilization at which the divergence happens is workload- and pool-dependent (87% on this T4 config, would be different on another). Calling the cliff a utilization point would have led to the wrong threshold derivation in Deliverable #9; instead, the threshold is `cliff (87%) − 10 × 1.05% (per-request KV)` where the cliff itself has to be re-measured per workload.

The GPU-util-as-scaling-signal mistake (Day 28) is the most operationally consequential. The intuition was that GPU utilization differentiates healthy from degraded states. The data showed `nvidia-smi` GPU utilization reads ~98% in both, and even DCGM SM-active reads similarly because the scheduler is busy cycling through preemption rather than running inference kernels. The implication is that no GPU-util-based HPA could ever fire during the cliff event, and any CPU-util-based HPA would also miss it (CPU was 20-40%). This is the kind of mistake that gets a frontier-lab interview question: "Why is queue depth a better autoscaling signal than GPU utilization?" The wrong answer is "queue depth fires earlier." The right answer is "GPU utilization is structurally blind to KV pressure because it measures the wrong dimension; the kernel is busy whether the work is forward progress or recompute churn."

The signal-lag mistakes from Day 28 (metric name with labels, TTFT measured as total request time, prompts too short to fill KV) are sharper than they look. Each represents a place where the measurement instrument was wrong in a way that would have produced a confident-sounding but wrong conclusion. They are not interview-grade individually, but they collectively make the point that "I measured X" is a claim that requires verification of the instrument, not just the data.

The exit assessment is nine questions; the hard gate is that I can answer all of them confidently in writing. Sample questions from the v4 syllabus: why does p99 TTFT explode before average GPU utilization looks concerning (the recompute feedback loop, mechanism not just correlation); walk through a retry storm with my actual data (amplification 1.56 from the geometric series, cascade math 33pp, recovery as a drain problem); why do GPUs under TP behave like a convoy (synchronous all-reduce, slowest GPU sets pace); why is queue depth better than GPU utilization (different dimensions, structural blindness); explain prefill/decode interference (4.57x p99 penalty non-chunked, mitigation via chunking or PD-disagg); TP versus PP (TP for fast interconnect and latency-sensitive workloads, PP when bandwidth-bound and bubble-tolerant); NCCL operation TP uses (per-layer all-reduce, comm fraction at concurrency 16 was Y%); straggler at TP=4 with one GPU at 70% (system-wide degradation linear in slowest clock); Deliverable #11 critique (where the design breaks first, what it is not optimizing for).

The portfolio audit makes everything self-consistent. Each deliverable cross-references the others; each number cited matches the source measurement; each graph has axes, units, legend, title; each writing pass strips "I learned that" constructions and replaces them with declarative claims backed by numbers. The audit catches the drift that compounds over an 18-day program: a number cited from memory rather than from data; a regime claim made about chunked prefill that does not hold above 70% utilization; a graph titled correctly but showing the wrong run.

Discussion seeds:
- Why is `num_kv_heads=2` for Qwen2.5-3B not derivable from "it has 36 layers and hidden_dim 2048"? What does GQA do mechanically, and why does the KV cost depend on the smaller number?
- The cliff is defined by divergence ratio, not raw utilization. Why does this matter for transferring the cliff finding to a different workload, and what would I have gotten wrong if I had used a raw KV percentage?
- "GPU utilization is the wrong autoscaling signal", what does "wrong dimension" mean specifically? At the Day 24 cliff, what is the GPU actually doing while utilization reads 98%, and why does that work make the signal useless?

## Day 40+ parallel track, KubeRay autoscaling re-execution

The autoscaling memo from Day 29 made claims grounded in in-process measurements: cliff at 87%, signal lag for queue-depth HPA, scale-down hazard via SIGTERM 30s silent hang, composite trigger derivation from the cliff math. The structure assumes the metrics exist, the controller reads them, and the scale-out decision propagates to a new replica. Each of those assumptions is a separate piece of infrastructure, each adds latency, and none of the latencies are in the standalone measurement.

Three layers of latency the standalone memo cannot see. The metrics pipeline runs on a scrape interval, typically 15-30 seconds. The HPA controller polls the custom metrics API on its own loop, with a stabilization window typically defaulting to 30 seconds. The new pod, once decided, takes the 94.2 seconds Day 28 measured for cold start. The Day 29 memo treats t_cold as the dominant term and concludes that reactive autoscaling is structurally inadequate for this configuration. The KubeRay re-run gives me the *other* terms (scrape latency and controller latency) as separately measurable numbers, and the comparison reveals which slice of the total scale-out time each layer contributes.

The "fake vLLM" plumbing strategy on a CPU node pool de-risks the entire pipeline before the GPU meter starts. The trick is mirroring the metric exactly: same gauge name (`vllm_gpu_cache_usage_perc`), same units, same label keys. If the fake exposes the metric correctly, the rest of the pipeline (Prometheus scrape, prometheus-adapter rule, custom metrics API surface, HPA reference) is independent of whether the workload is real. Validating the entire chain with no GPU costs almost nothing in dollars, surfaces the configuration mistakes that would otherwise burn GPU hours, and leaves the GPU swap as a near-noop. This is the same discipline as the Day 6 prefill-regression sequence: change one variable at a time, anchor the experiment to a known-good baseline, and pay attention to instrumentation correctness before drawing conclusions.

The new findings the cluster surfaces are not in the standalone memo. Pod cold-start time on a real cluster can differ from the in-process measurement because of image pull, init containers, readiness probe delay, and KubeRay's own scheduling overhead. The HPA controller loop latency and stabilization window introduce another delay between "metric crosses threshold" and "scale-out fires" that the standalone analysis does not see. The graceful-drain story from Day 28 plays out differently when the load balancer is K8s Service rather than a local socket: connection draining, endpoint slice updates, and the readiness-probe handshake all add behavior that does not exist in the standalone case.

The defensibility for the portfolio is "I have operated KubeRay on GPU." Not "I have read the docs." The autoscaling memo becomes, after this addendum, the only autoscaling artifact in the portfolio backed by an actual cluster orchestrator I have stood up myself. That is the leverage relative to my background, where the orchestrator side is already a strength.

Tools: KubeRay operator, RayService for the workload, kube-prometheus-stack with prometheus-adapter, HPA referencing a custom metric, Locust for traffic, fake-vLLM in FastAPI for the CPU plumbing phase.

Discussion seeds:
- The standalone Day 29 memo says reactive autoscaling fails because t_cold (94.2s) dominates t_headroom (seconds). On a real cluster, what additional latency layers exist between "metric crosses threshold" and "new pod serves traffic," and which of them are eliminated by pre-warming versus only by t_cold reduction?
- The fake-vLLM plumbing strategy validates the metric pipeline on CPU before the GPU swap. Why does this work, and what specifically would go wrong if I skipped this step and went straight to GPU?
- The autoscaling memo's composite trigger fires at `queue > 10 AND KV > 76% for 30s`. On a real cluster, the "for 30s" condition is enforced by the HPA stabilization window, not by the gateway. What does this mean operationally, and what could go wrong if the stabilization window is too short or too long?

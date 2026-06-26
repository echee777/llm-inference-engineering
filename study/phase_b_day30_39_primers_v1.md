# Phase B Remaining Schedule with Primers

Days 30 to 39 in execution order. Plus an opportunistic parallel track for KubeRay work that can be picked up between Phase B days or after Day 39. Plain-language goal per day with technical terms in parentheses for cross-reference to `phase_b_day29_40_compressed_v4.md`.

## Schedule

```
Day 30  Multi-GPU environment up
Day 31  Measure how fast two GPUs talk
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

The wires between GPUs (interconnect: NVLink vs PCIe vs cross-root PCIe) matter more than the GPUs themselves for inference work. Measure what wires you actually have (`nvidia-smi topo -m`). Prove the two GPUs can talk (NCCL all-reduce smoke test, `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2`) before measuring how well they talk. Morning is Track 1 polish (Deliverables #5 to #9 graph audit, number consistency, writing sweep). Afternoon is the hard gate that has to clear before any later experiment is valid.

Walk-away: confidence the measurement environment isn't lying to you (silent topology fallback, TP=2 silently running on a single GPU, NCCL warnings degrading bandwidth 5x to 10x).

## Day 31, measure how fast two GPUs talk to each other

GPU-to-GPU communication speed isn't a single number. NCCL all-reduce bandwidth depends on message size. Tiny messages: setup overhead dominates, wire mostly idle (latency-bound regime, sub-MB). Large messages: wire saturates, you hit the physical ceiling (bandwidth-bound regime, above saturation, typically 1 MB to 4 MB on PCIe Gen4). Sweep 1K to 512M to find the inflection. Then deploy a model split across both GPUs (vLLM with `--tensor-parallel-size 2`, called TP=2) and capture the single-request baseline (TTFT, throughput at concurrency 1).

Walk-away: a curve showing which message-size regime your inference workload (per-layer all-reduce calls during forward pass) actually lives in.

## Day 32, figure out how much runtime is talking vs working

Splitting a model across GPUs (TP) adds a communication tax: every transformer layer needs an all-reduce to combine partial results. Quantify the tax (communication fraction = NCCL time / total iteration time) using a profiler (torch.profiler). As concurrency rises, balance shifts: low concurrency means high comm fraction; high concurrency means matmul amortizes the comm and compute can dominate. Plus a profiling extension showing what's actually happening on the chip (DCGM tensor-core utilization via metric `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, NCCL subsystem trace via `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`).

Walk-away: ability to say "at concurrency N, iteration time is bounded by X, dominated by Y because [mechanism]". The headline interview-grade analysis skill of the multi-GPU week.

## Day 33, try the alternative split, and break a GPU on purpose

Two ways to split a model across GPUs. Within each layer (Tensor Parallelism, TP, what you've been doing, all-reduce per layer, bandwidth-hungry). Or across layers (Pipeline Parallelism, PP, point-to-point send/recv at stage boundaries, PCIe-friendly but introduces "bubbles" where GPUs sit idle during pipeline fill/drain). Compare measured if vLLM PP is available, theoretical otherwise. Then throttle one GPU (`nvidia-smi -i 1 -lgc 900`) to show the convoy effect: synchronous all-reduce means the system runs at the speed of the slowest GPU.

Walk-away: defensible "TP vs PP" decision rule with measurement. Plus understanding of why production fleets care about GPU homogeneity (no mixed generations in a TP group, per-GPU health monitoring, straggler detection as a production requirement).

## Day 34, run on premium hardware to see what good looks like

Every measurement so far has been on cost-conscious hardware (A10G, PCIe Gen4, no NVLink, no FP8 support, GDDR6 memory). The cheap interconnect makes the multi-GPU tax painful, which is instructive. Spend $60 to $100 for one day on Hopper-class hardware (2x H100 SXM, NVLink 4.0 at ~450 GB/s/GPU vs PCIe Gen4 ~32 GB/s, HBM3 memory) to see how much the cheap-hardware lessons transfer. Two contrasts: NVLink-fast TP communication (much lower comm fraction), and FP8 throughput (Hopper Tensor Cores running `fp8_e4m3`, ~1.5-2x throughput gain over BF16). Use Llama 3.1 8B or larger; Qwen2.5-3B is too small to show the FP8 delta cleanly. Hard cap: spin down by EOD.

Walk-away: defensible "I have run on H100" claim. Measured numbers showing which lessons depend on hardware tier and which don't.

## Day 35, write the multi-GPU serving story

Synthesis day. Pull every number into one architecture document (Deliverable #10, "Multi-GPU Serving Architecture Document", 7 sections: NCCL profiles, TP under load, TP vs single-GPU trade study, PP analysis, straggler impact, Amdahl recommendations, hardware contrast). The headline is the Amdahl crossover: at what TP degree (TP=4, TP=8 hypotheticals) does communication time exceed compute time? Two data points (A10G PCIe and H100 NVLink) bound the answer in two regimes. Mandatory Decision Section ("Given X, I would choose Y because [data]. What gets worse: Z. Not optimizing for: W. What would change this: V").

Walk-away: Deliverable #10. The seven-section document an interviewer can flip through to see the analysis chain end-to-end.

## Day 36, re-find the breaking point under multi-GPU

In single-GPU work you found where the system breaks (Day 24 cliff: TTFT divergence ratio p99/p50 > 2x, occurring at ~87% KV cache utilization, sharpness 2.4x p99 jump over 1.7pp). Does that breaking point shift under TP=2? Per-GPU KV budget differs (model weights take less per-GPU, more headroom for KV cache). Does the cliff get sharper or softer? Preemption under TP=2 requires all-reduce synchronization, which may change preemption cost.

Walk-away: side-by-side TP=1 vs TP=2 cliff plot with mechanistic explanation of any shift. One-page comparison note filed as appendix to Deliverable #10.

## Day 37, re-run the retry storm under multi-GPU

You already measured how retries amplify load (Day 26: amplification factor peak under 60% load, 2s timeout, 3 retries, burst trigger). Re-measure under TP=2. Write the multi-GPU failure-mode story end-to-end: combine TP=2 cliff (Day 36) and TP=2 retry storm into a single appendix for Deliverable #10. Then pre-fill the Deliverable #11 Number Sheet (`deliverable_11_numbers.md`, 21 fields) so Day 38 is assembly not derivation.

Walk-away: unified multi-GPU failure-mode story. Staff-signal sentence: "Under TP=2, the cliff shifted from X% to Y% because [mechanism], and retry amplification changed from Zx to Wx because [mechanism]". Plus a complete Number Sheet.

## Day 38, write the master platform design

End-to-end synthesis. Take every number from Phase A and Phase B and write the design document you would hand a team to build this serving system (Deliverable #11, "End-to-End Inference Platform Design", 4 to 6 pages, 7 sections: Workload Model, Capacity Model, Admission Control, Serving Architecture, Autoscaling, Failure Handling, Known Failure Modes). Every claim cites a specific measurement, no hypotheticals. Day 38 is assembly only; all numbers should already be in the Number Sheet from Day 37 PM.

Walk-away: Deliverable #11. The portfolio-cover artifact. The single document a hiring manager would read first.

## Day 39, top 5 mistakes, portfolio audit, exit interview

Your own weaknesses, on the record. Pick the five times during the residency where your mental model was wrong and the data corrected you (Deliverable #12, "Top 5 Mistakes I Made"). Quality gates per entry: required running an experiment (not predictable from docs), references a specific number, caused a concrete design change. Audit all 8 Phase B deliverables for consistency (every claim has a number, graphs publication-quality, cross-references intact). Answer 9 exit assessment questions in writing (TTFT cliff explanation, retry storm walkthrough with amplification factor, TP convoy effect mechanism, queue depth vs GPU util as scaling signal, prefill/decode interference, TP vs PP decision rule, NCCL all-reduce comm fraction, straggler at TP=4 with one GPU at 70% clock, Deliverable #11 critique).

Walk-away: Phase B complete. Portfolio coherent. You know exactly which questions you can answer cold and which need work. Hard gate: do not proceed to Phase C until all 9 exit questions can be answered confidently.

## Day 40+ parallel track, KubeRay autoscaling re-execution

Standalone work that can be picked up between Phase B days when you have spare cycles, or after Day 39 if easier to batch. Not blocking anything in Phase B's main flow (Days 30 to 39 don't read its outputs). Deliverable #9 ships with a marked "addendum pending" note until this lands.

The goal: take the autoscaling memo from Day 29 and run its claims on a real Kubernetes cluster (your existing `infra_kuberay` EKS setup, KubeRay operator, RayService manifest, kube-prometheus-stack with prometheus-adapter mapping `vllm_gpu_cache_usage_perc` to `external.metrics.k8s.io/v1beta1`, HPA referencing the custom metric). The memo's measurements were taken in-process. A real cluster adds layers: a metrics pipeline scraping on an interval (15-30s typical), a controller polling the metrics (HPA stabilization window, 30s default), pods that take time to actually start (model load time, 30-120s cold start). Each adds latency the standalone memo doesn't account for.

Walk-away: the lag the memo missed (delta between "controller decided to scale" and "new pod is actually serving traffic"). Plus a defensible "I have operated KubeRay on GPU" claim tying the autoscaling memo to the actual infrastructure repo.

Two-phase plan to control GPU spend:

```
Phase 1, morning, ~4 hrs, ~$0
  Build fake-vLLM (FastAPI service exposing /metrics with the same gauge
    name and label keys as real vLLM)
  Terraform the KubeRay + Prometheus + Adapter + HPA pipeline on a CPU
    node pool
  Validate end-to-end with Locust traffic, no GPU confounders

Phase 2, afternoon, ~4 hrs, ~$5 GPU spend
  Provision GPU node pool (g5.xlarge spot), swap fake-vLLM image for
    real vLLM
  Re-run Day 28 three-policy comparison (CPU-based HPA, GPU-util HPA,
    queue-depth HPA) on the real cluster
  Capture cluster-level signals new vs Day 28 standalone (pod cold-start
    time, HPA controller loop latency, in-flight request handling on
    scale-down)
```

If deferred past Day 39, Top 5 Mistakes and Phase B narrative won't include KubeRay-specific entries. That's the trade-off for deferring.

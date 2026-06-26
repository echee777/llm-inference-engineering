# Day 23 Session State

## Transfer Instructions

Copy these files to your laptop:
```
phase-b/day22/run_a_mixed.csv            # Chunked mixed traffic (3,504 short + 912 long)
phase-b/day22/run_b_short.csv            # Chunked short-only control (9,312 short)
phase-b/day22/profiling-notes.md         # Day 22 profiling methodology
phase-b/day23/day23-syllabus-v4.md       # Full syllabus
phase-b/day23/session-state.md           # This file
phase-b/day23/experiment-results.md      # Full experiment results summary
phase-b/day23/run_nochunked_mixed.csv    # Non-chunked mixed traffic (3,504 short + 912 long)
phase-b/day23/run_nochunked_short.csv    # Non-chunked short-only control (9,216 short)
```

## Where We Are

Day 23 /teachme session COMPLETE. All 9 concepts taught, interview checkpoint passed, Deliverable #6 written.
**ALL GPU EXPERIMENTS ARE COMPLETE.** All plots generated. Deliverable written.

## Critical Discovery

vLLM 0.17.1 V1 has chunked prefill enabled by default (max_num_batched_tokens=2048).
Day 22 data was collected WITH chunked prefill. The experiment was inverted:
we ran with `--no-enable-chunked-prefill` to get the non-chunked baseline.

## Concept Map (9 concepts, in order)

```
#  Concept                                           Mode     Status
──────────────────────────────────────────────────────────────────────────
1  Prefill/decode asymmetry (compute vs bandwidth)   [QUIZ]   DONE
2  Iteration-level serialization in vLLM V1          [QUIZ]   DONE
3  Decode starvation window calculation              [QUIZ]   DONE
4  Why GPU utilization is misleading                 [QUIZ]   DONE
5  CDF plots as diagnostic tools                     [DO]     DONE
6  Chunked prefill mechanism and tradeoff            [TEACH]  DONE
7  Chunk size as control knob                        [QUIZ]   DONE
8  Continuous batching misconception                 [QUIZ]   DONE
9  Production mitigations (3 levels)                 [TEACH]  DONE
```

## Full Experiment Data

```
                          Non-Chunked                 Chunked (default)
                          Mixed     Short-only        Mixed     Short-only
Short TTFT p50            181.5ms   138.6ms           180.4ms   135.8ms
Short TTFT p99            1051.9ms  230.4ms           445.5ms   209.5ms
Short TTFT max            1056.7ms  332.9ms           675.2ms   309.6ms

Interference penalty      4.57x                       2.13x
  (p99 mixed/isolated)

Avg tokens/iteration      411.5     ~32               275.9     31.8
Long throughput (tok/s)   778.2     n/a               778.2     n/a
Total throughput (tok/s)  1525.8    n/a               1525.8    n/a
Overall req/s             7.36      15.36             7.36      15.52
```

## Key Numbers for Deliverable #6

```
Short-request TTFT p99 (isolated, chunked)      209.5ms
Short-request TTFT p99 (mixed, non-chunked)     1051.9ms
Short-request TTFT p99 (mixed, chunked)         445.5ms
p99 degradation factor (non-chunked)            4.57x
p99 degradation factor (chunked)                2.13x
Chunked prefill short p99 improvement           57.6% (1052 -> 446ms)
Chunked prefill long throughput cost             0% (778.2 tok/s both)
```

## Experiment setup

- gpu_memory_utilization=0.90, max-num-seqs=128, max-model-len=4096
- Split concurrency pool: 48 short slots + 48 long slots (guarantees 50/50 in-flight)
- Short: 64 prompt tokens, 128 max_new_tokens
- Long: 2048 prompt tokens, 512 max_new_tokens
- Zero preemptions, zero queuing in all runs

## What Can Be Done On Laptop (no GPU) -- EVERYTHING REMAINING

1. All 9 concepts (quiz/teach interactive session)
2. CDF plot generation from existing CSVs
3. Bimodal TTFT histogram from existing CSVs
4. Writing all 5 sections of Deliverable #6 (all data is available)

## Resume Prompt

On laptop: `/teachme day23` and then "resuming from session-state.md, all experiments done"

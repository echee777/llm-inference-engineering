# Day 9 — Mini Collapse Experiment

## Setup

- **Model:** Qwen2.5-3B-Instruct on Tesla T4 (15GB)
- **Flags:** `--gpu-memory-utilization 0.45 --max-model-len 2048 --max-num-seqs 32 --dtype half`
- **KV cache pool:** 1,034 blocks (16 tokens/block)
- **Load:** 80 concurrent requests, short prompt (~15 tokens), `max_tokens=1900`
- **Bottleneck mode:** Memory-limited (not concurrency-limited)

Earlier attempts at `0.90` and `0.55` utilization failed to trigger preemption — the block pool was large enough that `--max-num-seqs 32` throttled admission before memory exhausted. Lowering to `0.45` shrank the pool to 1,034 blocks, forcing the memory gate to fire before the concurrency gate.

---

## Timeline Table (from actual log timestamps)

| Elapsed   | Event                                      | free/total     | Detail                                                                                              |
| --------- | ------------------------------------------ | -------------- | --------------------------------------------------------------------------------------------------- |
| T+0.0s    | First BLOCK_ALLOC — 80 requests dispatched | 1033/1034      | Initial prefill allocations begin                                                                   |
| T+9.6s    | 50% utilization crossed                    | 517/1034       | Steady block consumption, ~32 seqs running                                                          |
| T+16.4s   | 80% utilization                            | 206/1034       | Decode-phase blocks accumulating                                                                    |
| T+18.5s   | 90% utilization                            | 103/1034       | Approaching exhaustion                                                                              |
| T+19.1s   | 95% utilization                            | 51/1034        | Critical zone                                                                                       |
| T+22.4s   | **free=0 — first PREEMPT + ALLOC_FAIL**    | 0/1034         | 4 requests preempted simultaneously (49–209 tokens lost)                                            |
| T+22–40s  | Preemption spiral (wave 1)                 | 0–51/1034      | 34 preemptions; blocks oscillate between 0 and ~51; re-preemption counts climb to #5–#6 per request |
| T+40–50s  | Brief partial recovery                     | up to 206/1034 | Some requests finish, free blocks briefly rise, then immediately re-consumed                        |
| T+50–78s  | Preemption spiral (wave 2)                 | 0–51/1034      | 37 more preemptions; same pattern — free hits 0, preempt, recover ~30 blocks, consumed again        |
| T+78–96s  | Quiet gap — batch draining                 | recovering     | Fewer active requests, pressure eases temporarily                                                   |
| T+96–115s | Preemption spiral (wave 3)                 | 0–51/1034      | 18 final preemptions; tokens_lost up to 688 per eviction                                            |
| T+120.3s  | Last event — final FREE                    | 1034/1034      | All requests completed or timed out                                                                 |

---

## Key Numbers

| Metric                                 | Value       |
| -------------------------------------- | ----------- |
| Total preemptions                      | 89          |
| Total ALLOC_FAIL events                | 1,398       |
| Time to first preemption               | 22.4s       |
| Total run duration                     | 120.3s      |
| Requests timed out (120s)              | 11 of 80    |
| Latency range (completed)              | 8.7s – 120s |
| Unique requests affected by preemption | ~20 of 80   |

---

## Preemption Spiral Examples

**Request `9474796f` — preempted 6 times:**

| Preemption # | Elapsed | Tokens Lost | Observation                                    |
| ------------ | ------- | ----------- | ---------------------------------------------- |
| 1            | T+38.0s | 45          | First eviction, barely started decode          |
| 2            | T+39.5s | 73          | Re-prefilled, evicted again 1.5s later         |
| 3            | T+53.4s | 403         | Made real progress, all discarded              |
| 4            | T+60.9s | 411         | Slightly further, evicted again                |
| 5            | T+68.1s | 557         | Over half of max_tokens generated, thrown away |
| 6            | T+72.5s | 667         | Nearly complete output, evicted and restarted  |

This request generated **2,156 tokens of decode work** that was discarded across 6 preemptions. At ~1,900 useful output tokens, the wasted compute exceeded what was needed to finish the request.

**Request `83201e9f` — preempted 4 times:**

| Preemption # | Elapsed | Tokens Lost |
| ------------ | ------- | ----------- |
| 1            | T+27.1s | 688         |
| 2            | T+27.9s | 704         |
| 3            | T+29.7s | 745         |
| 4            | T+31.1s | 769         |

This one entered late with a long decode history, got preempted 4 times in 4 seconds — each time re-prefilling, generating a few more tokens, then immediately evicted again. Classic thrashing.

---

## Anchor Note

Within seconds of the load hitting, free blocks dropped to 0/1034 and stayed there. The system spent 22 seconds filling the pool, then the next 98 seconds thrashing — the scheduler continuously evicting requests to make room, only to have those freed blocks immediately consumed by other waiting requests that would themselves be evicted moments later. The GPU was generating tokens fine. Compute was never the bottleneck. The bottleneck was 1,034 blocks of HBM trying to serve 32 concurrent long-decode requests that each needed ~60 blocks to finish.

The preemption spiral is the key observation: request `9474796f` was preempted 6 times, throwing away 2,156 tokens of completed work. Each recompute consumed prefill budget without advancing the overall queue. The more requests got preempted, the more recompute flooded the system, the less forward progress was made — a positive feedback loop. 1,398 ALLOC_FAIL events means the scheduler tried and failed to promote WAITING requests roughly 12 times per second for the entire run.

Watching it felt like watching a traffic jam form — everything looks fine, then suddenly nothing moves, and the more cars pile up the worse it gets.

# Block 2

Find the natural capacity boundary. You're sweeping concurrency to discover where the cliff is — the point at which the system transitions from healthy to
broken. That means you need parameters where low concurrency works fine and high concurrency doesn't.

If you keep --gpu-memory-utilization 0.45 (1,034 blocks), even moderate concurrency will hit the cliff immediately. The curve won't have a flat region — it'll be a cliff from
the start. That makes a useless graph.

You want to go back to a realistic utilization — the syllabus says 0.90 after Block 1's reset. With 0.90, you get a much larger block pool (~12,000+ blocks on T4). Then:

- At concurrency 1–4: flat TTFT, plenty of blocks
- At concurrency 8–12: slight rise, utilization climbing
- At some point (maybe 16, 20, or higher): cliff — blocks exhaust, preemption fires, TTFT spikes

The workload (short prompt + long decode) can stay the same — that pattern holds blocks longer and is more realistic for finding the memory cliff. But max_tokens might need
adjusting. With ~12,000 blocks and each request needing ~120 blocks for 1,900 tokens, you could fit ~100 concurrent requests before exhaustion. The cliff might be well beyond
concurrency 20.

## Server settings

I decided to make the requests longer instead of higher concurrency

Changes from Block 1:

- 0.90 utilization (up from 0.45) — realistic pool size, ~8,000–12,000 blocks
- 4096 max model len (up from 2048) — room for 3,500 token completions
- 64 max num seqs (up from 32) — ensures the concurrency gate never fires before block exhaustion across our entire sweep range (max 56)

If warmup OOMs at 64, fall back to --max-num-seqs 56 (still above our max sweep level).

```
VLLM_TARGET_DEVICE=cuda  python3 -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen2.5-3B-Instruct     --gpu-memory-utilization 0.90     --max-model-len 4096     --max-num-seqs 64     --dtype half     2>&1 | tee /tmp/vllm-block2.log
```

## Section 2: Client Driver Script

The script does:

1. Sweeps concurrency levels: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56
2. At each level, sends N requests simultaneously (N = concurrency level)
3. Each request uses "stream": true — measures time from request send to first SSE data: chunk (true TTFT)
4. Prompt is short (~15 tokens), max_tokens=3500
5. Collects all TTFTs per level, reports mean/min/max/p99
6. Prints a summary table at the end, plus a simple ASCII chart
7. Waits 5 seconds between levels to let the server drain and stabilize

```
((venv-vllm) ) [ssm-user@ip-10-99-0-199 tmp]$ python3 -u ttft_sweep.py 2>&1 | tee /tmp/ttft_sweep_results.log
TTFT vs Concurrency Sweep
Levels: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56]
max_tokens=3500, drain_delay=5s
Server: http://localhost:8000

==================================================
Concurrency level: 1
==================================================
  [c=1 req=0] TTFT=43ms
  >> mean=43ms  min=43ms  max=43ms
  Draining 5s before next level...

==================================================
Concurrency level: 2
==================================================
  [c=2 req=1] TTFT=41ms
  [c=2 req=0] TTFT=72ms
  >> mean=57ms  min=41ms  max=72ms
  Draining 5s before next level...

==================================================
Concurrency level: 4
==================================================
  [c=4 req=2] TTFT=43ms
  [c=4 req=0] TTFT=81ms
  [c=4 req=1] TTFT=81ms
  [c=4 req=3] TTFT=81ms
  >> mean=72ms  min=43ms  max=81ms
  Draining 5s before next level...

==================================================
Concurrency level: 8
==================================================
  [c=8 req=2] TTFT=46ms
  [c=8 req=4] TTFT=84ms
  [c=8 req=1] TTFT=85ms
  [c=8 req=6] TTFT=84ms
  [c=8 req=3] TTFT=85ms
  [c=8 req=7] TTFT=85ms
  [c=8 req=5] TTFT=85ms
  [c=8 req=0] TTFT=85ms
  >> mean=80ms  min=46ms  max=85ms
  Draining 5s before next level...

==================================================
Concurrency level: 16
==================================================
  [c=16 req=3] TTFT=63ms
  [c=16 req=10] TTFT=102ms
  [c=16 req=9] TTFT=103ms
  [c=16 req=4] TTFT=103ms
  [c=16 req=6] TTFT=103ms
  [c=16 req=11] TTFT=103ms
  [c=16 req=14] TTFT=103ms
  [c=16 req=0] TTFT=104ms
  [c=16 req=13] TTFT=103ms
  [c=16 req=1] TTFT=104ms
  [c=16 req=8] TTFT=104ms
  [c=16 req=7] TTFT=104ms
  [c=16 req=2] TTFT=104ms
  [c=16 req=12] TTFT=103ms
  [c=16 req=15] TTFT=103ms
  [c=16 req=5] TTFT=104ms
  >> mean=101ms  min=63ms  max=104ms
  Draining 5s before next level...

==================================================
Concurrency level: 24
==================================================
  [c=24 req=19] TTFT=66ms
  [c=24 req=6] TTFT=145ms
  [c=24 req=14] TTFT=145ms
  [c=24 req=5] TTFT=146ms
  [c=24 req=9] TTFT=145ms
  [c=24 req=12] TTFT=145ms
  [c=24 req=2] TTFT=146ms
  [c=24 req=10] TTFT=145ms
  [c=24 req=18] TTFT=145ms
  [c=24 req=22] TTFT=145ms
  [c=24 req=15] TTFT=145ms
  [c=24 req=0] TTFT=147ms
  [c=24 req=16] TTFT=145ms
  [c=24 req=4] TTFT=146ms
  [c=24 req=7] TTFT=146ms
  [c=24 req=17] TTFT=145ms
  [c=24 req=21] TTFT=145ms
  [c=24 req=8] TTFT=146ms
  [c=24 req=23] TTFT=145ms
  [c=24 req=11] TTFT=146ms
  [c=24 req=1] TTFT=146ms
  [c=24 req=20] TTFT=145ms
  [c=24 req=13] TTFT=146ms
  [c=24 req=3] TTFT=146ms
  >> mean=142ms  min=66ms  max=147ms
  Draining 5s before next level...

==================================================
Concurrency level: 32
==================================================
  [c=32 req=19] TTFT=81ms
  [c=32 req=5] TTFT=159ms
  [c=32 req=20] TTFT=158ms
  [c=32 req=29] TTFT=158ms
  [c=32 req=23] TTFT=158ms
  [c=32 req=17] TTFT=158ms
  [c=32 req=7] TTFT=159ms
  [c=32 req=10] TTFT=159ms
  [c=32 req=3] TTFT=159ms
  [c=32 req=25] TTFT=158ms
  [c=32 req=30] TTFT=158ms
  [c=32 req=15] TTFT=159ms
  [c=32 req=31] TTFT=158ms
  [c=32 req=1] TTFT=160ms
  [c=32 req=14] TTFT=159ms
  [c=32 req=11] TTFT=159ms
  [c=32 req=2] TTFT=160ms
  [c=32 req=16] TTFT=159ms
  [c=32 req=21] TTFT=159ms
  [c=32 req=26] TTFT=159ms
  [c=32 req=6] TTFT=160ms
  [c=32 req=0] TTFT=160ms
  [c=32 req=28] TTFT=159ms
  [c=32 req=8] TTFT=160ms
  [c=32 req=9] TTFT=160ms
  [c=32 req=22] TTFT=159ms
  [c=32 req=13] TTFT=160ms
  [c=32 req=12] TTFT=160ms
  [c=32 req=4] TTFT=160ms
  [c=32 req=18] TTFT=159ms
  [c=32 req=24] TTFT=159ms
  [c=32 req=27] TTFT=159ms
  >> mean=157ms  min=81ms  max=160ms
  Draining 5s before next level...

==================================================
Concurrency level: 40
==================================================
  [c=40 req=34] TTFT=81ms
  [c=40 req=4] TTFT=181ms
  [c=40 req=13] TTFT=181ms
  [c=40 req=25] TTFT=181ms
  [c=40 req=35] TTFT=180ms
  [c=40 req=18] TTFT=182ms
  [c=40 req=12] TTFT=182ms
  [c=40 req=15] TTFT=182ms
  [c=40 req=16] TTFT=182ms
  [c=40 req=9] TTFT=183ms
  [c=40 req=8] TTFT=184ms
  [c=40 req=5] TTFT=184ms
  [c=40 req=31] TTFT=182ms
  [c=40 req=24] TTFT=183ms
  [c=40 req=17] TTFT=184ms
  [c=40 req=6] TTFT=185ms
  [c=40 req=19] TTFT=184ms
  [c=40 req=38] TTFT=183ms
  [c=40 req=33] TTFT=183ms
  [c=40 req=32] TTFT=184ms
  [c=40 req=26] TTFT=185ms
  [c=40 req=3] TTFT=186ms
  [c=40 req=1] TTFT=187ms
  [c=40 req=30] TTFT=185ms
  [c=40 req=29] TTFT=186ms
  [c=40 req=20] TTFT=186ms
  [c=40 req=23] TTFT=186ms
  [c=40 req=11] TTFT=187ms
  [c=40 req=39] TTFT=185ms
  [c=40 req=10] TTFT=188ms
  [c=40 req=2] TTFT=188ms
  [c=40 req=27] TTFT=187ms
  [c=40 req=21] TTFT=188ms
  [c=40 req=22] TTFT=188ms
  [c=40 req=14] TTFT=188ms
  [c=40 req=7] TTFT=189ms
  [c=40 req=36] TTFT=187ms
  [c=40 req=0] TTFT=190ms
  [c=40 req=28] TTFT=189ms
  [c=40 req=37] TTFT=188ms
  >> mean=182ms  min=81ms  max=190ms
  Draining 5s before next level...

==================================================
Concurrency level: 48
==================================================
  [c=48 req=34] TTFT=91ms
  [c=48 req=27] TTFT=210ms
  [c=48 req=40] TTFT=210ms
  [c=48 req=26] TTFT=210ms
  [c=48 req=1] TTFT=212ms
  [c=48 req=11] TTFT=211ms
  [c=48 req=44] TTFT=210ms
  [c=48 req=30] TTFT=210ms
  [c=48 req=28] TTFT=211ms
  [c=48 req=38] TTFT=210ms
  [c=48 req=6] TTFT=212ms
  [c=48 req=45] TTFT=210ms
  [c=48 req=32] TTFT=211ms
  [c=48 req=37] TTFT=210ms
  [c=48 req=36] TTFT=210ms
  [c=48 req=14] TTFT=212ms
  [c=48 req=31] TTFT=211ms
  [c=48 req=43] TTFT=210ms
  [c=48 req=42] TTFT=210ms
  [c=48 req=25] TTFT=211ms
  [c=48 req=20] TTFT=212ms
  [c=48 req=7] TTFT=212ms
  [c=48 req=17] TTFT=212ms
  [c=48 req=10] TTFT=212ms
  [c=48 req=39] TTFT=211ms
  [c=48 req=24] TTFT=212ms
  [c=48 req=13] TTFT=212ms
  [c=48 req=47] TTFT=211ms
  [c=48 req=5] TTFT=213ms
  [c=48 req=2] TTFT=213ms
  [c=48 req=18] TTFT=212ms
  [c=48 req=3] TTFT=213ms
  [c=48 req=46] TTFT=211ms
  [c=48 req=35] TTFT=211ms
  [c=48 req=29] TTFT=212ms
  [c=48 req=8] TTFT=213ms
  [c=48 req=9] TTFT=213ms
  [c=48 req=21] TTFT=212ms
  [c=48 req=22] TTFT=212ms
  [c=48 req=4] TTFT=213ms
  [c=48 req=12] TTFT=213ms
  [c=48 req=41] TTFT=211ms
  [c=48 req=15] TTFT=212ms
  [c=48 req=16] TTFT=212ms
  [c=48 req=23] TTFT=212ms
  [c=48 req=33] TTFT=256ms
  [c=48 req=0] TTFT=258ms
  [c=48 req=19] TTFT=257ms
  >> mean=212ms  min=91ms  max=258ms
  Draining 5s before next level...

==================================================
Concurrency level: 56
==================================================
  [c=56 req=34] TTFT=89ms
  [c=56 req=26] TTFT=160ms
  [c=56 req=24] TTFT=160ms
  [c=56 req=21] TTFT=160ms
  [c=56 req=7] TTFT=161ms
  [c=56 req=23] TTFT=160ms
  [c=56 req=9] TTFT=161ms
  [c=56 req=40] TTFT=160ms
  [c=56 req=13] TTFT=161ms
  [c=56 req=37] TTFT=160ms
  [c=56 req=53] TTFT=160ms
  [c=56 req=52] TTFT=160ms
  [c=56 req=12] TTFT=162ms
  [c=56 req=54] TTFT=160ms
  [c=56 req=41] TTFT=161ms
  [c=56 req=25] TTFT=161ms
  [c=56 req=27] TTFT=161ms
  [c=56 req=55] TTFT=160ms
  [c=56 req=4] TTFT=163ms
  [c=56 req=38] TTFT=161ms
  [c=56 req=39] TTFT=161ms
  [c=56 req=36] TTFT=161ms
  [c=56 req=51] TTFT=161ms
  [c=56 req=6] TTFT=163ms
  [c=56 req=50] TTFT=161ms
  [c=56 req=11] TTFT=163ms
  [c=56 req=49] TTFT=161ms
  [c=56 req=8] TTFT=163ms
  [c=56 req=35] TTFT=162ms
  [c=56 req=10] TTFT=163ms
  [c=56 req=2] TTFT=164ms
  [c=56 req=20] TTFT=163ms
  [c=56 req=22] TTFT=163ms
  [c=56 req=28] TTFT=268ms
  [c=56 req=44] TTFT=267ms
  [c=56 req=46] TTFT=267ms
  [c=56 req=42] TTFT=268ms
  [c=56 req=3] TTFT=270ms
  [c=56 req=14] TTFT=270ms
  [c=56 req=1] TTFT=271ms
  [c=56 req=5] TTFT=270ms
  [c=56 req=0] TTFT=271ms
  [c=56 req=19] TTFT=270ms
  [c=56 req=33] TTFT=269ms
  [c=56 req=15] TTFT=270ms
  [c=56 req=17] TTFT=270ms
  [c=56 req=18] TTFT=270ms
  [c=56 req=32] TTFT=269ms
  [c=56 req=16] TTFT=270ms
  [c=56 req=30] TTFT=270ms
  [c=56 req=45] TTFT=269ms
  [c=56 req=29] TTFT=270ms
  [c=56 req=43] TTFT=270ms
  [c=56 req=31] TTFT=270ms
  [c=56 req=47] TTFT=270ms
  [c=56 req=48] TTFT=270ms
  >> mean=204ms  min=89ms  max=271ms
  Draining 5s before next level...

======================================================================
TTFT vs Concurrency — Summary
======================================================================
 Concurrency    N   Mean(ms)    Min(ms)    Max(ms)    P99(ms)
----------------------------------------------------------------------
           1    1         43         43         43         43
           2    2         57         41         72         72
           4    4         72         43         81         81
           8    8         80         46         85         85
          16   16        101         63        104        104
          24   24        142         66        147        147
          32   32        157         81        160        160
          40   40        182         81        190        190
          48   48        212         91        258        258
          56   56        204         89        271        271

======================================================================
TTFT vs Concurrency — Chart (mean TTFT)
======================================================================
  c=  1 | ########                                 | 43ms
  c=  2 | ##########                               | 57ms
  c=  4 | #############                            | 72ms
  c=  8 | ###############                          | 80ms
  c= 16 | ###################                      | 101ms
  c= 24 | ##########################               | 142ms
  c= 32 | #############################            | 157ms
  c= 40 | ##################################       | 182ms
  c= 48 | ######################################## | 212ms
  c= 56 | ######################################   | 204ms
```

## Analysis

Interesting result — and it's not what the syllabus predicted. Here's what the data shows:

The curve is a steady linear slope, not a cliff.

c= 1: 43ms
c= 8: 80ms (~2x)
c=24: 142ms (~3.3x)
c=48: 212ms (~5x)
c=56: 204ms (~4.7x)

TTFT grows linearly with concurrency. No vertical cliff anywhere. And c=56 actually dropped below c=48 in mean TTFT.

Why no cliff? Because TTFT measures the time to the first token — the prefill phase. At that moment, each request has allocated only ~1 block (for a ~15-token prompt). 56
requests x 1 block = 56 blocks out of 12,950. That's 0.4% utilization. Memory pressure is nowhere close.

The memory cliff happens during decode, minutes later, as each request grows to ~220 blocks. But by then, TTFT has already been recorded.

What the slope actually shows is compute contention during prefill, not memory pressure. Evidence:

- At c=48 and c=56, the requests split into two visible waves — a fast batch (~160ms) and a slow batch (~267ms). This means the scheduler processed them in two prefill
  iterations. The second wave waited for the first forward pass to finish.
- The slope is ~3ms per additional concurrent request — consistent with scheduling overhead, not block exhaustion.

To get the cliff the syllabus describes, the experiment needs sustained load — requests already in the system holding blocks before new requests arrive. A burst of
short-prompt requests all arriving simultaneously won't do it because they all get their first token before memory builds up.

Two options to actually capture the cliff:

1. Staggered sustained load — Keep a constant stream of long-decode requests running, then measure TTFT of new requests arriving into an already-loaded system
2. Use long prompts — If each request consumes ~60 blocks at prefill (1000-token prompt), then 56 requests = 3,360 blocks at prefill time, and the memory gate might fire
   before first token

## Q: ❯ Is it because we are terminating each request after first token?

No — look at the code after TTFT is captured:

async for \_ in resp.content:
pass

This drains the entire streaming response. The connection stays open, the server generates all 3,500 tokens per request. The requests are NOT terminated early.

The issue is timing. All N requests arrive at the same instant:

1. T+0ms: All 56 requests hit the server simultaneously
2. T+89ms: First request gets its first token (1 block allocated)
3. T+271ms: Last request gets its first token (still only ~56 blocks total used)
4. T+271ms→T+180s: All 56 decode in parallel, blocks grow from 56 to ~12,000
5. But TTFT was already recorded for all 56 before step 4 even started

There's nobody left to measure. The memory cliff happens during decode, but there are no new requests arriving whose TTFT would be impacted by the existing load.

To capture the cliff, the script needs to keep background load running while measuring TTFT of fresh requests arriving into the loaded system. Something like:

1. Launch 40 long-decode requests as background load (don't measure these)
2. Wait 30 seconds for blocks to fill up
3. Send 1 fresh request and measure its TTFT
4. Repeat at different background load levels

That way the fresh request's TTFT reflects the memory state it's entering, not an empty system. Want me to rewrite the script with this approach?

## What should we be seeing?

```
  TTFT (ms)
      |
   5000|                              *
      |                              *
      |                              *
      |                             *
   1000|                            *
      |                          *
    200|                       *
    100|     * * * * * * * * *
     50| * *
      |________________________________
        1  2  3  4  5  6  8  10 12 16 20
                  concurrency →
```

## Finally!

### Block 2 Client Setup — How We Got Here

v1 (burst, short prompts, large pool): Fired N requests simultaneously with short prompts (~15 tokens) and max_tokens=3500 against --gpu-memory-utilization 0.90 (12,950
blocks). Result: Smooth linear slope from 43ms to 212ms. No cliff. Problem: All N requests prefilled instantly (1 block each = 0.4% utilization at TTFT time). Memory pressure
only built during decode, after TTFT was already recorded. The slope was pure compute contention — bigger batch, slower forward pass.

v2 (background load + probe, large pool): Launched N background long-decode requests, waited 30 seconds for blocks to fill, then sent 3 probe requests to measure TTFT. Result:
Noisy flat curve, 68ms–158ms. No cliff. Problem: Total throughput ~147 tok/s means blocks accumulate at ~9 blocks/second regardless of batch size. After 30 seconds: ~276
blocks consumed out of 12,950 (2%). Memory pressure never materialized. Would need ~23 minutes of settle time per level to fill the pool.

v3 final (burst, long prompts, small pool): The fix required changing three things simultaneously:

1. Long prompts (~200 hex words = 1,203 tokens = 75 blocks for prefill). Now each request consumes significant blocks at prefill time, before TTFT is measured. This is the key insight — the cliff comes from prefill contention, not decode accumulation.

2. Small pool (--gpu-memory-utilization 0.45, 1,021 blocks). Ensures the pool can be exhausted by ~13 concurrent prefills.

3. Short decode (max_tokens=200 = 13 blocks). Keeps total blocks per request at
   ~88, putting the cliff at c=11-12 within our sweep range. Also means each level completes faster.

```
Server: --gpu-memory-utilization 0.45 --max-model-len 4096 --max-num-seqs 64 --dtype half
```

Cliff at c=12→14: 12 × 88 = 1,056 > 1,021 blocks. Requests 13+ can't prefill until earlier requests finish and free blocks. Max TTFT jumps from 671ms to 8,735ms — a 13x spike.

The hex prompt was also tricky — first attempt used 800 words (3,897 tokens) which exceeded max-model-len 4096. Reduced to 200 words × 6-char hex = 1,203 tokens, verified with
the actual Qwen tokenizer on EC2.

### Results

python3 -u /tmp/ttft_sweep_v3.py 2>&1 | tee /tmp/ttft_sweep_v3_results.log

```
❯
  ======================================================================
  TTFT vs Concurrency — Summary
  ======================================================================
   Concurrency    N   Mean(ms)    Min(ms)    Max(ms)    P99(ms)
  ----------------------------------------------------------------------
             1    1        428        428        428        428
             2    2        317        314        319        319
             4    4        462         68        634        634
             6    6        546        300        596        596
             8    8        596        589        598        598
            10   10        563         80        661        661
            12   12        589         93        671        671
            14   14       1440        316       8735       8735
            16   16       2934         98      10135      10135
            18   18       3711        109      10463      10463
            20   20       4954        336      11674      11674

  ======================================================================
  TTFT vs Concurrency — Chart (mean TTFT)
  ======================================================================
    c=  1 | ###                                      | 428ms
    c=  2 | ##                                       | 317ms
    c=  4 | ###                                      | 462ms
    c=  6 | ####                                     | 546ms
    c=  8 | ####                                     | 596ms
    c= 10 | ####                                     | 563ms
    c= 12 | ####                                     | 589ms
    c= 14 | ###########                              | 1440ms
    c= 16 | #######################                  | 2934ms
    c= 18 | #############################            | 3711ms
    c= 20 | ######################################## | 4954ms

  ======================================================================
  TTFT vs Concurrency — Chart (max TTFT)
  ======================================================================
    c=  1 | #                                        | 428ms
    c=  2 | #                                        | 319ms
    c=  4 | ##                                       | 634ms
    c=  6 | ##                                       | 596ms
    c=  8 | ##                                       | 598ms
    c= 10 | ##                                       | 661ms
    c= 12 | ##                                       | 671ms
    c= 14 | #############################            | 8735ms
    c= 16 | ##################################       | 10135ms
    c= 18 | ###################################      | 10463ms
    c= 20 | ######################################## | 11674ms

```

⏺ There's the cliff. Right between c=12 and c=14.

Mean TTFT: 589ms → 1,440ms → 2,934ms → 4,954ms (2.4x → 5x → 8.4x baseline)
Max TTFT: 671ms → 8,735ms → 10,135ms → 11,674ms (13x → 15x → 17x baseline)

The max TTFT chart shows it most clearly — flat at ~600ms through c=12, then vertical jump to 8,735ms at c=14. That's the shape the syllabus described: flat → cliff. No
gradual degradation. The system is fine at 12 concurrent requests and broken at 14.

The math confirms: 1,021 blocks / ~88 blocks per request = 11.6 requests max. At c=12 you're at the edge. At c=14, two requests can't be prefilled and wait ~9 seconds for
blocks to free up.

The worst-case request at c=20 waited 11.7 seconds for its first token — vs 428ms baseline. That's a 27x TTFT spike caused entirely by memory exhaustion, not compute.

## PREFILL CLIFF Explanation

At c=14, the first ~11 requests grab their 75 prefill blocks each (consuming ~825 blocks out of 1,021). Requests 12-14 arrive at the same instant but there aren't enough free
blocks for their 75-block prefill. They sit in WAITING.

They can't get their first token until a running request finishes its entire decode (200 tokens), frees all 88 blocks, and makes room for a new 75-block prefill.

That wait — sitting in WAITING because prefill can't be allocated — is the 8,735ms max TTFT at c=14 vs 671ms at c=12.

The decode blocks (13 per request) make it slightly worse because they reduce headroom — each running request grows from 75 to 88 blocks during decode, eating into what little
free space remains. But the fundamental cliff is: "I need 75 blocks to start and there aren't 75 blocks free."

---

# Block 3 — Continuous Batching (Orca Insight)

## Setup

- **Server:** `--gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 32 --dtype half`
- **Instrumentation:** Patched `vllm/v1/core/sched/scheduler.py` with `[SCHED_STEP]` logging after each `schedule()` call — logs active request IDs (8-char suffixes), new/running/finished counts, total scheduled tokens per iteration
- **Client:** 10 requests staggered 200ms apart, short prompts, `max_tokens=300`
- **Log:** 344 SCHED_STEP events over 19.6 seconds

## Timeline — Batch Composition Changes

| Elapsed | Batch Size | Event                                   | Active Set                       |
| ------- | ---------- | --------------------------------------- | -------------------------------- |
| T+0.0s  | 1          | new=1 (8e2ac2ba arrives)                | [8e2ac2ba]                       |
| T+0.2s  | 2          | new=1 (af2bb8f5 arrives)                | [8e2ac2ba, af2bb8f5]             |
| T+0.4s  | 3          | new=1 (b36da414 arrives)                | +b36da414                        |
| T+0.6s  | 4          | new=1 (96516904 arrives)                | +96516904                        |
| T+0.8s  | 5          | new=1 (953cd0cd arrives)                | +953cd0cd                        |
| T+1.0s  | 6          | new=1 (93a363b0 arrives)                | +93a363b0                        |
| T+1.2s  | 7          | new=1 (bccb5fc7 arrives)                | +bccb5fc7                        |
| T+1.4s  | 8          | new=1 (b7547ac1 arrives)                | +b7547ac1                        |
| T+10.3s | 10         | new=2 (809cc39c, bc3900aa arrive)       | all 10 active                    |
| T+17.3s | 9          | finished=[96516904]                     | first request completes          |
| T+18.3s | 6          | finished=[8e2ac2ba, bc3900aa, af2bb8f5] | three finish in rapid succession |
| T+18.6s | 5          | finished=[b36da414]                     |                                  |
| T+19.0s | 4          | finished=[953cd0cd]                     |                                  |
| T+19.2s | 3          | finished=[93a363b0]                     |                                  |
| T+19.5s | 0          | finished=[bccb5fc7, 809cc39c, b7547ac1] | all done                         |

## Key Numbers

| Metric                             | Value                               |
| ---------------------------------- | ----------------------------------- |
| Total scheduler iterations         | 344                                 |
| Total duration                     | 19.6 seconds                        |
| Iteration rate                     | ~17.5 steps/second (~57ms per step) |
| Unique requests processed          | 10                                  |
| Time to reach full batch (10/10)   | 10.3 seconds                        |
| Time from first finish to all done | 2.3 seconds (T+17.3s → T+19.6s)     |
| Stagger interval observed          | ~200ms (matches client)             |
| Preemptions                        | 0                                   |

## What the Data Shows — The Orca Insight

The batch is **not static**. Every single forward pass, the scheduler decides which requests participate in that iteration's batch. The active set changes continuously:

**Growth phase (T+0s → T+10.3s):** Each new request joins the running batch on the very next scheduler iteration after it arrives. Request `af2bb8f5` arrives at T+0.2s and immediately joins `8e2ac2ba` in the next forward pass. The scheduler doesn't wait for a batch to "fill up" or for existing requests to finish. It just adds the newcomer to whatever batch is already running.

Between arrivals, the scheduler runs multiple iterations with the same batch. For example, between request 2 arriving (T+0.2s) and request 3 arriving (T+0.4s), there are 6 iterations all with `running=2, new=0` — pure decode steps for those two requests.

**Steady state (T+10.3s → T+17.3s):** All 10 requests are active. The scheduler runs ~230 iterations with `running=10` — all ten decoding together in every forward pass.

**Drain phase (T+17.3s → T+19.6s):** Requests finish one at a time and immediately leave the batch. The batch shrinks from 10 → 9 → 8 → 6 → 5 → 4 → 3 → 0 in 2.3 seconds. Each request finishes independently — it doesn't wait for siblings.

## Why This Matters

Traditional batching (pre-Orca) processes a fixed batch and waits for all requests in the batch to finish before accepting new ones. If one request generates 300 tokens and another generates 10, the 10-token request sits idle for the remaining 290 tokens.

Continuous batching means:

- New requests join immediately (no waiting for a "batch window")
- Finished requests leave immediately (no waiting for siblings)
- The batch composition is different on every forward pass
- GPU utilization stays high because slots freed by finished requests are instantly available

In this run, request `96516904` finished at T+17.3s. The very next iteration ran with 9 requests instead of 10. No wasted GPU cycles. No idle slots.

## Anchor Note

The 344 scheduler iterations across 19.6 seconds show continuous batching in action. The batch starts with 1 request, grows to 10 as staggered requests arrive (each joining on the very next iteration after arrival), holds at 10 during steady-state decode, then shrinks back to 0 as requests independently finish and immediately leave. No request waits for any other request.

This is the Orca paper's core contribution: per-iteration scheduling. The batch is not a container that fills and empties — it's a rolling set that the scheduler rebuilds every ~57ms. A request's lifecycle (arrive → prefill → decode → finish) is independent of every other request's lifecycle. The scheduler just runs the union of whoever is currently active.

The contrast with Block 1 is instructive: there, 80 requests overwhelmed 1,034 blocks and triggered preemption spirals. Here, 10 requests with adequate memory show the system working as designed — the scheduler smoothly managing a changing batch with zero preemptions, zero wasted iterations, and requests joining/leaving on their own timelines.

## Appendix: Prefill vs Decode Iteration Latency

Do iterations that include a prefill (new=1) take longer than pure decode iterations (new=0)?

**Prefill iterations** (iteration where a new request joins):

| Batch Size | Delta (ms) | Note                                                        |
| ---------- | ---------- | ----------------------------------------------------------- |
| 1          | 188        | Cold start — first-ever iteration                           |
| 2          | 68         | 1 prefill + 1 decode (second-ever iteration, still warming) |
| 3          | 28         | 1 prefill + 2 decode                                        |
| 4          | 30         | 1 prefill + 3 decode                                        |
| 5          | 29         | 1 prefill + 4 decode                                        |
| 6          | 30         | 1 prefill + 5 decode                                        |
| 7          | 30         | 1 prefill + 6 decode                                        |
| 10         | 14         | 2 prefills + 8 decode (requests 9-10 arrived together)      |

**Decode-only iterations** by batch size:

| Batch Size | Count | Min (ms) | Mean (ms) | Max (ms) |
| ---------- | ----- | -------- | --------- | -------- |
| 2          | 7     | 4        | 25        | 29       |
| 3          | 19    | 28       | 29        | 31       |
| 4          | 12    | 29       | 30        | 31       |
| 5          | 19    | 30       | 30        | 31       |
| 6          | 13    | 30       | 31        | 31       |
| 7          | 5     | 30       | 31        | 32       |
| 9          | 31    | 30       | 31        | 32       |
| 10         | 226   | 17       | 31        | 45       |

**Finding:** At batch=3 and above, prefill iterations (28–30ms) are indistinguishable from decode-only iterations (29–31ms). The batch=2 prefill outlier at 68ms is likely CUDA warmth, not prefill cost.

**Why:** The prompts are short (~9 tokens). A 9-token prefill is computationally equivalent to one decode step — a single forward pass over ~9 tokens, same as generating 1 token for each of 9 running requests. With long prompts (like Block 2's 1,200-token prefill), you'd see a much larger gap because prefill processes all prompt tokens in one pass, making that iteration significantly heavier than a decode step.

---

# Block 4: Mapping Collapse to Scheduler Code

Re-examining the Day 7 scheduler trace (`vllm/v1/core/sched/scheduler.py`) with the collapse data above. Every event maps to a specific line in the unified `schedule()` method.

## 1,398 ALLOC_FAIL events → Phase 2 `break` (line ~266)

Every step, after scheduling RUNNING requests, the scheduler enters Phase 2 and tries to promote WAITING requests. It calls `allocate_slots()` — returns `None` because `free == 0`. The scheduler hits `break` and stops admitting. This happened ~12 times/second for the entire run. Each one is a step where a WAITING request _could have_ started but couldn't get blocks.

## 89 preemptions → Phase 1 preemption loop (lines ~267-277)

This fires when a RUNNING request needs a new block for its next decode token and `allocate_slots()` returns `None`. The scheduler enters the `while` loop, popping the last request from `self.running` and calling `_preempt_request()`. That victim's blocks are freed, `num_computed_tokens` reset to 0, and it gets prepended to `self.waiting`.

## The thrashing spiral → Phase 1 + Phase 2 interaction

The feedback loop in scheduler terms:

```
Step N:
  Phase 1: Request A needs 1 decode block → free=0 → preempt Request B
           → B.num_computed_tokens = 0, B goes to front of waiting queue
           → A gets B's freed blocks, continues decoding
  Phase 2: skipped (preemptions occurred this step)

Step N+1:
  Phase 1: A continues (already running)
  Phase 2: B is at front of waiting → allocate_slots() → gets blocks → RUNNING
           → B must re-prefill from scratch (all prior decode work lost)

Step N+2:
  Phase 1: B is now running (re-prefilling, consuming blocks fast)
           → Request C needs a decode block → free=0 → preempt B again
           → B.num_computed_tokens = 0 AGAIN
           → This is why request 9474796f was preempted 6 times
```

Key detail: **Phase 2 doesn't fire if preemptions occurred in Phase 1.** But preempted requests go to the _front_ of `self.waiting` via `prepend_request()`. So next step (no preemptions that step), they get promoted in Phase 2 — only to be preempted again in Phase 1 the step after that.

## The 51-block oscillation → one victim's worth of blocks

When `free` oscillates between 0 and ~51, that's roughly one request's worth of blocks. Preempt one request → free ~51 blocks → immediately consumed by the next request's prefill → back to 0. The `blocks_used` pattern of `0/1034 ↔ 51/1034` for 78 seconds is the scheduler executing this cycle every ~150ms.

## 11 request timeouts → starvation from repeated preemption

Each preemption resets `num_computed_tokens = 0`. A request that was 90% done (like `9474796f` at 667/1900 tokens) must restart its entire decode from scratch. But it re-enters the same pressure environment and gets preempted again. After 6 rounds, the 120s timeout fires — the request generated 2,156 tokens of total work but never finished because it kept losing its KV cache.

## What this means for capacity planning

The scheduler has no "forward progress guarantee." Under sustained pressure, the FCFS preemption policy (`self.running.pop()`) can repeatedly evict the same request. The system does more total compute (re-prefills) while making less forward progress (finished requests). This is the positive feedback loop: preemption → recompute → more block pressure → more preemption.

The fix isn't in the scheduler — it's in admission control. The cliff in Block 2 (c=12→14) shows that keeping concurrency below the memory boundary eliminates preemption entirely. Zero preemptions = zero wasted compute = predictable TTFT.

# AFTERNOON

## Walk me through what happens when vLLM runs out of KV cache blocks under current load. Must cover detection, preemption, TTFT spike, cascade

### ANS:

- for RUNNING requests, if KVcache runs out, a victim RUNNING request is preempted and moved to the head of the WAITING queue. It's existing kvcache is freed (compute wasted)
- Only when NO preemption ocurred, WAITING requests are considered for scheduling. To do so, blocks are attempted to be allocated for all slots of the request (i.e. the entire prompt). If successful
  the WAITING request is scheduled to RUNNING state. Otherwise, nothing else is scheduled.
- Thrashing happens when the system is out of memory and a preempted WAITING->RUNNING may be scheduled but then another request is promptly preempted RUNNING->WAITING
- TTFT spike happens when requests cannot be serviced quickly, i.e. they sit in the WAITING queue for a long time because of insufficient memory. This represents a cascade situation where the system
  is completely out of memory and most RUNNING requests cannot progress for lack of KVcache memory.

⏺ That's a strong answer. A few refinements:

Detection: You nailed it — allocate_slots() returns None for a RUNNING request that needs its next decode block. One clarification: WAITING requests don't allocate "all slots for the entire prompt" upfront.
V1 supports chunked prefill, so a WAITING request may receive fewer tokens than its full prompt in one step. It just gets as many as the token budget allows.

Preemption: Accurate. One detail to add: the victim selection is self.running.pop() under FCFS — that's the last (lowest-priority) request. And critically, num_computed_tokens is reset to 0, meaning the
request must re-prefill from scratch. All prior decode work is discarded, not just paused.

Thrashing: Correct. The specific mechanism worth adding: Phase 2 (WAITING promotion) is skipped when preemptions occurred in Phase 1. But the preempted request goes to the front of self.waiting via
prepend_request(). So it gets promoted the next step — only to trigger another preemption the step after. The 0↔51 block oscillation you saw is this exact cycle.

TTFT spike / cascade: Your description covers the symptom but could be sharper on the mechanism. The TTFT spike isn't just from sitting in WAITING — it's compounded by recompute cost. A preempted request
that had 667 decoded tokens must re-prefill and re-decode all of them. Meanwhile that recompute consumes blocks and token budget, crowding out other requests. That's the positive feedback loop: more
preemption → more recompute → more block pressure → more preemption. The system does more total compute while making less forward progress.

Overall a solid staff-level answer. Ready to start the architecture document?

### I don't understand the positive feedback loop. Preemption frees N KV blocks so when the preempted request is recomputed it won't need more than N KV blocks. But yes, there is wasted compute so we are

indeed making less forward COMPUTE (but not memory right)?

⏺ You're right to push back — your reasoning is correct. Preemption is memory-neutral in the short term: freeing N blocks from the victim, then the victim consumes at most N blocks when rescheduled. The pool
doesn't shrink.

The feedback loop operates through compute, not memory directly:

1. Request B is preempted → frees 51 blocks
2. Next step, B is re-admitted from front of waiting queue
3. B must re-prefill → consumes a large chunk of the token budget this step (e.g., 100 tokens for chunked prefill)
4. That's 100 tokens of budget that could have gone to 100 other requests' decode steps
5. Those other requests make slightly less decode progress → take longer to finish
6. Taking longer to finish → they hold their blocks longer
7. Blocks held longer → fewer blocks free up per unit time → more preemption pressure

So the loop is: preemption → wasted compute on re-prefill → fewer completions per unit time → blocks freed more slowly → more preemptions.

But honestly, "positive feedback loop" overstates it. The real problem is simpler: the system is overcommitted. 32 requests × ~60 blocks each = 1,920 blocks needed, but only 1,034 available. At most ~17 can
be fully resident. Preemption doesn't solve overcommitment — it just churns through requests, wasting compute without reducing demand. The system isn't getting worse per se; it's stuck in a bad steady
state where it does work that gets thrown away.

Good catch. The memory is zero-sum; the waste is compute.

## FINAL ARTIFACT PREP

I'm going to let you write the details for the final artifact but I'm going to practice the answers with you. I will not quote exactly file names and class names as that is minor. I will focus on the main concepts:

### 1. Request lifecycle

- Api server receives endpoint requests, tokenizes and creates SamplingParams.
  - engine.generate() is called which places request into ZeroMQ
  - Then creates a RequestOutput to track response and yields upon return
- In the separate process, the scheduler loop processes the queue
- scheduler loop operates on iterations with orca continuous batching, handling both requests both in prefill and decode
  - RUNNING requests in decode are prioritized over prefill
  - scheduler delegates to KVcacheManager by attempting to allocate(). failure to allocate triggers different behavior/code paths
  - on each iteration, a forward pass is delegated to a GPUWOrker
- GPUWorker
  - executes forward pass, prioritizing decode above prefill, handing off a CUDAGraph into GPU land
  - when the async call returns, logits are sampled and the token chosen
  - decode tokens are returned, potentially EOS tokens
- Back in scheduler land
  - If EOS tokens, max len, client cancellations or any other FINISH state triggers are invoked, the scheduler returns terminates the request by releasing all KV memory
  - Else the token is shipped
- Back in API server, the RequestOutput is shipped back and the token streamed back to the client via SSE

#### Strong answer. Here's my feedback on each section:

API server → generate(): Accurate. One nuance: generate() doesn't create the RequestOutput — it creates an async generator and yields RequestOutputs as they arrive from the OutputProcessor. The
OutputProcessor handles detokenization on the API server side. Minor distinction but shows you understand the async flow.

Scheduler loop: Correct. One refinement on the priority statement: it's not "decode over prefill" exactly — it's RUNNING requests over WAITING requests (Phase 1 before Phase 2). A RUNNING request could
still be in chunked prefill (not yet finished its prompt). The scheduler doesn't distinguish prefill vs decode — it just sees num_computed_tokens advancing toward num_tokens. The "decode first" ordering
happens later, inside the GPUModelRunner's prepare_inputs(), as an optimization for tensor layout.

GPUWorker: Mostly right. The "prioritizing decode above prefill" is a prepare_inputs() sort optimization (fewer tokens first), not a scheduling priority. The GPU sees one flat batch — it doesn't know which
tokens are prefill vs decode. Also worth noting the two-phase design: execute_model() runs the forward pass and stores hidden states, then sample_tokens() is a separate call that computes logits and
samples. Grammar bitmask computation overlaps with the GPU forward pass.

Finish handling: Accurate. The update_from_output() step checks all finish conditions and calls \_free_request() → kv_cache_manager.free().

SSE streaming: Correct. EngineCoreOutputs → ZMQ → OutputProcessor (detokenization) → async generator → SSE to client.

## Final Deliverable

The next item is the Week 2 Deliverable: Annotated vLLM Architecture Diagram — 6 required sections. The good news is you've already built most of the content across day7, day8, and day9 work files. This is
about assembling it into a single self-contained document.

### The 6 sections are:

1. Request Lifecycle — full trace with V1 file paths and function names
2. Request State Machine (V1) — the diagram + transition details
3. Preemption Cost Annotation — recompute cost model, tie to TTFT spike
4. Continuous Batching Path — when blocks are freed, when new requests can use them
5. Instrumentation Hook Points — Day 8 patch locations, fields, example output
6. Mini Collapse Observation — setup, timeline, cliff graph, capacity planning insight

Want me to draft the full document? I'll pull from your existing work files and write it as a single self-contained .md file that reads as if an interviewer will see it without you present.

❯ I'm going to let you write the details for the final artifact but I'm going to practice the answers with you. I will not quote exactly file names and class names as that is minor. I will focus on the
main concepts:

### 1. Request lifecycle

- Api server receives endpoint requests, tokenizes and creates SamplingParams.
  - engine.generate() is called which places request into ZeroMQ
  - Then creates a RequestOutput to track response and yields upon return
- In the separate process, the scheduler loop processes the queue
- scheduler loop operates on iterations with orca continuous batching, handling both requests both in prefill and decode
  - RUNNING requests in decode are prioritized over prefill
  - scheduler delegates to KVcacheManager by attempting to allocate(). failure to allocate triggers different behavior/code paths
  - on each iteration, a forward pass is delegated to a GPUWOrker
- GPUWorker
  - executes forward pass, prioritizing decode above prefill, handing off a CUDAGraph into GPU land
  - when the async call returns, logits are sampled and the token chosen
  - decode tokens are returned, potentially EOS tokens
- Back in scheduler land
  - If EOS tokens, max len, client cancellations or any other FINISH state triggers are invoked, the scheduler returns terminates the request by releasing all KV memory
  - Else the token is shipped
- Back in API server, the RequestOutput is shipped back and the token streamed back to the client via SSE

#### Strong answer. Here's my feedback on each section:

API server → generate(): Accurate. One nuance: generate() doesn't create the RequestOutput — it creates an async generator and yields RequestOutputs as they arrive from the OutputProcessor. The
OutputProcessor handles detokenization on the API server side. Minor distinction but shows you understand the async flow.

Scheduler loop: Correct. One refinement on the priority statement: it's not "decode over prefill" exactly — it's RUNNING requests over WAITING requests (Phase 1 before Phase 2). A RUNNING request could
still be in chunked prefill (not yet finished its prompt). The scheduler doesn't distinguish prefill vs decode — it just sees num_computed_tokens advancing toward num_tokens. The "decode first" ordering
happens later, inside the GPUModelRunner's prepare_inputs(), as an optimization for tensor layout.

GPUWorker: Mostly right. The "prioritizing decode above prefill" is a prepare_inputs() sort optimization (fewer tokens first), not a scheduling priority. The GPU sees one flat batch — it doesn't know which
tokens are prefill vs decode. Also worth noting the two-phase design: execute_model() runs the forward pass and stores hidden states, then sample_tokens() is a separate call that computes logits and
samples. Grammar bitmask computation overlaps with the GPU forward pass.

Finish handling: Accurate. The update_from_output() step checks all finish conditions and calls \_free_request() → kv_cache_manager.free().

SSE streaming: Correct. EngineCoreOutputs → ZMQ → OutputProcessor (detokenization) → async generator → SSE to client.

Ready for Section 2 (State Machine)?

### 2: State machine

3 states
WAITING - new requests or preempted requests. No KV memory allocated.
RUNNING - requests in chunked prefill or decode. Each has a KV cache associated.
FINISHED - requests decoded with EOS, reached max tokens, canceled by user etc.
Transitions:
WAITING -> RUNNING : Request at the head of the WAIT queue is promoted to prefill. Only if enough KV can be allocated for chunked prefill. Other critiria must be met such as max token budget, max
requests.
RUNNING -> WAITING : preempted when a RUNNING request needed memory to progress and the preempted was chosen as a victim.
RUNNING -> FINISHED : when one of 6 or more exit criteria area met

#### Solid and accurate. A few additions:

WAITING state: You could mention the sub-states — WAITING_FOR_FSM (structured output grammar compiling), WAITING_FOR_REMOTE_KVS (P/D disaggregation), WAITING_FOR_STREAMING_REQ (streaming input paused).
These go into a skipped_waiting queue, not the main waiting queue. Not critical for the interview but shows depth.

RUNNING → WAITING: Worth emphasizing the cost: num_computed_tokens resets to 0, meaning all prior decode work is discarded. The request re-prefills from scratch when rescheduled. This is the key V1 design
tradeoff — no swap-to-CPU path means simpler code but higher preemption cost (proportional to prompt length + generated tokens).

Victim selection: You said "chosen as a victim" — worth knowing the mechanism: self.running.pop() under FCFS (last/lowest-priority request), and the victim is prepended to the front of self.waiting via
prepend_request() for priority resumption.

WAITING → WAITING (implicit): You didn't mention the non-transition: when allocate_slots() returns None for a WAITING request, the scheduler just breaks — the request stays in WAITING. This is the dominant
steady-state under pressure (the 1,398 ALLOC_FAIL events from your collapse).

FINISHED states: Good that you said "6 or more." The six are: FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED, FINISHED_ERROR, FINISHED_REPETITION.

Ready for Section 3 (Preemption Cost)?

### Section 3: Preemption Cost Annotation — recompute cost model, tie to TTFT spike

TTFT spike happens when system is memory starved. RUNNING requests need preemption of other requests to progress. SPike happens because incoming requests pile up because overall rate of service cannot
meet the incoming request rate. This is compounded by thrashing when preemption keeps happening and compute is wasted so existing service rate actually drops.
Preempted requests throw away the entire KV cache for the request meaning all computed work is wasted. num computed tokens is reset to 0 for that request.

#### Correct on all points. One thing to add for interview completeness:

Quantify the cost. You described the mechanism but an interviewer would want to hear the math. A preempted request with a 1,000-token prompt that had generated 500 decode tokens must re-prefill all 1,000
prompt tokens when rescheduled. The recompute cost is proportional to prompt length, and then it must re-generate those 500 tokens again. Your collapse data shows this concretely: request 9474796f was
preempted 6 times, discarding 2,156 tokens of completed decode work — more than the 1,900 tokens needed to actually finish the request.

The TTFT distinction. You described system-level throughput degradation, which is right. But TTFT specifically is the time from request arrival to first token. For a new request arriving during collapse,
TTFT spikes because it sits in WAITING while allocate_slots() returns None every step (your 1,398 ALLOC_FAIL events = ~12 rejections/second). For a preempted request, the effective TTFT restarts — it must
wait in WAITING again, then re-prefill, before producing its "first" token again. Your Block 2 data showed this cliff: TTFT went from 428ms at c=1 to 4,954ms at c=20 — a 12x spike once the memory boundary
was crossed at c=14.

Prefix caching as mitigation. Worth a sentence: with prefix caching enabled (V1 default), a preempted request's prefix blocks may still be cached if no other request evicted them. Only the non-cached suffix
needs recomputation. This is why V1 chose recompute over swap — prefix caching makes the common case cheaper than swapping the full KV cache to CPU and back.

Ready for Section 4 (Continuous Batching)?

### Section 4: Continous Batching

Continuous Batching Path — when blocks are freed, when new requests can use them
On every iteration, the scheduler considers all RUNNING and WAITING requests, prioritizing RUNNING requests before WAITING.
RUNNING requests require more KV to progress and likewise for WAITING requests.
When a request in decode ends on EOS or maxlen or some other criteria, all KV for that request is freed.
Secondly, when a RUNNING request is victimized and preempted, it's KV is entirely freed.
In both these cases, other requests can use these blocks on attempted allocate()
New requests pile up into the WAITING queue and are promoted to RUNNING as described above.
Note that promotion of WAITING requests only occurs if no preemption was needed.
Presumably this is a heuristic to reduce thrashing when memory pressure is high (needing preemption)

#### Excellent — and that last observation about the "no promotion after preemption" heuristic is sharp. You're right that it's a thrashing reduction mechanism. If preemptions occurred, admitting new WAITING

requests would just add more block pressure and likely trigger more preemptions immediately.

Two timing details to sharpen:

When freed blocks become available. Blocks freed by finished requests in update_from_output() (end of step N) are available for schedule() at the start of step N+1 — the next iteration. Blocks freed by
preemption during schedule() Phase 1 are available immediately within the same schedule() call — either for the request that triggered the preemption (retry allocation) or for other RUNNING requests later
in Phase 1. This is a subtle but important distinction: preemption-freed blocks can be used same-step, finish-freed blocks require one step delay.

Why this makes continuous batching efficient. The key sentence for an interviewer: the GPU is never idle waiting for a full batch to drain. Your Block 3 data showed this — request 96516904 finished at
T+17.3s, and the very next iteration (57ms later) ran with 9 requests instead of 10. No wasted cycles. Static batching would have held the GPU idle for those freed slots until all 10 requests finished.

Ready for Section 5 (Instrumentation Hook Points)?

### Section 5) Instrumentation Hook Points — Day 8 patch locations, fields, example output

Locations:
BLOCK_PREEMPT: in scheduler:\_preempt_request()
BLOCK_FREE: in scheudler:\_free()
BLOCK_ALLOC: scheduler: after every invocation of allocate_new_blocks(), gated by non-zero allocations
BLOCK_ALLOC_FAIL: scheduler: if cannot allocate blocks in allocate_slots(); in kv cache manager: when no blocks are allocated.

Fields: Operation, time(epoch), request-id, num blocks to allocate, free blocks, total gpu blocks

Example output: (EngineCore_DP0 pid=9357) INFO 03-12 22:32:50 [kv_cache_manager.py:371] [BLOCK_ALLOC] ts=1773354770408 req=chatcmpl-9b0488459b278b31-b602974d alloc=2 free=30276/30278

#### Good recall. A few corrections:

File locations: BLOCK_ALLOC, BLOCK_ALLOC_FAIL, and BLOCK_FREE are all in kv_cache_manager.py, not the scheduler. Only BLOCK_PREEMPT is in scheduler.py. This makes sense architecturally — the scheduler
handles request lifecycle decisions (preempt), while the KV cache manager handles block operations (alloc, free).

BLOCK_PREEMPT logging order: Worth noting that PREEMPT is logged before free() is called, so the log captures computed_tokens_lost and the free count at the moment of the preemption decision — showing the
pressure state that caused it. PREEMPT also logs the preemption count (how many times this specific request has been preempted), which is how you tracked request 9474796f being preempted 6 times.

BLOCK_FREE field difference: FREE logs freed=N (how many blocks were released), not alloc=N. And the free count shown is the post-free count — so you can verify the math: pre-free count + freed = post-free
count.

The null block detail: Total GPU blocks is reported as num_gpu_blocks - 1 because block 0 is permanently reserved as a null block (sentinel for padding positions). This is why your logs show 30278 as total
instead of 30279.

Ready for Section 6 (Mini Collapse Observation)?

### Section 6: Mini Collapse Observation — setup, timeline, cliff graph, capacity planning insight

Cliff graph -- level -> slow ramp -> sharp vertical cliff. Y axis is TTFT. X axis is concurrency
Setup -- must constrain memory to hit KV cache memory exhaustion. Set gpu-memory-utilization 0.45. Requests must be big enough 2048 tokens. Concurrency must be sweeped to high enough 50+
transactions.
Capacity planning insight: An estimate of KV cache size \* expected concurrency (batch size) is needed for steady state system such that memory boundness is avoided. Basically we want the system to
have enough memory so that it reaches the compute bound steady state. KV cache memory must be enough i.e. GPU RAM minus model weights and other required running costs (activations etc)

#### Strong conceptual grasp. A few corrections on specifics:

Setup: Your Block 1 collapse used --gpu-memory-utilization 0.45 with 80 concurrent requests and max_tokens=1900 — that's correct. But your Block 2 cliff graph used different settings:
--gpu-memory-utilization 0.90 (realistic pool) with 200-word hex prompts (~1,203 tokens = 75 blocks per prefill) and max_tokens=200. The cliff graph needs a large enough pool to show the flat region before
the cliff — at 0.45, it's a cliff from the start with no flat baseline.

Cliff numbers: Your actual data showed the cliff at c=12→14, not 50+. At c=12, ~900/1,021 blocks consumed by prefills — slight TTFT rise to 589ms. At c=14, request 13+ can't get 75 blocks and waits 8.7
seconds — TTFT jumps to 1,440ms (a 13x spike). By c=20, TTFT hit 4,954ms. The cliff is sharp because it's a binary transition: either you get blocks or you don't. There's no graceful degradation.

Capacity planning formula: Your insight is right. More precisely from your data: the cliff occurs at concurrency × blocks_per_request ≈ total_blocks. In your case: 12 × 75 = 900 blocks, pool = 1,021. At 14
× 75 = 1,050 > 1,021 — that's the exact crossover. So the planning rule is: max_safe_concurrency = floor(available_blocks / blocks_per_request). Keep actual concurrency below that and you get zero
preemptions, predictable TTFT.

That's all 6 sections. Want me to write up the full architecture document now?

### Explain why 200-word hex prompts = ~1,203 tokens

⏺ Hex strings tokenize very inefficiently compared to natural language. BPE tokenizers are trained on natural language, so common words like "the" or "running" are single tokens. Random hex characters like
a3f9c1 aren't in the vocabulary as whole units — the tokenizer breaks them down to 1-2 characters per token.

Rough math:

- 200 hex "words" (e.g., "a3f9c1 b7e2d4 ...")
- Each hex word is ~6 random characters
- BPE encodes random alphanumeric at roughly 1-2 characters per token → ~3-6 tokens per hex word
- Plus space separators as tokens
- 200 words × ~6 tokens/word ≈ 1,200 tokens

That's why hex was chosen — it's a way to generate prompts with a predictable, high token-to-word ratio. Natural language at 200 words would only be ~250-300 tokens, which wouldn't consume enough blocks
(300 tokens / 16 tokens per block = ~19 blocks) to hit the memory cliff at reasonable concurrency. With hex, 200 words → ~1,203 tokens → 75 blocks per prefill, making the cliff visible at c=14.

#### Let me verify by checking the actual script.

⏺ Read 1 file (ctrl+o to expand)

⏺ Confirmed. Looking at the script:

- make_long_prompt() generates 200 hex words, each 6 characters from md5().hexdigest()[:6]
- The comment on line 11 says ~200 hex words ≈ ~1000 tokens and line 155 says ~1200 tokens
- Line 146-149 shows the actual capacity math: ~1000 tokens ≈ 63 blocks for prefill, plus max_tokens=200 adds ~13 blocks decode, total ~76 blocks per request

The ~1,203 token count was the empirical measurement from running the tokenizer on the actual prompt. The estimate range (1,000-1,200) varies because BPE tokenization of hex is unpredictable — some 6-char
hex strings like "000000" compress better than "a3f9c1". The MD5 seeding makes each word unique and non-repeating, which resists BPE compression further.

The reason hex was chosen over, say, lorem ipsum: natural language compresses well under BPE (~1.3 tokens/word), so 200 words would only give ~260 tokens. Hex gives ~6 tokens/word, making it easy to
generate large token counts without massive wall-of-text prompts.

#### How did we get pool=1021? What is the size of a single block

⏺ Found it. Here's the formula (kv_cache_interface.py:80-87):

real_page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size

The 2 is for K and V tensors. This is per layer. For FullAttentionSpec, it uses head_size + head_size_v instead (line 181-187), but they're typically equal.

Now for Qwen2.5-3B-Instruct specifically:

⏺ Searched for 1 pattern (ctrl+o to expand)

⏺ Web Search("Qwen2.5-3B-Instruct config.json num_key_value_heads num_hidden_layers head_dim")
⎿  Did 4 searches in 42s

⏺ Here's the full calculation:

Block size formula (from kv_cache_interface.py:80-87):

page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size

But that's per KV cache group (effectively per layer group). The total per block includes all layers.

Qwen2.5-3B-Instruct architecture:

- num_hidden_layers = 36
- num_key_value_heads = 2 (GQA — 2 KV heads shared across 16 query heads)
- head_dim = 128 (= hidden_size 2048 / num_attention_heads 16)
- dtype = fp16 = 2 bytes
- block_size = 16 tokens

Per-block bytes:

block_bytes = block_size × num_layers × num_kv_heads × head_dim × dtype_bytes × 2 (K+V)
= 16 × 36 × 2 × 128 × 2 × 2
= 589,824 bytes
≈ 576 KB per block

How pool=1,021 was derived:

vLLM profiles GPU memory at startup:
available_kv_cache_memory = (total_gpu_memory × gpu_memory_utilization) - model_weights - activations - cuda_graphs

On a T4 (15 GB) at --gpu-memory-utilization 0.45:
~15 GB × 0.45 = ~6.75 GB budget

- model weights (~6 GB for 3B params in fp16)
- activations, cuda graphs, etc.
  ≈ ~600 MB left for KV cache

600 MB / 576 KB per block ≈ ~1,021 blocks

The exact number comes from vLLM's runtime profiling, not a static formula — it allocates the model, measures actual memory usage, then fills the remainder with KV cache blocks.

Sources:

- Qwen2.5-3B config.json

# Day 13 Session Prime

## Context

I'm working through a structured ML inference syllabus (day13-syllabus.md). The topic is **Prefix Caching + FP8 KV Cache** in vLLM. I'm studying for staff-level ML infrastructure interviews.

## What's done

- **Steps 1-4:** Built and ran a prefix caching experiment (day13_prefix_cache.py, run_experiment.sh) benchmarking vLLM with Qwen2.5-3B on a T4 GPU. Tested across 4 system prompt lengths (100/500/1000/2000 tokens) and 5 hit ratios (0/25/50/75/100%). Went through three iterations of debugging cache contamination bugs. Clean results and analysis are in day13-work.md.

- **Step 5 (quiz):** Completed. Key topics covered:
  - Why TTFT speedup drops from 18x (syslen=1000) to 12.4x (syslen=2000) at 100% hit: uncached tokens still pay O(sequence_length) attention cost against all cached KV entries.
  - Prefill is O(N^2) not O(N), so "60% of tokens cached" != "60% compute saved." For a RAG pipeline (900 system + 600 RAG + 50 user tokens), actual savings are ~34%.
  - System design levers for RAG: deterministic chunk ordering + prefix-aware request routing.
  - Cache contamination detection: cache-OFF should be flat across hit ratios. If it's not, caching wasn't actually off.
  - Cross-condition isolation: unique variant IDs per condition, vLLM restart between syslens.
  - Prefix caching has near-zero overhead at 0% hit rate, which is why vLLM V1 defaults it on.
  - Multi-turn chat benefits from prefix caching (entire conversation history is the prefix for the next turn).

## What's next

- **Step 6: FP8 KV Cache Capacity Modeling** (T4 doesn't support FP8, so this is theoretical)
  - Part A: KV capacity calculator. We were about to do the FP16 KV bytes-per-token calculation from first principles for Qwen2.5-3B (2 KV heads, 128 head_dim, 36 layers). Challenge me on this rather than giving me the answer.
  - Part B: Document 6 production assumptions that make the theoretical capacity model wrong.
  - Part C: FP8 KV quality/backend nuance (standard vs Flash Attention 3, calibration pathways).
- **Step 7:** Update master tradeoff table with today's data.
- **Step 8:** Write interview insight paragraphs with measured numbers.
- **End-of-day checklist** is in the syllabus.

## How to work with me

- Challenge me on concepts instead of giving me answers directly. Push back when my reasoning is incomplete or wrong.
- Sound human. Conversational tone, no em-dashes.
- Keep responses concise.
- Reference day13-syllabus.md for step details and day13-work.md for completed work and experiment results.

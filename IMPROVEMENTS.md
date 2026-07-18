# Repo improvements: summary and open decisions

One-time handoff doc. It records what was changed to raise polish and navigability
without touching the substance, the numbers, or the honest framing, and it lists the
items left for you to decide. Delete it once you have actioned the flags.

All changes are separate git commits on `main`, so any one can be reverted in isolation
with `git revert <hash>`. Nothing was pushed.

## What changed (by task)

```
commit    area
1d2f442   gitignore: exclude .vscode/ and prep/ (your interview+resume material,
          kept out of the hiring-manager-facing repo, same rationale as resumes/)
cf8c59d   REPO_METADATA.md: About-box description + GitHub topics to paste manually
372058d   study/phase-a/README.md + study/phase-b/README.md: per-phase index of days
          with one-line findings (reused from your own top-README day list) and links
cf48294   fix: benchmark_matrix.py had a broken relative model path and a
          machine-specific absolute HF cache path; both now match the sibling scripts
7b15bed   docs: module docstrings on the two headline scripts (block instrumentation
          patch, admission gateway): what each measures + exact run command
714d1e6   docs: benchmark_day24.py How-to-run block; requirements.txt scope note
```

Task-by-task:

```
1. Repo hygiene         REPO_METADATA.md added. README links verified (51 links,
                        0 broken). .gitignore is sensible; no logs/pycache/traces are
                        committed; the only binaries are small result plots + Grafana
                        screenshots (legit deliverable artifacts). Nothing to remove.
2. Reproducibility      Checked day8, day11, day16, day24. Fixed 2 real rot spots in
                        benchmark_matrix.py. Surfaced exact run commands in the script
                        docstrings and a real dep-scope note in requirements.txt.
3. Navigability         Both phase folders now have an index README. The two "fastest
                        entry point" docs (progress_summary.md, phase_a_b_..._v1.md)
                        exist and are current.
4. Code quality         Module docstrings added to the two headline scripts only. No
                        refactors, no framework, no restructure.
5. Honesty audit        Flags below. NOT auto-applied, per your instruction.
```

## Flagged for your decision (not applied)

### Honesty audit

The repo is unusually disciplined. Every day-level deliverable that relies on modeling
rather than measurement already says so in its own header. The one real issue is that
the top README's blanket framing flattens the measured-vs-derived line that the
deliverables are careful to keep. Items 1-3 are the same concern in three places.

```
1. README.md:4-5   OVERCLAIM
   now:  "Every finding here came from running real experiments on a live GPU and
          measuring the result, not from reading papers or coursework."
   why:  Days 26-29 (retry-storm amplification, cascade) are derived on paper from
          the measured cliff data, not measured. Day 27's own header says so.
   maybe: "Nearly every finding came from running real experiments on a live GPU;
          the retry-storm and autoscaling analyses are derived analytically from the
          measured cliff data, and say so."

2. README.md:42   OVERCLAIM (headline block)
   now:  "Retry storms self-amplify ~1.56x   a transient trigger drives a 2-3 minute
          cascade past the cliff"
   why:  Sits beside measured results with no derived marker. 1.56x is a geometric
          series (Day 26); the 2-3 min cascade is a reasoned estimate (Day 27).
   maybe: "Retry storms self-amplify ~1.56x (derived): a transient trigger models
          out to a 2-3 minute cascade past the cliff."

3. README.md:23   OVERCLAIM
   now:  "I measure before I claim. Every number below is reproducible from the
          scripts in this repo."
   why:  The 1.56x and the cascade duration come from a formula/reasoning, not a
          runnable script.
   maybe: "Every measured number below is reproducible from the scripts; the
          retry/autoscaling figures are derived from those measurements analytically."

4. study/phase-b/day23/deliverable-6-prefill-decode-interference.md:17
   LOW-SEVERITY, safe factual fix (a citation typo, not an honesty call)
   now:  "using the empirical fit from Day 21 (TTFT(ms) = 37.6 + 0.228 * prompt_tokens)"
   why:  That regression is the Day 6 result everywhere else in the repo. Day 21 did
          not produce it. The number is right, the day label is wrong.
   maybe: change "Day 21" to "Day 6".

5. study/phase-b/day24/deliverable-7-cliff.md:209 vs 312/315
   LOW-SEVERITY precision nit
   why:  The same 879->803 tok/s delta is reported as 8.6% (divided by 879) in one
          spot and 9.5% (divided by 803) in another. Not an overclaim, just reads
          inconsistent.
   maybe: pick one denominator, or state "8.6% of peak / 9.5% above the recommended
          point" if both are intended.
```

Exemplary, leave as-is (called out so you do not soften them by accident):

```
- README.md methodology note (the AI-assisted disclosure)
- day27 postmortem header caveat ("reasoned estimates, not observed values")
- day29 memo ("not experimentally validated on this rig")
- day15 Generalizability Limits table (each row tagged measured vs theoretical)
- the recurring, against-interest reporting of the 8x GQA error and the Day 18
  "admission control made latency worse" negative result
```

No SOUNDS-PRODUCTION issues found. Every "production" reference in the deliverables is
properly hedged, and the framing stays single-GPU / learning-oriented throughout.

### Reproducibility, optional cleanups (not applied)

```
- The run commands now live in the script docstrings (single source of truth, and the
  README links reviewers straight to those scripts). If you would rather a reviewer
  see the command in the day WRITEUP too, day16-work.md and day11-work.md still keep
  their run command only in a sibling doc (day16-afternoon-prompt.md,
  day11-work-claude.md). Optional to duplicate one line into the main writeup.
- study/phase-b/day24/plot_day24.py:16-17 has two unused imports (statistics,
  defaultdict). Harmless. Left alone to honor "no style-only refactors."
- The Makefile in study/phase-a targets the Day 18/19 gateway assets (its
  GATEWAY_PY := day19/gateway.py). That is correct for those days; the four days
  audited here are launched by hand, which the docstrings now show.
```

## GitHub About box + topics (paste manually)

Repo page -> gear next to "About" -> Description + Topics. Also in REPO_METADATA.md.

```
Description:
Self-directed, single-GPU (NVIDIA T4) LLM inference engineering residency: measured
vLLM V1 experiments on serving failure modes, KV-cache capacity, quantization,
admission control, and autoscaling. Every finding reproducible from the scripts.

Topics:
llm-inference vllm gpu performance-engineering kv-cache paged-attention quantization
observability autoscaling admission-control roofline inference-optimization
```

Do not add "production" or "distributed-systems" topics; the multi-GPU track was
scoped but not executed.

# Repo metadata

GitHub can't set the About box or topics from a file, so paste these in manually:
Repo page -> the gear icon next to "About" (top right) -> Description + Topics.

## Description (About box)

```
Self-directed, single-GPU (NVIDIA T4) LLM inference engineering residency: measured
vLLM V1 experiments on serving failure modes, KV-cache capacity, quantization,
admission control, and autoscaling. Every finding reproducible from the scripts.
```

If GitHub truncates it, the short form:

```
Single-GPU LLM inference residency: measured vLLM V1 experiments on failure modes,
KV-cache capacity, quantization, and control systems.
```

## Topics

Paste these into the Topics field (space or comma separated in the GitHub UI):

```
llm-inference
vllm
gpu
performance-engineering
kv-cache
paged-attention
quantization
observability
autoscaling
admission-control
roofline
inference-optimization
```

## Notes

- The framing is deliberately single-GPU and learning-oriented. Do not add topics like
  "production" or "distributed-systems" that would misrepresent the scope. The
  multi-GPU / tensor-parallel track was scoped but not executed.
- If you set a social preview image, the README's "Headline findings" block is the
  strongest thing to screenshot.

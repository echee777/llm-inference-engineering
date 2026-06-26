# Day 18 Load Test Results

## Setup

- **GPU:** Tesla T4 (15 GiB)
- **Model:** Qwen/Qwen2.5-3B-Instruct (half precision, enforce-eager)
- **KV cache capacity:** 217,312 tokens
- **Admission budget:** 141,252 tokens (65% of 217,312)
- **Traffic:** 5-bucket two-dimensional matrix with min_tokens forcing long completions
- **Queue:** FIFO, max 50, 5-second timeout

## Test Matrix

Four tests were run, all with the same traffic mix and 10-minute duration:

```
Test                  Users  AC      Result
────────────────────────────────────────────────────
baseline_50u          50     OFF     Completed normally
controlled_50u        50     ON      Completed normally
baseline_100u         100    OFF     Completed normally
controlled_100u       100    ON      Completed, 678 rejections
baseline_200u         200    OFF     vLLM crashed
controlled_200u       200    ON      vLLM crashed
```

## Results: 50 Users

At 50 users, both tests are nearly identical. The system is well within
capacity. No preemptions, no rejections, stable TTFT.

```
50 Users               Baseline (AC OFF)    Controlled (AC ON)
────────────────────────────────────────────────────────────────
TTFT p50               ~160ms               ~160ms
TTFT p95               ~190-200ms           ~190-200ms
TTFT p99               ~210-230ms           ~210-270ms
TTFT max               533ms                640ms
Rejections             0                    0
Preemptions            0                    0
Budget utilization     n/a (no cap)         ~70-76% steady state
Total requests         1,254                1,279
```

**Interpretation:** 50 users doesn't stress the system. The Qwen 3B model
on a T4 has 217k tokens of KV cache, which is generous for this workload.
Admission control is active but never needs to reject.

## Results: 100 Users

At 100 users, the tests diverge significantly, but not in the direction
the syllabus anticipated.

```
100 Users              Baseline (AC OFF)    Controlled (AC ON)
────────────────────────────────────────────────────────────────
TTFT p50 (short)       210ms                210ms
TTFT p50 (long)        220ms                2,400ms
TTFT p50 (medium)      220ms                260ms
TTFT p95               260-290ms            4,600-4,900ms
TTFT p99               340-570ms            5,100-5,200ms
TTFT max               1,368ms              5,183ms
Rejections             0                    678 (32% of attempts)
Preemptions            0                    0
Budget utilization     n/a                  hit ceiling, queue active
Total admitted         987                  1,411
Total rejected         0                    678
```

**Interpretation:** The controlled test has worse TTFT than the baseline.
This is because:

1. The system has enough KV headroom that no preemptions occur even at
   100 users without admission control. The "cliff" never materializes.

2. With AC enabled, the budget ceiling is hit and requests queue for up
   to 5 seconds (MAX_WAIT_SECONDS) before either being admitted or
   rejected. This queue wait inflates gateway TTFT massively.

3. The 5-second TTFT values are requests that sat in the queue for the
   full timeout, then either got admitted late or were rejected.

**This is a valid and important finding:** Admission control adds latency
when the system doesn't need protection. The budget was calibrated for a
scenario where KV exhaustion leads to preemption cascades. On this hardware
with this model, that scenario doesn't occur at 100 users. The admission
controller is being overly conservative, rejecting 32% of traffic that
would have been served fine.

## Results: 200 Users

At 200 users, vLLM crashes in both tests. The failure mode is not KV
exhaustion but process-level overload (connection handling, internal
memory). Admission control cannot protect against this class of failure.

## Key Takeaways

### 1. The preemption cliff depends on the hardware/model ratio

The syllabus assumed a tighter KV budget where 50 users would trigger
preemptions. The Qwen 3B model has small KV vectors relative to the T4's
memory, giving 217k tokens of cache. The cliff would occur with a larger
model (7B+) or lower gpu-memory-utilization.

### 2. Admission control can hurt when miscalibrated

At 100 users, AC added 2-5 seconds of queue wait to requests that would
have been fine without it. The 65% utilization target was too conservative
for this workload. Options:
- Raise TARGET_UTILIZATION to 80-85% (less headroom, more throughput)
- Lower MAX_WAIT_SECONDS (faster rejection, less queue latency)
- Tune based on empirical preemption threshold, not theoretical budget

### 3. The 200-user crash reveals a different failure mode

vLLM died from process-level overload, not KV exhaustion. Admission
control gates on KV memory budget but cannot protect against:
- Connection/socket exhaustion
- Python asyncio task explosion
- Internal vLLM scheduler overload

This is analogous to the compute-bound blind spot discussed in the
quiz: AC protects one resource (KV memory) but not others.

### 4. Policy B progressive release works correctly

With min_tokens forcing long completions, Policy B released budget
progressively (correction_freed=11,672 tokens in the controlled 100u
test). Requests generating 1500-2000 tokens hit the 50-token release
checkpoint ~30 times each.

## Interview Narrative

"I ran load tests at 50, 100, and 200 concurrent users against a 3B
model on a T4. At 50 users the system was well within capacity and
admission control had no effect. At 100 users, admission control
actually hurt, adding 2-5s of queue latency to requests that would
have been served fine without it, because the KV cache was large enough
to absorb the load without preemptions. At 200 users, vLLM crashed
from process-level overload regardless of admission control.

This told me two things: first, the 65% utilization target was too
conservative for this hardware/model combination. Second, admission
control only protects against KV exhaustion, not against other failure
modes like connection overload or compute saturation. In production, I'd
calibrate the budget empirically by finding the actual preemption
threshold, and I'd add additional protections (connection limits,
request rate caps) for the failure modes that token-budget AC doesn't
cover."

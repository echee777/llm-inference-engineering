# Day 28 Questions

Three questions Day 28 must answer to complete Deliverable #9 (Autoscaling Memo).

---

## Q1: Queue depth trigger lag vs KV utilization trigger lag

How many seconds does queue depth lead KV utilization as a scale-out signal?

Experiment: ramp load from healthy to cliff. Record the timestamp when queue depth first exceeds threshold (e.g., > 5 sustained) and the timestamp when KV utilization first exceeds threshold (e.g., > 79%). The delta is the lead time advantage of queue-based scaling.

This number fills the "leads cliff by N seconds" cell in the Autoscaling Memo Section 3 comparison table and justifies the choice of queue depth over KV utilization as the primary scale-out trigger.

---

## Q2: Graceful drain vs naive termination

What breaks when a pod is terminated without draining?

Experiment: under moderate load (~60% KV), terminate a vLLM pod (a) immediately (kill -9) and (b) with drain (stop admission, wait for in-flight completion, then terminate). Measure: number of failed requests, client-visible error rate, retry spike on remaining pods, time to recovery.

This fills Autoscaling Memo Section 5 with measured consequences rather than mechanical argument.

---

## Q3: Composite signal threshold validation

Do the proposed scale-out thresholds (queue > 5 sustained 60s with positive gradient) trigger early enough to beat the cliff?

Experiment: ramp load toward cliff with the proposed thresholds as an alert rule. Record: when the alert would fire, when the cliff actually occurs, whether the 90-second model load window fits in the gap.

This validates or adjusts the thresholds in Autoscaling Memo Section 4.

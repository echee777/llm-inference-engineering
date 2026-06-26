# Day 18 (Wed) — Load Testing + Admission Control Validation

> **Version:** v3 (incorporating R1 × 4 + author catch + R2 × 1 + R3 × 3 + R4 × 2)
>
> **Change log:**
> - [FIX-1] Traffic matrix made two-dimensional: prompt length AND max_tokens both varied
> - [FIX-2] Results tables expanded: admitted throughput, total throughput, queue wait p95 added
> - [FIX-3] V1 recompute-only preemption clarification added inline
> - [FIX-4] Dashboard Panel 5 expanded to show admitted / rejected / queued as three time series
> - [FIX-5] stream=False → stream=True; TTFT instrumented as time-to-first-chunk (author catch, missed by R1)
> - [FIX-6] Gateway TTFT vs Model TTFT split into separate metrics and separate dashboard panel
> - [FIX-7] "Total throughput" definition clarified: offered load at ingress, not model-processed tokens
> - [FIX-8] Queue depth vs queue wait p95 relationship made explicit (Panel 2)
> - [FIX-9] Token estimation bias tied to concrete failure modes (over-admit cliff / under-utilize)
> - [FIX-10] Admission control compute-bound limitation added to Interview Readiness section
> - [FIX-11] Panel 7 extended with compute-bound vs memory-bound diagnostic statement

---

## Goal

Produce empirical proof that your admission control gateway protects latency SLOs under
load — and make that proof visible, defensible, and causally interpretable via Prometheus +
Grafana dashboards.

The two required outputs are:

1. **Load test results:** baseline (no admission) vs. controlled (admission enforced), same traffic,
   same ramp, directly comparable tables.
2. **Working dashboard:** seven panels that tell the full operator story during an incident.

---

## Morning Block (4 hrs) — Load Tests

---

### Step 1 — Traffic matrix + Locust setup (1 hr)

#### [FIX-1] Two-dimensional traffic matrix

The original single-dimension matrix only varied `max_tokens`. Because your admission budget
is `prompt_tokens + max_completion_tokens`, varying only completion length tests one half of
the budget equation. A long-prompt/short-output request and a short-prompt/long-output request
have completely different runtime profiles — prefill-heavy vs. decode-heavy — even when their
token budget reservation is identical.

**Corrected five-bucket matrix:**

| Bucket | Prompt Tokens | Max Completion | Budget Cost | Runtime Bottleneck | Traffic Weight |
|---|---|---|---|---|---|
| short | 50 | 100 | 150 | decode | 20% |
| medium | 200 | 500 | 700 | mixed | 30% |
| long | 500 | 2000 | 2500 | decode | 20% |
| long-prompt/short-output | 1500 | 100 | 1600 | **prefill** | 10% |
| short-prompt/long-output | 50 | 2000 | 2050 | **decode** | 20% |

> **Interview insight:** Admission control uses token budget, but runtime cost depends on
> *where* those tokens fall (prefill vs. decode). Two requests with identical budget can hit
> completely different GPU bottlenecks. This connects directly to the prefill/decode asymmetry
> from Day 11 benchmarks.

#### [FIX-5] Use stream=True — TTFT must be time-to-first-chunk

`stream=False` measures **total response latency**, not TTFT. For a 2000-token completion,
those numbers differ by seconds. Any TTFT metric collected against `stream=False` traffic is
invalid. All load test requests must use `stream=True`, with TTFT instrumented as
**time from request send to first SSE chunk received**.

#### Locust script

Install:

```bash
pip install locust
```

`locustfile.py`:

```python
from locust import HttpUser, task, between
import random, json, time

# [FIX-1] Two-dimensional traffic matrix
TRAFFIC_MIX = [
    # (name,               prompt_tokens, max_tokens, weight)
    ("short",              50,            100,        0.20),
    ("medium",             200,           500,        0.30),
    ("long",               500,           2000,       0.20),
    ("long_prompt_short",  1500,          100,        0.10),
    ("short_prompt_long",  50,            2000,       0.20),
]

def pick_bucket():
    r = random.random()
    cumulative = 0.0
    for name, p_tokens, c_tokens, weight in TRAFFIC_MIX:
        cumulative += weight
        if r < cumulative:
            return name, p_tokens, c_tokens
    return "medium", 200, 500

def build_prompt(n_tokens: int) -> str:
    # Approximate: 1 token ≈ 0.75 words for English prose
    words = max(1, int(n_tokens * 0.75))
    return ("The quick brown fox jumps over the lazy dog. " * (words // 9 + 1))[:words * 5]

class InferenceUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def send_request(self):
        name, p_tokens, c_tokens = pick_bucket()
        payload = {
            "model": "your-model-name",
            "messages": [{"role": "user", "content": build_prompt(p_tokens)}],
            "max_tokens": c_tokens,
            "stream": True,   # [FIX-5] must be True for valid TTFT measurement
        }

        request_start = time.monotonic()
        first_token_received = False

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name=f"/{name}"
        ) as resp:
            if resp.status_code == 429:
                resp.success()   # admission rejection — expected, not a Locust failure
                return
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            for chunk in resp.iter_lines():
                if chunk and not first_token_received:
                    # [FIX-5] TTFT = time to first SSE data chunk
                    ttft_ms = (time.monotonic() - request_start) * 1000
                    self.environment.events.request.fire(
                        request_type="TTFT",
                        name=f"ttft/{name}",
                        response_time=ttft_ms,
                        response_length=0,
                        exception=None,
                        context={}
                    )
                    first_token_received = True
```

**Ramp config:**
- Users: 1 → 50, spawn rate 2/sec
- Run duration: 10 minutes minimum per test
- Why 50 users: KV budget ~33,800 tokens at 65% utilization; at ~700 token average cost per
  request the system can admit ~48 concurrent requests. 50 users reliably enters the rejection
  zone.

---

### Step 2 — Test 1: Baseline, no admission control (1.5 hrs)

Disable admission check (`ADMISSION_ENABLED=false` env flag or comment out the check).
Run Locust, ramp to 50 users. Record every 5 minutes.

#### [FIX-3] V1 recompute-only preemption — read before starting

> In vLLM V1 there is no SWAPPED state and no CPU swap path. Any preemption event means
> the request is dropped from the KV pool and must **re-prefill from scratch**. TTFT spikes
> under memory pressure are **recompute symptoms**, not swap symptoms. Watch for this in
> your baseline run — log lines with "Preempting..." indicate re-prefill workload is being
> added on top of the existing queue.

Monitor during the run:

```bash
# Terminal 1 — GPU memory live
nvidia-smi dmon -s mc -d 2

# Terminal 2 — vLLM logs, preemption events
journalctl -fu vllm | grep -E "Preempt|OOM|recompute"
```

#### [FIX-2] Expanded baseline results table

| Concurrent Users | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Admitted Throughput (tok/s) | Total Throughput (tok/s) | Queue Wait p95 (ms) | Preemption Events | OOM? |
|---|---|---|---|---|---|---|---|---|
| 5 | | | | | | | | |
| 15 | | | | | | | | |
| 25 | | | | | | | | |
| 35 | | | | | | | | |
| 50 | | | | | | | | |

**Expected observation:** TTFT is stable through ~25–30 users, then hockey-sticks. Above ~35
users: preemption cascades, TTFT p99 goes to seconds, admitted throughput collapses. This is
the cliff from Day 9 reproduced systematically.

> **Narrative thread:** "I first observed this cliff in the Day 9 mini-collapse experiment. Day 18
> reproduces it systematically with a controlled traffic mix so I can measure the before/after
> contrast precisely."

---

### Step 3 — Test 2: With admission control (1.5 hrs)

Re-enable admission control. Config from Day 17:

```python
KV_CAPACITY_TOKENS  = 52_000
TARGET_UTILIZATION  = 0.65
ADMISSION_BUDGET    = int(KV_CAPACITY_TOKENS * TARGET_UTILIZATION)  # ~33,800
```

> **[FIX-9] Why 65% and not higher?** The 35% headroom absorbs token estimation error. If
> token estimation is biased low (you consistently undercount request cost), admission
> over-admits and reintroduces the KV exhaustion cliff. If estimation is biased high (you
> consistently overcount), you underutilize GPU capacity — safe, but wasteful. The 65% target
> is calibrated from the Day 17 token budget correction experiment, which measured actual
> completion length vs. `max_completion_tokens` on your traffic mix.

Run **identical traffic** — same Locust script, same ramp, same duration.

#### [FIX-2] Expanded admission-control results table

> **[FIX-7] Column definition — Total Throughput:** Total throughput is measured as attempted
> token rate at ingress (`prompt_tokens + max_tokens` per request), not actual tokens processed
> by the model. Rejected requests never reach the model, so "total throughput" measures offered
> load, not completed work. Admitted throughput is the subset the model actually processed.

| Concurrent Users | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Admitted Throughput (tok/s) | Total Throughput (tok/s) | Queue Wait p95 (ms) | Rejection Rate (%) |
|---|---|---|---|---|---|---|---|
| 5 | | | | | | | |
| 15 | | | | | | | |
| 25 | | | | | | | |
| 35 | | | | | | | |
| 50 | | | | | | | |

**Expected observation:**
- TTFT for admitted requests stays stable across all load levels
- Rejection rate climbs as load increases — excess load is shed
- Admitted throughput is lower than total attempted throughput — that gap is the cost of protection
- No preemption events, no OOM

> **Queue wait is the price paid before admission; TTFT is the price paid after admission.**
> These are causally distinct. Queue wait p95 staying low confirms requests aren't waiting
> long in the queue before a decision is made — the admission check itself is fast.

**The core tradeoff to articulate:** Admission control trades rejection rate for latency
stability. Total throughput drops (some requests are rejected). Admitted requests have
predictable, SLO-consistent latency. This is the correct tradeoff for any latency-SLO-bound
serving system.

---

## Afternoon Block (4 hrs) — Dashboards

---

### Step 4 — Prometheus + Grafana setup (2 hrs)

```bash
# Prometheus
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Grafana
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

`prometheus.yml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'gateway'
    static_configs:
      - targets: ['host.docker.internal:8001']

  - job_name: 'vllm'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

#### Gateway metrics instrumentation

```python
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import time

# Counters / gauges
REQUEST_COUNTER      = Counter('gateway_requests_total', 'Requests by disposition',
                               ['status'])  # status: admitted, rejected, queued
QUEUE_DEPTH          = Gauge('gateway_queue_depth', 'Current queue depth')
TOKEN_BUDGET_USED    = Gauge('gateway_token_budget_used_pct', 'Token budget utilization %')

# [FIX-5] Gateway TTFT: request arrival → first token returned to client
GATEWAY_TTFT         = Histogram('gateway_ttft_seconds', 'Gateway TTFT',
                                  buckets=[.05, .1, .25, .5, 1, 2, 5, 10])

# [FIX-6] Model TTFT: admission timestamp → first token emitted by vLLM
MODEL_TTFT           = Histogram('model_ttft_seconds', 'Model TTFT (post-admission)',
                                  buckets=[.05, .1, .25, .5, 1, 2, 5, 10])

# Queue wait: enqueue timestamp → admission timestamp
QUEUE_WAIT           = Histogram('gateway_queue_wait_seconds', 'Queue wait before admission',
                                  buckets=[.001, .005, .01, .05, .1, .5, 1, 5])

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

#### [FIX-6] Timestamp instrumentation in request lifecycle

```python
async def handle_request(request):
    arrival_time = time.monotonic()

    # --- Admission check ---
    estimated_cost = request.prompt_tokens + request.max_completion_tokens
    if active_token_budget + estimated_cost > ADMISSION_BUDGET:
        REQUEST_COUNTER.labels(status="rejected").inc()
        raise HTTPException(status_code=429)

    REQUEST_COUNTER.labels(status="admitted").inc()
    admitted_time = time.monotonic()

    # Queue wait = time from arrival to admission decision
    QUEUE_WAIT.observe(admitted_time - arrival_time)

    first_token = True
    async for chunk in stream_from_vllm(request):
        if first_token:
            now = time.monotonic()
            # [FIX-6] Model TTFT: admission → first token from backend
            MODEL_TTFT.observe(now - admitted_time)
            # [FIX-5] Gateway TTFT: arrival → first token to client
            GATEWAY_TTFT.observe(now - arrival_time)
            first_token = False
        yield chunk
```

vLLM exposes Prometheus metrics natively at `/metrics` when started with `--enable-metrics`.
Key vLLM metrics:

- `vllm:gpu_cache_usage_perc` — KV cache pool utilization
- `vllm:time_to_first_token_seconds` — vLLM's internal TTFT histogram
- `vllm:num_preemptions_total` — cumulative preemption counter
- `vllm:request_success_total` — completed request count

---

### Step 5 — Grafana dashboard (2 hrs)

Open Grafana at `http://localhost:3000` (default admin/admin). Add Prometheus as a data
source. Build one dashboard with seven panels.

---

**Panel 1 — Active Token Budget Utilization % (primary operational metric)**

```promql
gateway_token_budget_used_pct
```

This is the number to watch during an incident. If it approaches 100%, admitted traffic is
about to see latency climb. Target operating range: 55–70%.

---

**Panel 2 — Queue Depth**

```promql
gateway_queue_depth
```

Leading indicator. Queue depth spikes *before* TTFT spikes. Rising queue depth at stable
budget utilization means burst absorption is working. Rising queue depth *and* rising budget
utilization means you are approaching the admission cliff. **[FIX-8] Queue depth is a leading
indicator; queue wait p95 is the user-visible consequence of sustained depth.** Watch both
together: depth tells you the system is under pressure now; wait p95 tells you how much that
pressure has already cost users.

---

**Panel 3 — GPU KV Cache Utilization**

```promql
vllm:gpu_cache_usage_perc * 100
```

Validates that your token budget tracks actual GPU memory. These two should move together.
If token budget shows 65% but GPU cache shows 90%, your token estimation model is wrong.

---

**Panel 4 — TTFT p50 / p95 / p99 (three time series, one panel)**

```promql
histogram_quantile(0.50, rate(gateway_ttft_seconds_bucket[1m]))
histogram_quantile(0.95, rate(gateway_ttft_seconds_bucket[1m]))
histogram_quantile(0.99, rate(gateway_ttft_seconds_bucket[1m]))
```

The gap between p95 and p99 indicates tail fatness. With admission control working, all
three should stay flat under load. Without admission control, p99 breaks first, then p95.

---

**Panel 5 — [FIX-4] Admitted / Rejected / Queued Request Rate (three time series, one panel)**

```promql
rate(gateway_requests_total{status="admitted"}[1m])
rate(gateway_requests_total{status="rejected"}[1m])
rate(gateway_requests_total{status="queued"}[1m])
```

Shows the cost of protection alongside the benefit. As load increases, rejected rate rises
while admitted rate stays bounded. Queued is the third state most dashboards omit — it shows
burst absorption in action before admission decisions are made.

---

**Panel 6 — Token Budget Utilization vs. TTFT p99 (dual Y-axis)**

Left axis: `gateway_token_budget_used_pct`
Right axis: `histogram_quantile(0.99, rate(gateway_ttft_seconds_bucket[1m]))`

The money panel. With admission control: token budget is capped by policy, TTFT p99 stays
flat. Without admission control (baseline run): token budget climbs freely, TTFT p99 follows
with a lag then explodes. This single graph is the visual proof that admission control works.

---

**Panel 7 — [FIX-6] Gateway TTFT vs. Model TTFT p99 (two time series, one panel)**

```promql
histogram_quantile(0.99, rate(gateway_ttft_seconds_bucket[1m]))
histogram_quantile(0.99, rate(model_ttft_seconds_bucket[1m]))
```

The gap between these two lines is queue wait + network overhead. With admission control
working:

- Both should be stable and close together (queue wait near zero for admitted requests)
- Gateway TTFT slightly above Model TTFT — the delta is gateway overhead only

If gateway TTFT climbs while model TTFT stays flat, queue wait is growing — admission is
working but the queue is building up. If both climb together, the model backend is under
pressure despite admission control — investigate KV cache and preemption.

> **[FIX-11] Compute-bound diagnostic:** If model TTFT increases while KV cache utilization
> (Panel 3) remains stable, the system is compute-bound rather than memory-bound — prefill
> saturation, not KV exhaustion. This is a distinct failure mode that admission control does
> not prevent. A burst of long-prompt requests can produce this signature: token budget within
> limits, GPU compute saturated, TTFT rising.

> **Interview answer this unlocks:** "I tracked gateway TTFT and model TTFT separately to
> confirm admission control was doing real work, not just hiding backend degradation behind
> queueing. Both stayed flat under load — that proves the model was protected, not masked."

---

## End-of-Day Output Checklist

| Artifact | Status |
|---|---|
| Baseline results table (5 load levels, 8 columns) | |
| Admission-control results table (5 load levels, 8 columns) | |
| Written comparison paragraph: TTFT improvement + throughput cost quantified | |
| Working Prometheus + Grafana stack scraping gateway + vLLM | |
| Dashboard screenshot: all 7 panels populated with real data from load tests | |

---

## Interview Readiness — Anticipated Questions

**"Why 65% utilization target?"**
The KV pool is not uniformly allocated. Requests reserve `max_completion_tokens` upfront.
The remaining 35% is headroom for token estimation error and burst. Measured on Day 17: with
token budget correction (release excess as tokens stream), actual capacity improvement is X%.

**"What's your admission signal?"**
Active token budget utilization — not request count, not GPU Util%. GPU Util% is misleading
(empirically demonstrated on Day 1: a single lightweight kernel shows 100%).

**"What does your dashboard tell you during an incident?"**
Token budget utilization is the leading indicator. Queue depth confirms admission limits are
being hit. Gateway TTFT p99 confirms client-visible latency. Model TTFT p99 separately
confirms whether the backend is healthy or degrading. These four together tell the full causal
story.

**"You protected TTFT — but did you crush throughput?"**
Admitted throughput stays near-constant across all load levels. Total throughput (admitted +
rejected attempts) grows with load but the excess is shed as 429s. The system trades increased
visible rejection rate for stable latency on admitted traffic. That is the correct tradeoff
for a latency-SLO-bound serving system.

**"What happens under a long-prompt/short-output burst?"**
Those requests are prefill-heavy. They consume their token budget reservation quickly (short
completion) but spike GPU compute during admission. My traffic matrix includes this bucket
explicitly — I measured admission control behavior under prefill-heavy load as a distinct
test condition.

**"What is recompute-only preemption and why does it matter here?"**
In vLLM V1 there is no SWAPPED state. Preemption means the request's KV blocks are freed and
the request must re-prefill from scratch when re-admitted. Under high load without admission
control, preemption cascades create a recompute avalanche — each preempted request adds
prefill work on top of the existing queue. Admission control prevents this by keeping the KV
pool below the fragmentation threshold.

**"What does admission control not protect against?"**
Admission control gates on KV memory budget — it prevents KV exhaustion and recompute
cascades. It does not regulate prefill compute consumption. A burst of long-prompt requests
increases GPU GEMM workload during prefill even if token budget stays within the 65% target.
In this regime KV utilization remains stable but model TTFT rises due to compute saturation —
a distinct failure mode. On the dashboard this is identifiable: Panel 7 model TTFT climbs
while Panel 3 KV utilization stays flat. In production this is addressed with additional
controls — prefill rate limiting, prompt length caps, or separate admission budgets for
prefill-heavy vs. decode-heavy traffic. My traffic matrix includes a long-prompt/short-output
bucket explicitly to observe this boundary condition.

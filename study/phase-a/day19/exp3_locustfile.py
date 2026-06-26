# locustfile.py — Day 18: Two-dimensional traffic matrix load test
#
# Locust is a Python load testing framework. It spawns simulated "users"
# that send requests to your server concurrently. Each user runs the
# @task method in a loop with random pauses between requests.
#
# This script simulates realistic inference traffic with a mix of request
# sizes that exercise BOTH dimensions of the admission budget:
#   - prompt_tokens (prefill cost, compute-heavy)
#   - max_tokens (decode cost, memory-heavy)
#
# Key design decisions:
#   1. stream=True on all requests (required for valid TTFT measurement)
#   2. 429/503 responses marked as success (admission rejection is expected behavior)
#   3. Custom TTFT event fired on first SSE chunk (shows up in Locust stats)
#   4. All chunks consumed even after TTFT (so the gateway properly releases budget)

# HttpUser: base class for a simulated user. Each instance sends requests
#   in a loop with random wait times between them.
# task: decorator that marks a method as something the user "does"
# between: random wait time generator (e.g., between(0.5, 2.0) = 0.5 to 2s)
from locust import HttpUser, task, between
import random
import time

# ============================================================================
# TRAFFIC MATRIX
# ============================================================================
# This is the two-dimensional traffic mix from the syllabus [FIX-1].
# Each bucket specifies BOTH prompt length AND completion length.
#
# Why two dimensions? Because admission control budgets on
# prompt_tokens + max_completion_tokens. A request that's 1500 prompt + 100
# completion has a similar budget cost to 50 prompt + 2000 completion, but
# they stress completely different GPU resources:
#   - Long prompt + short output = prefill-heavy (GPU compute bound)
#   - Short prompt + long output = decode-heavy (KV memory bound)
#
# The weights approximate a production traffic distribution:
#   - "medium" is most common (30%)
#   - "long_prompt_short" is rare (10%) but included specifically to test
#     the compute-bound blind spot in admission control
TRAFFIC_MIX = [
    # (name,               prompt_tokens, max_tokens, min_tokens, weight)
    # min_tokens forces the model to actually generate long completions,
    # so KV cache is held for the full duration and budget pressure is real.
    # Without min_tokens, the model answers gibberish prompts in 30-70 tokens
    # regardless of max_tokens, and we never stress the system.
    ("short",              50,            100,        80,       0.20),   # light requests, decode-dominant
    ("medium",             200,           500,        400,      0.30),   # typical workload, mixed
    ("long",               500,           2000,       1500,     0.20),   # heavy requests, decode-dominant
    ("long_prompt_short",  1500,          100,        80,       0.10),   # prefill-heavy, tests compute bound
    ("short_prompt_long",  50,            2000,       1500,     0.20),   # decode-heavy, tests memory bound
]

# Must match the model name configured in vLLM and the gateway.
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def pick_bucket():
    """Weighted random selection from the traffic matrix.

    Generates a random float [0, 1) and walks through the weights
    cumulatively until we find which bucket it falls in.
    With weights [0.20, 0.30, 0.20, 0.10, 0.20]:
      r < 0.20         -> "short"
      r < 0.50 (0.2+0.3) -> "medium"
      r < 0.70         -> "long"
      r < 0.80         -> "long_prompt_short"
      r < 1.00         -> "short_prompt_long"
    """
    r = random.random()
    cumulative = 0.0
    for name, p_tokens, c_tokens, min_tokens, weight in TRAFFIC_MIX:
        cumulative += weight
        if r < cumulative:
            return name, p_tokens, c_tokens, min_tokens
    # Fallback (should never hit due to weights summing to 1.0)
    return "medium", 200, 500, 400


def build_prompt(n_tokens: int) -> str:
    """Generate filler text that's approximately n_tokens long.

    We use repeated English prose. The approximation is ~0.75 words per token
    for English text with a typical tokenizer. This doesn't need to be exact,
    it just needs to produce prompts of roughly the right size so the gateway's
    token counter (which uses the real tokenizer) sees the expected prompt length.
    """
    words = max(1, int(n_tokens * 0.75))
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * (words // 9 + 1))[:words * 5]


class InferenceUser(HttpUser):
    """A simulated user that sends inference requests to the gateway.

    Locust creates multiple instances of this class (up to --users count).
    Each instance runs send_request() in a loop, waiting a random 0.5-2.0
    seconds between requests. With 50 users, this generates enough concurrent
    load to push the admission controller into the rejection zone.
    """
    # Random pause between requests: simulates realistic user behavior.
    # Without this, each user would fire requests as fast as possible,
    # which isn't realistic and makes the ramp-up less controlled.
    wait_time = between(0.5, 2.0)

    @task
    def send_request(self):
        """Send one inference request to the gateway and measure TTFT.

        This is the core load test logic. Each call:
        1. Picks a random traffic bucket (weighted)
        2. Builds a prompt of the appropriate size
        3. Sends a streaming POST to the gateway
        4. Measures time to first SSE chunk (TTFT)
        5. Consumes remaining chunks (so the gateway releases budget)
        """
        name, p_tokens, c_tokens, min_toks = pick_bucket()
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": build_prompt(p_tokens)}],
            "max_tokens": c_tokens,
            # min_tokens forces the model to generate at least this many tokens.
            # Without this, the model answers our filler prompts in 30-70 tokens
            # regardless of max_tokens, and we never stress KV cache.
            "min_tokens": min_toks,
            # stream=True is CRITICAL for valid TTFT measurement.
            # With stream=False, the server buffers the entire completion
            # and sends it as one response. You'd measure total generation
            # time, not time-to-first-token. For a 2000-token completion,
            # the difference is seconds.
            "stream": True,
        }

        # Start the clock BEFORE sending the request.
        # TTFT = time from request send to first token received.
        request_start = time.monotonic()
        first_token_received = False

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            # stream=True tells the requests library to not buffer the
            # response, so we can iterate chunks as they arrive.
            stream=True,
            # catch_response=True lets us manually mark requests as
            # success or failure (instead of Locust auto-judging by status code).
            catch_response=True,
            # name groups requests in Locust's stats table by bucket.
            # Without this, all requests show as one "POST /v1/chat/completions"
            # entry. With it, we get separate stats for /short, /medium, etc.
            name=f"/{name}",
        ) as resp:
            # 429 (admission rejected) and 503 (queue full/timeout) are
            # EXPECTED behavior when the system is under load. Admission
            # control is working correctly when it rejects excess traffic.
            # If we let Locust count these as failures, the "error rate"
            # metric would be meaningless (it would just reflect load level,
            # not actual problems).
            if resp.status_code in (429, 503):
                resp.success()  # tell Locust this is normal, not an error
                return
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")  # real error
                return

            # Consume the SSE stream chunk by chunk.
            # resp.iter_lines() yields one line at a time from the stream.
            # Locust uses the `requests` library under the hood, so this
            # returns decoded strings (not bytes).
            for chunk in resp.iter_lines():
                if chunk and not first_token_received:
                    # FIRST SSE CHUNK WITH CONTENT = first token!
                    # This is our TTFT measurement point.
                    ttft_ms = (time.monotonic() - request_start) * 1000

                    # Fire a custom Locust event so TTFT appears as a
                    # separate row in Locust's stats table.
                    # request_type="TTFT" creates a new category (alongside
                    # the default "POST" entries). name="ttft/medium" lets
                    # us see TTFT broken down by traffic bucket.
                    self.environment.events.request.fire(
                        request_type="TTFT",
                        name=f"ttft/{name}",
                        response_time=ttft_ms,
                        response_length=0,
                        exception=None,
                        context={},
                    )
                    first_token_received = True

                # IMPORTANT: keep consuming ALL chunks even after measuring TTFT.
                # If we break early, the HTTP connection closes prematurely.
                # The gateway's stream_and_release() generator would error out,
                # and the finally block would still release budget, but the
                # request wouldn't complete cleanly. Consuming all chunks
                # ensures the gateway sees a normal completion and releases
                # budget at the right time.

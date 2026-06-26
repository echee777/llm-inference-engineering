"""Day 23: Dual-CDF plot comparing short-request TTFT in mixed vs isolated runs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
mixed = pd.read_csv(os.path.join(DIR, "run_nochunked_mixed.csv"))
short_only = pd.read_csv(os.path.join(DIR, "run_nochunked_short.csv"))

# Filter mixed to short requests only
mixed_short = mixed[mixed["type"] == "short"]["ttft_ms"].dropna().sort_values().values
isolated_short = short_only["ttft_ms"].dropna().sort_values().values

# Build CDFs
def make_cdf(data):
    sorted_d = np.sort(data)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    return sorted_d, cdf

x_mixed, y_mixed = make_cdf(mixed_short)
x_iso, y_iso = make_cdf(isolated_short)

# p99 values
p99_mixed = np.percentile(mixed_short, 99)
p99_iso = np.percentile(isolated_short, 99)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_iso, y_iso, label=f"Short (isolated) — p99={p99_iso:.0f}ms", color="#2196F3", linewidth=2)
ax.plot(x_mixed, y_mixed, label=f"Short (mixed w/ long) — p99={p99_mixed:.0f}ms", color="#F44336", linewidth=2)

# p99 annotation
ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
ax.annotate("", xy=(p99_mixed, 0.99), xytext=(p99_iso, 0.99),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
ax.text((p99_mixed + p99_iso) / 2, 0.96,
        f"p99 gap: {p99_mixed - p99_iso:.0f}ms ({p99_mixed / p99_iso:.1f}x)",
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

ax.set_xlabel("TTFT (ms)", fontsize=12)
ax.set_ylabel("Cumulative Fraction", fontsize=12)
ax.set_title("Decode Starvation: Short-Request TTFT — Non-Chunked Prefill", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(DIR, "cdf_nochunked.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

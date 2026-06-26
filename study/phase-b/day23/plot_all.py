"""Day 23: Generate all plots for Deliverable #6."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))
DAY22 = os.path.join(os.path.dirname(DIR), "day22")

# Load all four datasets
nochunked_mixed = pd.read_csv(os.path.join(DIR, "run_nochunked_mixed.csv"))
nochunked_short = pd.read_csv(os.path.join(DIR, "run_nochunked_short.csv"))
chunked_mixed = pd.read_csv(os.path.join(DAY22, "run_a_mixed.csv"))
chunked_short = pd.read_csv(os.path.join(DAY22, "run_b_short.csv"))


def make_cdf(data):
    s = np.sort(data)
    return s, np.arange(1, len(s) + 1) / len(s)


# --- Plot 1: Dual CDF, chunked vs non-chunked (short requests in mixed traffic) ---
fig, ax = plt.subplots(figsize=(10, 6))

nc_mixed_short = nochunked_mixed[nochunked_mixed["type"] == "short"]["ttft_ms"].values
c_mixed_short = chunked_mixed[chunked_mixed["type"] == "short"]["ttft_ms"].values
iso_short = nochunked_short["ttft_ms"].values

x1, y1 = make_cdf(iso_short)
x2, y2 = make_cdf(nc_mixed_short)
x3, y3 = make_cdf(c_mixed_short)

p99_iso = np.percentile(iso_short, 99)
p99_nc = np.percentile(nc_mixed_short, 99)
p99_c = np.percentile(c_mixed_short, 99)

ax.plot(x1, y1, label=f"Short isolated — p99={p99_iso:.0f}ms", color="#2196F3", linewidth=2)
ax.plot(x3, y3, label=f"Short mixed (chunked) — p99={p99_c:.0f}ms", color="#FF9800", linewidth=2)
ax.plot(x2, y2, label=f"Short mixed (non-chunked) — p99={p99_nc:.0f}ms", color="#F44336", linewidth=2)

ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
ax.set_xlabel("TTFT (ms)", fontsize=12)
ax.set_ylabel("Cumulative Fraction", fontsize=12)
ax.set_title("Short-Request TTFT: Isolated vs Mixed (Chunked vs Non-Chunked)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, "cdf_all_three.png"), dpi=150)
print("Saved: cdf_all_three.png")
plt.close()


# --- Plot 2: Bimodal TTFT histogram (non-chunked mixed run, all request types) ---
fig, ax = plt.subplots(figsize=(10, 6))

short_ttft = nochunked_mixed[nochunked_mixed["type"] == "short"]["ttft_ms"].values
long_ttft = nochunked_mixed[nochunked_mixed["type"] == "long"]["ttft_ms"].values

bins = np.linspace(0, max(nochunked_mixed["ttft_ms"].max(), 2500), 80)
ax.hist(short_ttft, bins=bins, alpha=0.7, label=f"Short requests (n={len(short_ttft)})", color="#2196F3")
ax.hist(long_ttft, bins=bins, alpha=0.7, label=f"Long requests (n={len(long_ttft)})", color="#F44336")

ax.set_xlabel("TTFT (ms)", fontsize=12)
ax.set_ylabel("Request Count", fontsize=12)
ax.set_title("Bimodal TTFT Distribution — Non-Chunked Mixed Traffic", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, "histogram_bimodal.png"), dpi=150)
print("Saved: histogram_bimodal.png")
plt.close()


# --- Plot 3: Chunked prefill comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Short-request TTFT p99
labels = ["Isolated", "Mixed\n(chunked)", "Mixed\n(non-chunked)"]
values = [p99_iso, p99_c, p99_nc]
colors = ["#2196F3", "#FF9800", "#F44336"]
bars = axes[0].bar(labels, values, color=colors, width=0.6)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f"{val:.0f}ms", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("TTFT p99 (ms)", fontsize=12)
axes[0].set_title("Short-Request TTFT p99", fontsize=13)
axes[0].grid(True, alpha=0.3, axis="y")

# Right: Degradation factors
labels2 = ["Non-chunked\nvs isolated", "Chunked\nvs isolated"]
factors = [p99_nc / p99_iso, p99_c / p99_iso]
colors2 = ["#F44336", "#FF9800"]
bars2 = axes[1].bar(labels2, factors, color=colors2, width=0.5)
for bar, val in zip(bars2, factors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}x", ha="center", fontsize=11, fontweight="bold")
axes[1].axhline(y=1.0, color="#2196F3", linestyle="--", linewidth=1.5, label="Baseline (1.0x)")
axes[1].set_ylabel("Degradation Factor (p99 ratio)", fontsize=12)
axes[1].set_title("Interference Penalty", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(DIR, "bar_comparison.png"), dpi=150)
print("Saved: bar_comparison.png")
plt.close()

print("\nAll plots generated.")

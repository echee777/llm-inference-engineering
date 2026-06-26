"""
Day 24 — Generate the cliff plots from day24_cliff_summary.csv.

Outputs two figures (syllabus Step 3):
  1. cliff_primary.png   — KV util %  vs  TTFT p50 / p95 / p99
                            with shaded variance bands in the transition zone
                            and an explicit cliff-point marker.
  2. cliff_secondary.png — KV util %  vs  preemption rate (events/min)

The cliff point is detected as the lowest KV-util bucket whose
divergence ratio (p99/p50) crosses 2.0 — matches the syllabus
definition.
"""

import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(DIR, "day24_cliff_combined_summary.csv")

DIVERGENCE_THRESHOLD = 2.0


def load_summary() -> pd.DataFrame:
    """Read the summary CSV and coerce numeric columns."""
    df = pd.read_csv(SUMMARY_CSV)
    numeric_cols = [
        "concurrency",
        "repeat",
        "completed",
        "errors",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "ttft_max_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "divergence_ratio",
        "throughput_tps",
        "kv_util_pct_mean",
        "kv_util_pct_max",
        "queue_depth_mean",
        "queue_depth_max",
        "queue_growth_frac",
        "preemption_rate_per_min",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_by_concurrency(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeats into mean ± std rows, sorted by concurrency."""
    rows = []
    for c, group in df.groupby("concurrency"):
        n = len(group)
        row = {
            "concurrency": c,
            "n_runs": n,
            "kv_util_pct_mean": group["kv_util_pct_mean"].mean(),
            "kv_util_pct_max": group["kv_util_pct_max"].max(),
            "ttft_p50_ms_mean": group["ttft_p50_ms"].mean(),
            "ttft_p50_ms_std": group["ttft_p50_ms"].std() if n > 1 else 0.0,
            "ttft_p95_ms_mean": group["ttft_p95_ms"].mean(),
            "ttft_p95_ms_std": group["ttft_p95_ms"].std() if n > 1 else 0.0,
            "ttft_p99_ms_mean": group["ttft_p99_ms"].mean(),
            "ttft_p99_ms_std": group["ttft_p99_ms"].std() if n > 1 else 0.0,
            "divergence_ratio_mean": group["divergence_ratio"].mean(),
            "preemption_rate_per_min_mean": group["preemption_rate_per_min"].mean(),
            "queue_depth_mean_mean": group["queue_depth_mean"].mean(),
            "queue_growth_frac_mean": group["queue_growth_frac"].mean(),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("kv_util_pct_mean").reset_index(drop=True)
    return out


def find_cliff_point(agg: pd.DataFrame) -> float | None:
    """Return the KV utilization at the first point where divergence ratio
    exceeds the syllabus-defined threshold (2.0)."""
    above = agg[agg["divergence_ratio_mean"] >= DIVERGENCE_THRESHOLD]
    if above.empty:
        return None
    return float(above.iloc[0]["kv_util_pct_mean"])


def plot_primary(agg: pd.DataFrame, cliff_kv: float | None) -> str:
    """Primary cliff curve: KV util vs TTFT p50/p95/p99 with variance bands."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = agg["kv_util_pct_mean"].values

    # Lines (means)
    ax.plot(
        x, agg["ttft_p50_ms_mean"], marker="o",
        color="#2196F3", linewidth=2, label="TTFT p50",
    )
    ax.plot(
        x, agg["ttft_p95_ms_mean"], marker="s",
        color="#FF9800", linewidth=2, label="TTFT p95",
    )
    ax.plot(
        x, agg["ttft_p99_ms_mean"], marker="^",
        color="#F44336", linewidth=2, label="TTFT p99",
    )

    # Variance bands where we have repeats
    repeat_mask = agg["n_runs"] > 1
    if repeat_mask.any():
        xr = agg.loc[repeat_mask, "kv_util_pct_mean"].values
        for col_mean, col_std, color in [
            ("ttft_p50_ms_mean", "ttft_p50_ms_std", "#2196F3"),
            ("ttft_p95_ms_mean", "ttft_p95_ms_std", "#FF9800"),
            ("ttft_p99_ms_mean", "ttft_p99_ms_std", "#F44336"),
        ]:
            mean = agg.loc[repeat_mask, col_mean].values
            std = agg.loc[repeat_mask, col_std].values
            ax.fill_between(xr, mean - std, mean + std, alpha=0.18, color=color)

    # Cliff point marker
    if cliff_kv is not None:
        ax.axvline(
            cliff_kv,
            color="#555",
            linestyle="--",
            alpha=0.7,
            linewidth=1.4,
        )
        ymax = ax.get_ylim()[1]
        ax.text(
            cliff_kv + 0.5,
            ymax * 0.92,
            f"cliff at {cliff_kv:.1f}% KV\n(div ratio ≥ 2.0)",
            fontsize=10,
            fontweight="bold",
            color="#222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#FFF8DC",
                edgecolor="#888",
            ),
        )

    ax.set_xlabel("KV Cache Utilization (%)", fontsize=12)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title(
        "Latency vs KV Utilization Cliff — Qwen2.5-3B-Instruct on T4",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DIR, "cliff_primary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_secondary(agg: pd.DataFrame, cliff_kv: float | None) -> str:
    """Secondary plot: KV util vs preemption rate."""
    fig, ax = plt.subplots(figsize=(11, 6))

    x = agg["kv_util_pct_mean"].values
    y = agg["preemption_rate_per_min_mean"].values

    ax.plot(x, y, marker="o", color="#7B1FA2", linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.15, color="#7B1FA2")

    if cliff_kv is not None:
        ax.axvline(
            cliff_kv,
            color="#555",
            linestyle="--",
            alpha=0.7,
            linewidth=1.4,
        )
        ymax = ax.get_ylim()[1]
        ax.text(
            cliff_kv + 0.5,
            ymax * 0.85,
            f"cliff at {cliff_kv:.1f}% KV",
            fontsize=10,
            fontweight="bold",
            color="#222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#FFF8DC",
                edgecolor="#888",
            ),
        )

    ax.set_xlabel("KV Cache Utilization (%)", fontsize=12)
    ax.set_ylabel("Preemption Rate (events/min)", fontsize=12)
    ax.set_title(
        "Preemption Onset vs KV Utilization — same workload",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DIR, "cliff_secondary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def print_summary_table(agg: pd.DataFrame, cliff_kv: float | None) -> None:
    """Pretty-print the aggregated table to stdout for the deliverable."""
    print("\n" + "=" * 100)
    print("AGGREGATED CLIFF SWEEP")
    print("=" * 100)
    header = (
        f"{'Conc':>5} {'Runs':>5} {'KV%mean':>8} {'KV%max':>8} "
        f"{'p50':>7} {'p95':>7} {'p99':>7} {'div':>5} "
        f"{'preempt/m':>10} {'qD':>5} {'qGr':>5}"
    )
    print(header)
    print("-" * 100)
    for _, r in agg.iterrows():
        print(
            f"{int(r['concurrency']):>5} "
            f"{int(r['n_runs']):>5} "
            f"{r['kv_util_pct_mean']:>8.2f} "
            f"{r['kv_util_pct_max']:>8.2f} "
            f"{r['ttft_p50_ms_mean']:>7.0f} "
            f"{r['ttft_p95_ms_mean']:>7.0f} "
            f"{r['ttft_p99_ms_mean']:>7.0f} "
            f"{r['divergence_ratio_mean']:>5.2f} "
            f"{r['preemption_rate_per_min_mean']:>10.1f} "
            f"{r['queue_depth_mean_mean']:>5.1f} "
            f"{r['queue_growth_frac_mean']:>5.2f}"
        )
    print("=" * 100)
    if cliff_kv is not None:
        print(f"\nCliff point detected at KV utilization = {cliff_kv:.1f}%")
        print(f"  (first point with divergence ratio >= {DIVERGENCE_THRESHOLD})")
    else:
        print("\nNo cliff detected in the swept range.")


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise SystemExit(f"Summary CSV not found: {SUMMARY_CSV}")
    df = load_summary()
    agg = aggregate_by_concurrency(df)
    cliff_kv = find_cliff_point(agg)

    print_summary_table(agg, cliff_kv)

    p1 = plot_primary(agg, cliff_kv)
    p2 = plot_secondary(agg, cliff_kv)

    print(f"\nWrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()

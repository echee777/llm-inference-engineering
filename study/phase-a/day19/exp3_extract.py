"""Extract TTFT p50/p95/p99 from Experiment 3 Locust CSV files.

Reads results/exp3_*_stats.csv and prints a summary table.
Run after all exp3 load levels complete.
"""
import csv
import glob
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    pattern = os.path.join(RESULTS_DIR, "exp3_*_stats.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found matching {pattern}")
        return

    print(f"{'Label':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12}")
    print("-" * 48)

    for f in files:
        label = os.path.basename(f).replace("exp3_", "").replace("_stats.csv", "")
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("Name", "")
                # Look for the aggregated TTFT row
                if "Aggregated" in name:
                    p50 = row.get("50%", row.get("Median Response Time", "?"))
                    p95 = row.get("95%", "?")
                    p99 = row.get("99%", "?")
                    print(f"{label:<12} {p50:<12} {p95:<12} {p99:<12}")
                    break


if __name__ == "__main__":
    main()

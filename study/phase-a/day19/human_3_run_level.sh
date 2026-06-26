#!/usr/bin/env bash
# Run one Locust load level for Experiment 3.
# Usage: ./human_3_run_level.sh <users> <label>
# Example: ./human_3_run_level.sh 12 25pct
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <users> <label>"
    echo "Example: $0 12 25pct"
    exit 1
fi

USERS=$1
LABEL=$2

echo "=== Exp3: Load level ${LABEL} (${USERS} users, 6 min) ==="
mkdir -p results

source ~/venv-locust/bin/activate

locust -f exp3_locustfile.py \
    --host http://localhost:8001 \
    --users "$USERS" \
    --spawn-rate 5 \
    --run-time 6m \
    --headless \
    --csv "results/exp3_${LABEL}" \
    --html "results/exp3_${LABEL}_report.html" \
    --print-stats

echo ""
echo "--- Metrics after ${LABEL} ---"
./human_3_check_metrics.sh | tee "results/exp3_${LABEL}_metrics.txt"

echo ""
echo "=== Load level ${LABEL} complete ==="
echo "CSVs: results/exp3_${LABEL}_stats.csv"
echo "HTML:  results/exp3_${LABEL}_report.html"
echo "Metrics: results/exp3_${LABEL}_metrics.txt"

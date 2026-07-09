#!/bin/bash
# Run static thread allocation profiling across all datasets.
#
# Usage:
#   ./calibration.sh [--dry-run] [--dataset deep100m|sift100m|spacev100m]
#
# Runs each dataset sequentially.  Results are written to:
#   <PATH>/results/threading/<dataset>_t32/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT
CONFIG_DIR="$REPO_ROOT/configs"
PROFILER="$REPO_ROOT/src/thread_calibration/static_profiler.py"

TOTAL_THREADS=32
TARGET_RECALLS="<RECALL_SWEEP>"  # e.g. "70 80 90 95 98"
SLEEP_BETWEEN=7
# Thread step: 4 → coarse sweep (7 combos: 4/28 8/24 12/20 16/16 20/12 24/8 28/4)
#              1 → full sweep (25 combos, ~3× more runs)
THREAD_STEP=4
DEVICE="nvme0n1"   # block device to monitor in /proc/diskstats

DRY_RUN=""
DATASETS=("deep100m" "sift100m" "spacev100m" "deep300m")

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN="--dry_run" ;;
        --dataset)      shift; DATASETS=("$1") ;;
        --device)       shift; DEVICE="$1" ;;
        --thread_step)  shift; THREAD_STEP="$1" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "QUINN Static Thread Profiling"
echo "  total_threads = $TOTAL_THREADS"
echo "  recalls       = $TARGET_RECALLS"
echo "  datasets      = ${DATASETS[*]}"
echo "  device        = $DEVICE"
echo "============================================================"

for DATASET in "${DATASETS[@]}"; do
    CONFIG="$CONFIG_DIR/${DATASET}/${DATASET}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "[SKIP] Config not found: $CONFIG"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Dataset: $DATASET"
    echo "------------------------------------------------------------"

    python "$PROFILER" \
        --config "$CONFIG" \
        --total_threads "$TOTAL_THREADS" \
        --target_recalls $TARGET_RECALLS \
        --thread_step "$THREAD_STEP" \
        --sleep "$SLEEP_BETWEEN" \
        --device "$DEVICE" \
        $DRY_RUN

    echo ""
    echo "Sleeping 10s before next dataset..."
    sleep 10
done

echo ""
echo "============================================================"
echo "All profiling complete."
echo "Results in: $REPO_ROOT/result/profiling/threading/"
echo "============================================================"

# Summarize best configs across all datasets
echo ""
echo "Best configs summary:"
python - <<'EOF'
import csv
import os
from pathlib import Path

results_dir = Path(os.environ['REPO_ROOT']) / 'result' / 'profiling' / 'threading'
# This inline script is executed after all datasets finish
for best_csv in sorted(results_dir.glob('*/best_configs.csv')):
    print(f"\n=== {best_csv.parent.name} ===")
    with open(best_csv) as f:
        rows = list(csv.DictReader(f))
    print(f"{'recall':>8}  {'threadS':>8}  {'threadD':>8}  {'achieved':>10}  {'QPS':>10}  {'p99_ms':>9}")
    print('-' * 65)
    for r in rows:
        ar = f"{float(r['achieved_recall']):.2f}" if r['achieved_recall'] else 'N/A'
        qps = f"{float(r['qps']):.1f}" if r['qps'] else 'N/A'
        p99 = f"{float(r['p99_latency_ms']):.1f}" if r['p99_latency_ms'] else 'N/A'
        print(f"{r['target_recall']:>8}  {r['threadS']:>8}  {r['threadD']:>8}  "
              f"{ar:>10}  {qps:>10}  {p99:>9}")
EOF

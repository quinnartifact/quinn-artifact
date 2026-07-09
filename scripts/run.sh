#!/bin/bash
# Example final benchmark driver across configured datasets and recalls,
# at a fixed alpha of 0.6. N_RUNS controls how many repetitions are executed
# locally (default 1; override with e.g. N_RUNS=3 scripts/run.sh).
#
# Results are saved under:
#   benchmark_final/<dataset>/alpha_0.6/recall_<r>/run_<n>/

set -euo pipefail

QUINN="$(cd "$(dirname "$0")/.." && pwd)"
CONTROLLER="$QUINN/src/controller/controller.py"
PYTHON=python3

ALPHA="0.6"
DATASETS=(sift100m spacev100m deep100m deep300m)
RECALLS=("<RECALL_SWEEP>")  # e.g. 70 75 80 85 90 95 96 97 98 99 99.5
N_RUNS="${N_RUNS:-1}"
OUT_BASE="$QUINN/benchmark_final"

# ---------------------------------------------------------------------------
# make_config <base_yaml> <benchmark_dir> <tmp_out>
#   Writes a copy of base_yaml with output.benchmark_dir replaced.
# ---------------------------------------------------------------------------
make_config() {
    local base="$1" bench_dir="$2" out="$3"
    $PYTHON - <<EOF
import yaml, sys
with open("$base") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("output", {})["benchmark_dir"] = "$bench_dir"
with open("$out", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)
EOF
}

# ---------------------------------------------------------------------------
# run_one <dataset> <recall> <run_idx>
# ---------------------------------------------------------------------------
run_one() {
    local dataset="$1" recall="$2" run="$3"
    local bench_dir="$OUT_BASE/$dataset/alpha_$ALPHA/recall_$recall/run_$run"
    local result_dir="$QUINN/result/${dataset}_final/alpha_${ALPHA}/recall_${recall}/run_${run}"
    local base_cfg="$QUINN/configs/${dataset}/${dataset}.yaml"
    local model_dir="$QUINN/model/$dataset/alpha_$ALPHA"
    local tmp_cfg
    tmp_cfg=$(mktemp /tmp/quinn_XXXXXX.yaml)

    mkdir -p "$bench_dir" "$result_dir"
    make_config "$base_cfg" "$bench_dir" "$tmp_cfg"

    echo "  [${dataset}] alpha=${ALPHA} recall=${recall} run=${run}"
    if $PYTHON "$CONTROLLER" \
            --config    "$tmp_cfg" \
            --target_recall "$recall" \
            --model_dir "$model_dir" \
            --output_dir "$result_dir"; then
        echo "  -> OK: $bench_dir"
    else
        echo "  -> FAILED (continuing)"
    fi

    rm -f "$tmp_cfg"
    sleep 5
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
total_runs=$(( ${#DATASETS[@]} * ${#RECALLS[@]} * N_RUNS ))
echo "================================================================"
echo "QUINN Final Benchmark"
echo "  alpha_$ALPHA × ${#RECALLS[@]} recalls × $N_RUNS runs × ${#DATASETS[@]} datasets = $total_runs runs"
echo "  Output : $OUT_BASE"
echo "================================================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $dataset ==="
    for recall in "${RECALLS[@]}"; do
        for run in $(seq 1 $N_RUNS); do
            run_one "$dataset" "$recall" "$run"
        done
    done
done

echo ""
echo "================================================================"
echo "All experiments completed."
echo "Results in: $OUT_BASE"
echo "================================================================"

#!/usr/bin/env bash
# Offline training pipeline: profiling → oracle labels → GBDT model.
# Run from the QUINN root: bash scripts/train.sh

set -euo pipefail

QUINN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VECTORDB_DIR="<PATH>"  # root of the external vectordb repo (query/GT/centroid data + baseline stats)
DATASET="<DATASET>"  # e.g. deep100m

# ── Data paths ────────────────────────────────────────────────────────────────
DISKANN_RESULT_DIR="${VECTORDB_DIR}/diskann/${DATASET}/other/result-learning"
DISKANN_STATS_DIR="${VECTORDB_DIR}/diskann/${DATASET}/stats_learning"
SPANN_RESULT_DIR="${VECTORDB_DIR}/spann/${DATASET}/result_learning"
SPANN_STATS_DIR="${VECTORDB_DIR}/spann/${DATASET}/stats_learning"
GT_FILE="<PATH>"  # ground-truth file for the offline training query sample
QUERY_FILE="<PATH>"  # offline training query sample (e.g. a held-out learn-set)
CENTROID_FILE="${VECTORDB_DIR}/spann/${DATASET}/index/SPTAGHeadVectors.bin"

PROFILING_CSV="${QUINN_DIR}/data/${DATASET}/profiling_results.csv"
ORACLE_CSV="${QUINN_DIR}/data/${DATASET}/oracle_labels_optimized.csv"
MODEL_DIR="${QUINN_DIR}/model/${DATASET}"

cd "${QUINN_DIR}"

echo "========================================================================"
echo "Step 1/3 — Offline profiling"
echo "========================================================================"
# b_s_start/end/step: SPANN nprobe sweep — how many posting lists SPANN
#   probes per query, scanned from start to end in steps of step.
# b_d_start/end/step: DiskANN L sweep — DiskANN's beam-search width per
#   query, scanned the same way (b_d_start=0 means "DiskANN unused").
# Pick a range wide enough to cover the recall/latency points you care
# about; wider ranges and smaller steps cost more offline profiling time.
python src/offline_training/offline_profiling.py \
    --diskann_result_dir "${DISKANN_RESULT_DIR}" \
    --diskann_stats_dir  "${DISKANN_STATS_DIR}" \
    --spann_result_dir   "${SPANN_RESULT_DIR}" \
    --spann_stats_dir    "${SPANN_STATS_DIR}" \
    --gt_file            "${GT_FILE}" \
    --output             "${PROFILING_CSV}" \
    --b_s_start "<B_S_START>" --b_s_end "<B_S_END>" --b_s_step "<B_S_STEP>" \
    --b_d_start "<B_D_START>" --b_d_end "<B_D_END>" --b_d_step "<B_D_STEP>" \
    --n_jobs "<TUNE_PER_HARDWARE>"  # e.g. 32

echo ""
echo "========================================================================"
echo "Step 2/3 — Oracle label generation"
echo "========================================================================"
# --target_recalls: comma-separated list, e.g. "80,85,90,95,97,99"
python src/offline_training/generate_oracle_labels_optimized.py \
    --profiling_csv  "${PROFILING_CSV}" \
    --query_file     "${QUERY_FILE}" \
    --centroid_file  "${CENTROID_FILE}" \
    --target_recalls "<RECALL_SWEEP>" \
    --output         "${ORACLE_CSV}"

echo ""
echo "========================================================================"
echo "Step 3/3 — GBDT training"
echo "========================================================================"
python src/offline_training/train_optimized_gbdt_regression.py \
    --data       "${ORACLE_CSV}" \
    --output_dir "${MODEL_DIR}"

echo ""
echo "========================================================================"
echo "Training complete. Models saved to ${MODEL_DIR}"
echo "========================================================================"

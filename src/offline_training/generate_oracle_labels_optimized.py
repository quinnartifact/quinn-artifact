#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/offline_training/generate_oracle_labels_optimized.py — Generate oracle labels (optimized version)

From the profiling CSV produced by offline_profiling.py, for each
(query, target_recall) combination, find the (b_S, b_D) combination with the
smallest total I/O that still meets the recall target, to use as the
LightGBM training label.

Differences from the old version (generate_oracle_labels.py):
  - Uses util.feature_utils (SPTAG) to compute features, keeping it fully
    consistent with controller.py's inference path
  - Supports processing multiple target_recall values in parallel
    (--target_recalls argument)
  - Outputs a complete CSV with features included, ready to feed directly
    into train_optimized_gbdt.py

Output CSV columns:
  qid, target_recall, b_S_optimal, b_D_optimal, io_total_optimal,
  recall_achieved, d1, d1_d2_ratio, q_dim0, ..., q_dimK

Usage:
  python generate_oracle_labels_optimized.py \\
    --profiling_data ./data/sift100m/profiling_results_full.csv \\
    --query_file ./data/sift100m/query.fvecs \\
    --spann_index ./data/sift100m/spann_index \\
    --output ./data/sift100m/oracle_labels.csv \\
    --target_recalls 80 90 95 98
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add path to src to invoke util.feature_utils
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from util.feature_utils import load_fvecs_or_fbin, compute_centroid_features


def find_optimal_budget(profiling_df, target_recall_pct):
    """
    Find optimal (min IO) budget for each query satisfying target_recall_pct
    """
    results = []

    # Get unique QIDs
    qids = profiling_df['qid'].unique()

    # Pre-filter for efficiency
    # Group by qid
    grouped = profiling_df.groupby('qid')
    
    for qid, group in tqdm(grouped, desc=f"  Mining Optimal Budgets (Recall {target_recall_pct})"):
        valid = group[group['recall'] >= target_recall_pct]
        
        if not valid.empty:
            # Case 1: Target recall is met. Pick budget with minimum total IO.
            best_idx = valid['io_total'].idxmin()
            best_row = valid.loc[best_idx]
            target_met = True
        else:
            # Case 2: Target recall cannot be met. 
            # Find the closest possible recall (which is the maximum recall available).
            max_recall = group['recall'].max()
            closest_rows = group[group['recall'] == max_recall]
            
            # Among those with the same max recall, pick the one with minimum IO.
            best_idx = closest_rows['io_total'].idxmin()
            best_row = closest_rows.loc[best_idx]
            target_met = False
            
        results.append({
            'qid': int(qid),
            'b_S_optimal': int(best_row['b_S']),
            'b_D_optimal': int(best_row['b_D']),
            'min_io': float(best_row['io_total']),
            'achieved_recall': float(best_row['recall']),
            'target_met': target_met
        })
        
    df_res = pd.DataFrame(results)
    
    # [FIX] Filter out rows where target is NOT met and achieved recall is too far from target.
    # If target is 99% but we only got 60%, we should NOT train on this label saying "this is the best for 99%".
    # Tolerance: 1.0 (1%)
    if not df_res.empty:
        # Keep if target met OR achieved recall is within 1.0 of target
        # For targets < 90, we can be more lenient? No, strict is better for model signal.
        mask = (df_res['target_met']) | (df_res['achieved_recall'] >= target_recall_pct - 1.0)
        df_res = df_res[mask]
        
    return df_res


def main():
    parser = argparse.ArgumentParser(description='Generate Oracle Labels (Optimized)')

    parser.add_argument('--profiling_csv', required=True, help='Profiling Results CSV')
    parser.add_argument('--query_file', required=True, help='Query vector file')
    parser.add_argument('--centroid_file', required=True, help='Centroid vector file (SPTAGHeadVectors.bin)')

    parser.add_argument('--target_recalls', type=str, required=True,
                       help='Target recall list (comma separated %%), e.g. "80,85,90,95,97,99"')

    parser.add_argument('--output', required=True, help='Output CSV')

    args = parser.parse_args()

    print("=" * 80)
    print("Generate Oracle Labels (Optimized with SPTAG Features)")
    print("=" * 80)

    # 1. Load Profiling Data
    print(f"\n[1/4] Loading Profiling Data: {args.profiling_csv}")
    profiling_df = pd.read_csv(args.profiling_csv)
    print(f"  Rows: {len(profiling_df):,}")

    # 2. Load Queries & Centroids
    print(f"\n[2/4] Loading Vectors via feature_utils...")
    queries = load_fvecs_or_fbin(args.query_file)
    centroids = load_fvecs_or_fbin(args.centroid_file)
    print(f"  Queries: {queries.shape}")
    print(f"  Centroids: {centroids.shape}")

    # FILTER QIDS
    max_qid = len(queries)
    print(f"  Filtering profiling data for QID < {max_qid}...")
    profiling_df = profiling_df[profiling_df['qid'] < max_qid]
    print(f"  Rows after filtering: {len(profiling_df):,}")

    # 3. Compute Features (Consistent Logic)
    print(f"\n[3/4] Computing Features (using SPTAG logic)...")
    d1, ratios, _ = compute_centroid_features(queries, centroids, centroid_file=args.centroid_file)
    
    # 4. Generate Labels for each Recall
    target_recalls_pct = [float(x) for x in args.target_recalls.split(',')]
    target_recalls = [x / 100.0 for x in target_recalls_pct]

    print(f"\n[4/4] Generating Labels for Recalls: {target_recalls_pct}")
    
    all_dfs = []
    
    for tr_pct, tr_val in zip(target_recalls_pct, target_recalls):
        opt_df = find_optimal_budget(profiling_df, tr_val)
        
        # Merge features
        # opt_df has 'qid'
        # features are index-aligned with qid (assuming qid 0..N-1)
        
        # Add features column-wise
        opt_df['target_recall'] = tr_pct
        opt_df['d1'] = opt_df['qid'].map(lambda x: d1[x])
        opt_df['d1_d2_ratio'] = opt_df['qid'].map(lambda x: ratios[x])
        
        # Add query dimensions (Optional, can be heavy for large N)
        # For simplicity and speed, let's vectorise this merge
        # Create a DF for features
        # feat_df = pd.DataFrame(queries, columns=[f'q_dim{i}' for i in range(queries.shape[1])])
        # feat_df['qid'] = np.arange(len(queries))
        # But merging 128 cols for every recall row is heavy.
        # Let's map dynamically? Or just create feature DF once.
        
        all_dfs.append(opt_df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Now add query vectors (broadcast)
    print("  Attaching Query Vectors...")
    # Convert queries to DF
    q_dims = pd.DataFrame(queries, columns=[f'q_dim{i}' for i in range(queries.shape[1])])
    q_dims['qid'] = np.arange(len(queries))
    
    # Merge
    final_df = pd.merge(final_df, q_dims, on='qid', how='left')
    
    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    
    print(f"\n  Saved to: {out_path}")
    print(f"  Total Labels: {len(final_df):,}")
    print(f"  d1 Mean: {final_df['d1'].mean():.2f} (Should be ~553 for SIFT100M with SPTAG)")

if __name__ == "__main__":
    main()

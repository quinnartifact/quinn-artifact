#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/offline_training/offline_profiling.py — Offline profiling: scans all (b_S, b_D) combinations

Reads the pre-computed DiskANN and SPANN search results for different budget
settings, and for each query and each (b_S, b_D) budget combination computes:
  - Recall@100: overlap ratio with ground truth
  - io_spann   : total posting-list length scanned by SPANN (controlled by nprobe)
  - io_diskann : DiskANN's I/O count (controlled by L, the beam search width)
  - io_total   : io_spann + io_diskann

Output format (CSV):
  qid, b_S, b_D, recall, io_spann, io_diskann, io_total

This profiling CSV is the starting point for all downstream training steps:
  generate_oracle_labels_optimized.py → oracle labels
  train_optimized_gbdt.py            → GBDT model

Usage:
  python offline_profiling.py \\
    --diskann_result_dir /path/to/diskann/result \\
    --diskann_stats_dir  /path/to/diskann/stats \\
    --spann_result_dir   /path/to/spann/result \\
    --spann_stats_dir    /path/to/spann/stats \\
    --gt_file /path/to/ground_truth.ivecs \\
  --b_s_start 10 \
  --b_s_end 40 \
  --b_s_step 5 \
  --b_d_start 0 \
  --b_d_end 150 \
  --b_d_step 10 \
  --output ./data/sift100m/profiling_results_full.csv
"""

import argparse
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_fbin(path):
    """Read a .fbin-format file (query vectors)"""
    with open(path, 'rb') as f:
        npts = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        vectors = np.fromfile(f, dtype='<f4', count=npts * dim)
        return vectors.reshape(npts, dim)


def read_ids_bin(path):
    """Read a DiskANN result file (binary format with uint32 IDs)"""
    with open(path, 'rb') as f:
        nq, k = struct.unpack('<II', f.read(8))
        ids = np.fromfile(f, dtype='<u4', count=nq * k)
        return nq, k, ids.reshape(nq, k)


def read_spann_result(path):
    """Read a SPANN result file (contains IDs and distances)"""
    with open(path, 'rb') as f:
        nq, k = struct.unpack('<II', f.read(8))
        ids = np.empty((nq, k), dtype='<u4')
        dists = np.empty((nq, k), dtype='<f4')
        for i in range(nq):
            buf = f.read(k * 8)
            arr = np.frombuffer(buf, dtype=[('id', '<u4'), ('dist', '<f4')], count=k)
            ids[i] = arr['id']
            dists[i] = arr['dist']
    return nq, k, ids, dists


def read_gt_bin(path):
    """Read binary ground truth file (int32)
    Format: [num_queries (int32)] [K (int32)] [ids (nq * K * int32)]
    """
    with open(path, 'rb') as f:
        nq = struct.unpack('<i', f.read(4))[0]
        k = struct.unpack('<i', f.read(4))[0]
        # Read all data
        ids = np.fromfile(f, dtype='<i4', count=nq * k)
        return ids.reshape(nq, k)

def read_ivecs(path):
    """Read .ivecs file format"""
    rows = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(4)
            if not header:
                break
            dim = struct.unpack('<i', header)[0]
            # .ivecs data is int32
            vec = np.frombuffer(f.read(4 * dim), dtype='<i4')
            rows.append(vec)
    return np.vstack(rows)

def load_ground_truth(path):
    path_str = str(path)
    if path_str.endswith('.ivecs'):
        return read_ivecs(path_str)
    else:
        # Default to binary format for .bin files
        return read_gt_bin(path_str)


def compute_recall(result_ids, gt_ids, gt_k=100):
    """
    Compute recall@gt_k

    Args:
        result_ids: shape (nq, k) - the search result IDs
        gt_ids: shape (nq, gt_k) - ground truth IDs
        gt_k: consider the first k ground-truth results

    Returns:
        recalls: shape (nq,) - recall for each query
    """
    nq = result_ids.shape[0]
    recalls = np.zeros(nq)

    for q in range(nq):
        # Filter valid IDs (exclude 0xFFFFFFFF)
        valid_mask = result_ids[q] != 0xFFFFFFFF
        result_set = set(int(x) for x in result_ids[q][valid_mask])
        gt_set = set(int(x) for x in gt_ids[q, :gt_k] if int(x) >= 0)

        if len(gt_set) > 0:
            recalls[q] = len(result_set & gt_set) / len(gt_set)
        else:
            recalls[q] = 0

    return recalls


def compute_merged_recall(spann_ids, diskann_ids, gt_ids, gt_k=100):
    """
    Compute recall@gt_k after merging the SPANN and DiskANN results

    Args:
        spann_ids: shape (nq, k_s) - SPANN search results
        diskann_ids: shape (nq, k_d) - DiskANN search results
        gt_ids: shape (nq, gt_k) - ground truth IDs
        gt_k: consider the first k ground-truth results

    Returns:
        recalls: shape (nq,) - recall for each query
    """
    nq = spann_ids.shape[0]
    recalls = np.zeros(nq)

    for q in range(nq):
        # SPANN results
        spann_valid_mask = spann_ids[q] != 0xFFFFFFFF
        spann_set = set(int(x) for x in spann_ids[q][spann_valid_mask])

        # DiskANN results
        diskann_valid_mask = diskann_ids[q] != 0xFFFFFFFF
        diskann_set = set(int(x) for x in diskann_ids[q][diskann_valid_mask])

        # Merged results
        merged_set = spann_set | diskann_set

        # Ground truth
        gt_set = set(int(x) for x in gt_ids[q, :gt_k] if int(x) >= 0)

        if len(gt_set) > 0:
            recalls[q] = len(merged_set & gt_set) / len(gt_set)
        else:
            recalls[q] = 0

    return recalls


def load_io_stats(csv_path):
    """
    Read an I/O stats CSV file

    Returns:
        dict: {qid: n_ios}
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['qid'], df['n_ios']))


# --- Global variables for multiprocess workers (relies on Linux fork copy-on-write) ---
_G_SPANN_RESULTS = None
_G_DISKANN_RESULTS = None
_G_GT_IDS = None
_G_SPANN_IO_STATS = None
_G_DISKANN_IO_STATS = None

def init_worker(spann_res, diskann_res, gt_ids, spann_io, diskann_io):
    global _G_SPANN_RESULTS, _G_DISKANN_RESULTS, _G_GT_IDS, _G_SPANN_IO_STATS, _G_DISKANN_IO_STATS
    _G_SPANN_RESULTS = spann_res
    _G_DISKANN_RESULTS = diskann_res
    _G_GT_IDS = gt_ids
    _G_SPANN_IO_STATS = spann_io
    _G_DISKANN_IO_STATS = diskann_io

def compute_combination_task(args_tuple):
    b_s, b_d, gt_k = args_tuple
    
    gt_ids = _G_GT_IDS
    nq = min(gt_ids.shape[0], _G_SPANN_RESULTS[b_s].shape[0], _G_DISKANN_RESULTS[b_d].shape[0])

    spann_ids = _G_SPANN_RESULTS[b_s][:nq]
    diskann_ids = _G_DISKANN_RESULTS[b_d][:nq]
    gt_ids = gt_ids[:nq]
    
    # Compute recall
    recalls = compute_merged_recall(spann_ids, diskann_ids, gt_ids, gt_k=gt_k)
    
    # Assemble the results
    res_list = []
    spann_io = _G_SPANN_IO_STATS[b_s]
    diskann_io = _G_DISKANN_IO_STATS[b_d]
    
    for q in range(nq):
        io_s = spann_io.get(q, 0)
        io_d = diskann_io.get(q, 0)
        res_list.append({
            'qid': q,
            'b_S': b_s,
            'b_D': b_d,
            'recall': recalls[q],
            'io_spann': io_s,
            'io_diskann': io_d,
            'io_total': io_s + io_d
        })
    return res_list


def main():
    parser = argparse.ArgumentParser(description='Offline Profiling for DiskANN + SPANN Multi-Index')

    # Data paths
    parser.add_argument('--diskann_result_dir', required=True)
    parser.add_argument('--diskann_stats_dir', required=True)
    parser.add_argument('--spann_result_dir', required=True)
    parser.add_argument('--spann_stats_dir', required=True)
    parser.add_argument('--gt_file', required=True)

    # Budget ranges — pick a range wide enough to cover the recall/latency
    # points you care about; there's no dataset-agnostic default.
    parser.add_argument('--b_s_start', type=int, required=True, help='SPANN nprobe start value')
    parser.add_argument('--b_s_end', type=int, required=True, help='SPANN nprobe end value')
    parser.add_argument('--b_s_step', type=int, required=True, help='SPANN nprobe step')
    parser.add_argument('--b_d_start', type=int, required=True, help='DiskANN L start value (0 means DiskANN is not used)')
    parser.add_argument('--b_d_end', type=int, required=True, help='DiskANN L end value')
    parser.add_argument('--b_d_step', type=int, required=True, help='DiskANN L step')

    # Other parameters
    parser.add_argument('--gt_k', type=int, default=100, help='The k value for ground truth')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode (uses only a small number of budget combinations)')
    parser.add_argument('--n_jobs', type=int, required=True, help='Number of parallel worker processes — set to your own core count')

    args = parser.parse_args()

    # Convert to Path objects
    diskann_result_dir = Path(args.diskann_result_dir)
    diskann_stats_dir = Path(args.diskann_stats_dir)
    spann_result_dir = Path(args.spann_result_dir)
    spann_stats_dir = Path(args.spann_stats_dir)

    print("=" * 80)
    print("Offline Profiling for DiskANN + SPANN Multi-Index")
    print("=" * 80)

    # Load ground truth
    print(f"\n[1/4] Loading ground truth...")
    print(f"  File: {args.gt_file}")
    gt_ids = load_ground_truth(args.gt_file)
    nq = gt_ids.shape[0]
    print(f"  Number of queries: {nq}")
    print(f"  Ground Truth K: {args.gt_k}")

    # Build the budget ranges
    if args.quick_test:
        b_s_range = [10, 50, 100]
        b_d_range = [0, 50, 100]
        print(f"\n[Quick test mode] Using a small number of budget combinations")
    else:
        b_s_range = list(range(args.b_s_start, args.b_s_end + 1, args.b_s_step))
        # DiskANN range: special-case b_D = 0
        if args.b_d_start == 0:
            b_d_range = [0] + list(range(args.b_d_step, args.b_d_end + 1, args.b_d_step))
        else:
            b_d_range = list(range(args.b_d_start, args.b_d_end + 1, args.b_d_step))

    print(f"\n[2/4] Loading all result files...")
    print(f"  SPANN nprobe range: {b_s_range}")
    print(f"  DiskANN L range: {b_d_range}")

    # Load all SPANN results and I/O stats
    spann_results = {}
    spann_io_stats = {}

    print(f"\n  Loading SPANN results...")
    for b_s in tqdm(b_s_range, desc="  SPANN"):
        result_file = spann_result_dir / f"result_K100_{b_s}.bin"
        stats_file = spann_stats_dir / f"stat_K100_{b_s}.csv"

        if not result_file.exists():
            print(f"    Warning: {result_file} not found")
            continue
        if not stats_file.exists():
            print(f"    Warning: {stats_file} not found")
            continue

        _, _, ids, _ = read_spann_result(str(result_file))
        spann_results[b_s] = ids
        spann_io_stats[b_s] = load_io_stats(str(stats_file))

    # Load all DiskANN results and I/O stats
    diskann_results = {}
    diskann_io_stats = {}

    print(f"\n  Loading DiskANN results...")
    for b_d in tqdm(b_d_range, desc="  DiskANN"):
        # Special-case b_D = 0: DiskANN is not used
        if b_d == 0:
            # Build an empty result matrix (all IDs set to the invalid value)
            diskann_results[0] = np.full((nq, 1), 0xFFFFFFFF, dtype='<u4')
            # I/O is all zero
            diskann_io_stats[0] = {q: 0 for q in range(nq)}
            continue

        result_file = diskann_result_dir / f"result_{b_d}_idx_uint32.bin"
        stats_file = diskann_stats_dir / f"stat_L{b_d}.csv"

        if not result_file.exists():
            print(f"    Warning: {result_file} not found")
            continue
        if not stats_file.exists():
            print(f"    Warning: {stats_file} not found")
            continue

        _, _, ids = read_ids_bin(str(result_file))
        diskann_results[b_d] = ids
        diskann_io_stats[b_d] = load_io_stats(str(stats_file))

    print(f"\n  Loaded {len(spann_results)} SPANN results")
    print(f"  Loaded {len(diskann_results)} DiskANN results")

    # Compute recall and I/O for all budget combinations
    print(f"\n[3/4] Computing recall and I/O for all budget combinations (using {args.n_jobs} processes)...")

    results = []
    
    # Prepare the task list
    tasks = []
    for b_s in sorted(spann_results.keys()):
        for b_d in sorted(diskann_results.keys()):
            tasks.append((b_s, b_d, args.gt_k))

    # Run in parallel using ProcessPoolExecutor
    # Note: on Linux this uses fork, so workers share the large data loaded by the main process
    with ProcessPoolExecutor(max_workers=args.n_jobs, 
                             initializer=init_worker, 
                             initargs=(spann_results, diskann_results, gt_ids, spann_io_stats, diskann_io_stats)) as executor:
        
        futures = [executor.submit(compute_combination_task, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="  Processing budget combinations"):
            results.extend(future.result())

    # Convert to a DataFrame and save
    print(f"\n[4/4] Saving results...")
    if not results:
        print("  Error: no budget combination results were produced! Check the input directories and budget ranges.")
        return

    df = pd.DataFrame(results)

    # Make sure the output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\n  Saved to: {output_path}")
    print(f"  Total {len(df)} records")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Show summary statistics
    print(f"\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"  Number of queries: {nq}")
    print(f"  SPANN budget count: {len(spann_results)}")
    print(f"  DiskANN budget count: {len(diskann_results)}")
    print(f"  Total budget combinations: {len(spann_results) * len(diskann_results)}")
    
    if len(df) > 0:
        print(f"\n  Recall statistics:")
        print(f"    Min: {df['recall'].min():.4f}")
        print(f"    Max: {df['recall'].max():.4f}")
        print(f"    Mean: {df['recall'].mean():.4f}")
        print(f"    Median: {df['recall'].median():.4f}")
        print(f"\n  I/O statistics:")
        print(f"    SPANN average I/O: {df['io_spann'].mean():.2f}")
        print(f"    DiskANN average I/O: {df['io_diskann'].mean():.2f}")
        print(f"    Total average I/O: {df['io_total'].mean():.2f}")
        print(f"    I/O range: {df['io_total'].min():.0f} ~ {df['io_total'].max():.0f}")
    else:
        print("\n  No data available for statistics.")

    # Show the first few rows
    print(f"\n  First 10 records:")
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
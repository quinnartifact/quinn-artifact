#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUINN Result Merger

Merges DiskANN and SPANN search results and computes union recall.

Functionality:
1. Read DiskANN results (diskann_100_idx.uint32.bin)
2. Read SPANN results (spann_result.bin)
3. Compute the union of results
4. Compare against ground truth to compute recall
5. Output a detailed statistics report
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import time


def load_diskann_results(result_file: str) -> np.ndarray:
    """
    Load a DiskANN result file

    Format: uint32 binary file
    Header: [num_queries, K]
    Data: num_queries × K uint32 IDs

    Args:
        result_file: path to the DiskANN result file

    Returns:
        (N, K) array of uint32 IDs
    """
    result_file = Path(result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"DiskANN result file not found: {result_file}")

    # Read uint32 binary
    data = np.fromfile(result_file, dtype=np.uint32)

    # DiskANN format: [num_queries, K, data...]
    if len(data) < 2:
        raise ValueError(f"Invalid DiskANN result file: too small ({len(data)} elements)")

    num_queries = data[0]
    K = data[1]

    expected_size = 2 + num_queries * K
    if len(data) < expected_size:
        raise ValueError(f"DiskANN result file size mismatch: expected {expected_size}, got {len(data)}")
    
    if len(data) > expected_size:
        print(f"  Warning: DiskANN result file is larger than expected ({len(data)} > {expected_size}). "
              f"Stale data detected? Only reading first {expected_size} elements.")
        data = data[:expected_size]

    # Parse the results
    results = data[2:].reshape(num_queries, K)

    print(f"  Loaded DiskANN results: {results.shape} (queries={num_queries}, K={K})")
    return results


def load_spann_results(result_file: str) -> np.ndarray:
    """
    Load a SPANN result file

    Format: binary file with header
    Header: num_queries (uint32), K (uint32)
    Data: num_queries × K pairs of (ID uint32, distance float32)

    Args:
        result_file: path to the SPANN result file

    Returns:
        (N, K) array of uint32 IDs
    """
    result_file = Path(result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"SPANN result file not found: {result_file}")

    import struct

    with open(result_file, 'rb') as f:
        # Read the header
        num_queries, K = struct.unpack('II', f.read(8))

        # Read the results (ID + distance pairs)
        results = np.zeros((num_queries, K), dtype=np.uint32)

        for i in range(num_queries):
            for j in range(K):
                # Read ID (uint32) and distance (float32)
                vid = struct.unpack('I', f.read(4))[0]
                dist = struct.unpack('f', f.read(4))[0]
                results[i, j] = vid

    print(f"  Loaded SPANN results: {results.shape} (queries={num_queries}, K={K})")
    return results


def load_ground_truth(gt_file: str, k: int = None) -> np.ndarray:
    """
    Load a ground truth file

    Supported formats:
    - .ivecs: fvecs format (int32)
    - .bin: binary format

    Args:
        gt_file: path to the ground truth file
        k: only load the first k results (loads all if None)

    Returns:
        (N, K) array of ground truth IDs
    """
    gt_file = Path(gt_file)
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    suffix = gt_file.suffix

    if suffix == '.ivecs':
        # ivecs format: each row starts with the dimension, followed by the data
        vectors = []
        with open(gt_file, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                vec = np.fromfile(f, dtype=np.int32, count=dim)
                if k is not None:
                    vec = vec[:k]
                vectors.append(vec)
        gt = np.array(vectors, dtype=np.int32)

    elif suffix == '.bin':
        # Binary format: header + data
        with open(gt_file, 'rb') as f:
            num_queries = np.fromfile(f, dtype=np.int32, count=1)[0]
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]

            if k is not None and k < dim:
                # Only read the first k
                gt = np.zeros((num_queries, k), dtype=np.int32)
                for i in range(num_queries):
                    row = np.fromfile(f, dtype=np.int32, count=dim)
                    gt[i] = row[:k]
            else:
                gt = np.fromfile(f, dtype=np.int32, count=num_queries * dim)
                gt = gt.reshape(num_queries, dim)

    else:
        raise ValueError(f"Unsupported ground truth format: {suffix}")

    print(f"  Loaded ground truth: {gt.shape}")
    return gt


def merge_results(diskann_results: np.ndarray, spann_results: np.ndarray) -> List[set]:
    """
    Merge DiskANN and SPANN results (union)

    Strategy: for each query, return the full union of the DiskANN and SPANN results

    Args:
        diskann_results: (N, K1) DiskANN results
        spann_results: (N, K2) SPANN results

    Returns:
        List of sets, each set is the union result for that query
    """
    num_queries = len(diskann_results)
    merged = []

    for i in range(num_queries):
        # Take the union directly
        diskann_set = set(diskann_results[i])
        spann_set = set(spann_results[i])
        union_set = diskann_set | spann_set
        merged.append(union_set)

    return merged


def calculate_recall(results, ground_truth: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
    """
    Compute recall@k

    Args:
        results: (N, K) search results (ndarray) or List[set] (union results)
        ground_truth: (N, K_gt) ground truth
        k: the k to compute recall@k for

    Returns:
        average_recall: the average recall
        per_query_recall: (N,) recall for each query
    """
    num_queries = len(results)
    per_query_recall = np.zeros(num_queries)

    for i in range(num_queries):
        # Determine the type of results
        if isinstance(results, list):
            # List[set] type (union result)
            result_set = results[i]
        else:
            # ndarray type
            result_ids = [vid for vid in results[i, :k] if vid >= 0]
            result_set = set(result_ids)

        gt_set = set(ground_truth[i, :k])

        # Recall = |intersection| / |ground_truth|
        intersection = result_set & gt_set
        per_query_recall[i] = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0

    average_recall = np.mean(per_query_recall)
    return average_recall, per_query_recall


def generate_report(
    diskann_results: np.ndarray,
    spann_results: np.ndarray,
    merged_results: np.ndarray,
    ground_truth: np.ndarray,
    k_values: List[int],
    output_file: str,
    search_times: Dict[str, float] = None,
    num_queries: int = None
):
    """
    Generate a detailed statistics report

    Args:
        diskann_results: DiskANN results
        spann_results: SPANN results
        merged_results: merged results
        ground_truth: ground truth
        k_values: list of k values to compute
        output_file: path to the output file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUINN Merged Results Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Basic information
        f.write("Dataset Information:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Number of queries: {len(diskann_results)}\n")
        f.write(f"  DiskANN result size: {diskann_results.shape[1]}\n")
        f.write(f"  SPANN result size: {spann_results.shape[1]}\n")
        f.write(f"  Ground truth size: {ground_truth.shape[1]}\n")
        f.write("\n")

        # QPS statistics (if search_times is provided)
        if search_times and num_queries:
            f.write("QPS Statistics:\n")
            f.write("-"*80 + "\n")

            diskann_time = search_times.get('diskann', 0) / 1000.0  # ms -> s
            spann_time = search_times.get('spann', 0) / 1000.0  # ms -> s

            # Service latency: max(T_diskann, T_spann)
            service_latency = max(diskann_time, spann_time)
            service_qps = num_queries / service_latency if service_latency > 0 else 0

            # Total work: T_diskann + T_spann
            total_work = diskann_time + spann_time
            total_qps = num_queries / total_work if total_work > 0 else 0

            f.write(f"  DiskANN search time: {diskann_time:.3f} s\n")
            f.write(f"  SPANN search time: {spann_time:.3f} s\n")
            f.write(f"  Service latency (max): {service_latency:.3f} s\n")
            f.write(f"  Service QPS: {service_qps:.2f} queries/s\n")
            f.write(f"  Total work (sum): {total_work:.3f} s\n")
            f.write(f"  Total QPS: {total_qps:.2f} queries/s\n")
            f.write("\n")

        # Compute recall for each k value
        f.write("Recall Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'K':<10}{'DiskANN':<15}{'SPANN':<15}{'Merged':<15}{'Improvement':<15}\n")
        f.write("-"*80 + "\n")

        for k in k_values:
            if k > min(diskann_results.shape[1], spann_results.shape[1], ground_truth.shape[1]):
                continue

            # Compute recall for each method
            diskann_recall, _ = calculate_recall(diskann_results, ground_truth, k)
            spann_recall, _ = calculate_recall(spann_results, ground_truth, k)
            merged_recall, _ = calculate_recall(merged_results, ground_truth, k)

            improvement = merged_recall - max(diskann_recall, spann_recall)

            f.write(f"{k:<10}{diskann_recall*100:<15.2f}{spann_recall*100:<15.2f}"
                   f"{merged_recall*100:<15.2f}{improvement*100:<15.2f}\n")

        f.write("\n")

        # Compute union statistics
        f.write("Union Statistics:\n")
        f.write("-"*80 + "\n")

        num_queries = len(diskann_results)
        overlap_sizes = []
        union_sizes = []

        for i in range(num_queries):
            diskann_set = set(diskann_results[i])
            spann_set = set(spann_results[i])
            overlap = diskann_set & spann_set
            union = diskann_set | spann_set

            overlap_sizes.append(len(overlap))
            union_sizes.append(len(union))

        f.write(f"  Average overlap size: {np.mean(overlap_sizes):.2f}\n")
        f.write(f"  Average union size: {np.mean(union_sizes):.2f}\n")
        f.write(f"  Average overlap ratio: {np.mean(overlap_sizes) / diskann_results.shape[1] * 100:.2f}%\n")
        f.write("\n")

        # Per-query analysis
        f.write("Per-Query Analysis (sample of first 10 queries):\n")
        f.write("-"*80 + "\n")

        k_analysis = k_values[-1] if k_values else 100
        diskann_recall_all, _ = calculate_recall(diskann_results, ground_truth, k_analysis)
        spann_recall_all, _ = calculate_recall(spann_results, ground_truth, k_analysis)
        merged_recall_all, per_query_merged = calculate_recall(merged_results, ground_truth, k_analysis)

        for i in range(min(100, num_queries)):
            diskann_r, _ = calculate_recall(diskann_results[i:i+1], ground_truth[i:i+1], k_analysis)
            spann_r, _ = calculate_recall(spann_results[i:i+1], ground_truth[i:i+1], k_analysis)
            merged_r = per_query_merged[i]

            f.write(f"  Query {i}: DiskANN={diskann_r*100:.2f}%, "
                   f"SPANN={spann_r*100:.2f}%, Merged={merged_r*100:.2f}%\n")

        f.write("\n")
        f.write("="*80 + "\n")

    print(f"  Report saved to: {output_file}")


def merge_and_evaluate(
    diskann_result_file: str,
    spann_result_file: str,
    ground_truth_file: str,
    output_dir: str,
    k_values: List[int] = [10, 20, 50, 100],
    search_times: Dict[str, float] = None,
    num_queries: int = None,
    io_stats: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Main function: merge results and evaluate

    Args:
        diskann_result_file: path to the DiskANN result file
        spann_result_file: path to the SPANN result file
        ground_truth_file: path to the ground truth file
        output_dir: output directory
        k_values: list of k values to evaluate

    Returns:
        a dict containing the various recall values
    """
    print("\n" + "="*80)
    print("QUINN Result Merger")
    print("="*80)

    # Load the results
    print("\n[Step 1] Loading results...")
    diskann_results = load_diskann_results(diskann_result_file)
    spann_results = load_spann_results(spann_result_file)

    # Load ground truth
    print("\n[Step 2] Loading ground truth...")
    max_k = max(k_values)
    ground_truth = load_ground_truth(ground_truth_file, k=max_k)

    # Check dimensions
    if len(diskann_results) != len(spann_results) or len(diskann_results) != len(ground_truth):
        raise ValueError(f"Query count mismatch: DiskANN={len(diskann_results)}, "
                        f"SPANN={len(spann_results)}, GT={len(ground_truth)}")

    # Compute the union
    print("\n[Step 3] Merging results (union)...")
    merged_results = merge_results(diskann_results, spann_results)

    # Compute union-size statistics
    union_sizes = [len(s) for s in merged_results]
    overlap_sizes = [len(set(diskann_results[i]) & set(spann_results[i])) for i in range(len(diskann_results))]
    print(f"  Average union size: {np.mean(union_sizes):.2f}")
    print(f"  Average overlap size: {np.mean(overlap_sizes):.2f}")

    # Compute recall
    print("\n[Step 4] Calculating recall for each k...")
    recalls = {}
    max_result_k = min(diskann_results.shape[1], spann_results.shape[1])

    for k in k_values:
        if k > max_result_k:
            print(f"  Warning: k={k} exceeds result size, skipping")
            continue

        diskann_recall, _ = calculate_recall(diskann_results, ground_truth, k)
        spann_recall, _ = calculate_recall(spann_results, ground_truth, k)
        merged_recall, _ = calculate_recall(merged_results, ground_truth, k)

        recalls[f'diskann_recall@{k}'] = diskann_recall
        recalls[f'spann_recall@{k}'] = spann_recall
        recalls[f'merged_recall@{k}'] = merged_recall

        print(f"  Recall@{k}:")
        print(f"    DiskANN: {diskann_recall*100:.2f}%")
        print(f"    SPANN:   {spann_recall*100:.2f}%")
        print(f"    Merged:  {merged_recall*100:.2f}%")
        print(f"    Gain:    {(merged_recall - max(diskann_recall, spann_recall))*100:.2f}%")

    # Generate the report
    print("\n[Step 5] Generating report...")
    output_dir = Path(output_dir)
    report_file = output_dir / 'merged_recall_report.txt'

    generate_report(diskann_results, spann_results, merged_results,
                   ground_truth, k_values, report_file,
                   search_times, num_queries)

    # Save merge statistics (not the full union, since its size varies)
    stats_file = output_dir / 'merged_stats.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Union size per query:\n")
        for i, size in enumerate(union_sizes[:100]):  # first 100 samples
            f.write(f"Query {i}: {size}\n")
        f.write(f"\nAverage union size: {np.mean(union_sizes):.2f}\n")
        f.write(f"Average overlap size: {np.mean(overlap_sizes):.2f}\n")

        if io_stats and num_queries:
            diskann_io = io_stats.get('diskann', 0.0)
            spann_io = io_stats.get('spann', 0.0)
            total_io = diskann_io + spann_io
            avg_io = total_io / num_queries
            f.write(f"Average IO per query (DiskANN+SPANN): {avg_io:.2f}\n")
            f.write(f"Total IO (DiskANN+SPANN): {total_io:.0f}\n")

    print(f"  Merge statistics saved to: {stats_file}")

    print("\n" + "="*80)
    print("Merge and evaluation completed!")
    print("="*80 + "\n")

    return recalls


if __name__ == '__main__':
    # For standalone testing
    import argparse

    parser = argparse.ArgumentParser(description='Merge DiskANN and SPANN results')
    parser.add_argument('--diskann_result', required=True, help='DiskANN result file')
    parser.add_argument('--spann_result', required=True, help='SPANN result file')
    parser.add_argument('--ground_truth', required=True, help='Ground truth file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--k_values', default='10,20,50,100', help='K values (comma-separated)')

    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(',')]

    merge_and_evaluate(
        args.diskann_result,
        args.spann_result,
        args.ground_truth,
        args.output_dir,
        k_values
    )

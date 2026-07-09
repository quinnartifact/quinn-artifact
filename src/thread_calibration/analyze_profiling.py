#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and summarize static profiling results.

Reads all profiling_results.csv files under results/threading/ and produces:
  1. A merged summary table printed to stdout
  2. A best_configs_all.csv across all datasets
  3. Optional: QPS heatmap per (dataset, recall)

Usage:
  python analyze_profiling.py
  python analyze_profiling.py --results_dir /path/to/results/threading
  python analyze_profiling.py --plot  # requires matplotlib
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_csv(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def float_or_none(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, '', 'None') else None
    except (TypeError, ValueError):
        return None


def find_profiling_csvs(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob('*/profiling_results.csv'))


def merge_rows(csv_paths: List[Path]) -> List[Dict]:
    all_rows = []
    for p in csv_paths:
        rows = load_csv(str(p))
        for r in rows:
            r['achieved_recall'] = float_or_none(r.get('achieved_recall'))
            r['qps'] = float_or_none(r.get('qps'))
            r['mean_latency_ms'] = float_or_none(r.get('mean_latency_ms'))
            r['p99_latency_ms'] = float_or_none(r.get('p99_latency_ms'))
            r['p999_latency_ms'] = float_or_none(r.get('p999_latency_ms'))
            r['threadS'] = int(r['threadS'])
            r['threadD'] = int(r['threadD'])
            r['total_threads'] = int(r['total_threads'])
            r['target_recall'] = float(r['target_recall'])
        all_rows.extend(rows)
    return all_rows


def print_table(rows: List[Dict], title: str = ''):
    if title:
        print(f"\n{'='*80}")
        print(title)
        print('='*80)

    datasets = sorted(set(r['dataset'] for r in rows))
    recalls = sorted(set(r['target_recall'] for r in rows))

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        print(f"{'recall':>8}  {'threadS':>8}  {'threadD':>8}  "
              f"{'achieved%':>10}  {'QPS':>10}  {'mean_ms':>9}  {'p99_ms':>9}  {'best':>6}")
        print('-' * 78)

        ds_rows = [r for r in rows if r['dataset'] == dataset]
        for recall in recalls:
            recall_rows = sorted(
                [r for r in ds_rows if r['target_recall'] == recall],
                key=lambda x: x['threadS']
            )
            for r in recall_rows:
                ar = f"{r['achieved_recall']:.2f}" if r['achieved_recall'] is not None else 'N/A'
                qps = f"{r['qps']:.1f}" if r['qps'] is not None else 'N/A'
                mean = f"{r['mean_latency_ms']:.1f}" if r['mean_latency_ms'] is not None else 'N/A'
                p99 = f"{r['p99_latency_ms']:.1f}" if r['p99_latency_ms'] is not None else 'N/A'
                best_flag = ' <BEST' if r.get('is_best', 'False') == 'True' else ''
                print(f"{recall:>8}  {r['threadS']:>8}  {r['threadD']:>8}  "
                      f"{ar:>10}  {qps:>10}  {mean:>9}  {p99:>9}  {best_flag}")


def print_best_table(rows: List[Dict]):
    """Print a compact best-config lookup table per (dataset, recall)."""
    best_rows = [r for r in rows if r.get('is_best', 'False') == 'True']
    if not best_rows:
        print("\nNo best configs found (is_best column missing or all False).")
        return

    print(f"\n{'='*80}")
    print("BEST STATIC THREAD CONFIGS  (totalThread=32, maximize QPS @ target recall)")
    print('='*80)
    print(f"{'dataset':>12}  {'recall':>8}  {'threadS':>8}  {'threadD':>8}  "
          f"{'achieved%':>10}  {'QPS':>10}  {'p99_ms':>9}")
    print('-' * 80)

    datasets = sorted(set(r['dataset'] for r in best_rows))
    recalls = sorted(set(r['target_recall'] for r in best_rows))

    for dataset in datasets:
        for recall in recalls:
            matches = [r for r in best_rows
                       if r['dataset'] == dataset and r['target_recall'] == recall]
            if not matches:
                continue
            r = matches[0]
            ar = f"{r['achieved_recall']:.2f}" if r['achieved_recall'] is not None else 'N/A'
            qps = f"{r['qps']:.1f}" if r['qps'] is not None else 'N/A'
            p99 = f"{r['p99_latency_ms']:.1f}" if r['p99_latency_ms'] is not None else 'N/A'
            print(f"{dataset:>12}  {recall:>8}  {r['threadS']:>8}  {r['threadD']:>8}  "
                  f"{ar:>10}  {qps:>10}  {p99:>9}")


def save_best_csv(rows: List[Dict], output_path: str):
    best_rows = [r for r in rows if r.get('is_best', 'False') == 'True']
    if not best_rows:
        return
    fieldnames = ['dataset', 'target_recall', 'total_threads', 'threadS', 'threadD',
                  'achieved_recall', 'qps', 'mean_latency_ms', 'p99_latency_ms',
                  'p999_latency_ms', 'diskann_time_s', 'spann_time_s',
                  'diskann_io_latency_ms', 'queue_wait_time_ms']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"\nBest configs saved to: {output_path}")


def plot_heatmaps(rows: List[Dict], output_dir: Path):
    """
    For each dataset, plot a heatmap of QPS over (recall, threadS).
    Requires matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    datasets = sorted(set(r['dataset'] for r in rows))
    recalls = sorted(set(r['target_recall'] for r in rows))

    for dataset in datasets:
        ds_rows = [r for r in rows if r['dataset'] == dataset]
        all_splits = sorted(set(r['threadS'] for r in ds_rows))

        # QPS matrix: rows=recalls, cols=threadS values
        qps_matrix = np.full((len(recalls), len(all_splits)), np.nan)
        for r in ds_rows:
            ri = recalls.index(r['target_recall'])
            si = all_splits.index(r['threadS'])
            if r['qps'] is not None:
                qps_matrix[ri, si] = r['qps']

        fig, ax = plt.subplots(figsize=(max(6, len(all_splits) * 0.9), max(4, len(recalls) * 0.6)))
        im = ax.imshow(qps_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label='QPS')

        ax.set_xticks(range(len(all_splits)))
        ax.set_xticklabels([f"S{s}/D{32-s}" for s in all_splits], rotation=45, ha='right')
        ax.set_yticks(range(len(recalls)))
        ax.set_yticklabels([str(r) for r in recalls])
        ax.set_xlabel('Thread Split (threadS / threadD)')
        ax.set_ylabel('Target Recall (%)')
        ax.set_title(f'{dataset}  —  QPS Heatmap (totalThread=32)')

        # Annotate cells
        for ri in range(len(recalls)):
            for si in range(len(all_splits)):
                val = qps_matrix[ri, si]
                if not np.isnan(val):
                    ax.text(si, ri, f'{val:.0f}', ha='center', va='center',
                            fontsize=7, color='black')

        plt.tight_layout()
        fig_path = output_dir / f'{dataset}_qps_heatmap.png'
        plt.savefig(str(fig_path), dpi=150)
        plt.close()
        print(f"Saved heatmap: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze static profiling results')
    parser.add_argument(
        '--results_dir', type=str, default=None,
        help='Directory containing profiling results. Default: <repo>/results/threading'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate QPS heatmap plots per dataset'
    )
    args = parser.parse_args()

    if args.results_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        results_dir = repo_root / 'result' / 'profiling' / 'threading'
    else:
        results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    csvs = find_profiling_csvs(results_dir)
    if not csvs:
        print(f"[ERROR] No profiling_results.csv files found under {results_dir}")
        sys.exit(1)

    print(f"Found {len(csvs)} profiling CSV(s):")
    for p in csvs:
        print(f"  {p}")

    rows = merge_rows(csvs)
    print(f"\nTotal rows loaded: {len(rows)}")

    print_best_table(rows)
    print_table(rows, title='Full Profiling Results')

    # Save merged best configs
    save_best_csv(rows, str(results_dir / 'best_configs_all.csv'))

    if args.plot:
        plot_heatmaps(rows, results_dir)


if __name__ == '__main__':
    main()

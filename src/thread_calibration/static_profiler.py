#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Thread Allocation Profiler for QUINN Hybrid Index

For a fixed totalThread budget, grid-searches all valid (threadS, threadD)
configurations to find, for each dataset × target_recall combination, the
static thread allocation that maximizes hybrid QPS.

Constraints:
  threadS + threadD = totalThread
  threadS >= 4
  threadD >= 4
  threadS, threadD are integers

Usage:
  python static_profiler.py --config /path/to/deep100m.yaml --total_threads 32
  python static_profiler.py --config /path/to/deep100m.yaml --total_threads 32 \\
      --target_recalls 70 80 90 95 99 --output_dir results/threading/deep100m

Output:
  <output_dir>/profiling_results.csv  — full profiling table
  <output_dir>/best_configs.csv       — best config per (dataset, recall)
  <output_dir>/<recall>/<threadS>_<threadD>/merged_recall_report.txt  — per-run benchmark
"""

import argparse
import copy
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from io_monitor import DiskstatsMonitor


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_ini_text(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def save_ini_text(text: str, path: str):
    with open(path, 'w') as f:
        f.write(text)


def set_ini_threads(ini_text: str, n_threads: int) -> str:
    """Replace NumberOfThreads=<N> in .ini text."""
    return re.sub(
        r'(NumberOfThreads\s*=\s*)\d+',
        lambda m: m.group(1) + str(n_threads),
        ini_text
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_benchmark_report(report_path: str) -> Optional[Dict]:
    """
    Parse merged_recall_report.txt and return a dict of metrics.
    Returns None if the file doesn't exist or can't be parsed.
    """
    p = Path(report_path)
    if not p.exists():
        return None

    text = p.read_text()
    metrics = {}

    # QPS
    m = re.search(r'Final Service QPS:\s+([\d.]+)', text)
    if m:
        metrics['qps'] = float(m.group(1))
    else:
        m = re.search(r'Service QPS:\s+([\d.]+)', text)
        metrics['qps'] = float(m.group(1)) if m else None

    # Latency
    m = re.search(r'Per-Query Lat\. Mean:\s+([\d.]+)', text)
    metrics['mean_latency_ms'] = float(m.group(1)) if m else None

    m = re.search(r'Per-Query Lat\. p99:\s+([\d.]+)', text)
    metrics['p99_latency_ms'] = float(m.group(1)) if m else None

    m = re.search(r'Per-Query Lat\. p99\.9:\s+([\d.]+)', text)
    metrics['p999_latency_ms'] = float(m.group(1)) if m else None

    # Recall@10 merged  (table line: "10    xx.xx  xx.xx  xx.xx  ...")
    m = re.search(r'^10\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
    metrics['achieved_recall'] = float(m.group(3)) if m else None

    # DiskANN / SPANN wall times
    m = re.search(r'DiskANN search time:\s+([\d.]+)\s*s', text)
    metrics['diskann_time_s'] = float(m.group(1)) if m else None

    m = re.search(r'SPANN search time:\s+([\d.]+)\s*s', text)
    metrics['spann_time_s'] = float(m.group(1)) if m else None

    return metrics


# ---------------------------------------------------------------------------
# Valid thread combinations
# ---------------------------------------------------------------------------

def valid_thread_splits(total: int, min_each: int = 4, step: int = 1) -> List[Tuple[int, int]]:
    """Return list of (threadS, threadD) pairs satisfying constraints.

    Args:
        total:    Total thread budget.
        min_each: Minimum threads per component.
        step:     Increment between threadS values (e.g. 4 → only multiples of 4).
                  Always includes the min_each endpoints.
    """
    if step <= 1:
        candidates = range(min_each, total - min_each + 1)
    else:
        # Multiples of step, plus ensure endpoints are included
        raw = set(range(min_each, total - min_each + 1, step))
        raw.add(min_each)
        raw.add(total - min_each)
        candidates = sorted(raw)

    splits = []
    for tS in candidates:
        tD = total - tS
        if tD >= min_each:
            splits.append((tS, tD))
    return splits


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_one(
    base_yaml: str,
    base_ini: str,
    threadS: int,
    threadD: int,
    target_recall: float,
    run_output_dir: str,
    controller_script: str,
    tmp_dir: str,
    dry_run: bool = False,
    device: str = 'nvme0n1',
) -> Optional[Dict]:
    """
    Create temp configs, invoke controller.py, return parsed metrics.

    Args:
        base_yaml:        Path to the dataset's base YAML config
        base_ini:         Path to the dataset's SPANN .ini config
        threadS:          SPANN thread count
        threadD:          DiskANN thread count
        target_recall:    Target recall (%)
        run_output_dir:   Directory for this run's benchmark output
        controller_script: Path to controller.py
        tmp_dir:          Directory for temp config files
        dry_run:          If True, print the command but don't execute

    Returns:
        Dict of parsed metrics, or None on failure.
    """
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # --- Build temp YAML ---
    cfg = load_yaml(base_yaml)
    cfg['target_recall'] = target_recall

    # Set DiskANN threads
    cfg.setdefault('diskann', {}).setdefault('args', {})['num_threads'] = threadD

    # Redirect ALL output paths to the per-run directory so we never
    # touch the existing benchmark/ or result/ data from other runs.
    cfg.setdefault('output', {})['benchmark_dir'] = run_output_dir
    cfg['output']['output_dir'] = run_output_dir

    # Point temp SPANN ini to our modified copy
    ini_name = f'spann_s{threadS}_d{threadD}_r{target_recall}.ini'
    tmp_ini_path = str(Path(tmp_dir) / ini_name)
    cfg.setdefault('spann', {})['config_file'] = tmp_ini_path

    yaml_name = f'config_s{threadS}_d{threadD}_r{target_recall}.yaml'
    tmp_yaml_path = str(Path(tmp_dir) / yaml_name)
    save_yaml(cfg, tmp_yaml_path)

    # --- Build temp INI ---
    ini_text = load_ini_text(base_ini)
    ini_text = set_ini_threads(ini_text, threadS)
    save_ini_text(ini_text, tmp_ini_path)

    # --- Run controller ---
    cmd = [sys.executable, controller_script, '--config', tmp_yaml_path]
    print(f"\n[Run] threadS={threadS} threadD={threadD} recall={target_recall}")
    print(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        return {'qps': None, 'achieved_recall': None,
                'mean_latency_ms': None, 'p99_latency_ms': None,
                'p999_latency_ms': None, 'diskann_time_s': None,
                'spann_time_s': None, 'avg_read_bw_mb': None,
                'avg_queue_depth': None}

    t0 = time.time()
    monitor = DiskstatsMonitor(device=device)
    monitor.start()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # stream stdout/stderr to terminal
            text=True,
            cwd=str(Path(controller_script).parent),
        )
    except Exception as e:
        monitor.stop()
        print(f"  [ERROR] Subprocess failed: {e}")
        return None
    monitor.stop()
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] Controller exited with code {result.returncode}")
        return None

    print(f"  Completed in {elapsed:.1f}s")

    avg_bw = monitor.avg_read_bw_mb()
    avg_qd = monitor.avg_queue_depth()
    if avg_bw is not None:
        print(f"  IO avg BW={avg_bw:.1f} MB/s  avg queue depth={avg_qd:.2f}")
    monitor.save_timeseries(str(Path(run_output_dir) / 'diskstats.csv'))

    # --- Parse output ---
    report_path = str(Path(run_output_dir) / 'merged_recall_report.txt')
    metrics = parse_benchmark_report(report_path)
    if metrics is None:
        print(f"  [WARN] Could not parse report at {report_path}")
    else:
        print(f"  recall={metrics.get('achieved_recall'):.2f}%  "
              f"QPS={metrics.get('qps'):.1f}")

    if metrics is not None:
        metrics['avg_read_bw_mb']  = avg_bw
        metrics['avg_queue_depth'] = avg_qd
    return metrics


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

def run_profiling(
    base_yaml: str,
    total_threads: int,
    target_recalls: List[float],
    output_dir: str,
    dry_run: bool = False,
    min_threads: int = 4,
    thread_step: int = 1,
    sleep_between: float = 5.0,
    device: str = 'nvme0n1',
):
    """
    Grid-search all valid (threadS, threadD) for each target_recall,
    aggregate results, and write CSV outputs.
    """
    base_yaml = Path(base_yaml).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_base = load_yaml(str(base_yaml))
    dataset = cfg_base.get('dataset', 'unknown')

    # Resolve SPANN ini path (no placeholder substitution needed for ini path itself)
    base_ini = cfg_base.get('spann', {}).get('config_file', '')
    if not base_ini:
        raise ValueError("spann.config_file not found in base config")
    base_ini = Path(base_ini).resolve()
    if not base_ini.exists():
        raise FileNotFoundError(f"SPANN ini not found: {base_ini}")

    # Find controller.py
    controller_script = Path(__file__).resolve().parents[1] / 'controller' / 'controller.py'
    if not controller_script.exists():
        raise FileNotFoundError(f"controller.py not found at {controller_script}")

    splits = valid_thread_splits(total_threads, min_threads, thread_step)
    print(f"\n{'='*70}")
    print(f"Static Profiling: dataset={dataset}  totalThread={total_threads}")
    print(f"Target recalls: {target_recalls}")
    print(f"Thread splits ({len(splits)} combos): {splits}")
    print(f"Total runs: {len(target_recalls) * len(splits)}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}")

    tmp_dir = str(output_dir / '_tmp_configs')

    # Load existing results so incremental runs don't overwrite them
    existing_csv = output_dir / 'profiling_results.csv'
    rows = []
    existing_keys = set()
    if existing_csv.exists():
        existing_rows = []
        with open(existing_csv, newline='') as f:
            for r in csv.DictReader(f):
                existing_rows.append(r)
                existing_keys.add((float(r['target_recall']), int(r['threadS']), int(r['threadD'])))
        rows = existing_rows
        print(f"  Loaded {len(rows)} existing rows from {existing_csv}")

    for target_recall in target_recalls:
        for threadS, threadD in splits:
            run_bench_dir = str(output_dir / f"recall{target_recall}" / f"threadS{threadS}_threadD{threadD}")

            # Skip if this (recall, threadS, threadD) combo already exists
            if (float(target_recall), int(threadS), int(threadD)) in existing_keys:
                print(f"  [Skip] threadS={threadS} threadD={threadD} recall={target_recall} (already done)")
                continue

            metrics = run_one(
                base_yaml=str(base_yaml),
                base_ini=str(base_ini),
                threadS=threadS,
                threadD=threadD,
                target_recall=target_recall,
                run_output_dir=run_bench_dir,
                controller_script=str(controller_script),
                tmp_dir=tmp_dir,
                dry_run=dry_run,
                device=device,
            )

            row = {
                'dataset': dataset,
                'target_recall': target_recall,
                'total_threads': total_threads,
                'threadS': threadS,
                'threadD': threadD,
                'achieved_recall': metrics.get('achieved_recall') if metrics else None,
                'qps': metrics.get('qps') if metrics else None,
                'mean_latency_ms': metrics.get('mean_latency_ms') if metrics else None,
                'p99_latency_ms': metrics.get('p99_latency_ms') if metrics else None,
                'p999_latency_ms': metrics.get('p999_latency_ms') if metrics else None,
                'diskann_time_s': metrics.get('diskann_time_s') if metrics else None,
                'spann_time_s': metrics.get('spann_time_s') if metrics else None,
                'avg_read_bw_mb': metrics.get('avg_read_bw_mb') if metrics else None,
                'avg_queue_depth': metrics.get('avg_queue_depth') if metrics else None,
                'is_best': False,
                'status': 'ok' if metrics else 'failed',
                'run_dir': run_bench_dir,
            }
            rows.append(row)

            # Save incrementally after each run
            _write_csv(rows, str(output_dir / 'profiling_results.csv'))

            if sleep_between > 0 and not dry_run:
                print(f"  Sleeping {sleep_between}s before next run...")
                time.sleep(sleep_between)

    # --- Mark best configs ---
    rows = _mark_best(rows)

    # --- Write final outputs ---
    _write_csv(rows, str(output_dir / 'profiling_results.csv'))

    best_rows = [r for r in rows if r['is_best']]
    _write_csv(best_rows, str(output_dir / 'best_configs.csv'))

    # --- Cleanup temp configs ---
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print(f"PROFILING COMPLETE — {dataset}")
    print(f"{'='*70}")
    _print_summary(rows, total_threads)

    print(f"\nFull results:  {output_dir}/profiling_results.csv")
    print(f"Best configs:  {output_dir}/best_configs.csv")

    return rows


def _to_float(v):
    try:
        return float(v) if v not in (None, '', 'None') else None
    except (TypeError, ValueError):
        return None


def _mark_best(rows: List[Dict]) -> List[Dict]:
    """
    For each (dataset, target_recall) group, mark the config with the
    highest QPS among those meeting achieved_recall >= target_recall as best.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r['dataset'], r['target_recall'])
        groups[key].append(i)

    # Reset all first to avoid stale True values from previous runs
    for r in rows:
        r['is_best'] = False

    for key, indices in groups.items():
        target_recall = key[1]
        candidates = [
            i for i in indices
            if _to_float(rows[i]['qps']) is not None
        ]
        if candidates:
            best_i = max(candidates, key=lambda i: _to_float(rows[i]['qps']))
            rows[best_i]['is_best'] = True

    return rows


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict], total_threads: int):
    """Print a human-readable summary table."""
    print(f"\n{'recall':>8}  {'threadS':>8}  {'threadD':>8}  "
          f"{'achieved':>10}  {'QPS':>10}  {'mean_ms':>10}  {'p99_ms':>9}  best")
    print('-' * 80)
    for r in rows:
        flag = '  <-- BEST' if str(r['is_best']) == 'True' else ''
        ar_v = _to_float(r['achieved_recall'])
        qps_v = _to_float(r['qps'])
        mean_v = _to_float(r['mean_latency_ms'])
        p99_v = _to_float(r['p99_latency_ms'])
        ar = f"{ar_v:.2f}" if ar_v is not None else 'N/A'
        qps = f"{qps_v:.1f}" if qps_v is not None else 'N/A'
        mean = f"{mean_v:.1f}" if mean_v is not None else 'N/A'
        p99 = f"{p99_v:.1f}" if p99_v is not None else 'N/A'
        print(f"{r['target_recall']:>8}  {r['threadS']:>8}  {r['threadD']:>8}  "
              f"{ar:>10}  {qps:>10}  {mean:>10}  {p99:>9}{flag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='QUINN Static Thread Allocation Profiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', required=True,
        help='Base YAML config for the dataset (e.g. configs/deep100m.yaml)'
    )
    parser.add_argument(
        '--total_threads', type=int, default=32,
        help='Total thread budget (threadS + threadD = this value). Default: 32'
    )
    parser.add_argument(
        '--target_recalls', type=float, nargs='+',
        default=[70, 80, 90, 95, 98],
        help='List of target recall values to sweep. Default: 70 80 90 95 98'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory. Default: results/threading/<dataset>_t<total_threads>'
    )
    parser.add_argument(
        '--min_threads', type=int, default=4,
        help='Minimum threads for each component. Default: 4'
    )
    parser.add_argument(
        '--thread_step', type=int, default=1,
        help='Increment between threadS values (1=all, 4=coarse sweep). Default: 1'
    )
    parser.add_argument(
        '--sleep', type=float, default=5.0,
        help='Seconds to sleep between runs (to let SSD cool down). Default: 5'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--device', type=str, default='nvme0n1',
        help='Block device name to monitor in /proc/diskstats. Default: nvme0n1'
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = cfg.get('dataset', 'unknown')

    if args.output_dir is None:
        quinn_root = Path(__file__).resolve().parents[2]
        args.output_dir = str(
            quinn_root / 'result' / 'profiling' / 'threading' / f'{dataset}_t{args.total_threads}'
        )

    run_profiling(
        base_yaml=args.config,
        total_threads=args.total_threads,
        target_recalls=args.target_recalls,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        min_threads=args.min_threads,
        thread_step=args.thread_step,
        sleep_between=args.sleep,
        device=args.device,
    )


if __name__ == '__main__':
    main()

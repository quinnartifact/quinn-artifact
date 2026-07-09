#!/usr/bin/env python3
"""
Collect per-100ms I/O queue depth & bandwidth during hybrid (SPANN+DiskANN) runs.

For each thread combo at a given recall, runs the full QUINN controller and
simultaneously samples /proc/diskstats every 100ms to record:
  - Queue depth  : weighted_io_ms_delta / interval_ms  (= iostat avgqu-sz)
  - Read BW (MB/s): sectors_read_delta * 512 / interval_s / 1e6

Usage:
  python monitor.py --recalls 80 90 95 98
  python monitor.py --dataset sift100m --recalls 80 90 95 98

Results saved to:
  results/io_monitor/<dataset>_recall<R>/<combo>/diskstats.csv
  results/io_monitor/<dataset>_recall<R>/queue_depth.png
  results/io_monitor/<dataset>_recall<R>/read_bw.png
"""

import argparse
import os
import sys
import subprocess
import tempfile
import time
import threading
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

QUINN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(QUINN, 'src', 'thread_calibration'))
from static_profiler import load_yaml, save_yaml, load_ini_text, save_ini_text, set_ini_threads
from io_monitor import read_diskstats, compute_timeseries, DiskstatsMonitor

CONTROLLER = os.path.join(QUINN, 'src/controller/controller.py')
PYTHON     = sys.executable
DEVICE     = 'nvme0n1'

TOTAL     = 32
STEP      = 4
COMBOS    = [(s, TOTAL - s) for s in range(STEP, TOTAL, STEP)]
INTERVAL  = 0.1   # 100ms


# read_diskstats, compute_timeseries imported from src/thread_calibration/io_monitor


# ---------------------------------------------------------------------------
# Run one hybrid combo with diskstats monitoring
# ---------------------------------------------------------------------------
def run_combo(thread_s, thread_d, recall, out_dir, tmp_dir, base_yaml, base_ini):
    combo = f"S{thread_s}D{thread_d}"
    csv_path = os.path.join(out_dir, 'diskstats.csv')

    if os.path.exists(csv_path):
        print(f"  [{combo}] already exists, loading")
        times, qdepth, bw = [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                times.append(float(row['time_ms']))
                qdepth.append(float(row['queue_depth']))
                bw.append(float(row['read_bw_mb']))
        return times, qdepth, bw

    os.makedirs(out_dir, exist_ok=True)

    dump_path = os.path.join(out_dir, 'latency_raw.npz')

    # Build temp config
    cfg = load_yaml(base_yaml)
    cfg['target_recall'] = recall
    cfg.setdefault('threading', {})['mode'] = 'off'
    cfg.setdefault('diskann', {}).setdefault('args', {})['num_threads'] = thread_d
    cfg.setdefault('output', {})['output_dir']    = out_dir
    cfg['output']['benchmark_dir']                 = out_dir
    cfg['output']['latency_dump_path']             = dump_path

    tmp_ini  = os.path.join(tmp_dir, f'spann_s{thread_s}_d{thread_d}_r{recall}.ini')
    ini_text = load_ini_text(base_ini)
    ini_text = set_ini_threads(ini_text, thread_s)
    save_ini_text(ini_text, tmp_ini)
    cfg['spann']['config_file'] = tmp_ini

    tmp_yaml = os.path.join(tmp_dir, f'config_s{thread_s}_d{thread_d}_r{recall}.yaml')
    save_yaml(cfg, tmp_yaml)

    clock_offset = time.time() - time.monotonic()

    monitor = DiskstatsMonitor(device=DEVICE, interval=INTERVAL)
    monitor.start()

    cmd = [PYTHON, CONTROLLER, '--config', tmp_yaml]
    print(f"  [{combo}] running hybrid (S={thread_s} D={thread_d} recall={recall}) ...")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=QUINN, capture_output=True)
    elapsed = time.time() - t0
    print(f"  [{combo}] done in {elapsed:.1f}s, exit={ret.returncode}")

    monitor.stop()
    times, qdepth, bw = monitor.timeseries()
    records = monitor._records

    # Align to DiskANN search start using diskann_start_ns (CLOCK_REALTIME ns).
    # Convert to monotonic: mono_s = realtime_ns/1e9 - clock_offset
    search_start_s = None
    if os.path.exists(dump_path):
        data = np.load(dump_path)
        ds = data['diskann_start_ns']
        valid = ds[ds > 0]
        if len(valid) > 0:
            search_start_s = valid.min() / 1e9 - clock_offset

    if search_start_s is not None and times:
        # abs_ts[j] = records[j+1]['ts'] (time.monotonic()) — same base as search_start_s now
        abs_ts = np.array([records[i]['ts'] for i in range(1, len(records))])
        rel_ms = (abs_ts - search_start_s) * 1000.0
        q_arr  = np.array(qdepth)
        b_arr  = np.array(bw)
        mask = rel_ms >= 0
        times  = rel_ms[mask].tolist()
        qdepth = q_arr[mask].tolist()
        bw     = b_arr[mask].tolist()
        print(f"  [{combo}] aligned to DiskANN search start ({len(times)} samples kept)")
    elif times:
        # Fallback: trim from first IO activity
        t_arr = np.array(times)
        q_arr = np.array(qdepth)
        b_arr = np.array(bw)
        active = np.where((q_arr > 0.5) | (b_arr > 10.0))[0]
        if len(active) > 0:
            s = active[0]
            t_offset = t_arr[s]
            times  = (t_arr[s:] - t_offset).tolist()
            qdepth = q_arr[s:].tolist()
            bw     = b_arr[s:].tolist()

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_ms', 'queue_depth', 'read_bw_mb'])
        for t, q, b in zip(times, qdepth, bw):
            w.writerow([f'{t:.1f}', f'{q:.4f}', f'{b:.4f}'])

    print(f"  [{combo}] saved {len(times)} samples to {csv_path}")
    return times, qdepth, bw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def align_to_search_start(times, vals, threshold=1.0):
    """Shift time axis so t=0 = first sample where val > threshold."""
    times  = np.array(times)
    vals   = np.array(vals)
    active = np.where(vals > threshold)[0]
    if len(active) == 0:
        return times, vals
    t0 = times[active[0]]
    return times - t0, vals


def plot_metric(series_dict, out_path, ylabel, title, threshold=1.0):
    fig, ax = plt.subplots(figsize=(13, 6))
    colors  = cm.tab10(np.linspace(0, 1, len(COMBOS)))
    for (s, d), color in zip(COMBOS, colors):
        label = f"S{s}D{d}"
        if label not in series_dict:
            continue
        times, vals = series_dict[label]
        if not times:
            continue
        t = np.array(times)
        v = np.array(vals)
        ax.plot(t, v, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Time since search start (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot-only mode: read diskstats.csv from static_profiler output directory
# ---------------------------------------------------------------------------
def _load_diskstats_csv(path):
    times, qdepth, bw = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            times.append(float(row['time_ms']))
            qdepth.append(float(row['queue_depth']))
            bw.append(float(row['read_bw_mb']))
    return times, qdepth, bw


def plot_from_static_profiler(source_dir, recalls, dataset):
    """
    Read diskstats.csv files produced by static_profiler.py and plot BW time series.

    Expected directory layout (static_profiler output):
      <source_dir>/recall{R}/threadS{s}_threadD{d}/diskstats.csv

    Plots are written next to the recall directory:
      <source_dir>/recall{R}/queue_depth.png
      <source_dir>/recall{R}/read_bw.png
    """
    source_dir = os.path.abspath(source_dir)
    for recall in recalls:
        recall_dir = os.path.join(source_dir, f'recall{recall}')
        if not os.path.isdir(recall_dir):
            print(f"  [SKIP] {recall_dir} not found")
            continue

        print(f"\n{'='*60}\nrecall={recall}  (reading from {recall_dir})\n{'='*60}")
        qdepth_series = {}
        bw_series     = {}

        for entry in sorted(os.scandir(recall_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            csv_path = os.path.join(entry.path, 'diskstats.csv')
            if not os.path.exists(csv_path):
                print(f"  [SKIP] no diskstats.csv in {entry.name}")
                continue
            # Convert "threadS8_threadD24" → label "S8D24"
            name = entry.name
            label = name.replace('threadS', 'S').replace('_threadD', 'D')
            times, qdepth, bw = _load_diskstats_csv(csv_path)
            qdepth_series[label] = (times, qdepth)
            bw_series[label]     = (times, bw)
            print(f"  Loaded {len(times)} samples  [{label}]")

        if not bw_series:
            print(f"  [SKIP] no diskstats.csv files found under {recall_dir}")
            continue

        plot_metric(
            qdepth_series,
            os.path.join(recall_dir, 'queue_depth.png'),
            'Queue Depth (avgqu-sz)',
            f'I/O Queue Depth per 100ms — hybrid\n{dataset} · recall={recall} · T=32',
        )
        plot_metric(
            bw_series,
            os.path.join(recall_dir, 'read_bw.png'),
            'Read Bandwidth (MB/s)',
            f'I/O Read Bandwidth per 100ms — hybrid\n{dataset} · recall={recall} · T=32',
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='spacev100m')
    parser.add_argument('--recalls', nargs='+', type=int, default=[80, 90, 98])
    parser.add_argument('--device', default=DEVICE,
                        help='Block device name in /proc/diskstats (default: nvme0n1)')
    parser.add_argument(
        '--plot_dir', default=None,
        help='Plot-only mode: path to static_profiler output dir '
             '(contains recall<R>/threadS<s>_threadD<d>/diskstats.csv). '
             'No experiments are run.'
    )
    args = parser.parse_args()

    # ── Plot-only mode ──────────────────────────────────────────────────────
    if args.plot_dir:
        plot_from_static_profiler(args.plot_dir, args.recalls, args.dataset)
        return

    # ── Full run mode ────────────────────────────────────────────────────────
    base_yaml = os.path.join(QUINN, f'configs/{args.dataset}/{args.dataset}.yaml')
    base_ini  = os.path.join(QUINN, f'configs/{args.dataset}/{args.dataset}_searchconfig.ini')

    device = args.device
    with tempfile.TemporaryDirectory() as tmp_dir:
        for recall in args.recalls:
            out_base = os.path.join(QUINN, f'result/profiling/io_monitor/{args.dataset}_recall{recall}')
            os.makedirs(out_base, exist_ok=True)
            print(f"\n{'='*60}\nrecall={recall}\n{'='*60}")

            qdepth_series = {}
            bw_series     = {}

            for s, d in COMBOS:
                label   = f"S{s}D{d}"
                out_dir = os.path.join(out_base, label)
                times, qdepth, bw = run_combo(s, d, recall, out_dir, tmp_dir, base_yaml, base_ini)
                qdepth_series[label] = (times, qdepth)
                bw_series[label]     = (times, bw)

            plot_metric(
                qdepth_series,
                os.path.join(out_base, 'queue_depth.png'),
                'Queue Depth (avgqu-sz)',
                f'I/O Queue Depth per 100ms — hybrid\n{args.dataset} · recall={recall} · T=32  [{device}]'
            )
            plot_metric(
                bw_series,
                os.path.join(out_base, 'read_bw.png'),
                'Read Bandwidth (MB/s)',
                f'I/O Read Bandwidth per 100ms — hybrid\n{args.dataset} · recall={recall} · T=32  [{device}]'
            )


if __name__ == '__main__':
    main()

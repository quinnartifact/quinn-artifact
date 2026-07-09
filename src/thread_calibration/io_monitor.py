#!/usr/bin/env python3
"""
Diskstats-based I/O monitor for QUINN calibration runs.

Samples /proc/diskstats at a fixed interval while a workload runs, then
computes per-interval read bandwidth and IO queue depth.

Primary API: DiskstatsMonitor (start/stop or context manager).

Also exported for use by scripts/monitor.py:
  read_diskstats, compute_timeseries
"""

import threading
import time
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level diskstats reader
# ---------------------------------------------------------------------------

def read_diskstats(device: str) -> Optional[dict]:
    with open('/proc/diskstats') as f:
        for line in f:
            parts = line.split()
            if parts[2] == device:
                return {
                    'ts':             time.monotonic(),
                    'sectors_read':   int(parts[5]),
                    'weighted_io_ms': int(parts[13]),
                    'io_in_progress': int(parts[11]),
                }
    return None


def compute_timeseries(
    records: List[dict],
) -> Tuple[List[float], List[float], List[float]]:
    """Convert raw diskstats records into per-interval metrics.

    Returns:
        times_ms    — elapsed time from first sample (ms)
        queue_depth — avgqu-sz equivalent per interval
        read_bw_mb  — read bandwidth in MB/s per interval
    """
    if len(records) < 2:
        return [], [], []

    times_ms, queue_dep, read_bw = [], [], []
    t0 = records[0]['ts']
    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        dt_s  = curr['ts'] - prev['ts']
        dt_ms = dt_s * 1000

        d_weighted = curr['weighted_io_ms'] - prev['weighted_io_ms']
        qd = d_weighted / dt_ms if dt_ms > 0 else 0.0

        d_sectors = curr['sectors_read'] - prev['sectors_read']
        bw_mb = (d_sectors * 512) / dt_s / 1e6 if dt_s > 0 else 0.0

        times_ms.append((curr['ts'] - t0) * 1000)
        queue_dep.append(max(0.0, qd))
        read_bw.append(max(0.0, bw_mb))

    return times_ms, queue_dep, read_bw


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class DiskstatsMonitor:
    """Background I/O monitor — samples /proc/diskstats every `interval` s.

    Usage (explicit):
        monitor = DiskstatsMonitor(device='nvme0n1')
        monitor.start()
        subprocess.run(...)        # workload
        monitor.stop()
        bw = monitor.avg_read_bw_mb()
        qd = monitor.avg_queue_depth()

    Usage (context manager):
        with DiskstatsMonitor() as mon:
            subprocess.run(...)
        bw = mon.avg_read_bw_mb()
    """

    def __init__(self, device: str = 'nvme0n1', interval: float = 0.1):
        self.device   = device
        self.interval = interval
        self._records: List[dict] = []
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        self._records = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # -- internal ------------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            rec = read_diskstats(self.device)
            if rec:
                self._records.append(rec)
            time.sleep(self.interval)

    # -- results -------------------------------------------------------------

    def timeseries(self) -> Tuple[List[float], List[float], List[float]]:
        """Return (times_ms, queue_depth, read_bw_mb) lists."""
        return compute_timeseries(self._records)

    def avg_read_bw_mb(self) -> Optional[float]:
        _, _, bw = self.timeseries()
        return sum(bw) / len(bw) if bw else None

    def avg_queue_depth(self) -> Optional[float]:
        _, qd, _ = self.timeseries()
        return sum(qd) / len(qd) if qd else None

    def save_timeseries(self, path: str):
        """Save the full time series to a CSV file for later plotting."""
        import csv as _csv
        times, qdepth, bw = self.timeseries()
        with open(path, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['time_ms', 'queue_depth', 'read_bw_mb'])
            for t, q, b in zip(times, qdepth, bw):
                w.writerow([f'{t:.1f}', f'{q:.4f}', f'{b:.4f}'])

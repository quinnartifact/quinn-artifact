#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUINN Dynamic Thread Controller

IO-bandwidth-driven thread allocation for hybrid SPANN+DiskANN search.

Each monitoring window (interval_ms):
  1. Read IO read bandwidth from /proc/diskstats.
  2. Read Mean IO Latency and Mean Query Latency from LatencyShmWriter
     (only queries whose IO completed within this window).
  3. If a previous adjustment is pending a revert check:
       - If BOTH Mean IO Latency AND Mean Query Latency increased by more
         than revert_multiplier x pre-adjustment values → revert and
         permanently freeze that direction.
  4. Adjust based on IO BW:
       - BW > optimal_bw_high  →  reduce SPANN threads (give to DiskANN)
       - BW < optimal_bw_low   →  increase SPANN threads (take from DiskANN)
       - BW in range           →  hold
  5. When SPANN finishes: set thread_d = post_spann_diskann_threads.
"""

import threading
import time
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Diskstats helper
# ---------------------------------------------------------------------------

def _read_diskstats(device: str) -> Optional[dict]:
    """Read sectors_read and weighted_io_ms for device from /proc/diskstats."""
    try:
        with open('/proc/diskstats') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 14 and parts[2] == device:
                    return {
                        'ts':             time.monotonic(),
                        'sectors_read':   int(parts[5]),
                        'weighted_io_ms': int(parts[13]),
                    }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# ThresholdController
# ---------------------------------------------------------------------------

class ThresholdController:
    """
    IO-bandwidth-driven threshold controller.

    Main signal : IO read bandwidth (MB/s) from /proc/diskstats.
    Secondary   : Mean IO Latency and Mean Query Latency from SHM
                  (used only for revert decisions).

    Adjustment rules per window:
      BW > optimal_bw_high  →  SPANN -= step  (more threads for DiskANN)
      BW < optimal_bw_low   →  SPANN += step  (more threads for SPANN)
      otherwise             →  hold

    Revert rule (checked the window AFTER an adjustment):
      If BOTH mean_io_lat AND mean_q_lat increased by > revert_multiplier
      compared to pre-adjustment values → undo the step, freeze that
      direction permanently (never try it again).

    Config keys (all parameterised):
      total_threads            int    total thread budget
      init_thread_s            int    initial SPANN thread count
      step                     int    adjustment step size
      min_each                 int    minimum threads per side
      optimal_bw_low           float  MB/s lower bound of optimal BW range
      optimal_bw_high          float  MB/s upper bound of optimal BW range
      revert_multiplier        float  latency increase ratio that triggers revert
      device                   str    disk device name for /proc/diskstats
    """

    def __init__(self, total_threads: int, init_s: int,
                 step: int, min_each: int,
                 optimal_bw_low: float, optimal_bw_high: float,
                 revert_multiplier: float,
                 bw_growth_multiplier: float = 1.1,
                 device: str = 'nvme0n1',
                 cooldown_windows: int = 3):
        self.total        = int(total_threads)
        self._step        = int(step)
        self._min_each    = int(min_each)
        self._bw_low      = float(optimal_bw_low)
        self._bw_high     = float(optimal_bw_high)
        self._revert_mult = float(revert_multiplier)
        self._bw_growth_mult = float(bw_growth_multiplier)
        self._device          = device
        self._cooldown_total  = cooldown_windows

        self._thread_s = max(min_each, min(init_s, total_threads - min_each))

        # Freeze flags — once set, that direction is locked forever
        self._frozen_incr_d = False   # locked: cannot reduce SPANN further
        self._frozen_decr_d = False   # locked: cannot increase SPANN further

        # Revert tracking
        self._pending_dir = None   # +1 = gave DiskANN more; -1 = gave SPANN more
        self._pre_adj_lat = None   # mean latency before last adjustment
        self._pre_adj_bw  = None   # BW before last bw_low (+SPANN) adjustment

        # Cooldown: windows remaining before next adjustment is allowed
        self._cooldown_left  = 0

        # Set by notify_spann_done — stops further adjustments
        self._spann_done     = False

        # Diskstats baseline for BW delta
        self._prev_ds = _read_diskstats(device)

    @property
    def thread_s(self) -> int:
        return self._thread_s

    @property
    def thread_d(self) -> int:
        return self.total - self._thread_s

    def read_bw(self) -> Optional[float]:
        """Read IO read bandwidth (MB/s) since the last call."""
        curr = _read_diskstats(self._device)
        if curr is None or self._prev_ds is None:
            self._prev_ds = curr
            return None
        dt = curr['ts'] - self._prev_ds['ts']
        if dt <= 0:
            return None
        sectors_delta = curr['sectors_read'] - self._prev_ds['sectors_read']
        bw = max(0.0, sectors_delta * 512 / dt / 1e6)
        self._prev_ds = curr
        return bw

    def adjust(self,
               bw:       Optional[float],
               mean_lat: Optional[float],
               ) -> Tuple[int, int, bool, str, str]:
        """
        Decide new thread allocation for the current window.

        mean_lat: windowed mean DiskANN query latency (us), windowed by diskann_end.
                  Used only for revert decisions.

        Returns: (thread_s, thread_d, changed: bool, action_description: str, trigger: str)
          trigger codes: bw_high | bw_low | revert | freeze | momentum | spann_done | cooldown | hold | no_bw
        """
        # ── Guard: SPANN is done — no further adjustments ─────────────
        if self._spann_done:
            return self._thread_s, self.thread_d, False, 'spann_done(hold)', 'spann_done'

        # ── Cooldown: wait before allowing next adjustment ─────────────
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return self._thread_s, self.thread_d, False, f'cooldown({self._cooldown_left})', 'cooldown'

        # ── Step 1: Revert check from previous adjustment ──────────────
        if self._pending_dir is not None:
            direction = self._pending_dir

            # Latency revert: if latency got worse after any adjustment
            lat_worse = (
                mean_lat is not None and
                self._pre_adj_lat is not None and
                self._pre_adj_lat > 0 and
                mean_lat > self._pre_adj_lat * self._revert_mult
            )

            # BW growth revert: if +SPANN (direction=-1) didn't raise BW enough
            bw_no_growth = (
                direction == -1 and
                bw is not None and
                self._pre_adj_bw is not None and
                self._pre_adj_bw > 0 and
                bw < self._pre_adj_bw * self._bw_growth_mult
            )

            # Save for string formatting before any clearing
            pre_bw = self._pre_adj_bw

            def _dist_to_safe(b):
                if b is None: return float('inf')
                if b < self._bw_low:  return self._bw_low - b
                if b > self._bw_high: return b - self._bw_high
                return 0.0

            # Both-outside: if pre AND post BW are both outside the safe zone,
            # pick whichever thread allocation is closer to the safe zone and freeze.
            pre_outside  = pre_bw is not None and _dist_to_safe(pre_bw) > 0
            post_outside = bw    is not None and _dist_to_safe(bw)     > 0
            both_outside = pre_outside and post_outside

            # Momentum: bw_high adjustment dropped BW into the safe zone → keep going
            bw_effective = (
                direction == +1 and
                not lat_worse and
                not self._frozen_incr_d and
                bw is not None and
                pre_bw is not None and
                not both_outside and
                bw < pre_bw
            )

            if lat_worse:
                # Latency degraded → revert + freeze
                new_s = self._thread_s + direction * self._step
                new_s = max(self._min_each, min(self.total - self._min_each, new_s))
                self._thread_s = new_s
                if direction > 0: self._frozen_incr_d = True
                else:             self._frozen_decr_d = True
                self._pending_dir = None
                self._pre_adj_lat = None
                self._pre_adj_bw  = None
                self._cooldown_left = self._cooldown_total
                action = (
                    f'REVERT(dir={direction:+d},reason=lat) S={new_s} D={self.thread_d} '
                    f'[FROZEN dir={direction:+d}]'
                )
                return self._thread_s, self.thread_d, True, action, 'revert'

            elif bw_no_growth:
                # +SPANN adjustment didn't raise BW enough → revert + freeze
                new_s = self._thread_s + direction * self._step
                new_s = max(self._min_each, min(self.total - self._min_each, new_s))
                self._thread_s = new_s
                self._frozen_decr_d = True
                self._pending_dir = None
                self._pre_adj_lat = None
                self._pre_adj_bw  = None
                self._cooldown_left = self._cooldown_total
                action = (
                    f'REVERT(dir={direction:+d},reason=bw_no_growth({bw:.0f}<{pre_bw * self._bw_growth_mult:.0f}MB/s)) '
                    f'S={new_s} D={self.thread_d} [FROZEN dir={direction:+d}]'
                )
                return self._thread_s, self.thread_d, True, action, 'revert'

            elif both_outside:
                # Both pre and post BW outside safe zone → pick closer allocation + freeze
                dist_pre  = _dist_to_safe(pre_bw)
                dist_post = _dist_to_safe(bw)
                if direction > 0: self._frozen_incr_d = True
                else:             self._frozen_decr_d = True
                self._pending_dir = None
                self._pre_adj_lat = None
                self._pre_adj_bw  = None
                self._cooldown_left = self._cooldown_total
                if dist_pre <= dist_post:
                    # pre-adjustment allocation was closer → revert
                    new_s = self._thread_s + direction * self._step
                    new_s = max(self._min_each, min(self.total - self._min_each, new_s))
                    self._thread_s = new_s
                    reason = f'both_outside(revert,dist_pre={dist_pre:.0f}<dist_post={dist_post:.0f})'
                    changed = True
                else:
                    # post-adjustment allocation is closer → keep current
                    reason = f'both_outside(keep,dist_post={dist_post:.0f}<dist_pre={dist_pre:.0f})'
                    changed = False
                action = (
                    f'FREEZE(dir={direction:+d},reason={reason}) S={self._thread_s} D={self.thread_d} '
                    f'[FROZEN dir={direction:+d}]'
                )
                return self._thread_s, self.thread_d, changed, action, 'freeze'

            elif bw_effective:
                # BW moved in the right direction and lat is fine → keep going
                candidate = max(self._min_each, self._thread_s - self._step)
                self._pending_dir = None
                self._pre_adj_lat = None
                self._pre_adj_bw  = None
                if candidate != self._thread_s:
                    self._thread_s = candidate
                    self._pre_adj_lat = mean_lat if (mean_lat and mean_lat >= 100.0) else None
                    self._pre_adj_bw  = bw
                    self._pending_dir = +1
                    self._cooldown_left = self._cooldown_total
                    action = (
                        f'MOMENTUM(dir=+1) BW={bw:.0f}<{pre_bw:.0f}MB/s '
                        f'-> -SPANN S={candidate} D={self.total - candidate}'
                    )
                    return self._thread_s, self.thread_d, True, action, 'momentum'
                # else: already at min, fall through to hold

            else:
                self._pending_dir = None
                self._pre_adj_lat = None
                self._pre_adj_bw  = None

        # ── Step 2: No BW data → hold ──────────────────────────────────
        if bw is None:
            return self._thread_s, self.thread_d, False, 'no_bw_data', 'no_bw'

        # ── Step 3: Adjust based on IO BW ─────────────────────────────
        direction = 0
        new_s     = self._thread_s
        action    = f'hold(BW={bw:.0f}MB/s)'
        trigger   = 'hold'

        if bw > self._bw_high and not self._frozen_incr_d:
            candidate = max(self._min_each, self._thread_s - self._step)
            if candidate != self._thread_s:
                new_s     = candidate
                direction = +1
                trigger   = 'bw_high'
                action    = (f'BW={bw:.0f}>{self._bw_high:.0f}MB/s '
                             f'-> -SPANN S={new_s} D={self.total - new_s}')

        elif bw < self._bw_low and not self._frozen_decr_d:
            candidate = min(self.total - self._min_each, self._thread_s + self._step)
            if candidate != self._thread_s:
                new_s     = candidate
                direction = -1
                trigger   = 'bw_low'
                action    = (f'BW={bw:.0f}<{self._bw_low:.0f}MB/s '
                             f'-> +SPANN S={new_s} D={self.total - new_s}')

        if direction != 0:
            # Only store baseline if it's a plausible DiskANN latency (>= 100us).
            # Very small values (< 100us) appear at t=0 before queries are in flight
            # and would cause spurious reverts on the next real measurement.
            self._pre_adj_lat = mean_lat if (mean_lat and mean_lat >= 100.0) else None
            # Record current BW for post-adjustment verification:
            #   direction=+1 (bw_high): verify BW dropped (momentum check)
            #   direction=-1 (bw_low):  verify BW grew   (growth check)
            self._pre_adj_bw = bw
            self._pending_dir = direction
            self._thread_s    = new_s
            self._cooldown_left = self._cooldown_total
            return self._thread_s, self.thread_d, True, action, trigger

        return self._thread_s, self.thread_d, False, action, trigger


# ---------------------------------------------------------------------------
# DynamicThreadMonitor
# ---------------------------------------------------------------------------

class DynamicThreadMonitor:
    """
    Background thread that drives ThresholdController each window.

    Each interval_ms window:
      - Reads IO BW via ThresholdController.read_bw()
      - Reads Mean IO Latency and Mean Query Latency from LatencyShmWriter
        (only queries whose IO completed within this window)
      - Calls ThresholdController.adjust() and updates ThreadCountShmWriter

    notify_spann_done(post_d_threads):
      Call when SPANN finishes. Immediately sets thread_d = post_d_threads,
      thread_s = total - post_d_threads in SHM.
    """

    def __init__(self,
                 latency_shm_writer,
                 thread_count_shm_writer,
                 controller: ThresholdController,
                 interval_ms: int = 100,
                 timeseries_log: Optional[str] = None):
        self._lat_shm        = latency_shm_writer
        self._tc_shm         = thread_count_shm_writer
        self._ctrl           = controller
        self._interval_s     = interval_ms / 1000.0
        self._timeseries_log = timeseries_log

        self._stop_event = threading.Event()
        self._thread     = None
        self._history    = []       # (t, action, s, d, bw, mean_lat)
        self._log_handle = None

    def start(self):
        if self._timeseries_log:
            try:
                import os
                os.makedirs(
                    os.path.dirname(os.path.abspath(self._timeseries_log)),
                    exist_ok=True)
                self._log_handle = open(self._timeseries_log, 'w')
                self._log_handle.write(
                    'time_s,thread_s,thread_d,bw_mb,mean_lat_us,'
                    'changed,trigger,frozen_s_to_d,frozen_d_to_s,action\n')
                self._log_handle.flush()
            except Exception as e:
                print(f"[DynamicThreadMonitor] Warning: cannot open log: {e}")
                self._log_handle = None

        self._thread = threading.Thread(
            target=self._run, daemon=True, name='DynamicThreadMonitor')
        self._thread.start()
        print(f"[DynamicThreadMonitor] Started "
              f"(interval={self._interval_s*1000:.0f}ms "
              f"S={self._ctrl.thread_s} D={self._ctrl.thread_d} "
              f"BW_range=[{self._ctrl._bw_low:.0f},{self._ctrl._bw_high:.0f}]MB/s "
              f"revert_mult={self._ctrl._revert_mult})")

    def stop_log(self):
        """Close the trace log immediately (e.g. when DiskANN finishes).
        The monitor thread keeps running for SPANN thread handoff."""
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
        print("[DynamicThreadMonitor] Stopped.")

    def notify_spann_done(self, post_d_threads: int):
        """Immediately hand remaining threads to DiskANN when SPANN finishes.
        Also sets _spann_done=True to freeze further adjustments."""
        post_s = max(0, self._ctrl.total - post_d_threads)
        self._ctrl._thread_s  = post_s
        self._ctrl._spann_done = True
        try:
            self._tc_shm.update(post_s, post_d_threads)
            print(f"[DynamicThreadMonitor] SPANN done "
                  f"-> thread_s={post_s} thread_d={post_d_threads} (adjustments frozen)")
        except Exception as e:
            print(f"[DynamicThreadMonitor] Warning: notify_spann_done failed: {e}")

    def _read_window_metrics(self, window_ns: int) -> Optional[float]:
        """
        Return mean DiskANN total query latency (us) for queries whose DiskANN
        query completed within the last window_ns nanoseconds.

        Formula: mean(diskann_end - diskann_start), windowed by diskann_end.
        Same metric as hybrid_io_timeseries.py's total_us / stat_L100.csv.
        """
        try:
            _ss, _se, ds, de = self._lat_shm.read_all()
        except Exception:
            return None

        now_ns = time.time_ns()
        mask = (
            (ds > 0) & (de > 0) &
            (de > now_ns - window_ns) &
            (de <= now_ns)
        )
        if not np.any(mask):
            return None

        return float(np.mean(de[mask] - ds[mask])) / 1000.0

    def _run(self):
        t0        = time.monotonic()
        window_ns = int(self._interval_s * 1e9)
        first_tick = True

        while not self._stop_event.is_set():
            t_tick = time.monotonic()

            bw       = self._ctrl.read_bw()
            mean_lat = self._read_window_metrics(window_ns)

            # Skip the first tick: BW reading covers the gap from __init__ to
            # first tick and is not representative of steady-state throughput.
            if first_tick:
                first_tick = False
                self._stop_event.wait(max(0.0, self._interval_s - (time.monotonic() - t_tick)))
                continue

            new_s, new_d, changed, action, trigger = self._ctrl.adjust(bw, mean_lat)

            if changed:
                try:
                    self._tc_shm.update(new_s, new_d)
                    print(f"[DynamicThreadMonitor] {action}")
                except Exception as e:
                    print(f"[DynamicThreadMonitor] Warning: SHM update failed: {e}")

            ts = t_tick - t0
            self._history.append((ts, action, new_s, new_d, bw, mean_lat))

            if self._log_handle:
                bw_s  = f'{bw:.1f}'      if bw       is not None else ''
                lat_s = f'{mean_lat:.1f}' if mean_lat is not None else ''
                frozen_s2d = int(self._ctrl._frozen_incr_d)
                frozen_d2s = int(self._ctrl._frozen_decr_d)
                self._log_handle.write(
                    f'{ts:.3f},{new_s},{new_d},{bw_s},{lat_s},'
                    f'{int(changed)},{trigger},{frozen_s2d},{frozen_d2s},'
                    f'{action.replace(",", ";")}\n')
                self._log_handle.flush()

            elapsed = time.monotonic() - t_tick
            self._stop_event.wait(max(0.0, self._interval_s - elapsed))

    def print_summary(self):
        print("\n[DynamicThreadMonitor] Adjustment summary:")
        print(f"  Total monitoring ticks: {len(self._history)}")
        if not self._history:
            return

        prev_s = prev_d = None
        transitions = []
        for ts, act, s, d, bw, lat in self._history:
            if s != prev_s or d != prev_d:
                transitions.append((ts, act, s, d, bw, lat))
                prev_s, prev_d = s, d

        print(f"  Thread count transitions: {len(transitions)}")
        for ts, act, s, d, bw, lat in transitions:
            bw_s  = f'{bw:.0f}MB/s'  if bw  is not None else 'N/A'
            lat_s = f'{lat:.0f}us'   if lat is not None else 'N/A'
            print(f'    t={ts:7.2f}s  S={s:3d} D={d:3d}  '
                  f'BW={bw_s:10s} Lat={lat_s:10s}  | {act}')

        final = self._history[-1]
        print(f"  Final: thread_s={final[2]}  thread_d={final[3]}")
        if self._timeseries_log:
            print(f"  Time-series log: {self._timeseries_log}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/controller/shm/shm.py — POSIX shared memory writer implementation

Defines the Writer classes the controller side uses to create and
initialize the various SHM segments. Each class corresponds to one data
format read on the C++ side; both the Python and C++ sides use the same
little-endian binary layout and magic-number verification.

BudgetShmWriter      format: CASA header (16B) + (bS uint16, bD uint16) × N
EarlyExitShmWriter   format: EXIT header (20B) + float[N] + uint32[N×topk_k]
HubCacheShmWriter    format: HUBC header (32B) + per-query slot[N]
                           slot = num_hubs (uint32) + HubEntry[max_hubs]
                           HubEntry = diskann_id + degree + neighbors + coords
LatencyShmWriter     format: LATC header (16B) + int64[N×4]
                           fields: spann_start_ns, spann_end_ns,
                                 diskann_start_ns, diskann_io_ns
ThreadCountShmWriter format: THRC header (16B) = magic + version + thread_s + thread_d

Usage (all support the context manager protocol):
    with BudgetShmWriter('/quinn_budget_123', budgets) as shm:
        subprocess.Popen([diskann_bin, '--budget_shm', '/quinn_budget_123', ...])
"""

import mmap
import os
import struct
import numpy as np
from pathlib import Path

class BudgetShmWriter:
    """
    Python implementation of the Budget Shared Memory Writer

    Follows the format defined in budget_shm.h:
    - Header: magic (4B), version (4B), num_queries (4B), entry_size (4B)
    - Entries: array of (bS uint16, bD uint16)
    """

    MAGIC = 0x43415341  # 'CASA'
    VERSION = 1
    ENTRY_SIZE = 4  # uint16 + uint16

    def __init__(self, shm_name: str, budgets: np.ndarray):
        """
        Create and populate the shared memory

        Args:
            shm_name: Shared memory name (must start with '/')
            budgets: (N, 2) array of (bS, bD) pairs (uint16)
        """
        if not shm_name.startswith('/'):
            raise ValueError("shm_name must start with '/'")

        if budgets.ndim != 2 or budgets.shape[1] != 2:
            raise ValueError("budgets must be (N, 2) array")

        self.shm_name = shm_name
        self.budgets = budgets.astype(np.uint16)
        self.num_queries = len(budgets)

        # Compute the size
        self.header_size = 16  # 4 * uint32
        self.data_size = self.num_queries * self.ENTRY_SIZE
        self.total_size = self.header_size + self.data_size

        self._create_and_fill()

    def _create_and_fill(self):
        """Create the shared memory and write the data"""
        import posix_ipc

        # Create the shared memory
        try:
            # First try to delete an old one (if it exists)
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass

            # Create a new one
            self.shm = posix_ipc.SharedMemory(
                self.shm_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                mode=0o600,
                size=self.total_size
            )

            # Truncate to size
            os.ftruncate(self.shm.fd, self.total_size)

            # mmap
            self.mapfile = mmap.mmap(self.shm.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)

            # Write the header
            header = struct.pack('<IIII', self.MAGIC, self.VERSION, self.num_queries, self.ENTRY_SIZE)
            self.mapfile[0:16] = header

            # Write the entries
            offset = self.header_size
            # Write in vectorized form for efficiency
            # Convert the (N, 2) uint16 array to bytes and write it all at once
            entries_bytes = self.budgets.tobytes()
            self.mapfile[offset:offset+len(entries_bytes)] = entries_bytes

            # Flush
            self.mapfile.flush()

            print(f"[BudgetShmWriter] Created shm: {self.shm_name} ({self.num_queries} queries, {self.total_size} bytes)")

        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")

    def cleanup(self):
        """Release resources"""
        import posix_ipc

        try:
            if hasattr(self, 'mapfile'):
                self.mapfile.close()

            if hasattr(self, 'shm'):
                self.shm.close_fd()
                posix_ipc.unlink_shared_memory(self.shm_name)

            print(f"[BudgetShmWriter] Cleaned up shm: {self.shm_name}")

        except Exception as e:
            print(f"[BudgetShmWriter] Warning: cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class LatencyShmWriter:
    """
    Python controller side of per-query latency timestamp SHM.

    C++ counterpart: quinn/latency_shm.h  (LatencyShmAccessor)

    Layout:
      Header  (16B): magic(4B) version(4B) num_queries(4B) pad(4B)
      Entries       : int64[num_queries * 4] — zero-initialised
        per query:  spann_start_ns, spann_end_ns, diskann_start_ns, diskann_io_ns
                    NOTE: diskann_io_ns is a DURATION (QueryStats::io_us * 1000), not a timestamp.
    """

    MAGIC       = 0x4C415443  # 'LATC'
    VERSION     = 1
    HEADER_SIZE = 16          # 4 × uint32
    ENTRY_SIZE  = 32          # 4 × int64

    def __init__(self, shm_name: str, num_queries: int):
        if not shm_name.startswith('/'):
            raise ValueError("shm_name must start with '/'")
        self.shm_name    = shm_name
        self.num_queries = int(num_queries)
        self.total_size  = self.HEADER_SIZE + self.num_queries * self.ENTRY_SIZE
        self._create_and_fill()

    def _create_and_fill(self):
        import posix_ipc
        try:
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass

            self.shm = posix_ipc.SharedMemory(
                self.shm_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                mode=0o600,
                size=self.total_size
            )
            os.ftruncate(self.shm.fd, self.total_size)
            self.mapfile = mmap.mmap(self.shm.fd, self.total_size,
                                     mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)

            header = struct.pack('<IIII', self.MAGIC, self.VERSION, self.num_queries, 0)
            self.mapfile[0:self.HEADER_SIZE] = header
            self.mapfile[self.HEADER_SIZE:self.total_size] = bytes(self.num_queries * self.ENTRY_SIZE)
            self.mapfile.flush()

            print(f"[LatencyShmWriter] Created shm: {self.shm_name} "
                  f"({self.num_queries} queries, {self.total_size} bytes)")
        except Exception as e:
            raise RuntimeError(f"Failed to create latency SHM: {e}")

    def read_all(self):
        """Return (spann_start_ns, spann_end_ns, diskann_start_ns, diskann_end_ns).

        All four values are wall-clock timestamps (CLOCK_REALTIME ns).
        0 means "not recorded" (service disabled or not yet finished).
        """
        self.mapfile.seek(self.HEADER_SIZE)
        raw = self.mapfile.read(self.num_queries * self.ENTRY_SIZE)
        arr = np.frombuffer(raw, dtype=np.int64).reshape(self.num_queries, 4).copy()
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    def compute_stats(self):
        """Compute per-query latency statistics (ms).
        Returns (mean, p50, p90, p99, p99_9).

        Per-query latency = max(spann self-duration, diskann self-duration).
        Using max() of each service's own duration avoids batch-mode drift
        (where independent queues make cross-service timestamps incomparable).
        Falls back to whichever service recorded data if the other is missing.
        """
        ss, se, ds, de = self.read_all()
        spann_ok   = (ss > 0) & (se > 0)
        diskann_ok = (ds > 0) & (de > 0)

        spann_dur   = np.where(spann_ok,   se - ss, 0)
        diskann_dur = np.where(diskann_ok, de - ds, 0)

        valid  = spann_ok | diskann_ok
        lat_ms = np.maximum(spann_dur[valid], diskann_dur[valid]) / 1e6
        if len(lat_ms) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return float(np.mean(lat_ms)), \
               float(np.percentile(lat_ms, 50)), \
               float(np.percentile(lat_ms, 90)), \
               float(np.percentile(lat_ms, 99)), \
               float(np.percentile(lat_ms, 99.9))

    def cleanup(self):
        import posix_ipc
        try:
            if hasattr(self, 'mapfile'):
                self.mapfile.close()
            if hasattr(self, 'shm'):
                self.shm.close_fd()
                posix_ipc.unlink_shared_memory(self.shm_name)
            print(f"[LatencyShmWriter] Cleaned up shm: {self.shm_name}")
        except Exception as e:
            print(f"[LatencyShmWriter] Warning: cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class HubCacheShmWriter:
    """
    Python implementation of the Hub Cache Shared Memory Writer

    Follows the format defined in hub_cache_shm.h v1:
    - Header (32B): magic, version, num_queries, max_hubs, hub_k,
                    coord_dim, coord_bytes, element_size
    - Per-query slots: uint32 num_hubs (atomic) + HubEntry[max_hubs]
    - HubEntry: uint32 diskann_id + uint32 degree + uint32[hub_k] neighbors
                + uint8[coord_bytes] coords

    Created and initialized by the Python controller; SPANN writes to it, DiskANN reads from it.
    """

    MAGIC      = 0x48554243  # 'HUBC'
    VERSION    = 1
    HEADER_SIZE = 32         # 8 × uint32

    def __init__(self, shm_name: str, num_queries: int,
                 max_hubs: int, hub_k: int,
                 coord_dim: int, element_size: int = 4):
        """
        Args:
            shm_name:     Shared memory name (must start with '/')
            num_queries:  Number of queries N
            max_hubs:     Max hub entries per query (e.g. searchInternalResultNum)
            hub_k:        Hub neighbor count per entry (= HubNeighborCount)
            coord_dim:    Vector dimension
            element_size: Bytes per coordinate element (4 for float32, 1 for uint8)
        """
        if not shm_name.startswith('/'):
            raise ValueError("shm_name must start with '/'")

        self.shm_name    = shm_name
        self.num_queries = int(num_queries)
        self.max_hubs    = int(max_hubs)
        self.hub_k       = int(hub_k)
        self.coord_dim   = int(coord_dim)
        self.element_size = int(element_size)
        self.coord_bytes = self.coord_dim * self.element_size

        # entry_size = 4 (diskann_id) + 4 (degree) + hub_k*4 (neighbors) + coord_bytes
        self.entry_size = 4 + 4 + self.hub_k * 4 + self.coord_bytes
        # slot_size = 4 (num_hubs) + max_hubs * entry_size
        self.slot_size = 4 + self.max_hubs * self.entry_size
        self.total_size = self.HEADER_SIZE + self.num_queries * self.slot_size

        self._create_and_fill()

    def _create_and_fill(self):
        """Create the shared memory and zero-initialize it"""
        import posix_ipc

        try:
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass

            self.shm = posix_ipc.SharedMemory(
                self.shm_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                mode=0o600,
                size=self.total_size
            )

            import os
            os.ftruncate(self.shm.fd, self.total_size)
            import mmap as mmap_mod
            self.mapfile = mmap_mod.mmap(self.shm.fd, self.total_size,
                                         mmap_mod.MAP_SHARED,
                                         mmap_mod.PROT_WRITE | mmap_mod.PROT_READ)

            # Write header: magic, version, num_queries, max_hubs, hub_k,
            #               coord_dim, coord_bytes, element_size
            import struct
            header = struct.pack('<IIIIIIII',
                                 self.MAGIC, self.VERSION,
                                 self.num_queries, self.max_hubs,
                                 self.hub_k, self.coord_dim,
                                 self.coord_bytes, self.element_size)
            self.mapfile[0:self.HEADER_SIZE] = header

            # Zero-fill all per-query slots (num_hubs=0 signals "not yet finalized")
            import numpy as np
            data_size = self.total_size - self.HEADER_SIZE
            zeros = bytes(data_size)
            self.mapfile[self.HEADER_SIZE:self.total_size] = zeros

            self.mapfile.flush()

            print(f"[HubCacheShmWriter] Created shm: {self.shm_name} "
                  f"({self.num_queries} queries, max_hubs={self.max_hubs}, "
                  f"hub_k={self.hub_k}, dim={self.coord_dim}, "
                  f"total={self.total_size} bytes)")

        except Exception as e:
            raise RuntimeError(f"Failed to create hub cache shared memory: {e}")

    def cleanup(self):
        """Release resources"""
        import posix_ipc

        try:
            if hasattr(self, 'mapfile'):
                self.mapfile.close()
            if hasattr(self, 'shm'):
                self.shm.close_fd()
                posix_ipc.unlink_shared_memory(self.shm_name)
            print(f"[HubCacheShmWriter] Cleaned up shm: {self.shm_name}")
        except Exception as e:
            print(f"[HubCacheShmWriter] Warning: cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class EarlyExitShmWriter:
    """
    Python implementation of the Early Exit Shared Memory Writer

    Follows the format defined in early_exit_shm.h v2:
    - Header (20B): magic, version, num_queries, entry_size, topk_k
    - Float section: float[num_queries], initialized to FLT_MAX
    - TopK IDs section: uint32[num_queries][topk_k], initialized to UINT32_MAX (if topk_k > 0)
    """

    MAGIC      = 0x45584954  # 'EXIT'
    VERSION    = 2
    ENTRY_SIZE = 4           # sizeof(float)
    HEADER_SIZE = 20         # 5 * uint32

    def __init__(self, shm_name: str, num_queries: int, topk_k: int = 0):
        """
        Args:
            shm_name:    Shared memory name (must start with '/')
            num_queries: Number of queries N
            topk_k:      Number of topk IDs per query to store (0 = disabled)
        """
        if not shm_name.startswith('/'):
            raise ValueError("shm_name must start with '/'")

        self.shm_name    = shm_name
        self.num_queries = int(num_queries)
        self.topk_k      = int(topk_k)

        # Layout: header + float[N] + uint32[N][K]
        self.float_size = self.num_queries * 4
        self.ids_size   = self.num_queries * self.topk_k * 4
        self.total_size = self.HEADER_SIZE + self.float_size + self.ids_size

        self._create_and_fill()

    def _create_and_fill(self):
        """Create the shared memory and initialize the data"""
        import posix_ipc

        try:
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass

            self.shm = posix_ipc.SharedMemory(
                self.shm_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                mode=0o600,
                size=self.total_size
            )

            os.ftruncate(self.shm.fd, self.total_size)
            self.mapfile = mmap.mmap(self.shm.fd, self.total_size,
                                     mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)

            # Header: magic, version, num_queries, entry_size, topk_k
            header = struct.pack('<IIIII', self.MAGIC, self.VERSION,
                                 self.num_queries, self.ENTRY_SIZE, self.topk_k)
            self.mapfile[0:self.HEADER_SIZE] = header

            # Float section: init to FLT_MAX
            flt_max = np.finfo(np.float32).max
            floats = np.full(self.num_queries, flt_max, dtype=np.float32)
            float_bytes = floats.tobytes()
            self.mapfile[self.HEADER_SIZE:self.HEADER_SIZE + self.float_size] = float_bytes

            # TopK IDs section: init to UINT32_MAX
            if self.topk_k > 0:
                ids = np.full(self.num_queries * self.topk_k, 0xFFFFFFFF, dtype=np.uint32)
                ids_bytes = ids.tobytes()
                offset = self.HEADER_SIZE + self.float_size
                self.mapfile[offset:offset + self.ids_size] = ids_bytes

            self.mapfile.flush()

            print(f"[EarlyExitShmWriter] Created shm: {self.shm_name} "
                  f"({self.num_queries} queries, topk_k={self.topk_k}, {self.total_size} bytes)")

        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")

    def cleanup(self):
        """Release resources"""
        import posix_ipc

        try:
            if hasattr(self, 'mapfile'):
                self.mapfile.close()

            if hasattr(self, 'shm'):
                self.shm.close_fd()
                posix_ipc.unlink_shared_memory(self.shm_name)

            print(f"[EarlyExitShmWriter] Cleaned up shm: {self.shm_name}")

        except Exception as e:
            print(f"[EarlyExitShmWriter] Warning: cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ThreadCountShmWriter:
    """
    Python-side writer for the dynamic thread count shared memory.

    C++ counterpart: quinn/thread_count_shm.h (ThreadCountShmReader)

    Layout (16 bytes total):
      magic    uint32  0x54485243 ('THRC')
      version  uint32  1
      thread_s int32   allowed SPANN threads
      thread_d int32   allowed DiskANN threads
    """

    MAGIC   = 0x54485243  # 'THRC'
    VERSION = 1
    SIZE    = 16  # 4 x 4 bytes

    def __init__(self, shm_name: str, thread_s: int, thread_d: int):
        if not shm_name.startswith('/'):
            raise ValueError("shm_name must start with '/'")
        self.shm_name = shm_name
        self._create_and_fill(thread_s, thread_d)

    def _create_and_fill(self, thread_s: int, thread_d: int):
        import posix_ipc

        try:
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass

            self.shm = posix_ipc.SharedMemory(
                self.shm_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                mode=0o600,
                size=self.SIZE
            )
            os.ftruncate(self.shm.fd, self.SIZE)
            self.mapfile = mmap.mmap(self.shm.fd, self.SIZE,
                                     mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)

            data = struct.pack('<IIii', self.MAGIC, self.VERSION, thread_s, thread_d)
            self.mapfile[0:self.SIZE] = data
            self.mapfile.flush()

            print(f"[ThreadCountShmWriter] Created shm: {self.shm_name} "
                  f"(thread_s={thread_s}, thread_d={thread_d})")

        except Exception as e:
            raise RuntimeError(f"Failed to create thread count shared memory: {e}")

    def update(self, thread_s: int, thread_d: int):
        """Atomically update thread counts."""
        data = struct.pack('<IIii', self.MAGIC, self.VERSION, thread_s, thread_d)
        self.mapfile.seek(0)
        self.mapfile.write(data)
        self.mapfile.flush()

    def cleanup(self):
        import posix_ipc

        try:
            if hasattr(self, 'mapfile'):
                self.mapfile.close()
            if hasattr(self, 'shm'):
                self.shm.close_fd()
                posix_ipc.unlink_shared_memory(self.shm_name)
            print(f"[ThreadCountShmWriter] Cleaned up shm: {self.shm_name}")
        except Exception as e:
            print(f"[ThreadCountShmWriter] Warning: cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

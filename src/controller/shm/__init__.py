"""
src/controller/shm — QUINN POSIX Shared Memory Writers

The Python-side shared memory management module. controller.py uses it to
create the various SHM segments before launching the DiskANN/SPANN
subprocesses, and automatically cleans them up once the search finishes.

Exported Writer classes:
  BudgetShmWriter      — per-query (b_S, b_D) budget, read by the C++ binary
  EarlyExitShmWriter   — early-exit distance threshold (float) + SPANN topk IDs (optional)
  HubCacheShmWriter    — coordinates and neighbor info for SPANN head nodes, for DiskANN to prefetch
  LatencyShmWriter     — per-query SPANN/DiskANN start/end timestamps (ns)
  ThreadCountShmWriter — dynamic thread allocation counts (SPANN threads / DiskANN threads)

All Writers implement the context manager protocol and automatically unlink
the SHM segment on exiting the with block.
The corresponding C++ Reader header is in shm/budget_shm.h.
"""
from .shm import BudgetShmWriter, EarlyExitShmWriter, HubCacheShmWriter, LatencyShmWriter, ThreadCountShmWriter

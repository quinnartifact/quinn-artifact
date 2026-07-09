# QUINN

## Introduction

QUINN runs DiskANN (graph index) and SPTAG/SPANN (SPTAG's disk-based
partitioned-index algorithm) in parallel per query and merges their results.
A GBDT-based controller predicts a per-query I/O budget split between the
two engines (`b_S` for SPANN, `b_D` for DiskANN) and communicates it at
runtime over POSIX shared memory. An optional dynamic thread scheduler
rebalances CPU threads between the two engines mid-run based on observed
disk bandwidth.

```
query → [controller.py: predict budgets] → SHM → {DiskANN, SPANN} (parallel)
                                                         ↓
                                              [merger.py: combine + score]
```

This repository provides an anonymized submission-time snapshot of the QUINN
implementation and experimental workflow. It is not a one-command
reproduction package for the full evaluation. Full-scale
experiments require public benchmark datasets, large disk-resident indexes, 
and long-running evaluations.

## Layout

```
configs/            per-dataset controller configs (YAML) + SPTAG search configs (INI)
scripts/            entry-point scripts (see Usage below)
src/
  controller/       runtime: controller.py orchestrates DiskANN+SPANN, allocator.py
                    predicts budgets, merger.py combines results, shm/ for SHM I/O
  offline_training/ offline: saved profiling results → oracle labels → GBDT training
  threading/        runtime dynamic thread scheduler (rebalances threads by disk BW)
  thread_calibration/  static thread-allocation profiling + I/O bandwidth monitoring
  io_trace/         header-only Chrome-Trace-format I/O tracing (view in Perfetto)
  util/             vector I/O (fvecs/fbin) and format-conversion utilities
results/            selected processed CSV summaries for reference; full raw logs are not included
third_party/        vendored DiskANN and SPTAG/SPANN, modified to read
                    per-query budgets from POSIX shared memory
```

## Build

1. **DiskANN and SPTAG/SPANN**, from the repository root:
   ```bash
   cmake -S third_party/diskann -B third_party/diskann/build -DCMAKE_BUILD_TYPE=Release
   cmake --build third_party/diskann/build -j

   cmake -S third_party/sptag -B third_party/sptag/build -DCMAKE_BUILD_TYPE=Release
   cmake --build third_party/sptag/build -j
   ```
   Both are modified from upstream to read per-query search budgets from
   POSIX shared memory (`include/quinn/budget_shm.h` in DiskANN,
   `AnnService/inc/quinn/` in SPTAG). This produces the `search_disk_index`
   and `ssdserving` binaries referenced as `diskann_bin` / `spann_bin` in
   the configs. See each project's own `README.md` for dependencies.

2. **A disk-resident index per dataset**, for both engines — DiskANN's
   on-disk graph index and SPTAG/SPANN's partitioned index — referenced as
   `index_path_prefix` / `IndexDirectory` in the configs.

3. **An external data root** (`<PATH>` throughout the configs) with, per
   dataset: base/query/ground-truth vectors and the SPTAG head/centroid
   file (`SPTAGHeadVectors.bin`, used to compute routing features).

4. **Independent offline profiling runs** — only needed before
   `scripts/train.sh`. Run DiskANN and SPANN once each, independently, over
   a sweep of budgets against a held-out query sample, saving one result +
   one stats file per budget value:

   | Engine  | Budget swept | Result file (`--..._result_dir`)     | Stats file (`--..._stats_dir`)  |
   |---------|--------------|---------------------------------------|----------------------------------|
   | SPANN   | `nprobe`     | `result_K100_<nprobe>.bin` — binary result file containing top-k uint32 ids and float32 distances | `stat_K100_<nprobe>.csv` — columns `qid, n_ios` |
   | DiskANN | `L`          | `result_<L>_idx_uint32.bin` — binary, `(nq, k)` uint32 ids (DiskANN's native output format) | `stat_L<L>.csv` — columns `qid, n_ios` |

   `scripts/train.sh` (Step 1, `offline_profiling.py`) reads these files
   directly to build one row per `(query, b_S, b_D)` combination — it does
   not run the DiskANN/SPANN sweep itself; see that script's own docstring
   for the full CLI.

## Usage

`scripts/*` resolve the repository root internally; the commands below
assume they're invoked **from the repository root**.

```bash
scripts/calibration.sh       # static thread-allocation profiling  → threading.mode: static
python scripts/monitor.py    # hybrid I/O bandwidth monitoring     → threading.mode: dynamic thresholds
scripts/train.sh             # offline training: saved profiling files → oracle labels → GBDT model
scripts/run.sh               # example final benchmark driver (datasets × recalls × configured runs)
```

- **`calibration.sh`** sweeps `(threadS, threadD)` splits per dataset/recall
  via `src/thread_calibration/static_profiler.py` to find the split that
  maximizes QPS.
- **`monitor.py`** runs the full controller for a given thread split and
  samples `/proc/diskstats` every 100ms to record queue depth and read
  bandwidth — used to pick the `optimal_bw_low` / `optimal_bw_high`
  thresholds.
- **`train.sh`** consumes the saved profiling files from Build step 4, then
  runs `src/offline_training/{offline_profiling,
  generate_oracle_labels_optimized, train_optimized_gbdt_regression}.py` in
  sequence to produce oracle labels and the GBDT budget-predictor model.
- **`run.sh`** runs the controller across configured datasets and recall
  targets, writing outputs to
  `benchmark_final/<dataset>/alpha_0.6/recall_<r>/run_<n>/`.

Each script's own header comment documents its exact inputs.

## Configs

`configs/<dataset>/<dataset>.yaml` (controller config) and
`configs/<dataset>/<dataset>_searchconfig.ini` (SPTAG native config) exist
for `deep100m`, `deep300m`, `sift100m`, `spacev100m`, and share an identical
structure. Two kinds of placeholders appear throughout:

- **`<PATH>`** — a path into your own build/data layout (binaries, indexes,
  query/GT/centroid files, output directories). Each occurrence has a
  same-line (YAML) or line-above (INI) comment saying what it should point to.
- **`<TUNE_PER_HARDWARE>` / `<TUNE_RECALLS>` / `<TARGET_RECALL>` / etc.** —
  a value that has no dataset-agnostic default: thread counts, I/O-bandwidth
  thresholds, and recall targets all depend on your hardware and workload.
  `optimal_bw_low`/`optimal_bw_high` in particular should be set from
  `scripts/monitor.py`'s output on your own disk, not guessed.

`config_file:` inside each `.yaml` points at the sibling `.ini` with a real
relative path (`configs/<dataset>/<dataset>_searchconfig.ini`) — that file
ships with this repo, so it's left as a working reference rather than a
placeholder.
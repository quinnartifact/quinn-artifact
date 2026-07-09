#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUINN Controller

Integrates the Budget Allocator and Shared Memory management, responsible for:
1. Loading query vectors and computing features (d1, d2, etc.)
2. Using the Allocator to predict per-query budgets
3. Writing budgets into POSIX shared memory
4. Launching the DiskANN and SPANN CLIs (via subprocess)
5. Waiting for execution to finish and cleaning up resources

Usage:
    python controller.py \
        --model_dir ./model/deep100m \
        --query_file /path/to/queries.fbin \
        --centroid_file /path/to/centroids.bin \
        --target_recall 90 \
        --diskann_bin ./search_disk_index \
        --spann_bin ./ssd_serving \
        --diskann_args "--index /path/to/diskann --k 100" \
        --spann_args "--index /path/to/spann --k 100"
"""

import argparse
import ctypes
import mmap
import os
import select
import struct
import subprocess
import shutil
import shlex
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import yaml

import json as _json_module
from shm import BudgetShmWriter, EarlyExitShmWriter, HubCacheShmWriter, LatencyShmWriter, ThreadCountShmWriter
from allocator import Allocator

# Threading / dynamic controller (imported lazily to avoid hard dependency)
try:
    import sys as _sys
    _threading_dir = str(Path(__file__).resolve().parents[1] / 'threading')
    if _threading_dir not in _sys.path:
        _sys.path.insert(0, _threading_dir)
    from dynamic_thread_scheduler import ThresholdController, DynamicThreadMonitor
    _DYNAMIC_AVAILABLE = True
except ImportError as _e:
    _DYNAMIC_AVAILABLE = False
    print(f"[Init] dynamic_thread_scheduler not available: {_e}")

# Add path to src and system_monitoring
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
vectordb_util_dir = current_dir.parents[2] / 'vectordb' / 'util'

for p in [str(src_dir), str(vectordb_util_dir)]:
    if p not in sys.path:
        sys.path.append(p)

from util.feature_utils import load_fvecs_or_fbin, compute_centroid_features

try:
    import system_monitoring
    print(f"[Init] system_monitoring loaded from {vectordb_util_dir}")
except ImportError as e:
    print(f"[Warning] Could not import system_monitoring from {vectordb_util_dir}: {e}")
    system_monitoring = None

# -----------------------------
# Configuration Management
# -----------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and substitute variable placeholders in paths

    Supported placeholders:
    - {dataset}: dataset name
    - {target_recall}: target recall value

    Args:
        config_path: path to the YAML config file

    Returns:
        the config dict
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract global variables
    dataset = config.get('dataset', 'unknown')
    target_recall = config.get('target_recall', 0)

    # Recursively substitute placeholders in the config
    def replace_placeholders(obj):
        if isinstance(obj, dict):
            return {k: replace_placeholders(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_placeholders(item) for item in obj]
        elif isinstance(obj, str):
            return obj.format(dataset=dataset, target_recall=target_recall)
        else:
            return obj

    config = replace_placeholders(config)

    print(f"[Config] Loaded configuration from: {config_path}")
    print(f"[Config] Dataset: {dataset}, Target Recall: {target_recall}%")

    return config


def build_diskann_args(diskann_config: Dict[str, Any]) -> List[str]:
    """
    Build DiskANN command-line arguments from a config dict

    Args:
        diskann_config: the DiskANN config dict

    Returns:
        list of command-line arguments
    """
    args = []
    arg_dict = diskann_config.get('args', {})

    for key, value in arg_dict.items():
        # Convert snake_case to --kebab-case
        arg_name = '--' + key

        # Handle different value types
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        elif isinstance(value, list):
            # e.g. search_list: [100, 200, 300]
            args.append(arg_name)
            args.extend([str(v) for v in value])
        elif value is not None:
            args.append(arg_name)
            args.append(str(value))

    return args


def build_spann_args(spann_config: Dict[str, Any]) -> List[str]:
    """
    Build SPANN command-line arguments from a config dict

    Args:
        spann_config: the SPANN config dict

    Returns:
        list of command-line arguments
    """
    args = []

    # SPANN mainly uses config.ini
    if 'config_file' in spann_config:
        args.append(spann_config['config_file'])

    # Extra arguments
    if 'extra_args' in spann_config:
        args.extend(spann_config['extra_args'])

    return args


# -----------------------------
# I/O Trace Merge Utility
# -----------------------------

def merge_io_traces(diskann_trace: str, spann_trace: str, output: str):
    """Merge DiskANN and SPANN I/O trace JSON files into one Perfetto-compatible trace.
    
    Assigns pid=1 to DiskANN events and pid=2 to SPANN events so they appear as
    separate process tracks in Perfetto UI.
    """
    import json

    merged_events = []
    
    for path, pid, name in [(diskann_trace, 1, "diskann"), (spann_trace, 2, "spann")]:
        if not path or not Path(path).exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            events = data.get("traceEvents", [])
            for evt in events:
                evt["pid"] = pid
                # Update process_name metadata events
                if evt.get("ph") == "M" and evt.get("name") == "process_name":
                    evt["args"]["name"] = name
            merged_events.extend(events)
            print(f"  Loaded {len(events)} trace events from {name}")
        except Exception as e:
            print(f"  Warning: Failed to load {name} trace from {path}: {e}")

    if merged_events:
        with open(output, 'w') as f:
            json.dump({"traceEvents": merged_events}, f)
        print(f"  Combined trace written to: {output}")
        print(f"  Open in https://ui.perfetto.dev to visualize I/O waterfall")
    else:
        print(f"  No trace events to merge")


# -----------------------------
# Controller Main Logic
# -----------------------------

def _init_threads_from_budget_ratio(bs_sum: float, bd_sum: float, total: int, step: int = 4) -> int:
    """Map bS/bD budget ratio to initial SPANN thread count."""
    ratio = bs_sum / bd_sum if bd_sum > 0 else float('inf')
    if ratio > 3.0:
        raw = total * 5 / 8
    elif ratio > 1.5:
        raw = total / 2
    elif ratio > 0.6:
        raw = total / 4
    else:
        raw = total / 8
    init_s = round(raw / step) * step
    return int(max(step, min(total - step, init_s)))


def run_controller(
    allocator: Allocator,
    queries: np.ndarray,
    d1s: np.ndarray,
    d1_d2_ratios: np.ndarray,
    target_recall: float,
    diskann_bin: str,
    spann_bin: str,
    diskann_args: List[str],
    spann_args: List[str],
    ground_truth_file: str,
    output_dir: Optional[str] = None,
    benchmark_dir: Optional[str] = None,
    diskann_result_file: Optional[str] = None,
    spann_result_file: Optional[str] = None,
    k_values: Optional[List[int]] = None,
    diskann_cgroup: Optional[str] = None,
    spann_cgroup: Optional[str] = None,
    cgroup_wrapper: str = 'cgexec',
    diskann_cpu_affinity: Optional[str] = None,
    spann_cpu_affinity: Optional[str] = None,
    diskann_priority: int = 0,
    spann_priority: int = 0,
    dataset_name: str = "unknown",
    memory_search_time: float = 0.0,
    io_trace_dir: Optional[str] = None,
    enable_early_exit: bool = False,
    eps_stop: float = 0.05,
    tau_k_spann: int = 100,
    tau_k_disk: int = 100,
    patience: int = 1,
    k_ref: int = 1,
    phi: float = 0.0,
    hop_trace: bool = False,
    topk_k: int = 0,
    enable_hub_cache: bool = False,
    hub_k: int = 32,
    max_hubs: int = 100,
    min_b_D: int = 0,
    seed_indices: str = "",
    seed_k: int = 0,
    wait_for_spann: bool = False,
    deprioritize_spann: bool = False,
    thread_count_shm_name: Optional[str] = None,
    dynamic_config: Optional[Dict[str, Any]] = None,
    latency_dump_path: Optional[str] = None,
):
    """
    Run the full Controller pipeline

    Args:
        allocator: Budget Allocator
        queries: Query vectors
        target_recall: Target recall (%)
        diskann_bin: DiskANN binary path
        spann_bin: SPANN binary path
        diskann_args: Additional args for DiskANN
        spann_args: Additional args for SPANN
        ground_truth_file: Ground truth file
        output_dir: Output directory
        benchmark_dir: Benchmark directory
        diskann_result_file: DiskANN result file
        spann_result_file: SPANN result file
        k_values: K values for recall@k
        memory_search_time: Time spent in SPTAG memory search (s)
        predict_time: Time spent in GBDT budget prediction (s)
        diskann_cgroup: Cgroup path for DiskANN
        spann_cgroup: Cgroup path for SPANN
        cgroup_wrapper: Binary to execute process in cgroup (default: cgexec)
        diskann_cpu_affinity: CPU cores for DiskANN (e.g. "0-15")
        spann_cpu_affinity: CPU cores for SPANN (e.g. "16-31")
        diskann_priority: Process priority (nice value), default 0
        spann_priority: Process priority (nice value), default 0
        dataset_name: Dataset name for logging
        phi: Pruning threshold (min * phi < max => min = 0)
    """
    num_queries = len(queries)
    print(f"\n{'='*80}")
    print(f"QUINN Controller Started")
    print(f"{'='*80}")
    print(f"Queries: {num_queries}")
    print(f"Target Recall: {target_recall}%")
    print(f"DiskANN bin: {diskann_bin}")
    print(f"SPANN bin: {spann_bin}")
    print(f"{'='*80}\n")

    # Step 1: Predict budgets
    print(f"[Step 1] Predicting budgets...")
    start_time = time.time()

    budgets = allocator.predict_batch(
        target_recalls=target_recall,
        d1s=d1s,
        d1_d2_ratios=d1_d2_ratios,
        query_vectors=queries,
        phi=phi
    )

    predict_time = time.time() - start_time
    prepare_overhead = memory_search_time + predict_time
    print(f"  Prediction completed in {predict_time:.3f}s")
    print(f"  b_S range: [{budgets[:, 0].min()}, {budgets[:, 0].max()}]")
    print(f"  b_D range: [{budgets[:, 1].min()}, {budgets[:, 1].max()}]")

    if min_b_D > 0:
        skipped = int(np.sum(budgets[:, 1] < min_b_D))
        budgets[budgets[:, 1] < min_b_D, 1] = 0
        print(f"  [min_b_D={min_b_D}] Skipping DiskANN for {skipped}/{num_queries} queries (b_D zeroed)")


    # Save budgets (optional)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        budgets_file = output_dir / 'budgets.csv'
        np.savetxt(budgets_file, budgets, fmt='%d', delimiter=',', header='bS,bD', comments='')
        print(f"  Budgets saved to: {budgets_file}")

    # Step 2: Create shared memory
    print(f"\n[Step 2] Creating shared memory...")
    shm_name = f"/quinn_budget_{os.getpid()}"

    with BudgetShmWriter(shm_name, budgets) as shm_writer:
        # Use ExitStack to manage optional context managers
        from contextlib import ExitStack
        with ExitStack() as stack:
            if enable_early_exit:
                early_exit_shm_name = f"/quinn_early_exit_{os.getpid()}"
                print(f"[Step 2b] Creating Early Exit SHM: {early_exit_shm_name}")
                early_exit_ctx = stack.enter_context(EarlyExitShmWriter(early_exit_shm_name, num_queries, topk_k=topk_k))

            latency_shm_name = f"/quinn_latency_{os.getpid()}"
            print(f"[Step 2d] Creating Latency SHM: {latency_shm_name}")
            latency_shm_ctx = stack.enter_context(LatencyShmWriter(latency_shm_name, num_queries))

            # QUINN: Dynamic thread count SHM
            thread_count_shm_ctx = None
            if thread_count_shm_name and dynamic_config is not None:
                _init_s = dynamic_config.get('init_thread_s', dynamic_config.get('total_threads', 16) // 2)
                _init_d = dynamic_config.get('total_threads', 16) - _init_s
                print(f"[Step 2e] Creating ThreadCount SHM: {thread_count_shm_name} "
                      f"(thread_s={_init_s}, thread_d={_init_d})")
                thread_count_shm_ctx = stack.enter_context(
                    ThreadCountShmWriter(thread_count_shm_name, _init_s, _init_d))

            hub_cache_shm_name = None
            if enable_hub_cache:
                hub_cache_shm_name = f"/quinn_hub_cache_{os.getpid()}"
                coord_dim = int(queries.shape[1])
                element_size = int(queries.dtype.itemsize)
                print(f"[Step 2c] Creating Hub Cache SHM: {hub_cache_shm_name} "
                      f"(num_queries={num_queries}, max_hubs={max_hubs}, hub_k={hub_k}, "
                      f"coord_dim={coord_dim}, element_size={element_size})")
                stack.enter_context(HubCacheShmWriter(
                    shm_name=hub_cache_shm_name,
                    num_queries=num_queries,
                    max_hubs=max_hubs,
                    hub_k=hub_k,
                    coord_dim=coord_dim,
                    element_size=element_size,
                ))

            # Make sure the result path is clean (avoids a size mismatch if DiskANN doesn't truncate on write)
            if diskann_result_file:
                drf = Path(diskann_result_file)
                if drf.exists():
                    print(f"  [Init] Cleaning up old DiskANN result file: {drf.name}")
                    drf.unlink()

            print(f"\n[Step 3] Launching search processes...")

            # Prepare command-line arguments
            diskann_cmd_base = [diskann_bin, '--budget_shm', shm_name,
                                '--latency_shm', latency_shm_name] + diskann_args
            spann_cmd_base = [spann_bin] + spann_args + ['--budget_shm', shm_name,
                                                         '--latency_shm', latency_shm_name]

            # QUINN: Dynamic thread count SHM args
            if thread_count_shm_name and thread_count_shm_ctx is not None:
                diskann_cmd_base += ['--thread_count_shm', thread_count_shm_name]
                spann_cmd_base   += ['--thread_count_shm', thread_count_shm_name]
                print(f"  [DynamicThread] SHM args added to both processes")

            if enable_early_exit:
                # Add Early Exit args
                diskann_cmd_base += [
                    '--early_exit_shm', early_exit_shm_name,
                    '--eps_stop', str(eps_stop),
                    '--tau_k_spann', str(tau_k_spann),
                    '--tau_k_disk', str(tau_k_disk),
                    '--patience', str(patience)
                ]
                spann_cmd_base += [
                    '--early_exit_shm', early_exit_shm_name,
                    '--k_ref', str(k_ref)
                ]
                if seed_indices:
                    diskann_cmd_base += ['--seed_indices', seed_indices]
                if seed_k > 0:
                    diskann_cmd_base += ['--seed_k', str(seed_k)]
                if wait_for_spann:
                    diskann_cmd_base += ['--wait_for_spann']
                if deprioritize_spann:
                    diskann_cmd_base += ['--deprioritize_spann']
                print(f"  [EarlyExit] Enabled: eps_stop={eps_stop}, tau_k_spann={tau_k_spann}, tau_k_disk={tau_k_disk}, patience={patience}, k_ref={k_ref}, seed_indices={seed_indices!r}, seed_k={seed_k}, wait_for_spann={wait_for_spann}, deprioritize_spann={deprioritize_spann}")

            if enable_hub_cache:
                diskann_cmd_base += ['--hub_cache_shm', hub_cache_shm_name]
                spann_cmd_base   += ['--hub_cache_shm', hub_cache_shm_name]
                print(f"  [HubCache] Enabled: hub_k={hub_k}, max_hubs={max_hubs}")

            # QUINN: Add --io_trace args if io_trace_dir is specified
            diskann_trace_path = None
            spann_trace_path = None
            if io_trace_dir:
                io_trace_dir_abs = Path(io_trace_dir).resolve()
                io_trace_dir_abs.mkdir(parents=True, exist_ok=True)
                diskann_trace_path = io_trace_dir_abs / 'diskann_io_trace.json'
                spann_trace_path = io_trace_dir_abs / 'spann_io_trace.json'
                
                diskann_cmd_base += ['--io_trace', str(diskann_trace_path)]
                spann_cmd_base += ['--io_trace', str(spann_trace_path)]
                
                print(f"[Trace] I/O Tracing enabled:")
                print(f"    Trace directory: {io_trace_dir_abs}")
                print(f"    DiskANN trace: {diskann_trace_path}")
                print(f"    SPANN trace:   {spann_trace_path}")

            # QUINN: Add --hop_trace if enabled
            hop_trace_path = None
            if hop_trace and output_dir:
                diskann_result_dir = Path(output_dir) / 'diskann'
                diskann_result_dir.mkdir(parents=True, exist_ok=True)
                hop_trace_path = diskann_result_dir / 'hop_trace.csv'
                diskann_cmd_base += ['--hop_trace', str(hop_trace_path)]
                print(f"[HopTrace] Enabled: {hop_trace_path}")

            # Function to apply taskset (CPU Affinity) and nice (Priority)
            def apply_process_control(cmd_list, cpu_list, priority, name_for_log):
                new_cmd = cmd_list
                log_parts = []

                # 1. Apply Priority (nice) - Inner wrapper
                # nice MUST be applied before the command, but taskset usually wraps nice
                # Order: taskset -c <cores> nice -n <prio> <cmd>
                if priority != 0:
                    if shutil.which('nice'):
                        new_cmd = ['nice', '-n', str(priority)] + new_cmd
                        log_parts.append(f"Priority={priority}")
                    else:
                        print(f"  Warning: nice not found! Priority for {name_for_log} will not be set.")

                # 2. Apply CPU Affinity (taskset) - Outer wrapper
                if cpu_list:
                    if shutil.which('taskset'):
                        new_cmd = ['taskset', '-c', cpu_list] + new_cmd
                        log_parts.append(f"Affinity={cpu_list}")
                    else:
                        print(f"  Warning: taskset not found! CPU affinity for {name_for_log} will not be set.")

                if log_parts:
                    print(f"  {name_for_log} Process Control: {', '.join(log_parts)}")
                
                return new_cmd
            
            diskann_cmd_base = apply_process_control(diskann_cmd_base, diskann_cpu_affinity, diskann_priority, "DiskANN")
            spann_cmd_base = apply_process_control(spann_cmd_base, spann_cpu_affinity, spann_priority, "SPANN")

            # Apply cgroup wrappers if specified
            # Strategy:
            # 1. If path starts with '/', assume absolute FS path (Cgroup v2 style). Use shell wrapper.
            # 2. Otherwise, assume name/controller format. Use cgexec (Cgroup v1 style).
            
            def apply_cgroup(cmd_list, cgroup_val, name_for_log):
                if not cgroup_val:
                    return cmd_list
                
                # Case 1: Absolute path (Native Cgroup V2)
                if cgroup_val.startswith('/'):
                    cgroup_procs = Path(cgroup_val) / "cgroup.procs"
                    if not cgroup_procs.parent.exists():
                        print(f"  Warning: Cgroup path {cgroup_val} does not exist!")
                    
                    # Construct shell wrapper: "echo $$ > Path/cgroup.procs && exec Cmd..."
                    # We use sh -c to ensure the PID of the shell (which becomes the process) is registered
                    # Use shlex.join to safely quote arguments for the shell
                    cmd_str = shlex.join(cmd_list)
                    
                    print(f"  {name_for_log} Cgroup (V2): {cgroup_val}")
                    return [
                        'sh', '-c',
                        f'echo $$ > {cgroup_procs} && exec {cmd_str}'
                    ]
                
                # Case 2: Legacy cgexec (Cgroup v1)
                else:
                    # Check wrapper existence
                    if not shutil.which(cgroup_wrapper):
                        print(f"  Warning: {cgroup_wrapper} not found in PATH! Cgroup limits may not apply.")
                    
                    print(f"  {name_for_log} Cgroup (V1/Tool): {cgroup_val}")
                    return [cgroup_wrapper, '-g', cgroup_val, '--sticky'] + cmd_list

            diskann_cmd = apply_cgroup(diskann_cmd_base, diskann_cgroup, "DiskANN")
            spann_cmd = apply_cgroup(spann_cmd_base, spann_cgroup, "SPANN")

            # Force line-buffered stdout on both subprocesses so "READY" is not
            # held in the C runtime's block buffer when stdout is a pipe.
            if shutil.which('stdbuf'):
                diskann_cmd = ['stdbuf', '-oL'] + diskann_cmd
                spann_cmd   = ['stdbuf', '-oL'] + spann_cmd

            print(f"  DiskANN command: {' '.join(diskann_cmd)}")
            print(f"  SPANN command: {' '.join(spann_cmd)}")


            # TODO: uncomment the following code to actually execute
            try:
                # Prepare the log directory
                log_dir = Path(output_dir) if output_dir else Path('./logs')
                log_dir.mkdir(parents=True, exist_ok=True)

                diskann_log = log_dir / 'diskann.log'
                spann_log = log_dir / 'spann.log'

                print(f"  DiskANN log: {diskann_log}")
                print(f"  SPANN log: {spann_log}")

                # Launch subprocesses, using PIPE to implement the READY/START protocol
                # stdout: used for the READY/START/DONE protocol
                # stderr: redirected to the log file

                # Extract the config.ini path from spann_args
                spann_cwd = None
                if spann_args:
                    config_file = spann_args[0]
                    if Path(config_file).exists():
                        spann_cwd = str(Path(config_file).parent)
                        print(f"  SPANN working directory: {spann_cwd}")

                # Open log files for writing (manual writing from stdout)
                diskann_log_handle = open(diskann_log, 'w')
                spann_log_handle = open(spann_log, 'w')

                try:
                    # Launch DiskANN (stderr -> stdout)
                    diskann_proc = subprocess.Popen(
                        diskann_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1  # Line buffered
                    )

                    # Launch SPANN (stderr -> stdout)
                    spann_proc = subprocess.Popen(
                        spann_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=spann_cwd,
                        text=True,
                        bufsize=1  # Line buffered
                    )

                    processes = {
                        diskann_proc.stdout.fileno(): ('diskann', diskann_proc),
                        spann_proc.stdout.fileno(): ('spann', spann_proc)
                    }

                    print(f"  DiskANN PID: {diskann_proc.pid}")
                    print(f"  SPANN PID: {spann_proc.pid}")

                    # Wait for the READY signal
                    # ... (Ready signal logic remains same, but reading from stdout which now includes stderr)
                    # Note: We need to handle the ready check loop similarly?
                    # The ready check loop uses `proc.stdout.readline()`.
                    
                    ready_status = {'diskann': False, 'spann': False}
                    search_times = {}
                    io_stats = {}
                    print(f"\n[Step 4] Waiting for READY signals...")

                    while not all(ready_status.values()):
                        readable, _, _ = select.select(list(processes.keys()), [], [], 60.0)
                        for fd in readable:
                            name, proc = processes[fd]
                            line = proc.stdout.readline()
                            if not line: raise RuntimeError(f"{name} exited unexpectedly")
                            
                            # Write everything to log as well
                            if name == 'diskann':
                                diskann_log_handle.write(line)
                                diskann_log_handle.flush()
                            else:
                                spann_log_handle.write(line)
                                spann_log_handle.flush()

                            line = line.strip()
                            if line == 'READY':
                                print(f"  ✓ {name.upper()} is READY")
                                ready_status[name] = True
                    
                    print(f"\n  ✅ Both processes are READY!")
                    
                    # Start Monitoring
                    monitor_thread = None
                    log_path = None
                    plot_path = None
                    
                    if system_monitoring:
                        try:
                            mon_dir = Path(__file__).resolve().parents[2] / 'result' / 'monitoring' / 'quinn'
                            mon_dir.mkdir(parents=True, exist_ok=True)
                            
                            rec_str = str(target_recall).replace('.', '_')
                            log_path = str(mon_dir / f"{dataset_name}_recall{rec_str}.log")
                            plot_path = str(mon_dir / f"{dataset_name}_recall{rec_str}")
                            
                            print(f"\n[Term] Starting system monitoring...")
                            print(f"  Log: {log_path}")
                            print(f"  Plot: {plot_path}.png")
                            
                            monitor_thread = system_monitoring.start_monitoring()
                        except Exception as e:
                            print(f"  Warning: Failed to start monitoring: {e}")

                    print(f"\n[Step 5] Sending START signal...")

                    # Send START to both subprocesses simultaneously
                    diskann_proc.stdin.write("START\n")
                    diskann_proc.stdin.flush()
                    spann_proc.stdin.write("START\n")
                    spann_proc.stdin.flush()

                    print(f"  START signal sent to both processes")

                    # QUINN: Start dynamic thread monitor if configured
                    _dynamic_monitor = None
                    if (thread_count_shm_ctx is not None and dynamic_config is not None
                            and _DYNAMIC_AVAILABLE):
                        try:
                            _total          = dynamic_config.get('total_threads', 32)
                            _init_s         = dynamic_config.get('init_thread_s', _total // 2)
                            _step           = dynamic_config.get('step', 4)
                            _min_each       = dynamic_config.get('min_each', 4)
                            _interval_ms    = dynamic_config.get('interval_ms', 100)
                            _timeseries_log = _fix_tr(dynamic_config.get('timeseries_log', None))
                            _bw_low         = dynamic_config.get('optimal_bw_low', 1000.0)
                            _bw_high        = dynamic_config.get('optimal_bw_high', 2500.0)
                            _revert_mult    = dynamic_config.get('revert_multiplier', 1.2)
                            _bw_growth_mult = dynamic_config.get('bw_growth_multiplier', 1.1)
                            _device         = dynamic_config.get('device', 'nvme0n1')
                            _cooldown       = dynamic_config.get('cooldown_windows', 3)

                            _ctrl = ThresholdController(
                                total_threads=_total,
                                init_s=_init_s,
                                step=_step,
                                min_each=_min_each,
                                optimal_bw_low=_bw_low,
                                optimal_bw_high=_bw_high,
                                revert_multiplier=_revert_mult,
                                bw_growth_multiplier=_bw_growth_mult,
                                device=_device,
                                cooldown_windows=_cooldown,
                            )
                            _dynamic_monitor = DynamicThreadMonitor(
                                latency_shm_writer=latency_shm_ctx,
                                thread_count_shm_writer=thread_count_shm_ctx,
                                controller=_ctrl,
                                interval_ms=_interval_ms,
                                timeseries_log=_timeseries_log
                            )
                            _dynamic_monitor.start()
                        except Exception as _e:
                            print(f"  [DynamicThread] Warning: failed to start monitor: {_e}")
                            _dynamic_monitor = None

                    print(f"\n[Step 6] Waiting for search completion...")

                    # Wait for both subprocesses to finish searching, and read until EOF
                    done_status = {'diskann': False, 'spann': False}
                    eof_status = {'diskann': False, 'spann': False}
                    monitoring_stopped = False

                    # Keep track of active pipes for select
                    active_pipes = list(processes.keys())

                    while active_pipes:
                        readable, _, _ = select.select(active_pipes, [], [], 60.0)

                        for fd in readable:
                            name, proc = processes[fd]
                            line = proc.stdout.readline()
                            
                            if not line: # EOF
                                eof_status[name] = True
                                if fd in active_pipes:
                                    active_pipes.remove(fd)
                                continue

                            # Mirror to log immediately
                            if name == 'diskann':
                                diskann_log_handle.write(line)
                                diskann_log_handle.flush()
                            else:
                                spann_log_handle.write(line)
                                spann_log_handle.flush()
                                
                            # Debug print to help identify missing lines
                            # print(f"DEBUG [{name}]: {line.strip()}") 

                            line = line.strip()
                            if line.startswith('SEARCH_TIME_MS'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    search_time_ms = float(parts[1])
                                search_times[name] = search_time_ms
                                print(f"  {name.upper()}: search_time = {search_time_ms:.3f} ms")
                                # SPANN finished its search — immediately hand threads to DiskANN
                                if name == 'spann' and _dynamic_monitor is not None:
                                    _post_d = dynamic_config.get('post_spann_diskann_threads', 30)
                                    _dynamic_monitor.notify_spann_done(_post_d)
                            elif line.startswith('IO_TOTAL'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    io_count = float(parts[1])
                                    io_stats[name] = io_count
                                    print(f"  {name.upper()}: io_total = {io_count}")
                            elif line == 'DONE':
                                print(f"  ✓ {name.upper()} is DONE")
                                done_status[name] = True
                                if name == 'diskann' and _dynamic_monitor is not None:
                                    _dynamic_monitor.stop_log()

                        # Check if both are DONE to stop monitoring (perform once)
                        if all(done_status.values()) and not monitoring_stopped:
                             # logic to calculate QPS and stop monitoring...
                             # We perform this here to ensure monitoring stops exactly at search end
                             monitoring_stopped = True
                             
                             if search_times:
                                 print(f"\n[Step 7] Calculating aligned QPS...")
                                 diskann_time = search_times.get('diskann', 0) / 1000.0
                                 spann_time = search_times.get('spann', 0) / 1000.0
                                 service_latency = max(diskann_time, spann_time)
                                 # New Service QPS formula: includes preparation overhead (Memory Search + Predict)
                                 total_service_time = service_latency + prepare_overhead
                                 service_qps = num_queries / total_service_time if total_service_time > 0 else 0
                                 
                                 total_work = diskann_time + spann_time
                                 total_qps = num_queries / total_work if total_work > 0 else 0

                                 # Per-query latency stats from SHM timestamps
                                 try:
                                     mean_lat, p50_lat, p90_lat, p99_lat, p999_lat = latency_shm_ctx.compute_stats()
                                 except Exception as e:
                                     print(f"  Warning: Failed to compute latency stats: {e}")
                                     mean_lat = p50_lat = p90_lat = p99_lat = p999_lat = 0.0

                                 # Optionally dump raw per-query SHM data for time-series analysis
                                 if latency_dump_path:
                                     try:
                                         import numpy as _np
                                         ss, se, ds, de = latency_shm_ctx.read_all()
                                         _np.savez(latency_dump_path,
                                                   spann_start_ns=ss, spann_end_ns=se,
                                                   diskann_start_ns=ds, diskann_io_ns=de)
                                         print(f"  Latency SHM dumped to: {latency_dump_path}")
                                     except Exception as e:
                                         print(f"  Warning: Failed to dump latency SHM: {e}")

                                 print(f"  Memory search time: {memory_search_time:.3f} s")
                                 print(f"  Prediction time: {predict_time:.3f} s")
                                 print(f"  DiskANN search time: {diskann_time:.3f} s")
                                 print(f"  SPANN search time: {spann_time:.3f} s")
                                 print(f"  Service latency (max): {service_latency:.3f} s")
                                 print(f"  Preparation overhead: {prepare_overhead:.3f} s")
                                 print(f"  Service QPS (Incl. prepare): {service_qps:.2f} queries/s")
                                 print(f"  Total work: {total_work:.3f} s")
                                 print(f"  Total QPS: {total_qps:.2f} queries/s")
                                 print(f"  Per-query latency  mean: {mean_lat:.3f} ms  p50: {p50_lat:.3f} ms  p90: {p90_lat:.3f} ms  p99: {p99_lat:.3f} ms  p99.9: {p999_lat:.3f} ms")
                                 
                                 if monitor_thread and system_monitoring:
                                     try:
                                         print(f"\n[Term] Starting system monitoring...")
                                         system_monitoring.stop_monitoring(monitor_thread, file_name=log_path, fig_name=plot_path, limit_duration=service_latency)
                                     except Exception as e:
                                         print(f"  Warning: Failed to start monitoring: {e}")
                             else:
                                 # Fallback stop if done but no times (error case?)
                                 if monitor_thread and system_monitoring:
                                     try:
                                         system_monitoring.stop_monitoring(monitor_thread, file_name=log_path, fig_name=plot_path)
                                     except Exception as e: pass

                    # QUINN: Stop dynamic thread monitor
                    if _dynamic_monitor is not None:
                        _dynamic_monitor.stop()
                        _dynamic_monitor.print_summary()

                    # Wait for the subprocesses to exit and get their return codes
                    diskann_ret = diskann_proc.wait()
                    spann_ret = spann_proc.wait()

                finally:
                    diskann_log_handle.close()
                    spann_log_handle.close()

                # QUINN: Merge I/O traces from DiskANN and SPANN
                if io_trace_dir and diskann_trace_path and spann_trace_path:
                    print(f"\n[Trace] Merging I/O traces...")
                    combined_trace = str(Path(io_trace_dir) / 'combined_io_trace.json')
                    merge_io_traces(diskann_trace_path, spann_trace_path, combined_trace)

                # Move stat.csv to the result directory and rename it to spann_stat.csv
                if spann_cwd and output_dir:
                    stat_csv_source = Path(spann_cwd) / 'stat.csv'
                    if stat_csv_source.exists():
                        spann_result_dir = Path(output_dir) / 'spann'
                        diskann_result_dir = Path(output_dir) / 'diskann'
                        spann_result_dir.mkdir(parents=True, exist_ok=True)
                        diskann_result_dir.mkdir(parents=True, exist_ok=True)
                        stat_csv_dest = spann_result_dir / 'spann_stat.csv'
                        stat_csv_dest_diskann = diskann_result_dir / 'stat_L100.csv'


                        # Move even if the destination file already exists
                        if stat_csv_dest.exists():
                            stat_csv_dest.unlink()
                        shutil.move(str(stat_csv_source), str(stat_csv_dest))

                        if stat_csv_dest_diskann.exists():
                            stat_csv_dest_diskann.unlink()
                        shutil.move("./stat_L100.csv", str(stat_csv_dest_diskann))
                        for leftover in Path(".").glob("stat_L*.csv"):
                            leftover.unlink(missing_ok=True)

                        print(f"\n  Moved stat.csv to: {stat_csv_dest}")
                        print(f"  Moved stat_L100 to: {stat_csv_dest_diskann}")

                        # Parse IO stats
                        try:
                            import csv
                            # DiskANN stats
                            if stat_csv_dest_diskann.exists():
                                with open(stat_csv_dest_diskann, 'r') as f:
                                    # Strip any surrounding whitespace
                                    lines = (line.strip() for line in f)
                                    reader = csv.DictReader(lines)
                                    # DiskANN CSV header might have spaces, handle flexibly
                                    total_io = 0.0
                                    for row in reader:
                                        # Find key that looks like 'n_ios'
                                        ios_key = next((k for k in row.keys() if k and 'n_ios' in k.strip()), None)
                                        if ios_key:
                                            total_io += float(row[ios_key])
                                    io_stats['diskann'] = total_io
                                    print(f"  DiskANN IO total (from csv): {total_io}")

                            # SPANN stats
                            if stat_csv_dest.exists():
                                with open(stat_csv_dest, 'r') as f:
                                    lines = (line.strip() for line in f)
                                    reader = csv.DictReader(lines)
                                    total_io = 0.0
                                    for row in reader:
                                        ios_key = next((k for k in row.keys() if k and 'n_ios' in k.strip()), None)
                                        if ios_key:
                                            total_io += float(row[ios_key])
                                    io_stats['spann'] = total_io
                                    print(f"  SPANN IO total (from csv): {total_io}")
                        except Exception as e:
                            print(f"  Warning: Failed to parse IO stats from CSV: {e}")

                print(f"  DiskANN exit code: {diskann_ret}")
                print(f"  SPANN exit code: {spann_ret}")

                # If execution failed, show the tail of the log
                if diskann_ret != 0:
                    print(f"\n  DiskANN failed! Last 50 lines of log:")
                    with open(diskann_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-50:]:
                            print(f"    {line.rstrip()}")

                if spann_ret != 0:
                    print(f"\n  SPANN failed! Last 50 lines of log:")
                    with open(spann_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-50:]:
                            print(f"    {line.rstrip()}")

                # If both succeeded, show key statistics
                if diskann_ret == 0 and spann_ret == 0:
                    print(f"\n  ✅ Both processes completed successfully!")
                    print(f"\n  Extracting key metrics from logs...")

                    # Extract recall from the DiskANN log
                    try:
                        with open(diskann_log, 'r') as f:
                            for line in f:
                                sline = line.strip()
                                if 'Recall@' in line or 'QPS' in line or 'Latency' in line:
                                    print(f"    [DiskANN] {line.rstrip()}")
                                # Capture lines starting with digit (Data Row)
                                elif sline and sline[0].isdigit():
                                    print(f"    [DiskANN] {line.rstrip()}")
                    except:
                        pass

                    # Extract recall from the SPANN log
                    try:
                        with open(spann_log, 'r') as f:
                            for line in f:
                                if 'Recall@' in line or 'QPS' in line or 'Latency' in line or 'MRR' in line:
                                    print(f"    [SPANN] {line.rstrip()}")
                    except:
                        pass
        
            except Exception as e:
                print(f"  Error launching processes: {e}")
                raise

    # Step 4: Cleanup complete (handled automatically by the context manager)
    print(f"\n[Step 5] Cleanup completed")

    # Step 5: Merge results and compute merged recall (if the required parameters were provided)
    if (diskann_result_file and spann_result_file and ground_truth_file and
        benchmark_dir and k_values):
        print(f"\n[Step 6] Merging results and calculating recall...")

        try:
            from merger import merge_and_evaluate

            recalls = merge_and_evaluate(
                diskann_result_file=diskann_result_file,
                spann_result_file=spann_result_file,
                ground_truth_file=ground_truth_file,
                output_dir=benchmark_dir,
                k_values=k_values,
                search_times=search_times,
                num_queries=num_queries,
                io_stats=io_stats
            )

            # Append extra performance metrics to the report
            report_file = Path(benchmark_dir) / 'merged_recall_report.txt'
            if report_file.exists():
                with open(report_file, 'a') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write("Preparation & Overhead Analysis (All terms in Parallel-Service Latency)\n")
                    f.write("="*80 + "\n")
                    f.write(f"Memory Search Time:  {memory_search_time:.3f} s\n")
                    f.write(f"Prediction Time:     {predict_time:.3f} s\n")
                    f.write(f"Preparation Total:   {prepare_overhead:.3f} s\n")
                    f.write(f"Search Serv. Lat.:   {service_latency:.3f} s\n")
                    f.write(f"Total Service Lat.:  {total_service_time:.3f} s\n")
                    f.write(f"Final Service QPS:   {service_qps:.2f} queries/s\n")
                    f.write(f"Per-Query Lat. Mean: {mean_lat:.3f} ms\n")
                    f.write(f"Per-Query Lat. p50:  {p50_lat:.3f} ms\n")
                    f.write(f"Per-Query Lat. p90:  {p90_lat:.3f} ms\n")
                    f.write(f"Per-Query Lat. p99:  {p99_lat:.3f} ms\n")
                    f.write(f"Per-Query Lat. p99.9:{p999_lat:.3f} ms\n")
                    f.write("="*80 + "\n")

            # Show key results
            print(f"\n  Key Results:")
            for key, value in recalls.items():
                print(f"    {key}: {value*100:.2f}%")

        except Exception as e:
            print(f"  Warning: Failed to merge results: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Controller finished successfully")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='QUINN Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a config file
  python controller.py --config configs/deep100m_recall90.yaml

  # Using a config file and overriding some parameters
  python controller.py --config configs/deep100m_recall90.yaml --target_recall 95

  # Using command-line arguments (legacy style)
  python controller.py --model_dir ./model --query_file queries.fbin ...
        """
    )

    # Config file (used preferentially)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to the YAML config file (recommended)')

    # Model & data (can be overridden by config)
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Model directory')
    parser.add_argument('--model_type', default='auto', choices=['auto', 'gbdt'],
                       help='Model type')
    parser.add_argument('--device', default='cpu',
                       help='Compute device (cpu/cuda)')

    parser.add_argument('--query_file', type=str, default=None,
                       help='Query vectors file (.fbin or .fvecs)')
    parser.add_argument('--centroid_file', type=str, default=None,
                       help='Centroid vectors file (used to compute d1, d2)')

    # Parameters
    parser.add_argument('--target_recall', type=float, default=None,
                       help='Target recall (%)')
    parser.add_argument('--max_queries', type=int, default=None,
                       help='Maximum number of queries (for testing)')

    # Binaries
    parser.add_argument('--diskann_bin', type=str, default=None,
                       help='Path to the DiskANN executable')
    parser.add_argument('--spann_bin', type=str, default=None,
                       help='Path to the SPANN executable')

    # Additional arguments
    parser.add_argument('--diskann_args', default='',
                       help='Additional DiskANN arguments (space-separated, overrides config)')
    parser.add_argument('--spann_args', default='',
                       help='Additional SPANN arguments (space-separated, overrides config)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (stores budgets.csv, etc.)')

    # Cgroup I/O Control
    parser.add_argument('--diskann_cgroup', type=str, default=None,
                       help='Cgroup path for DiskANN (e.g., blkio:quinn/diskann)')
    parser.add_argument('--spann_cgroup', type=str, default=None,
                       help='Cgroup path for SPANN (e.g., blkio:quinn/spann)')
    parser.add_argument('--cgroup_wrapper', type=str, default='cgexec',
                       help='Cgroup wrapper binary (default: cgexec)')
    
    # Resource Isolation
    parser.add_argument('--diskann_cores', type=str, default=None,
                       help='CPU cores for DiskANN (e.g., 0-15)')
    parser.add_argument('--spann_cores', type=str, default=None,
                       help='CPU cores for SPANN (e.g., 16-31)')
    parser.add_argument('--diskann_priority', type=int, default=None,
    help='Process priority for DiskANN (nice value)')
    parser.add_argument('--spann_priority', type=int, default=None,
    help='Process priority for SPANN (nice value)')
    parser.add_argument('--io_trace', type=str, default=None,
                       help='Output directory for I/O trace JSON files')

    # Early Exit Arguments (Incumbent-gated frontier bound)
    parser.add_argument('--enable_early_exit', action='store_true', help='Enable early exit (incumbent-gated frontier bound)')
    parser.set_defaults(enable_early_exit=None)
    parser.add_argument('--eps_stop', type=float, default=None, help='Slack factor for frontier bound (e.g. 0.02~0.10)')
    parser.add_argument('--tau_k_spann', type=int, default=None, help='Use SPANN k-th nearest distance as tau_spann')
    parser.add_argument('--tau_k_disk', type=int, default=None, help='Use DiskANN k-th nearest distance as tau_disk')
    parser.add_argument('--patience', type=int, default=None, help='Consecutive rounds frontier bound must hold before terminating')
    parser.add_argument('--k_ref', type=int, default=None, help='SPANN side: which k-th distance to publish to SHM')
    parser.add_argument('--phi', type=float, default=None, help='Pruning threshold (bD * phi < bS => bD = 0)')
    parser.add_argument('--hop_trace', action='store_true', default=None, help='Enable hop trace recording (CSV output to DiskANN result dir)')
    parser.add_argument('--topk_k', type=int, default=None, help='Number of SPANN topk IDs to store in early exit SHM per query (0 = disabled)')
    parser.add_argument('--seed_indices', type=str, default=None, help='Comma-separated 0-based positions in SPANN topk to use as DiskANN seeds (e.g. "0,25,50,75"). Empty = sequential top-N.')
    parser.add_argument('--seed_k', type=int, default=None, help='Number of SPANN top-K IDs to inject as DiskANN seeds (sequential mode). 0 = no injection. Ignored when --seed_indices is set.')
    parser.add_argument('--wait_for_spann', action='store_true', default=None, help='DiskANN spin-waits until SPANN has written topk IDs to SHM before starting beam search.')
    parser.add_argument('--deprioritize_spann', action='store_true', default=None, help='Move SPANN top-K nodes to back of beam frontier each iteration. Non-SPANN candidates expand first.')

    # Hub Cache Arguments (Hub Piggybacking)
    parser.add_argument('--enable_hub_cache', action='store_true', default=None, help='Enable hub piggybacking cache (SPANN head nodes pre-loaded into DiskANN SHM)')
    parser.add_argument('--hub_k', type=int, default=None, help='Hub neighbor count per entry (= HubNeighborCount in SPANN searchconfig)')
    parser.add_argument('--max_hubs', type=int, default=None, help='Max hub entries per query (= SPANN searchInternalResultNum)')
    parser.add_argument('--min_b_D', type=int, default=None, help='Skip DiskANN for queries with b_D below this threshold (0 = disabled)')

    args = parser.parse_args()

    # Handle the config file
    config = {}
    if args.config:
        config = load_config(args.config)
    else:
        # No config file — check required command-line arguments
        required_args = ['model_dir', 'query_file', 'centroid_file', 'target_recall', 'diskann_bin', 'spann_bin']
        missing = [arg for arg in required_args if getattr(args, arg) is None]
        if missing:
            parser.error(f"Without --config, the following arguments are required: {', '.join('--' + m for m in missing)}")

    # Get parameters from config or args (args take priority)
    def get_param(key: str, section: str = None, required: bool = False):
        """Get a parameter from args or config, args take priority"""
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            return arg_value

        if section and section in config:
            config_value = config[section].get(key)
            if config_value is not None:
                return config_value

        if required:
            raise ValueError(f"Required parameter '{key}' not found in config or args")

        return None

    # Extract all parameters
    model_dir = get_param('model_dir', 'model', required=True)
    model_type = get_param('model_type', 'model') or 'auto'
    device = get_param('device', 'model') or 'cpu'

    query_file = get_param('query_file', 'data', required=True)
    centroid_file = get_param('centroid_file', 'data', required=True)
    max_queries = get_param('max_queries', 'data')
    dataset = config.get('dataset', 'unknown')

    # Target recall: prefer the top-level setting, fall back to the search section
    target_recall = config.get('target_recall')
    if target_recall is None:
        target_recall = get_param('target_recall', 'search', required=True)
    # Command-line arguments override the config
    config_target_recall = target_recall   # original value used by load_config substitution
    if args.target_recall is not None:
        target_recall = args.target_recall

    # Re-substitute {target_recall} in path strings if CLI overrode the value
    def _fix_tr(s):
        if isinstance(s, str) and config_target_recall is not None:
            return s.replace(str(config_target_recall), str(target_recall))
        return s

    diskann_bin = get_param('diskann_bin', 'diskann', required=True) or config.get('diskann', {}).get('binary')
    spann_bin = get_param('spann_bin', 'spann', required=True) or config.get('spann', {}).get('binary')

    output_dir = _fix_tr(get_param('output_dir', 'output') or config.get('output', {}).get('output_dir'))

    # Merge-related parameters
    benchmark_dir = _fix_tr(config.get('output', {}).get('benchmark_dir') if config else None)
    diskann_result_file = config.get('results', {}).get('diskann_result') if config else None
    spann_result_file = config.get('results', {}).get('spann_result') if config else None
    k_values = config.get('results', {}).get('k_values', [10, 20, 50, 100]) if config else [10, 20, 50, 100]
    gt_file = config.get('data', {}).get('gt_file') or config.get('diskann', {}).get('args', {}).get('gt_file')

    # --- Static threading: override num_threads / NumberOfThreads from profiling ---
    threading_mode = config.get('threading', {}).get('mode', 'off')
    if threading_mode == 'static':
        import json as _json
        import re as _re
        import tempfile as _tempfile

        best_configs_file = config.get('threading', {}).get('best_configs_file', '')
        if not best_configs_file or not Path(best_configs_file).exists():
            raise FileNotFoundError(
                f"[Threading] best_configs_file not found: {best_configs_file!r}\n"
                "  Run static_profiler.py first, then set threading.best_configs_file in your config."
            )

        best_all = _json.loads(Path(best_configs_file).read_text())
        recall_key = str(float(target_recall))
        thread_cfg = best_all.get(dataset, {}).get(recall_key)

        if thread_cfg is None:
            # Try nearest available recall
            available = sorted(best_all.get(dataset, {}).keys(), key=float)
            if available:
                nearest = min(available, key=lambda k: abs(float(k) - float(target_recall)))
                thread_cfg = best_all[dataset][nearest]
                print(f"[Threading] WARN: no exact match for recall={target_recall}, "
                      f"using nearest={nearest}")
            else:
                raise KeyError(
                    f"[Threading] No profiling data for dataset={dataset!r} in {best_configs_file}"
                )

        threadS = thread_cfg['threadS']
        threadD = thread_cfg['threadD']
        print(f"[Threading] mode=static  dataset={dataset}  recall={target_recall}"
              f"  → threadS={threadS}  threadD={threadD}"
              f"  (profiled QPS={thread_cfg.get('qps', 'N/A'):.0f})")

        # Override DiskANN num_threads in config
        config.setdefault('diskann', {}).setdefault('args', {})['num_threads'] = threadD

        # Override SPANN NumberOfThreads: patch the ini file into a temp copy
        spann_ini_path = config.get('spann', {}).get('config_file', '')
        if not spann_ini_path or not Path(spann_ini_path).exists():
            raise FileNotFoundError(
                f"[Threading] SPANN config_file not found: {spann_ini_path!r}"
            )
        ini_text = Path(spann_ini_path).read_text()
        ini_text = _re.sub(
            r'(NumberOfThreads\s*=\s*)\d+',
            lambda m: m.group(1) + str(threadS),
            ini_text
        )
        # Place the temp ini next to the original so SPANN's cwd stays the same
        tmp_ini_path = Path(spann_ini_path).parent / f'_quinn_tmp_{dataset}_s{threadS}.ini'
        tmp_ini_path.write_text(ini_text)
        config['spann']['config_file'] = str(tmp_ini_path)
        print(f"[Threading] Temp SPANN ini written: {tmp_ini_path}")

    # dynamic threading initialized after features (needs allocator + d1s for budget ratio)
    _thread_count_shm_name = None
    _dynamic_config = None

    # Step 1: Load the Allocator
    print(f"[Init] Loading Allocator...")
    allocator = Allocator(
        model_dir=model_dir,
        model_type=model_type,
        device=device
    )

    # Step 2: Load queries
    t0 = time.time()
    print(f"\n[Init] Loading queries from {query_file}...")
    queries = load_fvecs_or_fbin(query_file)
    t1 = time.time()
    print(f"  Loaded {len(queries)} queries, dim={queries.shape[1]} (took {t1-t0:.3f}s)")

    if max_queries:
        queries = queries[:max_queries]
        print(f"  Limited to {len(queries)} queries for testing")

    # Step 3: Load centroids and compute features
    t0 = time.time()
    print(f"\n[Init] Loading centroids from {centroid_file}...")
    centroids = load_fvecs_or_fbin(centroid_file)
    t1 = time.time()
    print(f"  Loaded {len(centroids)} centroids, dim={centroids.shape[1]} (took {t1-t0:.3f}s)")

    t0 = time.time()
    print(f"\n[Init] Computing centroid features...")
    d1s, d1_d2_ratios, mem_search_time = compute_centroid_features(queries, centroids, centroid_file=centroid_file)
    t1 = time.time()
    print(f"  Computing features took {t1-t0:.3f}s")

    # Step 4: Dynamic threading — initialized here so budget ratio can use allocator + features
    if threading_mode == 'dynamic':
        import re as _re

        if not _DYNAMIC_AVAILABLE:
            raise ImportError(
                "[Threading] dynamic_thread_scheduler.py not found — "
                "cannot use threading.mode=dynamic"
            )

        _dyn_sub = config.get('threading', {}).get('dynamic', {})
        _total = int(_dyn_sub.get('total_threads', 32))
        _step  = int(_dyn_sub.get('step', 4))

        if 'init_thread_s' in _dyn_sub:
            threadS = int(_dyn_sub['init_thread_s'])
            print(f"[Threading] mode=dynamic  dataset={dataset}  recall={target_recall}"
                  f"  → initial threadS={threadS}  threadD={_total - threadS} (from config)")
        else:
            _phi = float(get_param('phi', 'model') or 0.0)
            _budgets_preview = allocator.predict_batch(
                target_recalls=target_recall,
                d1s=d1s,
                d1_d2_ratios=d1_d2_ratios,
                query_vectors=queries,
                phi=_phi,
            )
            _bs_sum = float(_budgets_preview[:, 0].sum())
            _bd_sum = float(_budgets_preview[:, 1].sum())
            _ratio  = _bs_sum / _bd_sum if _bd_sum > 0 else float('inf')
            threadS = _init_threads_from_budget_ratio(_bs_sum, _bd_sum, _total, _step)
            print(f"[Threading] mode=dynamic  dataset={dataset}  recall={target_recall}"
                  f"  bS/bD={_ratio:.3f}  → initial threadS={threadS}  threadD={_total - threadS} (from budget ratio)")

        threadD = _total - threadS
        config.setdefault('diskann', {}).setdefault('args', {})['num_threads'] = threadD

        spann_ini_path = config.get('spann', {}).get('config_file', '')
        if spann_ini_path and Path(spann_ini_path).exists():
            ini_text = Path(spann_ini_path).read_text()
            ini_text = _re.sub(
                r'(NumberOfThreads\s*=\s*)\d+',
                lambda m: m.group(1) + str(threadS),
                ini_text
            )
            tmp_ini_path = Path(spann_ini_path).parent / f'_quinn_tmp_{dataset}_dyn_s{threadS}.ini'
            tmp_ini_path.write_text(ini_text)
            config['spann']['config_file'] = str(tmp_ini_path)
            print(f"[Threading/dynamic] Temp SPANN ini written: {tmp_ini_path}")

        _thread_count_shm_name = f"/quinn_thread_count_{os.getpid()}"
        _dynamic_config = dict(_dyn_sub)
        _dynamic_config['total_threads'] = threadS + threadD
        _dynamic_config['init_thread_s'] = threadS
        print(f"[Threading/dynamic] Config: {_dynamic_config}")

    # Build DiskANN and SPANN arguments (after threading so config has correct num_threads)
    if args.diskann_args:
        diskann_args = args.diskann_args.split()
    elif 'diskann' in config:
        diskann_args = build_diskann_args(config['diskann'])
    else:
        diskann_args = []

    if args.spann_args:
        spann_args = args.spann_args.split()
    elif 'spann' in config:
        spann_args = build_spann_args(config['spann'])
    else:
        spann_args = []

    # Step 5: Run the Controller
    run_controller(
        allocator=allocator,
        queries=queries,
        d1s=d1s,
        d1_d2_ratios=d1_d2_ratios,
        target_recall=target_recall,
        diskann_bin=diskann_bin,
        spann_bin=spann_bin,
        diskann_args=diskann_args,
        spann_args=spann_args,
        ground_truth_file=gt_file,
        output_dir=output_dir,
        benchmark_dir=benchmark_dir,
        diskann_result_file=diskann_result_file,
        spann_result_file=spann_result_file,
        k_values=k_values,
        diskann_cgroup=args.diskann_cgroup,
        spann_cgroup=args.spann_cgroup,
        cgroup_wrapper=args.cgroup_wrapper,
        diskann_cpu_affinity=get_param('diskann_cores', 'resource_isolation'),
        spann_cpu_affinity=get_param('spann_cores', 'resource_isolation'),
        diskann_priority=int(get_param('diskann_priority', 'resource_isolation') or 0),
        spann_priority=int(get_param('spann_priority', 'resource_isolation') or 0),
        dataset_name=dataset,
        memory_search_time=mem_search_time,
        io_trace_dir=args.io_trace,
        enable_early_exit=get_param('enable_early_exit', 'early_exit') or False,
        eps_stop=float(get_param('eps_stop', 'early_exit') or 0.05),
        tau_k_spann=int(get_param('tau_k_spann', 'early_exit') or 100),
        tau_k_disk=int(get_param('tau_k_disk', 'early_exit') or 100),
        patience=int(get_param('patience', 'early_exit') or 1),
        k_ref=int(get_param('k_ref', 'early_exit') or 1),
        phi=float(get_param('phi', 'model') or 0.0),
        hop_trace=get_param('hop_trace', 'early_exit') or False,
        topk_k=int(get_param('topk_k', 'early_exit') or 0),
        enable_hub_cache=get_param('enable_hub_cache', 'hub_cache') or False,
        hub_k=int(get_param('hub_k', 'hub_cache') or 32),
        max_hubs=int(get_param('max_hubs', 'hub_cache') or 100),
        min_b_D=int(get_param('min_b_D', 'diskann') or 0),
        seed_indices=str(get_param('seed_indices', 'early_exit') or ""),
        seed_k=int(get_param('seed_k', 'early_exit') or 0),
        wait_for_spann=bool(get_param('wait_for_spann', 'early_exit') or False),
        deprioritize_spann=bool(get_param('deprioritize_spann', 'early_exit') or False),
        thread_count_shm_name=_thread_count_shm_name,
        dynamic_config=_dynamic_config,
        latency_dump_path=config.get('output', {}).get('latency_dump_path'),
    )


if __name__ == '__main__':
    main()

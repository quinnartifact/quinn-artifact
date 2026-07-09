#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/controller/test/test_shm.py — Shared memory read/write verification

End-to-end test of the binary layout between Python BudgetShmWriter and the
C++ BudgetShmReader:

1. On the Python side, BudgetShmWriter writes a (b_S, b_D) budget array to POSIX SHM
2. Launch the C++ binary reader (test_shm_reader) to verify the magic number and field values
3. Compare the values written by Python against the values read by C++, to confirm endianness and struct alignment are correct

Usage:
  python test_shm.py --shm_name /quinn_budget_test --n_queries 100

Requires test_shm_reader under src/controller/shm/ to be compiled first (see the Makefile).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path to find controller/allocator/shm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shm import BudgetShmWriter


def test_shared_memory(num_queries: int = 100, use_cpp_reader: bool = True):
    """Test shared memory write and read"""

    print("="*80)
    print("Shared Memory Test")
    print("="*80)

    # Step 1: Generate test budgets
    print(f"\n[Step 1] Generating test budgets ({num_queries} queries)...")
    np.random.seed(42)
    budgets = np.random.randint(10, 201, size=(num_queries, 2), dtype=np.uint16)
    budgets[:, 0] = (budgets[:, 0] // 10) * 10  # b_S: 10, 20, ..., 200
    budgets[:, 1] = (budgets[:, 1] // 10) * 10  # b_D: 10, 20, ..., 200

    print(f"  Generated {len(budgets)} budget entries")
    print(f"  b_S range: [{budgets[:, 0].min()}, {budgets[:, 0].max()}]")
    print(f"  b_D range: [{budgets[:, 1].min()}, {budgets[:, 1].max()}]")
    print(f"\n  First 5 entries:")
    for i in range(min(5, len(budgets))):
        print(f"    [{i}] b_S={budgets[i, 0]}, b_D={budgets[i, 1]}")

    # Step 2: Create shared memory
    print(f"\n[Step 2] Creating shared memory...")
    shm_name = f"/quinn_budget_test_{os.getpid()}"

    try:
        with BudgetShmWriter(shm_name, budgets) as writer:
            print(f"✓ Shared memory created: {shm_name}")

            # Step 3: Verify using the C++ reader (if compiled)
            if use_cpp_reader:
                print(f"\n[Step 3] Testing C++ reader...")
                cpp_reader_path = Path(__file__).parent / 'test_shm_reader'

                if not cpp_reader_path.exists():
                    print(f"  Warning: C++ reader not found at {cpp_reader_path}")
                    print(f"  Compile it with:")
                    print(f"    $ g++ -std=c++17 -o test_shm_reader test_shm_reader.cpp -lrt")
                    print(f"  Skipping C++ reader test...")
                else:
                    # Run the C++ reader
                    try:
                        result = subprocess.run(
                            [str(cpp_reader_path), shm_name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        print(f"  C++ reader exit code: {result.returncode}")

                        if result.returncode == 0:
                            print(f"✓ C++ reader test PASSED")
                            print(f"\n  C++ reader output:")
                            for line in result.stdout.split('\n'):
                                if line.strip():
                                    print(f"    {line}")
                        else:
                            print(f"✗ C++ reader test FAILED")
                            print(f"  stderr: {result.stderr}")
                            return False

                    except subprocess.TimeoutExpired:
                        print(f"✗ C++ reader timeout")
                        return False
                    except Exception as e:
                        print(f"✗ C++ reader error: {e}")
                        return False

            # Step 4: Verify the shared memory contents (Python)
            print(f"\n[Step 4] Verifying shared memory content (Python)...")
            try:
                import posix_ipc
                import mmap
                import struct

                # Open shared memory (read-only)
                shm = posix_ipc.SharedMemory(shm_name)
                mapfile = mmap.mmap(shm.fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)

                # Read the header
                header = struct.unpack('<IIII', mapfile[0:16])
                magic, version, num_queries_read, entry_size = header

                print(f"  Header:")
                print(f"    Magic: 0x{magic:08X} (expected: 0x43415341)")
                print(f"    Version: {version} (expected: 1)")
                print(f"    Num queries: {num_queries_read} (expected: {num_queries})")
                print(f"    Entry size: {entry_size} (expected: 4)")

                # Verify the header
                if magic != 0x43415341:
                    print(f"✗ Invalid magic number")
                    return False

                if version != 1:
                    print(f"✗ Invalid version")
                    return False

                if num_queries_read != num_queries:
                    print(f"✗ Num queries mismatch")
                    return False

                if entry_size != 4:
                    print(f"✗ Entry size mismatch")
                    return False

                # Read and verify the first few entries
                print(f"\n  Verifying first 5 entries:")
                offset = 16
                for i in range(min(5, num_queries)):
                    bS, bD = struct.unpack('<HH', mapfile[offset:offset+4])
                    expected_bS, expected_bD = budgets[i]

                    status = "✓" if (bS == expected_bS and bD == expected_bD) else "✗"
                    print(f"    [{i}] {status} Read: b_S={bS}, b_D={bD} | Expected: b_S={expected_bS}, b_D={expected_bD}")

                    if bS != expected_bS or bD != expected_bD:
                        print(f"✗ Entry mismatch at index {i}")
                        mapfile.close()
                        shm.close_fd()
                        return False

                    offset += 4

                mapfile.close()
                shm.close_fd()

                print(f"✓ Shared memory content verification PASSED")

            except Exception as e:
                print(f"✗ Verification failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    except Exception as e:
        print(f"✗ Shared memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("All tests PASSED ✓")
    print("="*80)

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Shared Memory')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries to test')
    parser.add_argument('--no_cpp_reader', action='store_true',
                       help='Skip the C++ reader test')

    args = parser.parse_args()

    success = test_shared_memory(args.num_queries, not args.no_cpp_reader)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

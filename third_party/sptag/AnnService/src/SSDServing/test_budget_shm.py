#!/usr/bin/env python3
"""
測試 SPANN 的 Budget Shared Memory 整合

這個腳本會：
1. 創建一個測試用的 shared memory
2. 等待用戶手動執行 SPANN
3. 清理 shared memory
"""

import os
import sys
import numpy as np
import argparse

# 添加 QUINN controller 路徑
sys.path.insert(0, '<REPO_ROOT>/src/controller')

try:
    from controller import BudgetShmWriter
except ImportError as e:
    print(f"Error: Cannot import BudgetShmWriter from QUINN controller")
    print(f"  {e}")
    print(f"\nPlease make sure QUINN controller is set up correctly.")
    print(f"Required: pip install posix_ipc numpy")
    sys.exit(1)


def create_test_budgets(num_queries, nprobe_min=10, nprobe_max=200):
    """
    創建測試用的 budgets

    策略：
    - 前 1/3 queries: 低 nprobe (10-50)
    - 中 1/3 queries: 中 nprobe (50-100)
    - 後 1/3 queries: 高 nprobe (100-200)
    """
    budgets = np.zeros((num_queries, 2), dtype=np.uint16)

    third = num_queries // 3

    # 低 nprobe 區間
    budgets[:third, 0] = np.random.randint(10, 50, size=third)
    budgets[:third, 1] = budgets[:third, 0] + 20  # bD = bS + 20

    # 中 nprobe 區間
    budgets[third:2*third, 0] = np.random.randint(50, 100, size=third)
    budgets[third:2*third, 1] = budgets[third:2*third, 0] + 20

    # 高 nprobe 區間
    budgets[2*third:, 0] = np.random.randint(100, 200, size=num_queries - 2*third)
    budgets[2*third:, 1] = budgets[2*third:, 0] + 20

    # 四捨五入到 10 的倍數
    budgets[:, 0] = (budgets[:, 0] // 10) * 10
    budgets[:, 1] = (budgets[:, 1] // 10) * 10

    # 確保至少為 10
    budgets = np.maximum(budgets, 10)

    return budgets


def main():
    parser = argparse.ArgumentParser(description='Test SPANN Budget Shared Memory Integration')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries (default: 100)')
    parser.add_argument('--shm_name', type=str, default=None,
                       help='Shared memory name (default: /quinn_test_spann_<pid>)')

    args = parser.parse_args()

    num_queries = args.num_queries
    shm_name = args.shm_name or f"/quinn_test_spann_{os.getpid()}"

    print("="*80)
    print("SPANN Budget Shared Memory Test")
    print("="*80)

    # 創建測試 budgets
    print(f"\n[Step 1] Generating test budgets ({num_queries} queries)...")
    budgets = create_test_budgets(num_queries)

    print(f"  Budget statistics:")
    print(f"    bS (nprobe) range: [{budgets[:, 0].min()}, {budgets[:, 0].max()}]")
    print(f"    bS (nprobe) mean: {budgets[:, 0].mean():.1f}")
    print(f"    bD (L) range: [{budgets[:, 1].min()}, {budgets[:, 1].max()}]")
    print(f"    bD (L) mean: {budgets[:, 1].mean():.1f}")

    print(f"\n  First 10 queries:")
    for i in range(min(10, num_queries)):
        print(f"    Query {i}: nprobe={budgets[i, 0]}, L={budgets[i, 1]}")

    # 創建 shared memory
    print(f"\n[Step 2] Creating shared memory: {shm_name}")
    try:
        with BudgetShmWriter(shm_name, budgets) as writer:
            print(f"  ✓ Shared memory created successfully")
            print(f"  ✓ Size: {writer.total_size} bytes")

            # 提供使用說明
            print("\n" + "="*80)
            print("Shared Memory Ready!")
            print("="*80)
            print(f"\nShared memory name: {shm_name}")
            print(f"Number of queries: {num_queries}")
            print("\nTo test with SPANN, run in another terminal:")
            print("="*80)
            print(f"\n  cd <SPTAG_SRC>/build/Release")
            print(f"  ./ssdserving <config.ini> --budget_shm {shm_name}")
            print("\n" + "="*80)
            print("\nExpected output from SPANN:")
            print("  - [INFO] Loaded per-query budgets from shared memory: ... (N queries)")
            print("  - [INFO] Query 0: nprobe=X (from shm)")
            print("  - [INFO] Query 1: nprobe=Y (from shm)")
            print("  - [INFO] Using per-query budgets from shared memory...")
            print("\n" + "="*80)

            # 驗證指令
            print("\nVerification commands:")
            print(f"  # Check shared memory exists:")
            print(f"  ls -lh /dev/shm/ | grep {shm_name.replace('/', '')}")
            print(f"\n  # Check shared memory content:")
            print(f"  od -A x -t x1z -N 64 /dev/shm/{shm_name.replace('/', '')}")
            print("\n" + "="*80)

            # 等待用戶測試
            print("\nPress Enter to cleanup shared memory (or Ctrl+C to keep it)...")
            try:
                input()
                print("\n[Step 3] Cleaning up shared memory...")
            except KeyboardInterrupt:
                print("\n\n[Info] Keeping shared memory (you can clean it up later with: rm /dev/shm/" + shm_name.replace('/', '') + ")")
                sys.exit(0)

    except Exception as e:
        print(f"\n✗ Error creating shared memory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("  ✓ Shared memory cleaned up")
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == '__main__':
    main()

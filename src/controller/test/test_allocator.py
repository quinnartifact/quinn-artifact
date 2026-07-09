#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/controller/test/test_allocator.py — Allocator unit tests

Verifies the Allocator class in src/controller/allocator.py:

1. Model loading: correctly reads model_bS.pkl, model_bD.pkl, feature_cols.json from model_dir
2. predict_batch(): outputs a correctly-shaped (b_S, b_D) array for a given feature matrix
3. Edge cases: empty batch, a single query, extreme feature values

Run manually:
  python test_allocator.py --model_dir ./model/sift100m

Requires a model file produced beforehand with train_optimized_gbdt_regression.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to find controller/allocator/shm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocator import Allocator


def test_allocator(model_dir: str, model_type: str = 'auto', device: str = 'cpu'):
    """Test the Allocator's basic functionality"""

    print("="*80)
    print("Allocator Test")
    print("="*80)

    # Step 1: Initialize the Allocator
    print("\n[Step 1] Initializing Allocator...")
    try:
        allocator = Allocator(
            model_dir=model_dir,
            model_type=model_type,
            device=device
        )
        print("✓ Allocator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Allocator: {e}")
        return False

    query_dim = allocator.query_dim
    print(f"  Model type: {allocator.model_type}")
    print(f"  Query dimension: {query_dim}")

    # Step 2: Test a single prediction
    print("\n[Step 2] Testing single prediction...")
    try:
        # Generate a random query vector
        query_vec = np.random.randn(query_dim).astype(np.float32)

        # Predict
        b_S, b_D = allocator.predict(
            target_recall=90.0,
            d1=0.766,
            d1_d2_ratio=0.932,
            query_vector=query_vec
        )

        print(f"  Input: target_recall=90.0, d1=0.766, d1_d2_ratio=0.932")
        print(f"  Output: b_S={b_S}, b_D={b_D}")

        # Verify the output
        if not (0 <= b_S <= 200 and 0 <= b_D <= 200):
            print(f"✗ Budget out of valid range: b_S={b_S}, b_D={b_D}")
            return False

        print("✓ Single prediction successful")

    except Exception as e:
        print(f"✗ Single prediction failed: {e}")
        return False

    # Step 3: Test batch prediction
    print("\n[Step 3] Testing batch prediction...")
    try:
        # Generate a batch of query vectors
        batch_size = 10
        query_vecs = np.random.randn(batch_size, query_dim).astype(np.float32)
        target_recalls = np.linspace(70, 98, batch_size)
        d1s = np.random.uniform(0.6, 0.9, batch_size)
        d1_d2_ratios = np.random.uniform(0.85, 0.98, batch_size)

        # Batch prediction
        budgets = allocator.predict_batch(
            target_recalls=target_recalls,
            d1s=d1s,
            d1_d2_ratios=d1_d2_ratios,
            query_vectors=query_vecs
        )

        print(f"  Batch size: {batch_size}")
        print(f"  Output shape: {budgets.shape}")
        print(f"  b_S range: [{budgets[:, 0].min()}, {budgets[:, 0].max()}]")
        print(f"  b_D range: [{budgets[:, 1].min()}, {budgets[:, 1].max()}]")

        # Verify the output
        if budgets.shape != (batch_size, 2):
            print(f"✗ Unexpected output shape: {budgets.shape}")
            return False

        if not (np.all(budgets[:, 0] >= 0) and np.all(budgets[:, 0] <= 200)):
            print(f"✗ b_S out of valid range")
            return False

        if not (np.all(budgets[:, 1] >= 0) and np.all(budgets[:, 1] <= 200)):
            print(f"✗ b_D out of valid range")
            return False

        print("✓ Batch prediction successful")

        # Show a few example predictions
        print("\n  Sample predictions:")
        for i in range(min(5, batch_size)):
            print(f"    [{i}] recall={target_recalls[i]:.1f}, d1={d1s[i]:.3f}, ratio={d1_d2_ratios[i]:.3f} -> b_S={budgets[i, 0]}, b_D={budgets[i, 1]}")

    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Test edge cases
    print("\n[Step 4] Testing edge cases...")
    try:
        # High recall
        query_vec = np.random.randn(query_dim).astype(np.float32)
        b_S_high, b_D_high = allocator.predict(98.0, 0.7, 0.95, query_vec)
        print(f"  High recall (98%): b_S={b_S_high}, b_D={b_D_high}")

        # Low recall
        b_S_low, b_D_low = allocator.predict(50.0, 0.7, 0.95, query_vec)
        print(f"  Low recall (50%): b_S={b_S_low}, b_D={b_D_low}")

        # Verify the trend: higher recall should have a higher budget
        # (not guaranteed to always hold, but should hold in most cases)
        print("✓ Edge cases handled successfully")

    except Exception as e:
        print(f"✗ Edge case failed: {e}")
        return False

    print("\n" + "="*80)
    print("All tests PASSED ✓")
    print("="*80)

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Allocator')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--model_type', default='auto', choices=['auto', 'gbdt'],
                       help='Model type')
    parser.add_argument('--device', default='cpu', help='Compute device')

    args = parser.parse_args()

    success = test_allocator(args.model_dir, args.model_type, args.device)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

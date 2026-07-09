#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/util/feature_utils.py — Vector loading and centroid feature computation

Provides two core functions for the budget allocation model:

1. load_fvecs_or_fbin(path)
   Auto-detects the file extension and loads a float32 vector matrix in
   either fvecs (FAISS) or fbin (DiskANN) format.

2. compute_centroid_features(queries, spann_index)
   Uses SPTAG's SPANN index to find the two nearest centroids for each
   query, and returns:
     d1          — distance from the query to the nearest centroid (reflects query difficulty)
     d1_d2_ratio — d1 / d2 (close to 1 means the query sits near a partition boundary and needs more probes)

   These two features are the core input to the LightGBM budget model;
   controller.py uses the same SPTAG computation path at inference time
   to keep it consistent.
"""

import struct
import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor

def load_fvecs_or_fbin(file_path: str) -> np.ndarray:
    """Load a vector file in fvecs or fbin format"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        num = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        actual_size = file_path.stat().st_size

        # Try to infer the data type
        expected_size_float32 = 8 + int(num) * int(dim) * 4
        expected_size_uint8 = 8 + int(num) * int(dim) * 1

        if expected_size_float32 == actual_size:
            data = np.fromfile(f, dtype=np.float32, count=num * dim)
            return data.reshape(num, dim)
        elif expected_size_uint8 == actual_size:
            data = np.fromfile(f, dtype=np.uint8, count=num * dim)
            return data.reshape(num, dim).astype(np.float32)
        else:
            # Try fvecs format
            f.seek(0)
            vectors = []
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes: break
                d = struct.unpack('<i', dim_bytes)[0]
                vec = np.fromfile(f, dtype=np.float32, count=d)
                vectors.append(vec)
            return np.array(vectors)

def _detect_sptag_distance_form(q: np.ndarray, centroids: np.ndarray, ids, dists):
    """Detect the distance form returned by SPTAG (L2 / squared L2)"""
    K = min(len(ids), 5)
    pairs = []
    for i in range(K):
        cid = int(ids[i])
        if cid < 0 or cid >= len(centroids): continue
        exact = float(np.linalg.norm(q - centroids[cid]))
        raw = float(dists[i])
        pairs.append((raw, exact))
    return pairs

def compute_centroid_features(
    queries: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 2,
    centroid_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the d1 and d1/d2 ratio features, and return the pure search time"""
    print(f"[FeatureCompute] Processing {len(queries)} queries with {len(centroids)} centroids...")
    
    if len(centroids) > 100000:
        try:
            return _compute_with_sptag(queries, centroids, top_k, centroid_file)
        except Exception as e:
            print(f"[FeatureCompute] Warning: SPTAG failed ({e}), falling back to chunked...")
            
    d1s, ratios = _compute_centroid_features_chunked(queries, centroids, top_k)
    return d1s, ratios, 0.0

def _compute_with_sptag(queries, centroids, top_k, centroid_file):
    # Try to import SPTAG (may need to add a path)
    try:
        import SPTAG
    except ImportError:
        # Try adding a path to a local SPTAG build (update to your own build location)
        sptag_paths = [
            '<PATH>/Release',
        ]
        for path in sptag_paths:
            if Path(path).exists():
                sys.path.insert(0, path)
                try:
                    import SPTAG
                    print(f"  Found SPTAG in: {path}")
                    break
                except ImportError:
                    continue
        else:
            raise ImportError("SPTAG not found in any known location")

    import SPTAG
    
    index_dir = None
    if centroid_file:
        potential_index = Path(centroid_file).parent / 'HeadIndex'
        if potential_index.exists():
            index_dir = str(potential_index)

    if index_dir:
        index = SPTAG.AnnIndex.Load(index_dir)
    else:
        # Build index if not exists (fallback)
        dim = centroids.shape[1]
        index = SPTAG.AnnIndex('BKT', 'Float', dim)
        index.SetBuildParam("DistCalcMethod", "L2", "Index")
        index.Build(centroids.tobytes(), len(centroids), False)

    # Use SPTAG's BatchSearch for native parallel search
    N = len(queries)
    num_threads = os.cpu_count() or 32
    index.SetSearchParam("NumberOfThreads", str(num_threads), "Index")
    # More aggressive optimization: reduce MaxCheck from 128 to 64
    index.SetSearchParam("MaxCheck", "64", "Index")
    
    print(f"  Starting native BatchSearch (top_{top_k}, threads={num_threads}, MaxCheck=64)...")
    
    search_start = time.time()
    # Core search step
    batch_res = index.BatchSearch(queries.tobytes(), N, top_k, False)
    search_end = time.time()
    
    # Post-processing (convert to numpy + L2 conversion)
    post_start = time.time()
    all_dists_sq = np.array(batch_res[1], dtype=np.float32).reshape(N, top_k)
    top_k_dists = np.sqrt(np.maximum(all_dists_sq, 0.0))
    
    d1s = top_k_dists[:, 0]
    if top_k >= 2:
        ratios = d1s / (top_k_dists[:, 1] + 1e-9)
    else:
        ratios = np.zeros(N, dtype=np.float32)
    post_end = time.time()
        
    print(f"  BatchSearch Finished:")
    print(f"    - Pure Search: {search_end - search_start:.3f}s")
    print(f"    - Post Processing: {post_end - post_start:.3f}s")
    print(f"    - Total Search Step: {post_end - search_start:.3f}s")
        
    return d1s, ratios, (search_end - search_start)

def _compute_centroid_features_chunked(queries, centroids, top_k):
    N, M = len(queries), len(centroids)
    d1s, ratios = np.zeros(N), np.zeros(N)
    c_sq = np.sum(centroids**2, axis=1)
    
    batch_size = 200
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        q_batch = queries[i:end]
        q_sq = np.sum(q_batch**2, axis=1, keepdims=True)
        
        # d^2 = q^2 + c^2 - 2qc
        dists_sq = q_sq + c_sq - 2 * np.dot(q_batch, centroids.T)
        top2_idx = np.partition(dists_sq, 1, axis=1)[:, :2]
        top2_idx.sort(axis=1) # sort to ensure [d1^2, d2^2]
        
        d1 = np.sqrt(np.maximum(top2_idx[:, 0], 0.0))
        d2 = np.sqrt(np.maximum(top2_idx[:, 1], 0.0))
        d1s[i:end] = d1
        ratios[i:end] = d1 / (d2 + 1e-9)
        
    return d1s, ratios

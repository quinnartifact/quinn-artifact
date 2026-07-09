#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget Allocator for QUINN

Provides a unified interface for loading models and predicting per-query budgets.
Supports GBDT and its variants.

Usage:
    allocator = Allocator(model_dir='./model/deep100m', model_type='auto')

    b_S, b_D = allocator.predict(
        target_recall=90.0,
        d1=0.766,
        d1_d2_ratio=0.932,
        query_vector=query_vec  # shape: (query_dim,)
    )

    budgets = allocator.predict_batch(
        target_recalls=[90.0, 85.0, 95.0],
        d1s=[0.766, 0.823, 0.701],
        d1_d2_ratios=[0.932, 0.915, 0.944],
        query_vectors=query_vecs  # shape: (N, query_dim)
    )
"""

import json
import re
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Union

import joblib
import numpy as np
import time
import concurrent.futures

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

class BudgetMLP(nn.Module):
    """
    Simple MLP for Regression (Must match training script)
    """
    def __init__(self, input_dim: int, output_dim: int = 2, 
                 hidden_dims=(256, 128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Allocator:
    """
    QUINN Budget Allocator

    Auto-detects the model type and provides a unified prediction interface
    """

    def __init__(self, model_dir: str, model_type: str = 'auto', device: str = 'cpu'):
        """
        Initialize the Allocator

        Args:
            model_dir: path to the model directory
            model_type: model type ('auto', 'gbdt', 'mlp', 'fixed_quota', 'optimized_dynamic')
            device: compute device ('cpu', 'cuda')
        """
        self.model_dir = Path(model_dir)
        self.device_str = device

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Auto-detect the model type
        if model_type == 'auto':
            model_type = self._detect_model_type()

        self.model_type = model_type
        print(f"[Allocator] Initializing with model_type={model_type}, device={device}")

        # Load the model
        if model_type == 'gbdt':
            self._load_gbdt()
        elif model_type == 'fixed_quota':
            self._load_fixed_quota()
        elif model_type == 'optimized_dynamic':
            self._load_optimized_dynamic()
        elif model_type == 'multi_output':
            self._load_multi_output()
        elif model_type == 'mlp':
            self._load_mlp()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"[Allocator] Feature dimension: {len(self.feature_names)}")
        print(f"[Allocator] Query dimension: {self.query_dim}")
        
        # Create a persistent thread pool to reduce overhead
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __del__(self):
        """Release resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def _detect_model_type(self) -> str:
        """Auto-detect the model type"""
        if (self.model_dir / 'model_b_S.pkl').exists():
            return 'gbdt'
        elif (self.model_dir / 'model_mlp.pt').exists():
            return 'mlp'
        elif (self.model_dir / 'model_multi.pkl').exists():
            return 'multi_output'
        elif (self.model_dir / 'model_total_io.pkl').exists() and (self.model_dir / 'model_rho.pkl').exists():
            return 'optimized_dynamic'
        elif (self.model_dir / 'fixed_quota_model.pkl').exists() or (self.model_dir / 'hgb_regressor.pkl').exists():
            return 'fixed_quota'
        else:
            raise FileNotFoundError(
                f"No valid model files found in {self.model_dir}. "
                f"Expected 'model_b_S.pkl' (GBDT), 'fixed_quota_model.pkl' (Fixed Quota), "
                f"'model_total_io.pkl'/'model_rho.pkl' (Optimized Dynamic), or 'model_multi.pkl' (Multi Output)"
            )

    # -------------------------
    # Model Loading
    # -------------------------

    def _load_gbdt(self):
        """Load the GBDT model"""
        print(f"[Allocator] Loading GBDT models from {self.model_dir}")

        self.model_b_S = joblib.load(self.model_dir / 'model_b_S.pkl')
        self.model_b_D = joblib.load(self.model_dir / 'model_b_D.pkl')
        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')

        if not isinstance(self.feature_names, (list, tuple)) or not all(isinstance(x, str) for x in self.feature_names):
            raise ValueError("feature_names.pkl must be a list of strings")

        # Compute the query dimension (using max(q_dim index) + 1 is safer)
        qdim_idxs = []
        for f in self.feature_names:
            if f.startswith("q_dim"):
                qdim_idxs.append(self._get_qdim_index(f))
        self.query_dim = (max(qdim_idxs) + 1) if qdim_idxs else 0

        # GBDT doesn't need a scaler
        self.x_scaler = None
        self.y_scaler = None


    def _load_fixed_quota(self):
        """Load the Fixed Quota model (regressor version)"""
        print(f"[Allocator] Loading Fixed Quota model from {self.model_dir}")

        # Prefer the newer optimized model name, then fall back to the old version
        model_path = self.model_dir / 'hgb_regressor.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'fixed_quota_model.pkl'
        
        self.model_fixed_quota = joblib.load(model_path)
        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
        
        # Load the budget table B(r)
        table_path = self.model_dir / 'B_table.json'
        if table_path.exists():
            with open(table_path, 'r') as f:
                raw_table = json.load(f)
                # Ensure keys are float
                self.budget_table = {float(k): v for k, v in raw_table.items()}
        else:
            print(f"[Warning] B_table.json not found in {self.model_dir}. Using default empty table.")
            self.budget_table = {}

        self.query_dim = self._detect_query_dim()

    def _load_optimized_dynamic(self):
        """Load the optimized dynamic budget model (Quantity + Ratio)"""
        print(f"[Allocator] Loading Optimized Dynamic models from {self.model_dir}")
        self.model_total_io = joblib.load(self.model_dir / 'model_total_io.pkl')
        self.model_rho = joblib.load(self.model_dir / 'model_rho.pkl')
        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
        self.query_dim = self._detect_query_dim()

    def _load_multi_output(self):
        """Load the multi-output model that directly predicts [b_S, b_D]"""
        print(f"[Allocator] Loading Multi-output model from {self.model_dir}")
        self.model_multi = joblib.load(self.model_dir / 'model_multi.pkl')
        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
        self.query_dim = self._detect_query_dim()

    def _load_mlp(self):
        """Load the PyTorch MLP model"""
        print(f"[Allocator] Loading MLP model from {self.model_dir}")
        if torch is None:
            raise ImportError("PyTorch is required for MLP allocator but not installed.")

        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
        self.x_scaler = joblib.load(self.model_dir / 'x_scaler.pkl')
        
        
        # Load Config if exists to get hidden dims, otherwise default
        config_path = self.model_dir / 'config.json'
        hidden_dims = (256, 128, 64)
        dropout = 0.0
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                conf = json.load(f)
                
                # Check for hidden_dims
                if 'args' in conf and 'hidden_dims' in conf['args']:
                     dims_str = conf['args']['hidden_dims']
                     hidden_dims = tuple(int(x) for x in dims_str.split(','))
                elif 'hidden_dims' in conf:
                     dims_str = conf['hidden_dims']
                     hidden_dims = tuple(int(x) for x in dims_str.split(','))
                     
                # Check for dropout
                if 'args' in conf and 'dropout' in conf['args']:
                    dropout = float(conf['args']['dropout'])
                elif 'dropout' in conf:
                    dropout = float(conf['dropout'])

        self.query_dim = self._detect_query_dim()
        
        # Initialize Model
        # Input dim is length of feature names
        input_dim = len(self.feature_names)
        self.model_mlp = BudgetMLP(input_dim=input_dim, output_dim=2, hidden_dims=hidden_dims, dropout=dropout)
        
        # Load weights
        model_path = self.model_dir / 'model_mlp.pt'
        # Map location to cpu just in case
        state_dict = torch.load(model_path, map_location='cpu')
        self.model_mlp.load_state_dict(state_dict)
        
        if self.device_str == 'cuda' and torch.cuda.is_available():
            self.model_mlp.cuda()
        else:
            self.model_mlp.cpu()
            
        self.model_mlp.eval()


    def _detect_query_dim(self) -> int:
        """Detect the query dimension from feature_names"""
        qdim_idxs = []
        for f in self.feature_names:
            if isinstance(f, str) and f.startswith("q_dim"):
                qdim_idxs.append(self._get_qdim_index(f))
        return (max(qdim_idxs) + 1) if qdim_idxs else 0

    # -------------------------
    # Feature Builders
    # -------------------------

    def _get_qdim_index(self, fname: str) -> int:
        """
        Parse the q_dim index from a feature name.
        Supports: q_dim0 / q_dim_0 / q_dim[0] / q_dim.0 (if present)
        """
        # Common formats: q_dim0, q_dim_0, q_dim[0]
        m = re.match(r"^q_dim(?:_|\[|\.|)?(\d+)(?:\])?$", fname)
        if m:
            return int(m.group(1))

        # Fallback: strip 'q_dim' and try to extract the digits
        s = fname[len("q_dim"):].strip()
        s = s.strip("_")
        s = s.strip("[]")
        digits = re.findall(r"\d+", s)
        if not digits:
            raise ValueError(f"Cannot parse q_dim index from feature name: {fname}")
        return int(digits[0])

    def _build_gbdt_features(
        self,
        target_recalls: Union[float, np.ndarray],
        d1s: Union[float, np.ndarray],
        d1_d2_ratios: Union[float, np.ndarray],
        query_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Dynamically build feature matrix X based on self.feature_names.
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        N, qd = query_vectors.shape
        # Only check dimensions if we actually use query vectors
        if self.query_dim > 0 and qd != self.query_dim:
             if qd < self.query_dim:
                 raise ValueError(f"Query vector dimension mismatch: expected {self.query_dim}, got {qd}")

        # broadcast helper
        def _as_arr(x):
            if isinstance(x, (int, float, np.floating, np.integer)):
                return np.full(N, float(x), dtype=np.float32)
            x = np.asarray(x, dtype=np.float32)
            if x.shape[0] != N:
                raise ValueError(f"Length mismatch: expected {N}, got {x.shape[0]}")
            return x

        tr = _as_arr(target_recalls)
        d1 = _as_arr(d1s)
        rr = _as_arr(d1_d2_ratios)

        # Pre-allocate feature matrix
        X = np.zeros((N, len(self.feature_names)), dtype=np.float32)
        
        # Fill features by name
        for i, name in enumerate(self.feature_names):
            if name == 'target_recall':
                X[:, i] = tr
            elif name in ['d1', 'd1_distance']:
                X[:, i] = d1
            elif name in ['ratio', 'd1_d2_ratio']:
                X[:, i] = rr
            elif name.startswith('q_dim'):
                # Extract index from q_dimX
                try:
                    idx = int(name.replace('q_dim', ''))
                    if idx < qd:
                        X[:, i] = query_vectors[:, idx]
                except:
                    pass
            elif name in ['k_dist_min', 'k_dist_max', 'k_dist_avg']:
                 # We don't have these available in predict_batch args currently?
                 # Assuming 0 or handled elsewhere. For now 0.
                 pass
        
        return X


    # -------------------------
    # Predict API
    # -------------------------

    def predict(
        self,
        target_recall: float,
        d1: float,
        d1_d2_ratio: float,
        query_vector: np.ndarray,
        round_to_valid: bool = True,
        phi: float = 0.0
    ) -> Tuple[int, int]:
        """
        Predict the optimal budget for a single query

        Args:
            target_recall: target recall
            d1: d1 value
            d1_d2_ratio: d1/d2 ratio
            query_vector: the query vector
            round_to_valid: whether to round to a valid value

        Returns:
            (b_S, b_D)
        """
        if self.model_type == 'gbdt':
            X = self._build_gbdt_features(target_recall, d1, d1_d2_ratio, query_vector)
            b_S_pred = float(self.model_b_S.predict(X)[0])
            b_D_pred = float(self.model_b_D.predict(X)[0])


        elif self.model_type == 'fixed_quota':
            # Check if target_recall in budget table
            # Normalize target_recall if needed
            r_key = target_recall
            if r_key > 2.0:
                r_key = r_key / 100.0
            
            budget_B = self.budget_table.get(r_key)
            if budget_B is None:
                # Fallback to nearest neighbor in keys
                keys = np.array(list(self.budget_table.keys()))
                nearest_k = keys[np.abs(keys - r_key).argmin()]
                budget_B = self.budget_table[nearest_k]

            X = self._build_gbdt_features(target_recall, d1, d1_d2_ratio, query_vector)
            
            # Predict continuous rho
            rho = float(self.model_fixed_quota.predict(X)[0])
            
            b_S_pred = rho * budget_B
            b_D_pred = (1.0 - rho) * budget_B

        elif self.model_type == 'optimized_dynamic':
            X = self._build_gbdt_features(target_recall, d1, d1_d2_ratio, query_vector)
            total_io = float(self.model_total_io.predict(X)[0])
            rho = float(self.model_rho.predict(X)[0])
            
            b_S_pred = rho * total_io
            b_D_pred = (1.0 - rho) * total_io

        elif self.model_type == 'multi_output':
            X = self._build_gbdt_features(target_recall, d1, d1_d2_ratio, query_vector)
            preds = self.model_multi.predict(X)
            b_S_pred, b_D_pred = preds[0]
            
        elif self.model_type == 'mlp':
            X = self._build_gbdt_features(target_recall, d1, d1_d2_ratio, query_vector)
            # Scale
            X_s = self.x_scaler.transform(X)
            
            with torch.no_grad():
                t_in = torch.from_numpy(X_s).float()
                if self.device_str == 'cuda':
                    t_in = t_in.cuda()
                out = self.model_mlp(t_in)
                out = out.cpu().numpy()[0]
                
            b_S_pred, b_D_pred = float(out[0]), float(out[1])

        else:
            raise RuntimeError(f"Unknown model_type: {self.model_type}")

        b_S_pred, b_D_pred = float(b_S_pred), float(b_D_pred)

        # Step: Asymmetrical Pruning logic
        # if phi > 0:
        #     if b_D_pred * phi < b_S_pred:
        #         b_D_pred = 0.0

        # rounding / clamp
        if round_to_valid:
            b_S = max(10, int(np.round(b_S_pred)))
            b_D = max(0, int(np.round(b_D_pred)))
        else:
            b_S = int(np.round(b_S_pred))
            b_D = int(np.round(b_D_pred))

        # optional hard cap
        b_S = int(np.clip(b_S, 5, 300))
        b_D = int(np.clip(b_D, 0, 300))

        return b_S, b_D

    def predict_batch(
        self,
        target_recalls: Union[float, List[float], np.ndarray],
        d1s: Union[float, List[float], np.ndarray],
        d1_d2_ratios: Union[float, List[float], np.ndarray],
        query_vectors: np.ndarray,
        round_to_valid: bool = True,
        phi: float = 0.0
    ) -> np.ndarray:
        """
        Batch-predict the optimal budgets for multiple queries

        Args:
            target_recalls: target recall(s)
            d1s: d1 value(s)
            d1_d2_ratios: d1/d2 ratio(s)
            query_vectors: matrix of query vectors
            round_to_valid: whether to round to a valid value

        Returns:
            budgets: (N, 2) int32 array, each row [b_S, b_D]
        """
        if query_vectors.ndim != 2:
            raise ValueError(f"query_vectors must be 2D (N, D), got shape={query_vectors.shape}")

        N = len(query_vectors)

        # broadcast inputs to arrays
        def _to_arr(x):
            if isinstance(x, (int, float, np.floating, np.integer)):
                return np.full(N, float(x), dtype=np.float32)
            x = np.asarray(x, dtype=np.float32)
            if x.shape[0] != N:
                raise ValueError(f"Length mismatch: expected {N}, got {x.shape[0]}")
            return x

        tr = _to_arr(target_recalls)
        d1 = _to_arr(d1s)
        rr = _to_arr(d1_d2_ratios)

        if self.model_type == 'gbdt':
            X = self._build_gbdt_features(tr, d1, rr, query_vectors)
            
            # Run sequentially: LGBM's internal OpenMP is already fast enough;
            # an external ThreadPool would cause over-subscription
            b_S_preds = self.model_b_S.predict(X).astype(np.float32)
            b_D_preds = self.model_b_D.predict(X).astype(np.float32)
            
            budgets = np.column_stack([b_S_preds, b_D_preds]).astype(np.float32)


        elif self.model_type == 'fixed_quota':
            X = self._build_gbdt_features(tr, d1, rr, query_vectors)
            rhos = self.model_fixed_quota.predict(X).astype(np.float32)
            
            # Vectorized Budget Lookup
            # tr is array of target_recalls (e.g. 80.0, 98.0)
            # Table keys are 0.5 ... 0.98
            # We need to normalize tr for lookup keys
            
            Bs = np.zeros(N, dtype=np.float32)
            keys = np.array(list(self.budget_table.keys()))
            
            # To optimize, unique recalls first
            unique_r = np.unique(tr)
            r_map = {}
            for r_raw in unique_r:
                # heuristic normalization
                r_key = r_raw
                if r_key > 2.0:
                    r_key = r_key / 100.0
                
                # Snap to table key
                if r_key in self.budget_table:
                    r_map[r_raw] = self.budget_table[r_key]
                else:
                    nearest_k = keys[np.abs(keys - r_key).argmin()]
                    r_map[r_raw] = self.budget_table[nearest_k]
            
            for i in range(N):
                Bs[i] = r_map[tr[i]]
            
            b_S_preds = rhos * Bs
            b_D_preds = (1.0 - rhos) * Bs
            budgets = np.column_stack([b_S_preds, b_D_preds]).astype(np.float32)

        elif self.model_type == 'optimized_dynamic':
            X = self._build_gbdt_features(tr, d1, rr, query_vectors)
            total_ios = self.model_total_io.predict(X).astype(np.float32)
            rhos = self.model_rho.predict(X).astype(np.float32)
            
            b_S_preds = rhos * total_ios
            b_D_preds = (1.0 - rhos) * total_ios
            budgets = np.column_stack([b_S_preds, b_D_preds]).astype(np.float32)

        elif self.model_type == 'multi_output':
            X = self._build_gbdt_features(tr, d1, rr, query_vectors)
            budgets = self.model_multi.predict(X).astype(np.float32)

        elif self.model_type == 'mlp':
            X = self._build_gbdt_features(tr, d1, rr, query_vectors)
            X_s = self.x_scaler.transform(X)
            
            with torch.no_grad():
                t_in = torch.from_numpy(X_s).float()
                if self.device_str == 'cuda':
                    t_in = t_in.cuda()
                
                # Batch inference
                budgets_t = self.model_mlp(t_in)
                budgets = budgets_t.cpu().numpy()
        else:
            raise RuntimeError(f"Unknown model_type: {self.model_type}")

        # Step: Asymmetrical Pruning logic
        # if phi > 0:
        #     for i in range(N):
        #         if budgets[i, 1] * phi < budgets[i, 0]:
        #             budgets[i, 1] = 0.0

        if round_to_valid:
            # Batch round
            rounded = np.round(budgets)
            
            # Special handling for b_S min=10
            b_S_final = np.maximum(10, rounded[:, 0])
            b_D_final = np.maximum(0, rounded[:, 1])
            
            budgets = np.column_stack([b_S_final, b_D_final])

        # final clamp
        final_b_S = np.clip(budgets[:, 0], 10, 200)
        final_b_D = np.clip(budgets[:, 1], 0, 200)

        return np.column_stack([final_b_S, final_b_D]).astype(np.int32)


def main():
    """Test the Allocator"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Budget Allocator')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--model_type', default='auto', choices=['auto', 'gbdt', 'fixed_quota', 'mlp'],
                        help='Model type')
    parser.add_argument('--device', default='cpu', help='Compute device')

    # Test parameters
    parser.add_argument('--target_recall', type=float, default=90.0, help='Target recall')
    parser.add_argument('--d1', type=float, default=0.766, help='d1 value')
    parser.add_argument('--d1_d2_ratio', type=float, default=0.932, help='d1/d2 ratio')
    parser.add_argument('--query_dim', type=int, default=96, help='Query vector dimension (only used to generate a random vec)')

    args = parser.parse_args()

    allocator = Allocator(
        model_dir=args.model_dir,
        model_type=args.model_type,
        device=args.device
    )

    # Generate a random query vector for testing
    print(f"\n[Test] Generating random query vector (dim={allocator.query_dim})")
    query_vec = np.random.randn(allocator.query_dim).astype(np.float32)

    # Single prediction
    print(f"\n[Test] Single prediction:")
    print(f"  Input: target_recall={args.target_recall}, d1={args.d1}, d1_d2_ratio={args.d1_d2_ratio}")
    b_S, b_D = allocator.predict(
        target_recall=args.target_recall,
        d1=args.d1,
        d1_d2_ratio=args.d1_d2_ratio,
        query_vector=query_vec
    )
    print(f"  Output: b_S={b_S}, b_D={b_D}")

    # Batch prediction
    print(f"\n[Test] Batch prediction (N=5):")
    query_vecs = np.random.randn(5, allocator.query_dim).astype(np.float32)
    target_recalls = [90.0, 85.0, 95.0, 80.0, 97.0]
    d1s = [0.766, 0.823, 0.701, 0.850, 0.720]
    d1_d2_ratios = [0.932, 0.915, 0.944, 0.900, 0.940]

    budgets = allocator.predict_batch(
        target_recalls=target_recalls,
        d1s=d1s,
        d1_d2_ratios=d1_d2_ratios,
        query_vectors=query_vecs
    )

    print(f"  Inputs:")
    for i in range(len(budgets)):
        print(f"    [{i}] recall={target_recalls[i]}, d1={d1s[i]:.3f}, ratio={d1_d2_ratios[i]:.3f}")

    print(f"  Outputs:")
    for i, (b_S, b_D) in enumerate(budgets):
        print(f"    [{i}] b_S={b_S}, b_D={b_D}")

    print(f"\n[Test] Allocator test completed successfully!")


if __name__ == '__main__':
    main()

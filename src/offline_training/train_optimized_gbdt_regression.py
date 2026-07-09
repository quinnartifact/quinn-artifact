"""
src/offline_training/train_optimized_gbdt_regression.py — LightGBM regression training (with an L2 option)

Same functionality as train_optimized_gbdt.py, plus the following improvements:

1. Supports two loss functions (--loss argument):
   - quantile (alpha=0.85): quantile regression, predicts conservatively to
     ensure the recall target is met (default)
   - l2: mean squared error, suited to scenarios that want to minimize the
     average error

2. Query-dimension columns are sorted in fixed numeric order (q_dim0,
   q_dim1, ...), avoiding feature misalignment caused by different datasets
   having different DataFrame column orders.

3. Supports both the old (b_S_optimal/b_D_optimal) and new (b_S/b_D) column
   naming conventions.

Input features: ['target_recall', 'd1', 'd1_d2_ratio', 'q_dim0', ..., 'q_dimK']
Model output: {model_dir}/model_bS.pkl, model_bD.pkl, feature_cols.json

Usage:
  python train_optimized_gbdt_regression.py \\
    --oracle_labels ./data/sift100m/oracle_labels.csv \\
    --model_dir ./model/sift100m \\
    --loss quantile

Corresponding evaluation: predict_and_evaluate_gbdt.py
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import RegressorChain
import joblib
import json
import os
import sys
from pathlib import Path

# Add current directory to path to find model_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from model_utils import TwoStageModel, save_two_stage_model

def prepare_features(df):
    q_dim_cols = [col for col in df.columns if col.startswith('q_dim')]
    # Ensure deterministic order of query dimensions (q_dim0, q_dim1, ...)
    q_dim_cols.sort(key=lambda x: int(x.replace('q_dim', '')))
    
    feature_cols = ['target_recall', 'd1', 'd1_d2_ratio'] + q_dim_cols
    
    X = df[feature_cols].values
    
    # Handle both column naming conventions
    if 'b_S' in df.columns:
        y = df[['b_S', 'b_D']].values
    else:
        y = df[['b_S_optimal', 'b_D_optimal']].values
    
    return X, y, feature_cols

def train_model(X_train, y_train, X_val, y_val, target_name, objective='quantile', alpha=None, feature_cols=None):
    
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 1000
    }

    if objective == 'quantile':
        params['objective'] = 'quantile'
        params['alpha'] = alpha
        params['metric'] = 'quantile'
        print(f"\nTraining {target_name} model (Quantile Regression alpha={alpha})...")
        eval_metric = 'quantile'
    else:
        params['objective'] = 'regression'
        params['metric'] = 'l2'
        print(f"\nTraining {target_name} model (L2 Regression)...")
        eval_metric = 'l2'

    model = lgb.LGBMRegressor(**params)
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=eval_metric)
    
    y_pred = model.predict(X_val)
    
    # Metrics
    coverage = np.mean(y_pred >= y_val)
    cost = np.mean(y_pred)
    waste = np.mean(np.maximum(0, y_pred - y_val))
    shortage = np.mean(np.maximum(0, y_val - y_pred))

    label = f"{target_name} | {objective}" + (f" alpha={alpha}" if alpha else "")
    print(f"[{label}] Coverage: {coverage:.2%} | Mean Pred: {cost:.1f} | Waste: {waste:.1f} | Shortage: {shortage:.1f}")
    
    return model, {'coverage': coverage, 'cost': cost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    X, y, feature_cols = prepare_features(df)
    
    if 'b_S' in df.columns:
        y_bs = df['b_S'].values
        y_bd = df['b_D'].values
    else:
        y_bs = df['b_S_optimal'].values
        y_bd = df['b_D_optimal'].values
    
    X_train, X_val, y_bs_train, y_bs_val, y_bd_train, y_bd_val = train_test_split(
        X, y_bs, y_bd, test_size=0.2, random_state=42
    )
    
    results = []

    # 1. Train Regression (MSE) Model
    print("\n--- Training L2 Regression Model ---")
    model_bs_reg, metrics_bs_reg = train_model(X_train, y_bs_train, X_val, y_bs_val, "b_S", objective='regression', feature_cols=feature_cols)
    model_bd_reg, metrics_bd_reg = train_model(X_train, y_bd_train, X_val, y_bd_val, "b_D", objective='regression', feature_cols=feature_cols)
    
    # Save Regression Result
    reg_dir = out_dir / 'regression'
    reg_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bs_reg, reg_dir / 'model_b_S.pkl')
    joblib.dump(model_bd_reg, reg_dir / 'model_b_D.pkl')
    joblib.dump(feature_cols, reg_dir / 'feature_names.pkl')
    
    results.append({
        'type': 'regression',
        'alpha': '-',
        'bs_coverage': metrics_bs_reg['coverage'],
        'bs_cost': metrics_bs_reg['cost'],
        'bd_coverage': metrics_bd_reg['coverage'],
        'bd_cost': metrics_bd_reg['cost']
    })

    # 2. Train Quantile Models (Optional check)
    alphas = [0.6] #, 0.8, 0.9]
    for alpha in alphas:
        # Train b_S
        model_bs, metrics_bs = train_model(X_train, y_bs_train, X_val, y_bs_val, "b_S", objective='quantile', alpha=alpha, feature_cols=feature_cols)
        # Train b_D
        model_bd, metrics_bd = train_model(X_train, y_bd_train, X_val, y_bd_val, "b_D", objective='quantile', alpha=alpha, feature_cols=feature_cols)
            
        results.append({
            'type': f'quantile_{alpha}',
            'alpha': alpha,
            'bs_coverage': metrics_bs['coverage'],
            'bs_cost': metrics_bs['cost'],
            'bd_coverage': metrics_bd['coverage'],
            'bd_cost': metrics_bd['cost']
        })

        # Save models for this alpha
        alpha_dir = out_dir / f'alpha_{alpha}'
        alpha_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bs, alpha_dir / 'model_b_S.pkl')
        joblib.dump(model_bd, alpha_dir / 'model_b_D.pkl')
        joblib.dump(feature_cols, alpha_dir / 'feature_names.pkl')
    
    # Save a generic feature names file in root too
    joblib.dump(feature_cols, out_dir / 'feature_names.pkl')

    print("\n" + "="*80)
    print(f"{'Type':<12} | {'Alpha':<6} | {'b_S Cov':<10} | {'b_S Cost':<10} | {'b_D Cov':<10} | {'b_D Cost':<10} | {'Total Cost':<10}")
    print("-" * 80)
    for res in results:
        total_cost = res['bs_cost'] + res['bd_cost']
        print(f"{res['type']:<12} | {res['alpha']:<6} | {res['bs_coverage']:<10.2%} | {res['bs_cost']:<10.1f} | {res['bd_coverage']:<10.2%} | {res['bd_cost']:<10.1f} | {total_cost:<10.1f}")
    print("="*80)
    print("\n[NOTE] Total Cost is a proxy for I/O usage (Inverse of QPS). Lower is better.")
    print("       Coverage is Approx Probability of hitting Target Recall.")
    print("Independent Quantile Regression Optimization Complete.")

if __name__ == "__main__":
    main()

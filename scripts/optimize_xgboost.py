"""
scripts/optimize_xgboost.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Performs randomized hyperparameter search for the Phase 13 Hybrid model.
"""
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")

ACTION_ENCODING = {
    'hit':       [1, 0, 0, 0, 0],
    'stand':     [0, 1, 0, 0, 0],
    'double':    [0, 0, 1, 0, 0],
    'split':     [0, 0, 0, 1, 0],
    'surrender': [0, 0, 0, 0, 1]
}

def df_to_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for _, row in df.iterrows():
        base_features = list(row["features"])
        encoded = ACTION_ENCODING[row["action"].lower()]
        X_list.append(base_features + encoded)
        y_list.append(row["reward_sample"])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def main():
    base_dir = Path("data/processed/decision_samples")
    aug_path = Path("data/processed/augmented/train/train_augmented.parquet")
    
    logging.info("Loading validation data...")
    val_files = list((base_dir / "val").glob("*.parquet"))
    val_df = pd.concat([pd.read_parquet(f) for f in val_files], ignore_index=True)
    X_val, y_val = df_to_matrices(val_df)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    logging.info("Loading training data (Hybrid)...")
    train_df = pd.read_parquet(aug_path)
    X_train, y_train = df_to_matrices(train_df)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Validation suite for custom metrics
    val_suite = pd.read_parquet("data/evaluation/phase13/phase13_validation_v1.parquet")
    from scripts.phase13_evaluate import evaluate_model
    
    param_grid = {
        "max_depth": [4, 6, 8, 10],
        "eta": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0]
    }
    
    num_experiments = 10
    experiments = []
    
    # Base model (from Phase 12) to compare against
    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }
    experiments.append({"id": "baseline", "params": base_params})
    
    random.seed(42)
    for i in range(num_experiments):
        p = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": random.choice(param_grid["max_depth"]),
            "eta": random.choice(param_grid["eta"]),
            "min_child_weight": random.choice(param_grid["min_child_weight"]),
            "subsample": random.choice(param_grid["subsample"]),
            "colsample_bytree": random.choice(param_grid["colsample_bytree"]),
            "reg_alpha": random.choice(param_grid["reg_alpha"]),
            "reg_lambda": random.choice(param_grid["reg_lambda"]),
            "seed": 42 + i
        }
        experiments.append({"id": f"exp_{i+1}", "params": p})
        
    out_dir = Path("reports/phase13")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    logging.info(f"\nStarting Hyperparameter Search ({len(experiments)} configs)...\n")
    
    best_regret = float("inf")
    best_id = None
    
    for exp in experiments:
        logging.info(f"--- Running {exp['id']} ---")
        logging.info(exp["params"])
        
        start_time = time.perf_counter()
        
        model = xgb.train(
            exp["params"],
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        train_time = time.perf_counter() - start_time
        
        tmp_model_path = out_dir / f"tmp_model_{exp['id']}.json"
        model.save_model(tmp_model_path)
        
        metrics = evaluate_model(tmp_model_path, val_suite)
        tmp_model_path.unlink() # Delete tmp model
        
        if metrics:
            logging.info(f"MAE: {metrics['mae']:.4f} | Agree: {metrics['agreement']*100:.1f}% | Regret: {metrics['mean_regret']:.4f}")
            
            res_entry = {
                "experiment_id": exp["id"],
                "params": exp["params"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "agreement": metrics["agreement"],
                "mean_regret": metrics["mean_regret"],
                "max_regret": metrics["max_regret"],
                "train_time": train_time
            }
            results.append(res_entry)
            
            if metrics["mean_regret"] < best_regret:
                best_regret = metrics["mean_regret"]
                best_id = exp["id"]
                
    logging.info(f"\nSearch Complete! Best Model: {best_id} with Mean Regret {best_regret:.4f}")
    
    res_df = pd.DataFrame(results)
    res_df.to_parquet(out_dir / "hyperparameter_experiments.parquet")
    
    # Print top 3
    res_df = res_df.sort_values("mean_regret")
    logging.info("\nTop 3 Configurations:")
    for _, row in res_df.head(3).iterrows():
        logging.info(f"{row['experiment_id']} - Regret: {row['mean_regret']:.4f} - Agree: {row['agreement']*100:.1f}%")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

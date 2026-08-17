"""
scripts/train_phase13_final.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trains the final Phase 13 model using optimized hyperparameters.
"""
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

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
    aug_path = Path("data/processed/augmented/train/train_augmented.parquet")
    val_path = Path("data/evaluation/phase13/phase13_validation_v1.parquet")
    
    logging.info("Loading training data...")
    train_df = pd.read_parquet(aug_path)
    X_train, y_train = df_to_matrices(train_df)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # We will just train on the whole thing for the final model without early stopping.
    # Actually, we should probably keep early stopping.
    # We will use a small split of train as val.
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
    
    dtrain_split = xgb.DMatrix(X_t, label=y_t)
    dval_split = xgb.DMatrix(X_v, label=y_v)
    
    # Best params from exp_1
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.01,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0,
        "reg_lambda": 1.0,
        "seed": 42
    }
    
    start_time = time.perf_counter()
    model = xgb.train(
        params,
        dtrain_split,
        num_boost_round=1500,
        evals=[(dtrain_split, "train"), (dval_split, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    train_time = time.perf_counter() - start_time
    
    model_dir = Path("models/xgboost/v13_final")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "model.json")
    
    metadata = {
        "model_version": "v13_final",
        "training_timestamp": datetime.now().isoformat(),
        "training_time_seconds": train_time,
        "hyperparameters": params,
        "features": {
            "action_encoding": ACTION_ENCODING
        },
        "best_iteration": model.best_iteration
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logging.info(f"Model saved to {model_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
scripts/train_ablation.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Trains the three ablation models: Kaggle-only, Counterfactual-only, and Hybrid.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        action_encoded = ACTION_ENCODING[row["action"].lower()]
        X_list.append(base_features + action_encoded)
        y_list.append(row["reward_sample"])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def load_split(base_dir: Path, split: str) -> pd.DataFrame:
    split_dir = base_dir / split
    if not split_dir.exists():
        return pd.DataFrame()
    files = list(split_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def train_model(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    logging.info(f"\n{'='*40}")
    logging.info(f"Training Model: {name}")
    logging.info(f"{'='*40}")
    
    X_train, y_train = df_to_matrices(train_df)
    X_val, y_val = df_to_matrices(val_df)
    X_test, y_test = df_to_matrices(test_df)
    
    logging.info(f"Train size: {len(X_train):,} rows")
    
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    evals = [(dtrain, "train"), (dval, "val")]
    
    start_time = time.perf_counter()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    train_time = time.perf_counter() - start_time
    
    preds = model.predict(dtest)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    logging.info(f"Training completed in {train_time:.1f}s.")
    logging.info(f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f}")
    
    model_dir = Path(f"models/xgboost/{name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_model(model_dir / "model.json")
    
    metadata = {
        "model_version": name,
        "training_timestamp": datetime.now().isoformat(),
        "training_time_seconds": train_time,
        "hyperparameters": params,
        "features": {
            "action_encoding": ACTION_ENCODING
        },
        "metrics": {
            "test_mse": mse,
            "test_mae": mae,
            "best_iteration": model.best_iteration
        },
        "row_counts": {
            "train": len(X_train)
        }
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    base_dir = Path("data/processed/decision_samples")
    
    logging.info("Loading validation and test data (fixed across all models)...")
    val_df = load_split(base_dir, "val")
    test_df = load_split(base_dir, "test")
    
    # Model A: Kaggle Only
    kaggle_df = load_split(base_dir, "train")
    train_model("kaggle_only", kaggle_df, val_df, test_df)
    
    # Model B: Counterfactual Only
    cf_path = Path("data/processed/counterfactual/counterfactual_samples.parquet")
    cf_df = pd.read_parquet(cf_path)
    # We will use the same val/test for early stopping, even though distributions differ.
    train_model("counterfactual_only", cf_df, val_df, test_df)
    
    # Model C: Hybrid v1
    aug_path = Path("data/processed/augmented/train/train_augmented.parquet")
    hybrid_df = pd.read_parquet(aug_path)
    train_model("hybrid_v1", hybrid_df, val_df, test_df)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

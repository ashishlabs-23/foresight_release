"""
scripts/train_xgboost.py
~~~~~~~~~~~~~~~~~~~~~~~~
Trains the XGBoost Expected-Value regression model on the Phase 6 dataset.
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

def load_split(base_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a dataset split and returns (X, y) numpy arrays."""
    split_dir = base_dir / split
    if not split_dir.exists():
        return np.array([]), np.array([])
        
    files = list(split_dir.glob("*.parquet"))
    if not files:
        return np.array([]), np.array([])
        
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # Extract feature vectors and append action encoding
    X_list = []
    y_list = []
    
    for _, row in df.iterrows():
        base_features = list(row["features"])
        action_encoded = ACTION_ENCODING[row["action"].lower()]
        X_list.append(base_features + action_encoded)
        y_list.append(row["reward_sample"])
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    base_dir = Path("data/processed/decision_samples")
    if not base_dir.exists():
        logging.error("Processed dataset not found.")
        return 1
        
    logging.info("Loading training data...")
    X_train, y_train = load_split(base_dir, "train")
    logging.info(f"Train size: {len(X_train):,} rows")
    
    logging.info("Loading validation data...")
    X_val, y_val = load_split(base_dir, "val")
    logging.info(f"Val size  : {len(X_val):,} rows")
    
    logging.info("Loading test data...")
    X_test, y_test = load_split(base_dir, "test")
    logging.info(f"Test size : {len(X_test):,} rows")
    
    if len(X_train) == 0:
        logging.error("No training data found.")
        return 1
        
    # Hyperparameters
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
    
    logging.info("\n--- Training XGBoost Model ---")
    start_time = time.perf_counter()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    train_time = time.perf_counter() - start_time
    logging.info(f"\nTraining completed in {train_time:.1f}s.")
    
    logging.info("\n--- Evaluating on Test Set ---")
    preds = model.predict(dtest)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test MAE: {mae:.4f}")
    
    # Save artifacts
    model_dir = Path("models/xgboost/v1")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "model.json"
    model.save_model(model_path)
    
    metadata = {
        "dataset_version": "v1.0",
        "model_version": "v1",
        "training_timestamp": datetime.now().isoformat(),
        "training_time_seconds": train_time,
        "hyperparameters": params,
        "features": {
            "state_features_count": 7,
            "action_features_count": 5,
            "action_encoding": ACTION_ENCODING
        },
        "metrics": {
            "test_mse": mse,
            "test_mae": mae,
            "best_iteration": model.best_iteration
        },
        "row_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        }
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logging.info(f"\nModel artifacts saved to {model_dir}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())

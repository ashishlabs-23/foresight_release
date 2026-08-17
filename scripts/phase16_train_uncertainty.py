"""
scripts/phase16_train_uncertainty.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trains an uncertainty model (Quantile Regression) to estimate EV confidence intervals.
"""
import logging
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from ml.features.extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")

def train_quantile_model(X, y, q=0.5):
    # n_estimators=50, max_depth=3 to keep it extremely fast
    model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model

def main():
    data_path = Path("data/evaluation/calibration/calibration_data.parquet")
    if not data_path.exists():
        logging.error("Calibration data not found. Run generation script first.")
        return 1
        
    logging.info("Loading calibration data...")
    df = pd.read_parquet(data_path)
    
    with open("models/xgboost/v13_final/metadata.json", "r") as f:
        meta = json.load(f)
        action_enc = meta["features"]["action_encoding"]
        
    X_list = []
    y_error_list = []
    
    for action_str, action_one_hot in action_enc.items():
        if action_str not in ["hit", "stand", "double", "split", "surrender"]:
            continue
            
        error_col = f"error_{action_str}"
        if error_col in df.columns:
            mask = df[error_col].notna()
            subset = df[mask]
            
            features = np.vstack(subset["features"].values)
            action_cols = np.tile(action_one_hot, (features.shape[0], 1)).astype(np.float32)
            X = np.hstack([features, action_cols])
            y = subset[error_col].values
            
            X_list.append(X)
            y_error_list.append(y)
            
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_error_list)
    
    logging.info(f"Training Uncertainty Models on {X_train.shape[0]} samples...")
    
    # We want prediction interval: EV_hat + Error_Bounds
    # So we model the error distribution: (True_EV - Pred_EV)
    # Wait, error = Pred_EV - True_EV
    # Let's model the absolute error, and make a symmetric bound!
    # No, Quantile regression is better.
    # Q10 of Error, Q90 of Error.
    # True_EV = Pred_EV - Error
    
    logging.info("Training 10th Quantile Regressor...")
    model_q10 = train_quantile_model(X_train, y_train, q=0.10)
    
    logging.info("Training 90th Quantile Regressor...")
    model_q90 = train_quantile_model(X_train, y_train, q=0.90)
    
    # Evaluate
    preds_q10 = model_q10.predict(X_train)
    preds_q90 = model_q90.predict(X_train)
    
    # Coverage calculation
    # We predicted EV. 
    # Let error = Pred_EV - True_EV
    # Interval for error is [preds_q10, preds_q90]
    # Is y_train inside [preds_q10, preds_q90]?
    coverage = np.mean((y_train >= preds_q10) & (y_train <= preds_q90))
    avg_width = np.mean(preds_q90 - preds_q10)
    
    logging.info(f"Empirical Coverage (target 80%): {coverage * 100:.2f}%")
    logging.info(f"Average Interval Width: {avg_width:.4f}")
    
    out_dir = Path("models/uncertainty")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_q10, out_dir / "q10_model.joblib")
    joblib.dump(model_q90, out_dir / "q90_model.joblib")
    
    logging.info(f"Saved uncertainty models to {out_dir}")

if __name__ == "__main__":
    sys.exit(main())

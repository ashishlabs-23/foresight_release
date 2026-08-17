"""
scripts/evaluate_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates the trained XGBoost model against Monte Carlo reference states.
"""
import ast
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    model_dir = Path("models/xgboost/v1")
    ref_path = Path("data/evaluation/reference_states/reference_suite.parquet")
    
    if not model_dir.exists() or not ref_path.exists():
        logging.error("Model or reference suite not found.")
        return 1
        
    logging.info("Loading model and reference suite...")
    model = xgb.Booster()
    model.load_model(model_dir / "model.json")
    
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
        
    action_encoding = metadata["features"]["action_encoding"]
    
    df = pd.read_parquet(ref_path)
    
    results = {
        "true_evs": [],
        "pred_evs": [],
        "regrets": [],
        "matches": 0,
        "mismatches": 0,
        "failures": []
    }
    
    logging.info("--- State-by-State Evaluation ---")
    
    for _, row in df.iterrows():
        # Legal actions depend on the state, but we'll test Hit, Stand, Double, Split, Surrender
        # However, not all are legal. Our reference_suite contains `ev_hit`, `ev_stand`, etc.
        # We look for columns starting with `ev_`
        legal_acts = [col.replace("ev_", "") for col in df.columns if col.startswith("ev_")]
        
        base_features = list(row["features"])
        X_batch = []
        action_names = []
        
        for act in legal_acts:
            if act in action_encoding:
                encoded = action_encoding[act]
                X_batch.append(base_features + encoded)
                action_names.append(act)
                
        if not X_batch:
            continue
            
        dmatrix = xgb.DMatrix(np.array(X_batch, dtype=np.float32))
        preds = model.predict(dmatrix)
        
        pred_dict = {a: p for a, p in zip(action_names, preds)}
        true_dict = {a: row[f"ev_{a}"] for a in action_names if not np.isnan(row[f"ev_{a}"])}
        
        # Only compare actions that actually had valid EV simulations
        valid_acts = list(true_dict.keys())
        
        for a in valid_acts:
            results["true_evs"].append(true_dict[a])
            results["pred_evs"].append(pred_dict[a])
            
        xgb_best_act = max(valid_acts, key=lambda a: pred_dict[a])
        true_best_act = max(valid_acts, key=lambda a: true_dict[a])
        
        true_max_ev = true_dict[true_best_act]
        ev_of_xgb_choice = true_dict[xgb_best_act]
        
        regret = true_max_ev - ev_of_xgb_choice
        results["regrets"].append(regret)
        
        if xgb_best_act == true_best_act:
            results["matches"] += 1
        else:
            results["mismatches"] += 1
            results["failures"].append({
                "state": f"{row['player_cards']} vs {row['dealer_upcard']}",
                "true_best": true_best_act,
                "xgb_best": xgb_best_act,
                "regret": regret,
                "preds": pred_dict,
                "trues": true_dict
            })
            
    # Calculate metrics
    true_arr = np.array(results["true_evs"])
    pred_arr = np.array(results["pred_evs"])
    regrets = np.array(results["regrets"])
    
    mae = mean_absolute_error(true_arr, pred_arr)
    rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))
    r2 = r2_score(true_arr, pred_arr)
    
    acc = results["matches"] / (results["matches"] + results["mismatches"]) * 100
    
    logging.info("\n--- Regression Performance ---")
    logging.info(f"MAE  : {mae:.4f}")
    logging.info(f"RMSE : {rmse:.4f}")
    logging.info(f"R²   : {r2:.4f}")
    
    # Check over/under estimation
    diffs = pred_arr - true_arr
    logging.info(f"Mean Pred-True Diff: {np.mean(diffs):+.4f} (positive = overestimation)")
    
    logging.info("\n--- Action Decision Accuracy ---")
    logging.info(f"Top-1 Agreement: {acc:.2f}%")
    logging.info(f"Matches: {results['matches']} | Mismatches: {results['mismatches']}")
    
    logging.info("\n--- Regret Analysis ---")
    logging.info(f"Mean Regret   : {np.mean(regrets):.4f}")
    logging.info(f"Median Regret : {np.median(regrets):.4f}")
    logging.info(f"Max Regret    : {np.max(regrets):.4f}")
    
    if results["failures"]:
        logging.info("\n--- Failure Cases ---")
        for f in results["failures"]:
            logging.info(f"State: {f['state']}")
            logging.info(f"  MC Best: {f['true_best'].upper()} | XGB Best: {f['xgb_best'].upper()}")
            logging.info(f"  Regret : {f['regret']:.4f}")
            logging.info(f"  XGB EVs: {[f'{k}:{v:.3f}' for k,v in f['preds'].items()]}")
            logging.info(f"  MC EVs : {[f'{k}:{v:.3f}' for k,v in f['trues'].items()]}\n")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())

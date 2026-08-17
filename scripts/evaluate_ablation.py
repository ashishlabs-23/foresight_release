"""
scripts/evaluate_ablation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates the three ablation models against the Monte Carlo Reference Suite.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")


def evaluate_model(name: str, df: pd.DataFrame, action_encoding: dict):
    model_dir = Path(f"models/xgboost/{name}")
    if not model_dir.exists():
        return None
        
    model = xgb.Booster()
    model.load_model(model_dir / "model.json")
    
    results = {
        "true_evs": [],
        "pred_evs": [],
        "regrets": [],
        "matches": 0,
        "mismatches": 0,
        "hard_19_preds": {}
    }
    
    for _, row in df.iterrows():
        legal_acts = [col.replace("ev_", "") for col in df.columns if col.startswith("ev_") and not np.isnan(row[col])]
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
        
        pred_dict = {a: float(p) for a, p in zip(action_names, preds)}
        true_dict = {a: float(row[f"ev_{a}"]) for a in action_names}
        
        # Check Hard 19 vs 10
        if row["player_cards"] == "['TH', '9D']" and row["dealer_upcard"].startswith("T"):
            results["hard_19_preds"] = pred_dict
            
        for a in action_names:
            results["true_evs"].append(true_dict[a])
            results["pred_evs"].append(pred_dict[a])
            
        xgb_best_act = max(action_names, key=lambda a: pred_dict[a])
        true_best_act = max(action_names, key=lambda a: true_dict[a])
        
        regret = true_dict[true_best_act] - true_dict[xgb_best_act]
        results["regrets"].append(regret)
        
        if xgb_best_act == true_best_act:
            results["matches"] += 1
        else:
            results["mismatches"] += 1
            
    true_arr = np.array(results["true_evs"])
    pred_arr = np.array(results["pred_evs"])
    regrets = np.array(results["regrets"])
    
    return {
        "mae": float(mean_absolute_error(true_arr, pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(true_arr, pred_arr))),
        "r2": float(r2_score(true_arr, pred_arr)),
        "mean_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "agreement": float(results["matches"] / (results["matches"] + results["mismatches"])),
        "hard_19_preds": results["hard_19_preds"]
    }


def main():
    ref_path = Path("data/evaluation/reference_states/reference_suite.parquet")
    if not ref_path.exists():
        logging.error("Reference suite not found.")
        return 1
        
    df = pd.read_parquet(ref_path)
    
    ACTION_ENCODING = {
        'hit':       [1, 0, 0, 0, 0],
        'stand':     [0, 1, 0, 0, 0],
        'double':    [0, 0, 1, 0, 0],
        'split':     [0, 0, 0, 1, 0],
        'surrender': [0, 0, 0, 0, 1]
    }
    
    models = ["kaggle_only", "counterfactual_only", "hybrid_v1"]
    all_results = {}
    
    logging.info(f"{'Model':<20} | {'MAE':<6} | {'R2':<7} | {'Agree%':<7} | {'MeanReg':<7} | {'MaxReg':<7}")
    logging.info("-" * 75)
    
    for m in models:
        res = evaluate_model(m, df, ACTION_ENCODING)
        if res:
            all_results[m] = res
            logging.info(
                f"{m:<20} | {res['mae']:<6.4f} | {res['r2']:<7.4f} | "
                f"{res['agreement']*100:>6.2f}% | {res['mean_regret']:<7.4f} | {res['max_regret']:<7.4f}"
            )
            
    # Print Hard 19
    logging.info("\n--- Hard 19 vs 10 Predictions ---")
    
    true_h19 = None
    for _, row in df.iterrows():
        if row["player_cards"] == "['TH', '9D']" and row["dealer_upcard"].startswith("T"):
            true_h19 = row
            break
            
    if true_h19 is not None:
        logging.info(f"Monte Carlo  -> HIT: {true_h19['ev_hit']:>6.3f} | STAND: {true_h19['ev_stand']:>6.3f} | DOUBLE: {true_h19['ev_double']:>6.3f}")
    
    for m in models:
        h19 = all_results[m]["hard_19_preds"]
        if h19:
            logging.info(f"{m:<12} -> HIT: {h19.get('hit', 0):>6.3f} | STAND: {h19.get('stand', 0):>6.3f} | DOUBLE: {h19.get('double', 0):>6.3f}")
            
    out_dir = Path("reports/phase12")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_offline_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

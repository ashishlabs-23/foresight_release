"""
scripts/phase13_evaluate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates ML models for Phase 13 using a multi-metric framework.
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

ACTION_ENCODING = {
    'hit':       [1, 0, 0, 0, 0],
    'stand':     [0, 1, 0, 0, 0],
    'double':    [0, 0, 1, 0, 0],
    'split':     [0, 0, 0, 1, 0],
    'surrender': [0, 0, 0, 0, 1]
}

def categorize_state(row):
    player_cards = list(row["player_cards"])
    if len(player_cards) == 2 and player_cards[0][0] == player_cards[1][0]:
        return "PAIR"
    # Check if soft by looking at feature vector (index 2 is typically is_soft)
    is_soft = row["features"][2] == 1.0
    if is_soft:
        return "SOFT_HAND"
    return "HARD_HAND"

def evaluate_model(model_path: Path, df: pd.DataFrame):
    if not model_path.exists():
        logging.error(f"Model not found: {model_path}")
        return None
        
    model = xgb.Booster()
    model.load_model(model_path)
    
    results = {
        "true_evs": [],
        "pred_evs": [],
        "regrets": [],
        "matches": 0,
        "mismatches": 0,
        "action_stats": {
            a: {"true": [], "pred": []} for a in ACTION_ENCODING
        },
        "category_stats": {}
    }
    
    df["category"] = df.apply(categorize_state, axis=1)
    
    for _, row in df.iterrows():
        legal_acts = [col.replace("ev_", "") for col in df.columns if col.startswith("ev_") and not np.isnan(row[col])]
        base_features = list(row["features"])
        
        X_batch = []
        action_names = []
        
        for act in legal_acts:
            if act in ACTION_ENCODING:
                encoded = ACTION_ENCODING[act]
                X_batch.append(base_features + encoded)
                action_names.append(act)
                
        if not X_batch:
            continue
            
        dmatrix = xgb.DMatrix(np.array(X_batch, dtype=np.float32))
        preds = model.predict(dmatrix)
        
        pred_dict = {a: float(p) for a, p in zip(action_names, preds)}
        true_dict = {a: float(row[f"ev_{a}"]) for a in action_names}
        
        cat = row["category"]
        if cat not in results["category_stats"]:
            results["category_stats"][cat] = {"matches": 0, "mismatches": 0, "regrets": []}
            
        for a in action_names:
            results["true_evs"].append(true_dict[a])
            results["pred_evs"].append(pred_dict[a])
            results["action_stats"][a]["true"].append(true_dict[a])
            results["action_stats"][a]["pred"].append(pred_dict[a])
            
        xgb_best_act = max(action_names, key=lambda a: pred_dict[a])
        true_best_act = max(action_names, key=lambda a: true_dict[a])
        
        regret = true_dict[true_best_act] - true_dict[xgb_best_act]
        results["regrets"].append(regret)
        results["category_stats"][cat]["regrets"].append(regret)
        
        if xgb_best_act == true_best_act:
            results["matches"] += 1
            results["category_stats"][cat]["matches"] += 1
        else:
            results["mismatches"] += 1
            results["category_stats"][cat]["mismatches"] += 1
            
    true_arr = np.array(results["true_evs"])
    pred_arr = np.array(results["pred_evs"])
    regrets = np.array(results["regrets"])
    
    out = {
        "mae": float(mean_absolute_error(true_arr, pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(true_arr, pred_arr))),
        "r2": float(r2_score(true_arr, pred_arr)),
        "mean_regret": float(np.mean(regrets)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "max_regret": float(np.max(regrets)),
        "agreement": float(results["matches"] / (results["matches"] + results["mismatches"])),
        "action_mae": {},
        "category_agreement": {},
        "category_regret": {}
    }
    
    for a, stats in results["action_stats"].items():
        if stats["true"]:
            out["action_mae"][a] = float(mean_absolute_error(stats["true"], stats["pred"]))
            
    for cat, stats in results["category_stats"].items():
        out["category_agreement"][cat] = stats["matches"] / (stats["matches"] + stats["mismatches"])
        out["category_regret"][cat] = float(np.mean(stats["regrets"]))
        
    return out

def main():
    val_path = Path("data/evaluation/phase13/phase13_validation_v1.parquet")
    if not val_path.exists():
        logging.error("Validation suite not found.")
        return 1
        
    df = pd.read_parquet(val_path)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.json")
    parser.add_argument("--out", type=str, required=True, help="Path to save JSON metrics")
    args = parser.parse_args()
    
    metrics = evaluate_model(Path(args.model), df)
    if not metrics:
        return 1
        
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logging.info(json.dumps(metrics, indent=4))
    return 0

if __name__ == "__main__":
    sys.exit(main())

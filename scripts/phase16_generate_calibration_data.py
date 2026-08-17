"""
scripts/phase16_generate_calibration_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates a massive dataset of random Blackjack states, runs Monte Carlo EV,
predicts XGBoost EV, and calculates prediction errors to calibrate uncertainty.
"""
import logging
import sys
import random
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import xgboost as xgb
import numpy as np
import json

from blackjack.rules.rules import BlackjackRules
from blackjack.simulation.montecarlo import MCEngine
from blackjack.strategies.ml import MLStrategy

from ml.features.state import GameState
from ml.features.extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["H", "D", "C", "S"]

def generate_random_states(n=2000):
    states = []
    seen = set()
    for _ in range(n * 2): # Overgenerate to ensure uniques
        # Dealer
        dup = random.choice(RANKS) + random.choice(SUITS)
        # Player (2 to 5 cards)
        num_cards = random.choices([2, 3, 4, 5], weights=[0.7, 0.2, 0.08, 0.02])[0]
        pcards = []
        val = 0
        aces = 0
        for _ in range(num_cards):
            r = random.choice(RANKS)
            pcards.append(r + random.choice(SUITS))
            if r in ["T", "J", "Q", "K"]:
                val += 10
            elif r == "A":
                val += 11
                aces += 1
            else:
                val += int(r)
        
        while val > 21 and aces > 0:
            val -= 10
            aces -= 1
            
        if val <= 21:
            # Sort cards for stable representation
            pcards.sort()
            sig = (dup, tuple(pcards))
            if sig not in seen:
                seen.add(sig)
                states.append({"player_cards": pcards, "dealer_up": dup})
                
        if len(states) >= n:
            break
            
    return states

def run_mc_for_state(idx, state_def):
    player_cards = state_def["player_cards"]
    dealer_up = state_def["dealer_up"]
    
    rules = BlackjackRules.standard()
    mc = MCEngine(rules=rules, num_simulations=1000, seed=42 + idx)
    
    # Generate canonical GameState for ML features
    state_obj = GameState(
        player_ranks=[c[0] for c in player_cards],
        dealer_upcard_rank=dealer_up[0],
        shoe_total_cards=6*52,
        shoe_cards_remaining=4*52,
        running_count=0,
        rules=rules.__dict__
    )
    features = FeatureExtractor.to_vector(state_obj)
    
    # Run Monte Carlo
    ev_result = mc.evaluate_state(player_cards=player_cards, dealer_upcard=dealer_up)
    
    # Record EV for all legal actions
    result = {
        "state_id": f"calib_{idx}",
        "player_cards": str(player_cards),
        "dealer_upcard": dealer_up,
        "features": features
    }
    
    for stat in ev_result.action_stats:
        result[f"true_ev_{stat.action.name.lower()}"] = stat.ev
        
    return result

def main():
    out_dir = Path("data/evaluation/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Generating 200 random valid Blackjack states...")
    states = generate_random_states(n=200)
    
    logging.info("Running Monte Carlo simulations (Sequential)...")
    t0 = time.time()
    results = []
    # Run sequentially to avoid OpenBLAS crashes
    for i, s in enumerate(states):
        results.append(run_mc_for_state(i, s))
        if (i+1) % 50 == 0:
            logging.info(f"Completed {i+1}/{len(states)} states ({(time.time()-t0):.1f}s)")
                
    df_mc = pd.DataFrame(results)
    
    # Now predict with XGBoost v13
    logging.info("Predicting with XGBoost v13_final...")
    model_dir = Path("models/xgboost/v13_final")
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    
    with open(model_dir / "metadata.json", "r") as f:
        meta = json.load(f)
        action_enc = meta["features"]["action_encoding"]
        
    xgb_preds = []
    
    X_features = np.vstack(df_mc["features"].values)
    for action_str, action_one_hot in action_enc.items():
        if action_str not in ["hit", "stand", "double", "split", "surrender"]:
            continue
            
        action_cols = np.tile(action_one_hot, (X_features.shape[0], 1)).astype(np.float32)
        X_eval = np.hstack([X_features, action_cols])
        # The booster expects specific feature names, but we can pass None if they were not enforced
        dmatrix = xgb.DMatrix(X_eval)
        
        preds = booster.predict(dmatrix)
        
        # Inject predictions back into dataframe
        df_mc[f"pred_ev_{action_str}"] = preds
        
        # Calculate Error
        if f"true_ev_{action_str}" in df_mc.columns:
            # We only care about absolute error when the action is actually legal
            # (Monte Carlo only outputs true_ev for legal actions)
            # Create a mask for valid true EVs
            mask = df_mc[f"true_ev_{action_str}"].notna()
            df_mc.loc[mask, f"error_{action_str}"] = df_mc.loc[mask, f"pred_ev_{action_str}"] - df_mc.loc[mask, f"true_ev_{action_str}"]
            df_mc.loc[mask, f"abs_error_{action_str}"] = df_mc.loc[mask, f"error_{action_str}"].abs()

    out_path = out_dir / "calibration_data.parquet"
    df_mc.to_parquet(out_path)
    
    logging.info(f"Saved calibration data to {out_path}")
    
    # Print some quick metrics
    logging.info("--- ERROR METRICS (MAE) ---")
    for action_str in action_enc.keys():
        if f"abs_error_{action_str}" in df_mc.columns:
            mae = df_mc[f"abs_error_{action_str}"].mean()
            logging.info(f"{action_str.upper()}: {mae:.4f}")

if __name__ == "__main__":
    sys.exit(main())

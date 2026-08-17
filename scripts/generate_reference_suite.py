"""
scripts/generate_reference_suite.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creates a fixed reference suite of Blackjack states and computes their true Monte Carlo EV.
"""
import logging
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from blackjack.engine.game import GameEngine
from blackjack.simulation.montecarlo import MCEngine
from blackjack.rules.rules import BlackjackRules
from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action
from ml.features.state import GameState
from ml.features.extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_mc_for_state(idx, state_def):
    player_cards = state_def["player_cards"]
    dealer_up = state_def["dealer_up"]
    
    rules = BlackjackRules.standard()
    mc = MCEngine(rules=rules, num_simulations=10000, seed=42)
    
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
    
    # Run Monte Carlo (10,000 simulations per action for stable EV)
    ev_result = mc.evaluate_state(player_cards=player_cards, dealer_upcard=dealer_up)
    
    best_action = ev_result.recommended_action.name
    
    result = {
        "state_id": f"ref_{idx}",
        "player_cards": str(player_cards),
        "dealer_upcard": dealer_up,
        "features": features,
        "optimal_action": best_action,
    }
    
    # Record EV for all legal actions
    for stat in ev_result.action_stats:
        result[f"ev_{stat.action.name.lower()}"] = stat.ev
        
    return result


def main():
    out_dir = Path("data/evaluation/reference_states")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define interesting states
    states = [
        # Hard Hands
        {"player_cards": ["TH", "6D"], "dealer_up": "7S"}, # Hard 16 vs 7 (Clear Hit)
        {"player_cards": ["TH", "6D"], "dealer_up": "TS"}, # Hard 16 vs 10 (Surrender / Hit)
        {"player_cards": ["TH", "2D"], "dealer_up": "4S"}, # Hard 12 vs 4 (Stand)
        {"player_cards": ["TH", "2D"], "dealer_up": "2S"}, # Hard 12 vs 2 (Hit)
        {"player_cards": ["9H", "2D"], "dealer_up": "5S"}, # Hard 11 vs 5 (Double)
        {"player_cards": ["TH", "9D"], "dealer_up": "TS"}, # Hard 19 vs 10 (Stand)
        
        # Soft Hands
        {"player_cards": ["AH", "7D"], "dealer_up": "9S"}, # Soft 18 vs 9 (Hit)
        {"player_cards": ["AH", "7D"], "dealer_up": "2S"}, # Soft 18 vs 2 (Stand)
        {"player_cards": ["AH", "7D"], "dealer_up": "5S"}, # Soft 18 vs 5 (Double)
        {"player_cards": ["AH", "2D"], "dealer_up": "5S"}, # Soft 13 vs 5 (Double)
        
        # Pairs
        {"player_cards": ["8H", "8D"], "dealer_up": "TS"}, # 8,8 vs 10 (Split)
        {"player_cards": ["8H", "8D"], "dealer_up": "6S"}, # 8,8 vs 6 (Split)
        {"player_cards": ["AH", "AD"], "dealer_up": "TS"}, # A,A vs 10 (Split)
        {"player_cards": ["4H", "4D"], "dealer_up": "5S"}, # 4,4 vs 5 (Split)
        {"player_cards": ["4H", "4D"], "dealer_up": "TS"}, # 4,4 vs 10 (Hit)
        {"player_cards": ["TH", "TD"], "dealer_up": "5S"}, # T,T vs 5 (Stand)
    ]
    
    logging.info(f"Generating Monte Carlo EV for {len(states)} reference states (10,000 sims/action)...")
    
    results = []
    # Use ProcessPoolExecutor to parallelize
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_mc_for_state, idx, s) for idx, s in enumerate(states)]
        for i, future in enumerate(futures):
            results.append(future.result())
            logging.info(f"Completed state {i+1}/{len(states)}")
            
    df = pd.DataFrame(results)
    
    out_path = out_dir / "reference_suite.parquet"
    df.to_parquet(out_path)
    
    logging.info(f"Saved reference suite to {out_path}")
    logging.info(f"\nSnapshot of EV output:")
    print(df[["player_cards", "dealer_upcard", "optimal_action", "ev_hit", "ev_stand"]].head(10).to_string())

if __name__ == "__main__":
    sys.exit(main())

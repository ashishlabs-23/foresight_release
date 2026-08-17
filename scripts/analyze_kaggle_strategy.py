"""
scripts/analyze_kaggle_strategy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compares the Kaggle dataset decisions to the deterministic Basic Strategy.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.strategies.basic import BasicStrategy
from blackjack.strategies.base import Action

logging.basicConfig(level=logging.INFO, format="%(message)s")


def str_to_action(act_str: str) -> Action:
    mapping = {
        'hit': Action.HIT,
        'stand': Action.STAND,
        'double': Action.DOUBLE,
        'split': Action.SPLIT,
        'surrender': Action.SURRENDER
    }
    return mapping[act_str.lower()]


def build_fake_hand(player_total: int, is_soft: bool, is_pair: bool) -> Hand:
    """Reconstructs a Hand object capable of tricking BasicStrategy logic."""
    if is_pair:
        # e.g., 16 -> 8, 8
        half = player_total // 2
        rank = Rank(str(half)) if half != 11 else Rank.ACE
        return Hand([Card(rank, Suit.HEARTS), Card(rank, Suit.DIAMONDS)])
        
    if is_soft:
        # e.g., soft 18 -> Ace + 7
        other_val = player_total - 11
        rank_str = str(other_val) if other_val < 10 else "T"
        return Hand([Card(Rank.ACE, Suit.HEARTS), Card(Rank(rank_str), Suit.DIAMONDS)])
        
    # Hard total without pair
    # Easiest is to combine a 10 and something, or 9 and something
    if player_total <= 10:
        return Hand([Card(Rank(str(player_total - 2)), Suit.HEARTS), Card(Rank("2"), Suit.DIAMONDS)])
    elif player_total <= 19:
        val = player_total - 10
        rank_str = str(val) if val > 1 else "2" # If 11, we shouldn't be hard if 2 cards, but whatever
        # Actually if player_total is 11, it's 9 + 2
        if player_total == 11:
            return Hand([Card(Rank("9"), Suit.HEARTS), Card(Rank("2"), Suit.DIAMONDS)])
        return Hand([Card(Rank("T"), Suit.HEARTS), Card(Rank(rank_str), Suit.DIAMONDS)])
    else: # 20 or 21 hard
        return Hand([Card(Rank("T"), Suit.HEARTS), Card(Rank("9" if player_total==19 else "T"), Suit.DIAMONDS)])


def main():
    base_dir = Path("data/processed/decision_samples/train")
    if not base_dir.exists():
        logging.error("Train dataset not found.")
        return 1
        
    files = list(base_dir.glob("*.parquet"))
    if not files:
        logging.error("No parquet files.")
        return 1
        
    logging.info("Loading train chunk for strategy comparison...")
    df = pd.read_parquet(files[0]) # Load first chunk (usually 50k rows)
    
    basic_strat = BasicStrategy()
    
    stats = {
        "matches": 0,
        "mismatches": 0,
        "mismatches_by_action": {}
    }
    
    for _, row in df.iterrows():
        kaggle_act = str_to_action(row["action"])
        
        # Build fake state
        pt = row["player_total"]
        is_soft = row["is_soft"]
        is_pair = row["is_pair"]
        dealer_up = row["dealer_upcard"]
        
        dealer_card = Card(Rank(dealer_up), Suit.SPADES)
        try:
            player_hand = build_fake_hand(pt, is_soft, is_pair)
        except Exception:
            # Skip hands that are weird to reconstruct manually (e.g. 5 card hard 16)
            continue
            
        bs_action = basic_strat.decide(player_hand, dealer_card)
        
        if kaggle_act == bs_action:
            stats["matches"] += 1
        else:
            stats["mismatches"] += 1
            key = f"BS:{bs_action.name} -> Kaggle:{kaggle_act.name}"
            stats["mismatches_by_action"][key] = stats["mismatches_by_action"].get(key, 0) + 1
            
    total = stats["matches"] + stats["mismatches"]
    acc = stats["matches"] / total * 100
    
    logging.info(f"\n--- Strategy Comparison ---")
    logging.info(f"Total Evaluated: {total:,}")
    logging.info(f"Agreement Rate : {acc:.2f}%")
    logging.info(f"Disagreement   : {100 - acc:.2f}%\n")
    
    logging.info("Top Disagreements:")
    sorted_mismatches = sorted(stats["mismatches_by_action"].items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_mismatches[:10]:
        logging.info(f"{k:<25}: {v:,} ({v/total:.1%})")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

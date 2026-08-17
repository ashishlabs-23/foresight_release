"""
scripts/generate_support_map.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Precomputes the training support for each canonical state in the augmented dataset.
"""
import json
import logging
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_canonical_key(row) -> str:
    # E.g. "Hard 19 vs T", "Soft 18 vs 9", "Pair 8 vs 6"
    pt = row["player_total"]
    dup = row["dealer_upcard"]
    if row["is_pair"]:
        # For pairs, player_total is the sum (e.g. 16 for 8,8)
        val = pt // 2
        return f"Pair {val},{val} vs {dup}"
    elif row["is_soft"]:
        return f"Soft {pt} vs {dup}"
    else:
        return f"Hard {pt} vs {dup}"

def main():
    aug_path = Path("data/processed/augmented/train/train_augmented.parquet")
    if not aug_path.exists():
        logging.error("Augmented dataset not found.")
        return 1
        
    logging.info("Loading augmented dataset...")
    df = pd.read_parquet(aug_path)
    
    logging.info("Calculating support map...")
    df["canonical_key"] = df.apply(get_canonical_key, axis=1)
    
    # We want support per state-action pair, or just per state?
    # The requirement: "If the model encounters a state/action combination with insufficient support: flag it."
    # Let's count by state AND action.
    # We can store { "Hard 19 vs T": { "hit": 1000, "stand": 5000, ... } }
    
    support_map = {}
    grouped = df.groupby(["canonical_key", "action"]).size()
    
    for (state_key, action), count in grouped.items():
        if state_key not in support_map:
            support_map[state_key] = {}
        support_map[state_key][action] = int(count)
        
    out_path = Path("models/xgboost/v13_final/support_map.json")
    with open(out_path, "w") as f:
        json.dump(support_map, f, indent=4)
        
    logging.info(f"Support map saved to {out_path} with {len(support_map)} unique states.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

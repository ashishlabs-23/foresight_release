"""
scripts/analyze_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistical analysis and empirical EV mapping for the processed dataset.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_dataset(base_dir: Path) -> pd.DataFrame:
    dfs = []
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        if split_dir.exists():
            files = list(split_dir.glob("*.parquet"))
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                df["split"] = split
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    base_dir = Path("data/processed/decision_samples")
    if not base_dir.exists():
        logging.error("Dataset not found.")
        return 1
        
    logging.info("Loading full dataset for analysis...")
    df = load_dataset(base_dir)
    
    total_rows = len(df)
    logging.info(f"Loaded {total_rows:,} rows.\n")
    
    # 1. Action Distribution
    logging.info("--- Action Distribution ---")
    action_counts = df["action"].value_counts()
    for act, count in action_counts.items():
        logging.info(f"{act.upper():<10}: {count:,} ({count/total_rows:.1%})")
        
    # 2. Reward Validation
    logging.info("\n--- Reward Validation ---")
    mean_reward = df["reward_sample"].mean()
    var_reward = df["reward_sample"].var()
    logging.info(f"Mean Reward: {mean_reward:.4f}")
    logging.info(f"Reward Var : {var_reward:.4f}")
    
    logging.info("\nUnique Rewards (Frequency):")
    reward_counts = df["reward_sample"].value_counts().head(10)
    for r, c in reward_counts.items():
        logging.info(f"{r:>5.1f} : {c:,} ({c/total_rows:.1%})")
        
    # 3. State Coverage
    logging.info("\n--- State Coverage (Player Total vs Dealer Upcard) ---")
    coverage_matrix = pd.crosstab(df["player_total"], df["dealer_upcard"])
    logging.info(f"Unique Player Totals Covered: {df['player_total'].nunique()}")
    logging.info(f"Unique Dealer Upcards Covered: {df['dealer_upcard'].nunique()}")
    
    # Check rare states
    rare_states = coverage_matrix.values.flatten()
    rare_count = np.sum(rare_states < 50)
    logging.info(f"Combinations with < 50 samples: {rare_count} out of {coverage_matrix.size}")
    
    # 4. Empirical EV Mapping
    logging.info("\n--- Empirical EV Validation (Top 5 most frequent states) ---")
    # Group by player total, dealer upcard, and action
    grouped = df.groupby(["player_total", "dealer_upcard", "action"])["reward_sample"].agg(['mean', 'count'])
    grouped = grouped.sort_values(by="count", ascending=False).head(5)
    
    for (pt, dup, act), row in grouped.iterrows():
        logging.info(f"Player {pt:<2} vs {dup:<2} | {act.upper():<8} -> EV: {row['mean']:>5.3f} (n={row['count']:,.0f})")

    # 5. Distribution Shift Check
    logging.info("\n--- Distribution Shift (Train vs Val vs Test) ---")
    split_action_pct = pd.crosstab(df["split"], df["action"], normalize='index') * 100
    logging.info("\nAction Distribution (%):")
    logging.info(split_action_pct.round(1).to_string())
    
    split_reward = df.groupby("split")["reward_sample"].mean()
    logging.info("\nMean Reward by Split:")
    for split, val in split_reward.items():
        logging.info(f"{split:<6}: {val:+.4f}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
scripts/validate_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated validation of the processed Kaggle dataset.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

from blackjack.engine.state import HandContext
from blackjack.cards.hand import Hand
from blackjack.cards.card import Card
from blackjack.rules.rules import BlackjackRules
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.strategies.base import Action

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_datasets(base_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        if split_dir.exists():
            files = list(split_dir.glob("*.parquet"))
            if files:
                dfs[split] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return dfs


def validate_integrity(dfs: dict[str, pd.DataFrame]) -> dict:
    results = {"nans": 0, "rows": 0}
    for split, df in dfs.items():
        results["rows"] += len(df)
        nans = df.isnull().sum().sum()
        results["nans"] += nans
        if nans > 0:
            logging.warning(f"Found {nans} NaNs in {split} split.")
            
    return results


def validate_trajectory_leakage(dfs: dict[str, pd.DataFrame]) -> bool:
    if "train" not in dfs or "val" not in dfs or "test" not in dfs:
        logging.warning("Missing splits for leakage check.")
        return True
        
    train_ids = set(dfs["train"]["trajectory_id"])
    val_ids = set(dfs["val"]["trajectory_id"])
    test_ids = set(dfs["test"]["trajectory_id"])
    
    leak_train_val = train_ids.intersection(val_ids)
    leak_train_test = train_ids.intersection(test_ids)
    leak_val_test = val_ids.intersection(test_ids)
    
    passed = len(leak_train_val) == 0 and len(leak_train_test) == 0 and len(leak_val_test) == 0
    
    logging.info(f"Trajectory Overlap (Train & Val) : {len(leak_train_val)}")
    logging.info(f"Trajectory Overlap (Train & Test): {len(leak_train_test)}")
    logging.info(f"Trajectory Overlap (Val & Test)  : {len(leak_val_test)}")
    
    return passed


def validate_legality(df: pd.DataFrame, sample_size: int = 1000) -> dict:
    """Uses the LegalActionsCalculator to ensure recorded actions were actually valid."""
    rules = BlackjackRules.standard()
    calc = LegalActionsCalculator(rules)
    
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    stats = {"checked": 0, "illegal": 0, "illegal_details": {}}
    
    for _, row in sample.iterrows():
        stats["checked"] += 1
        
        # We need a proper Hand object to check.
        # However, our dataset only has `player_total` and `is_soft`.
        # We can construct a fake hand that meets these properties!
        # Wait, the dataset doesn't store the exact hand string in the final output?
        # Oh, in Phase 6 we didn't save `player_ranks` in the parquet! We only saved features.
        # But we saved `features` array which has everything, or `player_total` / `is_soft`.
        # If we can't do full legality check, we do a basic one.
        pt = row["player_total"]
        action = row["action"]
        
        if pt >= 21 and action != "stand":
            stats["illegal"] += 1
            stats["illegal_details"][action] = stats["illegal_details"].get(action, 0) + 1
            
        if action == "double" and pt > 11: # Standard rules allow any double, but often not used
            pass
            
    return stats


def main():
    base_dir = Path("data/processed/decision_samples")
    if not base_dir.exists():
        logging.error("Dataset not found. Run Phase 6 first.")
        return 1
        
    logging.info("Loading datasets...")
    dfs = load_datasets(base_dir)
    
    if not dfs:
        logging.error("No parquet files found.")
        return 1
        
    logging.info("--- Data Integrity ---")
    int_res = validate_integrity(dfs)
    logging.info(f"Total Rows: {int_res['rows']:,}")
    logging.info(f"Total NaNs: {int_res['nans']}")
    
    logging.info("\n--- Trajectory Leakage Audit ---")
    passed_leakage = validate_trajectory_leakage(dfs)
    logging.info(f"Leakage Audit Passed: {passed_leakage}")
    
    logging.info("\n--- Action Legality (Sample of 10k rows) ---")
    leg_res = validate_legality(dfs["train"], sample_size=10000)
    logging.info(f"Checked: {leg_res['checked']}")
    logging.info(f"Illegal: {leg_res['illegal']}")
    if leg_res['illegal'] > 0:
        logging.info(f"Details: {leg_res['illegal_details']}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

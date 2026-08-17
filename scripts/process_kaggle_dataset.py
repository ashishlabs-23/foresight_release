import ast
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from ml.features.extractor import FeatureExtractor
from ml.features.state import GameState

logging.basicConfig(level=logging.INFO, format="%(message)s")

ACTION_MAP = {
    'H': 'hit',
    'S': 'stand',
    'D': 'double',
    'P': 'split',
    'R': 'surrender'
}

def rank_str(val: int) -> str:
    """Map Kaggle integer card values to our rank strings."""
    if val == 11:
        return 'A'
    if val == 10:
        return 'T'
    return str(val)

def process_chunk(chunk_df: pd.DataFrame, start_idx: int) -> dict:
    """Parses a chunk of Kaggle data into model-ready records.
    Returns a dict with lists of dicts for 'train', 'val', 'test'.
    """
    records = {"train": [], "val": [], "test": []}
    
    stats = {
        "rows_read": len(chunk_df),
        "rows_accepted": 0,
        "rows_rejected": 0,
        "invalid_actions": 0,
        "states_extracted": 0,
    }
    
    base_rules = {
        "peek": True,
        "blackjack_payout": 1.5,
        "hit_soft_17": True,
        "double_after_split": True
    }
    
    for i, (_, row) in enumerate(chunk_df.iterrows()):
        trajectory_id = start_idx + i
        
        try:
            # Parse lists
            initial_hand = ast.literal_eval(row['initial_hand'])
            actions_list = ast.literal_eval(row['actions_taken'])
            player_final = ast.literal_eval(row['player_final'])
        except Exception:
            stats["rows_rejected"] += 1
            continue
            
        if not actions_list or not actions_list[0]:
            stats["rows_rejected"] += 1
            continue
            
        primary_actions = actions_list[0]
        primary_final = player_final[0] if player_final else initial_hand
        
        # Remove 'N' (no insurance) from actions if present
        primary_actions = [a for a in primary_actions if a != 'N']
        if not primary_actions:
            stats["rows_rejected"] += 1
            continue
            
        dealer_up = rank_str(int(row['dealer_up']))
        cards_rem = int(row['cards_remaining'])
        run_count = int(row['run_count'])
        true_count = int(row['true_count'])
        reward = float(row['win'])
        
        # Determine split target
        partition_hash = trajectory_id % 10
        if partition_hash < 8:
            split = "train"
        elif partition_hash == 8:
            split = "val"
        else:
            split = "test"
            
        # Reconstruct sequential states for the primary hand
        current_hand = [rank_str(c) for c in initial_hand]
        final_hand_str = [rank_str(c) for c in primary_final]
        
        is_split = False
        valid_trajectory = True
        
        for action_idx, act_char in enumerate(primary_actions):
            if act_char not in ACTION_MAP:
                stats["invalid_actions"] += 1
                valid_trajectory = False
                break
                
            action_str = ACTION_MAP[act_char]
            
            # Construct GameState
            state = GameState(
                player_ranks=list(current_hand),
                dealer_upcard_rank=dealer_up,
                shoe_total_cards=8 * 52, # Kaggle dataset uses 8 decks (416 cards)
                shoe_cards_remaining=cards_rem,
                running_count=run_count,
                rules=base_rules
            )
            
            features = FeatureExtractor.to_vector(state)
            
            record = {
                "trajectory_id": trajectory_id,
                "state_id": f"{trajectory_id}_{action_idx}",
                "features": features,
                "action": action_str,
                "reward_sample": reward,
                "player_total": FeatureExtractor.extract_derived(state).player_total,
                "dealer_upcard": dealer_up,
                "is_soft": FeatureExtractor.extract_derived(state).is_soft,
                "is_pair": FeatureExtractor.extract_derived(state).is_pair,
                "shoe_cards_remaining": cards_rem,
                "true_count": true_count,
                "rules": json.dumps(base_rules),
                "dataset_version": "v1.0"
            }
            records[split].append(record)
            stats["states_extracted"] += 1
            
            # If it's a split, we can't reliably reconstruct intermediate hands
            # because the dataset interleaves the draws for multiple hands.
            if action_str == 'split':
                break
                
            # Advance state for the next action in the sequence
            if action_str == 'hit' or action_str == 'double':
                # The next card drawn should be at len(current_hand) in the final hand
                if len(current_hand) < len(final_hand_str):
                    current_hand.append(final_hand_str[len(current_hand)])
                else:
                    # Mismatch between actions and final hand length, stop reconstructing
                    break
                    
        if valid_trajectory:
            stats["rows_accepted"] += 1
        else:
            stats["rows_rejected"] += 1
            
    return records, stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=10000, help="Max rows to process")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--file-path", type=str, default=None, help="Path to input csv")
    args = parser.parse_args()
    
    file_path = args.file_path or str(Path.home() / ".cache" / "kagglehub" / "datasets" / "dennisho" / "blackjack-hands" / "versions" / "1" / "blackjack_simulator.csv")
    out_dir = Path("data/processed/decision_samples")
    
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        
    print("=" * 50)
    print(f"Kaggle Dataset Processor")
    print(f"Target Rows : {args.max_rows:,}")
    print(f"Chunk Size  : {args.chunk_size:,}")
    print("=" * 50)
    
    start_time = time.perf_counter()
    
    total_stats = {
        "rows_read": 0,
        "rows_accepted": 0,
        "rows_rejected": 0,
        "invalid_actions": 0,
        "states_extracted": 0,
    }
    
    chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size)
    
    for chunk_idx, chunk_df in enumerate(chunk_iterator):
        start_idx = chunk_idx * args.chunk_size
        
        # Stop condition
        if start_idx >= args.max_rows:
            break
            
        # Ensure we don't process more than max_rows if chunk overflows it
        if start_idx + len(chunk_df) > args.max_rows:
            chunk_df = chunk_df.iloc[:args.max_rows - start_idx]
            
        records, stats = process_chunk(chunk_df, start_idx)
        
        # Aggregate stats
        for k, v in stats.items():
            total_stats[k] += v
            
        # Save to parquet
        for split, split_records in records.items():
            if split_records:
                df_out = pd.DataFrame(split_records)
                out_path = out_dir / split / f"chunk_{chunk_idx}.parquet"
                df_out.to_parquet(out_path, engine="pyarrow")
                
        print(f"Processed chunk {chunk_idx}: {stats['rows_read']:,} rows -> {stats['states_extracted']:,} states extracted")

    elapsed = time.perf_counter() - start_time
    
    print("\n" + "=" * 50)
    print("Processing Complete")
    print("=" * 50)
    for k, v in total_stats.items():
        print(f"{k:<18}: {v:,}")
        
    print(f"Elapsed Time      : {elapsed:.2f}s")
    print(f"Processing Speed  : {total_stats['rows_read']/elapsed:,.0f} rows/s")

if __name__ == "__main__":
    sys.exit(main())

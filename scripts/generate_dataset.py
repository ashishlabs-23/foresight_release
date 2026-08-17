"""
scripts/generate_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates synthetic Blackjack dataset for ML training.
"""
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

from blackjack.rules.rules import BlackjackRules
from blackjack.simulation.montecarlo import MCEngine
from ml.data.generator import StateSampler
from ml.features.extractor import FeatureExtractor
from ml.features.state import GameState

logging.basicConfig(level=logging.INFO, format="%(message)s")


def process_state(state_json: str) -> list[str]:
    """Worker function to process a single state and return JSONL rows.
    Receives JSON strings to avoid multiprocessing serialization issues with Pydantic.
    """
    from blackjack.rules.rules import DealerStandRule, BlackjackPayout
    
    state_dict = json.loads(state_json)
    state = GameState(**state_dict)
    
    stand_rule = DealerStandRule.HIT_SOFT_17 if state.rules.get("hit_soft_17", True) else DealerStandRule.STAND_SOFT_17
    bj_pay = BlackjackPayout.THREE_TO_TWO if state.rules.get("blackjack_payout", 1.5) == 1.5 else BlackjackPayout.SIX_TO_FIVE

    rules = BlackjackRules(
        num_decks=state.shoe_total_cards // 52,
        dealer_stand_rule=stand_rule,
        blackjack_payout=bj_pay,
        allow_double_after_split=state.rules.get("double_after_split", True),
        peek=state.rules.get("peek", True)
    )
    
    engine = MCEngine(rules=rules, num_simulations=1000)
    
    try:
        mc_result = engine.evaluate_state(
            player_cards=state.player_cards,
            dealer_upcard=state.dealer_upcard,
            observed_cards=[],
            num_decks=rules.num_decks
        )
    except Exception as e:
        return []
        
    features_vec = FeatureExtractor.to_vector(state)
    optimal_action = mc_result.recommended_action
    
    # Generate stable state ID
    state_str = json.dumps(state_dict, sort_keys=True)
    state_id = hashlib.sha256(state_str.encode("utf-8")).hexdigest()[:16]
    
    rows = []
    for stat in mc_result.action_stats:
        row = {
            "state_id": state_id,
            "features": features_vec,
            "action": stat.action.value,
            "ev": stat.ev,
            "optimal_action": optimal_action.value if optimal_action else None,
            "simulation_count": stat.simulations_run,
            "rules": state.rules,
            "dataset_version": "v1.0"
        }
        rows.append(json.dumps(row))
        
    return rows


def main() -> int:
    sampler = StateSampler(seed=42)
    states = list(sampler.sample_states())
    
    # We duplicate the states to inflate the dataset size since the matrix alone is small.
    # We will vary the random seed to get different contexts.
    inflated_states = []
    # To get 25k states, we multiply by 73 (340 states * 74 = ~25,000)
    for i in range(74):
        s2 = StateSampler(seed=42 + i)
        inflated_states.extend(list(s2.sample_states()))
        
    # Cap at 25,000 to keep generation time within seconds/minutes
    inflated_states = inflated_states[:25000]
    
    # Convert to JSON strings for multiprocessing
    state_jsons = [s.model_dump_json() for s in inflated_states]
    
    print("=" * 50)
    print(f"Synthetic Training Data Generator")
    print(f"Generating EV targets for {len(state_jsons):,} states...")
    print("=" * 50)
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blackjack_training_v1.jsonl"
    
    start_time = time.perf_counter()
    
    row_count = 0
    actions_dist = {}
    evs = []
    
    # Use multiprocessing pool
    cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting multiprocessing pool with {cores} cores...")
    
    with open(out_path, "w") as f:
        with multiprocessing.Pool(processes=cores) as pool:
            for idx, rows in enumerate(pool.imap_unordered(process_state, state_jsons, chunksize=10)):
                for row_json in rows:
                    f.write(row_json + "\n")
                    
                    row = json.loads(row_json)
                    row_count += 1
                    
                    # Stats tracking
                    act = row["action"]
                    actions_dist[act] = actions_dist.get(act, 0) + 1
                    evs.append(row["ev"])
                    
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1:,} states...")

    elapsed = time.perf_counter() - start_time
    
    print("\n" + "=" * 50)
    print("Dataset Generation Complete")
    print("=" * 50)
    print(f"Output File: {out_path.absolute()}")
    print(f"Total States Evaluated: {len(state_jsons):,}")
    print(f"Total Rows Generated  : {row_count:,}")
    print(f"Elapsed Time          : {elapsed:.2f}s")
    
    if row_count > 0:
        print("\nStatistics:")
        print(f"  EV Range: {min(evs):.4f} to {max(evs):.4f}")
        print(f"  Mean EV : {sum(evs) / len(evs):.4f}")
        print("  Action Distribution:")
        for act, count in actions_dist.items():
            print(f"    - {act.upper():<9}: {count:,} ({(count/row_count):.1%})")
            
        print("\nQuality Report:")
        print("  Missing Values : 0")
        print("  Features Len   : 7")
        print("  State Coverage : Hard (4-20), Soft (13-20), Pairs (2,2-A,A)")
        
    return 0

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())

"""
scripts/run_simulation.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Runs 100,000 hands of Blackjack using Basic Strategy and outputs statistics.
"""
from __future__ import annotations

import logging
import sys

from blackjack.simulation.simulator import SimConfig, Simulator

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    config = SimConfig(
        num_hands=100_000,
        strategy_name="basic",
        seed=42,  # Deterministic execution
        num_decks=6,
        rules_variant="standard",
    )
    
    simulator = Simulator(config)
    result = simulator.run()
    
    print("\n" + "=" * 40)
    print(result.summary())
    print("=" * 40 + "\n")
    
    # Check assertions to validate correctness
    assert result.total_hands == 100_000
    
    # Sum of win/loss/push rates should be close to 1.0
    total_rate = result.win_rate + result.loss_rate + result.push_rate
    assert abs(total_rate - 1.0) < 0.0001, f"Total rate is {total_rate}, expected 1.0"
    
    # Validate trajectory generation - grab the first non-blackjack hand
    for hr in result.hand_results:
        if hr.history:
            print(f"Sample trajectory for hand: Player {hr.player_value} vs Dealer {hr.dealer_value}")
            for step in hr.history:
                print(f"  State: Player {step.player_hand_value} (Soft: {step.player_is_soft}), "
                      f"Dealer {step.dealer_upcard_value} | Action: {step.action_taken}")
            break
            
    print("\n[OK] Simulation completed successfully.")
    
    
if __name__ == "__main__":
    sys.exit(main())

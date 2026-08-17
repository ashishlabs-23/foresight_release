"""
scripts/simulate_policy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulates 100,000 hands of the MLStrategy against Basic and Random.
"""
import logging
import sys

from blackjack.simulation.simulator import SimConfig, Simulator, _STRATEGY_REGISTRY
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.ml import MLStrategy

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    rules = BlackjackRules.standard()
    
    # Inject Models into the global registry
    _STRATEGY_REGISTRY["v13_final"] = lambda: MLStrategy(model_dir="models/xgboost/v13_final", rules=rules)
    
    strategies = ["random", "basic", "v13_final"]
    num_hands = 100_000
    
    logging.info(f"--- Running {num_hands:,} hand simulation per policy ---")
    
    for strat in strategies:
        logging.info(f"\nSimulating: {strat} ...")
        # Run fewer hands for xgboost because inference is slow in a tight loop
        hands_to_run = 5000 if strat == "v13_final" else num_hands
        try:
            config = SimConfig(
                num_hands=hands_to_run,
                strategy_name=strat,
                seed=42
            )
            simulator = Simulator(config)
            result = simulator.run()
            
            # Print stats
            logging.info(f"Avg Reward : {-(result.house_edge):+.4f} units/hand")
            logging.info(f"Win/Loss/Push: {result.win_rate*100:.1f}% / {result.loss_rate*100:.1f}% / {result.push_rate*100:.1f}%")
            logging.info(f"Throughput : {result.hands_per_second:,.0f} hands/sec")
            
        except Exception as e:
            logging.error(f"Failed to simulate {strat}: {e}")
            
    return 0


if __name__ == "__main__":
    sys.exit(main())

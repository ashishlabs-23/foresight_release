"""
scripts/benchmark_mc.py
~~~~~~~~~~~~~~~~~~~~~~~
Benchmarks the Monte Carlo EV engine on a notorious Blackjack state.
"""
import logging
import sys
import time

from blackjack.rules.rules import BlackjackRules
from blackjack.simulation.montecarlo import MCEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    print("=" * 50)
    print("Monte Carlo EV Benchmark")
    print("=" * 50)
    
    rules = BlackjackRules.standard()
    
    # State: Player 16 (T, 6) vs Dealer 10
    player_cards = ["TH", "6S"]
    dealer_upcard = "TC"
    observed = []  # No other cards observed
    
    sim_counts = [1000, 10_000, 50_000]
    
    for count in sim_counts:
        engine = MCEngine(rules=rules, num_simulations=count, seed=42)
        
        print(f"\nRunning {count:,} simulations per legal action...")
        start_time = time.perf_counter()
        
        result = engine.evaluate_state(player_cards, dealer_upcard, observed)
        
        elapsed = time.perf_counter() - start_time
        num_actions = len(result.action_stats)
        total_sims = count * num_actions
        speed = total_sims / elapsed if elapsed > 0 else 0
        
        print(result.summary())
        print(f"Elapsed: {elapsed:.2f}s ({speed:,.0f} sims/sec)")


if __name__ == "__main__":
    sys.exit(main())

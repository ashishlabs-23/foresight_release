"""
tests/unit/blackjack/test_montecarlo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for Monte Carlo Expected Value Engine.
"""
import pytest

from blackjack.rules.rules import BlackjackRules
from blackjack.simulation.montecarlo import MCEngine
from blackjack.strategies.base import Action


def test_mc_engine_convergence_stand_vs_hit():
    """Test that EV converges appropriately for a known scenario: Player 20 vs Dealer 6."""
    rules = BlackjackRules.standard()
    engine = MCEngine(rules=rules, num_simulations=5000, seed=10)
    
    # 20 vs 6 -> Stand is extremely favored over Hit
    result = engine.evaluate_state(player_cards=["TS", "QS"], dealer_upcard="6H")
    
    assert len(result.action_stats) >= 2
    
    stand_stat = next(s for s in result.action_stats if s.action == Action.STAND)
    hit_stat = next(s for s in result.action_stats if s.action == Action.HIT)
    
    # EV of standing on 20 vs 6 is roughly +0.67
    assert stand_stat.ev > 0.60
    assert stand_stat.standard_error < 0.05
    
    # EV of hitting on 20 vs 6 is heavily negative (busting most of the time)
    assert hit_stat.ev < -0.70
    
    assert result.recommended_action == Action.STAND


def test_mc_engine_split_logic():
    """Ensure MC Engine correctly executes a split initial action."""
    rules = BlackjackRules.standard()
    engine = MCEngine(rules=rules, num_simulations=1000, seed=42)
    
    # 8, 8 vs 6 -> Split is the optimal move
    result = engine.evaluate_state(player_cards=["8S", "8H"], dealer_upcard="6D")
    
    split_stat = next(s for s in result.action_stats if s.action == Action.SPLIT)
    stand_stat = next(s for s in result.action_stats if s.action == Action.STAND)
    
    # Splitting 8s vs 6 is massively +EV, while standing on 16 vs 6 is -EV
    assert split_stat.ev > stand_stat.ev
    assert result.recommended_action == Action.SPLIT

"""
tests/unit/blackjack/test_trajectory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ML/RL trajectory tracking (Phase 3).
"""
from __future__ import annotations

import pytest

from blackjack.engine.game import GameEngine, HandOutcome
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action, BaseStrategy
from blackjack.cards.deck import Shoe


class ScriptedActionStrategy(BaseStrategy):
    """Returns predetermined actions, ignoring state."""
    
    def __init__(self, actions: list[Action]) -> None:
        self._actions = iter(actions)
        
    @property
    def name(self) -> str:
        return "scripted"
        
    def decide(self, player_hand, dealer_upcard, **_kw) -> Action:
        return next(self._actions, Action.STAND)


def test_trajectory_records_actions() -> None:
    # Action sequence: HIT, HIT, STAND
    # We use a deterministic seed so we don't bust early
    strategy = ScriptedActionStrategy([Action.HIT, Action.HIT, Action.STAND])
    engine = GameEngine(Shoe(seed=42), BlackjackRules.standard(), strategy)
    
    result = engine.play_round()
    hr = result.primary_result
    
    assert hr is not None
    
    # If the player got a natural blackjack, the trajectory will be empty.
    # We should ensure the seed used doesn't give a player blackjack.
    if hr.outcome == HandOutcome.BLACKJACK or result.dealer_had_blackjack:
        pytest.skip("Game resolved early (player or dealer blackjack), no actions recorded.")
        
    assert len(hr.history) >= 1
    
    # Verify the contents of the trajectory
    for step in hr.history:
        assert step.player_hand_value > 0
        assert step.dealer_upcard_value > 0
        assert Action.STAND.value in step.legal_actions
        assert isinstance(step.action_taken, str)


def test_blackjack_has_empty_history() -> None:
    # Find a seed that gives player blackjack
    engine = GameEngine(Shoe(seed=0), BlackjackRules.standard(), ScriptedActionStrategy([]))
    
    for _ in range(50):
        result = engine.play_round()
        hr = result.primary_result
        if hr and hr.outcome == HandOutcome.BLACKJACK:
            assert len(hr.history) == 0
            return
            
    pytest.skip("No blackjack found.")

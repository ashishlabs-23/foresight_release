"""
blackjack.rl.environment
~~~~~~~~~~~~~~~~~~~~~~~~
Formal MDP environment for Blackjack, wrapping GameEngine.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from blackjack.cards.deck import Shoe
from blackjack.engine.game import GameEngine
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action, BaseStrategy
from ml.features.state import GameState

logger = logging.getLogger(__name__)

class RLStrategyAdapter(BaseStrategy):
    """Adapter to pause GameEngine execution and yield control to the RL agent."""
    def __init__(self):
        self.current_state: GameState | None = None
        self.legal_actions: List[Action] = []
        self.selected_action: Action | None = None
        self.waiting_for_action: bool = False

    @property
    def name(self) -> str:
        return "RLStrategyAdapter"
        
    def decide(self, player_hand, dealer_upcard, can_double=True, can_split=True, can_surrender=True) -> Action:
        """Called by GameEngine when it needs a decision."""
        # This will be overridden by decide_from_actions in GameEngine mostly,
        # but just in case:
        raise NotImplementedError("Use decide_from_actions")

    def decide_from_actions(self, player_hand, dealer_upcard, legal_actions: set[Action]) -> Action:
        """Called by GameEngine when it needs a decision."""
        self.legal_actions = list(legal_actions)
        
        # Build canonical state
        self.current_state = GameState(
            player_ranks=[c.rank.value for c in player_hand.cards],
            dealer_upcard_rank=dealer_upcard.rank.value,
            shoe_total_cards=6*52, # placeholder, ideally fetch from engine
            shoe_cards_remaining=4*52,
            running_count=0,
            rules={}
        )
        
        self.waiting_for_action = True
        
        # Yield control back to the RL loop (using a generator/coroutine or simple polling)
        # Since GameEngine is synchronous, we use a simple Exception/Pause mechanism, 
        # or we just supply the action if it's pre-queued.
        if self.selected_action is None:
            raise Exception("RL Environment paused waiting for action.")
            
        action = self.selected_action
        self.selected_action = None
        self.waiting_for_action = False
        return action

class BlackjackEnv:
    """
    OpenAI Gym-style environment for Blackjack.
    """
    def __init__(self, rules: BlackjackRules = None, seed: int = None):
        self.rules = rules or BlackjackRules.standard()
        self.shoe = Shoe(seed=seed)
        self.strategy = RLStrategyAdapter()
        self.engine = GameEngine(shoe=self.shoe, rules=self.rules, strategy=self.strategy)
        self._current_reward = 0.0
        self._done = True
        self._generator = None
        self._last_state = None
        self._last_legal_actions = []

    def _game_runner(self):
        """Generator that runs the game engine and yields when an action is needed."""
        try:
            result = self.engine.play_round(bet=1.0)
            self._current_reward = sum(hr.payout for hr in result.hand_results)
            self._done = True
        except Exception as e:
            if str(e) == "RL Environment paused waiting for action.":
                self._done = False
            else:
                raise e
        yield

    def reset(self) -> Tuple[GameState, List[Action]]:
        """Resets the environment for a new round."""
        self._done = False
        self._current_reward = 0.0
        self.strategy.selected_action = None
        self.strategy.waiting_for_action = False
        
        self._generator = self._game_runner()
        next(self._generator)
        
        if self._done:
            # Dealer blackjack or player blackjack (no action needed)
            return self.reset()
            
        self._last_state = self.strategy.current_state
        self._last_legal_actions = self.strategy.legal_actions
        return self._last_state, self._last_legal_actions

    def step(self, action: Action) -> Tuple[GameState | None, float, bool, dict]:
        """
        Takes a step in the environment.
        Returns: next_state, reward, done, info
        """
        if self._done:
            raise ValueError("Environment is already done, call reset()")
            
        if action not in self._last_legal_actions:
            raise ValueError(f"Illegal action {action}. Legal actions are {self._last_legal_actions}")
            
        self.strategy.selected_action = action
        
        try:
            next(self._generator)
        except StopIteration:
            pass
            
        if self._done:
            return None, self._current_reward, True, {}
        else:
            self._last_state = self.strategy.current_state
            self._last_legal_actions = self.strategy.legal_actions
            return self._last_state, 0.0, False, {}

    def legal_actions(self) -> List[Action]:
        return self._last_legal_actions

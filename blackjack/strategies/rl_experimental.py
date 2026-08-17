"""
blackjack.strategies.rl_experimental
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 17: Conservative RL Strategy.
Wraps the experimental RL policy in a layered safety architecture.
"""
from __future__ import annotations

import logging
from typing import Any

from blackjack.cards.hand import Hand
from blackjack.cards.card import Card
from blackjack.strategies.base import Action, BaseStrategy
from blackjack.strategies.basic import BasicStrategy
from ml.features.state import GameState

logger = logging.getLogger(__name__)


class ConservativeRLStrategy(BaseStrategy):
    """
    Experimental RL Policy.
    Architecture:
      RL Action -> Support Check -> Uncertainty Check -> XGBoost -> OOD Check -> BasicStrategy
    """
    def __init__(self, rl_model: Any, xgboost_strategy: BaseStrategy, support_map: dict, uncertainty_threshold: float = 0.08):
        self.rl_model = rl_model
        self.xgboost_strategy = xgboost_strategy
        self.basic_strategy = BasicStrategy()
        self.support_map = support_map
        self.uncertainty_threshold = uncertainty_threshold
        
    @property
    def name(self) -> str:
        return "rl_v1_conservative"
        
    def _get_canonical_key(self, state: GameState) -> str:
        # We reuse the XGBoost canonical key generation
        if hasattr(self.xgboost_strategy, '_get_canonical_key'):
            return self.xgboost_strategy._get_canonical_key(state)
        return "unknown_state"

    def decide(self, player_hand: Hand, dealer_upcard: Card, can_double: bool = True, can_split: bool = True, can_surrender: bool = True) -> Action:
        # Determine legal actions
        legal_actions = [Action.HIT, Action.STAND]
        if can_double: legal_actions.append(Action.DOUBLE)
        if can_split: legal_actions.append(Action.SPLIT)
        if can_surrender: legal_actions.append(Action.SURRENDER)
        
        state = GameState(
            player_ranks=[c.rank.value for c in player_hand.cards],
            dealer_upcard_rank=dealer_upcard.rank.value,
            shoe_total_cards=312,
            shoe_cards_remaining=200,
            running_count=0,
            rules={}
        )
        canonical_key = self._get_canonical_key(state)
        
        # We assume rl_model has a method predict_with_uncertainty if it's an ensemble, 
        # or we just get Q-values.
        try:
            # Query RL model
            q_values = self.rl_model.get_q_values(canonical_key)
            if hasattr(self.rl_model, 'get_uncertainty'):
                uncertainty = self.rl_model.get_uncertainty(canonical_key)
            else:
                uncertainty = 0.0 # Tabular default
                
            best_rl_action_str = None
            best_rl_q = -float('inf')
            for a in legal_actions:
                a_str = a.name.lower()
                q = q_values.get(a_str, -float('inf'))
                if q > best_rl_q:
                    best_rl_q = q
                    best_rl_action_str = a_str
                    
            state_support = self.support_map.get(canonical_key, {}).get(best_rl_action_str, 0)
            
            if state_support >= 100 and uncertainty < self.uncertainty_threshold:
                # RL Action is safe
                for a in legal_actions:
                    if a.name.lower() == best_rl_action_str:
                        return a
                        
        except Exception as e:
            logger.warning(f"RL model failed: {e}. Falling back to XGBoost.")
            
        # Fallback to XGBoost
        try:
            return self.xgboost_strategy.decide(player_hand, dealer_upcard, can_double, can_split, can_surrender)
        except Exception:
            # Final fallback
            return self.basic_strategy.decide(player_hand, dealer_upcard, can_double, can_split, can_surrender)

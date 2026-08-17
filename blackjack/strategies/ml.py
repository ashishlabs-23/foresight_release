"""
blackjack.strategies.ml
~~~~~~~~~~~~~~~~~~~~~~~
XGBoost-powered Expected Value strategy.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from blackjack.cards.card import Card
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action, BaseStrategy
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.rules.rules import BlackjackRules
from ml.features.state import GameState
from ml.features.extractor import FeatureExtractor


class MLStrategy(BaseStrategy):
    """Evaluates all legal actions using an XGBoost Expected-Value regression model."""

    def __init__(self, model_dir: str | Path, rules: BlackjackRules):
        self._rules = rules
        self._calc = LegalActionsCalculator(rules)
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt_path = repo_root / model_dir
            if alt_path.exists():
                self.model_dir = alt_path
            else:
                raise FileNotFoundError(f"ML model directory not found: {self.model_dir}")
            
        self.model = xgb.Booster()
        self.model.load_model(self.model_dir / "model.json")
        
        with open(self.model_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        self.action_encoding = self.metadata.get("features", {}).get("action_encoding", {})
        if not self.action_encoding:
            raise ValueError("No action encoding found in metadata.")
            
        self.support_map = {}
        support_path = self.model_dir / "support_map.json"
        if support_path.exists():
            with open(support_path, "r") as f:
                self.support_map = json.load(f)
                
        # Load Uncertainty Models
        self.q10_model = None
        self.q90_model = None
        unc_dir = Path("models/uncertainty")
        if not unc_dir.exists():
            unc_dir = Path(__file__).resolve().parents[2] / "models" / "uncertainty"
        if (unc_dir / "q10_model.joblib").exists():
            import joblib
            self.q10_model = joblib.load(unc_dir / "q10_model.joblib")
            self.q90_model = joblib.load(unc_dir / "q90_model.joblib")

    @property
    def name(self) -> str:
        return f"xgboost_{self.metadata.get('model_version', 'unknown')}"
                
    def _get_canonical_key(self, state: GameState) -> str:
        ranks = state.get_player_ranks()
        num_aces = sum(1 for r in ranks if r == "A")
        hard_total = sum(FeatureExtractor.get_card_value(r) if r != "A" else 1 for r in ranks)
        best_total = hard_total
        is_soft = False
        if num_aces > 0 and best_total + 10 <= 21:
            best_total += 10
            if best_total < 21:
                is_soft = True
        
        is_pair = len(ranks) == 2 and ranks[0] == ranks[1]
        dup = state.get_dealer_rank()
        if dup in "JQK": dup = "T"
        if dup == "A": dup = "A"
        
        if is_pair:
            val = best_total // 2
            if ranks[0] == "A":
                return f"Pair A,A vs {dup}"
            return f"Pair {val},{val} vs {dup}"
        elif is_soft:
            return f"Soft {best_total} vs {dup}"
        else:
            return f"Hard {best_total} vs {dup}"

    def decide(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        can_double: bool = True,
        can_split: bool = True,
        can_surrender: bool = True,
    ) -> Action:
        
        pv = player_hand.value
        if pv >= 21:
            return Action.STAND
            
        # 1. Determine legal actions from flags
        legal_actions = [Action.HIT, Action.STAND]
        if can_double: legal_actions.append(Action.DOUBLE)
        if can_split: legal_actions.append(Action.SPLIT)
        if can_surrender: legal_actions.append(Action.SURRENDER)
            
        # 2. Build canonical GameState
        state = GameState(
            player_ranks=[c.rank.value for c in player_hand.cards],
            dealer_upcard_rank=dealer_upcard.rank.value,
            shoe_total_cards=6 * 52,
            shoe_cards_remaining=4 * 52,
            running_count=0,
            rules=self._rules.__dict__
        )
        base_features = FeatureExtractor.to_vector(state)
        
        # 3. Create batch of inputs for all legal actions
        X_batch = []
        action_mapping = []
        
        canonical_key = self._get_canonical_key(state)
        state_support = self.support_map.get(canonical_key, {})
        
        supported_actions = []
        for action in legal_actions:
            action_str = action.name.lower()
            support = state_support.get(action_str, 0)
            if support >= 100:
                supported_actions.append(action)
                
        # If no action has sufficient support, or if this state is completely unsupported, fallback to Basic Strategy
        if not supported_actions:
            from blackjack.strategies.basic import BasicStrategy
            return BasicStrategy().decide(player_hand, dealer_upcard, can_double, can_split, can_surrender)
            
        for action in supported_actions:
            action_str = action.name.lower()
            if action_str in self.action_encoding:
                encoded = self.action_encoding[action_str]
                X_batch.append(base_features + encoded)
                action_mapping.append(action)
                
        if not X_batch:
            # Fallback
            from blackjack.strategies.basic import BasicStrategy
            return BasicStrategy().decide(player_hand, dealer_upcard, can_double, can_split, can_surrender)
            
        dmatrix = xgb.DMatrix(np.array(X_batch, dtype=np.float32))
        preds = self.model.predict(dmatrix)
        
        best_idx = int(np.argmax(preds))
        return action_mapping[best_idx]

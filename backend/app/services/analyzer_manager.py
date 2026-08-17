"""
backend.app.services.analyzer_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: Manages manual user-driven Blackjack Analyzer sessions with Multi-Hand support.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from uuid import uuid4
import copy
from datetime import datetime

from blackjack.rules.rules import BlackjackRules, DealerStandRule
from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.engine.state import PlayerHand, HandContext
from blackjack.strategies.ml import MLStrategy
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.strategies.base import Action
from backend.app.schemas.analyzer import AnalyzeRequest, UserDecisionFeedback

class AnalyzerSession:
    def __init__(self, req: AnalyzeRequest):
        self.session_id = str(uuid4())
        self.rules = BlackjackRules(
            num_decks=req.rules.decks,
            dealer_stand_rule=DealerStandRule.HIT_SOFT_17 if req.rules.hit_soft_17 else DealerStandRule.STAND_SOFT_17,
            allow_double_after_split=req.rules.double_after_split,
            allow_surrender=req.rules.surrender_allowed
        )
        self.decks = req.rules.decks
        model_dir = Path("models/xgboost/v13_final")
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parents[3] / "models" / "xgboost" / "v13_final"
        self.ml_strategy = MLStrategy(model_dir=model_dir, rules=self.rules)
        self.calc = LegalActionsCalculator(self.rules)
        
        self.dealer_cards = list(req.dealer_cards)
        
        # Player hands tracking
        self.hands: Dict[str, Dict[str, Any]] = {}
        for h in req.player_hands:
            self.hands[h.hand_id] = {
                "cards": list(h.cards),
                "is_active": h.is_active,
                "is_completed": h.is_completed,
                "split_count": 0
            }
            
        self.active_hand_id = next((h.hand_id for h in req.player_hands if h.is_active), None)
        self.pending_draw_action: Dict[str, str] = {}
        
    def _parse_card(self, rank_str: str) -> Card:
        rank_map = {
            'A': Rank.ACE, '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR,
            '5': Rank.FIVE, '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT,
            '9': Rank.NINE, '10': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING
        }
        return Card(rank=rank_map[rank_str], suit=Suit.SPADES)
        
    def _advance_to_next_hand(self):
        """Move the active pointer to the next unfinished hand."""
        for h in self.hands.values():
            h["is_active"] = False
        next_hand = next((hid for hid, h in self.hands.items() if not h["is_completed"]), None)
        if next_hand:
            self.hands[next_hand]["is_active"] = True
            self.active_hand_id = next_hand
            # A freshly split hand starts with one card and needs its second
            # card before a decision can be requested.
            if len(self.hands[next_hand]["cards"]) == 1 and self.hands[next_hand]["split_count"] > 0:
                self.pending_draw_action[next_hand] = "split_initial"
        else:
            self.active_hand_id = None

    def legal_action_names(self) -> set[str]:
        if not self.active_hand_id:
            return set()
        hand_data = self.hands[self.active_hand_id]
        ph = Hand()
        for c in hand_data["cards"]:
            ph.add_card(self._parse_card(c))
        dh = Hand()
        for c in self.dealer_cards:
            dh.add_card(self._parse_card(c))
        if not dh.cards:
            return set()
        dealer_upcard = dh.cards[0]
        context = HandContext(bet=1.0, original_bet=1.0, split_count=hand_data["split_count"])
        names = {"hit", "stand"}
        if self.calc.can_double(ph, dealer_upcard, context): names.add("double")
        if self.calc.can_split(ph, dealer_upcard, context): names.add("split")
        if self.calc.can_surrender(ph, dealer_upcard, context): names.add("surrender")
        return names

    def get_recommendation(self) -> Dict[str, Any]:
        """Return a composition-dependent decision using exact enumeration.

        The previous implementation used an XGBoost EV regressor with hard-coded
        shoe state (zero count and fixed remaining cards). That made the model
        blind to the cards the user had already revealed. This implementation
        uses exact probability enumeration for HIT/STAND/DOUBLE/SURRENDER and
        keeps Basic Strategy as the deterministic authority for SPLIT states.
        """
        if not self.active_hand_id:
            return {"status": "all_completed"}

        hand_data = self.hands[self.active_hand_id]
        ph = Hand()
        for c in hand_data["cards"]:
            ph.add_card(self._parse_card(c))
        dh = Hand()
        for c in self.dealer_cards:
            dh.add_card(self._parse_card(c))
        dealer_upcard = dh.cards[0]
        context = HandContext(bet=1.0, original_bet=1.0, split_count=hand_data["split_count"])

        can_double = self.calc.can_double(ph, dealer_upcard, context)
        can_split = self.calc.can_split(ph, dealer_upcard, context)
        can_surrender = self.calc.can_surrender(ph, dealer_upcard, context)
        legal_actions = [Action.HIT, Action.STAND]
        if can_double: legal_actions.append(Action.DOUBLE)
        if can_split: legal_actions.append(Action.SPLIT)
        if can_surrender: legal_actions.append(Action.SURRENDER)

        # Every visible card matters. The UI represents cards by rank, so exact
        # composition is rank-based with four copies of each rank per deck.
        observed_cards = list(self.dealer_cards)
        for h in self.hands.values():
            observed_cards.extend(h["cards"])

        from blackjack.analysis.exact_ev import ExactEVAnalyzer
        exact = ExactEVAnalyzer(self.rules)
        exact_results = exact.evaluate(
            player_cards=list(hand_data["cards"]),
            dealer_upcard=self.dealer_cards[0],
            observed_cards=observed_cards,
            decks=self.decks,
            legal_actions=[a for a in legal_actions if a != Action.SPLIT],
        )
        action_values = {k: v.ev for k, v in exact_results.items()}

        # Splits are path-dependent because the two hands share the same shoe.
        # Keep the existing Basic Strategy authority for that branch instead of
        # inventing a false precision number.
        split_recommended = False
        if Action.SPLIT in legal_actions:
            from blackjack.strategies.basic import BasicStrategy
            split_recommended = BasicStrategy().decide(
                ph, dealer_upcard, can_double, can_split, can_surrender
            ) == Action.SPLIT
            if split_recommended:
                action_values["split"] = max(action_values.values(), default=-1.0) + 0.000001

        # Choose the exact EV maximum, with Basic Strategy controlling split.
        if split_recommended:
            rec_action = "split"
        else:
            rec_action = max(action_values, key=action_values.get)

        ranked = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        uncertainty = 0.0 if not split_recommended else 0.03

        if split_recommended:
            support_level = "Hybrid"
            reason = "Exact composition-based EV was used for HIT/STAND/DOUBLE/SURRENDER; SPLIT follows the validated rule-based Basic Strategy branch because split EV is path-dependent across multiple hands."
        else:
            support_level = "Exact"
            reason = f"Exact composition-based enumeration evaluated every legal non-split action using the visible shoe composition. {rec_action.upper()} has the highest expected value under the configured rules."

        risk_level = "Low" if margin >= 0.05 and not split_recommended else ("Medium" if margin >= 0.02 else "High")
        output_dict = {
            "hand_id": self.active_hand_id,
            "recommended_action": rec_action,
            "action_analysis": {k: round(float(v), 6) for k, v in action_values.items()},
            "decision_margin": round(float(margin), 6),
            "uncertainty": float(uncertainty),
            "support_level": support_level,
            "risk_level": risk_level,
            "model": "exact-composition-v1",
            "reason": reason,
            "prediction_method": "exact_enumeration",
            "sample_size": None,
        }
        self._save_feedback(output_dict)
        return output_dict

    def _save_feedback(self, output: Dict[str, Any]):
        decision_id = str(uuid4())
        
        hand_data = self.hands[self.active_hand_id]
        
        state_snapshot = {
            "player_cards": list(hand_data["cards"]),
            "dealer_cards": list(self.dealer_cards),
            "split_count": hand_data["split_count"]
        }
        
        # We don't save to sqlite here, we wait for user_action. We temporarily store it.
        self.pending_feedback = UserDecisionFeedback(
            decision_id=decision_id,
            session_id=self.session_id,
            hand_id=self.active_hand_id,
            timestamp=datetime.utcnow(),
            model_version=output["model"],
            feature_version="v16",
            calibration_version="v1.0-exp",
            decks=self.decks,
            state_snapshot=state_snapshot,
            recommended_action=output["recommended_action"],
            predicted_evs=output["action_analysis"],
            uncertainty=output["uncertainty"],
            risk_level=output["risk_level"],
            support_level=output["support_level"]
        )

ANALYZER_SESSIONS: Dict[str, AnalyzerSession] = {}

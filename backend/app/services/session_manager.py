from typing import Dict, Any, List, Optional
from pathlib import Path
from uuid import uuid4

from blackjack.cards.deck import Shoe
from blackjack.cards.hand import Hand
from blackjack.cards.card import Card
from blackjack.engine.state import PlayerHand, HandContext
from blackjack.rules.rules import BlackjackRules
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.rules.payout import PayoutCalculator
from blackjack.strategies.ml import MLStrategy
from blackjack.strategies.base import Action
from backend.app.schemas.decision import DecisionRecord
import time

class GameSession:
    def __init__(self, rules: BlackjackRules = None):
        self.session_id = str(uuid4())
        self.rules = rules or BlackjackRules()
        self.shoe = Shoe(num_decks=self.rules.num_decks)
        model_dir = Path("models/xgboost/v13_final")
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parents[3] / "models" / "xgboost" / "v13_final"
        self.ml_strategy = MLStrategy(model_dir=model_dir, rules=self.rules)
        self.calc = LegalActionsCalculator(self.rules)
        self.payout = PayoutCalculator(self.rules)
        
        self.player_hands: List[PlayerHand] = []
        self.dealer_hand = Hand()
        self.current_hand_idx = 0
        self.status = "idle" # idle, decision_required, complete
        self.bet = 1.0
        
        self.decision_history: List[DecisionRecord] = []
        self.pending_decision: Optional[DecisionRecord] = None
        
        # Start the first round automatically
        self.start_round()
        
    def start_round(self):
        # Reset
        self.player_hands = [PlayerHand(hand=Hand(), context=HandContext(bet=self.bet, original_bet=self.bet))]
        self.dealer_hand = Hand()
        self.current_hand_idx = 0
        self.status = "playing"
        
        # Deal
        self.player_hands[0].hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal())
        self.player_hands[0].hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal(hidden=True))
        
        if self.dealer_hand.is_blackjack and self.rules.peek:
            self.shoe.reveal_hidden()
            self.status = "complete"
            return
            
        if self.player_hands[0].hand.is_blackjack:
            self.shoe.reveal_hidden()
            self.status = "complete"
            return
            
        self.status = "decision_required"
        
    def get_public_state(self) -> Dict[str, Any]:
        if not self.player_hands:
            return {"status": "idle"}
            
        safe_idx = min(self.current_hand_idx, len(self.player_hands) - 1)
        ph = self.player_hands[safe_idx]
        dealer_upcard = self.dealer_hand.cards[0]
        
        legal_actions = []
        if self.status == "decision_required":
            can_double = self.calc.can_double(ph.hand, dealer_upcard, ph.context)
            can_split = self.calc.can_split(ph.hand, dealer_upcard, ph.context)
            can_surrender = self.calc.can_surrender(ph.hand, dealer_upcard, ph.context)
            legal_actions = ["hit", "stand"]
            if can_double: legal_actions.append("double")
            if can_split: legal_actions.append("split")
            if can_surrender: legal_actions.append("surrender")
            
        return {
            "session_id": self.session_id,
            "player_hands": [[str(c) for c in h.hand.cards] for h in self.player_hands],
            "current_hand_idx": self.current_hand_idx,
            "dealer_upcard": str(dealer_upcard),
            "dealer_hand": [str(c) for c in self.dealer_hand.cards] if self.status == "complete" else [str(dealer_upcard), "hidden"],
            "player_total": ph.hand.value,
            "is_soft": any(c.is_ace for c in ph.hand.cards) and ph.hand.value <= 21,
            "legal_actions": legal_actions,
            "status": self.status,
            "history": [d.dict() for d in self.decision_history]
        }
        
    def _play_dealer(self):
        self.shoe.reveal_hidden()
        # Dealer hits soft 17 rule
        while self.dealer_hand.value < 17 or (self.dealer_hand.value == 17 and self.rules.hit_soft_17 and any(c.is_ace for c in self.dealer_hand.cards)):
            self.dealer_hand.add_card(self.shoe.deal())
            
        self.status = "complete"
        
        # Calculate rewards and patch them to the decisions
        for idx, ph in enumerate(self.player_hands):
            pv = ph.hand.value
            dv = self.dealer_hand.value
            
            if ph.surrendered:
                reward = -0.5 * ph.context.original_bet
            elif ph.hand.is_bust:
                reward = -1.0 * ph.context.bet
            elif dv > 21:
                reward = 1.0 * ph.context.bet
            elif pv > dv:
                reward = 1.0 * ph.context.bet
            elif dv > pv:
                reward = -1.0 * ph.context.bet
            else:
                reward = 0.0
                
            # If player got a blackjack
            if ph.hand.is_blackjack and not self.dealer_hand.is_blackjack:
                reward = self.rules.blackjack_payout * ph.context.bet
                
            # Naively attach reward to all decisions made for this hand (this is approximate for splits)
            # A more robust solution maps decisions to hands.
            for dec in self.decision_history:
                if dec.reward is None:
                    dec.reward = reward
                    dec.resulting_cards = [str(c) for c in ph.hand.cards]
                    # Phase 18: Update monitoring DB with rewards
                    from backend.app.monitoring.logger import log_decision_event
                    log_decision_event(dec.dict())

    def apply_action(self, action_str: str):
        if self.status != "decision_required":
            raise ValueError("No decision required")
            
        action_str = action_str.lower()
        
        # Log user action against the pending decision
        if self.pending_decision:
            self.pending_decision.user_action = action_str
            if self.pending_decision.action_values:
                rec_action = self.pending_decision.recommended_action
                u_ev = self.pending_decision.action_values.get(action_str, 0.0)
                r_ev = self.pending_decision.action_values.get(rec_action, 0.0)
                self.pending_decision.user_action_ev = u_ev
                self.pending_decision.ev_difference = r_ev - u_ev
            
            self.decision_history.append(self.pending_decision)
            
            # Phase 18: Log to monitoring DB
            from backend.app.monitoring.logger import log_decision_event
            log_decision_event(self.pending_decision.dict())
            
            self.pending_decision = None
            
        ph = self.player_hands[self.current_hand_idx]
        
        if action_str == "hit":
            ph.hand.add_card(self.shoe.deal())
            if ph.hand.is_bust:
                ph.is_complete = True
                self.current_hand_idx += 1
        elif action_str == "stand":
            ph.is_complete = True
            self.current_hand_idx += 1
        elif action_str == "double":
            ph.context.doubled = True
            ph.context.bet *= 2
            ph.hand.add_card(self.shoe.deal())
            ph.is_complete = True
            self.current_hand_idx += 1
        elif action_str == "split":
            # Add split logic
            ph.context.split_count += 1
            card2 = ph.hand.cards.pop()
            new_hand = PlayerHand(hand=Hand(), context=HandContext(bet=ph.context.bet, original_bet=ph.context.original_bet, split_count=ph.context.split_count))
            new_hand.hand.add_card(card2)
            self.player_hands.insert(self.current_hand_idx + 1, new_hand)
            # Deal 2nd card to current hand
            ph.hand.add_card(self.shoe.deal())
            
        elif action_str == "surrender":
            ph.surrendered = True
            ph.is_complete = True
            self.current_hand_idx += 1
            
        if self.current_hand_idx >= len(self.player_hands):
            self._play_dealer()
            
        # Ensure new split hand gets a second card if needed
        elif len(self.player_hands[self.current_hand_idx].hand.cards) == 1:
            self.player_hands[self.current_hand_idx].hand.add_card(self.shoe.deal())
            
    def get_recommendation(self) -> Dict[str, Any]:
        start_t = time.time()
        if self.status != "decision_required":
            return {"error": "No decision required"}
            
        ph = self.player_hands[self.current_hand_idx]
        dealer_upcard = self.dealer_hand.cards[0]
        
        can_double = self.calc.can_double(ph.hand, dealer_upcard, ph.context)
        can_split = self.calc.can_split(ph.hand, dealer_upcard, ph.context)
        can_surrender = self.calc.can_surrender(ph.hand, dealer_upcard, ph.context)
        
        # Override decide logic slightly to intercept EVs
        legal_actions = [Action.HIT, Action.STAND]
        if can_double: legal_actions.append(Action.DOUBLE)
        if can_split: legal_actions.append(Action.SPLIT)
        if can_surrender: legal_actions.append(Action.SURRENDER)
        
        from ml.features.state import GameState
        state = GameState(
            player_ranks=[c.rank.value for c in ph.hand.cards],
            dealer_upcard_rank=dealer_upcard.rank.value,
            shoe_total_cards=6*52,
            shoe_cards_remaining=4*52,
            running_count=0,
            rules=self.rules.__dict__
        )
        
        from ml.features.extractor import FeatureExtractor
        base_features = FeatureExtractor.to_vector(state)
        
        import xgboost as xgb
        import numpy as np
        
        X_batch = []
        action_mapping = []
        supported_actions = []
        
        canonical_key = self.ml_strategy._get_canonical_key(state)
        state_support = self.ml_strategy.support_map.get(canonical_key, {})
        
        for action in legal_actions:
            action_str = action.name.lower()
            if state_support.get(action_str, 0) >= 100:
                supported_actions.append(action)
                
        fallback_used = False
        if not supported_actions:
            fallback_used = True
            from blackjack.strategies.basic import BasicStrategy
            rec = BasicStrategy().decide(ph.hand, dealer_upcard, can_double, can_split, can_surrender)
            pt = ph.hand.value
            dup = dealer_upcard.rank.value
            
            output_dict = {
                "recommended_action": rec.name.lower(),
                "fallback_used": True,
                "reason": f"This state has insufficient model support. Basic Strategy fallback was used.",
                "action_values": {},
                "support_status": "FALLBACK",
                "model_version": self.ml_strategy.name,
                "prediction_interval": None,
                "uncertainty_score": None,
                "risk_level": "UNSUPPORTED_DECISION",
                "calibration_version": "v1.0-exp",
                "uncertainty_method": "none",
                "decision_margin": 0.0,
                "decision_strength": "Basic Strategy Fallback"
            }
            
            self.pending_decision = DecisionRecord.create(
                session_id=self.session_id,
                state_summary={"player_total": pt, "dealer_upcard": str(dealer_upcard), "is_soft": ph.hand.is_soft, "is_pair": ph.hand.is_pair},
                ml_output=output_dict,
                latency_ms=(time.time() - start_t) * 1000
            )
            return output_dict
            
        for action in supported_actions:
            action_str = action.name.lower()
            if action_str in self.ml_strategy.action_encoding:
                encoded = self.ml_strategy.action_encoding[action_str]
                X_batch.append(base_features + encoded)
                action_mapping.append(action)
                
        dmatrix = xgb.DMatrix(np.array(X_batch, dtype=np.float32))
        preds = self.ml_strategy.model.predict(dmatrix)
        
        best_idx = int(np.argmax(preds))
        rec_action = action_mapping[best_idx].name.lower()
        
        action_values = {a.name.lower(): float(p) for a, p in zip(action_mapping, preds)}
        
        # Build explanation
        pt = ph.hand.value
        dup = dealer_upcard.rank.value
        
        strength = "N/A"
        margin = 0.0
        if len(action_mapping) >= 2:
            sorted_evs = sorted(preds, reverse=True)
            margin = sorted_evs[0] - sorted_evs[1]
            if margin < 0.05:
                strength = "Close Decision"
                reason = f"Your total is {pt} against a dealer {dup}. {rec_action.upper()} and the second-best action have similar estimated values. The model's advantage for {rec_action.upper()} is small."
            else:
                strength = "Strong Preference" if margin >= 0.15 else "Moderate Preference"
                reason = f"Your total is {pt} against a dealer {dup}. The model estimates {rec_action.upper()} has the highest expected value among the available actions."
        else:
            reason = f"Your total is {pt} against a dealer {dup}. {rec_action.upper()} is the only supported/legal action."
        
        # Phase 16: Uncertainty Fields (Quantile Regression)
        # Query the uncertainty models if they exist
        prediction_interval = [preds[best_idx] - 0.05, preds[best_idx] + 0.05]
        uncertainty_score = 0.05
        uncertainty_method = "fallback-static"
        
        if hasattr(self.ml_strategy, "q10_model") and self.ml_strategy.q10_model is not None:
            # Reconstruct the optimal action's feature vector
            best_action_features = np.array([X_batch[best_idx]], dtype=np.float32)
            q10_pred = self.ml_strategy.q10_model.predict(best_action_features)[0]
            q90_pred = self.ml_strategy.q90_model.predict(best_action_features)[0]
            
            # The model predicted error.
            # True_EV = Pred_EV - Error => Error = Pred_EV - True_EV
            # If Q10(Error) < Error < Q90(Error)
            # Pred - Q90 < True < Pred - Q10
            # Wait, no, we modeled absolute error or signed error?
            # In phase16_train_uncertainty.py, y = Pred_EV - True_EV
            # Therefore True = Pred - y. So Interval is [Pred - Q90, Pred - Q10].
            # Let's ensure Q10 is lower bound.
            lower_bound = float(preds[best_idx] - q90_pred)
            upper_bound = float(preds[best_idx] - q10_pred)
            
            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound
                
            prediction_interval = [lower_bound, upper_bound]
            uncertainty_score = float((upper_bound - lower_bound) / 2.0)
            uncertainty_method = "quantile-regression"
            
        if uncertainty_score > 0.08:
            risk_level = "UNCERTAIN_DECISION"
        elif margin < 0.05:
            risk_level = "CLOSE_DECISION"
        else:
            risk_level = "STRONG_DECISION"
            
        output_dict = {
            "recommended_action": rec_action,
            "fallback_used": False,
            "reason": reason,
            "action_values": action_values,
            "support_status": "HIGH_SUPPORT",
            "model_version": self.ml_strategy.name,
            "prediction_interval": [float(prediction_interval[0]), float(prediction_interval[1])],
            "uncertainty_score": float(uncertainty_score),
            "risk_level": risk_level,
            "calibration_version": "v1.0-exp",
            "uncertainty_method": uncertainty_method,
            "decision_margin": float(margin),
            "decision_strength": strength
        }
        
        self.pending_decision = DecisionRecord.create(
            session_id=self.session_id,
            state_summary={"player_total": pt, "dealer_upcard": str(dealer_upcard), "is_soft": ph.hand.is_soft, "is_pair": ph.hand.is_pair},
            ml_output=output_dict,
            latency_ms=(time.time() - start_t) * 1000
        )
        
        return output_dict

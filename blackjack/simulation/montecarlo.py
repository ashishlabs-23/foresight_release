"""
blackjack/simulation/montecarlo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Monte Carlo Expected Value Engine.
Evaluates the EV of all legal actions from an arbitrary game state.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Sequence

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.deck import Shoe
from blackjack.cards.hand import Hand
from blackjack.engine.game import GameEngine, HandResult
from blackjack.engine.outcomes import HandOutcome
from blackjack.engine.state import HandContext, PlayerHand
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.rules.payout import PayoutCalculator
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action, BaseStrategy
from blackjack.strategies.basic import BasicStrategy

logger = logging.getLogger(__name__)


@dataclass
class MCActionStats:
    """Statistics for a single action after MC rollouts."""
    
    action: Action
    ev: float
    simulations_run: int
    standard_error: float


@dataclass
class MCEVResult:
    """The result of evaluating a state using Monte Carlo simulation."""
    
    player_cards: list[str]
    dealer_upcard: str
    action_stats: list[MCActionStats]
    recommended_action: Action | None
    
    def get_best_action(self) -> Action | None:
        if not self.action_stats:
            return None
        return max(self.action_stats, key=lambda s: s.ev).action

    def summary(self) -> str:
        if not self.action_stats:
            return "No legal actions evaluated."
            
        lines = [f"Monte Carlo EV Evaluation"]
        lines.append(f"Player: {', '.join(self.player_cards)} | Dealer: {self.dealer_upcard}")
        
        # Sort by EV descending
        sorted_stats = sorted(self.action_stats, key=lambda s: s.ev, reverse=True)
        for stat in sorted_stats:
            lines.append(
                f"  {stat.action.value.upper():<9}: "
                f"EV = {stat.ev:+.4f} \u00b1 {stat.standard_error:.4f} "
                f"({stat.simulations_run} sims)"
            )
            
        best = self.get_best_action()
        if best:
            lines.append(f"\nRecommendation: {best.value.upper()}")
            
        return "\n".join(lines)


class ScriptedInitialActionStrategy(BaseStrategy):
    """A proxy strategy that returns a specific FIRST action, then falls back to a base strategy."""
    
    def __init__(self, initial_action: Action, fallback: BaseStrategy):
        self._initial_action = initial_action
        self._fallback = fallback
        self._action_taken = False
        
    @property
    def name(self) -> str:
        return f"MC_Rollout_{self._initial_action.value}"

    def decide(self, player_hand, dealer_upcard, **kwargs) -> Action:
        # Fallback to the underlying strategy once the scripted action is consumed.
        # However, due to GameEngine's architecture, if we split, the new hands
        # might ask for an action. The *very first* decision made by this strategy
        # instance is the scripted one.
        if not self._action_taken:
            self._action_taken = True
            return self._initial_action
            
        return self._fallback.decide(player_hand, dealer_upcard, **kwargs)


class MCEngine:
    """Evaluates the Expected Value of actions using Monte Carlo rollouts."""
    
    def __init__(
        self,
        rules: BlackjackRules,
        num_simulations: int = 1000,
        rollout_strategy: BaseStrategy | None = None,
        seed: int | None = None,
    ) -> None:
        self._rules = rules
        self._num_simulations = num_simulations
        self._rollout_strategy = rollout_strategy or BasicStrategy()
        self._rng = random.Random(seed)
        self._calc = LegalActionsCalculator(rules)
        self._payout = PayoutCalculator(rules)

    def evaluate_state(
        self,
        player_cards: list[str],
        dealer_upcard: str,
        observed_cards: list[str] | None = None,
        num_decks: int = 6,
    ) -> MCEVResult:
        """
        Evaluate all legal actions from the given state.
        
        Parameters
        ----------
        player_cards : list[str]
            Cards the player currently holds (e.g. ["TH", "6D"]).
        dealer_upcard : str
            The dealer's visible upcard (e.g. "7C").
        observed_cards : list[str], optional
            Other cards known to have been removed from the shoe.
            If None, only the player's cards and dealer's upcard are removed.
        num_decks : int
            Number of decks in the shoe.
            
        Returns
        -------
        MCEVResult containing EVs for each legal action.
        """
        if observed_cards is None:
            observed_cards = []
            
        # All cards we know are out of the shoe
        all_observed = list(player_cards) + [dealer_upcard] + observed_cards
        
        # Build Hand to determine legal actions
        hand = Hand()
        for c_str in player_cards:
            hand.add_card(self._parse_card(c_str))
            
        dupcard = self._parse_card(dealer_upcard)
        
        # Determine context - we assume first action, bet = 1.0, not from split.
        # For a completely generalized MCEngine, these would be parameters.
        ctx = HandContext(bet=1.0, original_bet=1.0, is_first_action=len(player_cards) == 2)
        
        legal_actions = self._calc.get_legal_actions(hand, dupcard, ctx)
        
        # Evaluate each legal action
        stats: list[MCActionStats] = []
        for action in legal_actions:
            action_stat = self._evaluate_action(action, player_cards, dealer_upcard, all_observed, num_decks, ctx)
            stats.append(action_stat)
            
        res = MCEVResult(
            player_cards=player_cards,
            dealer_upcard=dealer_upcard,
            action_stats=stats,
            recommended_action=None,
        )
        res.recommended_action = res.get_best_action()
        return res

    def _evaluate_action(
        self,
        action: Action,
        player_cards: list[str],
        dealer_upcard: str,
        all_observed: list[str],
        num_decks: int,
        ctx: HandContext,
    ) -> MCActionStats:
        """Run N rollouts for a specific initial action."""
        
        total_reward = 0.0
        reward_sq_sum = 0.0
        
        for i in range(self._num_simulations):
            # 1. Generate synthetic shoe
            shoe = Shoe.create_synthetic(num_decks, all_observed, seed=self._rng.randint(0, 999999))
            
            # 2. Setup initial game state
            strategy = ScriptedInitialActionStrategy(action, self._rollout_strategy)
            engine = GameEngine(shoe, self._rules, strategy)
            
            # Since GameEngine.play_round() deals a fresh hand, we must bypass it 
            # and set up the state manually to resume from the given state.
            
            # Recreate player hand
            primary_hand = Hand()
            for c_str in player_cards:
                primary_hand.add_card(self._parse_card(c_str))
                
            # Copy the context to avoid mutating the original
            run_ctx = HandContext(
                bet=ctx.bet,
                original_bet=ctx.original_bet,
                is_first_action=ctx.is_first_action,
                doubled=ctx.doubled,
                split_count=ctx.split_count,
                from_split_aces=ctx.from_split_aces,
            )
            ph = PlayerHand(hand=primary_hand, context=run_ctx)
            
            # Recreate dealer hand
            d_hand = Hand()
            d_hand.add_card(self._parse_card(dealer_upcard))
            # Deal the hidden hole card
            d_hand.add_card(shoe.deal(hidden=True))
            
            # Resolve US peek if applicable (this logic mirrors GameEngine)
            dealer_bj = d_hand.is_blackjack
            player_bj = ph.hand.is_blackjack
            
            net_payout = 0.0
            round_over = False
            
            if self._rules.peek and dealer_bj:
                shoe.reveal_hidden()
                if player_bj:
                    net_payout = 0.0
                else:
                    net_payout = self._payout.net(HandOutcome.LOSS, bet=run_ctx.bet)
                round_over = True
            elif player_bj:
                shoe.reveal_hidden()
                net_payout = self._payout.net(HandOutcome.BLACKJACK, original_bet=run_ctx.original_bet)
                round_over = True
                
            if not round_over:
                # Play player hands
                player_hands = [ph]
                hand_idx = 0
                while hand_idx < len(player_hands):
                    current_ph = player_hands[hand_idx]
                    # Note: engine._play_player_hand uses engine._strategy which is our Scripted Strategy
                    engine._play_player_hand(current_ph, d_hand.cards[0], player_hands, hand_idx, run_ctx.original_bet)
                    hand_idx += 1
                    
                # Play dealer hand
                shoe.reveal_hidden()
                any_live = any(not h.hand.is_bust and not h.surrendered for h in player_hands)
                if any_live:
                    engine._play_dealer(d_hand)
                    
                # Calculate payouts
                for current_ph in player_hands:
                    outcome = engine._determine_outcome(current_ph, d_hand)
                    net = self._payout.net(
                        outcome,
                        bet=current_ph.context.bet,
                        original_bet=current_ph.context.original_bet,
                    )
                    net_payout += net
                    
            total_reward += net_payout
            reward_sq_sum += net_payout * net_payout
            
        # Calculate stats
        mean_ev = total_reward / self._num_simulations
        variance = (reward_sq_sum / self._num_simulations) - (mean_ev * mean_ev)
        
        # Fix floating point inaccuracies that can cause negative zero variance
        if variance < 0:
            variance = 0.0
            
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(self._num_simulations)
        
        return MCActionStats(
            action=action,
            ev=mean_ev,
            simulations_run=self._num_simulations,
            standard_error=std_err,
        )

    @staticmethod
    def _parse_card(card_str: str) -> Card:
        rank_char, suit_char = card_str[0], card_str[1]
        
        rank = next(r for r in Rank if r.value == rank_char)
        suit = next(s for s in Suit if s.value == suit_char)
        
        return Card(rank, suit)

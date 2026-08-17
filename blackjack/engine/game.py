"""
blackjack.engine.game
~~~~~~~~~~~~~~~~~~~~~
GameEngine — orchestrates a full round of Blackjack including splits.

Phase 2 changes vs Phase 1
---------------------------
* :meth:`play_round` — NEW primary API; returns a :class:`RoundResult` that
  can contain multiple player hands (from splits).
* Full split support: second card removal, new hand queuing, one-card-only
  rule for split Aces, and max-splits enforcement via
  :class:`~blackjack.rules.legal_actions.LegalActionsCalculator`.
* :class:`~blackjack.rules.payout.PayoutCalculator` handles all net-payout
  arithmetic (no magic numbers in the engine).
* US-style peek (hole-card check) before player acts.
* :meth:`play_hand` is retained as a backwards-compatible wrapper that
  returns a single :class:`HandResult` (aggregated if splits occurred).

Dependency rule
---------------
This module imports from blackjack.cards, blackjack.engine.state,
blackjack.rules, and blackjack.strategies ONLY.
No ML, backend, or HTTP imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from blackjack.cards.card import Card
from blackjack.cards.deck import Shoe
from blackjack.cards.hand import Hand
from blackjack.engine.state import HandContext, PlayerHand, RoundResult
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action, BaseStrategy

logger = logging.getLogger(__name__)

from blackjack.engine.outcomes import HandOutcome, HandResult, PlayerActionRecord


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------


class GameEngine:
    """Orchestrates a Blackjack round, including full split support.

    Parameters
    ----------
    shoe     : Shoe to draw cards from. Shared across multiple rounds.
    rules    : Rule configuration.
    strategy : Strategy to use for player decisions.

    Example
    -------
    >>> from blackjack.cards.deck import Shoe
    >>> from blackjack.rules.rules import BlackjackRules
    >>> from blackjack.strategies.basic import BasicStrategy
    >>> engine = GameEngine(Shoe(seed=1), BlackjackRules.standard(), BasicStrategy())
    >>> result = engine.play_round(bet=1.0)
    >>> print(result)
    """

    def __init__(
        self,
        shoe: Shoe,
        rules: BlackjackRules,
        strategy: BaseStrategy,
    ) -> None:
        self._shoe = shoe
        self._rules = rules
        self._strategy = strategy
        
        from blackjack.rules.legal_actions import LegalActionsCalculator
        from blackjack.rules.payout import PayoutCalculator
        self._calc = LegalActionsCalculator(rules)
        self._payout = PayoutCalculator(rules)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def play_round(self, bet: float = 1.0) -> RoundResult:
        """Deal and play one complete round, returning all hand outcomes.

        Supports splits: the returned :class:`RoundResult` may contain
        multiple :class:`HandResult` objects if the player split.

        Parameters
        ----------
        bet : Initial bet per hand in arbitrary units (default 1.0).
        """
        # -------- Initial deal ----------------------------------------
        primary_ctx = HandContext(bet=bet, original_bet=bet)
        primary = PlayerHand(hand=Hand(), context=primary_ctx)
        dealer_hand = Hand()

        primary.hand.add_card(self._shoe.deal())
        dealer_hand.add_card(self._shoe.deal())
        primary.hand.add_card(self._shoe.deal())
        dealer_hand.add_card(self._shoe.deal(hidden=True))  # hole card

        dealer_upcard: Card = dealer_hand.cards[0]
        dealer_bj = dealer_hand.is_blackjack
        player_bj = primary.hand.is_blackjack

        logger.debug(
            f"Round started - player={primary.hand}, dealer_upcard={dealer_upcard}, "
            f"dealer_bj={dealer_bj}, player_bj={player_bj}"
        )

        # -------- US peek: resolve immediately on dealer blackjack ------
        if self._rules.peek and dealer_bj:
            self._shoe.reveal_hidden()
            return self._resolve_dealer_blackjack(primary, dealer_hand, bet, player_bj)

        # -------- Player natural blackjack ------------------------------
        if player_bj:
            self._shoe.reveal_hidden()
            payout = self._payout.net(HandOutcome.BLACKJACK, original_bet=bet)
            result = HandResult(
                outcome=HandOutcome.BLACKJACK,
                player_value=21,
                dealer_value=dealer_hand.value,
                payout=payout,
                player_cards=list(primary.hand.cards),
                dealer_cards=list(dealer_hand.cards),
                bet=bet,
                original_bet=bet,
            )
            primary.is_complete = True
            return RoundResult(
                player_hands=[primary],
                dealer_hand=dealer_hand,
                dealer_cards=list(dealer_hand.cards),
                hand_results=[result],
                total_net_payout=payout,
                dealer_had_blackjack=False,
            )

        # -------- Player action phase -----------------------------------
        player_hands: list[PlayerHand] = [primary]
        hand_idx = 0

        while hand_idx < len(player_hands):
            ph = player_hands[hand_idx]

            # For a newly queued split hand (1 card), deal the second card
            if len(ph.hand) == 1 and ph.context.split_count > 0:
                ph.hand.add_card(self._shoe.deal())

                # Split Aces rule: only one card each, no further actions
                if (
                    ph.hand.cards[0].is_ace
                    and not self._rules.hit_split_aces
                ):
                    ph.is_complete = True
                    hand_idx += 1
                    continue

            # Play this hand to completion
            self._play_player_hand(ph, dealer_upcard, player_hands, hand_idx, bet)
            hand_idx += 1

        # -------- Dealer phase ------------------------------------------
        # Reveal hole card first
        self._shoe.reveal_hidden()
        
        # Only deal if at least one player hand is alive (not bust/surrendered)
        any_live = any(
            not ph.hand.is_bust and not ph.surrendered
            for ph in player_hands
        )
        if any_live:
            self._play_dealer(dealer_hand)

        # -------- Payout calculation ------------------------------------
        hand_results: list[HandResult] = []
        total_payout = 0.0

        for ph in player_hands:
            outcome = self._determine_outcome(ph, dealer_hand)
            net = self._payout.net(
                outcome,
                bet=ph.context.bet,
                original_bet=ph.context.original_bet,
            )
            hr = HandResult(
                outcome=outcome,
                player_value=ph.hand.value,
                dealer_value=dealer_hand.value,
                payout=net,
                player_cards=list(ph.hand.cards),
                dealer_cards=list(dealer_hand.cards),
                bet=ph.context.bet,
                original_bet=ph.context.original_bet,
                doubled=ph.context.doubled,
                split_count=ph.context.split_count,
                history=ph.history,
            )
            hand_results.append(hr)
            total_payout += net

        logger.debug(
            f"Round complete - num_hands={len(hand_results)}, total_payout={total_payout}, "
            f"dealer_value={dealer_hand.value}"
        )

        return RoundResult(
            player_hands=player_hands,
            dealer_hand=dealer_hand,
            dealer_cards=list(dealer_hand.cards),
            hand_results=hand_results,
            total_net_payout=total_payout,
            dealer_had_blackjack=False,
        )

    def play_hand(self) -> HandResult:
        """Backwards-compatible wrapper: play one round at unit bet.

        If the round produces multiple hands (from splits), returns an
        aggregate :class:`HandResult` based on total net payout.
        """
        result = self.play_round(bet=1.0)
        if len(result.hand_results) == 1:
            return result.hand_results[0]

        # Aggregate multiple split hands into one result for the Simulator
        total = result.total_net_payout
        if total > 0:
            agg_outcome = HandOutcome.WIN
        elif total < 0:
            agg_outcome = HandOutcome.LOSS
        else:
            agg_outcome = HandOutcome.PUSH

        primary = result.hand_results[0]
        return HandResult(
            outcome=agg_outcome,
            player_value=primary.player_value,
            dealer_value=primary.dealer_value,
            payout=total,
            player_cards=primary.player_cards,
            dealer_cards=primary.dealer_cards,
            bet=1.0,
            original_bet=1.0,
            split_count=result.player_hands[0].context.split_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _play_player_hand(
        self,
        ph: PlayerHand,
        dealer_upcard: Card,
        player_hands: list[PlayerHand],
        hand_idx: int,
        original_bet: float,
    ) -> None:
        """Iterate player actions until the hand is complete."""
        while ph.is_active:
            legal = self._calc.get_legal_actions(ph.hand, dealer_upcard, ph.context)
            action = self._strategy.decide_from_actions(ph.hand, dealer_upcard, legal)

            # Record trajectory for ML / simulator stats
            record = PlayerActionRecord(
                player_hand_value=ph.hand.value,
                player_is_soft=ph.hand.is_soft,
                dealer_upcard_value=dealer_upcard.value,
                legal_actions=frozenset(a.value for a in legal),
                action_taken=action.value,
            )
            ph.history.append(record)

            logger.debug(f"Player action - action={action.value}, hand={ph.hand}")

            match action:
                case Action.STAND:
                    ph.is_complete = True

                case Action.HIT:
                    ph.hand.add_card(self._shoe.deal())
                    ph.context.mark_acted()
                    # Bust → hand is over (is_active will return False)

                case Action.DOUBLE:
                    ph.context.mark_doubled()   # doubles bet, sets is_first_action=False
                    ph.hand.add_card(self._shoe.deal())
                    ph.is_complete = True        # exactly one card on double

                case Action.SPLIT:
                    self._execute_split(ph, player_hands, hand_idx, original_bet)
                    # After split, the current hand has a fresh 2-card state
                    # and is_first_action is reset; loop continues on this hand.

                case Action.SURRENDER:
                    ph.surrendered = True
                    ph.is_complete = True

    def _execute_split(
        self,
        ph: PlayerHand,
        player_hands: list[PlayerHand],
        hand_idx: int,
        original_bet: float,
    ) -> None:
        """Remove the second card, deal a replacement, create a new queued hand."""
        new_hand = ph.hand.split()   # removes second card, returns it as a new 1-card Hand

        new_split_count = ph.context.split_count + 1
        from_aces = ph.hand.cards[0].is_ace

        # Deal replacement card to the current hand
        ph.hand.add_card(self._shoe.deal())

        # Update current hand context (fresh first-action state for the new 2-card hand)
        ph.context.split_count = new_split_count
        ph.context.from_split_aces = from_aces
        ph.context.is_first_action = True
        ph.context.doubled = False
        # Keep original_bet and bet intact for this hand

        # Queue the new split hand (it has 1 card; second card dealt when processed)
        new_ctx = HandContext(
            bet=original_bet,
            original_bet=original_bet,
            is_first_action=True,
            split_count=new_split_count,
            from_split_aces=new_hand.cards[0].is_ace,
        )
        player_hands.insert(hand_idx + 1, PlayerHand(hand=new_hand, context=new_ctx))

        # Split Aces: one card only, mark current hand done after the split card was dealt
        if from_aces and not self._rules.hit_split_aces:
            ph.is_complete = True

    def _play_dealer(self, dealer_hand: Hand) -> None:
        """Draw cards for the dealer according to the configured stand rule."""
        while self._rules.dealer_must_hit(dealer_hand.value, dealer_hand.is_soft):
            dealer_hand.add_card(self._shoe.deal())

    @staticmethod
    def _determine_outcome(ph: PlayerHand, dealer_hand: Hand) -> HandOutcome:
        """Compare player and dealer hands and return the hand outcome."""
        if ph.surrendered:
            return HandOutcome.SURRENDER
        if ph.hand.is_bust:
            return HandOutcome.LOSS
        if dealer_hand.is_bust:
            return HandOutcome.WIN
        pv, dv = ph.hand.value, dealer_hand.value
        if pv > dv:
            return HandOutcome.WIN
        if pv < dv:
            return HandOutcome.LOSS
        return HandOutcome.PUSH

    def _resolve_dealer_blackjack(
        self,
        primary: PlayerHand,
        dealer_hand: Hand,
        bet: float,
        player_bj: bool,
    ) -> RoundResult:
        """Resolve immediately when dealer has blackjack (US peek rule)."""
        if player_bj:
            outcome, net = HandOutcome.PUSH, 0.0
        else:
            outcome = HandOutcome.LOSS
            net = self._payout.net(HandOutcome.LOSS, bet=bet)

        result = HandResult(
            outcome=outcome,
            player_value=primary.hand.value,
            dealer_value=dealer_hand.value,
            payout=net,
            player_cards=list(primary.hand.cards),
            dealer_cards=list(dealer_hand.cards),
            bet=bet,
            original_bet=bet,
        )
        primary.is_complete = True
        return RoundResult(
            player_hands=[primary],
            dealer_hand=dealer_hand,
            dealer_cards=list(dealer_hand.cards),
            hand_results=[result],
            total_net_payout=net,
            dealer_had_blackjack=True,
        )

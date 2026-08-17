"""
blackjack.rules.legal_actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LegalActionsCalculator — given a player hand state, return the exact set
of actions that are currently permitted.

This is the single authoritative source of truth for action legality.
The game engine, strategy adapters, and tests all use this class.

Dependency rule: imports only from blackjack.cards and blackjack.rules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from blackjack.cards.card import Card
from blackjack.cards.hand import Hand
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action

if TYPE_CHECKING:
    from blackjack.engine.state import HandContext


class LegalActionsCalculator:
    """Computes the set of legal player actions for a given game state.

    Parameters
    ----------
    rules : The active rule configuration.

    Example
    -------
    >>> from blackjack.cards.card import Card, Rank, Suit
    >>> from blackjack.cards.hand import Hand
    >>> from blackjack.engine.state import HandContext
    >>> from blackjack.rules.rules import BlackjackRules
    >>> calc = LegalActionsCalculator(BlackjackRules.standard())
    >>> hand = Hand.from_cards(Card(Rank.EIGHT, Suit.SPADES), Card(Rank.EIGHT, Suit.HEARTS))
    >>> ctx  = HandContext(is_first_action=True, split_count=0)
    >>> upcard = Card(Rank.SIX, Suit.DIAMONDS)
    >>> Action.SPLIT in calc.get_legal_actions(hand, upcard, ctx)
    True
    """

    # HIT and STAND are always available (unless 21+ but engine won't ask then)
    _BASE = frozenset({Action.HIT, Action.STAND})

    def __init__(self, rules: BlackjackRules) -> None:
        self._rules = rules

    def get_legal_actions(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        context: HandContext,
    ) -> frozenset[Action]:
        """Return the frozenset of legal actions for the current state.

        HIT and STAND are always included (the hand is assumed live).
        DOUBLE, SPLIT, and SURRENDER are added based on rules and context.

        Parameters
        ----------
        player_hand   : The player's current hand.
        dealer_upcard : The dealer's face-up card.
        context       : Betting / action context for this hand.
        """
        actions: set[Action] = set(self._BASE)
        rules = self._rules

        # -- DOUBLE --
        # Requires: first action, 2 cards, double_on restriction satisfied
        if (
            context.is_first_action
            and len(player_hand) == 2
            and rules.can_double_on_hand(
                player_hand.value,
                player_hand.is_soft,
                context.split_count,
            )
        ):
            actions.add(Action.DOUBLE)

        # -- SPLIT --
        # Requires: first action, 2 cards of same rank, within max_splits,
        #           not blocked by "no resplit aces" rule
        if (
            context.is_first_action
            and player_hand.is_pair
            and rules.can_split_on_context(
                context.split_count,
                context.from_split_aces,
            )
        ):
            actions.add(Action.SPLIT)

        # -- SURRENDER --
        # Late surrender: first action, 2 cards, not after a split (unless rules allow)
        if (
            context.is_first_action
            and len(player_hand) == 2
            and rules.can_surrender_in_context(context.split_count)
        ):
            actions.add(Action.SURRENDER)

        return frozenset(actions)

    # ------------------------------------------------------------------
    # Convenience: boolean probes (useful in tests)
    # ------------------------------------------------------------------

    def can_double(
        self, player_hand: Hand, dealer_upcard: Card, context: HandContext
    ) -> bool:
        return Action.DOUBLE in self.get_legal_actions(player_hand, dealer_upcard, context)

    def can_split(
        self, player_hand: Hand, dealer_upcard: Card, context: HandContext
    ) -> bool:
        return Action.SPLIT in self.get_legal_actions(player_hand, dealer_upcard, context)

    def can_surrender(
        self, player_hand: Hand, dealer_upcard: Card, context: HandContext
    ) -> bool:
        return Action.SURRENDER in self.get_legal_actions(player_hand, dealer_upcard, context)

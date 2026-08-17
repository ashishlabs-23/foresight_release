"""
blackjack.strategies.base
~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract Strategy interface — all strategy implementations must subclass this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from blackjack.cards.card import Card
from blackjack.cards.hand import Hand


class Action(str, Enum):
    """All possible player actions in Blackjack."""

    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


class BaseStrategy(ABC):
    """Abstract base class for all Blackjack strategy implementations.

    Implementing classes must define :meth:`decide` and :attr:`name`.

    The interface deliberately keeps context to the minimum needed
    for a strategy decision:
      - The player's current hand
      - The dealer's visible upcard
      - Which special actions (double, split, surrender) are currently legal

    Strategies must be stateless across hands — all state needed for
    card-counting or RL should be managed externally by the game engine.

    Phase 2 addition
    ----------------
    :meth:`decide_from_actions` accepts a ``frozenset[Action]`` of legal
    actions and has a default implementation that converts them to the
    bool flags expected by :meth:`decide`.  Subclasses may override
    ``decide_from_actions`` for richer, set-aware logic.
    """

    @abstractmethod
    def decide(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        can_double: bool = True,
        can_split: bool = True,
        can_surrender: bool = True,
    ) -> Action:
        """Return the recommended action for the current game state.

        Parameters
        ----------
        player_hand  : The player's current hand.
        dealer_upcard: The dealer's face-up card.
        can_double   : Whether doubling down is currently permitted.
        can_split    : Whether splitting is currently permitted.
        can_surrender: Whether surrendering is currently permitted.

        Returns
        -------
        Action
            One of HIT, STAND, DOUBLE, SPLIT, SURRENDER.
            Must only return DOUBLE/SPLIT/SURRENDER if the corresponding
            ``can_*`` flag is True.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier (used in reports and logging)."""
        ...

    def decide_from_actions(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        legal_actions: frozenset[Action],
    ) -> Action:
        """Return an action from the given set of legal actions.

        Default implementation: converts ``legal_actions`` to bool flags
        and delegates to :meth:`decide`, then validates the result.

        Subclasses may override this method to access the full action set
        directly (e.g., for RL policies that operate on action spaces).

        Parameters
        ----------
        player_hand   : The player's current hand.
        dealer_upcard : The dealer's face-up card.
        legal_actions : Frozenset of currently permitted Actions.

        Returns
        -------
        Action
            A member of ``legal_actions``. Falls back to STAND or HIT
            if the strategy returns an illegal action.
        """
        action = self.decide(
            player_hand=player_hand,
            dealer_upcard=dealer_upcard,
            can_double=Action.DOUBLE in legal_actions,
            can_split=Action.SPLIT in legal_actions,
            can_surrender=Action.SURRENDER in legal_actions,
        )
        # Safety: ensure the returned action is actually legal
        if action in legal_actions:
            return action
        if Action.STAND in legal_actions:
            return Action.STAND
        return Action.HIT

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

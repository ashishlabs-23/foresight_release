"""
blackjack.strategies.random_strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Random strategy — uniformly samples from the set of legal actions.

Used as a baseline for:
  - Comparing RL agents against a floor of random play
  - Stress-testing the game engine (all code paths are exercised)
  - Monte Carlo sanity checks (expected house edge should be very high)
"""
from __future__ import annotations

import random

from blackjack.cards.card import Card
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action, BaseStrategy


class RandomStrategy(BaseStrategy):
    """Selects uniformly at random from all currently legal actions.

    Parameters
    ----------
    seed : int | None
        Optional random seed for reproducible simulations.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def decide(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        can_double: bool = True,
        can_split: bool = True,
        can_surrender: bool = True,
    ) -> Action:
        """Return a uniformly random legal action."""
        legal: list[Action] = [Action.HIT, Action.STAND]

        if can_double:
            legal.append(Action.DOUBLE)
        if can_split and player_hand.can_split:
            legal.append(Action.SPLIT)
        if can_surrender:
            legal.append(Action.SURRENDER)

        return self._rng.choice(legal)

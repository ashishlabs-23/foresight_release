"""
blackjack.engine.outcomes
~~~~~~~~~~~~~~~~~~~~~~~~~
Hand outcome definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blackjack.cards.card import Card


class HandOutcome(str, Enum):
    """Result of a single hand for the player."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    BLACKJACK = "blackjack"
    SURRENDER = "surrender"


@dataclass
class PlayerActionRecord:
    """Records a single state transition / action taken by the player."""
    
    player_hand_value: int
    player_is_soft: bool
    dealer_upcard_value: int
    legal_actions: frozenset[str]
    action_taken: str


@dataclass
class HandResult:
    """Detailed outcome of one player hand.

    Attributes
    ----------
    outcome      : Win / Loss / Push / Blackjack / Surrender
    player_value : Final player hand value.
    dealer_value : Final dealer hand value.
    payout       : Net units won/lost for this hand (sign convention:
                   positive = player won money, negative = player lost).
    player_cards : Cards in the player's final hand.
    dealer_cards : All dealer cards (including hole card).
    bet          : Bet amount for this hand (doubled if player doubled).
    original_bet : Original bet before any doubling.
    doubled      : Whether the player doubled down.
    split_count  : How many splits produced this hand.
    history      : Sequence of actions taken during this hand.
    """

    outcome: HandOutcome
    player_value: int
    dealer_value: int
    payout: float
    player_cards: list[Card] = field(default_factory=list)
    dealer_cards: list[Card] = field(default_factory=list)
    bet: float = 1.0
    original_bet: float = 1.0
    doubled: bool = False
    split_count: int = 0
    history: list[PlayerActionRecord] = field(default_factory=list)

    def __str__(self) -> str:
        doubled_tag = " [DBL]" if self.doubled else ""
        return (
            f"{self.outcome.value.upper()}{doubled_tag} | "
            f"Player: {self.player_value} | "
            f"Dealer: {self.dealer_value} | "
            f"Payout: {self.payout:+.2f}"
        )

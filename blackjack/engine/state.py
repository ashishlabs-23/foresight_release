"""
blackjack.engine.state
~~~~~~~~~~~~~~~~~~~~~~
Per-round state objects for the Phase 2 multi-hand game engine.

These classes carry the mutable state needed to track a round that may
produce multiple player hands (via splitting).

Dependency rule: imports only from blackjack.cards and blackjack.engine.game
(HandOutcome, HandResult). No ML, backend, or HTTP imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from blackjack.cards.hand import Hand

if TYPE_CHECKING:
    from blackjack.cards.card import Card
    from blackjack.engine.outcomes import HandOutcome, HandResult, PlayerActionRecord


@dataclass
class HandContext:
    """Mutable betting / action context for one player hand within a round.

    Attributes
    ----------
    bet              : Current bet amount (in arbitrary units). Doubles on double-down.
    original_bet     : Bet at the start of this hand (before any double). Useful for
                       surrender payout (always half the ORIGINAL bet).
    is_first_action  : True until the player takes any action on this hand.
    split_count      : Total number of splits that have occurred in this round
                       (shared across the split tree so max_splits is respected).
    from_split_aces  : True if this hand was produced by splitting a pair of Aces.
    doubled          : True after the player doubles down on this hand.
    """

    bet: float = 1.0
    original_bet: float = 1.0
    is_first_action: bool = True
    split_count: int = 0
    from_split_aces: bool = False
    doubled: bool = False

    def mark_doubled(self) -> None:
        """Double the current bet and mark the hand as doubled."""
        self.bet *= 2
        self.doubled = True
        self.is_first_action = False

    def mark_acted(self) -> None:
        """Mark that the first action has been taken."""
        self.is_first_action = False


@dataclass
class PlayerHand:
    """One player hand during an active round.

    A round starts with a single PlayerHand. When the player splits, a second
    PlayerHand is inserted into the processing queue.

    Attributes
    ----------
    hand          : The underlying card collection.
    context       : Betting / action context.
    surrendered   : True if the player surrendered this hand.
    is_complete   : True when no more actions are taken on this hand.
    """

    hand: Hand = field(default_factory=Hand)
    context: HandContext = field(default_factory=HandContext)
    surrendered: bool = False
    is_complete: bool = False
    history: list[PlayerActionRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_bust(self) -> bool:
        return self.hand.is_bust

    @property
    def is_active(self) -> bool:
        """True while this hand is still in play (not done, bust, or surrendered)."""
        return not self.is_complete and not self.surrendered and not self.hand.is_bust


@dataclass
class RoundResult:
    """Complete outcome of one round (may contain multiple split hands).

    Attributes
    ----------
    player_hands     : All player hands played, in order.
    dealer_hand      : The dealer's final hand.
    dealer_cards     : Dealer cards as a list snapshot.
    hand_results     : Per-hand outcomes (one per player hand).
    total_net_payout : Algebraic sum of all hand payouts.
                       Positive = player profits overall.
    dealer_had_blackjack : Whether the dealer held a natural (affects Phase 3 insurance).
    """

    player_hands: list[PlayerHand]
    dealer_hand: Hand
    dealer_cards: list[Card]
    hand_results: list[HandResult]
    total_net_payout: float
    dealer_had_blackjack: bool = False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def num_player_hands(self) -> int:
        """Total player hands played (rounds + extra split hands)."""
        return len(self.player_hands)

    @property
    def primary_result(self) -> HandResult | None:
        """The outcome of the first (primary) player hand."""
        return self.hand_results[0] if self.hand_results else None

    def __str__(self) -> str:
        lines = [
            f"Round: {self.num_player_hands} hand(s) | "
            f"net payout: {self.total_net_payout:+.2f} | "
            f"dealer: {self.dealer_hand.value}"
        ]
        for i, hr in enumerate(self.hand_results):
            lines.append(f"  Hand {i + 1}: {hr}")
        return "\n".join(lines)

"""
blackjack.rules.rules
~~~~~~~~~~~~~~~~~~~~~
Configurable rule variants for Blackjack.

Supports common casino rule-sets:
- Vegas Strip (standard, 6D, S17, 3:2)
- Vegas Downtown (1D, H17, 3:2)
- Unfavourable (6D, H17, 6:5)

Phase 2 additions
-----------------
* ``allow_double``        — global double-down toggle.
* ``double_on``           — restricts which totals allow doubling.
* ``hit_split_aces``      — whether player may draw more than one card to split Aces.
* ``peek``                — US-style hole-card peek before player acts.
* ``surrender_after_split``— rarely offered; disabled by default.
* Helper query methods on the rules object so callers don't need to
  re-implement rule logic:
    - :meth:`can_double_on_hand`
    - :meth:`can_split_on_context`
    - :meth:`can_surrender_in_context`
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DealerStandRule(str, Enum):
    """Whether the dealer stands or hits on soft 17."""

    STAND_SOFT_17 = "S17"  # Player-favourable
    HIT_SOFT_17 = "H17"    # House-favourable


class BlackjackPayout(str, Enum):
    """Payout multiplier for a natural blackjack."""

    THREE_TO_TWO = "3:2"   # Standard — pays 1.5×
    SIX_TO_FIVE = "6:5"    # House-favourable — pays 1.2×


class DoubleRule(str, Enum):
    """Which hand totals are eligible for doubling down.

    ANY        — any initial two-card total (standard Vegas Strip).
    NINE_ELEVEN — only hard 9, 10, or 11.
    TEN_ELEVEN  — only hard 10 or 11.
    HARD_ONLY   — any hard total (Aces-as-1), no soft doubles.
    """

    ANY = "any"
    NINE_ELEVEN = "9-11"
    TEN_ELEVEN = "10-11"
    HARD_ONLY = "hard_only"


@dataclass(frozen=True)
class BlackjackRules:
    """Immutable rule configuration for a blackjack variant.

    All fields have sensible Vegas Strip defaults.

    Attributes
    ----------
    num_decks                : Number of decks in the shoe (1–8).
    dealer_stand_rule        : S17 or H17.
    blackjack_payout         : 3:2 or 6:5.
    allow_double             : Global toggle — can the player double at all?
    double_on                : Restricts which totals allow doubling.
    allow_double_after_split : DAS allowed.
    allow_resplit_aces       : Resplitting Aces allowed.
    hit_split_aces           : Player may draw multiple cards to split-Ace hands.
    allow_surrender          : Late surrender allowed.
    surrender_after_split    : Surrender after split (very rare; off by default).
    max_splits               : Maximum number of re-splits per round.
    peek                     : US-style: dealer peeks for BJ before player acts.
    reshuffle_penetration    : Shoe penetration before reshuffling.
    """

    num_decks: int = 6
    dealer_stand_rule: DealerStandRule = DealerStandRule.STAND_SOFT_17
    blackjack_payout: BlackjackPayout = BlackjackPayout.THREE_TO_TWO
    # --- Doubling ---
    allow_double: bool = True
    double_on: DoubleRule = DoubleRule.ANY
    allow_double_after_split: bool = True
    # --- Splitting ---
    allow_resplit_aces: bool = False
    hit_split_aces: bool = False
    max_splits: int = 3
    # --- Surrender ---
    allow_surrender: bool = True
    surrender_after_split: bool = False
    # --- Deal mechanics ---
    peek: bool = True               # US hole-card peek
    reshuffle_penetration: float = 0.75

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def standard(cls) -> "BlackjackRules":
        """Vegas Strip: 6 decks, S17, 3:2 payout, DAS, late surrender."""
        return cls()

    @classmethod
    def vegas_downtown(cls) -> "BlackjackRules":
        """Vegas Downtown: 1 deck, H17, 3:2 payout."""
        return cls(
            num_decks=1,
            dealer_stand_rule=DealerStandRule.HIT_SOFT_17,
            blackjack_payout=BlackjackPayout.THREE_TO_TWO,
        )

    @classmethod
    def unfavourable(cls) -> "BlackjackRules":
        """House-edge maximised: 6 decks, H17, 6:5 payout, no surrender."""
        return cls(
            blackjack_payout=BlackjackPayout.SIX_TO_FIVE,
            dealer_stand_rule=DealerStandRule.HIT_SOFT_17,
            allow_surrender=False,
        )

    @classmethod
    def single_deck(cls) -> "BlackjackRules":
        """Classic single-deck game, S17, 3:2."""
        return cls(num_decks=1)

    @classmethod
    def atlantic_city(cls) -> "BlackjackRules":
        """Atlantic City: 8 decks, S17, 3:2, DAS, resplit aces allowed."""
        return cls(
            num_decks=8,
            dealer_stand_rule=DealerStandRule.STAND_SOFT_17,
            blackjack_payout=BlackjackPayout.THREE_TO_TWO,
            allow_resplit_aces=True,
            max_splits=3,
        )

    @classmethod
    def european(cls) -> "BlackjackRules":
        """European No-Hole-Card: 6D, S17, no peek, no surrender."""
        return cls(
            num_decks=6,
            dealer_stand_rule=DealerStandRule.STAND_SOFT_17,
            peek=False,
            allow_surrender=False,
        )

    @classmethod
    def restrictive_double(cls) -> "BlackjackRules":
        """Double allowed on 10–11 only (common in some casinos)."""
        return cls(double_on=DoubleRule.TEN_ELEVEN)

    # ------------------------------------------------------------------
    # Dealer logic
    # ------------------------------------------------------------------

    def dealer_must_hit(self, hand_value: int, is_soft: bool) -> bool:
        """Return True when the dealer is required to draw another card.

        Parameters
        ----------
        hand_value : Current hand total.
        is_soft    : Whether the hand contains an Ace counted as 11.
        """
        if hand_value < 17:
            return True
        if (
            hand_value == 17
            and is_soft
            and self.dealer_stand_rule == DealerStandRule.HIT_SOFT_17
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Action legality helpers
    # ------------------------------------------------------------------

    def can_double_on_hand(
        self,
        hand_value: int,
        is_soft: bool,
        split_count: int,
    ) -> bool:
        """True if the rules permit doubling on this hand.

        Parameters
        ----------
        hand_value  : Current hand total.
        is_soft     : Whether the total is soft (Ace = 11).
        split_count : How many splits have occurred this round.
        """
        if not self.allow_double:
            return False
        if split_count > 0 and not self.allow_double_after_split:
            return False

        match self.double_on:
            case DoubleRule.ANY:
                return True
            case DoubleRule.NINE_ELEVEN:
                return not is_soft and hand_value in (9, 10, 11)
            case DoubleRule.TEN_ELEVEN:
                return not is_soft and hand_value in (10, 11)
            case DoubleRule.HARD_ONLY:
                return not is_soft
            case _:
                return True

    def can_split_on_context(
        self,
        split_count: int,
        from_split_aces: bool,
    ) -> bool:
        """True if the rules permit another split given the current round context.

        Parameters
        ----------
        split_count      : Splits already executed this round.
        from_split_aces  : Whether the current hand came from a previous Ace split.
        """
        if split_count >= self.max_splits:
            return False
        if from_split_aces and not self.allow_resplit_aces:
            return False
        return True

    def can_surrender_in_context(self, split_count: int) -> bool:
        """True if surrender is available in the current game context.

        Parameters
        ----------
        split_count : Splits already executed this round.
        """
        if not self.allow_surrender:
            return False
        if split_count > 0 and not self.surrender_after_split:
            return False
        return True

    # ------------------------------------------------------------------
    # Payout helper
    # ------------------------------------------------------------------

    def blackjack_multiplier(self) -> float:
        """Payout multiplier for a natural blackjack (excluding the original bet)."""
        match self.blackjack_payout:
            case BlackjackPayout.THREE_TO_TWO:
                return 1.5
            case BlackjackPayout.SIX_TO_FIVE:
                return 1.2
            case _:
                return 1.5

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"BlackjackRules({self.num_decks}D, {self.dealer_stand_rule.value}, "
            f"{self.blackjack_payout.value}, double_on={self.double_on.value})"
        )

"""
blackjack.strategies.basic
~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistically optimal Basic Strategy for 6-deck, S17, DAS blackjack.

Based on the standard Basic Strategy charts published by the Wizard of Odds
(wizardofodds.com). The lookup tables encode:
  - Hard totals (4–17+)
  - Soft totals (soft 13–20)
  - Pair splits (2-2 through A-A)

Fallback ordering when an action is not legal:
  DOUBLE → HIT (if can_double=False)
  SPLIT  → falls through to hard/soft table
  SURRENDER → HIT (if can_surrender=False)
"""
from __future__ import annotations

from blackjack.cards.card import Card, Rank
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action, BaseStrategy

# ---------------------------------------------------------------------------
# Lookup tables
# Outer key: player total (hard) or soft total
# Inner key: dealer upcard value (2–10, Ace=11)
# ---------------------------------------------------------------------------

# Hard total table: player_total → {dealer_upcard → Action}
_HARD: dict[int, dict[int, Action]] = {
    4:  dict.fromkeys(range(2, 12), Action.HIT),
    5:  dict.fromkeys(range(2, 12), Action.HIT),
    6:  dict.fromkeys(range(2, 12), Action.HIT),
    7:  dict.fromkeys(range(2, 12), Action.HIT),
    8:  dict.fromkeys(range(2, 12), Action.HIT),
    9: {
        2: Action.HIT,    3: Action.DOUBLE, 4: Action.DOUBLE,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    10: {
        2: Action.DOUBLE,  3: Action.DOUBLE,  4: Action.DOUBLE,
        5: Action.DOUBLE,  6: Action.DOUBLE,  7: Action.DOUBLE,
        8: Action.DOUBLE,  9: Action.DOUBLE, 10: Action.HIT, 11: Action.HIT,
    },
    11: {
        2: Action.DOUBLE,  3: Action.DOUBLE,  4: Action.DOUBLE,
        5: Action.DOUBLE,  6: Action.DOUBLE,  7: Action.DOUBLE,
        8: Action.DOUBLE,  9: Action.DOUBLE, 10: Action.DOUBLE, 11: Action.HIT,
    },
    12: {
        2: Action.HIT,    3: Action.HIT,    4: Action.STAND,
        5: Action.STAND,  6: Action.STAND,  7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    13: {
        2: Action.STAND,  3: Action.STAND,  4: Action.STAND,
        5: Action.STAND,  6: Action.STAND,  7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    14: {
        2: Action.STAND,  3: Action.STAND,  4: Action.STAND,
        5: Action.STAND,  6: Action.STAND,  7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    15: {
        2: Action.STAND,  3: Action.STAND,  4: Action.STAND,
        5: Action.STAND,  6: Action.STAND,  7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.SURRENDER, 11: Action.HIT,
    },
    16: {
        2: Action.STAND,  3: Action.STAND,  4: Action.STAND,
        5: Action.STAND,  6: Action.STAND,  7: Action.HIT,
        8: Action.HIT,    9: Action.SURRENDER, 10: Action.SURRENDER,
        11: Action.SURRENDER,
    },
    17: dict.fromkeys(range(2, 12), Action.STAND),
}

# Soft total table (Ace counted as 11): soft_total → {dealer_upcard → Action}
_SOFT: dict[int, dict[int, Action]] = {
    13: {  # A+2
        2: Action.HIT,    3: Action.HIT,    4: Action.HIT,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    14: {  # A+3
        2: Action.HIT,    3: Action.HIT,    4: Action.HIT,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    15: {  # A+4
        2: Action.HIT,    3: Action.HIT,    4: Action.DOUBLE,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    16: {  # A+5
        2: Action.HIT,    3: Action.HIT,    4: Action.DOUBLE,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    17: {  # A+6
        2: Action.HIT,    3: Action.DOUBLE, 4: Action.DOUBLE,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.HIT,
        8: Action.HIT,    9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    18: {  # A+7
        2: Action.STAND,  3: Action.DOUBLE, 4: Action.DOUBLE,
        5: Action.DOUBLE, 6: Action.DOUBLE, 7: Action.STAND,
        8: Action.STAND,  9: Action.HIT,   10: Action.HIT, 11: Action.HIT,
    },
    19: {  # A+8
        2: Action.STAND,  3: Action.STAND,  4: Action.STAND,
        5: Action.STAND,  6: Action.DOUBLE, 7: Action.STAND,
        8: Action.STAND,  9: Action.STAND, 10: Action.STAND, 11: Action.STAND,
    },
    20: dict.fromkeys(range(2, 12), Action.STAND),  # A+9 — always stand
}

# Ranks that always split / never split
_ALWAYS_SPLIT = frozenset({Rank.ACE, Rank.EIGHT})
_NEVER_SPLIT = frozenset({Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.FIVE})


def _dealer_key(upcard: Card) -> int:
    """Normalise dealer upcard to lookup key (2–10, Ace=11)."""
    return 11 if upcard.is_ace else min(upcard.value, 10)


class BasicStrategy(BaseStrategy):
    """Statistically optimal 6-deck Basic Strategy (S17, DAS, late surrender).

    Lookup priority:
    1. Pair split table (if can_split and hand.can_split)
    2. Soft total table (if hand.is_soft)
    3. Hard total table (fallback)
    """

    @property
    def name(self) -> str:
        return "basic"

    def decide(
        self,
        player_hand: Hand,
        dealer_upcard: Card,
        can_double: bool = True,
        can_split: bool = True,
        can_surrender: bool = True,
    ) -> Action:
        pv = player_hand.value

        # Automatic stand on 21 or above
        if pv >= 21:
            return Action.STAND

        dk = _dealer_key(dealer_upcard)

        # --- 1. Pairs ---
        if can_split and player_hand.can_split:
            split_action = self._pair_decision(player_hand, dk)
            if split_action is not None:
                return split_action

        # --- 2. Soft totals ---
        if player_hand.is_soft and pv in _SOFT:
            action = _SOFT[pv].get(dk, Action.HIT)
            if action == Action.DOUBLE and not can_double:
                return Action.HIT
            return action

        # --- 3. Hard totals ---
        clamped = max(4, min(pv, 17))
        action = _HARD.get(clamped, {}).get(dk, Action.STAND)

        if action == Action.DOUBLE and not can_double:
            return Action.HIT
        if action == Action.SURRENDER and not can_surrender:
            return Action.HIT

        return action

    # ------------------------------------------------------------------
    # Pair logic
    # ------------------------------------------------------------------

    def _pair_decision(self, hand: Hand, dealer_key: int) -> Action | None:
        """Return SPLIT or None (fall through to hard/soft table)."""
        rank = hand.cards[0].rank

        if rank in _ALWAYS_SPLIT:
            return Action.SPLIT
        if rank in _NEVER_SPLIT:
            return None  # treat as hard total

        # 2s, 3s, 6s, 7s, 9s: split vs dealer 2–6
        if rank in (Rank.TWO, Rank.THREE, Rank.SIX, Rank.SEVEN, Rank.NINE):
            return Action.SPLIT if dealer_key <= 6 else None

        # 4s: split vs dealer 5–6 only
        if rank == Rank.FOUR:
            return Action.SPLIT if dealer_key in (5, 6) else None

        return None

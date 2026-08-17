"""
blackjack.cards.hand
~~~~~~~~~~~~~~~~~~~~
Hand — a mutable collection of Cards with blackjack value semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from blackjack.cards.card import Card


@dataclass
class Hand:
    """A player's or dealer's hand.

    Value semantics
    ---------------
    * Aces count as 11 until doing so would bust the hand, then as 1.
    * A *soft* hand contains an Ace counted as 11.
    * Blackjack requires exactly 2 cards totalling 21.

    Example
    -------
    >>> from blackjack.cards.card import Card, Rank, Suit
    >>> h = Hand()
    >>> h.add_card(Card(Rank.ACE, Suit.SPADES))
    >>> h.add_card(Card(Rank.KING, Suit.HEARTS))
    >>> h.value
    21
    >>> h.is_blackjack
    True
    >>> h.is_soft
    False
    """

    cards: list[Card] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_cards(cls, *cards: Card) -> "Hand":
        """Create a hand pre-loaded with the given cards.

        Example
        -------
        >>> h = Hand.from_cards(Card(Rank.ACE, Suit.SPADES), Card(Rank.SIX, Suit.HEARTS))
        >>> h.value
        17
        """
        h = cls()
        for card in cards:
            h.cards.append(card)
        return h

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_card(self, card: Card) -> None:
        """Append a card to this hand."""
        self.cards.append(card)

    def clear(self) -> None:
        """Remove all cards (reuse hand object between rounds)."""
        self.cards.clear()

    def split(self) -> "Hand":
        """Remove the second card and return it in a new one-card Hand.

        After calling this the receiver has exactly one card left; the
        caller is responsible for dealing a new card to both hands.

        Raises
        ------
        ValueError
            If the hand does not satisfy :attr:`can_split`.
        """
        if not self.can_split:
            raise ValueError(
                f"Cannot split: need exactly 2 cards of the same rank — got {self}"
            )
        second = self.cards.pop()
        return Hand(cards=[second])

    # ------------------------------------------------------------------
    # Value properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> int:
        """Best non-busting total. Aces are reduced from 11 → 1 as needed."""
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.is_ace)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    @property
    def hard_value(self) -> int:
        """Total counting ALL Aces as 1 (no soft reduction)."""
        return sum(1 if c.is_ace else c.value for c in self.cards)

    @property
    def soft_value(self) -> int:
        """Total counting the FIRST Ace as 11 (may exceed 21 — use .value for safe total)."""
        base = self.hard_value
        if any(c.is_ace for c in self.cards):
            return base + 10
        return base

    @property
    def num_aces(self) -> int:
        """Number of Ace cards in the hand."""
        return sum(1 for c in self.cards if c.is_ace)

    @property
    def is_bust(self) -> bool:
        """True when the hand value exceeds 21."""
        return self.value > 21

    @property
    def is_blackjack(self) -> bool:
        """True for a natural blackjack (exactly 2 cards, value == 21)."""
        return len(self.cards) == 2 and self.value == 21

    # Alias used in casino literature
    is_natural = is_blackjack  # type: ignore[assignment]

    @property
    def is_soft(self) -> bool:
        """True when the hand contains an Ace counted as 11 (not yet busted)."""
        hard_total = sum(1 if c.is_ace else c.value for c in self.cards)
        has_ace = any(c.is_ace for c in self.cards)
        return has_ace and (hard_total + 10) < 21

    @property
    def is_pair(self) -> bool:
        """True when the hand has exactly 2 cards of the same rank (splittable)."""
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    @property
    def can_split(self) -> bool:
        """Alias for :attr:`is_pair` — kept for backwards compatibility."""
        return self.is_pair

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        cards_str = " ".join(str(c) for c in self.cards)
        soft_tag = " (soft)" if self.is_soft else ""
        return f"[{cards_str}] = {self.value}{soft_tag}"

    def __repr__(self) -> str:
        return f"Hand(cards={self.cards!r}, value={self.value})"

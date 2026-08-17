"""
blackjack.cards.card
~~~~~~~~~~~~~~~~~~~~
Immutable Card dataclass with Suit, Rank, and blackjack value semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(str, Enum):
    """The four suits of a standard deck."""

    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"

    def __str__(self) -> str:
        symbols = {
            Suit.SPADES: "♠",
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
        }
        return symbols[self]


class Rank(str, Enum):
    """The thirteen ranks of a standard deck."""

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


# Blackjack values: Aces start at 11 — Hand logic reduces to 1 on bust.
RANK_VALUES: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 10,
    Rank.QUEEN: 10,
    Rank.KING: 10,
    Rank.ACE: 11,
}


@dataclass(frozen=True, slots=True)
class Card:
    """An immutable playing card.

    Attributes
    ----------
    rank : Rank
    suit : Suit

    Properties
    ----------
    value     : blackjack hard value (Ace = 11)
    is_ace    : True if rank is Ace
    is_face   : True if Jack, Queen, or King
    """

    rank: Rank
    suit: Suit

    @property
    def value(self) -> int:
        """Blackjack value (Ace = 11 by default)."""
        return RANK_VALUES[self.rank]

    @property
    def is_ace(self) -> bool:
        return self.rank == Rank.ACE

    @property
    def is_face(self) -> bool:
        return self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING)

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit}"

    def __repr__(self) -> str:
        return f"Card(rank={self.rank!r}, suit={self.suit!r})"

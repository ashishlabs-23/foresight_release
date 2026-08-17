"""
blackjack.cards.deck
~~~~~~~~~~~~~~~~~~~~
Deck (52 cards) and Shoe (multi-deck, auto-reshuffle).
"""
from __future__ import annotations

import random
from blackjack.cards.card import Card, Rank, Suit


class Deck:
    """A standard 52-card deck (unshuffled by default).

    Useful as a building block; for actual gameplay use :class:`Shoe`.
    """

    def __init__(self) -> None:
        self._cards: list[Card] = [
            Card(rank, suit) for suit in Suit for rank in Rank
        ]

    @property
    def cards(self) -> list[Card]:
        """Return a defensive copy of the card list."""
        return list(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __repr__(self) -> str:
        return f"Deck(cards={len(self._cards)})"


class Shoe:
    """A multi-deck shoe that auto-reshuffles at a configurable penetration point.

    Parameters
    ----------
    num_decks : int
        Number of 52-card decks in the shoe (default 6).
    reshuffle_penetration : float
        Fraction of cards dealt before reshuffling (0 < p ≤ 1).
        Default 0.75 means reshuffle when 75 % of cards have been dealt.
    seed : int | None
        Random seed for reproducible simulations.
    """

    def __init__(
        self,
        num_decks: int = 6,
        reshuffle_penetration: float = 0.75,
        seed: int | None = None,
    ) -> None:
        if num_decks < 1:
            raise ValueError(f"num_decks must be >= 1, got {num_decks}")
        if not (0.0 < reshuffle_penetration <= 1.0):
            raise ValueError(
                f"reshuffle_penetration must be in (0, 1], got {reshuffle_penetration}"
            )

        self._num_decks = num_decks
        self._reshuffle_penetration = reshuffle_penetration
        self._total_cards: int = num_decks * 52
        self._rng = random.Random(seed)
        self._cards: list[Card] = []
        self._reshuffle_count: int = 0
        self._running_count: int = 0
        self._hidden_cards: list[Card] = []
        self.shuffle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shuffle(self) -> None:
        """Build a fresh set of cards and shuffle them."""
        self._cards = [
            Card(rank, suit)
            for _ in range(self._num_decks)
            for suit in Suit
            for rank in Rank
        ]
        self._rng.shuffle(self._cards)
        self._reshuffle_count += 1
        self._running_count = 0
        self._hidden_cards.clear()

    @classmethod
    def create_synthetic(
        cls,
        num_decks: int,
        observed_cards: list[str],
        seed: int | None = None,
    ) -> Shoe:
        """Create a Shoe omitting specific observed cards, for Monte Carlo rollouts."""
        shoe = cls(num_decks=num_decks, seed=seed)
        
        # Build the exact composition of a full shoe
        full_deck = [Card(rank, suit) for _ in range(num_decks) for suit in Suit for rank in Rank]
        
        # Remove observed cards
        for card_str in observed_cards:
            rank_str, suit_str = card_str[0], card_str[1]
            # Find a matching card in full_deck
            for i, c in enumerate(full_deck):
                if c.rank.value == rank_str and c.suit.value == suit_str:
                    full_deck.pop(i)
                    break
                    
        shoe._cards = full_deck
        shoe._rng.shuffle(shoe._cards)
        shoe._total_cards = num_decks * 52
        return shoe

    def deal(self, hidden: bool = False) -> Card:
        """Deal one card from the top of the shoe.

        Automatically reshuffles if the penetration threshold is reached.
        If hidden is True, the card is not added to the running count until revealed.
        """
        if self.needs_reshuffle:
            self.shuffle()
        card = self._cards.pop()
        
        if hidden:
            self._hidden_cards.append(card)
        else:
            self._update_running_count(card)
            
        return card

    def reveal_hidden(self) -> None:
        """Reveal all hidden cards and add them to the running count."""
        for card in self._hidden_cards:
            self._update_running_count(card)
        self._hidden_cards.clear()

    def _update_running_count(self, card: Card) -> None:
        """Hi-Lo system: +1 for 2-6, 0 for 7-9, -1 for 10-A."""
        # Ace value is 11, Face cards are 10
        if 2 <= card.value <= 6:
            self._running_count += 1
        elif card.value >= 10:
            self._running_count -= 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cards_remaining(self) -> int:
        """Number of cards left in the shoe."""
        return len(self._cards)

    @property
    def needs_reshuffle(self) -> bool:
        """True when the shoe has been depleted past the penetration threshold."""
        threshold = int(self._total_cards * (1.0 - self._reshuffle_penetration))
        return len(self._cards) <= threshold

    @property
    def penetration(self) -> float:
        """Fraction of cards that have been dealt since last shuffle (0.0–1.0)."""
        dealt = self._total_cards - len(self._cards)
        return dealt / self._total_cards

    @property
    def reshuffle_count(self) -> int:
        """How many times the shoe has been reshuffled (including the initial shuffle)."""
        return self._reshuffle_count

    @property
    def num_decks(self) -> int:
        return self._num_decks

    @property
    def running_count(self) -> int:
        """Hi-Lo running count of all revealed cards."""
        return self._running_count

    def __repr__(self) -> str:
        return (
            f"Shoe(num_decks={self._num_decks}, "
            f"cards_remaining={self.cards_remaining}, "
            f"penetration={self.penetration:.1%})"
        )

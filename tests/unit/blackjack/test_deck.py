"""
tests/unit/blackjack/test_deck.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Deck and Shoe.
"""
from __future__ import annotations

import pytest

from blackjack.cards.deck import Deck, Shoe
from blackjack.cards.card import Card


class TestDeck:
    def test_deck_has_52_cards(self) -> None:
        deck = Deck()
        assert len(deck) == 52

    def test_deck_cards_are_unique(self) -> None:
        deck = Deck()
        assert len(set(deck.cards)) == 52

    def test_deck_cards_returns_copy(self) -> None:
        deck = Deck()
        cards = deck.cards
        cards.pop()
        assert len(deck) == 52  # original unchanged


class TestShoe:
    def test_shoe_default_has_6_decks(self) -> None:
        shoe = Shoe(seed=1)
        assert shoe.num_decks == 6
        assert shoe.cards_remaining == 312  # 6 × 52

    def test_shoe_single_deck(self) -> None:
        shoe = Shoe(num_decks=1, seed=1)
        assert shoe.cards_remaining == 52

    def test_deal_reduces_count(self) -> None:
        shoe = Shoe(num_decks=1, seed=1)
        card = shoe.deal()
        assert isinstance(card, Card)
        assert shoe.cards_remaining == 51

    def test_shoe_invalid_num_decks(self) -> None:
        with pytest.raises(ValueError, match="num_decks"):
            Shoe(num_decks=0)

    def test_shoe_invalid_penetration(self) -> None:
        with pytest.raises(ValueError, match="reshuffle_penetration"):
            Shoe(reshuffle_penetration=0.0)
        with pytest.raises(ValueError, match="reshuffle_penetration"):
            Shoe(reshuffle_penetration=1.1)

    def test_penetration_starts_at_zero(self) -> None:
        shoe = Shoe(seed=1)
        assert shoe.penetration == 0.0

    def test_penetration_increases_on_deal(self) -> None:
        shoe = Shoe(num_decks=1, seed=1)
        shoe.deal()
        assert shoe.penetration > 0.0

    def test_reshuffle_on_penetration(self) -> None:
        """Shoe should auto-reshuffle when penetration threshold is hit."""
        shoe = Shoe(num_decks=1, reshuffle_penetration=0.1, seed=1)
        # Dealing past 10% of 52 = 5 cards should trigger reshuffle
        for _ in range(10):
            shoe.deal()
        assert shoe.reshuffle_count >= 2

    def test_seed_produces_reproducible_sequence(self) -> None:
        shoe1 = Shoe(num_decks=1, seed=99)
        shoe2 = Shoe(num_decks=1, seed=99)
        cards1 = [shoe1.deal() for _ in range(10)]
        cards2 = [shoe2.deal() for _ in range(10)]
        assert cards1 == cards2

    def test_different_seeds_produce_different_sequences(self) -> None:
        shoe1 = Shoe(num_decks=1, seed=1)
        shoe2 = Shoe(num_decks=1, seed=2)
        cards1 = [shoe1.deal() for _ in range(10)]
        cards2 = [shoe2.deal() for _ in range(10)]
        assert cards1 != cards2

"""
tests/unit/blackjack/test_card.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Card, Rank, Suit, and RANK_VALUES.
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit, RANK_VALUES


class TestRankValues:
    def test_all_ranks_have_values(self) -> None:
        assert len(RANK_VALUES) == 13

    def test_number_cards_have_correct_values(self) -> None:
        assert RANK_VALUES[Rank.TWO] == 2
        assert RANK_VALUES[Rank.NINE] == 9

    def test_face_cards_value_ten(self) -> None:
        assert RANK_VALUES[Rank.JACK] == 10
        assert RANK_VALUES[Rank.QUEEN] == 10
        assert RANK_VALUES[Rank.KING] == 10

    def test_ace_value_eleven(self) -> None:
        assert RANK_VALUES[Rank.ACE] == 11


class TestCard:
    def test_card_creation(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_card_value_matches_rank(self) -> None:
        assert Card(Rank.SEVEN, Suit.HEARTS).value == 7
        assert Card(Rank.KING, Suit.CLUBS).value == 10

    def test_ace_is_ace(self) -> None:
        ace = Card(Rank.ACE, Suit.DIAMONDS)
        assert ace.is_ace is True
        assert Card(Rank.KING, Suit.SPADES).is_ace is False

    def test_face_card_detection(self) -> None:
        assert Card(Rank.JACK, Suit.HEARTS).is_face is True
        assert Card(Rank.QUEEN, Suit.CLUBS).is_face is True
        assert Card(Rank.KING, Suit.SPADES).is_face is True
        assert Card(Rank.TEN, Suit.DIAMONDS).is_face is False
        assert Card(Rank.ACE, Suit.HEARTS).is_face is False

    def test_card_is_frozen(self) -> None:
        card = Card(Rank.TWO, Suit.CLUBS)
        with pytest.raises(AttributeError):
            card.rank = Rank.THREE  # type: ignore[misc]

    def test_card_str_repr(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        s = str(card)
        assert "A" in s
        assert "S" in s or "♠" in s

    def test_card_equality(self) -> None:
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        c3 = Card(Rank.ACE, Suit.HEARTS)
        assert c1 == c2
        assert c1 != c3

    def test_card_hashable(self) -> None:
        """Cards must be hashable to be used in sets/dict keys."""
        cards = {Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)}
        assert len(cards) == 2

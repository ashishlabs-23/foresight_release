"""
tests/unit/blackjack/test_hand.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Hand — value calculation, blackjack detection, soft/hard hands.
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand


def _card(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def _hand(*ranks: Rank) -> Hand:
    h = Hand()
    for rank in ranks:
        h.add_card(_card(rank))
    return h


class TestHandValue:
    def test_simple_hard_total(self) -> None:
        h = _hand(Rank.SEVEN, Rank.EIGHT)
        assert h.value == 15

    def test_ace_counts_as_eleven(self) -> None:
        h = _hand(Rank.ACE, Rank.SEVEN)
        assert h.value == 18

    def test_ace_reduces_to_one_on_bust(self) -> None:
        h = _hand(Rank.ACE, Rank.SEVEN, Rank.SIX)
        assert h.value == 14  # 11 + 7 + 6 > 21 → 1 + 7 + 6 = 14

    def test_two_aces_reduced_correctly(self) -> None:
        h = _hand(Rank.ACE, Rank.ACE)
        assert h.value == 12  # 11 + 1

    def test_blackjack_value(self) -> None:
        h = _hand(Rank.ACE, Rank.KING)
        assert h.value == 21

    def test_bust_value(self) -> None:
        h = _hand(Rank.TEN, Rank.KING, Rank.FIVE)
        assert h.value == 25


class TestHandProperties:
    def test_is_blackjack_true(self) -> None:
        h = _hand(Rank.ACE, Rank.KING)
        assert h.is_blackjack is True

    def test_is_blackjack_false_on_21_with_three_cards(self) -> None:
        h = _hand(Rank.SEVEN, Rank.SEVEN, Rank.SEVEN)
        assert h.value == 21
        assert h.is_blackjack is False

    def test_is_bust_true(self) -> None:
        h = _hand(Rank.KING, Rank.QUEEN, Rank.THREE)
        assert h.is_bust is True

    def test_is_bust_false(self) -> None:
        h = _hand(Rank.KING, Rank.NINE)
        assert h.is_bust is False

    def test_is_soft_true(self) -> None:
        h = _hand(Rank.ACE, Rank.SEVEN)
        assert h.is_soft is True

    def test_is_soft_false_when_ace_forced_to_one(self) -> None:
        h = _hand(Rank.ACE, Rank.SEVEN, Rank.SIX)
        assert h.is_soft is False  # ace is now counted as 1

    def test_can_split_same_rank(self) -> None:
        h = _hand(Rank.EIGHT, Rank.EIGHT)
        assert h.can_split is True

    def test_can_split_different_ranks(self) -> None:
        h = _hand(Rank.EIGHT, Rank.NINE)
        assert h.can_split is False

    def test_can_split_requires_two_cards(self) -> None:
        h = _hand(Rank.EIGHT, Rank.EIGHT, Rank.EIGHT)
        assert h.can_split is False

    def test_add_card_increases_length(self) -> None:
        h = Hand()
        assert len(h) == 0
        h.add_card(_card(Rank.ACE))
        assert len(h) == 1

    def test_clear_empties_hand(self) -> None:
        h = _hand(Rank.ACE, Rank.KING)
        h.clear()
        assert len(h) == 0

"""
tests/unit/blackjack/test_basic_strategy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for BasicStrategy decision logic.
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action
from blackjack.strategies.basic import BasicStrategy


def _card(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def _hand(*ranks: Rank) -> Hand:
    h = Hand()
    for rank in ranks:
        h.add_card(_card(rank))
    return h


@pytest.fixture
def strategy() -> BasicStrategy:
    return BasicStrategy()


class TestBasicStrategyName:
    def test_name(self, strategy: BasicStrategy) -> None:
        assert strategy.name == "basic"


class TestHardTotals:
    def test_hard_8_always_hit(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.THREE, Rank.FIVE)
        for rank in [Rank.TWO, Rank.SIX, Rank.TEN, Rank.ACE]:
            action = strategy.decide(hand, _card(rank))
            assert action == Action.HIT, f"Expected HIT for hard 8 vs {rank}"

    def test_hard_17_always_stand(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.TEN, Rank.SEVEN)
        for rank in [Rank.TWO, Rank.SIX, Rank.TEN, Rank.ACE]:
            action = strategy.decide(hand, _card(rank))
            assert action == Action.STAND, f"Expected STAND for hard 17 vs {rank}"

    def test_hard_11_double_vs_low_card(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.SIX, Rank.FIVE)
        action = strategy.decide(hand, _card(Rank.SIX))
        assert action == Action.DOUBLE

    def test_hard_11_hit_when_cannot_double(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.SIX, Rank.FIVE)
        action = strategy.decide(hand, _card(Rank.SIX), can_double=False)
        assert action == Action.HIT

    def test_surrender_16_vs_10(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.NINE, Rank.SEVEN)
        action = strategy.decide(hand, _card(Rank.TEN))
        assert action == Action.SURRENDER

    def test_no_surrender_when_disallowed(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.NINE, Rank.SEVEN)
        action = strategy.decide(hand, _card(Rank.TEN), can_surrender=False)
        assert action == Action.HIT


class TestSoftTotals:
    def test_soft_18_stand_vs_7(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.SEVEN)
        action = strategy.decide(hand, _card(Rank.SEVEN))
        assert action == Action.STAND

    def test_soft_18_double_vs_6(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.SEVEN)
        action = strategy.decide(hand, _card(Rank.SIX))
        assert action == Action.DOUBLE

    def test_soft_18_hit_vs_9(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.SEVEN)
        action = strategy.decide(hand, _card(Rank.NINE))
        assert action == Action.HIT

    def test_soft_20_always_stand(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.NINE)
        for rank in [Rank.TWO, Rank.SIX, Rank.TEN, Rank.ACE]:
            action = strategy.decide(hand, _card(rank))
            assert action == Action.STAND


class TestPairSplitting:
    def test_always_split_aces(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.ACE)
        for rank in [Rank.TWO, Rank.TEN]:
            action = strategy.decide(hand, _card(rank))
            assert action == Action.SPLIT

    def test_always_split_eights(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.EIGHT, Rank.EIGHT)
        for rank in [Rank.TWO, Rank.TEN]:
            action = strategy.decide(hand, _card(rank))
            assert action == Action.SPLIT

    def test_never_split_tens(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.TEN, Rank.TEN)
        action = strategy.decide(hand, _card(Rank.SIX))
        assert action != Action.SPLIT

    def test_no_split_when_disallowed(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.ACE)
        action = strategy.decide(hand, _card(Rank.SIX), can_split=False)
        # Should fall through to hard/soft table, not split
        assert action != Action.SPLIT


class TestEdgeCases:
    def test_stand_on_21(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.ACE, Rank.KING)
        action = strategy.decide(hand, _card(Rank.TEN))
        assert action == Action.STAND

    def test_stand_on_bust(self, strategy: BasicStrategy) -> None:
        hand = _hand(Rank.TEN, Rank.KING, Rank.FIVE)  # 25 — busted
        action = strategy.decide(hand, _card(Rank.TEN))
        assert action == Action.STAND

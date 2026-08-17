"""
tests/unit/blackjack/test_hand_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive hand evaluation tests — Phase 2.

Covers ALL hand value edge cases including:
- Hard totals
- Soft totals
- Multiple aces
- New Phase 2 properties (hard_value, soft_value, num_aces, is_pair)
- The split() method
- from_cards() constructor
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand


def c(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def h(*ranks: Rank) -> Hand:
    return Hand.from_cards(*[c(r) for r in ranks])


# ---------------------------------------------------------------------------
# Hard total values
# ---------------------------------------------------------------------------

class TestHardTotals:
    def test_two_face_cards(self) -> None:
        assert h(Rank.KING, Rank.QUEEN).value == 20

    def test_face_and_number(self) -> None:
        assert h(Rank.TEN, Rank.SIX).value == 16

    def test_three_cards_no_ace(self) -> None:
        assert h(Rank.SEVEN, Rank.FIVE, Rank.THREE).value == 15

    def test_min_four_cards(self) -> None:
        assert h(Rank.TWO, Rank.TWO, Rank.TWO, Rank.TWO).value == 8

    def test_21_three_cards(self) -> None:
        assert h(Rank.TEN, Rank.SIX, Rank.FIVE).value == 21

    def test_bust_hard(self) -> None:
        assert h(Rank.TEN, Rank.KING, Rank.FIVE).value == 25

    def test_22_bust(self) -> None:
        assert h(Rank.TEN, Rank.TEN, Rank.TWO).value == 22


# ---------------------------------------------------------------------------
# Soft totals — Ace counted as 11
# ---------------------------------------------------------------------------

class TestSoftTotals:
    """A + 6 = soft 17; A + 6 + 10 = hard 17."""

    def test_ace_six_soft_17(self) -> None:
        hand = h(Rank.ACE, Rank.SIX)
        assert hand.value == 17
        assert hand.is_soft is True

    def test_ace_six_ten_hard_17(self) -> None:
        hand = h(Rank.ACE, Rank.SIX, Rank.TEN)
        assert hand.value == 17
        assert hand.is_soft is False

    def test_ace_king_blackjack(self) -> None:
        hand = h(Rank.ACE, Rank.KING)
        assert hand.value == 21
        assert hand.is_blackjack is True
        assert hand.is_soft is False  # BJ is NOT called soft

    def test_ace_seven_soft_18(self) -> None:
        hand = h(Rank.ACE, Rank.SEVEN)
        assert hand.value == 18
        assert hand.is_soft is True

    def test_ace_nine_soft_20(self) -> None:
        hand = h(Rank.ACE, Rank.NINE)
        assert hand.value == 20
        assert hand.is_soft is True

    def test_ace_two_soft_13(self) -> None:
        hand = h(Rank.ACE, Rank.TWO)
        assert hand.value == 13
        assert hand.is_soft is True

    def test_soft_after_adding_card(self) -> None:
        hand = h(Rank.ACE, Rank.FOUR)  # soft 15
        assert hand.is_soft is True
        hand.add_card(c(Rank.TWO))      # soft 17
        assert hand.is_soft is True

    def test_soft_to_hard_by_hit(self) -> None:
        hand = h(Rank.ACE, Rank.SEVEN)  # soft 18
        hand.add_card(c(Rank.FIVE))      # would be 23 soft → 13 hard
        assert hand.value == 13
        assert hand.is_soft is False


# ---------------------------------------------------------------------------
# Ace edge cases
# ---------------------------------------------------------------------------

class TestAceEdgeCases:
    """Corresponds to user-specified examples + edge cases."""

    def test_example_a_plus_6(self) -> None:
        """A + 6 → soft 17"""
        hand = h(Rank.ACE, Rank.SIX)
        assert hand.value == 17
        assert hand.is_soft is True

    def test_example_a_plus_6_plus_10(self) -> None:
        """A + 6 + 10 → hard 17"""
        hand = h(Rank.ACE, Rank.SIX, Rank.TEN)
        assert hand.value == 17
        assert hand.is_soft is False

    def test_example_a_plus_a(self) -> None:
        """A + A → 12 (one ace as 11, one as 1)"""
        hand = h(Rank.ACE, Rank.ACE)
        assert hand.value == 12

    def test_example_10_plus_6_plus_8_bust(self) -> None:
        """10 + 6 + 8 → bust (24)"""
        hand = h(Rank.TEN, Rank.SIX, Rank.EIGHT)
        assert hand.value == 24
        assert hand.is_bust is True

    def test_three_aces(self) -> None:
        """A + A + A → 13 (one as 11, two as 1)"""
        hand = h(Rank.ACE, Rank.ACE, Rank.ACE)
        assert hand.value == 13

    def test_four_aces(self) -> None:
        """A + A + A + A → 14"""
        hand = h(Rank.ACE, Rank.ACE, Rank.ACE, Rank.ACE)
        assert hand.value == 14

    def test_ace_reduces_on_bust(self) -> None:
        """A + 9 + 5 = 15 (ace reduced from 11 to 1)"""
        hand = h(Rank.ACE, Rank.NINE, Rank.FIVE)
        assert hand.value == 15
        assert hand.is_soft is False

    def test_bust_with_multiple_aces(self) -> None:
        """A + A + 9 + 9 → 20 (both aces forced to 1)"""
        hand = h(Rank.ACE, Rank.ACE, Rank.NINE, Rank.NINE)
        assert hand.value == 20

    def test_ace_remaining_soft_after_reduction(self) -> None:
        """A + 3 + 4 → 18 (still soft)"""
        hand = h(Rank.ACE, Rank.THREE, Rank.FOUR)
        assert hand.value == 18
        assert hand.is_soft is True


# ---------------------------------------------------------------------------
# Phase 2 new properties
# ---------------------------------------------------------------------------

class TestPhase2Properties:
    def test_hard_value_all_aces_as_one(self) -> None:
        hand = h(Rank.ACE, Rank.SIX)
        assert hand.hard_value == 7   # 1 + 6

    def test_hard_value_no_aces(self) -> None:
        hand = h(Rank.SEVEN, Rank.EIGHT)
        assert hand.hard_value == 15

    def test_soft_value_with_ace(self) -> None:
        hand = h(Rank.ACE, Rank.SIX)
        assert hand.soft_value == 17  # 1 + 6 + 10 (first ace as 11)

    def test_soft_value_no_ace(self) -> None:
        hand = h(Rank.SEVEN, Rank.EIGHT)
        assert hand.soft_value == 15  # same as hard_value

    def test_num_aces_zero(self) -> None:
        assert h(Rank.TEN, Rank.SEVEN).num_aces == 0

    def test_num_aces_one(self) -> None:
        assert h(Rank.ACE, Rank.SEVEN).num_aces == 1

    def test_num_aces_two(self) -> None:
        assert h(Rank.ACE, Rank.ACE).num_aces == 2

    def test_is_pair_same_rank(self) -> None:
        assert h(Rank.EIGHT, Rank.EIGHT).is_pair is True

    def test_is_pair_different_rank(self) -> None:
        assert h(Rank.EIGHT, Rank.NINE).is_pair is False

    def test_is_pair_aces(self) -> None:
        assert h(Rank.ACE, Rank.ACE).is_pair is True

    def test_is_pair_three_cards(self) -> None:
        assert h(Rank.EIGHT, Rank.EIGHT, Rank.EIGHT).is_pair is False

    def test_is_natural_alias(self) -> None:
        hand = h(Rank.ACE, Rank.KING)
        assert hand.is_blackjack == hand.is_natural  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Split method
# ---------------------------------------------------------------------------

class TestSplitMethod:
    def test_split_returns_new_one_card_hand(self) -> None:
        hand = h(Rank.EIGHT, Rank.EIGHT)
        new_hand = hand.split()
        assert len(new_hand) == 1
        assert new_hand.cards[0].rank == Rank.EIGHT

    def test_original_hand_has_one_card_after_split(self) -> None:
        hand = h(Rank.EIGHT, Rank.EIGHT)
        hand.split()
        assert len(hand) == 1
        assert hand.cards[0].rank == Rank.EIGHT

    def test_split_ace_pair(self) -> None:
        hand = h(Rank.ACE, Rank.ACE)
        new_hand = hand.split()
        assert new_hand.cards[0].is_ace
        assert hand.cards[0].is_ace

    def test_split_raises_on_non_pair(self) -> None:
        hand = h(Rank.EIGHT, Rank.NINE)
        with pytest.raises(ValueError, match="Cannot split"):
            hand.split()

    def test_split_raises_on_three_cards(self) -> None:
        hand = h(Rank.EIGHT, Rank.EIGHT, Rank.EIGHT)
        with pytest.raises(ValueError, match="Cannot split"):
            hand.split()


# ---------------------------------------------------------------------------
# from_cards constructor
# ---------------------------------------------------------------------------

class TestFromCards:
    def test_empty(self) -> None:
        hand = Hand.from_cards()
        assert len(hand) == 0

    def test_single_card(self) -> None:
        hand = Hand.from_cards(c(Rank.ACE))
        assert len(hand) == 1
        assert hand.cards[0].is_ace

    def test_two_cards(self) -> None:
        hand = Hand.from_cards(c(Rank.ACE), c(Rank.SIX))
        assert hand.value == 17
        assert hand.is_soft is True


# ---------------------------------------------------------------------------
# Bust and blackjack detection
# ---------------------------------------------------------------------------

class TestBustAndBlackjack:
    def test_not_bust_at_21(self) -> None:
        assert h(Rank.TEN, Rank.ACE).is_bust is False

    def test_bust_at_22(self) -> None:
        assert h(Rank.TEN, Rank.TEN, Rank.TWO).is_bust is True

    def test_blackjack_ace_ten(self) -> None:
        assert h(Rank.ACE, Rank.TEN).is_blackjack is True

    def test_blackjack_ace_king(self) -> None:
        assert h(Rank.ACE, Rank.KING).is_blackjack is True

    def test_21_three_cards_not_blackjack(self) -> None:
        assert h(Rank.SEVEN, Rank.SEVEN, Rank.SEVEN).is_blackjack is False

    def test_blackjack_value_21(self) -> None:
        assert h(Rank.ACE, Rank.QUEEN).value == 21

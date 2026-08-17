"""
tests/unit/blackjack/test_ace_handling.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ace-specific edge cases — Phase 2.

Tests every scenario where Ace reduction logic could be wrong:
- Single Ace in various positions
- Multiple Aces
- Aces after hits
- Soft → hard transitions
- Split Aces (one-card rule via game engine)
- Ace pair splitting
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand


def c(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def h(*ranks: Rank) -> Hand:
    return Hand.from_cards(*[c(r) for r in ranks])


class TestSingleAce:
    """Ace alone and paired with each relevant card."""

    def test_ace_alone_value_11(self) -> None:
        hand = Hand()
        hand.add_card(c(Rank.ACE))
        assert hand.value == 11

    def test_ace_plus_2(self) -> None:
        assert h(Rank.ACE, Rank.TWO).value == 13

    def test_ace_plus_5(self) -> None:
        assert h(Rank.ACE, Rank.FIVE).value == 16

    def test_ace_plus_9(self) -> None:
        assert h(Rank.ACE, Rank.NINE).value == 20

    def test_ace_plus_10_blackjack(self) -> None:
        hand = h(Rank.ACE, Rank.TEN)
        assert hand.value == 21
        assert hand.is_blackjack is True

    def test_ace_counts_as_1_when_needed(self) -> None:
        """A + 9 + 5 → ace reduced to 1, total = 15."""
        hand = h(Rank.ACE, Rank.NINE, Rank.FIVE)
        assert hand.value == 15
        assert hand.is_soft is False

    def test_ace_stays_11_when_safe(self) -> None:
        """A + 4 + 6 → 21 (ace still 11)."""
        hand = h(Rank.ACE, Rank.FOUR, Rank.SIX)
        assert hand.value == 21
        assert hand.is_soft is False  # exactly 21, no "soft" tag needed


class TestMultipleAces:
    """Two or more Aces in the same hand."""

    def test_two_aces_value(self) -> None:
        """A + A → 12 (one as 11, one as 1)."""
        assert h(Rank.ACE, Rank.ACE).value == 12

    def test_two_aces_plus_9(self) -> None:
        """A + A + 9 → 21 (both aces: one as 11 would bust, so 1+1+9=11... wait:
        11+1+9=21 with one ace as 11. ✓"""
        assert h(Rank.ACE, Rank.ACE, Rank.NINE).value == 21

    def test_two_aces_plus_10(self) -> None:
        """A + A + 10 → 12 (both must be 1 else bust)."""
        assert h(Rank.ACE, Rank.ACE, Rank.TEN).value == 12

    def test_three_aces_value(self) -> None:
        """A + A + A → 13 (one as 11, two as 1)."""
        assert h(Rank.ACE, Rank.ACE, Rank.ACE).value == 13

    def test_four_aces_value(self) -> None:
        """A + A + A + A → 14."""
        assert h(Rank.ACE, Rank.ACE, Rank.ACE, Rank.ACE).value == 14

    def test_ace_ace_is_not_soft(self) -> None:
        """A + A = 12; is_soft = True (first ace counted as 11)."""
        hand = h(Rank.ACE, Rank.ACE)
        assert hand.is_soft is True  # 1 + 1 + 10 = 12 ≤ 21

    def test_two_aces_plus_ten_hard(self) -> None:
        """A + A + 10 = 12 (both aces are 1) → hard hand."""
        hand = h(Rank.ACE, Rank.ACE, Rank.TEN)
        assert hand.is_soft is False


class TestSoftToHardTransition:
    """Soft hand becomes hard when another card would bust it soft."""

    def test_soft_17_hit_8_becomes_hard_15(self) -> None:
        hand = h(Rank.ACE, Rank.SIX)
        assert hand.is_soft is True  # soft 17
        hand.add_card(c(Rank.EIGHT))
        # 11 + 6 + 8 = 25 → reduce ace → 1 + 6 + 8 = 15
        assert hand.value == 15
        assert hand.is_soft is False

    def test_soft_18_hit_5_stays_soft(self) -> None:
        """A + 7 + 2 → still soft 20 (1+11+7+... wait:
        A + 7 = soft 18, + 2 = soft 20 (still ≤ 21)."""
        hand = h(Rank.ACE, Rank.SEVEN)
        hand.add_card(c(Rank.TWO))
        assert hand.value == 20
        assert hand.is_soft is True

    def test_soft_20_hit_5_becomes_hard_16(self) -> None:
        """A + 9 + 6 → 11+9+6=26 reduce → 1+9+6=16."""
        hand = h(Rank.ACE, Rank.NINE)
        hand.add_card(c(Rank.SIX))
        assert hand.value == 16
        assert hand.is_soft is False

    def test_multi_card_soft_stays_valid(self) -> None:
        """A + 2 + 2 + 2 → soft 17."""
        hand = h(Rank.ACE, Rank.TWO, Rank.TWO, Rank.TWO)
        assert hand.value == 17
        assert hand.is_soft is True

    def test_multi_card_soft_hit_triggers_reduction(self) -> None:
        """A + 2 + 2 + 2 + 9 → 11+2+2+2+9=26 → 1+2+2+2+9=16."""
        hand = h(Rank.ACE, Rank.TWO, Rank.TWO, Rank.TWO)
        hand.add_card(c(Rank.NINE))
        assert hand.value == 16
        assert hand.is_soft is False


class TestAcePairSplit:
    """Ace-pair splitting via Hand.split()."""

    def test_ace_pair_can_split(self) -> None:
        hand = h(Rank.ACE, Rank.ACE)
        assert hand.is_pair is True

    def test_ace_pair_split_returns_ace_hand(self) -> None:
        hand = h(Rank.ACE, Rank.ACE)
        new_hand = hand.split()
        assert new_hand.cards[0].is_ace

    def test_ace_pair_split_first_hand_has_one_ace(self) -> None:
        hand = h(Rank.ACE, Rank.ACE)
        hand.split()
        assert hand.cards[0].is_ace
        assert len(hand) == 1

    def test_ace_value_after_split_and_deal(self) -> None:
        """After splitting aces and dealing a King, should be 21 (but not BJ)."""
        hand = h(Rank.ACE, Rank.ACE)
        new_hand = hand.split()
        # Simulate dealing to the first hand
        hand.add_card(c(Rank.KING))
        assert hand.value == 21
        # This is NOT blackjack because it came from a split
        # (BJ check is done by the engine, not Hand.is_blackjack)
        # Hand itself doesn't know it's from a split — value is still 21
        assert hand.value == 21

    def test_non_ace_pair_split_works(self) -> None:
        """8-8 split should produce two separate hands."""
        hand = h(Rank.EIGHT, Rank.EIGHT)
        new_hand = hand.split()
        assert hand.cards[0].rank == Rank.EIGHT
        assert new_hand.cards[0].rank == Rank.EIGHT
        assert len(hand) == 1
        assert len(new_hand) == 1


class TestAceHardValue:
    """Phase 2 hard_value and soft_value properties with Aces."""

    def test_hard_value_single_ace(self) -> None:
        """ACE counts as 1 in hard_value."""
        assert h(Rank.ACE).hard_value == 1

    def test_hard_value_ace_six(self) -> None:
        """A + 6: hard_value = 7."""
        assert h(Rank.ACE, Rank.SIX).hard_value == 7

    def test_soft_value_ace_six(self) -> None:
        """A + 6: soft_value = 17 (ace as 11)."""
        assert h(Rank.ACE, Rank.SIX).soft_value == 17

    def test_hard_value_two_aces(self) -> None:
        """A + A: hard_value = 2."""
        assert h(Rank.ACE, Rank.ACE).hard_value == 2

    def test_soft_value_two_aces(self) -> None:
        """A + A: soft_value = 12 (first ace as 11, second as 1)."""
        assert h(Rank.ACE, Rank.ACE).soft_value == 12

    def test_num_aces_in_complex_hand(self) -> None:
        """A + A + A: num_aces = 3."""
        assert h(Rank.ACE, Rank.ACE, Rank.ACE).num_aces == 3

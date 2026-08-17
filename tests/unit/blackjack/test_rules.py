"""
tests/unit/blackjack/test_rules.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for BlackjackRules and dealer logic.
"""
from __future__ import annotations

import pytest

from blackjack.rules.rules import BlackjackRules, BlackjackPayout, DealerStandRule


class TestBlackjackRulesFactories:
    def test_standard_rules_defaults(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.num_decks == 6
        assert rules.dealer_stand_rule == DealerStandRule.STAND_SOFT_17
        assert rules.blackjack_payout == BlackjackPayout.THREE_TO_TWO
        assert rules.allow_double_after_split is True
        assert rules.allow_surrender is True

    def test_vegas_downtown(self) -> None:
        rules = BlackjackRules.vegas_downtown()
        assert rules.num_decks == 1
        assert rules.dealer_stand_rule == DealerStandRule.HIT_SOFT_17

    def test_unfavourable(self) -> None:
        rules = BlackjackRules.unfavourable()
        assert rules.blackjack_payout == BlackjackPayout.SIX_TO_FIVE
        assert rules.dealer_stand_rule == DealerStandRule.HIT_SOFT_17

    def test_single_deck(self) -> None:
        rules = BlackjackRules.single_deck()
        assert rules.num_decks == 1


class TestDealerMustHit:
    def test_dealer_hits_below_17(self) -> None:
        rules = BlackjackRules.standard()
        for v in range(4, 17):
            assert rules.dealer_must_hit(v, is_soft=False) is True

    def test_dealer_stands_hard_17_s17(self) -> None:
        rules = BlackjackRules.standard()  # S17
        assert rules.dealer_must_hit(17, is_soft=False) is False

    def test_dealer_stands_soft_17_s17(self) -> None:
        rules = BlackjackRules.standard()  # S17 — stands on soft 17
        assert rules.dealer_must_hit(17, is_soft=True) is False

    def test_dealer_hits_soft_17_h17(self) -> None:
        rules = BlackjackRules.vegas_downtown()  # H17
        assert rules.dealer_must_hit(17, is_soft=True) is True

    def test_dealer_stands_hard_17_h17(self) -> None:
        rules = BlackjackRules.vegas_downtown()  # H17
        assert rules.dealer_must_hit(17, is_soft=False) is False

    def test_dealer_stands_on_18_plus(self) -> None:
        rules = BlackjackRules.standard()
        for v in range(18, 22):
            assert rules.dealer_must_hit(v, is_soft=True) is False
            assert rules.dealer_must_hit(v, is_soft=False) is False


class TestBlackjackMultiplier:
    def test_three_to_two(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.blackjack_multiplier() == 1.5

    def test_six_to_five(self) -> None:
        rules = BlackjackRules.unfavourable()
        assert rules.blackjack_multiplier() == 1.2

    def test_rules_are_immutable(self) -> None:
        rules = BlackjackRules.standard()
        with pytest.raises(Exception):
            rules.num_decks = 1  # type: ignore[misc]

"""
tests/unit/blackjack/test_rule_variations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for BlackjackRules rule-set configurations and factories.
"""
from __future__ import annotations

import pytest

from blackjack.rules.rules import (
    BlackjackRules,
    DealerStandRule,
    BlackjackPayout,
    DoubleRule,
)


class TestRuleFactories:
    def test_standard_rules(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.num_decks == 6
        assert rules.dealer_stand_rule == DealerStandRule.STAND_SOFT_17
        assert rules.blackjack_payout == BlackjackPayout.THREE_TO_TWO
        assert rules.allow_double is True
        assert rules.allow_surrender is True
        assert rules.peek is True

    def test_vegas_downtown_rules(self) -> None:
        rules = BlackjackRules.vegas_downtown()
        assert rules.num_decks == 1
        assert rules.dealer_stand_rule == DealerStandRule.HIT_SOFT_17
        assert rules.blackjack_payout == BlackjackPayout.THREE_TO_TWO

    def test_unfavourable_rules(self) -> None:
        rules = BlackjackRules.unfavourable()
        assert rules.dealer_stand_rule == DealerStandRule.HIT_SOFT_17
        assert rules.blackjack_payout == BlackjackPayout.SIX_TO_FIVE
        assert rules.allow_surrender is False

    def test_single_deck_rules(self) -> None:
        rules = BlackjackRules.single_deck()
        assert rules.num_decks == 1

    def test_atlantic_city_rules(self) -> None:
        rules = BlackjackRules.atlantic_city()
        assert rules.num_decks == 8
        assert rules.allow_resplit_aces is True

    def test_european_rules(self) -> None:
        rules = BlackjackRules.european()
        assert rules.peek is False
        assert rules.allow_surrender is False

    def test_restrictive_double_rules(self) -> None:
        rules = BlackjackRules.restrictive_double()
        assert rules.double_on == DoubleRule.TEN_ELEVEN


class TestDealerMustHit:
    def test_s17_dealer_stands_on_soft_17(self) -> None:
        rules = BlackjackRules(dealer_stand_rule=DealerStandRule.STAND_SOFT_17)
        assert rules.dealer_must_hit(17, is_soft=True) is False

    def test_h17_dealer_hits_on_soft_17(self) -> None:
        rules = BlackjackRules(dealer_stand_rule=DealerStandRule.HIT_SOFT_17)
        assert rules.dealer_must_hit(17, is_soft=True) is True

    def test_dealer_hits_below_17(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.dealer_must_hit(16, is_soft=False) is True
        assert rules.dealer_must_hit(16, is_soft=True) is True

    def test_dealer_stands_on_hard_17(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.dealer_must_hit(17, is_soft=False) is False
        
    def test_dealer_stands_above_17(self) -> None:
        rules = BlackjackRules.standard()
        assert rules.dealer_must_hit(18, is_soft=False) is False
        assert rules.dealer_must_hit(18, is_soft=True) is False


class TestActionLegalityHelpers:
    def test_can_split_max_splits_reached(self) -> None:
        rules = BlackjackRules(max_splits=3)
        assert rules.can_split_on_context(split_count=3, from_split_aces=False) is False
        assert rules.can_split_on_context(split_count=2, from_split_aces=False) is True

    def test_can_resplit_aces(self) -> None:
        # Not allowed
        rules1 = BlackjackRules(allow_resplit_aces=False)
        assert rules1.can_split_on_context(split_count=1, from_split_aces=True) is False
        # Allowed
        rules2 = BlackjackRules(allow_resplit_aces=True)
        assert rules2.can_split_on_context(split_count=1, from_split_aces=True) is True

    def test_can_surrender(self) -> None:
        rules = BlackjackRules(allow_surrender=True, surrender_after_split=False)
        assert rules.can_surrender_in_context(split_count=0) is True
        assert rules.can_surrender_in_context(split_count=1) is False

        rules2 = BlackjackRules(allow_surrender=True, surrender_after_split=True)
        assert rules2.can_surrender_in_context(split_count=1) is True

    def test_can_double_after_split(self) -> None:
        rules1 = BlackjackRules(allow_double_after_split=False)
        assert rules1.can_double_on_hand(10, False, split_count=1) is False
        
        rules2 = BlackjackRules(allow_double_after_split=True)
        assert rules2.can_double_on_hand(10, False, split_count=1) is True

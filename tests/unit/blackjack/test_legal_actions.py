"""
tests/unit/blackjack/test_legal_actions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for LegalActionsCalculator.

Covers every branch of the legal-action computation:
- HIT and STAND always present
- DOUBLE conditions (first action, 2 cards, double_on restrictions, DAS)
- SPLIT conditions (first action, pair, max_splits, resplit aces)
- SURRENDER conditions (first action, 2 cards, after-split block)
- Interaction of multiple rules
"""
from __future__ import annotations

import pytest

from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.engine.state import HandContext
from blackjack.rules.legal_actions import LegalActionsCalculator
from blackjack.rules.rules import BlackjackRules, DoubleRule
from blackjack.strategies.base import Action


def c(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def h(*ranks: Rank) -> Hand:
    return Hand.from_cards(*[c(r) for r in ranks])


def upcard(rank: Rank = Rank.SIX) -> Card:
    return c(rank)


def ctx(
    is_first_action: bool = True,
    split_count: int = 0,
    from_split_aces: bool = False,
    doubled: bool = False,
) -> HandContext:
    return HandContext(
        is_first_action=is_first_action,
        split_count=split_count,
        from_split_aces=from_split_aces,
        doubled=doubled,
    )


@pytest.fixture
def std_calc() -> LegalActionsCalculator:
    return LegalActionsCalculator(BlackjackRules.standard())


# ---------------------------------------------------------------------------
# HIT and STAND are always legal
# ---------------------------------------------------------------------------

class TestBaseActions:
    def test_hit_always_legal(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(h(Rank.SEVEN, Rank.EIGHT), upcard(), ctx())
        assert Action.HIT in legal

    def test_stand_always_legal(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(h(Rank.SEVEN, Rank.EIGHT), upcard(), ctx())
        assert Action.STAND in legal

    def test_hit_and_stand_after_first_action(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(
            h(Rank.SEVEN, Rank.EIGHT, Rank.TWO), upcard(), ctx(is_first_action=False)
        )
        assert Action.HIT in legal
        assert Action.STAND in legal


# ---------------------------------------------------------------------------
# DOUBLE legality
# ---------------------------------------------------------------------------

class TestDouble:
    def test_double_on_first_action(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(h(Rank.SIX, Rank.FIVE), upcard(), ctx())
        assert Action.DOUBLE in legal

    def test_no_double_after_hit(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(
            h(Rank.SIX, Rank.FIVE, Rank.TWO), upcard(), ctx(is_first_action=False)
        )
        assert Action.DOUBLE not in legal

    def test_no_double_with_three_cards(self, std_calc: LegalActionsCalculator) -> None:
        legal = std_calc.get_legal_actions(
            h(Rank.THREE, Rank.THREE, Rank.FIVE), upcard(), ctx()
        )
        assert Action.DOUBLE not in legal

    def test_no_double_when_globally_disabled(self) -> None:
        rules = BlackjackRules(allow_double=False)
        calc = LegalActionsCalculator(rules)
        legal = calc.get_legal_actions(h(Rank.TEN, Rank.ACE), upcard(), ctx())
        assert Action.DOUBLE not in legal

    def test_double_on_9_11_only_allows_11(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.NINE_ELEVEN)
        calc = LegalActionsCalculator(rules)
        # hard 11 → allowed
        assert Action.DOUBLE in calc.get_legal_actions(h(Rank.SIX, Rank.FIVE), upcard(), ctx())
        # hard 8 → not allowed
        assert Action.DOUBLE not in calc.get_legal_actions(h(Rank.THREE, Rank.FIVE), upcard(), ctx())

    def test_double_on_9_11_allows_9(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.NINE_ELEVEN)
        calc = LegalActionsCalculator(rules)
        assert Action.DOUBLE in calc.get_legal_actions(h(Rank.FIVE, Rank.FOUR), upcard(), ctx())

    def test_double_on_10_11_blocks_9(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.TEN_ELEVEN)
        calc = LegalActionsCalculator(rules)
        assert Action.DOUBLE not in calc.get_legal_actions(h(Rank.FOUR, Rank.FIVE), upcard(), ctx())

    def test_double_on_10_11_allows_10(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.TEN_ELEVEN)
        calc = LegalActionsCalculator(rules)
        assert Action.DOUBLE in calc.get_legal_actions(h(Rank.SIX, Rank.FOUR), upcard(), ctx())

    def test_hard_only_blocks_soft_double(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.HARD_ONLY)
        calc = LegalActionsCalculator(rules)
        # A+7 = soft 18 — not allowed with hard_only
        assert Action.DOUBLE not in calc.get_legal_actions(h(Rank.ACE, Rank.SEVEN), upcard(), ctx())
        # Hard 10 — allowed
        assert Action.DOUBLE in calc.get_legal_actions(h(Rank.SIX, Rank.FOUR), upcard(), ctx())

    def test_das_allowed_after_split(self) -> None:
        rules = BlackjackRules(allow_double_after_split=True)
        calc = LegalActionsCalculator(rules)
        legal = calc.get_legal_actions(h(Rank.TEN, Rank.ACE), upcard(), ctx(split_count=1))
        assert Action.DOUBLE in legal

    def test_no_das_after_split(self) -> None:
        rules = BlackjackRules(allow_double_after_split=False)
        calc = LegalActionsCalculator(rules)
        legal = calc.get_legal_actions(h(Rank.TEN, Rank.ACE), upcard(), ctx(split_count=1))
        assert Action.DOUBLE not in legal

    def test_double_on_any_allows_soft(self) -> None:
        rules = BlackjackRules(double_on=DoubleRule.ANY)
        calc = LegalActionsCalculator(rules)
        assert Action.DOUBLE in calc.get_legal_actions(h(Rank.ACE, Rank.SEVEN), upcard(), ctx())


# ---------------------------------------------------------------------------
# SPLIT legality
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_pair_first_action(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SPLIT in std_calc.get_legal_actions(
            h(Rank.EIGHT, Rank.EIGHT), upcard(), ctx()
        )

    def test_no_split_non_pair(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SPLIT not in std_calc.get_legal_actions(
            h(Rank.EIGHT, Rank.NINE), upcard(), ctx()
        )

    def test_no_split_after_hit(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SPLIT not in std_calc.get_legal_actions(
            h(Rank.EIGHT, Rank.EIGHT), upcard(), ctx(is_first_action=False)
        )

    def test_split_respects_max_splits(self) -> None:
        rules = BlackjackRules(max_splits=1)
        calc = LegalActionsCalculator(rules)
        # At split_count=1 (max reached), split not allowed
        assert Action.SPLIT not in calc.get_legal_actions(
            h(Rank.EIGHT, Rank.EIGHT), upcard(), ctx(split_count=1)
        )
        # At split_count=0, split allowed
        assert Action.SPLIT in calc.get_legal_actions(
            h(Rank.EIGHT, Rank.EIGHT), upcard(), ctx(split_count=0)
        )

    def test_split_aces_blocked_by_resplit_rule(self) -> None:
        rules = BlackjackRules(allow_resplit_aces=False)
        calc = LegalActionsCalculator(rules)
        # from_split_aces=True → no resplit
        assert Action.SPLIT not in calc.get_legal_actions(
            h(Rank.ACE, Rank.ACE), upcard(), ctx(split_count=1, from_split_aces=True)
        )

    def test_resplit_aces_when_allowed(self) -> None:
        rules = BlackjackRules(allow_resplit_aces=True, max_splits=3)
        calc = LegalActionsCalculator(rules)
        assert Action.SPLIT in calc.get_legal_actions(
            h(Rank.ACE, Rank.ACE), upcard(), ctx(split_count=1, from_split_aces=True)
        )

    def test_ace_pair_can_split(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SPLIT in std_calc.get_legal_actions(
            h(Rank.ACE, Rank.ACE), upcard(), ctx()
        )


# ---------------------------------------------------------------------------
# SURRENDER legality
# ---------------------------------------------------------------------------

class TestSurrender:
    def test_surrender_first_action(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SURRENDER in std_calc.get_legal_actions(
            h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx()
        )

    def test_no_surrender_after_hit(self, std_calc: LegalActionsCalculator) -> None:
        assert Action.SURRENDER not in std_calc.get_legal_actions(
            h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx(is_first_action=False)
        )

    def test_no_surrender_when_disabled(self) -> None:
        rules = BlackjackRules(allow_surrender=False)
        calc = LegalActionsCalculator(rules)
        assert Action.SURRENDER not in calc.get_legal_actions(
            h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx()
        )

    def test_no_surrender_after_split_by_default(self) -> None:
        # surrender_after_split=False (default) → no surrender after split
        rules = BlackjackRules(allow_surrender=True, surrender_after_split=False)
        calc = LegalActionsCalculator(rules)
        assert Action.SURRENDER not in calc.get_legal_actions(
            h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx(split_count=1)
        )

    def test_surrender_after_split_when_allowed(self) -> None:
        rules = BlackjackRules(allow_surrender=True, surrender_after_split=True)
        calc = LegalActionsCalculator(rules)
        assert Action.SURRENDER in calc.get_legal_actions(
            h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx(split_count=1)
        )

    def test_no_surrender_with_three_cards(self, std_calc: LegalActionsCalculator) -> None:
        """Surrender is only valid on the first two cards."""
        assert Action.SURRENDER not in std_calc.get_legal_actions(
            h(Rank.FIVE, Rank.FIVE, Rank.SIX), upcard(Rank.TEN), ctx()
        )


# ---------------------------------------------------------------------------
# Convenience boolean probes
# ---------------------------------------------------------------------------

class TestConvenienceProbes:
    def test_can_double_probe(self) -> None:
        calc = LegalActionsCalculator(BlackjackRules.standard())
        assert calc.can_double(h(Rank.SIX, Rank.FIVE), upcard(), ctx()) is True

    def test_can_split_probe(self) -> None:
        calc = LegalActionsCalculator(BlackjackRules.standard())
        assert calc.can_split(h(Rank.EIGHT, Rank.EIGHT), upcard(), ctx()) is True

    def test_cannot_split_probe(self) -> None:
        calc = LegalActionsCalculator(BlackjackRules.standard())
        assert calc.can_split(h(Rank.EIGHT, Rank.NINE), upcard(), ctx()) is False

    def test_can_surrender_probe(self) -> None:
        calc = LegalActionsCalculator(BlackjackRules.standard())
        assert calc.can_surrender(h(Rank.NINE, Rank.SEVEN), upcard(Rank.TEN), ctx()) is True

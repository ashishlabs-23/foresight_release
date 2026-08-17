"""
tests/unit/blackjack/test_payout.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for PayoutCalculator.

Covers:
- All HandOutcome values
- Normal vs doubled bets
- Blackjack payout variants (3:2, 6:5)
- Surrender half-bet refund
- expected_value_per_unit helper
"""
from __future__ import annotations

import pytest

from blackjack.engine.outcomes import HandOutcome
from blackjack.rules.payout import PayoutCalculator
from blackjack.rules.rules import BlackjackRules, BlackjackPayout


@pytest.fixture
def standard_calc() -> PayoutCalculator:
    return PayoutCalculator(BlackjackRules.standard())


@pytest.fixture
def six_five_calc() -> PayoutCalculator:
    return PayoutCalculator(BlackjackRules.unfavourable())


class TestNormalPayouts:
    def test_win_one_unit(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.WIN, bet=1.0) == pytest.approx(1.0)

    def test_loss_minus_one_unit(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.LOSS, bet=1.0) == pytest.approx(-1.0)

    def test_push_zero(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.PUSH, bet=1.0) == pytest.approx(0.0)

    def test_win_arbitrary_bet(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.WIN, bet=25.0) == pytest.approx(25.0)

    def test_loss_arbitrary_bet(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.LOSS, bet=50.0) == pytest.approx(-50.0)


class TestBlackjackPayout:
    def test_three_to_two(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.BLACKJACK, bet=1.0) == pytest.approx(1.5)

    def test_three_to_two_on_large_bet(self, standard_calc: PayoutCalculator) -> None:
        assert standard_calc.net(HandOutcome.BLACKJACK, bet=100.0) == pytest.approx(150.0)

    def test_six_to_five(self, six_five_calc: PayoutCalculator) -> None:
        assert six_five_calc.net(HandOutcome.BLACKJACK, bet=1.0) == pytest.approx(1.2)

    def test_six_to_five_on_large_bet(self, six_five_calc: PayoutCalculator) -> None:
        assert six_five_calc.net(HandOutcome.BLACKJACK, bet=100.0) == pytest.approx(120.0)

    def test_three_to_two_uses_original_bet(self, standard_calc: PayoutCalculator) -> None:
        """BJ payout is on original_bet, not current bet."""
        payout = standard_calc.net(
            HandOutcome.BLACKJACK, bet=1.0, original_bet=1.0
        )
        assert payout == pytest.approx(1.5)


class TestDoubledPayouts:
    def test_doubled_win(self, standard_calc: PayoutCalculator) -> None:
        """Doubled bet = 2.0; win → +2.0"""
        assert standard_calc.net(HandOutcome.WIN, bet=2.0) == pytest.approx(2.0)

    def test_doubled_loss(self, standard_calc: PayoutCalculator) -> None:
        """Doubled bet = 2.0; loss → -2.0"""
        assert standard_calc.net(HandOutcome.LOSS, bet=2.0) == pytest.approx(-2.0)

    def test_doubled_push(self, standard_calc: PayoutCalculator) -> None:
        """Doubled bet; push → 0.0"""
        assert standard_calc.net(HandOutcome.PUSH, bet=2.0) == pytest.approx(0.0)


class TestSurrender:
    def test_surrender_half_original_bet(self, standard_calc: PayoutCalculator) -> None:
        payout = standard_calc.net(HandOutcome.SURRENDER, original_bet=1.0)
        assert payout == pytest.approx(-0.5)

    def test_surrender_always_uses_original_bet(self, standard_calc: PayoutCalculator) -> None:
        """Surrender refunds half original, not current bet."""
        payout = standard_calc.net(
            HandOutcome.SURRENDER, bet=1.0, original_bet=1.0
        )
        assert payout == pytest.approx(-0.5)

    def test_surrender_large_bet(self, standard_calc: PayoutCalculator) -> None:
        payout = standard_calc.net(HandOutcome.SURRENDER, original_bet=100.0)
        assert payout == pytest.approx(-50.0)


class TestExpectedValue:
    def test_zero_ev_with_balanced_outcomes(self, standard_calc: PayoutCalculator) -> None:
        ev = standard_calc.expected_value_per_unit(
            win_prob=0.5, lose_prob=0.5, push_prob=0.0
        )
        assert ev == pytest.approx(0.0)

    def test_positive_ev_more_wins(self, standard_calc: PayoutCalculator) -> None:
        ev = standard_calc.expected_value_per_unit(
            win_prob=0.6, lose_prob=0.4, push_prob=0.0
        )
        assert ev > 0

    def test_blackjack_adds_ev(self, standard_calc: PayoutCalculator) -> None:
        """Adding blackjack probability increases EV."""
        ev_no_bj = standard_calc.expected_value_per_unit(0.43, 0.48, 0.09)
        ev_with_bj = standard_calc.expected_value_per_unit(0.38, 0.48, 0.09, bj_prob=0.048)
        # BJ pays 1.5× so it should add to EV
        assert ev_with_bj > ev_no_bj

    def test_surrender_reduces_ev(self, standard_calc: PayoutCalculator) -> None:
        ev_no_surr = standard_calc.expected_value_per_unit(0.43, 0.53, 0.04)
        ev_with_surr = standard_calc.expected_value_per_unit(0.43, 0.48, 0.04, surrender_prob=0.05)
        # Surrender (-0.5) is better than losing (-1.0) so EV should be higher
        assert ev_with_surr > ev_no_surr


class TestPayoutCalculatorInit:
    def test_three_to_two_rule(self) -> None:
        calc = PayoutCalculator(BlackjackRules(blackjack_payout=BlackjackPayout.THREE_TO_TWO))
        assert calc.net(HandOutcome.BLACKJACK, bet=1.0) == pytest.approx(1.5)

    def test_six_to_five_rule(self) -> None:
        calc = PayoutCalculator(BlackjackRules(blackjack_payout=BlackjackPayout.SIX_TO_FIVE))
        assert calc.net(HandOutcome.BLACKJACK, bet=1.0) == pytest.approx(1.2)

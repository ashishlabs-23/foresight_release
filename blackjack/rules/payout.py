"""
blackjack.rules.payout
~~~~~~~~~~~~~~~~~~~~~~~~
PayoutCalculator — calculates net monetary gain/loss for a hand outcome.

All amounts are expressed in units of the *original* bet placed for that hand.

Payout table
------------
+-------------------+------------------+-------------------------------+
| Outcome           | Normal hand      | Doubled hand (bet * 2)        |
+===================+==================+===============================+
| WIN               | +bet             | +bet * 2                      |
| LOSS              | -bet             | -bet * 2                      |
| PUSH              | 0                | 0                             |
| BLACKJACK (3:2)   | +bet * 1.5       | n/a (BJ is pre-double)        |
| BLACKJACK (6:5)   | +bet * 1.2       | n/a                           |
| SURRENDER         | -original_bet/2  | n/a (surrender is pre-double) |
+-------------------+------------------+-------------------------------+

Notes
-----
* ``bet`` in the context object tracks the CURRENT bet (already doubled if
  the player doubled down).  ``original_bet`` is the starting amount.
* Surrender is always half the ORIGINAL bet (players cannot double then
  surrender — those are mutually exclusive first actions).
* Blackjack can only occur on the initial two-card deal; the bet is never
  doubled at that point.
"""
from __future__ import annotations

from blackjack.engine.outcomes import HandOutcome
from blackjack.rules.rules import BlackjackRules


class PayoutCalculator:
    """Converts a HandOutcome + bet amount to a net monetary gain/loss.

    Parameters
    ----------
    rules : Active rule configuration (needed for BJ payout ratio).

    Example
    -------
    >>> from blackjack.engine.outcomes import HandOutcome
    >>> from blackjack.rules.rules import BlackjackRules
    >>> calc = PayoutCalculator(BlackjackRules.standard())
    >>> calc.net(HandOutcome.BLACKJACK, bet=1.0)
    1.5
    >>> calc.net(HandOutcome.WIN, bet=2.0)   # doubled win
    2.0
    >>> calc.net(HandOutcome.SURRENDER, original_bet=1.0)
    -0.5
    """

    def __init__(self, rules: BlackjackRules) -> None:
        self._rules = rules

    def net(
        self,
        outcome: HandOutcome,
        bet: float = 1.0,
        original_bet: float | None = None,
    ) -> float:
        """Calculate net gain/loss in monetary units.

        Parameters
        ----------
        outcome      : The resolved outcome of the hand.
        bet          : Current bet (post-double if applicable).
        original_bet : Bet before any double (used only for surrender).
                       Defaults to ``bet`` when not provided.

        Returns
        -------
        float
            Positive = player wins that amount.
            Negative = player loses that amount.
        """
        if original_bet is None:
            original_bet = bet

        match outcome:
            case HandOutcome.WIN:
                return +bet
            case HandOutcome.LOSS:
                return -bet
            case HandOutcome.PUSH:
                return 0.0
            case HandOutcome.BLACKJACK:
                # BJ payout is on the original (non-doubled) bet
                return original_bet * self._rules.blackjack_multiplier()
            case HandOutcome.SURRENDER:
                # Surrender refunds half the original bet
                return -(original_bet / 2.0)
            case _:
                return 0.0

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def expected_value_per_unit(
        self,
        win_prob: float,
        lose_prob: float,
        push_prob: float,
        bj_prob: float = 0.0,
        surrender_prob: float = 0.0,
    ) -> float:
        """Theoretical expected value per unit wagered.

        Parameters
        ----------
        win_prob      : Probability of a regular win.
        lose_prob     : Probability of a loss.
        push_prob     : Probability of a push.
        bj_prob       : Probability of player blackjack.
        surrender_prob: Probability of surrender.

        Returns
        -------
        float
            Expected net gain per unit bet. Negative = house edge.
        """
        bj_mult = self._rules.blackjack_multiplier()
        return (
            win_prob * 1.0
            + lose_prob * (-1.0)
            + push_prob * 0.0
            + bj_prob * bj_mult
            + surrender_prob * (-0.5)
        )

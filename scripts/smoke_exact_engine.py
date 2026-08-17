"""Offline smoke test for the exact decision engine.

Run with:
    PYTHONPATH=. python scripts/smoke_exact_engine.py
"""
from blackjack.analysis.exact_ev import ExactEVAnalyzer
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import Action


def main() -> None:
    analyzer = ExactEVAnalyzer(BlackjackRules(num_decks=6))

    cases = [
        (("10", "6"), "7", "hit"),
        (("10", "10"), "7", "stand"),
        (("A", "7"), "9", "hit"),
    ]
    for player, dealer, expected in cases:
        observed = list(player) + [dealer]
        results = analyzer.evaluate(
            list(player), dealer, observed, 6,
            [Action.HIT, Action.STAND, Action.DOUBLE],
        )
        best = max(results, key=lambda name: results[name].ev)
        assert best == expected, (player, dealer, best, expected)

    bj = analyzer.evaluate(["A", "K"], "6", ["A", "K", "6"], 6, [Action.STAND])
    assert abs(bj["stand"].ev - 1.5) < 1e-9
    print("Exact engine smoke test: PASS")


if __name__ == "__main__":
    main()

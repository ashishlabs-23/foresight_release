"""
tests/unit/blackjack/test_simulator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Simulator, SimConfig, and SimResult.
"""
from __future__ import annotations

import pytest

from blackjack.simulation.simulator import SimConfig, SimResult, Simulator


class TestSimConfig:
    def test_default_config(self) -> None:
        cfg = SimConfig()
        assert cfg.num_hands == 1_000
        assert cfg.strategy_name == "basic"
        assert cfg.num_decks == 6

    def test_invalid_num_hands(self) -> None:
        with pytest.raises(ValueError, match="num_hands"):
            SimConfig(num_hands=0)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            SimConfig(strategy_name="nonexistent")

    def test_valid_strategies(self) -> None:
        for name in ["basic", "random"]:
            cfg = SimConfig(strategy_name=name)
            assert cfg.strategy_name == name


class TestSimulator:
    def test_simulator_runs_correct_number_of_hands(self) -> None:
        cfg = SimConfig(num_hands=50, strategy_name="basic", seed=42)
        sim = Simulator(cfg)
        result = sim.run()
        assert result.total_hands == 50

    def test_result_rates_sum_to_one(self) -> None:
        cfg = SimConfig(num_hands=200, strategy_name="basic", seed=42)
        result = Simulator(cfg).run()
        total = result.win_rate + result.loss_rate + result.push_rate
        assert abs(total - 1.0) < 0.01, f"Rates should sum to ~1.0, got {total}"

    def test_reproducible_with_seed(self) -> None:
        cfg1 = SimConfig(num_hands=100, strategy_name="basic", seed=7)
        cfg2 = SimConfig(num_hands=100, strategy_name="basic", seed=7)
        r1 = Simulator(cfg1).run()
        r2 = Simulator(cfg2).run()
        assert r1.total_payout == r2.total_payout

    def test_different_seeds_differ(self) -> None:
        r1 = Simulator(SimConfig(num_hands=200, seed=1)).run()
        r2 = Simulator(SimConfig(num_hands=200, seed=2)).run()
        # Extremely unlikely to match with 200 hands
        assert r1.total_payout != r2.total_payout

    def test_random_strategy_runs(self) -> None:
        cfg = SimConfig(num_hands=50, strategy_name="random", seed=42)
        result = Simulator(cfg).run()
        assert result.total_hands == 50

    def test_house_edge_basic_strategy_reasonable(self) -> None:
        """Basic strategy house edge should be < 5% (theoretically ~0.5%)."""
        cfg = SimConfig(num_hands=5_000, strategy_name="basic", seed=42)
        result = Simulator(cfg).run()
        assert result.house_edge < 0.05, f"House edge too high: {result.house_edge:.4%}"

    def test_elapsed_seconds_positive(self) -> None:
        cfg = SimConfig(num_hands=10, seed=42)
        result = Simulator(cfg).run()
        assert result.elapsed_seconds > 0

    def test_hands_per_second_positive(self) -> None:
        cfg = SimConfig(num_hands=10, seed=42)
        result = Simulator(cfg).run()
        assert result.hands_per_second > 0

    def test_summary_contains_key_info(self) -> None:
        cfg = SimConfig(num_hands=10, seed=42)
        result = Simulator(cfg).run()
        summary = result.summary()
        assert "Win rate" in summary
        assert "House edge" in summary

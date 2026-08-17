"""
tests/unit/backend/test_schemas.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Pydantic request/response schemas.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.simulation import SimulationRequest, SimulationResponse


class TestSimulationRequest:
    def test_default_values(self) -> None:
        req = SimulationRequest()
        assert req.num_hands == 1_000
        assert req.strategy == "basic"
        assert req.num_decks == 6
        assert req.rules_variant == "standard"
        assert req.seed is None

    def test_valid_strategies(self) -> None:
        for s in ["basic", "random"]:
            req = SimulationRequest(strategy=s)
            assert req.strategy == s

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(strategy="cheating")

    def test_valid_rules_variants(self) -> None:
        for v in ["standard", "vegas_downtown", "unfavourable"]:
            req = SimulationRequest(rules_variant=v)
            assert req.rules_variant == v

    def test_invalid_rules_variant_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(rules_variant="atlantis_casino")

    def test_num_hands_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(num_hands=0)

    def test_num_hands_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(num_hands=2_000_000)

    def test_num_decks_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(num_decks=0)
        with pytest.raises(ValidationError):
            SimulationRequest(num_decks=9)

    def test_seed_can_be_none(self) -> None:
        req = SimulationRequest(seed=None)
        assert req.seed is None

    def test_seed_can_be_int(self) -> None:
        req = SimulationRequest(seed=42)
        assert req.seed == 42


class TestSimulationResponse:
    def test_valid_response(self) -> None:
        resp = SimulationResponse(
            total_hands=1000,
            strategy="basic",
            rules_variant="standard",
            win_rate=0.43,
            loss_rate=0.49,
            push_rate=0.08,
            blackjack_rate=0.048,
            house_edge=0.005,
            elapsed_seconds=0.5,
            hands_per_second=2000.0,
        )
        assert resp.total_hands == 1000
        assert resp.strategy == "basic"

    def test_response_serialises_to_dict(self) -> None:
        resp = SimulationResponse(
            total_hands=100,
            strategy="random",
            rules_variant="standard",
            win_rate=0.4,
            loss_rate=0.5,
            push_rate=0.1,
            blackjack_rate=0.04,
            house_edge=0.02,
            elapsed_seconds=0.1,
            hands_per_second=1000.0,
        )
        d = resp.model_dump()
        assert isinstance(d, dict)
        assert "win_rate" in d
        assert "house_edge" in d

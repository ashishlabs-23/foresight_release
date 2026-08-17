"""
tests/integration/test_simulation_api.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the full simulation API flow.

These tests exercise: HTTP request → FastAPI router → service → blackjack engine → response.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from backend.app.main import app
    return TestClient(app)


class TestSimulationEndpoint:
    def test_simulate_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"num_hands": 100, "strategy": "basic", "seed": 42},
        )
        assert response.status_code == 200

    def test_simulate_response_schema(self, client: TestClient) -> None:
        data = client.post(
            "/api/v1/simulate",
            json={"num_hands": 50, "strategy": "basic", "seed": 1},
        ).json()

        required_fields = [
            "total_hands", "strategy", "rules_variant",
            "win_rate", "loss_rate", "push_rate",
            "blackjack_rate", "house_edge",
            "elapsed_seconds", "hands_per_second",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_simulate_total_hands_matches_request(self, client: TestClient) -> None:
        data = client.post(
            "/api/v1/simulate",
            json={"num_hands": 75, "strategy": "basic", "seed": 99},
        ).json()
        assert data["total_hands"] == 75

    def test_simulate_rates_sum_to_one(self, client: TestClient) -> None:
        data = client.post(
            "/api/v1/simulate",
            json={"num_hands": 200, "strategy": "basic", "seed": 7},
        ).json()
        total = data["win_rate"] + data["loss_rate"] + data["push_rate"]
        assert abs(total - 1.0) < 0.02, f"Rates sum: {total}"

    def test_simulate_random_strategy(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"num_hands": 50, "strategy": "random", "seed": 42},
        )
        assert response.status_code == 200

    def test_simulate_vegas_downtown_rules(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"num_hands": 50, "rules_variant": "vegas_downtown", "seed": 1},
        )
        assert response.status_code == 200
        assert response.json()["rules_variant"] == "vegas_downtown"

    def test_simulate_invalid_strategy_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"num_hands": 100, "strategy": "cheat"},
        )
        assert response.status_code == 422

    def test_simulate_invalid_num_hands_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"num_hands": 0},
        )
        assert response.status_code == 422

    def test_simulate_reproducibility_via_api(self, client: TestClient) -> None:
        """Same seed → same result on two separate HTTP calls."""
        payload = {"num_hands": 100, "strategy": "basic", "seed": 55}
        r1 = client.post("/api/v1/simulate", json=payload).json()
        r2 = client.post("/api/v1/simulate", json=payload).json()
        assert r1["win_rate"] == r2["win_rate"]
        assert r1["house_edge"] == r2["house_edge"]

    def test_health_and_simulate_both_reachable(self, client: TestClient) -> None:
        assert client.get("/api/v1/health").status_code == 200
        assert client.post(
            "/api/v1/simulate", json={"num_hands": 10, "seed": 1}
        ).status_code == 200

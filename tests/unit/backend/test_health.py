"""
tests/unit/backend/test_health.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the health check endpoint.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from backend.app.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data

    def test_health_status_ok(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"

    def test_health_version_present(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["version"] != ""

    def test_health_environment_valid(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["environment"] in ("development", "staging", "production")

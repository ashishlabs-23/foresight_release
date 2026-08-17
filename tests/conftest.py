"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures for all test suites.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure we always load development settings in tests
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep test output clean


@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient for integration tests.

    Session-scoped so the app is only created once per test run.
    """
    from backend.app.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def standard_shoe():
    """A fresh 6-deck shoe with a fixed seed."""
    from blackjack.cards.deck import Shoe
    return Shoe(num_decks=6, seed=42)


@pytest.fixture
def single_deck_shoe():
    """A fresh single-deck shoe with a fixed seed."""
    from blackjack.cards.deck import Shoe
    return Shoe(num_decks=1, seed=42)


@pytest.fixture
def standard_rules():
    """Standard Vegas Strip rules."""
    from blackjack.rules.rules import BlackjackRules
    return BlackjackRules.standard()


@pytest.fixture
def basic_strategy():
    """BasicStrategy instance."""
    from blackjack.strategies.basic import BasicStrategy
    return BasicStrategy()


@pytest.fixture
def random_strategy():
    """RandomStrategy with a fixed seed."""
    from blackjack.strategies.random_strategy import RandomStrategy
    return RandomStrategy(seed=42)

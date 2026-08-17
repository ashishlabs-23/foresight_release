"""
blackjack — Pure Blackjack game engine.

Layers
------
cards/      Card, Deck, Shoe, Hand
rules/      BlackjackRules variants
strategies/ BaseStrategy, BasicStrategy, RandomStrategy
engine/     GameEngine (orchestrates a full hand)
simulation/ Simulator (batch runner)

This package has NO dependency on ml/, backend/, or any HTTP framework.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("blackjack-ai")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]

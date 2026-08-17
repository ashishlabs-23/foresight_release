"""
ml/features/derived.py
~~~~~~~~~~~~~~~~~~~~~~
DerivedFeatures schema definition.
Extracts logical abstractions from the GameState.
"""
from pydantic import BaseModel, Field


class DerivedFeatures(BaseModel):
    """Features logically derived from the raw GameState without external info."""
    
    player_total: int = Field(..., ge=2, le=30, description="Best hard/soft total of player cards")
    is_soft: bool = Field(..., description="Whether the player total includes an Ace counted as 11")
    is_pair: bool = Field(..., description="Whether the player hand is a pair (can be split)")
    num_cards: int = Field(..., ge=1, description="Number of cards in player hand")
    
    decks_total: float = Field(..., gt=0.0, description="Total number of decks in the shoe")
    decks_remaining: float = Field(..., ge=0.0, description="Estimated decks remaining (cards / 52)")
    true_count: float = Field(..., description="Running count divided by decks remaining")
    penetration: float = Field(..., ge=0.0, le=1.0, description="Fraction of shoe dealt")

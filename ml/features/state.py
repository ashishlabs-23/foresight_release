"""
ml/features/state.py
~~~~~~~~~~~~~~~~~~~~
Raw observable GameState definition using Pydantic.
This represents the pure observation a player has at any decision point.
"""
from typing import Any
from pydantic import BaseModel, Field, field_validator


class GameState(BaseModel):
    """Canonical representation of the raw observable state."""
    
    # Cards represented as strings, e.g., 'TH', 'AS', '2D'
    player_cards: list[str] | None = Field(default=None, description="Player's current cards")
    dealer_upcard: str | None = Field(default=None, description="Dealer's visible upcard")
    
    # Ranks (for datasets lacking suit information)
    player_ranks: list[str] | None = Field(default=None, description="Player's current card ranks (e.g. ['5', 'T'])")
    dealer_upcard_rank: str | None = Field(default=None, description="Dealer's visible upcard rank")
    
    # Shoe state
    shoe_total_cards: int = Field(..., ge=52, description="Total cards in a freshly shuffled shoe")
    shoe_cards_remaining: int = Field(..., ge=0, description="Cards left in the shoe")
    running_count: int = Field(..., description="Hi-Lo running count of all revealed cards")
    
    # Game rules
    rules: dict[str, Any] = Field(..., description="Serialized BlackjackRules configuration")
    
    @field_validator("player_cards")
    @classmethod
    def validate_cards(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for card in v:
            if len(card) != 2:
                raise ValueError(f"Invalid card format: {card}")
            if card[0] not in "23456789TJQKA":
                raise ValueError(f"Invalid rank in card: {card}")
            if card[1] not in "HDCS":
                raise ValueError(f"Invalid suit in card: {card}")
        return v
        
    @field_validator("dealer_upcard")
    @classmethod
    def validate_upcard(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(f"Invalid dealer upcard format: {v}")
        if v[0] not in "23456789TJQKA":
            raise ValueError(f"Invalid rank in dealer upcard: {v}")
        if v[1] not in "HDCS":
            raise ValueError(f"Invalid suit in dealer upcard: {v}")
        return v
        
    def get_player_ranks(self) -> list[str]:
        """Returns the list of ranks, extracting from player_cards if needed."""
        if self.player_ranks is not None:
            return self.player_ranks
        if self.player_cards is not None:
            return [card[0] for card in self.player_cards]
        raise ValueError("GameState must have either player_cards or player_ranks")
        
    def get_dealer_rank(self) -> str:
        """Returns the dealer upcard rank, extracting from dealer_upcard if needed."""
        if self.dealer_upcard_rank is not None:
            return self.dealer_upcard_rank
        if self.dealer_upcard is not None:
            return self.dealer_upcard[0]
        raise ValueError("GameState must have either dealer_upcard or dealer_upcard_rank")

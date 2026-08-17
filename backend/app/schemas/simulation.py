"""
backend.app.schemas.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic schemas for simulation request and response.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SimulationRequest(BaseModel):
    """Request body for POST /api/v1/simulate."""

    num_hands: int = Field(
        default=1_000,
        ge=1,
        le=1_000_000,
        description="Number of hands to simulate.",
    )
    strategy: str = Field(
        default="basic",
        description="Strategy name: 'basic' or 'random'.",
    )
    num_decks: int = Field(
        default=6,
        ge=1,
        le=8,
        description="Number of decks in the shoe.",
    )
    rules_variant: str = Field(
        default="standard",
        description="Rule variant: 'standard', 'vegas_downtown', or 'unfavourable'.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. None = random.",
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"basic", "random"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("rules_variant")
    @classmethod
    def validate_rules_variant(cls, v: str) -> str:
        allowed = {"standard", "vegas_downtown", "unfavourable"}
        if v not in allowed:
            raise ValueError(f"rules_variant must be one of {sorted(allowed)}, got '{v}'")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "num_hands": 10000,
            "strategy": "basic",
            "num_decks": 6,
            "rules_variant": "standard",
            "seed": 42,
        }
    }}


class SimulationResponse(BaseModel):
    """Response body from POST /api/v1/simulate."""

    total_hands: int
    strategy: str
    rules_variant: str
    win_rate: float = Field(description="Fraction of hands won (excluding pushes).")
    loss_rate: float = Field(description="Fraction of hands lost.")
    push_rate: float = Field(description="Fraction of hands pushed.")
    blackjack_rate: float = Field(description="Fraction of natural blackjacks dealt to player.")
    house_edge: float = Field(description="Expected loss per hand (positive = player disadvantage).")
    elapsed_seconds: float
    hands_per_second: float

    model_config = {"json_schema_extra": {
        "example": {
            "total_hands": 10000,
            "strategy": "basic",
            "rules_variant": "standard",
            "win_rate": 0.425,
            "loss_rate": 0.491,
            "push_rate": 0.084,
            "blackjack_rate": 0.048,
            "house_edge": 0.0051,
            "elapsed_seconds": 0.42,
            "hands_per_second": 23809.52,
        }
    }}


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "ok"
    version: str
    environment: str

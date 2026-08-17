"""
backend.app.api.v1.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulation endpoints.

POST /api/v1/simulate
  Body   : SimulationRequest
  Returns: SimulationResponse
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.simulation import SimulationRequest, SimulationResponse
from backend.app.services.simulation_service import simulation_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Simulation"])


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a Blackjack simulation",
    description=(
        "Simulates N hands of Blackjack using the specified strategy and rule variant. "
        "Returns aggregate statistics including win rate and house edge."
    ),
)
def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Execute a batch Blackjack simulation and return aggregate results."""
    logger.info(
        "Simulation request received",
        num_hands=request.num_hands,
        strategy=request.strategy,
    )

    try:
        result = simulation_service.run_simulation(
            num_hands=request.num_hands,
            strategy_name=request.strategy,
            num_decks=request.num_decks,
            rules_variant=request.rules_variant,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during simulation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation failed — see server logs for details.",
        ) from exc

    return SimulationResponse(
        total_hands=result.total_hands,
        strategy=request.strategy,
        rules_variant=request.rules_variant,
        win_rate=result.win_rate,
        loss_rate=result.loss_rate,
        push_rate=result.push_rate,
        blackjack_rate=result.blackjack_rate,
        house_edge=result.house_edge,
        elapsed_seconds=result.elapsed_seconds,
        hands_per_second=result.hands_per_second,
    )

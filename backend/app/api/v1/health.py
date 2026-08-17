"""
backend.app.api.v1.health
~~~~~~~~~~~~~~~~~~~~~~~~~
Health check endpoint.

GET /api/v1/health
  Returns: { "status": "ok", "version": "0.1.0", "environment": "development" }
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.schemas.simulation import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service status, version, and current environment.",
)
def health_check() -> HealthResponse:
    """Liveness probe for load balancers and monitoring tools."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
    )

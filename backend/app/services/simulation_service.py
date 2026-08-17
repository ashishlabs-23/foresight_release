"""
backend.app.services.simulation_service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin service wrapping the blackjack simulation engine.

This is the ONLY place in backend/ that imports from blackjack/.
The API layer (routes) calls this service and never touches the engine directly.
"""
from __future__ import annotations

import logging

from blackjack.simulation.simulator import SimConfig, SimResult, Simulator

logger = logging.getLogger(__name__)


class SimulationService:
    """Orchestrates simulation runs on behalf of the API layer.

    In Phase 1 this is a thin wrapper.
    In Phase 4 it will add caching, database persistence, and async support.
    """

    def run_simulation(
        self,
        num_hands: int,
        strategy_name: str = "basic",
        num_decks: int = 6,
        rules_variant: str = "standard",
        seed: int | None = None,
    ) -> SimResult:
        """Run a simulation with the given parameters.

        Parameters
        ----------
        num_hands     : Number of hands to simulate.
        strategy_name : 'basic' or 'random'.
        num_decks     : Number of decks in the shoe.
        rules_variant : 'standard' | 'vegas_downtown' | 'unfavourable'.
        seed          : Optional reproducibility seed.

        Returns
        -------
        SimResult — aggregate simulation statistics.
        """
        logger.info(
            "SimulationService.run_simulation called",
            num_hands=num_hands,
            strategy=strategy_name,
        )

        config = SimConfig(
            num_hands=num_hands,
            strategy_name=strategy_name,
            num_decks=num_decks,
            rules_variant=rules_variant,
            seed=seed,
        )
        simulator = Simulator(config)
        return simulator.run()


# Module-level singleton (stateless, safe to share)
simulation_service = SimulationService()

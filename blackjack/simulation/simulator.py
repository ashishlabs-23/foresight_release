"""
blackjack.simulation.simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Batch simulator — runs N rounds and returns aggregate statistics.

This is the primary interface between the game engine and all consumers
(backend services, ML training data generation, CLI scripts).

Usage
-----
>>> from blackjack.simulation import Simulator, SimConfig
>>> config = SimConfig(num_hands=10_000, strategy_name="basic", seed=42)
>>> sim = Simulator(config)
>>> result = sim.run()
>>> print(f"House edge: {result.house_edge:.2%}")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from blackjack.cards.deck import Shoe
from blackjack.engine.game import GameEngine
from blackjack.engine.outcomes import HandOutcome, HandResult
from blackjack.rules.rules import BlackjackRules
from blackjack.strategies.base import BaseStrategy
from blackjack.strategies.basic import BasicStrategy
from blackjack.strategies.random_strategy import RandomStrategy

logger = logging.getLogger(__name__)

# Registry of available strategies by name
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "basic": BasicStrategy,
    "random": RandomStrategy,
}


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a simulation run.

    Attributes
    ----------
    num_hands       : Number of rounds to simulate.
    strategy_name   : Strategy identifier — 'basic' or 'random'.
    num_decks       : Decks in the shoe.
    penetration     : Shoe reshuffle penetration.
    seed            : Random seed (None = non-deterministic).
    rules_variant   : 'standard' | 'vegas_downtown' | 'unfavourable'
    """

    num_hands: int = 1_000
    strategy_name: str = "basic"
    num_decks: int = 6
    penetration: float = 0.75
    seed: int | None = None
    rules_variant: str = "standard"

    def __post_init__(self) -> None:
        if self.num_hands < 1:
            raise ValueError(f"num_hands must be >= 1, got {self.num_hands}")
        if self.strategy_name not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{self.strategy_name}'. "
                f"Available: {list(_STRATEGY_REGISTRY)}"
            )


@dataclass
class SimResult:
    """Aggregate results from a simulation run.

    All rates are expressed as fractions (0.0–1.0) based on the
    number of *rounds* played (not individual split hands).
    """

    config: SimConfig
    total_hands: int = 0          # rounds played
    total_splits: int = 0         # extra hands produced by splits
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blackjacks: int = 0
    surrenders: int = 0
    player_busts: int = 0
    dealer_busts: int = 0
    total_payout: float = 0.0     # net, summed across all hands (inc. splits)
    elapsed_seconds: float = 0.0
    hand_results: list[HandResult] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def total_player_hands(self) -> int:
        """Total number of individual hands played (rounds + splits)."""
        return self.total_hands + self.total_splits

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_hands if self.total_hands else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total_hands if self.total_hands else 0.0

    @property
    def push_rate(self) -> float:
        return self.pushes / self.total_hands if self.total_hands else 0.0

    @property
    def blackjack_rate(self) -> float:
        return self.blackjacks / self.total_hands if self.total_hands else 0.0

    @property
    def player_bust_rate(self) -> float:
        """Percentage of individual player hands that busted."""
        return self.player_busts / self.total_player_hands if self.total_player_hands else 0.0

    @property
    def dealer_bust_rate(self) -> float:
        """Percentage of rounds where the dealer busted."""
        return self.dealer_busts / self.total_hands if self.total_hands else 0.0

    @property
    def house_edge(self) -> float:
        """Expected value per round (negative = house advantage)."""
        return -(self.total_payout / self.total_hands) if self.total_hands else 0.0

    @property
    def hands_per_second(self) -> float:
        return self.total_hands / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def summary(self) -> str:
        split_note = f" ({self.total_splits} split hands)" if self.total_splits else ""
        return (
            f"Simulation Summary\n"
            f"  Strategy    : {self.config.strategy_name}\n"
            f"  Hands       : {self.total_hands:,}{split_note}\n"
            f"  Win rate    : {self.win_rate:.2%}\n"
            f"  Loss rate   : {self.loss_rate:.2%}\n"
            f"  Push rate   : {self.push_rate:.2%}\n"
            f"  Blackjacks  : {self.blackjack_rate:.2%}\n"
            f"  Player busts: {self.player_bust_rate:.2%}\n"
            f"  Dealer busts: {self.dealer_bust_rate:.2%}\n"
            f"  House edge  : {self.house_edge:.4%}\n"
            f"  Speed       : {self.hands_per_second:,.0f} hands/sec\n"
            f"  Elapsed     : {self.elapsed_seconds:.2f}s"
        )


class Simulator:
    """Runs a batch of Blackjack rounds using a given strategy and rule-set.

    Parameters
    ----------
    config : SimConfig describing the run.
    """

    _RULES_FACTORIES = {
        "standard": BlackjackRules.standard,
        "vegas_downtown": BlackjackRules.vegas_downtown,
        "unfavourable": BlackjackRules.unfavourable,
    }

    def __init__(self, config: SimConfig) -> None:
        self._config = config

    def run(self) -> SimResult:
        """Execute the simulation and return aggregate results."""
        config = self._config
        logger.info(
            f"Starting simulation - num_hands={config.num_hands}, strategy={config.strategy_name}, seed={config.seed}"
        )

        rules_factory = self._RULES_FACTORIES.get(
            config.rules_variant, BlackjackRules.standard
        )
        rules = rules_factory()

        shoe = Shoe(
            num_decks=config.num_decks,
            reshuffle_penetration=config.penetration,
            seed=config.seed,
        )

        strategy_cls = _STRATEGY_REGISTRY[config.strategy_name]
        strategy = strategy_cls(seed=config.seed) if config.strategy_name == "random" else strategy_cls()  # type: ignore[call-arg]

        engine = GameEngine(shoe, rules, strategy)

        result = SimResult(config=config)
        start = time.perf_counter()

        for _ in range(config.num_hands):
            round_result = engine.play_round(bet=1.0)
            result.total_hands += 1
            result.total_payout += round_result.total_net_payout

            # Track extra hands produced by splits
            if round_result.num_player_hands > 1:
                result.total_splits += round_result.num_player_hands - 1
                
            # Track dealer busts (one per round)
            if round_result.dealer_hand.is_bust:
                result.dealer_busts += 1
                
            # Track player busts across all hands
            for hr in round_result.hand_results:
                if hr.player_value > 21:
                    result.player_busts += 1

            # Store the primary hand result for the hand_results list
            primary = round_result.primary_result
            if primary:
                result.hand_results.append(primary)

            # Win/loss/push counters are based on the PRIMARY hand outcome
            # (the first hand dealt in this round)
            if primary:
                match primary.outcome:
                    case HandOutcome.WIN:
                        result.wins += 1
                    case HandOutcome.LOSS:
                        result.losses += 1
                    case HandOutcome.PUSH:
                        result.pushes += 1
                    case HandOutcome.BLACKJACK:
                        result.blackjacks += 1
                        result.wins += 1
                    case HandOutcome.SURRENDER:
                        result.surrenders += 1
                        result.losses += 1

        result.elapsed_seconds = time.perf_counter() - start

        logger.info(
            f"Simulation complete - total_hands={result.total_hands}, total_splits={result.total_splits}, "
            f"house_edge={result.house_edge:.4%}, elapsed={result.elapsed_seconds:.2f}s"
        )

        return result

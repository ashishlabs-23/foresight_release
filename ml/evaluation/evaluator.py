"""
ml.evaluation.evaluator
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract Evaluator interface.

Phase 1: interface only.
Phase 3: implement ModelEvaluator that compares RL agent vs BasicStrategy
         across multiple rule variants and produces an evaluation report.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvaluationReport:
    """Summary of a model evaluation run."""

    model_name: str
    num_episodes: int
    win_rate: float
    loss_rate: float
    push_rate: float
    house_edge: float
    vs_basic_strategy_delta: float | None = None   # positive = better than basic
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Evaluation Report — {self.model_name}",
            f"  Episodes    : {self.num_episodes:,}",
            f"  Win rate    : {self.win_rate:.2%}",
            f"  Loss rate   : {self.loss_rate:.2%}",
            f"  House edge  : {self.house_edge:.4%}",
        ]
        if self.vs_basic_strategy_delta is not None:
            sign = "+" if self.vs_basic_strategy_delta >= 0 else ""
            lines.append(f"  vs Basic    : {sign}{self.vs_basic_strategy_delta:.4%}")
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Abstract model evaluator."""

    @abstractmethod
    def evaluate(self, num_episodes: int) -> EvaluationReport:
        """Run the evaluation and return a structured report.

        Parameters
        ----------
        num_episodes : Number of hands to play during evaluation.
        """
        ...

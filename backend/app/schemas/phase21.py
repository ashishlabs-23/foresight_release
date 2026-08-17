"""
backend.app.schemas.phase21
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 21: Canonical Pydantic models for the Real-World Feedback,
Decision Quality & Model Monitoring System.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    DECISION_CREATED = "DECISION_CREATED"
    USER_ACTION = "USER_ACTION"
    CARD_REVEALED = "CARD_REVEALED"
    HAND_SPLIT = "HAND_SPLIT"
    HAND_RESOLVED = "HAND_RESOLVED"
    DEALER_REVEALED = "DEALER_REVEALED"
    RESULT_RECORDED = "RESULT_RECORDED"
    DECISION_EVALUATED = "DECISION_EVALUATED"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    ERROR_EVENT = "ERROR_EVENT"


class DecisionQuality(str, Enum):
    """
    EV-regret thresholds (documented):
      OPTIMAL                regret ≤ 0.005
      NEAR_OPTIMAL     0.005 < regret ≤ 0.050
      SUBOPTIMAL       0.050 < regret ≤ 0.150
      SEVERELY_SUBOPTIMAL    regret > 0.150
      UNKNOWN                evaluation not yet run
    """
    OPTIMAL = "OPTIMAL"
    NEAR_OPTIMAL = "NEAR_OPTIMAL"
    SUBOPTIMAL = "SUBOPTIMAL"
    SEVERELY_SUBOPTIMAL = "SEVERELY_SUBOPTIMAL"
    UNKNOWN = "UNKNOWN"


class SubjectiveRating(str, Enum):
    YES = "YES"
    NO = "NO"
    NOT_SURE = "NOT_SURE"


class SubjectiveReason(str, Enum):
    ANOTHER_STRATEGY = "I followed another strategy"
    MISUNDERSTOOD = "I misunderstood the recommendation"
    WRONG_INPUT = "The cards/rules were entered incorrectly"
    LOOKED_WRONG = "The recommendation looked wrong"
    OTHER = "Other"


class DriftStatus(str, Enum):
    CURRENT = "CURRENT"
    WARNING = "WARNING_DRIFT"
    CRITICAL = "CRITICAL_DRIFT"


class ModelStatus(str, Enum):
    PRODUCTION = "PRODUCTION"
    EXPERIMENTAL = "EXPERIMENTAL"
    ARCHIVED = "ARCHIVED"
    CANDIDATE = "CANDIDATE"


# ---------------------------------------------------------------------------
# Core Event Model
# ---------------------------------------------------------------------------

class DecisionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    hand_id: str
    decision_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType
    payload: Dict[str, Any] = Field(default_factory=dict)
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Canonical Feedback Record (immutable)
# ---------------------------------------------------------------------------

class FeedbackRecord(BaseModel):
    """
    Canonical, append-only feedback record.
    Never overwrite — create new events instead.
    """
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    hand_id: str
    decision_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str
    feature_version: str
    # Rule configuration
    rule_configuration: Dict[str, Any]
    deck_count: int
    # State
    player_cards: List[str]
    dealer_visible_cards: List[str]
    action_space: List[str]
    # Model output
    predicted_evs: Dict[str, float]
    recommended_action: str
    uncertainty: float
    support: str
    risk_level: str
    ood_status: bool
    fallback_used: bool
    # User interaction (filled in later via events)
    user_action: Optional[str] = None
    next_card: Optional[str] = None
    subsequent_state: Optional[Dict[str, Any]] = None
    final_result: Optional[str] = None
    dealer_final_cards: List[str] = Field(default_factory=list)
    # Evaluation (filled in asynchronously)
    decision_quality: Optional[DecisionQuality] = None
    ai_regret: Optional[float] = None
    user_regret: Optional[float] = None
    monte_carlo_verified: bool = False


# ---------------------------------------------------------------------------
# Evaluation Result (from Monte Carlo Auditor)
# ---------------------------------------------------------------------------

class ActionEV(BaseModel):
    action: str
    ev: float


class EvaluationResult(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str
    session_id: str
    hand_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Per-action EVs
    action_evs: Dict[str, float]
    optimal_action: str
    optimal_ev: float
    ai_action: str
    ai_ev: float
    user_action: Optional[str] = None
    user_ev: Optional[float] = None
    ai_regret: float
    user_regret: Optional[float] = None
    decision_quality: DecisionQuality
    monte_carlo_simulations: int = 0
    model_version: str = ""


# ---------------------------------------------------------------------------
# Session Summary
# ---------------------------------------------------------------------------

class HandSummary(BaseModel):
    hand_id: str
    decisions: int
    optimal_count: int
    near_optimal_count: int
    suboptimal_count: int
    severely_suboptimal_count: int
    unknown_count: int
    avg_ai_regret: float
    avg_user_regret: float
    final_result: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    total_hands: int
    total_decisions: int
    ai_user_agreement_pct: float
    avg_ai_regret: float
    avg_user_regret: float
    high_risk_decisions: int
    ood_fallbacks: int
    monte_carlo_audited: int
    decision_quality_breakdown: Dict[str, int]
    hand_summaries: List[HandSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Subjective Feedback (user-facing YES/NO/NOT_SURE)
# ---------------------------------------------------------------------------

class SubjectiveFeedbackRequest(BaseModel):
    decision_id: str
    session_id: str
    rating: SubjectiveRating
    reason: Optional[SubjectiveReason] = None


# ---------------------------------------------------------------------------
# Drift Report
# ---------------------------------------------------------------------------

class DriftDimension(BaseModel):
    name: str
    psi_score: float
    status: DriftStatus
    details: Dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_status: DriftStatus
    dimensions: List[DriftDimension]
    sample_size: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Analytics Response Models
# ---------------------------------------------------------------------------

class ModelPerformanceMetrics(BaseModel):
    model_version: str
    mae: float
    rmse: float
    mean_regret: float
    top1_action_agreement: float
    fallback_rate: float
    ood_rate: float
    avg_uncertainty: float
    monte_carlo_coverage: float
    sample_size: int


class DecisionQualityMetrics(BaseModel):
    optimal_pct: float
    near_optimal_pct: float
    suboptimal_pct: float
    severely_suboptimal_pct: float
    unknown_pct: float
    avg_ai_regret: float
    avg_user_regret: float
    ai_regret_distribution: List[float]
    user_regret_distribution: List[float]
    sample_size: int


class UncertaintyMetrics(BaseModel):
    avg_uncertainty: float
    uncertainty_regret_correlation: float
    support_regret_correlation: float
    ood_rate: float
    uncertainty_buckets: List[Dict[str, float]]


class UserBehaviorMetrics(BaseModel):
    ai_user_agreement_pct: float
    top_disagreement_states: List[Dict[str, Any]]
    ai_action_distribution: Dict[str, float]
    user_action_distribution: Dict[str, float]
    training_action_distribution: Dict[str, float]
    deck_distribution: Dict[str, int]
    sample_size: int


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    model_version: str
    model_type: str
    status: ModelStatus
    training_dataset_version: str
    feature_version: str
    training_date: str
    validation_metrics: Dict[str, float]
    promotion_status: str
    supported_rules: List[str]
    model_hash: str
    is_production: bool = False


# ---------------------------------------------------------------------------
# API request/response helpers
# ---------------------------------------------------------------------------

class SessionTimelineResponse(BaseModel):
    session_id: str
    events: List[Dict[str, Any]]
    reconstructed_hands: Dict[str, Any]

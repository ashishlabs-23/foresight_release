from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

class DecisionRecord(BaseModel):
    decision_id: str
    session_id: str
    timestamp: datetime
    
    # State Capture
    player_total: int
    dealer_upcard: str
    is_soft: bool
    is_pair: bool
    
    # ML Capture
    model_version: str
    feature_version: str
    legal_actions: List[str]
    action_values: Dict[str, float]
    
    # Decision Analysis
    recommended_action: str
    support_status: str  # HIGH_SUPPORT, LIMITED_SUPPORT, FALLBACK
    fallback_used: bool
    fallback_reason: Optional[str] = None
    decision_strength: str  # Close Decision, Strong Preference, etc.
    decision_margin: float
    
    # Phase 16: Uncertainty & Risk
    prediction_interval: Optional[List[float]] = None
    uncertainty_score: Optional[float] = None
    risk_level: Optional[str] = None
    calibration_version: Optional[str] = None
    uncertainty_method: Optional[str] = None
    
    # Phase 17: Research RL Fields
    policy_type: Optional[str] = None
    policy_version: Optional[str] = None
    q_values: Optional[Dict[str, float]] = None
    q_uncertainty: Optional[float] = None
    policy_support: Optional[int] = None
    policy_fallback: Optional[bool] = None
    baseline_action: Optional[str] = None
    baseline_EV: Optional[float] = None
    RL_action: Optional[str] = None
    RL_Q: Optional[float] = None
    agreement_with_baseline: Optional[bool] = None
    
    # Player Alignment
    user_action: Optional[str] = None
    user_action_ev: Optional[float] = None
    ev_difference: Optional[float] = None  # (recommended EV - user EV)
    
    # Outcome
    resulting_cards: List[str] = []
    reward: Optional[float] = None
    latency_ms: float = 0.0

    @classmethod
    def create(cls, session_id: str, state_summary: Dict, ml_output: Dict, latency_ms: float):
        # Calculate Margin
        action_vals = ml_output.get("action_values", {})
        margin = 0.0
        strength = "N/A"
        
        if action_vals and len(action_vals) >= 2:
            sorted_evs = sorted(action_vals.values(), reverse=True)
            margin = sorted_evs[0] - sorted_evs[1]
            if margin < 0.05:
                strength = "Close Decision"
            elif margin < 0.15:
                strength = "Moderate Preference"
            else:
                strength = "Strong Preference"
                
        return cls(
            decision_id=str(uuid4()),
            session_id=session_id,
            timestamp=datetime.utcnow(),
            player_total=state_summary["player_total"],
            dealer_upcard=state_summary["dealer_upcard"],
            is_soft=state_summary["is_soft"],
            is_pair=state_summary["is_pair"],
            model_version=ml_output.get("model_version", "unknown"),
            feature_version="v1",
            legal_actions=list(action_vals.keys()) if action_vals else [ml_output.get("recommended_action", "")],
            action_values=action_vals,
            recommended_action=ml_output.get("recommended_action", ""),
            support_status=ml_output.get("support_status", "unknown"),
            fallback_used=ml_output.get("fallback_used", False),
            fallback_reason=ml_output.get("reason") if ml_output.get("fallback_used") else None,
            decision_strength=strength,
            decision_margin=margin,
            prediction_interval=ml_output.get("prediction_interval"),
            uncertainty_score=ml_output.get("uncertainty_score"),
            risk_level=ml_output.get("risk_level"),
            calibration_version=ml_output.get("calibration_version"),
            uncertainty_method=ml_output.get("uncertainty_method"),
            policy_type=ml_output.get("policy_type"),
            policy_version=ml_output.get("policy_version"),
            q_values=ml_output.get("q_values"),
            q_uncertainty=ml_output.get("q_uncertainty"),
            policy_support=ml_output.get("policy_support"),
            policy_fallback=ml_output.get("policy_fallback"),
            baseline_action=ml_output.get("baseline_action"),
            baseline_EV=ml_output.get("baseline_EV"),
            RL_action=ml_output.get("RL_action"),
            RL_Q=ml_output.get("RL_Q"),
            agreement_with_baseline=ml_output.get("agreement_with_baseline"),
            latency_ms=latency_ms
        )

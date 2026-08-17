"""
backend.app.schemas.analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: Pydantic schemas for Multi-Hand User-Driven Blackjack AI interface.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, root_validator
from datetime import datetime
from backend.app.services.rule_validator import RuleConfigurationValidator
from backend.app.services.card_accounting import CardAvailabilityEngine

class GameRulesConfig(BaseModel):
    decks: int = Field(default=6, ge=1, le=8)
    hit_soft_17: bool = Field(default=True)
    blackjack_payout: float = Field(default=1.5)
    double_allowed: bool = Field(default=True)
    double_after_split: bool = Field(default=True)
    split_allowed: bool = Field(default=True)
    resplit_allowed: bool = Field(default=True)
    surrender_allowed: bool = Field(default=True)
    late_surrender: bool = Field(default=True)
    early_surrender: bool = Field(default=False)
    
    @root_validator(skip_on_failure=True)
    def validate_rules(cls, values):
        RuleConfigurationValidator.validate(values)
        return values

class PlayerHandState(BaseModel):
    hand_id: str
    cards: List[str]
    is_active: bool = False
    is_completed: bool = False

class AnalyzeRequest(BaseModel):
    rules: GameRulesConfig
    player_hands: List[PlayerHandState] = Field(default_factory=list)
    player_cards: Optional[List[str]] = None
    dealer_cards: List[str]
    
    @root_validator(pre=True)
    def handle_legacy_player_cards(cls, values):
        if "player_cards" in values and ("player_hands" not in values or not values["player_hands"]):
            values["player_hands"] = [{
                "hand_id": "h1",
                "cards": values["player_cards"],
                "is_active": True,
                "is_completed": False
            }]
        return values

    @root_validator(skip_on_failure=True)
    def validate_cards(cls, values):
        player_hands = values.get('player_hands', [])
        dealer_cards = values.get('dealer_cards', [])
        rules = values.get('rules')
        
        all_cards = list(dealer_cards)
        for h in player_hands:
            all_cards.extend(h.cards)
            
        if rules:
            CardAvailabilityEngine.validate_availability(rules.decks, all_cards)
                    
        return values

class DecisionActionRequest(BaseModel):
    session_id: str
    hand_id: Optional[str] = None
    user_action: str

class DecisionCardRequest(BaseModel):
    session_id: str
    hand_id: Optional[str] = None
    target: str = Field(..., description="Either 'player' or 'dealer'")
    card: str

class DecisionResultRequest(BaseModel):
    session_id: str
    hand_results: Optional[Dict[str, str]] = Field(default=None, description="Map of hand_id to final_result")
    final_result: Optional[str] = None

    @root_validator(pre=True)
    def handle_legacy_final_result(cls, values):
        if "final_result" in values and "hand_results" not in values:
            values["hand_results"] = {}
        return values
    
class UserDecisionFeedback(BaseModel):
    decision_id: str
    session_id: str
    hand_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str
    feature_version: str
    calibration_version: str
    decks: int
    state_snapshot: Dict[str, Any]
    recommended_action: str
    predicted_evs: Dict[str, float]
    uncertainty: float
    risk_level: str
    support_level: str
    user_action: Optional[str] = None
    new_cards: List[str] = Field(default_factory=list)
    final_result: Optional[str] = None

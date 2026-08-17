"""
backend.app.api.v1.analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: Endpoints for the manual Blackjack Decision Analyzer with Multi-Hand support.
"""
from fastapi import APIRouter, HTTPException
from backend.app.schemas.analyzer import (
    AnalyzeRequest, DecisionActionRequest, DecisionCardRequest, DecisionResultRequest
)
from backend.app.services.analyzer_manager import AnalyzerSession, ANALYZER_SESSIONS
from blackjack.cards.hand import Hand
from uuid import uuid4

router = APIRouter(prefix="/analyzer", tags=["analyzer"])

@router.post("/analyze")
def start_analysis(req: AnalyzeRequest):
    try:
        session = AnalyzerSession(req)
        ANALYZER_SESSIONS[session.session_id] = session
        
        output = session.get_recommendation()
        
        return {
            "session_id": session.session_id,
            "hands": session.hands,
            "active_hand_id": session.active_hand_id,
            "recommendation": output
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/decision/action")
def log_action(req: DecisionActionRequest):
    session = ANALYZER_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    target_hand_id = req.hand_id or session.active_hand_id
    if not target_hand_id or session.active_hand_id != target_hand_id:
        raise HTTPException(status_code=400, detail="Action submitted for non-active hand")

    action = req.user_action.lower().strip()
    if action not in session.legal_action_names():
        raise HTTPException(status_code=400, detail=f"Illegal action '{action}' for the current hand")

    # Persist the exact decision BEFORE generating the next recommendation.
    # This fixes the old feedback race where get_recommendation() replaced the
    # pending record before the user's HIT/DOUBLE choice could be saved.
    if getattr(session, "pending_feedback", None):
        session.pending_feedback.user_action = action
        from backend.app.monitoring.feedback import save_feedback
        save_feedback(session.pending_feedback.dict())
        session.pending_feedback = None

    hand_data = session.hands[target_hand_id]
    if action in {"hit", "double", "split"}:
        session.pending_draw_action[target_hand_id] = action

    if action == "stand":
        hand_data["is_completed"] = True
        hand_data["is_active"] = False
        session._advance_to_next_hand()
    elif action == "surrender":
        hand_data["is_completed"] = True
        hand_data["is_active"] = False
        session._advance_to_next_hand()
    elif action == "split":
        if len(hand_data["cards"]) != 2:
            raise HTTPException(status_code=400, detail="Split requires exactly two cards")
        card1, card2 = hand_data["cards"]
        hand_data["cards"] = [card1]
        hand_data["split_count"] += 1
        new_hand_id = str(uuid4())
        session.hands[new_hand_id] = {
            "cards": [card2], "is_active": False, "is_completed": False,
            "split_count": hand_data["split_count"]
        }

    output = session.get_recommendation()
    return {"status":"success","hands":session.hands,"active_hand_id":session.active_hand_id,"recommendation":output}

@router.post("/decision/card")
def add_card(req: DecisionCardRequest):
    session = ANALYZER_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.target == "player":
        target_hand_id = req.hand_id or session.active_hand_id
        if not target_hand_id or target_hand_id not in session.hands:
            raise HTTPException(status_code=400, detail="Invalid player hand target")
        if target_hand_id != session.active_hand_id:
            raise HTTPException(status_code=400, detail="Card submitted for a non-active hand")

        hand_data = session.hands[target_hand_id]
        hand_data["cards"].append(req.card)
        pending = session.pending_draw_action.pop(target_hand_id, None)

        # Attach the actual dealt card to the decision that the user just made.
        from backend.app.monitoring.feedback import append_new_card
        append_new_card(session.session_id, target_hand_id, req.card)

        parsed = Hand()
        for rank in hand_data["cards"]:
            parsed.add_card(session._parse_card(rank))

        # DOUBLE always ends after its one card. A natural 21/bust also ends.
        if pending == "double" or parsed.value >= 21:
            hand_data["is_completed"] = True
            hand_data["is_active"] = False
            session._advance_to_next_hand()
        if pending == "split_initial":
            # The second split hand has now received its initial second card;
            # it is ready for a normal decision.
            pass
        # SPLIT/HIT continue with the same hand unless it has naturally ended.

    elif req.target == "dealer":
        if session.active_hand_id is not None:
            raise HTTPException(status_code=400, detail="Dealer final cards can only be entered after all player hands are complete")
        session.dealer_cards.append(req.card)
    else:
        raise HTTPException(status_code=400, detail="Invalid target")

    output = session.get_recommendation()
    active_cards = session.hands[target_hand_id]["cards"] if req.target == "player" and target_hand_id in session.hands else []
    return {
        "session_id": session.session_id,
        "hands": session.hands,
        "active_hand_id": session.active_hand_id,
        "player_cards": active_cards,
        "recommendation": output,
        "dealer_cards": session.dealer_cards
    }

@router.post("/decision/result")
def log_result(req: DecisionResultRequest):
    session = ANALYZER_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    from backend.app.monitoring.feedback import update_results
    
    hand_results = req.hand_results or {}
    if not hand_results and req.final_result:
        hand_results = {hid: req.final_result for hid in session.hands}

    if not hand_results:
        raise HTTPException(status_code=400, detail="At least one hand result is required")
    unknown = set(hand_results) - set(session.hands)
    if unknown:
        raise HTTPException(status_code=400, detail="Result contains an unknown hand")
    if session.active_hand_id is not None:
        raise HTTPException(status_code=400, detail="Session still has an active hand")

    allowed = {"win","loss","push","blackjack","bust"}
    invalid = {v for v in hand_results.values() if v not in allowed}
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid result(s): {', '.join(sorted(invalid))}")

    for hand_id, result_str in hand_results.items():
        update_results(session.session_id, hand_id, result_str)

    del ANALYZER_SESSIONS[req.session_id]
    return {"status":"success","closed":True,"session_id":req.session_id}

@router.get("/decision/history")
def get_history():
    from backend.app.monitoring.feedback import get_history
    return get_history()

@router.get("/decision/{decision_id}")
def get_decision(decision_id: str):
    from backend.app.monitoring.feedback import get_decision
    res = get_decision(decision_id)
    if not res:
        raise HTTPException(status_code=404, detail="Decision not found")
    return res

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel

from backend.app.services.session_manager import GameSession

router = APIRouter(prefix="/game", tags=["game"])

# In-memory store
SESSIONS: Dict[str, GameSession] = {}

class ActionRequest(BaseModel):
    session_id: str
    action: str

@router.post("/session")
def create_session() -> Dict[str, Any]:
    session = GameSession()
    SESSIONS[session.session_id] = session
    return session.get_public_state()
    
@router.get("/state/{session_id}")
def get_state(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSIONS[session_id].get_public_state()
    
@router.post("/action")
def apply_action(req: ActionRequest) -> Dict[str, Any]:
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[req.session_id]
    try:
        session.apply_action(req.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return session.get_public_state()
    
@router.post("/reset/{session_id}")
def reset_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    session.start_round()
    return session.get_public_state()

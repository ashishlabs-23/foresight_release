from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from backend.app.api.v1.game import SESSIONS

router = APIRouter(prefix="/ai", tags=["ai"])

class RecommendRequest(BaseModel):
    session_id: str

@router.post("/recommend")
def recommend_action(req: RecommendRequest) -> Dict[str, Any]:
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[req.session_id]
    try:
        return session.get_recommendation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

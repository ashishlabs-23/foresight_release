from fastapi import APIRouter, HTTPException, Request, Header
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

router = APIRouter(prefix="/admin", tags=["admin"])

# Very simple mock authorization
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization != "Bearer mock-admin-token":
        raise HTTPException(status_code=401, detail="Unauthorized admin access")

@router.get("/model-health")
async def get_model_health(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    verify_token(authorization)
    
    # In a real system, we might query the DB directly, but here we read the daily report
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_file = Path(f"reports/production/{today_str}/model_health.json")
    
    if not report_file.exists():
        return {
            "status": "UNKNOWN",
            "health_score": 100,
            "message": "No evaluation report generated for today yet. Try running the shadow evaluator."
        }
        
    with open(report_file, "r") as f:
        return json.load(f)

@router.get("/model-registry")
async def get_model_registry(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    verify_token(authorization)
    
    registry_file = Path("ml/registry/model_registry.json")
    if not registry_file.exists():
        raise HTTPException(status_code=404, detail="Registry not found")
        
    with open(registry_file, "r") as f:
        return json.load(f)

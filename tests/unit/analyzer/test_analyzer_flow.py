"""
tests.unit.analyzer.test_analyzer_flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 19: End-to-End test for the Manual Analyzer Flow.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_analyzer_flow():
    # 1. Analyze Initial State (6 decks, Player: 10, 6, Dealer: 7)
    res = client.post("/api/v1/analyzer/analyze", json={
        "rules": {
            "decks": 6,
            "hit_soft_17": False,
            "blackjack_payout": 1.5,
            "double_after_split": True,
            "surrender_allowed": True
        },
        "player_cards": ["10", "6"],
        "dealer_cards": ["7"]
    })
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    rec = data["recommendation"]
    
    assert rec["recommended_action"] in ["hit", "stand"]
    
    # 2. User action: HIT
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "user_action": "hit"
    })
    assert res.status_code == 200
    
    # 3. Next Card: 4
    res = client.post("/api/v1/analyzer/decision/card", json={
        "session_id": session_id,
        "target": "player",
        "card": "4"
    })
    assert res.status_code == 200
    data = res.json()
    
    # New state should be 20 vs 7, Recommendation should be STAND
    assert "4" in data["player_cards"]
    rec = data["recommendation"]
    assert rec["recommended_action"] == "stand"
    
    # 4. User action: STAND
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "user_action": "stand"
    })
    assert res.status_code == 200
    
    # 5. Final Result: WIN
    res = client.post("/api/v1/analyzer/decision/result", json={
        "session_id": session_id,
        "final_result": "win"
    })
    assert res.status_code == 200
    
    # 6. Verify in History
    res = client.get("/api/v1/analyzer/decision/history")
    assert res.status_code == 200
    history = res.json()
    assert any(h["session_id"] == session_id and h["final_result"] == "win" for h in history)

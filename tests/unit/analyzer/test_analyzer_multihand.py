"""
tests.unit.analyzer.test_analyzer_multihand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: E2E test for the Manual Analyzer Flow with Split scenarios.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_analyzer_split_flow():
    # 1. Start Analysis with 8,8 vs 6
    res = client.post("/api/v1/analyzer/analyze", json={
        "rules": {
            "decks": 6,
            "hit_soft_17": False,
            "blackjack_payout": 1.5,
            "double_after_split": True,
            "split_allowed": True,
            "resplit_allowed": True,
            "surrender_allowed": True,
            "late_surrender": True,
            "early_surrender": False
        },
        "player_hands": [{"hand_id": "h1", "cards": ["8", "8"], "is_active": True, "is_completed": False}],
        "dealer_cards": ["6"]
    })
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    active_hand = data["active_hand_id"]
    assert active_hand == "h1"
    
    # 2. User Action: SPLIT
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "hand_id": "h1",
        "user_action": "split"
    })
    assert res.status_code == 200
    data = res.json()
    
    # There should now be two hands. One is active.
    assert len(data["hands"]) == 2
    active_hand = data["active_hand_id"]
    assert data["hands"][active_hand]["cards"] == ["8"]
    
    # 3. Add Card to Hand 1 (Say, a 3)
    res = client.post("/api/v1/analyzer/decision/card", json={
        "session_id": session_id,
        "hand_id": active_hand,
        "target": "player",
        "card": "3"
    })
    assert res.status_code == 200
    data = res.json()
    
    # AI should recommend HIT on 11 vs 6
    rec = data["recommendation"]
    assert rec["recommended_action"] in ["hit", "double"]
    
    # 4. User Action: HIT
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "hand_id": active_hand,
        "user_action": "hit"
    })
    
    # 5. Add Card to Hand 1 (Say, a 7 -> 18)
    res = client.post("/api/v1/analyzer/decision/card", json={
        "session_id": session_id,
        "hand_id": active_hand,
        "target": "player",
        "card": "7"
    })
    
    # 6. User Action: STAND (completes Hand 1)
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "hand_id": active_hand,
        "user_action": "stand"
    })
    data = res.json()
    
    # Active hand should now have shifted to Hand 2
    new_active = data["active_hand_id"]
    assert new_active != active_hand
    
    # 7. Add Card to Hand 2 (Say, a 10 -> 18)
    res = client.post("/api/v1/analyzer/decision/card", json={
        "session_id": session_id,
        "hand_id": new_active,
        "target": "player",
        "card": "10"
    })
    
    # 8. User Action: STAND (completes Hand 2)
    res = client.post("/api/v1/analyzer/decision/action", json={
        "session_id": session_id,
        "hand_id": new_active,
        "user_action": "stand"
    })
    data = res.json()
    
    # No more active hands
    assert data["active_hand_id"] is None
    
    # 9. Record Final Results
    res = client.post("/api/v1/analyzer/decision/result", json={
        "session_id": session_id,
        "hand_results": {
            active_hand: "win",
            new_active: "push"
        }
    })
    assert res.status_code == 200

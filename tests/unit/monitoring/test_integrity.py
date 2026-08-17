"""
tests.unit.monitoring.test_integrity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 18: Tests model hash and feature schema integrity at startup.
"""
import pytest
import json
from pathlib import Path
from backend.app.main import create_app

def test_startup_integrity_check_failure(monkeypatch, tmp_path):
    """
    Test that the application fails to start if the model registry specifies 
    a hash that doesn't match the loaded model.
    """
    mock_registry = {
        "active_production": "xgboost_v16",
        "models": {
            "xgboost_v16": {
                "version": "v16",
                "status": "PRODUCTION",
                "model_hash": "INVALID_HASH",
                "feature_schema_hash": "f1e2d3c4b5a6"
            }
        }
    }
    
    registry_path = tmp_path / "model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(mock_registry, f)
        
    # Mock the registry path in the integrity checker
    monkeypatch.setattr("backend.app.main.REGISTRY_PATH", str(registry_path))
    
    with pytest.raises(RuntimeError, match="Model integrity check failed: Hash mismatch"):
        # The app should refuse to start
        app = create_app()

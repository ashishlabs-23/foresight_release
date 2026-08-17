"""
ml.monitoring.drift
~~~~~~~~~~~~~~~~~~~
Phase 18: Drift Detection utilities.
Implements PSI (Population Stability Index) for detecting feature and state drift.
"""
import numpy as np

def calculate_psi(expected_dist: np.ndarray, actual_dist: np.ndarray, buckets=10, epsilon=1e-6) -> float:
    """
    Calculate the Population Stability Index (PSI) between two discrete distributions.
    """
    expected_dist = np.asarray(expected_dist)
    actual_dist = np.asarray(actual_dist)
    
    # Ensure they are normalized probabilities
    expected_prob = expected_dist / (np.sum(expected_dist) + epsilon)
    actual_prob = actual_dist / (np.sum(actual_dist) + epsilon)
    
    # Avoid log(0)
    expected_prob = np.maximum(expected_prob, epsilon)
    actual_prob = np.maximum(actual_prob, epsilon)
    
    # PSI = sum((actual - expected) * ln(actual / expected))
    psi_values = (actual_prob - expected_prob) * np.log(actual_prob / expected_prob)
    return float(np.sum(psi_values))

def evaluate_drift(psi_value: float) -> str:
    """
    Standard PSI drift interpretation.
    """
    if psi_value < 0.1:
        return "NO_DRIFT"
    elif psi_value < 0.2:
        return "WARNING_DRIFT"
    else:
        return "CRITICAL_DRIFT"

# Simulated baseline distributions for Blackjack states (Phase 16 Validation Baseline)
BASELINE_DISTRIBUTIONS = {
    "recommended_action": {
        "hit": 0.45,
        "stand": 0.40,
        "double": 0.10,
        "split": 0.03,
        "surrender": 0.02
    },
    "support_status": {
        "HIGH_SUPPORT": 0.92,
        "LIMITED_SUPPORT": 0.07,
        "FALLBACK": 0.01
    }
}

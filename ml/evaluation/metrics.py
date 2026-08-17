"""
ml.evaluation.metrics
~~~~~~~~~~~~~~~~~~~~~
Phase 16: Uncertainty calibration metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_calibration_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate coverage, average width, sharpness, and other calibration metrics.
    
    df should contain: ['true_ev', 'predicted_ev', 'lower_bound', 'upper_bound']
    """
    if 'lower_bound' not in df.columns or 'upper_bound' not in df.columns:
        return {}
        
    df['is_covered'] = (df['true_ev'] >= df['lower_bound']) & (df['true_ev'] <= df['upper_bound'])
    df['interval_width'] = df['upper_bound'] - df['lower_bound']
    
    coverage = df['is_covered'].mean()
    avg_width = df['interval_width'].mean()
    
    # Coverage error: assume 80% target coverage as a generic baseline if not specified
    target_coverage = 0.80 
    coverage_error = np.abs(coverage - target_coverage)
    
    # Sharpness: variance of the widths or similar metric (smaller is sharper)
    sharpness = df['interval_width'].std()
    
    metrics = {
        'coverage': float(coverage),
        'avg_interval_width': float(avg_width),
        'coverage_error': float(coverage_error),
        'sharpness': float(sharpness)
    }
    
    if 'predicted_ev' in df.columns:
        error = df['predicted_ev'] - df['true_ev']
        metrics['mae'] = float(np.abs(error).mean())
        metrics['rmse'] = float(np.sqrt((error ** 2).mean()))
        
    return metrics

def calculate_action_agreement(df: pd.DataFrame) -> float:
    """
    Calculate percentage of times the model's recommended action matches the optimal action.
    df must contain ['predicted_action', 'optimal_action']
    """
    if 'predicted_action' not in df.columns or 'optimal_action' not in df.columns:
        return 0.0
    return float((df['predicted_action'] == df['optimal_action']).mean())

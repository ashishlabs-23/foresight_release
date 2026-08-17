"""
ml.evaluation.calibration
~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 16: Calibration metrics and dataset builder for EV uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def build_calibration_dataset(
    predictions: pd.DataFrame, 
    monte_carlo_truth: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge XGBoost predictions with Monte Carlo true EVs.
    
    predictions should contain: ['state_id', 'action', 'predicted_ev', 'support_score', 'margin']
    monte_carlo_truth should contain: ['state_id', 'action', 'true_ev', 'sample_count']
    """
    merged = pd.merge(predictions, monte_carlo_truth, on=['state_id', 'action'], how='inner')
    merged['error'] = merged['predicted_ev'] - merged['true_ev']
    merged['absolute_error'] = np.abs(merged['error'])
    merged['squared_error'] = merged['error'] ** 2
    return merged

def calculate_error_distributions(df: pd.DataFrame, groupby_col: str | None = None) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, bias, and percentile absolute errors.
    """
    def _calc_stats(group: pd.DataFrame) -> pd.Series:
        mae = group['absolute_error'].mean()
        rmse = np.sqrt(group['squared_error'].mean())
        bias = group['error'].mean()
        median_ae = group['absolute_error'].median()
        p90_ae = group['absolute_error'].quantile(0.90)
        p95_ae = group['absolute_error'].quantile(0.95)
        p99_ae = group['absolute_error'].quantile(0.99)
        max_ae = group['absolute_error'].max()
        
        return pd.Series({
            'count': len(group),
            'MAE': mae,
            'RMSE': rmse,
            'bias': bias,
            'median_ae': median_ae,
            'p90_ae': p90_ae,
            'p95_ae': p95_ae,
            'p99_ae': p99_ae,
            'max_ae': max_ae
        })
        
    if groupby_col:
        return df.groupby(groupby_col).apply(_calc_stats).reset_index()
    else:
        return _calc_stats(df).to_frame().T

def analyze_error_vs_support(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Group by support_score bins and calculate absolute error stats.
    """
    df_copy = df.copy()
    df_copy['support_bin'] = pd.qcut(df_copy['support_score'], q=bins, duplicates='drop')
    return calculate_error_distributions(df_copy, groupby_col='support_bin')

def analyze_error_vs_margin(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Group by margin bins and calculate absolute error stats.
    """
    df_copy = df.copy()
    if 'margin' in df_copy.columns:
        df_copy['margin_bin'] = pd.qcut(df_copy['margin'], q=bins, duplicates='drop')
        return calculate_error_distributions(df_copy, groupby_col='margin_bin')
    return pd.DataFrame()

def analyze_regret_vs_margin(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Calculate regret statistics grouped by decision margin.
    """
    if 'regret' not in df.columns or 'margin' not in df.columns:
        return pd.DataFrame()
        
    df_copy = df.copy()
    df_copy['margin_bin'] = pd.qcut(df_copy['margin'], q=bins, duplicates='drop')
    
    return df_copy.groupby('margin_bin').agg(
        count=('regret', 'size'),
        avg_regret=('regret', 'mean'),
        median_regret=('regret', 'median'),
        p95_regret=('regret', lambda x: x.quantile(0.95))
    ).reset_index()

"""
ml.rl.data_converter
~~~~~~~~~~~~~~~~~~~~
Phase 17: Converts dataset of trajectories into offline RL transitions (s, a, r, s', done).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

def convert_trajectories_to_transitions(trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a dataframe of sequential hand trajectories into MDP transitions.
    Assumes trajectories_df is grouped by `hand_id` and sorted by `step` sequentially.
    Columns expected: ['hand_id', 'step', 'state_id', 'action', 'final_reward']
    
    IMPORTANT: We do NOT blindly apply final_reward to all steps.
    Intermediate steps receive a reward of 0.
    Terminal steps receive the final_reward.
    """
    transitions = []
    
    # Group by individual hand
    grouped = trajectories_df.groupby('hand_id')
    
    for hand_id, group in grouped:
        # Sort by step
        group = group.sort_values('step').reset_index(drop=True)
        num_steps = len(group)
        
        for i in range(num_steps):
            row = group.iloc[i]
            
            s = row['state_id']
            a = row['action']
            
            if i == num_steps - 1:
                # Terminal step
                r = row['final_reward']
                s_next = None
                done = True
            else:
                # Intermediate step
                r = 0.0
                s_next = group.iloc[i+1]['state_id']
                done = False
                
            transitions.append({
                'hand_id': hand_id,
                'state': s,
                'action': a,
                'reward': r,
                'next_state': s_next,
                'done': done
            })
            
    return pd.DataFrame(transitions)

def analyze_dataset_coverage(transitions_df: pd.DataFrame) -> dict:
    """
    Analyzes coverage of states, actions, and state-action pairs.
    """
    if transitions_df.empty:
        return {}
        
    num_transitions = len(transitions_df)
    unique_states = transitions_df['state'].nunique()
    
    # Action distribution
    action_dist = transitions_df['action'].value_counts(normalize=True).to_dict()
    
    # State-Action coverage
    transitions_df['state_action'] = transitions_df['state'].astype(str) + "_" + transitions_df['action'].astype(str)
    unique_sa = transitions_df['state_action'].nunique()
    
    sa_counts = transitions_df['state_action'].value_counts()
    rare_sa_count = (sa_counts < 100).sum()
    
    return {
        "num_transitions": num_transitions,
        "unique_states": unique_states,
        "unique_state_action_pairs": unique_sa,
        "action_distribution": action_dist,
        "rare_state_action_pairs": rare_sa_count,
    }

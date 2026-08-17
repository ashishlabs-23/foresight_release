"""
scripts/analyze_coverage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Identifies action coverage gaps in the Kaggle dataset.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    base_dir = Path("data/processed/decision_samples/train")
    if not base_dir.exists():
        logging.error("Train dataset not found.")
        return 1
        
    logging.info("Loading training data for coverage analysis...")
    files = list(base_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    total_rows = len(df)
    logging.info(f"Loaded {total_rows:,} rows.")
    
    # Define Canonical State
    # We will use player_total, dealer_upcard, is_soft, is_pair
    # We will compute the cross-tabulation of State x Action
    
    # Create a string representation for the state
    def make_state_key(row):
        pt = row["player_total"]
        dup = row["dealer_upcard"]
        if row["is_pair"]:
            return f"Pair {pt//2},{pt//2} vs {dup}"
        elif row["is_soft"]:
            return f"Soft {pt} vs {dup}"
        else:
            return f"Hard {pt} vs {dup}"
            
    df["canonical_state"] = df.apply(make_state_key, axis=1)
    
    coverage = pd.crosstab(df["canonical_state"], df["action"]).fillna(0).astype(int)
    
    # Add missing action columns if they don't exist
    for act in ["hit", "stand", "double", "split", "surrender"]:
        if act not in coverage.columns:
            coverage[act] = 0
            
    # Save the raw coverage matrix
    out_dir = Path("reports/phase11")
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(out_dir / "action_coverage_before.parquet")
    
    # Identify Gaps (threshold = 1000)
    MIN_SAMPLES = 1000
    
    report_lines = [
        "# Phase 11: Action Coverage Analysis (Before)",
        "",
        f"**Total States Analyzed**: {len(coverage)}",
        f"**Total Rows**: {total_rows:,}",
        f"**Minimum Sample Threshold**: {MIN_SAMPLES}",
        "",
        "## Critical Coverage Gaps Identified",
        "The following common states are missing crucial counterfactual action support:",
        ""
    ]
    
    gaps_found = 0
    for state, row in coverage.iterrows():
        # Only look at a few example gaps to keep the report concise
        if gaps_found > 20:
            break
            
        # Is it a standard state we care about? (e.g. not 21, not hard 4 vs Ace)
        if "Hard 21" in state or "Hard 20" in state: 
            continue
            
        actions_str = []
        has_gap = False
        
        for act in ["hit", "stand", "double", "split", "surrender"]:
            count = row.get(act, 0)
            
            # Simple legality heuristics for reporting (precise legality handled in generation)
            is_legal = True
            if act == "split" and "Pair" not in state: is_legal = False
            if act == "double" and ("Hard" in state and int(state.split(" ")[1]) > 11): is_legal = True # Legal but rarely done
            
            if is_legal:
                if count < MIN_SAMPLES:
                    has_gap = True
                    actions_str.append(f"**{act.upper()}**: {count} ⚠️")
                else:
                    actions_str.append(f"{act.upper()}: {count}")
                    
        if has_gap:
            report_lines.append(f"- **{state}** -> " + " | ".join(actions_str))
            gaps_found += 1
            
    with open(out_dir / "action_coverage_before.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logging.info(f"Coverage analysis saved to {out_dir}/action_coverage_before.md")
    return 0

if __name__ == "__main__":
    sys.exit(main())

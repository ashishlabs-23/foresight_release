"""
scripts/augment_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Combines the Kaggle observational data with the Counterfactual simulation data.
"""
import logging
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    train_dir = Path("data/processed/decision_samples/train")
    cf_path = Path("data/processed/counterfactual/counterfactual_samples.parquet")
    
    if not train_dir.exists() or not cf_path.exists():
        logging.error("Source datasets not found.")
        return 1
        
    logging.info("Loading Kaggle training data...")
    files = list(train_dir.glob("*.parquet"))
    kaggle_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    logging.info("Loading Counterfactual data...")
    cf_df = pd.read_parquet(cf_path)
    
    # Kaggle IDs might be int64, while CF IDs are strings. Unify them as strings.
    kaggle_df["trajectory_id"] = kaggle_df["trajectory_id"].astype(str)
    kaggle_df["state_id"] = kaggle_df["state_id"].astype(str)
    cf_df["trajectory_id"] = cf_df["trajectory_id"].astype(str)
    cf_df["state_id"] = cf_df["state_id"].astype(str)
    
    # Ensure source column exists
    if "source" not in kaggle_df.columns:
        kaggle_df["source"] = "kaggle"
        
    # Combine
    logging.info("Combining datasets...")
    augmented_df = pd.concat([kaggle_df, cf_df], ignore_index=True)
    
    logging.info(f"Kaggle Rows         : {len(kaggle_df):,}")
    logging.info(f"Counterfactual Rows : {len(cf_df):,}")
    logging.info(f"Augmented Total     : {len(augmented_df):,}")
    
    # Analyze Coverage After
    def make_state_key(row):
        pt = row["player_total"]
        dup = row["dealer_upcard"]
        if row["is_pair"]:
            return f"Pair {pt//2},{pt//2} vs {dup}"
        elif row["is_soft"]:
            return f"Soft {pt} vs {dup}"
        else:
            return f"Hard {pt} vs {dup}"
            
    augmented_df["canonical_state"] = augmented_df.apply(make_state_key, axis=1)
    coverage = pd.crosstab(augmented_df["canonical_state"], augmented_df["action"]).fillna(0).astype(int)
    
    for act in ["hit", "stand", "double", "split", "surrender"]:
        if act not in coverage.columns:
            coverage[act] = 0
            
    out_dir = Path("reports/phase11")
    coverage.to_parquet(out_dir / "action_coverage_after.parquet")
    
    MIN_SAMPLES = 1000
    report_lines = [
        "# Phase 11: Action Coverage Analysis (After Augmentation)",
        "",
        f"**Total States Analyzed**: {len(coverage)}",
        f"**Total Augmented Rows**: {len(augmented_df):,}",
        f"**Counterfactuals Added**: {len(cf_df):,}",
        "",
        "## Selected Coverage Improvements",
        ""
    ]
    
    improvements_shown = 0
    for state, row in coverage.iterrows():
        if improvements_shown > 20: break
        
        # Did we add counterfactuals here?
        cf_matches = cf_df[cf_df.apply(make_state_key, axis=1) == state]
        if not cf_matches.empty:
            actions_str = []
            for act in ["hit", "stand", "double", "split", "surrender"]:
                count = row.get(act, 0)
                cf_count = len(cf_matches[cf_matches["action"] == act])
                if cf_count > 0:
                    actions_str.append(f"**{act.upper()}**: {count} (+{cf_count} cf) ✅")
                elif count > MIN_SAMPLES:
                    actions_str.append(f"{act.upper()}: {count}")
            report_lines.append(f"- **{state}** -> " + " | ".join(actions_str))
            improvements_shown += 1
            
    with open(out_dir / "action_coverage_after.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logging.info(f"Coverage analysis saved to {out_dir}/action_coverage_after.md")
    
    # Save Augmented Dataset
    aug_dir = Path("data/processed/augmented/train")
    aug_dir.mkdir(parents=True, exist_ok=True)
    augmented_df.drop(columns=["canonical_state"]).to_parquet(aug_dir / "train_augmented.parquet")
    
    logging.info(f"Augmented dataset saved to {aug_dir}/train_augmented.parquet")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

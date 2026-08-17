import pandas as pd
import json

def inspect_dataset(file_path, num_rows=10000):
    print(f"Inspecting first {num_rows} rows of {file_path}")
    df = pd.read_csv(file_path, nrows=num_rows)
    
    print("\n--- Columns & Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Unique Value Counts ---")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} unique values")
        if unique_vals < 20:
            print(f"  Values: {df[col].unique().tolist()}")
            
    print("\n--- Action Parsing Sandbox ---")
    actions = df["actions_taken"].unique()
    print("Unique Action Strings (first 20):")
    for act in list(actions)[:20]:
        print(act)
        
    print("\n--- Hand Parsing Sandbox ---")
    hands = df["initial_hand"].unique()
    print("Unique Initial Hands (first 20):")
    for h in list(hands)[:20]:
        print(h)
        
    print("\n--- Player Final Hand Parsing Sandbox ---")
    finals = df["player_final"].unique()
    print("Unique Player Final Hands (first 20):")
    for h in list(finals)[:20]:
        print(h)
        
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default=None, help="Path to blackjack_simulator.csv")
    parser.add_argument("--num-rows", type=int, default=10000)
    args = parser.parse_args()

    file_path = args.file_path
    if not file_path:
        default_p = Path.home() / ".cache" / "kagglehub" / "datasets" / "dennisho" / "blackjack-hands" / "versions" / "1" / "blackjack_simulator.csv"
        file_path = str(default_p)

    inspect_dataset(file_path, args.num_rows)


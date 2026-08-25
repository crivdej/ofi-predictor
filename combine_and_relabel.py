"""
Combine all raw collected sessions into one larger, correctly-labeled training set.

Each raw CSV is a separate collection session (different day/time). Each one is
relabeled INDEPENDENTLY before concatenating — add_labels() looks ahead by elapsed
time within a sorted timestamp series, so relabeling the concatenation directly
would let the horizon lookup for the last row of one session "see" the first row
of a much later session as if it were 5 seconds ahead. Relabel-then-concat avoids
that entirely.

Usage:
    python combine_and_relabel.py
"""
import glob
import pandas as pd
from src.feature_engineer import FeatureEngineer
from relabel_data import RAW_COLS

OUTPUT_PATH = "data/training_data_combined_relabeled.csv"


def combine_and_relabel(prediction_horizon=5, deadzone_multiplier=0.0):
    raw_files = sorted(glob.glob("data/training_data_*.csv"))
    raw_files = [f for f in raw_files if "relabeled" not in f and "combined" not in f]

    all_labeled = []
    print(f"Found {len(raw_files)} raw session file(s):\n")
    for path in raw_files:
        df = pd.read_csv(path)[RAW_COLS]
        fe = FeatureEngineer(prediction_horizon=prediction_horizon, deadzone_multiplier=deadzone_multiplier)
        labeled = fe.add_labels(df)
        labeled['source_session'] = path
        all_labeled.append(labeled)
        print(f"  {path}: {len(df)} raw -> {len(labeled)} labeled")

    combined = pd.concat(all_labeled, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCombined: {len(combined)} labeled rows from {len(raw_files)} sessions")
    print(f"Label distribution:\n{combined['price_increase'].value_counts()}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    return combined


if __name__ == "__main__":
    combine_and_relabel()

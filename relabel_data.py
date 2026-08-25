"""
Relabel existing training data with the corrected time-based-horizon + deadzone
logic in FeatureEngineer.add_labels().

Doesn't need raw order book replay — mid_price/spread/timestamp per snapshot are
already saved in the CSV, which is all the fixed labeling logic needs. This exists
so the old (mislabeled) CSVs can be reprocessed without re-collecting live data.

Usage:
    python relabel_data.py data/training_data_20251118_152030.csv
"""
import sys
import pandas as pd
from src.feature_engineer import FeatureEngineer

# Columns that come from extract_features() — i.e. everything except the labels
# add_labels() itself produces (future_price, actual_horizon_seconds, price_increase).
RAW_COLS = [
    'timestamp', 'mid_price', 'spread', 'spread_pct', 'ofi_current', 'ofi_momentum',
    'total_bid_volume', 'total_ask_volume', 'volume_imbalance', 'volume_ratio',
    'bid_depth_l1', 'ask_depth_l1',
]


def relabel(input_path, prediction_horizon=5, deadzone_multiplier=0.0):
    df = pd.read_csv(input_path)
    df = df[RAW_COLS]  # drop stale future_price/price_increase from the old labeling

    fe = FeatureEngineer(prediction_horizon=prediction_horizon, deadzone_multiplier=deadzone_multiplier)
    relabeled = fe.add_labels(df)

    output_path = input_path.replace('.csv', '_relabeled.csv')
    relabeled.to_csv(output_path, index=False)

    print(f"\n{len(df)} raw snapshots -> {len(relabeled)} labeled rows "
          f"({len(df) - len(relabeled)} dropped: end-of-session or inside the deadzone)")
    print(f"Actual horizon achieved: mean={relabeled['actual_horizon_seconds'].mean():.2f}s, "
          f"min={relabeled['actual_horizon_seconds'].min():.2f}s, "
          f"max={relabeled['actual_horizon_seconds'].max():.2f}s")
    print(f"\nLabel distribution:")
    print(relabeled['price_increase'].value_counts())
    print(f"\nSaved to: {output_path}")
    return relabeled


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/training_data_20251118_152030.csv"
    relabel(path)

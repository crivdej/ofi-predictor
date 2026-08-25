"""
Feature Engineer: Converts order book data to ML features
"""
import numpy as np
import pandas as pd

class FeatureEngineer:
    """Extracts ML features from order book snapshots and OFI calculations"""
    
    def __init__(self, prediction_horizon=5, deadzone_multiplier=0.0):
        """
        Args:
            prediction_horizon: seconds ahead to predict. Applied by actual elapsed
                time (see add_labels), not by assuming a fixed row-to-row interval.
            deadzone_multiplier: a price move only counts as a real up/down move if
                it's larger than `deadzone_multiplier * spread`; smaller moves are
                dropped rather than forced into a binary label (see add_labels).
                Default 0.0 still drops exact ties (no move at all) but keeps any
                real nonzero move — chosen empirically: on the Nov 18 session,
                larger multipliers (tried up to 0.5x spread) threw away 50-90% of
                rows for no accuracy benefit, because mean spread (~$60) is wide
                relative to how far BTC actually moves in 5-6s during a quiet
                (Saturday night) session. Revisit this once there's data from a
                more liquid/active period — a nonzero deadzone may earn its keep
                once tighter spreads make it a meaningful noise filter again.
        """
        self.prediction_horizon = prediction_horizon
        self.deadzone_multiplier = deadzone_multiplier
        self.feature_history = []
    
    def extract_features(self, order_book, ofi_result, prev_ofi=None):
        """Extract features from current order book and OFI"""
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Price features
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100
        
        # OFI features
        ofi_current = ofi_result['ofi_total']
        ofi_momentum = ofi_current - prev_ofi if prev_ofi is not None else 0
        
        # Volume features
        total_bid_vol = ofi_result['total_bid_volume']
        total_ask_vol = ofi_result['total_ask_volume']
        volume_imbalance = ofi_result['volume_imbalance']
        volume_ratio = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
        
        # Depth features
        bid_depth_l1 = float(bids[0][1]) if len(bids) > 0 else 0
        ask_depth_l1 = float(asks[0][1]) if len(asks) > 0 else 0
        
        return {
            'timestamp': order_book['timestamp'],
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            'ofi_current': ofi_current,
            'ofi_momentum': ofi_momentum,
            'total_bid_volume': total_bid_vol,
            'total_ask_volume': total_ask_vol,
            'volume_imbalance': volume_imbalance,
            'volume_ratio': volume_ratio,
            'bid_depth_l1': bid_depth_l1,
            'ask_depth_l1': ask_depth_l1,
        }
    
    def add_labels(self, features_df):
        """Add price movement labels `prediction_horizon` seconds ahead.

        Two correctness fixes vs. the original version:
        1. Looks ahead by actual elapsed TIME (searchsorted on timestamp), not by a
           fixed one-row shift. Snapshots come from REST polling, not a clean tick
           stream, so "next row" was never really "prediction_horizon seconds
           ahead" — it was whatever the polling interval happened to be that tick.
        2. Applies a deadzone: only moves bigger than `deadzone_multiplier * spread`
           count as a real up/down. In this dataset, ~54% of consecutive mid-price
           snapshots move by under $0.01 — without a deadzone, that noise gets
           forced into the "decrease" class (since the old code used a strict `>`)
           and swamps whatever real signal OFI has.
        """
        df = features_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        horizon = pd.Timedelta(seconds=self.prediction_horizon)
        target_times = df['timestamp'] + horizon
        future_idx = df['timestamp'].searchsorted(target_times, side='left')

        # Rows with no snapshot far enough in the future (end of the session) can't
        # be labeled. Look up against the FULL arrays first (clipping just to avoid
        # an out-of-bounds index — those rows get dropped right after anyway), then
        # filter — indexing after filtering would point at the wrong rows entirely.
        mid_price_arr = df['mid_price'].to_numpy()
        timestamp_arr = df['timestamp'].to_numpy()
        valid = future_idx < len(df)
        safe_idx = np.clip(future_idx, 0, len(df) - 1)

        df['future_price'] = mid_price_arr[safe_idx]
        df['actual_horizon_seconds'] = (timestamp_arr[safe_idx] - timestamp_arr) / np.timedelta64(1, 's')
        df = df.loc[valid].reset_index(drop=True)

        price_change = df['future_price'] - df['mid_price']
        deadzone = self.deadzone_multiplier * df['spread']

        label = pd.Series(np.nan, index=df.index)
        label[price_change > deadzone] = 1
        label[price_change < -deadzone] = 0
        df['price_increase'] = label

        before = len(df)
        df = df.dropna(subset=['price_increase']).reset_index(drop=True)
        df['price_increase'] = df['price_increase'].astype(int)
        dropped = before - len(df)
        print(f"[feature_engineer] dropped {dropped}/{before} rows inside the "
              f"deadzone (move smaller than {self.deadzone_multiplier}x spread)")

        return df
    
    def create_training_data(self, feature_list):
        """Convert feature list to labeled training DataFrame"""
        if len(feature_list) < 2:
            return None
        
        df = pd.DataFrame(feature_list)
        df = self.add_labels(df)
        
        return df
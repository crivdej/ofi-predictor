"""
Feature Engineer: Converts order book data to ML features
"""
import numpy as np
import pandas as pd

class FeatureEngineer:
    """Extracts ML features from order book snapshots and OFI calculations"""
    
    def __init__(self, prediction_horizon=5):
        self.prediction_horizon = prediction_horizon
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
        """Add price movement labels"""
        features_df = features_df.copy()
        features_df['future_price'] = features_df['mid_price'].shift(-1)
        features_df['price_increase'] = (
            features_df['future_price'] > features_df['mid_price']
        ).astype(int)
        return features_df[:-1]  # Remove last row (no future price)
    
    def create_training_data(self, feature_list):
        """Convert feature list to labeled training DataFrame"""
        if len(feature_list) < 2:
            return None
        
        df = pd.DataFrame(feature_list)
        df = self.add_labels(df)
        
        return df
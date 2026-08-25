"""
OFI Calculator: Measures order flow imbalance
"""
import numpy as np

class OFICalculator:
    """
    Calculates Order Flow Imbalance between consecutive order book snapshots.
    Positive OFI = buying pressure, Negative OFI = selling pressure
    """
    
    def __init__(self, num_levels=5):
        self.num_levels = num_levels
    
    def calculate_ofi(self, prev_book, curr_book):
        """
        Calculate OFI between two order book snapshots
        
        Returns:
            dict with OFI metrics
        """
        # Extract volumes
        prev_bid_vols = [float(bid[1]) for bid in prev_book['bids'][:self.num_levels]]
        curr_bid_vols = [float(bid[1]) for bid in curr_book['bids'][:self.num_levels]]
        prev_ask_vols = [float(ask[1]) for ask in prev_book['asks'][:self.num_levels]]
        curr_ask_vols = [float(ask[1]) for ask in curr_book['asks'][:self.num_levels]]
        
        # Calculate changes
        bid_changes = np.array(curr_bid_vols) - np.array(prev_bid_vols)
        ask_changes = np.array(curr_ask_vols) - np.array(prev_ask_vols)
        
        # OFI = bid increases - ask increases
        ofi_total = np.sum(bid_changes) - np.sum(ask_changes)
        # Per-level breakdown. Not yet wired up: FeatureEngineer.extract_features()
        # only reads ofi_total from this dict, so ofi_by_level currently only feeds
        # calculate_weighted_ofi() below (itself not called anywhere yet either).
        # Left in as plausible future feature scope rather than removed.
        ofi_by_level = bid_changes - ask_changes
        
        # Volume metrics
        total_bid_volume = np.sum(curr_bid_vols)
        total_ask_volume = np.sum(curr_ask_vols)
        volume_imbalance = total_bid_volume - total_ask_volume
        
        return {
            'ofi_total': ofi_total,
            'ofi_by_level': ofi_by_level,
            'bid_changes': bid_changes,
            'ask_changes': ask_changes,
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'volume_imbalance': volume_imbalance
        }
    
    def calculate_weighted_ofi(self, prev_book, curr_book, weights=None):
        """Calculate weighted OFI with higher weight for closer levels.

        Not currently called anywhere in the pipeline (feature_engineer.py only
        uses ofi_total from calculate_ofi()) — left in as unused-but-plausible
        future feature scope rather than removed.
        """
        if weights is None:
            weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2][:self.num_levels])
        
        ofi_result = self.calculate_ofi(prev_book, curr_book)
        weighted_ofi = np.sum(ofi_result['ofi_by_level'] * weights)
        
        return weighted_ofi
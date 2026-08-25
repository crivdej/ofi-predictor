"""
Collect Training Data: Run data collector and save features for ML training
"""
from src.data_collector import OrderBookCollector
from src.ofi_calculator import OFICalculator
from src.feature_engineer import FeatureEngineer
import time
import pandas as pd
from datetime import datetime

class TrainingDataCollector:
    """
    Collects order book data, calculates OFI, extracts features, and saves for training
    """
    
    def __init__(self, duration_minutes=10, update_interval=2.0):
        """
        Args:
            duration_minutes: How long to collect data (default: 10 minutes)
            update_interval: Seconds between snapshots (default: 2)
        """
        self.duration_minutes = duration_minutes
        self.update_interval = update_interval
        self.collector = OrderBookCollector(update_interval=update_interval)
        self.ofi_calculator = OFICalculator(num_levels=5)
        self.feature_engineer = FeatureEngineer(prediction_horizon=5)
        self.features = []
        self.prev_ofi = None
        
    def collect(self):
        """Collect data for specified duration"""
        print(f" Collecting training data for {self.duration_minutes} minutes...")
        print(f"⏱️  Update interval: {self.update_interval} seconds")
        print(f"🎯 Prediction horizon: {self.feature_engineer.prediction_horizon} seconds")
        print("=" * 80)
        
        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        snapshot_count = 0
        
        try:
            while time.time() < end_time:
                # Get order book
                order_book = self.collector.get_order_book()
                
                if order_book:
                    self.collector.order_books.append(order_book)
                    snapshot_count += 1
                    
                    # Calculate OFI if we have previous snapshot
                    if len(self.collector.order_books) >= 2:
                        ofi_result = self.ofi_calculator.calculate_ofi(
                            self.collector.order_books[-2],
                            self.collector.order_books[-1]
                        )
                        
                        # Extract features
                        features = self.feature_engineer.extract_features(
                            order_book,
                            ofi_result,
                            self.prev_ofi
                        )
                        self.features.append(features)
                        self.prev_ofi = ofi_result['ofi_total']
                        
                        # Display progress
                        if snapshot_count % 10 == 0:
                            elapsed = (time.time() - start_time) / 60
                            remaining = self.duration_minutes - elapsed
                            print(f"✓ Collected {snapshot_count} snapshots | "
                                  f"{len(self.features)} features | "
                                  f"{remaining:.1f} min remaining | "
                                  f"OFI: {ofi_result['ofi_total']:+.4f}")
                    
                    # Memory management
                    if len(self.collector.order_books) > 100:
                        self.collector.order_books.pop(0)
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Collection interrupted by user")
        
        print("\n" + "=" * 80)
        print(f"✅ Collection complete!")
        print(f" Total snapshots: {snapshot_count}")
        print(f" Total features: {len(self.features)}")
        
        return self.save_data()
    
    def save_data(self):
        """Save collected data to CSV"""
        if len(self.features) < 2:
            print("❌ Not enough data collected (need at least 2 snapshots)")
            return None
        
        # Create training DataFrame
        df = self.feature_engineer.create_training_data(self.features)
        
        if df is None or len(df) == 0:
            print("❌ Failed to create training data")
            return None
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/training_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Data saved to: {filename}")
        print(f" Shape: {df.shape}")
        print(f"\n Feature columns:")
        for col in df.columns:
            if col != 'price_increase':
                print(f"   - {col}")
        
        print(f"\n🎯 Label distribution:")
        print(df['price_increase'].value_counts())
        print(f"\n   Baseline accuracy (always predict majority): "
              f"{df['price_increase'].value_counts().max() / len(df) * 100:.1f}%")
        
        return df

if __name__ == "__main__":
    # Collect 120 minutes of data (or stop early with Ctrl+C)
    collector = TrainingDataCollector(duration_minutes=120, update_interval=2.0)
    df = collector.collect()
    
    if df is not None:
        print("\n✅ Training data ready for modeling!")
        print("   Next step: Train prediction model with this data")
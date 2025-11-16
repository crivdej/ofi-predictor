"""
Live Predictor: Real-time price movement predictions on streaming market data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_collector import OrderBookCollector
from src.ofi_calculator import OFICalculator
from src.feature_engineer import FeatureEngineer
from src.predictor import OFIPredictor
import time
from datetime import datetime
from collections import deque

class LivePredictor:
    """Makes real-time predictions on live market data"""
    
    def __init__(self, model_path='models/ofi_predictor.pkl', update_interval=2.0):
        self.update_interval = update_interval
        
        # Initialize components
        self.collector = OrderBookCollector(update_interval=update_interval)
        self.ofi_calculator = OFICalculator(num_levels=5)
        self.feature_engineer = FeatureEngineer(prediction_horizon=5)
        
        # Load model
        print("Loading trained model...")
        self.predictor = OFIPredictor()
        self.predictor.load_model(model_path)
        
        # Tracking
        self.predictions = deque(maxlen=100)
        self.prev_ofi = None
        self.prediction_count = 0
        
    def run(self, duration_minutes=None):
        """
        Run live predictions
        
        Args:
            duration_minutes: Duration to run (None = infinite)
        """
        print("\n" + "=" * 80)
        print("LIVE OFI PREDICTOR")
        print("=" * 80)
        print(f"Update interval: {self.update_interval}s")
        print(f"Prediction horizon: 5s")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes else None
        
        try:
            while True:
                if end_time and time.time() >= end_time:
                    break
                
                order_book = self.collector.get_order_book()
                
                if order_book and len(self.collector.order_books) >= 1:
                    self.collector.order_books.append(order_book)
                    
                    if len(self.collector.order_books) >= 2:
                        ofi_result = self.ofi_calculator.calculate_ofi(
                            self.collector.order_books[-2],
                            self.collector.order_books[-1]
                        )
                        
                        features = self.feature_engineer.extract_features(
                            order_book, ofi_result, self.prev_ofi
                        )
                        
                        prediction = self.predictor.predict(features)
                        
                        self._display_prediction(order_book, ofi_result, prediction, features)
                        
                        self.predictions.append({
                            'timestamp': datetime.now(),
                            'prediction': prediction['prediction'],
                            'confidence': prediction['confidence'],
                            'mid_price': features['mid_price'],
                            'ofi': ofi_result['ofi_total']
                        })
                        
                        self.prev_ofi = ofi_result['ofi_total']
                        self.prediction_count += 1
                    
                    if len(self.collector.order_books) > 100:
                        self.collector.order_books.pop(0)
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping predictions...")
            self._show_summary()
    
    def _display_prediction(self, order_book, ofi_result, prediction, features):
        """Display current prediction (compact format)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        mid_price = features['mid_price']
        spread = features['spread']
        ofi = ofi_result['ofi_total']
        
        pred = "UP" if prediction['prediction'] == 1 else "DOWN"
        prob = prediction['probability_increase'] if prediction['prediction'] == 1 else prediction['probability_decrease']
        conf_level = "HIGH" if prediction['confidence'] > 0.7 else "MED" if prediction['confidence'] > 0.55 else "LOW"
        
        ofi_dir = "BUY" if ofi > 0 else "SELL" if ofi < 0 else "NEUTRAL"
        
        print(f"[{timestamp}] #{self.prediction_count + 1:3d} | "
              f"Price: ${mid_price:>9,.2f} | "
              f"OFI: {ofi:>+7.3f} ({ofi_dir:7s}) | "
              f"Pred: {pred:4s} {prob:>5.1%} ({conf_level:4s})")
    
    def _show_summary(self):
        """Display session statistics"""
        if len(self.predictions) == 0:
            return
        
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        
        total = len(self.predictions)
        up_preds = sum(1 for p in self.predictions if p['prediction'] == 1)
        down_preds = total - up_preds
        avg_conf = sum(p['confidence'] for p in self.predictions) / total
        
        print(f"Total predictions: {total}")
        print(f"  UP predictions:   {up_preds:3d} ({up_preds/total:>5.1%})")
        print(f"  DOWN predictions: {down_preds:3d} ({down_preds/total:>5.1%})")
        print(f"  Average confidence: {avg_conf:.1%}")
        
        if len(self.predictions) > 1:
            start_price = self.predictions[0]['mid_price']
            end_price = self.predictions[-1]['mid_price']
            price_change = end_price - start_price
            price_change_pct = (price_change / start_price) * 100
            
            print(f"\nPrice movement:")
            print(f"  Start: ${start_price:,.2f}")
            print(f"  End:   ${end_price:,.2f}")
            print(f"  Change: ${price_change:+,.2f} ({price_change_pct:+.2f}%)")

if __name__ == "__main__":
    live = LivePredictor(
        model_path='models/ofi_predictor.pkl',
        update_interval=2.0
    )
    live.run(duration_minutes=5)
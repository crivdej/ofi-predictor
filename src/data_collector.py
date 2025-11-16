"""
Data Collector: Fetches order book data from Binance.US
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from binance.client import Client
from datetime import datetime
from config import SYMBOL
import time

class OrderBookCollector:
    """Collects order book data from Binance.US using REST API"""
    
    def __init__(self, update_interval=1.0):
        self.symbol = SYMBOL
        self.client = Client(api_key=None, api_secret=None, tld='us')
        self.update_interval = update_interval
        self.order_books = []
        
    def get_order_book(self):
        """Fetch current order book snapshot"""
        try:
            depth = self.client.get_order_book(symbol=self.symbol, limit=5)
            return {
                'timestamp': datetime.now(),
                'bids': depth['bids'],
                'asks': depth['asks']
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None
    
    def start(self, verbose=True):
        """Start collecting order book data"""
        if verbose:
            print(f"Connecting to Binance.US {self.symbol}...")
            print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                order_book = self.get_order_book()
                
                if order_book:
                    if verbose:
                        self.display_order_book(order_book)
                    
                    self.order_books.append(order_book)
                    
                    if len(self.order_books) > 100:
                        self.order_books.pop(0)
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            if verbose:
                print(f"\nStopped. Collected {len(self.order_books)} snapshots")
    
    def display_order_book(self, order_book):
        """Display order book snapshot"""
        timestamp = order_book['timestamp'].strftime("%H:%M:%S")
        bids = order_book['bids']
        asks = order_book['asks']
        
        print(f"[{timestamp}]")
        print("📗 BIDS:")
        for i, (price, qty) in enumerate(bids, 1):
            print(f"  L{i}: ${float(price):,.2f} x {float(qty):.6f}")
        
        print("📕 ASKS:")
        for i, (price, qty) in enumerate(asks, 1):
            print(f"  L{i}: ${float(price):,.2f} x {float(qty):.6f}")
        
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            mid_price = (best_bid + best_ask) / 2
            
            print(f"Mid: ${mid_price:,.2f} | Spread: ${spread:.2f} ({spread_pct:.4f}%)")
        
        print("-" * 60 + "\n")

if __name__ == "__main__":
    collector = OrderBookCollector(update_interval=2.0)
    collector.start()
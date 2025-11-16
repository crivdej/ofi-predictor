"""
Main entry point for running the data collector
Run this from project root: python run_collector.py
"""

from src.data_collector import OrderBookCollector

if __name__ == "__main__":
    collector = OrderBookCollector()
    collector.start()
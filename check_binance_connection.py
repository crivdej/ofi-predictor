"""
Manual smoke-test script for the Binance.US REST connection — not a pytest
suite (run directly with `python check_binance_connection.py`; makes a real
network call). Renamed from test_binance.py so `pytest` doesn't try to
collect and import it, which would fire that network call as a side effect.
"""
from binance.client import Client

# Create client for Binance.US
client = Client(api_key=None, api_secret=None, tld='us')

# Get exchange info
print("Testing Binance.US connection...\n")

try:
    # Get all trading pairs
    exchange_info = client.get_exchange_info()
    
    # Find BTC pairs
    btc_symbols = [s['symbol'] for s in exchange_info['symbols'] if 'BTC' in s['symbol'] and 'USD' in s['symbol']]
    
    print("Available BTC/USD trading pairs:")
    for symbol in btc_symbols[:10]:
        print(f"  - {symbol}")
    
    # Try to get order book for BTCUSD
    print("\nTrying BTCUSD order book:")
    depth = client.get_order_book(symbol='BTCUSD', limit=5)
    print(f"  ✓ Bids: {len(depth['bids'])} levels")
    print(f"  ✓ Asks: {len(depth['asks'])} levels")
    print(f"  Best bid: ${float(depth['bids'][0][0]):,.2f}")
    print(f"  Best ask: ${float(depth['asks'][0][0]):,.2f}")
    
except Exception as e:
    print(f"Error: {e}")
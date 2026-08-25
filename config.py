"""
Configuration file for OFI Predictor project

Only values actually read by the code live here (see SYMBOL, used by
src/data_collector.py). Several other settings that used to live in this file
(OFI level count, update interval, prediction horizon, train/test split, log
config, Monte Carlo params, result-directory paths, etc.) were never imported
anywhere — every place that needed one of those values hardcoded its own
literal instead, so the config here was silently disconnected from actual
behavior. Removed rather than kept as unread, misleading documentation.
"""

# Exchange Configuration
SYMBOL = "BTCUSD"

# Defined but intentionally not wired up: data_collector.py polls the REST API
# every 2s rather than opening this WebSocket stream (see README's Data
# Collection section). Kept here as a marker of that unimplemented path rather
# than silently removed.
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/bnbbtc@depth"
"""
Configuration file for OFI Predictor project
"""

# Exchange Configuration
EXCHANGE = "binance"
SYMBOL = "BTCUSD"
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/bnbbtc@depth"

# OFI Calculation Parameters
OFI_LEVELS = 5
UPDATE_FREQUENCY = 1.0
LOOKBACK_WINDOW = 100

# Model Parameters
PREDICTION_HORIZON = 5
TRAIN_TEST_SPLIT = 0.8
MIN_TRAINING_SAMPLES = 1000

# Monte Carlo Parameters
N_SIMULATIONS = 1000
SIMULATION_LENGTH = 3600
VOLATILITY_SCENARIOS = [0.005, 0.01, 0.02, 0.03, 0.05]

# File Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "ofi_predictor.log"
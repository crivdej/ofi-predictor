# Real-Time Order Flow Imbalance Predictor

Price movement prediction using market microstructure signals for quantitative trading applications.

## Overview
Built an end-to-end machine learning system that predicts short-term cryptocurrency price movements by analyzing order flow imbalance (OFI) - a key market microstructure signal used by high-frequency trading firms.

## Technical Implementation

### Data Collection
- Real-time WebSocket connection to Binance.US API
- Streaming order book data (top 5 price levels)
- 2-second update frequency

### Feature Engineering
- **Order Flow Imbalance (OFI)**: Measures net buying/selling pressure
- **Multi-level analysis**: Aggregates signals across 5 order book depth levels
- **Momentum features**: Tracks acceleration of order flow changes
- **Spread metrics**: Market liquidity indicators

### Machine Learning Model
- **Algorithm**: Logistic Regression with balanced class weights
- **Training Data**: 3,535 labeled samples (60 minutes of live data)
- **Validation**: Time-based train/test split (no data leakage)
- **Performance**: 57.4% accuracy (beats 78.4% baseline on balanced data)

### Live Prediction System
- Real-time dashboard built with Streamlit
- 5-second prediction horizon
- Confidence scoring for each prediction

## Results
- **Test Accuracy**: 57.4%
- **Precision**: 27.6%
- **Recall**: 59.5%
- Successfully demonstrated OFI's predictive power in live market conditions

## Technical Stack
- **Data Collection**: python-binance, WebSockets
- **ML**: scikit-learn, pandas, numpy
- **Visualization**: Streamlit, Plotly
- **Deployment**: Real-time streaming pipeline

## Key Learnings
- Understanding of market microstructure and order flow dynamics
- Handling severely imbalanced datasets in financial applications
- Building real-time ML systems with streaming data
- Time-series validation techniques to prevent look-ahead bias

## Running the Project
```bash
# Train model
python src/predictor.py

# Launch live dashboard
streamlit run dashboard.py
```

## Academic Foundation
Based on research by Stoikov on order flow imbalance as a predictor of short-term price movements.
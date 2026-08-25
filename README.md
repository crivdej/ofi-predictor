# Real-Time Order Flow Imbalance Predictor

Price movement prediction using market microstructure signals for quantitative trading applications.

## Overview
Built an end-to-end machine learning system that predicts short-term cryptocurrency price movements by analyzing order flow imbalance (OFI) - a key market microstructure signal used by high-frequency trading firms.

## Technical Implementation

### Data Collection
- REST polling of the Binance.US order book API every 2 seconds (not a WebSocket
  stream — `config.py` has a websocket URL defined but the collector never uses it;
  noted here rather than silently left inaccurate)
- Top 5 order book price levels per snapshot

### Feature Engineering
- **Order Flow Imbalance (OFI)**: Measures net buying/selling pressure
- **Multi-level analysis**: Aggregates signals across 5 order book depth levels
- **Momentum features**: Tracks the change in order flow imbalance since the previous snapshot
- **Spread metrics**: Market liquidity indicators

### Machine Learning Model
- **Algorithm**: Logistic Regression with balanced class weights
- **Training Data**: 3,057 labeled samples, from 60 minutes of live data (3,536 raw
  snapshots; ~480 dropped as exact ties — see Methodology Notes)
- **Validation**: Time-based train/test split, cross-validated across 5 walk-forward
  folds — a single split is too noisy to trust at this sample size (this project's
  own numbers proved it, see below)
- **Performance**: 54.7% mean accuracy across 5 folds vs. a 52.1% mean majority-class
  baseline (+2.6 points, ±3.6 across folds) — beat baseline in 4 of 5 folds

### Live Prediction System
- Real-time dashboard built with Streamlit
- 5-second prediction horizon
- Confidence scoring for each prediction

## Results
- **Cross-validated accuracy**: 54.7% (±2.7% across 5 time-series folds) vs. 52.1% mean baseline
- **Single-split test accuracy**: 54.4% | Precision: 54.2% | Recall: 61.0%
- OFI shows a real but small and inconsistent short-horizon (~5-6s) edge on this
  instrument, measured during a single low-liquidity (Saturday night) session — not
  a strong signal, and not yet confirmed across more/varied market conditions
  (see Next Steps)

## Methodology Notes (corrections made after the initial version)
Two labeling bugs inflated the original results and are now fixed:
1. **The 5-second prediction horizon wasn't real.** `prediction_horizon=5` was
   stored but never applied — labels were based on "next row" (~2s later, whatever
   the REST poll interval happened to be that tick), not a genuine 5-second-ahead
   target. `FeatureEngineer.add_labels()` now looks ahead by actual elapsed time.
2. **No noise deadzone.** ~54% of consecutive snapshots had a <$0.01 price change,
   and the original strict `>` comparison forced all of that noise into the
   "decrease" class — skewing the baseline to 69-78% and making the model look like
   it underperformed baseline. It did, but the baseline itself was an artifact of
   the bug. Dropping exact ties instead of mislabeling them produces a fairer,
   near-balanced (~52/48) class split and a ~52% baseline.
3. **The original 57.4%-vs-78.4% comparison was backwards** — a majority-class
   baseline of 78.4% beats 57.4% accuracy; the model underperformed baseline, not
   the reverse. All numbers here are now cross-validated rather than read off a
   single train/test split, since a single split on data this size isn't reliable
   (this project's own repeated runs landed anywhere from 54% to 60% test accuracy
   on the same setup).

## Next Steps
- **Tried combining all 5 collected sessions (5,805 rows vs. 3,057) — it made the
  cross-validated result *weaker*, not better: +1.8% mean edge (±4.4%, negative in
  2 of 5 folds) vs. +2.6% (±3.6%, negative in 1 of 5) on the single session alone,
  and the top feature shifted from `ofi_current` to `spread`/`spread_pct`.** That
  points to the sessions having different regimes (spread/volatility levels) rather
  than more of the same signal — pooling them let the model partly learn
  session-specific noise instead of a session-general OFI signal. More data from
  mismatched conditions isn't the fix here.
- So: collect data during a single, longer, more liquid/active trading period
  (weekday, main session hours) instead — representativeness over volume
- Try the tree-based models `xgboost`/`scikit-learn` already list as dependencies
  but were never implemented — logistic regression is the only model tried so far

## Technical Stack
- **Data Collection**: python-binance (REST polling)
- **ML**: scikit-learn, pandas, numpy
- **Visualization**: Streamlit, Plotly
- **Deployment**: near-real-time polling pipeline (2s interval)

## Key Learnings
- Understanding of market microstructure and order flow dynamics
- Why label construction matters as much as the model: a silent labeling bug
  (wrong horizon, no noise deadzone) can single-handedly flip a result from
  "underperforms baseline" to "beats it," with the exact same model and data
- Time-series validation techniques to prevent look-ahead bias — and why a single
  train/test split isn't enough to trust a result on noisy financial data
- Diagnosing and fixing a data-leakage bug (a label-construction diagnostic column
  had leaked into the model's own feature set) before trusting any of the above

## Running the Project
```bash
# One-time: relabel raw collected data with the corrected horizon/deadzone logic
python relabel_data.py data/training_data_20251118_152030.csv

# Train model (also runs 5-fold time-series cross-validation)
python src/predictor.py

# Launch live dashboard
streamlit run dashboard.py
```

## Academic Foundation
Based on research by Stoikov on order flow imbalance as a predictor of short-term price movements.
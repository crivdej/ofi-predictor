"""
Real-Time OFI Prediction Dashboard
Run with: streamlit run dashboard.py
"""
import streamlit as st
import sys
sys.path.append('.')

from src.data_collector import OrderBookCollector
from src.ofi_calculator import OFICalculator
from src.feature_engineer import FeatureEngineer
from src.predictor import OFIPredictor
import plotly.graph_objects as go
import time
from datetime import datetime
from collections import deque

st.set_page_config(
    page_title="OFI Live Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .stMetric {background-color: #f8f9fa; padding: 0.7rem; border-radius: 0.3rem;}
    .stMetric label {font-size: 0.5rem;}
    /* Target metric values directly */
    div[data-testid="stMetricValue"] {
        font-size: 0.95rem !important;
    }

    h1 {
        color: #1f77b4;
        font-weight: 5300;
    }
    h3 {
        color: #444;
        font-weight: 200;
        font-size: 1.0rem;
    }
    .order-book {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        line-height: 0.8;
    }
    /* Vertical dividers*/
    [data-testid="column"]:not(:last-child) {
        border-right: 1px solid #e0e0e0;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collector = OrderBookCollector(update_interval=2.0)
    st.session_state.ofi_calculator = OFICalculator(num_levels=5)
    st.session_state.feature_engineer = FeatureEngineer(prediction_horizon=5)
    st.session_state.predictor = OFIPredictor()
    st.session_state.predictor.load_model('models/ofi_predictor.pkl')
    st.session_state.prev_ofi = None
    st.session_state.predictions = deque(maxlen=50)
    st.session_state.prices = deque(maxlen=50)
    st.session_state.ofis = deque(maxlen=50)
    st.session_state.timestamps = deque(maxlen=50)
    st.session_state.initialized = True

# Sidebar - Model Info
with st.sidebar:
    st.markdown("### Model Performance")
    metrics = st.session_state.predictor.metrics
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics['test_accuracy']:.1%}")
        st.metric("Precision", f"{metrics['test_precision']:.1%}")
    with col2:
        st.metric("Recall", f"{metrics['test_recall']:.1%}")
        st.metric("Baseline", f"{metrics['baseline_accuracy']:.1%}")
    
    st.markdown("---")
    st.markdown("### Model Details")
    st.caption("Type: Logistic Regression")
    st.caption(f"Features: {len(st.session_state.predictor.feature_columns)}")
    st.caption("Update Interval: 2 seconds")
    st.caption("Prediction Horizon: 5 seconds")

# Header
st.title("Live Order Flow Imbalance Predictor")
st.caption("Real-time price movement prediction using market microstructure signals")
st.markdown("---")

# Main layout
col1, col2, col3 = st.columns([3.5, 2, 2])

with col1:
    st.markdown("### Market Data")
    price_display = st.empty()
    st.markdown("")
    order_book_display = st.empty()

with col2:
    st.markdown("### Model Prediction")
    prediction_display = st.empty()
    st.markdown("")
    ofi_display = st.empty()

with col3:
    st.markdown("### Session Stats")
    stats_display = st.empty()

# Historical Data Charts
st.markdown("---")
st.markdown("### Order Flow Imbalance History")
ofi_chart = st.empty()


# Main loop
while True:
    order_book = st.session_state.collector.get_order_book()
    
    if order_book:
        st.session_state.collector.order_books.append(order_book)
        
        if len(st.session_state.collector.order_books) >= 2:
            # Calculate OFI
            ofi_result = st.session_state.ofi_calculator.calculate_ofi(
                st.session_state.collector.order_books[-2],
                st.session_state.collector.order_books[-1]
            )
            
            # Extract features and predict
            features = st.session_state.feature_engineer.extract_features(
                order_book, ofi_result, st.session_state.prev_ofi
            )
            prediction = st.session_state.predictor.predict(features)
            
            # Store data
            timestamp = datetime.now()
            st.session_state.timestamps.append(timestamp)
            st.session_state.prices.append(features['mid_price'])
            st.session_state.ofis.append(ofi_result['ofi_total'])
            st.session_state.predictions.append(prediction)
            st.session_state.prev_ofi = ofi_result['ofi_total']
            
            # Extract values
            bids = order_book['bids']
            asks = order_book['asks']
            mid_price = features['mid_price']
            spread = features['spread']
            spread_pct = features['spread_pct']
            ofi = ofi_result['ofi_total']
            ofi_momentum = features['ofi_momentum']
            
            # Price metrics
            with price_display.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Mid Price", f"${mid_price:,.2f}")
                m2.metric("Spread", f"${spread:.2f}")
                m3.metric("Spread %", f"{spread_pct:.4f}%")
            
            # Order book
            with order_book_display.container():
                st.markdown('<div class="order-book">', unsafe_allow_html=True)
                ob1, ob2 = st.columns(2)
                
                with ob1:
                    st.markdown("**Bids**")
                    for i, (price, qty) in enumerate(bids[:5], 1):
                        st.text(f"{i}  ${float(price):>10,.2f}  ×  {float(qty):>6.4f}")
                
                with ob2:
                    st.markdown("**Asks**")
                    for i, (price, qty) in enumerate(asks[:5], 1):
                        st.text(f"{i}  ${float(price):>10,.2f}  ×  {float(qty):>6.4f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Predictions
            with prediction_display.container():
                pred_direction = "UP" if prediction['prediction'] == 1 else "DOWN"
                pred_prob = prediction['probability_increase'] if prediction['prediction'] == 1 else prediction['probability_decrease']
                conf_level = prediction['confidence']
                
                # Prediction box
                if prediction['prediction'] == 1:
                    st.success(f"**Predicted Direction:** ↗ {pred_direction}")
                else:
                    st.error(f"**Predicted Direction:** ↘ {pred_direction}")
                
                # Probability bar
                st.progress(pred_prob)
                st.caption(f"Probability: {pred_prob:.1%}")
                
                # Confidence
                if conf_level > 0.7:
                    st.info("**Confidence:** High")
                elif conf_level > 0.55:
                    st.warning("**Confidence:** Medium")
                else:
                    st.warning("**Confidence:** Low")
            
            # OFI display
            with ofi_display.container():
                ofi_direction = "Buying Pressure" if ofi > 0 else "Selling Pressure" if ofi < 0 else "Balanced"
                
                if ofi > 0:
                    st.success(f"**{ofi_direction}**")
                elif ofi < 0:
                    st.error(f"**{ofi_direction}**")
                else:
                    st.info(f"**{ofi_direction}**")
                
                c1, c2 = st.columns(2)
                c1.metric("OFI", f"{ofi:+.4f}")
                c2.metric("Momentum", f"{ofi_momentum:+.4f}")
            
            # Statistics
            with stats_display.container():
                total = len(st.session_state.predictions)
                up_preds = sum(1 for p in st.session_state.predictions if p['prediction'] == 1)
                down_preds = total - up_preds
                
                st.metric("Predictions", total)
                st.markdown("")
                
                s1, s2 = st.columns(2)
                s1.metric("↗ Up", f"{up_preds}")
                s1.caption(f"{up_preds/total*100:.0f}%" if total > 0 else "0%")
                
                s2.metric("↘ Down", f"{down_preds}")
                s2.caption(f"{down_preds/total*100:.0f}%" if total > 0 else "0%")
            
            # OFI chart
            if len(st.session_state.ofis) > 1:
                with ofi_chart.container():
                    fig = go.Figure()
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in st.session_state.ofis]
                    fig.add_trace(go.Bar(
                        x=list(st.session_state.timestamps),
                        y=list(st.session_state.ofis),
                        marker_color=colors,
                        name='OFI'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    fig.update_layout(
                        title="Order Flow Imbalance (Last 50 Updates)",
                        height=280,
                        margin=dict(l=20, r=20, t=35, b=20),
                        showlegend=False,
                        xaxis_title="Time",
                        yaxis_title="OFI",
                        hovermode='x unified',
                        plot_bgcolor='white',
                        yaxis=dict(gridcolor='lightgray', zeroline=True)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"ofi_{len(st.session_state.ofis)}")
            
            # Memory management
            if len(st.session_state.collector.order_books) > 100:
                st.session_state.collector.order_books.pop(0)
    
    time.sleep(2)
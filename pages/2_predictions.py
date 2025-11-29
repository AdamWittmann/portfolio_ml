# pages/2_🔮_Predictions.py - Stock Predictions

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Predictions", page_icon="🔮", layout="wide")

# Title
st.title("🔮 Stock Predictions")
st.markdown("### Get ML-powered buy/sell signals with confidence scores")

# Load model
@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/stock_classifier.json')
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Please run `model_pipeline.py` first to train the model.")
    st.stop()
else:
    st.success("✅ Model loaded successfully")

# Sidebar - Input controls
st.sidebar.header("Prediction Settings")

ticker_input = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Enter a valid stock ticker (e.g., AAPL, TSLA, NVDA)"
).upper()

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence to trigger a BUY signal"
)

lookback_days = st.sidebar.number_input(
    "Lookback Period (days)",
    min_value=200,
    max_value=1000,
    value=365,
    help="How many days of historical data to fetch"
)

# Feature calculation functions
def calculate_features(df):
    """Calculate all 17 features used in training"""
    df = df.copy()
    
    # Price features (already have open, high, low, close, Volume)
    
    # SMAs
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Volatility (20-day rolling std of returns)
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Drawdown (distance from peak)
    df['peak'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['peak']) / df['peak']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (2 * std_20)
    df['bb_lower'] = sma_20 - (2 * std_20)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    return df

# Feature columns (must match training)
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'Volume',
    'sma_20', 'sma_50', 'sma_200', 'daily_return', 'volatility',
    'drawdown', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width'
]

# Fetch and process data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_predict(ticker, days):
    """Fetch stock data and generate prediction"""
    try:
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return None, "No data found for this ticker"
        
        # Prepare dataframe - ensure we get clean 1D arrays
        df = pd.DataFrame(index=stock_data.index)
        df['open'] = stock_data['Open'].values.flatten() if hasattr(stock_data['Open'].values, 'flatten') else stock_data['Open'].values
        df['high'] = stock_data['High'].values.flatten() if hasattr(stock_data['High'].values, 'flatten') else stock_data['High'].values
        df['low'] = stock_data['Low'].values.flatten() if hasattr(stock_data['Low'].values, 'flatten') else stock_data['Low'].values
        df['close'] = stock_data['Close'].values.flatten() if hasattr(stock_data['Close'].values, 'flatten') else stock_data['Close'].values
        df['Volume'] = stock_data['Volume'].values.flatten() if hasattr(stock_data['Volume'].values, 'flatten') else stock_data['Volume'].values
        
        # Calculate features
        df = calculate_features(df)
        
        # Drop NaN rows (from rolling calculations)
        df = df.dropna()
        
        if len(df) == 0:
            return None, "Not enough data to calculate features"
        
        # Get most recent row for prediction
        latest = df[FEATURE_COLS].iloc[-1:].copy()
        
        return df, latest
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Main prediction area
st.header(f"Prediction for {ticker_input}")

with st.spinner(f"Fetching data for {ticker_input}..."):
    result, latest = fetch_and_predict(ticker_input, lookback_days)
    
    if result is None:
        st.error(f"❌ {latest}")
        st.stop()
    
    df = result

# Make prediction
pred_proba = model.predict_proba(latest)[:, 1][0]
pred_class = 1 if pred_proba >= threshold else 0

# Display prediction
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Prediction",
        "🟢 BUY" if pred_class == 1 else "🔴 HOLD/SELL",
        delta=f"{pred_proba:.1%} confidence"
    )

with col2:
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        delta=f"{price_change:+.2f}%"
    )

with col3:
    st.metric(
        "7-Day Target",
        f">${current_price * 1.03:.2f}",
        delta="+3.0% (model target)"
    )

# Confidence gauge
st.subheader("Confidence Score")

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=pred_proba * 100,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Buy Confidence (%)"},
    delta={'reference': threshold * 100, 'suffix': "% vs threshold"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkgreen" if pred_class == 1 else "darkred"},
        'steps': [
            {'range': [0, 30], 'color': "lightgray"},
            {'range': [30, 50], 'color': "gray"},
            {'range': [50, 70], 'color': "lightblue"},
            {'range': [70, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': threshold * 100
        }
    }
))

fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Feature values
st.subheader("📊 Current Technical Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("RSI (14)", f"{latest['rsi'].values[0]:.1f}")
    st.metric("Volatility", f"{latest['volatility'].values[0]:.4f}")

with col2:
    st.metric("MACD", f"{latest['macd'].values[0]:.2f}")
    st.metric("MACD Signal", f"{latest['macd_signal'].values[0]:.2f}")

with col3:
    st.metric("SMA 20", f"${latest['sma_20'].values[0]:.2f}")
    st.metric("SMA 50", f"${latest['sma_50'].values[0]:.2f}")

with col4:
    st.metric("BB Width", f"${latest['bb_width'].values[0]:.2f}")
    st.metric("Drawdown", f"{latest['drawdown'].values[0]:.2%}")

# Price chart with prediction
st.subheader("📈 Price History & Indicators")

fig = go.Figure()

# Price
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['close'],
    name='Close Price',
    line=dict(color='black', width=2)
))

# SMAs
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['sma_20'],
    name='SMA 20',
    line=dict(color='blue', width=1, dash='dot')
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['sma_50'],
    name='SMA 50',
    line=dict(color='orange', width=1, dash='dot')
))

# Bollinger Bands
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['bb_upper'],
    name='BB Upper',
    line=dict(color='red', width=1, dash='dash'),
    opacity=0.5
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['bb_lower'],
    name='BB Lower',
    line=dict(color='green', width=1, dash='dash'),
    fill='tonexty',
    opacity=0.3
))

# Prediction marker
fig.add_trace(go.Scatter(
    x=[df.index[-1]],
    y=[df['close'].iloc[-1]],
    mode='markers',
    name='Current Position',
    marker=dict(
        size=15,
        color='green' if pred_class == 1 else 'red',
        symbol='star'
    )
))

fig.update_layout(
    title=f"{ticker_input} Price Chart (Last {lookback_days} days)",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Explanation
st.subheader("ℹ️ How to Interpret")

st.info("""
**Signal Interpretation:**
- 🟢 **BUY**: Model predicts >3% return in next 7 days with confidence above threshold
- 🔴 **HOLD/SELL**: Model confidence below threshold or predicts <3% return

**Confidence Score:**
- 50-60%: Weak signal
- 60-70%: Moderate signal
- 70%+: Strong signal

**Remember:** This is a signal generator, not a complete trading system. Real trading requires:
- Entry/exit strategy
- Risk management
- Transaction cost consideration
- Market timing
""")

# Download predictions
st.subheader("💾 Export Data")

if st.button("Generate Detailed Report"):
    report_df = pd.DataFrame({
        'Ticker': [ticker_input],
        'Date': [df.index[-1]],
        'Current Price': [current_price],
        'Prediction': ['BUY' if pred_class == 1 else 'HOLD/SELL'],
        'Confidence': [f"{pred_proba:.2%}"],
        'RSI': [latest['rsi'].values[0]],
        'MACD': [latest['macd'].values[0]],
        'Volatility': [latest['volatility'].values[0]],
        'SMA_20': [latest['sma_20'].values[0]],
        'SMA_50': [latest['sma_50'].values[0]]
    })
    
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name=f"{ticker_input}_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
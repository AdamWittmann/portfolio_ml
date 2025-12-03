# pages/2_🔮_Predictions.py - Multi-Stock Forecast Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / 'models' / 'stock_classifier.json'

st.set_page_config(page_title="Stock Predictions", page_icon="", layout="wide")

# Title
st.title("Multi-Stock Forecast Dashboard")
st.markdown("### Compare ML predictions across multiple stocks simultaneously")

# Load model
@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error(f"❌ Model not found at {MODEL_PATH}! Please run `model_pipeline.py` first to train the model.")
    st.stop()
else:
    st.success("✅ Model loaded successfully")

# Sidebar - Input controls
st.sidebar.header("Forecast Settings")

# Stock ticker list input
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
ticker_input = st.sidebar.text_area(
    "Stock Tickers (one per line)",
    value='\n'.join(default_tickers),
    height=200,
    help="Enter stock ticker symbols, one per line"
)
tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

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

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Display Filters")
show_only_buys = st.sidebar.checkbox("Show only BUY signals", value=False)
min_confidence_display = st.sidebar.slider(
    "Minimum Confidence to Display",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Hide stocks below this confidence level"
)

# Feature calculation functions
def calculate_features(df):
    """Calculate all 17 features used in training"""
    df = df.copy()
    
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

# Fetch and process data for a single ticker
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, days):
    """Fetch stock data and calculate features"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return None
        
        # Prepare dataframe - ensure we get clean 1D arrays
        df = pd.DataFrame(index=stock_data.index)
        df['open'] = stock_data['Open'].values.flatten() if hasattr(stock_data['Open'].values, 'flatten') else stock_data['Open'].values
        df['high'] = stock_data['High'].values.flatten() if hasattr(stock_data['High'].values, 'flatten') else stock_data['High'].values
        df['low'] = stock_data['Low'].values.flatten() if hasattr(stock_data['Low'].values, 'flatten') else stock_data['Low'].values
        df['close'] = stock_data['Close'].values.flatten() if hasattr(stock_data['Close'].values, 'flatten') else stock_data['Close'].values
        df['Volume'] = stock_data['Volume'].values.flatten() if hasattr(stock_data['Volume'].values, 'flatten') else stock_data['Volume'].values
        
        # Calculate features
        df = calculate_features(df)
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) == 0:
            return None
        
        return df
        
    except Exception as e:
        return None

# Generate predictions for all stocks
st.header("Forecast Results")

with st.spinner(f"Analyzing {len(tickers)} stocks..."):
    results = []
    
    for ticker in tickers:
        df = fetch_stock_data(ticker, lookback_days)
        
        if df is None or len(df) == 0:
            continue
        
        # Get most recent row for prediction
        latest = df[FEATURE_COLS].iloc[-1:].copy()
        
        # Make prediction
        pred_proba = model.predict_proba(latest)[:, 1][0]
        pred_class = 1 if pred_proba >= threshold else 0
        
        # Get current metrics
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        results.append({
            'Ticker': ticker,
            'Signal': 'BUY' if pred_class == 1 else 'HOLD',
            'Confidence': pred_proba,
            'Current Price': current_price,
            'Price Change %': price_change_pct,
            'Target Price': current_price * 1.03,
            'RSI': latest['rsi'].values[0],
            'MACD': latest['macd'].values[0],
            'Volatility': latest['volatility'].values[0],
            'Drawdown %': latest['drawdown'].values[0] * 100,
            'SMA 20': latest['sma_20'].values[0],
            'SMA 50': latest['sma_50'].values[0]
        })

if len(results) == 0:
    st.error("❌ No valid data found for any tickers. Please check ticker symbols.")
    st.stop()

# Create results dataframe
results_df = pd.DataFrame(results)

# Apply filters
filtered_df = results_df.copy()
if show_only_buys:
    filtered_df = filtered_df[filtered_df['Signal'] == 'BUY']
if min_confidence_display > 0:
    filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence_display]

# Sort by confidence (highest first)
filtered_df = filtered_df.sort_values('Confidence', ascending=False)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    buy_count = (results_df['Signal'] == 'BUY').sum()
    st.metric("BUY Signals", f"{buy_count}/{len(results_df)}")

with col2:
    avg_confidence = results_df['Confidence'].mean()
    st.metric("Average Confidence", f"{avg_confidence:.1%}")

with col3:
    high_conf_count = (results_df['Confidence'] >= 0.7).sum()
    st.metric("High Confidence (>70%)", f"{high_conf_count}")

with col4:
    avg_price_change = results_df['Price Change %'].mean()
    st.metric("Avg Price Change", f"{avg_price_change:+.2f}%")

# Main forecast table
st.subheader(f"🎯 Predictions ({len(filtered_df)} stocks)")

# Format the dataframe for display
display_df = filtered_df.copy()
display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
display_df['Price Change %'] = display_df['Price Change %'].apply(lambda x: f"{x:+.2f}%")
display_df['Target Price'] = display_df['Target Price'].apply(lambda x: f"${x:.2f}")
display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
display_df['MACD'] = display_df['MACD'].apply(lambda x: f"{x:.2f}")
display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.4f}")
display_df['Drawdown %'] = display_df['Drawdown %'].apply(lambda x: f"{x:.2f}%")
display_df['SMA 20'] = display_df['SMA 20'].apply(lambda x: f"${x:.2f}")
display_df['SMA 50'] = display_df['SMA 50'].apply(lambda x: f"${x:.2f}")

# Color-code the signal column
def color_signal(val):
    if val == 'BUY':
        return 'background-color: #90EE90'  # Light green
    else:
        return 'background-color: #FFB6C1'  # Light red

styled_df = display_df.style.applymap(color_signal, subset=['Signal'])

st.dataframe(styled_df, use_container_width=True, height=400)

# Visualization tabs
tab1, tab2, tab3 = st.tabs(["Confidence Distribution", "Top Picks", "Price vs Confidence"])

with tab1:
    # Confidence distribution histogram
    fig = px.histogram(
        results_df,
        x='Confidence',
        nbins=20,
        color='Signal',
        title="Confidence Score Distribution",
        labels={'Confidence': 'Confidence Score', 'count': 'Number of Stocks'},
        color_discrete_map={'BUY': 'green', 'HOLD': 'red'}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="black", 
                  annotation_text=f"Threshold: {threshold:.0%}")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Top 5 BUY recommendations
    st.subheader("Top 5 BUY Opportunities (Highest Confidence)")
    
    buy_signals = results_df[results_df['Signal'] == 'BUY'].nlargest(5, 'Confidence')
    
    if len(buy_signals) == 0:
        st.warning("No BUY signals at current threshold.")
    else:
        for idx, row in buy_signals.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
                
                with col1:
                    st.markdown(f"### {row['Ticker']}")
                    st.markdown(f"**{row['Confidence']:.1%}** confidence")
                
                with col2:
                    st.metric("Current", f"${row['Current Price']:.2f}", 
                             delta=f"{row['Price Change %']:+.2f}%")
                
                with col3:
                    st.metric("Target", f"${row['Target Price']:.2f}", 
                             delta="+3.0%")
                
                with col4:
                    st.write(f"**RSI:** {row['RSI']:.1f} | **MACD:** {row['MACD']:.2f}")
                    st.write(f"**Vol:** {row['Volatility']:.4f} | **DD:** {row['Drawdown %']:.2f}%")
                
                st.markdown("---")

with tab3:
    # Scatter plot: Price change vs Confidence
    fig = px.scatter(
        results_df,
        x='Confidence',
        y='Price Change %',
        color='Signal',
        size='Volatility',
        hover_data=['Ticker', 'RSI', 'MACD'],
        title="Price Change vs Model Confidence",
        labels={'Confidence': 'Model Confidence', 'Price Change %': 'Recent Price Change (%)'},
        color_discrete_map={'BUY': 'green', 'HOLD': 'red'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# Interpretation guide
with st.expander("ℹ️ How to Interpret This Dashboard"):
    st.markdown("""
    **Signal Types:**
    - 🟢 **BUY**: Model confidence ≥ threshold, predicts >3% return in 7 days
    - 🔴 **HOLD**: Model confidence < threshold or predicts <3% return
    
    **Confidence Levels:**
    - **50-60%**: Weak signal (barely above random chance)
    - **60-70%**: Moderate signal (meaningful edge)
    - **70%+**: Strong signal (high conviction)
    
    **Key Metrics:**
    - **RSI**: Relative Strength Index (>70 overbought, <30 oversold)
    - **MACD**: Trend momentum (positive = bullish, negative = bearish)
    - **Volatility**: Recent price fluctuation (higher = riskier)
    - **Drawdown**: Distance from recent peak (negative = below peak)
    
    **Important Reminders:**
    - This is a **signal generator**, not financial advice
    - Real trading requires entry/exit rules, risk management, and cost consideration
    - Past performance (62.4% win rate, 3.64 Sharpe) doesn't guarantee future results
    - Always do your own research and consider multiple factors
    """)
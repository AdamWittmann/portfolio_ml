# pages/1_EDA.py - Exploratory Data Analysis

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

# Title
st.title("📊 Exploratory Data Analysis")
st.markdown("### Interactive Stock Market Analysis")

# Sidebar - Stock Selection
st.sidebar.header("Settings")
default_tickers = ["AAPL", "MTB", "NVDA", "TSLA", "META", "MSFT", "HD", "NFLX", "CSCO", "SPY"]
selected_tickers = st.sidebar.multiselect(
    "Select Stocks",
    options=["AAPL", "MTB", "NVDA", "TSLA", "META", "MSFT", "HD", "NFLX", "CSCO", "SPY", "VOO", "UVIX", "PATH", "NLR", "OKLO"],
    default=default_tickers
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(pd.to_datetime("2022-01-01"), pd.to_datetime("2024-10-04")),
    max_value=pd.to_datetime("today")
)

# Load data
@st.cache_data
def load_data(tickers, start, end):
    """Load stock data from Yahoo Finance"""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data

if len(selected_tickers) == 0:
    st.warning("⚠️ Please select at least one stock ticker from the sidebar")
    st.stop()

with st.spinner("Loading stock data..."):
    try:
        data = load_data(selected_tickers, date_range[0], date_range[1])
        prices = data['Close']
        volume = data['Volume']
        returns = prices.pct_change()
        
        st.success(f"✅ Loaded {len(prices)} days of data for {len(selected_tickers)} stocks")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Charts", 
    "🔥 Correlation Heatmap", 
    "📊 Technical Indicators",
    "💹 Sharpe Ratio",
    "📉 Returns Distribution"
])

# TAB 1: Price Charts
with tab1:
    st.header("Stock Price History")
    
    # Plotly interactive chart
    fig = go.Figure()
    
    for ticker in selected_tickers:
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[ticker] if len(selected_tickers) > 1 else prices,
            mode='lines',
            name=ticker,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(prices.describe().T.style.format("{:.2f}"))

# TAB 2: Correlation Heatmap
with tab2:
    st.header("Stock Correlation Matrix")
    st.markdown("*How stocks move together (1 = perfect correlation, -1 = inverse)*")
    
    if len(selected_tickers) > 1:
        corr_matrix = prices.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=600,
            xaxis=dict(side='bottom')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Positive Correlation:**
            - 0.7-1.0: Strong (move together)
            - 0.3-0.7: Moderate
            """)
        with col2:
            st.info("""
            **Negative/Weak:**
            - -0.3-0.3: Little relationship
            - < -0.3: Negative (move opposite)
            """)
    else:
        st.warning("Select multiple stocks to see correlations")

# TAB 3: Technical Indicators
with tab3:
    st.header("Technical Indicators")
    
    # Stock selector for detailed view
    selected_stock = st.selectbox("Select stock for detailed analysis:", selected_tickers)
    
    # Get data for selected stock
    stock_prices = prices[selected_stock] if len(selected_tickers) > 1 else prices
    stock_volume = volume[selected_stock] if len(selected_tickers) > 1 else volume
    
    # Calculate indicators
    rsi = ta.momentum.RSIIndicator(stock_prices, window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(stock_prices)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(stock_prices, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    
    # EMAs
    ema_12 = stock_prices.ewm(span=12, adjust=False).mean()
    ema_26 = stock_prices.ewm(span=26, adjust=False).mean()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{selected_stock} Price & Bollinger Bands',
            'MACD',
            'RSI (14-day)',
            'EMA Crossover (12/26)'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price + Bollinger Bands
    fig.add_trace(go.Scatter(x=stock_prices.index, y=stock_prices, name='Price', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper, name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower, name='BB Lower', line=dict(color='green', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_mid.index, y=bb_mid, name='BB Mid', line=dict(color='blue', dash='dot')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name='Signal', line=dict(color='red')), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # EMA
    fig.add_trace(go.Scatter(x=ema_12.index, y=ema_12, name='EMA 12', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=ema_26.index, y=ema_26, name='EMA 26', line=dict(color='red')), row=4, col=1)
    
    fig.update_layout(height=1000, showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="EMA", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current RSI", f"{rsi.iloc[-1]:.1f}", 
                delta="Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral")
    col2.metric("MACD", f"{macd_line.iloc[-1]:.2f}")
    col3.metric("EMA 12", f"${ema_12.iloc[-1]:.2f}")
    col4.metric("EMA 26", f"${ema_26.iloc[-1]:.2f}")

# TAB 4: Sharpe Ratio
with tab4:
    st.header("Rolling Sharpe Ratio (252-day)")
    st.markdown("*Risk-adjusted returns over time*")
    
    # Calculate rolling Sharpe
    window = 252
    risk_free_rate = 0.037 / 252
    
    fig = go.Figure()
    
    for ticker in selected_tickers:
        ticker_returns = returns[ticker] if len(selected_tickers) > 1 else returns
        excess_returns = ticker_returns - risk_free_rate
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = ticker_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name=ticker
        ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Good (>1)")
    fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Excellent (>2)")
    
    fig.update_layout(
        title="Rolling 1-Year Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio (Annualized)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Sharpe Ratio Interpretation:**
    - < 0: Losing money
    - 0-1: Positive but subpar
    - 1-2: Good
    - > 2: Excellent (rare to sustain)
    """)

# TAB 5: Returns Distribution
with tab5:
    st.header("Daily Returns Distribution")
    
    # Create histogram for each stock
    fig = make_subplots(
        rows=len(selected_tickers), cols=1,
        subplot_titles=[f'{ticker} Daily Returns' for ticker in selected_tickers],
        vertical_spacing=0.05
    )
    
    for i, ticker in enumerate(selected_tickers, 1):
        ticker_returns = returns[ticker].dropna() if len(selected_tickers) > 1 else returns.dropna()
        
        fig.add_trace(
            go.Histogram(
                x=ticker_returns,
                name=ticker,
                nbinsx=50,
                marker_color=px.colors.qualitative.Plotly[i-1]
            ),
            row=i, col=1
        )
        
        fig.update_xaxes(title_text="Daily Return", row=i, col=1)
        fig.update_yaxes(title_text="Frequency", row=i, col=1)
    
    fig.update_layout(height=300*len(selected_tickers), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Return Statistics")
    stats_df = pd.DataFrame({
        'Mean Return': returns.mean() * 252,  # Annualized
        'Volatility': returns.std() * np.sqrt(252),  # Annualized
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })
    st.dataframe(stats_df.style.format("{:.4f}"))